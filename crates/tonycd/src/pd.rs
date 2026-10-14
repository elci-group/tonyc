//! Preemptive Dynamism (directive §9, §24; Phase 6).
//!
//! PD predicts *future computational demand* from Tonyc's own demand log —
//! nothing else is observed — and prepares answers before they are asked
//! for, when the expected value clears the cost:
//!
//! - a request is a candidate only if it was asked for at least
//!   `min_demand` times within the window, and no fresh answer to it exists
//!   (it went stale, expired, is about to expire, or was never produced);
//! - P(asked within the horizon) = 1 − exp(−λ·horizon), with λ the request's
//!   observed demand rate;
//! - it is prepared only if P ≥ the probability threshold — which rises
//!   while measured yield is low — and P × conventional cost avoided ≥
//!   `value_ratio` × estimated operating cost;
//! - it runs as an ordinary SPECULATIVE job cloned from the request's own
//!   envelope (same locality, quality, scope), so it yields to pressure and
//!   urgent work and is bounded by the speculative daily budget and
//!   `max_pending`.
//!
//! Every prediction is tracked to an outcome — consumed (a PD hit) or
//! discarded — and Preemption Yield (consumed ÷ produced) is reported, so PD
//! cannot quietly degenerate into gratuitous background inference. PD's own
//! inference is logged as `speculative`, never as demand.

use crate::daemon::Daemon;
use crate::economics;
use crate::store::{Store, json_of, now_ms};
use anyhow::Result;
use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tonyc_core::cognition::ObjectState;
use tonyc_core::envelope::{Urgency, WorkEnvelope};
use tonyc_core::model::estimate_tokens;
use tonyc_core::plan::PlanInput;
use tonyc_core::{JobState, Millis, plan};

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS pd_templates (
    semantic_key TEXT PRIMARY KEY,
    envelope     TEXT NOT NULL,
    updated_at   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pd_predictions (
    id           TEXT PRIMARY KEY,
    at           INTEGER NOT NULL,
    semantic_key TEXT NOT NULL,
    objective    TEXT NOT NULL,
    reason       TEXT NOT NULL,
    probability  REAL NOT NULL,
    value_gbp    REAL NOT NULL,
    cost_gbp     REAL NOT NULL,
    evidence     TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    state        TEXT NOT NULL,
    object_id    TEXT,
    produced_at  INTEGER,
    settled_at   INTEGER,
    outcome      TEXT,
    compute_ms   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS pd_predictions_state ON pd_predictions (state, at);
CREATE INDEX IF NOT EXISTS pd_predictions_job ON pd_predictions (job_id);
"#;

/// Demand kinds that count as an agent asking for something.
const ASKS: &str = "'try_miss', 'try_hit', 'inference', 'reuse_hit'";

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub id: String,
    pub at: Millis,
    pub semantic_key: String,
    pub objective: String,
    pub reason: String,
    pub probability: f64,
    pub value_gbp: f64,
    pub cost_gbp: f64,
    pub evidence: Value,
    pub job_id: String,
    /// QUEUED (job submitted), PRODUCED, CONSUMED, DISCARDED, FAILED.
    pub state: String,
    pub object_id: Option<String>,
    pub produced_at: Option<Millis>,
    pub settled_at: Option<Millis>,
    pub outcome: Option<String>,
    pub compute_ms: i64,
}

const PRED_COLS: &str =
    "id, at, semantic_key, objective, reason, probability, value_gbp, cost_gbp, evidence, job_id, state,
    object_id, produced_at, settled_at, outcome, compute_ms";

fn pred_from_row(r: &rusqlite::Row) -> rusqlite::Result<Prediction> {
    Ok(Prediction {
        id: r.get(0)?,
        at: r.get(1)?,
        semantic_key: r.get(2)?,
        objective: r.get(3)?,
        reason: r.get(4)?,
        probability: r.get(5)?,
        value_gbp: r.get(6)?,
        cost_gbp: r.get(7)?,
        evidence: json_of(r.get(8)?)?,
        job_id: r.get(9)?,
        state: r.get(10)?,
        object_id: r.get(11)?,
        produced_at: r.get(12)?,
        settled_at: r.get(13)?,
        outcome: r.get(14)?,
        compute_ms: r.get(15)?,
    })
}

/// A request's demand within the window.
#[derive(Debug, Clone, Serialize)]
pub struct Demand {
    pub semantic_key: String,
    pub asks: i64,
    pub try_misses: i64,
    pub last_at: Millis,
    pub objective: String,
}

impl Store {
    /// Remembers how a request was asked, so PD can ask it again.
    pub fn upsert_template(&self, env: &WorkEnvelope) -> Result<()> {
        if env.task.pin_model.is_some() || !env.reuse.cache {
            return Ok(());
        }
        self.conn().execute(
            "INSERT INTO pd_templates (semantic_key, envelope, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(semantic_key) DO UPDATE SET envelope = excluded.envelope, updated_at = excluded.updated_at",
            params![env.semantic_hash, serde_json::to_string(env)?, now_ms()],
        )?;
        Ok(())
    }

    pub fn template(&self, key: &str) -> Result<Option<WorkEnvelope>> {
        let conn = self.conn();
        Ok(conn
            .query_row("SELECT envelope FROM pd_templates WHERE semantic_key = ?1", [key], |r| json_of(r.get(0)?))
            .optional()?)
    }

    pub fn demand_by_key(&self, since: Millis) -> Result<Vec<Demand>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!(
            "SELECT semantic_key, COUNT(*), SUM(kind = 'try_miss'), MAX(at), MAX(objective) FROM demand
             WHERE at >= ?1 AND semantic_key IS NOT NULL AND kind IN ({ASKS})
             GROUP BY semantic_key ORDER BY COUNT(*) DESC, MAX(at) DESC"
        ))?;
        Ok(stmt
            .query_map([since], |r| {
                Ok(Demand {
                    semantic_key: r.get(0)?,
                    asks: r.get(1)?,
                    try_misses: r.get(2)?,
                    last_at: r.get(3)?,
                    objective: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn insert_prediction(&self, p: &Prediction) -> Result<()> {
        self.conn().execute(
            &format!("INSERT INTO pd_predictions ({PRED_COLS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)"),
            params![
                p.id,
                p.at,
                p.semantic_key,
                p.objective,
                p.reason,
                p.probability,
                p.value_gbp,
                p.cost_gbp,
                p.evidence.to_string(),
                p.job_id,
                p.state,
                p.object_id,
                p.produced_at,
                p.settled_at,
                p.outcome,
                p.compute_ms
            ],
        )?;
        Ok(())
    }

    pub fn predictions(&self, state: Option<&str>, limit: i64) -> Result<Vec<Prediction>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!(
            "SELECT {PRED_COLS} FROM pd_predictions WHERE ?1 IS NULL OR state = ?1 ORDER BY at DESC LIMIT ?2"
        ))?;
        Ok(stmt.query_map(params![state, limit], pred_from_row)?.collect::<rusqlite::Result<_>>()?)
    }

    pub fn prediction_for_job(&self, job: &str) -> Result<Option<Prediction>> {
        let conn = self.conn();
        Ok(conn
            .query_row(&format!("SELECT {PRED_COLS} FROM pd_predictions WHERE job_id = ?1"), [job], pred_from_row)
            .optional()?)
    }

    fn settle_prediction(
        &self,
        id: &str,
        state: &str,
        outcome: &str,
        object: Option<&str>,
        compute_ms: Option<i64>,
    ) -> Result<()> {
        let now = now_ms();
        let produced = state == "PRODUCED";
        self.conn().execute(
            "UPDATE pd_predictions SET state = ?2, outcome = ?3,
                 object_id = COALESCE(?4, object_id),
                 compute_ms = COALESCE(?5, compute_ms),
                 produced_at = CASE WHEN ?6 THEN ?7 ELSE produced_at END,
                 settled_at = CASE WHEN ?6 THEN settled_at ELSE ?7 END
             WHERE id = ?1",
            params![id, state, outcome, object, compute_ms, produced, now],
        )?;
        Ok(())
    }

    /// Counts by state among predictions made since a time.
    pub fn prediction_counts(&self, since: Millis) -> Result<(i64, i64, i64, i64, i64, i64)> {
        let conn = self.conn();
        Ok(conn.query_row(
            "SELECT COALESCE(SUM(state = 'QUEUED'), 0), COALESCE(SUM(state = 'PRODUCED'), 0),
                    COALESCE(SUM(state = 'CONSUMED'), 0), COALESCE(SUM(state = 'DISCARDED'), 0),
                    COALESCE(SUM(CASE WHEN state = 'CONSUMED' THEN compute_ms ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN state = 'DISCARDED' THEN compute_ms ELSE 0 END), 0)
             FROM pd_predictions WHERE at >= ?1",
            [since],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
        )?)
    }

    /// The conventional cost recorded the last time this request ran.
    fn last_conventional(&self, key: &str) -> Result<Option<f64>> {
        let conn = self.conn();
        Ok(conn
            .query_row(
                "SELECT l.conventional_gbp FROM ledger l JOIN jobs j ON j.id = l.job_id
                 WHERE j.semantic_hash = ?1 AND l.kind = 'inference' ORDER BY l.at DESC LIMIT 1",
                [key],
                |r| r.get(0),
            )
            .optional()?)
    }
}

/// Records that a PD-prepared answer was used: a PD hit. The prediction is
/// found through the job that produced the object, so a hit counts even if
/// it comes before PD has next settled the prediction as produced.
pub fn note_consumed(store: &Store, object_id: &str, how: &str) -> Result<()> {
    let found: Option<(String, String)> = store
        .conn()
        .query_row(
            "SELECT p.id, p.job_id FROM pd_predictions p JOIN results r ON r.job_id = p.job_id
             WHERE r.object_id = ?1 AND p.state IN ('QUEUED', 'PRODUCED')",
            [object_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((id, job)) = found else { return Ok(()) };
    let compute = store.ledger_row(&job)?.map(|l| l.wall_ms).unwrap_or(0);
    let now = now_ms();
    store.conn().execute(
        "UPDATE pd_predictions SET state = 'CONSUMED', outcome = ?2, object_id = ?3,
             produced_at = COALESCE(produced_at, ?4), settled_at = ?4, compute_ms = ?5
         WHERE id = ?1",
        params![id, format!("used: {how}"), object_id, now, compute],
    )?;
    Ok(())
}

/// Measured yield and the probability threshold it implies.
#[derive(Debug, Clone, Serialize)]
pub struct Yield {
    pub produced: i64,
    pub consumed: i64,
    pub discarded: i64,
    /// Consumed ÷ produced (directive §24).
    pub preemption_yield: Option<f64>,
    /// Consumed ÷ settled: what the threshold adapts to.
    pub settled_yield: Option<f64>,
    pub threshold: f64,
    pub compute_used_ms: i64,
    pub compute_discarded_ms: i64,
}

/// Yield over the window. Once five predictions have settled, the
/// threshold is the configured one divided by the settled yield (floored
/// at a quarter): PD gets pickier while its work goes unused.
pub fn measure(d: &Daemon) -> Result<Yield> {
    let since = now_ms() - d.cfg.pd.window_ms()?;
    let (queued, produced_open, consumed, discarded, used_ms, discarded_ms) = d.store.prediction_counts(since)?;
    let _ = queued;
    let produced = produced_open + consumed + discarded;
    let settled = consumed + discarded;
    let settled_yield = (settled > 0).then(|| consumed as f64 / settled as f64);
    let base = d.cfg.pd.min_probability;
    let threshold = match settled_yield {
        Some(y) if settled >= 5 => (base / y.max(0.25)).min(1.0),
        _ => base,
    };
    Ok(Yield {
        produced,
        consumed,
        discarded,
        preemption_yield: (produced > 0).then(|| consumed as f64 / produced as f64),
        settled_yield,
        threshold,
        compute_used_ms: used_ms,
        compute_discarded_ms: discarded_ms,
    })
}

/// Moves predictions to their outcomes.
pub fn settle(d: &Daemon) -> Result<()> {
    let horizon = d.cfg.pd.horizon_ms()?;
    let now = now_ms();
    for p in d.store.predictions(Some("QUEUED"), 1000)? {
        let Some(job) = d.store.get(&p.job_id)? else { continue };
        if job.state == JobState::Ready {
            let result = d.store.result(&p.job_id)?;
            let compute = d.store.ledger_row(&p.job_id)?.map(|l| l.wall_ms);
            match result.and_then(|r| r.object_id) {
                Some(obj) => d.store.settle_prediction(&p.id, "PRODUCED", "prepared", Some(&obj), compute)?,
                None => d.store.settle_prediction(&p.id, "FAILED", "produced no reusable object", None, compute)?,
            }
        } else if job.state.is_terminal() {
            d.store.settle_prediction(&p.id, "FAILED", &format!("job ended {}", job.state), None, None)?;
        }
    }
    for p in d.store.predictions(Some("PRODUCED"), 1000)? {
        let Some(obj_id) = &p.object_id else { continue };
        let outcome = match d.store.object(obj_id)? {
            None => Some("its answer was deleted before use".to_string()),
            Some(o) if o.state == ObjectState::Stale => {
                Some(format!("went stale before use: {}", o.stale_reason.unwrap_or_default()))
            }
            Some(_) if p.produced_at.is_some_and(|t| t + horizon < now) => {
                Some("not asked for within the horizon".into())
            }
            Some(_) => None,
        };
        if let Some(why) = outcome {
            d.store.settle_prediction(&p.id, "DISCARDED", &why, None, None)?;
        }
    }
    Ok(())
}

/// One request PD looked at, and what it decided.
#[derive(Debug, Clone, Serialize)]
pub struct Considered {
    pub semantic_key: String,
    pub objective: String,
    pub prepared: bool,
    pub reason: String,
    pub probability: Option<f64>,
    pub value_gbp: Option<f64>,
    pub cost_gbp: Option<f64>,
    pub demand: Demand,
}

/// One evaluation: what PD considered and why.
#[derive(Debug, Clone, Serialize)]
pub struct Evaluation {
    pub at: Millis,
    pub status: String,
    pub threshold: f64,
    pub considered: Vec<Considered>,
    pub submitted: Vec<String>,
}

pub fn paused(store: &Store) -> bool {
    store.setting("pd_paused").ok().flatten().is_some_and(|v| v == "true")
}

/// One PD pass: settle outcomes, evaluate demand, submit what pays.
pub fn evaluate(d: &Arc<Daemon>) -> Result<Evaluation> {
    settle(d)?;
    let now = now_ms();
    let y = measure(d)?;
    let mut eval = Evaluation {
        at: now,
        status: "evaluated".into(),
        threshold: y.threshold,
        considered: vec![],
        submitted: vec![],
    };
    if !d.cfg.pd.enabled {
        eval.status = "disabled in configuration".into();
        return Ok(eval);
    }
    if paused(&d.store) {
        eval.status = "paused".into();
        return Ok(eval);
    }
    let pending = d.store.predictions(Some("QUEUED"), 1000)?.len();
    let room = d.cfg.pd.max_pending.saturating_sub(pending);
    let window = d.cfg.pd.window_ms()?;
    let horizon = d.cfg.pd.horizon_ms()?;
    let econ = &d.cfg.economics;

    let mut accepted: Vec<(f64, Considered, WorkEnvelope)> = Vec::new();
    for demand in d.store.demand_by_key(now - window)? {
        let key = demand.semantic_key.clone();
        let mut c = Considered {
            semantic_key: key.clone(),
            objective: demand.objective.clone(),
            prepared: false,
            reason: String::new(),
            probability: None,
            value_gbp: None,
            cost_gbp: None,
            demand: demand.clone(),
        };
        let mut skip = |c: &mut Considered, why: String| {
            c.reason = why;
            eval.considered.push(c.clone());
        };
        if demand.asks < d.cfg.pd.min_demand as i64 {
            skip(&mut c, format!("asked {}× (needs {})", demand.asks, d.cfg.pd.min_demand));
            continue;
        }
        let Some(template) = d.store.template(&key)? else {
            skip(&mut c, "no record of how it was asked".into());
            continue;
        };
        let open = d.store.predictions(None, 1000)?.into_iter().find(|p| {
            p.semantic_key == key
                && (p.state == "QUEUED"
                    || (p.state == "PRODUCED"
                        && p.object_id.as_ref().is_some_and(|o| {
                            d.store.object(o).ok().flatten().is_some_and(|o| o.state == ObjectState::Fresh)
                        })))
        });
        if let Some(p) = open {
            skip(&mut c, format!("already prepared ({} {})", p.id, p.state));
            continue;
        }
        // Is there a fresh answer already, and for how long?
        let mut why = format!("asked {}× in the last {}", demand.asks, tonyc_core::duration::Human(window));
        if demand.try_misses > 0 {
            why += &format!(" ({} try miss(es))", demand.try_misses);
        }
        let objects = d.store.objects_by_key(&key)?;
        let fresh = objects.iter().find(|o| {
            o.state == ObjectState::Fresh
                && o.scope.visible_to(&template.reuse.scope)
                && crate::cognition::check(&d.store, o, now).ok().is_some_and(|r| r.is_ok())
        });
        match fresh {
            Some(o) => match o.expires_at {
                Some(t) if t - now < horizon / 4 => why += "; its answer expires soon",
                _ => {
                    skip(&mut c, "a fresh answer exists".into());
                    continue;
                }
            },
            None => match objects.iter().find(|o| o.state == ObjectState::Stale) {
                Some(o) => why += &format!("; its answer went stale ({})", o.stale_reason.clone().unwrap_or_default()),
                None => why += "; no answer exists",
            },
        }
        // Probability of being asked within the horizon (Poisson).
        let rate = demand.asks as f64 / window as f64;
        let p = 1.0 - (-rate * horizon as f64).exp();
        c.probability = Some(p);
        if p < y.threshold {
            skip(&mut c, format!("{why}; P {p:.2} < threshold {:.2}", y.threshold));
            continue;
        }
        // Value: the conventional cost avoided if it is asked for.
        let prompt_tokens =
            estimate_tokens(template.task.objective.len() + template.task.input_text.as_ref().map_or(0, |t| t.len()));
        let value = d
            .store
            .last_conventional(&key)?
            .unwrap_or_else(|| economics::conventional_cost(econ, prompt_tokens, template.task.max_tokens as u64));
        c.value_gbp = Some(value);
        // Cost: what running it would take now, from the planner.
        let mut spec = template.clone();
        spec.latency.class = Urgency::Speculative;
        spec.latency.deadline_at = None;
        let cands = crate::refinement::candidates(d);
        let planned = plan(&PlanInput {
            envelope: &spec,
            candidates: &cands,
            ceiling: &d.ceiling(),
            now,
            prompt_tokens,
            validator_available: true,
            budget_left_gbp: Some(
                (d.cfg.budgets.remote_monthly_gbp - d.store.spend_since(economics::month_start(now))?).max(0.0),
            ),
            reuse_potential: 0.0,
            future_value: p,
        });
        let decision = match planned {
            Ok(dec) => dec,
            Err(f) => {
                skip(&mut c, format!("{why}; cannot run now: {}", f.reason));
                continue;
            }
        };
        let domain = crate::daemon::domain_of(&decision.runtime);
        let energy = economics::energy_wh(econ, &domain, false, decision.predicted_runtime_ms);
        let cost = energy / 1000.0 * econ.electricity_gbp_per_kwh + decision.estimated_cost_gbp;
        c.cost_gbp = Some(cost);
        if p * value < d.cfg.pd.value_ratio * cost {
            skip(
                &mut c,
                format!("{why}; expected value £{:.5} < {}× cost £{cost:.5}", p * value, d.cfg.pd.value_ratio),
            );
            continue;
        }
        c.reason = format!(
            "{why}; P(asked within {}) {p:.2}; value £{value:.5} vs cost £{cost:.5}",
            tonyc_core::duration::Human(horizon)
        );
        accepted.push((p * value - cost, c, spec));
    }
    accepted.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (i, (_, mut c, mut spec)) in accepted.into_iter().enumerate() {
        if i >= room {
            c.reason += &format!("; waiting: {} speculative job(s) already pending", d.cfg.pd.max_pending);
            eval.considered.push(c);
            continue;
        }
        let job_id = crate::api::new_id();
        spec.id = job_id.clone();
        spec.submitted_at = now_ms();
        d.store.insert_job(&spec)?;
        let pred = Prediction {
            id: format!("pd_{}", &job_id[4..16]),
            at: now_ms(),
            semantic_key: c.semantic_key.clone(),
            objective: c.objective.clone(),
            reason: c.reason.clone(),
            probability: c.probability.unwrap_or(0.0),
            value_gbp: c.value_gbp.unwrap_or(0.0),
            cost_gbp: c.cost_gbp.unwrap_or(0.0),
            evidence: json!({"demand": c.demand, "threshold": y.threshold}),
            job_id: job_id.clone(),
            state: "QUEUED".into(),
            object_id: None,
            produced_at: None,
            settled_at: None,
            outcome: None,
            compute_ms: 0,
        };
        d.store.insert_prediction(&pred)?;
        d.store.note(&job_id, "speculative", json!({"prediction": pred.id, "reason": pred.reason}))?;
        c.prepared = true;
        eval.submitted.push(job_id);
        eval.considered.push(c);
    }
    if !eval.submitted.is_empty() {
        d.wake.notify_one();
    }
    Ok(eval)
}
