//! Economics (directive §12, §21, §24; Phase 4).
//!
//! Three records feed everything here, all append-only:
//! - **observations**: how each completed inference actually performed;
//! - **quality observations**: benchmark cases a model passed or failed;
//! - **the ledger**: one accounting row per completed job — energy,
//!   operating cost, spend, and what the same tokens would have cost on the
//!   reference premium API.
//!
//! Measured profiles replace static priors once a model has run a few
//! times; budgets are checked against the ledger; the report derives the
//! Inference Avoidance Ratio from it. Every money figure is an estimate
//! under stated assumptions (`[economics]` in the configuration).

use crate::config::{BudgetsConfig, EconomicsConfig};
use crate::store::{Store, now_ms};
use anyhow::Result;
use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use std::collections::BTreeMap;
use tonyc_core::Millis;
use tonyc_core::envelope::Urgency;
use tonyc_core::model::{ModelProfile, ProfileSource, QualityEvidence, Timings};

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS observations (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    at          INTEGER NOT NULL,
    job_id      TEXT NOT NULL,
    model       TEXT NOT NULL,
    domain      TEXT NOT NULL,
    tokens_in   INTEGER NOT NULL,
    tokens_out  INTEGER NOT NULL,
    wall_ms     INTEGER NOT NULL,
    load_ms     INTEGER,
    prefill_ms  INTEGER,
    generate_ms INTEGER
);
CREATE INDEX IF NOT EXISTS observations_model ON observations (model, seq);
CREATE TABLE IF NOT EXISTS quality_observations (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    at         INTEGER NOT NULL,
    model      TEXT NOT NULL,
    capability TEXT NOT NULL,
    case_id    TEXT NOT NULL,
    passed     INTEGER NOT NULL,
    run_id     TEXT NOT NULL,
    job_id     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS quality_observations_model ON quality_observations (model);
CREATE TABLE IF NOT EXISTS ledger (
    job_id           TEXT PRIMARY KEY,
    at               INTEGER NOT NULL,
    kind             TEXT NOT NULL,
    class            TEXT NOT NULL,
    model            TEXT NOT NULL,
    domain           TEXT NOT NULL,
    leaves_host      INTEGER NOT NULL,
    tokens_in        INTEGER NOT NULL,
    tokens_out       INTEGER NOT NULL,
    wall_ms          INTEGER NOT NULL,
    energy_wh        REAL NOT NULL,
    energy_gbp       REAL NOT NULL,
    spend_gbp        REAL NOT NULL,
    conventional_gbp REAL NOT NULL,
    context_objects  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_at ON ledger (at);
CREATE TRIGGER IF NOT EXISTS observations_no_update BEFORE UPDATE ON observations
    BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_delete BEFORE DELETE ON observations
    BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS quality_no_update BEFORE UPDATE ON quality_observations
    BEGIN SELECT RAISE(ABORT, 'quality observations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS quality_no_delete BEFORE DELETE ON quality_observations
    BEGIN SELECT RAISE(ABORT, 'quality observations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
    BEGIN SELECT RAISE(ABORT, 'the ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
    BEGIN SELECT RAISE(ABORT, 'the ledger is append-only'); END;
"#;

/// Runs a model needs before measurements replace its static priors.
pub const MIN_RUNS: u32 = 3;
/// Most recent runs measurements are drawn from.
const WINDOW: i64 = 50;
/// A load longer than this means the run started cold.
const COLD_LOAD_MS: i64 = 1000;

/// What has been measured for one model.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Observed {
    pub runs: u32,
    pub tokens_per_sec: Option<f64>,
    pub prefill_tokens_per_sec: Option<f64>,
    /// 75th percentile overhead of cold runs.
    pub cold_startup_ms: Option<i64>,
    /// Median overhead of warm runs.
    pub warm_startup_ms: Option<i64>,
    /// Capability → (passed, total) from benchmarks.
    pub quality: BTreeMap<String, (u32, u32)>,
}

impl Observed {
    /// Applies measurements to a profile (before any configured override).
    pub fn apply(&self, p: &mut ModelProfile) {
        p.evidence.runs = self.runs;
        if self.runs >= MIN_RUNS {
            let mut measured = false;
            if let Some(t) = self.tokens_per_sec {
                p.performance.tokens_per_sec = t;
                measured = true;
            }
            if let Some(t) = self.prefill_tokens_per_sec {
                p.performance.prefill_tokens_per_sec = t;
                measured = true;
            }
            if let Some(ms) = self.cold_startup_ms {
                p.performance.startup_ms = ms;
                measured = true;
            }
            if let Some(ms) = self.warm_startup_ms {
                p.performance.warm_startup_ms = ms;
                measured = true;
            }
            if measured {
                p.source = ProfileSource::Observed;
            }
        }
        // Priors are read from the profile as it was before any blending, so
        // that a measured `general` does not become the prior of others.
        let base = p.capabilities.clone();
        for (cap, (passed, total)) in &self.quality {
            if *total == 0 {
                continue;
            }
            let prior = base.get(cap).or_else(|| base.get("general")).copied().unwrap_or(0.5);
            let ev = QualityEvidence { passed: *passed, total: *total, prior };
            p.capabilities.insert(cap.clone(), ev.estimate());
            p.evidence.quality.insert(cap.clone(), ev);
            p.source = ProfileSource::Observed;
        }
    }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    percentile(&mut v, 0.5)
}

fn percentile(v: &mut [f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    // Linear interpolation between the closest ranks.
    let pos = (v.len() - 1) as f64 * q;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    Some(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64))
}

#[derive(Debug, Clone, Serialize)]
pub struct ObservationRow {
    pub at: Millis,
    pub job_id: String,
    pub domain: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub wall_ms: i64,
    pub timings: Option<Timings>,
}

/// One accounting row.
#[derive(Debug, Clone, Serialize)]
pub struct LedgerRow {
    pub job_id: String,
    pub at: Millis,
    /// `inference` or `reuse`.
    pub kind: String,
    pub class: Urgency,
    pub model: String,
    pub domain: String,
    pub leaves_host: bool,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub wall_ms: i64,
    pub energy_wh: f64,
    pub energy_gbp: f64,
    /// Money actually spent (paid runtimes).
    pub spend_gbp: f64,
    /// What the tokens would have cost on the reference API.
    pub conventional_gbp: f64,
    pub context_objects: u32,
}

impl LedgerRow {
    pub fn operating_gbp(&self) -> f64 {
        self.energy_gbp + self.spend_gbp
    }
}

pub fn conventional_cost(cfg: &EconomicsConfig, tokens_in: u64, tokens_out: u64) -> f64 {
    (tokens_in as f64 * cfg.reference_input_gbp_per_mtok + tokens_out as f64 * cfg.reference_output_gbp_per_mtok) / 1e6
}

/// Energy drawn by a run of `wall_ms` on a domain.
pub fn energy_wh(cfg: &EconomicsConfig, domain: &str, gpu: bool, wall_ms: i64) -> f64 {
    let watts = if gpu {
        cfg.gpu_watts
    } else if domain == "local" {
        cfg.local_watts
    } else {
        cfg.node_watts
    };
    watts * wall_ms.max(0) as f64 / 3_600_000.0
}

impl Store {
    pub fn record_observation(&self, model: &str, o: &ObservationRow) -> Result<()> {
        let t = o.timings.unwrap_or_default();
        let has = o.timings.is_some();
        self.conn().execute(
            "INSERT INTO observations (at, job_id, model, domain, tokens_in, tokens_out, wall_ms, load_ms, prefill_ms, generate_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                o.at,
                o.job_id,
                model,
                o.domain,
                o.tokens_in as i64,
                o.tokens_out as i64,
                o.wall_ms,
                has.then_some(t.load_ms),
                has.then_some(t.prefill_ms),
                has.then_some(t.generate_ms)
            ],
        )?;
        Ok(())
    }

    pub fn observations(&self, model: &str, limit: i64) -> Result<Vec<ObservationRow>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT at, job_id, domain, tokens_in, tokens_out, wall_ms, load_ms, prefill_ms, generate_ms
             FROM observations WHERE model = ?1 ORDER BY seq DESC LIMIT ?2",
        )?;
        Ok(stmt
            .query_map(params![model, limit], |r| {
                let load: Option<i64> = r.get(6)?;
                Ok(ObservationRow {
                    at: r.get(0)?,
                    job_id: r.get(1)?,
                    domain: r.get(2)?,
                    tokens_in: r.get::<_, i64>(3)? as u64,
                    tokens_out: r.get::<_, i64>(4)? as u64,
                    wall_ms: r.get(5)?,
                    timings: match load {
                        Some(load_ms) => {
                            Some(Timings { load_ms, prefill_ms: r.get::<_, i64>(7)?, generate_ms: r.get::<_, i64>(8)? })
                        }
                        None => None,
                    },
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn observed_models(&self) -> Result<Vec<String>> {
        let conn = self.conn();
        let mut stmt =
            conn.prepare("SELECT model FROM observations UNION SELECT model FROM quality_observations ORDER BY 1")?;
        Ok(stmt.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    }

    pub fn record_quality(
        &self,
        model: &str,
        capability: &str,
        case: &str,
        passed: bool,
        run: &str,
        job: &str,
    ) -> Result<()> {
        self.conn().execute(
            "INSERT INTO quality_observations (at, model, capability, case_id, passed, run_id, job_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![now_ms(), model, capability, case, passed, run, job],
        )?;
        Ok(())
    }

    /// Measurements for a model, from its most recent runs.
    pub fn observed(&self, model: &str) -> Result<Observed> {
        let rows = self.observations(model, WINDOW)?;
        let total_runs: i64 =
            self.conn().query_row("SELECT COUNT(*) FROM observations WHERE model = ?1", [model], |r| r.get(0))?;
        let mut generation = Vec::new();
        let mut prefill = Vec::new();
        let mut cold = Vec::new();
        let mut warm = Vec::new();
        for o in &rows {
            let Some(t) = o.timings else { continue };
            if o.tokens_out >= 8 && t.generate_ms > 0 {
                generation.push(o.tokens_out as f64 * 1000.0 / t.generate_ms as f64);
            }
            if o.tokens_in >= 8 && t.prefill_ms > 0 {
                prefill.push(o.tokens_in as f64 * 1000.0 / t.prefill_ms as f64);
            }
            let overhead = (o.wall_ms - t.prefill_ms - t.generate_ms).max(0) as f64;
            if t.load_ms > COLD_LOAD_MS { cold.push(overhead) } else { warm.push(overhead) }
        }
        let mut quality = BTreeMap::new();
        {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT capability, SUM(passed), COUNT(*) FROM quality_observations WHERE model = ?1 GROUP BY capability",
            )?;
            for row in
                stmt.query_map([model], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))?
            {
                let (cap, passed, total) = row?;
                quality.insert(cap, (passed as u32, total as u32));
            }
        }
        Ok(Observed {
            runs: total_runs as u32,
            tokens_per_sec: median(generation),
            prefill_tokens_per_sec: median(prefill),
            cold_startup_ms: percentile(&mut cold, 0.75).map(|v| v as i64),
            warm_startup_ms: median(warm).map(|v| v as i64),
            quality,
        })
    }

    pub fn record_ledger(&self, l: &LedgerRow) -> Result<()> {
        self.conn().execute(
            "INSERT OR IGNORE INTO ledger (job_id, at, kind, class, model, domain, leaves_host, tokens_in, tokens_out,
                 wall_ms, energy_wh, energy_gbp, spend_gbp, conventional_gbp, context_objects)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                l.job_id,
                l.at,
                l.kind,
                l.class.as_str(),
                l.model,
                l.domain,
                l.leaves_host,
                l.tokens_in as i64,
                l.tokens_out as i64,
                l.wall_ms,
                l.energy_wh,
                l.energy_gbp,
                l.spend_gbp,
                l.conventional_gbp,
                l.context_objects
            ],
        )?;
        Ok(())
    }

    pub fn ledger_row(&self, job: &str) -> Result<Option<LedgerRow>> {
        Ok(self.ledger_where("job_id = ?1", &[&job])?.into_iter().next())
    }

    pub fn ledger_since(&self, since: Millis) -> Result<Vec<LedgerRow>> {
        self.ledger_where("at >= ?1", &[&since])
    }

    fn ledger_where(&self, clause: &str, args: &[&dyn rusqlite::ToSql]) -> Result<Vec<LedgerRow>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!(
            "SELECT job_id, at, kind, class, model, domain, leaves_host, tokens_in, tokens_out, wall_ms, energy_wh,
                    energy_gbp, spend_gbp, conventional_gbp, context_objects FROM ledger WHERE {clause} ORDER BY at"
        ))?;
        Ok(stmt
            .query_map(args, |r| {
                Ok(LedgerRow {
                    job_id: r.get(0)?,
                    at: r.get(1)?,
                    kind: r.get(2)?,
                    class: r.get::<_, String>(3)?.parse().map_err(|e: String| {
                        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, e.into())
                    })?,
                    model: r.get(4)?,
                    domain: r.get(5)?,
                    leaves_host: r.get(6)?,
                    tokens_in: r.get::<_, i64>(7)? as u64,
                    tokens_out: r.get::<_, i64>(8)? as u64,
                    wall_ms: r.get(9)?,
                    energy_wh: r.get(10)?,
                    energy_gbp: r.get(11)?,
                    spend_gbp: r.get(12)?,
                    conventional_gbp: r.get(13)?,
                    context_objects: r.get(14)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Money spent since a time.
    pub fn spend_since(&self, since: Millis) -> Result<f64> {
        Ok(self
            .conn()
            .query_row("SELECT COALESCE(SUM(spend_gbp), 0) FROM ledger WHERE at >= ?1", [since], |r| r.get(0))?)
    }

    /// Compute time used by a class since a time.
    pub fn compute_since(&self, class: Urgency, since: Millis) -> Result<i64> {
        Ok(self.conn().query_row(
            "SELECT COALESCE(SUM(wall_ms), 0) FROM ledger WHERE class = ?1 AND at >= ?2 AND kind = 'inference'",
            params![class.as_str(), since],
            |r| r.get(0),
        )?)
    }

    /// How often this exact request was asked for before (demand log).
    pub fn prior_demand(&self, key: &str, excluding_job: &str) -> Result<i64> {
        Ok(self.conn().query_row(
            "SELECT COUNT(*) FROM demand WHERE semantic_key = ?1 AND (job_id IS NULL OR job_id != ?2)
             AND kind IN ('try_miss', 'try_hit', 'inference', 'reuse_hit')",
            params![key, excluding_job],
            |r| r.get(0),
        )?)
    }

    pub fn count_jobs_in_state_since(&self, state: &str, since: Millis) -> Result<i64> {
        Ok(self.conn().query_row(
            "SELECT COUNT(*) FROM jobs WHERE state = ?1 AND updated_at >= ?2",
            params![state, since],
            |r| r.get(0),
        )?)
    }

    pub fn demand_since(&self, kind: &str, since: Millis) -> Result<i64> {
        Ok(self.conn().query_row(
            "SELECT COUNT(*) FROM demand WHERE kind = ?1 AND at >= ?2",
            params![kind, since],
            |r| r.get(0),
        )?)
    }

    /// The conventional cost recorded for the job that produced an object.
    pub fn conventional_of_job(&self, job: &str) -> Result<Option<(f64, u64, u64)>> {
        let conn = self.conn();
        Ok(conn
            .query_row(
                "SELECT conventional_gbp, tokens_in, tokens_out FROM ledger WHERE job_id = ?1 AND kind = 'inference'",
                [job],
                |r| Ok((r.get(0)?, r.get::<_, i64>(1)? as u64, r.get::<_, i64>(2)? as u64)),
            )
            .optional()?)
    }
}

// ----------------------------------------------------------------- time ---

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// Start of the current UTC day.
pub fn day_start(now: Millis) -> Millis {
    now.div_euclid(86_400_000) * 86_400_000
}

/// Start of the current UTC calendar month.
pub fn month_start(now: Millis) -> Millis {
    let (y, m, _) = civil_from_days(now.div_euclid(86_400_000));
    days_from_civil(y, m, 1) * 86_400_000
}

// --------------------------------------------------------------- report ---

#[derive(Debug, Serialize)]
pub struct Report {
    pub since: Millis,
    pub reference: Value,
    pub jobs: Value,
    pub tokens: Value,
    pub compute: Value,
    pub money: Value,
    pub inference_avoidance_ratio: Option<f64>,
    pub budgets: Value,
    /// Preemptive Dynamism (directive §24): Preemption Yield and the fate
    /// of speculative work.
    pub preemption: Value,
    /// The same ledger per orchestrator or agent system (directive §28).
    pub by_origin: std::collections::BTreeMap<String, Value>,
}

use serde_json::{Value, json};

pub fn report(store: &Store, cfg: &EconomicsConfig, budgets: &BudgetsConfig, since: Millis) -> Result<Report> {
    let (_, open, consumed, discarded, used_ms, discarded_ms) = store.prediction_counts(since)?;
    let produced = open + consumed + discarded;
    let rows = store.ledger_since(since)?;
    let inference: Vec<&LedgerRow> = rows.iter().filter(|r| r.kind == "inference").collect();
    let reuse: Vec<&LedgerRow> = rows.iter().filter(|r| r.kind == "reuse").collect();
    let local = inference.iter().filter(|r| !r.leaves_host).count();
    let off_host = inference.len() - local;

    let mut by_domain: BTreeMap<String, (i64, usize)> = BTreeMap::new();
    for r in &inference {
        let e = by_domain.entry(r.domain.clone()).or_default();
        e.0 += r.wall_ms;
        e.1 += 1;
    }
    let energy_wh: f64 = rows.iter().map(|r| r.energy_wh).sum();
    let energy_gbp: f64 = rows.iter().map(|r| r.energy_gbp).sum();
    let spend: f64 = rows.iter().map(|r| r.spend_gbp).sum();
    let conventional: f64 = rows.iter().map(|r| r.conventional_gbp).sum();
    let operating = energy_gbp + spend;
    let avoided = (conventional - spend).max(0.0);

    let now = now_ms();
    let month_spend = store.spend_since(month_start(now))?;
    let mut daily = serde_json::Map::new();
    for (class, limit) in budgets.daily_limits()? {
        let used = store.compute_since(class, day_start(now))?;
        daily.insert(class.as_str().into(), json!({"used_ms": used, "limit_ms": limit}));
    }

    Ok(Report {
        since,
        reference: json!({
            "name": cfg.reference_name,
            "input_gbp_per_mtok": cfg.reference_input_gbp_per_mtok,
            "output_gbp_per_mtok": cfg.reference_output_gbp_per_mtok,
            "electricity_gbp_per_kwh": cfg.electricity_gbp_per_kwh,
            "watts": {"local": cfg.local_watts, "node": cfg.node_watts, "gpu": cfg.gpu_watts},
        }),
        jobs: json!({
            "inference": inference.len(),
            "inference_local": local,
            "inference_off_host": off_host,
            "served_from_store": reuse.len(),
            "with_context": inference.iter().filter(|r| r.context_objects > 0).count(),
            "try_hits": store.demand_since("try_hit", since)?,
            "try_misses": store.demand_since("try_miss", since)?,
            "deadline_misses": store.count_jobs_in_state_since("DEADLINE_EXCEEDED", since)?,
            "insufficient_quality": store.count_jobs_in_state_since("INSUFFICIENT_QUALITY", since)?,
            "remote_inference_avoided": local + reuse.len(),
        }),
        tokens: json!({
            "tokens_in": inference.iter().map(|r| r.tokens_in).sum::<u64>(),
            "tokens_out": inference.iter().map(|r| r.tokens_out).sum::<u64>(),
        }),
        compute: json!({
            "hours_by_domain": by_domain.iter().map(|(d, (ms, n))| (d.clone(), json!({"hours": *ms as f64 / 3.6e6, "jobs": n}))).collect::<serde_json::Map<_, _>>(),
            "energy_wh": energy_wh,
        }),
        money: json!({
            "conventional_gbp": conventional,
            "remote_spend_gbp": spend,
            "energy_gbp": energy_gbp,
            "operating_gbp": operating,
            "estimated_cost_avoided_gbp": avoided,
        }),
        inference_avoidance_ratio: (operating > 0.0).then(|| avoided / operating),
        budgets: json!({
            "remote_monthly": {"spent_gbp": month_spend, "limit_gbp": budgets.remote_monthly_gbp},
            "daily_compute": daily,
            "memory_max_mb": budgets.memory_max_mb,
        }),
        preemption: json!({
            "produced": produced,
            "pd_hits": consumed,
            "discarded": discarded,
            "outstanding": open,
            "preemption_yield": (produced > 0).then(|| consumed as f64 / produced as f64),
            "speculative_used_hours": used_ms as f64 / 3.6e6,
            "speculative_discarded_hours": discarded_ms as f64 / 3.6e6,
        }),
        by_origin: store.ledger_by_origin(since)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_boundaries() {
        // 2026-10-13 08:00:00Z
        let t = 1_791_878_400_000;
        assert_eq!(day_start(t), 1_791_849_600_000);
        assert_eq!(month_start(t), 1_790_812_800_000); // 2026-10-01
        assert_eq!(month_start(1_790_812_800_000), 1_790_812_800_000);
    }

    #[test]
    fn measurements_replace_priors_after_enough_runs() {
        let store = Store::open_in_memory().unwrap();
        let obs = |job: &str, load: i64, wall: i64| ObservationRow {
            at: 1,
            job_id: job.into(),
            domain: "local".into(),
            tokens_in: 100,
            tokens_out: 50,
            wall_ms: wall,
            timings: Some(Timings { load_ms: load, prefill_ms: 1000, generate_ms: 5000 }),
        };
        store.record_observation("m/a", &obs("1", 20_000, 26_000)).unwrap();
        store.record_observation("m/a", &obs("2", 10, 6_200)).unwrap();
        let o = store.observed("m/a").unwrap();
        let mut p = crate::runtimes::test_profile("m/a");
        o.apply(&mut p);
        assert_eq!(p.source, ProfileSource::StaticPrior, "two runs are not enough");
        store.record_observation("m/a", &obs("3", 10, 6_300)).unwrap();
        let o = store.observed("m/a").unwrap();
        o.apply(&mut p);
        assert_eq!(p.source, ProfileSource::Observed);
        assert_eq!(p.performance.tokens_per_sec, 10.0);
        assert_eq!(p.performance.prefill_tokens_per_sec, 100.0);
        assert_eq!(p.performance.startup_ms, 20_000, "cold overhead");
        assert_eq!(p.performance.warm_startup_ms, 250, "median warm overhead");
    }

    #[test]
    fn benchmark_results_shrink_toward_the_prior() {
        let store = Store::open_in_memory().unwrap();
        for (i, ok) in [true, false, false, false].into_iter().enumerate() {
            store.record_quality("m/a", "general", &format!("c{i}"), ok, "run", "job").unwrap();
        }
        let mut p = crate::runtimes::test_profile("m/a");
        let prior = p.capabilities["general"];
        store.observed("m/a").unwrap().apply(&mut p);
        let q = p.capabilities["general"];
        assert!((q - (prior * 4.0 + 1.0) / 8.0).abs() < 1e-9);
        assert_eq!(p.evidence.quality["general"].total, 4);
    }
}
