//! Refinement (directive §14, Phase 5): draft, check, validate, revise,
//! escalate — inside the job's deadline, budget and locality.
//!
//! Every pass is an ordinary inference: it is planned with the model pinned
//! (so locality, health, resources, budget and deadline all apply), waits
//! for a slot in its domain, is recorded in the boundary audit, and is
//! measured like any run. Confidence moves only by checks (a failed check
//! proves an answer wrong) and by validator verdicts, weighted by each
//! validator's measured accuracy (`tonyc_core::refine`). If the floor or the
//! verification requirement cannot be met within the job's means, the job
//! ends INSUFFICIENT_QUALITY with its evidence and best attempt.

use crate::daemon::{Daemon, StopReason, domain_of};
use crate::economics::{self, ObservationRow};
use crate::host;
use crate::runtimes::AdapterError;
use crate::store::{Store, now_ms};
use anyhow::Result;
use rusqlite::params;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonyc_core::envelope::{OutputSchema, Refine, Verification, WorkEnvelope};
use tonyc_core::model::estimate_tokens;
use tonyc_core::plan::PlanInput;
use tonyc_core::refine::{ValidatorStats, Verdict, check_all, if_all_pass, parse_judgement, update};
use tonyc_core::{Candidate, Exposure, InferenceRequest, JobState, LocalityPolicy, Millis, plan};

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS passes (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    at         INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    role       TEXT NOT NULL,
    model      TEXT NOT NULL,
    domain     TEXT NOT NULL,
    verdict    TEXT,
    confidence REAL NOT NULL,
    detail     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passes_job ON passes (job_id, seq);
CREATE TRIGGER IF NOT EXISTS passes_no_update BEFORE UPDATE ON passes
    BEGIN SELECT RAISE(ABORT, 'passes are append-only'); END;
CREATE TRIGGER IF NOT EXISTS passes_no_delete BEFORE DELETE ON passes
    BEGIN SELECT RAISE(ABORT, 'passes are append-only'); END;
"#;

#[derive(Debug, Clone, Serialize)]
pub struct PassRow {
    pub seq: i64,
    pub at: Millis,
    pub started_at: Millis,
    /// `draft`, `check`, `validate`, `tie_break`, `revise`.
    pub role: String,
    pub model: String,
    pub domain: String,
    pub verdict: Option<String>,
    /// Confidence after this pass.
    pub confidence: f64,
    pub detail: Value,
}

impl Store {
    #[allow(clippy::too_many_arguments)]
    pub fn record_pass(
        &self,
        job: &str,
        started_at: Millis,
        role: &str,
        model: &str,
        domain: &str,
        verdict: Option<Verdict>,
        confidence: f64,
        detail: &Value,
    ) -> Result<()> {
        let verdict = verdict.map(|v| serde_json::to_value(v).unwrap().as_str().unwrap().to_string());
        self.conn().execute(
            "INSERT INTO passes (job_id, at, started_at, role, model, domain, verdict, confidence, detail)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![job, now_ms(), started_at, role, model, domain, verdict, confidence, detail.to_string()],
        )?;
        Ok(())
    }

    pub fn passes(&self, job: &str) -> Result<Vec<PassRow>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT seq, at, started_at, role, model, domain, verdict, confidence, detail FROM passes
             WHERE job_id = ?1 ORDER BY seq",
        )?;
        Ok(stmt
            .query_map([job], |r| {
                Ok(PassRow {
                    seq: r.get(0)?,
                    at: r.get(1)?,
                    started_at: r.get(2)?,
                    role: r.get(3)?,
                    model: r.get(4)?,
                    domain: r.get(5)?,
                    verdict: r.get(6)?,
                    confidence: r.get(7)?,
                    detail: crate::store::json_of(r.get(8)?)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
}

/// Accumulated cost of a job's passes, for the ledger.
#[derive(Debug, Clone, Default)]
pub struct Totals {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub wall_ms: i64,
    pub energy_wh: f64,
    pub spend_gbp: f64,
    pub domains: BTreeSet<String>,
}

impl Totals {
    pub fn add(&mut self, o: &PassOut) {
        self.tokens_in += o.tokens_in;
        self.tokens_out += o.tokens_out;
        self.wall_ms += o.wall_ms;
        self.energy_wh += o.energy_wh;
        self.spend_gbp += o.spend_gbp;
        self.domains.insert(o.domain.clone());
    }

    pub fn domain_label(&self) -> String {
        self.domains.iter().cloned().collect::<Vec<_>>().join("+")
    }
}

#[derive(Debug, Clone)]
pub struct PassOut {
    pub text: String,
    pub model: String,
    pub domain: String,
    pub exposure: Exposure,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub wall_ms: i64,
    pub energy_wh: f64,
    pub spend_gbp: f64,
    pub started_at: Millis,
}

#[derive(Debug)]
pub enum PassErr {
    /// The model cannot take this pass now (policy, health, resources,
    /// budget, deadline): refinement stops, it does not fail the job.
    Unavailable(String),
    Stopped(StopReason),
    Deadline,
    Failed(String),
}

/// Energy and spend of a run.
pub fn costs(d: &Daemon, domain: &str, runtime: &str, wall_ms: i64, tokens: u64, cost_per_mtok: f64) -> (f64, f64) {
    let gpu = match domain.strip_prefix("node:") {
        Some(name) => d.node(name).and_then(|n| n.caps()).is_some_and(|c| !c.gpus.is_empty()),
        None => runtime != "builtin" && !d.host.gpus.is_empty(),
    };
    let energy = economics::energy_wh(&d.cfg.economics, domain, gpu, wall_ms);
    (energy, tokens as f64 / 1e6 * cost_per_mtok)
}

/// Candidates for planning, annotated with pressure and resource fit.
pub fn candidates(d: &Daemon) -> Vec<Candidate> {
    let mut c = d.registry.candidates_with(|domain| d.has_free_slot(domain));
    crate::scheduler::annotate(d, &mut c, &host::sample(&d.cfg.pressure, d.host.cores));
    c
}

/// An envelope for a pass on exactly `model`, under `privacy`.
fn pinned(env: &WorkEnvelope, privacy: &LocalityPolicy, model: &str, max_tokens: u32) -> WorkEnvelope {
    let mut e = env.clone();
    e.task.pin_model = Some(model.to_string());
    e.task.max_tokens = max_tokens;
    e.quality.floor = 0.0;
    e.quality.verification = Verification::None;
    e.privacy = privacy.clone();
    e
}

/// Whether `model` could take a pass for this job right now.
fn admissible(
    d: &Daemon,
    env: &WorkEnvelope,
    privacy: &LocalityPolicy,
    cands: &[Candidate],
    model: &str,
    prompt_tokens: u64,
) -> Result<(), String> {
    let e = pinned(env, privacy, model, 128);
    let budget = d.cfg.budgets.remote_monthly_gbp
        - d.store.spend_since(economics::month_start(now_ms())).map_err(|e| e.to_string())?;
    plan(&PlanInput {
        envelope: &e,
        candidates: cands,
        ceiling: &d.ceiling(),
        now: now_ms(),
        prompt_tokens,
        validator_available: false,
        budget_left_gbp: Some(budget.max(0.0)),
        reuse_potential: 0.0,
        future_value: 0.0,
    })
    .map(|_| ())
    .map_err(|f| f.reason)
}

/// Informative validators this job may use, best first.
pub fn validators(d: &Daemon, env: &WorkEnvelope, privacy: &LocalityPolicy, prompt_tokens: u64) -> Vec<ValidatorStats> {
    let cands = candidates(d);
    let mut out: Vec<ValidatorStats> = cands
        .iter()
        .filter(|c| c.healthy && c.blocked.is_none())
        .map(|c| ValidatorStats::of(&c.profile))
        .filter(|v| v.informative())
        .filter(|v| admissible(d, env, privacy, &cands, &v.model, prompt_tokens).is_ok())
        .collect();
    out.sort_by(|a, b| b.accuracy().total_cmp(&a.accuracy()).then(a.model.cmp(&b.model)));
    out
}

/// Models that may revise an answer, by expected quality for the
/// capability, best first.
pub fn ladder(d: &Daemon, env: &WorkEnvelope, privacy: &LocalityPolicy, prompt_tokens: u64) -> Vec<(String, f64)> {
    let cands = candidates(d);
    let mut out: Vec<(String, f64)> = cands
        .iter()
        .filter(|c| c.healthy && c.blocked.is_none())
        .filter_map(|c| c.profile.quality_for(&env.task.capability).map(|(q, _)| (c.profile.id.clone(), q)))
        .filter(|(m, _)| admissible(d, env, privacy, &cands, m, prompt_tokens).is_ok())
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// One pass to run: which model, and what to ask it.
pub struct PassRequest<'a> {
    pub model: &'a str,
    pub prompt: String,
    pub max_tokens: u32,
    pub json: bool,
}

/// Runs one pass on a pinned model through the normal machinery.
pub async fn run_pass(
    d: &Arc<Daemon>,
    env: &WorkEnvelope,
    privacy: &LocalityPolicy,
    pass: PassRequest<'_>,
    token: &CancellationToken,
) -> Result<PassOut, PassErr> {
    let id = env.id.as_str();
    let PassRequest { model, prompt, max_tokens, json: json_out } = pass;
    let e = pinned(env, privacy, model, max_tokens);
    let cands = candidates(d);
    let budget = d.cfg.budgets.remote_monthly_gbp
        - d.store.spend_since(economics::month_start(now_ms())).map_err(|e| PassErr::Failed(e.to_string()))?;
    let decision = plan(&PlanInput {
        envelope: &e,
        candidates: &cands,
        ceiling: &d.ceiling(),
        now: now_ms(),
        prompt_tokens: estimate_tokens(prompt.len()),
        validator_available: false,
        budget_left_gbp: Some(budget.max(0.0)),
        reuse_potential: 0.0,
        future_value: 0.0,
    })
    .map_err(|f| PassErr::Unavailable(f.reason))?;
    let (adapter, local) = d
        .registry
        .resolve(&decision.model)
        .ok_or_else(|| PassErr::Unavailable(format!("no adapter serves {model}")))?;
    let exposure = adapter.exposure();
    privacy.permits(&exposure).and_then(|_| d.ceiling().permits(&exposure)).map_err(PassErr::Unavailable)?;
    let domain = adapter.domain();
    let _permit = tokio::select! {
        p = d.slot(&domain).acquire_owned() => p.expect("slot semaphores are never closed"),
        _ = token.cancelled() => return Err(PassErr::Stopped(d.stop_reason(id).unwrap_or(StopReason::User))),
        _ = until(env.latency.deadline_at) => return Err(PassErr::Deadline),
    };
    let req = InferenceRequest { model: local, prompt, max_tokens, json: json_out };
    let wire = adapter.encode(&req);
    let rec = |sent: u64, received: u64, summary: String, payload: Option<&[u8]>| {
        d.store
            .record_boundary(id, &exposure, sent, received, &summary, payload)
            .map_err(|e| PassErr::Failed(e.to_string()))
    };
    rec(wire.len() as u64, 0, format!("refinement pass dispatched to {model}"), Some(&wire))?;
    let started = now_ms();
    let out = tokio::select! {
        r = adapter.run(&req) => r,
        _ = token.cancelled() => return Err(PassErr::Stopped(d.stop_reason(id).unwrap_or(StopReason::User))),
        _ = until(env.latency.deadline_at) => return Err(PassErr::Deadline),
    };
    let out = match out {
        Ok(o) => o,
        Err(AdapterError::NodeUnreachable(e)) | Err(AdapterError::Unreachable(e)) => {
            rec(0, 0, format!("refinement pass to {model} failed: {e}"), None)?;
            return Err(PassErr::Unavailable(e));
        }
        Err(AdapterError::Failed(e)) => return Err(PassErr::Failed(e)),
    };
    rec(0, out.bytes_received, format!("refinement response from {model}"), None)?;
    let wall_ms = now_ms() - started;
    let fail = |e: anyhow::Error| PassErr::Failed(e.to_string());
    d.store
        .record_observation(
            model,
            &ObservationRow {
                at: now_ms(),
                job_id: id.to_string(),
                domain: domain.clone(),
                tokens_in: out.tokens_in,
                tokens_out: out.tokens_out,
                wall_ms,
                timings: out.timings,
            },
        )
        .map_err(fail)?;
    d.refresh_observed(model).map_err(fail)?;
    let cost =
        cands.iter().find(|c| c.profile.id == model).map(|c| c.profile.economics.cost_per_mtok_gbp).unwrap_or(0.0);
    let (energy_wh, spend_gbp) = costs(d, &domain, adapter.runtime(), wall_ms, out.tokens_in + out.tokens_out, cost);
    Ok(PassOut {
        text: out.text,
        model: model.to_string(),
        domain: domain_of(adapter.runtime()),
        exposure,
        tokens_in: out.tokens_in,
        tokens_out: out.tokens_out,
        wall_ms,
        energy_wh,
        spend_gbp,
        started_at: started,
    })
}

async fn until(deadline: Option<Millis>) {
    match deadline {
        Some(at) => tokio::time::sleep(Duration::from_millis((at - now_ms()).max(0) as u64)).await,
        None => std::future::pending().await,
    }
}

// --------------------------------------------------------------- prompts ---

fn validate_prompt(env: &WorkEnvelope, inputs: &str, answer: &str) -> String {
    judge_prompt(&env.task.objective, inputs, answer)
}

/// What a validator is asked. Also used by benchmark judge cases, so that
/// a validator is measured on exactly the prompt it is trusted with.
pub fn judge_prompt(objective: &str, inputs: &str, answer: &str) -> String {
    format!(
        "You are checking an answer for correctness. Be strict.\n\nTask: {}\n{}\nProposed answer:\n{}\n\n\
         Reply with JSON only: {{\"verdict\": \"pass\" or \"fail\", \"issues\": [\"...\"]}}. \
         Use \"fail\" if anything is wrong, missing, or not supported by the inputs.",
        objective,
        if inputs.trim().is_empty() { String::new() } else { format!("Inputs:\n{inputs}\n") },
        answer
    )
}

fn revise_prompt(env: &WorkEnvelope, inputs: &str, answer: &str, issues: &[String]) -> String {
    let list: String = issues.iter().map(|i| format!("- {i}\n")).collect();
    format!(
        "{}\n{}\nA previous answer was judged incorrect. Problems found:\n{}\nPrevious answer:\n{}\n\n\
         Write a corrected answer to the task. Give only the answer.{}",
        env.task.objective,
        inputs,
        if list.is_empty() { "- (unspecified)\n".into() } else { list },
        answer,
        if env.output.schema == OutputSchema::Json {
            " Respond with a single JSON value and nothing else."
        } else {
            ""
        }
    )
}

// ------------------------------------------------------------ the loop ---

/// What refinement concluded.
#[derive(Debug)]
pub struct Outcome {
    pub text: String,
    pub model: String,
    pub exposure: Exposure,
    pub confidence: f64,
    pub verified: bool,
    pub satisfied: bool,
    pub reason: String,
    pub rounds: u32,
}

pub enum Ended {
    Done(Outcome),
    Stopped(StopReason),
    Deadline,
}

/// The draft, as produced by the job's first dispatch.
pub struct Draft {
    pub text: String,
    pub model: String,
    pub exposure: Exposure,
    pub quality: f64,
    pub started_at: Millis,
    pub domain: String,
}

/// Whether this job's answer goes through the refinement loop at all.
/// Checks are always enforced — with `refine: never` a failing answer is
/// simply not revised.
pub fn wanted(env: &WorkEnvelope, pipeline: bool) -> bool {
    !env.quality.checks.is_empty()
        || (env.quality.refine != Refine::Never && (pipeline || env.output.schema == OutputSchema::Json))
}

fn schema_problems(env: &WorkEnvelope, answer: &str) -> Vec<String> {
    if env.output.schema == OutputSchema::Json && serde_json::from_str::<Value>(answer).is_err() {
        vec!["is not valid JSON (json/v1)".into()]
    } else {
        vec![]
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn refine(
    d: &Arc<Daemon>,
    env: &WorkEnvelope,
    privacy: &LocalityPolicy,
    inputs: &str,
    draft: Draft,
    validators: &[ValidatorStats],
    ladder: &[(String, f64)],
    totals: &mut Totals,
    token: &CancellationToken,
) -> Result<Ended> {
    let id = env.id.as_str();
    let floor = env.quality.floor;
    let mut p = draft.quality;
    let mut answer = draft.text;
    let mut answerer = draft.model.clone();
    let mut exposure = draft.exposure;
    let mut tried: BTreeSet<String> = BTreeSet::from([answerer.clone()]);
    let mut rounds = 0u32;
    d.store.record_pass(id, draft.started_at, "draft", &answerer, &draft.domain, None, p, &json!({"quality": p}))?;
    let mut best: Option<(f64, bool, String, String, Exposure)> = None;
    let stop_reason: String;

    macro_rules! pass {
        ($model:expr, $prompt:expr, $max:expr, $json:expr) => {
            match run_pass(
                d,
                env,
                privacy,
                PassRequest { model: $model, prompt: $prompt, max_tokens: $max, json: $json },
                token,
            )
            .await
            {
                Ok(o) => {
                    totals.add(&o);
                    Ok(o)
                }
                Err(PassErr::Stopped(r)) => return Ok(Ended::Stopped(r)),
                Err(PassErr::Deadline) => return Ok(Ended::Deadline),
                Err(e) => Err(e),
            }
        };
    }

    loop {
        // Deterministic checks first: a failure proves the answer wrong.
        let mut problems = check_all(&env.quality.checks, &answer);
        problems.extend(schema_problems(env, &answer));
        let mut issues = problems.clone();
        let (mut passes, mut fails) = (0usize, 0usize);
        if !problems.is_empty() {
            p = 0.0;
            d.store.record_pass(
                id,
                now_ms(),
                "check",
                "checks",
                "local",
                Some(Verdict::Fail),
                p,
                &json!({"problems": problems}),
            )?;
        } else {
            if !env.quality.checks.is_empty() {
                d.store.record_pass(
                    id,
                    now_ms(),
                    "check",
                    "checks",
                    "local",
                    Some(Verdict::Pass),
                    p,
                    &json!({"checks": env.quality.checks.len()}),
                )?;
            }
            // Independent validators: never the answer's own model.
            let pool: Vec<&ValidatorStats> = validators.iter().filter(|v| v.model != answerer).collect();
            let mut used = 0usize;
            for (i, v) in pool.iter().enumerate() {
                let tie_break = used >= 2;
                if tie_break && !(passes > 0 && fails > 0 && passes == fails) {
                    break;
                }
                let prompt = validate_prompt(env, inputs, &answer);
                let o = match pass!(&v.model, prompt, 160, true) {
                    Ok(o) => o,
                    Err(e) => {
                        let why = format!("{e:?}");
                        d.store.record_pass(
                            id,
                            now_ms(),
                            "validate",
                            &v.model,
                            "-",
                            Some(Verdict::Abstain),
                            p,
                            &json!({"unavailable": why}),
                        )?;
                        continue;
                    }
                };
                used += 1;
                let j = parse_judgement(&o.text);
                p = update(p, j.verdict, v);
                match j.verdict {
                    Verdict::Pass => passes += 1,
                    Verdict::Fail => fails += 1,
                    Verdict::Abstain => {}
                }
                issues.extend(j.issues.iter().cloned());
                let role = if tie_break { "tie_break" } else { "validate" };
                d.store.record_pass(
                    id,
                    o.started_at,
                    role,
                    &v.model,
                    &o.domain,
                    Some(j.verdict),
                    p,
                    &json!({"issues": j.issues, "sensitivity": v.sensitivity, "specificity": v.specificity, "measured": v.measured}),
                )?;
                d.store.note(
                    id,
                    "pass",
                    json!({"role": role, "model": v.model, "verdict": j.verdict, "confidence": p, "n": i}),
                )?;
            }
        }
        let checks_define = !env.quality.checks.is_empty();
        let verified =
            problems.is_empty() && ((passes > fails && passes > 0) || (checks_define && passes == 0 && fails == 0));
        let satisfied =
            problems.is_empty() && p >= floor && (env.quality.verification != Verification::Required || verified);
        if best.as_ref().is_none_or(|(bp, ..)| p > *bp) {
            best = Some((p, verified, answer.clone(), answerer.clone(), exposure.clone()));
        }
        if satisfied {
            return Ok(Ended::Done(Outcome {
                text: answer,
                model: answerer,
                exposure,
                confidence: p,
                verified,
                satisfied: true,
                reason: format!("confidence {p:.2} ≥ floor {floor:.2}{}", if verified { ", verified" } else { "" }),
                rounds,
            }));
        }
        if env.quality.refine == Refine::Never {
            stop_reason = "refinement disabled for this job".into();
            break;
        }
        if rounds >= env.quality.max_rounds {
            stop_reason = format!("{rounds} revision(s), the most this job allows");
            break;
        }
        // Revise: escalate to the best model not yet tried that is expected
        // to do better; else the best not yet tried; else retry the same.
        let current_q = ladder.iter().find(|(m, _)| *m == answerer).map(|(_, q)| *q).unwrap_or(0.0);
        let reviser = ladder
            .iter()
            .find(|(m, q)| !tried.contains(m) && *q > current_q)
            .or_else(|| ladder.iter().find(|(m, _)| !tried.contains(m)))
            .or_else(|| ladder.iter().find(|(m, _)| *m == answerer))
            .cloned();
        let Some((reviser, rq)) = reviser else {
            stop_reason = "no model is available to revise the answer".into();
            break;
        };
        if issues.is_empty() {
            issues.push(format!("confidence {p:.2} is below the floor {floor:.2}"));
        }
        let prompt = revise_prompt(env, inputs, &answer, &issues);
        let o = match pass!(&reviser, prompt, env.task.max_tokens, env.output.schema == OutputSchema::Json) {
            Ok(o) => o,
            Err(e) => {
                stop_reason = format!("revision on {reviser} not possible: {e:?}");
                break;
            }
        };
        rounds += 1;
        tried.insert(reviser.clone());
        answer = o.text;
        answerer = reviser.clone();
        exposure = o.exposure.clone();
        p = rq;
        let escalated = rq > current_q;
        d.store.record_pass(
            id,
            o.started_at,
            "revise",
            &reviser,
            &o.domain,
            None,
            p,
            &json!({"issues": issues, "escalated": escalated, "round": rounds}),
        )?;
        d.store.note(
            id,
            "pass",
            json!({"role": "revise", "model": reviser, "escalated": escalated, "confidence": p}),
        )?;
    }
    let (bp, bv, btext, bmodel, bexp) = best.expect("at least the draft was assessed");
    Ok(Ended::Done(Outcome {
        text: btext,
        model: bmodel,
        exposure: bexp,
        confidence: bp,
        verified: bv,
        satisfied: false,
        reason: format!(
            "best confidence {bp:.2} below floor {floor:.2}{}; stopped: {stop_reason}",
            if env.quality.verification == Verification::Required && !bv { " or unverified" } else { "" }
        ),
        rounds,
    }))
}

/// How a job will be produced.
#[derive(Debug, Clone, Serialize)]
pub struct Strategy {
    pub pipeline: bool,
    pub validators: Vec<ValidatorStats>,
    pub ladder: Vec<(String, f64)>,
    pub explain: Vec<String>,
}

/// Decides between a single pass and a refinement pipeline, before any
/// inference. `single` is the plain plan's verdict for this job.
pub fn strategy(
    d: &Daemon,
    env: &WorkEnvelope,
    privacy: &LocalityPolicy,
    prompt_tokens: u64,
    single_ok: bool,
    single_quality_shortfall: bool,
) -> std::result::Result<Strategy, (JobState, String)> {
    let validators = validators(d, env, privacy, prompt_tokens);
    let ladder = ladder(d, env, privacy, prompt_tokens);
    let mut explain = Vec::new();
    let need_verify = env.quality.verification == Verification::Required;
    let always = env.quality.refine == Refine::Always;
    let pipeline =
        env.quality.refine != Refine::Never && (need_verify || always || (!single_ok && single_quality_shortfall));
    if !pipeline {
        return Ok(Strategy { pipeline: false, validators, ladder, explain });
    }
    if need_verify && validators.is_empty() && env.quality.checks.is_empty() {
        return Err((
            JobState::InsufficientQuality,
            "verification required, but no informative validator may take this job and it has no checks".into(),
        ));
    }
    // Could refinement reach the floor at all? The best case: the strongest
    // admissible answerer, passed by the two best validators other than it.
    let best = ladder.first().cloned();
    if let Some((model, q)) = &best {
        let vs: Vec<ValidatorStats> = validators.iter().filter(|v| &v.model != model).take(2).cloned().collect();
        let reach = if_all_pass(*q, &vs);
        explain.push(format!(
            "pipeline: best answerer {model} ({q:.2}), {} validator(s) → up to {reach:.2} if they pass",
            vs.len()
        ));
        // Checks can only rule an answer out, never raise confidence: they
        // do not change what is reachable.
        if reach < env.quality.floor {
            return Err((
                JobState::InsufficientQuality,
                format!(
                    "even with {} validator(s) passing {model}'s answer, confidence would reach only {reach:.2} < floor {:.2}",
                    vs.len(),
                    env.quality.floor
                ),
            ));
        }
    } else {
        return Err((JobState::InsufficientQuality, "no admissible model can answer this job".into()));
    }
    for v in validators.iter().take(3) {
        explain.push(format!(
            "validator {}: sensitivity {:.2}, specificity {:.2} ({})",
            v.model,
            v.sensitivity,
            v.specificity,
            if v.measured { "measured" } else { "assumed from general quality" }
        ));
    }
    Ok(Strategy { pipeline: true, validators, ladder, explain })
}
