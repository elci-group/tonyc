//! The planner (directive §13): envelope + candidates → decision.
//!
//! Hard constraints are applied first and in a fixed order, each stage
//! eliminating candidates with a recorded reason:
//!
//! 0. pin — a job pinned to a model (benchmarks) considers only it;
//! 1. locality — the job's policy and the daemon ceiling must both permit
//!    the candidate's exposure (never weighed against anything);
//! 2. health — the execution domain must be reachable;
//! 3. resources — the model must fit (memory) where it would run;
//! 4. pressure — work that yields does not run on a pressured domain;
//!    (then, softly: domains with a free slot are preferred over busy ones)
//! 5. verification — `required` needs a validator;
//! 6. quality — expected quality must meet the floor;
//! 7. budget — estimated spend must fit the remaining budget;
//! 8. deadline — predicted completion must fit before the deadline.
//!
//! If a stage eliminates every remaining candidate, planning fails with the
//! state that stage implies (RESOURCE_BLOCKED means "wait", not "fail").
//! Survivors are ranked by the utility `U = Q + R + V − C − L − E`. R grows
//! with the result's reuse potential and with quality: an answer likely to
//! be asked for again is worth a better model. V does the same for
//! speculative work, by the probability that the prepared answer will be
//! asked for. The privacy term P never appears because privacy is a filter,
//! not a weight.

use crate::Millis;
use crate::envelope::{Energy, Urgency, Verification, WorkEnvelope};
use crate::locality::{Exposure, LocalityPolicy};
use crate::model::ModelProfile;
use crate::state::JobState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Candidate {
    pub profile: ModelProfile,
    pub exposure: Exposure,
    pub healthy: bool,
    /// Whether the candidate's execution domain has a free slot now.
    pub available: bool,
    /// Whether its domain is under pressure (busy host or node); work that
    /// yields to pressure does not run there.
    pub pressured: bool,
    /// Why the model cannot run there now, if it cannot (e.g. memory).
    pub blocked: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rejection {
    pub model: String,
    pub stage: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtilityTerms {
    pub quality: f64,
    pub reuse: f64,
    pub future_value: f64,
    pub cost: f64,
    pub latency: f64,
    pub energy: f64,
    pub total: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub model: String,
    pub runtime: String,
    pub exposure: Exposure,
    pub estimated_quality: f64,
    pub quality_basis: String,
    pub predicted_runtime_ms: i64,
    pub deadline_margin_ms: Option<i64>,
    pub estimated_cost_gbp: f64,
    pub estimated_energy_wh: f64,
    pub utility: UtilityTerms,
    pub reasons: Vec<String>,
    pub considered: usize,
    pub rejected: Vec<Rejection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[error("{state}: {reason}")]
pub struct PlanFailure {
    pub state: JobState,
    pub reason: String,
    pub considered: usize,
    pub rejected: Vec<Rejection>,
}

pub struct PlanInput<'a> {
    pub envelope: &'a WorkEnvelope,
    pub candidates: &'a [Candidate],
    pub ceiling: &'a LocalityPolicy,
    pub now: Millis,
    pub prompt_tokens: u64,
    pub validator_available: bool,
    /// Monetary budget left for this plan; `None` is unlimited.
    pub budget_left_gbp: Option<f64>,
    /// How likely the result is to be reused (0–1), from demand history.
    pub reuse_potential: f64,
    /// For speculative work: the probability the prepared answer will be
    /// asked for (0–1), from Preemptive Dynamism's prediction.
    pub future_value: f64,
}

struct Survivor<'a> {
    cand: &'a Candidate,
    quality: f64,
    basis: String,
    runtime_ms: i64,
}

pub fn plan(input: &PlanInput) -> Result<Decision, PlanFailure> {
    let env = input.envelope;
    let mut rejected = Vec::new();
    let mut cands: Vec<&Candidate> = input.candidates.iter().collect();
    cands.sort_by(|a, b| a.profile.id.cmp(&b.profile.id));
    let considered = cands.len();
    let fail = |state, reason: String, rejected: Vec<Rejection>| PlanFailure { state, reason, considered, rejected };

    if cands.is_empty() {
        return Err(fail(JobState::ModelFailed, "no models are registered with any enabled runtime".into(), rejected));
    }

    // 0. Pin.
    if let Some(pin) = &env.task.pin_model {
        cands.retain(|c| {
            if &c.profile.id != pin {
                rejected.push(Rejection {
                    model: c.profile.id.clone(),
                    stage: "pin".into(),
                    reason: format!("job is pinned to {pin}"),
                });
            }
            &c.profile.id == pin
        });
        if cands.is_empty() {
            return Err(fail(JobState::ModelFailed, format!("pinned model {pin} is not available"), rejected));
        }
    }

    // 1. Locality: job policy and daemon ceiling, both hard.
    cands.retain(|c| {
        let verdict = env
            .privacy
            .permits(&c.exposure)
            .map_err(|r| format!("job policy: {r}"))
            .and_then(|_| input.ceiling.permits(&c.exposure).map_err(|r| format!("daemon ceiling: {r}")));
        match verdict {
            Ok(()) => true,
            Err(reason) => {
                rejected.push(Rejection { model: c.profile.id.clone(), stage: "locality".into(), reason });
                false
            }
        }
    });
    if cands.is_empty() {
        return Err(fail(
            JobState::PolicyBlocked,
            format!("no execution domain satisfies locality {}", env.privacy.class),
            rejected,
        ));
    }

    // 2. Health.
    cands.retain(|c| {
        if !c.healthy {
            rejected.push(Rejection {
                model: c.profile.id.clone(),
                stage: "health".into(),
                reason: format!("runtime {} is unreachable", c.profile.runtime),
            });
        }
        c.healthy
    });
    if cands.is_empty() {
        return Err(fail(JobState::ModelFailed, "every admissible runtime is unreachable".into(), rejected));
    }

    // 3. Resources.
    cands.retain(|c| match &c.blocked {
        Some(reason) => {
            rejected.push(Rejection { model: c.profile.id.clone(), stage: "resources".into(), reason: reason.clone() });
            false
        }
        None => true,
    });
    if cands.is_empty() {
        return Err(fail(
            JobState::ResourceBlocked,
            "no admissible model fits the resources available now".into(),
            rejected,
        ));
    }

    // 4. Pressure, for work that yields to it.
    if env.yields_to_pressure() {
        cands.retain(|c| {
            if c.pressured {
                rejected.push(Rejection {
                    model: c.profile.id.clone(),
                    stage: "pressure".into(),
                    reason: "its domain is under pressure and this work yields".into(),
                });
            }
            !c.pressured
        });
        if cands.is_empty() {
            return Err(fail(JobState::ResourceBlocked, "every admissible domain is under pressure".into(), rejected));
        }
    }

    // Capacity is a preference, not a constraint: busy domains are dropped
    // only if some admissible domain is free; otherwise the job waits.
    if cands.iter().any(|c| c.available) {
        cands.retain(|c| {
            if !c.available {
                rejected.push(Rejection {
                    model: c.profile.id.clone(),
                    stage: "capacity".into(),
                    reason: "its execution domain has no free slot".into(),
                });
            }
            c.available
        });
    }

    // 3. Verification.
    if env.quality.verification == Verification::Required && !input.validator_available {
        for c in &cands {
            rejected.push(Rejection {
                model: c.profile.id.clone(),
                stage: "verification".into(),
                reason: "verification required but no validator can take this job".into(),
            });
        }
        return Err(fail(
            JobState::InsufficientQuality,
            "verification: required cannot be met; no validator can take this job".into(),
            rejected,
        ));
    }

    // 4. Quality floor.
    let mut survivors: Vec<Survivor> = Vec::new();
    for c in cands {
        match c.profile.quality_for(&env.task.capability) {
            Some((q, basis)) if q >= env.quality.floor => survivors.push(Survivor {
                cand: c,
                quality: q,
                basis: basis.to_string(),
                runtime_ms: c.profile.predict_runtime_ms(input.prompt_tokens, env.task.max_tokens),
            }),
            Some((q, basis)) => rejected.push(Rejection {
                model: c.profile.id.clone(),
                stage: "quality".into(),
                reason: format!("expected quality {q:.2} ({basis}) is below floor {:.2}", env.quality.floor),
            }),
            None => rejected.push(Rejection {
                model: c.profile.id.clone(),
                stage: "quality".into(),
                reason: format!("no capability profile for `{}` or `general`", env.task.capability),
            }),
        }
    }
    if survivors.is_empty() {
        return Err(fail(
            JobState::InsufficientQuality,
            format!("no model meets quality floor {:.2} for `{}`", env.quality.floor, env.task.capability),
            rejected,
        ));
    }

    // Budget.
    let total_tokens = input.prompt_tokens + env.task.max_tokens as u64;
    if let Some(left) = input.budget_left_gbp {
        survivors.retain(|s| {
            let cost = total_tokens as f64 / 1e6 * s.cand.profile.economics.cost_per_mtok_gbp;
            let fits = cost <= left + 1e-12;
            if !fits {
                rejected.push(Rejection {
                    model: s.cand.profile.id.clone(),
                    stage: "budget".into(),
                    reason: format!("estimated £{cost:.4} exceeds the £{:.4} budget left", left.max(0.0)),
                });
            }
            fits
        });
        if survivors.is_empty() {
            return Err(fail(
                JobState::PolicyBlocked,
                "no admissible model fits the remaining budget".into(),
                rejected,
            ));
        }
    }

    // Deadline.
    if let Some(deadline) = env.latency.deadline_at {
        let remaining = deadline - input.now;
        survivors.retain(|s| {
            let fits = s.runtime_ms <= remaining;
            if !fits {
                rejected.push(Rejection {
                    model: s.cand.profile.id.clone(),
                    stage: "deadline".into(),
                    reason: format!("predicted {}ms exceeds remaining {}ms", s.runtime_ms, remaining.max(0)),
                });
            }
            fits
        });
        if survivors.is_empty() {
            return Err(fail(
                JobState::DeadlineExceeded,
                "no admissible model is predicted to finish before the deadline".into(),
                rejected,
            ));
        }
    }

    // Rank survivors by utility.
    let mut scored: Vec<(Survivor, UtilityTerms, f64, f64)> = survivors
        .into_iter()
        .map(|s| {
            let econ = &s.cand.profile.economics;
            let cost_gbp = total_tokens as f64 / 1e6 * econ.cost_per_mtok_gbp;
            let energy_wh = total_tokens as f64 / 1e3 * econ.energy_wh_per_ktok;
            let terms = utility(env, &s, cost_gbp, energy_wh, input.now, input.reuse_potential, input.future_value);
            (s, terms, cost_gbp, energy_wh)
        })
        .collect();
    scored.sort_by(|(a, ua, ..), (b, ub, ..)| {
        ub.total
            .total_cmp(&ua.total)
            .then(a.runtime_ms.cmp(&b.runtime_ms))
            .then(a.cand.profile.id.cmp(&b.cand.profile.id))
    });

    let runner_up = scored.get(1).map(|(s, u, ..)| (s.cand.profile.id.clone(), u.total));
    let (best, terms, cost_gbp, energy_wh) = scored.swap_remove(0);
    let margin = env.latency.deadline_at.map(|d| d - input.now - best.runtime_ms);

    let mut reasons = vec![
        format!("expected quality {:.2} ({}) ≥ floor {:.2}", best.quality, best.basis, env.quality.floor),
        format!("estimated monetary cost £{cost_gbp:.4}"),
        format!(
            "predicted runtime {}ms ({} profile{}, model {})",
            best.runtime_ms,
            match best.cand.profile.source {
                crate::model::ProfileSource::Observed => "observed",
                crate::model::ProfileSource::Configured => "configured",
                crate::model::ProfileSource::StaticPrior => "static-prior",
            },
            match best.cand.profile.evidence.runs {
                0 => String::new(),
                n => format!(", {n} runs"),
            },
            if best.cand.profile.loaded { "warm" } else { "cold" }
        ),
    ];
    if input.reuse_potential > 0.0 {
        reasons.push(format!("reuse potential {:.2} from demand history", input.reuse_potential));
    }
    if input.future_value > 0.0 {
        reasons.push(format!("speculative: predicted {:.2} likely to be asked for", input.future_value));
    }
    if let Some(m) = margin {
        reasons.push(format!("deadline margin {m}ms"));
    }
    if let Some((id, u)) = runner_up {
        reasons.push(format!("utility {:.3} beat {id} at {u:.3}", terms.total));
    }
    if env.quality.verification == Verification::Preferred && !input.validator_available {
        reasons.push("verification preferred but no validator can take this job; result is unverified".into());
    }

    Ok(Decision {
        model: best.cand.profile.id.clone(),
        runtime: best.cand.profile.runtime.clone(),
        exposure: best.cand.exposure.clone(),
        estimated_quality: best.quality,
        quality_basis: best.basis,
        predicted_runtime_ms: best.runtime_ms,
        deadline_margin_ms: margin,
        estimated_cost_gbp: cost_gbp,
        estimated_energy_wh: energy_wh,
        utility: terms,
        reasons,
        considered,
        rejected,
    })
}

/// Weight on monetary cost: one penny outweighs 0.1 of quality.
const COST_PER_GBP: f64 = 10.0;

/// Weight of reuse potential: a result certain to be reused gains a quarter
/// of its quality again.
const REUSE_WEIGHT: f64 = 0.25;

fn utility(
    env: &WorkEnvelope,
    s: &Survivor,
    cost_gbp: f64,
    energy_wh: f64,
    now: Millis,
    reuse: f64,
    future: f64,
) -> UtilityTerms {
    let secs = s.runtime_ms as f64 / 1000.0;
    // Only urgent classes pay for wall time; deferred work is free to be slow.
    let mut latency = match env.latency.class {
        Urgency::Now => 0.01 * secs,
        Urgency::Soon => 0.002 * secs,
        _ => 0.0,
    };
    if let Some(deadline) = env.latency.deadline_at {
        let remaining = (deadline - now).max(1) as f64;
        // Penalise plans that consume most of the remaining window.
        let used = s.runtime_ms as f64 / remaining;
        if used > 0.5 {
            latency += 0.2 * (used - 0.5);
        }
    }
    let energy_weight = match env.resources.energy {
        Energy::Economical => 0.02,
        Energy::Balanced => 0.01,
        Energy::Performance => 0.002,
    };
    let terms = UtilityTerms {
        quality: s.quality,
        reuse: REUSE_WEIGHT * reuse.clamp(0.0, 1.0) * s.quality,
        future_value: REUSE_WEIGHT * future.clamp(0.0, 1.0) * s.quality,
        cost: COST_PER_GBP * cost_gbp,
        latency,
        energy: energy_weight * energy_wh,
        total: 0.0,
    };
    UtilityTerms {
        total: terms.quality + terms.reuse + terms.future_value - terms.cost - terms.latency - terms.energy,
        ..terms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{CompileDefaults, WorkRequest, compile};
    use crate::locality::Locality;
    use crate::model::{Economics, ModelResources, Performance, ProfileSource};
    use std::collections::BTreeMap;

    fn profile(id: &str, q: f64, tps: f64, cost: f64) -> ModelProfile {
        ModelProfile {
            id: id.into(),
            runtime: id.split('/').next().unwrap().into(),
            digest: None,
            capabilities: BTreeMap::from([("general".into(), q)]),
            performance: Performance {
                tokens_per_sec: tps,
                prefill_tokens_per_sec: tps * 10.0,
                startup_ms: 100,
                warm_startup_ms: 10,
            },
            resources: ModelResources::default(),
            economics: Economics { cost_per_mtok_gbp: cost, energy_wh_per_ktok: 0.5 },
            source: ProfileSource::StaticPrior,
            loaded: false,
            evidence: Default::default(),
        }
    }

    fn cand(id: &str, q: f64, tps: f64, cost: f64, exposure: Exposure) -> Candidate {
        Candidate {
            profile: profile(id, q, tps, cost),
            exposure,
            healthy: true,
            available: true,
            pressured: false,
            blocked: None,
        }
    }

    fn env(yaml: &str) -> WorkEnvelope {
        compile(&WorkRequest::parse(yaml).unwrap(), "tny_t", 0, &CompileDefaults::default()).unwrap()
    }

    fn run(env: &WorkEnvelope, cands: &[Candidate], ceiling: &LocalityPolicy) -> Result<Decision, PlanFailure> {
        plan_with(env, cands, ceiling, None, 0.0)
    }

    fn plan_with(
        env: &WorkEnvelope,
        cands: &[Candidate],
        ceiling: &LocalityPolicy,
        budget: Option<f64>,
        reuse: f64,
    ) -> Result<Decision, PlanFailure> {
        plan(&PlanInput {
            envelope: env,
            candidates: cands,
            ceiling,
            now: 0,
            prompt_tokens: 100,
            validator_available: false,
            budget_left_gbp: budget,
            reuse_potential: reuse,
            future_value: 0.0,
        })
    }

    fn unrestricted() -> LocalityPolicy {
        LocalityPolicy { class: Locality::Unrestricted, ssh_allowlist: vec![], remote_allowlist: vec![] }
    }

    fn remote_cheap_excellent() -> Candidate {
        cand("cloud/frontier", 0.99, 500.0, 0.0, Exposure::Remote { host: "api.example.com".into() })
    }

    #[test]
    fn locality_beats_quality_and_cost() {
        let e = env("task: {objective: x}");
        let cands = [remote_cheap_excellent(), cand("local/small", 0.6, 20.0, 0.0, Exposure::InProcess)];
        let d = run(&e, &cands, &unrestricted()).unwrap();
        assert_eq!(d.model, "local/small");
        assert!(d.rejected.iter().any(|r| r.model == "cloud/frontier" && r.stage == "locality"));
    }

    #[test]
    fn local_only_with_only_remote_is_policy_blocked() {
        let e = env("task: {objective: x}");
        let err = run(&e, &[remote_cheap_excellent()], &unrestricted()).unwrap_err();
        assert_eq!(err.state, JobState::PolicyBlocked);
    }

    #[test]
    fn ceiling_applies_even_when_job_allows_more() {
        let e = env("task: {objective: x}\nprivacy: {locality: unrestricted}");
        let err = run(&e, &[remote_cheap_excellent()], &LocalityPolicy::local_only()).unwrap_err();
        assert_eq!(err.state, JobState::PolicyBlocked);
        assert!(err.rejected[0].reason.starts_with("daemon ceiling"));
    }

    #[test]
    fn quality_floor_fails_explicitly() {
        let e = env("task: {objective: x}\nquality: {floor: 0.9}");
        let err = run(&e, &[cand("local/small", 0.6, 20.0, 0.0, Exposure::InProcess)], &unrestricted()).unwrap_err();
        assert_eq!(err.state, JobState::InsufficientQuality);
    }

    #[test]
    fn required_verification_without_validator_fails() {
        let e = env("task: {objective: x}\nquality: {verification: required}");
        let err = run(&e, &[cand("local/small", 0.9, 20.0, 0.0, Exposure::InProcess)], &unrestricted()).unwrap_err();
        assert_eq!(err.state, JobState::InsufficientQuality);
    }

    #[test]
    fn deadline_eliminates_slow_models() {
        // 512 tokens at 1 tok/s ≈ 8.5 min; deadline 1 minute.
        let e = env("task: {objective: x}\nlatency: {deadline: 1m}");
        let cands = [
            cand("local/slow", 0.95, 1.0, 0.0, Exposure::InProcess),
            cand("local/fast", 0.7, 100.0, 0.0, Exposure::InProcess),
        ];
        let d = run(&e, &cands, &unrestricted()).unwrap();
        assert_eq!(d.model, "local/fast");
        let err = run(&e, &cands[..1], &unrestricted()).unwrap_err();
        assert_eq!(err.state, JobState::DeadlineExceeded);
    }

    #[test]
    fn deferred_work_accepts_slow_cheap_models() {
        // A deferred job prefers a slower free model over a faster paid one.
        let e = env("task: {objective: x}\nprivacy: {locality: unrestricted}");
        let cands = [
            cand("local/slow", 0.85, 2.0, 0.0, Exposure::InProcess),
            cand("cloud/fast", 0.87, 300.0, 20.0, Exposure::Remote { host: "api.example.com".into() }),
        ];
        assert_eq!(run(&e, &cands, &unrestricted()).unwrap().model, "local/slow");
        // The same choice under NOW pays for latency and picks the fast one.
        let now = env("task: {objective: x}\nprivacy: {locality: unrestricted}\nlatency: {class: now}");
        assert_eq!(run(&now, &cands, &unrestricted()).unwrap().model, "cloud/fast");
    }

    #[test]
    fn unhealthy_runtimes_are_skipped() {
        let e = env("task: {objective: x}");
        let mut c = cand("local/a", 0.9, 20.0, 0.0, Exposure::InProcess);
        c.healthy = false;
        assert_eq!(run(&e, &[c], &unrestricted()).unwrap_err().state, JobState::ModelFailed);
    }

    #[test]
    fn free_domains_are_preferred_but_busy_ones_are_not_fatal() {
        let e = env("task: {objective: x}");
        let mut busy = cand("local/best", 0.9, 20.0, 0.0, Exposure::InProcess);
        busy.available = false;
        let free = cand("local/ok", 0.7, 20.0, 0.0, Exposure::InProcess);
        let d = run(&e, &[busy.clone(), free], &unrestricted()).unwrap();
        assert_eq!(d.model, "local/ok");
        assert!(d.rejected.iter().any(|r| r.stage == "capacity"));
        assert_eq!(run(&e, &[busy], &unrestricted()).unwrap().model, "local/best");
    }

    #[test]
    fn pinned_jobs_consider_only_their_model() {
        let e = env("task: {objective: x, pin_model: local/b}");
        let cands = [
            cand("local/a", 0.9, 20.0, 0.0, Exposure::InProcess),
            cand("local/b", 0.6, 20.0, 0.0, Exposure::InProcess),
        ];
        assert_eq!(run(&e, &cands, &unrestricted()).unwrap().model, "local/b");
        let e = env("task: {objective: x, pin_model: local/zzz}");
        assert_eq!(run(&e, &cands, &unrestricted()).unwrap_err().state, JobState::ModelFailed);
    }

    #[test]
    fn models_that_do_not_fit_make_the_job_wait() {
        let e = env("task: {objective: x}");
        let mut big = cand("local/big", 0.9, 20.0, 0.0, Exposure::InProcess);
        big.blocked = Some("needs 9000 MB, 2000 MB available".into());
        let small = cand("local/small", 0.6, 20.0, 0.0, Exposure::InProcess);
        assert_eq!(run(&e, &[big.clone(), small], &unrestricted()).unwrap().model, "local/small");
        assert_eq!(run(&e, &[big], &unrestricted()).unwrap_err().state, JobState::ResourceBlocked);
    }

    #[test]
    fn yielding_work_avoids_pressured_domains() {
        let mut busy = cand("local/a", 0.9, 20.0, 0.0, Exposure::InProcess);
        busy.pressured = true;
        let idle_node = cand("node.x/b", 0.6, 20.0, 0.0, Exposure::Ssh { node: "x".into(), shared: false });
        let open = unrestricted();
        let idle = env("task: {objective: x}\nlatency: {class: idle}\nprivacy: {locality: unrestricted}");
        assert_eq!(run(&idle, &[busy.clone(), idle_node], &open).unwrap().model, "node.x/b");
        assert_eq!(run(&idle, std::slice::from_ref(&busy), &open).unwrap_err().state, JobState::ResourceBlocked);
        let now = env("task: {objective: x}\nlatency: {class: now}");
        assert_eq!(run(&now, &[busy], &open).unwrap().model, "local/a", "urgent work does not yield");
    }

    #[test]
    fn budgets_bound_spend() {
        let e = env("task: {objective: x}\nprivacy: {locality: unrestricted}");
        let paid = cand("cloud/paid", 0.95, 300.0, 10.0, Exposure::Remote { host: "api.example.com".into() });
        let free = cand("local/free", 0.6, 20.0, 0.0, Exposure::InProcess);
        let open = unrestricted();
        assert_eq!(plan_with(&e, &[paid.clone(), free], &open, Some(0.0), 0.0).unwrap().model, "local/free");
        let err = plan_with(&e, std::slice::from_ref(&paid), &open, Some(0.0), 0.0).unwrap_err();
        assert_eq!(err.state, JobState::PolicyBlocked);
        assert_eq!(plan_with(&e, &[paid], &open, Some(1.0), 0.0).unwrap().model, "cloud/paid");
    }

    #[test]
    fn reuse_potential_favours_quality() {
        // Equal utility without reuse: the slower, better model wins once the
        // answer is likely to be reused.
        let e = env("task: {objective: x}\nlatency: {class: soon}");
        let better = cand("local/better", 0.70, 20.0, 0.0, Exposure::InProcess);
        let faster = cand("local/faster", 0.66, 200.0, 0.0, Exposure::InProcess);
        let cands = [better, faster];
        assert_eq!(plan_with(&e, &cands, &unrestricted(), None, 0.0).unwrap().model, "local/faster");
        let d = plan_with(&e, &cands, &unrestricted(), None, 1.0).unwrap();
        assert_eq!(d.model, "local/better");
        assert!(d.utility.reuse > 0.0);
    }

    #[test]
    fn future_value_counts_for_speculative_work() {
        let e = env("task: {objective: x}\nlatency: {class: speculative}");
        let c = [cand("local/a", 0.8, 20.0, 0.0, Exposure::InProcess)];
        let d = plan(&PlanInput {
            envelope: &e,
            candidates: &c,
            ceiling: &unrestricted(),
            now: 0,
            prompt_tokens: 100,
            validator_available: false,
            budget_left_gbp: None,
            reuse_potential: 0.0,
            future_value: 0.5,
        })
        .unwrap();
        assert!((d.utility.future_value - 0.25 * 0.5 * 0.8).abs() < 1e-12);
        assert!(d.reasons.iter().any(|r| r.contains("speculative")));
    }

    #[test]
    fn warm_models_predict_faster() {
        let mut p = profile("local/a", 0.8, 10.0, 0.0);
        p.performance.startup_ms = 20_000;
        let cold = p.predict_runtime_ms(0, 10);
        p.loaded = true;
        assert_eq!(cold - p.predict_runtime_ms(0, 10), 20_000 - 10);
    }

    #[test]
    fn planning_is_deterministic() {
        let e = env("task: {objective: x}");
        let a = [
            cand("local/b", 0.8, 20.0, 0.0, Exposure::InProcess),
            cand("local/a", 0.8, 20.0, 0.0, Exposure::InProcess),
        ];
        let b = [a[1].clone(), a[0].clone()];
        assert_eq!(run(&e, &a, &unrestricted()).unwrap(), run(&e, &b, &unrestricted()).unwrap());
        assert_eq!(run(&e, &a, &unrestricted()).unwrap().model, "local/a");
    }
}
