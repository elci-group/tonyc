//! The work scheduler.
//!
//! Each tick: expire waiting jobs whose deadline has passed, sample host
//! pressure, make running yield-class work give way to pressure, then start
//! waiting jobs in priority order (urgency class, earliest deadline,
//! submission order) while execution slots (local workers plus node slots)
//! are free. A started job plans, then waits for a slot in the domain it
//! chose. When every slot is busy
//! and urgent work is waiting, a running job of a yielding class is
//! preempted and returned to the queue (invariant T5).

use crate::bench;
use crate::cognition;
use crate::cogstore::CognitiveObject;
use crate::daemon::{Daemon, StopReason};
use crate::economics;
use crate::host;
use crate::refinement;
use crate::runtimes::AdapterError;
use crate::store::{ResultRow, SourceRef, now_ms};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonyc_core::cognition::{ObjectState, ValidationState};
use tonyc_core::envelope::{NodeFailure, OutputSchema, Verification, WorkEnvelope};
use tonyc_core::model::estimate_tokens;
use tonyc_core::plan::PlanInput;
use tonyc_core::{Candidate, Exposure, InferenceRequest, JobState, Locality, LocalityPolicy, plan};

pub async fn run(d: Arc<Daemon>) {
    let tick = Duration::from_millis(d.cfg.daemon.tick_ms.max(10));
    loop {
        if let Err(e) = tick_once(&d).await {
            tracing::error!("scheduler tick failed: {e:#}");
        }
        tokio::select! {
            _ = d.wake.notified() => {}
            _ = tokio::time::sleep(tick) => {}
            _ = d.shutdown.cancelled() => break,
        }
    }
}

async fn tick_once(d: &Arc<Daemon>) -> anyhow::Result<()> {
    let now = now_ms();
    let pressure = host::sample(&d.cfg.pressure, d.host.cores);
    let waiting = d.store.waiting()?;

    for job in &waiting {
        if let Some(deadline) = job.envelope.latency.deadline_at
            && deadline <= now
        {
            let failure = json!({"stage": "queue", "reason": "deadline passed before the job could start"});
            if d.store.transition(
                &job.id,
                &[JobState::Queued, JobState::ResourceBlocked],
                JobState::DeadlineExceeded,
                "deadline_exceeded",
                Some(failure.clone()),
            )? {
                d.store.set_failure(&job.id, &failure)?;
            }
        }
    }

    if pressure.busy {
        // Only work on this host yields to this host's pressure.
        d.stop_running(
            |r| r.yields && r.domain.as_deref().is_none_or(|dom| dom == "local"),
            || StopReason::Pressure(pressure.reasons.join("; ")),
        );
    }

    let mut limits = d.cfg.budgets.daily_limits()?;
    if d.cfg.pd.enabled {
        limits.entry(tonyc_core::envelope::Urgency::Speculative).or_insert(d.cfg.pd.daily_compute_ms()?);
    }
    let mut used_today = std::collections::BTreeMap::new();
    for class in limits.keys() {
        used_today.insert(*class, d.store.compute_since(*class, economics::day_start(now))?);
    }

    for job in d.store.waiting()? {
        let env = &job.envelope;
        if !d.retry_due(&job.id, now) {
            continue;
        }
        let block = |reason: String| -> anyhow::Result<()> {
            if job.state == JobState::Queued {
                d.store.transition(
                    &job.id,
                    &[JobState::Queued],
                    JobState::ResourceBlocked,
                    "resource_blocked",
                    Some(json!({"reason": reason})),
                )?;
            }
            Ok(())
        };
        if let Some(limit) = limits.get(&env.latency.class) {
            let used = used_today[&env.latency.class];
            if used >= *limit {
                block(format!(
                    "daily compute budget for {} used: {} of {}",
                    env.latency.class,
                    tonyc_core::duration::Human(used),
                    tonyc_core::duration::Human(*limit)
                ))?;
                continue;
            }
        }
        // Yielding work waits while this host is busy — unless an idle node
        // it may use exists (the planner then avoids the busy host).
        if env.yields_to_pressure() && pressure.busy && d.idle_node_for(env).is_none() {
            block(format!("host pressure: {}", pressure.reasons.join("; ")))?;
            continue;
        }
        if d.running_count() >= d.capacity() {
            if !env.latency.class.yields() {
                // Urgent work is waiting: ask the least urgent yielding job
                // to give up its slot. It returns to the queue.
                let rank = env.latency.class.rank();
                let id = job.id.clone();
                d.preempt_one(rank, &id);
            }
            break;
        }
        start(d, job.envelope.clone(), job.state)?;
    }

    // Score finished benchmark cases; their models' profiles change.
    for model in bench::score_pending(&d.store)? {
        d.refresh_observed(&model)?;
    }
    Ok(())
}

/// Marks each candidate with its domain's pressure and whether the model
/// fits there now (directive §20: resource-aware scheduling).
pub(crate) fn annotate(d: &Daemon, cands: &mut [Candidate], pressure: &host::Pressure) {
    let total_mb = d.host.memory_mb;
    let available_mb = host::available_memory_mb();
    let cap_mb = d.cfg.budgets.memory_max_mb;
    for c in cands.iter_mut() {
        let p = &c.profile;
        let need = if p.loaded { 0 } else { p.resources.ram_mb };
        match crate::daemon::domain_of(&p.runtime).strip_prefix("node:") {
            None => {
                c.pressured = pressure.busy;
                if cap_mb > 0 && need > cap_mb {
                    c.blocked = Some(format!("needs {need} MB; the memory budget allows {cap_mb} MB"));
                } else if total_mb > 0 && need > total_mb {
                    c.blocked = Some(format!("needs {need} MB; this host has {total_mb} MB"));
                } else if need > available_mb {
                    // Loading it now would push other work out of memory:
                    // background work should not; urgent work may.
                    c.pressured = true;
                }
            }
            Some(name) => {
                if let Some(node) = d.node(name) {
                    c.pressured = d.node_pressured(&node);
                    if let Some(caps) = node.caps()
                        && caps.memory_mb > 0
                        && need > caps.memory_mb * 9 / 10
                    {
                        c.blocked = Some(format!("needs {need} MB; node {name} has {} MB", caps.memory_mb));
                    }
                }
            }
        }
    }
}

fn start(d: &Arc<Daemon>, env: WorkEnvelope, from: JobState) -> anyhow::Result<()> {
    // Register before the state change so a cancel request always finds
    // the job either waiting in the store or in the running registry.
    let token = d.register_running(&env);
    if !d.store.transition(&env.id, &[from], JobState::Planning, "planning", None)? {
        d.unregister_running(&env.id);
        return Ok(());
    }
    let d2 = d.clone();
    tokio::spawn(async move {
        let id = env.id.clone();
        if let Err(e) = execute(&d2, &env, &token).await {
            tracing::error!(job = %id, "execution error: {e:#}");
            let failure = json!({"stage": "daemon", "reason": format!("{e:#}")});
            let _ = d2.store.set_failure(&id, &failure);
            let _ = d2.store.transition(
                &id,
                &[JobState::Planning, JobState::Running],
                JobState::ModelFailed,
                "failed",
                Some(failure),
            );
        }
        d2.unregister_running(&id);
        d2.wake.notify_one();
    });
    Ok(())
}

/// Reads input files, hashing each one as provenance. Returns the input
/// section of the prompt and the sources.
fn gather_inputs(env: &WorkEnvelope, max_bytes: u64) -> Result<(String, Vec<SourceRef>), String> {
    let mut body = String::new();
    if let Some(text) = &env.task.input_text {
        body.push('\n');
        body.push_str(text);
        body.push('\n');
    }
    let mut sources = Vec::new();
    for path in &env.task.input_files {
        let meta = std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
        if meta.len() > max_bytes {
            return Err(format!("{path} is {} bytes, over the {max_bytes}-byte input limit", meta.len()));
        }
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let hash = blake3::hash(&bytes).to_hex().to_string();
        let text = String::from_utf8(bytes).map_err(|_| format!("{path} is not UTF-8 text"))?;
        sources.push(SourceRef { path: path.clone(), blake3: hash, bytes: text.len() as u64 });
        body.push_str(&format!("\n--- {path} ---\n{text}\n"));
    }
    if env.output.schema == OutputSchema::Json {
        body.push_str("\nRespond with a single JSON value and nothing else.\n");
    }
    Ok((body, sources))
}

async fn execute(d: &Arc<Daemon>, env: &WorkEnvelope, token: &CancellationToken) -> anyhow::Result<()> {
    let id = env.id.as_str();
    let fail = |state: JobState, kind: &str, failure: serde_json::Value| -> anyhow::Result<()> {
        d.store.set_failure(id, &failure)?;
        d.store.transition(
            id,
            &[JobState::Planning, JobState::Running, JobState::Refining],
            state,
            kind,
            Some(failure),
        )?;
        Ok(())
    };

    let (input_body, sources) = match gather_inputs(env, d.cfg.policy.max_input_bytes) {
        Ok(v) => v,
        Err(reason) => return fail(JobState::ModelFailed, "failed", json!({"stage": "input", "reason": reason})),
    };

    // Cognitive Store: existing cognition first (directive §16).
    let mut cognition = json!({});
    if env.reuse.cache {
        let (hit, skipped) = cognition::find_direct(&d.store, env)?;
        if !skipped.is_empty() {
            cognition["direct_skipped"] = json!(skipped);
        }
        if let Some(hit) = hit {
            return complete_from_store(d, env, hit, cognition);
        }
    } else {
        cognition["cache"] = json!("disabled by request");
    }
    let job_paths: Vec<String> = sources.iter().map(|s| s.path.clone()).collect();
    let context =
        if env.reuse.cache { cognition::select_context(&d.store, env, &job_paths, &d.cfg.cognition)? } else { vec![] };
    if !context.is_empty() {
        cognition["context"] = serde_json::to_value(&context)?;
    }

    // Sensitive corpus sources narrow the job to LOCAL_ONLY, structurally.
    let (sensitivity, why) = cognition::sensitivity_of(&d.store, &sources, &context)?;
    let mut planned = env.clone();
    if sensitivity.forces_local() && env.privacy.class != Locality::LocalOnly {
        planned.privacy = LocalityPolicy::local_only();
        cognition["narrowed"] = json!({"from": env.privacy.class, "to": Locality::LocalOnly, "because": why});
    }
    d.store.set_cognition(id, &cognition)?;

    let prompt = format!(
        "{}\n{}{}",
        env.task.objective,
        cognition::context_block(&context, d.cfg.cognition.context_chars),
        input_body
    );
    let prompt_tokens = estimate_tokens(prompt.len());
    // An answer asked for before is likely to be asked for again.
    let reuse_potential = (d.store.prior_demand(&env.semantic_hash, id)? as f64 / 3.0).min(1.0);
    // Speculative work carries its prediction's probability as V.
    let prediction = d.store.prediction_for_job(id)?;
    let future_value = prediction.as_ref().map(|p| p.probability).unwrap_or(0.0);
    // Strategy: one pass, or a refinement pipeline (directive §14).
    let strategy = {
        let mut cands = d.registry.candidates_with(|domain| d.has_free_slot(domain));
        annotate(d, &mut cands, &host::sample(&d.cfg.pressure, d.host.cores));
        let budget_left = d.cfg.budgets.remote_monthly_gbp - d.store.spend_since(economics::month_start(now_ms()))?;
        let prelim = plan(&PlanInput {
            envelope: &planned,
            candidates: &cands,
            ceiling: &d.ceiling(),
            now: now_ms(),
            prompt_tokens,
            validator_available: true,
            budget_left_gbp: Some(budget_left.max(0.0)),
            reuse_potential,
            future_value,
        });
        let shortfall = matches!(&prelim, Err(f) if f.state == JobState::InsufficientQuality);
        if prelim.is_err() && !shortfall {
            // Some other constraint: the dispatch loop reports it.
            refinement::Strategy { pipeline: false, validators: vec![], ladder: vec![], explain: vec![] }
        } else {
            match refinement::strategy(d, &planned, &planned.privacy, prompt_tokens, prelim.is_ok(), shortfall) {
                Ok(s) => s,
                Err((state, reason)) => {
                    return fail(state, "plan_failed", json!({"stage": "refinement plan", "reason": reason}));
                }
            }
        }
    };
    // A pipeline drafts by utility alone — cheaply — and earns its
    // confidence through validation.
    let mut draft_env = planned.clone();
    if strategy.pipeline {
        draft_env.quality.floor = 0.0;
        draft_env.quality.verification = Verification::None;
        d.store.note(id, "strategy", json!({"pipeline": true, "explain": strategy.explain}))?;
    }

    // Plan, wait for a slot in the chosen domain, dispatch. If an SSH node
    // fails, re-plan without it (same locality rules) or fail, per the job's
    // `on_node_failure` policy (gate G7).
    let mut excluded: Vec<String> = Vec::new();
    let mut node_failures: Vec<String> = Vec::new();
    let (decision, exposure, out, started, permit, domain, candidates) = loop {
        let mut candidates: Vec<_> = d
            .registry
            .candidates_with(|domain| d.has_free_slot(domain))
            .into_iter()
            .filter(|c| !excluded.contains(&c.profile.runtime))
            .collect();
        annotate(d, &mut candidates, &host::sample(&d.cfg.pressure, d.host.cores));
        let ceiling = d.ceiling();
        let budget_left = d.cfg.budgets.remote_monthly_gbp - d.store.spend_since(economics::month_start(now_ms()))?;
        let planned_result = plan(&PlanInput {
            envelope: &draft_env,
            candidates: &candidates,
            ceiling: &ceiling,
            now: now_ms(),
            prompt_tokens,
            validator_available: false,
            budget_left_gbp: Some(budget_left.max(0.0)),
            reuse_potential,
            future_value,
        });
        let decision = match planned_result {
            Ok(decision) => decision,
            Err(f) if !node_failures.is_empty() => {
                let reason = format!("{}; no fallback: {}", node_failures.join("; "), f.reason);
                let plan = serde_json::to_value(&f)?;
                return fail(
                    JobState::NodeUnreachable,
                    "node_unreachable",
                    json!({"stage": "fallback", "reason": reason, "plan": plan}),
                );
            }
            Err(f) if f.state == JobState::ResourceBlocked => {
                // Not a failure: the job waits and is planned again later.
                let retry_at = d.defer(id);
                let detail = json!({"reason": f.reason, "plan": serde_json::to_value(&f)?, "retry_at": retry_at});
                d.store.transition(
                    id,
                    &[JobState::Planning],
                    JobState::ResourceBlocked,
                    "resource_blocked",
                    Some(detail),
                )?;
                return Ok(());
            }
            Err(f) => {
                let failure = serde_json::to_value(&f)?;
                return fail(f.state, "plan_failed", json!({"stage": "plan", "reason": f.reason, "plan": failure}));
            }
        };
        d.store.set_decision(id, &serde_json::to_value(&decision)?)?;

        if let Some(reason) = d.stop_reason(id) {
            return settle_stopped(d, id, reason, JobState::Planning);
        }
        let Some((adapter, model)) = d.registry.resolve(&decision.model) else {
            return fail(
                JobState::ModelFailed,
                "failed",
                json!({"stage": "dispatch", "reason": format!("no adapter serves {}", decision.model)}),
            );
        };
        // Locality is re-checked against the adapter's exposure at dispatch
        // time: if the runtime's classification changed since planning, the
        // job does not run.
        let exposure = adapter.exposure();
        if let Err(r) = planned.privacy.permits(&exposure).and_then(|_| ceiling.permits(&exposure)) {
            return fail(JobState::PolicyBlocked, "policy_blocked", json!({"stage": "dispatch", "reason": r}));
        }

        let domain = adapter.domain();
        let permit = tokio::select! {
            p = d.slot(&domain).acquire_owned() => p.expect("slot semaphores are never closed"),
            _ = token.cancelled() => {
                let reason = d.stop_reason(id).unwrap_or(StopReason::User);
                return settle_stopped(d, id, reason, JobState::Planning);
            }
            _ = until_deadline(env) => {
                let reason = format!("deadline reached while waiting for a slot on {domain}");
                return fail(JobState::DeadlineExceeded, "deadline_exceeded", json!({"stage": "slot", "reason": reason}));
            }
        };
        d.set_running_domain(id, &domain);
        d.clear_retry(id);
        d.store.transition(
            id,
            &[JobState::Planning],
            JobState::Running,
            "running",
            Some(json!({"model": decision.model, "exposure": exposure, "domain": domain})),
        )?;

        let req = InferenceRequest {
            model,
            prompt: prompt.clone(),
            max_tokens: env.task.max_tokens,
            json: env.output.schema == OutputSchema::Json,
        };
        let wire = adapter.encode(&req);
        d.store.record_boundary(
            id,
            &exposure,
            wire.len() as u64,
            0,
            &format!("request dispatched to {}", decision.model),
            Some(&wire),
        )?;

        let started = now_ms();
        let outcome = tokio::select! {
            r = adapter.run(&req) => Some(r),
            _ = token.cancelled() => None,
            _ = until_deadline(env) => {
                return fail(JobState::DeadlineExceeded, "deadline_exceeded", json!({"stage": "run", "reason": "deadline reached while running"}));
            }
        };
        let Some(result) = outcome else {
            let reason = d.stop_reason(id).unwrap_or(StopReason::User);
            return settle_stopped(d, id, reason, JobState::Running);
        };
        match result {
            Ok(out) => break (decision, exposure, out, started, permit, domain, candidates),
            Err(AdapterError::NodeUnreachable(e)) => {
                let summary = format!("dispatch to {} failed: {e}", decision.model);
                d.store.record_boundary(id, &exposure, 0, 0, &summary, None)?;
                node_failures.push(format!("{domain} unreachable: {e}"));
                if env.resources.on_node_failure == NodeFailure::Fail {
                    let reason = format!("{domain} unreachable: {e}");
                    return fail(
                        JobState::NodeUnreachable,
                        "node_unreachable",
                        json!({"stage": "run", "reason": reason, "policy": "fail"}),
                    );
                }
                excluded.push(adapter.runtime().to_string());
                drop(permit);
                let reason = format!("{domain} unreachable ({e}); re-planning without it");
                d.store.transition(
                    id,
                    &[JobState::Running],
                    JobState::Planning,
                    "fallback",
                    Some(json!({"reason": reason})),
                )?;
            }
            Err(AdapterError::Unreachable(e)) => {
                let reason = format!("runtime unreachable: {e}");
                return fail(JobState::ModelFailed, "failed", json!({"stage": "run", "reason": reason}));
            }
            Err(AdapterError::Failed(e)) => {
                return fail(JobState::ModelFailed, "failed", json!({"stage": "run", "reason": e}));
            }
        }
    };
    d.store.record_boundary(
        id,
        &exposure,
        0,
        out.bytes_received,
        &format!("response received from {}", decision.model),
        None,
    )?;

    // Slots are held per inference, not per job: refinement passes queue
    // for their own.
    drop(permit);
    if out.text.trim().is_empty() {
        return fail(JobState::ModelFailed, "failed", json!({"stage": "output", "reason": "model returned no output"}));
    }

    // The draft's own measurements and costs.
    let draft_wall = now_ms() - started;
    d.store.record_observation(
        &decision.model,
        &economics::ObservationRow {
            at: now_ms(),
            job_id: id.to_string(),
            domain: domain.clone(),
            tokens_in: out.tokens_in,
            tokens_out: out.tokens_out,
            wall_ms: draft_wall,
            timings: out.timings,
        },
    )?;
    d.refresh_observed(&decision.model)?;
    let profile_of = |model: &str| candidates.iter().find(|c| c.profile.id == model).map(|c| c.profile.clone());
    let cost_per_mtok = profile_of(&decision.model).map(|p| p.economics.cost_per_mtok_gbp).unwrap_or(0.0);
    let (draft_energy, draft_spend) =
        refinement::costs(d, &domain, &decision.runtime, draft_wall, out.tokens_in + out.tokens_out, cost_per_mtok);
    let mut totals = refinement::Totals {
        tokens_in: out.tokens_in,
        tokens_out: out.tokens_out,
        wall_ms: draft_wall,
        energy_wh: draft_energy,
        spend_gbp: draft_spend,
        domains: [domain.clone()].into(),
    };
    let econ = d.cfg.economics.clone();
    let ledger = |kind: &str, model: &str, totals: &refinement::Totals, conventional: f64| -> anyhow::Result<()> {
        d.store.record_ledger(&economics::LedgerRow {
            job_id: id.to_string(),
            at: now_ms(),
            kind: kind.into(),
            class: env.latency.class,
            model: model.to_string(),
            domain: totals.domain_label(),
            leaves_host: totals.domains.iter().any(|dom| dom != "local"),
            tokens_in: totals.tokens_in,
            tokens_out: totals.tokens_out,
            wall_ms: totals.wall_ms,
            energy_wh: totals.energy_wh,
            energy_gbp: totals.energy_wh / 1000.0 * econ.electricity_gbp_per_kwh,
            spend_gbp: totals.spend_gbp,
            conventional_gbp: conventional,
            context_objects: context.len() as u32,
        })
    };

    // The answer: the draft as it is, or refined (checks, validation,
    // revision, escalation).
    let (text, final_model, final_runtime, final_exposure, confidence, verified, digest) = if refinement::wanted(
        env,
        strategy.pipeline,
    ) {
        d.store.transition(
            id,
            &[JobState::Running],
            JobState::Refining,
            "refining",
            Some(json!({"pipeline": strategy.pipeline, "validators": strategy.validators.len()})),
        )?;
        let draft = refinement::Draft {
            text: out.text.clone(),
            model: decision.model.clone(),
            exposure: exposure.clone(),
            quality: decision.estimated_quality,
            started_at: started,
            domain: domain.clone(),
        };
        let ended = refinement::refine(
            d,
            env,
            &planned.privacy,
            &input_body,
            draft,
            // Validators run in a pipeline; a single pass refined only for
            // its checks is revised, not judged.
            if strategy.pipeline { &strategy.validators } else { &[] },
            &strategy.ladder,
            &mut totals,
            token,
        )
        .await?;
        match ended {
            refinement::Ended::Stopped(reason) => return settle_stopped(d, id, reason, JobState::Refining),
            refinement::Ended::Deadline => {
                ledger("failed", &decision.model, &totals, 0.0)?;
                let reason = "deadline reached while refining";
                return fail(
                    JobState::DeadlineExceeded,
                    "deadline_exceeded",
                    json!({"stage": "refine", "reason": reason}),
                );
            }
            refinement::Ended::Done(o) if !o.satisfied => {
                // Inference was spent and nothing usable came of it: the
                // ledger records the cost, with nothing avoided.
                ledger("failed", &o.model, &totals, 0.0)?;
                let best: String = o.text.chars().take(2000).collect();
                return fail(
                    JobState::InsufficientQuality,
                    "insufficient_quality",
                    json!({"stage": "refine", "reason": o.reason, "confidence": o.confidence,
                               "verified": o.verified, "rounds": o.rounds, "best_model": o.model, "best_attempt": best}),
                );
            }
            refinement::Ended::Done(o) => {
                let runtime = d.registry.resolve(&o.model).map(|(a, _)| a.runtime().to_string()).unwrap_or_default();
                let digest = if o.model == decision.model {
                    out.model_digest.clone()
                } else {
                    profile_of(&o.model).and_then(|p| p.digest)
                };
                (o.text, o.model, runtime, o.exposure, o.confidence, o.verified, digest)
            }
        }
    } else {
        if env.output.schema == OutputSchema::Json && serde_json::from_str::<serde_json::Value>(&out.text).is_err() {
            let reason = "output is not valid JSON (json/v1)";
            return fail(JobState::ModelFailed, "failed", json!({"stage": "output", "reason": reason}));
        }
        (
            out.text.clone(),
            decision.model.clone(),
            decision.runtime.clone(),
            exposure.clone(),
            decision.estimated_quality,
            false,
            out.model_digest.clone(),
        )
    };

    let finished = now_ms();
    // A premium API would have answered in one call of the draft's size;
    // refinement passes are Tonyc's own method and count as operating cost.
    ledger("inference", &final_model, &totals, economics::conventional_cost(&econ, out.tokens_in, out.tokens_out))?;
    let object_id = if env.reuse.cache {
        let object = CognitiveObject {
            id: new_object_id(id),
            kind: env.output.kind,
            semantic_key: env.semantic_hash.clone(),
            scope: env.reuse.scope.clone(),
            objective: env.task.objective.clone(),
            content: text.clone(),
            schema: env.output.schema.as_str().to_string(),
            confidence,
            validation: if verified { ValidationState::Verified } else { ValidationState::Unverified },
            freshness: env.reuse.freshness.clone(),
            expires_at: env.reuse.freshness.expires_at(finished),
            state: ObjectState::Fresh,
            stale_reason: None,
            stale_at: None,
            created_at: finished,
            job_id: id.to_string(),
            model: final_model.clone(),
            model_digest: digest.clone(),
            runtime: final_runtime.clone(),
            exposure: final_exposure.clone(),
            privacy: planned.privacy.clone(),
            sensitivity,
            sources: sources.clone(),
            parents: cognition::parents_of(&context),
        };
        d.store.insert_object(&object)?;
        cognition["produced"] = json!(object.id);
        d.store.set_cognition(id, &cognition)?;
        Some(object.id)
    } else {
        None
    };
    // PD's own work is never demand — it must not feed itself — and nor are
    // benchmarks, which are Tonyc measuring itself.
    let kind = match (&prediction, &env.task.pin_model) {
        (Some(_), _) => "speculative",
        (None, Some(_)) => "benchmark",
        _ => "inference",
    };
    d.store.record_demand(
        kind,
        Some(&env.semantic_hash),
        Some(&env.reuse.scope),
        Some(&env.task.objective),
        object_id.as_deref(),
        Some(id),
    )?;

    d.store.put_result(&ResultRow {
        job_id: id.to_string(),
        output: text,
        schema: env.output.schema.as_str().to_string(),
        model: final_model,
        runtime: final_runtime,
        model_digest: digest,
        exposure: final_exposure,
        semantic_hash: env.semantic_hash.clone(),
        sources,
        tokens_in: totals.tokens_in,
        tokens_out: totals.tokens_out,
        started_at: started,
        finished_at: finished,
        reused_from: None,
        object_id,
    })?;
    d.store.transition(
        id,
        &[JobState::Running, JobState::Refining],
        JobState::Ready,
        "ready",
        Some(json!({"elapsed_ms": finished - started, "confidence": confidence, "verified": verified})),
    )?;
    Ok(())
}

/// Completes a job from an existing object: no planning, no inference, no
/// data leaves the daemon (gate G10).
fn complete_from_store(
    d: &Arc<Daemon>,
    env: &WorkEnvelope,
    hit: cognition::DirectHit,
    mut cognition: serde_json::Value,
) -> anyhow::Result<()> {
    let id = env.id.as_str();
    let o = hit.object;
    let now = now_ms();
    cognition["reused"] = json!({
        "object": o.id,
        "produced_by": o.job_id,
        "model": o.model,
        "created_at": o.created_at,
        "notes": hit.notes,
    });
    d.store.set_cognition(id, &cognition)?;
    d.store.put_result(&ResultRow {
        job_id: id.to_string(),
        output: o.content.clone(),
        schema: o.schema.clone(),
        model: o.model.clone(),
        runtime: "cognitive-store".into(),
        model_digest: o.model_digest.clone(),
        exposure: Exposure::InProcess,
        semantic_hash: env.semantic_hash.clone(),
        sources: o.sources.clone(),
        tokens_in: 0,
        tokens_out: 0,
        started_at: now,
        finished_at: now,
        reused_from: Some(o.id.clone()),
        object_id: Some(o.id.clone()),
    })?;
    d.store.note_object(&o.id, "reused", json!({"job": id}))?;
    crate::pd::note_consumed(&d.store, &o.id, &format!("served job {id}"))?;
    let (conventional, tokens_in, tokens_out) = d.store.conventional_of_job(&o.job_id)?.unwrap_or_else(|| {
        let out = estimate_tokens(o.content.len());
        (economics::conventional_cost(&d.cfg.economics, 0, out), 0, out)
    });
    d.store.record_ledger(&economics::LedgerRow {
        job_id: id.to_string(),
        at: now,
        kind: "reuse".into(),
        class: env.latency.class,
        model: o.model.clone(),
        domain: "local".into(),
        leaves_host: false,
        tokens_in,
        tokens_out,
        wall_ms: 0,
        energy_wh: 0.0,
        energy_gbp: 0.0,
        spend_gbp: 0.0,
        conventional_gbp: conventional,
        context_objects: 0,
    })?;
    d.store.record_demand(
        "reuse_hit",
        Some(&env.semantic_hash),
        Some(&env.reuse.scope),
        Some(&env.task.objective),
        Some(&o.id),
        Some(id),
    )?;
    d.store.transition(
        id,
        &[JobState::Planning],
        JobState::Ready,
        "reused",
        Some(json!({"object": o.id, "reason": format!("satisfied by existing cognition {}", o.id)})),
    )?;
    Ok(())
}

/// Resolves when the job's deadline passes (never, without one).
async fn until_deadline(env: &WorkEnvelope) {
    match env.latency.deadline_at {
        Some(at) => tokio::time::sleep(Duration::from_millis((at - now_ms()).max(0) as u64)).await,
        None => std::future::pending().await,
    }
}

fn new_object_id(job: &str) -> String {
    let mut h = blake3::Hasher::new();
    h.update(job.as_bytes());
    h.update(uuid::Uuid::now_v7().as_bytes());
    format!("obj_{}", &h.finalize().to_hex()[..16])
}

fn settle_stopped(d: &Arc<Daemon>, id: &str, reason: StopReason, from: JobState) -> anyhow::Result<()> {
    let (to, kind, detail) = match reason {
        StopReason::User => (JobState::Cancelled, "cancelled", json!({"reason": "cancelled by request"})),
        StopReason::Preempted(by) => {
            (JobState::Queued, "preempted", json!({"reason": format!("yielded to more urgent job {by}")}))
        }
        StopReason::Pressure(why) => {
            (JobState::ResourceBlocked, "yielded", json!({"reason": format!("host pressure: {why}")}))
        }
        StopReason::Shutdown => {
            (JobState::Queued, "interrupted", json!({"reason": "daemon shutting down; job requeued"}))
        }
    };
    d.store.transition(id, &[from], to, kind, Some(detail))?;
    Ok(())
}
