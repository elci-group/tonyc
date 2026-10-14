//! The local HTTP API, served on a Unix socket (and optionally a loopback
//! TCP port). Agents and the `tonyc` CLI speak only this.

use crate::cognition;
use crate::cogstore::CognitiveObject;
use crate::daemon::{Daemon, StopReason};
use crate::fabric::{self, NodeAdapter, NodeSpec};
use crate::host;
use crate::runtimes::ModelAdapter;
use crate::store::{JobRow, now_ms};
use crate::trace;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tonyc_core::cognition::{CognitiveKind, Doc, ObjectState, Scope, Sensitivity, rank};
use tonyc_core::envelope::{WorkRequest, compile};
use tonyc_core::{JobState, Locality};

type D = State<Arc<Daemon>>;

pub fn router(d: Arc<Daemon>) -> Router {
    Router::new()
        .route("/v1/protocol", get(protocol))
        .route("/v1/events", get(events_feed))
        .route("/v1/status", get(status))
        .route("/v1/host", get(host_info))
        .route("/v1/policy", get(policy).post(set_policy))
        .route("/v1/nodes", get(nodes).post(add_node))
        .route("/v1/nodes/{name}", get(node).delete(remove_node))
        .route("/v1/nodes/{name}/probe", post(probe_node))
        .route("/v1/audit", get(audit))
        .route("/v1/audit/verify", get(verify_audit))
        .route("/v1/models", get(models))
        .route("/v1/models/refresh", post(refresh_models))
        .route("/v1/model", get(model_detail))
        .route("/v1/models/benchmark", post(benchmark))
        .route("/v1/benchmarks", get(benchmarks))
        .route("/v1/benchmarks/{id}", get(benchmark_detail))
        .route("/v1/economics", get(economics_report))
        .route("/v1/pd", get(pd_status))
        .route("/v1/pd/explain", get(pd_explain))
        .route("/v1/pd/evaluate", post(pd_evaluate))
        .route("/v1/pd/pause", post(pd_pause))
        .route("/v1/pd/resume", post(pd_resume))
        .route("/v1/jobs", get(list_jobs).post(submit))
        .route("/v1/jobs/{id}", get(job))
        .route("/v1/jobs/{id}/trace", get(job_trace))
        .route("/v1/jobs/{id}/await", get(await_job))
        .route("/v1/jobs/{id}/cancel", post(cancel))
        .route("/v1/try", post(try_existing))
        .route("/v1/cognition", get(query))
        .route("/v1/cognition/scan", post(scan))
        .route("/v1/cognition/{id}", get(object))
        .route("/v1/invalidate", post(invalidate))
        .route("/v1/cache/prune", post(prune))
        .route("/v1/corpus", get(corpus_list).post(corpus_add))
        .route("/v1/corpus/forget", post(corpus_forget))
        .fallback(|| async { err(StatusCode::NOT_FOUND, "no such endpoint") })
        .with_state(d)
}

struct ApiError(StatusCode, Value);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": format!("{e:#}")}))
    }
}

fn err(code: StatusCode, msg: impl Into<String>) -> ApiError {
    ApiError(code, json!({"error": msg.into()}))
}

type ApiResult = Result<Response, ApiError>;

async fn status(State(d): D) -> ApiResult {
    let pressure = host::sample(&d.cfg.pressure, d.host.cores);
    let counts: serde_json::Map<String, Value> =
        d.store.counts()?.into_iter().map(|(s, n)| (s.to_string(), json!(n))).collect();
    let runtimes: Vec<Value> = d.registry.adapters().iter().map(|a| a.describe()).collect();
    Ok(Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "pid": std::process::id(),
        "uptime_ms": now_ms() - d.started_at,
        "socket": d.cfg.socket_path(),
        "data_dir": d.cfg.data_dir(),
        "workers": d.cfg.daemon.workers,
        "running": d.running_ids(),
        "jobs": counts,
        "ceiling": d.ceiling(),
        "pressure": pressure,
        "runtimes": runtimes,
        "off_host_crossings": d.store.off_host_crossings()?,
        "capacity": d.capacity(),
        "nodes": d.nodes().iter().map(|n| json!({"name": n.spec.name, "healthy": n.healthy(), "last_error": n.last_error()})).collect::<Vec<_>>(),
        "cognition": {
            "fresh": d.store.object_counts()?.0,
            "stale": d.store.object_counts()?.1,
            "corpus_files": d.store.corpus(None)?.len(),
            "demand": d.store.demand_counts()?.into_iter().collect::<std::collections::BTreeMap<_, _>>(),
        },
    }))
    .into_response())
}

async fn host_info(State(d): D) -> ApiResult {
    let pressure = host::sample(&d.cfg.pressure, d.host.cores);
    Ok(Json(json!({ "host": d.host, "pressure": pressure, "thresholds": d.cfg.pressure })).into_response())
}

fn policy_view(d: &Daemon) -> Value {
    let defaults = d.defaults();
    json!({
        "ceiling": d.ceiling(),
        "defaults": {
            "locality": defaults.locality,
            "quality_floor": defaults.quality_floor,
            "max_tokens": defaults.max_tokens,
            "max_tokens_ceiling": defaults.max_tokens_ceiling,
            "max_input_bytes": d.cfg.policy.max_input_bytes,
        },
        "sources": d.policy_sources(),
        "localities": Locality::ALL,
        "nodes": d.node_names(),
        "remote_domains_enabled": d.registry.adapters().iter().any(|a| a.exposure().leaves_host() && a.healthy()),
    })
}

async fn policy(State(d): D) -> ApiResult {
    Ok(Json(policy_view(&d)).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicySet {
    max_locality: Option<String>,
    default_locality: Option<String>,
    /// Drop runtime overrides and return to the configuration file.
    #[serde(default)]
    reset: bool,
}

/// `tonyc policy set`: an explicit, persisted change to the locality
/// ceiling or default. Nothing else can widen where data may go.
async fn set_policy(State(d): D, body: String) -> ApiResult {
    let r: PolicySet = json_body(&body)?;
    if r.reset {
        for k in crate::daemon::POLICY_KEYS {
            d.store.put_setting(k, "")?;
        }
    }
    let max = match &r.max_locality {
        Some(v) => v.parse::<Locality>().map_err(bad)?,
        None => d.ceiling().class,
    };
    let default = match &r.default_locality {
        Some(v) => v.parse::<Locality>().map_err(bad)?,
        None => d.defaults().locality,
    };
    if default > max {
        return Err(bad(format!("default_locality {default} would exceed max_locality {max}")));
    }
    if r.max_locality.is_some() {
        d.store.put_setting("max_locality", max.as_str())?;
    }
    if r.default_locality.is_some() {
        d.store.put_setting("default_locality", default.as_str())?;
    }
    tracing::info!(max_locality = %max, default_locality = %default, "policy changed");
    d.wake.notify_one();
    Ok(Json(policy_view(&d)).into_response())
}

async fn models(State(d): D) -> ApiResult {
    let ceiling = d.ceiling();
    let models: Vec<Value> = d
        .registry
        .candidates()
        .into_iter()
        .map(|c| {
            let admissible = ceiling.permits(&c.exposure).err();
            json!({
                "profile": c.profile,
                "exposure": c.exposure,
                "healthy": c.healthy,
                "ceiling_blocks": admissible,
            })
        })
        .collect();
    let runtimes: Vec<Value> = d.registry.adapters().iter().map(|a| a.describe()).collect();
    Ok(Json(json!({ "models": models, "runtimes": runtimes })).into_response())
}

async fn refresh_models(State(d): D) -> ApiResult {
    d.registry.refresh().await;
    d.wake.notify_one();
    models(State(d)).await
}

async fn protocol(State(d): D) -> ApiResult {
    Ok(Json(crate::protocol::describe(&d.cfg.socket_path(), d.cfg.daemon.tcp.as_deref())).into_response())
}

#[derive(Deserialize)]
struct FeedQuery {
    after: Option<i64>,
    limit: Option<usize>,
    wait_ms: Option<u64>,
    /// Only jobs whose `origin.system` is this.
    system: Option<String>,
}

/// The durable event feed: every job state change after a cursor, oldest
/// first. With `wait_ms`, long-polls until there is at least one event. An
/// orchestrator keeps `next` and resumes from it after a restart (its own
/// or the daemon's).
async fn events_feed(State(d): D, Query(q): Query<FeedQuery>) -> ApiResult {
    let after = match q.after {
        Some(a) if a < 0 => d.store.feed_head()?,
        Some(a) => a,
        None => 0,
    };
    let limit = q.limit.unwrap_or(100).clamp(1, 1000);
    let system = q.system.as_deref().map(str::to_ascii_lowercase);
    let until = tokio::time::Instant::now() + Duration::from_millis(q.wait_ms.unwrap_or(0).min(60_000));
    loop {
        let events = d.store.feed(after, limit, system.as_deref())?;
        if !events.is_empty() || tokio::time::Instant::now() >= until {
            let next = events.last().map_or(after, |e| e.seq);
            return Ok(Json(json!({"events": events, "next": next})).into_response());
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            _ = d.shutdown.cancelled() => return Err(err(StatusCode::SERVICE_UNAVAILABLE, "daemon shutting down")),
        }
    }
}

async fn submit(State(d): D, headers: axum::http::HeaderMap, body: String) -> ApiResult {
    let req = WorkRequest::parse(&body)
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, json!({"error": e.to_string(), "field": e.field})))?;
    let key = match headers.get("idempotency-key") {
        None => None,
        Some(v) => match v.to_str() {
            Ok(k) if crate::protocol::valid_key(k) => Some(k.to_string()),
            _ => return Err(bad("Idempotency-Key must be 1-200 printable ASCII characters without spaces")),
        },
    };
    let id = new_id();
    let env = compile(&req, &id, now_ms(), &d.defaults())
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, json!({"error": e.to_string(), "field": e.field})))?;
    for f in &env.task.input_files {
        if !std::path::Path::new(f).is_file() {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                json!({"error": format!("task.input.files: {f} is not a readable file"), "field": "task.input.files"}),
            ));
        }
    }
    if let Some(key) = key {
        use crate::protocol::Submitted;
        match d.store.insert_job_idempotent(&env, &key, &crate::protocol::request_hash(&req))? {
            Submitted::Created => {}
            Submitted::Replayed(id) => {
                let job = find(&d, &id)?;
                return Ok(Json(json!({"id": job.id, "state": job.state, "envelope": job.envelope, "replayed": true}))
                    .into_response());
            }
            Submitted::Conflict(id) => {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    json!({"error": format!("Idempotency-Key was already used for a different request (job {id})"), "job_id": id}),
                ));
            }
        }
    } else {
        d.store.insert_job(&env)?;
    }
    d.store.upsert_template(&env)?;
    d.wake.notify_one();
    Ok((StatusCode::CREATED, Json(json!({ "id": env.id, "state": JobState::Queued, "envelope": env }))).into_response())
}

#[derive(Deserialize)]
struct ListQuery {
    state: Option<String>,
    limit: Option<usize>,
    /// Only jobs whose `origin.system` is this.
    origin: Option<String>,
}

async fn list_jobs(State(d): D, Query(q): Query<ListQuery>) -> ApiResult {
    let state =
        q.state.as_deref().map(str::parse::<JobState>).transpose().map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let limit = q.limit.unwrap_or(50).min(1000);
    let jobs: Vec<Value> = match q.origin.as_deref().map(str::to_ascii_lowercase) {
        None => d.store.list(state, limit)?.iter().map(summary).collect(),
        Some(system) => d
            .store
            .list(state, usize::MAX >> 1)?
            .iter()
            .filter(|j| j.envelope.origin.as_ref().is_some_and(|o| o.system == system))
            .take(limit)
            .map(summary)
            .collect(),
    };
    Ok(Json(json!({ "jobs": jobs })).into_response())
}

fn summary(j: &JobRow) -> Value {
    let env = &j.envelope;
    json!({
        "id": j.id,
        "state": j.state,
        "class": env.latency.class,
        "locality": env.privacy.class,
        "objective": env.task.objective,
        "submitted_at": j.submitted_at,
        "updated_at": j.updated_at,
        "deadline_at": env.latency.deadline_at,
        "model": j.decision.as_ref().and_then(|d| d.get("model")).cloned(),
        "failure": j.failure.as_ref().and_then(|f| f.get("reason")).cloned(),
        "origin": env.origin,
    })
}

/// Resolves a full id or a unique prefix of one.
fn find(d: &Daemon, id: &str) -> Result<JobRow, ApiError> {
    if let Some(job) = d.store.get(id)? {
        return Ok(job);
    }
    let needle = if id.starts_with("tny_") { id.to_string() } else { format!("tny_{id}") };
    if needle.len() < 8 {
        return Err(err(StatusCode::NOT_FOUND, format!("no job {id}")));
    }
    let matches: Vec<JobRow> =
        d.store.list(None, usize::MAX >> 1)?.into_iter().filter(|j| j.id.starts_with(&needle)).collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => Err(err(StatusCode::NOT_FOUND, format!("no job {id}"))),
        n => Err(err(StatusCode::CONFLICT, format!("`{id}` matches {n} jobs; use more characters"))),
    }
}

fn detail(d: &Daemon, job: &JobRow) -> Result<Value, ApiError> {
    Ok(json!({
        "job": job,
        "result": d.store.result(&job.id)?,
        "events": d.store.events(&job.id)?,
        "boundary": d.store.boundary(&job.id)?,
        "ledger": d.store.ledger_row(&job.id)?,
        "passes": d.store.passes(&job.id)?,
    }))
}

async fn job(State(d): D, Path(id): Path<String>) -> ApiResult {
    let job = find(&d, &id)?;
    Ok(Json(detail(&d, &job)?).into_response())
}

async fn job_trace(State(d): D, Path(id): Path<String>) -> ApiResult {
    let job = find(&d, &id)?;
    let text = trace::render(
        &job,
        d.store.result(&job.id)?.as_ref(),
        &d.store.events(&job.id)?,
        &d.store.boundary(&job.id)?,
        d.store.ledger_row(&job.id)?.as_ref(),
        &d.store.passes(&job.id)?,
    );
    Ok(([("content-type", "text/plain; charset=utf-8")], text).into_response())
}

#[derive(Deserialize)]
struct AwaitQuery {
    timeout_ms: Option<u64>,
}

/// Long-polls until the job is terminal or the timeout passes (202).
async fn await_job(State(d): D, Path(id): Path<String>, Query(q): Query<AwaitQuery>) -> ApiResult {
    let id = find(&d, &id)?.id;
    let until = tokio::time::Instant::now() + Duration::from_millis(q.timeout_ms.unwrap_or(30_000).min(600_000));
    loop {
        let job = find(&d, &id)?;
        if job.state.is_terminal() {
            return Ok(Json(detail(&d, &job)?).into_response());
        }
        if tokio::time::Instant::now() >= until {
            return Ok((StatusCode::ACCEPTED, Json(detail(&d, &job)?)).into_response());
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            _ = d.shutdown.cancelled() => return Err(err(StatusCode::SERVICE_UNAVAILABLE, "daemon shutting down")),
        }
    }
}

async fn cancel(State(d): D, Path(id): Path<String>) -> ApiResult {
    let job = find(&d, &id)?;
    if job.state.is_terminal() {
        return Err(err(StatusCode::CONFLICT, format!("job {} is already {}", job.id, job.state)));
    }
    if d.stop(&job.id, StopReason::User) {
        return Ok((StatusCode::ACCEPTED, Json(json!({"id": job.id, "state": "CANCELLING"}))).into_response());
    }
    let detail = json!({"reason": "cancelled by request"});
    if d.store.transition(
        &job.id,
        &[JobState::Queued, JobState::ResourceBlocked],
        JobState::Cancelled,
        "cancelled",
        Some(detail),
    )? {
        return Ok(Json(json!({"id": job.id, "state": JobState::Cancelled})).into_response());
    }
    // It started between the read and the transition.
    if d.stop(&job.id, StopReason::User) {
        return Ok((StatusCode::ACCEPTED, Json(json!({"id": job.id, "state": "CANCELLING"}))).into_response());
    }
    Err(err(StatusCode::CONFLICT, format!("job {} could not be cancelled", job.id)))
}

pub(crate) fn new_id() -> String {
    let seed = uuid::Uuid::now_v7();
    let mut h = blake3::Hasher::new();
    h.update(seed.as_bytes());
    h.update(&std::process::id().to_le_bytes());
    format!("tny_{}", &h.finalize().to_hex()[..16])
}

// --------------------------------------------------------- cognition ---

/// Parses a JSON request body regardless of its content type (an empty body
/// reads as `{}`), so that clients need not set one.
fn json_body<T: serde::de::DeserializeOwned>(body: &str) -> Result<T, ApiError> {
    let body = if body.trim().is_empty() { "{}" } else { body };
    serde_json::from_str(body).map_err(|e| bad(format!("invalid JSON body: {e}")))
}

fn bad(e: impl std::fmt::Display) -> ApiError {
    err(StatusCode::BAD_REQUEST, e.to_string())
}

/// A listing view of an object. Secret content is withheld from listings;
/// it is shown only when the object is fetched by id.
fn object_view(o: &CognitiveObject, score: Option<f64>) -> Value {
    let withheld = o.sensitivity == Sensitivity::Secret;
    json!({
        "id": o.id,
        "kind": o.kind,
        "state": o.state,
        "scope": o.scope,
        "objective": o.objective,
        "model": o.model,
        "confidence": o.confidence,
        "validation": o.validation,
        "freshness": o.freshness.describe(),
        "sensitivity": o.sensitivity,
        "created_at": o.created_at,
        "stale_reason": o.stale_reason,
        "sources": o.sources.len(),
        "parents": o.parents.len(),
        "job_id": o.job_id,
        "score": score,
        "content": if withheld { Value::Null } else { json!(o.content) },
        "withheld": withheld,
    })
}

/// §19: return existing cognition for a request without queueing work.
async fn try_existing(State(d): D, body: String) -> ApiResult {
    let req = WorkRequest::parse(&body)
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, json!({"error": e.to_string(), "field": e.field})))?;
    let env = compile(&req, "tny_try", now_ms(), &d.defaults())
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, json!({"error": e.to_string(), "field": e.field})))?;
    let (hit, skipped) = cognition::find_direct(&d.store, &env)?;
    let scope = Some(&env.reuse.scope);
    d.store.upsert_template(&env)?;
    match hit {
        Some(h) => {
            d.store.note_object(&h.object.id, "reused", json!({"via": "try"}))?;
            crate::pd::note_consumed(&d.store, &h.object.id, "try")?;
            d.store.record_demand(
                "try_hit",
                Some(&env.semantic_hash),
                scope,
                Some(&env.task.objective),
                Some(&h.object.id),
                None,
            )?;
            let mut view = object_view(&h.object, None);
            view["content"] = json!(h.object.content);
            Ok(Json(json!({"hit": true, "object": view, "notes": h.notes, "skipped": skipped})).into_response())
        }
        None => {
            // The miss is evidence of demand (consumed by Phase 6).
            d.store.record_demand(
                "try_miss",
                Some(&env.semantic_hash),
                scope,
                Some(&env.task.objective),
                None,
                None,
            )?;
            let fresh = d.store.objects_in_state(ObjectState::Fresh)?;
            let visible: Vec<&CognitiveObject> =
                fresh.iter().filter(|o| o.scope.visible_to(&env.reuse.scope)).collect();
            let texts: Vec<String> = visible.iter().map(|o| format!("{} {}", o.objective, o.content)).collect();
            let docs: Vec<Doc> = visible.iter().zip(&texts).map(|(o, t)| Doc { id: &o.id, text: t }).collect();
            let nearest: Vec<Value> = rank(&env.task.objective, &docs)
                .into_iter()
                .take(3)
                .filter_map(|(id, score)| visible.iter().find(|o| o.id == id).map(|o| object_view(o, Some(score))))
                .collect();
            Ok(Json(json!({"hit": false, "skipped": skipped, "nearest": nearest})).into_response())
        }
    }
}

#[derive(Deserialize)]
struct CognitionQuery {
    text: Option<String>,
    scope: Option<String>,
    kind: Option<String>,
    /// FRESH (default), STALE or ALL.
    state: Option<String>,
    limit: Option<usize>,
}

/// §18 `query`: existing cognition, ranked by relevance to `text`.
async fn query(State(d): D, Query(q): Query<CognitionQuery>) -> ApiResult {
    let scope = q.scope.as_deref().map(str::parse::<Scope>).transpose().map_err(bad)?;
    let kind = q.kind.as_deref().map(str::parse::<CognitiveKind>).transpose().map_err(bad)?;
    let state = match q.state.as_deref().map(str::to_ascii_uppercase).as_deref() {
        None | Some("FRESH") => Some(ObjectState::Fresh),
        Some("STALE") => Some(ObjectState::Stale),
        Some("ALL") => None,
        Some(other) => return Err(bad(format!("unknown state `{other}` (FRESH, STALE, ALL)"))),
    };
    let limit = q.limit.unwrap_or(20).min(500);
    let mut objects: Vec<CognitiveObject> = d
        .store
        .all_objects()?
        .into_iter()
        .filter(|o| scope.as_ref().is_none_or(|s| &o.scope == s) && kind.is_none_or(|k| o.kind == k))
        .collect();
    // Fresh-looking objects are verified now, so listed states are true.
    for o in objects.iter_mut().filter(|o| o.state == ObjectState::Fresh) {
        if !cognition::verify(&d.store, o)? {
            *o = d.store.object(&o.id)?.unwrap_or(o.clone());
        }
    }
    objects.retain(|o| state.is_none_or(|s| o.state == s));
    let ranked: Vec<(CognitiveObject, Option<f64>)> = match q.text.as_deref().filter(|t| !t.trim().is_empty()) {
        Some(text) => {
            d.store.record_demand("query", None, scope.as_ref(), Some(text), None, None)?;
            let texts: Vec<String> = objects.iter().map(|o| format!("{} {}", o.objective, o.content)).collect();
            let docs: Vec<Doc> = objects.iter().zip(&texts).map(|(o, t)| Doc { id: &o.id, text: t }).collect();
            rank(text, &docs)
                .into_iter()
                .filter_map(|(id, score)| objects.iter().find(|o| o.id == id).map(|o| (o.clone(), Some(score))))
                .collect()
        }
        None => objects.into_iter().map(|o| (o, None)).collect(),
    };
    let out: Vec<Value> = ranked.iter().take(limit).map(|(o, s)| object_view(o, *s)).collect();
    Ok(Json(json!({ "objects": out })).into_response())
}

fn find_object(d: &Daemon, id: &str) -> Result<CognitiveObject, ApiError> {
    if let Some(o) = d.store.object(id)? {
        return Ok(o);
    }
    let needle = if id.starts_with("obj_") { id.to_string() } else { format!("obj_{id}") };
    if needle.len() < 8 {
        return Err(err(StatusCode::NOT_FOUND, format!("no object {id}")));
    }
    let matches: Vec<CognitiveObject> =
        d.store.all_objects()?.into_iter().filter(|o| o.id.starts_with(&needle)).collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => Err(err(StatusCode::NOT_FOUND, format!("no object {id}"))),
        n => Err(err(StatusCode::CONFLICT, format!("`{id}` matches {n} objects; use more characters"))),
    }
}

/// An object with its full derivation: sources, ancestors, descendants,
/// the producing job and its lifecycle events (gate G8).
async fn object(State(d): D, Path(id): Path<String>) -> ApiResult {
    let o = find_object(&d, &id)?;
    let verdict = cognition::check(&d.store, &o, now_ms())?;
    let mut ancestors = Vec::new();
    let mut frontier: Vec<(String, String, usize)> =
        o.parents.iter().map(|p| (p.id.clone(), p.relation.clone(), 1)).collect();
    let mut seen = std::collections::BTreeSet::new();
    while let Some((pid, relation, depth)) = frontier.pop() {
        if depth > 16 || !seen.insert(pid.clone()) {
            continue;
        }
        match d.store.object(&pid)? {
            Some(p) => {
                frontier.extend(p.parents.iter().map(|g| (g.id.clone(), g.relation.clone(), depth + 1)));
                ancestors.push(json!({"id": p.id, "relation": relation, "depth": depth, "state": p.state,
                    "model": p.model, "job_id": p.job_id, "sources": p.sources}));
            }
            None => ancestors.push(json!({"id": pid, "relation": relation, "depth": depth, "state": "DELETED"})),
        }
    }
    Ok(Json(json!({
        "object": o,
        "fresh_now": verdict.is_ok(),
        "stale_because": verdict.err(),
        "ancestors": ancestors,
        "children": d.store.object_children(&o.id)?,
        "events": d.store.object_events(&o.id)?,
    }))
    .into_response())
}

async fn scan(State(d): D) -> ApiResult {
    let store = d.store.clone();
    let report =
        tokio::task::spawn_blocking(move || cognition::scan(&store)).await.map_err(|e| anyhow::anyhow!(e))??;
    Ok(Json(json!(report)).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InvalidateRequest {
    object: Option<String>,
    path: Option<String>,
    scope: Option<String>,
    event: Option<String>,
    reason: Option<String>,
}

/// §18 `invalidate`: by object, source path (file or directory), scope or
/// event tag. Cascades to dependent cognition.
async fn invalidate(State(d): D, body: String) -> ApiResult {
    let r: InvalidateRequest = json_body(&body)?;
    let reason = r.reason.clone().unwrap_or_else(|| "invalidated by request".into());
    let mut ids: Vec<String> = Vec::new();
    let mut selectors = 0;
    if let Some(o) = &r.object {
        selectors += 1;
        ids.push(find_object(&d, o)?.id);
    }
    if let Some(p) = &r.path {
        selectors += 1;
        ids.extend(d.store.objects_with_source_under(p)?);
    }
    if let Some(s) = &r.scope {
        selectors += 1;
        ids.extend(d.store.object_ids_in_scope(&s.parse::<Scope>().map_err(bad)?)?);
    }
    if let Some(e) = &r.event {
        selectors += 1;
        ids.extend(
            d.store
                .objects_in_state(ObjectState::Fresh)?
                .into_iter()
                .filter(|o| o.freshness.events.contains(e))
                .map(|o| o.id),
        );
    }
    if selectors != 1 {
        return Err(bad("give exactly one of object, path, scope or event"));
    }
    ids.sort();
    ids.dedup();
    let detail = match (&r.event, &r.path) {
        (Some(e), _) => format!("{reason} (event {e})"),
        (_, Some(p)) => format!("{reason} (path {p})"),
        _ => reason,
    };
    let report = cognition::invalidate(&d.store, ids.into_iter().map(|id| (id, detail.clone())).collect())?;
    Ok(Json(json!(report)).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PruneRequest {
    /// Only prune objects stale for at least this long (default: all stale).
    older_than: Option<String>,
}

async fn prune(State(d): D, body: String) -> ApiResult {
    let r: PruneRequest = json_body(&body)?;
    let min_age = match &r.older_than {
        Some(t) => tonyc_core::duration::parse_millis(t).map_err(bad)?,
        None => 0,
    };
    let cutoff = now_ms() - min_age;
    let ids: Vec<String> = d
        .store
        .objects_in_state(ObjectState::Stale)?
        .into_iter()
        .filter(|o| o.stale_at.unwrap_or(0) <= cutoff)
        .map(|o| o.id)
        .collect();
    let n = d.store.delete_objects(&ids, "pruned")?;
    Ok(Json(json!({"pruned": n, "objects": ids})).into_response())
}

#[derive(Deserialize)]
struct CorpusQuery {
    scope: Option<String>,
}

async fn corpus_list(State(d): D, Query(q): Query<CorpusQuery>) -> ApiResult {
    let scope = q.scope.as_deref().map(str::parse::<Scope>).transpose().map_err(bad)?;
    Ok(Json(json!({"entries": d.store.corpus(scope.as_ref())?})).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusAdd {
    paths: Vec<String>,
    scope: Option<String>,
    sensitivity: Option<String>,
}

async fn corpus_add(State(d): D, body: String) -> ApiResult {
    let r: CorpusAdd = json_body(&body)?;
    let scope = r.scope.as_deref().unwrap_or("user").parse::<Scope>().map_err(bad)?;
    let sensitivity = r.sensitivity.as_deref().unwrap_or("normal").parse::<Sensitivity>().map_err(bad)?;
    if r.paths.is_empty() {
        return Err(bad("no paths given"));
    }
    let store = d.store.clone();
    let max = d.cfg.cognition.max_corpus_files;
    let added = tokio::task::spawn_blocking(move || cognition::corpus_add(&store, &r.paths, &scope, sensitivity, max))
        .await
        .map_err(|e| anyhow::anyhow!(e))?
        .map_err(bad)?;
    Ok(Json(json!({"added": added.len(), "entries": added})).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusForget {
    target: String,
}

async fn corpus_forget(State(d): D, body: String) -> ApiResult {
    let r: CorpusForget = json_body(&body)?;
    let report = cognition::corpus_forget(&d.store, &r.target).map_err(bad)?;
    Ok(Json(json!(report)).into_response())
}

// ------------------------------------------------------------- fabric ---

fn node_view(d: &Daemon, n: &NodeAdapter) -> Value {
    let exposure = n.exposure();
    json!({
        "name": n.spec.name,
        "target": n.spec.target,
        "port": n.spec.port,
        "identity": n.spec.identity,
        "runtime_url": n.spec.runtime_url,
        "shared": n.spec.shared,
        "slots": n.spec.slots,
        "slots_free": d.slot(&n.domain()).available_permits(),
        "healthy": n.healthy(),
        "last_error": n.last_error(),
        "exposure": exposure,
        "ceiling_blocks": d.ceiling().permits(&exposure).err(),
        "host_keys": n.spec.host_keys.len(),
        "added_at": n.spec.added_at,
        "capabilities": n.caps(),
        "models": n.models().into_iter().map(|m| m.id).collect::<Vec<_>>(),
    })
}

async fn nodes(State(d): D) -> ApiResult {
    let nodes: Vec<Value> = d.nodes().iter().map(|n| node_view(&d, n)).collect();
    Ok(Json(json!({"nodes": nodes, "known_hosts": d.known_hosts_path()})).into_response())
}

fn find_node(d: &Daemon, name: &str) -> Result<std::sync::Arc<NodeAdapter>, ApiError> {
    d.node(name).ok_or_else(|| err(StatusCode::NOT_FOUND, format!("no node {name}")))
}

async fn node(State(d): D, Path(name): Path<String>) -> ApiResult {
    let n = find_node(&d, &name)?;
    Ok(Json(node_view(&d, &n)).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AddNode {
    name: String,
    ssh: String,
    port: Option<u16>,
    identity: Option<String>,
    runtime_url: Option<String>,
    /// `private` (default) or `shared`.
    policy: Option<String>,
    slots: Option<usize>,
    /// `SHA256:...` of the host key to pin, verified out of band. Without
    /// it the host must already be in the user's known_hosts.
    host_key: Option<String>,
}

/// `tonyc node add`: pin the host key, probe over SSH, register.
async fn add_node(State(d): D, body: String) -> ApiResult {
    let r: AddNode = json_body(&body)?;
    if d.node(&r.name).is_some() {
        return Err(err(StatusCode::CONFLICT, format!("node {} exists; remove it first", r.name)));
    }
    let shared = match r.policy.as_deref().unwrap_or("private") {
        "private" => false,
        "shared" => true,
        other => return Err(bad(format!("node policy `{other}` must be private or shared"))),
    };
    let mut spec = NodeSpec {
        name: r.name,
        target: r.ssh,
        port: r.port,
        identity: r.identity,
        runtime_url: r.runtime_url.unwrap_or_else(|| "http://127.0.0.1:11434".into()),
        shared,
        slots: r.slots.unwrap_or(1),
        host_keys: vec![],
        added_at: now_ms(),
    };
    fabric::validate(&spec).map_err(bad)?;
    spec.host_keys = fabric::pin_host_keys(&d.ssh, &spec.target, spec.port, r.host_key.as_deref())
        .await
        .map_err(|e| err(StatusCode::BAD_REQUEST, format!("{e:#}")))?;

    // The probe needs the new key pinned; undo that if the node fails it.
    let mut all = d.store.nodes()?;
    all.push(spec.clone());
    fabric::write_known_hosts(&d.known_hosts_path(), &all)?;
    let probe_node = NodeAdapter::new(spec.clone(), d.ssh.clone());
    if let Err(e) = probe_node.probe().await {
        fabric::write_known_hosts(&d.known_hosts_path(), &d.store.nodes()?)?;
        return Err(err(StatusCode::BAD_GATEWAY, format!("could not reach {} over ssh: {e}", spec.target)));
    }
    d.store.put_node(&spec)?;
    let adapter = d.attach_node(spec);
    adapter.refresh().await;
    d.wake.notify_one();
    tracing::info!(node = %adapter.spec.name, "node added");
    Ok((StatusCode::CREATED, Json(node_view(&d, &adapter))).into_response())
}

async fn remove_node(State(d): D, Path(name): Path<String>) -> ApiResult {
    find_node(&d, &name)?;
    d.detach_node(&name);
    d.store.remove_node(&name)?;
    fabric::write_known_hosts(&d.known_hosts_path(), &d.store.nodes()?)?;
    tracing::info!(node = %name, "node removed");
    Ok(Json(json!({"removed": name})).into_response())
}

async fn probe_node(State(d): D, Path(name): Path<String>) -> ApiResult {
    let n = find_node(&d, &name)?;
    n.refresh().await;
    d.wake.notify_one();
    Ok(Json(node_view(&d, &n)).into_response())
}

#[derive(Deserialize)]
struct AuditQuery {
    job: Option<String>,
    payload: Option<bool>,
    limit: Option<usize>,
}

/// Every crossing of the host boundary, with the exact payload on request.
async fn audit(State(d): D, Query(q): Query<AuditQuery>) -> ApiResult {
    let job = match &q.job {
        Some(j) => Some(find(&d, j)?.id),
        None => None,
    };
    let rows = d.store.audit(job.as_deref(), q.payload.unwrap_or(false), q.limit.unwrap_or(100).min(10_000))?;
    Ok(Json(json!({"crossings": rows, "total_off_host": d.store.off_host_crossings()?})).into_response())
}

/// Re-checks every off-host crossing against the locality policy of the job
/// that made it (after any sensitivity narrowing). A violation here would
/// mean data left the host that policy did not allow (gate G13).
async fn verify_audit(State(d): D) -> ApiResult {
    let crossings = d.store.audit(None, false, usize::MAX >> 1)?;
    let mut violations = Vec::new();
    for c in &crossings {
        let Some(job) = d.store.get(&c.job_id)? else {
            violations.push(json!({"seq": c.seq, "job": c.job_id, "reason": "crossing belongs to no known job"}));
            continue;
        };
        let narrowed = job.cognition.as_ref().is_some_and(|v| v.get("narrowed").is_some());
        let policy = if narrowed { tonyc_core::LocalityPolicy::local_only() } else { job.envelope.privacy.clone() };
        if let Err(reason) = policy.permits(&c.exposure) {
            violations.push(json!({"seq": c.seq, "job": c.job_id, "exposure": c.exposure, "reason": reason}));
        }
    }
    Ok(Json(json!({"checked": crossings.len(), "violations": violations})).into_response())
}

// ---------------------------------------------------------- economics ---

#[derive(Deserialize)]
struct ModelQuery {
    id: String,
}

/// `tonyc model inspect`: prior, measurements and the effective profile.
async fn model_detail(State(d): D, Query(q): Query<ModelQuery>) -> ApiResult {
    let effective = d
        .registry
        .candidates()
        .into_iter()
        .find(|c| c.profile.id == q.id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("no model {}", q.id)))?;
    Ok(Json(json!({
        "effective": effective.profile,
        "exposure": effective.exposure,
        "healthy": effective.healthy,
        "prior": d.registry.prior(&q.id),
        "observed": d.store.observed(&q.id)?,
        "min_runs": crate::economics::MIN_RUNS,
        "configured_override": d.registry.has_override(&q.id),
        "recent": d.store.observations(&q.id, 20)?,
    }))
    .into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkRequest {
    model: String,
    capability: Option<String>,
    cases: Option<usize>,
    class: Option<String>,
}

async fn benchmark(State(d): D, body: String) -> ApiResult {
    let r: BenchmarkRequest = json_body(&body)?;
    let class = r.class.as_deref().unwrap_or("idle").parse().map_err(bad)?;
    let run = crate::bench::start(&d, &r.model, r.capability.as_deref(), class, r.cases).map_err(bad)?;
    Ok((StatusCode::CREATED, Json(json!(run))).into_response())
}

async fn benchmarks(State(d): D) -> ApiResult {
    Ok(Json(json!({"runs": crate::bench::runs(&d.store, 50)?})).into_response())
}

async fn benchmark_detail(State(d): D, Path(id): Path<String>) -> ApiResult {
    let run =
        crate::bench::run(&d.store, &id)?.ok_or_else(|| err(StatusCode::NOT_FOUND, format!("no benchmark {id}")))?;
    Ok(Json(json!({"run": run, "cases": crate::bench::cases_of(&d.store, &id)?})).into_response())
}

#[derive(Deserialize)]
struct EconomicsQuery {
    /// How far back, e.g. `7d`; default: everything.
    since: Option<String>,
}

/// §24: the economics ledger, summarised (gate G12).
async fn economics_report(State(d): D, Query(q): Query<EconomicsQuery>) -> ApiResult {
    let since = match &q.since {
        Some(s) => now_ms() - tonyc_core::duration::parse_millis(s).map_err(bad)?,
        None => 0,
    };
    let report = crate::economics::report(&d.store, &d.cfg.economics, &d.cfg.budgets, since)?;
    Ok(Json(json!(report)).into_response())
}

// ------------------------------------------------- preemptive dynamism ---

fn pd_summary(d: &Daemon) -> Result<Value, ApiError> {
    let y = crate::pd::measure(d)?;
    let now = now_ms();
    let used = d.store.compute_since(tonyc_core::envelope::Urgency::Speculative, crate::economics::day_start(now))?;
    let limit = d
        .cfg
        .budgets
        .daily_limits()?
        .get(&tonyc_core::envelope::Urgency::Speculative)
        .copied()
        .unwrap_or(d.cfg.pd.daily_compute_ms()?);
    Ok(json!({
        "enabled": d.cfg.pd.enabled,
        "paused": crate::pd::paused(&d.store),
        "window": d.cfg.pd.window,
        "horizon": d.cfg.pd.horizon,
        "min_demand": d.cfg.pd.min_demand,
        "max_pending": d.cfg.pd.max_pending,
        "pending": d.store.predictions(Some("QUEUED"), 1000)?.len(),
        "measured": y,
        "budget": {"used_ms": used, "limit_ms": limit},
        "last_evaluated": d.pd_last.lock().unwrap().as_ref().map(|e| e.at),
    }))
}

async fn pd_status(State(d): D) -> ApiResult {
    Ok(Json(pd_summary(&d)?).into_response())
}

/// §25: what is Tonyc computing speculatively, and why it believes the work
/// will become useful.
async fn pd_explain(State(d): D) -> ApiResult {
    let mut active = Vec::new();
    for p in d.store.predictions(None, 50)? {
        let job_state = d.store.get(&p.job_id)?.map(|j| j.state);
        active.push(json!({"prediction": p, "job_state": job_state}));
    }
    Ok(Json(json!({
        "status": pd_summary(&d)?,
        "predictions": active,
        "last_evaluation": *d.pd_last.lock().unwrap(),
    }))
    .into_response())
}

/// Runs an evaluation now (rather than at the next interval).
async fn pd_evaluate(State(d): D) -> ApiResult {
    let e = crate::pd::evaluate(&d)?;
    *d.pd_last.lock().unwrap() = Some(e.clone());
    Ok(Json(json!(e)).into_response())
}

async fn pd_pause(State(d): D) -> ApiResult {
    d.store.put_setting("pd_paused", "true")?;
    Ok(Json(pd_summary(&d)?).into_response())
}

async fn pd_resume(State(d): D) -> ApiResult {
    d.store.put_setting("pd_paused", "false")?;
    Ok(Json(pd_summary(&d)?).into_response())
}
