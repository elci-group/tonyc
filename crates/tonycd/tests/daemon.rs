//! End-to-end daemon tests over the real Unix-socket API, mapped to the
//! v0.1 acceptance gates that Phase 1 owns (directive §30).

mod common;

use common::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tonyc_core::{Exposure, InferenceOutput, InferenceRequest, Locality, ModelProfile};
use tonycd::runtimes::{AdapterError, BoxFuture, ModelAdapter, Registry};

// --------------------------------------------------------------- gates ---

/// G1, G8: an agent submits deferred work; the result carries its lineage.
#[tokio::test]
async fn g1_g8_deferred_work_completes_with_provenance() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let file = h.data.parent().unwrap().join("input.txt");
    std::fs::write(&file, "alpha beta gamma").unwrap();
    let id = h
        .submit(&format!(
            "task:\n  objective: summarise\n  input: {{files: [{}]}}\nlatency: {{class: deferred, deadline: 10m}}\n",
            file.display()
        ))
        .await;
    let v = h.wait(&id).await;
    assert_eq!(v["job"]["state"], "READY");
    let r = &v["result"];
    assert!(r["output"].as_str().unwrap().contains("alpha beta gamma"));
    assert_eq!(r["model"], "builtin/echo");
    assert_eq!(r["model_digest"], "builtin-echo-v1");
    assert_eq!(r["sources"][0]["blake3"], blake3::hash(b"alpha beta gamma").to_hex().as_str());
    assert_eq!(r["semantic_hash"], v["job"]["envelope"]["semantic_hash"]);
    assert_eq!(events(&v), ["submitted", "planning", "running", "ready"]);
    assert_eq!(v["job"]["decision"]["model"], "builtin/echo");
    h.stop().await;
}

/// G2: queued and in-flight work survives a daemon restart.
#[tokio::test]
async fn g2_work_survives_restart() {
    // 200 ms per token × 6 words: the job is mid-run when we stop, even on a
    // loaded machine.
    let mut h = Harness::new(|_| {}, vec![builtin(200)]).await;
    let running = h.submit("task: {objective: one two three four five six}").await;
    let queued = h.submit("task: {objective: queued behind}").await;
    h.wait_for_state(&running, "RUNNING").await;
    h.restart(|_| {}, vec![builtin(0)]).await;
    for id in [&running, &queued] {
        let v = h.wait(id).await;
        assert_eq!(v["job"]["state"], "READY", "{id}");
    }
    let (_, v) = h.call("GET", &format!("/v1/jobs/{running}"), "").await;
    let kinds = events(&v);
    assert!(kinds.contains(&"interrupted".to_string()), "{kinds:?}");
    assert_eq!(v["job"]["attempts"], 2);
    h.stop().await;
}

/// G2 (crash): a job left RUNNING by a killed daemon is requeued on start.
#[tokio::test]
async fn g2_crash_recovery_requeues_in_flight_jobs() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path(), 0);
    std::fs::create_dir_all(cfg.data_dir()).unwrap();
    {
        let store = tonycd::store::Store::open(&cfg.data_dir().join("state.db")).unwrap();
        let env = tonyc_core::compile(
            &tonyc_core::WorkRequest::parse("task: {objective: crashed}").unwrap(),
            "tny_crashed",
            tonycd::store::now_ms(),
            &Default::default(),
        )
        .unwrap();
        store.insert_job(&env).unwrap();
        store.transition("tny_crashed", &[], tonyc_core::JobState::Running, "running", None).unwrap();
    }
    let h = Harness {
        socket: cfg.socket_path(),
        data: cfg.data_dir(),
        running: Some(tonycd::start_with(cfg, Registry::new(vec![builtin(0)], BTreeMap::new())).await.unwrap()),
        _dir: dir,
    };
    let v = h.wait("tny_crashed").await;
    assert_eq!(v["job"]["state"], "READY");
    assert!(events(&v).contains(&"recovered".to_string()));
    h.stop().await;
}

/// G3: deadlines shape scheduling — earliest deadline first, infeasible
/// plans fail before running, and running work is stopped at its deadline.
#[tokio::test]
async fn g3_deadlines_influence_scheduling() {
    // 512 tokens at 50 ms/token cannot fit in 2 s.
    let slow = Harness::new(|_| {}, vec![builtin(50)]).await;
    let infeasible = slow.submit("task: {objective: x}\nlatency: {deadline: 2s}").await;
    let v = slow.wait(&infeasible).await;
    assert_eq!(v["job"]["state"], "DEADLINE_EXCEEDED");
    assert_eq!(v["job"]["failure"]["stage"], "plan");
    slow.stop().await;

    // Earliest deadline first within a class.
    let blocker = Harness::new(|c| c.daemon.workers = 1, vec![builtin(120)]).await;
    let first = blocker.submit("task: {objective: a b c d e f g h}\nlatency: {class: now}").await;
    blocker.wait_for_state(&first, "RUNNING").await;
    let late = blocker.submit("task: {objective: late}\nlatency: {deadline: 1h}").await;
    let early = blocker.submit("task: {objective: early}\nlatency: {deadline: 10m}").await;
    let (e, l) = (blocker.wait(&early).await, blocker.wait(&late).await);
    let started = |v: &Value| v["result"]["started_at"].as_i64().unwrap();
    assert!(started(&e) <= started(&l), "earlier deadline should start first");
    blocker.stop().await;

    // A running job that reaches its deadline stops with DEADLINE_EXCEEDED.
    // The model is *predicted* fast (override) but runs slowly.
    let lying = Harness::new(
        |c| {
            let fast = tonycd::config::ModelOverride { tokens_per_sec: Some(1e6), ..Default::default() };
            c.models.insert("builtin/echo".into(), fast);
        },
        vec![builtin(200)],
    )
    .await;
    let id = lying
        .submit("task: {objective: one two three four five six seven eight nine ten}\nlatency: {deadline: 1s}")
        .await;
    let v = lying.wait(&id).await;
    assert_eq!(v["job"]["state"], "DEADLINE_EXCEEDED");
    assert_eq!(v["job"]["failure"]["stage"], "run");
    lying.stop().await;
}

/// G4, G13: LOCAL_ONLY never reaches a remote runtime, however attractive;
/// a job cannot widen the daemon ceiling; remote use is never implicit.
#[tokio::test]
async fn g4_g13_local_only_is_structural() {
    let remote = Arc::new(FakeRemote::default());
    let h = Harness::new(|c| c.policy.max_locality = Locality::Unrestricted, vec![remote.clone(), builtin(0)]).await;

    // Default locality is LOCAL_ONLY: the local model wins despite worse numbers.
    let id = h.submit("task: {objective: secret}").await;
    let v = h.wait(&id).await;
    assert_eq!(v["result"]["model"], "builtin/echo");
    let rejected = v["job"]["decision"]["rejected"].as_array().unwrap();
    assert!(rejected.iter().any(|r| r["model"] == "cloud/frontier" && r["stage"] == "locality"));
    h.stop().await;

    // Only the remote model exists: the job is blocked, not leaked.
    let h = Harness::new(|c| c.policy.max_locality = Locality::Unrestricted, vec![remote.clone()]).await;
    let id = h.submit("task: {objective: secret}\nprivacy: {locality: lan_allowed}").await;
    assert_eq!(h.wait(&id).await["job"]["state"], "POLICY_BLOCKED");
    h.stop().await;

    // The job asks for UNRESTRICTED but the daemon ceiling is LOCAL_ONLY.
    let h = Harness::new(|_| {}, vec![remote.clone()]).await;
    let id = h.submit("task: {objective: secret}\nprivacy: {locality: unrestricted}").await;
    let v = h.wait(&id).await;
    assert_eq!(v["job"]["state"], "POLICY_BLOCKED");
    let reason = v["job"]["failure"]["plan"]["rejected"][0]["reason"].as_str().unwrap();
    assert!(reason.starts_with("daemon ceiling"), "{reason}");
    let (_, status) = h.call("GET", "/v1/status", "").await;
    assert_eq!(status["off_host_crossings"], 0);
    h.stop().await;

    assert_eq!(remote.calls(), 0, "the remote runtime must never be invoked");
}

/// G5: runtimes are replaceable — the same envelope runs on whichever
/// adapter is registered, and the agent contract does not change.
#[tokio::test]
async fn g5_runtime_is_replaceable() {
    struct Upper;
    impl ModelAdapter for Upper {
        fn runtime(&self) -> &str {
            "upper"
        }
        fn exposure(&self) -> Exposure {
            Exposure::InProcess
        }
        fn healthy(&self) -> bool {
            true
        }
        fn models(&self) -> Vec<ModelProfile> {
            let mut p = FakeRemote::default().models().remove(0);
            p.id = "upper/v1".into();
            p.runtime = "upper".into();
            p.capabilities.insert("general".into(), 0.7);
            vec![p]
        }
        fn refresh(&self) -> BoxFuture<'_, ()> {
            Box::pin(async {})
        }
        fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
            Box::pin(async move {
                Ok(InferenceOutput {
                    text: req.prompt.to_uppercase(),
                    tokens_in: 1,
                    tokens_out: 1,
                    model_digest: None,
                    bytes_sent: 0,
                    bytes_received: 0,
                    timings: None,
                })
            })
        }
        fn describe(&self) -> Value {
            serde_json::json!({"runtime": "upper"})
        }
    }
    let h = Harness::new(|_| {}, vec![Arc::new(Upper)]).await;
    let id = h.submit("task: {objective: hello}").await;
    let v = h.wait(&id).await;
    assert_eq!(v["result"]["model"], "upper/v1");
    assert!(v["result"]["output"].as_str().unwrap().starts_with("HELLO"));
    h.stop().await;
}

/// Cancellation of queued and running work.
#[tokio::test]
async fn cancellation() {
    let h = Harness::new(|_| {}, vec![builtin(100)]).await;
    let running = h.submit("task: {objective: one two three four five six seven eight nine ten}").await;
    let queued = h.submit("task: {objective: waiting}").await;
    h.wait_for_state(&running, "RUNNING").await;
    let (status, _) = h.call("POST", &format!("/v1/jobs/{queued}/cancel"), "").await;
    assert_eq!(status, 200);
    let (status, _) = h.call("POST", &format!("/v1/jobs/{running}/cancel"), "").await;
    assert_eq!(status, 202);
    assert_eq!(h.wait(&running).await["job"]["state"], "CANCELLED");
    assert_eq!(h.wait(&queued).await["job"]["state"], "CANCELLED");
    let (status, _) = h.call("POST", &format!("/v1/jobs/{queued}/cancel"), "").await;
    assert_eq!(status, 409, "terminal jobs cannot be cancelled again");
    h.stop().await;
}

/// T5 / G11 groundwork: urgent work preempts a running IDLE job, which is
/// requeued and completes afterwards.
#[tokio::test]
async fn urgent_work_preempts_idle_work() {
    let h = Harness::new(|c| c.daemon.workers = 1, vec![builtin(150)]).await;
    let idle = h.submit("task: {objective: a b c d e f g h i j k l m n o p}\nlatency: {class: idle}").await;
    h.wait_for_state(&idle, "RUNNING").await;
    let now = h.submit("task: {objective: urgent}\nlatency: {class: now}").await;
    let n = h.wait(&now).await;
    assert_eq!(n["job"]["state"], "READY");
    let i = h.wait(&idle).await;
    assert_eq!(i["job"]["state"], "READY");
    assert!(events(&i).contains(&"preempted".to_string()), "{:?}", events(&i));
    assert!(n["result"]["finished_at"].as_i64() < i["result"]["finished_at"].as_i64());
    h.stop().await;
}

/// G11 groundwork: under host pressure, yielding classes wait as
/// RESOURCE_BLOCKED while urgent work still runs.
#[tokio::test]
async fn pressure_blocks_yielding_classes_only() {
    let h = Harness::new(
        |c| {
            c.pressure.disabled = false;
            c.pressure.min_available_memory = 1.01; // always "busy"
        },
        vec![builtin(0)],
    )
    .await;
    let idle = h.submit("task: {objective: background}\nlatency: {class: speculative}").await;
    let now = h.submit("task: {objective: foreground}\nlatency: {class: now}").await;
    assert_eq!(h.wait(&now).await["job"]["state"], "READY");
    h.wait_for_state(&idle, "RESOURCE_BLOCKED").await;
    h.stop().await;
}

/// Explicit failure instead of manufactured confidence.
#[tokio::test]
async fn explicit_failures() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let id = h.submit("task: {objective: x}\nquality: {floor: 0.9}").await;
    assert_eq!(h.wait(&id).await["job"]["state"], "INSUFFICIENT_QUALITY");
    let id = h.submit("task: {objective: x}\nquality: {verification: required}").await;
    assert_eq!(h.wait(&id).await["job"]["state"], "INSUFFICIENT_QUALITY");
    let id = h.submit("task: {objective: x}\noutput: {schema: json/v1}").await;
    let v = h.wait(&id).await;
    assert_eq!(v["job"]["state"], "READY", "builtin returns JSON when asked: {v}");
    serde_json::from_str::<Value>(v["result"]["output"].as_str().unwrap()).unwrap();

    let (status, v) = h.call("POST", "/v1/jobs", "task: {objective: x}\nquality: {floor: 2}").await;
    assert_eq!((status, v["field"].as_str()), (400, Some("quality.floor")));
    let (status, _) = h.call("POST", "/v1/jobs", "task: {objective: x, input: {files: [/nonexistent/file]}}").await;
    assert_eq!(status, 400);
    h.stop().await;
}

/// G14 groundwork: every decision renders as a readable trace.
#[tokio::test]
async fn traces_explain_decisions() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let id = h.submit("task: {objective: explain me}\nlatency: {deadline: 5m}").await;
    h.wait(&id).await;
    let prefix = &id[..12];
    let (status, Value::String(text)) = h.call("GET", &format!("/v1/jobs/{prefix}/trace"), "").await else { panic!() };
    assert_eq!(status, 200);
    for needle in [
        "Requested:",
        "LOCAL_ONLY",
        "Selected:",
        "builtin/echo",
        "Reason:",
        "U = Q",
        "Remote data transmitted:",
        "NONE",
    ] {
        assert!(text.contains(needle), "trace lacks `{needle}`:\n{text}");
    }
    h.stop().await;
}

/// Two daemons cannot share a state directory.
#[tokio::test]
async fn single_daemon_per_state_dir() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let mut cfg = config(h._dir.path(), 0);
    cfg.daemon.socket = Some(h._dir.path().join("other.sock"));
    let err = tonycd::start_with(cfg, Registry::new(vec![builtin(0)], BTreeMap::new())).await.err().unwrap();
    assert!(err.to_string().contains("another tonycd"), "{err}");
    h.stop().await;
}
