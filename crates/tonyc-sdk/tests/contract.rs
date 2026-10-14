//! The SDK against a real daemon: the whole §18–19 contract, plus the
//! orchestrator primitives.

use std::sync::Arc;
use std::time::Duration;
use tonyc_sdk::{Client, Error, Invalidate, JobState, Query, Served, TryOutcome, Urgency, Work};
use tonycd::config::{BuiltinConfig, Config};
use tonycd::runtimes::{Registry, builtin::Builtin};

struct Daemon {
    _dir: tempfile::TempDir,
    running: Option<tonycd::Running>,
    client: Client,
}

async fn daemon(ms_per_token: u64) -> Daemon {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = Config::default();
    cfg.daemon.socket = Some(dir.path().join("run/tonycd.sock"));
    cfg.daemon.data_dir = Some(dir.path().join("data"));
    cfg.daemon.tick_ms = 20;
    cfg.pressure.disabled = true;
    cfg.pd.enabled = false;
    cfg.runtimes.ollama.enabled = false;
    let b = BuiltinConfig { enabled: true, ms_per_token, quality: 0.6 };
    cfg.runtimes.builtin = b.clone();
    let socket = cfg.socket_path();
    let running =
        tonycd::start_with(cfg, Registry::new(vec![Arc::new(Builtin::new(b))], Default::default())).await.unwrap();
    Daemon { _dir: dir, running: Some(running), client: Client::unix(socket) }
}

impl Daemon {
    async fn stop(mut self) {
        self.running.take().unwrap().stop().await;
    }
}

const MINUTE: Duration = Duration::from_secs(60);

#[tokio::test]
async fn the_agent_contract() {
    let d = daemon(0).await;
    let tonyc = d.client.clone().with_origin("sdk-test");

    let p = tonyc.protocol().await.unwrap();
    assert!(p.supports("try") && p.supports("events"));

    // submit → await → result with provenance.
    let work = Work::new("list three primes").input("2 3 5 7 11").urgency(Urgency::Soon).scope("project:sdk");
    let job = tonyc.submit(&work).await.unwrap();
    assert_eq!(job.state, JobState::Queued);
    assert_eq!(job.envelope.origin.as_ref().unwrap().system, "sdk-test");
    let done = tonyc.wait(&job.id, MINUTE).await.unwrap();
    assert_eq!(done.state(), JobState::Ready);
    let result = done.result.as_ref().unwrap();
    assert_eq!(result.model, "builtin/echo");
    assert!(!result.exposure.leaves_host());
    assert!(done.output().unwrap().contains("list three primes"));

    // try: the same question is now answered from the store.
    let TryOutcome::Hit(object) = tonyc.try_existing(&work).await.unwrap() else { panic!("expected a hit") };
    assert_eq!(object.content.as_deref(), done.output());
    // query finds it by relevance and scope.
    let found = tonyc.query(&Query::text("primes").scope("project:sdk")).await.unwrap();
    assert_eq!(found[0].id, object.id);

    // invalidate the scope → the next try misses.
    let report = tonyc.invalidate(Invalidate::Scope("project:sdk".into()), Some("test")).await.unwrap();
    assert!(report.to_string().contains(&object.id), "{report}");
    assert!(tonyc.try_existing(&work).await.unwrap().hit().is_none());

    // A refused request names the field at fault.
    let err = tonyc.submit(&Work::new("x").floor(2.0)).await.unwrap_err();
    assert!(matches!(&err, Error::Api { status: 400, field: Some(f), .. } if f == "quality.floor"), "{err}");
    d.stop().await;
}

#[tokio::test]
async fn cancel_and_await_timeouts() {
    let d = daemon(200).await;
    let tonyc = &d.client;
    let job = tonyc.submit(&Work::new("a slow one with many many words to echo back")).await.unwrap();
    let pending = tonyc.await_job(&job.id, Duration::from_millis(50)).await.unwrap();
    assert!(!pending.is_terminal());
    let state = tonyc.cancel(&job.id).await.unwrap();
    assert!(state == "CANCELLED" || state == "CANCELLING", "{state}");
    let done = tonyc.wait(&job.id, MINUTE).await.unwrap();
    assert_eq!(done.state(), JobState::Cancelled);
    assert!(done.output().is_none());
    d.stop().await;
}

#[tokio::test]
async fn orchestrator_primitives() {
    let d = daemon(0).await;
    let lucid = d.client.clone().with_agent("lucid", "cto");

    // Idempotent submission.
    let work = Work::new("triage the failing build");
    let a = lucid.submit_idempotent("lucid:t1", &work).await.unwrap();
    let b = lucid.submit_idempotent("lucid:t1", &work).await.unwrap();
    assert_eq!(a.id, b.id);
    assert!(!a.replayed && b.replayed);
    let conflict = lucid.submit_idempotent("lucid:t1", &Work::new("other")).await.unwrap_err();
    assert_eq!(conflict.status(), Some(409));

    // The event feed follows the job to its end without polling the job.
    let mut cursor = 0;
    let mut terminal = None;
    for _ in 0..50 {
        let page = lucid.events(cursor, Duration::from_secs(2), Some("lucid")).await.unwrap();
        cursor = page.next;
        if let Some(e) = page.events.iter().find(|e| e.job_id == a.id && e.terminal) {
            terminal = e.state;
            break;
        }
    }
    assert_eq!(terminal, Some(JobState::Ready));

    // Three planes: a miss runs the caller's hot path; a hit does not.
    let fresh = Work::new("a question nobody asked before");
    match lucid.try_or(&fresh, async { "hot answer" }).await.unwrap() {
        Served::Hot(answer) => assert_eq!(answer, "hot answer"),
        Served::Existing(_) => panic!("nothing should exist yet"),
    }
    match lucid.try_or(&work, async { unreachable!("the store has this answer") }).await.unwrap() {
        Served::Existing(o) => assert_eq!(o.job_id.as_deref(), Some(a.id.as_str())),
        Served::Hot(()) => unreachable!(),
    }
    d.stop().await;
}
