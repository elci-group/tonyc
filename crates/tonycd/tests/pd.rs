//! Phase 6: Preemptive Dynamism — predicting demand, preparing answers
//! before they are asked for, and measuring whether that paid off.

mod common;

use common::*;
use serde_json::{Value, json};
use tonycd::config::Config;

fn pd_on(c: &mut Config) {
    c.pd.enabled = true;
    c.pd.interval_secs = 3600; // evaluations are triggered by the tests
    c.pd.window = "1h".into();
    c.pd.horizon = "1h".into();
}

async fn post(h: &Harness, path: &str, body: Value) -> Value {
    let (status, v) = h.call("POST", path, &body.to_string()).await;
    assert!(status < 300, "{path} → {status}: {v}");
    v
}

async fn evaluate(h: &Harness) -> Value {
    post(h, "/v1/pd/evaluate", json!({})).await
}

fn considered<'a>(e: &'a Value, objective: &str) -> &'a Value {
    e["considered"].as_array().unwrap().iter().find(|c| c["objective"] == objective).expect("considered")
}

/// A request asked for twice, whose answer went stale, is prepared again
/// before anyone asks — and when it is asked for, the answer is waiting: a
/// PD hit. PD's own work is not counted as demand.
#[tokio::test]
async fn repeated_demand_is_prepared_before_it_is_asked() {
    let h = Harness::new(pd_on, vec![builtin(0)]).await;
    let src = h.data.parent().unwrap().join("Cargo.toml");
    std::fs::write(&src, "serde = 1").unwrap();
    let req = format!("task: {{objective: list the dependencies, input: {{files: [{}]}}}}", src.display());
    h.wait(&h.submit(&req).await).await;
    let again = h.wait(&h.submit(&req).await).await;
    assert!(again["result"]["reused_from"].is_string(), "second ask is served from the store");

    std::fs::write(&src, "serde = 1\ntokio = 1").unwrap();
    post(&h, "/v1/cognition/scan", json!({})).await;

    let e = evaluate(&h).await;
    assert_eq!(e["submitted"].as_array().unwrap().len(), 1, "{e}");
    let c = considered(&e, "list the dependencies");
    assert_eq!(c["prepared"], true);
    let reason = c["reason"].as_str().unwrap();
    assert!(reason.contains("asked 2×") && reason.contains("went stale"), "{reason}");
    assert!(c["probability"].as_f64().unwrap() > 0.8);

    let spec = e["submitted"][0].as_str().unwrap();
    let v = h.wait(spec).await;
    assert_eq!(v["job"]["state"], "READY");
    assert_eq!(v["job"]["envelope"]["latency"]["class"], "SPECULATIVE");
    assert_eq!(v["job"]["envelope"]["privacy"]["class"], "LOCAL_ONLY", "same locality as the request");
    assert!(v["job"]["decision"]["utility"]["future_value"].as_f64().unwrap() > 0.0);
    assert!(v["result"]["output"].as_str().unwrap().contains("tokio"));

    // Settled as produced; not prepared twice.
    let e = evaluate(&h).await;
    assert!(considered(&e, "list the dependencies")["reason"].as_str().unwrap().contains("already prepared"));
    assert!(e["submitted"].as_array().unwrap().is_empty());

    // The agent asks: the prepared answer is there.
    let (_, t) = h.call("POST", "/v1/try", &req).await;
    assert_eq!(t["hit"], true);
    assert!(t["object"]["content"].as_str().unwrap().contains("tokio"));

    let pd = h.call("GET", "/v1/pd", "").await.1;
    assert_eq!(pd["measured"]["consumed"], 1);
    assert_eq!(pd["measured"]["preemption_yield"], 1.0);
    let econ = h.call("GET", "/v1/economics", "").await.1;
    assert_eq!(econ["preemption"]["pd_hits"], 1);
    let status = h.call("GET", "/v1/status", "").await.1;
    assert_eq!(status["cognition"]["demand"]["speculative"], 1, "PD's own run is not demand");
    assert_eq!(status["cognition"]["demand"]["inference"], 1);
    h.stop().await;
}

/// One-off requests, and requests whose answer is still fresh, are left
/// alone.
#[tokio::test]
async fn one_offs_and_fresh_answers_are_left_alone() {
    let h = Harness::new(pd_on, vec![builtin(0)]).await;
    h.wait(&h.submit("task: {objective: asked once}").await).await;
    h.wait(&h.submit("task: {objective: asked twice}").await).await;
    h.wait(&h.submit("task: {objective: asked twice}").await).await;
    let e = evaluate(&h).await;
    assert!(e["submitted"].as_array().unwrap().is_empty(), "{e}");
    assert!(considered(&e, "asked once")["reason"].as_str().unwrap().contains("needs 2"));
    assert_eq!(considered(&e, "asked twice")["reason"], "a fresh answer exists");
    h.stop().await;
}

/// `pd pause` stops speculation; `pd resume` restarts it. Try misses count
/// as demand (directive §19).
#[tokio::test]
async fn pause_and_try_misses() {
    let h = Harness::new(pd_on, vec![builtin(0)]).await;
    for _ in 0..2 {
        let (_, t) = h.call("POST", "/v1/try", "task: {objective: never computed}").await;
        assert_eq!(t["hit"], false);
    }
    post(&h, "/v1/pd/pause", json!({})).await;
    let e = evaluate(&h).await;
    assert_eq!(e["status"], "paused");
    assert!(e["submitted"].as_array().unwrap().is_empty());
    post(&h, "/v1/pd/resume", json!({})).await;
    let e = evaluate(&h).await;
    let c = considered(&e, "never computed");
    assert_eq!(c["prepared"], true, "{e}");
    assert!(
        c["reason"].as_str().unwrap().contains("2 try miss(es)")
            && c["reason"].as_str().unwrap().contains("no answer exists")
    );
    let job = e["submitted"][0].as_str().unwrap();
    h.wait(job).await;
    // Asked for before PD has next evaluated: the hit still counts.
    let (_, t) = h.call("POST", "/v1/try", "task: {objective: never computed}").await;
    assert_eq!(t["hit"], true, "the third ask finds it ready");
    let pd = h.call("GET", "/v1/pd", "").await.1;
    assert_eq!(pd["measured"]["consumed"], 1, "{pd}");
    h.stop().await;
}

/// Work that is never used is discarded, and while PD's yield is poor it
/// raises its own bar until it stops speculating on that evidence.
#[tokio::test]
async fn unused_work_is_discarded_and_pd_gets_pickier() {
    let h = Harness::new(pd_on, vec![builtin(0)]).await;
    let src = h.data.parent().unwrap().join("notes.md");
    std::fs::write(&src, "v0").unwrap();
    let req = format!("task: {{objective: summarise the notes, input: {{files: [{}]}}}}", src.display());
    h.wait(&h.submit(&req).await).await;
    h.wait(&h.submit(&req).await).await;
    for round in 1..=5 {
        std::fs::write(&src, format!("v{round}")).unwrap();
        post(&h, "/v1/cognition/scan", json!({})).await;
        let e = evaluate(&h).await;
        let job = e["submitted"][0].as_str().unwrap_or_else(|| panic!("round {round}: {e}")).to_string();
        h.wait(&job).await;
        evaluate(&h).await; // settles it as produced
        // The notes change again before anyone asks: the work was wasted.
        std::fs::write(&src, format!("v{round}b")).unwrap();
        post(&h, "/v1/cognition/scan", json!({})).await;
    }
    let e = evaluate(&h).await;
    let pd = h.call("GET", "/v1/pd", "").await.1;
    assert_eq!(pd["measured"]["discarded"], 5, "{pd}");
    assert_eq!(pd["measured"]["preemption_yield"], 0.0);
    assert_eq!(pd["measured"]["threshold"], 1.0, "0.3 ÷ a yield floored at 0.25, capped at 1");
    let reason = considered(&e, "summarise the notes")["reason"].as_str().unwrap();
    assert!(reason.contains("< threshold 1.00"), "{reason}");
    assert!(e["submitted"].as_array().unwrap().is_empty());
    let explain = h.call("GET", "/v1/pd/explain", "").await.1;
    let outcome = explain["predictions"][0]["prediction"]["outcome"].as_str().unwrap();
    assert!(outcome.contains("went stale before use"), "{outcome}");
    h.stop().await;
}

/// The value test: expected value must beat the operating cost.
#[tokio::test]
async fn work_that_does_not_pay_is_not_done() {
    let h = Harness::new(
        |c| {
            pd_on(c);
            c.pd.value_ratio = 1e12;
        },
        vec![builtin(5)],
    )
    .await;
    for _ in 0..2 {
        h.call("POST", "/v1/try", "task: {objective: expensive to prepare}").await;
    }
    let e = evaluate(&h).await;
    assert!(e["submitted"].as_array().unwrap().is_empty());
    let reason = considered(&e, "expensive to prepare")["reason"].as_str().unwrap();
    assert!(reason.contains("expected value") && reason.contains("cost"), "{reason}");
    h.stop().await;
}
