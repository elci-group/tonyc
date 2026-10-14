//! Phase 7: the parts of the protocol orchestrators depend on — version
//! negotiation, idempotent submission, the durable event feed and
//! attribution by origin.

mod common;

use common::*;
use serde_json::Value;

#[tokio::test]
async fn protocol_is_negotiable() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let (status, v) = h.call("GET", "/v1/protocol", "").await;
    assert_eq!(status, 200);
    assert_eq!(v["protocol"], "tonyc/1");
    assert_eq!(v["envelope"], "tonyc.envelope/1");
    let features: Vec<&str> = v["features"].as_array().unwrap().iter().map(|f| f.as_str().unwrap()).collect();
    for f in ["submit", "try", "events", "idempotency", "origin"] {
        assert!(features.contains(&f), "missing feature {f}");
    }
    h.stop().await;
}

#[tokio::test]
async fn an_idempotency_key_submits_once_even_across_a_restart() {
    let mut h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let key = [("Idempotency-Key", "lucid:task-7:1")];
    let (s1, a) = h.call_with("POST", "/v1/jobs", "task: {objective: summarise}", &key).await;
    assert_eq!(s1, 201, "{a}");
    let id = a["id"].as_str().unwrap().to_string();

    // A retry (re-encoded as JSON) gets the same job, not a second one.
    let (s2, b) = h.call_with("POST", "/v1/jobs", r#"{"task":{"objective":"summarise"}}"#, &key).await;
    assert_eq!(s2, 200, "{b}");
    assert_eq!(b["id"], id.as_str());
    assert_eq!(b["replayed"], true);

    h.restart(|_| {}, vec![builtin(0)]).await;
    let (s3, c) = h.call_with("POST", "/v1/jobs", "task: {objective: summarise}", &key).await;
    assert_eq!((s3, c["id"].as_str()), (200, Some(id.as_str())));

    // The key cannot be reused for different work.
    let (s4, d) = h.call_with("POST", "/v1/jobs", "task: {objective: something else}", &key).await;
    assert_eq!(s4, 409, "{d}");
    assert_eq!(d["job_id"], id.as_str());

    let (s5, _) = h.call_with("POST", "/v1/jobs", "task: {objective: x}", &[("Idempotency-Key", "has space")]).await;
    assert_eq!(s5, 400);

    let (_, list) = h.call("GET", "/v1/jobs", "").await;
    // Only the first submission created a job.
    assert_eq!(list["jobs"].as_array().unwrap().len(), 1, "{list}");
    h.stop().await;
}

fn kinds(v: &Value) -> Vec<String> {
    v["events"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap().to_string()).collect()
}

#[tokio::test]
async fn the_event_feed_resumes_from_a_cursor_and_long_polls() {
    let mut h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let a = h.submit("task: {objective: one}\norigin: {system: lucid, run: t1}").await;
    h.wait(&a).await;
    let (_, feed) = h.call("GET", "/v1/events", "").await;
    let ks = kinds(&feed);
    assert_eq!(ks.first().map(String::as_str), Some("submitted"), "{feed}");
    let last = feed["events"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["terminal"], true, "{feed}");
    assert_eq!(last["state"], "READY");
    assert_eq!(last["origin"]["system"], "lucid");
    let cursor = feed["next"].as_i64().unwrap();

    // Nothing new: an immediate answer with the cursor unchanged.
    let (_, empty) = h.call("GET", &format!("/v1/events?after={cursor}"), "").await;
    assert!(kinds(&empty).is_empty());
    assert_eq!(empty["next"].as_i64(), Some(cursor));

    // The cursor survives a restart, and a long poll wakes on new work.
    h.restart(|_| {}, vec![builtin(0)]).await;
    let socket = h.socket.clone();
    let waiter = tokio::spawn(async move {
        let probe = Probe { socket };
        probe.get(&format!("/v1/events?after={cursor}&wait_ms=5000")).await
    });
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let b = h.submit("task: {objective: two}").await;
    let woke = waiter.await.unwrap();
    assert_eq!(woke["events"][0]["job_id"], b.as_str(), "{woke}");
    assert_eq!(woke["events"][0]["kind"], "submitted");

    // `after=-1` starts from now; `system` filters by origin.
    let (_, head) = h.call("GET", "/v1/events?after=-1", "").await;
    assert!(kinds(&head).is_empty());
    h.wait(&b).await;
    let (_, lucid) = h.call("GET", "/v1/events?system=LUCID", "").await;
    assert!(lucid["events"].as_array().unwrap().iter().all(|e| e["job_id"] == a.as_str()), "{lucid}");
    h.stop().await;
}

/// A second connection for concurrent requests.
struct Probe {
    socket: std::path::PathBuf,
}

impl Probe {
    async fn get(&self, path: &str) -> Value {
        use http_body_util::{BodyExt, Empty};
        use hyper::body::Bytes;
        let stream = tokio::net::UnixStream::connect(&self.socket).await.unwrap();
        let (mut tx, conn) = hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(stream)).await.unwrap();
        tokio::spawn(conn);
        let req = hyper::Request::get(path).header("host", "localhost").body(Empty::<Bytes>::new()).unwrap();
        let body = tx.send_request(req).await.unwrap().into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).unwrap()
    }
}

#[tokio::test]
async fn work_is_attributed_to_its_origin() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let a = h.submit("task: {objective: alpha}\norigin: {system: lucid, run: t1, agent: cto}").await;
    let b = h.submit("task: {objective: beta}\norigin: {system: terri}").await;
    let c = h.submit("task: {objective: gamma}").await;
    for id in [&a, &b, &c] {
        h.wait(id).await;
    }
    // The same question again from Lucid is served from the store.
    let a2 = h.submit("task: {objective: alpha}\norigin: {system: lucid, run: t2}").await;
    h.wait(&a2).await;

    let (_, lucid) = h.call("GET", "/v1/jobs?origin=lucid", "").await;
    let ids: Vec<&str> = lucid["jobs"].as_array().unwrap().iter().map(|j| j["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 2, "{lucid}");
    assert!(ids.contains(&a.as_str()) && ids.contains(&a2.as_str()));
    assert_eq!(lucid["jobs"][0]["origin"]["system"], "lucid");

    let (_, eco) = h.call("GET", "/v1/economics", "").await;
    let by = &eco["by_origin"];
    assert_eq!(by["lucid"]["inference"], 1, "{eco}");
    assert_eq!(by["lucid"]["served_from_store"], 1, "{eco}");
    assert_eq!(by["terri"]["inference"], 1);
    assert_eq!(by["unattributed"]["inference"], 1);

    let (status, v) = h.call("POST", "/v1/jobs", "task: {objective: x}\norigin: {run: r}").await;
    assert_eq!((status, v["field"].as_str()), (400, Some("origin.system")));
    h.stop().await;
}
