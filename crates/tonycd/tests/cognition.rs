//! Phase 2: the Cognitive Store, over the real API (gates G8, G9, G10).

mod common;

use common::*;
use serde_json::{Value, json};
use std::sync::Arc;
use tonyc_core::Locality;
use tonycd::runtimes::ModelAdapter;

fn file(h: &Harness, name: &str, text: &str) -> String {
    let p = h.data.parent().unwrap().join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

async fn post(h: &Harness, path: &str, body: Value) -> Value {
    let (status, v) = h.call("POST", path, &body.to_string()).await;
    assert!(status < 300, "{path} → {status}: {v}");
    v
}

async fn try_raw(h: &Harness, request: &str) -> Value {
    let (status, v) = h.call("POST", "/v1/try", request).await;
    assert_eq!(status, 200, "{v}");
    v
}

async fn object(h: &Harness, id: &str) -> Value {
    let (status, v) = h.call("GET", &format!("/v1/cognition/{id}"), "").await;
    assert_eq!(status, 200, "{v}");
    v
}

fn produced(v: &Value) -> String {
    v["result"]["object_id"].as_str().expect("job produced an object").to_string()
}

/// G10: existing cognition eliminates a redundant inference.
#[tokio::test]
async fn g10_existing_cognition_eliminates_inference() {
    let model = Counting::new(builtin(0));
    let h = Harness::new(|_| {}, vec![model.clone()]).await;
    let a = h.wait(&h.submit("task: {objective: Analyse repository dependency architecture}").await).await;
    assert_eq!(a["job"]["state"], "READY");
    assert_eq!(model.calls(), 1);

    let b = h
        .wait(&h.submit("task: {objective: Analyse repository dependency architecture}\nlatency: {class: now}").await)
        .await;
    assert_eq!(b["job"]["state"], "READY");
    assert_eq!(model.calls(), 1, "the second request must not run inference");
    assert_eq!(b["result"]["reused_from"], a["result"]["object_id"]);
    assert_eq!(b["result"]["output"], a["result"]["output"]);
    assert_eq!(b["result"]["runtime"], "cognitive-store");
    assert!(b["boundary"].as_array().unwrap().is_empty(), "nothing was dispatched anywhere");
    assert_eq!(events(&b), ["submitted", "planning", "reused"]);

    let id = b["job"]["id"].as_str().unwrap();
    let (_, Value::String(trace)) = h.call("GET", &format!("/v1/jobs/{id}/trace"), "").await else { panic!() };
    assert!(trace.contains("satisfied by existing cognition"), "{trace}");
    assert!(trace.contains("no inference"), "{trace}");

    let (_, status) = h.call("GET", "/v1/status", "").await;
    assert_eq!(status["cognition"]["demand"]["reuse_hit"], 1);
    assert_eq!(status["cognition"]["fresh"], 1);
    h.stop().await;
}

/// G9: a source change invalidates dependent cognition (by scan), the job
/// goes STALE, and the next request refreshes from the stale answer.
#[tokio::test]
async fn g9_source_change_invalidates_and_refreshes() {
    let model = Counting::new(builtin(0));
    let h = Harness::new(|_| {}, vec![model.clone()]).await;
    let src = file(&h, "Cargo.toml", "[dependencies]\nserde = \"1\"\n");
    let req = format!("task: {{objective: list dependencies, input: {{files: [{src}]}}}}");
    let a = h.wait(&h.submit(&req).await).await;
    let obj_a = produced(&a);
    assert_eq!(object(&h, &obj_a).await["object"]["freshness"]["policy"], "source_hash");

    std::fs::write(&src, "[dependencies]\nserde = \"1\"\ntokio = \"1\"\n").unwrap();
    let scan = post(&h, "/v1/cognition/scan", json!({})).await;
    assert_eq!(scan["invalidated"]["objects"], json!([obj_a]));
    let a_id = a["job"]["id"].as_str().unwrap();
    assert_eq!(h.state(a_id).await, "STALE");
    let o = object(&h, &obj_a).await;
    assert_eq!(o["object"]["state"], "STALE");
    assert!(o["object"]["stale_reason"].as_str().unwrap().contains("changed"));

    let b = h.wait(&h.submit(&req).await).await;
    assert_eq!(model.calls(), 2, "stale cognition must not be served");
    assert!(b["result"]["output"].as_str().unwrap().contains("tokio"));
    let context = &b["job"]["cognition"]["context"][0];
    assert_eq!(context["relation"], "refresh_of");
    assert_eq!(context["changed_sources"], json!([src]));
    let obj_b = produced(&b);
    let ob = object(&h, &obj_b).await;
    assert_eq!(ob["object"]["state"], "FRESH");
    assert_eq!(ob["ancestors"][0]["id"], obj_a);
    assert_eq!(ob["ancestors"][0]["relation"], "refresh_of");
    h.stop().await;
}

/// The background scanner finds changed sources without being asked.
#[tokio::test]
async fn background_scan_marks_jobs_stale() {
    let h = Harness::new(|c| c.cognition.scan_interval_secs = 1, vec![builtin(0)]).await;
    let src = file(&h, "README.md", "first");
    let job = h.wait(&h.submit(&format!("task: {{objective: summarise, input: {{files: [{src}]}}}}")).await).await;
    std::fs::write(&src, "second").unwrap();
    h.wait_for_state(job["job"]["id"].as_str().unwrap(), "STALE").await;
    h.stop().await;
}

/// Freshness is verified at use time, not only by the scanner.
#[tokio::test]
async fn stale_cognition_never_masquerades_between_scans() {
    let model = Counting::new(builtin(0));
    let h = Harness::new(|c| c.cognition.scan_interval_secs = 3600, vec![model.clone()]).await;
    let src = file(&h, "notes.txt", "version one");
    let req = format!("task: {{objective: summarise, input: {{files: [{src}]}}}}");
    let a = h.wait(&h.submit(&req).await).await;
    std::fs::write(&src, "version two").unwrap();
    let b = h.wait(&h.submit(&req).await).await;
    assert_eq!(model.calls(), 2);
    assert!(b["result"]["reused_from"].is_null());
    assert!(b["result"]["output"].as_str().unwrap().contains("version two"));
    assert_eq!(object(&h, &produced(&a)).await["object"]["state"], "STALE");

    // try() is checked the same way.
    std::fs::write(&src, "version three").unwrap();
    let t = try_raw(&h, &req).await;
    assert_eq!(t["hit"], false);
    h.stop().await;
}

/// Context reuse and DEPENDENCY_INVALIDATED: an object built on another goes
/// stale with it, and so does its job.
#[tokio::test]
async fn dependency_invalidation_cascades() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let src = file(&h, "Cargo.lock", "serde tokio axum hyper");
    let a = h
        .wait(
            &h.submit(&format!("task: {{objective: Analyse the crate dependency graph, input: {{files: [{src}]}}}}"))
                .await,
        )
        .await;
    let b = h
        .wait(
            &h.submit(&format!(
                "task: {{objective: Which crate dependencies are likely unnecessary, input: {{files: [{src}]}}}}"
            ))
            .await,
        )
        .await;
    let (obj_a, obj_b) = (produced(&a), produced(&b));
    let ctx = &b["job"]["cognition"]["context"][0];
    assert_eq!((ctx["id"].as_str(), ctx["relation"].as_str()), (Some(obj_a.as_str()), Some("context")));

    let r = post(&h, "/v1/invalidate", json!({"object": obj_a, "reason": "upstream audit changed"})).await;
    assert_eq!(r["objects"], json!([obj_a, obj_b]));
    let ob = object(&h, &obj_b).await;
    assert!(ob["object"]["stale_reason"].as_str().unwrap().contains(&format!("context {obj_a} went stale")));
    for job in [&a, &b] {
        assert_eq!(h.state(job["job"]["id"].as_str().unwrap()).await, "STALE");
    }
    h.stop().await;
}

/// §19 try(): hits return cognition immediately; misses are recorded as demand.
#[tokio::test]
async fn try_returns_existing_cognition_or_records_demand() {
    let model = Counting::new(builtin(0));
    let h = Harness::new(|_| {}, vec![model.clone()]).await;
    let miss = try_raw(&h, "task: {objective: classify these build logs}").await;
    assert_eq!(miss["hit"], false);
    h.wait(&h.submit("task: {objective: classify these build logs}").await).await;
    let hit = try_raw(&h, "task: {objective: classify these build logs}").await;
    assert_eq!(hit["hit"], true);
    assert!(hit["object"]["content"].as_str().unwrap().contains("classify"));
    assert_eq!(model.calls(), 1, "try never runs inference");
    let (_, status) = h.call("GET", "/v1/status", "").await;
    assert_eq!(status["cognition"]["demand"]["try_miss"], 1);
    assert_eq!(status["cognition"]["demand"]["try_hit"], 1);
    h.stop().await;
}

/// §18 query(): ranked by relevance; scope and kind filters.
#[tokio::test]
async fn query_ranks_existing_cognition() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    h.wait(&h.submit("task: {objective: Explain the linker failure in the release build}").await).await;
    h.wait(&h.submit("task: {objective: Summarise dependency drift across workspace crates}\noutput: {kind: summary}\nreuse: {corpus: 'project:tonyc'}").await).await;
    let (_, v) = h.call("GET", "/v1/cognition?text=which%20dependencies%20drifted", "").await;
    let objects = v["objects"].as_array().unwrap();
    assert!(objects[0]["objective"].as_str().unwrap().contains("dependency drift"), "{v}");
    let (_, v) = h.call("GET", "/v1/cognition?kind=summary", "").await;
    assert_eq!(v["objects"].as_array().unwrap().len(), 1);
    let (_, v) = h.call("GET", "/v1/cognition?scope=project:tonyc", "").await;
    assert_eq!(v["objects"][0]["scope"], "project:tonyc");
    let (status, _) = h.call("GET", "/v1/cognition?kind=poem", "").await;
    assert_eq!(status, 400);
    h.stop().await;
}

/// TTL, event and manual freshness; cache off; scope isolation; floors.
#[tokio::test]
async fn freshness_policies_and_reuse_rules() {
    let model = Counting::new(builtin(0));
    let h = Harness::new(|_| {}, vec![model.clone()]).await;

    let ttl = "task: {objective: ttl job}\nreuse: {ttl: 300ms}";
    h.wait(&h.submit(ttl).await).await;
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    let again = h.wait(&h.submit(ttl).await).await;
    assert!(again["result"]["reused_from"].is_null(), "expired cognition is not reused");

    let ev = "task: {objective: event job}\nreuse: {freshness: event, events: [deps-updated]}";
    let e = h.wait(&h.submit(ev).await).await;
    let r = post(&h, "/v1/invalidate", json!({"event": "deps-updated"})).await;
    assert_eq!(r["objects"], json!([produced(&e)]));

    let before = model.calls();
    let off = "task: {objective: no cache}\nreuse: {cache: false}";
    let x = h.wait(&h.submit(off).await).await;
    assert!(x["result"]["object_id"].is_null(), "cache off produces no object");
    h.wait(&h.submit(off).await).await;
    assert_eq!(model.calls(), before + 2);

    // Same request, different scope: not visible, so inference runs.
    let before = model.calls();
    h.wait(&h.submit("task: {objective: scoped}\nreuse: {corpus: 'project:a'}").await).await;
    h.wait(&h.submit("task: {objective: scoped}\nreuse: {corpus: 'project:b'}").await).await;
    assert_eq!(model.calls(), before + 2);
    // User-scope cognition is visible to project jobs.
    h.wait(&h.submit("task: {objective: general knowledge}").await).await;
    let p = h.wait(&h.submit("task: {objective: general knowledge}\nreuse: {corpus: 'project:a'}").await).await;
    assert!(p["result"]["reused_from"].is_string());

    // An object below the new request's floor is not reused (builtin is 0.6).
    let low = h.wait(&h.submit("task: {objective: general knowledge}\nquality: {floor: 0.7}").await).await;
    assert_eq!(low["job"]["state"], "INSUFFICIENT_QUALITY");
    let skipped = low["job"]["cognition"]["direct_skipped"][0].as_str().unwrap();
    assert!(skipped.contains("below floor"), "{skipped}");
    h.stop().await;
}

/// Sensitive corpus sources structurally narrow a job to LOCAL_ONLY.
#[tokio::test]
async fn sensitive_sources_force_local_execution() {
    let remote = Arc::new(FakeRemote::default());
    let local = builtin(0);
    let h = Harness::new(|c| c.policy.max_locality = Locality::Unrestricted, vec![remote.clone(), local]).await;
    let open_file = file(&h, "public.md", "public notes");
    let secret_file = file(&h, "payroll.csv", "alice,100000");
    post(&h, "/v1/corpus", json!({"paths": [secret_file], "scope": "project:hr", "sensitivity": "sensitive"})).await;

    let open = h
        .wait(
            &h.submit(&format!(
                "task: {{objective: summarise, input: {{files: [{open_file}]}}}}\nprivacy: {{locality: unrestricted}}"
            ))
            .await,
        )
        .await;
    assert_eq!(open["result"]["model"], "cloud/frontier", "control: unrestricted work may go remote");

    let hr = h
        .wait(
            &h.submit(&format!(
                "task: {{objective: summarise, input: {{files: [{secret_file}]}}}}\nprivacy: {{locality: unrestricted}}"
            ))
            .await,
        )
        .await;
    assert_eq!(hr["result"]["model"], "builtin/echo");
    assert_eq!(hr["job"]["cognition"]["narrowed"]["to"], "LOCAL_ONLY");
    assert!(
        remote.prompts.lock().unwrap().iter().all(|p| !p.contains("alice")),
        "sensitive content never left the host"
    );
    h.stop().await;

    let remote = Arc::new(FakeRemote::default());
    let h = Harness::new(|c| c.policy.max_locality = Locality::Unrestricted, vec![remote.clone()]).await;
    let secret_file = file(&h, "payroll.csv", "alice,100000");
    post(&h, "/v1/corpus", json!({"paths": [secret_file], "sensitivity": "secret"})).await;
    let v = h
        .wait(
            &h.submit(&format!(
                "task: {{objective: summarise, input: {{files: [{secret_file}]}}}}\nprivacy: {{locality: unrestricted}}"
            ))
            .await,
        )
        .await;
    assert_eq!(v["job"]["state"], "POLICY_BLOCKED");
    assert_eq!(remote.calls(), 0);
    h.stop().await;
}

/// Cognition produced under LOCAL_ONLY never rides along as context into a
/// job that may leave the host.
#[tokio::test]
async fn local_cognition_is_not_context_for_wider_jobs() {
    let remote = Arc::new(FakeRemote::default());
    let local: Arc<dyn ModelAdapter> = builtin(0);
    let h = Harness::new(|c| c.policy.max_locality = Locality::Unrestricted, vec![remote.clone(), local]).await;
    let src = file(&h, "design.md", "queue scheduler design");
    h.wait(
        &h.submit(&format!(
            "task: {{objective: review the scheduler design, input: {{text: PRIVATEMARK, files: [{src}]}}}}"
        ))
        .await,
    )
    .await;
    let wide = h
        .wait(&h.submit(&format!("task: {{objective: critique the scheduler design, input: {{files: [{src}]}}}}\nprivacy: {{locality: unrestricted}}")).await)
        .await;
    assert_eq!(wide["result"]["model"], "cloud/frontier");
    assert!(wide["job"]["cognition"]["context"].is_null(), "{}", wide["job"]["cognition"]);
    assert!(remote.prompts.lock().unwrap().iter().all(|p| !p.contains("PRIVATEMARK")));
    h.stop().await;
}

/// Forgetting a corpus scope purges derived cognition and redacts results.
#[tokio::test]
async fn corpus_forget_purges_derived_cognition() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let dir = h.data.parent().unwrap().join("repo");
    std::fs::create_dir_all(dir.join(".git")).unwrap();
    std::fs::write(dir.join("main.rs"), "fn main() { println!(\"tonyc\") }").unwrap();
    std::fs::write(dir.join(".git/HEAD"), "ref").unwrap();
    let added = post(&h, "/v1/corpus", json!({"paths": [dir], "scope": "repository:demo"})).await;
    assert_eq!(added["added"], 1, "hidden directories are skipped");
    let main = dir.join("main.rs").to_string_lossy().into_owned();
    let job = h.wait(&h.submit(&format!("task: {{objective: explain, input: {{files: [{main}]}}}}")).await).await;
    let obj = produced(&job);

    let r = post(&h, "/v1/corpus/forget", json!({"target": "repository:demo"})).await;
    assert_eq!(r["purged_objects"], json!([obj]));
    let (status, _) = h.call("GET", &format!("/v1/cognition/{obj}"), "").await;
    assert_eq!(status, 404);
    let id = job["job"]["id"].as_str().unwrap();
    let (_, v) = h.call("GET", &format!("/v1/jobs/{id}"), "").await;
    assert_eq!(v["job"]["state"], "STALE");
    assert!(v["result"]["output"].as_str().unwrap().starts_with("[forgotten"));
    let (_, v) = h.call("GET", "/v1/corpus", "").await;
    assert!(v["entries"].as_array().unwrap().is_empty());
    h.stop().await;
}

/// G8: an object's full derivation is inspectable, and lineage events are
/// append-only.
#[tokio::test]
async fn g8_lineage_is_complete() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let src = file(&h, "api.rs", "pub fn submit() {}");
    let a = h
        .wait(&h.submit(&format!("task: {{objective: document the public api, input: {{files: [{src}]}}}}")).await)
        .await;
    let b = h
        .wait(
            &h.submit(&format!("task: {{objective: document the public api examples, input: {{files: [{src}]}}}}"))
                .await,
        )
        .await;
    let o = object(&h, &produced(&b)[..12]).await;
    let obj = &o["object"];
    assert_eq!(obj["job_id"], b["job"]["id"]);
    assert_eq!(obj["model"], "builtin/echo");
    assert_eq!(obj["model_digest"], "builtin-echo-v1");
    assert_eq!(obj["sources"][0]["path"], src);
    assert_eq!(obj["semantic_key"], b["job"]["envelope"]["semantic_hash"]);
    assert_eq!(obj["privacy"]["class"], "LOCAL_ONLY");
    assert_eq!(o["ancestors"][0]["id"], produced(&a));
    assert_eq!(o["ancestors"][0]["job_id"], a["job"]["id"]);
    assert_eq!(o["events"][0]["kind"], "created");
    assert_eq!(o["fresh_now"], true);
    h.stop().await;
}

/// Pruning removes stale objects only.
#[tokio::test]
async fn prune_removes_only_stale_objects() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let keep = produced(&h.wait(&h.submit("task: {objective: keep me}").await).await);
    let drop = produced(&h.wait(&h.submit("task: {objective: drop me}").await).await);
    post(&h, "/v1/invalidate", json!({"object": drop})).await;
    let r = post(&h, "/v1/cache/prune", json!({})).await;
    assert_eq!(r["objects"], json!([drop]));
    assert_eq!(object(&h, &keep).await["object"]["state"], "FRESH");
    h.stop().await;
}
