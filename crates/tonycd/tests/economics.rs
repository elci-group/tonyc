//! Phase 4: measured profiles, benchmarks, budgets, resource-aware
//! planning and the economics ledger (gate G12).

mod common;

use common::*;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tonyc_core::model::Timings;
use tonyc_core::{Exposure, Locality};
use tonycd::config::ModelOverride;
use tonycd::runtimes::ModelAdapter;

async fn get(h: &Harness, path: &str) -> Value {
    let (status, v) = h.call("GET", path, "").await;
    assert_eq!(status, 200, "{path}: {v}");
    v
}

async fn model(h: &Harness, id: &str) -> Value {
    get(h, &format!("/v1/model?id={id}")).await
}

/// §12: measured performance replaces static priors after a few runs, and
/// cold and warm starts are told apart.
#[tokio::test]
async fn measured_profiles_replace_priors() {
    let mut m = Scripted::new("timed/m", 0.7);
    m.tokens = (100, 20);
    m.timings = |n| Some(Timings { load_ms: if n == 0 { 3000 } else { 5 }, prefill_ms: 50, generate_ms: 1000 });
    let h = Harness::new(|_| {}, vec![Arc::new(m)]).await;
    let before = model(&h, "timed/m").await;
    assert_eq!(before["effective"]["source"], "static_prior");
    for i in 0..3 {
        h.wait(&h.submit(&format!("task: {{objective: measure me {i}}}")).await).await;
    }
    let after = model(&h, "timed/m").await;
    let eff = &after["effective"];
    assert_eq!(eff["source"], "observed", "{after}");
    assert_eq!(eff["evidence"]["runs"], 3);
    assert_eq!(eff["performance"]["tokens_per_sec"], 20.0);
    assert_eq!(eff["performance"]["prefill_tokens_per_sec"], 2000.0);
    assert_eq!(after["prior"]["performance"]["tokens_per_sec"], 1e6, "the prior is kept for comparison");
    assert!(after["observed"]["cold_startup_ms"].is_number());
    assert_eq!(after["recent"].as_array().unwrap().len(), 3);

    let id = h.submit("task: {objective: and again}").await;
    h.wait(&id).await;
    let (_, Value::String(trace)) = h.call("GET", &format!("/v1/jobs/{id}/trace"), "").await else { panic!() };
    assert!(trace.contains("observed profile, 3 runs"), "{trace}");
    h.stop().await;
}

fn oracle(prompt: &str) -> String {
    // Right about general knowledge, wrong about everything else.
    let p = prompt.to_lowercase();
    if p.contains("17 + 25") {
        "42".into()
    } else if p.contains("capital of france") {
        "Paris".into()
    } else if p.contains("leap year") {
        "366".into()
    } else if p.contains("exactly the word ok") {
        "OK".into()
    } else {
        "I am not sure.".into()
    }
}

/// Benchmarks score a model per capability, and the scores change what the
/// planner believes — and therefore decides.
#[tokio::test]
async fn benchmarks_measure_quality_per_capability() {
    let mut m = Scripted::new("oracle/m", 0.7);
    m.answer = oracle;
    let m = Arc::new(m);
    let h = Harness::new(|_| {}, vec![m.clone()]).await;
    let (status, run) =
        h.call("POST", "/v1/models/benchmark", &json!({"model": "oracle/m", "class": "now"}).to_string()).await;
    assert_eq!(status, 201, "{run}");
    let id = run["id"].as_str().unwrap().to_string();
    let mut done = Value::Null;
    for _ in 0..500 {
        done = get(&h, &format!("/v1/benchmarks/{id}")).await;
        if done["run"]["state"] == "COMPLETE" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(done["run"]["state"], "COMPLETE", "{done}");
    let cases = done["run"]["cases"].as_i64().unwrap();
    assert_eq!(done["run"]["passed"], 4);
    assert_eq!(done["run"]["errors"], 0, "an invalid JSON answer is a failed case, not an error");
    assert_eq!(m.calls() as i64, cases, "one inference per case");

    let eff = &model(&h, "oracle/m").await["effective"];
    let general = eff["capabilities"]["general"].as_f64().unwrap();
    let summarisation = eff["capabilities"]["summarisation"].as_f64().unwrap();
    assert!((general - (0.7 * 4.0 + 4.0) / 8.0).abs() < 1e-9, "{eff}");
    assert!((summarisation - (0.7 * 4.0) / 6.0).abs() < 1e-9, "{eff}");
    assert_eq!(eff["evidence"]["quality"]["general"]["total"], 4);
    assert_eq!(eff["evidence"]["quality"]["rust"]["prior"], 0.7, "priors come from before blending");
    assert_eq!(eff["evidence"]["quality"]["extraction"]["passed"], 0);

    // The prior (0.7) would have passed a 0.5 floor; the measurement does not.
    let v = h.wait(&h.submit("task: {objective: summarise this, capability: summarisation}").await).await;
    assert_eq!(v["job"]["state"], "INSUFFICIENT_QUALITY");
    let v = h.wait(&h.submit("task: {objective: general question}").await).await;
    assert_eq!(v["job"]["state"], "READY");
    h.stop().await;
}

/// G12: Tonyc reports estimated cost avoided, from the ledger.
#[tokio::test]
async fn g12_reports_cost_avoided() {
    let mut m = Scripted::new("local/m", 0.7);
    m.tokens = (1000, 500);
    let h = Harness::new(|_| {}, vec![Arc::new(m)]).await;
    let a = h.wait(&h.submit("task: {objective: first question}").await).await;
    h.wait(&h.submit("task: {objective: second question}").await).await;
    let r = h.wait(&h.submit("task: {objective: first question}").await).await;
    assert!(r["result"]["reused_from"].is_string());

    // Reference prices by default: £2.40 in, £12.00 out per million tokens.
    let per_job = (1000.0 * 2.40 + 500.0 * 12.00) / 1e6;
    assert!((a["ledger"]["conventional_gbp"].as_f64().unwrap() - per_job).abs() < 1e-12);
    assert_eq!(r["ledger"]["kind"], "reuse");
    assert!((r["ledger"]["conventional_gbp"].as_f64().unwrap() - per_job).abs() < 1e-12);

    let e = get(&h, "/v1/economics").await;
    assert_eq!(e["jobs"]["inference"], 2);
    assert_eq!(e["jobs"]["served_from_store"], 1);
    assert_eq!(e["jobs"]["remote_inference_avoided"], 3);
    let avoided = e["money"]["estimated_cost_avoided_gbp"].as_f64().unwrap();
    assert!((avoided - 3.0 * per_job).abs() < 1e-9, "{e}");
    assert_eq!(e["money"]["remote_spend_gbp"], 0.0);
    assert_eq!(e["preemption"]["produced"], 0, "no speculative work yet");
    assert!(e["preemption"]["preemption_yield"].is_null());

    let id = a["job"]["id"].as_str().unwrap();
    let (_, Value::String(trace)) = h.call("GET", &format!("/v1/jobs/{id}/trace"), "").await else { panic!() };
    assert!(trace.contains("Economics (estimates)") && trace.contains("estimated cost avoided"), "{trace}");
    h.stop().await;
}

fn paid() -> Scripted {
    let mut m = Scripted::new("cloud/paid", 0.95);
    m.cost_per_mtok = 1000.0;
    m.tokens = (400, 400);
    m.exposure = Exposure::Remote { host: "api.example.com".into() };
    m
}

/// §21: REMOTE_SPEND = 0 is valid and enforced; a monthly budget is spent
/// down and then holds.
#[tokio::test]
async fn remote_budget_is_enforced() {
    let open = |c: &mut tonycd::config::Config| c.policy.max_locality = Locality::Unrestricted;
    let h = Harness::new(open, vec![Arc::new(paid())]).await;
    let v = h.wait(&h.submit("task: {objective: x}\nprivacy: {locality: unrestricted}").await).await;
    assert_eq!(v["job"]["state"], "POLICY_BLOCKED");
    assert_eq!(v["job"]["failure"]["plan"]["rejected"][0]["stage"], "budget");
    h.stop().await;

    let h = Harness::new(
        |c| {
            open(c);
            c.budgets.remote_monthly_gbp = 1.0;
        },
        vec![Arc::new(paid())],
    )
    .await;
    // Estimated £0.61 (100 + 512 tokens) fits £1; actual spend is £0.80.
    let v = h.wait(&h.submit("task: {objective: x}\nprivacy: {locality: unrestricted}").await).await;
    assert_eq!(v["job"]["state"], "READY");
    assert!((v["ledger"]["spend_gbp"].as_f64().unwrap() - 0.8).abs() < 1e-9);
    let v = h.wait(&h.submit("task: {objective: y}\nprivacy: {locality: unrestricted}").await).await;
    assert_eq!(v["job"]["state"], "POLICY_BLOCKED", "£0.20 left is not enough");
    let e = get(&h, "/v1/economics").await;
    assert!((e["budgets"]["remote_monthly"]["spent_gbp"].as_f64().unwrap() - 0.8).abs() < 1e-9);
    h.stop().await;
}

/// §21: daily compute budgets per urgency class.
#[tokio::test]
async fn daily_compute_budget_holds_a_class() {
    let h = Harness::new(
        |c| {
            c.budgets.daily_compute.insert("speculative".into(), "50ms".into());
        },
        vec![builtin(30)],
    )
    .await;
    let first = h.wait(&h.submit("task: {objective: one two three}\nlatency: {class: speculative}").await).await;
    assert_eq!(first["job"]["state"], "READY");
    let second = h.submit("task: {objective: four five six}\nlatency: {class: speculative}").await;
    h.wait_for_state(&second, "RESOURCE_BLOCKED").await;
    let (_, v) = h.call("GET", &format!("/v1/jobs/{second}"), "").await;
    let reason = v["events"].as_array().unwrap().last().unwrap()["detail"]["reason"].as_str().unwrap().to_string();
    assert!(reason.contains("daily compute budget for SPECULATIVE"), "{reason}");
    assert_eq!(
        h.wait(&h.submit("task: {objective: urgent}\nlatency: {class: now}").await).await["job"]["state"],
        "READY"
    );
    h.stop().await;
}

/// Resource-aware planning: a model that does not fit is skipped; with no
/// alternative the job waits (RESOURCE_BLOCKED), it does not fail.
#[tokio::test]
async fn models_that_do_not_fit_are_skipped_or_waited_for() {
    let mut big = Scripted::new("local/big", 0.9);
    big.ram_mb = 5000;
    big.loaded = false;
    let small = Scripted::new("local/small", 0.6);
    let cap = |c: &mut tonycd::config::Config| c.budgets.memory_max_mb = 1000;
    let big: Arc<dyn ModelAdapter> = Arc::new(big);
    let h = Harness::new(cap, vec![big.clone(), Arc::new(small)]).await;
    let v = h.wait(&h.submit("task: {objective: x}").await).await;
    assert_eq!(v["result"]["model"], "local/small");
    let rejected = v["job"]["decision"]["rejected"].as_array().unwrap();
    assert!(
        rejected.iter().any(|r| r["stage"] == "resources" && r["reason"].as_str().unwrap().contains("memory budget"))
    );
    h.stop().await;

    let h = Harness::new(cap, vec![big]).await;
    let id = h.submit("task: {objective: x}").await;
    h.wait_for_state(&id, "RESOURCE_BLOCKED").await;
    h.stop().await;

    // An override can say the model is smaller than its prior suggests.
    let mut big = Scripted::new("local/big", 0.9);
    big.ram_mb = 5000;
    big.loaded = false;
    let h = Harness::new(
        |c| {
            cap(c);
            c.models.insert("local/big".into(), ModelOverride { ram_mb: Some(800), ..Default::default() });
        },
        vec![Arc::new(big)],
    )
    .await;
    assert_eq!(h.wait(&h.submit("task: {objective: x}").await).await["job"]["state"], "READY");
    h.stop().await;
}

/// R: an answer asked for before is worth more; the trace says so.
#[tokio::test]
async fn demand_history_raises_reuse_potential() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    for _ in 0..3 {
        h.call("POST", "/v1/try", "task: {objective: popular question}").await;
    }
    let v = h.wait(&h.submit("task: {objective: popular question}").await).await;
    assert!(v["job"]["decision"]["utility"]["reuse"].as_f64().unwrap() > 0.0, "{}", v["job"]["decision"]);
    let reasons = v["job"]["decision"]["reasons"].to_string();
    assert!(reasons.contains("reuse potential 1.00"), "{reasons}");
    let fresh = h.wait(&h.submit("task: {objective: never asked before}").await).await;
    assert_eq!(fresh["job"]["decision"]["utility"]["reuse"], 0.0);
    h.stop().await;
}

/// The ledger and the observations are append-only.
#[tokio::test]
async fn accounting_records_are_append_only() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    h.wait(&h.submit("task: {objective: x}").await).await;
    let db = rusqlite::Connection::open(h.data.join("state.db")).unwrap();
    let rows: i64 = db.query_row("SELECT COUNT(*) FROM ledger", [], |r| r.get(0)).unwrap();
    assert_eq!(rows, 1);
    for sql in [
        "UPDATE ledger SET spend_gbp = 0",
        "DELETE FROM ledger",
        "DELETE FROM observations",
        "UPDATE observations SET wall_ms = 0",
    ] {
        assert!(db.execute(sql, []).is_err(), "{sql} must be refused");
    }
    drop(db);
    h.stop().await;
}
