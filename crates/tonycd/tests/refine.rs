//! Phase 5: refinement — validation with measured validators, revision,
//! escalation, deterministic checks, tie-breaks, explicit failure.

mod common;

use common::*;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

const PASS: &str = r#"{"verdict": "pass", "issues": []}"#;
const FAIL: &str = r#"{"verdict": "fail", "issues": ["the answer is wrong"]}"#;

/// Judges like a sound validator: an answer that starts "wrong" fails.
fn judge(p: &str) -> Option<String> {
    p.contains("Proposed answer:").then(|| if p.contains("Proposed answer:\nwrong") { FAIL } else { PASS }.to_string())
}

fn says_right(p: &str) -> String {
    judge(p).unwrap_or_else(|| "right answer".into())
}

fn says_wrong(p: &str) -> String {
    judge(p).unwrap_or_else(|| "wrong answer".into())
}

fn model(id: &str, q: f64, energy: f64, answer: fn(&str) -> String) -> Arc<Scripted> {
    let mut m = Scripted::new(id, q);
    m.energy_wh_per_ktok = energy;
    m.answer = answer;
    Arc::new(m)
}

fn roles(v: &Value) -> Vec<String> {
    v["passes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            format!(
                "{}:{}:{}",
                p["role"].as_str().unwrap(),
                p["model"].as_str().unwrap(),
                p["verdict"].as_str().unwrap_or("-")
            )
        })
        .collect()
}

async fn object(h: &Harness, v: &Value) -> Value {
    let id = v["result"]["object_id"].as_str().unwrap();
    h.call("GET", &format!("/v1/cognition/{id}"), "").await.1
}

/// `verification: required` is met by an independent validator; the
/// verdict moves confidence by Bayes' rule; unverified cognition does not
/// satisfy a request that requires verification, verified cognition does.
#[tokio::test]
async fn verification_required_is_met_by_an_independent_validator() {
    // The drafter is cheap; the validator costs energy, so it does not draft.
    let drafter = model("draft/m", 0.7, 0.0, says_right);
    let validator = model("judge/m", 0.8, 100.0, says_right);
    let h = Harness::new(|_| {}, vec![drafter.clone(), validator.clone()]).await;

    let plain = h.wait(&h.submit("task: {objective: what is the answer}").await).await;
    assert_eq!(plain["job"]["state"], "READY");
    assert_eq!(object(&h, &plain).await["object"]["validation"], "unverified");
    assert_eq!(validator.calls(), 0, "no refinement was asked for");

    let v = h.wait(&h.submit("task: {objective: what is the answer}\nquality: {verification: required}").await).await;
    assert_eq!(v["job"]["state"], "READY", "{v}");
    assert!(v["job"]["cognition"]["direct_skipped"][0].as_str().unwrap().contains("unverified"));
    assert_eq!(roles(&v), ["draft:draft/m:-", "validate:judge/m:pass"]);
    let o = object(&h, &v).await;
    assert_eq!(o["object"]["validation"], "verified");
    let expected = 0.7 * 0.8 / (0.7 * 0.8 + 0.3 * 0.2);
    assert!((o["object"]["confidence"].as_f64().unwrap() - expected).abs() < 1e-9, "{o}");
    assert!(events(&v).contains(&"refining".to_string()));

    let again =
        h.wait(&h.submit("task: {objective: what is the answer}\nquality: {verification: required}").await).await;
    assert_eq!(again["result"]["reused_from"], v["result"]["object_id"], "verified cognition satisfies it");

    let id = v["job"]["id"].as_str().unwrap();
    let (_, Value::String(trace)) = h.call("GET", &format!("/v1/jobs/{id}/trace"), "").await else { panic!() };
    assert!(trace.contains("Refinement:") && trace.contains("validate") && trace.contains("judge/m"), "{trace}");
    h.stop().await;
}

/// A failed validation leads to a revision by a stronger model (escalation),
/// which is validated again. Every pass is accounted for in the ledger.
#[tokio::test]
async fn failed_validation_escalates_and_revalidates() {
    let cheap = model("cheap/m", 0.5, 0.0, says_wrong);
    let judge_m = model("judge/m", 0.85, 100.0, says_right);
    let expert = model("expert/m", 0.9, 200.0, says_right);
    let h = Harness::new(|_| {}, vec![cheap.clone(), judge_m.clone(), expert.clone()]).await;
    let v = h.wait(&h.submit("task: {objective: hard question}\nquality: {floor: 0.8, refine: always}").await).await;
    assert_eq!(v["job"]["state"], "READY", "{v}");
    assert_eq!(v["result"]["model"], "expert/m");
    assert_eq!(v["result"]["output"], "right answer");
    assert_eq!(
        roles(&v),
        [
            "draft:cheap/m:-",
            "validate:expert/m:fail",
            "validate:judge/m:fail",
            "revise:expert/m:-",
            "validate:judge/m:pass",
        ]
    );
    let revise = &v["passes"][3];
    assert_eq!(revise["detail"]["escalated"], true);
    assert!(revise["detail"]["issues"].to_string().contains("the answer is wrong"));
    let o = object(&h, &v).await;
    assert_eq!(o["object"]["validation"], "verified");
    assert!(o["object"]["confidence"].as_f64().unwrap() >= 0.8);

    // Five inference passes of (10, 20) tokens; a premium API would have
    // made one call of the draft's size.
    let l = &v["ledger"];
    assert_eq!((l["tokens_in"].as_u64(), l["tokens_out"].as_u64()), (Some(50), Some(100)));
    assert!((l["conventional_gbp"].as_f64().unwrap() - (10.0 * 2.40 + 20.0 * 12.00) / 1e6).abs() < 1e-12);
    assert_eq!(cheap.calls() + judge_m.calls() + expert.calls(), 5);
    h.stop().await;
}

/// When even passing validation could not reach the floor, the job fails up
/// front — before spending any inference.
#[tokio::test]
async fn unreachable_floors_fail_before_inference() {
    let a = model("a/m", 0.7, 0.0, says_right);
    let b = model("b/m", 0.8, 0.0, says_right);
    let h = Harness::new(|_| {}, vec![a.clone(), b.clone()]).await;
    let v = h.wait(&h.submit("task: {objective: x}\nquality: {floor: 0.99, verification: required}").await).await;
    assert_eq!(v["job"]["state"], "INSUFFICIENT_QUALITY");
    let reason = v["job"]["failure"]["reason"].as_str().unwrap();
    assert!(reason.contains("would reach only"), "{reason}");
    assert_eq!(a.calls() + b.calls(), 0);
    h.stop().await;
}

fn london_then_paris(p: &str) -> String {
    if p.contains("Problems found") { "Paris".into() } else { "London".into() }
}

fn always_london(_: &str) -> String {
    "London".into()
}

/// Deterministic checks: a failing answer is revised; checks define
/// verification when no validator is involved; when revision cannot fix
/// it, the job fails with its evidence and best attempt — and the ledger
/// records the wasted cost.
#[tokio::test]
async fn deterministic_checks_drive_revision() {
    let m = model("geo/m", 0.7, 0.0, london_then_paris);
    let h = Harness::new(|_| {}, vec![m.clone()]).await;
    let checks = "quality: {checks: [{contains: [paris]}]}";
    let v = h.wait(&h.submit(&format!("task: {{objective: capital of France}}\n{checks}")).await).await;
    assert_eq!(v["job"]["state"], "READY", "{v}");
    assert_eq!(v["result"]["output"], "Paris");
    assert_eq!(roles(&v), ["draft:geo/m:-", "check:checks:fail", "revise:geo/m:-", "check:checks:pass"]);
    assert_eq!(object(&h, &v).await["object"]["validation"], "verified");
    h.stop().await;

    let stubborn = model("stubborn/m", 0.7, 0.0, always_london);
    let h = Harness::new(|_| {}, vec![stubborn.clone()]).await;
    let v = h.wait(&h.submit(&format!("task: {{objective: capital of France}}\n{checks}")).await).await;
    assert_eq!(v["job"]["state"], "INSUFFICIENT_QUALITY");
    assert_eq!(v["job"]["failure"]["best_attempt"], "London");
    assert_eq!(v["job"]["failure"]["rounds"], 2);
    assert_eq!(stubborn.calls(), 3, "a draft and two revisions");
    let e = h.call("GET", "/v1/economics", "").await.1;
    assert!(e["money"]["estimated_cost_avoided_gbp"].as_f64().unwrap() == 0.0, "nothing was avoided: {e}");

    // refine: never still enforces checks, but does not revise.
    let v = h
        .wait(
            &h.submit("task: {objective: capital of Spain}\nquality: {refine: never, checks: [{contains: [madrid]}]}")
                .await,
        )
        .await;
    assert_eq!(v["job"]["state"], "INSUFFICIENT_QUALITY");
    assert_eq!(stubborn.calls(), 4);
    h.stop().await;
}

fn yes(p: &str) -> String {
    if p.contains("Proposed answer:") { PASS.into() } else { "yes".into() }
}

fn no(p: &str) -> String {
    if p.contains("Proposed answer:") { FAIL.into() } else { "no".into() }
}

/// (Each test model has its own runtime: model ids are `<runtime>/<model>`
/// and a runtime name identifies one adapter.)
///
/// Validators that disagree are reconciled by a tie-break; the majority
/// decides verification.
#[tokio::test]
async fn disagreement_is_settled_by_a_tie_break() {
    let drafter = model("draft/m", 0.6, 0.0, says_right);
    let v1 = model("v1/m", 0.9, 100.0, yes);
    let v2 = model("v2/m", 0.85, 100.0, no);
    let v3 = model("v3/m", 0.8, 100.0, yes);
    let h = Harness::new(|_| {}, vec![drafter, v1, v2, v3]).await;
    let v = h.wait(&h.submit("task: {objective: contested}\nquality: {refine: always}").await).await;
    assert_eq!(v["job"]["state"], "READY", "{v}");
    assert_eq!(roles(&v), ["draft:draft/m:-", "validate:v1/m:pass", "validate:v2/m:fail", "tie_break:v3/m:pass"]);
    assert_eq!(object(&h, &v).await["object"]["validation"], "verified");
    h.stop().await;
}

/// Validator accuracy is measured, not assumed: a validator that passes
/// everything is benchmarked, its specificity falls, and its pass no longer
/// carries a job over the floor.
#[tokio::test]
async fn measured_validator_accuracy_matters() {
    let drafter = model("draft/m", 0.55, 0.0, says_right);
    let yes_man = model("yesman/m", 0.9, 100.0, yes);
    let h = Harness::new(|_| {}, vec![drafter, yes_man]).await;
    let job = "task: {objective: anything}\nquality: {floor: 0.8, verification: required}\nreuse: {cache: false}";
    let before = h.wait(&h.submit(job).await).await;
    assert_eq!(before["job"]["state"], "READY", "assumed accuracy 0.9 lets one pass through: {before}");

    let (status, run) = h
        .call(
            "POST",
            "/v1/models/benchmark",
            &json!({"model": "yesman/m", "capability": "validation", "class": "now"}).to_string(),
        )
        .await;
    assert_eq!(status, 201, "{run}");
    let id = run["id"].as_str().unwrap();
    for _ in 0..500 {
        if h.call("GET", &format!("/v1/benchmarks/{id}"), "").await.1["run"]["state"] == "COMPLETE" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let m = h.call("GET", "/v1/model?id=yesman/m", "").await.1;
    let caps = &m["effective"]["capabilities"];
    let specificity = caps["validation.specificity"].as_f64().unwrap();
    assert!(caps["validation.sensitivity"].as_f64().unwrap() > 0.9 && specificity < 0.3, "{caps}");

    let after = h.wait(&h.submit(job).await).await;
    assert_eq!(after["job"]["state"], "INSUFFICIENT_QUALITY", "{after}");
    let first_validation = after["passes"][1]["confidence"].as_f64().unwrap();
    assert!(first_validation < 0.8, "a yes-man's pass is weak evidence: {first_validation}");
    h.stop().await;
}
