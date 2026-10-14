//! Conformance: the daemon against the language-neutral protocol document
//! (protocol/tonyc.openapi.json). Every documented operation is routed,
//! every routed endpoint is classified, and live responses validate against
//! the documented schemas — so the document cannot drift from the daemon.

mod common;

use common::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn spec() -> Value {
    serde_json::from_str(include_str!("../../../protocol/tonyc.openapi.json")).unwrap()
}

/// Validates `v` against a JSON Schema subset: $ref, type, const, enum,
/// properties, required, additionalProperties: false, items, oneOf,
/// minimum/maximum, minLength/maxLength.
fn validate(spec: &Value, schema: &Value, v: &Value, at: &str, errors: &mut Vec<String>) {
    if let Some(r) = schema.get("$ref").and_then(Value::as_str) {
        let name = r.trim_start_matches("#/components/schemas/");
        return validate(spec, &spec["components"]["schemas"][name], v, at, errors);
    }
    if let Some(options) = schema.get("oneOf").and_then(Value::as_array) {
        let ok = options.iter().filter(|o| {
            let mut e = vec![];
            validate(spec, o, v, at, &mut e);
            e.is_empty()
        });
        if ok.count() != 1 {
            errors.push(format!("{at}: matches no single oneOf option: {v}"));
        }
        return;
    }
    if let Some(t) = schema.get("type") {
        let types: Vec<&str> = match t {
            Value::String(s) => vec![s.as_str()],
            Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
            _ => vec![],
        };
        let actual = match v {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };
        if !types.iter().any(|t| *t == actual || (*t == "number" && actual == "integer")) {
            errors.push(format!("{at}: expected {types:?}, got {actual} ({v})"));
            return;
        }
    }
    if let Some(c) = schema.get("const")
        && c != v
    {
        errors.push(format!("{at}: expected {c}, got {v}"));
    }
    if let Some(e) = schema.get("enum").and_then(Value::as_array)
        && !e.contains(v)
    {
        errors.push(format!("{at}: {v} is not one of {e:?}"));
    }
    if let Some(n) = v.as_f64() {
        if schema.get("minimum").and_then(Value::as_f64).is_some_and(|m| n < m) {
            errors.push(format!("{at}: {n} below minimum"));
        }
        if schema.get("maximum").and_then(Value::as_f64).is_some_and(|m| n > m) {
            errors.push(format!("{at}: {n} above maximum"));
        }
    }
    if let Some(s) = v.as_str() {
        let len = s.chars().count() as u64;
        if schema.get("minLength").and_then(Value::as_u64).is_some_and(|m| len < m) {
            errors.push(format!("{at}: shorter than minLength"));
        }
    }
    if let Some(o) = v.as_object() {
        let props = schema.get("properties").and_then(Value::as_object);
        for r in schema.get("required").and_then(Value::as_array).into_iter().flatten() {
            if !o.contains_key(r.as_str().unwrap()) {
                errors.push(format!("{at}: missing required `{r}`"));
            }
        }
        for (k, val) in o {
            match props.and_then(|p| p.get(k)) {
                Some(s) => validate(spec, s, val, &format!("{at}.{k}"), errors),
                None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                    errors.push(format!("{at}: undocumented field `{k}`"))
                }
                None => {}
            }
        }
    }
    if let (Some(items), Some(a)) = (schema.get("items"), v.as_array()) {
        for (i, item) in a.iter().enumerate() {
            validate(spec, items, item, &format!("{at}[{i}]"), errors);
        }
    }
}

#[test]
fn the_validator_catches_drift() {
    let spec = spec();
    let schema = json!({"$ref": "#/components/schemas/FeedEvent"});
    let good = json!({"seq": 1, "at": 2, "job_id": "j", "kind": "submitted", "state": "QUEUED", "terminal": false, "origin": null});
    let mut errors = vec![];
    validate(&spec, &schema, &good, "ok", &mut errors);
    assert!(errors.is_empty(), "{errors:?}");
    for bad in [
        json!({"seq": 1, "at": 2, "job_id": "j", "kind": "submitted"}),
        json!({"seq": "1", "at": 2, "job_id": "j", "kind": "k", "terminal": false}),
        json!({"seq": 1, "at": 2, "job_id": "j", "kind": "k", "terminal": false, "state": "DONE"}),
    ] {
        let mut errors = vec![];
        validate(&spec, &schema, &bad, "bad", &mut errors);
        assert!(!errors.is_empty(), "{bad} passed");
    }
    let mut errors = vec![];
    let request = json!({"task": {"objective": "x", "colour": "red"}});
    validate(&spec, &json!({"$ref": "#/components/schemas/WorkRequest"}), &request, "req", &mut errors);
    assert!(errors.iter().any(|e| e.contains("colour")), "{errors:?}");
}

/// The documented response schema for an operation and status.
fn response_schema(spec: &Value, path: &str, method: &str, status: u16) -> Value {
    let op = &spec["paths"][path][method];
    let mut r = op["responses"][status.to_string()].clone();
    assert!(!r.is_null(), "{method} {path} does not document status {status}");
    if let Some(name) = r.get("$ref").and_then(Value::as_str) {
        r = spec["components"]["responses"][name.trim_start_matches("#/components/responses/")].clone();
    }
    r["content"]["application/json"]["schema"].clone()
}

fn conforms(spec: &Value, path: &str, method: &str, status: u16, body: &Value) {
    let schema = response_schema(spec, path, method, status);
    let mut errors = vec![];
    validate(spec, &schema, body, &format!("{method} {path} {status}"), &mut errors);
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[tokio::test]
async fn live_responses_match_the_protocol_document() {
    let spec = spec();
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let (s, v) = h.call("GET", "/v1/protocol", "").await;
    conforms(&spec, "/v1/protocol", "get", s, &v);

    let work = r#"{"task": {"objective": "conformance", "input": {"text": "a b c"}}, "reuse": {"corpus": "project:conf"}, "origin": {"system": "conformance", "run": "r1"}}"#;
    let key = [("Idempotency-Key", "conf-1")];
    let (s, v) = h.call_with("POST", "/v1/jobs", work, &key).await;
    assert_eq!(s, 201);
    conforms(&spec, "/v1/jobs", "post", s, &v);
    let id = v["id"].as_str().unwrap().to_string();
    let (s, v) = h.call_with("POST", "/v1/jobs", work, &key).await;
    assert_eq!(s, 200);
    conforms(&spec, "/v1/jobs", "post", s, &v);
    let (s, v) = h.call_with("POST", "/v1/jobs", r#"{"task": {"objective": "other"}}"#, &key).await;
    assert_eq!(s, 409);
    conforms(&spec, "/v1/jobs", "post", s, &v);
    let (s, v) = h.call("POST", "/v1/jobs", r#"{"task": {"objective": "x"}, "quality": {"floor": 9}}"#).await;
    assert_eq!(s, 400);
    conforms(&spec, "/v1/jobs", "post", s, &v);

    let (s, v) = h.call("GET", &format!("/v1/jobs/{id}/await?timeout_ms=10000"), "").await;
    assert_eq!(s, 200);
    conforms(&spec, "/v1/jobs/{id}/await", "get", s, &v);
    let (s, v) = h.call("GET", &format!("/v1/jobs/{id}"), "").await;
    conforms(&spec, "/v1/jobs/{id}", "get", s, &v);
    let (s, v) = h.call("GET", "/v1/jobs/tny_doesnotexist", "").await;
    assert_eq!(s, 404);
    conforms(&spec, "/v1/jobs/{id}", "get", s, &v);
    let (s, v) = h.call("GET", "/v1/jobs?origin=conformance", "").await;
    conforms(&spec, "/v1/jobs", "get", s, &v);

    let (s, v) = h.call("POST", "/v1/try", work).await;
    assert_eq!(v["hit"], true);
    conforms(&spec, "/v1/try", "post", s, &v);
    let (s, v) = h.call("POST", "/v1/try", r#"{"task": {"objective": "never asked"}}"#).await;
    assert_eq!(v["hit"], false);
    conforms(&spec, "/v1/try", "post", s, &v);
    let (s, v) = h.call("GET", "/v1/cognition?text=conformance&scope=project:conf", "").await;
    assert!(!v["objects"].as_array().unwrap().is_empty());
    conforms(&spec, "/v1/cognition", "get", s, &v);

    let (s, v) = h.call("GET", "/v1/events?after=0&limit=5", "").await;
    conforms(&spec, "/v1/events", "get", s, &v);
    let (s, v) = h.call("GET", "/v1/events?system=conformance", "").await;
    conforms(&spec, "/v1/events", "get", s, &v);

    let (s, v) = h.call("POST", "/v1/invalidate", r#"{"scope": "project:conf", "reason": "conformance"}"#).await;
    conforms(&spec, "/v1/invalidate", "post", s, &v);
    assert_eq!(v["jobs"][0], id.as_str());

    // Cancel: a queued job (the worker is kept busy by a slow one).
    h.stop().await;
    let h = Harness::new(|c| c.daemon.workers = 1, vec![builtin(200)]).await;
    let _busy = h.submit("task: {objective: slow one two three four five six seven eight}").await;
    let queued = h.submit("task: {objective: queued}").await;
    let (s, v) = h.call("POST", &format!("/v1/jobs/{queued}/cancel"), "").await;
    conforms(&spec, "/v1/jobs/{id}/cancel", "post", s, &v);
    let (s, v) = h.call("POST", &format!("/v1/jobs/{queued}/cancel"), "").await;
    assert_eq!(s, 409);
    conforms(&spec, "/v1/jobs/{id}/cancel", "post", s, &v);
    let (s, v) = h.call("GET", &format!("/v1/jobs/{queued}/await?timeout_ms=1"), "").await;
    conforms(&spec, "/v1/jobs/{id}/await", "get", s, &v);
    h.stop().await;
}

/// Operator endpoints, documented in docs/PROTOCOL.md rather than the
/// agent-contract document.
const OPERATOR_ROUTES: &[&str] = &[
    "/v1/status",
    "/v1/host",
    "/v1/policy",
    "/v1/nodes",
    "/v1/nodes/{name}",
    "/v1/nodes/{name}/probe",
    "/v1/audit",
    "/v1/audit/verify",
    "/v1/models",
    "/v1/models/refresh",
    "/v1/model",
    "/v1/models/benchmark",
    "/v1/benchmarks",
    "/v1/benchmarks/{id}",
    "/v1/economics",
    "/v1/pd",
    "/v1/pd/explain",
    "/v1/pd/evaluate",
    "/v1/pd/pause",
    "/v1/pd/resume",
    "/v1/jobs/{id}/trace",
    "/v1/cognition/scan",
    "/v1/cognition/{id}",
    "/v1/cache/prune",
    "/v1/corpus",
    "/v1/corpus/forget",
];

#[test]
fn every_route_is_documented_somewhere() {
    let spec = spec();
    let documented: BTreeSet<String> = spec["paths"].as_object().unwrap().keys().cloned().collect();
    let routed: BTreeSet<String> = include_str!("../src/api.rs")
        .split(".route(\"")
        .skip(1)
        .map(|rest| rest.split('"').next().unwrap().to_string())
        .collect();
    let operator: BTreeSet<String> = OPERATOR_ROUTES.iter().map(|s| s.to_string()).collect();
    let docs = include_str!("../../../docs/PROTOCOL.md");
    for r in &routed {
        assert!(
            documented.contains(r) || operator.contains(r),
            "route {r} is in neither the protocol document nor the operator list"
        );
        let mentioned = docs.contains(&format!("`{r}`")) || docs.contains(&format!("`{r}?"));
        assert!(mentioned, "docs/PROTOCOL.md does not mention `{r}`");
    }
    for d in &documented {
        assert!(routed.contains(d), "the protocol document describes {d}, which the daemon does not route");
    }
}

/// The documented request schema has exactly the fields the daemon accepts.
#[test]
fn the_request_schema_matches_the_request_type() {
    let spec = spec();
    let schema = &spec["components"]["schemas"]["WorkRequest"];
    let actual = serde_json::to_value(tonyc_core::WorkRequest::default()).unwrap();
    let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<BTreeSet<_>>();
    assert_eq!(keys(&actual), keys(&schema["properties"]));
    for (section, fields) in actual.as_object().unwrap() {
        if fields.is_object() {
            assert_eq!(keys(fields), keys(&schema["properties"][section]["properties"]), "section {section}");
        }
    }
    // A full serialised request — what the SDKs send — validates.
    let mut errors = vec![];
    let mut sent = actual.clone();
    sent["task"]["objective"] = json!("x");
    validate(&spec, schema, &sent, "WorkRequest", &mut errors);
    let example: Value = serde_yaml::from_str(include_str!("../../../examples/dependency-drift.yaml")).unwrap();
    validate(&spec, schema, &example, "example", &mut errors);
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
