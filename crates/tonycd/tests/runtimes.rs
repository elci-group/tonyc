//! Phase 7: runtime adapters beyond Ollama — an OpenAI-compatible server
//! (llama.cpp, vLLM, LM Studio...) and a local program — driven through the
//! daemon exactly as agents use it.

mod common;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use common::*;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tonycd::config::{CommandConfig, CommandModel, CommandProtocol, Config, OpenAiConfig};

#[derive(Default)]
struct Seen {
    auth: Vec<Option<String>>,
    bodies: Vec<Value>,
}

/// A minimal OpenAI-compatible server, shaped like llama.cpp's.
async fn fake_server() -> (String, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let app = axum::Router::new()
        .route("/v1/models", get(|| async { Json(json!({"object": "list", "data": [{"id": "tiny-1b"}, {"id": "other-7b"}]})) }))
        .route(
            "/v1/chat/completions",
            post(|State(seen): State<Arc<Mutex<Seen>>>, headers: HeaderMap, body: String| async move {
                let body: Value = serde_json::from_str(&body).unwrap();
                let prompt = body["messages"][0]["content"].as_str().unwrap_or("").to_string();
                let mut s = seen.lock().unwrap();
                s.auth.push(headers.get("authorization").map(|v| v.to_str().unwrap().to_string()));
                s.bodies.push(body);
                Json(json!({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": format!("answer to: {}", prompt.lines().last().unwrap_or(""))}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                    "timings": {"prompt_ms": 5.0, "predicted_ms": 20.0},
                }))
            }),
        )
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen)
}

async fn start(edit: impl FnOnce(&mut Config)) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), 0);
    cfg.runtimes.builtin.enabled = false;
    edit(&mut cfg);
    let (socket, data) = (cfg.socket_path(), cfg.data_dir());
    let running = tonycd::start(cfg).await.unwrap();
    Harness { _dir: dir, data, socket, running: Some(running) }
}

#[tokio::test]
async fn an_openai_compatible_server_is_a_runtime() {
    let (url, seen) = fake_server().await;
    // SAFETY: only this test reads this variable.
    unsafe { std::env::set_var("TONYC_TEST_LLAMA_KEY", "k-123") };
    let h = start(|cfg| {
        cfg.runtimes.openai.push(OpenAiConfig {
            name: "llamacpp".into(),
            url: url.clone(),
            api_key_env: Some("TONYC_TEST_LLAMA_KEY".into()),
            models: vec!["tiny-1b".into()],
            ..Default::default()
        });
    })
    .await;

    let (_, models) = h.call("GET", "/v1/models", "").await;
    let ids: Vec<&str> =
        models["models"].as_array().unwrap().iter().map(|m| m["profile"]["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["llamacpp/tiny-1b"], "{models}");

    let id = h.submit("task: {objective: name a colour, max_tokens: 16}\nquality: {floor: 0.3}").await;
    let done = h.wait(&id).await;
    assert_eq!(done["job"]["state"], "READY", "{done}");
    assert_eq!(done["result"]["model"], "llamacpp/tiny-1b");
    assert_eq!(done["result"]["exposure"]["kind"], "loopback");
    assert_eq!(done["result"]["tokens_out"], 4);
    assert!(done["result"]["output"].as_str().unwrap().starts_with("answer to:"));
    assert_eq!(done["boundary"][0]["leaves_host"], false, "{done}");

    {
        let s = seen.lock().unwrap();
        assert_eq!(s.auth, [Some("Bearer k-123".to_string())]);
        assert_eq!(s.bodies[0]["model"], "tiny-1b");
        assert_eq!(s.bodies[0]["max_tokens"], 16);
    }
    h.stop().await;
}

fn command(name: &str, script: &str, exposure: &str) -> CommandConfig {
    CommandConfig {
        name: name.into(),
        argv: vec!["sh".into(), "-c".into(), script.into()],
        protocol: CommandProtocol::Json,
        models: vec![CommandModel {
            name: "phi".into(),
            capabilities: [("general".to_string(), 0.7)].into(),
            tokens_per_sec: 50.0,
            startup_ms: 10,
            ram_mb: 0,
        }],
        exposure: exposure.into(),
        timeout_secs: 10,
    }
}

#[tokio::test]
async fn a_local_program_is_a_runtime() {
    let h = start(|cfg| {
        cfg.runtimes.command.push(command(
            "candle",
            r#"cat >/dev/null; echo '{"text":"42","tokens_in":7,"tokens_out":1,"timings":{"load_ms":3,"prefill_ms":1,"generate_ms":2}}'"#,
            "local",
        ));
    })
    .await;
    let id = h.submit("task: {objective: what is six times seven}").await;
    let done = h.wait(&id).await;
    assert_eq!(done["job"]["state"], "READY", "{done}");
    assert_eq!(done["result"]["model"], "candle/phi");
    assert_eq!(done["result"]["output"], "42");
    assert_eq!(done["result"]["exposure"], json!({"kind": "loopback", "endpoint": "exec:candle"}));
    h.stop().await;
}

#[tokio::test]
async fn a_program_that_declares_it_forwards_data_is_kept_from_local_only_work() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("called");
    let script = format!("touch {}; cat >/dev/null; echo '{{\"text\":\"leaked\"}}'", marker.display());
    let h = start(|cfg| {
        cfg.runtimes.command.push(command("forwarder", &script, "remote:api.example.com"));
    })
    .await;
    let id = h.submit("task: {objective: secret analysis}\nprivacy: {locality: local_only}").await;
    let done = h.wait(&id).await;
    assert_ne!(done["job"]["state"], "READY", "{done}");
    assert!(!marker.exists(), "a LOCAL_ONLY job ran on a runtime that declared remote exposure");
    h.stop().await;
}

#[test]
fn runtime_names_must_be_unique_and_well_formed() {
    let mut cfg = Config::default();
    cfg.runtimes.command.push(command("ollama", "true", "local"));
    assert!(cfg.validate().unwrap_err().to_string().contains("used twice"));
    let mut cfg = Config::default();
    cfg.runtimes.command.push(command("Bad.Name", "true", "local"));
    assert!(cfg.validate().is_err());
    let mut cfg = Config::default();
    cfg.runtimes.command.push(command("ok", "true", "sideways"));
    assert!(cfg.validate().unwrap_err().to_string().contains("exposure"));
}
