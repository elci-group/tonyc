//! Phase 3: the SSH compute fabric over the real API (gates G6, G7, G13).
//!
//! `ssh` is replaced by a script that runs the remote command locally (and
//! logs its arguments), `ssh-keyscan` by one that presents a real generated
//! host key, and the node's runtime by an in-process fake Ollama that
//! records the exact bytes it receives. `ssh-keygen` is the real one.

mod common;

use axum::Router;
use axum::routing::{get, post as route_post};
use common::*;
use serde_json::{Value, json};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tonyc_core::Locality;
use tonycd::config::{BuiltinConfig, Config};
use tonycd::runtimes::ModelAdapter;
use tonycd::runtimes::builtin::Builtin;

/// The fake node runtime: Ollama's two endpoints.
struct FakeRuntime {
    url: String,
    received: Arc<Mutex<Vec<Vec<u8>>>>,
}

async fn fake_runtime() -> FakeRuntime {
    let received: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
    let rec = received.clone();
    let app = Router::new()
        .route(
            "/api/tags",
            get(|| async {
                axum::Json(json!({"models": [{"name": "fake:1b", "size": 800_000_000u64, "digest": "fakedigest",
                    "details": {"parameter_size": "1B"}}]}))
            }),
        )
        .route(
            "/api/generate",
            route_post(move |body: axum::body::Bytes| {
                let rec = rec.clone();
                async move {
                    rec.lock().unwrap().push(body.to_vec());
                    let v: Value = serde_json::from_slice(&body).unwrap();
                    let prompt = v["prompt"].as_str().unwrap_or("");
                    let first: String = prompt.lines().next().unwrap_or("").chars().take(60).collect();
                    axum::Json(
                        json!({"response": format!("node says: {first}"), "prompt_eval_count": 7, "eval_count": 3}),
                    )
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    FakeRuntime { url, received }
}

/// Fake ssh tooling in its own directory.
struct Fixture {
    dir: tempfile::TempDir,
    fingerprint: String,
}

impl Fixture {
    fn new() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let key = d.join("hostkey");
        let ok = std::process::Command::new("ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-f"])
            .arg(&key)
            .status()
            .unwrap();
        assert!(ok.success());
        let pubkey = std::fs::read_to_string(key.with_extension("pub")).unwrap();
        let mut parts = pubkey.split_whitespace();
        let line = format!("[127.0.0.1]:2222 {} {}", parts.next().unwrap(), parts.next().unwrap());
        std::fs::write(d.join("hostkey.line"), format!("{line}\n")).unwrap();
        let fp = std::process::Command::new("ssh-keygen")
            .args(["-l", "-f"])
            .arg(key.with_extension("pub"))
            .output()
            .unwrap();
        let fingerprint = String::from_utf8_lossy(&fp.stdout).split_whitespace().nth(1).unwrap().to_string();

        script(
            &d.join("ssh"),
            &format!(
                r#"#!/usr/bin/env bash
echo "$*" >> {log}
for a in "$@"; do
  if [ "$a" = "-G" ]; then printf 'hostname 127.0.0.1\nport 2222\nuser tester\n'; exit 0; fi
done
if [ -e {down} ]; then echo "ssh: connect to host 127.0.0.1 port 2222: Connection refused" >&2; exit 255; fi
while [ $# -gt 0 ]; do
  case "$1" in
    -o|-p|-i|-F|-l) shift 2 ;;
    --) shift; break ;;
    -*) shift ;;
    *) break ;;
  esac
done
shift
exec sh -c "$1"
"#,
                log = d.join("ssh.log").display(),
                down = d.join("DOWN").display()
            ),
        );
        script(&d.join("ssh-keyscan"), &format!("#!/bin/sh\ncat {}\n", d.join("hostkey.line").display()));
        std::fs::write(d.join("user_known_hosts"), "").unwrap();
        Fixture { dir, fingerprint }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn configure(&self, c: &mut Config) {
        c.fabric.ssh_program = self.path("ssh").display().to_string();
        c.fabric.keyscan_program = self.path("ssh-keyscan").display().to_string();
        c.fabric.user_known_hosts = Some(self.path("user_known_hosts"));
        c.fabric.connect_timeout_secs = 2;
    }

    fn down(&self, down: bool) {
        if down {
            std::fs::write(self.path("DOWN"), "").unwrap();
        } else {
            let _ = std::fs::remove_file(self.path("DOWN"));
        }
    }

    fn ssh_log(&self) -> String {
        std::fs::read_to_string(self.path("ssh.log")).unwrap_or_default()
    }
}

fn script(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

async fn post(h: &Harness, path: &str, body: Value) -> (u16, Value) {
    h.call("POST", path, &body.to_string()).await
}

async fn add_forge(h: &Harness, fx: &Fixture, rt: &FakeRuntime, extra: Value) -> Value {
    let mut body = json!({"name": "forge", "ssh": "tester@forge", "runtime_url": rt.url, "host_key": fx.fingerprint});
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    let (status, v) = post(h, "/v1/nodes", body).await;
    assert_eq!(status, 201, "{v}");
    v
}

async fn allow_ssh(h: &Harness, class: &str) {
    let (status, v) = post(h, "/v1/policy", json!({"max_locality": class})).await;
    assert_eq!(status, 200, "{v}");
}

const ON_FORGE: &str = "privacy: {locality: ssh_allowlist, ssh_allowlist: [forge]}\nquality: {floor: 0.3}";

fn weak_local() -> Arc<dyn ModelAdapter> {
    Arc::new(Builtin::new(BuiltinConfig { enabled: true, ms_per_token: 0, quality: 0.35 }))
}

/// G6: an authorised SSH node executes work, under strict SSH settings, and
/// exactly what crossed the boundary is recorded.
#[tokio::test]
async fn g6_authorised_node_executes_work() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(|c| fx.configure(c), vec![]).await;
    let node = add_forge(&h, &fx, &rt, json!({})).await;
    assert_eq!(node["healthy"], true, "{node}");
    assert_eq!(node["models"], json!(["node.forge/fake:1b"]));
    assert_eq!(node["capabilities"]["curl"], true);
    let known_hosts = std::fs::read_to_string(h.data.join("known_hosts")).unwrap();
    assert!(known_hosts.starts_with("[127.0.0.1]:2222 ssh-ed25519 "), "{known_hosts}");

    allow_ssh(&h, "ssh_allowlist").await;
    let id = h.submit(&format!("task: {{objective: compile the release notes}}\n{ON_FORGE}")).await;
    let v = h.wait(&id).await;
    assert_eq!(v["job"]["state"], "READY", "{v}");
    assert_eq!(v["result"]["model"], "node.forge/fake:1b");
    assert_eq!(v["result"]["exposure"], json!({"kind": "ssh", "node": "forge"}));
    assert!(v["result"]["output"].as_str().unwrap().starts_with("node says: compile the release notes"));

    // The audit holds exactly the bytes the node's runtime received.
    let (_, audit) = h.call("GET", &format!("/v1/audit?job={id}&payload=true"), "").await;
    let sent =
        audit["crossings"].as_array().unwrap().iter().find(|c| c["bytes_sent"].as_u64() > Some(0)).unwrap().clone();
    let received = rt.received.lock().unwrap()[0].clone();
    assert_eq!(sent["payload"].as_str().unwrap().as_bytes(), received.as_slice());
    assert_eq!(sent["payload_blake3"], blake3::hash(&received).to_hex().as_str());
    assert_eq!(sent["bytes_sent"], received.len());

    // Strict, non-forwarding SSH on every connection.
    let log = fx.ssh_log();
    for opt in [
        "BatchMode=yes",
        "StrictHostKeyChecking=yes",
        &format!("UserKnownHostsFile={}", h.data.join("known_hosts").display()),
        "GlobalKnownHostsFile=/dev/null",
        "ForwardAgent=no",
        "ForwardX11=no",
        "ClearAllForwardings=yes",
        "PermitLocalCommand=no",
    ] {
        // Each invocation starts on a line of its own with `-T`; the remote
        // script that follows may span several lines.
        let invocations: Vec<&str> = log.lines().filter(|l| l.starts_with("-T ")).collect();
        assert!(invocations.len() >= 2, "{log}");
        assert!(invocations.iter().all(|l| l.contains(opt)), "missing {opt}:\n{log}");
    }

    let (_, Value::String(trace)) = h.call("GET", &format!("/v1/jobs/{id}/trace"), "").await else { panic!() };
    assert!(trace.contains("SSH node forge") && trace.contains("payload blake3"), "{trace}");
    let (_, verified) = h.call("GET", "/v1/audit/verify", "").await;
    assert_eq!(verified["violations"], json!([]));
    assert!(verified["checked"].as_u64().unwrap() >= 2);
    h.stop().await;
}

/// G13: a registered node is not used until the ceiling is raised
/// explicitly, nor by jobs that do not ask for it.
#[tokio::test]
async fn g13_nodes_need_an_explicit_ceiling_and_request() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(|c| fx.configure(c), vec![]).await;
    add_forge(&h, &fx, &rt, json!({})).await;
    let id = h.submit(&format!("task: {{objective: x}}\n{ON_FORGE}")).await;
    let v = h.wait(&id).await;
    assert_eq!(v["job"]["state"], "POLICY_BLOCKED");
    assert!(v["job"]["failure"]["plan"]["rejected"][0]["reason"].as_str().unwrap().contains("daemon ceiling"));

    allow_ssh(&h, "unrestricted").await;
    let v = h.wait(&h.submit("task: {objective: y}\nquality: {floor: 0.3}").await).await;
    assert_eq!(v["job"]["state"], "POLICY_BLOCKED", "the default LOCAL_ONLY job must not use the node");
    assert!(rt.received.lock().unwrap().is_empty());
    let (_, status) = h.call("GET", "/v1/status", "").await;
    assert_eq!(status["off_host_crossings"], 0);
    h.stop().await;
}

/// Shared nodes are used only when a job names them.
#[tokio::test]
async fn shared_nodes_must_be_named() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(|c| fx.configure(c), vec![]).await;
    add_forge(&h, &fx, &rt, json!({"policy": "shared"})).await;
    allow_ssh(&h, "unrestricted").await;
    let v =
        h.wait(&h.submit("task: {objective: x}\nprivacy: {locality: unrestricted}\nquality: {floor: 0.3}").await).await;
    assert_eq!(v["job"]["state"], "POLICY_BLOCKED");
    let v = h.wait(&h.submit(&format!("task: {{objective: x}}\n{ON_FORGE}")).await).await;
    assert_eq!(v["job"]["state"], "READY");
    h.stop().await;
}

/// Host keys are pinned, never trusted on first use.
#[tokio::test]
async fn host_keys_are_verified_not_assumed() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(|c| fx.configure(c), vec![]).await;
    let base = json!({"name": "forge", "ssh": "tester@forge", "runtime_url": rt.url});

    let (status, v) = post(&h, "/v1/nodes", base.clone()).await;
    assert_eq!(status, 400);
    let msg = v["error"].as_str().unwrap();
    assert!(msg.contains("not trusted yet") && msg.contains(&fx.fingerprint), "{msg}");

    let mut wrong = base.clone();
    wrong["host_key"] = json!("SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
    let (status, v) = post(&h, "/v1/nodes", wrong).await;
    assert_eq!(status, 400);
    assert!(v["error"].as_str().unwrap().contains("matches"), "{v}");
    let (_, nodes) = h.call("GET", "/v1/nodes", "").await;
    assert!(nodes["nodes"].as_array().unwrap().is_empty());

    // A host the user already trusts needs no fingerprint.
    std::fs::copy(fx.path("hostkey.line"), fx.path("user_known_hosts")).unwrap();
    let (status, v) = post(&h, "/v1/nodes", base).await;
    assert_eq!(status, 201, "{v}");
    h.stop().await;
}

/// G7: when the node fails, the job falls back to another admissible domain.
#[tokio::test]
async fn g7_node_failure_falls_back() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(|c| fx.configure(c), vec![weak_local()]).await;
    add_forge(&h, &fx, &rt, json!({})).await;
    allow_ssh(&h, "ssh_allowlist").await;
    fx.down(true);
    let v = h.wait(&h.submit(&format!("task: {{objective: fall back please}}\n{ON_FORGE}")).await).await;
    assert_eq!(v["job"]["state"], "READY", "{v}");
    assert_eq!(v["result"]["model"], "builtin/echo");
    let kinds = events(&v);
    assert!(kinds.contains(&"fallback".to_string()), "{kinds:?}");
    let boundary = v["boundary"].as_array().unwrap();
    assert!(boundary.iter().any(|b| b["summary"].as_str().unwrap().contains("failed")));
    let (_, node) = h.call("GET", "/v1/nodes/forge", "").await;
    assert_eq!(node["healthy"], false);
    h.stop().await;
}

/// G7: `on_node_failure: fail` ends the job NODE_UNREACHABLE even when a
/// fallback exists; with fallback but nowhere to go, it ends the same way.
#[tokio::test]
async fn g7_node_failure_fails_by_policy() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(|c| fx.configure(c), vec![weak_local()]).await;
    add_forge(&h, &fx, &rt, json!({})).await;
    allow_ssh(&h, "ssh_allowlist").await;
    fx.down(true);
    let v = h
        .wait(
            &h.submit(&format!("task: {{objective: strict}}\n{ON_FORGE}\nresources: {{on_node_failure: fail}}")).await,
        )
        .await;
    assert_eq!(v["job"]["state"], "NODE_UNREACHABLE");
    assert_eq!(v["job"]["failure"]["policy"], "fail");
    h.stop().await;

    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(|c| fx.configure(c), vec![]).await;
    add_forge(&h, &fx, &rt, json!({})).await;
    allow_ssh(&h, "ssh_allowlist").await;
    fx.down(true);
    let v = h.wait(&h.submit(&format!("task: {{objective: nowhere}}\n{ON_FORGE}")).await).await;
    assert_eq!(v["job"]["state"], "NODE_UNREACHABLE");
    assert!(v["job"]["failure"]["reason"].as_str().unwrap().contains("no fallback"), "{v}");
    h.stop().await;
}

/// A node whose runtime is down registers, but is unhealthy and unused.
#[tokio::test]
async fn node_without_runtime_is_registered_unhealthy() {
    let fx = Fixture::new();
    let h = Harness::new(|c| fx.configure(c), vec![]).await;
    let (status, v) = post(
        &h,
        "/v1/nodes",
        json!({"name": "cold", "ssh": "tester@cold", "runtime_url": "http://127.0.0.1:9", "host_key": fx.fingerprint}),
    )
    .await;
    assert_eq!(status, 201, "{v}");
    assert_eq!(v["healthy"], false);
    assert!(v["last_error"].as_str().unwrap().contains("runtime"), "{v}");
    // An unreachable node cannot be added at all.
    fx.down(true);
    let (status, _) = post(
        &h,
        "/v1/nodes",
        json!({"name": "gone", "ssh": "tester@gone", "runtime_url": "http://127.0.0.1:9", "host_key": fx.fingerprint}),
    )
    .await;
    assert_eq!(status, 502);
    h.stop().await;
}

/// Work is dispatched across domains: with the local slot busy, the next
/// job goes to the free node.
#[tokio::test]
async fn work_spreads_across_free_domains() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let slow_local: Arc<dyn ModelAdapter> =
        Arc::new(Builtin::new(BuiltinConfig { enabled: true, ms_per_token: 200, quality: 0.6 }));
    let h = Harness::new(|c| fx.configure(c), vec![slow_local]).await;
    add_forge(&h, &fx, &rt, json!({})).await;
    allow_ssh(&h, "ssh_allowlist").await;
    let a = h.submit(&format!("task: {{objective: a b c d e f g h i j}}\n{ON_FORGE}")).await;
    h.wait_for_state(&a, "RUNNING").await;
    let b = h.submit(&format!("task: {{objective: second job}}\n{ON_FORGE}")).await;
    let (va, vb) = (h.wait(&a).await, h.wait(&b).await);
    assert_eq!(va["result"]["model"], "builtin/echo");
    assert_eq!(vb["result"]["model"], "node.forge/fake:1b");
    assert!(vb["job"]["decision"]["rejected"].as_array().unwrap().iter().any(|r| r["stage"] == "capacity"));
    h.stop().await;
}

/// Nodes persist across restarts and are probed again on start.
#[tokio::test]
async fn nodes_persist_and_can_be_removed() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let mut h = Harness::new(|c| fx.configure(c), vec![]).await;
    add_forge(&h, &fx, &rt, json!({"slots": 2})).await;
    allow_ssh(&h, "ssh_allowlist").await;
    h.restart(|c| fx.configure(c), vec![]).await;
    let mut healthy = false;
    for _ in 0..200 {
        let (_, n) = h.call("GET", "/v1/nodes/forge", "").await;
        if n["healthy"] == true {
            assert_eq!(n["slots"], 2);
            healthy = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(healthy, "node was not re-probed after restart");
    let (_, p) = h.call("GET", "/v1/policy", "").await;
    assert_eq!(p["ceiling"]["class"], "SSH_ALLOWLIST");
    assert_eq!(p["sources"]["max_locality"], "set");
    assert_eq!(h.wait(&h.submit(&format!("task: {{objective: z}}\n{ON_FORGE}")).await).await["job"]["state"], "READY");

    let (status, _) = h.call("DELETE", "/v1/nodes/forge", "").await;
    assert_eq!(status, 200);
    assert!(!std::fs::read_to_string(h.data.join("known_hosts")).unwrap().contains("ssh-ed25519"));
    let (_, m) = h.call("GET", "/v1/models", "").await;
    assert!(m["models"].as_array().unwrap().is_empty());
    h.stop().await;
}

/// `policy set` cannot produce a default wider than the ceiling, and can
/// be reset to the configuration.
#[tokio::test]
async fn policy_set_is_validated_and_resettable() {
    let h = Harness::new(|_| {}, vec![builtin(0)]).await;
    let (status, _) = post(&h, "/v1/policy", json!({"default_locality": "unrestricted"})).await;
    assert_eq!(status, 400);
    let (status, _) =
        post(&h, "/v1/policy", json!({"max_locality": "lan_allowed", "default_locality": "lan_allowed"})).await;
    assert_eq!(status, 200);
    let v = h.wait(&h.submit("task: {objective: q}").await).await;
    assert_eq!(v["job"]["envelope"]["privacy"]["class"], "LAN_ALLOWED");
    let (_, p) = post(&h, "/v1/policy", json!({"reset": true})).await;
    assert_eq!(p["ceiling"]["class"], Locality::LocalOnly.as_str());
    assert_eq!(p["sources"]["default_locality"], "config");
    h.stop().await;
}

/// Resource-aware dispatch: while this host is under pressure, background
/// work runs on an idle node it may use, and waits if there is none.
#[tokio::test]
async fn background_work_moves_to_an_idle_node_under_host_pressure() {
    let (fx, rt) = (Fixture::new(), fake_runtime().await);
    let h = Harness::new(
        |c| {
            fx.configure(c);
            c.pressure.disabled = false;
            c.pressure.min_available_memory = 1.01; // this host: always busy
            c.pressure.busy_load_per_core = 1000.0; // nodes: never busy
            c.pressure.battery_is_busy = false;
            c.pressure.max_temperature_c = 1000.0;
        },
        vec![builtin(0)],
    )
    .await;
    add_forge(&h, &fx, &rt, json!({})).await;
    allow_ssh(&h, "ssh_allowlist").await;

    let v = h
        .wait(&h.submit(&format!("task: {{objective: background work}}\nlatency: {{class: idle}}\n{ON_FORGE}")).await)
        .await;
    assert_eq!(v["result"]["model"], "node.forge/fake:1b", "{v}");
    let rejected = v["job"]["decision"]["rejected"].as_array().unwrap();
    assert!(rejected.iter().any(|r| r["model"] == "builtin/echo" && r["stage"] == "pressure"), "{rejected:?}");

    // Urgent work still runs here; local-only background work waits.
    let now = h.wait(&h.submit("task: {objective: urgent}\nlatency: {class: now}").await).await;
    assert_eq!(now["result"]["model"], "builtin/echo");
    let id = h.submit("task: {objective: local background}\nlatency: {class: idle}").await;
    h.wait_for_state(&id, "RESOURCE_BLOCKED").await;
    h.stop().await;
}
