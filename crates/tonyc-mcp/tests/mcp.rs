//! The MCP adapter, driven as an MCP client would drive it: over stdio, as a
//! separate process, against a real daemon.

use serde_json::{Value, json};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tonycd::config::{BuiltinConfig, Config};
use tonycd::runtimes::{Registry, builtin::Builtin};

async fn daemon(dir: &std::path::Path) -> (tonycd::Running, std::path::PathBuf) {
    let mut cfg = Config::default();
    cfg.daemon.socket = Some(dir.join("run/tonycd.sock"));
    cfg.daemon.data_dir = Some(dir.join("data"));
    cfg.daemon.tick_ms = 20;
    cfg.pressure.disabled = true;
    cfg.pd.enabled = false;
    cfg.runtimes.ollama.enabled = false;
    let b = BuiltinConfig { enabled: true, ms_per_token: 0, quality: 0.6 };
    cfg.runtimes.builtin = b.clone();
    let socket = cfg.socket_path();
    let running =
        tonycd::start_with(cfg, Registry::new(vec![Arc::new(Builtin::new(b))], Default::default())).await.unwrap();
    (running, socket)
}

struct Mcp {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    next: u64,
}

impl Mcp {
    fn spawn(socket: &std::path::Path) -> Mcp {
        let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_tonyc-mcp"))
            .env("TONYC_SOCKET", socket)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        Mcp { child, stdin, lines, next: 0 }
    }

    async fn send(&mut self, msg: Value) {
        self.stdin.write_all(format!("{msg}\n").as_bytes()).await.unwrap();
    }

    async fn request(&mut self, method: &str, params: Value) -> Value {
        self.next += 1;
        let id = self.next;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).await;
        loop {
            let line = tokio::time::timeout(std::time::Duration::from_secs(20), self.lines.next_line())
                .await
                .expect("an answer in time")
                .unwrap()
                .expect("the server is still running");
            let v: Value = serde_json::from_str(&line).unwrap();
            if v["id"] == id {
                return v;
            }
        }
    }

    async fn tool(&mut self, name: &str, args: Value) -> Value {
        let r = self.request("tools/call", json!({"name": name, "arguments": args})).await;
        r["result"].clone()
    }
}

#[tokio::test]
async fn an_mcp_agent_defers_reuses_and_collects_work() {
    let dir = tempfile::tempdir().unwrap();
    let (running, socket) = daemon(dir.path()).await;
    let mut mcp = Mcp::spawn(&socket);

    let init = mcp
        .request(
            "initialize",
            json!({"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test-agent", "version": "1"}}),
        )
        .await;
    assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
    assert!(init["result"]["capabilities"]["tools"].is_object());
    mcp.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await;

    let tools = mcp.request("tools/list", json!({})).await;
    let names: Vec<&str> =
        tools["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        ["tonyc_try", "tonyc_submit", "tonyc_await", "tonyc_status", "tonyc_query", "tonyc_cancel", "tonyc_invalidate"]
    );

    // Nothing exists yet.
    let miss = mcp.tool("tonyc_try", json!({"objective": "summarise the release notes", "input": "v2 adds X"})).await;
    assert_eq!(miss["structuredContent"]["hit"], false, "{miss}");

    // Defer it, then collect it.
    let sub = mcp
        .tool("tonyc_submit", json!({"objective": "summarise the release notes", "input": "v2 adds X", "deadline": "1h", "idempotency_key": "agent-1"}))
        .await;
    assert_eq!(sub["isError"], false, "{sub}");
    let job = sub["structuredContent"]["job_id"].as_str().unwrap().to_string();
    let again = mcp.tool("tonyc_submit", json!({"objective": "summarise the release notes", "input": "v2 adds X", "deadline": "1h", "idempotency_key": "agent-1"})).await;
    assert_eq!(again["structuredContent"]["job_id"], job.as_str());
    assert_eq!(again["structuredContent"]["replayed"], true);

    let done = mcp.tool("tonyc_await", json!({"job_id": job, "timeout_seconds": 10})).await;
    assert_eq!(done["structuredContent"]["state"], "READY", "{done}");
    assert!(done["content"][0]["text"].as_str().unwrap().contains("READY"));

    // Now the cold path answers immediately.
    let hit = mcp.tool("tonyc_try", json!({"objective": "summarise the release notes", "input": "v2 adds X"})).await;
    assert_eq!(hit["structuredContent"]["hit"], true, "{hit}");

    // The work is attributed to the agent that connected.
    let client = tonyc_sdk::Client::unix(&socket);
    let status = client.status(&job).await.unwrap();
    let origin = status.info.envelope.origin.unwrap();
    assert_eq!((origin.system.as_str(), origin.agent.as_deref()), ("mcp", Some("test-agent")));

    // Agents cannot ask for remote inference, and mistakes are tool errors,
    // not protocol errors.
    let remote = mcp.tool("tonyc_submit", json!({"objective": "x", "locality": "UNRESTRICTED"})).await;
    assert_eq!(remote["isError"], true);
    let bad = mcp.tool("tonyc_submit", json!({"objective": "x", "quality_floor": 3})).await;
    assert_eq!(bad["isError"], true);
    assert!(bad["content"][0]["text"].as_str().unwrap().contains("quality.floor"), "{bad}");
    let unknown = mcp.request("resources/list", json!({})).await;
    assert_eq!(unknown["error"]["code"], -32601);

    drop(mcp.stdin);
    let _ = mcp.child.wait().await;
    running.stop().await;
}

#[tokio::test]
async fn a_missing_daemon_is_explained() {
    let server = tonyc_mcp::Server::new(tonyc_sdk::Client::unix("/nonexistent/tonycd.sock"));
    let r = server
        .handle(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "tonyc_try", "arguments": {"objective": "x"}}}))
        .await
        .unwrap();
    assert_eq!(r["result"]["isError"], true);
    assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("Is tonycd running?"));
    // Notifications get no answer.
    assert!(server.handle(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await.is_none());
}
