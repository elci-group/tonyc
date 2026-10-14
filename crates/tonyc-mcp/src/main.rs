//! `tonyc-mcp`: an MCP server on stdio exposing Tonyc to agents.
//!
//! ```sh
//! claude mcp add tonyc -- tonyc-mcp          # Claude Code
//! TONYC_SOCKET=/path/to/tonycd.sock tonyc-mcp  # a non-default daemon
//! ```

use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tonyc_mcp::Server;
use tonyc_sdk::Client;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if std::env::args().nth(1).is_some_and(|a| a == "--version" || a == "-V") {
        println!("tonyc-mcp {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let server = Arc::new(Server::new(Client::new()));
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // One writer, so concurrent responses never interleave on stdout.
    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(line) = rx.recv().await {
            if out.write_all(line.as_bytes()).await.is_err() || out.write_all(b"\n").await.is_err() {
                break;
            }
            let _ = out.flush().await;
        }
    });

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut tasks = tokio::task::JoinSet::new();
    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                let err = serde_json::json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}});
                let _ = tx.send(err.to_string());
                continue;
            }
        };
        // `initialize` completes before anything after it (it decides how
        // work is attributed); every other request runs on its own, so a
        // long tonyc_await does not hold up the others.
        if msg.get("method").and_then(|m| m.as_str()) == Some("initialize") {
            if let Some(reply) = server.handle(msg).await {
                let _ = tx.send(reply.to_string());
            }
            continue;
        }
        let (server, tx) = (server.clone(), tx.clone());
        tasks.spawn(async move {
            if let Some(reply) = server.handle(msg).await {
                let _ = tx.send(reply.to_string());
            }
        });
    }
    while tasks.join_next().await.is_some() {}
    drop(tx);
    let _ = writer.await;
}
