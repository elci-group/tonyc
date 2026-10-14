//! TonycFirst against a real tonycd and a counting stand-in for Lucid's
//! provider.

use async_trait::async_trait;
use lucid_platform::ports::{ChatMessage, ModelProvider, ModelRequest, ModelResponse, ProviderError, Usage};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tonyc_lucid::TonycFirst;
use tonyc_sdk::Client;
use tonycd::config::{BuiltinConfig, Config};
use tonycd::runtimes::{Registry, builtin::Builtin};

#[derive(Default)]
struct Counting {
    calls: AtomicUsize,
}

#[async_trait]
impl ModelProvider for Counting {
    fn name(&self) -> &str {
        "openai_compat:test"
    }
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ModelResponse {
            provider: "openai_compat:test".into(),
            model: request.model.clone(),
            content: Some("from the provider".into()),
            tool_calls: vec![],
            usage: Usage { input_tokens: 10, output_tokens: 3, reported: true },
            finish_reason: "stop".into(),
        })
    }
}

async fn daemon(dir: &std::path::Path) -> (tonycd::Running, std::path::PathBuf) {
    let mut cfg = Config::default();
    cfg.daemon.socket = Some(dir.join("run/tonycd.sock"));
    cfg.daemon.data_dir = Some(dir.join("data"));
    cfg.daemon.tick_ms = 20;
    cfg.pressure.disabled = true;
    cfg.pd.enabled = false;
    cfg.runtimes.ollama.enabled = false;
    let b = BuiltinConfig { enabled: true, ms_per_token: 0, quality: 0.8 };
    cfg.runtimes.builtin = b.clone();
    let socket = cfg.socket_path();
    let running =
        tonycd::start_with(cfg, Registry::new(vec![Arc::new(Builtin::new(b))], Default::default())).await.unwrap();
    (running, socket)
}

fn request(question: &str) -> ModelRequest {
    ModelRequest {
        model: "gpt-test".into(),
        messages: vec![
            ChatMessage::System { content: "You are the project CTO.".into() },
            ChatMessage::User { content: question.into() },
        ],
        tools: vec![],
        temperature: Some(0.2),
        max_output_tokens: 256,
    }
}

#[tokio::test]
async fn existing_cognition_is_served_before_the_provider_is_called() {
    let dir = tempfile::tempdir().unwrap();
    let (running, socket) = daemon(dir.path()).await;
    let provider = Arc::new(Counting::default());
    let lucid = TonycFirst::new(provider.clone(), Client::unix(&socket)).scope("project:p1").task("task-9");
    assert_eq!(lucid.name(), "tonyc+openai_compat:test");

    // Hot path: nothing exists, the provider answers.
    let req = request("Summarise the open risks");
    let first = lucid.complete(&req).await.unwrap();
    assert_eq!(first.content.as_deref(), Some("from the provider"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);

    // Warm path: Lucid defers the question; Tonyc produces it locally.
    let job = lucid.defer(&req, "1h", Some("lucid:task-9:risks")).await.unwrap();
    let again = lucid.defer(&req, "1h", Some("lucid:task-9:risks")).await.unwrap();
    assert_eq!(job.id, again.id);
    let origin = job.envelope.origin.clone().unwrap();
    assert_eq!(
        (origin.system.as_str(), origin.run.as_deref(), origin.agent.as_deref()),
        ("lucid", Some("task-9"), Some("gpt-test"))
    );
    let done = Client::unix(&socket).wait(&job.id, Duration::from_secs(30)).await.unwrap();
    assert_eq!(done.state(), tonyc_sdk::JobState::Ready);

    // Cold path: the same request is now served without a provider call.
    let served = lucid.complete(&req).await.unwrap();
    assert_eq!(served.provider, "tonyc");
    assert_eq!(served.model, "builtin/echo");
    assert_eq!(served.usage.output_tokens, 0);
    assert_eq!(served.content.as_deref(), done.output());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(lucid.stats().served.load(Ordering::SeqCst), 1);

    // Another project cannot see project p1's cognition.
    let other = TonycFirst::new(provider.clone(), Client::unix(&socket)).scope("project:p2");
    other.complete(&req).await.unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);

    // Tool use always goes to the provider.
    let mut tools = req.clone();
    tools.messages.push(ChatMessage::Assistant { content: None, tool_calls: vec![] });
    tools.messages.push(ChatMessage::Tool { call_id: "c1".into(), content: "ok".into() });
    assert!(lucid.work_for(&tools).is_none());

    // The economics are attributed to Lucid.
    let (_, eco) = raw_get(&socket, "/v1/economics").await;
    assert_eq!(eco["by_origin"]["lucid"]["inference"], 1, "{eco}");
    running.stop().await;
}

#[tokio::test]
async fn lucid_carries_on_when_tonyc_is_down() {
    let provider = Arc::new(Counting::default());
    let lucid = TonycFirst::new(provider.clone(), Client::unix("/nonexistent/tonycd.sock"));
    let r = lucid.complete(&request("anything")).await.unwrap();
    assert_eq!(r.content.as_deref(), Some("from the provider"));
    assert_eq!(lucid.stats().bypassed.load(Ordering::SeqCst), 1);
    assert!(matches!(lucid.defer(&request("x"), "1h", None).await, Err(ProviderError::Transient(_))));
}

async fn raw_get(socket: &std::path::Path, path: &str) -> (u16, serde_json::Value) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::UnixStream::connect(socket).await.unwrap();
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
    let mut buf = String::new();
    s.read_to_string(&mut buf).await.unwrap();
    let (head, body) = buf.split_once("\r\n\r\n").unwrap();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap())
}
