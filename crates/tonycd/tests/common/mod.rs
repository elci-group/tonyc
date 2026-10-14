//! Shared harness: a real daemon on a temporary Unix socket, spoken to over
//! HTTP exactly as agents and the CLI do.
#![allow(dead_code)]

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tonyc_core::model::{Economics, ModelResources, Performance, ProfileSource};
use tonyc_core::{Exposure, InferenceOutput, InferenceRequest, ModelProfile};
use tonycd::config::{BuiltinConfig, Config};
use tonycd::runtimes::{AdapterError, BoxFuture, ModelAdapter, Registry, builtin::Builtin};

// ------------------------------------------------------------- harness ---

pub struct Harness {
    pub _dir: tempfile::TempDir,
    pub data: PathBuf,
    pub socket: PathBuf,
    pub running: Option<tonycd::Running>,
}

pub fn config(dir: &Path, ms_per_token: u64) -> Config {
    let mut cfg = Config::default();
    cfg.daemon.socket = Some(dir.join("run/tonycd.sock"));
    cfg.daemon.data_dir = Some(dir.join("data"));
    cfg.daemon.tick_ms = 20;
    cfg.pressure.disabled = true;
    // Tests trigger scans explicitly unless they are testing the scanner.
    cfg.cognition.scan_interval_secs = 3600;
    // Preemptive Dynamism is opted into by the tests that exercise it.
    cfg.pd.enabled = false;
    cfg.runtimes.ollama.enabled = false;
    cfg.runtimes.builtin = BuiltinConfig { enabled: true, ms_per_token, quality: 0.6 };
    cfg
}

pub fn builtin(ms_per_token: u64) -> Arc<dyn ModelAdapter> {
    Arc::new(Builtin::new(BuiltinConfig { enabled: true, ms_per_token, quality: 0.6 }))
}

impl Harness {
    pub async fn new(cfg_edit: impl FnOnce(&mut Config), adapters: Vec<Arc<dyn ModelAdapter>>) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(dir.path(), 0);
        cfg_edit(&mut cfg);
        let socket = cfg.socket_path();
        let data = cfg.data_dir();
        let overrides = cfg.models.clone();
        let running = tonycd::start_with(cfg, Registry::new(adapters, overrides)).await.unwrap();
        Harness { _dir: dir, data, socket, running: Some(running) }
    }

    pub async fn restart(&mut self, cfg_edit: impl FnOnce(&mut Config), adapters: Vec<Arc<dyn ModelAdapter>>) {
        self.running.take().unwrap().stop().await;
        let mut cfg = config(self._dir.path(), 0);
        cfg_edit(&mut cfg);
        self.running = Some(tonycd::start_with(cfg, Registry::new(adapters, BTreeMap::new())).await.unwrap());
    }

    pub async fn call(&self, method: &str, path: &str, body: &str) -> (u16, Value) {
        self.call_with(method, path, body, &[]).await
    }

    pub async fn call_with(&self, method: &str, path: &str, body: &str, headers: &[(&str, &str)]) -> (u16, Value) {
        let stream = tokio::net::UnixStream::connect(&self.socket).await.unwrap();
        let (mut tx, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream)).await.unwrap();
        tokio::spawn(conn);
        let mut req = hyper::Request::builder().method(method).uri(path).header("host", "localhost");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let req = req.body(Full::new(Bytes::from(body.to_string()))).unwrap();
        let res = tx.send_request(req).await.unwrap();
        let status = res.status().as_u16();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let value =
            serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into()));
        (status, value)
    }

    pub async fn submit(&self, yaml: &str) -> String {
        let (status, v) = self.call("POST", "/v1/jobs", yaml).await;
        assert_eq!(status, 201, "submit failed: {v}");
        v["id"].as_str().unwrap().to_string()
    }

    pub async fn wait(&self, id: &str) -> Value {
        let (status, v) = self.call("GET", &format!("/v1/jobs/{id}/await?timeout_ms=10000"), "").await;
        assert_eq!(status, 200, "job {id} did not finish: {v}");
        v
    }

    pub async fn state(&self, id: &str) -> String {
        let (_, v) = self.call("GET", &format!("/v1/jobs/{id}"), "").await;
        v["job"]["state"].as_str().unwrap().to_string()
    }

    pub async fn wait_for_state(&self, id: &str, want: &str) {
        for _ in 0..500 {
            if self.state(id).await == want {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("job {id} never reached {want}; it is {}", self.state(id).await);
    }

    pub async fn stop(mut self) {
        self.running.take().unwrap().stop().await;
    }
}

pub fn events(v: &Value) -> Vec<String> {
    v["events"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap().to_string()).collect()
}

/// A runtime that claims to be excellent, free and fast — and sends data
/// off the host. It records every prompt so tests can prove what it saw.
#[derive(Default)]
pub struct FakeRemote {
    pub prompts: Mutex<Vec<String>>,
}

impl FakeRemote {
    pub fn calls(&self) -> usize {
        self.prompts.lock().unwrap().len()
    }
}

impl ModelAdapter for FakeRemote {
    fn runtime(&self) -> &str {
        "cloud"
    }
    fn exposure(&self) -> Exposure {
        Exposure::Remote { host: "api.example.com".into() }
    }
    fn healthy(&self) -> bool {
        true
    }
    fn models(&self) -> Vec<ModelProfile> {
        vec![ModelProfile {
            id: "cloud/frontier".into(),
            runtime: "cloud".into(),
            digest: None,
            capabilities: BTreeMap::from([("general".into(), 0.99)]),
            performance: Performance {
                tokens_per_sec: 1e6,
                prefill_tokens_per_sec: 1e6,
                startup_ms: 0,
                warm_startup_ms: 0,
            },
            resources: ModelResources::default(),
            economics: Economics { cost_per_mtok_gbp: 0.0, energy_wh_per_ktok: 0.0 },
            source: ProfileSource::Configured,
            loaded: true,
            evidence: Default::default(),
        }]
    }
    fn refresh(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        self.prompts.lock().unwrap().push(req.prompt.clone());
        Box::pin(async {
            Ok(InferenceOutput {
                text: "remote answer".into(),
                tokens_in: 1,
                tokens_out: 1,
                model_digest: None,
                bytes_sent: 1,
                bytes_received: 1,
                timings: None,
            })
        })
    }
    fn describe(&self) -> Value {
        serde_json::json!({"runtime": "cloud"})
    }
}

/// Wraps an adapter and counts inference calls.
pub struct Counting {
    pub inner: Arc<dyn ModelAdapter>,
    pub calls: AtomicUsize,
}

impl Counting {
    pub fn new(inner: Arc<dyn ModelAdapter>) -> Arc<Counting> {
        Arc::new(Counting { inner, calls: AtomicUsize::new(0) })
    }
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ModelAdapter for Counting {
    fn runtime(&self) -> &str {
        self.inner.runtime()
    }
    fn exposure(&self) -> Exposure {
        self.inner.exposure()
    }
    fn healthy(&self) -> bool {
        self.inner.healthy()
    }
    fn models(&self) -> Vec<ModelProfile> {
        self.inner.models()
    }
    fn refresh(&self) -> BoxFuture<'_, ()> {
        self.inner.refresh()
    }
    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.run(req)
    }
    fn describe(&self) -> Value {
        self.inner.describe()
    }
}

/// A model whose every property a test controls: quality, price, memory,
/// warmth, token counts, timings and answers.
pub struct Scripted {
    pub runtime: String,
    pub model: String,
    pub quality: f64,
    pub cost_per_mtok: f64,
    pub energy_wh_per_ktok: f64,
    pub ram_mb: u64,
    pub loaded: bool,
    pub tokens: (u64, u64),
    pub exposure: Exposure,
    pub answer: fn(&str) -> String,
    /// Timings for the n-th call (0-based).
    pub timings: fn(usize) -> Option<tonyc_core::model::Timings>,
    pub calls: AtomicUsize,
}

impl Scripted {
    pub fn new(id: &str, quality: f64) -> Scripted {
        let (runtime, model) = id.split_once('/').unwrap();
        Scripted {
            runtime: runtime.into(),
            model: model.into(),
            quality,
            cost_per_mtok: 0.0,
            energy_wh_per_ktok: 0.0,
            ram_mb: 0,
            loaded: true,
            tokens: (10, 20),
            exposure: Exposure::InProcess,
            answer: |p| format!("answer: {}", p.lines().next().unwrap_or("")),
            timings: |_| None,
            calls: AtomicUsize::new(0),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ModelAdapter for Scripted {
    fn runtime(&self) -> &str {
        &self.runtime
    }
    fn exposure(&self) -> Exposure {
        self.exposure.clone()
    }
    fn healthy(&self) -> bool {
        true
    }
    fn models(&self) -> Vec<ModelProfile> {
        vec![ModelProfile {
            id: format!("{}/{}", self.runtime, self.model),
            runtime: self.runtime.clone(),
            digest: None,
            capabilities: BTreeMap::from([("general".into(), self.quality)]),
            performance: Performance {
                tokens_per_sec: 1e6,
                prefill_tokens_per_sec: 1e6,
                startup_ms: 0,
                warm_startup_ms: 0,
            },
            resources: ModelResources { ram_mb: self.ram_mb, vram_mb: 0 },
            economics: Economics { cost_per_mtok_gbp: self.cost_per_mtok, energy_wh_per_ktok: self.energy_wh_per_ktok },
            source: ProfileSource::StaticPrior,
            loaded: self.loaded,
            evidence: Default::default(),
        }]
    }
    fn refresh(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(InferenceOutput {
                text: (self.answer)(&req.prompt),
                tokens_in: self.tokens.0,
                tokens_out: self.tokens.1,
                model_digest: None,
                bytes_sent: 0,
                bytes_received: 0,
                timings: (self.timings)(n),
            })
        })
    }
    fn describe(&self) -> Value {
        serde_json::json!({"runtime": self.runtime})
    }
}
