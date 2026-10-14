//! OpenAI-compatible adapter: one adapter for every local server that
//! speaks `/v1/models` and `/v1/chat/completions` — llama.cpp's
//! `llama-server`, vLLM, LM Studio, LocalAI and Ollama's `/v1` among them.
//!
//! Like the Ollama adapter it resolves the configured URL, classifies where
//! the address actually is (loopback, LAN, remote) and connects only to that
//! address. It is not a client for hosted APIs: only plain HTTP is accepted.

use super::net::{locate, parse_url};
use super::ollama::{TagDetails, TagModel, prior_profile};
use super::{AdapterError, BoxFuture, ModelAdapter, Warmth, http};
use crate::config::OpenAiConfig;
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::RwLock;
use std::time::Duration;
use tonyc_core::model::{Timings, estimate_tokens};
use tonyc_core::{Exposure, InferenceOutput, InferenceRequest, ModelProfile};

pub struct OpenAi {
    cfg: OpenAiConfig,
    state: RwLock<State>,
    warmth: Warmth,
}

struct State {
    target: Option<(SocketAddr, String)>,
    exposure: Exposure,
    healthy: bool,
    models: Vec<ModelProfile>,
    last_error: Option<String>,
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Deserialize)]
struct Completion {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
    /// llama.cpp reports where the time went.
    #[serde(default)]
    timings: Option<LlamaTimings>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Deserialize)]
struct LlamaTimings {
    #[serde(default)]
    prompt_ms: f64,
    #[serde(default)]
    predicted_ms: f64,
}

impl OpenAi {
    pub fn new(cfg: OpenAiConfig) -> OpenAi {
        let host = parse_url(&cfg.name, &cfg.url).map(|(h, _)| h).unwrap_or_else(|_| cfg.url.clone());
        OpenAi {
            cfg,
            state: RwLock::new(State {
                target: None,
                exposure: Exposure::Remote { host },
                healthy: false,
                models: vec![],
                last_error: Some("not yet probed".into()),
            }),
            warmth: Warmth::default(),
        }
    }

    fn headers(&self) -> Result<Vec<(&'static str, String)>, String> {
        match &self.cfg.api_key_env {
            None => Ok(vec![]),
            Some(var) => match std::env::var(var) {
                Ok(key) if !key.is_empty() => Ok(vec![("authorization", format!("Bearer {key}"))]),
                _ => Err(format!("api_key_env `{var}` is not set")),
            },
        }
    }

    async fn probe(&self) -> Result<((SocketAddr, String), Exposure, Vec<ModelProfile>), String> {
        let (host, port) = parse_url(&self.cfg.name, &self.cfg.url)?;
        let (addr, exposure) = locate(&host, port).await?;
        let host_header = format!("{host}:{port}");
        let reply =
            http::request(addr, &host_header, "GET", "/v1/models", &self.headers()?, None, Duration::from_secs(5))
                .await?;
        if !reply.status.is_success() {
            return Err(format!("/v1/models returned {}", reply.status));
        }
        let list: ModelList = serde_json::from_slice(&reply.body).map_err(|e| format!("/v1/models: {e}"))?;
        let mut models: Vec<ModelProfile> = list
            .data
            .into_iter()
            .filter(|m| self.cfg.models.is_empty() || self.cfg.models.contains(&m.id))
            .map(|m| profile(&self.cfg.name, &m.id))
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.dedup_by(|a, b| a.id == b.id);
        Ok(((addr, host_header), exposure, models))
    }
}

/// A static prior from the model name alone (the API does not report sizes).
fn profile(runtime: &str, name: &str) -> ModelProfile {
    let params = params_in_name(name).unwrap_or_else(|| "7B".into());
    let mut p = prior_profile(
        runtime,
        TagModel {
            name: name.to_string(),
            size: 0,
            digest: String::new(),
            details: TagDetails { parameter_size: params.clone() },
        },
        1.0,
    );
    p.digest = None;
    // Without a file size, estimate memory from the parameter count (~4.5
    // bits per weight, the common quantisation, plus overhead).
    let b = super::ollama::parse_params(&params).unwrap_or(7.0);
    p.resources.ram_mb = (b * 1024.0 * 0.6) as u64 + 256;
    p
}

/// `gemma3:1b` → `1b`; `Qwen2.5-7B-Instruct` → `7B`.
fn params_in_name(name: &str) -> Option<String> {
    name.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.'))
        .find(|t| {
            let (num, unit) = t.split_at(t.len().saturating_sub(1));
            matches!(unit, "b" | "B" | "m" | "M")
                && !num.is_empty()
                && num.parse::<f64>().is_ok()
                && num.starts_with(|c: char| c.is_ascii_digit())
        })
        .map(str::to_string)
}

/// The JSON body of a chat completion request for an inference request.
fn completion_body(req: &InferenceRequest) -> Vec<u8> {
    let mut body = serde_json::json!({
        "model": req.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "stream": false,
    });
    if req.json {
        body["response_format"] = serde_json::json!({"type": "json_object"});
    }
    serde_json::to_vec(&body).expect("a JSON value serialises")
}

impl ModelAdapter for OpenAi {
    fn runtime(&self) -> &str {
        &self.cfg.name
    }

    fn exposure(&self) -> Exposure {
        self.state.read().unwrap().exposure.clone()
    }

    fn healthy(&self) -> bool {
        self.state.read().unwrap().healthy
    }

    fn models(&self) -> Vec<ModelProfile> {
        let mut models = self.state.read().unwrap().models.clone();
        let prefix = format!("{}/", self.cfg.name);
        for m in &mut models {
            m.loaded = self.warmth.is_warm(m.id.trim_start_matches(&prefix));
        }
        models
    }

    fn refresh(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            let probed = self.probe().await;
            let mut st = self.state.write().unwrap();
            match probed {
                Ok((target, exposure, models)) => {
                    st.target = Some(target);
                    st.exposure = exposure;
                    st.models = models;
                    st.healthy = true;
                    st.last_error = None;
                }
                Err(e) => {
                    st.healthy = false;
                    st.last_error = Some(e);
                }
            }
        })
    }

    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        Box::pin(async move {
            let (addr, host_header) = self
                .state
                .read()
                .unwrap()
                .target
                .clone()
                .ok_or_else(|| AdapterError::Unreachable(format!("{} has not been reached", self.cfg.name)))?;
            let headers = self.headers().map_err(AdapterError::Unreachable)?;
            let timeout = Duration::from_secs(self.cfg.request_timeout_secs.max(1));
            let started = std::time::Instant::now();
            let reply = http::request(
                addr,
                &host_header,
                "POST",
                "/v1/chat/completions",
                &headers,
                Some(self.encode(req)),
                timeout,
            )
            .await
            .map_err(|e| {
                self.state.write().unwrap().healthy = false;
                AdapterError::Unreachable(e)
            })?;
            if !reply.status.is_success() {
                let msg = String::from_utf8_lossy(&reply.body).chars().take(300).collect::<String>();
                return Err(AdapterError::Failed(format!("{} returned {}: {msg}", self.cfg.name, reply.status)));
            }
            let out: Completion = serde_json::from_slice(&reply.body)
                .map_err(|e| AdapterError::Failed(format!("decoding reply: {e}")))?;
            let text = out
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .ok_or_else(|| AdapterError::Failed("the reply has no message content".into()))?;
            self.warmth.ran(&req.model);
            let (tokens_in, tokens_out) = match out.usage {
                Some(u) if u.completion_tokens > 0 => (u.prompt_tokens, u.completion_tokens),
                _ => (estimate_tokens(req.prompt.len()), estimate_tokens(text.len())),
            };
            // llama.cpp says where the time went; otherwise attribute the
            // whole wall time to generation (an overestimate of it).
            let timings = match out.timings {
                Some(t) if t.predicted_ms > 0.0 => {
                    Some(Timings { load_ms: 0, prefill_ms: t.prompt_ms as i64, generate_ms: t.predicted_ms as i64 })
                }
                _ => None,
            }
            .or_else(|| {
                (tokens_out > 0).then(|| Timings {
                    load_ms: 0,
                    prefill_ms: 0,
                    generate_ms: started.elapsed().as_millis() as i64,
                })
            });
            Ok(InferenceOutput {
                text,
                tokens_in,
                tokens_out,
                model_digest: None,
                bytes_sent: reply.bytes_sent,
                bytes_received: reply.body.len() as u64,
                timings,
            })
        })
    }

    fn encode(&self, req: &InferenceRequest) -> Vec<u8> {
        completion_body(req)
    }

    fn describe(&self) -> serde_json::Value {
        let st = self.state.read().unwrap();
        serde_json::json!({
            "runtime": self.cfg.name,
            "kind": "openai",
            "url": self.cfg.url,
            "healthy": st.healthy,
            "exposure": st.exposure,
            "models": st.models.len(),
            "last_error": st.last_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_parameter_counts_from_names() {
        assert_eq!(params_in_name("gemma3:1b").as_deref(), Some("1b"));
        assert_eq!(params_in_name("Qwen2.5-7B-Instruct-Q4_K_M").as_deref(), Some("7B"));
        assert_eq!(params_in_name("llama-3.1-8b-instruct").as_deref(), Some("8b"));
        assert_eq!(params_in_name("smollm-360M").as_deref(), Some("360M"));
        assert_eq!(params_in_name("gemma3n:e2b"), None);
        assert_eq!(params_in_name("mistral"), None);
    }

    #[test]
    fn priors_scale_with_size() {
        let small = profile("llamacpp", "gemma3:1b");
        let big = profile("llamacpp", "qwen2.5:32b");
        assert_eq!(small.id, "llamacpp/gemma3:1b");
        assert!(small.capabilities["general"] < big.capabilities["general"]);
        assert!(small.performance.tokens_per_sec > big.performance.tokens_per_sec);
        assert!(small.resources.ram_mb < big.resources.ram_mb);
    }

    #[test]
    fn asks_for_json_when_the_job_needs_it() {
        let req = InferenceRequest { model: "m".into(), prompt: "p".into(), max_tokens: 9, json: true };
        let body: serde_json::Value = serde_json::from_slice(&completion_body(&req)).unwrap();
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["max_tokens"], 9);
        assert_eq!(body["messages"][0]["content"], "p");
    }

    #[test]
    fn unprobed_adapter_is_remote_and_unhealthy() {
        let o = OpenAi::new(OpenAiConfig { name: "x".into(), url: "http://127.0.0.1:1".into(), ..Default::default() });
        assert!(!o.healthy());
        assert!(o.exposure().leaves_host());
    }
}
