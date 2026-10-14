//! Ollama adapter.
//!
//! The configured URL is resolved once per refresh and classified by where
//! it actually points: loopback addresses give a `Loopback` exposure, private
//! addresses `Lan`, anything else `Remote`. Requests then connect to that
//! exact address. Until a refresh succeeds the adapter is unhealthy and
//! classified `Remote`, the most restrictive reading.

use super::net::{classify, parse_url};
use super::{AdapterError, BoxFuture, ModelAdapter, http};
use crate::config::OllamaConfig;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::RwLock;
use std::time::Duration;
use tonyc_core::model::{Economics, ModelResources, Performance, ProfileSource};
use tonyc_core::{Exposure, InferenceOutput, InferenceRequest, ModelProfile};

pub struct Ollama {
    cfg: OllamaConfig,
    state: RwLock<State>,
    warmth: super::Warmth,
}

struct State {
    target: Option<Target>,
    exposure: Exposure,
    healthy: bool,
    models: Vec<ModelProfile>,
    last_error: Option<String>,
}

#[derive(Clone)]
struct Target {
    addr: SocketAddr,
    host_header: String,
}

#[derive(Deserialize)]
pub(crate) struct Tags {
    pub models: Vec<TagModel>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct TagModel {
    pub name: String,
    pub size: u64,
    pub digest: String,
    #[serde(default)]
    pub details: TagDetails,
}

#[derive(Deserialize, Default, Clone)]
pub(crate) struct TagDetails {
    #[serde(default)]
    pub parameter_size: String,
}

#[derive(Deserialize)]
pub(crate) struct Generated {
    pub response: String,
    #[serde(default)]
    pub prompt_eval_count: u64,
    #[serde(default)]
    pub eval_count: u64,
    /// Durations in nanoseconds.
    #[serde(default)]
    pub load_duration: u64,
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_duration: u64,
}

impl Generated {
    pub fn timings(&self) -> Option<tonyc_core::model::Timings> {
        (self.eval_duration > 0).then_some(tonyc_core::model::Timings {
            load_ms: (self.load_duration / 1_000_000) as i64,
            prefill_ms: (self.prompt_eval_duration / 1_000_000) as i64,
            generate_ms: (self.eval_duration / 1_000_000) as i64,
        })
    }
}

impl Ollama {
    pub fn new(cfg: OllamaConfig) -> Ollama {
        let host = parse_url("ollama", &cfg.url).map(|(h, _)| h).unwrap_or_else(|_| cfg.url.clone());
        Ollama {
            cfg,
            state: RwLock::new(State {
                target: None,
                exposure: Exposure::Remote { host },
                healthy: false,
                models: vec![],
                last_error: Some("not yet probed".into()),
            }),
            warmth: super::Warmth::default(),
        }
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.request_timeout_secs.max(1))
    }

    async fn probe(&self) -> Result<(Target, Exposure, Vec<ModelProfile>), String> {
        let (host, port) = parse_url("ollama", &self.cfg.url)?;
        let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
            .await
            .map_err(|e| format!("resolving {host}: {e}"))?
            .collect();
        let addr = *addrs.first().ok_or_else(|| format!("{host} resolved to no addresses"))?;
        let target = Target { addr, host_header: format!("{host}:{port}") };
        let exposure = classify(&host, port, &addrs);
        let reply =
            http::request(target.addr, &target.host_header, "GET", "/api/tags", &[], None, Duration::from_secs(5))
                .await?;
        if !reply.status.is_success() {
            return Err(format!("/api/tags returned {}", reply.status));
        }
        let tags: Tags = serde_json::from_slice(&reply.body).map_err(|e| format!("/api/tags: {e}"))?;
        let mut models: Vec<ModelProfile> = tags.models.into_iter().map(|m| prior_profile("ollama", m, 1.0)).collect();
        if let Ok(ps) =
            http::request(target.addr, &target.host_header, "GET", "/api/ps", &[], None, Duration::from_secs(5)).await
            && ps.status.is_success()
        {
            self.warmth.reported(&super::parse_loaded(&ps.body), Duration::from_secs(90));
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok((target, exposure, models))
    }
}

impl ModelAdapter for Ollama {
    fn runtime(&self) -> &str {
        "ollama"
    }

    fn exposure(&self) -> Exposure {
        self.state.read().unwrap().exposure.clone()
    }

    fn healthy(&self) -> bool {
        self.state.read().unwrap().healthy
    }

    fn models(&self) -> Vec<ModelProfile> {
        let mut models = self.state.read().unwrap().models.clone();
        for m in &mut models {
            m.loaded = self.warmth.is_warm(m.id.trim_start_matches("ollama/"));
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
                    // Keep the last known models so jobs fail as unreachable
                    // rather than as if the models never existed.
                    st.healthy = false;
                    st.last_error = Some(e);
                }
            }
        })
    }

    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        Box::pin(async move {
            let (target, digest) = {
                let st = self.state.read().unwrap();
                let target =
                    st.target.clone().ok_or_else(|| AdapterError::Unreachable("ollama has not been reached".into()))?;
                let digest =
                    st.models.iter().find(|m| m.id == format!("ollama/{}", req.model)).and_then(|m| m.digest.clone());
                (target, digest)
            };
            let payload = self.encode(req);
            let reply = http::request(
                target.addr,
                &target.host_header,
                "POST",
                "/api/generate",
                &[],
                Some(payload),
                self.timeout(),
            )
            .await
            .map_err(|e| {
                self.state.write().unwrap().healthy = false;
                AdapterError::Unreachable(e)
            })?;
            if !reply.status.is_success() {
                let msg = String::from_utf8_lossy(&reply.body).chars().take(300).collect::<String>();
                return Err(AdapterError::Failed(format!("ollama returned {}: {msg}", reply.status)));
            }
            let out: Generated = serde_json::from_slice(&reply.body)
                .map_err(|e| AdapterError::Failed(format!("decoding reply: {e}")))?;
            self.warmth.ran(&req.model);
            Ok(InferenceOutput {
                timings: out.timings(),
                text: out.response,
                tokens_in: out.prompt_eval_count,
                tokens_out: out.eval_count,
                model_digest: digest,
                bytes_sent: reply.bytes_sent,
                bytes_received: reply.body.len() as u64,
            })
        })
    }

    fn encode(&self, req: &InferenceRequest) -> Vec<u8> {
        generate_body(req)
    }

    fn describe(&self) -> serde_json::Value {
        let st = self.state.read().unwrap();
        serde_json::json!({
            "runtime": "ollama",
            "url": self.cfg.url,
            "healthy": st.healthy,
            "exposure": st.exposure,
            "models": st.models.len(),
            "last_error": st.last_error,
        })
    }
}

/// The JSON body of an Ollama `/api/generate` request.
pub(crate) fn generate_body(req: &InferenceRequest) -> Vec<u8> {
    let mut body = serde_json::json!({
        "model": req.model,
        "prompt": req.prompt,
        "stream": false,
        "options": { "num_predict": req.max_tokens },
    });
    if req.json {
        body["format"] = "json".into();
    }
    serde_json::to_vec(&body).expect("a JSON value serialises")
}

/// Static priors for a model Tonyc has not yet measured. They are rough by
/// design and marked `static_prior`; configuration overrides them now and
/// benchmarks will replace them (Phase 4). `speedup` scales generation
/// speed (e.g. for a node with a GPU large enough for the model).
pub(crate) fn prior_profile(runtime: &str, m: TagModel, speedup: f64) -> ModelProfile {
    let params_b = parse_params(&m.details.parameter_size).unwrap_or(m.size as f64 / 0.6e9);
    let general = match params_b {
        p if p < 1.5 => 0.45,
        p if p < 4.0 => 0.55,
        p if p < 9.0 => 0.65,
        p if p < 15.0 => 0.70,
        p if p < 35.0 => 0.75,
        _ => 0.80,
    };
    let tokens_per_sec = (40.0 / params_b.max(0.1)).clamp(1.0, 80.0) * speedup;
    let size_mb = m.size / (1024 * 1024);
    // Assume ~35 W of CPU package power while generating.
    let energy_wh_per_ktok = 35.0 * (1000.0 / tokens_per_sec) / 3600.0;
    ModelProfile {
        id: format!("{runtime}/{}", m.name),
        runtime: runtime.into(),
        digest: Some(m.digest),
        capabilities: BTreeMap::from([("general".into(), general)]),
        performance: Performance {
            tokens_per_sec,
            prefill_tokens_per_sec: tokens_per_sec * 8.0,
            startup_ms: 500 + size_mb as i64 / 2,
            warm_startup_ms: 150,
        },
        resources: ModelResources { ram_mb: size_mb + size_mb / 5, vram_mb: 0 },
        economics: Economics { cost_per_mtok_gbp: 0.0, energy_wh_per_ktok },
        source: ProfileSource::StaticPrior,
        loaded: false,
        evidence: Default::default(),
    }
}

pub(crate) fn parse_params(s: &str) -> Option<f64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last()? {
        'B' | 'b' => (&s[..s.len() - 1], 1.0),
        'M' | 'm' => (&s[..s.len() - 1], 0.001),
        _ => return None,
    };
    num.parse::<f64>().ok().map(|n| n * mult)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_parameter_sizes() {
        assert_eq!(parse_params("8.0B"), Some(8.0));
        assert!((parse_params("999.89M").unwrap() - 0.99989).abs() < 1e-9);
        assert_eq!(parse_params("big"), None);
    }

    #[test]
    fn unprobed_adapter_is_remote_and_unhealthy() {
        let o = Ollama::new(OllamaConfig::default());
        assert!(!o.healthy());
        assert!(o.exposure().leaves_host());
    }
}
