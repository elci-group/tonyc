//! `builtin/echo`: a deterministic in-process runtime. It does not infer;
//! it echoes a bounded slice of the prompt. It exists so the daemon, the
//! queue and the CLI can be exercised end to end without a model, and its
//! quality profile is low enough that real jobs never select it.

use super::{AdapterError, BoxFuture, ModelAdapter};
use crate::config::BuiltinConfig;
use std::collections::BTreeMap;
use std::time::Duration;
use tonyc_core::model::{Economics, ModelResources, Performance, ProfileSource, Timings, estimate_tokens};
use tonyc_core::{Exposure, InferenceOutput, InferenceRequest, ModelProfile};

pub struct Builtin {
    cfg: BuiltinConfig,
}

impl Builtin {
    pub fn new(cfg: BuiltinConfig) -> Builtin {
        Builtin { cfg }
    }
}

impl ModelAdapter for Builtin {
    fn runtime(&self) -> &str {
        "builtin"
    }

    fn exposure(&self) -> Exposure {
        Exposure::InProcess
    }

    fn healthy(&self) -> bool {
        true
    }

    fn models(&self) -> Vec<ModelProfile> {
        let tps = if self.cfg.ms_per_token == 0 { 1e6 } else { 1000.0 / self.cfg.ms_per_token as f64 };
        vec![ModelProfile {
            id: "builtin/echo".into(),
            runtime: "builtin".into(),
            digest: Some("builtin-echo-v1".into()),
            capabilities: BTreeMap::from([("general".into(), self.cfg.quality)]),
            performance: Performance {
                tokens_per_sec: tps,
                prefill_tokens_per_sec: 1e6,
                startup_ms: 0,
                warm_startup_ms: 0,
            },
            resources: ModelResources::default(),
            economics: Economics { cost_per_mtok_gbp: 0.0, energy_wh_per_ktok: 0.0001 },
            source: ProfileSource::Configured,
            loaded: true,
            evidence: Default::default(),
        }]
    }

    fn refresh(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }

    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        Box::pin(async move {
            if req.model != "echo" {
                return Err(AdapterError::Failed(format!("builtin has no model `{}`", req.model)));
            }
            let words: Vec<&str> = req.prompt.split_whitespace().take(req.max_tokens as usize).collect();
            let tokens_out = words.len() as u64;
            let generate_ms = if self.cfg.ms_per_token > 0 { self.cfg.ms_per_token * tokens_out.max(1) } else { 0 };
            if generate_ms > 0 {
                tokio::time::sleep(Duration::from_millis(generate_ms)).await;
            }
            let echoed = words.join(" ");
            let text = if req.json { serde_json::json!({ "echo": echoed }).to_string() } else { echoed };
            Ok(InferenceOutput {
                tokens_in: estimate_tokens(req.prompt.len()),
                tokens_out,
                model_digest: Some("builtin-echo-v1".into()),
                bytes_sent: 0,
                bytes_received: 0,
                timings: Some(Timings { load_ms: 0, prefill_ms: 0, generate_ms: generate_ms as i64 }),
                text,
            })
        })
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "runtime": "builtin", "healthy": true, "note": "deterministic echo; not inference" })
    }
}
