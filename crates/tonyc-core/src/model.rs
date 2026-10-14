//! Model capability profiles (directive §12) and the runtime-neutral shapes
//! every model adapter accepts and returns (§11). No runtime, provider or
//! architecture appears here (invariant T6).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Where a profile's numbers come from. Observed figures replace static
/// priors as Tonyc benchmarks models (Phase 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileSource {
    StaticPrior,
    Configured,
    Observed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProfile {
    /// `<runtime>/<model>`, e.g. `ollama/gemma3:1b`.
    pub id: String,
    pub runtime: String,
    /// Content digest of the weights when the runtime reports one.
    pub digest: Option<String>,
    /// Capability name → expected quality in `[0, 1]`. `general` is the
    /// fallback for capabilities the profile does not name.
    pub capabilities: BTreeMap<String, f64>,
    pub performance: Performance,
    pub resources: ModelResources,
    pub economics: Economics,
    pub source: ProfileSource,
    /// Whether the runtime currently holds the model in memory (warm).
    #[serde(default)]
    pub loaded: bool,
    /// What the figures rest on: measured runs and benchmark results.
    #[serde(default)]
    pub evidence: Evidence,
}

/// The measurements behind a profile.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Evidence {
    /// Completed inference runs observed for this model.
    pub runs: u32,
    /// Capability → benchmark outcome.
    #[serde(default)]
    pub quality: BTreeMap<String, QualityEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityEvidence {
    pub passed: u32,
    pub total: u32,
    /// The quality assumed before any benchmark.
    pub prior: f64,
}

/// Number of pseudo-observations the prior counts for when benchmark
/// results are blended in: a handful of cases moves the estimate, but
/// cannot swing it to 0 or 1.
pub const QUALITY_PRIOR_WEIGHT: f64 = 4.0;

impl QualityEvidence {
    pub fn estimate(&self) -> f64 {
        (self.prior * QUALITY_PRIOR_WEIGHT + self.passed as f64) / (QUALITY_PRIOR_WEIGHT + self.total as f64)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Performance {
    /// Generation speed.
    pub tokens_per_sec: f64,
    /// Prompt processing speed.
    pub prefill_tokens_per_sec: f64,
    /// Time to first token when the model must be loaded (cold).
    pub startup_ms: i64,
    /// Overhead when the model is already in memory (warm).
    #[serde(default)]
    pub warm_startup_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelResources {
    pub ram_mb: u64,
    pub vram_mb: u64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Economics {
    /// Monetary cost in GBP per million tokens (input and output alike).
    pub cost_per_mtok_gbp: f64,
    /// Estimated energy per thousand tokens processed.
    pub energy_wh_per_ktok: f64,
}

impl ModelProfile {
    /// Expected quality for a capability and the capability it was read
    /// from (the requested one, or `general` as a fallback).
    pub fn quality_for(&self, capability: &str) -> Option<(f64, &str)> {
        if let Some((k, q)) = self.capabilities.get_key_value(capability) {
            return Some((*q, k.as_str()));
        }
        self.capabilities.get_key_value("general").map(|(k, q)| (*q, k.as_str()))
    }

    /// Predicted wall time for a job of this size.
    pub fn predict_runtime_ms(&self, prompt_tokens: u64, max_tokens: u32) -> i64 {
        let prefill = prompt_tokens as f64 / self.performance.prefill_tokens_per_sec.max(0.001);
        let generate = max_tokens as f64 / self.performance.tokens_per_sec.max(0.001);
        let startup = if self.loaded { self.performance.warm_startup_ms } else { self.performance.startup_ms };
        startup + ((prefill + generate) * 1000.0).ceil() as i64
    }
}

/// What the daemon asks an adapter to compute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// The adapter-local model name (without the runtime prefix).
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    /// Ask the runtime to constrain output to JSON.
    pub json: bool,
}

/// Where a run's time went, when the runtime reports it.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Timings {
    pub load_ms: i64,
    pub prefill_ms: i64,
    pub generate_ms: i64,
}

/// What an adapter returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub model_digest: Option<String>,
    /// Bytes written to and read from the execution domain, for the
    /// data-boundary audit.
    pub bytes_sent: u64,
    pub bytes_received: u64,
    #[serde(default)]
    pub timings: Option<Timings>,
}

/// Rough token estimate used for planning before a runtime has tokenised.
pub fn estimate_tokens(text_bytes: usize) -> u64 {
    (text_bytes as u64).div_ceil(4)
}
