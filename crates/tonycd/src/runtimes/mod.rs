//! The model runtime layer (directive §11).
//!
//! Each runtime is a [`ModelAdapter`]: it declares the [`Exposure`] its use
//! implies, lists the models it can serve with their capability profiles,
//! and runs runtime-neutral [`InferenceRequest`]s. The scheduler never sees
//! anything runtime-specific, so a runtime can be replaced or added without
//! touching planning or the agent contract (gate G5).

pub mod builtin;
pub mod command;
mod http;
mod net;
pub mod ollama;
pub mod openai;

use crate::config::{Config, ModelOverride};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use tonyc_core::model::ProfileSource;
use tonyc_core::{Candidate, Exposure, InferenceOutput, InferenceRequest, ModelProfile};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum AdapterError {
    #[error("runtime unreachable: {0}")]
    Unreachable(String),
    /// An SSH node (or the runtime on it) could not be reached. The job may
    /// fall back to another domain, per its policy.
    #[error("node unreachable: {0}")]
    NodeUnreachable(String),
    #[error("model failed: {0}")]
    Failed(String),
}

pub trait ModelAdapter: Send + Sync {
    /// Runtime name; model ids are `<runtime>/<model>`.
    fn runtime(&self) -> &str;
    /// The execution domain whose slots this runtime uses: `local`, or
    /// `node:<name>` for an SSH node.
    fn domain(&self) -> String {
        "local".into()
    }
    /// Where data goes when this runtime is used.
    fn exposure(&self) -> Exposure;
    fn healthy(&self) -> bool;
    fn models(&self) -> Vec<ModelProfile>;
    /// Rediscovers models and health.
    fn refresh(&self) -> BoxFuture<'_, ()>;
    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>>;
    /// The exact bytes `run` sends for a request; the boundary audit hashes
    /// them, and keeps them when they leave the host.
    fn encode(&self, req: &InferenceRequest) -> Vec<u8> {
        req.prompt.as_bytes().to_vec()
    }
    /// Free-form status for `tonyc models` and `tonyc doctor`.
    fn describe(&self) -> serde_json::Value;
}

pub struct Registry {
    adapters: RwLock<Vec<Arc<dyn ModelAdapter>>>,
    overrides: BTreeMap<String, ModelOverride>,
    /// Measurements by model id; applied over static priors.
    observed: RwLock<std::collections::HashMap<String, crate::economics::Observed>>,
}

impl Registry {
    pub fn new(adapters: Vec<Arc<dyn ModelAdapter>>, overrides: BTreeMap<String, ModelOverride>) -> Registry {
        Registry { adapters: RwLock::new(adapters), overrides, observed: RwLock::default() }
    }

    pub fn set_observed(&self, model: &str, observed: crate::economics::Observed) {
        self.observed.write().unwrap_or_else(|p| p.into_inner()).insert(model.to_string(), observed);
    }

    /// The runtimes the configuration enables.
    pub fn from_config(cfg: &Config) -> Registry {
        let mut adapters: Vec<Arc<dyn ModelAdapter>> = Vec::new();
        if cfg.runtimes.builtin.enabled {
            adapters.push(Arc::new(builtin::Builtin::new(cfg.runtimes.builtin.clone())));
        }
        if cfg.runtimes.ollama.enabled {
            adapters.push(Arc::new(ollama::Ollama::new(cfg.runtimes.ollama.clone())));
        }
        for o in &cfg.runtimes.openai {
            adapters.push(Arc::new(openai::OpenAi::new(o.clone())));
        }
        for c in &cfg.runtimes.command {
            adapters.push(Arc::new(command::Command::new(c.clone())));
        }
        Registry::new(adapters, cfg.models.clone())
    }

    /// Refreshes every runtime concurrently.
    pub async fn refresh(&self) {
        let mut set = tokio::task::JoinSet::new();
        for a in self.adapters() {
            set.spawn(async move { a.refresh().await });
        }
        while set.join_next().await.is_some() {}
    }

    /// Refreshes the runtimes in one domain (`local`, `node:<name>`).
    pub async fn refresh_domain(&self, domain: &str) {
        for a in self.adapters().into_iter().filter(|a| a.domain() == domain) {
            a.refresh().await;
        }
    }

    pub fn adapters(&self) -> Vec<Arc<dyn ModelAdapter>> {
        self.adapters.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn add(&self, adapter: Arc<dyn ModelAdapter>) {
        let mut adapters = self.adapters.write().unwrap_or_else(|p| p.into_inner());
        adapters.retain(|a| a.runtime() != adapter.runtime());
        adapters.push(adapter);
    }

    pub fn remove(&self, runtime: &str) -> bool {
        let mut adapters = self.adapters.write().unwrap_or_else(|p| p.into_inner());
        let before = adapters.len();
        adapters.retain(|a| a.runtime() != runtime);
        adapters.len() != before
    }

    /// Every model the planner may consider, with configured overrides
    /// applied and disabled models removed. All domains count as free.
    pub fn candidates(&self) -> Vec<Candidate> {
        self.candidates_with(|_| true)
    }

    /// As [`Registry::candidates`], marking each by whether its domain has a
    /// free slot.
    pub fn candidates_with(&self, free: impl Fn(&str) -> bool) -> Vec<Candidate> {
        let mut out = Vec::new();
        for a in self.adapters() {
            let exposure = a.exposure();
            let healthy = a.healthy();
            let available = free(&a.domain());
            for mut profile in a.models() {
                // Static prior, then measurements, then the user's override.
                if let Some(obs) = self.observed.read().unwrap_or_else(|p| p.into_inner()).get(&profile.id) {
                    obs.apply(&mut profile);
                }
                if let Some(o) = self.overrides.get(&profile.id) {
                    if o.disabled {
                        continue;
                    }
                    apply_override(&mut profile, o);
                }
                out.push(Candidate {
                    profile,
                    exposure: exposure.clone(),
                    healthy,
                    available,
                    pressured: false,
                    blocked: None,
                });
            }
        }
        out
    }

    /// A model's profile as its runtime reports it: the static prior, before
    /// measurements or overrides.
    pub fn prior(&self, model_id: &str) -> Option<ModelProfile> {
        self.adapters().into_iter().flat_map(|a| a.models()).find(|m| m.id == model_id)
    }

    pub fn has_override(&self, model_id: &str) -> bool {
        self.overrides.contains_key(model_id)
    }

    /// The adapter serving a model id, and the model's runtime-local name.
    pub fn resolve(&self, model_id: &str) -> Option<(Arc<dyn ModelAdapter>, String)> {
        let (runtime, name) = model_id.split_once('/')?;
        let adapters: Vec<_> = self.adapters().into_iter().filter(|a| a.runtime() == runtime).collect();
        // The adapter that actually lists the model, if runtime names collide.
        let adapter =
            adapters.iter().find(|a| a.models().iter().any(|m| m.id == model_id)).or(adapters.first())?.clone();
        Some((adapter, name.to_string()))
    }
}

/// Which models a runtime holds in memory. Fed by the runtime's own report
/// (`/api/ps`) on refresh, and by successful runs (a model that just ran is
/// warm until the runtime's keep-alive lapses).
#[derive(Default)]
pub struct Warmth {
    until: std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
}

/// Ollama's default keep-alive is five minutes; assume a little less.
const WARM_AFTER_RUN: std::time::Duration = std::time::Duration::from_secs(240);

impl Warmth {
    pub fn ran(&self, model: &str) {
        self.until.lock().unwrap().insert(model.to_string(), std::time::Instant::now() + WARM_AFTER_RUN);
    }

    /// Replaces the set with what the runtime reports as loaded now.
    pub fn reported(&self, models: &[String], valid_for: std::time::Duration) {
        let mut until = self.until.lock().unwrap();
        until.clear();
        for m in models {
            until.insert(m.clone(), std::time::Instant::now() + valid_for);
        }
    }

    pub fn is_warm(&self, model: &str) -> bool {
        self.until.lock().unwrap().get(model).is_some_and(|t| *t > std::time::Instant::now())
    }
}

/// Parses Ollama's `/api/ps` body into loaded model names.
pub(crate) fn parse_loaded(body: &[u8]) -> Vec<String> {
    #[derive(serde::Deserialize)]
    struct Ps {
        #[serde(default)]
        models: Vec<PsModel>,
    }
    #[derive(serde::Deserialize)]
    struct PsModel {
        name: String,
    }
    serde_json::from_slice::<Ps>(body).map(|p| p.models.into_iter().map(|m| m.name).collect()).unwrap_or_default()
}

fn apply_override(p: &mut ModelProfile, o: &ModelOverride) {
    let mut touched = !o.capabilities.is_empty();
    for (k, v) in &o.capabilities {
        p.capabilities.insert(k.clone(), *v);
    }
    let mut set = |field: &mut f64, v: Option<f64>| {
        if let Some(v) = v {
            *field = v;
            touched = true;
        }
    };
    set(&mut p.performance.tokens_per_sec, o.tokens_per_sec);
    set(&mut p.performance.prefill_tokens_per_sec, o.prefill_tokens_per_sec);
    set(&mut p.economics.cost_per_mtok_gbp, o.cost_per_mtok_gbp);
    set(&mut p.economics.energy_wh_per_ktok, o.energy_wh_per_ktok);
    if let Some(ms) = o.startup_ms {
        p.performance.startup_ms = ms;
        touched = true;
    }
    if let Some(mb) = o.ram_mb {
        p.resources.ram_mb = mb;
        touched = true;
    }
    if touched {
        p.source = ProfileSource::Configured;
    }
}

/// A plain profile for unit tests.
#[cfg(test)]
pub(crate) fn test_profile(id: &str) -> ModelProfile {
    use tonyc_core::model::{Economics, ModelResources, Performance};
    ModelProfile {
        id: id.into(),
        runtime: id.split('/').next().unwrap_or("").into(),
        digest: None,
        capabilities: BTreeMap::from([("general".into(), 0.6)]),
        performance: Performance {
            tokens_per_sec: 5.0,
            prefill_tokens_per_sec: 40.0,
            startup_ms: 3000,
            warm_startup_ms: 150,
        },
        resources: ModelResources::default(),
        economics: Economics::default(),
        source: ProfileSource::StaticPrior,
        loaded: false,
        evidence: Default::default(),
    }
}
