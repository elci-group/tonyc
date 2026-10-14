//! Shared daemon state.

use crate::config::Config;
use crate::fabric::{self, NodeAdapter, NodeSpec, Ssh};
use crate::host::{self, HostInfo};
use crate::runtimes::{ModelAdapter, Registry};
use crate::store::{Store, now_ms};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;
use tonyc_core::envelope::{CompileDefaults, WorkEnvelope};
use tonyc_core::{Locality, LocalityPolicy, Millis};

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    User,
    /// Preempted by the named, more urgent job.
    Preempted(String),
    Pressure(String),
    Shutdown,
}

pub struct RunningJob {
    pub rank: i64,
    pub yields: bool,
    /// The domain it holds a slot in, once it has one.
    pub domain: Option<String>,
    pub token: CancellationToken,
    pub reason: Option<StopReason>,
}

pub struct Daemon {
    pub cfg: Config,
    pub store: Arc<Store>,
    pub registry: Registry,
    pub host: HostInfo,
    pub ssh: Arc<Ssh>,
    pub wake: Notify,
    pub shutdown: CancellationToken,
    pub started_at: Millis,
    running: Mutex<HashMap<String, RunningJob>>,
    nodes: RwLock<BTreeMap<String, Arc<NodeAdapter>>>,
    /// Execution slots per domain: `local`, `node:<name>`.
    slots: Mutex<HashMap<String, Arc<Semaphore>>>,
    /// Jobs waiting on resources: when to try again, and how many tries.
    retry: Mutex<HashMap<String, (Millis, u32)>>,
    /// Preemptive Dynamism's latest evaluation, for `tonyc pd explain`.
    pub pd_last: Mutex<Option<crate::pd::Evaluation>>,
}

/// Settings that `tonyc policy set` may change at runtime; they override
/// the configuration file and persist in the state database.
pub const POLICY_KEYS: [&str; 2] = ["max_locality", "default_locality"];

/// The execution domain a runtime belongs to.
pub fn domain_of(runtime: &str) -> String {
    match runtime.strip_prefix("node.") {
        Some(name) => format!("node:{name}"),
        None => "local".into(),
    }
}

impl Daemon {
    pub fn new(cfg: Config, store: Arc<Store>, registry: Registry) -> Daemon {
        let ssh = Arc::new(Ssh { cfg: cfg.fabric.clone(), known_hosts: cfg.data_dir().join("known_hosts") });
        let slots = HashMap::from([("local".to_string(), Arc::new(Semaphore::new(cfg.daemon.workers)))]);
        Daemon {
            cfg,
            store,
            registry,
            host: host::discover(),
            ssh,
            wake: Notify::new(),
            shutdown: CancellationToken::new(),
            started_at: now_ms(),
            running: Mutex::new(HashMap::new()),
            nodes: RwLock::new(BTreeMap::new()),
            slots: Mutex::new(slots),
            retry: Mutex::new(HashMap::new()),
            pd_last: Mutex::new(None),
        }
    }

    // ---------------------------------------------------- measurements ---

    /// Loads the measured profile of every model with observations.
    pub fn load_observed(&self) -> Result<()> {
        for model in self.store.observed_models()? {
            self.refresh_observed(&model)?;
        }
        Ok(())
    }

    pub fn refresh_observed(&self, model: &str) -> Result<()> {
        self.registry.set_observed(model, self.store.observed(model)?);
        Ok(())
    }

    // ----------------------------------------------------- waiting work ---

    /// Backs a job off after it could not be planned for lack of resources:
    /// 2 s, doubling to at most a minute.
    pub fn defer(&self, id: &str) -> Millis {
        let mut retry = self.retry.lock().unwrap();
        let tries = retry.get(id).map(|(_, n)| n + 1).unwrap_or(0);
        let wait = (2_000i64 << tries.min(5)).min(60_000);
        let at = now_ms() + wait;
        retry.insert(id.to_string(), (at, tries));
        at
    }

    pub fn retry_due(&self, id: &str, now: Millis) -> bool {
        self.retry.lock().unwrap().get(id).is_none_or(|(at, _)| *at <= now)
    }

    pub fn clear_retry(&self, id: &str) {
        self.retry.lock().unwrap().remove(id);
    }

    // --------------------------------------------------------- pressure ---

    /// Whether a node is under pressure (load per core over the threshold).
    pub fn node_pressured(&self, node: &NodeAdapter) -> bool {
        if self.cfg.pressure.disabled {
            return false;
        }
        node.caps().is_some_and(|c| c.cores > 0 && c.load_1m / c.cores as f64 > self.cfg.pressure.busy_load_per_core)
    }

    /// A healthy, unpressured node this job's locality (and the ceiling)
    /// would allow — somewhere yielding work can go while this host is busy.
    pub fn idle_node_for(&self, env: &WorkEnvelope) -> Option<String> {
        let ceiling = self.ceiling();
        self.nodes()
            .into_iter()
            .filter(|n| n.healthy() && !self.node_pressured(n))
            .find(|n| {
                let e = n.exposure();
                env.privacy.permits(&e).is_ok() && ceiling.permits(&e).is_ok()
            })
            .map(|n| n.spec.name.clone())
    }

    fn locality_setting(&self, key: &str, fallback: Locality) -> Locality {
        self.store.setting(key).ok().flatten().and_then(|v| v.parse().ok()).unwrap_or(fallback)
    }

    /// The daemon-wide locality ceiling every plan must also satisfy. Its
    /// SSH allowlist is the configured one plus every registered node:
    /// registering a node is the explicit act that authorises it, and the
    /// ceiling's class must still be raised for any node to be used.
    pub fn ceiling(&self) -> LocalityPolicy {
        let mut ssh = self.cfg.policy.ssh_allowlist.clone();
        ssh.extend(self.node_names());
        ssh.sort();
        ssh.dedup();
        LocalityPolicy {
            class: self.locality_setting("max_locality", self.cfg.policy.max_locality),
            ssh_allowlist: ssh,
            remote_allowlist: self.cfg.policy.remote_allowlist.clone(),
        }
    }

    pub fn defaults(&self) -> CompileDefaults {
        CompileDefaults {
            locality: self.locality_setting("default_locality", self.cfg.policy.default_locality),
            quality_floor: self.cfg.policy.default_quality_floor,
            max_tokens: self.cfg.policy.default_max_tokens,
            max_tokens_ceiling: self.cfg.policy.max_tokens_ceiling,
            ttl_ms: self.cfg.cognition.default_ttl_ms().unwrap_or(tonyc_core::envelope::DEFAULT_TTL_MS),
        }
    }

    /// Where each policy value comes from: `config` or `set`.
    pub fn policy_sources(&self) -> BTreeMap<&'static str, &'static str> {
        POLICY_KEYS
            .iter()
            .map(|k| {
                let set = self.store.setting(k).ok().flatten().is_some_and(|v| !v.is_empty());
                (*k, if set { "set" } else { "config" })
            })
            .collect()
    }

    // ------------------------------------------------------------ slots ---

    pub fn slot(&self, domain: &str) -> Arc<Semaphore> {
        self.slots.lock().unwrap().entry(domain.to_string()).or_insert_with(|| Arc::new(Semaphore::new(1))).clone()
    }

    pub fn has_free_slot(&self, domain: &str) -> bool {
        self.slot(domain).available_permits() > 0
    }

    /// Jobs that may be in flight at once: local workers plus the slots of
    /// every healthy node.
    pub fn capacity(&self) -> usize {
        let nodes: usize = self.nodes().iter().filter(|n| n.healthy()).map(|n| n.spec.slots).sum();
        self.cfg.daemon.workers + nodes
    }

    // ------------------------------------------------------------ nodes ---

    pub fn nodes(&self) -> Vec<Arc<NodeAdapter>> {
        self.nodes.read().unwrap().values().cloned().collect()
    }

    pub fn node(&self, name: &str) -> Option<Arc<NodeAdapter>> {
        self.nodes.read().unwrap().get(name).cloned()
    }

    pub fn node_names(&self) -> Vec<String> {
        self.nodes.read().unwrap().keys().cloned().collect()
    }

    pub fn known_hosts_path(&self) -> PathBuf {
        self.ssh.known_hosts.clone()
    }

    /// Makes a node an execution domain (registry, slots).
    pub fn attach_node(&self, spec: NodeSpec) -> Arc<NodeAdapter> {
        let adapter = Arc::new(NodeAdapter::new(spec.clone(), self.ssh.clone()));
        self.slots.lock().unwrap().insert(format!("node:{}", spec.name), Arc::new(Semaphore::new(spec.slots)));
        self.registry.add(adapter.clone() as Arc<dyn ModelAdapter>);
        self.nodes.write().unwrap().insert(spec.name.clone(), adapter.clone());
        adapter
    }

    pub fn detach_node(&self, name: &str) -> Option<Arc<NodeAdapter>> {
        let adapter = self.nodes.write().unwrap().remove(name)?;
        self.registry.remove(adapter.runtime());
        Some(adapter)
    }

    /// Attaches every persisted node and rewrites the pinned host keys.
    pub fn load_nodes(&self) -> Result<()> {
        let specs = self.store.nodes()?;
        fabric::write_known_hosts(&self.known_hosts_path(), &specs)?;
        for spec in specs {
            self.attach_node(spec);
        }
        Ok(())
    }

    fn running(&self) -> std::sync::MutexGuard<'_, HashMap<String, RunningJob>> {
        self.running.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn running_count(&self) -> usize {
        self.running().len()
    }

    pub fn running_ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.running().keys().cloned().collect();
        ids.sort();
        ids
    }

    pub fn register_running(&self, env: &WorkEnvelope) -> CancellationToken {
        let token = self.shutdown.child_token();
        self.running().insert(
            env.id.clone(),
            RunningJob {
                rank: env.latency.class.rank(),
                yields: env.latency.class.yields(),
                domain: None,
                token: token.clone(),
                reason: None,
            },
        );
        token
    }

    pub fn set_running_domain(&self, id: &str, domain: &str) {
        if let Some(job) = self.running().get_mut(id) {
            job.domain = Some(domain.to_string());
        }
    }

    pub fn unregister_running(&self, id: &str) {
        self.running().remove(id);
    }

    /// Why a running job was asked to stop, if it was. A job stopped by the
    /// shutdown token alone reports `Shutdown`.
    pub fn stop_reason(&self, id: &str) -> Option<StopReason> {
        let running = self.running();
        let job = running.get(id)?;
        match &job.reason {
            Some(r) => Some(r.clone()),
            None if job.token.is_cancelled() => Some(StopReason::Shutdown),
            None => None,
        }
    }

    /// Asks a running job to stop. Returns false if it is not running.
    pub fn stop(&self, id: &str, reason: StopReason) -> bool {
        let mut running = self.running();
        let Some(job) = running.get_mut(id) else { return false };
        if job.reason.is_none() {
            job.reason = Some(reason);
        }
        job.token.cancel();
        true
    }

    pub fn stop_running(&self, select: impl Fn(&RunningJob) -> bool, reason: impl Fn() -> StopReason) {
        let mut running = self.running();
        for job in running.values_mut().filter(|j| select(j) && j.reason.is_none()) {
            job.reason = Some(reason());
            job.token.cancel();
        }
    }

    /// Preempts the least urgent yielding job that is less urgent than
    /// `rank`, unless one is already on its way out.
    pub fn preempt_one(&self, rank: i64, by: &str) -> bool {
        let mut running = self.running();
        if running.values().any(|j| j.reason.is_some()) {
            return false;
        }
        let victim = running
            .iter_mut()
            .filter(|(_, j)| j.yields && j.rank > rank)
            .max_by(|(ia, a), (ib, b)| a.rank.cmp(&b.rank).then(ib.cmp(ia)));
        match victim {
            Some((_, job)) => {
                job.reason = Some(StopReason::Preempted(by.to_string()));
                job.token.cancel();
                true
            }
            None => false,
        }
    }
}
