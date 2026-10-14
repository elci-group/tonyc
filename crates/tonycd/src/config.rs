//! Daemon configuration (`tonycd.toml`). Every field has a default, and the
//! defaults are conservative: LOCAL_ONLY ceiling, no remote execution domain,
//! one worker slot, the built-in test runtime disabled.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tonyc_core::Locality;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub daemon: DaemonConfig,
    pub policy: PolicyConfig,
    pub pressure: PressureConfig,
    pub runtimes: RuntimesConfig,
    pub cognition: CognitionConfig,
    pub fabric: FabricConfig,
    pub economics: EconomicsConfig,
    pub budgets: BudgetsConfig,
    pub pd: PdConfig,
    /// Capability-profile overrides keyed by model id (`ollama/gemma3:1b`).
    pub models: BTreeMap<String, ModelOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    pub socket: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
    /// Optional loopback TCP listener, e.g. `127.0.0.1:7878`.
    pub tcp: Option<String>,
    /// Concurrent inference slots.
    pub workers: usize,
    /// Scheduler wake-up interval when idle.
    pub tick_ms: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        DaemonConfig { socket: None, data_dir: None, tcp: None, workers: 1, tick_ms: 500 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolicyConfig {
    /// The most exposure any job may request. Jobs asking for more are
    /// planned as if they had asked for this.
    pub max_locality: Locality,
    pub ssh_allowlist: Vec<String>,
    pub remote_allowlist: Vec<String>,
    /// Locality for jobs that do not state one.
    pub default_locality: Locality,
    pub default_quality_floor: f64,
    pub default_max_tokens: u32,
    pub max_tokens_ceiling: u32,
    /// Largest input file the daemon will read for a job.
    pub max_input_bytes: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        PolicyConfig {
            max_locality: Locality::LocalOnly,
            ssh_allowlist: vec![],
            remote_allowlist: vec![],
            default_locality: Locality::LocalOnly,
            default_quality_floor: 0.5,
            default_max_tokens: 512,
            max_tokens_ceiling: 8192,
            max_input_bytes: 1 << 20,
        }
    }
}

/// Thresholds for host pressure (directive §20).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PressureConfig {
    /// 1-minute load average per core above which the host is busy.
    pub busy_load_per_core: f64,
    /// Available memory fraction below which the host is busy.
    pub min_available_memory: f64,
    /// Whether running on battery counts as busy.
    pub battery_is_busy: bool,
    /// GPU utilisation (0–1) above which the host is busy.
    pub gpu_busy_utilisation: f64,
    /// Hottest thermal zone (°C) above which the host is busy.
    pub max_temperature_c: f64,
    /// Ignore host pressure entirely (testing).
    pub disabled: bool,
}

impl Default for PressureConfig {
    fn default() -> Self {
        PressureConfig {
            busy_load_per_core: 0.7,
            min_available_memory: 0.15,
            battery_is_busy: true,
            gpu_busy_utilisation: 0.8,
            max_temperature_c: 90.0,
            disabled: false,
        }
    }
}

/// Accounting assumptions (Phase 4). Every figure derived from these is an
/// estimate and is reported as one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EconomicsConfig {
    /// What "conventional inference" is priced as: a premium hosted API.
    pub reference_name: String,
    pub reference_input_gbp_per_mtok: f64,
    pub reference_output_gbp_per_mtok: f64,
    pub electricity_gbp_per_kwh: f64,
    /// Power drawn while inferring, by domain kind.
    pub local_watts: f64,
    pub node_watts: f64,
    /// Used instead of the above when the domain has a GPU.
    pub gpu_watts: f64,
}

impl Default for EconomicsConfig {
    fn default() -> Self {
        EconomicsConfig {
            reference_name: "premium hosted API (reference)".into(),
            reference_input_gbp_per_mtok: 2.40,
            reference_output_gbp_per_mtok: 12.00,
            electricity_gbp_per_kwh: 0.25,
            local_watts: 35.0,
            node_watts: 50.0,
            gpu_watts: 250.0,
        }
    }
}

/// Preemptive Dynamism (directive §9, Phase 6): preparing answers that the
/// demand log says will be asked for, before they are.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PdConfig {
    pub enabled: bool,
    pub interval_secs: u64,
    /// How far back demand is counted.
    pub window: String,
    /// How far ahead a prediction looks; an answer not asked for within it
    /// is discarded.
    pub horizon: String,
    /// Times a request must have been asked before it is predicted.
    pub min_demand: u32,
    /// Least probability of being asked within the horizon. Raised
    /// automatically while measured yield is low.
    pub min_probability: f64,
    /// Expected value (probability × conventional cost avoided) must exceed
    /// the estimated operating cost by this factor.
    pub value_ratio: f64,
    /// Most speculative jobs queued or running at once.
    pub max_pending: usize,
    /// Daily compute for SPECULATIVE work, unless `budgets.daily_compute`
    /// sets one.
    pub daily_compute: String,
}

impl Default for PdConfig {
    fn default() -> Self {
        PdConfig {
            enabled: true,
            interval_secs: 60,
            window: "7d".into(),
            horizon: "24h".into(),
            min_demand: 2,
            min_probability: 0.3,
            value_ratio: 2.0,
            max_pending: 2,
            daily_compute: "30m".into(),
        }
    }
}

impl PdConfig {
    pub fn window_ms(&self) -> anyhow::Result<i64> {
        tonyc_core::duration::parse_millis(&self.window).map_err(|e| anyhow::anyhow!("pd.window: {e}"))
    }

    pub fn horizon_ms(&self) -> anyhow::Result<i64> {
        tonyc_core::duration::parse_millis(&self.horizon).map_err(|e| anyhow::anyhow!("pd.horizon: {e}"))
    }

    pub fn daily_compute_ms(&self) -> anyhow::Result<i64> {
        tonyc_core::duration::parse_millis(&self.daily_compute).map_err(|e| anyhow::anyhow!("pd.daily_compute: {e}"))
    }
}

/// Resource budgets (directive §21).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetsConfig {
    /// Monetary spend allowed per calendar month (UTC). 0 is a valid
    /// zero-spend policy: paid runtimes are never used.
    pub remote_monthly_gbp: f64,
    /// Largest model (MB of RAM) Tonyc may load locally; 0 means only what
    /// is currently available.
    pub memory_max_mb: u64,
    /// Compute time per UTC day by urgency class, e.g. `speculative = "4h"`.
    /// Classes not listed are unlimited.
    pub daily_compute: BTreeMap<String, String>,
}

impl Default for BudgetsConfig {
    fn default() -> Self {
        BudgetsConfig { remote_monthly_gbp: 0.0, memory_max_mb: 0, daily_compute: BTreeMap::new() }
    }
}

impl BudgetsConfig {
    /// Daily compute limits by class, in milliseconds.
    pub fn daily_limits(&self) -> anyhow::Result<BTreeMap<tonyc_core::envelope::Urgency, i64>> {
        let mut out = BTreeMap::new();
        for (class, limit) in &self.daily_compute {
            let urgency: tonyc_core::envelope::Urgency =
                class.parse().map_err(|e: String| anyhow::anyhow!("budgets.daily_compute: {e}"))?;
            let ms = tonyc_core::duration::parse_millis(limit)
                .map_err(|e| anyhow::anyhow!("budgets.daily_compute.{class}: {e}"))?;
            out.insert(urgency, ms);
        }
        Ok(out)
    }
}

/// The SSH compute fabric (Phase 3). Nodes are registered at runtime
/// (`tonyc node add`) and persisted in the state database.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FabricConfig {
    /// The OpenSSH client and helpers.
    pub ssh_program: String,
    pub keyscan_program: String,
    pub keygen_program: String,
    /// Where already-trusted host keys are looked up when a node is added
    /// without an explicit fingerprint (default `~/.ssh/known_hosts`).
    pub user_known_hosts: Option<PathBuf>,
    pub connect_timeout_secs: u64,
    pub probe_interval_secs: u64,
    pub request_timeout_secs: u64,
}

impl Default for FabricConfig {
    fn default() -> Self {
        FabricConfig {
            ssh_program: "ssh".into(),
            keyscan_program: "ssh-keyscan".into(),
            keygen_program: "ssh-keygen".into(),
            user_known_hosts: None,
            connect_timeout_secs: 10,
            probe_interval_secs: 60,
            request_timeout_secs: 3600,
        }
    }
}

impl FabricConfig {
    pub fn user_known_hosts(&self) -> PathBuf {
        self.user_known_hosts.clone().unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/".into())).join(".ssh/known_hosts")
        })
    }
}

/// The Cognitive Store (Phase 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CognitionConfig {
    /// How often the corpus and every fresh object are re-checked.
    pub scan_interval_secs: u64,
    /// Lifetime of results with `ttl` freshness and no explicit ttl.
    pub default_ttl: String,
    /// Most prior objects offered to a model as context.
    pub context_objects: usize,
    /// Character budget for that context.
    pub context_chars: usize,
    /// Minimum relatedness (0–1) for an object to be offered as context.
    pub related_threshold: f64,
    pub max_corpus_files: usize,
}

impl Default for CognitionConfig {
    fn default() -> Self {
        CognitionConfig {
            scan_interval_secs: 30,
            default_ttl: "7d".into(),
            context_objects: 3,
            context_chars: 4000,
            related_threshold: 0.25,
            max_corpus_files: 10_000,
        }
    }
}

impl CognitionConfig {
    pub fn default_ttl_ms(&self) -> anyhow::Result<i64> {
        tonyc_core::duration::parse_millis(&self.default_ttl).map_err(|e| anyhow::anyhow!("cognition.default_ttl: {e}"))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimesConfig {
    pub ollama: OllamaConfig,
    pub builtin: BuiltinConfig,
    /// OpenAI-compatible servers: llama.cpp's `llama-server`, vLLM, LM
    /// Studio, LocalAI, Ollama's `/v1`. Plain HTTP on this host or the LAN.
    pub openai: Vec<OpenAiConfig>,
    /// Local programs speaking the command runtime protocol (any engine:
    /// Candle, ONNX Runtime, `llama-cli`, a script).
    pub command: Vec<CommandConfig>,
}

impl RuntimesConfig {
    fn validate(&self) -> anyhow::Result<()> {
        let mut seen = std::collections::BTreeSet::from(["builtin".to_string(), "ollama".to_string()]);
        let names = self.openai.iter().map(|o| &o.name).chain(self.command.iter().map(|c| &c.name));
        for name in names {
            anyhow::ensure!(
                !name.is_empty()
                    && name.len() <= 32
                    && name.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
                    && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "_-".contains(c)),
                "runtime name `{name}`: use 1-32 lowercase letters, digits, `_` or `-`"
            );
            anyhow::ensure!(seen.insert(name.clone()), "runtime name `{name}` is used twice (or is reserved)");
        }
        for c in &self.command {
            anyhow::ensure!(!c.argv.is_empty(), "runtimes.command `{}`: argv must not be empty", c.name);
            anyhow::ensure!(!c.models.is_empty(), "runtimes.command `{}`: declare at least one model", c.name);
            c.declared_exposure().map_err(|e| anyhow::anyhow!("runtimes.command `{}`: {e}", c.name))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiConfig {
    /// The runtime name; model ids are `<name>/<model>`.
    pub name: String,
    /// `http://host:port`. Like Ollama's, the exposure is derived from where
    /// it actually resolves.
    pub url: String,
    /// An environment variable holding a bearer token, for servers started
    /// with one (`llama-server --api-key`). Never stored in the database.
    pub api_key_env: Option<String>,
    /// Serve only these models (default: everything `/v1/models` lists).
    pub models: Vec<String>,
    pub request_timeout_secs: u64,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        OpenAiConfig {
            name: String::new(),
            url: String::new(),
            api_key_env: None,
            models: vec![],
            request_timeout_secs: 3600,
        }
    }
}

/// A local program as a runtime. Tonyc writes one request to its stdin and
/// reads one answer from its stdout; see docs/PROTOCOL.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CommandConfig {
    pub name: String,
    /// The program and its arguments; `{model}` is replaced by the model
    /// name. The prompt is never put on the command line (other users can
    /// read process arguments).
    pub argv: Vec<String>,
    /// `text`: the prompt on stdin, the answer as stdout. `json`: one
    /// `tonyc.runtime/1` request object in, one response object out.
    pub protocol: CommandProtocol,
    pub models: Vec<CommandModel>,
    /// Where the program sends data: `local` (default; it computes on this
    /// host), `lan:<host>` or `remote:<host>`. Tonyc cannot see inside the
    /// program, so a program that forwards prompts must say so here; the
    /// locality policy then treats it accordingly.
    pub exposure: String,
    pub timeout_secs: u64,
}

impl CommandConfig {
    pub fn declared_exposure(&self) -> Result<tonyc_core::Exposure, String> {
        use tonyc_core::Exposure;
        match self.exposure.split_once(':') {
            None if self.exposure == "local" => Ok(Exposure::Loopback { endpoint: format!("exec:{}", self.name) }),
            Some(("lan", host)) if !host.is_empty() => Ok(Exposure::Lan { host: host.into() }),
            Some(("remote", host)) if !host.is_empty() => Ok(Exposure::Remote { host: host.into() }),
            _ => Err(format!("exposure `{}` must be `local`, `lan:<host>` or `remote:<host>`", self.exposure)),
        }
    }
}

impl Default for CommandConfig {
    fn default() -> Self {
        CommandConfig {
            name: String::new(),
            argv: vec![],
            protocol: CommandProtocol::Text,
            models: vec![],
            exposure: "local".into(),
            timeout_secs: 600,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandProtocol {
    Text,
    Json,
}

/// A model a command runtime serves, with the prior Tonyc plans with until
/// it has measured the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CommandModel {
    pub name: String,
    pub capabilities: BTreeMap<String, f64>,
    pub tokens_per_sec: f64,
    pub startup_ms: i64,
    pub ram_mb: u64,
}

impl Default for CommandModel {
    fn default() -> Self {
        CommandModel {
            name: String::new(),
            capabilities: BTreeMap::new(),
            tokens_per_sec: 10.0,
            startup_ms: 1000,
            ram_mb: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OllamaConfig {
    pub enabled: bool,
    /// Must resolve to a loopback, private or explicitly allowed address;
    /// the exposure is derived from where it actually resolves.
    pub url: String,
    pub request_timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        OllamaConfig { enabled: true, url: "http://127.0.0.1:11434".into(), request_timeout_secs: 3600 }
    }
}

/// A deterministic in-process runtime for tests and smoke checks. Its output
/// is a transformation of the prompt, not inference; its quality profile is
/// deliberately low so it never wins a real job by accident.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BuiltinConfig {
    pub enabled: bool,
    /// Simulated generation delay per output token.
    pub ms_per_token: u64,
    pub quality: f64,
}

impl Default for BuiltinConfig {
    fn default() -> Self {
        BuiltinConfig { enabled: false, ms_per_token: 0, quality: 0.05 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelOverride {
    pub capabilities: BTreeMap<String, f64>,
    pub tokens_per_sec: Option<f64>,
    pub prefill_tokens_per_sec: Option<f64>,
    pub startup_ms: Option<i64>,
    pub cost_per_mtok_gbp: Option<f64>,
    pub energy_wh_per_ktok: Option<f64>,
    pub ram_mb: Option<u64>,
    /// Exclude the model from planning.
    pub disabled: bool,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Config> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let cfg: Config = toml::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.daemon.workers >= 1, "daemon.workers must be at least 1");
        anyhow::ensure!(
            self.policy.default_locality <= self.policy.max_locality,
            "policy.default_locality ({}) exceeds policy.max_locality ({})",
            self.policy.default_locality,
            self.policy.max_locality
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.policy.default_quality_floor),
            "policy.default_quality_floor must be in [0, 1]"
        );
        self.runtimes.validate()?;
        Ok(())
    }

    /// Default config location: `$XDG_CONFIG_HOME/tonyc/tonycd.toml`.
    pub fn default_path() -> PathBuf {
        xdg("XDG_CONFIG_HOME", ".config").join("tonyc").join("tonycd.toml")
    }

    pub fn socket_path(&self) -> PathBuf {
        self.daemon.socket.clone().unwrap_or_else(default_socket)
    }

    pub fn data_dir(&self) -> PathBuf {
        self.daemon.data_dir.clone().unwrap_or_else(|| xdg("XDG_DATA_HOME", ".local/share").join("tonyc"))
    }
}

pub use tonyc_core::transport::default_socket;

fn xdg(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/".into())).join(fallback),
    }
}
