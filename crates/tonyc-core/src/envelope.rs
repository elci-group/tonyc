//! The work envelope (directive §6).
//!
//! Agents submit a lenient [`WorkRequest`] (YAML or JSON, most fields
//! optional). [`compile`] validates it and fills every default, producing a
//! [`WorkEnvelope`]: the deterministic, fully explicit contract the scheduler
//! works from. The same request, id and submission time always compile to the
//! same envelope, and `semantic_hash` identifies the work independently of
//! who asked for it or when.

use crate::Millis;
use crate::cognition::{CognitiveKind, Freshness, FreshnessPolicy, Scope, word_enum};
use crate::duration;
use crate::locality::{Locality, LocalityPolicy};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

pub const ENVELOPE_VERSION: &str = "tonyc.envelope/1";

word_enum! {
    /// Semantic urgency classes (directive §6). Ordered most to least urgent.
    Urgency { Now = "NOW", Soon = "SOON", Deferred = "DEFERRED", Idle = "IDLE", Speculative = "SPECULATIVE" }
}

impl Urgency {
    /// Scheduling rank: lower runs first.
    pub fn rank(self) -> i64 {
        self as i64
    }

    /// Whether work of this class must give way to host pressure and to more
    /// urgent work (invariant T5).
    pub fn yields(self) -> bool {
        matches!(self, Urgency::Idle | Urgency::Speculative)
    }
}

word_enum! {
    /// How much verification a result needs before it is `READY`.
    Verification { None = "none", Preferred = "preferred", Required = "required" }
}

word_enum! {
    /// How a job may use a host resource.
    ResourceMode { Opportunistic = "opportunistic", Normal = "normal" }
}

word_enum! {
    /// Refinement (directive §14). `auto`: refine when the floor or the
    /// verification requirement needs it, or a check fails. `always`: draft
    /// cheaply and validate every answer. `never`: one pass only.
    Refine { Auto = "auto", Always = "always", Never = "never" }
}

word_enum! {
    /// What to do when an SSH node fails mid-dispatch (gate G7).
    NodeFailure { Fallback = "fallback", Fail = "fail" }
}

word_enum! {
    /// Energy preference; weights the energy term of the scheduler utility.
    Energy { Economical = "economical", Balanced = "balanced", Performance = "performance" }
}

word_enum! {
    /// Output contracts Tonyc can check.
    OutputSchema { Text = "text/v1", Json = "json/v1" }
}

// ---------------------------------------------------------------- request ---

/// What an agent submits. Unknown fields are rejected so typos fail loudly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkRequest {
    pub task: TaskRequest,
    #[serde(default)]
    pub latency: LatencyRequest,
    #[serde(default)]
    pub quality: QualityRequest,
    #[serde(default)]
    pub privacy: PrivacyRequest,
    #[serde(default)]
    pub resources: ResourcesRequest,
    #[serde(default)]
    pub reuse: ReuseRequest,
    #[serde(default)]
    pub output: OutputRequest,
    /// Who is asking, for attribution only (directive §28): it never changes
    /// what is asked for, where it may run, or what it may reuse.
    #[serde(default)]
    pub origin: OriginRequest,
}

/// The system (orchestrator or agent) a request came from.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginRequest {
    /// e.g. `lucid`, `terri`, `mcp`, `claude-code`.
    pub system: Option<String>,
    /// The caller's own reference for the work (its task or run id).
    pub run: Option<String>,
    /// The agent within that system.
    pub agent: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRequest {
    /// Ignored if present: the daemon assigns ids.
    pub id: Option<String>,
    pub objective: String,
    /// The capability the work needs, e.g. `summarisation`, `rust`.
    pub capability: Option<String>,
    #[serde(default)]
    pub input: InputRequest,
    pub max_tokens: Option<u32>,
    /// Run on exactly this model (`<runtime>/<model>`). For benchmarks and
    /// diagnostics; agents should state constraints, not models (T7).
    pub pin_model: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputRequest {
    pub text: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatencyRequest {
    pub class: Option<Urgency>,
    /// A duration from submission (`45m`) or an absolute Unix time in ms.
    pub deadline: Option<DeadlineSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeadlineSpec {
    AtMillis(i64),
    In(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityRequest {
    pub floor: Option<f64>,
    pub verification: Option<Verification>,
    /// Deterministic conditions every answer must meet.
    #[serde(default)]
    pub checks: Vec<crate::refine::Check>,
    /// Whether Tonyc may refine (validate, revise, escalate).
    pub refine: Option<Refine>,
    /// Most revisions after the first draft (default 2).
    pub max_rounds: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrivacyRequest {
    pub locality: Option<Locality>,
    #[serde(default)]
    pub ssh_allowlist: Vec<String>,
    #[serde(default)]
    pub remote_allowlist: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourcesRequest {
    pub cpu: Option<ResourceMode>,
    pub gpu: Option<ResourceMode>,
    pub energy: Option<Energy>,
    /// `fallback` (default): re-plan without the failed node, under the same
    /// locality rules. `fail`: end as NODE_UNREACHABLE.
    pub on_node_failure: Option<NodeFailure>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReuseRequest {
    /// Consult and contribute to the Cognitive Store (default true).
    pub cache: Option<bool>,
    /// The scope results belong to: `user` (default), `project:<name>`, ...
    pub corpus: Option<String>,
    /// How the result goes stale. Default: `source_hash` when the task reads
    /// files, else `ttl`.
    pub freshness: Option<FreshnessPolicy>,
    /// Lifetime for `ttl` freshness, e.g. `7d`.
    pub ttl: Option<String>,
    /// Event tags for `event` freshness.
    #[serde(default)]
    pub events: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputRequest {
    pub schema: Option<OutputSchema>,
    /// The kind of cognitive object the result is (default `analysis`).
    pub kind: Option<CognitiveKind>,
}

impl WorkRequest {
    /// Parses YAML (which also accepts JSON).
    pub fn parse(text: &str) -> Result<WorkRequest, CompileError> {
        serde_yaml::from_str(text).map_err(|e| CompileError::new("", e.to_string()))
    }
}

// --------------------------------------------------------------- envelope ---

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkEnvelope {
    pub version: String,
    pub id: String,
    pub submitted_at: Millis,
    pub semantic_hash: String,
    pub task: Task,
    pub latency: Latency,
    pub quality: Quality,
    pub privacy: LocalityPolicy,
    pub resources: Resources,
    pub reuse: Reuse,
    pub output: Output,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
}

/// Attribution: which system asked. Not part of the semantic hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    pub system: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub objective: String,
    pub capability: String,
    pub input_text: Option<String>,
    pub input_files: Vec<String>,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Latency {
    pub class: Urgency,
    pub deadline_at: Option<Millis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quality {
    pub floor: f64,
    pub verification: Verification,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<crate::refine::Check>,
    #[serde(default = "default_refine")]
    pub refine: Refine,
    #[serde(default = "default_rounds")]
    pub max_rounds: u32,
}

fn default_refine() -> Refine {
    Refine::Auto
}

fn default_rounds() -> u32 {
    2
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resources {
    pub cpu: ResourceMode,
    pub gpu: ResourceMode,
    pub energy: Energy,
    #[serde(default = "default_node_failure")]
    pub on_node_failure: NodeFailure,
}

fn default_node_failure() -> NodeFailure {
    NodeFailure::Fallback
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reuse {
    pub cache: bool,
    #[serde(default = "Scope::user")]
    pub scope: Scope,
    #[serde(default = "default_freshness")]
    pub freshness: Freshness,
}

fn default_freshness() -> Freshness {
    Freshness { policy: FreshnessPolicy::Ttl, ttl_ms: Some(DEFAULT_TTL_MS), events: vec![] }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub schema: OutputSchema,
    #[serde(default = "default_kind")]
    pub kind: CognitiveKind,
}

fn default_kind() -> CognitiveKind {
    CognitiveKind::Analysis
}

/// Lifetime of results that read no files and state no freshness policy.
pub const DEFAULT_TTL_MS: i64 = 7 * 86_400_000;

impl WorkEnvelope {
    /// Whether the job must wait while the host is under pressure.
    pub fn yields_to_pressure(&self) -> bool {
        self.latency.class.yields()
            || (self.latency.class == Urgency::Deferred
                && (self.resources.cpu == ResourceMode::Opportunistic
                    || self.resources.gpu == ResourceMode::Opportunistic))
    }
}

/// Defaults applied by [`compile`]; the daemon builds this from its policy.
#[derive(Debug, Clone)]
pub struct CompileDefaults {
    pub locality: Locality,
    pub quality_floor: f64,
    pub max_tokens: u32,
    pub max_tokens_ceiling: u32,
    pub ttl_ms: i64,
}

impl Default for CompileDefaults {
    fn default() -> Self {
        CompileDefaults {
            locality: Locality::LocalOnly,
            quality_floor: 0.5,
            max_tokens: 512,
            max_tokens_ceiling: 8192,
            ttl_ms: DEFAULT_TTL_MS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{}{message}", if field.is_empty() { String::new() } else { format!("{field}: ") })]
pub struct CompileError {
    pub field: String,
    pub message: String,
}

impl CompileError {
    pub fn new(field: &str, message: impl Into<String>) -> Self {
        CompileError { field: field.to_string(), message: message.into() }
    }
}

/// Compiles a request into an envelope. Pure: the caller supplies the id and
/// the submission time.
pub fn compile(
    req: &WorkRequest,
    id: &str,
    submitted_at: Millis,
    defaults: &CompileDefaults,
) -> Result<WorkEnvelope, CompileError> {
    let objective = req.task.objective.trim();
    if objective.is_empty() {
        return Err(CompileError::new("task.objective", "must not be empty"));
    }

    let capability = req.task.capability.as_deref().unwrap_or("general").trim().to_ascii_lowercase();
    if capability.is_empty() || !capability.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)) {
        return Err(CompileError::new("task.capability", "use letters, digits, `.`, `_` or `-`"));
    }

    let max_tokens = req.task.max_tokens.unwrap_or(defaults.max_tokens);
    if max_tokens == 0 || max_tokens > defaults.max_tokens_ceiling {
        return Err(CompileError::new(
            "task.max_tokens",
            format!("must be between 1 and {}", defaults.max_tokens_ceiling),
        ));
    }

    for f in &req.task.input.files {
        if !f.starts_with('/') {
            return Err(CompileError::new("task.input.files", format!("`{f}` must be an absolute path")));
        }
    }

    let class = req.latency.class.unwrap_or(Urgency::Deferred);
    let deadline_at = match &req.latency.deadline {
        None => None,
        Some(DeadlineSpec::AtMillis(at)) => {
            if *at <= submitted_at {
                return Err(CompileError::new("latency.deadline", "is already in the past"));
            }
            Some(*at)
        }
        Some(DeadlineSpec::In(text)) => {
            let ms = duration::parse_millis(text).map_err(|e| CompileError::new("latency.deadline", e.to_string()))?;
            Some(submitted_at + ms)
        }
    };

    let floor = req.quality.floor.unwrap_or(defaults.quality_floor);
    if !(0.0..=1.0).contains(&floor) || floor.is_nan() {
        return Err(CompileError::new("quality.floor", "must be between 0 and 1"));
    }

    if req.quality.max_rounds.is_some_and(|r| r > 5) {
        return Err(CompileError::new("quality.max_rounds", "must be at most 5"));
    }
    for (i, c) in req.quality.checks.iter().enumerate() {
        c.validate().map_err(|e| CompileError::new(&format!("quality.checks[{i}]"), e))?;
    }

    let locality = req.privacy.locality.unwrap_or(defaults.locality);
    let mut ssh = req.privacy.ssh_allowlist.clone();
    let mut remote = req.privacy.remote_allowlist.clone();
    ssh.sort();
    ssh.dedup();
    remote.sort();
    remote.dedup();
    match locality {
        Locality::SshAllowlist if ssh.is_empty() => {
            return Err(CompileError::new("privacy.ssh_allowlist", "SSH_ALLOWLIST needs at least one node"));
        }
        Locality::RemoteAllowlist if remote.is_empty() => {
            return Err(CompileError::new("privacy.remote_allowlist", "REMOTE_ALLOWLIST needs at least one host"));
        }
        _ => {}
    }

    let reuse = compile_reuse(&req.reuse, !req.task.input.files.is_empty(), defaults)?;
    let origin = compile_origin(&req.origin)?;

    let mut env = WorkEnvelope {
        version: ENVELOPE_VERSION.to_string(),
        id: id.to_string(),
        submitted_at,
        semantic_hash: String::new(),
        task: Task {
            objective: objective.to_string(),
            capability,
            input_text: req.task.input.text.clone(),
            input_files: req.task.input.files.clone(),
            max_tokens,
            pin_model: req.task.pin_model.clone(),
        },
        latency: Latency { class, deadline_at },
        quality: Quality {
            floor,
            verification: req.quality.verification.unwrap_or(Verification::None),
            checks: req.quality.checks.clone(),
            refine: req.quality.refine.unwrap_or(Refine::Auto),
            max_rounds: req.quality.max_rounds.unwrap_or(2),
        },
        privacy: LocalityPolicy { class: locality, ssh_allowlist: ssh, remote_allowlist: remote },
        resources: Resources {
            cpu: req.resources.cpu.unwrap_or(ResourceMode::Normal),
            gpu: req.resources.gpu.unwrap_or(ResourceMode::Normal),
            energy: req.resources.energy.unwrap_or(Energy::Balanced),
            on_node_failure: req.resources.on_node_failure.unwrap_or(NodeFailure::Fallback),
        },
        reuse,
        output: Output {
            schema: req.output.schema.unwrap_or(OutputSchema::Text),
            kind: req.output.kind.unwrap_or(CognitiveKind::Analysis),
        },
        origin,
    };
    env.semantic_hash = semantic_hash(&env);
    Ok(env)
}

fn compile_reuse(r: &ReuseRequest, reads_files: bool, defaults: &CompileDefaults) -> Result<Reuse, CompileError> {
    let scope = match &r.corpus {
        Some(text) => text.parse::<Scope>().map_err(|e| CompileError::new("reuse.corpus", e))?,
        None => Scope::user(),
    };
    let policy = r.freshness.unwrap_or(if reads_files { FreshnessPolicy::SourceHash } else { FreshnessPolicy::Ttl });
    if policy == FreshnessPolicy::SourceHash && !reads_files {
        return Err(CompileError::new("reuse.freshness", "source_hash needs task.input.files to hash"));
    }
    let ttl_ms = match (&r.ttl, policy) {
        (Some(t), FreshnessPolicy::Ttl) => {
            Some(duration::parse_millis(t).map_err(|e| CompileError::new("reuse.ttl", e.to_string()))?)
        }
        (None, FreshnessPolicy::Ttl) => Some(defaults.ttl_ms),
        (Some(_), _) => return Err(CompileError::new("reuse.ttl", "only applies to `ttl` freshness")),
        (None, _) => None,
    };
    let mut events = r.events.clone();
    events.sort();
    events.dedup();
    match policy {
        FreshnessPolicy::EventInvalidated if events.is_empty() => {
            return Err(CompileError::new("reuse.events", "`event` freshness needs at least one event tag"));
        }
        FreshnessPolicy::EventInvalidated => {}
        _ if !events.is_empty() => {
            return Err(CompileError::new("reuse.events", "only applies to `event` freshness"));
        }
        _ => {}
    }
    Ok(Reuse { cache: r.cache.unwrap_or(true), scope, freshness: Freshness { policy, ttl_ms, events } })
}

fn compile_origin(o: &OriginRequest) -> Result<Option<Origin>, CompileError> {
    let Some(system) = o.system.as_deref().map(str::trim) else {
        if o.run.is_some() || o.agent.is_some() {
            return Err(CompileError::new("origin.system", "is required when origin.run or origin.agent is set"));
        }
        return Ok(None);
    };
    let system = system.to_ascii_lowercase();
    if system.is_empty() || system.len() > 64 || !system.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
    {
        return Err(CompileError::new("origin.system", "use 1-64 letters, digits, `.`, `_` or `-`"));
    }
    let text = |field: &str, v: &Option<String>| -> Result<Option<String>, CompileError> {
        match v.as_deref().map(str::trim) {
            None | Some("") => Ok(None),
            Some(t) if t.len() > 200 || t.chars().any(char::is_control) => {
                Err(CompileError::new(field, "must be at most 200 printable characters"))
            }
            Some(t) => Ok(Some(t.to_string())),
        }
    };
    Ok(Some(Origin { system, run: text("origin.run", &o.run)?, agent: text("origin.agent", &o.agent)? }))
}

/// Hash of what is being asked for (task and output), excluding who asked,
/// when, how urgently, and how good the answer must be: the quality floor
/// and verification are requirements an existing answer is checked against,
/// not part of the question. Two envelopes with equal semantic hashes
/// request the same cognition. Input files contribute their paths only; the
/// Cognitive Store checks their content hashes when it reuses an answer.
pub fn semantic_hash(env: &WorkEnvelope) -> String {
    let key = serde_json::json!({
        "v": ENVELOPE_VERSION,
        "task": env.task,
        "output": env.output,
    });
    let canonical = serde_json::to_vec(&key).expect("envelope serialises");
    blake3::hash(&canonical).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
task:
  id: tny_ignored
  objective: "Analyse dependency drift"
latency:
  class: deferred
  deadline: 45m
quality:
  floor: 0.82
  verification: required
privacy:
  locality: local_only
resources:
  cpu: opportunistic
  gpu: opportunistic
  energy: economical
reuse:
  cache: true
  corpus: project
output:
  schema: text/v1
"#;

    fn compile_text(text: &str) -> Result<WorkEnvelope, CompileError> {
        compile(&WorkRequest::parse(text)?, "tny_1", 1_000, &CompileDefaults::default())
    }

    #[test]
    fn directive_example_compiles() {
        let env = compile_text(EXAMPLE).unwrap();
        assert_eq!(env.id, "tny_1");
        assert_eq!(env.latency.class, Urgency::Deferred);
        assert_eq!(env.latency.deadline_at, Some(1_000 + 45 * 60_000));
        assert_eq!(env.quality.floor, 0.82);
        assert_eq!(env.quality.verification, Verification::Required);
        assert_eq!(env.privacy.class, Locality::LocalOnly);
        assert_eq!(env.resources.energy, Energy::Economical);
        assert!(env.yields_to_pressure());
    }

    #[test]
    fn compilation_is_deterministic() {
        let a = compile_text(EXAMPLE).unwrap();
        let b = compile_text(EXAMPLE).unwrap();
        assert_eq!(a, b);
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    }

    #[test]
    fn semantic_hash_ignores_identity_urgency_and_requirements() {
        let a = compile_text("task: {objective: x}\nlatency: {class: now}\nquality: {floor: 0.9}").unwrap();
        let b = compile(
            &WorkRequest::parse("task: {objective: x}\nlatency: {class: idle}").unwrap(),
            "other",
            9,
            &CompileDefaults::default(),
        )
        .unwrap();
        assert_eq!(a.semantic_hash, b.semantic_hash);
        let c = compile_text("task: {objective: y}").unwrap();
        assert_ne!(a.semantic_hash, c.semantic_hash);
    }

    #[test]
    fn origin_is_attribution_only() {
        let plain = compile_text("task: {objective: x}").unwrap();
        assert_eq!(plain.origin, None);
        let from = compile_text("task: {objective: x}\norigin: {system: Lucid, run: task-7, agent: cto}").unwrap();
        let o = from.origin.clone().unwrap();
        assert_eq!((o.system.as_str(), o.run.as_deref(), o.agent.as_deref()), ("lucid", Some("task-7"), Some("cto")));
        assert_eq!(plain.semantic_hash, from.semantic_hash);
        assert_eq!(compile_text("task: {objective: x}\norigin: {run: r}").unwrap_err().field, "origin.system");
        assert_eq!(compile_text("task: {objective: x}\norigin: {system: 'a b'}").unwrap_err().field, "origin.system");
        // Old envelopes (stored before origin existed) still load.
        let mut v = serde_json::to_value(&plain).unwrap();
        v.as_object_mut().unwrap().remove("origin");
        assert_eq!(serde_json::from_value::<WorkEnvelope>(v).unwrap(), plain);
    }

    #[test]
    fn defaults_are_local_only_and_explicit() {
        let env = compile_text("task: {objective: hello}").unwrap();
        assert_eq!(env.privacy.class, Locality::LocalOnly);
        assert_eq!(env.latency.class, Urgency::Deferred);
        assert_eq!(env.output.schema, OutputSchema::Text);
        assert_eq!(env.task.capability, "general");
    }

    #[test]
    fn rejects_bad_requests() {
        assert_eq!(compile_text("task: {objective: '  '}").unwrap_err().field, "task.objective");
        assert_eq!(compile_text("task: {objective: x}\nquality: {floor: 1.5}").unwrap_err().field, "quality.floor");
        assert_eq!(
            compile_text("task: {objective: x}\nlatency: {deadline: soonish}").unwrap_err().field,
            "latency.deadline"
        );
        assert_eq!(
            compile_text("task: {objective: x}\nprivacy: {locality: ssh_allowlist}").unwrap_err().field,
            "privacy.ssh_allowlist"
        );
        assert_eq!(
            compile_text("task: {objective: x, input: {files: [rel.txt]}}").unwrap_err().field,
            "task.input.files"
        );
        assert!(compile_text("task: {objective: x}\nlatency: {class: whenever}").is_err());
        assert!(compile_text("task: {objective: x}\nprivcy: {}").is_err(), "unknown fields fail loudly");
    }

    #[test]
    fn reuse_defaults_and_rules() {
        let env = compile_text("task: {objective: x}").unwrap();
        assert_eq!(env.reuse.scope, Scope::user());
        assert_eq!(env.reuse.freshness.policy, FreshnessPolicy::Ttl);
        assert_eq!(env.reuse.freshness.ttl_ms, Some(DEFAULT_TTL_MS));
        let env = compile_text("task: {objective: x, input: {files: [/a]}}\nreuse: {corpus: 'project:tonyc'}").unwrap();
        assert_eq!(env.reuse.freshness.policy, FreshnessPolicy::SourceHash);
        assert_eq!(env.reuse.scope.to_string(), "project:tonyc");
        let env = compile_text("task: {objective: x}\nreuse: {freshness: event, events: [deps, deps]}").unwrap();
        assert_eq!(env.reuse.freshness.events, ["deps"]);
        assert_eq!(
            compile_text("task: {objective: x}\nreuse: {freshness: source_hash}").unwrap_err().field,
            "reuse.freshness"
        );
        assert_eq!(compile_text("task: {objective: x}\nreuse: {freshness: event}").unwrap_err().field, "reuse.events");
        assert_eq!(
            compile_text("task: {objective: x}\nreuse: {freshness: manual, ttl: 1d}").unwrap_err().field,
            "reuse.ttl"
        );
        assert_eq!(
            compile_text("task: {objective: x}\nreuse: {corpus: 'galaxy:far'}").unwrap_err().field,
            "reuse.corpus"
        );
    }

    #[test]
    fn phase1_envelopes_still_deserialize() {
        let mut v = serde_json::to_value(compile_text("task: {objective: x}").unwrap()).unwrap();
        v["reuse"] = serde_json::json!({"cache": true, "corpus": null});
        v["output"] = serde_json::json!({"schema": "text/v1"});
        let env: WorkEnvelope = serde_json::from_value(v).unwrap();
        assert_eq!(env.reuse.scope, Scope::user());
        assert_eq!(env.output.kind, CognitiveKind::Analysis);
    }

    #[test]
    fn quality_checks_and_refinement_compile() {
        let env = compile_text(
            "task: {objective: x}\nquality: {refine: always, max_rounds: 3, checks: [{contains: [paris]}, {json_fields: [name]}]}",
        )
        .unwrap();
        assert_eq!(env.quality.refine, Refine::Always);
        assert_eq!(env.quality.max_rounds, 3);
        assert_eq!(env.quality.checks.len(), 2);
        assert_eq!(
            compile_text("task: {objective: x}\nquality: {checks: [{}]}").unwrap_err().field,
            "quality.checks[0]"
        );
        assert_eq!(
            compile_text("task: {objective: x}\nquality: {max_rounds: 9}").unwrap_err().field,
            "quality.max_rounds"
        );
        let plain = compile_text("task: {objective: x}").unwrap();
        assert_eq!((plain.quality.refine, plain.quality.max_rounds), (Refine::Auto, 2));
    }

    #[test]
    fn accepts_json() {
        let env = compile_text(r#"{"task":{"objective":"x"},"latency":{"class":"NOW","deadline":"30s"}}"#).unwrap();
        assert_eq!(env.latency.class, Urgency::Now);
        assert_eq!(env.latency.deadline_at, Some(31_000));
    }
}
