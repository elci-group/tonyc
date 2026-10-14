//! Building work requests.

use std::path::Path;
use tonyc_core::cognition::{CognitiveKind, FreshnessPolicy};
use tonyc_core::envelope::{
    DeadlineSpec, Energy, NodeFailure, OriginRequest, OutputSchema, Refine, ResourceMode, Urgency, Verification,
    WorkRequest,
};
use tonyc_core::refine::Check;
use tonyc_core::{Locality, Millis};

/// A unit of cognition to ask Tonyc for. Everything but the objective is
/// optional; the daemon fills the rest from its policy (by default
/// `LOCAL_ONLY`, `DEFERRED`).
///
/// ```
/// use tonyc_sdk::{Locality, Urgency, Work};
/// let work = Work::new("Summarise the dependency changes")
///     .file("/srv/repo/Cargo.lock")
///     .urgency(Urgency::Deferred)
///     .deadline_in("2h")
///     .locality(Locality::LocalOnly)
///     .floor(0.7)
///     .scope("project:billing");
/// assert_eq!(work.request().task.objective, "Summarise the dependency changes");
/// ```
#[derive(Debug, Clone)]
pub struct Work(WorkRequest);

impl Work {
    pub fn new(objective: impl Into<String>) -> Work {
        let mut req = WorkRequest::default();
        req.task.objective = objective.into();
        Work(req)
    }

    /// Wraps a request built elsewhere (for example parsed from YAML).
    pub fn from_request(req: WorkRequest) -> Work {
        Work(req)
    }

    pub fn request(&self) -> &WorkRequest {
        &self.0
    }

    pub fn into_request(self) -> WorkRequest {
        self.0
    }

    /// Text the objective is about.
    pub fn input(mut self, text: impl Into<String>) -> Work {
        self.0.task.input.text = Some(text.into());
        self
    }

    /// A file the work reads (absolute path). Results depending on files are
    /// invalidated when the files change.
    pub fn file(mut self, path: impl AsRef<Path>) -> Work {
        self.0.task.input.files.push(path.as_ref().to_string_lossy().into_owned());
        self
    }

    /// The capability needed (`general`, `code.rust`, ...).
    pub fn capability(mut self, capability: impl Into<String>) -> Work {
        self.0.task.capability = Some(capability.into());
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Work {
        self.0.task.max_tokens = Some(n);
        self
    }

    /// Run on exactly this model (`<runtime>/<model>`).
    pub fn pin_model(mut self, model: impl Into<String>) -> Work {
        self.0.task.pin_model = Some(model.into());
        self
    }

    pub fn urgency(mut self, class: Urgency) -> Work {
        self.0.latency.class = Some(class);
        self
    }

    /// A relative deadline such as `45m` or `2h`.
    pub fn deadline_in(mut self, duration: impl Into<String>) -> Work {
        self.0.latency.deadline = Some(DeadlineSpec::In(duration.into()));
        self
    }

    /// An absolute deadline in milliseconds since the Unix epoch.
    pub fn deadline_at(mut self, at: Millis) -> Work {
        self.0.latency.deadline = Some(DeadlineSpec::AtMillis(at));
        self
    }

    /// The lowest acceptable estimated quality, 0 to 1.
    pub fn floor(mut self, floor: f64) -> Work {
        self.0.quality.floor = Some(floor);
        self
    }

    pub fn verification(mut self, v: Verification) -> Work {
        self.0.quality.verification = Some(v);
        self
    }

    /// A machine-checkable requirement on the answer.
    pub fn check(mut self, check: Check) -> Work {
        self.0.quality.checks.push(check);
        self
    }

    pub fn refine(mut self, refine: Refine) -> Work {
        self.0.quality.refine = Some(refine);
        self
    }

    /// The trust boundary the work must stay inside. The daemon's ceiling
    /// can only narrow it.
    pub fn locality(mut self, locality: Locality) -> Work {
        self.0.privacy.locality = Some(locality);
        self
    }

    /// SSH nodes allowed under `SSH_ALLOWLIST`.
    pub fn ssh_node(mut self, node: impl Into<String>) -> Work {
        self.0.privacy.ssh_allowlist.push(node.into());
        self
    }

    pub fn opportunistic(mut self) -> Work {
        self.0.resources.cpu = Some(ResourceMode::Opportunistic);
        self.0.resources.gpu = Some(ResourceMode::Opportunistic);
        self
    }

    pub fn energy(mut self, energy: Energy) -> Work {
        self.0.resources.energy = Some(energy);
        self
    }

    pub fn on_node_failure(mut self, policy: NodeFailure) -> Work {
        self.0.resources.on_node_failure = Some(policy);
        self
    }

    /// Whether existing cognition may answer (default true).
    pub fn cache(mut self, cache: bool) -> Work {
        self.0.reuse.cache = Some(cache);
        self
    }

    /// The cognition scope, e.g. `project:billing` (default `user`).
    pub fn scope(mut self, scope: impl Into<String>) -> Work {
        self.0.reuse.corpus = Some(scope.into());
        self
    }

    pub fn freshness(mut self, policy: FreshnessPolicy) -> Work {
        self.0.reuse.freshness = Some(policy);
        self
    }

    /// How long a `ttl` answer stays fresh, e.g. `24h`.
    pub fn ttl(mut self, ttl: impl Into<String>) -> Work {
        self.0.reuse.ttl = Some(ttl.into());
        self
    }

    /// Ask for a JSON object as the answer.
    pub fn json(mut self) -> Work {
        self.0.output.schema = Some(OutputSchema::Json);
        self
    }

    pub fn kind(mut self, kind: CognitiveKind) -> Work {
        self.0.output.kind = Some(kind);
        self
    }

    /// Attribute the work to an orchestrator or agent system.
    pub fn origin(mut self, system: impl Into<String>, run: Option<String>, agent: Option<String>) -> Work {
        self.0.origin = OriginRequest { system: Some(system.into()), run, agent };
        self
    }

    pub(crate) fn has_origin(&self) -> bool {
        self.0.origin.system.is_some()
    }
}

impl From<WorkRequest> for Work {
    fn from(req: WorkRequest) -> Work {
        Work(req)
    }
}
