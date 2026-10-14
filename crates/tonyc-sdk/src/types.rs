//! What the daemon answers with. Every type ignores fields it does not know,
//! so a client keeps working against a newer daemon (protocol revisions only
//! add fields).

use serde::Deserialize;
use serde_json::Value;
use tonyc_core::{Exposure, JobState, Millis, WorkEnvelope};

/// `GET /v1/protocol`.
#[derive(Debug, Clone, Deserialize)]
pub struct Protocol {
    pub protocol: String,
    pub revision: String,
    pub features: Vec<String>,
    pub envelope: String,
    #[serde(default)]
    pub daemon: Value,
}

impl Protocol {
    pub fn supports(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
}

/// The answer to a submission.
#[derive(Debug, Clone, Deserialize)]
pub struct Submission {
    pub id: String,
    pub state: JobState,
    pub envelope: WorkEnvelope,
    /// True when an idempotency key matched an earlier submission and this
    /// is that job.
    #[serde(default)]
    pub replayed: bool,
}

/// A job with its result, if it has one.
#[derive(Debug, Clone, Deserialize)]
pub struct Job {
    #[serde(rename = "job")]
    pub info: JobInfo,
    pub result: Option<JobResult>,
    #[serde(default)]
    pub events: Vec<JobEvent>,
}

impl Job {
    pub fn id(&self) -> &str {
        &self.info.id
    }

    pub fn state(&self) -> JobState {
        self.info.state
    }

    pub fn is_terminal(&self) -> bool {
        self.info.state.is_terminal()
    }

    /// The answer, when the job is `READY`.
    pub fn output(&self) -> Option<&str> {
        (self.info.state == JobState::Ready).then_some(()).and(self.result.as_ref()).map(|r| r.output.as_str())
    }

    /// Why the job did not produce an answer, when it failed.
    pub fn failure_reason(&self) -> Option<&str> {
        self.info.failure.as_ref().and_then(|f| f.get("reason")).and_then(Value::as_str)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobInfo {
    pub id: String,
    pub state: JobState,
    pub envelope: WorkEnvelope,
    pub submitted_at: Millis,
    pub updated_at: Millis,
    #[serde(default)]
    pub attempts: i64,
    /// The scheduling decision (model, domain, utility, rejected
    /// alternatives).
    pub decision: Option<Value>,
    pub failure: Option<Value>,
    pub cognition: Option<Value>,
}

/// A result with its provenance.
#[derive(Debug, Clone, Deserialize)]
pub struct JobResult {
    pub output: String,
    pub schema: String,
    pub model: String,
    pub runtime: String,
    pub model_digest: Option<String>,
    pub exposure: Exposure,
    pub semantic_hash: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub started_at: Millis,
    pub finished_at: Millis,
    /// Set when the answer came from the Cognitive Store without inference.
    pub reused_from: Option<String>,
    pub object_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobEvent {
    pub seq: i64,
    pub at: Millis,
    pub kind: String,
    pub state: Option<JobState>,
    pub detail: Option<Value>,
}

/// A cognitive object as listed by `query` or returned by `try`.
#[derive(Debug, Clone, Deserialize)]
pub struct CognitiveObject {
    pub id: String,
    pub kind: String,
    pub state: String,
    pub scope: String,
    pub objective: String,
    pub model: Option<String>,
    pub confidence: f64,
    /// `unverified`, `verified` or `failed`.
    #[serde(default)]
    pub validation: Option<String>,
    /// The most sensitive source it was derived from: `public`, `normal`,
    /// `sensitive` or `secret`. Sensitive and secret cognition was produced
    /// on this host only; do not forward it anywhere else.
    #[serde(default)]
    pub sensitivity: Option<String>,
    pub created_at: Millis,
    /// Withheld (null) in listings of `secret` objects.
    pub content: Option<String>,
    #[serde(default)]
    pub withheld: bool,
    pub score: Option<f64>,
    pub job_id: Option<String>,
    #[serde(default)]
    pub freshness: Value,
}

/// The answer to `try`.
#[derive(Debug, Clone)]
pub enum TryOutcome {
    /// Existing cognition satisfies the request.
    Hit(Box<CognitiveObject>),
    /// Nothing suitable exists now; `nearest` are related objects, and
    /// `skipped` explains candidates that did not qualify. The miss is
    /// recorded as demand, so repeated misses can lead Tonyc to prepare
    /// the answer ahead of time.
    Miss { nearest: Vec<CognitiveObject>, skipped: Vec<Value> },
}

impl TryOutcome {
    pub fn hit(&self) -> Option<&CognitiveObject> {
        match self {
            TryOutcome::Hit(o) => Some(o),
            TryOutcome::Miss { .. } => None,
        }
    }
}

/// One entry of the event feed.
#[derive(Debug, Clone, Deserialize)]
pub struct FeedEvent {
    pub seq: i64,
    pub at: Millis,
    pub job_id: String,
    pub kind: String,
    pub state: Option<JobState>,
    pub terminal: bool,
    pub origin: Option<Value>,
}

/// A page of the event feed. Resume from `next`.
#[derive(Debug, Clone, Deserialize)]
pub struct EventPage {
    pub events: Vec<FeedEvent>,
    pub next: i64,
}

/// What to invalidate: exactly one selector.
#[derive(Debug, Clone)]
pub enum Invalidate {
    Object(String),
    /// Everything derived from files under this path.
    Path(String),
    Scope(String),
    /// Everything with `event` freshness listening for this tag.
    Event(String),
}

/// A `query` for existing cognition.
#[derive(Debug, Clone, Default)]
pub struct Query {
    pub text: Option<String>,
    pub scope: Option<String>,
    pub kind: Option<String>,
    /// `FRESH` (default), `STALE` or `ALL`.
    pub state: Option<String>,
    pub limit: Option<usize>,
}

impl Query {
    pub fn text(text: impl Into<String>) -> Query {
        Query { text: Some(text.into()), ..Query::default() }
    }

    pub fn scope(mut self, scope: impl Into<String>) -> Query {
        self.scope = Some(scope.into());
        self
    }

    pub fn limit(mut self, limit: usize) -> Query {
        self.limit = Some(limit);
        self
    }
}
