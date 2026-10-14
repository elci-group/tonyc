//! Tonyc contracts (directive TONYC/1, Phase 0).
//!
//! Everything a caller or an execution domain can rely on lives here and is
//! runtime-neutral: the work envelope and how a request compiles into it,
//! locality semantics, the canonical job states, model capability profiles,
//! the adapter request/response shapes and the planner that turns an envelope
//! plus a set of candidate models into an inspectable decision.
//!
//! Nothing in this crate performs I/O. The daemon (`tonycd`) owns scheduling,
//! persistence and execution; this crate only states the rules.

pub mod cognition;
pub mod duration;
pub mod envelope;
pub mod locality;
pub mod model;
pub mod plan;
pub mod refine;
pub mod state;
pub mod transport;

pub use envelope::{CompileError, WorkEnvelope, WorkRequest, compile};
pub use locality::{Exposure, Locality, LocalityPolicy};
pub use model::{InferenceOutput, InferenceRequest, ModelProfile};
pub use plan::{Candidate, Decision, PlanFailure, PlanInput, Rejection, plan};
pub use state::JobState;

/// Milliseconds since the Unix epoch. All Tonyc timestamps use this unit.
pub type Millis = i64;
