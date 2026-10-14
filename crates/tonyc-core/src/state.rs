//! Canonical job states (directive §26). Tonyc fails explicitly.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobState {
    Queued,
    Planning,
    Running,
    Refining,
    Ready,
    Stale,
    Cancelled,
    DeadlineExceeded,
    /// Waiting: the host is under pressure and the job's class yields to it.
    ResourceBlocked,
    PolicyBlocked,
    ModelFailed,
    NodeUnreachable,
    InsufficientQuality,
}

impl JobState {
    pub const ALL: [JobState; 13] = [
        JobState::Queued,
        JobState::Planning,
        JobState::Running,
        JobState::Refining,
        JobState::Ready,
        JobState::Stale,
        JobState::Cancelled,
        JobState::DeadlineExceeded,
        JobState::ResourceBlocked,
        JobState::PolicyBlocked,
        JobState::ModelFailed,
        JobState::NodeUnreachable,
        JobState::InsufficientQuality,
    ];

    /// A terminal job never runs again. `READY` may later become `STALE`
    /// when its sources change (Phase 2), but it is not re-executed in place.
    pub fn is_terminal(self) -> bool {
        !matches!(
            self,
            JobState::Queued | JobState::Planning | JobState::Running | JobState::Refining | JobState::ResourceBlocked
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Queued => "QUEUED",
            JobState::Planning => "PLANNING",
            JobState::Running => "RUNNING",
            JobState::Refining => "REFINING",
            JobState::Ready => "READY",
            JobState::Stale => "STALE",
            JobState::Cancelled => "CANCELLED",
            JobState::DeadlineExceeded => "DEADLINE_EXCEEDED",
            JobState::ResourceBlocked => "RESOURCE_BLOCKED",
            JobState::PolicyBlocked => "POLICY_BLOCKED",
            JobState::ModelFailed => "MODEL_FAILED",
            JobState::NodeUnreachable => "NODE_UNREACHABLE",
            JobState::InsufficientQuality => "INSUFFICIENT_QUALITY",
        }
    }
}

impl fmt::Display for JobState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for JobState {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let norm = s.trim().to_ascii_uppercase().replace('-', "_");
        JobState::ALL.into_iter().find(|st| st.as_str() == norm).ok_or_else(|| format!("unknown job state `{s}`"))
    }
}
