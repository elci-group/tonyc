//! The ecosystem-facing parts of the protocol (directive Phase 7, §28):
//! version negotiation, idempotent submission, the durable event feed and
//! attribution of work to the orchestrator that asked for it.
//!
//! Orchestrators retry, restart and run several agents at once. They need to
//! submit without duplicating work, learn about state changes without
//! polling every job, and see what Tonyc did for each of them. All of this
//! lives in the state database, so it survives a daemon restart exactly as
//! the queue does.

use crate::store::{Store, now_ms};
use anyhow::Result;
use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tonyc_core::{JobState, Millis, WorkEnvelope};

/// The wire protocol name. A client checks it before anything else.
pub const PROTOCOL: &str = "tonyc/1";
/// The protocol revision: minor revisions only add endpoints, fields and
/// features; anything incompatible becomes `tonyc/2` under `/v2/`.
pub const REVISION: &str = "1.1";

/// What this daemon implements, so clients can degrade instead of guessing.
pub const FEATURES: &[&str] = &[
    "submit",
    "status",
    "await",
    "cancel",
    "try",
    "query",
    "invalidate",
    "events",
    "idempotency",
    "origin",
    "refinement",
    "preemptive_dynamism",
    "ssh_fabric",
    "economics",
];

/// How long an idempotency key is remembered.
pub const IDEMPOTENCY_TTL_MS: Millis = 24 * 3_600_000;

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS idempotency (
    key          TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    at           INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idempotency_at ON idempotency (at);
"#;

/// The outcome of an idempotent submission.
#[derive(Debug, PartialEq)]
pub enum Submitted {
    /// The job was created now.
    Created,
    /// The key was used before for the same request: this is that job.
    Replayed(String),
    /// The key was used before for a different request.
    Conflict(String),
}

/// Checks an idempotency key's shape: 1-200 printable ASCII characters.
pub fn valid_key(key: &str) -> bool {
    !key.is_empty() && key.len() <= 200 && key.bytes().all(|b| b.is_ascii_graphic())
}

/// A canonical hash of a request (its parsed form, so formatting and YAML
/// versus JSON do not matter).
pub fn request_hash(req: &tonyc_core::WorkRequest) -> String {
    let canonical = serde_json::to_vec(req).expect("a request serialises");
    blake3::hash(&canonical).to_hex().to_string()
}

/// One entry of the event feed.
#[derive(Debug, Clone, Serialize)]
pub struct FeedEvent {
    pub seq: i64,
    pub at: Millis,
    pub job_id: String,
    pub kind: String,
    pub state: Option<JobState>,
    pub terminal: bool,
    pub origin: Option<Value>,
}

impl Store {
    /// Inserts a job, or returns the job an earlier use of `key` created.
    /// The check and the insert are one transaction, so two concurrent
    /// submissions with the same key create one job.
    pub fn insert_job_idempotent(&self, env: &WorkEnvelope, key: &str, request_hash: &str) -> Result<Submitted> {
        {
            let mut conn = self.conn();
            let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            tx.execute("DELETE FROM idempotency WHERE at < ?1", [now_ms() - IDEMPOTENCY_TTL_MS])?;
            let earlier: Option<(String, String)> = tx
                .query_row("SELECT request_hash, job_id FROM idempotency WHERE key = ?1", [key], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .optional()?;
            if let Some((hash, job)) = earlier {
                return Ok(if hash == request_hash { Submitted::Replayed(job) } else { Submitted::Conflict(job) });
            }
            tx.execute(
                "INSERT INTO idempotency (key, request_hash, job_id, at) VALUES (?1, ?2, ?3, ?4)",
                params![key, request_hash, env.id, now_ms()],
            )?;
            tx.commit()?;
        }
        // The key is recorded before the job: if inserting the job fails the
        // key is released again, so a retry can succeed.
        if let Err(e) = self.insert_job(env) {
            self.conn().execute("DELETE FROM idempotency WHERE key = ?1", [key])?;
            return Err(e);
        }
        Ok(Submitted::Created)
    }

    /// Job events after `after`, oldest first, optionally only those of
    /// jobs from one origin system.
    pub fn feed(&self, after: i64, limit: usize, system: Option<&str>) -> Result<Vec<FeedEvent>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT e.seq, e.at, e.job_id, e.kind, e.state, json_extract(j.envelope, '$.origin')
               FROM events e JOIN jobs j ON j.id = e.job_id
              WHERE e.seq > ?1 AND (?2 IS NULL OR json_extract(j.envelope, '$.origin.system') = ?2)
              ORDER BY e.seq LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![after, system, limit as i64], |r| {
                let state: Option<String> = r.get(4)?;
                let state = state.and_then(|s| s.parse::<JobState>().ok());
                let origin: Option<String> = r.get(5)?;
                Ok(FeedEvent {
                    seq: r.get(0)?,
                    at: r.get(1)?,
                    job_id: r.get(2)?,
                    kind: r.get(3)?,
                    terminal: state.is_some_and(|s| s.is_terminal()),
                    state,
                    origin: origin.and_then(|o| serde_json::from_str(&o).ok()),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// The newest event sequence number (0 when there are none).
    pub fn feed_head(&self) -> Result<i64> {
        Ok(self.conn().query_row("SELECT COALESCE(MAX(seq), 0) FROM events", [], |r| r.get(0))?)
    }

    /// The ledger since `since`, grouped by the origin system of each job.
    /// Jobs submitted without an origin are grouped as `unattributed`.
    pub fn ledger_by_origin(&self, since: Millis) -> Result<BTreeMap<String, Value>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT COALESCE(json_extract(j.envelope, '$.origin.system'), 'unattributed'),
                    SUM(l.kind = 'inference'), SUM(l.kind = 'reuse'), SUM(l.wall_ms),
                    SUM(l.conventional_gbp), SUM(l.spend_gbp), SUM(l.energy_gbp)
               FROM ledger l JOIN jobs j ON j.id = l.job_id
              WHERE l.at >= ?1
              GROUP BY 1 ORDER BY 1",
        )?;
        let rows = stmt
            .query_map([since], |r| {
                let conventional: f64 = r.get(4)?;
                let spend: f64 = r.get(5)?;
                let energy: f64 = r.get(6)?;
                Ok((
                    r.get::<_, String>(0)?,
                    json!({
                        "inference": r.get::<_, i64>(1)?,
                        "served_from_store": r.get::<_, i64>(2)?,
                        "compute_hours": r.get::<_, i64>(3)? as f64 / 3.6e6,
                        "conventional_gbp": conventional,
                        "operating_gbp": spend + energy,
                        "estimated_cost_avoided_gbp": (conventional - spend).max(0.0),
                    }),
                ))
            })?
            .collect::<rusqlite::Result<BTreeMap<_, _>>>()?;
        Ok(rows)
    }
}

/// The body of `GET /v1/protocol`.
pub fn describe(socket: &std::path::Path, tcp: Option<&str>) -> Value {
    json!({
        "protocol": PROTOCOL,
        "revision": REVISION,
        "daemon": {"name": "tonycd", "version": env!("CARGO_PKG_VERSION")},
        "features": FEATURES,
        "envelope": tonyc_core::envelope::ENVELOPE_VERSION,
        "transport": {"unix": socket, "tcp": tcp},
        "limits": {
            "await_timeout_ms_max": 600_000,
            "events_wait_ms_max": 60_000,
            "events_limit_max": 1000,
            "idempotency_ttl_ms": IDEMPOTENCY_TTL_MS,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonyc_core::envelope::{CompileDefaults, WorkRequest, compile};

    fn env(id: &str, yaml: &str) -> (WorkEnvelope, String) {
        let req = WorkRequest::parse(yaml).unwrap();
        (compile(&req, id, now_ms(), &CompileDefaults::default()).unwrap(), request_hash(&req))
    }

    #[test]
    fn a_key_creates_one_job() {
        let s = Store::open_in_memory().unwrap();
        let (a, ha) = env("tny_a", "task: {objective: x}");
        assert_eq!(s.insert_job_idempotent(&a, "k1", &ha).unwrap(), Submitted::Created);
        // Same request, formatted differently: the same job.
        let (b, hb) = env("tny_b", "{\"task\": {\"objective\": \"x\"}}");
        assert_eq!(hb, ha);
        assert_eq!(s.insert_job_idempotent(&b, "k1", &hb).unwrap(), Submitted::Replayed("tny_a".into()));
        assert!(s.get("tny_b").unwrap().is_none());
        // A different request under the same key is refused.
        let (c, hc) = env("tny_c", "task: {objective: y}");
        assert_eq!(s.insert_job_idempotent(&c, "k1", &hc).unwrap(), Submitted::Conflict("tny_a".into()));
        assert_eq!(s.insert_job_idempotent(&c, "k2", &hc).unwrap(), Submitted::Created);
    }

    #[test]
    fn keys_expire() {
        let s = Store::open_in_memory().unwrap();
        let (a, h) = env("tny_a", "task: {objective: x}");
        s.insert_job_idempotent(&a, "k", &h).unwrap();
        s.raw(&format!("UPDATE idempotency SET at = {}", now_ms() - IDEMPOTENCY_TTL_MS - 1)).unwrap();
        let (b, h) = env("tny_b", "task: {objective: x}");
        assert_eq!(s.insert_job_idempotent(&b, "k", &h).unwrap(), Submitted::Created);
    }

    #[test]
    fn key_shape() {
        assert!(valid_key("lucid:task-7:attempt-1"));
        assert!(!valid_key(""));
        assert!(!valid_key("has space"));
        assert!(!valid_key(&"x".repeat(201)));
    }

    #[test]
    fn feed_filters_by_origin_and_pages() {
        let s = Store::open_in_memory().unwrap();
        s.insert_job(&env("tny_a", "task: {objective: a}\norigin: {system: lucid, run: t1}").0).unwrap();
        s.insert_job(&env("tny_b", "task: {objective: b}").0).unwrap();
        s.transition("tny_a", &[], JobState::Running, "started", None).unwrap();
        let all = s.feed(0, 100, None).unwrap();
        assert_eq!(all.len(), 3);
        let lucid = s.feed(0, 100, Some("lucid")).unwrap();
        assert_eq!(lucid.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>(), ["submitted", "started"]);
        assert_eq!(lucid[0].origin.as_ref().unwrap()["run"], "t1");
        assert!(all[1].origin.is_none());
        let page = s.feed(all[0].seq, 1, None).unwrap();
        assert_eq!(page[0].job_id, "tny_b");
        assert_eq!(s.feed_head().unwrap(), all[2].seq);
    }
}
