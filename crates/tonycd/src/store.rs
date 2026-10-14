//! Persistent state (SQLite).
//!
//! The queue, every state transition, every result and every boundary
//! crossing live here, so work survives a daemon restart. The `events` and
//! `boundary` tables are append-only: triggers reject UPDATE and DELETE, so
//! provenance cannot be rewritten through this API or by a stray query.

use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::sync::Mutex;
use tonyc_core::envelope::WorkEnvelope;
use tonyc_core::{Exposure, JobState, Millis};

const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    envelope      TEXT NOT NULL,
    semantic_hash TEXT NOT NULL,
    rank          INTEGER NOT NULL,
    state         TEXT NOT NULL,
    deadline_at   INTEGER,
    submitted_at  INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    decision      TEXT,
    failure       TEXT
);
CREATE INDEX IF NOT EXISTS jobs_waiting ON jobs (state, rank, deadline_at, submitted_at);
CREATE TABLE IF NOT EXISTS events (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL REFERENCES jobs(id),
    at      INTEGER NOT NULL,
    kind    TEXT NOT NULL,
    state   TEXT,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS events_job ON events (job_id, seq);
CREATE TABLE IF NOT EXISTS results (
    job_id        TEXT PRIMARY KEY REFERENCES jobs(id),
    output        TEXT NOT NULL,
    schema        TEXT NOT NULL,
    model         TEXT NOT NULL,
    runtime       TEXT NOT NULL,
    model_digest  TEXT,
    exposure      TEXT NOT NULL,
    semantic_hash TEXT NOT NULL,
    sources       TEXT NOT NULL,
    tokens_in     INTEGER NOT NULL,
    tokens_out    INTEGER NOT NULL,
    started_at    INTEGER NOT NULL,
    finished_at   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS boundary (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL REFERENCES jobs(id),
    at             INTEGER NOT NULL,
    exposure       TEXT NOT NULL,
    leaves_host    INTEGER NOT NULL,
    bytes_sent     INTEGER NOT NULL,
    bytes_received INTEGER NOT NULL,
    summary        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS objects (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    semantic_key  TEXT NOT NULL,
    scope         TEXT NOT NULL,
    objective     TEXT NOT NULL,
    content       TEXT NOT NULL,
    schema        TEXT NOT NULL,
    confidence    REAL NOT NULL,
    validation    TEXT NOT NULL,
    freshness     TEXT NOT NULL,
    expires_at    INTEGER,
    state         TEXT NOT NULL,
    stale_reason  TEXT,
    stale_at      INTEGER,
    created_at    INTEGER NOT NULL,
    job_id        TEXT NOT NULL,
    model         TEXT NOT NULL,
    model_digest  TEXT,
    runtime       TEXT NOT NULL,
    exposure      TEXT NOT NULL,
    privacy       TEXT NOT NULL,
    sensitivity   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS objects_key ON objects (semantic_key, state, created_at);
CREATE INDEX IF NOT EXISTS objects_state ON objects (state, created_at);
CREATE TABLE IF NOT EXISTS object_sources (
    object_id TEXT NOT NULL,
    path      TEXT NOT NULL,
    blake3    TEXT NOT NULL,
    bytes     INTEGER NOT NULL,
    PRIMARY KEY (object_id, path)
);
CREATE INDEX IF NOT EXISTS object_sources_path ON object_sources (path);
CREATE TABLE IF NOT EXISTS object_parents (
    object_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    relation  TEXT NOT NULL,
    PRIMARY KEY (object_id, parent_id)
);
CREATE INDEX IF NOT EXISTS object_parents_parent ON object_parents (parent_id);
CREATE TABLE IF NOT EXISTS cognition_events (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT NOT NULL,
    at        INTEGER NOT NULL,
    kind      TEXT NOT NULL,
    detail    TEXT
);
CREATE INDEX IF NOT EXISTS cognition_events_object ON cognition_events (object_id, seq);
CREATE TABLE IF NOT EXISTS corpus (
    path        TEXT PRIMARY KEY,
    scope       TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    blake3      TEXT,
    bytes       INTEGER,
    mode        INTEGER,
    added_at    INTEGER NOT NULL,
    scanned_at  INTEGER NOT NULL,
    present     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS file_hashes (
    path       TEXT PRIMARY KEY,
    size       INTEGER NOT NULL,
    mtime_ns   INTEGER NOT NULL,
    blake3     TEXT NOT NULL,
    checked_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS demand (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    at           INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    semantic_key TEXT,
    scope        TEXT,
    objective    TEXT,
    object_id    TEXT,
    job_id       TEXT
);
CREATE TABLE IF NOT EXISTS nodes (
    name     TEXT PRIMARY KEY,
    spec     TEXT NOT NULL,
    added_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS cognition_events_no_update BEFORE UPDATE ON cognition_events
    BEGIN SELECT RAISE(ABORT, 'cognition events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS cognition_events_no_delete BEFORE DELETE ON cognition_events
    BEGIN SELECT RAISE(ABORT, 'cognition events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS demand_no_update BEFORE UPDATE ON demand
    BEGIN SELECT RAISE(ABORT, 'demand records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS demand_no_delete BEFORE DELETE ON demand
    BEGIN SELECT RAISE(ABORT, 'demand records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS boundary_no_update BEFORE UPDATE ON boundary
    BEGIN SELECT RAISE(ABORT, 'boundary records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS boundary_no_delete BEFORE DELETE ON boundary
    BEGIN SELECT RAISE(ABORT, 'boundary records are append-only'); END;
"#;

pub struct Store {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobRow {
    pub id: String,
    pub state: JobState,
    pub envelope: WorkEnvelope,
    pub submitted_at: Millis,
    pub updated_at: Millis,
    pub attempts: i64,
    pub decision: Option<Value>,
    pub failure: Option<Value>,
    /// How the Cognitive Store took part: reuse, context, narrowing, the
    /// object produced.
    pub cognition: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventRow {
    pub seq: i64,
    pub at: Millis,
    pub kind: String,
    pub state: Option<JobState>,
    pub detail: Option<Value>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct SourceRef {
    pub path: String,
    pub blake3: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResultRow {
    pub job_id: String,
    pub output: String,
    pub schema: String,
    pub model: String,
    pub runtime: String,
    pub model_digest: Option<String>,
    pub exposure: Exposure,
    pub semantic_hash: String,
    pub sources: Vec<SourceRef>,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub started_at: Millis,
    pub finished_at: Millis,
    /// The cognitive object that satisfied this job without inference.
    pub reused_from: Option<String>,
    /// The cognitive object this result is (produced or reused).
    pub object_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundaryRow {
    pub seq: i64,
    pub at: Millis,
    pub exposure: Exposure,
    pub leaves_host: bool,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub summary: String,
    pub payload_blake3: Option<String>,
    /// Whether the exact payload is stored (always, for off-host records).
    pub payload_kept: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    pub job_id: String,
}

pub fn now_ms() -> Millis {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as Millis).unwrap_or(0)
}

pub(crate) fn state_of(text: String) -> rusqlite::Result<JobState> {
    text.parse()
        .map_err(|e: String| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, e.into()))
}

pub(crate) fn json_of<T: serde::de::DeserializeOwned>(text: String) -> rusqlite::Result<T> {
    serde_json::from_str(&text)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e)))
}

const JOB_COLS: &str = "id, state, envelope, submitted_at, updated_at, attempts, decision, failure, cognition";

fn job_from_row(r: &rusqlite::Row) -> rusqlite::Result<JobRow> {
    Ok(JobRow {
        id: r.get(0)?,
        state: state_of(r.get(1)?)?,
        envelope: json_of(r.get(2)?)?,
        submitted_at: r.get(3)?,
        updated_at: r.get(4)?,
        attempts: r.get(5)?,
        decision: r.get::<_, Option<String>>(6)?.map(json_of).transpose()?,
        failure: r.get::<_, Option<String>>(7)?.map(json_of).transpose()?,
        cognition: r.get::<_, Option<String>>(8)?.map(json_of).transpose()?,
    })
}

fn append_event(tx: &Connection, job: &str, kind: &str, state: Option<JobState>, detail: Option<&Value>) -> Result<()> {
    tx.execute(
        "INSERT INTO events (job_id, at, kind, state, detail) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![job, now_ms(), kind, state.map(|s| s.as_str()), detail.map(|d| d.to_string())],
    )?;
    Ok(())
}

impl Store {
    pub fn open(path: &Path) -> Result<Store> {
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        restrict_permissions(path)?;
        conn.execute_batch(SCHEMA).context("initialising schema")?;
        conn.execute_batch(crate::economics::SCHEMA).context("initialising economics schema")?;
        conn.execute_batch(crate::bench::SCHEMA).context("initialising benchmark schema")?;
        conn.execute_batch(crate::refinement::SCHEMA).context("initialising refinement schema")?;
        conn.execute_batch(crate::pd::SCHEMA).context("initialising PD schema")?;
        conn.execute_batch(crate::protocol::SCHEMA).context("initialising protocol schema")?;
        migrate(&conn)?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    pub fn open_in_memory() -> Result<Store> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        conn.execute_batch(crate::economics::SCHEMA)?;
        conn.execute_batch(crate::bench::SCHEMA)?;
        conn.execute_batch(crate::refinement::SCHEMA)?;
        conn.execute_batch(crate::pd::SCHEMA)?;
        conn.execute_batch(crate::protocol::SCHEMA)?;
        migrate(&conn)?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    pub(crate) fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn insert_job(&self, env: &WorkEnvelope) -> Result<()> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO jobs (id, envelope, semantic_hash, rank, state, deadline_at, submitted_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            params![
                env.id,
                serde_json::to_string(env)?,
                env.semantic_hash,
                env.latency.class.rank(),
                JobState::Queued.as_str(),
                env.latency.deadline_at,
                env.submitted_at,
            ],
        )?;
        append_event(&tx, &env.id, "submitted", Some(JobState::Queued), Some(&serde_json::to_value(env)?))?;
        tx.commit()?;
        Ok(())
    }

    /// Moves a job to `to` if it is currently in one of `from` (any state if
    /// `from` is empty). Returns whether the transition happened. The state
    /// change and its event are written atomically.
    pub fn transition(
        &self,
        id: &str,
        from: &[JobState],
        to: JobState,
        kind: &str,
        detail: Option<Value>,
    ) -> Result<bool> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        let current: Option<String> =
            tx.query_row("SELECT state FROM jobs WHERE id = ?1", [id], |r| r.get(0)).optional()?;
        let Some(current) = current else { return Err(anyhow!("no such job {id}")) };
        let current = state_of(current)?;
        if !from.is_empty() && !from.contains(&current) {
            return Ok(false);
        }
        let bump = i64::from(to == JobState::Running && current != JobState::Running);
        tx.execute(
            "UPDATE jobs SET state = ?2, updated_at = ?3, attempts = attempts + ?4 WHERE id = ?1",
            params![id, to.as_str(), now_ms(), bump],
        )?;
        append_event(&tx, id, kind, Some(to), detail.as_ref())?;
        tx.commit()?;
        Ok(true)
    }

    pub fn note(&self, id: &str, kind: &str, detail: Value) -> Result<()> {
        append_event(&self.conn(), id, kind, None, Some(&detail))
    }

    pub fn set_decision(&self, id: &str, decision: &Value) -> Result<()> {
        self.conn().execute("UPDATE jobs SET decision = ?2 WHERE id = ?1", params![id, decision.to_string()])?;
        Ok(())
    }

    pub fn set_cognition(&self, id: &str, cognition: &Value) -> Result<()> {
        self.conn().execute("UPDATE jobs SET cognition = ?2 WHERE id = ?1", params![id, cognition.to_string()])?;
        Ok(())
    }

    pub fn set_failure(&self, id: &str, failure: &Value) -> Result<()> {
        self.conn().execute("UPDATE jobs SET failure = ?2 WHERE id = ?1", params![id, failure.to_string()])?;
        Ok(())
    }

    /// Jobs the scheduler may start, most urgent first: urgency class, then
    /// earliest deadline (none last), then submission order.
    pub fn waiting(&self) -> Result<Vec<JobRow>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!(
            "SELECT {JOB_COLS} FROM jobs WHERE state IN ('QUEUED', 'RESOURCE_BLOCKED')
             ORDER BY rank, deadline_at IS NULL, deadline_at, submitted_at, id"
        ))?;
        let rows = stmt.query_map([], job_from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Returns interrupted jobs to the queue after a restart.
    pub fn recover(&self) -> Result<Vec<String>> {
        let ids: Vec<String> = {
            let conn = self.conn();
            let mut stmt = conn.prepare("SELECT id FROM jobs WHERE state IN ('PLANNING', 'RUNNING', 'REFINING')")?;
            stmt.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?
        };
        for id in &ids {
            self.transition(
                id,
                &[JobState::Planning, JobState::Running, JobState::Refining],
                JobState::Queued,
                "recovered",
                Some(serde_json::json!({"reason": "daemon restarted while the job was in flight"})),
            )?;
        }
        Ok(ids)
    }

    pub fn get(&self, id: &str) -> Result<Option<JobRow>> {
        let conn = self.conn();
        Ok(conn.query_row(&format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"), [id], job_from_row).optional()?)
    }

    pub fn list(&self, state: Option<JobState>, limit: usize) -> Result<Vec<JobRow>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!(
            "SELECT {JOB_COLS} FROM jobs WHERE (?1 IS NULL OR state = ?1) ORDER BY submitted_at DESC, id DESC LIMIT ?2"
        ))?;
        let rows = stmt
            .query_map(params![state.map(|s| s.as_str()), limit as i64], job_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn counts(&self) -> Result<Vec<(JobState, i64)>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT state, COUNT(*) FROM jobs GROUP BY state ORDER BY state")?;
        let rows =
            stmt.query_map([], |r| Ok((state_of(r.get(0)?)?, r.get(1)?)))?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn put_result(&self, r: &ResultRow) -> Result<()> {
        self.conn().execute(
            "INSERT INTO results (job_id, output, schema, model, runtime, model_digest, exposure, semantic_hash,
                                  sources, tokens_in, tokens_out, started_at, finished_at, reused_from, object_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                r.job_id,
                r.output,
                r.schema,
                r.model,
                r.runtime,
                r.model_digest,
                serde_json::to_string(&r.exposure)?,
                r.semantic_hash,
                serde_json::to_string(&r.sources)?,
                r.tokens_in as i64,
                r.tokens_out as i64,
                r.started_at,
                r.finished_at,
                r.reused_from,
                r.object_id,
            ],
        )?;
        Ok(())
    }

    pub fn result(&self, id: &str) -> Result<Option<ResultRow>> {
        let conn = self.conn();
        Ok(conn
            .query_row(
                "SELECT job_id, output, schema, model, runtime, model_digest, exposure, semantic_hash, sources,
                        tokens_in, tokens_out, started_at, finished_at, reused_from, object_id
                 FROM results WHERE job_id = ?1",
                [id],
                |r| {
                    Ok(ResultRow {
                        job_id: r.get(0)?,
                        output: r.get(1)?,
                        schema: r.get(2)?,
                        model: r.get(3)?,
                        runtime: r.get(4)?,
                        model_digest: r.get(5)?,
                        exposure: json_of(r.get(6)?)?,
                        semantic_hash: r.get(7)?,
                        sources: json_of(r.get(8)?)?,
                        tokens_in: r.get::<_, i64>(9)? as u64,
                        tokens_out: r.get::<_, i64>(10)? as u64,
                        started_at: r.get(11)?,
                        finished_at: r.get(12)?,
                        reused_from: r.get(13)?,
                        object_id: r.get(14)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn events(&self, id: &str) -> Result<Vec<EventRow>> {
        let conn = self.conn();
        let mut stmt =
            conn.prepare("SELECT seq, at, kind, state, detail FROM events WHERE job_id = ?1 ORDER BY seq")?;
        let rows = stmt
            .query_map([id], |r| {
                Ok(EventRow {
                    seq: r.get(0)?,
                    at: r.get(1)?,
                    kind: r.get(2)?,
                    state: r.get::<_, Option<String>>(3)?.map(state_of).transpose()?,
                    detail: r.get::<_, Option<String>>(4)?.map(json_of).transpose()?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Appends a boundary record. When data leaves the host, the exact
    /// payload is kept with its hash (directive §22: remote execution records
    /// exactly what crossed the boundary).
    pub fn record_boundary(
        &self,
        id: &str,
        exposure: &Exposure,
        sent: u64,
        received: u64,
        summary: &str,
        payload: Option<&[u8]>,
    ) -> Result<()> {
        let (hash, kept) = match payload {
            Some(p) => (Some(blake3::hash(p).to_hex().to_string()), exposure.leaves_host().then(|| p.to_vec())),
            None => (None, None),
        };
        self.conn().execute(
            "INSERT INTO boundary (job_id, at, exposure, leaves_host, bytes_sent, bytes_received, summary,
                                   payload_blake3, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                now_ms(),
                serde_json::to_string(exposure)?,
                exposure.leaves_host(),
                sent as i64,
                received as i64,
                summary,
                hash,
                kept
            ],
        )?;
        Ok(())
    }

    pub fn boundary(&self, id: &str) -> Result<Vec<BoundaryRow>> {
        self.boundary_where("job_id = ?1", &[&id], false, usize::MAX >> 1)
    }

    /// Off-host crossings, newest first, optionally for one job, optionally
    /// with their payloads.
    pub fn audit(&self, job: Option<&str>, with_payload: bool, limit: usize) -> Result<Vec<BoundaryRow>> {
        match job {
            Some(j) => self.boundary_where("leaves_host = 1 AND job_id = ?1", &[&j], with_payload, limit),
            None => self.boundary_where("leaves_host = 1", &[], with_payload, limit),
        }
    }

    fn boundary_where(
        &self,
        clause: &str,
        args: &[&dyn rusqlite::ToSql],
        with_payload: bool,
        limit: usize,
    ) -> Result<Vec<BoundaryRow>> {
        let conn = self.conn();
        let order = if clause.starts_with("job_id") { "seq" } else { "seq DESC" };
        let mut stmt = conn.prepare(&format!(
            "SELECT seq, at, exposure, leaves_host, bytes_sent, bytes_received, summary, payload_blake3, payload, job_id
             FROM boundary WHERE {clause} ORDER BY {order} LIMIT {limit}"
        ))?;
        let rows = stmt
            .query_map(args, |r| {
                let payload: Option<Vec<u8>> = r.get(8)?;
                Ok(BoundaryRow {
                    seq: r.get(0)?,
                    at: r.get(1)?,
                    exposure: json_of(r.get(2)?)?,
                    leaves_host: r.get(3)?,
                    bytes_sent: r.get::<_, i64>(4)? as u64,
                    bytes_received: r.get::<_, i64>(5)? as u64,
                    summary: r.get(6)?,
                    payload_blake3: r.get(7)?,
                    payload_kept: payload.is_some(),
                    payload: if with_payload {
                        payload.map(|p| String::from_utf8_lossy(&p).into_owned())
                    } else {
                        None
                    },
                    job_id: r.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Total boundary crossings that left the host, across all jobs.
    pub fn off_host_crossings(&self) -> Result<i64> {
        Ok(self.conn().query_row("SELECT COUNT(*) FROM boundary WHERE leaves_host = 1", [], |r| r.get(0))?)
    }

    #[cfg(test)]
    pub fn raw(&self, sql: &str) -> rusqlite::Result<usize> {
        self.conn().execute(sql, [])
    }
}

/// Current schema version. 1: Phase 1 queue. 2: Cognitive Store. 3: SSH
/// fabric (nodes, settings, boundary payloads). 4: economics
/// (observations, quality observations, ledger, benchmarks). 5: refinement
/// passes. 6: Preemptive Dynamism (templates, predictions). 7: idempotency
/// keys (and envelopes may carry an `origin`). New tables come
/// from `SCHEMA` and `economics::SCHEMA`; new columns on existing tables are
/// added here.
const SCHEMA_VERSION: i64 = 7;

fn migrate(conn: &Connection) -> Result<()> {
    let has_column = |table: &str, column: &str| -> Result<bool> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let names = stmt.query_map([], |r| r.get::<_, String>(1))?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(names.iter().any(|n| n == column))
    };
    for (table, column) in [
        ("results", "reused_from"),
        ("results", "object_id"),
        ("jobs", "cognition"),
        ("boundary", "payload_blake3"),
        ("boundary", "payload"),
    ] {
        if !has_column(table, column)? {
            conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {column} TEXT"), [])?;
        }
    }
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [SCHEMA_VERSION.to_string()],
    )?;
    Ok(())
}

fn restrict_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if path.exists() {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonyc_core::envelope::{CompileDefaults, WorkRequest, compile};

    fn env(id: &str, yaml: &str, at: Millis) -> WorkEnvelope {
        compile(&WorkRequest::parse(yaml).unwrap(), id, at, &CompileDefaults::default()).unwrap()
    }

    #[test]
    fn waiting_order_is_class_then_deadline_then_fifo() {
        let s = Store::open_in_memory().unwrap();
        s.insert_job(&env("a", "task: {objective: a}\nlatency: {class: idle}", 1)).unwrap();
        s.insert_job(&env("b", "task: {objective: b}", 2)).unwrap();
        s.insert_job(&env("c", "task: {objective: c}\nlatency: {deadline: 1h}", 3)).unwrap();
        s.insert_job(&env("d", "task: {objective: d}\nlatency: {deadline: 5m}", 4)).unwrap();
        s.insert_job(&env("e", "task: {objective: e}\nlatency: {class: now}", 5)).unwrap();
        let order: Vec<_> = s.waiting().unwrap().into_iter().map(|j| j.id).collect();
        assert_eq!(order, ["e", "d", "c", "b", "a"]);
    }

    #[test]
    fn provenance_is_append_only() {
        let s = Store::open_in_memory().unwrap();
        s.insert_job(&env("a", "task: {objective: a}", 1)).unwrap();
        s.record_boundary("a", &Exposure::InProcess, 1, 1, "x", None).unwrap();
        assert!(s.raw("UPDATE events SET kind = 'forged'").is_err());
        assert!(s.raw("DELETE FROM events").is_err());
        assert!(s.raw("UPDATE boundary SET bytes_sent = 0").is_err());
        assert!(s.raw("DELETE FROM boundary").is_err());
    }

    #[test]
    fn conditional_transitions_and_recovery() {
        let s = Store::open_in_memory().unwrap();
        s.insert_job(&env("a", "task: {objective: a}", 1)).unwrap();
        assert!(s.transition("a", &[JobState::Queued], JobState::Running, "started", None).unwrap());
        assert!(!s.transition("a", &[JobState::Queued], JobState::Cancelled, "cancelled", None).unwrap());
        assert_eq!(s.recover().unwrap(), ["a"]);
        let job = s.get("a").unwrap().unwrap();
        assert_eq!(job.state, JobState::Queued);
        assert_eq!(job.attempts, 1);
        let kinds: Vec<_> = s.events("a").unwrap().into_iter().map(|e| e.kind).collect();
        assert_eq!(kinds, ["submitted", "started", "recovered"]);
    }

    #[test]
    fn phase1_databases_migrate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.db");
        {
            // The Phase 1 shape of the two tables that gained columns.
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE jobs (id TEXT PRIMARY KEY, envelope TEXT NOT NULL, semantic_hash TEXT NOT NULL,
                     rank INTEGER NOT NULL, state TEXT NOT NULL, deadline_at INTEGER, submitted_at INTEGER NOT NULL,
                     updated_at INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, decision TEXT, failure TEXT);
                 CREATE TABLE results (job_id TEXT PRIMARY KEY, output TEXT NOT NULL, schema TEXT NOT NULL,
                     model TEXT NOT NULL, runtime TEXT NOT NULL, model_digest TEXT, exposure TEXT NOT NULL,
                     semantic_hash TEXT NOT NULL, sources TEXT NOT NULL, tokens_in INTEGER NOT NULL,
                     tokens_out INTEGER NOT NULL, started_at INTEGER NOT NULL, finished_at INTEGER NOT NULL);",
            )
            .unwrap();
        }
        let store = Store::open(&path).unwrap();
        store.insert_job(&env("a", "task: {objective: a}", 1)).unwrap();
        assert!(store.get("a").unwrap().unwrap().cognition.is_none());
        let version: String =
            store.conn().query_row("SELECT value FROM meta WHERE key = 'schema'", [], |r| r.get(0)).unwrap();
        assert_eq!(version, SCHEMA_VERSION.to_string());
        drop(store);
        Store::open(&path).unwrap(); // idempotent
    }
}
