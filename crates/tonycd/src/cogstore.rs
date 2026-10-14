//! Cognitive Store persistence: objects, their sources and parents, the
//! corpus, the file-hash cache and the demand log.

use crate::store::{SourceRef, Store, json_of, now_ms};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::Value;
use tonyc_core::cognition::{CognitiveKind, Freshness, ObjectState, Scope, Sensitivity, ValidationState};
use tonyc_core::{Exposure, LocalityPolicy, Millis};

#[derive(Debug, Clone, Serialize)]
pub struct ParentRef {
    pub id: String,
    /// `context` (shown to the model as prior cognition) or `refresh_of`
    /// (an earlier, stale answer to the same request).
    pub relation: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CognitiveObject {
    pub id: String,
    pub kind: CognitiveKind,
    pub semantic_key: String,
    pub scope: Scope,
    pub objective: String,
    pub content: String,
    pub schema: String,
    /// Expected quality of the producing model for this capability.
    pub confidence: f64,
    pub validation: ValidationState,
    pub freshness: Freshness,
    pub expires_at: Option<Millis>,
    pub state: ObjectState,
    pub stale_reason: Option<String>,
    pub stale_at: Option<Millis>,
    pub created_at: Millis,
    pub job_id: String,
    pub model: String,
    pub model_digest: Option<String>,
    pub runtime: String,
    pub exposure: Exposure,
    /// The locality the producing job ran under; bounds where the content
    /// may be sent as context later.
    pub privacy: LocalityPolicy,
    pub sensitivity: Sensitivity,
    pub sources: Vec<SourceRef>,
    pub parents: Vec<ParentRef>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorpusEntry {
    pub path: String,
    pub scope: Scope,
    pub sensitivity: Sensitivity,
    pub blake3: Option<String>,
    pub bytes: Option<u64>,
    pub mode: Option<u32>,
    pub added_at: Millis,
    pub scanned_at: Millis,
    pub present: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CognitionEvent {
    pub seq: i64,
    pub at: Millis,
    pub kind: String,
    pub detail: Option<Value>,
}

fn parse<T: std::str::FromStr<Err = String>>(text: String) -> rusqlite::Result<T> {
    text.parse()
        .map_err(|e: String| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, e.into()))
}

const OBJECT_COLS: &str =
    "id, kind, semantic_key, scope, objective, content, schema, confidence, validation, freshness,
    expires_at, state, stale_reason, stale_at, created_at, job_id, model, model_digest, runtime, exposure, privacy,
    sensitivity";

fn object_from_row(r: &rusqlite::Row) -> rusqlite::Result<CognitiveObject> {
    Ok(CognitiveObject {
        id: r.get(0)?,
        kind: parse(r.get(1)?)?,
        semantic_key: r.get(2)?,
        scope: parse(r.get(3)?)?,
        objective: r.get(4)?,
        content: r.get(5)?,
        schema: r.get(6)?,
        confidence: r.get(7)?,
        validation: parse(r.get(8)?)?,
        freshness: json_of(r.get(9)?)?,
        expires_at: r.get(10)?,
        state: parse(r.get(11)?)?,
        stale_reason: r.get(12)?,
        stale_at: r.get(13)?,
        created_at: r.get(14)?,
        job_id: r.get(15)?,
        model: r.get(16)?,
        model_digest: r.get(17)?,
        runtime: r.get(18)?,
        exposure: json_of(r.get(19)?)?,
        privacy: json_of(r.get(20)?)?,
        sensitivity: parse(r.get(21)?)?,
        sources: vec![],
        parents: vec![],
    })
}

fn fill(conn: &Connection, mut o: CognitiveObject) -> rusqlite::Result<CognitiveObject> {
    let mut stmt =
        conn.prepare_cached("SELECT path, blake3, bytes FROM object_sources WHERE object_id = ?1 ORDER BY path")?;
    o.sources = stmt
        .query_map([&o.id], |r| {
            Ok(SourceRef { path: r.get(0)?, blake3: r.get(1)?, bytes: r.get::<_, i64>(2)? as u64 })
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut stmt =
        conn.prepare_cached("SELECT parent_id, relation FROM object_parents WHERE object_id = ?1 ORDER BY parent_id")?;
    o.parents = stmt
        .query_map([&o.id], |r| Ok(ParentRef { id: r.get(0)?, relation: r.get(1)? }))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(o)
}

fn cognition_event(conn: &Connection, object: &str, kind: &str, detail: &Value) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO cognition_events (object_id, at, kind, detail) VALUES (?1, ?2, ?3, ?4)",
        params![object, now_ms(), kind, detail.to_string()],
    )?;
    Ok(())
}

impl Store {
    pub fn insert_object(&self, o: &CognitiveObject) -> Result<()> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        tx.execute(
            &format!(
                "INSERT INTO objects ({OBJECT_COLS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                      ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22)"
            ),
            params![
                o.id,
                o.kind.as_str(),
                o.semantic_key,
                o.scope.to_string(),
                o.objective,
                o.content,
                o.schema,
                o.confidence,
                o.validation.as_str(),
                serde_json::to_string(&o.freshness)?,
                o.expires_at,
                o.state.as_str(),
                o.stale_reason,
                o.stale_at,
                o.created_at,
                o.job_id,
                o.model,
                o.model_digest,
                o.runtime,
                serde_json::to_string(&o.exposure)?,
                serde_json::to_string(&o.privacy)?,
                o.sensitivity.as_str(),
            ],
        )?;
        for s in &o.sources {
            tx.execute(
                "INSERT INTO object_sources (object_id, path, blake3, bytes) VALUES (?1, ?2, ?3, ?4)",
                params![o.id, s.path, s.blake3, s.bytes as i64],
            )?;
        }
        for p in &o.parents {
            tx.execute(
                "INSERT INTO object_parents (object_id, parent_id, relation) VALUES (?1, ?2, ?3)",
                params![o.id, p.id, p.relation],
            )?;
        }
        cognition_event(
            &tx,
            &o.id,
            "created",
            &serde_json::json!({"job": o.job_id, "model": o.model, "parents": o.parents, "sources": o.sources}),
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn note_object(&self, id: &str, kind: &str, detail: Value) -> Result<()> {
        cognition_event(&self.conn(), id, kind, &detail)?;
        Ok(())
    }

    pub fn object(&self, id: &str) -> Result<Option<CognitiveObject>> {
        let conn = self.conn();
        let o = conn
            .query_row(&format!("SELECT {OBJECT_COLS} FROM objects WHERE id = ?1"), [id], object_from_row)
            .optional()?;
        Ok(o.map(|o| fill(&conn, o)).transpose()?)
    }

    fn objects_where(&self, clause: &str, args: &[&dyn rusqlite::ToSql]) -> Result<Vec<CognitiveObject>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!("SELECT {OBJECT_COLS} FROM objects WHERE {clause}"))?;
        let rows = stmt.query_map(args, object_from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows.into_iter().map(|o| fill(&conn, o)).collect::<rusqlite::Result<_>>()?)
    }

    /// Objects answering exactly this request, newest first.
    pub fn objects_by_key(&self, key: &str) -> Result<Vec<CognitiveObject>> {
        self.objects_where("semantic_key = ?1 ORDER BY created_at DESC, id DESC", &[&key])
    }

    pub fn objects_in_state(&self, state: ObjectState) -> Result<Vec<CognitiveObject>> {
        self.objects_where("state = ?1 ORDER BY created_at DESC, id DESC", &[&state.as_str()])
    }

    pub fn all_objects(&self) -> Result<Vec<CognitiveObject>> {
        self.objects_where("1 = 1 ORDER BY created_at DESC, id DESC", &[])
    }

    pub fn objects_with_source_under(&self, path: &str) -> Result<Vec<String>> {
        let conn = self.conn();
        let prefix = format!("{}/%", path.trim_end_matches('/'));
        let mut stmt =
            conn.prepare("SELECT DISTINCT object_id FROM object_sources WHERE path = ?1 OR path LIKE ?2 ORDER BY 1")?;
        Ok(stmt.query_map(params![path, prefix], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    }

    pub fn object_ids_in_scope(&self, scope: &Scope) -> Result<Vec<String>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT id FROM objects WHERE scope = ?1 ORDER BY id")?;
        Ok(stmt.query_map([scope.to_string()], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    }

    pub fn object_children(&self, id: &str) -> Result<Vec<ParentRef>> {
        let conn = self.conn();
        let mut stmt =
            conn.prepare("SELECT object_id, relation FROM object_parents WHERE parent_id = ?1 ORDER BY 1")?;
        Ok(stmt
            .query_map([id], |r| Ok(ParentRef { id: r.get(0)?, relation: r.get(1)? }))?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn object_events(&self, id: &str) -> Result<Vec<CognitionEvent>> {
        let conn = self.conn();
        let mut stmt =
            conn.prepare("SELECT seq, at, kind, detail FROM cognition_events WHERE object_id = ?1 ORDER BY seq")?;
        Ok(stmt
            .query_map([id], |r| {
                Ok(CognitionEvent {
                    seq: r.get(0)?,
                    at: r.get(1)?,
                    kind: r.get(2)?,
                    detail: r.get::<_, Option<String>>(3)?.map(json_of).transpose()?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Marks a fresh object stale. Returns whether it changed.
    pub fn mark_stale(&self, id: &str, reason: &str) -> Result<bool> {
        let conn = self.conn();
        let n = conn.execute(
            "UPDATE objects SET state = 'STALE', stale_reason = ?2, stale_at = ?3 WHERE id = ?1 AND state = 'FRESH'",
            params![id, reason, now_ms()],
        )?;
        if n > 0 {
            cognition_event(&conn, id, "invalidated", &serde_json::json!({"reason": reason}))?;
        }
        Ok(n > 0)
    }

    /// Deletes objects (their lineage events remain). Returns the count.
    pub fn delete_objects(&self, ids: &[String], why: &str) -> Result<usize> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        let mut n = 0;
        for id in ids {
            tx.execute("DELETE FROM object_sources WHERE object_id = ?1", [id])?;
            tx.execute("DELETE FROM object_parents WHERE object_id = ?1", [id])?;
            n += tx.execute("DELETE FROM objects WHERE id = ?1", [id])?;
            cognition_event(&tx, id, "deleted", &serde_json::json!({"reason": why}))?;
        }
        tx.commit()?;
        Ok(n)
    }

    /// Jobs whose result is, or was satisfied by, one of these objects.
    pub fn jobs_using_objects(&self, ids: &[String]) -> Result<Vec<String>> {
        let conn = self.conn();
        let mut out = Vec::new();
        let mut stmt = conn.prepare("SELECT job_id FROM results WHERE object_id = ?1 OR reused_from = ?1")?;
        for id in ids {
            for job in stmt.query_map([id], |r| r.get::<_, String>(0))? {
                out.push(job?);
            }
        }
        out.sort();
        out.dedup();
        Ok(out)
    }

    /// Replaces the stored output of results derived from these objects.
    pub fn redact_results(&self, ids: &[String], note: &str) -> Result<usize> {
        let conn = self.conn();
        let mut n = 0;
        for id in ids {
            n += conn.execute(
                "UPDATE results SET output = ?2 WHERE object_id = ?1 OR reused_from = ?1",
                params![id, note],
            )?;
        }
        Ok(n)
    }

    pub fn object_counts(&self) -> Result<(i64, i64)> {
        let conn = self.conn();
        let count = |state: &str| -> Result<i64> {
            Ok(conn.query_row("SELECT COUNT(*) FROM objects WHERE state = ?1", [state], |r| r.get(0))?)
        };
        Ok((count("FRESH")?, count("STALE")?))
    }

    // --------------------------------------------------------- corpus ---

    pub fn corpus_upsert(&self, e: &CorpusEntry) -> Result<()> {
        self.conn().execute(
            "INSERT INTO corpus (path, scope, sensitivity, blake3, bytes, mode, added_at, scanned_at, present)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(path) DO UPDATE SET scope = excluded.scope, sensitivity = excluded.sensitivity,
                 blake3 = excluded.blake3, bytes = excluded.bytes, mode = excluded.mode,
                 scanned_at = excluded.scanned_at, present = excluded.present",
            params![
                e.path,
                e.scope.to_string(),
                e.sensitivity.as_str(),
                e.blake3,
                e.bytes.map(|b| b as i64),
                e.mode,
                e.added_at,
                e.scanned_at,
                e.present
            ],
        )?;
        Ok(())
    }

    pub fn corpus(&self, scope: Option<&Scope>) -> Result<Vec<CorpusEntry>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT path, scope, sensitivity, blake3, bytes, mode, added_at, scanned_at, present FROM corpus
             WHERE ?1 IS NULL OR scope = ?1 ORDER BY path",
        )?;
        Ok(stmt
            .query_map([scope.map(|s| s.to_string())], |r| {
                Ok(CorpusEntry {
                    path: r.get(0)?,
                    scope: parse(r.get(1)?)?,
                    sensitivity: parse(r.get(2)?)?,
                    blake3: r.get(3)?,
                    bytes: r.get::<_, Option<i64>>(4)?.map(|b| b as u64),
                    mode: r.get(5)?,
                    added_at: r.get(6)?,
                    scanned_at: r.get(7)?,
                    present: r.get(8)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn corpus_entry(&self, path: &str) -> Result<Option<(Scope, Sensitivity)>> {
        let conn = self.conn();
        Ok(conn
            .query_row("SELECT scope, sensitivity FROM corpus WHERE path = ?1", [path], |r| {
                Ok((parse(r.get(0)?)?, parse(r.get(1)?)?))
            })
            .optional()?)
    }

    pub fn corpus_remove(&self, paths: &[String]) -> Result<usize> {
        let conn = self.conn();
        let mut n = 0;
        for p in paths {
            n += conn.execute("DELETE FROM corpus WHERE path = ?1", [p])?;
        }
        Ok(n)
    }

    // ------------------------------------------------------ file hashes ---

    pub fn cached_hash(&self, path: &str, size: u64, mtime_ns: i64) -> Result<Option<String>> {
        let conn = self.conn();
        Ok(conn
            .query_row(
                "SELECT blake3 FROM file_hashes WHERE path = ?1 AND size = ?2 AND mtime_ns = ?3",
                params![path, size as i64, mtime_ns],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn put_hash(&self, path: &str, size: u64, mtime_ns: i64, hash: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO file_hashes (path, size, mtime_ns, blake3, checked_at) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime_ns = excluded.mtime_ns,
                 blake3 = excluded.blake3, checked_at = excluded.checked_at",
            params![path, size as i64, mtime_ns, hash, now_ms()],
        )?;
        Ok(())
    }

    // ----------------------------------------------------------- demand ---

    /// Appends a demand record: evidence of what agents asked the store for,
    /// consumed later by economics (Phase 4) and prediction (Phase 6).
    pub fn record_demand(
        &self,
        kind: &str,
        key: Option<&str>,
        scope: Option<&Scope>,
        objective: Option<&str>,
        object: Option<&str>,
        job: Option<&str>,
    ) -> Result<()> {
        self.conn().execute(
            "INSERT INTO demand (at, kind, semantic_key, scope, objective, object_id, job_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![now_ms(), kind, key, scope.map(|s| s.to_string()), objective, object, job],
        )?;
        Ok(())
    }

    pub fn demand_counts(&self) -> Result<Vec<(String, i64)>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT kind, COUNT(*) FROM demand GROUP BY kind ORDER BY kind")?;
        Ok(stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?)
    }
}
