//! The Cognitive Store engine (directive §15–17, §19).
//!
//! Freshness is never assumed: an object is checked against its policy —
//! and, for `source_hash`, against the current bytes of every source — each
//! time it is about to be used, and a background scan does the same so that
//! staleness shows up in listings before anyone asks. Invalidation cascades
//! to objects that used a stale object as context, and the jobs whose
//! results they are move from READY to STALE.

use crate::cogstore::{CognitiveObject, CorpusEntry, ParentRef};
use crate::config::CognitionConfig;
use crate::store::{SourceRef, Store, now_ms};
use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use tonyc_core::cognition::{
    Doc, FreshnessPolicy, ObjectState, Scope, Sensitivity, ValidationState, context_admissible, rank, source_overlap,
};
use tonyc_core::envelope::{Verification, WorkEnvelope};
use tonyc_core::{JobState, Millis};

/// Files modified this recently are always re-hashed: their mtime may not
/// yet distinguish a further write within the same timestamp tick.
const RACY_WINDOW_NS: i64 = 2_000_000_000;

pub struct Fingerprint {
    pub blake3: String,
    pub bytes: u64,
    pub mode: u32,
}

/// The current content hash of a regular file, or `None` if it is missing
/// or not a regular file. Uses a (size, mtime) cache.
pub fn fingerprint(store: &Store, path: &str) -> Result<Option<Fingerprint>> {
    let meta = match std::fs::metadata(path) {
        Ok(m) if m.is_file() => m,
        Ok(_) => return Ok(None),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let size = meta.len();
    let mtime_ns = meta.mtime() * 1_000_000_000 + meta.mtime_nsec();
    let mode = meta.permissions().mode() & 0o7777;
    let racy = now_ms() * 1_000_000 - mtime_ns < RACY_WINDOW_NS;
    if !racy && let Some(h) = store.cached_hash(path, size, mtime_ns)? {
        return Ok(Some(Fingerprint { blake3: h, bytes: size, mode }));
    }
    let mut hasher = blake3::Hasher::new();
    let mut file = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let h = hasher.finalize().to_hex().to_string();
    store.put_hash(path, size, mtime_ns, &h)?;
    Ok(Some(Fingerprint { blake3: h, bytes: size, mode }))
}

/// Checks an object against its freshness policy right now.
pub fn check(store: &Store, o: &CognitiveObject, now: Millis) -> Result<std::result::Result<(), String>> {
    if o.state == ObjectState::Stale {
        return Ok(Err(o.stale_reason.clone().unwrap_or_else(|| "stale".into())));
    }
    if let Some(at) = o.expires_at
        && now >= at
    {
        return Ok(Err(format!("ttl expired ({})", o.freshness.describe())));
    }
    if o.freshness.policy == FreshnessPolicy::SourceHash {
        for s in &o.sources {
            match fingerprint(store, &s.path)? {
                None => return Ok(Err(format!("source {} is missing", s.path))),
                Some(f) if f.blake3 != s.blake3 => return Ok(Err(format!("source {} changed", s.path))),
                Some(_) => {}
            }
        }
    }
    if o.freshness.policy == FreshnessPolicy::Immutable {
        return Ok(Ok(()));
    }
    for p in o.parents.iter().filter(|p| p.relation == "context") {
        match store.object(&p.id)? {
            None => return Ok(Err(format!("context {} no longer exists", p.id))),
            Some(parent) if parent.state == ObjectState::Stale => {
                return Ok(Err(format!("context {} is stale", p.id)));
            }
            Some(_) => {}
        }
    }
    Ok(Ok(()))
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Invalidation {
    pub objects: Vec<String>,
    pub jobs: Vec<String>,
}

/// Marks objects stale, cascades to objects that used them as context, and
/// moves the jobs whose results they are from READY to STALE.
pub fn invalidate(store: &Store, roots: Vec<(String, String)>) -> Result<Invalidation> {
    let mut queue: VecDeque<(String, String)> = roots.into();
    let mut report = Invalidation::default();
    let mut seen = BTreeSet::new();
    while let Some((id, reason)) = queue.pop_front() {
        if !seen.insert(id.clone()) || !store.mark_stale(&id, &reason)? {
            continue;
        }
        report.objects.push(id.clone());
        for child in store.object_children(&id)?.into_iter().filter(|c| c.relation == "context") {
            let immutable = store.object(&child.id)?.is_some_and(|c| c.freshness.policy == FreshnessPolicy::Immutable);
            if !immutable {
                queue.push_back((child.id, format!("context {id} went stale: {reason}")));
            }
        }
    }
    stale_jobs(store, &report.objects, "derived cognition went stale", &mut report.jobs)?;
    Ok(report)
}

fn stale_jobs(store: &Store, objects: &[String], why: &str, out: &mut Vec<String>) -> Result<()> {
    for job in store.jobs_using_objects(objects)? {
        if store.transition(
            &job,
            &[JobState::Ready],
            JobState::Stale,
            "stale",
            Some(serde_json::json!({"reason": why})),
        )? {
            out.push(job);
        }
    }
    Ok(())
}

/// Checks an object now; if it fails, invalidates it. Returns whether it is
/// still fresh.
pub fn verify(store: &Store, o: &CognitiveObject) -> Result<bool> {
    match check(store, o, now_ms())? {
        Ok(()) => Ok(true),
        Err(reason) => {
            invalidate(store, vec![(o.id.clone(), reason)])?;
            Ok(false)
        }
    }
}

/// An existing object that satisfies the request outright (gate G10).
pub struct DirectHit {
    pub object: CognitiveObject,
    pub notes: Vec<String>,
}

/// Looks for fresh cognition answering exactly this request. Objects are
/// verified live; ones that fail are invalidated on the spot.
pub fn find_direct(store: &Store, env: &WorkEnvelope) -> Result<(Option<DirectHit>, Vec<String>)> {
    let mut skipped = Vec::new();
    for o in store.objects_by_key(&env.semantic_hash)? {
        if o.state != ObjectState::Fresh {
            continue;
        }
        if !o.scope.visible_to(&env.reuse.scope) {
            skipped.push(format!("{} is in scope {}, not visible from {}", o.id, o.scope, env.reuse.scope));
            continue;
        }
        if o.confidence < env.quality.floor {
            skipped.push(format!("{} confidence {:.2} is below floor {:.2}", o.id, o.confidence, env.quality.floor));
            continue;
        }
        let validation_ok = match env.quality.verification {
            Verification::Required => o.validation == ValidationState::Verified,
            _ => o.validation != ValidationState::Failed,
        };
        let problems = tonyc_core::refine::check_all(&env.quality.checks, &o.content);
        if !problems.is_empty() {
            skipped.push(format!("{} fails this request's checks: {}", o.id, problems.join("; ")));
            continue;
        }
        if !validation_ok {
            skipped
                .push(format!("{} is {}, request needs verification {}", o.id, o.validation, env.quality.verification));
            continue;
        }
        match check(store, &o, now_ms())? {
            Ok(()) => {
                let mut notes = vec![
                    format!("same semantic key {}", &env.semantic_hash[..16]),
                    format!("freshness {} verified now", o.freshness.describe()),
                    format!("confidence {:.2} ≥ floor {:.2}", o.confidence, env.quality.floor),
                ];
                if !o.sources.is_empty() {
                    notes.push(format!("{} source(s) re-hashed and unchanged", o.sources.len()));
                }
                return Ok((Some(DirectHit { object: o, notes }), skipped));
            }
            Err(reason) => {
                skipped.push(format!("{} went stale: {reason}", o.id));
                invalidate(store, vec![(o.id.clone(), reason)])?;
            }
        }
    }
    Ok((None, skipped))
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextPick {
    #[serde(skip)]
    pub object: CognitiveObject,
    pub id: String,
    pub relation: String,
    pub score: f64,
    /// For `refresh_of`: the sources that changed since the prior answer.
    pub changed_sources: Vec<String>,
}

/// Prior cognition to give the model: the newest stale answer to the same
/// request (incremental refresh), and the most related fresh objects.
pub fn select_context(
    store: &Store,
    env: &WorkEnvelope,
    job_sources: &[String],
    cfg: &CognitionConfig,
) -> Result<Vec<ContextPick>> {
    let mut picks = Vec::new();
    // Context never widens where data goes and never narrows the job: an
    // object is offered only if its own locality allows this job's, and
    // sensitive objects only to jobs that are already LOCAL_ONLY.
    let admissible = |o: &CognitiveObject| {
        o.scope.visible_to(&env.reuse.scope)
            && context_admissible(&o.privacy, &env.privacy)
            && (env.privacy.class == tonyc_core::Locality::LocalOnly || !o.sensitivity.forces_local())
    };

    if let Some(prev) =
        store.objects_by_key(&env.semantic_hash)?.into_iter().find(|o| o.state == ObjectState::Stale && admissible(o))
    {
        let mut changed = Vec::new();
        for s in &prev.sources {
            if fingerprint(store, &s.path)?.is_none_or(|f| f.blake3 != s.blake3) {
                changed.push(s.path.clone());
            }
        }
        picks.push(ContextPick {
            id: prev.id.clone(),
            object: prev,
            relation: "refresh_of".into(),
            score: 1.0,
            changed_sources: changed,
        });
    }

    let fresh: Vec<CognitiveObject> =
        store.objects_in_state(ObjectState::Fresh)?.into_iter().filter(|o| admissible(o)).collect();
    let texts: Vec<String> = fresh.iter().map(|o| format!("{} {}", o.objective, clip(&o.content, 2000))).collect();
    let docs: Vec<Doc> = fresh.iter().zip(&texts).map(|(o, t)| Doc { id: &o.id, text: t }).collect();
    let mut query = env.task.objective.clone();
    if let Some(t) = &env.task.input_text {
        query.push(' ');
        query.push_str(clip(t, 500));
    }
    let lexical: BTreeMap<String, f64> = rank(&query, &docs).into_iter().collect();
    let mut scored: Vec<(f64, CognitiveObject)> = fresh
        .into_iter()
        .map(|o| {
            let paths: Vec<String> = o.sources.iter().map(|s| s.path.clone()).collect();
            let score = 0.6 * lexical.get(&o.id).copied().unwrap_or(0.0) + 0.4 * source_overlap(&paths, job_sources);
            (score, o)
        })
        .filter(|(score, _)| *score >= cfg.related_threshold)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
    for (score, o) in scored {
        if picks.len() >= cfg.context_objects {
            break;
        }
        if verify(store, &o)? {
            picks.push(ContextPick {
                id: o.id.clone(),
                object: o,
                relation: "context".into(),
                score,
                changed_sources: vec![],
            });
        }
    }
    Ok(picks)
}

/// The prompt section presenting prior cognition, within a character budget.
pub fn context_block(picks: &[ContextPick], budget: usize) -> String {
    if picks.is_empty() {
        return String::new();
    }
    let per = (budget / picks.len()).max(200);
    let mut out =
        String::from("\nExisting cognition (from earlier work; prefer the inputs below where they disagree):\n");
    for p in picks {
        let o = &p.object;
        if p.relation == "refresh_of" {
            let changed =
                if p.changed_sources.is_empty() { "unknown".to_string() } else { p.changed_sources.join(", ") };
            out.push_str(&format!(
                "\n[{} · previous answer to this same request · {} · now stale; changed since: {changed}]\n{}\n",
                o.id,
                o.model,
                clip(&o.content, per)
            ));
        } else {
            out.push_str(&format!(
                "\n[{} · {} · {} · relevance {:.2}]\n{}\n",
                o.id,
                o.kind,
                o.model,
                p.score,
                clip(&o.content, per)
            ));
        }
    }
    out
}

fn clip(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The most restrictive corpus sensitivity among the given source paths and
/// context objects, with the items that set it.
pub fn sensitivity_of(
    store: &Store,
    sources: &[SourceRef],
    context: &[ContextPick],
) -> Result<(Sensitivity, Vec<String>)> {
    let mut level = Sensitivity::Normal;
    let mut why = Vec::new();
    for s in sources {
        if let Some((_, sens)) = store.corpus_entry(&s.path)? {
            if sens.forces_local() {
                why.push(format!("{} is {sens}", s.path));
            }
            level = level.max(sens);
        }
    }
    for p in context {
        if p.object.sensitivity.forces_local() {
            why.push(format!("context {} is {}", p.id, p.object.sensitivity));
        }
        level = level.max(p.object.sensitivity);
    }
    Ok((level, why))
}

pub fn parents_of(picks: &[ContextPick]) -> Vec<ParentRef> {
    picks.iter().map(|p| ParentRef { id: p.id.clone(), relation: p.relation.clone() }).collect()
}

#[derive(Debug, Default, Serialize)]
pub struct ScanReport {
    pub corpus_files: usize,
    pub corpus_changed: Vec<String>,
    pub objects_checked: usize,
    pub invalidated: Invalidation,
}

/// Re-hashes the corpus and re-checks every fresh object.
pub fn scan(store: &Store) -> Result<ScanReport> {
    let mut report = ScanReport::default();
    let now = now_ms();
    for mut e in store.corpus(None)? {
        report.corpus_files += 1;
        let fp = fingerprint(store, &e.path)?;
        let new_hash = fp.as_ref().map(|f| f.blake3.clone());
        if new_hash != e.blake3 {
            report.corpus_changed.push(e.path.clone());
        }
        e.present = fp.is_some();
        e.bytes = fp.as_ref().map(|f| f.bytes);
        e.mode = fp.as_ref().map(|f| f.mode);
        e.blake3 = new_hash;
        e.scanned_at = now;
        store.corpus_upsert(&e)?;
    }
    let mut failures = Vec::new();
    for o in store.objects_in_state(ObjectState::Fresh)? {
        report.objects_checked += 1;
        if let Err(reason) = check(store, &o, now)? {
            failures.push((o.id.clone(), reason));
        }
    }
    report.invalidated = invalidate(store, failures)?;
    Ok(report)
}

const SKIP_DIRS: &[&str] = &["target", "node_modules", "__pycache__"];

/// Adds files (directories recursively, skipping hidden entries and build
/// output) to the corpus.
pub fn corpus_add(
    store: &Store,
    paths: &[String],
    scope: &Scope,
    sensitivity: Sensitivity,
    max_files: usize,
) -> Result<Vec<CorpusEntry>> {
    let mut files = Vec::new();
    for p in paths {
        if !p.starts_with('/') {
            bail!("corpus paths must be absolute: {p}");
        }
        let path = Path::new(p);
        if path.is_dir() {
            walk(path, &mut files, max_files)?;
        } else if path.is_file() {
            files.push(p.clone());
        } else {
            bail!("{p} is not a file or directory");
        }
        if files.len() > max_files {
            bail!("more than {max_files} files; add a narrower path or raise cognition.max_corpus_files");
        }
    }
    let now = now_ms();
    let mut out = Vec::new();
    for f in files {
        let fp = fingerprint(store, &f)?;
        let entry = CorpusEntry {
            path: f,
            scope: scope.clone(),
            sensitivity,
            blake3: fp.as_ref().map(|x| x.blake3.clone()),
            bytes: fp.as_ref().map(|x| x.bytes),
            mode: fp.as_ref().map(|x| x.mode),
            added_at: now,
            scanned_at: now,
            present: fp.is_some(),
        };
        store.corpus_upsert(&entry)?;
        out.push(entry);
    }
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<String>, max: usize) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        if out.len() > max {
            return Ok(());
        }
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let ft = e.file_type()?;
        if ft.is_dir() && !SKIP_DIRS.contains(&name.as_str()) {
            walk(&e.path(), out, max)?;
        } else if ft.is_file() {
            out.push(e.path().to_string_lossy().into_owned());
        }
    }
    Ok(())
}

#[derive(Debug, Default, Serialize)]
pub struct Forgotten {
    pub paths: Vec<String>,
    pub purged_objects: Vec<String>,
    pub invalidated: Invalidation,
    pub redacted_results: usize,
}

/// Stops tracking a scope or path, and forgets what was derived from it:
/// derived objects are deleted, results built on them are redacted, and
/// objects that used them as context go stale.
pub fn corpus_forget(store: &Store, target: &str) -> Result<Forgotten> {
    let entries = store.corpus(None)?;
    let paths: Vec<String> = if target.starts_with('/') {
        let dir = format!("{}/", target.trim_end_matches('/'));
        entries.into_iter().filter(|e| e.path == target || e.path.starts_with(&dir)).map(|e| e.path).collect()
    } else {
        let scope: Scope = target.parse().map_err(|e: String| anyhow::anyhow!(e))?;
        entries.into_iter().filter(|e| e.scope == scope).map(|e| e.path).collect()
    };
    if paths.is_empty() {
        bail!("nothing in the corpus matches `{target}`");
    }
    let mut purge = BTreeSet::new();
    for p in &paths {
        purge.extend(store.objects_with_source_under(p)?);
    }
    let purge: Vec<String> = purge.into_iter().collect();
    let mut dependents = Vec::new();
    for id in &purge {
        for c in store.object_children(id)? {
            if !purge.contains(&c.id) {
                dependents.push((c.id, format!("context {id} was derived from forgotten corpus `{target}`")));
            }
        }
    }
    let mut out = Forgotten { paths: paths.clone(), ..Default::default() };
    out.invalidated = invalidate(store, dependents)?;
    stale_jobs(store, &purge, "derived from forgotten corpus", &mut out.invalidated.jobs)?;
    out.redacted_results = store.redact_results(&purge, &format!("[forgotten: derived from corpus `{target}`]"))?;
    store.delete_objects(&purge, &format!("corpus forget `{target}`"))?;
    store.corpus_remove(&paths)?;
    out.purged_objects = purge;
    Ok(out)
}
