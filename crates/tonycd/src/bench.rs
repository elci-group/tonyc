//! Model benchmarks (directive §12: "observed performance SHOULD
//! progressively replace vendor claims and static assumptions").
//!
//! A benchmark run is a set of ordinary jobs pinned to one model, at a low
//! urgency by default, with reuse off. They go through the normal queue, so
//! locality, pressure, slots and the boundary audit all apply. Each case has
//! a deterministic check; results become quality observations, blended with
//! the model's prior (see `QualityEvidence`). The suite is small and its
//! scores are coarse; sample sizes are always reported alongside them.

use crate::daemon::Daemon;
use crate::store::{Store, now_ms};
use anyhow::{Result, bail};
use rusqlite::params;
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tonyc_core::JobState;
use tonyc_core::envelope::{Urgency, WorkRequest, compile};

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS bench_runs (
    id          TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    capability  TEXT,
    class       TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    finished_at INTEGER,
    state       TEXT NOT NULL,
    cases       INTEGER NOT NULL,
    scored      INTEGER NOT NULL DEFAULT 0,
    passed      INTEGER NOT NULL DEFAULT 0,
    errors      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS bench_jobs (
    job_id     TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    case_id    TEXT NOT NULL,
    capability TEXT NOT NULL,
    scored     INTEGER NOT NULL DEFAULT 0,
    passed     INTEGER,
    note       TEXT
);
CREATE INDEX IF NOT EXISTS bench_jobs_pending ON bench_jobs (scored);
"#;

/// A deterministic check on a model's answer. Text matching is
/// case-insensitive.
#[derive(Debug, Clone, Default)]
pub struct Check {
    pub contains_all: &'static [&'static str],
    pub contains_none: &'static [&'static str],
    pub max_chars: Option<usize>,
    /// For JSON answers: key → expected (strings match by containment).
    pub json: &'static [(&'static str, Expect)],
}

#[derive(Debug, Clone, Copy)]
pub enum Expect {
    Text(&'static str),
    Number(f64),
}

pub struct Case {
    pub id: &'static str,
    pub capability: &'static str,
    pub prompt: &'static str,
    pub max_tokens: u32,
    pub json: bool,
    pub check: Check,
    /// A correct answer and a wrong one, for judge cases: a validator should
    /// pass the first and fail the second.
    pub good: &'static str,
    pub bad: &'static str,
}

/// The built-in suite: short prompts small models can answer, each with a
/// check that does not depend on phrasing.
pub fn suite() -> Vec<Case> {
    vec![
        Case {
            id: "general.arith",
            capability: "general",
            prompt: "What is 17 + 25? Answer with just the number.",
            max_tokens: 16,
            json: false,
            check: Check { contains_all: &["42"], ..Default::default() },
            good: "42",
            bad: "43",
        },
        Case {
            id: "general.capital",
            capability: "general",
            prompt: "What is the capital of France? Answer in one word.",
            max_tokens: 16,
            json: false,
            check: Check { contains_all: &["paris"], ..Default::default() },
            good: "Paris",
            bad: "Lyon",
        },
        Case {
            id: "general.leap",
            capability: "general",
            prompt: "How many days are there in a leap year? Answer with just the number.",
            max_tokens: 16,
            json: false,
            check: Check { contains_all: &["366"], ..Default::default() },
            good: "366",
            bad: "365",
        },
        Case {
            id: "general.obey",
            capability: "general",
            prompt: "Reply with exactly the word OK and nothing else.",
            max_tokens: 8,
            json: false,
            check: Check { contains_all: &["ok"], max_chars: Some(12), ..Default::default() },
            good: "OK",
            bad: "Sure! OK, happy to help.",
        },
        Case {
            id: "summarisation.build",
            capability: "summarisation",
            prompt: "Summarise in one short sentence: The Rust build failed because the crate `serde` was not listed in Cargo.toml. Adding serde = \"1\" to the dependencies fixed the build.",
            max_tokens: 60,
            json: false,
            check: Check { contains_all: &["serde"], max_chars: Some(300), ..Default::default() },
            good: "The build failed because serde was missing from Cargo.toml; adding it fixed the build.",
            bad: "The build failed because of a network outage and was fixed by retrying.",
        },
        Case {
            id: "summarisation.meeting",
            capability: "summarisation",
            prompt: "Summarise in one short sentence: The weekly planning meeting has moved from Tuesday morning to Thursday at 3pm because two team members are travelling on Tuesday.",
            max_tokens: 60,
            json: false,
            check: Check { contains_all: &["thursday"], max_chars: Some(300), ..Default::default() },
            good: "The planning meeting moved from Tuesday to Thursday at 3pm because two members are travelling.",
            bad: "The planning meeting was cancelled.",
        },
        Case {
            id: "classification.sentiment",
            capability: "classification",
            prompt: "Classify the sentiment of this review as positive or negative. Answer with one word.\nReview: The battery died after two days and support never replied.",
            max_tokens: 8,
            json: false,
            check: Check { contains_all: &["negative"], contains_none: &["positive"], ..Default::default() },
            good: "negative",
            bad: "positive",
        },
        Case {
            id: "classification.spam",
            capability: "classification",
            prompt: "Is this message spam or not spam? Answer with exactly `spam` or `not spam`.\nMessage: Congratulations! You won a free cruise, click this link now to claim your prize!",
            max_tokens: 8,
            json: false,
            check: Check { contains_all: &["spam"], contains_none: &["not spam"], ..Default::default() },
            good: "spam",
            bad: "not spam",
        },
        Case {
            id: "extraction.person",
            capability: "extraction",
            prompt: "Extract the person's name and age as JSON with the keys \"name\" and \"age\".\nText: Maria is 34 years old and lives in Lisbon.",
            max_tokens: 40,
            json: true,
            check: Check {
                json: &[("name", Expect::Text("maria")), ("age", Expect::Number(34.0))],
                ..Default::default()
            },
            good: "{\"name\": \"Maria\", \"age\": 34}",
            bad: "{\"name\": \"Maria\", \"age\": 43}",
        },
        Case {
            id: "rust.mutable",
            capability: "rust",
            prompt: "In Rust, how do you declare a variable named x that can be changed later? Answer with just the declaration keywords and name, for example `let y`.",
            max_tokens: 16,
            json: false,
            check: Check { contains_all: &["let mut x"], ..Default::default() },
            good: "let mut x",
            bad: "const x",
        },
        Case {
            id: "rust.iter",
            capability: "rust",
            prompt: "What does this Rust expression evaluate to? Answer with just the number.\n[1, 2, 3].iter().map(|x| x * 2).sum::<i32>()",
            max_tokens: 16,
            json: false,
            check: Check { contains_all: &["12"], ..Default::default() },
            good: "12",
            bad: "6",
        },
    ]
}

/// One benchmark job: answering a case, or judging a known answer to it.
pub enum Item<'a> {
    Answer(&'a Case),
    Judge(&'a Case, bool),
}

impl Item<'_> {
    pub fn id(&self) -> String {
        match self {
            Item::Answer(c) => c.id.to_string(),
            Item::Judge(c, good) => format!("judge.{}.{}", c.id, if *good { "good" } else { "bad" }),
        }
    }

    pub fn capability(&self) -> &'static str {
        match self {
            Item::Answer(c) => c.capability,
            Item::Judge(_, true) => "validation.sensitivity",
            Item::Judge(_, false) => "validation.specificity",
        }
    }

    fn request(&self, model: &str, class: Urgency) -> Value {
        let (objective, capability, max_tokens, json) = match self {
            Item::Answer(c) => (c.prompt.to_string(), c.capability, c.max_tokens, c.json),
            Item::Judge(c, good) => (
                crate::refinement::judge_prompt(c.prompt, "", if *good { c.good } else { c.bad }),
                "validation",
                160,
                true,
            ),
        };
        json!({
            "task": {"objective": objective, "capability": capability, "max_tokens": max_tokens, "pin_model": model},
            "latency": {"class": class.as_str()},
            "quality": {"floor": 0.0, "refine": "never"},
            "reuse": {"cache": false},
            "output": {"schema": if json { "json/v1" } else { "text/v1" }},
        })
    }

    pub fn score(&self, output: &str) -> (bool, Option<String>) {
        match self {
            Item::Answer(c) => score(c, output),
            Item::Judge(_, good) => {
                use tonyc_core::refine::{Verdict, parse_judgement};
                let want = if *good { Verdict::Pass } else { Verdict::Fail };
                let got = parse_judgement(output).verdict;
                if got == want { (true, None) } else { (false, Some(format!("judged {got:?}, should be {want:?}"))) }
            }
        }
    }
}

/// Finds a benchmark item by id.
pub fn item<'a>(cases: &'a [Case], id: &str) -> Option<Item<'a>> {
    if let Some(rest) = id.strip_prefix("judge.") {
        let (case, which) = rest.rsplit_once('.')?;
        let c = cases.iter().find(|c| c.id == case)?;
        return Some(Item::Judge(c, which == "good"));
    }
    cases.iter().find(|c| c.id == id).map(Item::Answer)
}

/// Scores an answer. Returns whether it passed and a note if it did not.
pub fn score(case: &Case, output: &str) -> (bool, Option<String>) {
    let text = output.trim().to_lowercase();
    for want in case.check.contains_all {
        if !text.contains(&want.to_lowercase()) {
            return (false, Some(format!("missing `{want}`")));
        }
    }
    for bad in case.check.contains_none {
        if text.contains(&bad.to_lowercase()) {
            return (false, Some(format!("contains `{bad}`")));
        }
    }
    if let Some(max) = case.check.max_chars
        && output.trim().chars().count() > max
    {
        return (false, Some(format!("longer than {max} characters")));
    }
    if !case.check.json.is_empty() {
        let start = output.find('{');
        let end = output.rfind('}');
        let Some(v) =
            start.zip(end).filter(|(s, e)| s < e).and_then(|(s, e)| serde_json::from_str::<Value>(&output[s..=e]).ok())
        else {
            return (false, Some("no JSON object".into()));
        };
        for (key, expect) in case.check.json {
            let ok = match (expect, &v[*key]) {
                (Expect::Text(t), Value::String(s)) => s.to_lowercase().contains(t),
                (Expect::Number(n), Value::Number(x)) => x.as_f64().is_some_and(|x| (x - n).abs() < 1e-9),
                (Expect::Number(n), Value::String(s)) => s.trim().parse::<f64>().is_ok_and(|x| (x - n).abs() < 1e-9),
                _ => false,
            };
            if !ok {
                return (false, Some(format!("JSON field `{key}` is wrong or missing")));
            }
        }
    }
    (true, None)
}

#[derive(Debug, Clone, Serialize)]
pub struct Run {
    pub id: String,
    pub model: String,
    pub capability: Option<String>,
    pub class: String,
    pub created_at: i64,
    pub finished_at: Option<i64>,
    pub state: String,
    pub cases: i64,
    pub scored: i64,
    pub passed: i64,
    pub errors: i64,
}

/// Starts a benchmark: one pinned, cache-free job per case.
pub fn start(
    d: &Arc<Daemon>,
    model: &str,
    capability: Option<&str>,
    class: Urgency,
    limit: Option<usize>,
) -> Result<Run> {
    if !d.registry.candidates().iter().any(|c| c.profile.id == model) {
        bail!("no model {model}; see `tonyc models`");
    }
    let all = suite();
    // `validation` measures judging only; no capability measures both.
    let mut items: Vec<Item> = Vec::new();
    for c in &all {
        if capability.is_none_or(|cap| cap == c.capability) {
            items.push(Item::Answer(c));
        }
    }
    if capability.is_none_or(|cap| cap == "validation") {
        for c in &all {
            items.push(Item::Judge(c, true));
            items.push(Item::Judge(c, false));
        }
    }
    items.truncate(limit.unwrap_or(usize::MAX));
    if items.is_empty() {
        bail!("no benchmark cases for capability `{}`", capability.unwrap_or(""));
    }
    let now = now_ms();
    let id = format!("bench_{}", &blake3::hash(format!("{model}{}", uuid::Uuid::now_v7()).as_bytes()).to_hex()[..12]);
    d.store.conn().execute(
        "INSERT INTO bench_runs (id, model, capability, class, created_at, state, cases) VALUES (?1, ?2, ?3, ?4, ?5, 'RUNNING', ?6)",
        params![id, model, capability, class.as_str(), now, items.len() as i64],
    )?;
    let ceiling = d.ceiling();
    for it in &items {
        let req = WorkRequest::parse(&it.request(model, class).to_string())?;
        let job = format!("tny_{}", &blake3::hash(format!("{id}{}", it.id()).as_bytes()).to_hex()[..16]);
        let mut env = compile(&req, &job, now_ms(), &d.defaults())?;
        // Synthetic prompts carry no user data: run wherever the ceiling
        // allows, so that node models can be measured too.
        env.privacy = ceiling.clone();
        d.store.insert_job(&env)?;
        d.store.conn().execute(
            "INSERT INTO bench_jobs (job_id, run_id, case_id, capability) VALUES (?1, ?2, ?3, ?4)",
            params![job, id, it.id(), it.capability()],
        )?;
    }
    d.wake.notify_one();
    Ok(run(&d.store, &id)?.expect("just inserted"))
}

pub fn run(store: &Store, id: &str) -> Result<Option<Run>> {
    Ok(runs_where(store, "id = ?1", &[&id])?.into_iter().next())
}

pub fn runs(store: &Store, limit: i64) -> Result<Vec<Run>> {
    runs_where(store, "1 = 1 ORDER BY created_at DESC LIMIT ?1", &[&limit])
}

fn runs_where(store: &Store, clause: &str, args: &[&dyn rusqlite::ToSql]) -> Result<Vec<Run>> {
    let conn = store.conn();
    let mut stmt = conn.prepare(&format!(
        "SELECT id, model, capability, class, created_at, finished_at, state, cases, scored, passed, errors
         FROM bench_runs WHERE {clause}"
    ))?;
    Ok(stmt
        .query_map(args, |r| {
            Ok(Run {
                id: r.get(0)?,
                model: r.get(1)?,
                capability: r.get(2)?,
                class: r.get(3)?,
                created_at: r.get(4)?,
                finished_at: r.get(5)?,
                state: r.get(6)?,
                cases: r.get(7)?,
                scored: r.get(8)?,
                passed: r.get(9)?,
                errors: r.get(10)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
}

pub fn cases_of(store: &Store, run_id: &str) -> Result<Vec<Value>> {
    let conn = store.conn();
    let mut stmt = conn.prepare(
        "SELECT b.job_id, b.case_id, b.capability, b.scored, b.passed, b.note, j.state
         FROM bench_jobs b JOIN jobs j ON j.id = b.job_id WHERE b.run_id = ?1 ORDER BY b.case_id",
    )?;
    Ok(stmt
        .query_map([run_id], |r| {
            Ok(json!({
                "job_id": r.get::<_, String>(0)?,
                "case": r.get::<_, String>(1)?,
                "capability": r.get::<_, String>(2)?,
                "scored": r.get::<_, bool>(3)?,
                "passed": r.get::<_, Option<bool>>(4)?,
                "note": r.get::<_, Option<String>>(5)?,
                "job_state": r.get::<_, String>(6)?,
            }))
        })?
        .collect::<rusqlite::Result<_>>()?)
}

/// Scores every finished, unscored benchmark job. Returns the models whose
/// quality evidence changed.
pub fn score_pending(store: &Store) -> Result<Vec<String>> {
    let pending: Vec<(String, String, String, String)> = {
        let conn = store.conn();
        let mut stmt = conn.prepare(
            "SELECT b.job_id, b.run_id, b.case_id, r.model FROM bench_jobs b JOIN bench_runs r ON r.id = b.run_id
             WHERE b.scored = 0",
        )?;
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.collect::<rusqlite::Result<_>>()?
    };
    let cases = suite();
    let mut changed = Vec::new();
    for (job, run_id, case_id, model) in pending {
        let Some(row) = store.get(&job)? else { continue };
        if !row.state.is_terminal() {
            continue;
        }
        let it = item(&cases, &case_id);
        let (passed, note) = match (row.state, store.result(&job)?, &it) {
            (JobState::Ready, Some(r), Some(it)) => {
                let (ok, note) = it.score(&r.output);
                store.record_quality(&model, it.capability(), &case_id, ok, &run_id, &job)?;
                changed.push(model.clone());
                (Some(ok), note)
            }
            // An answer that fails the output contract (e.g. not JSON) is
            // the model's failure, not the infrastructure's.
            (JobState::ModelFailed, _, Some(it)) if row.failure.as_ref().is_some_and(|f| f["stage"] == "output") => {
                store.record_quality(&model, it.capability(), &case_id, false, &run_id, &job)?;
                changed.push(model.clone());
                let why = row.failure.as_ref().and_then(|f| f["reason"].as_str()).unwrap_or("invalid output");
                (Some(false), Some(why.to_string()))
            }
            (state, ..) => (None, Some(format!("job ended {state}; not scored"))),
        };
        let conn = store.conn();
        conn.execute(
            "UPDATE bench_jobs SET scored = 1, passed = ?2, note = ?3 WHERE job_id = ?1",
            params![job, passed, note],
        )?;
        conn.execute(
            "UPDATE bench_runs SET scored = scored + 1, passed = passed + ?2, errors = errors + ?3 WHERE id = ?1",
            params![run_id, i64::from(passed == Some(true)), i64::from(passed.is_none())],
        )?;
        conn.execute(
            "UPDATE bench_runs SET state = 'COMPLETE', finished_at = ?2 WHERE id = ?1 AND scored >= cases",
            params![run_id, now_ms()],
        )?;
    }
    changed.sort();
    changed.dedup();
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str) -> Case {
        suite().into_iter().find(|c| c.id == id).unwrap()
    }

    #[test]
    fn checks_are_lenient_about_phrasing_and_strict_about_facts() {
        assert!(score(&case("general.arith"), "The answer is 42.").0);
        assert!(!score(&case("general.arith"), "43").0);
        assert!(score(&case("classification.spam"), "Spam").0);
        assert!(!score(&case("classification.spam"), "Not spam").0);
        assert!(score(&case("general.obey"), "OK").0);
        assert!(!score(&case("general.obey"), "OK! Happy to help with anything else you need.").0);
        assert!(score(&case("extraction.person"), "```json\n{\"name\": \"Maria\", \"age\": 34}\n```").0);
        assert!(score(&case("extraction.person"), "{\"name\": \"Maria\", \"age\": \"34\"}").0);
        assert!(!score(&case("extraction.person"), "{\"name\": \"Maria\"}").0);
        assert!(score(&case("rust.mutable"), "`let mut x`").0);
    }

    #[test]
    fn judge_items_round_trip_and_score_verdicts() {
        let cases = suite();
        let it = item(&cases, "judge.general.arith.bad").unwrap();
        assert_eq!(it.capability(), "validation.specificity");
        assert!(it.score("{\"verdict\": \"fail\", \"issues\": [\"wrong sum\"]}").0);
        assert!(!it.score("{\"verdict\": \"pass\"}").0);
        assert_eq!(item(&cases, "judge.general.arith.good").unwrap().capability(), "validation.sensitivity");
        assert!(matches!(item(&cases, "rust.iter"), Some(Item::Answer(_))));
        // Each case's good answer passes its own check and its bad one does not.
        for c in &cases {
            assert!(score(c, c.good).0, "{} good", c.id);
            assert!(!score(c, c.bad).0, "{} bad", c.id);
        }
    }

    #[test]
    fn every_case_has_a_capability_and_a_check() {
        for c in suite() {
            let has_check = !c.check.contains_all.is_empty() || !c.check.json.is_empty();
            assert!(has_check && !c.capability.is_empty(), "{}", c.id);
        }
    }
}
