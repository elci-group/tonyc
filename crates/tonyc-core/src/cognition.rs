//! Cognitive Store contracts (directive §10, §15–17).
//!
//! Inference results are kept as *cognitive objects*: typed, scoped,
//! traceable to their sources and the model that produced them, and governed
//! by a freshness policy so that cached cognition never silently stands in
//! for current truth. This module states the rules; the daemon stores the
//! objects and enforces them.

use crate::Millis;
use crate::duration;
use crate::locality::{Locality, LocalityPolicy};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

macro_rules! word_enum {
    ($(#[$m:meta])* $name:ident { $($variant:ident = $text:literal),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $name { $($variant),+ }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant),+];
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
        }
        impl FromStr for $name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, String> {
                let norm = s.trim().replace('-', "_");
                $name::ALL.iter().copied()
                    .find(|v| v.as_str().eq_ignore_ascii_case(&norm))
                    .ok_or_else(|| format!(
                        "unknown {} `{s}` (expected one of {})",
                        stringify!($name).to_ascii_lowercase(),
                        $name::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>().join(", ")))
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> { s.serialize_str(self.as_str()) }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
            }
        }
    };
}
pub(crate) use word_enum;

word_enum! {
    /// What a cognitive object is (directive §15).
    CognitiveKind {
        Fact = "fact", Analysis = "analysis", Summary = "summary",
        Classification = "classification", Prediction = "prediction",
        Artifact = "artifact", Embedding = "embedding", Validation = "validation",
        NegativeResult = "negative_result",
    }
}

word_enum! {
    /// Freshness strategies (directive §17).
    FreshnessPolicy {
        Immutable = "immutable", Ttl = "ttl", SourceHash = "source_hash",
        EventInvalidated = "event", DependencyInvalidated = "dependency", Manual = "manual",
    }
}

word_enum! {
    /// Validation state. Validators arrive with refinement (Phase 5); until
    /// then every object is `unverified`.
    ValidationState { Unverified = "unverified", Verified = "verified", Failed = "failed" }
}

word_enum! {
    ObjectState { Fresh = "FRESH", Stale = "STALE" }
}

word_enum! {
    /// Corpus sensitivity. `sensitive` and `secret` sources structurally
    /// narrow any job that reads them to LOCAL_ONLY; `secret` content is also
    /// withheld from query listings.
    Sensitivity { Public = "public", Normal = "normal", Sensitive = "sensitive", Secret = "secret" }
}

impl Sensitivity {
    pub fn forces_local(self) -> bool {
        self >= Sensitivity::Sensitive
    }
}

word_enum! {
    /// Corpus namespaces (directive §10).
    Namespace {
        User = "user", Project = "project", Repository = "repository",
        Workspace = "workspace", Domain = "domain", Ephemeral = "ephemeral",
    }
}

/// `namespace[:name]`, e.g. `project:tonyc`, `user`, `ephemeral`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Scope {
    pub namespace: Namespace,
    pub name: Option<String>,
}

impl Scope {
    pub fn user() -> Scope {
        Scope { namespace: Namespace::User, name: None }
    }

    /// Whether an object in `self` may serve a job in `job`: same scope, or
    /// a user-wide object serving any non-ephemeral job.
    pub fn visible_to(&self, job: &Scope) -> bool {
        self == job
            || (self.namespace == Namespace::User && self.name.is_none() && job.namespace != Namespace::Ephemeral)
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}:{n}", self.namespace),
            None => write!(f, "{}", self.namespace),
        }
    }
}

impl FromStr for Scope {
    type Err = String;
    fn from_str(s: &str) -> Result<Scope, String> {
        let (ns, name) = match s.trim().split_once(':') {
            Some((ns, name)) => (ns, Some(name.trim())),
            None => (s.trim(), None),
        };
        let namespace: Namespace = ns.parse()?;
        let name = match name {
            Some("") => return Err(format!("scope `{s}` has an empty name")),
            Some(n) if !n.chars().all(|c| c.is_ascii_alphanumeric() || "._-/".contains(c)) => {
                return Err(format!("scope name `{n}` may use letters, digits, `.`, `_`, `-`, `/`"));
            }
            other => other.map(str::to_string),
        };
        Ok(Scope { namespace, name })
    }
}

impl Serialize for Scope {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Scope {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

/// An object's freshness policy with its parameters. Every object is also
/// invalidated when an object it was derived from is (dependency cascade).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Freshness {
    pub policy: FreshnessPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<String>,
}

impl Freshness {
    /// When the object expires, if it does by time.
    pub fn expires_at(&self, created_at: Millis) -> Option<Millis> {
        match self.policy {
            FreshnessPolicy::Ttl => self.ttl_ms.map(|t| created_at + t),
            _ => None,
        }
    }

    pub fn describe(&self) -> String {
        match self.policy {
            FreshnessPolicy::Ttl => format!("ttl {}", duration::Human(self.ttl_ms.unwrap_or(0))),
            FreshnessPolicy::EventInvalidated => format!("event [{}]", self.events.join(", ")),
            p => p.to_string(),
        }
    }
}

/// Whether an object produced under `object` locality may be shown to a
/// model running a job under `job` locality. Data derived under a narrow
/// policy must not ride along into a job that may go wider.
pub fn context_admissible(object: &LocalityPolicy, job: &LocalityPolicy) -> bool {
    job.class == Locality::LocalOnly || object.class == Locality::Unrestricted || object == job
}

// ------------------------------------------------------------ retrieval ---

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what", "which", "why", "with", "you", "your",
    "me", "my", "we", "our", "these", "those", "there", "their", "them", "will", "would", "should", "any",
];

/// Lower-cased alphanumeric terms with stop words and one-letter tokens
/// removed, crudely stemmed (`dependencies` → `dependenc`).
pub fn terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(|w| w.to_lowercase())
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .map(|w| stem(&w))
        .collect()
}

fn stem(w: &str) -> String {
    for suffix in ["ies", "ing", "ed", "es", "s", "y"] {
        if w.len() > suffix.len() + 3
            && let Some(base) = w.strip_suffix(suffix)
        {
            return base.to_string();
        }
    }
    w.to_string()
}

/// A document for lexical retrieval.
pub struct Doc<'a> {
    pub id: &'a str,
    pub text: &'a str,
}

/// BM25 scores of `query` against `docs`, normalised to `[0, 1]` by the
/// query's best possible self-match, highest first. Lexical only: semantic
/// (embedding) retrieval needs an embedding model and plugs in later.
pub fn rank(query: &str, docs: &[Doc]) -> Vec<(String, f64)> {
    let q: BTreeSet<String> = terms(query).into_iter().collect();
    if q.is_empty() || docs.is_empty() {
        return vec![];
    }
    let tokenised: Vec<Vec<String>> = docs.iter().map(|d| terms(d.text)).collect();
    let n = docs.len() as f64;
    let avg = tokenised.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let mut df: BTreeMap<&str, f64> = BTreeMap::new();
    for t in &tokenised {
        let uniq: BTreeSet<&str> = t.iter().map(String::as_str).collect();
        for w in uniq {
            *df.entry(w).or_default() += 1.0;
        }
    }
    let (k1, b) = (1.2, 0.75);
    let idf = |w: &str| {
        let d = df.get(w).copied().unwrap_or(0.0);
        ((n - d + 0.5) / (d + 0.5) + 1.0).ln()
    };
    // The best a document could score: every query term present once in a
    // document of average length.
    let ceiling: f64 = q.iter().map(|w| idf(w.as_str()).max(idf_floor(n)) * (k1 + 1.0) / (1.0 + k1)).sum();
    let mut out: Vec<(String, f64)> = docs
        .iter()
        .zip(&tokenised)
        .filter_map(|(doc, toks)| {
            let len = toks.len() as f64;
            let score: f64 = q
                .iter()
                .map(|w| {
                    let tf = toks.iter().filter(|t| *t == w).count() as f64;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    idf(w) * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / avg.max(1.0)))
                })
                .sum();
            (score > 0.0).then(|| (doc.id.to_string(), (score / ceiling).min(1.0)))
        })
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

fn idf_floor(n: f64) -> f64 {
    // idf of a term present in every document.
    ((0.5) / (n + 0.5) + 1.0).ln()
}

/// Jaccard overlap of two sets of source paths.
pub fn source_overlap(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let a: BTreeSet<&String> = a.iter().collect();
    let b: BTreeSet<&String> = b.iter().collect();
    a.intersection(&b).count() as f64 / a.union(&b).count() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_parse_and_scope_visibility() {
        let p: Scope = "project:tonyc".parse().unwrap();
        assert_eq!(p.to_string(), "project:tonyc");
        assert_eq!("USER".parse::<Scope>().unwrap(), Scope::user());
        assert!("galaxy:x".parse::<Scope>().is_err());
        assert!("project:".parse::<Scope>().is_err());
        assert!(Scope::user().visible_to(&p));
        assert!(!p.visible_to(&Scope::user()));
        assert!(!p.visible_to(&"project:other".parse().unwrap()));
        assert!(!Scope::user().visible_to(&"ephemeral".parse().unwrap()));
    }

    #[test]
    fn narrow_cognition_does_not_ride_into_wide_jobs() {
        let local = LocalityPolicy::local_only();
        let open = LocalityPolicy { class: Locality::Unrestricted, ssh_allowlist: vec![], remote_allowlist: vec![] };
        assert!(context_admissible(&open, &local));
        assert!(context_admissible(&local, &local));
        assert!(!context_admissible(&local, &open));
    }

    #[test]
    fn lexical_ranking_prefers_related_documents() {
        let docs = [
            Doc { id: "deps", text: "Analyse repository dependency architecture: tokio, serde, axum dependencies" },
            Doc { id: "cats", text: "A poem about cats sleeping in the sun" },
            Doc { id: "build", text: "Explain the build failure in the linker step" },
        ];
        let ranked = rank("Which dependencies are likely unnecessary?", &docs);
        assert_eq!(ranked[0].0, "deps");
        assert!(ranked.iter().all(|(id, _)| id != "cats"));
        assert!(ranked[0].1 > 0.0 && ranked[0].1 <= 1.0);
        assert!(rank("the of and", &docs).is_empty());
    }

    #[test]
    fn overlap() {
        let a = vec!["/a".to_string(), "/b".to_string()];
        let b = vec!["/b".to_string()];
        assert_eq!(source_overlap(&a, &b), 0.5);
        assert_eq!(source_overlap(&a, &[]), 0.0);
    }

    #[test]
    fn sensitivity_ordering() {
        assert!(Sensitivity::Secret.forces_local());
        assert!(Sensitivity::Sensitive.forces_local());
        assert!(!Sensitivity::Normal.forces_local());
    }
}
