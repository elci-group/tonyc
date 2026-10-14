//! Refinement contracts (directive §14): deterministic checks, validator
//! statistics, and how verdicts move confidence.
//!
//! Confidence is a probability that the answer is correct. A draft starts
//! at its model's expected quality for the capability. A validator verdict
//! updates it by Bayes' rule using that validator's *measured* sensitivity
//! (it passes correct answers) and specificity (it fails wrong ones). A
//! failed deterministic check proves the answer wrong. Validators are
//! treated as independent of the drafter and of each other — the reason a
//! model never validates its own answer — and the posterior is capped so
//! that no chain of verdicts claims certainty.

use crate::model::ModelProfile;
use serde::{Deserialize, Serialize};

/// The most confidence any combination of verdicts can establish.
pub const MAX_CONFIDENCE: f64 = 0.99;
/// A validator must beat chance by this margin (sensitivity + specificity
/// > 1 + margin) for its verdicts to count.
pub const INFORMATIVE_MARGIN: f64 = 0.2;

/// A deterministic condition on an answer. Text matching is
/// case-insensitive.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    /// Every one of these must appear.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contains: Vec<String>,
    /// None of these may appear.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub absent: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<usize>,
    /// The answer must be a JSON object with these fields present and
    /// not null.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_fields: Vec<String>,
}

impl Check {
    pub fn validate(&self) -> Result<(), String> {
        if self.contains.is_empty() && self.absent.is_empty() && self.max_chars.is_none() && self.json_fields.is_empty()
        {
            return Err("a check needs contains, absent, max_chars or json_fields".into());
        }
        if self.contains.iter().chain(&self.absent).chain(&self.json_fields).any(|s| s.trim().is_empty()) {
            return Err("check strings must not be empty".into());
        }
        Ok(())
    }

    /// The problems with an answer, if any.
    pub fn problems(&self, answer: &str) -> Vec<String> {
        let text = answer.to_lowercase();
        let mut out = Vec::new();
        for want in &self.contains {
            if !text.contains(&want.to_lowercase()) {
                out.push(format!("does not contain `{want}`"));
            }
        }
        for bad in &self.absent {
            if text.contains(&bad.to_lowercase()) {
                out.push(format!("contains `{bad}`"));
            }
        }
        if let Some(max) = self.max_chars {
            let n = answer.trim().chars().count();
            if n > max {
                out.push(format!("is {n} characters long; at most {max} allowed"));
            }
        }
        if !self.json_fields.is_empty() {
            match first_json_object(answer) {
                None => out.push("is not a JSON object".into()),
                Some(v) => {
                    for f in &self.json_fields {
                        if v.get(f).is_none_or(|x| x.is_null()) {
                            out.push(format!("JSON field `{f}` is missing"));
                        }
                    }
                }
            }
        }
        out
    }
}

/// Problems an answer has against all checks.
pub fn check_all(checks: &[Check], answer: &str) -> Vec<String> {
    checks.iter().flat_map(|c| c.problems(answer)).collect()
}

/// The first `{...}` in a text, parsed.
pub fn first_json_object(text: &str) -> Option<serde_json::Value> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (start < end)
        .then(|| serde_json::from_str(&text[start..=end]).ok())
        .flatten()
        .filter(|v: &serde_json::Value| v.is_object())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
    /// The validator gave no usable verdict; confidence does not move.
    Abstain,
}

/// A validator's verdict and the issues it named.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Judgement {
    pub verdict: Verdict,
    pub issues: Vec<String>,
}

/// Reads a validator's reply: JSON `{"verdict": ..., "issues": [...]}`, or
/// failing that a bare PASS/FAIL. Anything else abstains.
pub fn parse_judgement(reply: &str) -> Judgement {
    if let Some(v) = first_json_object(reply) {
        let verdict = v.get("verdict").and_then(|x| x.as_str()).map(str::to_lowercase);
        let issues = v
            .get("issues")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|i| i.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        match verdict.as_deref() {
            Some(t) if t.contains("fail") => return Judgement { verdict: Verdict::Fail, issues },
            Some(t) if t.contains("pass") => return Judgement { verdict: Verdict::Pass, issues },
            _ => {}
        }
    }
    let upper = reply.to_uppercase();
    let (pass, fail) = (upper.contains("PASS"), upper.contains("FAIL"));
    let verdict = match (pass, fail) {
        (true, false) => Verdict::Pass,
        (false, true) => Verdict::Fail,
        _ => Verdict::Abstain,
    };
    Judgement { verdict, issues: vec![] }
}

/// How well a model judges answers, from its profile: measured
/// `validation.sensitivity` / `validation.specificity` when benchmarked,
/// else its general quality for both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatorStats {
    pub model: String,
    pub sensitivity: f64,
    pub specificity: f64,
    pub measured: bool,
}

impl ValidatorStats {
    pub fn of(p: &ModelProfile) -> ValidatorStats {
        let general = p.capabilities.get("general").copied().unwrap_or(0.5);
        let s = p.capabilities.get("validation.sensitivity").copied();
        let t = p.capabilities.get("validation.specificity").copied();
        ValidatorStats {
            model: p.id.clone(),
            sensitivity: s.unwrap_or(general),
            specificity: t.unwrap_or(general),
            measured: s.is_some() || t.is_some(),
        }
    }

    /// Whether its verdicts carry information at all.
    pub fn informative(&self) -> bool {
        self.sensitivity + self.specificity > 1.0 + INFORMATIVE_MARGIN
    }

    /// Balanced accuracy, for ranking validators.
    pub fn accuracy(&self) -> f64 {
        (self.sensitivity + self.specificity) / 2.0
    }
}

/// Bayes' rule: confidence after a verdict.
pub fn update(p: f64, verdict: Verdict, v: &ValidatorStats) -> f64 {
    let (s, t) = (v.sensitivity.clamp(0.01, 0.99), v.specificity.clamp(0.01, 0.99));
    let p = p.clamp(0.0, 1.0);
    let post = match verdict {
        Verdict::Pass => p * s / (p * s + (1.0 - p) * (1.0 - t)),
        Verdict::Fail => p * (1.0 - s) / (p * (1.0 - s) + (1.0 - p) * t),
        Verdict::Abstain => p,
    };
    if post.is_nan() { p } else { post.min(MAX_CONFIDENCE) }
}

/// The confidence reached if every one of `validators` passes the answer.
pub fn if_all_pass(p: f64, validators: &[ValidatorStats]) -> f64 {
    validators.iter().fold(p, |acc, v| update(acc, Verdict::Pass, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: f64, t: f64) -> ValidatorStats {
        ValidatorStats { model: "m/v".into(), sensitivity: s, specificity: t, measured: true }
    }

    #[test]
    fn verdicts_move_confidence_by_bayes() {
        let good = v(0.8, 0.8);
        assert!((update(0.7, Verdict::Pass, &good) - 0.56 / 0.62).abs() < 1e-12);
        assert!((update(0.7, Verdict::Fail, &good) - 0.14 / 0.38).abs() < 1e-12);
        assert_eq!(update(0.7, Verdict::Abstain, &good), 0.7);
        assert!(update(0.999, Verdict::Pass, &good) <= MAX_CONFIDENCE);
        // A validator no better than chance leaves confidence where it was.
        let coin = v(0.5, 0.5);
        assert!((update(0.7, Verdict::Pass, &coin) - 0.7).abs() < 1e-12);
        assert!(!coin.informative() && good.informative());
        // One that passes everything is not informative either.
        assert!(!v(1.0, 0.05).informative());
    }

    #[test]
    fn independent_passes_compound() {
        let p = if_all_pass(0.6, &[v(0.8, 0.8), v(0.8, 0.8)]);
        assert!(p > update(0.6, Verdict::Pass, &v(0.8, 0.8)) && p < MAX_CONFIDENCE);
    }

    #[test]
    fn checks() {
        let c = Check {
            contains: vec!["Paris".into()],
            absent: vec!["london".into()],
            max_chars: Some(20),
            ..Default::default()
        };
        assert!(c.problems("It is paris.").is_empty());
        assert_eq!(c.problems("London, not Paris").len(), 1);
        assert_eq!(c.problems("Berlin").len(), 1);
        let j = Check { json_fields: vec!["name".into()], ..Default::default() };
        assert!(j.problems("```json\n{\"name\": \"x\"}\n```").is_empty());
        assert_eq!(j.problems("{\"name\": null}"), ["JSON field `name` is missing"]);
        assert_eq!(j.problems("no json"), ["is not a JSON object"]);
        assert!(Check::default().validate().is_err());
    }

    #[test]
    fn judgements_parse_leniently() {
        assert_eq!(parse_judgement("{\"verdict\": \"PASS\", \"issues\": []}").verdict, Verdict::Pass);
        let j = parse_judgement("Here you go: {\"verdict\": \"fail\", \"issues\": [\"wrong year\"]}");
        assert_eq!((j.verdict, j.issues), (Verdict::Fail, vec!["wrong year".to_string()]));
        assert_eq!(parse_judgement("FAIL").verdict, Verdict::Fail);
        assert_eq!(parse_judgement("pass or fail, hard to say").verdict, Verdict::Abstain);
        assert_eq!(parse_judgement("maybe").verdict, Verdict::Abstain);
    }
}
