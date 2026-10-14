//! Human-readable decision traces (directive §23): what was asked, what was
//! chosen, why, what was rejected, and exactly what crossed the host
//! boundary.

use crate::store::{BoundaryRow, EventRow, JobRow, ResultRow};
use std::fmt::Write;
use tonyc_core::Millis;
use tonyc_core::duration::Human;
use tonyc_core::plan::{Decision, PlanFailure};

pub fn render(
    job: &JobRow,
    result: Option<&ResultRow>,
    events: &[EventRow],
    boundary: &[BoundaryRow],
    ledger: Option<&crate::economics::LedgerRow>,
    passes: &[crate::refinement::PassRow],
) -> String {
    let env = &job.envelope;
    let mut s = String::new();
    let _ = writeln!(s, "Job {}  {}", job.id, job.state);
    let _ = writeln!(s, "  submitted {}  attempts {}", utc(job.submitted_at), job.attempts);

    let _ = writeln!(s, "\nRequested:");
    let _ = writeln!(s, "  objective: {}", one_line(&env.task.objective, 100));
    match env.latency.deadline_at {
        Some(at) => {
            let _ = writeln!(s, "  {} ≤ {} (deadline {})", env.latency.class, Human(at - env.submitted_at), utc(at));
        }
        None => {
            let _ = writeln!(s, "  {} (no deadline)", env.latency.class);
        }
    }
    let mut locality = env.privacy.class.to_string();
    if !env.privacy.ssh_allowlist.is_empty() {
        locality += &format!(" ssh=[{}]", env.privacy.ssh_allowlist.join(", "));
    }
    if !env.privacy.remote_allowlist.is_empty() {
        locality += &format!(" remote=[{}]", env.privacy.remote_allowlist.join(", "));
    }
    let _ = writeln!(s, "  {locality}");
    let _ = writeln!(
        s,
        "  quality ≥ {:.2} for `{}`, verification {}",
        env.quality.floor, env.task.capability, env.quality.verification
    );
    let _ = writeln!(s, "  output {} {}, up to {} tokens", env.output.kind, env.output.schema, env.task.max_tokens);
    let _ = writeln!(
        s,
        "  reuse {} in scope {}, freshness {}",
        if env.reuse.cache { "on" } else { "off" },
        env.reuse.scope,
        env.reuse.freshness.describe()
    );

    if let Some(c) = &job.cognition {
        write_cognition(&mut s, c);
    }

    if let Some(d) = job.decision.as_ref().and_then(|v| serde_json::from_value::<Decision>(v.clone()).ok()) {
        let _ = writeln!(s, "\nSelected:");
        let _ = writeln!(s, "  {} via {}", d.model, d.exposure.label());
        let _ = writeln!(s, "\nReason:");
        for r in &d.reasons {
            let _ = writeln!(s, "  - {r}");
        }
        let u = &d.utility;
        let _ = writeln!(
            s,
            "  - U = Q {:.3} + R {:.3} + V {:.3} − C {:.3} − L {:.3} − E {:.4} = {:.3}",
            u.quality, u.reuse, u.future_value, u.cost, u.latency, u.energy, u.total
        );
        write_rejections(&mut s, d.considered, &d.rejected);
    }

    if let Some(f) = &job.failure {
        let _ = writeln!(s, "\nFailure:");
        let _ = writeln!(s, "  stage {}: {}", f["stage"].as_str().unwrap_or("?"), f["reason"].as_str().unwrap_or("?"));
        if let Some(p) = f.get("plan").and_then(|p| serde_json::from_value::<PlanFailure>(p.clone()).ok()) {
            write_rejections(&mut s, p.considered, &p.rejected);
        }
    }

    if let Some(r) = result.filter(|r| r.reused_from.is_some()) {
        let _ = writeln!(s, "\nCompleted:");
        let _ = writeln!(s, "  {} from the Cognitive Store — no inference", utc(r.finished_at));
    } else if let Some(r) = result {
        let _ = writeln!(s, "\nCompleted:");
        let _ = writeln!(
            s,
            "  {} after {} running ({} total)",
            utc(r.finished_at),
            Human(r.finished_at - r.started_at),
            Human(r.finished_at - job.submitted_at)
        );
        let _ = writeln!(
            s,
            "  {} tokens in, {} out; model digest {}",
            r.tokens_in,
            r.tokens_out,
            r.model_digest.as_deref().unwrap_or("unknown")
        );
        for src in &r.sources {
            let _ = writeln!(s, "  source {} ({} bytes, blake3 {})", src.path, src.bytes, &src.blake3[..16]);
        }
    }

    if !passes.is_empty() {
        let _ = writeln!(s, "\nRefinement:");
        let t0 = passes[0].started_at;
        for p in passes {
            let verdict = p.verdict.as_deref().map(|v| format!(" {v}")).unwrap_or_default();
            let mut note = String::new();
            if let Some(issues) = p.detail.get("issues").and_then(|i| i.as_array()).filter(|a| !a.is_empty()) {
                let list: Vec<&str> = issues.iter().filter_map(|i| i.as_str()).take(3).collect();
                note = format!(" — {}", list.join("; "));
            }
            if let Some(problems) = p.detail.get("problems").and_then(|i| i.as_array()) {
                let list: Vec<&str> = problems.iter().filter_map(|i| i.as_str()).take(3).collect();
                note = format!(" — {}", list.join("; "));
            }
            if p.detail.get("escalated").and_then(|e| e.as_bool()) == Some(true) {
                note.push_str(" (escalated)");
            }
            let _ = writeln!(
                s,
                "  T+{:<8} {:<9} {:<28}{verdict:<8} confidence {:.2}{note}",
                Human(p.started_at - t0).to_string(),
                p.role,
                p.model,
                p.confidence
            );
        }
    }

    if let Some(l) = ledger {
        let _ = writeln!(s, "\nEconomics (estimates):");
        let _ =
            writeln!(s, "  conventional cost £{:.5} (one reference-API call of the answer's size)", l.conventional_gbp);
        let _ = writeln!(
            s,
            "  this job used {} tokens in, {} out{}",
            l.tokens_in,
            l.tokens_out,
            if passes.len() > 1 {
                format!(" across {} passes", passes.iter().filter(|p| p.model != "checks").count())
            } else {
                String::new()
            }
        );
        let _ = writeln!(
            s,
            "  operating cost £{:.5} (energy {:.3} Wh on {}, spend £{:.5})",
            l.operating_gbp(),
            l.energy_wh,
            l.domain,
            l.spend_gbp
        );
        let _ = writeln!(s, "  estimated cost avoided £{:.5}", (l.conventional_gbp - l.spend_gbp).max(0.0));
    }

    let off_host: Vec<_> = boundary.iter().filter(|b| b.leaves_host).collect();
    let _ = writeln!(s, "\nRemote data transmitted:");
    if off_host.is_empty() {
        let local = boundary.len();
        let _ = writeln!(
            s,
            "  NONE{}",
            if local > 0 { format!(" ({local} on-host dispatch records)") } else { String::new() }
        );
    } else {
        for b in off_host {
            let hash =
                b.payload_blake3.as_deref().map(|h| format!(", payload blake3 {}", &h[..16])).unwrap_or_default();
            let _ = writeln!(
                s,
                "  {} {}: {} bytes sent, {} received{hash} — {}",
                utc(b.at),
                b.exposure.label(),
                b.bytes_sent,
                b.bytes_received,
                b.summary
            );
        }
        let _ = writeln!(s, "  (exact payloads: `tonyc audit --job {} --payload`)", job.id);
    }

    let _ = writeln!(s, "\nHistory:");
    for e in events {
        let state = e.state.map(|st| st.to_string()).unwrap_or_default();
        let detail = e
            .detail
            .as_ref()
            .and_then(|d| d.get("reason").and_then(|r| r.as_str()).map(str::to_string))
            .unwrap_or_default();
        let _ = writeln!(s, "  {}  {:<18} {:<20} {}", utc(e.at), e.kind, state, detail);
    }
    s
}

fn write_cognition(s: &mut String, c: &serde_json::Value) {
    let _ = writeln!(s, "\nCognition:");
    if let Some(r) = c.get("reused") {
        let _ = writeln!(
            s,
            "  satisfied by existing cognition {} (job {}, {})",
            r["object"].as_str().unwrap_or("?"),
            r["produced_by"].as_str().unwrap_or("?"),
            r["model"].as_str().unwrap_or("?")
        );
        for n in r["notes"].as_array().into_iter().flatten() {
            let _ = writeln!(s, "  - {}", n.as_str().unwrap_or(""));
        }
    }
    if let Some(msg) = c.get("cache").and_then(|v| v.as_str()) {
        let _ = writeln!(s, "  cache {msg}");
    }
    for n in c.get("direct_skipped").and_then(|v| v.as_array()).into_iter().flatten() {
        let _ = writeln!(s, "  not reused: {}", n.as_str().unwrap_or(""));
    }
    for p in c.get("context").and_then(|v| v.as_array()).into_iter().flatten() {
        let id = p["id"].as_str().unwrap_or("?");
        if p["relation"] == "refresh_of" {
            let changed: Vec<&str> =
                p["changed_sources"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).collect();
            let _ = writeln!(
                s,
                "  refreshing stale {id}; changed: {}",
                if changed.is_empty() { "unknown".into() } else { changed.join(", ") }
            );
        } else {
            let _ = writeln!(s, "  context {id} (relevance {:.2})", p["score"].as_f64().unwrap_or(0.0));
        }
    }
    if let Some(n) = c.get("narrowed") {
        let because: Vec<&str> = n["because"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).collect();
        let _ = writeln!(
            s,
            "  locality narrowed {} → LOCAL_ONLY: {}",
            n["from"].as_str().unwrap_or("?"),
            because.join("; ")
        );
    }
    if let Some(id) = c.get("produced").and_then(|v| v.as_str()) {
        let _ = writeln!(s, "  produced {id}");
    }
    if c.as_object().is_some_and(|m| m.is_empty()) {
        let _ = writeln!(s, "  no existing cognition applied");
    }
}

fn write_rejections(s: &mut String, considered: usize, rejected: &[tonyc_core::Rejection]) {
    let _ = writeln!(s, "\nConsidered {considered} model(s); rejected {}:", rejected.len());
    for r in rejected {
        let _ = writeln!(s, "  - {} [{}]: {}", r.model, r.stage, r.reason);
    }
}

fn one_line(text: &str, max: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max { flat } else { format!("{}…", flat.chars().take(max).collect::<String>()) }
}

/// `2026-10-13 09:41:07Z`
pub fn utc(ms: Millis) -> String {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}Z", tod / 3600, tod % 3600 / 60, tod % 60)
}

#[cfg(test)]
mod tests {
    #[test]
    fn formats_utc() {
        assert_eq!(super::utc(0), "1970-01-01 00:00:00Z");
        assert_eq!(super::utc(1_791_878_400_000), "2026-10-13 08:00:00Z");
    }
}
