//! Tonyc as Model Context Protocol tools (directive Phase 7: agent
//! adapters).
//!
//! Any agent that speaks MCP — Claude Code, Claude Desktop, IDE agents —
//! gets Tonyc's contract as tools: look for existing cognition before
//! inferring (`tonyc_try`), hand off work it does not need now
//! (`tonyc_submit`), and collect it later (`tonyc_await`). The agent never
//! sees models, nodes or runtimes (§18), and nothing it asks for can go
//! beyond the daemon's locality ceiling.

use serde_json::{Value, json};
use std::sync::RwLock;
use std::time::Duration;
use tonyc_sdk::{Client, Error, Invalidate, Job, Locality, Query, TryOutcome, Urgency, Work};

/// MCP revisions this server can speak, newest first.
pub const MCP_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

pub struct Server {
    client: RwLock<Client>,
    base: Client,
}

impl Server {
    pub fn new(client: Client) -> Server {
        Server { client: RwLock::new(client.clone().with_origin("mcp")), base: client }
    }

    fn client(&self) -> Client {
        self.client.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Handles one JSON-RPC message; returns the response, if it needs one.
    pub async fn handle(&self, msg: Value) -> Option<Value> {
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or_default();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        // Notifications (no id) get no answer.
        let id = id?;
        let result = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tools() })),
            "tools/call" => Ok(self.call(&params).await),
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        Some(match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err((code, message)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
        })
    }

    fn initialize(&self, params: &Value) -> Value {
        let asked = params["protocolVersion"].as_str().unwrap_or_default();
        let version = MCP_VERSIONS.iter().find(|v| **v == asked).copied().unwrap_or(MCP_VERSIONS[0]);
        // Work is attributed to the agent that connected.
        if let Some(agent) = params["clientInfo"]["name"].as_str().filter(|a| !a.is_empty()) {
            *self.client.write().unwrap_or_else(|p| p.into_inner()) = self.base.clone().with_agent("mcp", agent);
        }
        json!({
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tonyc", "version": env!("CARGO_PKG_VERSION")},
            "instructions": INSTRUCTIONS,
        })
    }

    async fn call(&self, params: &Value) -> Value {
        let name = params["name"].as_str().unwrap_or_default();
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let outcome = match name {
            "tonyc_try" => self.try_existing(&args).await,
            "tonyc_submit" => self.submit(&args).await,
            "tonyc_await" => self.await_job(&args).await,
            "tonyc_status" => self.status(&args).await,
            "tonyc_query" => self.query(&args).await,
            "tonyc_cancel" => self.cancel(&args).await,
            "tonyc_invalidate" => self.invalidate(&args).await,
            _ => Err(format!("unknown tool `{name}`")),
        };
        match outcome {
            Ok((text, structured)) => json!({
                "content": [{"type": "text", "text": text}],
                "structuredContent": structured,
                "isError": false,
            }),
            Err(message) => json!({"content": [{"type": "text", "text": message}], "isError": true}),
        }
    }

    async fn try_existing(&self, args: &Value) -> Outcome {
        let work = work(args)?;
        match self.client().try_existing(&work).await.map_err(describe)? {
            TryOutcome::Hit(o) => {
                let text = format!(
                    "Existing cognition (object {}, model {}, confidence {:.2}):\n\n{}",
                    o.id,
                    o.model.as_deref().unwrap_or("?"),
                    o.confidence,
                    o.content.as_deref().unwrap_or("[withheld: secret]")
                );
                Ok((text, json!({"hit": true, "object_id": o.id, "content": o.content, "confidence": o.confidence})))
            }
            TryOutcome::Miss { nearest, .. } => {
                let mut text = String::from(
                    "No existing cognition answers this. Infer it yourself if you need it now, or tonyc_submit it if it can wait.",
                );
                if !nearest.is_empty() {
                    text.push_str("\n\nRelated objects:");
                    for o in &nearest {
                        text.push_str(&format!("\n- {} ({}): {}", o.id, o.kind, o.objective));
                    }
                }
                let ids: Vec<&str> = nearest.iter().map(|o| o.id.as_str()).collect();
                Ok((text, json!({"hit": false, "nearest": ids})))
            }
        }
    }

    async fn submit(&self, args: &Value) -> Outcome {
        let work = work(args)?;
        let client = self.client();
        let s = match args["idempotency_key"].as_str() {
            Some(key) => client.submit_idempotent(key, &work).await,
            None => client.submit(&work).await,
        }
        .map_err(describe)?;
        let env = &s.envelope;
        let deadline =
            env.latency.deadline_at.map(|d| format!(", due within {}", span(d - env.submitted_at))).unwrap_or_default();
        let text = format!(
            "{} job {} ({}, {}{deadline}). Collect it with tonyc_await.",
            if s.replayed { "Already submitted:" } else { "Submitted" },
            s.id,
            env.latency.class,
            env.privacy.class,
        );
        Ok((text, json!({"job_id": s.id, "state": s.state, "replayed": s.replayed})))
    }

    async fn await_job(&self, args: &Value) -> Outcome {
        let id = required(args, "job_id")?;
        let secs = args["timeout_seconds"].as_u64().unwrap_or(30).min(600);
        let job = self.client().await_job(id, Duration::from_secs(secs)).await.map_err(describe)?;
        Ok(job_view(&job))
    }

    async fn status(&self, args: &Value) -> Outcome {
        let job = self.client().status(required(args, "job_id")?).await.map_err(describe)?;
        Ok(job_view(&job))
    }

    async fn query(&self, args: &Value) -> Outcome {
        let q = Query {
            text: args["text"].as_str().map(str::to_string),
            scope: args["scope"].as_str().map(str::to_string),
            limit: Some(args["limit"].as_u64().unwrap_or(5).min(50) as usize),
            ..Query::default()
        };
        let objects = self.client().query(&q).await.map_err(describe)?;
        if objects.is_empty() {
            return Ok(("No matching cognition.".into(), json!({"objects": []})));
        }
        let mut text = String::new();
        for o in &objects {
            text.push_str(&format!(
                "## {} — {}\n{}\n\n",
                o.id,
                o.objective,
                o.content.as_deref().unwrap_or("[withheld: secret; fetch by id]")
            ));
        }
        let ids: Vec<&str> = objects.iter().map(|o| o.id.as_str()).collect();
        Ok((text.trim_end().to_string(), json!({"objects": ids})))
    }

    async fn cancel(&self, args: &Value) -> Outcome {
        let id = required(args, "job_id")?;
        let state = self.client().cancel(id).await.map_err(describe)?;
        Ok((format!("Job {id}: {state}"), json!({"job_id": id, "state": state})))
    }

    async fn invalidate(&self, args: &Value) -> Outcome {
        let pick = |k: &str| args[k].as_str().map(str::to_string);
        let what = match (pick("object"), pick("path"), pick("scope"), pick("event")) {
            (Some(o), None, None, None) => Invalidate::Object(o),
            (None, Some(p), None, None) => Invalidate::Path(p),
            (None, None, Some(s), None) => Invalidate::Scope(s),
            (None, None, None, Some(e)) => Invalidate::Event(e),
            _ => return Err("give exactly one of object, path, scope or event".into()),
        };
        let report = self.client().invalidate(what, args["reason"].as_str()).await.map_err(describe)?;
        let n = report["objects"].as_array().map_or(0, Vec::len);
        Ok((format!("Invalidated {n} object(s) and everything derived from them."), report))
    }
}

type Outcome = Result<(String, Value), String>;

fn required<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args[key].as_str().filter(|s| !s.is_empty()).ok_or_else(|| format!("`{key}` is required"))
}

fn describe(e: Error) -> String {
    match e {
        Error::Connect { .. } => format!("{e}. Is tonycd running? (`tonyc daemon start`)"),
        Error::Api { field: Some(field), message, .. } => format!("refused: {message} (field {field})"),
        other => other.to_string(),
    }
}

fn parse_word<T: std::str::FromStr>(args: &Value, key: &str) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    args[key].as_str().map(|s| s.parse::<T>().map_err(|e| format!("{key}: {e}"))).transpose()
}

/// Builds work from tool arguments.
fn work(args: &Value) -> Result<Work, String> {
    let mut w = Work::new(required(args, "objective")?);
    if let Some(t) = args["input"].as_str() {
        w = w.input(t);
    }
    for f in args["files"].as_array().into_iter().flatten() {
        w = w.file(f.as_str().ok_or("files: expected strings")?);
    }
    if let Some(u) = parse_word::<Urgency>(args, "urgency")? {
        w = w.urgency(u);
    }
    if let Some(d) = args["deadline"].as_str() {
        w = w.deadline_in(d);
    }
    if let Some(l) = parse_word::<Locality>(args, "locality")? {
        // Remote inference is never something an agent can ask for through
        // this adapter; an operator grants it per request, deliberately.
        if matches!(l, Locality::RemoteAllowlist | Locality::Unrestricted) {
            return Err(format!("locality {l} cannot be requested through MCP"));
        }
        w = w.locality(l);
    }
    for n in args["ssh_nodes"].as_array().into_iter().flatten() {
        w = w.ssh_node(n.as_str().ok_or("ssh_nodes: expected strings")?);
    }
    if let Some(f) = args["quality_floor"].as_f64() {
        w = w.floor(f);
    }
    if let Some(s) = args["scope"].as_str() {
        w = w.scope(s);
    }
    if let Some(c) = args["capability"].as_str() {
        w = w.capability(c);
    }
    if let Some(n) = args["max_tokens"].as_u64() {
        w = w.max_tokens(n.min(u32::MAX as u64) as u32);
    }
    if args["json"].as_bool() == Some(true) {
        w = w.json();
    }
    Ok(w)
}

/// `90_000` → `1m 30s`; `7_200_000` → `2h`.
fn span(ms: i64) -> String {
    let s = ms.max(0) / 1000;
    let (d, h, m, sec) = (s / 86_400, s / 3600 % 24, s / 60 % 60, s % 60);
    let parts: Vec<String> = [(d, "d"), (h, "h"), (m, "m"), (sec, "s")]
        .iter()
        .filter(|(v, _)| *v > 0)
        .map(|(v, u)| format!("{v}{u}"))
        .collect();
    if parts.is_empty() { "0s".into() } else { parts.join(" ") }
}

fn job_view(job: &Job) -> (String, Value) {
    let state = job.state();
    // `output()` is only Some when there is a result, so both match here.
    let text = match (job.output().zip(job.result.as_ref()), job.failure_reason()) {
        (Some((out, r)), _) => {
            let source = match &r.reused_from {
                Some(o) => format!("reused from object {o}"),
                None => format!("{} on {}", r.model, r.exposure.label()),
            };
            format!("Job {} is READY ({source}):\n\n{out}", job.id())
        }
        (None, Some(reason)) => format!("Job {} is {state}: {reason}", job.id()),
        (None, None) if job.is_terminal() => format!("Job {} is {state}.", job.id()),
        (None, None) => format!("Job {} is {state}; not finished yet. Call tonyc_await again later.", job.id()),
    };
    let structured = json!({
        "job_id": job.id(),
        "state": state,
        "terminal": job.is_terminal(),
        "output": job.output(),
        "model": job.result.as_ref().map(|r| &r.model),
        "failure": job.failure_reason(),
    });
    (text, structured)
}

const INSTRUCTIONS: &str = "Tonyc produces cognition when, where and as cheaply as policy allows, and reuses \
what it has produced before. Before inferring something expensive that may have been asked before, call \
tonyc_try. For work you do not need in this turn (background analysis, summaries for later, checks with a \
deadline), call tonyc_submit with a deadline instead of doing it now, and collect it with tonyc_await. Data \
never leaves the boundary the daemon's policy allows, whatever locality you ask for.";

fn tools() -> Value {
    let work_props = json!({
        "objective": {"type": "string", "description": "What to produce, as an instruction."},
        "input": {"type": "string", "description": "Text the objective is about."},
        "files": {"type": "array", "items": {"type": "string"}, "description": "Absolute paths of files to read. Answers are invalidated when the files change."},
        "scope": {"type": "string", "description": "Cognition scope, e.g. project:<name> (default user)."},
        "capability": {"type": "string", "description": "Capability needed, e.g. general, code.rust."},
    });
    let mut submit_props = work_props.clone();
    let extra = json!({
        "urgency": {"type": "string", "enum": ["NOW", "SOON", "DEFERRED", "IDLE"], "description": "How soon it is needed (default DEFERRED)."},
        "deadline": {"type": "string", "description": "When it stops being useful, relative: 45m, 2h, 1d."},
        "locality": {"type": "string", "enum": ["LOCAL_ONLY", "LAN_ALLOWED", "SSH_ALLOWLIST"], "description": "Trust boundary; the daemon's ceiling still applies (default LOCAL_ONLY)."},
        "ssh_nodes": {"type": "array", "items": {"type": "string"}, "description": "SSH nodes allowed under SSH_ALLOWLIST."},
        "quality_floor": {"type": "number", "minimum": 0, "maximum": 1, "description": "Lowest acceptable estimated quality."},
        "max_tokens": {"type": "integer", "minimum": 1},
        "json": {"type": "boolean", "description": "Answer must be a JSON object."},
        "idempotency_key": {"type": "string", "description": "Retrying with the same key returns the same job."},
    });
    for (k, v) in extra.as_object().into_iter().flatten() {
        submit_props[k] = v.clone();
    }
    let job_id = json!({"job_id": {"type": "string", "description": "A job id (tny_...) or a unique prefix of one."}});
    json!([
        {
            "name": "tonyc_try",
            "description": "Return existing cognition for a request immediately, or say there is none. Never runs a model; a miss is remembered, so questions asked repeatedly get prepared ahead of time.",
            "inputSchema": {"type": "object", "properties": work_props, "required": ["objective"]},
            "annotations": {"readOnlyHint": true},
        },
        {
            "name": "tonyc_submit",
            "description": "Hand off work you do not need right now. Tonyc runs it when and where it is cheapest within the deadline and the privacy boundary, or reuses an existing answer. Returns a job id.",
            "inputSchema": {"type": "object", "properties": submit_props, "required": ["objective"]},
        },
        {
            "name": "tonyc_await",
            "description": "Wait for a submitted job (up to timeout_seconds, default 30, max 600) and return its answer, or its state if it is not finished.",
            "inputSchema": {"type": "object", "properties": {
                "job_id": job_id["job_id"],
                "timeout_seconds": {"type": "integer", "minimum": 0, "maximum": 600},
            }, "required": ["job_id"]},
            "annotations": {"readOnlyHint": true},
        },
        {
            "name": "tonyc_status",
            "description": "A job's state, and its answer if it is ready. Does not wait.",
            "inputSchema": {"type": "object", "properties": job_id, "required": ["job_id"]},
            "annotations": {"readOnlyHint": true},
        },
        {
            "name": "tonyc_query",
            "description": "Search existing cognition (fresh answers Tonyc has produced) by relevance.",
            "inputSchema": {"type": "object", "properties": {
                "text": {"type": "string"},
                "scope": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            }, "required": ["text"]},
            "annotations": {"readOnlyHint": true},
        },
        {
            "name": "tonyc_cancel",
            "description": "Cancel a job that is no longer needed.",
            "inputSchema": {"type": "object", "properties": job_id, "required": ["job_id"]},
            "annotations": {"destructiveHint": true},
        },
        {
            "name": "tonyc_invalidate",
            "description": "Mark cognition stale (and everything derived from it) because what it was based on changed. Give exactly one of object, path, scope or event.",
            "inputSchema": {"type": "object", "properties": {
                "object": {"type": "string"},
                "path": {"type": "string", "description": "Invalidate answers derived from files under this path."},
                "scope": {"type": "string"},
                "event": {"type": "string"},
                "reason": {"type": "string"},
            }},
            "annotations": {"destructiveHint": true},
        },
    ])
}

#[cfg(test)]
mod tests {
    #[test]
    fn spans_read_naturally() {
        assert_eq!(super::span(90_000), "1m 30s");
        assert_eq!(super::span(7_200_000), "2h");
        assert_eq!(super::span(90_061_000), "1d 1h 1m 1s");
        assert_eq!(super::span(-5), "0s");
    }
}
