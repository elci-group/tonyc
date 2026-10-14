//! Command adapter: a local program as a runtime (directive §11: llama.cpp,
//! ONNX Runtime, Candle "and other compatible local engines").
//!
//! Tonyc starts the program once per request, writes the request to its
//! stdin and reads the answer from its stdout. With `protocol = "text"` the
//! request is the bare prompt and the answer is the whole of stdout. With
//! `protocol = "json"` both are one `tonyc.runtime/1` object, so a wrapper
//! in any language can report token counts and timings too.
//!
//! The prompt never appears on the command line, the environment is cleared
//! except for `PATH`, `HOME` and locale variables, and the process is killed
//! if the job is cancelled or times out.

use super::{AdapterError, BoxFuture, ModelAdapter};
use crate::config::{CommandConfig, CommandProtocol};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::sync::RwLock;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tonyc_core::model::{Economics, ModelResources, Performance, ProfileSource, Timings, estimate_tokens};
use tonyc_core::{Exposure, InferenceOutput, InferenceRequest, ModelProfile};

pub const RUNTIME_PROTOCOL: &str = "tonyc.runtime/1";

/// Most a program may write before Tonyc stops reading.
const MAX_OUTPUT: u64 = 16 * 1024 * 1024;

pub struct Command {
    cfg: CommandConfig,
    exposure: Exposure,
    state: RwLock<Option<String>>,
}

#[derive(Serialize)]
struct JsonRequest<'a> {
    protocol: &'a str,
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    json: bool,
}

#[derive(Deserialize)]
struct JsonResponse {
    text: String,
    #[serde(default)]
    tokens_in: Option<u64>,
    #[serde(default)]
    tokens_out: Option<u64>,
    #[serde(default)]
    timings: Option<Timings>,
    #[serde(default)]
    error: Option<String>,
}

impl Command {
    /// The configuration must have been validated.
    pub fn new(cfg: CommandConfig) -> Command {
        let exposure = cfg.declared_exposure().unwrap_or(Exposure::Remote { host: cfg.name.clone() });
        Command { cfg, exposure, state: RwLock::new(Some("not yet checked".into())) }
    }

    fn argv(&self, model: &str) -> Vec<String> {
        self.cfg.argv.iter().map(|a| a.replace("{model}", model)).collect()
    }

    async fn invoke(&self, req: &InferenceRequest) -> Result<(Vec<u8>, i64), AdapterError> {
        let argv = self.argv(&req.model);
        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for var in ["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"] {
            if let Some(v) = std::env::var_os(var) {
                cmd.env(var, v);
            }
        }
        let started = std::time::Instant::now();
        let mut child = cmd.spawn().map_err(|e| {
            *self.state.write().unwrap() = Some(format!("{}: {e}", argv[0]));
            AdapterError::Unreachable(format!("starting {}: {e}", argv[0]))
        })?;
        let input = self.encode(req);
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let mut stdout = child.stdout.take().expect("stdout is piped").take(MAX_OUTPUT);
        let mut stderr = child.stderr.take().expect("stderr is piped").take(64 * 1024);
        let work = async {
            // Write and read concurrently, so a program that answers before
            // reading all its input cannot deadlock against us.
            let write = async {
                let r = stdin.write_all(&input).await;
                drop(stdin);
                r
            };
            let mut out = Vec::new();
            let mut err = Vec::new();
            let (w, o, e) = tokio::join!(write, stdout.read_to_end(&mut out), stderr.read_to_end(&mut err));
            // A program may legitimately exit without reading its input.
            if let Err(e) = w
                && e.kind() != std::io::ErrorKind::BrokenPipe
            {
                return Err(format!("writing the request: {e}"));
            }
            o.map_err(|e| format!("reading the answer: {e}"))?;
            let _ = e;
            let status = child.wait().await.map_err(|e| e.to_string())?;
            Ok((status, out, err))
        };
        let timeout = Duration::from_secs(self.cfg.timeout_secs.max(1));
        let (status, out, err) = tokio::time::timeout(timeout, work)
            .await
            .map_err(|_| AdapterError::Failed(format!("{} timed out after {}s", self.cfg.name, timeout.as_secs())))?
            .map_err(AdapterError::Failed)?;
        if !status.success() {
            let tail = String::from_utf8_lossy(&err);
            let tail: String = tail.trim().chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
            return Err(AdapterError::Failed(format!("{} exited with {status}: {tail}", self.cfg.name)));
        }
        Ok((out, started.elapsed().as_millis() as i64))
    }
}

/// Whether `program` names an executable file, directly or on `PATH`.
fn executable(program: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let is_exec = |p: &std::path::Path| p.metadata().is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0);
    if program.contains('/') {
        return is_exec(std::path::Path::new(program));
    }
    std::env::var_os("PATH").is_some_and(|path| std::env::split_paths(&path).any(|d| is_exec(&d.join(program))))
}

impl ModelAdapter for Command {
    fn runtime(&self) -> &str {
        &self.cfg.name
    }

    fn exposure(&self) -> Exposure {
        self.exposure.clone()
    }

    fn healthy(&self) -> bool {
        self.state.read().unwrap().is_none()
    }

    fn models(&self) -> Vec<ModelProfile> {
        self.cfg
            .models
            .iter()
            .map(|m| {
                let mut capabilities = m.capabilities.clone();
                capabilities.entry("general".into()).or_insert(0.5);
                let tps = m.tokens_per_sec.max(0.01);
                ModelProfile {
                    id: format!("{}/{}", self.cfg.name, m.name),
                    runtime: self.cfg.name.clone(),
                    digest: None,
                    capabilities,
                    performance: Performance {
                        tokens_per_sec: tps,
                        prefill_tokens_per_sec: tps * 8.0,
                        startup_ms: m.startup_ms,
                        // A fresh process per request: never warm.
                        warm_startup_ms: m.startup_ms,
                    },
                    resources: ModelResources { ram_mb: m.ram_mb, vram_mb: 0 },
                    economics: Economics { cost_per_mtok_gbp: 0.0, energy_wh_per_ktok: 35.0 * (1000.0 / tps) / 3600.0 },
                    source: ProfileSource::Configured,
                    loaded: false,
                    evidence: Default::default(),
                }
            })
            .collect()
    }

    fn refresh(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            let program = &self.cfg.argv[0];
            *self.state.write().unwrap() =
                if executable(program) { None } else { Some(format!("{program} is not an executable file")) };
        })
    }

    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        Box::pin(async move {
            if !self.cfg.models.iter().any(|m| m.name == req.model) {
                return Err(AdapterError::Failed(format!("{} has no model `{}`", self.cfg.name, req.model)));
            }
            let (out, wall_ms) = self.invoke(req).await?;
            let bytes_received = out.len() as u64;
            let bytes_sent = self.encode(req).len() as u64;
            let (text, tokens_in, tokens_out, timings) = match self.cfg.protocol {
                CommandProtocol::Text => {
                    let text = String::from_utf8_lossy(&out).trim_end().to_string();
                    (text, None, None, None)
                }
                CommandProtocol::Json => {
                    let r: JsonResponse = serde_json::from_slice(&out).map_err(|e| {
                        AdapterError::Failed(format!(
                            "{} did not answer with a {RUNTIME_PROTOCOL} object: {e}",
                            self.cfg.name
                        ))
                    })?;
                    if let Some(e) = r.error {
                        return Err(AdapterError::Failed(format!("{}: {e}", self.cfg.name)));
                    }
                    (r.text, r.tokens_in, r.tokens_out, r.timings)
                }
            };
            let tokens_out = tokens_out.unwrap_or_else(|| estimate_tokens(text.len()));
            Ok(InferenceOutput {
                tokens_in: tokens_in.unwrap_or_else(|| estimate_tokens(req.prompt.len())),
                tokens_out,
                model_digest: None,
                bytes_sent,
                bytes_received,
                timings: timings.or_else(|| {
                    (tokens_out > 0).then_some(Timings { load_ms: 0, prefill_ms: 0, generate_ms: wall_ms })
                }),
                text,
            })
        })
    }

    fn encode(&self, req: &InferenceRequest) -> Vec<u8> {
        match self.cfg.protocol {
            CommandProtocol::Text => req.prompt.as_bytes().to_vec(),
            CommandProtocol::Json => serde_json::to_vec(&JsonRequest {
                protocol: RUNTIME_PROTOCOL,
                model: &req.model,
                prompt: &req.prompt,
                max_tokens: req.max_tokens,
                json: req.json,
            })
            .expect("a request serialises"),
        }
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "runtime": self.cfg.name,
            "kind": "command",
            "program": self.cfg.argv[0],
            "protocol": self.cfg.protocol,
            "healthy": self.healthy(),
            "exposure": self.exposure,
            "models": self.cfg.models.len(),
            "last_error": *self.state.read().unwrap(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CommandModel;

    fn cmd(argv: &[&str], protocol: CommandProtocol) -> Command {
        Command::new(CommandConfig {
            name: "t".into(),
            argv: argv.iter().map(|s| s.to_string()).collect(),
            protocol,
            models: vec![CommandModel { name: "m".into(), ..Default::default() }],
            timeout_secs: 5,
            ..Default::default()
        })
    }

    fn req(prompt: &str) -> InferenceRequest {
        InferenceRequest { model: "m".into(), prompt: prompt.into(), max_tokens: 50, json: false }
    }

    #[tokio::test]
    async fn text_protocol_pipes_the_prompt() {
        let c = cmd(&["tr", "a-z", "A-Z"], CommandProtocol::Text);
        c.refresh().await;
        assert!(c.healthy());
        assert_eq!(c.exposure(), Exposure::Loopback { endpoint: "exec:t".into() });
        let out = c.run(&req("hello world")).await.unwrap();
        assert_eq!(out.text, "HELLO WORLD");
        assert_eq!(out.bytes_sent, 11);
    }

    #[tokio::test]
    async fn json_protocol_reports_tokens() {
        let script = r#"read -r line; case "$line" in *'"protocol":"tonyc.runtime/1"'*) echo '{"text":"ok","tokens_in":3,"tokens_out":1}';; *) exit 3;; esac"#;
        let c = cmd(&["sh", "-c", script], CommandProtocol::Json);
        let out = c.run(&req("hi")).await.unwrap();
        assert_eq!((out.text.as_str(), out.tokens_in, out.tokens_out), ("ok", 3, 1));
        let err = cmd(&["sh", "-c", r#"cat >/dev/null; echo '{"text":"","error":"no gpu"}'"#], CommandProtocol::Json)
            .run(&req("hi"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no gpu"), "{err}");
    }

    #[tokio::test]
    async fn failures_and_timeouts_are_reported() {
        let err = cmd(&["sh", "-c", "echo boom >&2; exit 4"], CommandProtocol::Text).run(&req("x")).await.unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
        let mut slow = cmd(&["sleep", "5"], CommandProtocol::Text);
        slow.cfg.timeout_secs = 1;
        let err = slow.run(&req("x")).await.unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
        let missing = cmd(&["/nonexistent/engine"], CommandProtocol::Text);
        missing.refresh().await;
        assert!(!missing.healthy());
    }

    #[tokio::test]
    async fn the_prompt_is_not_on_the_command_line_and_the_environment_is_clean() {
        // SAFETY: tests in this module do not read this variable concurrently.
        unsafe { std::env::set_var("TONYC_TEST_SECRET", "s3cret") };
        let c = cmd(&["sh", "-c", "cat >/dev/null; echo \"$0 $TONYC_TEST_SECRET {model}\""], CommandProtocol::Text);
        let out = c.run(&req("private prompt")).await.unwrap();
        assert_eq!(out.text, "sh  m");
    }

    #[test]
    fn declared_exposure_is_parsed() {
        let mut cfg = CommandConfig { name: "x".into(), ..Default::default() };
        assert!(!cfg.declared_exposure().unwrap().leaves_host());
        cfg.exposure = "remote:api.example.com".into();
        assert_eq!(cfg.declared_exposure().unwrap(), Exposure::Remote { host: "api.example.com".into() });
        cfg.exposure = "elsewhere".into();
        assert!(cfg.declared_exposure().is_err());
    }
}
