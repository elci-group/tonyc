//! The SSH compute fabric (directive §8, Phase 3).
//!
//! Authorised machines become execution domains through the system OpenSSH
//! client — Tonyc invents no transport or authentication of its own. Every
//! connection runs non-interactively against Tonyc's own pinned
//! `known_hosts` with strict host-key checking, and with agent, X11 and port
//! forwarding disabled, so a node gets the payload on stdin and nothing that
//! reaches back into this host. Nodes run inference with the runtime already
//! on them (Ollama over its local HTTP port, called with `curl`); nothing is
//! installed remotely.

use crate::config::FabricConfig;
use crate::runtimes::ollama::{Generated, TagModel, Tags, generate_body, prior_profile};
use crate::runtimes::{AdapterError, BoxFuture, ModelAdapter};
use crate::store::{Store, json_of, now_ms};
use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tonyc_core::{Exposure, InferenceOutput, InferenceRequest, Millis, ModelProfile};

/// A registered node, as persisted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    pub name: String,
    /// What `ssh` is given: `user@host`, `host` or an alias from
    /// `~/.ssh/config`.
    pub target: String,
    pub port: Option<u16>,
    pub identity: Option<String>,
    /// The inference runtime's URL as seen *on the node*.
    pub runtime_url: String,
    /// Shared nodes are used only by jobs that name them.
    pub shared: bool,
    /// Concurrent jobs this node accepts.
    pub slots: usize,
    /// The pinned `known_hosts` lines for this node.
    pub host_keys: Vec<String>,
    pub added_at: Millis,
}

/// What a probe learned about a node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeCaps {
    pub arch: String,
    pub cores: usize,
    pub memory_mb: u64,
    pub load_1m: f64,
    pub gpus: Vec<NodeGpu>,
    pub curl: bool,
    pub runtime_ok: bool,
    pub runtime_error: Option<String>,
    pub models: Vec<String>,
    pub probed_at: Millis,
    pub probe_ms: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeGpu {
    pub name: String,
    pub vram_mb: Option<u64>,
}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 32
        && name.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Only plain `http://host[:port]` URLs: they are placed in a remote shell
/// command, so anything else is refused rather than escaped.
pub fn valid_runtime_url(url: &str) -> bool {
    url.strip_prefix("http://").is_some_and(|rest| {
        let rest = rest.trim_end_matches('/');
        !rest.is_empty() && rest.chars().all(|c| c.is_ascii_alphanumeric() || ".-:[]".contains(c))
    })
}

fn valid_target(target: &str) -> bool {
    !target.is_empty()
        && !target.starts_with('-')
        && target.chars().all(|c| c.is_ascii_alphanumeric() || "@.-_:[]".contains(c))
}

/// Single-quotes a string for a POSIX shell.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

pub struct SshOutput {
    pub code: i32,
    pub stdout: Vec<u8>,
    pub stderr: String,
}

/// The OpenSSH client, configured for Tonyc.
pub struct Ssh {
    pub cfg: FabricConfig,
    pub known_hosts: PathBuf,
}

impl Ssh {
    fn args(&self, node: &NodeSpec) -> Vec<String> {
        let mut a: Vec<String> = vec!["-T".into()];
        for o in [
            "BatchMode=yes".to_string(),
            format!("ConnectTimeout={}", self.cfg.connect_timeout_secs.max(1)),
            "ServerAliveInterval=15".into(),
            "ServerAliveCountMax=3".into(),
            "StrictHostKeyChecking=yes".into(),
            format!("UserKnownHostsFile={}", self.known_hosts.display()),
            "GlobalKnownHostsFile=/dev/null".into(),
            "UpdateHostKeys=no".into(),
            "ForwardAgent=no".into(),
            "ForwardX11=no".into(),
            "ClearAllForwardings=yes".into(),
            "PermitLocalCommand=no".into(),
            "RequestTTY=no".into(),
        ] {
            a.push("-o".into());
            a.push(o);
        }
        if let Some(p) = node.port {
            a.push("-p".into());
            a.push(p.to_string());
        }
        if let Some(i) = &node.identity {
            a.extend(["-i".into(), i.clone(), "-o".into(), "IdentitiesOnly=yes".into()]);
        }
        a.push("--".into());
        a.push(node.target.clone());
        a
    }

    /// Runs `remote` on the node with `stdin` as its input.
    pub async fn run(
        &self,
        node: &NodeSpec,
        remote: &str,
        stdin: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<SshOutput, String> {
        let mut cmd = Command::new(&self.cfg.ssh_program);
        cmd.args(self.args(node))
            .arg(remote)
            .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| format!("starting {}: {e}", self.cfg.ssh_program))?;
        if let (Some(data), Some(mut pipe)) = (stdin, child.stdin.take()) {
            let data = data.to_vec();
            tokio::spawn(async move {
                let _ = pipe.write_all(&data).await;
                let _ = pipe.shutdown().await;
            });
        }
        let out = tokio::time::timeout(timeout, child.wait_with_output())
            .await
            .map_err(|_| format!("timed out after {}s", timeout.as_secs()))?
            .map_err(|e| e.to_string())?;
        Ok(SshOutput {
            code: out.status.code().unwrap_or(-1),
            stdout: out.stdout,
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        })
    }

    /// `ssh -G`: the hostname and port the target really resolves to under
    /// the user's ssh configuration.
    pub async fn resolve(&self, target: &str, port: Option<u16>) -> Result<(String, u16)> {
        let mut cmd = Command::new(&self.cfg.ssh_program);
        cmd.arg("-G");
        if let Some(p) = port {
            cmd.args(["-p", &p.to_string()]);
        }
        let out = cmd.args(["--", target]).output().await.context("running ssh -G")?;
        if !out.status.success() {
            bail!("ssh -G {target}: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let field = |k: &str| text.lines().find_map(|l| l.strip_prefix(k).map(|v| v.trim().to_string()));
        let host = field("hostname ").ok_or_else(|| anyhow!("ssh -G gave no hostname for {target}"))?;
        let port = field("port ").and_then(|p| p.parse().ok()).unwrap_or(22);
        Ok((host, port))
    }

    /// The host's public keys as `known_hosts` lines (not yet trusted).
    pub async fn keyscan(&self, host: &str, port: u16) -> Result<Vec<String>> {
        let out = Command::new(&self.cfg.keyscan_program)
            .args(["-T", &self.cfg.connect_timeout_secs.max(1).to_string(), "-p", &port.to_string(), "--", host])
            .output()
            .await
            .context("running ssh-keyscan")?;
        let lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect();
        if lines.is_empty() {
            bail!("no host keys from {host}:{port} ({})", String::from_utf8_lossy(&out.stderr).trim());
        }
        Ok(lines)
    }

    /// `SHA256:...` fingerprint of a `known_hosts` line.
    pub async fn fingerprint(&self, line: &str) -> Result<String> {
        let mut child = Command::new(&self.cfg.keygen_program)
            .args(["-l", "-f", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("running ssh-keygen")?;
        let mut pipe = child.stdin.take().expect("piped");
        pipe.write_all(line.as_bytes()).await?;
        pipe.write_all(b"\n").await?;
        drop(pipe);
        let out = child.wait_with_output().await?;
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .nth(1)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("ssh-keygen could not fingerprint a host key"))
    }

    /// Host keys the user already trusts for this host.
    pub async fn already_trusted(&self, host: &str, port: u16) -> Result<Vec<String>> {
        let file = self.cfg.user_known_hosts();
        if !file.exists() {
            return Ok(vec![]);
        }
        let name = if port == 22 { host.to_string() } else { format!("[{host}]:{port}") };
        let out = Command::new(&self.cfg.keygen_program)
            .args(["-F", &name, "-f"])
            .arg(&file)
            .output()
            .await
            .context("running ssh-keygen -F")?;
        Ok(String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect())
    }
}

/// Decides which host keys to pin for a new node. An explicit fingerprint
/// must match a key the host presents; otherwise the host must already be
/// in the user's `known_hosts`. Nothing is trusted on first use.
pub async fn pin_host_keys(
    ssh: &Ssh,
    target: &str,
    port: Option<u16>,
    fingerprint: Option<&str>,
) -> Result<Vec<String>> {
    let (host, port) = ssh.resolve(target, port).await?;
    if fingerprint.is_none() {
        let trusted = ssh.already_trusted(&host, port).await?;
        if !trusted.is_empty() {
            return Ok(trusted);
        }
    }
    let scanned = ssh.keyscan(&host, port).await?;
    let mut prints = Vec::new();
    for line in &scanned {
        prints.push((ssh.fingerprint(line).await?, line.clone()));
    }
    match fingerprint {
        Some(want) => {
            let pinned: Vec<String> = prints.iter().filter(|(fp, _)| fp == want).map(|(_, l)| l.clone()).collect();
            if pinned.is_empty() {
                bail!(
                    "no host key of {host}:{port} matches {want}; it presents: {}",
                    prints.iter().map(|(fp, _)| fp.as_str()).collect::<Vec<_>>().join(", ")
                );
            }
            Ok(pinned)
        }
        None => bail!(
            "the host key of {host}:{port} is not trusted yet. Verify one of these fingerprints out of band, \
             then add the node again with --host-key <fingerprint>: {}",
            prints.iter().map(|(fp, _)| fp.as_str()).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// The remote probe: one `sh` script, printing `key=value` lines and then
/// the runtime's model list.
fn probe_script(runtime_url: &str) -> String {
    format!(
        r#"echo "arch=$(uname -m)"
echo "cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
echo "mem_kb=$(awk '/MemTotal/ {{print $2}}' /proc/meminfo 2>/dev/null)"
echo "load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)"
if command -v curl >/dev/null 2>&1; then echo curl=yes; else echo curl=no; fi
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | sed 's/^/gpu=/'; fi
echo ---tags---
curl -sS -m 5 {} 2>&1
echo
echo ---ps---
curl -sS -m 5 {} 2>/dev/null"#,
        sh_quote(&format!("{}/api/tags", runtime_url.trim_end_matches('/'))),
        sh_quote(&format!("{}/api/ps", runtime_url.trim_end_matches('/')))
    )
}

fn parse_probe(text: &str) -> (NodeCaps, Option<Vec<TagModel>>, Vec<String>) {
    let (head, rest) = text.split_once("---tags---").unwrap_or((text, ""));
    let (tags, ps) = rest.split_once("---ps---").unwrap_or((rest, ""));
    let loaded = crate::runtimes::parse_loaded(ps.trim().as_bytes());
    let mut caps = NodeCaps::default();
    for line in head.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        let v = v.trim();
        match k {
            "arch" => caps.arch = v.into(),
            "cores" => caps.cores = v.parse().unwrap_or(0),
            "mem_kb" => caps.memory_mb = v.parse::<u64>().unwrap_or(0) / 1024,
            "load" => caps.load_1m = v.parse().unwrap_or(0.0),
            "curl" => caps.curl = v == "yes",
            "gpu" => {
                let mut parts = v.split(',').map(str::trim);
                let name = parts.next().unwrap_or("").to_string();
                caps.gpus.push(NodeGpu { name, vram_mb: parts.next().and_then(|m| m.parse().ok()) });
            }
            _ => {}
        }
    }
    let tags = tags.trim();
    match serde_json::from_str::<Tags>(tags) {
        Ok(t) => {
            caps.runtime_ok = true;
            caps.models = t.models.iter().map(|m| m.name.clone()).collect();
            (caps, Some(t.models), loaded)
        }
        Err(_) => {
            caps.runtime_error =
                Some(if tags.is_empty() { "no response".into() } else { tags.chars().take(200).collect() });
            (caps, None, loaded)
        }
    }
}

struct NodeState {
    healthy: bool,
    caps: Option<NodeCaps>,
    models: Vec<ModelProfile>,
    last_error: Option<String>,
}

/// An SSH node as an execution domain.
pub struct NodeAdapter {
    pub spec: NodeSpec,
    runtime: String,
    ssh: Arc<Ssh>,
    state: RwLock<NodeState>,
    warmth: crate::runtimes::Warmth,
}

impl NodeAdapter {
    pub fn new(spec: NodeSpec, ssh: Arc<Ssh>) -> NodeAdapter {
        NodeAdapter {
            runtime: format!("node.{}", spec.name),
            spec,
            ssh,
            state: RwLock::new(NodeState {
                healthy: false,
                caps: None,
                models: vec![],
                last_error: Some("not yet probed".into()),
            }),
            warmth: crate::runtimes::Warmth::default(),
        }
    }

    pub fn caps(&self) -> Option<NodeCaps> {
        self.state.read().unwrap().caps.clone()
    }

    pub fn last_error(&self) -> Option<String> {
        self.state.read().unwrap().last_error.clone()
    }

    fn mark_down(&self, why: &str) {
        let mut st = self.state.write().unwrap();
        st.healthy = false;
        st.last_error = Some(why.to_string());
    }

    /// Probes the node: reachability, hardware, runtime and models.
    pub async fn probe(&self) -> Result<NodeCaps, String> {
        let started = now_ms();
        let timeout = Duration::from_secs(self.ssh.cfg.connect_timeout_secs.max(1) + 15);
        let out = self.ssh.run(&self.spec, &probe_script(&self.spec.runtime_url), None, timeout).await?;
        if out.code == 255 {
            return Err(format!("ssh: {}", out.stderr));
        }
        let (mut caps, tags, loaded) = parse_probe(&String::from_utf8_lossy(&out.stdout));
        self.warmth.reported(&loaded, Duration::from_secs(self.ssh.cfg.probe_interval_secs.clamp(5, 3600) + 30));
        caps.probed_at = now_ms();
        caps.probe_ms = caps.probed_at - started;
        let gpu_vram = caps.gpus.iter().filter_map(|g| g.vram_mb).max().unwrap_or(0);
        let models: Vec<ModelProfile> = tags
            .unwrap_or_default()
            .into_iter()
            .map(|m| {
                // A GPU that holds the whole model generates far faster.
                let fits = gpu_vram > 0 && m.size / (1024 * 1024) < gpu_vram;
                prior_profile(&self.runtime, m, if fits { 4.0 } else { 1.0 })
            })
            .collect();
        let mut st = self.state.write().unwrap();
        st.caps = Some(caps.clone());
        st.models = models;
        if !caps.curl {
            st.healthy = false;
            st.last_error = Some("curl is not installed on the node".into());
        } else if !caps.runtime_ok {
            st.healthy = false;
            st.last_error = Some(format!(
                "runtime at {} on the node: {}",
                self.spec.runtime_url,
                caps.runtime_error.clone().unwrap_or_default()
            ));
        } else {
            st.healthy = true;
            st.last_error = None;
        }
        Ok(caps)
    }
}

impl ModelAdapter for NodeAdapter {
    fn runtime(&self) -> &str {
        &self.runtime
    }

    fn domain(&self) -> String {
        format!("node:{}", self.spec.name)
    }

    fn exposure(&self) -> Exposure {
        Exposure::Ssh { node: self.spec.name.clone(), shared: self.spec.shared }
    }

    fn healthy(&self) -> bool {
        self.state.read().unwrap().healthy
    }

    fn models(&self) -> Vec<ModelProfile> {
        let prefix = format!("{}/", self.runtime);
        let mut models = self.state.read().unwrap().models.clone();
        for m in &mut models {
            m.loaded = self.warmth.is_warm(m.id.trim_start_matches(&prefix));
        }
        models
    }

    fn refresh(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            if let Err(e) = self.probe().await {
                self.mark_down(&e);
            }
        })
    }

    fn run<'a>(&'a self, req: &'a InferenceRequest) -> BoxFuture<'a, Result<InferenceOutput, AdapterError>> {
        Box::pin(async move {
            let payload = self.encode(req);
            let secs = self.ssh.cfg.request_timeout_secs.max(1);
            let remote = format!(
                "curl -sS -m {secs} -H 'Content-Type: application/json' --data-binary @- -w '\\n%{{http_code}}' {}",
                sh_quote(&format!("{}/api/generate", self.spec.runtime_url.trim_end_matches('/')))
            );
            let out = self.ssh.run(&self.spec, &remote, Some(&payload), Duration::from_secs(secs + 30)).await.map_err(
                |e| {
                    self.mark_down(&e);
                    AdapterError::NodeUnreachable(e)
                },
            )?;
            match out.code {
                0 => {}
                255 => {
                    let why = format!("ssh: {}", out.stderr);
                    self.mark_down(&why);
                    return Err(AdapterError::NodeUnreachable(why));
                }
                // curl: 6/7 cannot resolve/connect, 28 timeout, 127 missing.
                6 | 7 | 28 | 127 => {
                    let why = format!("runtime on node {} (curl exit {}): {}", self.spec.name, out.code, out.stderr);
                    self.mark_down(&why);
                    return Err(AdapterError::NodeUnreachable(why));
                }
                code => return Err(AdapterError::Failed(format!("remote command exited {code}: {}", out.stderr))),
            }
            let text = String::from_utf8_lossy(&out.stdout);
            let (body_text, status) = text.rsplit_once('\n').unwrap_or(("", text.as_ref()));
            if status.trim() != "200" {
                let msg: String = body_text.chars().take(300).collect();
                return Err(AdapterError::Failed(format!("runtime on node returned {}: {msg}", status.trim())));
            }
            let g: Generated =
                serde_json::from_str(body_text).map_err(|e| AdapterError::Failed(format!("decoding reply: {e}")))?;
            let digest = self
                .models()
                .into_iter()
                .find(|m| m.id == format!("{}/{}", self.runtime, req.model))
                .and_then(|m| m.digest);
            self.warmth.ran(&req.model);
            Ok(InferenceOutput {
                timings: g.timings(),
                text: g.response,
                tokens_in: g.prompt_eval_count,
                tokens_out: g.eval_count,
                model_digest: digest,
                bytes_sent: payload.len() as u64,
                bytes_received: out.stdout.len() as u64,
            })
        })
    }

    fn encode(&self, req: &InferenceRequest) -> Vec<u8> {
        generate_body(req)
    }

    fn describe(&self) -> serde_json::Value {
        let st = self.state.read().unwrap();
        serde_json::json!({
            "runtime": self.runtime,
            "node": self.spec.name,
            "target": self.spec.target,
            "healthy": st.healthy,
            "exposure": self.exposure(),
            "models": st.models.len(),
            "last_error": st.last_error,
        })
    }
}

// ------------------------------------------------------------- storage ---

impl Store {
    pub fn put_node(&self, n: &NodeSpec) -> Result<()> {
        self.conn().execute(
            "INSERT INTO nodes (name, spec, added_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET spec = excluded.spec",
            params![n.name, serde_json::to_string(n)?, n.added_at],
        )?;
        Ok(())
    }

    pub fn nodes(&self) -> Result<Vec<NodeSpec>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT spec FROM nodes ORDER BY name")?;
        Ok(stmt.query_map([], |r| json_of(r.get(0)?))?.collect::<rusqlite::Result<_>>()?)
    }

    pub fn node(&self, name: &str) -> Result<Option<NodeSpec>> {
        let conn = self.conn();
        Ok(conn.query_row("SELECT spec FROM nodes WHERE name = ?1", [name], |r| json_of(r.get(0)?)).optional()?)
    }

    pub fn remove_node(&self, name: &str) -> Result<bool> {
        Ok(self.conn().execute("DELETE FROM nodes WHERE name = ?1", [name])? > 0)
    }

    pub fn setting(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn();
        Ok(conn.query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0)).optional()?)
    }

    pub fn put_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            params![key, value, now_ms()],
        )?;
        Ok(())
    }
}

/// Rewrites Tonyc's `known_hosts` from the pinned keys of every node.
pub fn write_known_hosts(path: &std::path::Path, nodes: &[NodeSpec]) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut lines: Vec<&str> = nodes.iter().flat_map(|n| n.host_keys.iter().map(String::as_str)).collect();
    lines.sort();
    lines.dedup();
    let mut text = lines.join("\n");
    text.push('\n');
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text)?;
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Validates a node before anything is contacted.
pub fn validate(spec: &NodeSpec) -> Result<()> {
    if !valid_name(&spec.name) {
        bail!("node name `{}` must be 1–32 lowercase letters, digits, `-` or `_`", spec.name);
    }
    if !valid_target(&spec.target) {
        bail!("ssh target `{}` must look like user@host, host or an ssh config alias", spec.target);
    }
    if !valid_runtime_url(&spec.runtime_url) {
        bail!("runtime url `{}` must be http://host[:port]", spec.runtime_url);
    }
    if spec.slots == 0 || spec.slots > 64 {
        bail!("slots must be between 1 and 64");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation() {
        assert!(valid_name("forge") && valid_name("gpu-2") && !valid_name("-x") && !valid_name("Forge"));
        assert!(valid_runtime_url("http://127.0.0.1:11434") && valid_runtime_url("http://[::1]:11434/"));
        assert!(!valid_runtime_url("http://x;rm -rf /") && !valid_runtime_url("https://x"));
        assert!(valid_target("sal@10.0.0.42") && !valid_target("-oProxyCommand=x") && !valid_target("a b"));
    }

    #[test]
    fn quoting() {
        assert_eq!(sh_quote("it's"), r"'it'\''s'");
    }

    #[test]
    fn parses_probe_output() {
        let text = "arch=x86_64\ncores=16\nmem_kb=32768000\nload=0.50\ncurl=yes\ngpu=NVIDIA RTX 4090, 24564\n---tags---\n\
                    {\"models\":[{\"name\":\"qwen:7b\",\"size\":4000000000,\"digest\":\"abc\",\"details\":{\"parameter_size\":\"7B\"}}]}";
        let (caps, tags, _) = parse_probe(text);
        assert_eq!((caps.arch.as_str(), caps.cores, caps.memory_mb), ("x86_64", 16, 32000));
        assert_eq!(caps.gpus[0].vram_mb, Some(24564));
        assert!(caps.curl && caps.runtime_ok);
        assert_eq!(tags.unwrap()[0].name, "qwen:7b");
        let (caps, tags, _) = parse_probe("arch=aarch64\ncurl=yes\n---tags---\ncurl: (7) Failed to connect");
        assert!(!caps.runtime_ok && tags.is_none());
        assert!(caps.runtime_error.unwrap().contains("Failed to connect"));
        let (_, _, loaded) =
            parse_probe("curl=yes\n---tags---\n{\"models\":[]}\n---ps---\n{\"models\":[{\"name\":\"qwen:7b\"}]}");
        assert_eq!(loaded, ["qwen:7b"]);
    }
}
