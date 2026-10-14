//! The Tonyc Rust SDK.
//!
//! Tonyc is a latency-arbitrage and cognition-reuse layer for agents: work
//! that is not needed immediately is submitted with its constraints and
//! produced when, where and as cheaply as policy allows; work that has been
//! done before is reused. This crate speaks the `tonyc/1` protocol to a
//! running `tonycd` over its Unix socket (or loopback TCP).
//!
//! The contract is deliberately small (directive §18–19):
//!
//! ```no_run
//! # async fn demo() -> Result<(), tonyc_sdk::Error> {
//! use std::time::Duration;
//! use tonyc_sdk::{Client, TryOutcome, Work};
//!
//! let tonyc = Client::new().with_origin("my-agent");
//! let work = Work::new("Summarise the open incidents").deadline_in("2h");
//!
//! // Cold path: is the answer already known?
//! if let TryOutcome::Hit(object) = tonyc.try_existing(&work).await? {
//!     println!("{}", object.content.unwrap_or_default());
//!     return Ok(());
//! }
//! // Warm path: submit it and collect it later.
//! let job = tonyc.submit(&work).await?;
//! let done = tonyc.wait(&job.id, Duration::from_secs(3600)).await?;
//! println!("{:?}", done.output());
//! # Ok(()) }
//! ```

mod types;
mod work;

pub use tonyc_core::cognition::{CognitiveKind, FreshnessPolicy};
pub use tonyc_core::envelope::{Energy, NodeFailure, Refine, Urgency, Verification, WorkRequest};
pub use tonyc_core::refine::Check;
pub use tonyc_core::{Exposure, JobState, Locality};
pub use types::*;
pub use work::Work;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// The protocol this SDK speaks.
pub const PROTOCOL: &str = "tonyc/1";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The daemon is not running or not reachable.
    #[error("cannot reach tonycd at {at}: {source}")]
    Connect { at: String, source: std::io::Error },
    /// The daemon refused the request.
    #[error("tonycd answered {status}: {message}")]
    Api { status: u16, message: String, field: Option<String>, body: Value },
    #[error("transport: {0}")]
    Transport(String),
    #[error("unexpected answer: {0}")]
    Decode(String),
    /// The daemon speaks a different protocol.
    #[error("tonycd speaks {0}, this SDK speaks {PROTOCOL}")]
    Protocol(String),
    #[error("job {0} did not finish in time")]
    Timeout(String),
}

impl Error {
    /// The HTTP status of an API error.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Api { status, .. } => Some(*status),
            _ => None,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Clone)]
enum Endpoint {
    Unix(PathBuf),
    Tcp(SocketAddr),
}

/// A client for one daemon. Cheap to clone; each request opens its own
/// connection, so a client can be shared freely across tasks.
#[derive(Debug, Clone)]
pub struct Client {
    endpoint: Endpoint,
    origin: Option<(String, Option<String>)>,
}

impl Default for Client {
    fn default() -> Self {
        Client::new()
    }
}

impl Client {
    /// The daemon at the default socket: `$TONYC_SOCKET`, else
    /// `$XDG_RUNTIME_DIR/tonyc/tonycd.sock`, else
    /// `~/.local/state/tonyc/tonycd.sock`.
    pub fn new() -> Client {
        Client::unix(tonyc_core::transport::default_socket())
    }

    pub fn unix(path: impl Into<PathBuf>) -> Client {
        Client { endpoint: Endpoint::Unix(path.into()), origin: None }
    }

    /// A daemon listening on loopback TCP (`daemon.tcp`).
    pub fn tcp(addr: SocketAddr) -> Client {
        Client { endpoint: Endpoint::Tcp(addr), origin: None }
    }

    /// Attributes every submission without its own origin to `system`
    /// (`lucid`, `terri`, `my-agent`, ...). Attribution never changes what
    /// is computed or where.
    pub fn with_origin(mut self, system: impl Into<String>) -> Client {
        self.origin = Some((system.into(), None));
        self
    }

    /// As [`Client::with_origin`], naming the agent too.
    pub fn with_agent(mut self, system: impl Into<String>, agent: impl Into<String>) -> Client {
        self.origin = Some((system.into(), Some(agent.into())));
        self
    }

    // ------------------------------------------------------- transport ---

    async fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<Vec<u8>>,
        headers: &[(&str, &str)],
    ) -> Result<(u16, Bytes)> {
        let req = {
            let mut b = hyper::Request::builder().method(method).uri(path).header("host", "tonycd");
            if body.is_some() {
                b = b.header("content-type", "application/json");
            }
            for (k, v) in headers {
                b = b.header(*k, *v);
            }
            b.body(Full::new(Bytes::from(body.unwrap_or_default()))).map_err(|e| Error::Transport(e.to_string()))?
        };
        let res = match &self.endpoint {
            Endpoint::Unix(path) => {
                let stream = tokio::net::UnixStream::connect(path)
                    .await
                    .map_err(|source| Error::Connect { at: path.display().to_string(), source })?;
                exchange(stream, req).await?
            }
            Endpoint::Tcp(addr) => {
                let stream = tokio::net::TcpStream::connect(addr)
                    .await
                    .map_err(|source| Error::Connect { at: addr.to_string(), source })?;
                exchange(stream, req).await?
            }
        };
        Ok(res)
    }

    async fn call<T: DeserializeOwned>(&self, method: &str, path: &str, body: Option<Vec<u8>>) -> Result<T> {
        self.call_with(method, path, body, &[]).await.map(|(_, v)| v)
    }

    async fn call_with<T: DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        body: Option<Vec<u8>>,
        headers: &[(&str, &str)],
    ) -> Result<(u16, T)> {
        let (status, bytes) = self.send(method, path, body, headers).await?;
        if !(200..300).contains(&status) {
            let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            let message = body
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
            let field = body.get("field").and_then(Value::as_str).map(str::to_string);
            return Err(Error::Api { status, message, field, body });
        }
        let value = serde_json::from_slice(&bytes).map_err(|e| Error::Decode(format!("{method} {path}: {e}")))?;
        Ok((status, value))
    }

    fn body(&self, work: &Work) -> Result<Vec<u8>> {
        let mut req = work.request().clone();
        if !work.has_origin()
            && let Some((system, agent)) = &self.origin
        {
            req.origin.system = Some(system.clone());
            req.origin.agent.clone_from(agent);
        }
        serde_json::to_vec(&req).map_err(|e| Error::Decode(e.to_string()))
    }

    // ---------------------------------------------------------- contract ---

    /// The protocol the daemon speaks and the features it has. Fails with
    /// [`Error::Protocol`] if it is not `tonyc/1`.
    pub async fn protocol(&self) -> Result<Protocol> {
        let p: Protocol = self.call("GET", "/v1/protocol", None).await?;
        if p.protocol != PROTOCOL {
            return Err(Error::Protocol(p.protocol));
        }
        Ok(p)
    }

    /// `submit(work) -> JobId`: queues the work. The daemon compiles it
    /// into an explicit envelope (returned) or refuses it with the field at
    /// fault.
    pub async fn submit(&self, work: &Work) -> Result<Submission> {
        self.call("POST", "/v1/jobs", Some(self.body(work)?)).await
    }

    /// As [`Client::submit`], but a retry with the same key (within 24
    /// hours, across daemon restarts) returns the original job instead of
    /// creating another. Reusing a key for different work is an error
    /// (409).
    pub async fn submit_idempotent(&self, key: &str, work: &Work) -> Result<Submission> {
        self.call_with("POST", "/v1/jobs", Some(self.body(work)?), &[("idempotency-key", key)]).await.map(|(_, s)| s)
    }

    /// `status(job) -> JobStatus`.
    pub async fn status(&self, id: &str) -> Result<Job> {
        self.call("GET", &format!("/v1/jobs/{}", encode(id)), None).await
    }

    /// `await(job)`: waits up to `timeout` (at most ten minutes per call)
    /// and returns the job, terminal or not.
    pub async fn await_job(&self, id: &str, timeout: Duration) -> Result<Job> {
        let ms = timeout.as_millis().min(600_000);
        self.call("GET", &format!("/v1/jobs/{}/await?timeout_ms={ms}", encode(id)), None).await
    }

    /// Waits until the job is terminal, for as long as `timeout`.
    pub async fn wait(&self, id: &str, timeout: Duration) -> Result<Job> {
        let until = tokio::time::Instant::now() + timeout;
        loop {
            let left = until.saturating_duration_since(tokio::time::Instant::now());
            let job = self.await_job(id, left.min(Duration::from_secs(600))).await?;
            if job.is_terminal() {
                return Ok(job);
            }
            if left.is_zero() {
                return Err(Error::Timeout(id.to_string()));
            }
        }
    }

    /// `cancel(job)`. Returns the state the job moved to (`CANCELLED`, or
    /// `CANCELLING` while a running job stops).
    pub async fn cancel(&self, id: &str) -> Result<String> {
        let v: Value = self.call("POST", &format!("/v1/jobs/{}/cancel", encode(id)), None).await?;
        Ok(v["state"].as_str().unwrap_or_default().to_string())
    }

    /// `tonyc.try()` (§19): existing cognition now, or a miss. Never queues
    /// work and never runs a model.
    pub async fn try_existing(&self, work: &Work) -> Result<TryOutcome> {
        let v: Value = self.call("POST", "/v1/try", Some(self.body(work)?)).await?;
        let decode =
            |v: &Value| serde_json::from_value::<CognitiveObject>(v.clone()).map_err(|e| Error::Decode(e.to_string()));
        if v["hit"].as_bool() == Some(true) {
            return Ok(TryOutcome::Hit(Box::new(decode(&v["object"])?)));
        }
        let nearest = v["nearest"].as_array().map(|a| a.iter().map(decode).collect()).transpose()?.unwrap_or_default();
        let skipped = v["skipped"].as_array().cloned().unwrap_or_default();
        Ok(TryOutcome::Miss { nearest, skipped })
    }

    /// `query(cognitive_query) -> Vec<CognitiveObject>`, ranked by relevance
    /// when `text` is given.
    pub async fn query(&self, q: &Query) -> Result<Vec<CognitiveObject>> {
        let mut params = Vec::new();
        let mut add = |k: &str, v: &Option<String>| {
            if let Some(v) = v {
                params.push(format!("{k}={}", encode(v)));
            }
        };
        add("text", &q.text);
        add("scope", &q.scope);
        add("kind", &q.kind);
        add("state", &q.state);
        add("limit", &q.limit.map(|l| l.to_string()));
        let path =
            if params.is_empty() { "/v1/cognition".into() } else { format!("/v1/cognition?{}", params.join("&")) };
        let v: Value = self.call("GET", &path, None).await?;
        serde_json::from_value(v["objects"].clone()).map_err(|e| Error::Decode(e.to_string()))
    }

    /// `invalidate(scope)`: marks cognition stale, with everything derived
    /// from it. Returns the daemon's report of what was invalidated.
    pub async fn invalidate(&self, what: Invalidate, reason: Option<&str>) -> Result<Value> {
        let (k, v) = match &what {
            Invalidate::Object(v) => ("object", v),
            Invalidate::Path(v) => ("path", v),
            Invalidate::Scope(v) => ("scope", v),
            Invalidate::Event(v) => ("event", v),
        };
        let mut body = serde_json::json!({ k: v });
        if let Some(r) = reason {
            body["reason"] = r.into();
        }
        self.call("POST", "/v1/invalidate", Some(body.to_string().into_bytes())).await
    }

    /// One page of the event feed after `after` (0 for the beginning, -1
    /// for "from now"), long-polling up to `wait` for something to happen.
    pub async fn events(&self, after: i64, wait: Duration, system: Option<&str>) -> Result<EventPage> {
        let mut path = format!("/v1/events?after={after}&wait_ms={}", wait.as_millis().min(60_000));
        if let Some(s) = system {
            path.push_str(&format!("&system={}", encode(s)));
        }
        self.call("GET", &path, None).await
    }

    // --------------------------------------------------- three planes ---

    /// The three-plane pattern (§27): serve existing cognition if there is
    /// any (cold path); otherwise run `hot` — the caller's own immediate
    /// inference — and return its answer. The miss is recorded as demand,
    /// so a question asked often enough is prepared ahead of time.
    pub async fn try_or<T, F>(&self, work: &Work, hot: F) -> Result<Served<T>>
    where
        F: Future<Output = T>,
    {
        match self.try_existing(work).await? {
            TryOutcome::Hit(o) => Ok(Served::Existing(o)),
            TryOutcome::Miss { .. } => Ok(Served::Hot(hot.await)),
        }
    }
}

/// Where an answer came from.
#[derive(Debug)]
pub enum Served<T> {
    Existing(Box<CognitiveObject>),
    Hot(T),
}

async fn exchange<S>(stream: S, req: hyper::Request<Full<Bytes>>) -> Result<(u16, Bytes)>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|e| Error::Transport(e.to_string()))?;
    let driver = tokio::spawn(conn);
    let res = sender.send_request(req).await.map_err(|e| Error::Transport(e.to_string()))?;
    let status = res.status().as_u16();
    let body = res.into_body().collect().await.map_err(|e| Error::Transport(e.to_string()))?.to_bytes();
    driver.abort();
    Ok((status, body))
}

/// Percent-encodes a path segment or query value.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_query_values() {
        assert_eq!(encode("project:a b/c"), "project%3Aa%20b%2Fc");
        assert_eq!(encode("tny_ab-1.2~"), "tny_ab-1.2~");
    }

    #[test]
    fn a_built_request_compiles_as_written() {
        let work = Work::new("x").input("y").deadline_in("2h").floor(0.7).scope("project:p").json().max_tokens(64);
        let client = Client::unix("/nonexistent").with_agent("lucid", "cto");
        let body = client.body(&work).unwrap();
        let back = WorkRequest::parse(std::str::from_utf8(&body).unwrap()).unwrap();
        let env = tonyc_core::compile(&back, "tny_1", 1, &Default::default()).unwrap();
        assert_eq!(env.task.input_text.as_deref(), Some("y"));
        assert_eq!(env.latency.deadline_at, Some(1 + 7_200_000));
        assert_eq!(env.reuse.scope.to_string(), "project:p");
        let origin = env.origin.unwrap();
        assert_eq!((origin.system.as_str(), origin.agent.as_deref()), ("lucid", Some("cto")));
        // An explicit origin wins over the client's default.
        let own = Work::new("x").origin("terri", Some("r1".into()), None);
        let back: WorkRequest = serde_json::from_slice(&client.body(&own).unwrap()).unwrap();
        assert_eq!(back.origin.system.as_deref(), Some("terri"));
    }

    #[tokio::test]
    async fn a_missing_daemon_is_a_connect_error() {
        let err = Client::unix("/nonexistent/tonycd.sock").protocol().await.unwrap_err();
        assert!(matches!(err, Error::Connect { .. }), "{err}");
    }
}
