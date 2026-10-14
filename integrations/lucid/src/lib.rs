//! Tonyc for Lucid (directive §28).
//!
//! Lucid decides how work contributes to the user's objective; Tonyc decides
//! when, where and how cheaply cognition is produced. [`TonycFirst`] wraps
//! any Lucid [`ModelProvider`]:
//!
//! - **cold path**: before calling the provider, it asks Tonyc whether the
//!   answer already exists (`try`); a hit is returned without spending a
//!   provider call. A miss is recorded by Tonyc as demand, so a question
//!   Lucid keeps asking gets prepared ahead of time.
//! - **hot path**: otherwise — and whenever Tonyc is down — the wrapped
//!   provider answers, exactly as before. Tonyc can never block Lucid.
//! - **warm path**: [`TonycFirst::defer`] hands a request Lucid will need
//!   later (with a deadline) to Tonyc, to be produced locally when cheapest;
//!   when the same request is made later, the cold path serves it.
//!
//! Only plain completions take part: requests that offer tools, or carry
//! tool calls or tool results, always go to the provider. Cognition derived
//! from `sensitive` or `secret` sources is never returned, because a Lucid
//! conversation may be sent on to a remote provider.

use async_trait::async_trait;
use lucid_platform::ports::{ChatMessage, ModelProvider, ModelRequest, ModelResponse, ProviderError, Usage};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tonyc_sdk::{Client, Submission, TryOutcome, Work};

/// The largest answer Tonyc will be asked for (its default ceiling).
const MAX_TOKENS: u32 = 8192;

pub struct TonycFirst {
    inner: Arc<dyn ModelProvider>,
    tonyc: Client,
    name: String,
    scope: Option<String>,
    floor: f64,
    run: Option<String>,
    stats: Stats,
}

/// What the wrapper did, for Lucid's own metrics.
#[derive(Default)]
pub struct Stats {
    /// Requests answered from Tonyc without a provider call.
    pub served: AtomicU64,
    /// Requests Tonyc had no answer for.
    pub missed: AtomicU64,
    /// Requests that could not take part (tools, or Tonyc unreachable).
    pub bypassed: AtomicU64,
}

impl TonycFirst {
    /// Wraps `inner`, asking the daemon `tonyc` first. Work is attributed to
    /// origin system `lucid`.
    pub fn new(inner: Arc<dyn ModelProvider>, tonyc: Client) -> TonycFirst {
        let name = format!("tonyc+{}", inner.name());
        TonycFirst {
            inner,
            tonyc: tonyc.with_origin("lucid"),
            name,
            scope: None,
            floor: 0.7,
            run: None,
            stats: Stats::default(),
        }
    }

    /// Restricts reuse to one cognition scope, e.g. `project:<lucid project id>`.
    pub fn scope(mut self, scope: impl Into<String>) -> TonycFirst {
        self.scope = Some(scope.into());
        self
    }

    /// The lowest confidence an existing answer needs to be served (default 0.7).
    pub fn floor(mut self, floor: f64) -> TonycFirst {
        self.floor = floor;
        self
    }

    /// Lucid's task id, recorded as the origin `run` of work deferred to Tonyc.
    pub fn task(mut self, task_id: impl Into<String>) -> TonycFirst {
        self.run = Some(task_id.into());
        self
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The Tonyc work a request corresponds to, or `None` if it cannot take
    /// part. The mapping is deterministic, so the same request always asks
    /// Tonyc the same question: the final user message is the objective,
    /// and the conversation before it is the input.
    pub fn work_for(&self, req: &ModelRequest) -> Option<Work> {
        if !req.tools.is_empty() || req.temperature.is_some_and(|t| t >= 1.0) {
            return None;
        }
        let (last, earlier) = req.messages.split_last()?;
        let ChatMessage::User { content: objective } = last else { return None };
        let mut context = String::new();
        for m in earlier {
            let (role, text) = match m {
                ChatMessage::System { content } => ("system", content.as_str()),
                ChatMessage::User { content } => ("user", content.as_str()),
                ChatMessage::Assistant { content, tool_calls } if tool_calls.is_empty() => {
                    ("assistant", content.as_deref().unwrap_or(""))
                }
                _ => return None,
            };
            context.push_str(&format!("{role}: {text}\n"));
        }
        let mut work =
            Work::new(objective.trim()).max_tokens(req.max_output_tokens.clamp(1, MAX_TOKENS)).floor(self.floor);
        if !context.is_empty() {
            work = work.input(context);
        }
        if let Some(s) = &self.scope {
            work = work.scope(s);
        }
        Some(work.origin("lucid", self.run.clone(), Some(req.model.clone())))
    }

    /// Warm path: asks Tonyc to produce the answer to `req` before
    /// `deadline` (e.g. `2h`), locally, when it is cheapest. A later
    /// [`complete`](ModelProvider::complete) of the same request is then
    /// served from Tonyc. `key` makes retries safe (an idempotency key).
    pub async fn defer(
        &self,
        req: &ModelRequest,
        deadline: &str,
        key: Option<&str>,
    ) -> Result<Submission, ProviderError> {
        let work = self
            .work_for(req)
            .ok_or_else(|| ProviderError::Rejected("this request cannot be deferred (tools or sampling)".into()))?
            .deadline_in(deadline);
        let submitted = match key {
            Some(k) => self.tonyc.submit_idempotent(k, &work).await,
            None => self.tonyc.submit(&work).await,
        };
        submitted.map_err(|e| match e {
            tonyc_sdk::Error::Api { status: 400, message, .. } => ProviderError::Rejected(message),
            other => ProviderError::Transient(other.to_string()),
        })
    }
}

#[async_trait]
impl ModelProvider for TonycFirst {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let Some(work) = self.work_for(request) else {
            self.stats.bypassed.fetch_add(1, Ordering::Relaxed);
            return self.inner.complete(request).await;
        };
        match self.tonyc.try_existing(&work).await {
            Ok(TryOutcome::Hit(object)) => {
                let shareable = matches!(object.sensitivity.as_deref(), None | Some("public") | Some("normal"));
                if let (true, Some(content)) = (shareable, object.content) {
                    self.stats.served.fetch_add(1, Ordering::Relaxed);
                    return Ok(ModelResponse {
                        provider: "tonyc".into(),
                        model: object.model.unwrap_or_else(|| "tonyc".into()),
                        content: Some(content),
                        tool_calls: vec![],
                        // No provider tokens were spent.
                        usage: Usage { input_tokens: 0, output_tokens: 0, reported: true },
                        finish_reason: "stop".into(),
                    });
                }
                self.stats.missed.fetch_add(1, Ordering::Relaxed);
            }
            Ok(TryOutcome::Miss { .. }) => {
                self.stats.missed.fetch_add(1, Ordering::Relaxed);
            }
            // Tonyc is optional: if it is down, Lucid carries on.
            Err(_) => {
                self.stats.bypassed.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.inner.complete(request).await
    }
}
