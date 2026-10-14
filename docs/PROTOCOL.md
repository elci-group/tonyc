# The Tonyc protocol (`tonyc/1`)

This is the language-neutral contract between `tonycd` and everything that uses it: agents, orchestrators,
SDKs and the CLI. Anything that can send HTTP/1.1 over a Unix socket can use Tonyc; nothing needs to
link against it (directive §5).

The machine-readable form of the agent contract is [`protocol/tonyc.openapi.json`](../protocol/tonyc.openapi.json)
(OpenAPI 3.1). The test `crates/tonycd/tests/conformance.rs` checks that every live response from the
daemon validates against it, and that every route the daemon serves is documented in it or here.

## Conformance levels

| Level | What a client or server implements |
|---|---|
| **core** | `/v1/protocol`, `/v1/jobs` (submit, list), `/v1/jobs/{id}`, `/v1/jobs/{id}/await`, `/v1/jobs/{id}/cancel`, `/v1/try`, `/v1/cognition`, `/v1/invalidate`, `/v1/events` — the agent contract of §18–19 plus the orchestrator primitives. All in the OpenAPI document. |
| **operator** | Everything else below: status, models, nodes, policy, audit, benchmarks, economics, Preemptive Dynamism, corpus. Used by the `tonyc` CLI and dashboards; agents never need it. |

## Transport

- HTTP/1.1, JSON bodies, on a Unix socket. The socket is found the same way by every client:
  `$TONYC_SOCKET`, else `$XDG_RUNTIME_DIR/tonyc/tonycd.sock`, else `$XDG_STATE_HOME/tonyc/tonycd.sock`
  (default `~/.local/state/tonyc/tonycd.sock`). The socket is mode `0600` in a `0700` directory: being
  able to connect is the authorisation.
- Optionally a loopback TCP port (`daemon.tcp = "127.0.0.1:7878"`); non-loopback addresses are refused.
- Any `Host` header. Bodies need no `Content-Type`. Work requests (`POST /v1/jobs`, `POST /v1/try`) may
  be YAML as well as JSON.
- Ids: job and object ids accept any unique prefix (8+ characters including `tny_`/`obj_`).

## Versioning

`GET /v1/protocol` answers:

```json
{"protocol": "tonyc/1", "revision": "1.1", "envelope": "tonyc.envelope/1",
 "daemon": {"name": "tonycd", "version": "0.1.0"},
 "features": ["submit", "status", "await", "cancel", "try", "query", "invalidate", "events",
              "idempotency", "origin", "refinement", "preemptive_dynamism", "ssh_fabric", "economics"],
 "transport": {"unix": "/run/user/1000/tonyc/tonycd.sock", "tcp": null},
 "limits": {"await_timeout_ms_max": 600000, "events_wait_ms_max": 60000,
            "events_limit_max": 1000, "idempotency_ttl_ms": 86400000}}
```

A client checks `protocol` first and refuses anything but `tonyc/1`. Revisions within `tonyc/1` only
add endpoints, response fields and features, so clients must ignore fields they do not know and use
`features` to decide what to call. An incompatible change becomes `tonyc/2` under `/v2/`.

## Errors

Every error is JSON: `{"error": "<message>"}`, plus `"field"` (the request field at fault) on 400s
from compiling a work request, and `"job_id"` on an idempotency conflict.

| Status | Meaning |
|---|---|
| 400 | Invalid request; the work request names the field. Unknown fields are rejected. |
| 404 | No such job, object or endpoint. |
| 409 | A conflict: an id prefix matches several jobs, a job is already terminal, or an `Idempotency-Key` was used for different work. |
| 503 | The daemon is shutting down (long polls end with this). |

## The agent contract (core)

| Method | Path | §18–19 | Notes |
|---|---|---|---|
| GET | `/v1/protocol` | — | as above |
| POST | `/v1/jobs` | `submit(work) -> JobId` | `201` with the compiled envelope; `Idempotency-Key` header (below) |
| GET | `/v1/jobs?state=&limit=&origin=` | — | newest first |
| GET | `/v1/jobs/{id}` | `status(job)` | job, decision, result with provenance, events, boundary records, ledger row, refinement passes |
| GET | `/v1/jobs/{id}/await?timeout_ms=` | `await(job)` | long-polls; `200` when terminal, `202` (same body) when the timeout passes first |
| POST | `/v1/jobs/{id}/cancel` | `cancel(job)` | `200` cancelled, `202` while a running job stops, `409` if already terminal |
| POST | `/v1/try` | `tonyc.try()` | existing cognition now, or `{"hit": false, "nearest": [...]}`; never queues work |
| GET | `/v1/cognition?text=&scope=&kind=&state=&limit=` | `query(q)` | ranked by relevance when `text` is given |
| POST | `/v1/invalidate` | `invalidate(scope)` | exactly one of `object`, `path`, `scope`, `event`; cascades to derived cognition |
| GET | `/v1/events?after=&limit=&wait_ms=&system=` | — | the durable event feed (below) |

### Work requests

A work request states constraints, never models (T7):

```yaml
task:
  objective: "Summarise what changed in the lockfile"   # required
  input: {text: "...", files: [/srv/repo/Cargo.lock]}   # files: absolute paths
  capability: general            # or summarisation, code.rust, ...
  max_tokens: 512
latency: {class: DEFERRED, deadline: 2h}                # NOW SOON DEFERRED IDLE SPECULATIVE
quality: {floor: 0.7, verification: preferred, checks: [{contains: [version]}], refine: auto}
privacy: {locality: LOCAL_ONLY}                         # the daemon's ceiling can only narrow it
resources: {cpu: opportunistic, gpu: opportunistic, energy: economical, on_node_failure: fallback}
reuse: {cache: true, corpus: "project:billing", freshness: source_hash}
output: {schema: text/v1, kind: summary}
origin: {system: lucid, run: task-118, agent: cto}      # attribution only
```

The daemon compiles it into an explicit `tonyc.envelope/1` (every default filled in) and returns that
envelope, so a caller can see exactly what was agreed. Its `semantic_hash` identifies the question
(task and output only): two requests with the same hash ask for the same cognition, whoever asks and
however urgently.

### Job states

`QUEUED → PLANNING → RUNNING → (REFINING) → READY`, or a terminal failure: `CANCELLED`,
`DEADLINE_EXCEEDED`, `POLICY_BLOCKED`, `MODEL_FAILED`, `NODE_UNREACHABLE`, `INSUFFICIENT_QUALITY`.
`RESOURCE_BLOCKED` is waiting, not terminal. `READY` can later become `STALE` when the cognition it
produced is invalidated. A failed job's `failure.reason` says why, in words.

### Idempotency

Orchestrators retry. A `POST /v1/jobs` with `Idempotency-Key: <1-200 printable ASCII, no spaces>`
creates the job once: a repeat with the same key and the same request (compared after parsing, so YAML
versus JSON or formatting does not matter) answers `200` with the original job and `"replayed": true`;
the same key with a different request answers `409`. Keys are remembered for 24 hours, in the state
database, so they survive a daemon restart. Check-and-insert is one transaction: concurrent retries
create one job.

### The event feed

`GET /v1/events?after=<cursor>` returns job events after the cursor, oldest first:

```json
{"events": [{"seq": 41, "at": 1791963357471, "job_id": "tny_3d14...", "kind": "finished",
             "state": "READY", "terminal": true, "origin": {"system": "lucid", "run": "task-118"}}],
 "next": 41}
```

Keep `next` and pass it as `after` next time; the feed is the append-only event log in the state
database, so a cursor stays valid across restarts of either side. `after=-1` starts from now.
`wait_ms` (up to 60 s) long-polls until there is at least one event. `system=<name>` keeps only jobs
from that origin. An orchestrator follows its work with one long poll instead of polling every job.

### Origin and attribution

`origin` (`system`, `run`, `agent`) says who asked. It changes nothing about what is computed, where, or
what may be reused — it is not part of the semantic hash, so Lucid can be served cognition TERRI paid
for. It appears in job listings and the event feed, `GET /v1/jobs?origin=` filters by it, and
`GET /v1/economics` reports the ledger `by_origin` (inference runs, answers served from the store,
compute hours, conventional cost, operating cost, cost avoided), with unattributed work grouped as
`unattributed`.

### The three planes (§27)

Tonyc owns the warm and cold paths; the agent keeps the hot one:

```text
POST /v1/try ──hit──► consume                          (cold: prepared or reused)
      │
     miss ──► need it now?  yes ──► the agent's own inference   (hot)
                            no  ──► POST /v1/jobs with a deadline; collect later (warm)
```

The miss is recorded as demand; questions that miss repeatedly are prepared ahead of time by Preemptive
Dynamism (when the expected value beats the cost) and the next `try` hits.

## Operator endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/status` | daemon, queue, runtimes, nodes, pressure, ceiling, cognition counts |
| GET | `/v1/host` | host resources and pressure |
| GET/POST | `/v1/policy` | the locality ceiling; set `{"max_locality"?, "default_locality"?, "reset"?}` |
| GET | `/v1/models` | every candidate model with its effective profile and whether the ceiling blocks it |
| POST | `/v1/models/refresh` | rediscover runtimes |
| GET | `/v1/model?id=` | prior, measured and effective profile, recent runs |
| POST | `/v1/models/benchmark` | `{"model", "capability"?, "cases"?, "class"?}` |
| GET | `/v1/benchmarks`, `/v1/benchmarks/{id}` | benchmark runs and per-case results |
| GET | `/v1/jobs/{id}/trace` | the decision trace as text |
| GET | `/v1/cognition/{id}` | an object with sources, ancestors, children, events, live freshness |
| POST | `/v1/cognition/scan` | re-check the corpus and every fresh object now |
| POST | `/v1/cache/prune` | `{"older_than"?}`: delete stale objects |
| GET/POST | `/v1/corpus` | list / add `{"paths", "scope"?, "sensitivity"?}` |
| POST | `/v1/corpus/forget` | `{"target": scope or path}` |
| GET/POST | `/v1/nodes` | list / add an SSH node `{"name", "ssh", "host_key"?, "port"?, "identity"?, "runtime_url"?, "policy"?, "slots"?}` |
| GET/DELETE | `/v1/nodes/{name}` | one node / unregister it |
| POST | `/v1/nodes/{name}/probe` | probe again |
| GET | `/v1/audit?job=&payload=&limit=` | off-host crossings, with exact payloads on request |
| GET | `/v1/audit/verify` | re-check every crossing against its job's policy |
| GET | `/v1/economics?since=` | the ledger, summarised, with `by_origin` |
| GET | `/v1/pd`, `/v1/pd/explain` | Preemptive Dynamism status; predictions with reasons and outcomes |
| POST | `/v1/pd/evaluate`, `/v1/pd/pause`, `/v1/pd/resume` | evaluate demand now; stop or restart speculation |

## Runtime adapters

Runtimes are the other side of the protocol: the planner sees only runtime-neutral model profiles, and
any engine can be added through one of four adapters (gate G5).

| Adapter | Config | Speaks | Exposure |
|---|---|---|---|
| `builtin` | `[runtimes.builtin]` | in-process echo, for tests | in-process |
| `ollama` | `[runtimes.ollama]` | Ollama's `/api/tags`, `/api/ps`, `/api/generate` | from where the URL resolves |
| `openai` | `[[runtimes.openai]]` | `/v1/models`, `/v1/chat/completions`: llama.cpp `llama-server`, vLLM, LM Studio, LocalAI, Ollama's `/v1` | from where the URL resolves |
| `command` | `[[runtimes.command]]` | `tonyc.runtime/1` on stdin/stdout: Candle, ONNX Runtime, `llama-cli`, any program | declared (`local` by default) |

HTTP runtimes accept plain `http://` only: they are for this host and the LAN. The URL is resolved,
every address it resolves to is classified (loopback, private, public), and requests go to that exact
address, so DNS cannot move inference somewhere the locality policy did not see.

```toml
[[runtimes.openai]]
name = "llamacpp"                 # model ids become llamacpp/<model>
url = "http://127.0.0.1:8080"
api_key_env = "LLAMA_API_KEY"     # optional; read from the environment, never stored
models = []                       # optional allowlist; default: whatever /v1/models lists

[[runtimes.command]]
name = "candle"
argv = ["/opt/candle/phi", "--model", "{model}"]   # {model} is the model name
protocol = "json"                 # or "text"
exposure = "local"                # or lan:<host> / remote:<host> if the program forwards prompts
timeout_secs = 600
[[runtimes.command.models]]
name = "phi-2"
capabilities = {general = 0.55, "code.rust" = 0.4}
tokens_per_sec = 12.0
startup_ms = 2500
ram_mb = 3000
```

### `tonyc.runtime/1` (command adapter)

Tonyc runs the program once per request with a cleared environment (only `PATH`, `HOME`, `LANG`,
`LC_ALL`, `TMPDIR`), never puts the prompt on the command line, and kills it on cancellation or timeout.

- `protocol = "text"`: stdin is the prompt; stdout (trailing whitespace trimmed) is the answer. Token
  counts are estimated.
- `protocol = "json"`: stdin is one object, stdout one object:

```json
{"protocol": "tonyc.runtime/1", "model": "phi-2", "prompt": "...", "max_tokens": 512, "json": false}
```

```json
{"text": "...", "tokens_in": 211, "tokens_out": 87,
 "timings": {"load_ms": 900, "prefill_ms": 120, "generate_ms": 4100}}
```

`tokens_*` and `timings` are optional; `{"text": "", "error": "..."}` reports a failure. A non-zero exit
status is a failure too, with the end of stderr as the reason. When `json` is true the answer must be a
JSON object.

Tonyc cannot see inside a program. A program that sends prompts anywhere must declare it with
`exposure`; the locality policy then keeps `LOCAL_ONLY` work away from it like any other off-host
domain.

## Clients

| Client | Where | Notes |
|---|---|---|
| Rust SDK | `crates/tonyc-sdk` | the full core contract, `Work` builder, `try_or` for the three planes |
| MCP server | `crates/tonyc-mcp` (`tonyc-mcp`) | the contract as MCP tools for any MCP agent |
| Python | `sdk/python/tonyc.py` | standard library only; one file |
| Tov | `cli/client.tov` | used by the `tonyc` CLI |
| Lucid | `integrations/lucid` | a `ModelProvider` that serves existing cognition before calling the real provider |
| curl | — | `curl --unix-socket $SOCK http://tonycd/v1/protocol` |
