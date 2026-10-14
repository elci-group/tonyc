# Tonyc

**A latency-arbitrage and cognition-reuse layer for agentic systems** (directive TONYC/1).

Agents usually buy immediate remote inference for work that could wait, run locally, or reuse earlier
results. Tonyc is a local daemon that takes work *with its real constraints* (how urgent, how good, where
the data may go) and chooses the cheapest execution that satisfies them. It may choose a twenty-minute
local computation over a twenty-second remote one. That is the point.

```
agents, orchestrators ──► Rust SDK · MCP · Python · Tov CLI ──tonyc/1 over a unix socket──► tonycd (Rust)
                                                   ├─ envelope compiler   (tonyc-core)
                                                   ├─ planner: locality filter → quality → deadline → utility
                                                   ├─ scheduler: urgency classes, EDF, pressure, preemption
                                                   ├─ Cognitive Store: reuse, context, freshness, invalidation
                                                   ├─ refinement, Preemptive Dynamism, economics ledger
                                                   ├─ runtimes: ollama, OpenAI-compatible (llama.cpp, vLLM, …),
                                                   │            local programs, builtin, SSH nodes (OpenSSH)
                                                   └─ SQLite: queue, objects, append-only provenance + audits
```

- **Rust** — `crates/tonyc-core` (contracts; no I/O), `crates/tonycd` (the daemon), `crates/tonyc-sdk`
  (the Rust SDK) and `crates/tonyc-mcp` (Tonyc as MCP tools for agents).
- **Tov** — `cli/`: the `tonyc` command and its client (`cli/client.tov`), compiled to a native binary.
- **Protocol** — [`docs/PROTOCOL.md`](docs/PROTOCOL.md) and [`protocol/tonyc.openapi.json`](protocol/tonyc.openapi.json):
  the language-neutral contract every client speaks; `sdk/python/tonyc.py` is a one-file client in it.
- **Integrations** — `integrations/lucid`: a Lucid `ModelProvider` that serves existing cognition first.

## Status

All eight phases of the directive's programme are built and tested: 0 (contracts), 1 (local daemon),
2 (Cognitive Store), 3 (SSH compute fabric), 4 (economics), 5 (refinement), 6 (Preemptive Dynamism) and
7 (ecosystem: Rust SDK, language-neutral protocol, agent and runtime adapters, orchestrator
integration). The crates are ready to publish but have not been published. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the acceptance-gate status.

## Build

Needs Rust ≥ 1.88, [Tov](https://tov.sh) 0.0.1 and a C compiler.

```sh
cargo build --release                       # target/release/tonycd
(cd cli && CC=gcc tov build main.tov -o tonyc)
scripts/check.sh                            # the full suite: fmt, clippy, Rust tests, Tov tests, e2e
```

`CC=gcc`: on Ubuntu, Tov's default (clang) can select a GCC installation without `libstdc++` and fail
to link anything that uses `fetch`. `scripts/check.sh` sets it for you.

## Use

```sh
tonycd &                                    # LOCAL_ONLY ceiling; discovers Ollama on 127.0.0.1:11434
tonyc doctor
tonyc submit --objective "Summarise this changelog" --file CHANGELOG.md --deadline 30m
tonyc jobs
tonyc inspect tny_3d63              # decision trace: what ran, why, what was rejected, what left the host
tonyc await tny_3d63
```

Results are kept as *cognitive objects* and reused:

```sh
tonyc try --objective "Summarise this changelog" --file CHANGELOG.md   # existing answer now, or exit 5
tonyc query release notes                   # search existing cognition
tonyc corpus add ~/src/app --scope project:app --sensitivity sensitive # tracked; forces LOCAL_ONLY
tonyc cache inspect obj_61be                # lineage, sources, freshness
tonyc invalidate --path ~/src/app/Cargo.toml
tonyc corpus forget project:app             # deletes derived cognition, redacts results
```

An identical request is answered from the store without inference, if the stored answer is still fresh
(its sources re-hashed and unchanged, its TTL unexpired), visible from the request's scope, and meets
the request's quality floor. Related answers are offered to the model as context; a stale answer to the
same request is offered as the starting point for a refresh. When a source changes, everything derived
from it goes `STALE`, including the jobs whose results it was.

Machines you authorise become compute nodes over standard SSH:

```sh
tonyc node add forge --ssh me@10.0.0.42 --host-key SHA256:…   # or a host already in ~/.ssh/known_hosts
tonyc nodes                                  # health, hardware, models
tonyc policy set --max-locality ssh_allowlist   # explicit: nothing uses a node until you do this
tonyc submit --objective "…" --locality ssh_allowlist --ssh-allow forge
tonyc audit --job tny_… --payload            # the exact bytes that went to the node
```

Tonyc uses the system OpenSSH client in batch mode, against its own pinned `known_hosts` with strict
host-key checking, and with agent, X11 and port forwarding off. Host keys are never trusted on first
use. Nodes need `curl` and an Ollama on their own loopback; nothing is installed on them. If a node fails
during dispatch, the job re-plans without it (`on_node_failure: fallback`, the default) or ends
`NODE_UNREACHABLE` (`fail`).

Tonyc measures instead of assuming, and accounts for what it saves:

```sh
tonyc model benchmark ollama/gemma3:1b --wait   # checkable tasks, run as low-priority jobs
tonyc model inspect ollama/gemma3:1b            # prior vs measured speed, warm/cold start, quality evidence
tonyc models                                    # effective profiles (measured after 3 runs)
tonyc economics --since 7d                      # cost avoided, operating cost, Inference Avoidance Ratio
```

Every completed inference is measured (Ollama's load/prefill/generate timings), and after three runs the
measurements replace the static priors; cold and warm starts are told apart, using the runtime's own
list of loaded models. Benchmark results blend into each capability's quality estimate. Models that do
not fit in memory are skipped, background work avoids busy hosts and nodes (load, memory, GPU, heat,
battery), and budgets cap monthly spend (default £0), daily compute per urgency class and model size.
Every job's ledger row prices its tokens at a reference premium API (configurable, £2.40/£12.00 per
million tokens in/out by default) against the energy it used; all such figures are estimates.

Time is spent on quality rather than on a bigger model. A job that needs more than one cheap pass can
get it:

```sh
tonyc submit --objective "…" --verification required --floor 0.8   # draft cheaply, validate independently
tonyc submit --objective "…" --check-contains canberra               # deterministic checks; failures are revised
tonyc model benchmark ollama/gemma3:latest --capability validation  # measure a model as a judge
```

A cheap model drafts; independent validators judge the draft, and each verdict moves the answer's
confidence by Bayes' rule using that validator's *measured* sensitivity and specificity; a failing
answer is revised with the critique's issues, escalating to a stronger model; disagreeing validators are
settled by a tie-break. If the floor or the verification requirement cannot be met — or provably could
not be, before any inference — the job ends `INSUFFICIENT_QUALITY` with its evidence and best attempt.
`tonyc inspect` shows the refinement timeline.

When useful cognition is predictable, the cheapest time to produce it may be before anyone asks.
Preemptive Dynamism watches Tonyc's own demand log (nothing else): a request asked for repeatedly —
`try` misses included — whose answer is missing, stale or about to expire, is prepared ahead as
`SPECULATIVE` work if the probability of it being asked for again within the horizon, times the cost it
would avoid, beats its operating cost. Every prepared answer is tracked until it is used (a PD hit) or
discarded, and PD raises its own bar while its Preemption Yield is low.

```sh
tonyc pd explain      # what is being prepared, and why Tonyc expects it to be used
tonyc pd status       # pending work, speculative budget, Preemption Yield
tonyc pd pause        # stop speculating (tonyc pd resume to restart)
```

A request can also be a YAML or JSON file (`tonyc submit job.yaml`, or `-` for stdin); see
[examples/dependency-drift.yaml](examples/dependency-drift.yaml). Only `task.objective` is required.
Defaults are conservative: `DEFERRED`, `LOCAL_ONLY`, quality floor 0.5, `text/v1`, reuse on in scope
`user`, freshness `source_hash` when the task reads files and a 7-day TTL otherwise.

Configuration lives in `~/.config/tonyc/tonycd.toml`; [examples/tonycd.toml](examples/tonycd.toml)
lists every key. `tonycd --print-config` shows the effective configuration.

### Exit codes (`tonyc`)

| code | meaning |
|---|---|
| 0 | success (for `await`/`result`/`submit --wait`: the job is `READY`) |
| 1 | daemon unreachable, or the API rejected the request |
| 2 | usage error |
| 3 | the job ended in a non-`READY` state (`INSUFFICIENT_QUALITY`, `POLICY_BLOCKED`, `STALE`, …) |
| 4 | still pending when the wait timed out |
| 5 | `try`: no existing cognition satisfies the request |

## Agents and orchestrators

Everything speaks `tonyc/1`: HTTP/1.1 with JSON on the daemon's Unix socket (`$TONYC_SOCKET`, else
`$XDG_RUNTIME_DIR/tonyc/tonycd.sock`, else `~/.local/state/tonyc/tonycd.sock`; mode 0600). The contract
is deliberately small — `submit`, `status`, `await`, `cancel`, `try`, `query`, `invalidate` — plus what
orchestrators need: idempotent submission, a durable event feed, and attribution by origin. The full
reference is [docs/PROTOCOL.md](docs/PROTOCOL.md).

**Rust** (`crates/tonyc-sdk`):

```rust
use tonyc_sdk::{Client, Served, Work};
let tonyc = Client::new().with_origin("my-agent");
let work = Work::new("Summarise the open incidents").deadline_in("2h");
let answer = match tonyc.try_or(&work, my_own_inference()).await? {
    Served::Existing(object) => object.content.unwrap_or_default(), // cold path: already known
    Served::Hot(answer) => answer,                                   // hot path: the agent's own call
};
let job = tonyc.submit_idempotent("incident-digest-2026-10-14", &work).await?;  // warm path
let done = tonyc.wait(&job.id, std::time::Duration::from_secs(7200)).await?;
```

**MCP** (any MCP agent — Claude Code, IDE agents): `cargo install --path crates/tonyc-mcp`, then
`claude mcp add tonyc -- tonyc-mcp`. The agent gets `tonyc_try`, `tonyc_submit`, `tonyc_await`,
`tonyc_status`, `tonyc_query`, `tonyc_cancel` and `tonyc_invalidate`; it cannot ask for remote inference.

**Python** (standard library only): `sdk/python/tonyc.py`.

```python
import tonyc
t = tonyc.Client(origin="my-agent")
job = t.submit({"task": {"objective": "Summarise the open incidents"}, "latency": {"deadline": "2h"}},
               idempotency_key="incident-digest-2026-10-14")
print(t.wait(job["id"])["result"]["output"])
```

**Orchestrators** follow their work with one long poll instead of polling each job, and see what Tonyc
did for them:

```sh
tonyc events --system lucid --follow       # GET /v1/events?after=<cursor>&wait_ms=60000&system=lucid
tonyc jobs --origin lucid
tonyc economics                             # … with a line per origin system
```

**Lucid**: `integrations/lucid` wraps any Lucid `ModelProvider`: existing cognition is served before the
provider is called, `defer` hands work Lucid will need later to Tonyc, and Lucid carries on unchanged if
Tonyc is down. It builds against a Lucid checkout next to this repository (`scripts/integrations.sh`).

**Runtimes** beyond Ollama: any OpenAI-compatible server on this host or the LAN (llama.cpp's
`llama-server`, vLLM, LM Studio, LocalAI) and any local program (Candle, ONNX Runtime, `llama-cli`, a
script) speaking `tonyc.runtime/1` on stdin/stdout. See [docs/PROTOCOL.md](docs/PROTOCOL.md#runtime-adapters)
and `examples/tonycd.toml`.

## Licence

MIT; see [LICENSE](LICENSE).
