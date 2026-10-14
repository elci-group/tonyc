# Tonyc architecture (TONYC/1, Phases 0–7)

This describes what exists. The directive is the design target; this file tracks how far the code has
reached and where it deliberately differs.

## Crates and modules

| path | role |
|---|---|
| `crates/tonyc-core` | Phase 0 contracts. Pure: no I/O, no async, no runtime knowledge. |
| `  envelope.rs` | `WorkRequest` (lenient input) → `compile` → `WorkEnvelope` (explicit, deterministic) |
| `  locality.rs` | `Locality` classes, `Exposure` of an execution domain, `LocalityPolicy::permits` |
| `  state.rs` | the 13 canonical job states (§26) |
| `  model.rs` | `ModelProfile` (§12), runtime-neutral `InferenceRequest`/`InferenceOutput` |
| `  plan.rs` | the planner: hard filters, then utility ranking; returns a `Decision` or `PlanFailure` |
| `  cognition.rs` | object kinds, freshness policies, scopes, sensitivity, context admissibility, lexical retrieval |
| `crates/tonycd` | Phase 1 daemon |
| `  store.rs` | SQLite: jobs, append-only `events` and `boundary` tables (triggers reject UPDATE/DELETE); schema migration |
| `  cogstore.rs` | objects, sources, parents, corpus, file-hash cache, append-only `cognition_events` and `demand` |
| `  cognition.rs` | Phase 2 engine: fingerprinting, freshness checks, invalidation cascade, reuse, context, scan, corpus |
| `  fabric.rs` | Phase 3: OpenSSH transport, host-key pinning, node probe, `NodeAdapter`, node persistence |
| `  economics.rs` | Phase 4: observations, measured profiles, ledger, budgets, the economics report |
| `  bench.rs` | Phase 4: the benchmark suite (answer and judge cases), runs as pinned jobs, scoring |
| `  refinement.rs` | Phase 5: strategy, pinned passes, the draft/check/validate/revise loop |
| `  pd.rs` | Phase 6: request templates, demand, predictions, settlement, Preemption Yield |
| `crates/tonyc-core/src/refine.rs` | Phase 5 contracts: checks, verdict parsing, validator statistics, Bayes updates |
| `  scheduler.rs` | ticks, deadline expiry, pressure gating, preemption, execution |
| `  daemon.rs` | shared state; running-job registry with cancellation tokens and stop reasons |
| `  runtimes/` | `ModelAdapter` trait; `ollama`, `openai` (OpenAI-compatible servers), `command` (local programs), `builtin`; `net.rs` resolves and classifies runtime addresses |
| `  protocol.rs` | Phase 7: protocol description, idempotency keys, the event feed, the ledger by origin |
| `  host.rs` | CPU/memory/GPU discovery (`/proc`, `/sys`, `nvidia-smi`); pressure sampling |
| `  api.rs`, `trace.rs` | HTTP API over the Unix socket; decision-trace rendering |
| `crates/tonyc-core/src/transport.rs` | where the daemon listens; shared by the daemon, SDKs and CLI |
| `crates/tonyc-sdk` | Phase 7: the Rust SDK — `Client`, the `Work` builder, typed answers, `try_or` |
| `crates/tonyc-mcp` | Phase 7: `tonyc-mcp`, the contract as Model Context Protocol tools over stdio |
| `protocol/tonyc.openapi.json` | Phase 7: the agent contract, machine-readable (OpenAPI 3.1); `docs/PROTOCOL.md` is the prose |
| `sdk/python/tonyc.py` | Phase 7: a one-file standard-library Python client |
| `integrations/lucid` | Phase 7: a Lucid `ModelProvider` that serves Tonyc's cognition first (outside the workspace) |
| `cli/` (Tov) | `client.tov` (client), `args.tov`, `format.tov`, `main.tov` (`tonyc`) |

## Work envelope

`compile(request, id, submitted_at, defaults)` is a pure function: the same inputs give the same
envelope, byte for byte. It fills every default from daemon policy, turns relative deadlines into
absolute ones, rejects unknown fields and invalid values with the field path, and computes a
`semantic_hash` over task and output only — two envelopes asking for the same cognition share it
regardless of who asked, when, how urgently, or how good the answer must be (the quality floor is a
requirement an existing answer is checked against, not part of the question). Reuse keys on it.

Urgency classes: `NOW`, `SOON`, `DEFERRED`, `IDLE`, `SPECULATIVE`. `IDLE` and `SPECULATIVE` *yield*: they
wait while the host is busy and can be preempted. `DEFERRED` work marked `cpu`/`gpu: opportunistic` also
waits for an idle host but is never preempted.

## Locality (T1, §7)

Every runtime declares an `Exposure`: `InProcess`, `Loopback`, `Lan`, `Ssh`, or `Remote`. A plan is
admissible only if the job's policy **and** the daemon ceiling (`policy.max_locality`, default
`LOCAL_ONLY`) both permit the exposure. This is the planner's first stage, before any quality or cost is
considered, so no score can override it. It is checked again at dispatch against the adapter's current
exposure.

The Ollama adapter derives its exposure from where its URL *resolves* (loopback → `Loopback`, private
addresses → `Lan`, otherwise `Remote`, mixed → most exposed), then connects only to that resolved
address, so a DNS change cannot move a request somewhere the policy did not see. Before its first
successful probe it reads as `Remote` and unhealthy.

Every dispatch and every response is appended to the `boundary` table with its exposure, byte counts and
whether it left the host. `tonyc inspect` prints "Remote data transmitted: NONE" only when no record for
the job left the host; `tonyc doctor` checks that a `LOCAL_ONLY` daemon has zero off-host records.

## Planning (§13)

Stages, each recording a rejection reason per eliminated candidate:

1. **locality** → `POLICY_BLOCKED` if nothing survives
2. **health** → `MODEL_FAILED` (runtime unreachable)
3. **verification** — `required` with no validator → `INSUFFICIENT_QUALITY`
4. **quality** — profile quality for the capability (else `general`) ≥ floor → `INSUFFICIENT_QUALITY`
5. **deadline** — predicted runtime ≤ time remaining → `DEADLINE_EXCEEDED`

Survivors are ranked by `U = Q + R + V − C − L − E`. `R` (reuse) and `V` (predicted future value) are
zero until Phases 2 and 6. `C` weighs monetary cost (£0.01 ≈ 0.1 quality). `L` charges wall time only
for `NOW`/`SOON` and for plans that use more than half the remaining deadline window — deferred work is
free to be slow. `E` weighs energy by the `energy` preference. Privacy is never a term. Ties break on
predicted runtime, then model id, so planning is deterministic.

## Scheduling and execution

Waiting jobs are ordered by class, then earliest deadline, then submission. Each tick: expire waiting jobs
past their deadline; if the host is busy, stop running yield-class jobs (they return as
`RESOURCE_BLOCKED`); start jobs while slots are free; if slots are full and non-yielding work waits,
preempt the least urgent running yield-class job (it returns to `QUEUED`). A running job is stopped at
its deadline (`DEADLINE_EXCEEDED`). On shutdown, running jobs are interrupted and requeued; after a crash,
jobs found `PLANNING`/`RUNNING`/`REFINING` are requeued on start. A file lock prevents two daemons sharing
a state directory.

Input files are read by the daemon at run time and recorded with their BLAKE3 hashes in the result
(`sources`), together with the model id, runtime, model digest, exposure, token counts and the envelope's
semantic hash.

## Cognitive Store (Phase 2, §10, §15–17, §19)

Every successful job with `reuse.cache` on (the default) produces a **cognitive object**: kind, semantic
key, scope, content, confidence (the planner's expected quality), validation state (`unverified` until
Phase 5), freshness policy, the producing job, model, digest, runtime and exposure, the locality it was
produced under, the corpus sensitivity of its inputs, its sources with BLAKE3 hashes, and its parents.
Lifecycle events (`created`, `reused`, `invalidated`, `deleted`) are append-only.

**Freshness** (§17): `source_hash` (default when a task reads files), `ttl` (default otherwise, 7 days),
`event` (invalidated by tag), `manual`, `immutable`, `dependency`. Every policy except `immutable` also
goes stale when an object it used as context does. Freshness is checked **at every use** — direct reuse,
`try`, context selection, `query` — by re-hashing sources through a (size, mtime) cache that distrusts
files modified in the last two seconds; a background scan (default every 30 s) does the same so that
listings are truthful before anyone asks. Cached cognition therefore never stands in for current truth.

**Reuse** (§16), in order:

1. *Direct* — a fresh object with the same semantic key, visible from the job's scope, with confidence ≥
   the job's floor and an acceptable validation state, completes the job `READY` with no planning, no
   inference and no boundary record. The job's result names `reused_from`.
2. *Refresh* — the newest stale object with the same key is shown to the model as the previous answer,
   with the list of sources that changed since.
3. *Context* — up to three fresh, related objects (score = 0.6 × BM25 relevance + 0.4 × source overlap,
   ≥ 0.25) are shown to the model; the new object records them as parents.

Context never widens where data goes and never narrows the job: an object is offered only if its
producing locality permits the job's (`LOCAL_ONLY` jobs may see anything), and sensitive objects only to
jobs that are already `LOCAL_ONLY`.

**Invalidation** cascades through `context` parents; every affected job whose result was, or was served
from, an invalidated object moves `READY → STALE`. `tonyc result` still prints a stale result but exits
3. Invalidate by object, path (file or directory), scope or event tag.

**Scopes** are `namespace[:name]` over the §10 namespaces. Objects are visible to jobs in the same scope;
plain `user` objects are visible to every non-ephemeral job.

**Corpus** entries are tracked files (directories are walked, skipping hidden entries, `target`,
`node_modules`) with scope, sensitivity, hash, size and mode. `sensitive` and `secret` sources
structurally narrow any job that reads them to `LOCAL_ONLY`, recorded in the trace. `secret` content is
withheld from `query`/`try` listings. `corpus forget` stops tracking, deletes derived objects, redacts
the results built on them and marks dependants stale.

**Demand** (`try_hit`, `try_miss`, `reuse_hit`, `inference`, `query`) is logged append-only as evidence
for economics (Phase 4) and Preemptive Dynamism (Phase 6).

## SSH compute fabric (Phase 3, §8)

**Transport.** The system OpenSSH client, nothing custom. Every invocation carries `BatchMode=yes`,
`StrictHostKeyChecking=yes`, `UserKnownHostsFile=<state>/known_hosts`, `GlobalKnownHostsFile=/dev/null`,
`UpdateHostKeys=no`, `ForwardAgent=no`, `ForwardX11=no`, `ClearAllForwardings=yes`,
`PermitLocalCommand=no`, `RequestTTY=no` and a connect timeout; the target is passed after `--`. A node
receives the request on stdin and nothing that reaches back into this host. The user's `~/.ssh/config`
still applies (aliases, users, identities); the forced options override it.

**Host keys.** `node add` resolves the target with `ssh -G`. If no fingerprint is given, the host must
already be in the user's `known_hosts`, and those entries are pinned. Otherwise the host's keys are
scanned and only one matching the given `SHA256:` fingerprint is pinned; a mismatch, or no fingerprint
for an unknown host, is refused and the presented fingerprints are listed. Tonyc's `known_hosts` holds
only pinned node keys and is rewritten when nodes change; a node that fails its first probe is unpinned.

**Nodes.** A probe (one `sh` script over SSH) reports architecture, cores, memory, load, GPUs
(`nvidia-smi`), whether `curl` exists, and the models of the runtime on the node's loopback
(`runtime_url`, default `http://127.0.0.1:11434`). Inference runs as `curl` on the node; nothing is
installed there. Node models are `node.<name>/<model>` with static priors (×4 generation speed when a
GPU holds the model) until Phase 4 measures them. Nodes persist in the state database, are probed on
start, every `fabric.probe_interval_secs` (10 s while down), and on `tonyc node probe`.

**Policy.** A node's exposure is `Ssh{node}`. The daemon ceiling's SSH allowlist is the configured one
plus every registered node — registering is the authorising act — but its class must also be raised
(`tonyc policy set --max-locality ssh_allowlist`, persisted, shown with its source in `policy show`),
and each job must ask: `SSH_ALLOWLIST` naming the node, or `UNRESTRICTED`. A `shared` node is used only
when named, even by `UNRESTRICTED` jobs. Sensitive corpus sources still force `LOCAL_ONLY`.

**Dispatch.** Execution slots are per domain: `daemon.workers` for `local`, `slots` per node. The
scheduler starts jobs while total capacity allows; the planner prefers candidates whose domain has a
free slot (busy ones are rejected at stage `capacity` unless every admissible domain is busy, in which
case the job waits for its chosen domain's slot).

**Failure (G7).** SSH exit 255, a timeout, or the node's runtime being unreachable (`curl` exit 6, 7, 28,
127) mark the node down and raise `NodeUnreachable`. With `resources.on_node_failure: fallback` (the
default) the job returns to `PLANNING` (event `fallback`) and re-plans without that node under the same
locality rules; if nothing admissible remains it ends `NODE_UNREACHABLE` with both reasons. With `fail`
it ends `NODE_UNREACHABLE` at once. A node already known to be down is simply not planned onto.

**Boundary audit.** Each dispatch is recorded before it is sent, with the BLAKE3 of the exact bytes
the runtime transmits (`ModelAdapter::encode`); when the exposure leaves the host the bytes themselves
are kept (`tonyc audit --payload`). Failed dispatches are recorded too, since the request may have been
partly sent. `GET /v1/audit/verify` (used by `tonyc doctor`) re-checks every off-host crossing against
the locality policy of the job that made it, after any sensitivity narrowing.

## Economics (Phase 4, §12, §20, §21, §24)

**Measured profiles.** Every completed inference appends an observation: tokens in and out, wall time
and, from Ollama (local or on a node), load, prefill and generation durations. Over a model's last 50
runs Tonyc takes the median generation and prefill speeds, the 75th-percentile overhead of *cold*
runs (load > 1 s) and the median overhead of *warm* ones. After three runs these replace the static
priors (`source: observed`). Which models are warm comes from the runtime itself (`/api/ps`, on each
refresh or probe) and from runs (a model that just ran stays warm for four minutes), so the planner
predicts with the warm or the cold start as appropriate. Configuration overrides still win.

**Benchmarks.** `tonyc model benchmark` queues the built-in suite (11 cases across general,
summarisation, classification, extraction and Rust) as ordinary jobs pinned to the model
(`task.pin_model`), reuse off, `IDLE` by default — so locality, pressure, slots and the boundary audit
all apply. Each case has a deterministic check (required/forbidden substrings, length, JSON fields); an
answer that fails the output contract counts as a failed case. A capability's quality becomes
`(prior × 4 + passed) / (4 + total)`: four pseudo-observations of the prior, so a few cases move it but
cannot swing it to 0 or 1. Priors come from the profile before any blending.

**Planning.** New stages: *pin*; *resources* — a model that is not loaded and needs more memory than the
memory budget or the host (or 90% of a node) is skipped, and if no model fits the job waits
`RESOURCE_BLOCKED` with backoff rather than failing; *pressure* — work that yields does not run on a
pressured domain (this host: load, memory, battery, GPU utilisation, temperature; a node: load per
core from its probe; a local model that needs more than the memory now available also counts); and
*budget* — estimated spend must fit what is left of the monthly budget. Background work blocked by this
host's pressure may still start if an idle node it is allowed to use exists. The utility's R term is
now live: reuse potential is `min(1, prior demand for this semantic key / 3)` and adds `0.25 × R × Q`.

**Ledger.** One append-only row per completed job: class, model, domain, tokens, wall time, energy
(`watts × wall time`, watts by domain kind and GPU), its cost at the electricity price, actual spend
(profile price × tokens), and the conventional cost — the same tokens at the reference API's prices.
A job served from the store is credited with the conventional cost of the job that produced the
answer, at zero energy. `tonyc economics` reports jobs by kind, calls avoided, estimated cost avoided
(conventional − spend), operating cost (energy + spend), the **Inference Avoidance Ratio** (avoided ÷
operating), compute hours by domain, deadline misses, budgets used, and says which §24 metrics await
Phase 6.

**Budgets.** `budgets.remote_monthly_gbp` (default 0: paid runtimes are never chosen),
`budgets.daily_compute` per urgency class (a class over its budget waits until the next UTC day), and
`budgets.memory_max_mb`.

## Refinement (Phase 5, §14)

**Confidence** is the probability an answer is correct. A draft starts at its model's expected quality
for the capability. A deterministic check (`quality.checks`: must contain / must not contain / maximum
length / JSON fields; plus the `json/v1` contract) that fails proves the answer wrong (confidence 0). A
validator verdict updates confidence by Bayes' rule with the validator's sensitivity (passes correct
answers) and specificity (fails wrong ones); an unparseable reply abstains and changes nothing. The
posterior is capped at 0.99.

**Validators are measured.** Benchmark *judge* cases give a model a case's known-good and known-bad
answer under the exact prompt validators receive; results become `validation.sensitivity` and
`validation.specificity` (blended with the prior like any capability). Unmeasured models are assumed to
judge as well as their general quality. A validator whose sensitivity + specificity ≤ 1.2 carries too
little information and is not used. A model never validates its own answer; validators are assumed
independent of the drafter and of each other.

**Strategy.** Before any inference: if `refine` is `always`, verification is `required`, or no single
model meets the floor, Tonyc plans a pipeline — otherwise one pass, refined only if a check or the JSON
contract fails. A pipeline must be able to reach the floor: the strongest admissible answerer, passed by
the two best validators other than it; if even that falls short, or verification is required with no
informative validator and no checks, the job ends `INSUFFICIENT_QUALITY` at once. A pipeline drafts by
utility with the floor relaxed — cheaply — and earns its confidence.

**The loop** (state `REFINING`): checks → up to two validators (a third as tie-break when they split) →
if the floor and any verification requirement are met, `READY` (`verified` when the majority of verdicts
passed, or when the job's checks passed and no validator was involved); otherwise revise, with the
checks' and validators' issues in the prompt, by the best model not yet tried that is expected to do
better (escalation), else the best untried, else the same model; repeat up to `max_rounds` (default
2). Every pass is planned with its model pinned — locality, health, resources, budget and the deadline
all apply — holds a slot only while it runs, is in the boundary audit, and is measured. If the
deadline, budget or available models stop refinement, or the rounds run out, the job ends
`INSUFFICIENT_QUALITY` with its best attempt, confidence and the full timeline (`passes`, append-only).
`refine: never` still enforces checks; it only forbids revision.

**Accounting.** The ledger charges every pass's energy and spend; the conventional cost credited is one
reference-API call of the draft's size. A failed refinement is recorded as `kind: failed` with nothing
avoided. Direct reuse also respects refinement: an unverified object never satisfies a request that
requires verification, and an object must pass the new request's checks.

## Preemptive Dynamism (Phase 6, §9, §24)

**Evidence.** Only Tonyc's demand log: each `try` (hit or miss), each inference an agent asked for, each
answer served from the store. PD's own runs are logged as `speculative` and benchmarks as `benchmark`;
neither counts as demand, so PD cannot feed itself. Every request's envelope is kept as a template
(keyed by semantic hash; pinned and cache-off jobs excluded) so that a prediction re-asks exactly what
was asked — same objective, inputs, quality, locality and scope.

**Prediction.** Each evaluation (every `pd.interval_secs`, or `tonyc pd evaluate`) settles outcomes and
then considers every request asked within `pd.window`:

1. asked fewer than `pd.min_demand` times → skip;
2. a fresh answer exists (checked live) that will not expire within a quarter of the horizon → skip;
   already being prepared, or prepared and still fresh → skip;
3. P(asked within the horizon) = 1 − exp(−λ·horizon), λ = asks ÷ window; P below the threshold → skip;
4. value = the conventional cost recorded the last time it ran (else estimated from its size); cost =
   the planner's prediction for running it now (energy at the electricity price, plus spend); if it
   cannot be planned now (policy, quality, resources) → skip, saying why; if P × value < `value_ratio`
   × cost → skip;
5. the rest are ranked by P × value − cost and submitted up to `max_pending` as `SPECULATIVE` jobs
   (their V utility term is P), so they yield to pressure and urgent work, are preempted first, and are
   bounded by the speculative daily budget (`pd.daily_compute` unless `budgets.daily_compute` sets one).

**Outcomes.** A prediction is `QUEUED` until its job ends; `PRODUCED` when it leaves a reusable answer;
`CONSUMED` (a PD hit) when that answer serves a job or a `try` — counted even if that happens before
the next settlement; `DISCARDED` if the answer goes stale, is deleted, or is not asked for within the
horizon; `FAILED` otherwise. Preemption Yield = consumed ÷ produced, reported with speculative compute
used and discarded (`tonyc pd status`, `tonyc economics`). Once five predictions have settled, the
probability threshold becomes `min_probability ÷ max(settled yield, 0.25)`, capped at 1: while its work
goes unused, PD gets pickier until it stops.

`tonyc pd explain` lists what is being prepared, the evidence and arithmetic behind each decision,
recent outcomes, and why everything else in the last evaluation was not prepared.

## Ecosystem (Phase 7, §5, §11, §18–19, §27–28)

**The protocol is the product boundary.** `tonyc/1` is HTTP/1.1 and JSON on the daemon's Unix socket,
and everything — the Tov CLI, the Rust SDK, the MCP server, the Python client, the Lucid adapter —
speaks only that. `docs/PROTOCOL.md` describes it and `protocol/tonyc.openapi.json` is the agent
contract in OpenAPI 3.1. They cannot drift silently: `tests/conformance.rs` validates live daemon
responses (successes and errors) against the document's schemas, checks that the documented request
schema has exactly the fields `WorkRequest` accepts, and fails if the daemon routes an endpoint that is
in neither the OpenAPI document nor the operator table of `PROTOCOL.md`. `GET /v1/protocol` names the
protocol, its revision and its features; revisions only add.

**Orchestrator primitives** (§28: Tonyc as a compute-economic primitive several orchestrators share):

- *Idempotency.* `Idempotency-Key` on submit: same key and same parsed request → the original job
  (`200`, `replayed`); same key, different request → `409`. Check and insert are one immediate
  transaction, keys live 24 hours in the state database, and survive restarts.
- *The event feed.* `GET /v1/events?after=` reads the append-only job event log with a cursor, long-polls
  with `wait_ms`, filters by origin. A cursor is a sequence number in the database, so it survives
  restarts of either side; an orchestrator follows all its work with one long poll.
- *Attribution.* `origin {system, run, agent}` on a request is recorded in the envelope but excluded
  from the semantic hash, so it never changes what is computed or what may be reused — one system's
  cognition serves another's identical question. Jobs, events and the economics report (`by_origin`)
  carry it.

**Agent adapters.** The Rust SDK covers the §18 contract and the §19 `try`, with `try_or` for the
three-plane pattern of §27 (existing cognition, else the caller's own hot inference). `tonyc-mcp` puts
the same contract in front of any MCP agent as seven tools; it attributes work to the connecting
client, refuses `REMOTE_ALLOWLIST`/`UNRESTRICTED` (an agent cannot ask for remote inference through it),
and reports a missing daemon as a tool error rather than failing the session.

**Runtime adapters** (§11, G5). Two adapters cover "llama.cpp, ONNX Runtime, Candle and other local
engines" without linking any of them:

- `openai`: any server with `/v1/models` and `/v1/chat/completions`. Like the Ollama adapter it resolves
  the URL, classifies every address (loopback, private, public) and connects to that exact address. An
  optional bearer token is read from an environment variable. Priors come from the parameter count in
  the model name (`7B`, `1b`), then measurement replaces them; llama.cpp's `timings` are used when
  present.
- `command`: a program run per request with `tonyc.runtime/1` on stdin/stdout (`text` or `json`). The
  prompt never appears in its arguments, its environment is cleared, and it is killed on cancellation or
  timeout. Tonyc cannot see inside a program, so the operator declares its exposure (`local`,
  `lan:<host>`, `remote:<host>`) and the locality policy treats it accordingly — a test proves a program
  declared `remote:` is never started for `LOCAL_ONLY` work.

**Lucid** (`integrations/lucid`). `TonycFirst` wraps any Lucid `ModelProvider`. For plain completions
(no tools, temperature below 1) it maps the request deterministically to a Tonyc question — the final
user message is the objective, earlier messages are the input, Lucid's model id is the origin agent —
asks `try` first and returns a hit with zero provider tokens; otherwise, or if Tonyc is unreachable, the
wrapped provider answers as before. `defer(request, deadline)` is the warm path: Lucid hands Tonyc a
question it will need later, and the same request is served from the store when it comes. Cognition
derived from `sensitive` or `secret` sources is never returned, because a Lucid conversation may go on
to a remote provider. It compiles against Lucid's real `ports::ModelProvider` from a checkout next to
this repository; `scripts/integrations.sh` builds and tests it and skips it when there is no checkout.
TERRI has no code yet (its checkout next to this repository is empty); it would use the same protocol.

## Acceptance gates (§30)

| gate | status | evidence |
|---|---|---|
| G1 submit deferred work | met | `g1_g8_…`, e2e |
| G2 survives restart | met (graceful and crash) | `g2_work_survives_restart`, `g2_crash_recovery_…` |
| G3 deadlines influence scheduling | met | `g3_…`: EDF order, infeasible plans fail, running jobs stopped |
| G4 LOCAL_ONLY → zero network inference | met for Phase 1 domains | `g4_g13_…`: a free, perfect remote runtime is never invoked |
| G5 replaceable runtimes | met | `ModelAdapter` with four adapters (Ollama, OpenAI-compatible, command, builtin); `g5_runtime_is_replaceable`, `tests/runtimes.rs` |
| G6 SSH node executes work | met (hermetic); live up to authentication | `g6_…` (fake ssh running the real remote command); live: real `ssh -G`, `ssh-keyscan`, pinning and strict host-key checking against this host's sshd |
| G7 SSH failure falls back per policy | met | `g7_node_failure_falls_back`, `g7_node_failure_fails_by_policy` |
| G8 complete derivation provenance | met (incl. refinement passes) | `g8_lineage_is_complete`: job, model + digest, sources, ancestors, events |
| G9 source changes invalidate cognition | met | `g9_…`, `stale_cognition_never_masquerades_between_scans`, `dependency_invalidation_cascades` |
| G10 existing cognition avoids inference | met | `g10_…`: a counted adapter is called once for two identical jobs |
| G11 background yields under pressure | met | load, memory, battery, GPU utilisation, temperature; node load; preemption; `background_work_moves_to_an_idle_node_under_host_pressure` |
| G12 reports estimated cost avoided | met | `g12_reports_cost_avoided`; `tonyc economics`; per-job trace |
| G13 no implicit remote boundary | met | ceiling defaults to `LOCAL_ONLY`; nodes need `policy set` *and* a job that asks (`g13_…`); `audit/verify` |
| G14 decisions inspectable | met for job planning | `tonyc inspect`; queue decisions are events with reasons |

## Known limitations

- **Not published.** `tonyc-core` and `tonyc-sdk` carry publishing metadata, but nothing has been
  published to crates.io (or anywhere else): publishing is an outward action left to the maintainer.
- **The Lucid adapter lives here, not in Lucid,** and builds against whatever Lucid checkout sits next
  to this repository; a change to Lucid's `ModelProvider` breaks `scripts/integrations.sh`, not Tonyc's
  own suite. Lucid does not yet construct `TonycFirst` in its provider factory — that is a change to
  Lucid.
- **Exact-question reuse only, for Lucid too.** `TonycFirst` serves a hit only for the same final
  question with the same conversation before it; paraphrases go to the provider.
- **Command runtimes** start a process per request (no warm state), and their exposure is declared,
  not observed.
- **Live run (Phase 7).** Against the development host's Ollama: through the OpenAI-compatible adapter,
  a `json/v1` job pinned to `gemma3:latest` answered `{"city": "Rome"}` in 69 s (cold load), on
  loopback; `gemma3:1b` returned valid JSON without the requested field and Tonyc refused it after two
  revisions (`INSUFFICIENT_QUALITY`). The command adapter ran `ollama run gemma3:1b` as a subprocess
  and answered "Paris" in 3 s. Through `tonyc-mcp`, `tonyc_try` served that answer in 13 ms, and a
  `tonyc_submit`/`tonyc_await` pair returned a `gemma3:latest` answer after 48 s.

- **PD's model is simple by design.** Demand is a Poisson rate per exact request (semantic key); PD does
  not generalise across similar requests, schedule around time of day, or predict requests never seen.
  The defaults (7-day window, 24-hour horizon, P ≥ 0.3) need three asks in a week before it acts.
- **Live run (Phase 6).** With the defaults, two `try` misses for a question gave P 0.25 and PD held back;
  a third gave P 0.35 (value £0.0010 against cost £0.00002), PD prepared the answer on `gemma3:1b` in
  4 s of speculative compute, and the fourth ask was served from the store in 14 ms: Preemption Yield
  1.00 on a sample of one.

- **Refinement's independence assumption.** Bayes updates treat validators as independent of the
  drafter and of each other; models from one family may share blind spots, which would make combined
  confidence optimistic. There is no separate generative "synthesis" step: disagreements are settled
  by a tie-break vote, and the validators' issues serve as the critique for revisions.
- **Refinement can cost more than it saves** on small answers: on the development host, verifying a
  16-token answer used £0.0004 of energy against a £0.00009 reference-API price. The ledger shows this.
- **Live run (Phase 5).** As judges, `gemma3:1b` failed 10 of 11 correct answers (sensitivity 0.19,
  specificity 0.79 — not informative, so never used as a validator), while `gemma3:latest` scored 0.84 /
  0.71. A `verification: required`, floor 0.8 question was drafted by `gemma3:1b` (0.72) and passed by
  `gemma3:latest`, reaching 0.88 and `verified` in 23 s.

- **Live SSH execution not yet demonstrated end to end.** On the development host, the real OpenSSH
  path was exercised through host-key verification; authentication failed because the account has no
  `authorized_keys`. Inference over SSH is covered by tests whose `ssh` runs the real remote command
  locally.
- **Node runtimes.** Only an Ollama-compatible runtime reachable with `curl` on the node. Node load is
  probed and shown but not yet used for scheduling (Phase 4 is resource-aware scheduling).
- **Fallback scope.** Fallback applies to node failures at dispatch. A local runtime failing ends the
  job `MODEL_FAILED` as before.

- **Retrieval is lexical.** Relatedness is BM25 over objectives and content plus source overlap; there
  is no embedding model yet, so paraphrases with no shared terms are not found. The `Doc`/`rank`
  interface is where an embedding retriever plugs in.
- **Context adds tokens rather than replacing inputs.** Prior cognition is offered alongside the inputs;
  substituting a summary for a large unchanged input is not attempted. Its quality effect is unmeasured
  until Phase 4.
- **`forget` scope.** It deletes derived objects and redacts derived results, but a job's envelope (its
  objective and any inline `input.text`) stays in the job history.

- **Estimates, stated.** Cost avoided depends on the reference prices and wattages in `[economics]`;
  energy is wall time × an assumed wattage, not a measurement. The live numbers in this repository's
  history are from a CPU-only laptop.
- **Small benchmark.** Eleven cases, two to four per capability; quality estimates carry their sample
  sizes and lean on the prior until more evidence accumulates. Code capabilities beyond a token Rust
  pair are not measured.
- **Memory checks use estimated model sizes** (download size × 1.2 until overridden with `ram_mb`).
- **Live run (Phase 4).** On the development host, benchmarking `gemma3:1b` took 17 s for 11 cases
  (9 passed; both Rust cases failed); its measured generation speed was 22.6 tok/s against a prior of
  40, prefill 65 tok/s against 320, and its `rust` quality fell to 0.30. `gemma4` (≈11 GB) was skipped
  on the 7.5 GB host. Over the twelve jobs, the Inference Avoidance Ratio was 8.4×.
- **No encryption at rest** (§22). State is protected by file permissions only (directory 0700, database
  and socket 0600).
- **Linux only** for host discovery; pressure ignores GPU utilisation and thermal state.
- **Input failures** (a file deleted between submit and run) end as `MODEL_FAILED` with stage `input`;
  the canonical state set has no closer match.
- **Live run.** On this host, a dependency analysis of a `Cargo.toml` took 1 m 31 s on `gemma3`; the
  identical request was then served from the store in 0.12 s, and the related "which dependencies are
  unnecessary?" request received the earlier analysis as context (relevance 0.76).
- **Tov 0.0.1 workarounds** in `cli/`: `fetch`/`Response` are imported from `std/http` explicitly (web
  globals are only injected when the entry file names them), and `doctor` avoids `.filter` with arrow
  functions (a codegen bug with two differently typed arrow parameters in one async function).
