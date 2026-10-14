# tonyc-sdk

The Rust SDK for [Tonyc](https://github.com/elci-group/tonyc), a latency-arbitrage and cognition-reuse
layer for agents. It speaks the `tonyc/1` protocol to a running `tonycd` over its Unix socket.

```rust
use std::time::Duration;
use tonyc_sdk::{Client, TryOutcome, Work};

let tonyc = Client::new().with_origin("my-agent");
let work = Work::new("Summarise the open incidents").deadline_in("2h").floor(0.7);

// Cold path: is the answer already known?
if let TryOutcome::Hit(object) = tonyc.try_existing(&work).await? {
    return Ok(object.content);
}
// Warm path: hand it to Tonyc and collect it later. A retry with the same
// key returns the same job.
let job = tonyc.submit_idempotent("incident-digest-2026-10-14", &work).await?;
let done = tonyc.wait(&job.id, Duration::from_secs(7200)).await?;
```

| Method | Protocol |
|---|---|
| `protocol()` | `GET /v1/protocol` — refuses anything but `tonyc/1` |
| `submit`, `submit_idempotent` | `POST /v1/jobs` |
| `status`, `await_job`, `wait` | `GET /v1/jobs/{id}`, `/await` |
| `cancel` | `POST /v1/jobs/{id}/cancel` |
| `try_existing`, `try_or` | `POST /v1/try` (and the three-plane pattern) |
| `query` | `GET /v1/cognition` |
| `invalidate` | `POST /v1/invalidate` |
| `events` | `GET /v1/events` — the durable event feed for orchestrators |

`Work` builds requests that state constraints — urgency, deadline, quality floor, locality, scope —
never models. Responses ignore fields they do not know, so the SDK keeps working against newer daemons.
The protocol is described in `docs/PROTOCOL.md` and `protocol/tonyc.openapi.json`.

MIT licensed.
