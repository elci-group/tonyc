#!/usr/bin/env bash
# End-to-end test: the Tov `tonyc` CLI against a real `tonycd` daemon using
# only the deterministic built-in runtime (no model needed).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tonycd="$root/target/debug/tonycd"
tonyc="$root/cli/tonyc"
work="$(mktemp -d)"
trap 'kill "$pid" "${ollama_pid:-}" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; rm -rf "$work"' EXIT

cat > "$work/tonycd.toml" <<TOML
[daemon]
socket = "$work/run/tonycd.sock"
data_dir = "$work/data"
tick_ms = 20
[pressure]
disabled = true
[cognition]
scan_interval_secs = 3600   # scans in this script are explicit (tonyc cache scan)
[pd]
interval_secs = 3600        # evaluations in this script are explicit (tonyc pd evaluate)
window = "1h"
horizon = "1h"
[runtimes.ollama]
enabled = false
[runtimes.builtin]
enabled = true
quality = 0.6
TOML
export TONYC_SOCKET="$work/run/tonycd.sock"

start() {
  "$tonycd" --config "$work/tonycd.toml" 2>>"$work/daemon.log" &
  pid=$!
  for _ in $(seq 100); do [ -S "$TONYC_SOCKET" ] && return; sleep 0.05; done
  echo "daemon did not start"; cat "$work/daemon.log"; exit 1
}

pass=0
check() { # check <name> <expected-exit> <expected-substring> -- command...
  local name="$1" want_exit="$2" want="$3"; shift 4
  local got_exit=0 got
  got="$("$@" 2>&1)" || got_exit=$?
  if [ "$got_exit" != "$want_exit" ] || [[ "$got" != *"$want"* ]]; then
    echo "FAIL $name: exit $got_exit (want $want_exit)"; echo "$got" | sed 's/^/  | /'; exit 1
  fi
  echo "ok   $name"; pass=$((pass + 1))
}

start
check "doctor" 0 "admissible models" -- "$tonyc" doctor
check "status" 0 "ceiling    LOCAL_ONLY" -- "$tonyc" status
check "submit --wait" 0 "hello from the queue" -- "$tonyc" submit --objective "hello from the queue" --wait --timeout 10s

echo "alpha beta gamma" > "$work/input.txt"
id="$("$tonyc" submit --objective summarise --file "$work/input.txt" --deadline 5m)"
check "await" 0 "alpha beta gamma" -- "$tonyc" await "$id" --timeout 10s
check "inspect trace" 0 "Remote data transmitted:" -- "$tonyc" inspect "$id"
check "inspect source hash" 0 "source $work/input.txt" -- "$tonyc" inspect "$id"
check "inspect --json" 0 "\"semantic_hash\"" -- "$tonyc" inspect "$id" --json
check "prefix ids" 0 "alpha beta gamma" -- "$tonyc" result "${id:0:10}"
check "jobs" 0 "READY" -- "$tonyc" jobs

printf 'task:\n  objective: from a file\nlatency:\n  class: soon\n' > "$work/req.yaml"
check "submit file" 0 "from a file" -- "$tonyc" submit "$work/req.yaml" --wait
check "submit stdin" 0 "from stdin" -- sh -c "printf 'task: {objective: from stdin}' | '$tonyc' submit - --wait"
check "json schema" 0 '{"echo"' -- "$tonyc" submit --objective "give json" --schema json/v1 --wait

check "floor too high" 3 "INSUFFICIENT_QUALITY" -- "$tonyc" submit --objective x --floor 0.99 --wait
check "bad deadline" 1 "latency.deadline" -- "$tonyc" submit --objective x --deadline soonish
check "unknown option" 2 "unknown option" -- "$tonyc" jobs --frob
check "unknown command" 2 "unknown command" -- "$tonyc" frobnicate
check "unknown job" 1 "no job" -- "$tonyc" inspect tny_doesnotexist

# Cognitive Store (Phase 2).
check "try miss" 5 "no existing cognition" -- "$tonyc" try --objective "explain the release process"
"$tonyc" submit --objective "explain the release process" --wait > /dev/null
check "try hit" 0 "explain the release process" -- "$tonyc" try --objective "explain the release process"
again="$("$tonyc" submit --objective "explain the release process" --class now)"
"$tonyc" await "$again" > /dev/null
check "reuse without inference" 0 "satisfied by existing cognition" -- "$tonyc" inspect "$again"
check "query" 0 "explain the release process" -- "$tonyc" query release process
check "status reuse" 0 "1 job(s) served without inference" -- "$tonyc" status

echo "serde = 1" > "$work/Cargo.toml"
check "corpus add" 0 "tracking 1 file(s) in project:e2e (sensitive)" -- "$tonyc" corpus add "$work/Cargo.toml" --scope project:e2e --sensitivity sensitive
check "corpus inspect" 0 "project:e2e" -- "$tonyc" corpus inspect
deps="$("$tonyc" submit --objective "list the dependencies" --file "$work/Cargo.toml" --scope project:e2e --locality unrestricted)"
"$tonyc" await "$deps" > /dev/null
check "sensitive narrows locality" 0 "locality narrowed UNRESTRICTED → LOCAL_ONLY" -- "$tonyc" inspect "$deps"
echo "serde = 1
tokio = 1" > "$work/Cargo.toml"
check "scan invalidates" 0 "jobs now STALE: $deps" -- "$tonyc" cache scan
refresh="$("$tonyc" submit --objective "list the dependencies" --file "$work/Cargo.toml" --scope project:e2e)"
"$tonyc" await "$refresh" > /dev/null
check "refresh from stale" 0 "refreshing stale obj_" -- "$tonyc" inspect "$refresh"
obj="$("$tonyc" inspect "$refresh" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["object_id"])')"
check "object lineage" 0 "refresh_of" -- "$tonyc" cache inspect "$obj"
check "stale listing" 0 "STALE" -- "$tonyc" cache inspect --stale
check "prune" 0 "pruned" -- "$tonyc" cache prune
"$tonyc" submit --objective "tagged" --freshness event --event deps-updated --wait > /dev/null
check "invalidate by event" 0 "invalidated 1 object(s)" -- "$tonyc" invalidate --event deps-updated
check "corpus forget" 0 "deleted 1 derived object(s)" -- "$tonyc" corpus forget project:e2e
check "forgotten result" 3 "[forgotten" -- "$tonyc" result "$refresh"

# SSH compute fabric (Phase 3), with a fake ssh that runs the remote command
# here, a real generated host key, and a tiny fake Ollama as the node runtime.
ssh-keygen -q -t ed25519 -N "" -f "$work/hostkey"
read -r ktype kdata _ < "$work/hostkey.pub"
echo "[127.0.0.1]:2222 $ktype $kdata" > "$work/hostkey.line"
fp="$(ssh-keygen -lf "$work/hostkey.pub" | awk '{print $2}')"
cat > "$work/fake-ssh" <<SH
#!/usr/bin/env bash
for a in "\$@"; do [ "\$a" = "-G" ] && { printf 'hostname 127.0.0.1\nport 2222\n'; exit 0; }; done
[ -e "$work/DOWN" ] && { echo "ssh: connect to host 127.0.0.1 port 2222: Connection refused" >&2; exit 255; }
while [ \$# -gt 0 ]; do case "\$1" in -o|-p|-i) shift 2;; --) shift; break;; -*) shift;; *) break;; esac; done
shift; exec sh -c "\$1"
SH
printf '#!/bin/sh\ncat %s\n' "$work/hostkey.line" > "$work/fake-keyscan"
chmod +x "$work/fake-ssh" "$work/fake-keyscan"
cat > "$work/fake-ollama.py" <<'PY'
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self.reply({"models": [{"name": "fake:1b", "size": 800000000, "digest": "d", "details": {"parameter_size": "1B"}}]})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.reply({"response": "node says: " + req["prompt"].splitlines()[0], "prompt_eval_count": 5, "eval_count": 3})
s = http.server.HTTPServer(("127.0.0.1", 0), H)
print(s.server_address[1], flush=True)
s.serve_forever()
PY
python3 -I "$work/fake-ollama.py" > "$work/ollama.port" &
ollama_pid=$!
for _ in $(seq 50); do [ -s "$work/ollama.port" ] && break; sleep 0.05; done
runtime="http://127.0.0.1:$(cat "$work/ollama.port")"
kill "$pid"; wait "$pid" || true
cat >> "$work/tonycd.toml" <<TOML
[fabric]
ssh_program = "$work/fake-ssh"
keyscan_program = "$work/fake-keyscan"
user_known_hosts = "$work/user_known_hosts"
connect_timeout_secs = 2
[models."node.forge/fake:1b"]
capabilities = { general = 0.9 }
TOML
start

check "node add needs a trusted key" 1 "not trusted yet" -- "$tonyc" node add forge --ssh tester@forge --runtime-url "$runtime"
check "node add" 0 "host key pinned; healthy" -- "$tonyc" node add forge --ssh tester@forge --runtime-url "$runtime" --host-key "$fp"
check "node add warns about ceiling" 0 "not usable yet" -- "$tonyc" node probe forge
check "nodes" 0 "forge        up" -- "$tonyc" nodes
blocked="$("$tonyc" submit --objective "on the node" --locality ssh_allowlist --ssh-allow forge --floor 0.3)"
"$tonyc" await "$blocked" > /dev/null
check "node blocked by ceiling" 0 "node.forge/fake:1b [locality]: daemon ceiling" -- "$tonyc" inspect "$blocked"
check "policy set" 0 "ceiling           SSH_ALLOWLIST   (set)" -- "$tonyc" policy set --max-locality ssh_allowlist
remote_job="$("$tonyc" submit --objective "summarise on the node" --locality ssh_allowlist --ssh-allow forge --floor 0.3)"
check "runs on the node" 0 "node says: summarise on the node" -- "$tonyc" await "$remote_job"
check "trace shows the node" 0 "SSH node forge" -- "$tonyc" inspect "$remote_job"
check "audit payload" 0 '"prompt":"summarise on the node' -- "$tonyc" audit --job "$remote_job" --payload
touch "$work/DOWN"
check "fallback to local" 0 "fallback" -- sh -c "'$tonyc' submit --objective 'node is down' --locality ssh_allowlist --ssh-allow forge --floor 0.3 --wait >/dev/null && '$tonyc' inspect \$('$tonyc' jobs --json --limit 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"jobs\"][0][\"id\"])')"
rm -f "$work/DOWN"
check "node recovers on probe" 0 "forge        up" -- "$tonyc" node probe forge
touch "$work/DOWN"
check "fail by policy" 3 "NODE_UNREACHABLE" -- "$tonyc" submit --objective "strict" --locality ssh_allowlist --ssh-allow forge --floor 0.3 --on-node-failure fail --wait
rm -f "$work/DOWN"
check "node remove" 0 "removed forge" -- "$tonyc" node remove forge
check "policy reset" 0 "ceiling           LOCAL_ONLY   (config)" -- "$tonyc" policy set --reset
kill "$ollama_pid"

# Economics (Phase 4).
check "models show profiles" 0 "builtin/echo" -- "$tonyc" models
check "benchmark runs" 0 "COMPLETE" -- "$tonyc" model benchmark builtin/echo --capability general --class now --wait --timeout 60s
check "benchmarks listed" 0 "builtin/echo" -- "$tonyc" benchmarks
check "model inspect" 0 "benchmarks 0/4" -- "$tonyc" model inspect builtin/echo
check "measured profile" 0 "run(s) measured" -- "$tonyc" model inspect builtin/echo
check "economics" 0 "Inference Avoidance Ratio" -- "$tonyc" economics
check "economics window" 0 "over the last 1h" -- "$tonyc" economics --since 1h
check "trace economics" 0 "Economics (estimates)" -- "$tonyc" inspect "$id"

# Refinement (Phase 5). The echo model repeats its prompt, so a check for
# a word it must not contain fails on every revision. Its benchmark above
# measured its quality at 0.30, so these jobs use a lower floor.
check "unreachable floor fails up front" 3 "would reach only" -- "$tonyc" submit --objective "say paris" --check-contains paris --wait
check "checks pass" 0 "say paris" -- "$tonyc" submit --objective "say paris" --check-contains paris --floor 0.2 --wait
stuck="$("$tonyc" submit --objective "say london" --check-absent london --max-rounds 1 --floor 0.2)"
check "unfixable check fails" 3 "INSUFFICIENT_QUALITY" -- "$tonyc" await "$stuck" --timeout 30s
check "refinement timeline" 0 "revise" -- "$tonyc" inspect "$stuck"
check "best attempt kept" 0 '"best_attempt"' -- "$tonyc" inspect "$stuck" --json
check "no validator, no verification" 3 "no informative validator" -- "$tonyc" submit --objective "verify me" --verification required --wait
check "refine never still checks" 3 "refinement disabled" -- "$tonyc" submit --objective "say rome" --check-absent rome --refine never --floor 0.2 --wait

# Preemptive Dynamism (Phase 6): two try misses are demand; PD prepares the
# answer, and the third ask finds it.
"$tonyc" try --objective "weekly status digest" --floor 0.2 > /dev/null 2>&1 || true
"$tonyc" try --objective "weekly status digest" --floor 0.2 > /dev/null 2>&1 || true
check "pd pause" 0 "paused" -- "$tonyc" pd pause
check "paused pd prepares nothing" 0 "0 prepared" -- "$tonyc" pd evaluate
check "pd resume" 0 "active" -- "$tonyc" pd resume
check "pd prepares repeated demand" 0 "1 prepared" -- "$tonyc" pd evaluate
check "pd explain says why" 0 "2 try miss(es)" -- "$tonyc" pd explain
spec="$("$tonyc" jobs --state READY --limit 50 --json | python3 -c 'import json,sys; print(next(j["id"] for j in json.load(sys.stdin)["jobs"] if j["class"] == "SPECULATIVE"))' 2>/dev/null || true)"
for _ in $(seq 100); do [ -n "$spec" ] && break; sleep 0.1; spec="$("$tonyc" jobs --state READY --limit 50 --json | python3 -c 'import json,sys; print(next((j["id"] for j in json.load(sys.stdin)["jobs"] if j["class"] == "SPECULATIVE"), ""))')"; done
check "prepared answer is waiting" 0 "weekly status digest" -- "$tonyc" try --objective "weekly status digest" --floor 0.2
check "pd hit counted" 0 "1 used (PD hits)" -- "$tonyc" economics
check "pd status yield" 0 "1 of 1 prepared answer(s) used" -- "$tonyc" pd status

# Ecosystem (Phase 7): protocol negotiation, idempotent submission from an
# orchestrator, the event feed, attribution, and the other clients.
check "protocol" 0 "tonyc/1 revision" -- "$tonyc" protocol
orch="$("$tonyc" submit --objective "orchestrated work" --origin lucid:task-1 --agent cto --idempotency-key lucid-task-1 --floor 0.2)"
check "idempotent resubmit" 0 "already submitted" -- "$tonyc" submit --objective "orchestrated work" --origin lucid:task-1 --agent cto --idempotency-key lucid-task-1 --floor 0.2
check "same job on retry" 0 "$orch" -- "$tonyc" submit --objective "orchestrated work" --origin lucid:task-1 --agent cto --idempotency-key lucid-task-1 --floor 0.2
check "key reused for other work" 1 "already used for a different request" -- "$tonyc" submit --objective "other work" --idempotency-key lucid-task-1
check "orchestrated job finishes" 0 "orchestrated work" -- "$tonyc" await "$orch" --timeout 10s
check "jobs by origin" 0 "orchestrated work" -- "$tonyc" jobs --origin lucid
check "event feed" 0 "READY (terminal)  lucid:task-1" -- "$tonyc" events --system lucid
check "economics by origin" 0 "origin      lucid" -- "$tonyc" economics
check "python client" 0 "python client: all checks passed" -- python3 "$root/sdk/python/test_tonyc.py"
mcp_in='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"e2e"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"tonyc_try","arguments":{"objective":"orchestrated work","quality_floor":0.2}}}'
check "mcp agent adapter" 0 "Existing cognition" -- sh -c "printf '%s\n' '$mcp_in' | '$root/target/debug/tonyc-mcp'"

# State survives a restart: results and provenance are still there.
# (In-flight and queued work across restarts: crates/tonycd/tests/daemon.rs.)
kill "$pid"; wait "$pid" || true
"$tonycd" --config "$work/tonycd.toml" --print-config > /dev/null
start
check "after restart" 0 "alpha beta gamma" -- "$tonyc" result "$id"
check "boundary audit" 0 "0 outside their job's policy" -- "$tonyc" doctor

echo "e2e: $pass checks passed"
