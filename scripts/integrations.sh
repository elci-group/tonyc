#!/usr/bin/env bash
# Orchestrator integrations that build against another project's checkout.
# Each is skipped (not failed) when that checkout is absent, so Tonyc's own
# suite (scripts/check.sh) never depends on them.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

lucid="$root/../dreamsequence/lucid/platform"
if [ -f "$lucid/Cargo.toml" ]; then
  echo "== Lucid ($lucid)"
  (cd "$root/integrations/lucid" && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test)
else
  echo "== Lucid: skipped (no checkout at $lucid)"
fi
