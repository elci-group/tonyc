#!/usr/bin/env bash
# The full Tonyc test suite: Rust format, lints and tests; Tov type-check,
# unit tests and build; then the CLI-against-daemon end-to-end test.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
# Tov compiles through C. On Ubuntu, clang can select a GCC install that has
# no libstdc++; gcc always links. Override with CC=... if needed.
export CC="${CC:-gcc}"

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build

cd cli
tov check --json . | grep -q '"errors":0' || { tov check .; exit 1; }
tov test .
tov build main.tov -o tonyc
cd ..

scripts/e2e.sh
