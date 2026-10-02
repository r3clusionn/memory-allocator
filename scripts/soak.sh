#!/usr/bin/env bash
# Runs the whole test suite repeatedly and keeps the full output of any failing run.
# Usage: scripts/soak.sh [RUNS] [OUTDIR]
set -u
cd "$(dirname "$0")/.."
runs=${1:-100}
out=${2:-target/soak}
mkdir -p "$out"
export CARGO_TARGET_DIR=target/soak-build
cargo test --release --no-run >/dev/null 2>&1
fails=0
for i in $(seq 1 "$runs"); do
  if ! cargo test --release > "$out/run.txt" 2>&1; then
    fails=$((fails + 1))
    cp "$out/run.txt" "$out/fail_$i.txt"
  fi
done
echo "$runs runs, $fails failed"
