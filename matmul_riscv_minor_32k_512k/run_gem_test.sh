#!/usr/bin/env bash
set -euo pipefail

# --- Path settings ---
GEM5_BIN="$HOME/gem5/build/ALL/gem5.opt"
SCRIPT="$HOME/gem5/configs/deprecated/example/se.py"

# Current benchmark folder
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"

# RISC-V workload binary path
WORKLOAD="$BENCH_DIR/matmul_rv64"

# Output folder inside the benchmark directory
OUTDIR="$BENCH_DIR/output"

# --- Run gem5 ---
mkdir -p "$OUTDIR"

$GEM5_BIN --outdir="$OUTDIR" "$SCRIPT" \
  --cpu-type=RiscvO3CPU \
  --num-cpus=1 \
  --caches \
  --l1i_size=32kB \
  --l1d_size=32kB \
  --l2cache \
  --l2_size=512kB \
  --cpu-clock=1GHz \
  -c "$WORKLOAD"
