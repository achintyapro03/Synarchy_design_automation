#!/usr/bin/env bash
set -euo pipefail

# --- Path settings ---
GEM5_BIN="$HOME/gem5/build/ALL/gem5.opt"
SCRIPT="$HOME/gem5/configs/deprecated/example/se.py"

# ARM workload binary (64-bit)
WORKLOAD="$HOME/gem5/benchmarks/bin/matmul_aarch64"

# Output folder inside the benchmark directory
OUTDIR="./output"

# --- Run gem5 ---
mkdir -p "$OUTDIR"

echo "[INFO] Running gem5 with ArmO3CPU CPU on $(basename "$WORKLOAD")..."
"$GEM5_BIN" --outdir="$OUTDIR" "$SCRIPT" \
  --cpu-type=ArmO3CPU \
  --num-cpus=1 \
  --caches \
  --l1i_size=32kB \
  --l1d_size=32kB \
  --l2cache \
  --l2_size=512kB \
  --cpu-clock=1GHz \
  -c "$WORKLOAD"

echo "[INFO] gem5 run completed. Output in $OUTDIR"
