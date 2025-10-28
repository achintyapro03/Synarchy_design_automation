#!/usr/bin/env bash
set -euo pipefail

# Source C file
SRC="$HOME/gem5_fast/benchmarks/bin/matmul.c"
OUT_DIR=$(dirname "$SRC")
OUT="$OUT_DIR/matmul_x86"

echo "[INFO] Compiling $SRC → $OUT"

gcc -static -O2 -march=x86-64 -o "$OUT" "$SRC"

# Check the resulting binary
file "$OUT"
