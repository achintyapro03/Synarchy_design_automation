#!/usr/bin/env bash
set -euo pipefail

# Source C file
SRC="$HOME/gem5/benchmarks/bin/matmul.c"
OUT_DIR=$(dirname "$SRC")
OUT="$OUT_DIR/matmul_aarch64"

echo "[INFO] Compiling $SRC → $OUT"

# Cross-compile for 64-bit ARM (AArch64)
aarch64-linux-gnu-gcc -static -O2 -march=armv8-a -o "$OUT" "$SRC"

# Check the resulting binary
file "$OUT"
