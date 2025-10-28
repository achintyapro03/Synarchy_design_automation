#!/usr/bin/env bash
set -euo pipefail

# Compile matmul.c for RISC-V (RV64, static)
SRC="matmul.c"
OUT="matmul_rv64"

echo "[INFO] Compiling $SRC → $OUT"

riscv64-linux-gnu-gcc -static -O2 -march=rv64gc -mabi=lp64d -o "$OUT" "$SRC"

file "$OUT"
