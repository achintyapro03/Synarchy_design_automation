#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURATION ===
GEM5_BIN="$HOME/gem5/build/ALL/gem5.opt"
SE_SCRIPT="$HOME/gem5/configs/deprecated/example/se.py"
WORKLOAD="$HOME/gem5/benchmarks/bin/matmul_aarch64"
OUTDIR="$PWD/output"

# Paths for McPAT + Parser
MCPAT_BIN="$HOME/mcpat/mcpat"
PARSER_DIR="$HOME/Gem5McPATParser"
TEMPLATE="$PARSER_DIR/templates/template_arm.xml"   # Make sure this exists

# === SANITY CHECKS ===
[[ -x "$GEM5_BIN" ]] || { echo "[ERROR] gem5 binary not found at $GEM5_BIN"; exit 1; }
[[ -x "$MCPAT_BIN" ]] || { echo "[ERROR] McPAT binary not found at $MCPAT_BIN"; exit 1; }
[[ -f "$SE_SCRIPT" ]] || { echo "[ERROR] SE script not found at $SE_SCRIPT"; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "[ERROR] Workload binary not found at $WORKLOAD"; exit 1; }
[[ -f "$TEMPLATE" ]] || { echo "[ERROR] ARM McPAT template not found at $TEMPLATE"; exit 1; }

mkdir -p "$OUTDIR"

# === GEM5 RUN ===
echo "[INFO] Running gem5 simulation on $(basename "$WORKLOAD")..."
"$GEM5_BIN" \
  --outdir="$OUTDIR" \
  --stats-file="$OUTDIR/stats.txt" \
  --dump-config="$OUTDIR/config.ini" \
  "$SE_SCRIPT" \
  --cpu-type=ArmO3CPU \
  --num-cpus=1 \
  --caches \
  --l1i_size=32kB \
  --l1d_size=32kB \
  --l2cache \
  --l2_size=512kB \
  --cpu-clock=1GHz \
  -c "$WORKLOAD"

echo "[INFO] gem5 run completed"


# === CONVERT GEM5 OUTPUT TO McPAT XML ===
echo "[INFO] Converting gem5 config + stats to McPAT XML..."
python3 "$PARSER_DIR/Gem5McPATParser.py" \
  --config "$OUTDIR/config.json" \
  --stats "$OUTDIR/stats.txt" \
  --template "$TEMPLATE" \
  --output "$OUTDIR/mcpat.xml"

# echo "[INFO] XML generated at $OUTDIR/mcpat.xml"

# # === RUN MCPAT ===
# echo "[INFO] Running McPAT..."
# "$MCPAT_BIN" -infile "$OUTDIR/mcpat.xml" -print_level 5 > "$OUTDIR/mcpat_power.txt"

# echo "[INFO] McPAT power report generated at $OUTDIR/mcpat_power.txt"

# # === EXTRACT TOTAL POWER ===
# echo "[INFO] Power summary:"
# if ! grep -E "Processor:|Total Leakage|Runtime Dynamic|Peak Power|Total Power" "$OUTDIR/mcpat_power.txt"; then
#   echo "[WARN] Could not find power summary lines in mcpat_power.txt"
# fi
