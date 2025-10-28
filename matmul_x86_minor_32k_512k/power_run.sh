#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURATION ===
GEM5_BIN="$HOME/gem5_fast/build/ALL/gem5.fast"
SE_SCRIPT="$HOME/gem5_fast/configs/deprecated/example/se.py"
WORKLOAD="$HOME/gem5_fast/benchmarks/bin/matmul_x86"
OUTDIR="$PWD/output"

# Paths for McPAT + Parser
MCPAT_BIN="$HOME/MCPAT_versions/mcpat_original/mcpat"
PARSER_DIR="$HOME/Gem5McPATParser"
TEMPLATE="$PARSER_DIR/templates/template_x86_fixed.xml"

# === SANITY CHECKS ===
[[ -x "$GEM5_BIN" ]] || { echo "gem5 binary not found"; exit 1; }
[[ -x "$MCPAT_BIN" ]] || { echo "McPAT binary not found"; exit 1; }
[[ -f "$SE_SCRIPT" ]] || { echo "SE script not found"; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "Workload binary not found"; exit 1; }

mkdir -p "$OUTDIR"

# === GEM5 RUN ===
# echo "[INFO] Running gem5 simulation..."
# "$GEM5_BIN" \
#   --outdir="$OUTDIR" \
#   --stats-file="$OUTDIR/stats.txt" \
#   --dump-config="$OUTDIR/config.ini" \
#   "$SE_SCRIPT" \
#   --cpu-type=X86O3CPU \
#   --num-cpus=1 \
#   --caches \
#   --l1i_size=32kB \
#   --l1d_size=32kB \
#   --l2cache \
#   --l2_size=512kB \
#   --cpu-clock=1GHz \
#   -c "$WORKLOAD"

# echo "[INFO] gem5 run completed"

# # === CONVERT config.ini → config.json ===
# echo "[INFO] Generating config.json..."
# python3 "$HOME/gem5/util/config_to_json.py" "$OUTDIR/config.ini" > "$OUTDIR/config.json"
# echo "[INFO] config.json generated at $OUTDIR/config.json"

# === CONVERT GEM5 OUTPUT TO McPAT XML ===
# echo "[INFO] Converting gem5 config + stats to McPAT XML..."
# python3 "$PARSER_DIR/Gem5McPATParser.py" \
#   --config "$OUTDIR/config.json" \
#   --stats "$OUTDIR/stats.txt" \
#   --template "$TEMPLATE" \
#   --output "$OUTDIR/mcpat.xml"

# echo "[INFO] XML generated at $OUTDIR/mcpat.xml"

# # === RUN MCPAT ===
echo "[INFO] Running McPAT..."
"$MCPAT_BIN" -infile "$OUTDIR/mcpat.xml" -print_level 5 > "$OUTDIR/mcpat_power.txt"

# echo "[INFO] McPAT power report generated at $OUTDIR/mcpat_power.txt"

# # === EXTRACT TOTAL POWER ===
# echo "[INFO] Power summary:"
# grep -E "Processor:|Total Leakage|Runtime Dynamic|Peak Power|Total Power" "$OUTDIR/mcpat_power.txt" || true
