#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURATION ===
GEM5_BIN="$HOME/gem5_fast/build/ALL/gem5.fast"
SE_SCRIPT="$HOME/gem5_fast/configs/deprecated/example/se.py"
WORKLOAD="$HOME/gem5_fast/benchmarks/bin/microbench_x86"
OUTDIR="$PWD/output"

# -----------------------------
# CONFIGURATION FROM ML MODEL
# -----------------------------
cpu_clock_GHz=1.008737
l1i_kb=16
l1d_kb=64
l1_assoc=2
l2_kb=1024
l2_assoc=2
fetchWidth=12
decodeWidth=11
issueWidth=8
commitWidth=12
numROBEntries=206
numIQEntries=94
LQEntries=56
SQEntries=37
BP="TAGE_SC_L_64KB"


# Paths for McPAT + Parser
MCPAT_BIN="$HOME/MCPAT_versions/mcpat_original/mcpat"
PARSER_DIR="$HOME/Gem5McPATParser"
TEMPLATE="$PARSER_DIR/templates/template_x86_${BP}.xml"

# === SANITY CHECKS ===
[[ -x "$GEM5_BIN" ]] || { echo "gem5 binary not found"; exit 1; }
[[ -x "$MCPAT_BIN" ]] || { echo "McPAT binary not found"; exit 1; }
[[ -f "$SE_SCRIPT" ]] || { echo "SE script not found"; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "Workload binary not found"; exit 1; }

mkdir -p "$OUTDIR"

# === GEM5 RUN ===
echo "[INFO] Running gem5 simulation..."
"$GEM5_BIN" \
  --outdir="$OUTDIR" \
  --stats-file="$OUTDIR/stats.txt" \
  --dump-config="$OUTDIR/config.ini" \
  "$SE_SCRIPT" \
  --cpu-type=X86O3CPU \
  --num-cpus=1 \
  --caches \
  --l1i_size=${l1i_kb}kB \
  --l1d_size=${l1d_kb}kB \
  --l2cache \
  --l2_size=${l2_kb}kB \
  --cpu-clock=${cpu_clock_GHz}GHz \
  --bp-type="${BP}" \
  -c "$WORKLOAD" \
  -P "system.cpu[0].fetchWidth=${fetchWidth}" \
  -P "system.cpu[0].decodeWidth=${decodeWidth}" \
  -P "system.cpu[0].issueWidth=${issueWidth}" \
  -P "system.cpu[0].commitWidth=${commitWidth}" \
  -P "system.cpu[0].numROBEntries=${numROBEntries}" \
  -P "system.cpu[0].numIQEntries=${numIQEntries}" \
  -P "system.cpu[0].LQEntries=${LQEntries}" \
  -P "system.cpu[0].SQEntries=${SQEntries}"

# === McPAT CONVERSION ===
echo "[INFO] Converting gem5 config + stats to McPAT XML..."
python3 "$PARSER_DIR/Gem5McPATParser.py" \
  --config "$OUTDIR/config.json" \
  --stats "$OUTDIR/stats.txt" \
  --template "$TEMPLATE" \
  --output "$OUTDIR/mcpat.xml"

echo "[INFO] gem5 + McPAT workflow finished."
