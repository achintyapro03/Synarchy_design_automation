#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURATION ===
GEM5_BIN="$HOME/gem5/build/ALL/gem5.opt"
SE_SCRIPT="$HOME/gem5/configs/deprecated/example/se.py"
WORKLOAD="$HOME/gem5/benchmarks/bin/matmul_rv64"
OUTDIR="$HOME/gem5/benchmarks/matmul_riscv_minor_32k_512k/output"

# Paths for McPAT + Parser
MCPAT_BIN="$HOME/mcpat/mcpat"
PARSER_DIR="$HOME/Gem5McPATParser"
TEMPLATE="$PARSER_DIR/templates/template_x86.xml"

# === GEM5 RUN ===
echo "▶ Running gem5 simulation..."
mkdir -p "$OUTDIR"

"$GEM5_BIN" \
  --outdir="$OUTDIR" \
  --stats-file="$OUTDIR/stats.txt" \
  --dump-config="$OUTDIR/config.ini" \
  "$SE_SCRIPT" \
  --cpu-type=RiscvO3CPU \
  --num-cpus=1 \
  --caches \
  --l1i_size=32kB \
  --l1d_size=32kB \
  --l2cache \
  --l2_size=512kB \
  --cpu-clock=1GHz \
  -c "$WORKLOAD"

echo "✅ gem5 run completed"

# === CONVERT config.ini → config.json ===
echo "▶ Generating config.json..."
python3 "$HOME/gem5/util/config_to_json.py" \
  "$OUTDIR/config.ini" > "$OUTDIR/config.json"

echo "✅ config.json generated at $OUTDIR/config.json"

ls -l "$OUTDIR/config.json"

# # === CONVERT GEM5 OUTPUT TO McPAT XML ===
# echo "▶ Converting gem5 config + stats to McPAT XML..."
# python3 "$PARSER_DIR/Gem5McPATParser.py" \
#   --config "$OUTDIR/config.json" \
#   --stats "$OUTDIR/stats.txt" \
#   --template "$TEMPLATE" \
#   --output "$OUTDIR/mcpat.xml"

# echo "✅ XML generated at $OUTDIR/mcpat.xml"

# # === RUN MCPAT ===
# echo "▶ Running McPAT..."
# "$MCPAT_BIN" -infile "$OUTDIR/mcpat.xml" -print_level 5 > "$OUTDIR/mcpat_power.txt"

# echo "✅ McPAT power report generated: $OUTDIR/mcpat_power.txt"

# # === EXTRACT TOTAL POWER ===
# echo "▶ Power summary:"
# grep -E "Processor:|Total Leakage|Runtime Dynamic|Peak Power|Total Power" "$OUTDIR/mcpat_power.txt" || true
