#!/usr/bin/env bash
# run_all.sh — run every verification script in order and write reports/verify/<date>/SUMMARY.md.
#
#   bash scripts/verify/run_all.sh            # everything (needs the Wright control socket for the *_b scripts)
#   bash scripts/verify/run_all.sh local      # only the scripts that need no Wright access
#
# Prerequisites for the Wright parts (see 00_env.sh): ssh -M -S /tmp/wright-codex.sock -fN wright
# Long background jobs the scripts consume must have finished on Wright: ~/verify_20260917/run_audits.sh
# (02b), run_recompute*.sh (03a), audit_behavior (06b); and locally the BHV re-run in $SCRATCH/bhv_robust (04a).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/00_env.sh"
mode=${1:-all}
rm -f "$REPORT_DIR/checks.tsv"
order=(01_trees_a 01_trees_b 01_trees_c 02_training_a 02_training_b 02_training_c 03_cubes_b 03_cubes_a 03_cubes_c
       04_reconstruction_b 04_reconstruction_a 04_reconstruction_c 05_analysis_a 05_analysis_b 05_analysis_c
       06_behavior_b 06_behavior_a 06_behavior_c)
for s in "${order[@]}"; do
  if [[ $mode == local && $s == *_b ]]; then continue; fi
  if [[ $mode == local && $s == 02_training_a ]]; then echo "(02_training_a's run-list regeneration needs Wright; skipped in local mode)"; continue; fi
  echo; echo "################ $s"; bash "$HERE/$s.sh" 2>&1 | tee "$REPORT_DIR/$s.log" | grep -E "^\s+\[|^PASS|^FAIL"
done
$PY "$HERE/make_summary.py" "$REPORT_DIR" "$PAPER_TEX"
echo; echo "summary: $REPORT_DIR/SUMMARY.md"
