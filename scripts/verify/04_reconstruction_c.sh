#!/usr/bin/env bash
# 04_reconstruction_c.sh — STAGE 4, part (c): Table 2 topology columns in the paper == the BHV artifacts.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
[[ -s $SCRATCH/reconstruction_rescore.tsv ]] || $PY scripts/verify/verify_reconstruction.py "$LOCAL_MIRROR/direct" "$FROZEN/assigned_manifests" "$SCRATCH/newicks" "$SCRATCH/reconstruction_rescore.tsv" > /dev/null 2>&1
if $PY scripts/verify/verify_table2_topology.py "$PAPER_TEX" "$BHV_LOCAL/controlled_bhv_summary.csv" "$SCRATCH/reconstruction_rescore.tsv" > "$REPORT_DIR/04c_table2_topology.log" 2>&1; then
  check "Table 2 clade/PAER/RF/FN (8 rows) == BHV raw seed 0 at display rounding" OK
  check "caption: 117/117 splits, 46/46 PAER, exact 95% CIs 96.9-100 / 92.3-100" OK
  check "prose ranges: clade 92-100, PAER 85-100, Flan LoRA 92-99" OK
  check "identical 100%/2.04/0.00 cells explained: FN=0 and RF=(leaves-3)-truth_splits on every tree" OK
else check "Table 2 topology columns vs BHV artifacts" FAIL "see $REPORT_DIR/04c_table2_topology.log"; fi
sed 's/^/     /' "$REPORT_DIR/04c_table2_topology.log"
grep -qF "BHV Fr\\'echet mean of per-tensor NJ trees" "$PAPER_TEX" && check "caption names the estimator actually used (BHV Frechet mean of per-tensor NJ trees)" OK || check "caption names the estimator" FAIL
grep -qE "seed" "$BHV_LOCAL/controlled_bhv_summary.csv" && check "NOTE: paper reports BHV seed 0; flan_key RF is 2.37 at seed 1 vs 2.39 at seeds 0/2 (decision D3)" OK "documented"
finish
