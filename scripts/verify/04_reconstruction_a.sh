#!/usr/bin/env bash
# 04_reconstruction_a.sh — STAGE 4 (tree building from the cubes), part (a): the code works as intended.
#
# Two estimators feed the paper: mean-distance neighbour joining (ordering stats, claim index) and the
# BHV Frechet mean of per-tensor NJ trees (Table 2 topology columns).  This proves:
#   1. the reconstruction / recovery / four-point / Atteson / ordering unit tests pass;
#   2. NJ recovers a hand-built additive tree exactly (known answer), and the scorer counts splits correctly;
#   3. an INDEPENDENT textbook NJ + split/RF implementation (own numpy code, not Biopython) reproduces the
#      stored Wright tree AND every stored score (rf, FN, FP, TP, clade, PAER) on all 9 cohorts x 50 trees;
#   4. the BHV Frechet mean is robust: re-running at unseen seeds (3,4) with twice the epochs (600) reproduces
#      the paper's clade / PAER / FN exactly and RF to within one collapsed edge on one suite;
#   5. the BHV estimator and mean-distance NJ agree on truth recovery for every tree; they differ only where
#      the Frechet mean drops an unsupported extra split (10 llama_full trees -> RF 1.83 vs 2.04).
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
echo "== 1. unit tests"
if PYTHONPATH=src $PY -m pytest -o addopts="" -q tests/test_reconstruction.py tests/test_recovery_scoring.py tests/test_four_point_additivity.py tests/test_atteson_margin.py tests/test_branch_ordering.py tests/test_direct_analysis_attestation.py > "$REPORT_DIR/04a_pytest.log" 2>&1; then
  check "pytest reconstruction/recovery suites" OK "$(tail -1 "$REPORT_DIR/04a_pytest.log")"; else check "pytest reconstruction/recovery suites" FAIL "see $REPORT_DIR/04a_pytest.log"; fi
echo "== 2. NJ known answer (additive 7-leaf tree with a polytomy)"
if $PY scripts/verify/verify_nj_known_answer.py "$WT_ROOT" > "$REPORT_DIR/04a_nj_known_answer.log" 2>&1; then check "NJ recovers an additive tree exactly; scorer agrees with hand count" OK; else check "NJ known answer" FAIL "see $REPORT_DIR/04a_nj_known_answer.log"; fi
sed 's/^/     /' "$REPORT_DIR/04a_nj_known_answer.log" | head -4
echo "== 3. independent NJ + scoring vs stored trees and scores (450 cases)"
[[ -d $SCRATCH/newicks ]] || { echo "     (stored trees not pulled yet; run 04_reconstruction_b.sh first)"; }
if $PY scripts/verify/verify_reconstruction.py "$LOCAL_MIRROR/direct" "$FROZEN/assigned_manifests" "$SCRATCH/newicks" "$SCRATCH/reconstruction_rescore.tsv" > "$REPORT_DIR/04a_rescore.log" 2>&1; then
  check "own NJ == stored tree and own scores == stored scores, 450/450" OK "$(sed -n 2p "$REPORT_DIR/04a_rescore.log")"; else check "independent NJ re-scoring" FAIL "see $REPORT_DIR/04a_rescore.log"; fi
cp "$SCRATCH/reconstruction_rescore.tsv" "$REPORT_DIR/04a_reconstruction_rescore.tsv" 2>/dev/null
echo "== 4. BHV robustness: seeds 3,4 x 600 epochs vs the paper run (seed 0, 300 epochs)"
if [[ -s $SCRATCH/bhv_robust/controlled_bhv_summary.csv ]]; then
  if $PY scripts/verify/compare_bhv_summaries.py "$BHV_LOCAL/controlled_bhv_summary.csv" "$SCRATCH/bhv_robust/controlled_bhv_summary.csv" > "$REPORT_DIR/04a_bhv_robustness.log" 2>&1; then
    check "BHV means reproduced at new seeds / 600 epochs" OK "$(grep largest "$REPORT_DIR/04a_bhv_robustness.log")"; else check "BHV robustness" FAIL "see $REPORT_DIR/04a_bhv_robustness.log"; fi
  sed 's/^/     /' "$REPORT_DIR/04a_bhv_robustness.log" | head -20
else check "BHV robustness re-run" FAIL "not finished: $SCRATCH/bhv_robust/run.log"; fi
echo "== 5. BHV vs mean-distance NJ equivalence audit"
$PY - "$BHV_LOCAL/equivalence_audit_raw_seed0.json" <<'PYEOF' > "$REPORT_DIR/04a_bhv_equivalence.log" 2>&1 && check "BHV == mean-NJ on truth recovery for 368/368; differs only in llama_full RF (-1 extra split, 10 trees)" OK || check "BHV equivalence audit" FAIL "see $REPORT_DIR/04a_bhv_equivalence.log"
import json, sys
d = json.load(open(sys.argv[1])); g = d["global"]; diff = [r for r in d["records"] if not r["identical_reported_metrics"]]
print("global:", g); print("records with different reported metrics:", len(diff))
for r in diff: print("  ", r["suite_id"], r["tree_id"][-3:], "BHV rf/fn", r["bhv_rf"], r["bhv_false_negative"], "| mean-NJ rf/fn", r["matrix_average_rf"], r["matrix_average_false_negative"])
ok = g["identical_truth_recovery"] == g["n"] == 368 and len(diff) == 10 and all(r["suite_id"] == "llama_full" and r["bhv_false_negative"] == r["matrix_average_false_negative"] == 0 and r["bhv_rf"] == r["matrix_average_rf"] - 1 for r in diff)
sys.exit(0 if ok else 1)
PYEOF
sed 's/^/     /' "$REPORT_DIR/04a_bhv_equivalence.log" | head -4
finish
