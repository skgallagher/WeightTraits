#!/usr/bin/env bash
# 05_analysis_a.sh — STAGE 5 (analysis of the trees), part (a): the code works as intended.
#
# Covers the ordering statistics (Table 2 left columns), the tensor-subset table (App. B), Figure 2
# (summaries + paired-tree bootstrap), the PhyloLM comparison, and the LayerTrace profiles.  This proves:
#   1. the analysis / paper-builder unit tests pass;
#   2. the ordering statistics recompute INDEPENDENTLY with scipy (Mann-Whitney U -> rank-biserial;
#      point-biserial Pearson r; Fisher-z mean) for all 450 trees and all 9 cohort summaries, to 1e-9;
#   3. re-running the checked-in builders today reproduces the committed artifacts byte for byte:
#      layer-subset table (2 files), Figure 2 summaries (3 files), and — via the R figure script, the
#      builder the 09-14 runbook thought was missing — the 2,000-replicate paired bootstrap CSV and the PNG;
#   4. the PhyloLM table (35 / 24 / 52 %, SE 5) recomputes from the 138 per-run metric rows over the same
#      46 eligible trees, with the four excluded trees absent.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
echo "== 1. unit tests"
if PYTHONPATH=src $PY -m pytest -o addopts="" -q tests/test_branch_ordering.py tests/test_direct_weight_table.py tests/test_layer_subset_table.py tests/test_paper_results.py tests/test_paper_figures.py tests/test_paper_diagnostics.py tests/test_phylolm_*.py tests/test_analysis_completion.py > "$REPORT_DIR/05a_pytest.log" 2>&1; then
  check "pytest analysis/paper suites" OK "$(tail -1 "$REPORT_DIR/05a_pytest.log")"; else check "pytest analysis/paper suites" FAIL "see $REPORT_DIR/05a_pytest.log"; fi
echo "== 2. ordering statistics, independent recomputation (scipy)"
if $PY scripts/verify/verify_ordering.py "$LOCAL_MIRROR/direct" "$FROZEN/assigned_manifests" "$SCRATCH/ordering_rescore.tsv" > "$REPORT_DIR/05a_ordering.log" 2>&1; then
  check "rank-biserial / r_branch / SEs reproduced for 450 trees + 9 cohorts (1e-9)" OK; else check "ordering statistics recompute" FAIL "see $REPORT_DIR/05a_ordering.log"; fi
sed 's/^/     /' "$REPORT_DIR/05a_ordering.log" | head -11; cp "$SCRATCH/ordering_rescore.tsv" "$REPORT_DIR/05a_ordering_rescore.tsv"
echo "== 3. builders reproduce the committed artifacts byte for byte"
RG="$SCRATCH/regen_paper"; mkdir -p "$RG"
PYTHONPATH=src $PY scripts/summarize_final_layertrace.py --analysis-root "$LOCAL_MIRROR/direct/flan_full/analysis" --json-out "$RG/final_fixed2000_layertrace.json" --csv-out "$RG/final_fixed2000_layertrace_profile.csv" > "$REPORT_DIR/05a_regen.log" 2>&1
PYTHONPATH=src $PY scripts/build_final_weight_figure2.py --weighttraits-root "$LOCAL_MIRROR" --phylolm-metrics "$ELLM_ROOT/results/aggregate/weighttraits_fresh_phylolm/per_run_metrics.csv" --out "$RG/merged_weight_figure2.json" --csv-out "$RG/merged_weight_figure2.csv" --records-out "$RG/merged_weight_figure2_records.csv" >> "$REPORT_DIR/05a_regen.log" 2>&1
for f in final_fixed2000_layertrace.json final_fixed2000_layertrace_profile.csv merged_weight_figure2.json merged_weight_figure2.csv merged_weight_figure2_records.csv; do
  cmp -s "$RG/$f" "reports/paper/$f" && check "regenerated == committed: $f" OK "$(shasum -a 256 reports/paper/$f | cut -c1-16)" || check "regenerated == committed: $f" FAIL; done
# R: paired-tree bootstrap + figure, run in a scratch replica so nothing committed is overwritten
RF="$SCRATCH/r_fig2"; rm -rf "$RF"; mkdir -p "$RF/WeightTraits/reports/paper" "$RF/ELLMTrees/scripts/ggplot" "$RF/ELLMTrees/results/aggregate" "$RF/paper/figures"
cp reports/paper/merged_weight_figure2.csv reports/paper/merged_weight_figure2_records.csv "$RF/WeightTraits/reports/paper/"; cp "$ELLM_ROOT/scripts/ggplot/theme_weighttraits.R" "$RF/ELLMTrees/scripts/ggplot/"
if (cd "$RF" && Rscript "$ELLM_ROOT/scripts/ggplot/make_merged_weight_figure2.R" "$RF/WeightTraits" "$RF/paper" 2000 > "$REPORT_DIR/05a_r_fig2.log" 2>&1); then
  cmp -s "$RF/WeightTraits/reports/paper/merged_weight_figure2_bootstrap.csv" reports/paper/merged_weight_figure2_bootstrap.csv && check "R paired bootstrap (2000 reps, seed 20260805) == committed CSV: OR 15.67 [9.70,58.55], 5.94 [4.66,8.38]" OK || check "R paired bootstrap == committed CSV" FAIL
  cmp -s "$RF/paper/figures/fig4_coherence_atteson.png" "$PAPER_ROOT/figures/fig4_coherence_atteson.png" && check "regenerated fig4_coherence_atteson.png == paper's (PDF identical except creation date)" OK || check "regenerated Figure 2 PNG == paper's" FAIL
else check "R Figure 2 script" FAIL "see $REPORT_DIR/05a_r_fig2.log"; fi
echo "== 4. PhyloLM table from per-run metrics"
$PY - "$ELLM_ROOT/results/aggregate/weighttraits_fresh_phylolm" <<'PYEOF' > "$REPORT_DIR/05a_phylolm.log" 2>&1 && check "PhyloLM clade 35/24/52 (SE 5) from 3 x 46 per-run rows; excluded trees absent" OK || check "PhyloLM table" FAIL "see $REPORT_DIR/05a_phylolm.log"
import csv, statistics, math, sys
d = sys.argv[1]; rows = list(csv.DictReader(open(d + "/per_run_metrics.csv"))); tab = {r["group"]: r for r in csv.DictReader(open(d + "/table_metrics.csv"))}
want = {"lora_qkv_r8": 35, "lora_qkv_r64": 24, "full_finetune": 52}; ok = True
for g in sorted({r["group"] for r in rows}):
    rs = [r for r in rows if r["group"] == g]; cl = [float(r["clade_recovery"]) for r in rs]; m = statistics.mean(cl); se = statistics.stdev(cl) / math.sqrt(len(cl))
    key = next(k for k in want if k in g); good = len(rs) == 46 and not ({r["run"][-3:] for r in rs} & {"015", "020", "037", "047"}) and round(100 * m) == want[key] and round(100 * se) == 5 and abs(m - float(tab[g]["clade_recovery"])) < 1e-5
    ok &= good; print(g, "n", len(rs), f"clade {m:.5f} SE {se:.4f} -> {round(100*m)}% ({round(100*se)}%)", "ok" if good else "MISMATCH")
sys.exit(0 if ok else 1)
PYEOF
sed 's/^/     /' "$REPORT_DIR/05a_phylolm.log"
finish
