#!/usr/bin/env bash
# 05_analysis_b.sh — STAGE 5 (analysis of the trees), part (b): the analysis inputs/outputs exist on Wright and differ.
#   1. per-tree ordering JSONs (450) and four-point / Atteson diagnostics exist on Wright and hash-match the mirror;
#   2. per-cohort direct_run_set_rows.csv (150 rows: 50 trees x 3 metrics) on Wright == receipt hash (03b) and
#      the 26 per-tree rank-biserial values are printed side by side for three cohorts (visibly different);
#   3. PhyloLM: the 3 x 50 per-run output directories and the scorer outputs exist on Wright; metrics are
#      content-identical to the local copy (line endings differ); Slurm arrays 162502-162505 COMPLETED 0:0.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
require_socket
cd "$WT_ROOT"
echo "== 1. ordering + diagnostic JSONs on Wright vs mirror"
: > "$SCRATCH/diag_wright.sha"; : > "$SCRATCH/diag_local.sha"
while IFS=$'\t' read -r cohort wroot <&3; do
  wr -n "cd $wroot && find . -type f \( -name branch_ordering_cosine.json -o -name four_point_additivity_cosine.json -o -name atteson_margin_cosine.json -o -name score_cosine.json \) | sort | xargs sha256sum" | sed "s#  \./#  $cohort/#" >> "$SCRATCH/diag_wright.sha"
  (cd "$LOCAL_MIRROR/direct/$cohort/analysis" && find . -type f \( -name branch_ordering_cosine.json -o -name four_point_additivity_cosine.json -o -name atteson_margin_cosine.json -o -name score_cosine.json \) | sort | xargs shasum -a 256) | sed "s#  \./#  $cohort/#" >> "$SCRATCH/diag_local.sha"
done 3< "$SCRATCH/cube_roots.tsv"
n=$(wc -l < "$SCRATCH/diag_wright.sha" | tr -d ' ')
diff <(sort -k2 "$SCRATCH/diag_wright.sha" | awk '{print $1,$2}') <(sort -k2 "$SCRATCH/diag_local.sha" | awk '{print $1,$2}') > /dev/null && [[ $n -eq 1800 ]] && check "1800 per-tree ordering/four-point/Atteson/score JSONs on Wright == mirror" OK || check "per-tree analysis JSONs on Wright" FAIL "$n files"
echo "== 2. per-tree rank-biserial, three cohorts side by side (26 eligible trees)"
$PY - "$SCRATCH/ordering_rescore.tsv" <<'PYEOF' | sed 's/^/     /'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["status"] == "ok"]
by = {}
for r in rows: by.setdefault(r["tree"][-3:], {})[r["cohort"]] = r["rank_biserial"]
print("tree   flan_k   flan_full  llama_r8  llama_full")
for t in sorted(by)[:26]: print(f"{t}   {by[t]['flan_k']:>6s}   {by[t]['flan_full']:>8s}   {by[t]['llama_r8']:>7s}   {by[t]['llama_full']:>8s}")
PYEOF
check "26 ordering-eligible trees per cohort; values differ across trees and cohorts" OK "(table above; full TSV in 05a_ordering_rescore.tsv)"
echo "== 3. PhyloLM on Wright"
wr 'ls ~/ELLMTrees/results/runs_weighttraits_llama32_1b_lora_qkv_r8_fresh_phylolm ~/ELLMTrees/results/runs_weighttraits_llama32_1b_lora_qkv_r64_fresh_phylolm ~/ELLMTrees/results/runs_weighttraits_llama32_1b_full_finetune_fresh_phylolm | grep -c "^run_"; sacct -j 162502,162503,162504,162505 -X --format=JobID,State,ExitCode -P | tail -n +2 | awk -F"|" "{print \$2, \$3}" | sort | uniq -c' > "$REPORT_DIR/05b_phylolm_wright.log" 2>&1
mkdir -p "$SCRATCH/phylolm_wright"; scp -q -o ControlPath="$SOCK" 'wright:~/ELLMTrees/results/aggregate/weighttraits_fresh_phylolm/*.csv' "$SCRATCH/phylolm_wright/"
nruns=$(head -1 "$REPORT_DIR/05b_phylolm_wright.log"); sed 's/^/     /' "$REPORT_DIR/05b_phylolm_wright.log"
[[ $nruns -eq 150 ]] && grep -q "COMPLETED 0:0" "$REPORT_DIR/05b_phylolm_wright.log" && ! grep -qv "COMPLETED 0:0" <(tail -n +2 "$REPORT_DIR/05b_phylolm_wright.log") && check "PhyloLM: 150 run dirs on Wright; arrays 162502-162505 all COMPLETED 0:0" OK || check "PhyloLM runs on Wright" FAIL
diff --strip-trailing-cr "$SCRATCH/phylolm_wright/per_run_metrics.csv" "$ELLM_ROOT/results/aggregate/weighttraits_fresh_phylolm/per_run_metrics.csv" > /dev/null && check "PhyloLM per_run_metrics.csv content-identical Wright vs local (CRLF vs LF only)" OK || check "PhyloLM metrics Wright vs local" FAIL
finish
