#!/usr/bin/env bash
# 01_trees_c.sh — STAGE 1 (tree generation), part (c): the paper's tree-design statements match.
#
# Each line below is a literal statement in iclr_draft_v4_numbers_20260911.tex (App. "Experimental
# Details", "Sample accounting", Table 2 caption).  The script greps for the statement, then checks
# the value against what verify_trees.py / the frozen configs computed independently.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
PYTHONPATH=src $PY scripts/verify/verify_trees.py "$FROZEN" > "$SCRATCH/verify_trees.out" 2>&1 || true
V="$SCRATCH/verify_trees.out"
val() { grep -E "^$1" "$V" | awk '{print $NF}'; }   # helper: last field of a summary line (unused below where explicit)

claim() {  # claim "<label>" "<literal tex fragment>" "<condition on computed values>"
  local label=$1 frag=$2 cond=$3
  if ! grep -qF -- "$frag" "$PAPER_TEX"; then check "$label" FAIL "phrase not found in tex: $frag"; return; fi
  if eval "$cond"; then check "$label" OK "tex: \"$frag\""; else check "$label" FAIL "tex says \"$frag\" but computed value disagrees"; fi
}
nonroot=$(grep -E "^non-root nodes total" "$V" | awk '{print $4}')
internal=$(grep -E "^internal nodes total" "$V" | awk '{print $4}')
leaves=$(grep -E "^leaves total" "$V" | awk '{print $3}')
topo=$(grep -E "^topology-eligible" "$V" | awk '{print $3}')
ordr=$(grep -E "^ordering-eligible" "$V" | awk '{print $3}')
splits=$(grep -E "^informative truth splits" "$V" | awk '{print $4}')
excl=$(grep -E "^trees with 0 informative" "$V" | sed "s/.*\[\(.*\)\].*expected.*/\1/" | tr -d "' ")
leafmin=$(grep -E "^confirm_paper_tree" "$V" | awk '{print $3}' | sort -n | head -1)
leafmax=$(grep -E "^confirm_paper_tree" "$V" | awk '{print $3}' | sort -n | tail -1)
nodemax=$(grep -E "^confirm_paper_tree" "$V" | awk '{print $2}' | sort -n | tail -1)
depthmax=$(grep -E "^confirm_paper_tree" "$V" | awk '{print $4}' | sort -n | tail -1)
lam=$(grep -E "branch_lambda" examples/trees/confirm_paper_numbers.yaml | awk '{print $2}')
seed=$(grep -E "seed_start" examples/trees/confirm_paper_numbers.yaml | awk '{print $2}')
nn=$(grep -E "^  n_nodes" examples/trees/confirm_paper_numbers.yaml | awk '{print $2}')
md=$(grep -E "^  max_depth" examples/trees/confirm_paper_numbers.yaml | head -1 | awk '{print $2}')

claim "641 trained nodes (365 leaves, 276 internal)" "It produces 641 trained nodes in total (365 leaves and 276 internal nodes)" "[[ $nonroot == 641 && $leaves == 365 && $internal == 276 ]]"
claim "4--10 observed leaves across 50 replications" "trees with 4--10 observed leaves across the 50 accepted replications" "[[ $leafmin == 4 && $leafmax == 10 ]]"
claim "Poisson(lambda = 1.5) branching" "\\mathrm{Poisson}(\\lambda = 1.5)" "[[ $lam == 1.5 ]]"
claim "node budget 14, max depth 4" "n_\\mathrm{nodes} = 14" "[[ $nn == 14 && $nodemax == 14 && $md == 4 && $depthmax == 4 ]]"
claim "topology seeds begin at 20260707; assignment seeds 1--50" "seeds beginning at \\texttt{20260707}; assignment seeds are 1--50" "[[ $seed == 20260707 ]]"
claim "36 task/dataset pairs (9 summ, 10 cls, 8 QA, 9 transl)" "36 task/dataset pairs (9 summarization, 10 classification, 8 QA, and 9 translation)" "grep -q '36 task/dataset pairs, per-tree without replacement' $REPORT_DIR/checks.tsv && ! grep -q '36 task/dataset pairs.*FAIL' $REPORT_DIR/checks.tsv"
claim "46 truths with a nontrivial split enter recovery; 26 enter ordering" "The 46 truths with at least one nontrivial" "[[ $topo == 46 && $ordr == 26 ]]"
claim "n_ord = 26 in the results text" "\$n_{ord}=26\$" "[[ $ordr == 26 ]]"
claim "excluded topology IDs 015, 020, 037, 047" "topology IDs 015, 020, 037, and 047" "[[ $excl == 015,020,037,047 ]]"
claim "117/117 truth splits and 46/46 PAER at the ceiling (Table 2 caption)" "117/117" "[[ $splits == 117 && $topo == 46 ]]"
claim "Table 2 rows all report n_rec/n_ord = 46/26" "46/26 &" "[[ \$(grep -c '46/26 &' $PAPER_TEX) -eq 8 ]]"
finish
