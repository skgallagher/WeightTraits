#!/usr/bin/env bash
# 01_trees_b.sh — STAGE 1 (tree generation), part (b): the trees exist on Wright and differ.
#
# This proves:
#   1. the sealed training stage on Wright contains the SAME 50 topology manifests, 50 assigned
#      manifests, tree-set summary and generator config as this repo (sha256 equal on both sides);
#   2. those files are listed in the sealed STAGE_SHA256SUMS manifest with the same hashes, and
#      that manifest itself still hashes to the value recorded in the runbook;
#   3. the 50 trees are 50 different topologies (0 duplicate topology fingerprints).
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
require_socket
cd "$WT_ROOT"
REL=examples/training/confirm_paper_numbers

echo "== 1. sealed stage manifest is intact"
got=$(wr "sha256sum $WRIGHT_STAGE/.stage_provenance/STAGE_SHA256SUMS" | cut -d' ' -f1)
[[ $got == "$STAGE_SUMS_SHA" ]] && check "STAGE_SHA256SUMS sha256 == runbook value" OK "$got" || check "STAGE_SHA256SUMS sha256 == runbook value" FAIL "$got"
wr "cat $WRIGHT_STAGE/.stage_provenance/STAGE_SHA256SUMS" > "$SCRATCH/STAGE_SHA256SUMS"
echo "     ($(wc -l < "$SCRATCH/STAGE_SHA256SUMS" | tr -d ' ') files listed in the sealed manifest; local copy in $SCRATCH)"

echo "== 2. local frozen tree files == Wright stage files (sha256 both sides)"
(cd "$WT_ROOT" && lsha $REL/trees/*.manifest.jsonl $REL/assigned_manifests/*.manifest.jsonl $REL/tree_set_summary.json $REL/assignment_summary.json examples/trees/confirm_paper_numbers.yaml $REL/paper_task_families.yaml) | sort -k2 > "$SCRATCH/trees_local.sha"
wr "cd $WRIGHT_STAGE && sha256sum $REL/trees/*.manifest.jsonl $REL/assigned_manifests/*.manifest.jsonl $REL/tree_set_summary.json $REL/assignment_summary.json examples/trees/confirm_paper_numbers.yaml $REL/paper_task_families.yaml" | sort -k2 > "$SCRATCH/trees_wright.sha"
nl=$(wc -l < "$SCRATCH/trees_local.sha" | tr -d ' '); nw=$(wc -l < "$SCRATCH/trees_wright.sha" | tr -d ' ')
if diff <(awk '{print $1, $2}' "$SCRATCH/trees_local.sha") <(awk '{print $1, $2}' "$SCRATCH/trees_wright.sha") > "$REPORT_DIR/01b_tree_hash_diff.log"; then
  check "104 tree/assignment files identical local vs Wright" OK "$nl local / $nw Wright files, all hashes equal"
else check "104 tree/assignment files identical local vs Wright" FAIL "see $REPORT_DIR/01b_tree_hash_diff.log"; fi

echo "== 3. those files are in the sealed manifest with the same hashes"
miss=0; bad=0
while read -r h f; do
  line=$(grep -F "  ./$f" "$SCRATCH/STAGE_SHA256SUMS" | head -1)
  if [[ -z $line ]]; then miss=$((miss+1)); echo "     not in manifest: $f" >> "$REPORT_DIR/01b_manifest_check.log"
  elif [[ ${line%% *} != "$h" ]]; then bad=$((bad+1)); echo "     hash differs: $f" >> "$REPORT_DIR/01b_manifest_check.log"; fi
done < "$SCRATCH/trees_wright.sha"
[[ $miss -eq 0 && $bad -eq 0 ]] && check "all tree files listed in STAGE_SHA256SUMS with matching hash" OK || check "all tree files listed in STAGE_SHA256SUMS with matching hash" FAIL "$miss missing, $bad differing (see $REPORT_DIR/01b_manifest_check.log)"

echo "== 4. 50 different topologies (fingerprint table)"
PYTHONPATH=src $PY scripts/verify/verify_trees.py "$FROZEN" > "$SCRATCH/verify_trees.out" 2>&1
grep -E "^confirm_paper_tree" "$SCRATCH/verify_trees.out" | awk '{print $1, $2, $3, $NF}' | column -t | sed 's/^/     /' | head -8; echo "     ... ($(grep -c "^confirm_paper_tree" "$SCRATCH/verify_trees.out") trees; full table in $REPORT_DIR/01a_verify_trees.log)"
d=$(grep -E "^distinct topologies" "$SCRATCH/verify_trees.out")
[[ $d == *"duplicate fingerprints: 0"* ]] && check "50 distinct topology fingerprints" OK "$d" || check "50 distinct topology fingerprints" FAIL "$d"
finish
