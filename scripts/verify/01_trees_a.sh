#!/usr/bin/env bash
# 01_trees_a.sh — STAGE 1 (tree generation), part (a): the code works as intended.
#
# This proves:
#   1. the tree-generation unit tests pass;
#   2. re-running the generator + task assigner from the frozen config/seeds reproduces
#      the 50 frozen manifests BYTE FOR BYTE (the generator is deterministic);
#   3. an independent, plain-Python read of the manifests confirms every structural claim
#      (641 = 276 + 365 nodes, 46/26 eligibility, the 4 excluded IDs, 117 truth splits,
#      50 distinct topologies) and agrees with the library's own split code.
# PASS means all three hold.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"

echo "== 1. unit tests (trees + task assignment)"
if PYTHONPATH=src $PY -m pytest -o addopts="" -q tests/test_tree_generation.py tests/test_tree_examples.py tests/test_task_assignment.py > "$REPORT_DIR/01a_pytest.log" 2>&1; then
  check "pytest tree/assignment suites" OK "$(tail -1 "$REPORT_DIR/01a_pytest.log")"
else check "pytest tree/assignment suites" FAIL "see $REPORT_DIR/01a_pytest.log"; fi

echo "== 2. regenerate trees + assignments from the frozen config and diff against the frozen files"
RG="$SCRATCH/regen_trees"; rm -rf "$RG"; mkdir -p "$RG"
PYTHONPATH=src $PY -m weighttraits.cli generate-tree-set \
  --config examples/trees/confirm_paper_numbers.yaml \
  --out-dir "$RG/trees" --summary-out "$RG/tree_set_summary.json" > "$REPORT_DIR/01a_regen.log" 2>&1
if diff -rq "$RG/trees" "$FROZEN/trees" >> "$REPORT_DIR/01a_regen.log" 2>&1; then
  check "regenerated 50 topology manifests == frozen" OK "$(ls "$RG/trees" | wc -l | tr -d ' ') files, byte-identical"
else check "regenerated 50 topology manifests == frozen" FAIL "diff in $REPORT_DIR/01a_regen.log"; fi
# the summary embeds the out_dir path, so compare it with that one field neutralised
$PY - "$RG/tree_set_summary.json" "$FROZEN/tree_set_summary.json" <<'PYEOF' && check "regenerated tree_set_summary.json == frozen (paths ignored)" OK || check "regenerated tree_set_summary.json == frozen (paths ignored)" FAIL
import json, sys
def norm(p):
    d = json.load(open(p)); d.pop("out_dir", None)
    for t in d["trees"]: t.pop("manifest", None)
    return d
sys.exit(0 if norm(sys.argv[1]) == norm(sys.argv[2]) else 1)
PYEOF
# assignments must be re-run on the FROZEN tree set summary (its manifest paths are repo-relative)
PYTHONPATH=src $PY -m weighttraits.cli assign-task-data-set \
  --tree-set "$FROZEN/tree_set_summary.json" \
  --config "$FROZEN/paper_task_families.yaml" \
  --out-dir "$RG/assigned" --summary-out "$RG/assignment_summary.json" \
  --seed-start 1 --policy per_node_without_replacement >> "$REPORT_DIR/01a_regen.log" 2>&1
if diff -rq "$RG/assigned" "$FROZEN/assigned_manifests" >> "$REPORT_DIR/01a_regen.log" 2>&1; then
  check "regenerated 50 assigned manifests == frozen" OK "byte-identical"
else check "regenerated 50 assigned manifests == frozen" FAIL "diff in $REPORT_DIR/01a_regen.log"; fi

echo "== 3. independent structural check of the frozen trees"
if PYTHONPATH=src $PY scripts/verify/verify_trees.py "$FROZEN" > "$REPORT_DIR/01a_verify_trees.log" 2>&1; then
  check "verify_trees.py (641/276/365, 46/26, 117 splits, 50 distinct)" OK "table in $REPORT_DIR/01a_verify_trees.log"
else check "verify_trees.py (641/276/365, 46/26, 117 splits, 50 distinct)" FAIL "see $REPORT_DIR/01a_verify_trees.log"; fi
grep -E "^(non-root|internal|leaves|trees with|topology-eligible|informative|ordering|distinct|library)" "$REPORT_DIR/01a_verify_trees.log" | sed 's/^/     /'

echo "== 4. task/dataset pool is what the paper says (36 pairs: 10 cls / 9 transl / 9 summ / 8 QA; seeds 1-50; no repeats within a tree)"
$PY - "$FROZEN" <<'PYEOF' && check "36 task/dataset pairs, per-tree without replacement, seeds 1..50" OK || check "36 task/dataset pairs, per-tree without replacement, seeds 1..50" FAIL
import json, glob, sys, collections
root = sys.argv[1]; pairs=set(); seeds=[]
for p in sorted(glob.glob(f"{root}/assigned_manifests/*.jsonl")):
    rows=[json.loads(l) for l in open(p) if l.strip()]
    seeds.append(rows[0]["task_assignment_seed"])
    ds=[r["dataset_id"] for r in rows]
    assert len(ds)==len(set(ds)), p
    assert all(r["task_assignment_policy"]=="per_node_without_replacement" for r in rows)
    pairs |= {(r["task_family"], r["dataset_id"]) for r in rows}
fam = collections.Counter(f for f,_ in pairs)
ok = len(pairs)==36 and fam=={"classification":10,"translation":9,"summarization":9,"qa":8} and sorted(seeds)==list(range(1,51))
print("pairs", len(pairs), dict(fam), "seeds", min(seeds), "..", max(seeds)); sys.exit(0 if ok else 1)
PYEOF
finish
