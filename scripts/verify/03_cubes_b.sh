#!/usr/bin/env bash
# 03_cubes_b.sh — STAGE 3 (distance cubes), part (b): the cubes exist on Wright and differ per simulation.
#
# This proves:
#   1. every file pinned by the 9 local receipts (roll-up JSON/CSV, analysis report, completion binding,
#      job config, training summary) exists on Wright with the recorded sha256, and each receipt is valid;
#   2. for each cohort, all 50 per-tree analysis directories on Wright hold the cube files, and their sha256
#      equals the local mirror's (so everything analysed locally IS the sealed Wright data);
#   3. the 450 cosine matrices are all different (no tree's result is a copy of another's), shown as a
#      per-tree fingerprint table.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
require_socket
cd "$WT_ROOT"

echo "== 1. receipt-pinned files: sha256 on Wright == receipt"
$PY - "$LOCAL_MIRROR/direct" > "$SCRATCH/receipt_files.tsv" <<'PYEOF'
import json, glob, sys
for r in sorted(glob.glob(sys.argv[1] + "/*/receipt.json")):
    d = json.load(open(r)); cohort = r.split("/")[-2]
    for k in ("rollup_json", "rollup_csv", "analysis_report", "completion_receipt", "input_config", "training_summary"):
        print(f"{cohort}\t{k}\t{d[k]['sha256']}\t{d[k]['path']}\t{d['valid']}\t{d['n_tree_summaries']}")
PYEOF
cut -f4 "$SCRATCH/receipt_files.tsv" | sort -u > "$SCRATCH/receipt_paths.txt"
wr "sha256sum $(paste -sd' ' "$SCRATCH/receipt_paths.txt")" > "$SCRATCH/receipt_wright.sha" 2>"$REPORT_DIR/03b_receipt_errors.log"
bad=0; n=0
while IFS=$'\t' read -r cohort key sha path valid nts <&3; do
  n=$((n+1)); got=$(grep -F "  $path" "$SCRATCH/receipt_wright.sha" | cut -d' ' -f1)
  [[ $got == "$sha" && $valid == True && $nts == 50 ]] || { bad=$((bad+1)); echo "MISMATCH $cohort $key expected $sha got ${got:-MISSING}" >> "$REPORT_DIR/03b_receipt_errors.log"; }
done 3< "$SCRATCH/receipt_files.tsv"
[[ $bad -eq 0 ]] && check "$n receipt-pinned files present on Wright with matching sha256; receipts valid, 50 trees" OK || check "receipt-pinned files on Wright" FAIL "$bad problems, see $REPORT_DIR/03b_receipt_errors.log"

echo "== 2. per-tree cube files: sha256 Wright == local mirror (9 cohorts x 50 trees x 4 files)"
$PY - "$LOCAL_MIRROR/direct" > "$SCRATCH/cube_roots.tsv" <<'PYEOF'
import json, glob, sys, os
for r in sorted(glob.glob(sys.argv[1] + "/*/receipt.json")):
    d = json.load(open(r)); cohort = r.split("/")[-2]
    print(f"{cohort}\t{os.path.dirname(d['rollup_json']['path'])}/analysis")
PYEOF
: > "$SCRATCH/cubes_wright.sha"; : > "$SCRATCH/cubes_local.sha"
while IFS=$'\t' read -r cohort wroot <&3; do
  wr -n "cd $wroot && find . -type f \( -name distance_matrix_cosine.npy -o -name direct_distance_layers.npz -o -name models.json -o -name layers.json \) | sort | xargs sha256sum" | sed "s#  \./#  $cohort/#" >> "$SCRATCH/cubes_wright.sha"
  (cd "$LOCAL_MIRROR/direct/$cohort/analysis" && find . -type f \( -name distance_matrix_cosine.npy -o -name direct_distance_layers.npz -o -name models.json -o -name layers.json \) | sort | xargs shasum -a 256) | sed "s#  \./#  $cohort/#" >> "$SCRATCH/cubes_local.sha"
done 3< "$SCRATCH/cube_roots.tsv"
sort -k2 "$SCRATCH/cubes_wright.sha" > "$SCRATCH/cw.sorted"; sort -k2 "$SCRATCH/cubes_local.sha" > "$SCRATCH/cl.sorted"
nw=$(wc -l < "$SCRATCH/cw.sorted" | tr -d ' '); nl=$(wc -l < "$SCRATCH/cl.sorted" | tr -d ' ')
if diff "$SCRATCH/cw.sorted" "$SCRATCH/cl.sorted" > "$REPORT_DIR/03b_cube_hash_diff.log"; then
  check "cube files identical Wright vs local mirror" OK "$nw files on Wright, $nl local, all sha256 equal (expect 1800)"
else check "cube files identical Wright vs local mirror" FAIL "$nw Wright / $nl local; see $REPORT_DIR/03b_cube_hash_diff.log"; fi
[[ $nw -eq 1800 ]] && check "9 cohorts x 50 trees x 4 cube files present on Wright" OK || check "9 cohorts x 50 trees x 4 cube files present on Wright" FAIL "$nw files"

echo "== 3. fingerprint table: every tree's cosine matrix is different"
[[ -s $SCRATCH/cube_fingerprints.tsv ]] || $PY scripts/verify/verify_cube_sanity.py "$LOCAL_MIRROR/direct" "$FROZEN/assigned_manifests" "$SCRATCH/cube_fingerprints.tsv" > /dev/null 2>&1
cp "$SCRATCH/cube_fingerprints.tsv" "$REPORT_DIR/03b_cube_fingerprints.tsv"
dups=$(cut -f6 "$SCRATCH/cube_fingerprints.tsv" | tail -n +2 | sort | uniq -d | wc -l | tr -d ' ')
rows=$(tail -n +2 "$SCRATCH/cube_fingerprints.tsv" | wc -l | tr -d ' ')
[[ $dups -eq 0 && $rows -eq 450 ]] && check "450 cosine matrices, 0 duplicates across trees/cohorts" OK "table: $REPORT_DIR/03b_cube_fingerprints.tsv" || check "450 cosine matrices, 0 duplicates" FAIL "$rows rows, $dups duplicate hashes"
echo "     (first rows; same tree, three cohorts — visibly different matrices)"
awk -F'\t' 'NR==1 || $2=="confirm_paper_tree_001"' "$SCRATCH/cube_fingerprints.tsv" | cut -c1-110 | sed 's/^/     /'
finish
