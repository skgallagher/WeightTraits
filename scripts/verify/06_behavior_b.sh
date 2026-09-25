#!/usr/bin/env bash
# 06_behavior_b.sh — BEHAVIOR track, part (b): the behavior artifacts exist on Wright and differ per simulation.
#   1. the 4 cohort terminal receipts + 15 runset receipts on Wright hash-match the local mirror and say
#      status=completed, valid, 50 trees, 641 nodes (276 internal + 365 leaves), root excluded;
#   2. 650 per-tree semantic matrices + analyses and 200 direct all-node matrices are pulled from Wright;
#      every semantic matrix is different;
#   3. raw inference: 641 response files per cohort (1,490 rows each for Llama, 1,400 for Flan), re-hashed
#      on Wright; response files differ between nodes except identical-recipe Flan nodes (deterministic);
#   4. the empty-output rates in the paper's EOS tables are RECOUNTED from the raw text on Wright;
#   5. the Slurm jobs of the final lineage all COMPLETED 0:0.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
require_socket
cd "$WT_ROOT"
DS=/home/export/sgallagh/WeightTraits-downstream-20260831-native/outputs/allnode_v1
echo "== 1. receipts"
: > "$REPORT_DIR/06b_receipts.log"
$PY - "$LOCAL_MIRROR/behavior" > "$SCRATCH/beh_receipts.tsv" <<'PYEOF'
import json, glob, sys, hashlib
for f in sorted(glob.glob(sys.argv[1] + "/*/terminal_receipt.json") + glob.glob(sys.argv[1] + "/*/*/runset_receipt.json")):
    d = json.load(open(f)); rel = f.split("/behavior/")[1]
    fields = [d.get("status"), d.get("valid"), d.get("n_trees"), d.get("n_nodes"), d.get("n_internal"), d.get("n_leaves"), d.get("root_excluded")]
    parts = rel.split("/"); DS = "/home/export/sgallagh/WeightTraits-downstream-20260831-native/outputs/allnode_v1"
    wpath = f"{DS}/semantic/{parts[0]}/terminal_receipt.json" if len(parts) == 2 else f"{DS}/semantic/{parts[0]}/{parts[1]}/runset_receipt.json"
    print("\t".join([rel, hashlib.sha256(open(f, "rb").read()).hexdigest(), wpath] + [str(x) for x in fields]))
PYEOF
paths=$(cut -f3 "$SCRATCH/beh_receipts.tsv" | tr '\n' ' ')
wr "sha256sum $paths" > "$SCRATCH/beh_receipts_wright.sha" 2>/dev/null
n=0; bad=0
while IFS=$'\t' read -r rel sha wpath status valid nt nn ni nl rx <&3; do
  n=$((n+1)); grep -q "^$sha " "$SCRATCH/beh_receipts_wright.sha" || { bad=$((bad+1)); echo "hash differs on Wright: $rel" >> "$REPORT_DIR/06b_receipts.log"; }
  if [[ $rel == */runset_receipt.json ]]; then [[ $status == completed && $valid == True && $nt == 50 && $nn == 641 && $ni == 276 && $nl == 365 && $rx == True ]] || { bad=$((bad+1)); echo "fields: $rel $status $valid $nt $nn $ni $nl $rx" >> "$REPORT_DIR/06b_receipts.log"; }; fi
done 3< "$SCRATCH/beh_receipts.tsv"
got=$(wr "sha256sum $DS/direct/flan_notrans/terminal_receipt.json $DS/semantic/flan_notrans/translation/runset_receipt.json" | awk '{printf "%s%s", (NR>1?" ":""), $1}')
[[ $got == "7f8cba724b3efa0a86835944808731d733c492eb04d45a3d54f2551a0f2f98c6 2d49afd1b026fb0238c365498774f5b7876be9299538847dc465981d4e6724a6" ]] && check "runbook-pinned hashes: flan_notrans direct terminal 7f8cba72…, translation runset 2d49afd1…" OK || check "runbook-pinned flan_notrans hashes" FAIL "$got"
[[ $bad -eq 0 && $n -eq 17 ]] && check "17 behavior receipts: Wright sha256 == mirror; runsets completed/valid, 50 trees, 641 = 276 + 365 nodes, root excluded" OK || check "behavior receipts" FAIL "$bad problems of $n; see $REPORT_DIR/06b_receipts.log"
echo "== 2. per-tree semantic + direct matrices from Wright"
if [[ ! -d $SCRATCH/behavior_trees ]]; then
  wr "cd $DS && tar cf ~/verify_20260917/behavior_trees.tar \$(find semantic -name tree_analysis.json -o -name semantic_matrix.json -o -name runset_receipt.json | sort) \$(find direct -name '*.cosine.all_nodes.matrix.json' -o -name terminal_receipt.json | sort)"
  scp -q -o ControlPath="$SOCK" wright:~/verify_20260917/behavior_trees.tar "$SCRATCH/" && mkdir -p "$SCRATCH/behavior_trees" && tar xf "$SCRATCH/behavior_trees.tar" -C "$SCRATCH/behavior_trees"; fi
nsm=$(find "$SCRATCH/behavior_trees/semantic" -name semantic_matrix.json | wc -l | tr -d ' '); nta=$(find "$SCRATCH/behavior_trees/semantic" -name tree_analysis.json | wc -l | tr -d ' '); ndm=$(find "$SCRATCH/behavior_trees/direct" -name "*.cosine.all_nodes.matrix.json" | wc -l | tr -d ' ')
dups=$($PY -c "
import glob, json, collections
by = collections.defaultdict(list)
for f in glob.glob('$SCRATCH/behavior_trees/semantic/*/*/*/semantic_matrix.json'): by[json.load(open(f))['matrix_receipt']['matrix_sha256']].append(f.split('/')[-4:-1])
# a duplicate is legitimate only between a cohort and its matched no-translation twin, same probe, same tree (a tree with no translation node = same experiment)
bad = [v for v in by.values() if len(v) > 1 and not (len(v) == 2 and v[0][1] == v[1][1] and v[0][2] == v[1][2] and {v[0][0], v[1][0]} in ({'flan_full', 'flan_notrans'}, {'llama_full', 'llama_notrans'}))]
twin = sum(1 for v in by.values() if len(v) > 1) - len(bad)
print(f'{len(bad)} {twin}')")
[[ $nsm -eq 650 && $nta -eq 650 && $ndm -eq 200 && ${dups%% *} -eq 0 ]] && check "650 semantic matrices + 650 tree analyses + 200 direct matrices; all distinct except ${dups##* } full/notrans twin tree(s) with no translation node" OK || check "per-tree behavior files" FAIL "$nsm/$nta/$ndm, unexplained duplicates=${dups%% *}"
echo "== 3-4. raw inference audit on Wright (responses per node, hashes, empty-output recount)"
if ! wr 'test -f ~/verify_20260917/audit_behavior.done'; then check "behavior inference audit finished on Wright" FAIL "still running: ~/verify_20260917/audit_behavior_*.log"; finish; fi
mkdir -p "$SCRATCH/audit_behavior"; scp -q -o ControlPath="$SOCK" 'wright:~/verify_20260917/audit_behavior_*.json' 'wright:~/verify_20260917/audit_behavior_*.tsv' "$SCRATCH/audit_behavior/"
$PY - "$SCRATCH/audit_behavior" "$SCRATCH/audit" "$LOCAL_MIRROR/table8_eos_sensitivity" <<'PYEOF' > "$REPORT_DIR/06b_inference_audit.log" 2>&1 && check "641 response files/cohort, expected rows, distinct except same-recipe Flan nodes; empty rates recounted == EOS tables" OK || check "raw inference audit" FAIL "see $REPORT_DIR/06b_inference_audit.log"
import csv, json, sys, collections
A, TA, T8 = sys.argv[1:4]; ok = True
chains = {}
for lab in ("flan_full", "flan_notrans", "llama_full", "llama_notrans"):
    try: chains[lab] = {(r["tree"], r["node"]): r["chain"] for r in csv.DictReader(open(f"{TA}/audit_{lab}.tsv"), delimiter="\t")}
    except FileNotFoundError: chains[lab] = {}
for lab in ("llama_full", "flan_full", "llama_notrans", "flan_notrans"):
    s = json.load(open(f"{A}/audit_behavior_{lab}.json")); rows = list(csv.DictReader(open(f"{A}/audit_behavior_{lab}.tsv"), delimiter="\t"))
    groups = collections.defaultdict(list)
    for r in rows: groups[r["sha256"]].append((r["tree"], r["node"]))
    dup_groups = [g for g in groups.values() if len(g) > 1]; same_recipe = sum(1 for g in dup_groups if chains[lab] and len({chains[lab][k] for k in g}) == 1)
    expected_rows = 200 if lab.endswith("notrans") else (1490 if lab.startswith("llama") else 1400)
    good = s["n_trees"] == 50 and s["n_nodes"] == 641 and s["rows_per_node"] == [[expected_rows, 641]] and same_recipe == len(dup_groups)
    ok &= good
    print(f"{lab:14s} trees={s['n_trees']} nodes={s['n_nodes']} rows/node={s['rows_per_node']} distinct_files={s['n_distinct_sha256']} dup_groups={len(dup_groups)} (same recipe: {same_recipe})  {'ok' if good else 'PROBLEM'}")
    print("   leaf empty %:", {k: round(v, 3) for k, v in s["leaf_empty_percent"].items()})
# recount vs the sealed EOS artifacts (llama_full)
s = json.load(open(f"{A}/audit_behavior_llama_full.json")); dep = json.load(open(f"{T8}/depth_summary.json"))
for probe in ("hellaswag", "arc_challenge", "mmlu", "truthfulqa", "dolly_open_ended"):
    t8 = json.load(open(f"{T8}/{probe}.json"))["counts"]; mine = s["leaf_empty_counts"][probe]
    good = mine == [t8["leaf_empty"], t8["leaf_outputs"]]; ok &= good
    dd = [round(s["depth_empty_percent"].get(f"{probe}@{d}", 0.0), 4) for d in (1, 2, 3, 4)]; td = [round(dep["depth"][probe][str(d)]["percent"], 4) for d in (1, 2, 3, 4)]
    good2 = dd == td; ok &= good2
    print(f"   {probe:16s} recount leaf empty {mine} vs sealed [{t8['leaf_empty']}, {t8['leaf_outputs']}] {'ok' if good else 'MISMATCH'} | by depth {dd} vs {td} {'ok' if good2 else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PYEOF
sed 's/^/     /' "$REPORT_DIR/06b_inference_audit.log"
echo "== 5. Slurm lineage"
wr 'sacct -j 187973,187974,187975,187976,187977,187978,187979,187980,195552,195764,195765,187982,187983,187984,187999 -X --format=JobID,JobName%30,State,ExitCode -P | tail -n +2' > "$REPORT_DIR/06b_sacct.log" 2>&1
nj=$(grep -c "" "$REPORT_DIR/06b_sacct.log"); nc=$(grep -c "COMPLETED|0:0" "$REPORT_DIR/06b_sacct.log")
awk -F'|' '{print $3}' "$REPORT_DIR/06b_sacct.log" | sort | uniq -c | sed 's/^/     /'
# The runbook documents flan_notrans inference array 187979 stopping at task 461 (preempted; 462+ cancelled), replaced by
# array 195552 + retry 195764_461 + aggregation 195765; the dependent 187980 therefore FAILED and was superseded.
unexpected=$(grep -v "COMPLETED|0:0" "$REPORT_DIR/06b_sacct.log" | grep -v "^187980|.*|FAILED\|^187979_461|.*|PREEMPTED\|^187979_\(462\|\[463-640%4\]\)|.*|CANCELLED" | wc -l | tr -d ' ')
repl=$(grep -c "^19555[2]\|^195764\|^195765" "$REPORT_DIR/06b_sacct.log"); replok=$(grep "^19555[2]\|^195764\|^195765" "$REPORT_DIR/06b_sacct.log" | grep -c "COMPLETED|0:0")
[[ $unexpected -eq 0 && $repl -gt 0 && $repl -eq $replok ]] && check "Slurm lineage: $nc/$nj rows COMPLETED 0:0; only the documented superseded flan_notrans attempts (187979_461+, 187980) are not; replacements 195552/195764/195765 COMPLETED ($replok/$repl)" OK || check "Slurm lineage" FAIL "$unexpected unexpected non-completed rows; replacements $replok/$repl"
finish
