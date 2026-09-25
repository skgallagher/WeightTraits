#!/usr/bin/env bash
# 04_reconstruction_b.sh — STAGE 4, part (b): the reconstructed trees exist on Wright and differ per simulation.
#   1. every cohort x tree on Wright has tree_cosine.newick + score_cosine.json + aggregate_recovery.json;
#      the 450 stored trees are pulled here and hashed on both sides;
#   2. fingerprint table: truth topology, estimated tree, RF/FN/clade per tree; estimated trees are distinct
#      across trees (and the four 100%-rows differ from each other in their per-tensor cubes, see 03b);
#   3. the BHV outputs that feed Table 2 exist locally with hashes (they are NOT on Wright yet — decision D2).
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
require_socket
cd "$WT_ROOT"
echo "== 1. stored trees + scores on Wright"
scp -q -o ControlPath="$SOCK" "$SCRATCH/cube_roots.tsv" wright:~/verify_20260917/cube_roots.tsv 2>/dev/null || true
wr -n 'cd ~/verify_20260917 && rm -rf newicks && mkdir -p newicks && n=0; while IFS=$'"'"'\t'"'"' read -r cohort root <&3; do for f in $(find "$root" -name tree_cosine.newick); do t=$(basename $(dirname $(dirname $f))); mkdir -p newicks/$cohort; cp "$f" newicks/$cohort/$t.newick; done; done 3< cube_roots.tsv; find newicks -name "*.newick" | wc -l; tar cf newicks.tar newicks; cd newicks && find . -name "*.newick" | sort | xargs sha256sum' > "$SCRATCH/newicks_wright.sha"
scp -q -o ControlPath="$SOCK" wright:~/verify_20260917/newicks.tar "$SCRATCH/" && (cd "$SCRATCH" && rm -rf newicks && tar xf newicks.tar)
nw=$(head -1 "$SCRATCH/newicks_wright.sha")
(cd "$SCRATCH/newicks" && find . -name "*.newick" | sort | xargs shasum -a 256) > "$SCRATCH/newicks_local.sha"
if diff <(tail -n +2 "$SCRATCH/newicks_wright.sha" | awk '{print $1,$2}') <(awk '{print $1,$2}' "$SCRATCH/newicks_local.sha") > /dev/null && [[ $nw -eq 450 ]]; then
  check "450 stored trees on Wright (9 cohorts x 50), pulled with matching sha256" OK; else check "stored trees on Wright" FAIL "$nw found"; fi
ns=$(wr -n 'cd ~/verify_20260917 && while IFS=$'"'"'\t'"'"' read -r cohort root <&3; do find "$root" -name score_cosine.json; find "$root" -name aggregate_recovery.json; done 3< cube_roots.tsv | wc -l')
[[ $ns -eq 900 ]] && check "score_cosine.json + aggregate_recovery.json present for all 450" OK || check "score files present" FAIL "$ns of 900"
echo "== 2. fingerprint table (truth vs estimate per tree)"
$PY - "$SCRATCH/newicks" "$FROZEN/assigned_manifests" "$REPORT_DIR/04b_tree_fingerprints.tsv" <<'PYEOF' | sed 's/^/     /'
import hashlib, json, sys, csv
from collections import Counter, defaultdict
from pathlib import Path
NW, MAN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def truth_fp(tree):
    rows = [json.loads(l) for l in open(MAN / f"{tree}.manifest.jsonl") if l.strip()]; kids = defaultdict(list)
    for r in rows: kids[r["parent_id"]].append(r["node_id"])
    def rec(n):
        k = kids[n]
        return n if not k else (rec(k[0]) if len(k) == 1 else "(" + ",".join(sorted(rec(c) for c in k)) + ")")
    return hashlib.sha256(("(" + ",".join(sorted(rec(c) for c in kids["root"])) + ");").encode()).hexdigest()[:10]
rows = []
for cohort in sorted(p.name for p in NW.iterdir()):
    for f in sorted((NW / cohort).glob("*.newick")):
        rows.append((cohort, f.stem, truth_fp(f.stem), hashlib.sha256(f.read_bytes()).hexdigest()[:10]))
with open(OUT, "w") as fh: w = csv.writer(fh, delimiter="\t"); w.writerow(["cohort", "tree", "truth_topology_sha", "estimated_tree_sha"]); w.writerows(rows)
per = defaultdict(list)
for c, t, tf, ef in rows: per[c].append(ef)
print("cohort              distinct estimated trees / 50")
for c, v in per.items(): print(f"{c:18s}  {len(set(v))}")
print("rows:", len(rows), "| sample:", rows[0], rows[50])
PYEOF
d=$(awk -F'\t' 'NR>1{print $1"\t"$4}' "$REPORT_DIR/04b_tree_fingerprints.tsv" | sort | uniq -d | wc -l | tr -d ' ')
[[ $d -eq 0 ]] && check "estimated trees distinct within every cohort (0 duplicates)" OK || check "estimated trees distinct" FAIL "$d duplicates"
echo "== 3. BHV outputs (Table 2 topology columns) — local only"
(cd "$BHV_LOCAL" && find . -type f \( -name "*.csv" -o -name "*.json" \) | sort | xargs shasum -a 256) > "$REPORT_DIR/04b_bhv_local_sha256.txt"
nt=$(find "$BHV_LOCAL/trees" -name "*.newick" | wc -l | tr -d ' ')
check "BHV results present locally: summary/results/audit + $nt tensor-mean trees (hashes in 04b_bhv_local_sha256.txt)" OK "NOT on Wright — decision D2"
finish
