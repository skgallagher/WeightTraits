#!/usr/bin/env python
"""Independent recomputation of the direct-lineage (ordering) statistics in Table 2.

usage: verify_ordering.py <local_mirror_direct_dir> <assigned_manifest_dir> <out.tsv>
For every cohort x tree: label leaf pairs same-/cross-root-branch from the manifest (branch = the
root's child on the leaf's path), take pair distances from the sealed mean cosine matrix, and compute
with scipy/numpy: Mann-Whitney U -> rank-biserial = 2U/(n_same*n_cross) - 1 (positive = cross-branch
pairs farther), and the point-biserial Pearson r between the same-branch indicator and distance.
Compare per tree with branch_ordering_cosine.json and per cohort (mean, SE, Fisher-z mean over the
26 eligible trees) with direct_run_set_summary.json.
"""
import csv, json, math, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr

D, MAN, OUT = (Path(p) for p in sys.argv[1:4])
def branches(tree):
    rows = [json.loads(l) for l in open(MAN / f"{tree}.manifest.jsonl") if l.strip()]
    kids = defaultdict(list)
    for r in rows: kids[r["parent_id"]].append(r["node_id"])
    return {r["node_id"]: r["path"][1] for r in rows if not kids[r["node_id"]]}
fails = []; out = []; summary_rows = []
for cohort in sorted(p.name for p in D.iterdir() if (p / "receipt.json").exists()):
    rbs, rs, n_ok, n_missing = [], [], 0, 0
    for t in sorted(p for p in (D / cohort / "analysis").iterdir() if p.is_dir()):
        a = next(t.glob("*_leaf_analysis")); m = np.load(a / "distance_matrix_cosine.npy"); models = json.load(open(a / "models.json"))
        stored = json.load(open(a / "branch_ordering_cosine.json")); br = branches(t.name)
        same, cross, ind, dist = [], [], [], []
        for i, j in combinations(range(len(models)), 2):
            s = br[models[i]] == br[models[j]]; d = float(m[i, j]); ind.append(1.0 if s else 0.0); dist.append(d); (same if s else cross).append(d)
        if not same or not cross:
            ok = stored["branch_ordering_status"] == "missing_branch_class"; n_missing += 1
            out.append((cohort, t.name, "missing_branch_class", "", "", "", "yes" if ok else "NO")); fails += [] if ok else [(cohort, t.name, "status")]; continue
        U = mannwhitneyu(cross, same, alternative="greater").statistic; rb = 2 * U / (len(cross) * len(same)) - 1
        r = max(-0.999, min(0.999, pearsonr(ind, dist)[0]))
        ok = stored["branch_ordering_status"] == "ok" and abs(stored["branch_rank_biserial"] - rb) < 1e-9 and abs(stored["branch_within_run_r"] - r) < 1e-9
        n_ok += 1; rbs.append(rb); rs.append(r)
        out.append((cohort, t.name, "ok", f"{rb:.4f}", f"{r:.4f}", f"{stored['branch_rank_biserial']:.4f}", "yes" if ok else "NO")); fails += [] if ok else [(cohort, t.name, rb, stored["branch_rank_biserial"], r, stored["branch_within_run_r"])]
    summ = json.load(open(D / cohort / "direct_run_set_summary.json"))["aggregate_by_metric"]["cosine"]
    mine = dict(n_ordering_trees=n_ok, rb_mean=float(np.mean(rbs)), rb_se=float(np.std(rbs, ddof=1) / math.sqrt(len(rbs))),
                r_fisher=math.tanh(float(np.mean([math.atanh(x) for x in rs]))), r_se=float(np.std(rs, ddof=1) / math.sqrt(len(rs))))
    theirs = dict(n_ordering_trees=summ["n_ordering_trees"], rb_mean=summ["branch_rank_biserial_mean"], rb_se=summ["branch_rank_biserial_se"],
                  r_fisher=summ["branch_within_run_r_fisher_z_mean"], r_se=summ["branch_within_run_r_se"])
    agree = all(abs(mine[k] - theirs[k]) < 1e-9 for k in mine)
    fails += [] if agree else [(cohort, "cohort summary", mine, theirs)]
    summary_rows.append((cohort, n_ok, n_missing, mine["rb_mean"], mine["rb_se"], mine["r_fisher"], mine["r_se"], agree))
with open(OUT, "w") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["cohort", "tree", "status", "rank_biserial", "within_run_r", "stored_rank_biserial", "matches_stored"]); w.writerows(out)
print(f"{'cohort':18s} {'ok':>3s} {'undef':>5s} {'rank-biserial (SE)':>20s} {'r_branch Fisher-z (SE)':>24s}  == pipeline summary")
for c, n_ok, n_missing, rb, rbse, rz, rse, agree in summary_rows:
    print(f"{c:18s} {n_ok:>3d} {n_missing:>5d} {rb:>12.3f} ({rbse:.3f}) {rz:>16.3f} ({rse:.3f})  {'yes' if agree else 'NO'}")
print(f"\nper-tree agreement with branch_ordering_cosine.json: {sum(1 for r in out if r[-1]=='yes')}/{len(out)}")
for f in fails[:8]: print("  MISMATCH", f)
print("PASS verify_ordering" if not fails else f"FAIL verify_ordering ({len(fails)})"); sys.exit(1 if fails else 0)
