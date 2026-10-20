#!/usr/bin/env python
"""Independent re-derivation of the weight->behavior bridge (Table 4 / App. F / heterogeneity).

usage: verify_behavior_meta.py <behavior_trees_dir> <out.tsv>
For every cohort x probe runset and every tree: from the sealed semantic matrix (mean paired
response-embedding cosine distance) and the sealed direct cosine matrix over all trained nodes,
recompute with numpy only: Pearson r between semantic similarity (1 - distance) and weight distance
over all node pairs; Fisher z; delete-one-node jackknife variance of z (drop every pair incident to
the node).  Compare with tree_analysis.json.  Then pool the 50 trees with DerSimonian-Laird
(Q, C, tau^2, RE weights), CI = tanh(z +- z_{0.975} se), I^2, p from z/se, and compare with the
runset receipt's `dl` block for the primary stratum (all trained node pairs) and the leaf-only
sensitivity stratum.
"""
import csv, glob, itertools, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm, chi2

B, OUT = Path(sys.argv[1]), sys.argv[2]
def r_and_jack(D, Sm, idx):
    pairs = list(itertools.combinations(idx, 2))
    def r_of(ps):
        x = np.array([D[i, j] for i, j in ps]); y = np.array([1 - Sm[i, j] for i, j in ps])
        if x.std() == 0 or y.std() == 0: return None
        return float(np.corrcoef(x, y)[0, 1])
    r = r_of(pairs)
    if r is None: return None, None
    zj = []
    for k in idx:
        rk = r_of([p for p in pairs if k not in p])
        if rk is None: return r, None
        zj.append(math.atanh(max(-0.999999, min(0.999999, rk))))
    n = len(idx); var = (n - 1) / n * sum((v - np.mean(zj)) ** 2 for v in zj)
    return r, var
def dl(zs, vs):
    zs, vs = np.array(zs), np.array(vs); w = 1 / vs; zf = (w * zs).sum() / w.sum()
    Q = float((w * (zs - zf) ** 2).sum()); df = len(zs) - 1; C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C); ws = 1 / (vs + tau2); z = float((ws * zs).sum() / ws.sum()); se = math.sqrt(1 / ws.sum())
    p = 2 * (1 - norm.cdf(abs(z / se))); i2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return dict(fisher_z=z, correlation=math.tanh(z), ci_low_r=math.tanh(z - norm.ppf(0.975) * se), ci_high_r=math.tanh(z + norm.ppf(0.975) * se), se_z=se, q=Q, q_df=df, tau2=tau2, i2=i2, p_value=p, n_trees=len(zs))
rows = []; fails = []
for rr in sorted(glob.glob(str(B / "semantic/*/*/runset_receipt.json"))):
    rec = json.load(open(rr)); cohort, probe = rr.split("/")[-3], rr.split("/")[-2]
    strata = {s["stratum"]: s for s in rec["analysis"]["strata"]}
    per_tree = {"all_trained_node_pairs": [], "trained_leaf_pairs": []}; n_tree_ok = 0; n_tree = 0
    for ta_path in sorted(glob.glob(str(B / f"semantic/{cohort}/{probe}/*/tree_analysis.json"))):
        tree = ta_path.split("/")[-2]; ta = json.load(open(ta_path)); sm = json.load(open(Path(ta_path).parent / "semantic_matrix.json"))
        dm = json.load(open(B / f"direct/{cohort}/{tree}/{tree}.cosine.all_nodes.matrix.json"))
        ids = sm["matrix_receipt"]["ordered_node_ids"]; assert ids == dm["ordered_node_ids"]
        D = np.array(dm["matrix"]); Sm = np.array(sm["matrix"]); n = len(ids); n_tree += 1
        st = {s["stratum"]: s for s in ta["strata"]}
        ok = True
        for stratum, idx in (("all_trained_node_pairs", list(range(n))), ("trained_leaf_pairs", list(dm["leaf_sensitivity_indices"]))):
            r, var = r_and_jack(D, Sm, idx); s = st[stratum]
            if not s["defined"]:
                continue
            if s["jackknife_variance_fisher_z"] is None:
                # pipeline rule: a jackknife replicate with <= 3 pairs leaves the variance undefined -> tree excluded from pooling
                m = len(idx); replicate_pairs = (m - 1) * (m - 2) // 2
                if r is None or abs(r - s["pearson_r"]) > 1e-9 or replicate_pairs > 3: ok = False
                continue
            if r is None or var is None or abs(r - s["pearson_r"]) > 1e-9 or abs(var - s["jackknife_variance_fisher_z"]) > 1e-9: ok = False
            else: per_tree[stratum].append((math.atanh(r), var))
        n_tree_ok += ok
        if not ok: fails.append((cohort, probe, tree))
    line = [cohort, probe, n_tree, n_tree_ok]
    for stratum in ("all_trained_node_pairs", "trained_leaf_pairs"):
        ref = strata[stratum]["dl"]; zs, vs = zip(*per_tree[stratum]) if per_tree[stratum] else ([], [])
        mine = dl(zs, vs) if zs else None
        keys = ("correlation", "ci_low_r", "ci_high_r", "i2", "tau2", "q", "p_value", "n_trees")
        agree = mine is not None and all(abs(mine[k] - ref[k]) < (1e-8 if k != "p_value" else 1e-12 + 1e-6 * ref[k]) for k in keys)
        if not agree: fails.append((cohort, probe, stratum, mine, {k: ref[k] for k in keys}))
        line += [f"{mine['correlation']:.4f}" if mine else "-", f"[{mine['ci_low_r']:.3f},{mine['ci_high_r']:.3f}]" if mine else "-", f"{100*mine['i2']:.0f}%" if mine else "-", mine["n_trees"] if mine else 0, "yes" if agree else "NO"]
    rows.append(line)
with open(OUT, "w") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["cohort", "probe", "trees", "trees_r_and_jackknife_match", "r_DL_all_nodes", "CI", "I2", "k", "matches_receipt", "r_DL_leaf_only", "CI_leaf", "I2_leaf", "k_leaf", "matches_receipt_leaf"]); w.writerows(rows)
print(f"{'cohort':14s} {'probe':16s} {'trees':>5s} {'r+jack ok':>9s} {'r_DL':>8s} {'95% CI':>16s} {'I2':>4s} {'k':>3s} {'==receipt':>9s} | {'leaf r_DL':>9s} {'CI':>16s} {'==':>3s}")
for l in rows: print(f"{l[0]:14s} {l[1]:16s} {l[2]:>5d} {l[3]:>9d} {l[4]:>8s} {l[5]:>16s} {l[6]:>4s} {l[7]:>3d} {l[8]:>9s} | {l[9]:>9s} {l[10]:>16s} {l[13]:>3s}")
for f in fails[:6]: print("  MISMATCH", str(f)[:300])
print("PASS verify_behavior_meta" if not fails else f"FAIL verify_behavior_meta ({len(fails)})"); sys.exit(1 if fails else 0)
