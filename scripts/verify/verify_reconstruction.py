#!/usr/bin/env python
"""Independent re-scoring of the mean-distance neighbour-joining pipeline.

usage: verify_reconstruction.py <local_mirror_direct_dir> <assigned_manifest_dir> <newick_dir> <out.tsv>

For every cohort x tree (9 x 50): run a textbook Saitou-Nei neighbour joining (own numpy code,
not Biopython) on the sealed cosine matrix, take the unrooted split set, and compare with
(1) the split set of the stored tree_cosine.newick from Wright (must be identical), and
(2) the stored score_cosine.json: rf, false_negative, false_positive, true_positive,
    clade_recovery, polytomy_aware_exact_recovery, n_truth_splits, n_estimate_splits —
    all recomputed here from split sets alone (RF = |S_est xor S_truth|).
Also checks the truth split set against the manifest with the same independent code as stage 1.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

D, MAN, NW, OUT = (Path(p) for p in sys.argv[1:5])

def truth_splits(tree):
    rows = [json.loads(l) for l in open(MAN / f"{tree}.manifest.jsonl") if l.strip()]
    kids = defaultdict(list)
    for r in rows: kids[r["parent_id"]].append(r["node_id"])
    leaves = frozenset(r["node_id"] for r in rows if not kids[r["node_id"]])
    under = {}
    def rec(n):
        s = frozenset([n]) if not kids[n] else frozenset().union(*(rec(c) for c in kids[n]))
        under[n] = s; return s
    for c in kids["root"]: rec(c)
    return {c for s in under.values() if (c := canon(s, leaves))}, leaves

def canon(side, leaves):
    side = frozenset(side) & leaves; other = leaves - side
    if len(side) <= 1 or len(other) <= 1: return None
    if len(side) != len(other): return side if len(side) < len(other) else other
    return min(side, other, key=lambda v: tuple(sorted(v)))

def nj(labels, D):
    """Textbook neighbour joining; returns the set of canonical unrooted splits of the result."""
    n = len(labels); D = D.astype(float).copy()
    clusters = [frozenset([l]) for l in labels]; leaves = frozenset(labels); splits = set()
    while len(clusters) > 3:
        m = len(clusters); r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]; np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j: i, j = j, i
        new = clusters[i] | clusters[j]
        c = canon(new, leaves)
        if c: splits.add(c)
        dk = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        keep = [k for k in range(m) if k not in (i, j)]
        D2 = np.zeros((m - 1, m - 1)); D2[:-1, :-1] = D[np.ix_(keep, keep)]; D2[-1, :-1] = dk[keep]; D2[:-1, -1] = dk[keep]
        clusters = [clusters[k] for k in keep] + [new]; D = D2
    return splits

def newick_splits(text, leaves):
    """Parse a Newick string and return its canonical unrooted splits (own tiny parser)."""
    pos = 0; splits = set()
    def parse():
        nonlocal pos
        if text[pos] == "(":
            pos += 1; members = set()
            while True:
                members |= parse()
                if text[pos] == ",": pos += 1; continue
                if text[pos] == ")": pos += 1; break
            # skip label / branch length
            while pos < len(text) and text[pos] not in ",);": pos += 1
            c = canon(members, leaves)
            if c: splits.add(c)
            return members
        start = pos
        while text[pos] not in ",);:": pos += 1
        name = text[start:pos]
        while pos < len(text) and text[pos] not in ",);": pos += 1
        return {name}
    parse(); return splits

fails = []; rows_out = []; n = 0
for cohort in sorted(p.name for p in D.iterdir() if (p / "receipt.json").exists()):
    for t in sorted(p for p in (D / cohort / "analysis").iterdir() if p.is_dir()):
        a = next(t.glob("*_leaf_analysis")); tree = t.name
        m = np.load(a / "distance_matrix_cosine.npy"); models = json.load(open(a / "models.json")); sc = json.load(open(a / "score_cosine.json"))
        S_truth, leaves = truth_splits(tree); assert set(models) == set(leaves)
        S_mine = nj(models, m)
        S_stored = newick_splits(open(NW / cohort / f"{tree}.newick").read().strip(), leaves)
        tp = len(S_mine & S_truth); fn = len(S_truth - S_mine); fp = len(S_mine - S_truth); rf = fn + fp
        clade = tp / len(S_truth) if S_truth else None; paer = (fn == 0)
        same_tree = (S_mine == S_stored)
        same_score = (sc["rf"] == rf and sc["false_negative"] == fn and sc["false_positive"] == fp and sc["true_positive"] == tp
                      and sc["n_truth_splits"] == len(S_truth) and sc["n_estimate_splits"] == len(S_mine) and bool(sc["polytomy_aware_exact_recovery"]) == paer
                      and (clade is None or abs(sc["clade_recovery"] - clade) < 1e-12))
        n += 1
        if not (same_tree and same_score): fails.append((cohort, tree, same_tree, same_score, sc["rf"], rf, sc["false_negative"], fn))
        rows_out.append((cohort, tree, len(leaves), len(S_truth), len(S_mine), tp, fn, fp, rf, "" if clade is None else f"{clade:.4f}", "yes" if same_tree else "NO", "yes" if same_score else "NO"))
with open(OUT, "w") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["cohort", "tree", "n_leaves", "truth_splits", "estimate_splits", "TP", "FN", "FP", "RF", "clade_recovery", "own_NJ==stored_tree", "own_score==stored_score"]); w.writerows(rows_out)
print(f"{n} cohort x tree reconstructions re-scored with an independent NJ + split/RF implementation")
print(f"own NJ split set == stored Wright tree: {sum(1 for r in rows_out if r[10]=='yes')}/{n};  own scores == stored score_cosine.json: {sum(1 for r in rows_out if r[11]=='yes')}/{n}")
for f in fails[:10]: print("  MISMATCH", f)
print("PASS verify_reconstruction" if not fails else f"FAIL verify_reconstruction ({len(fails)} mismatches)"); sys.exit(1 if fails else 0)
