#!/usr/bin/env python
"""NJ known answer: an additive distance matrix from a hand-built 7-leaf tree with a polytomy must be
reconstructed exactly (all truth splits recovered, FN = 0) by the project's neighbor_joining_newick, and
the project's split scorer must agree with a hand count.  usage: verify_nj_known_answer.py <wt_root>"""
import sys, itertools
sys.path.insert(0, sys.argv[1] + "/src")
import numpy as np
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.splits import splits_from_newick_text
from weighttraits.phylo.recovery import score_split_recovery
# tree: ((A:1,B:2):3,(C:1,D:1,E:2):2,(F:4,G:1):1)  -- (C,D,E) is a polytomy
edges = {"A": ("x", 1), "B": ("x", 2), "C": ("y", 1), "D": ("y", 1), "E": ("y", 2), "F": ("z", 4), "G": ("z", 1), "x": ("r", 3), "y": ("r", 2), "z": ("r", 1)}
def path(n):
    out = []
    while n in edges: p, w = edges[n]; out.append((n, w)); n = p
    return out
leaves = list("ABCDEFG"); n = len(leaves); D = np.zeros((n, n))
for i, j in itertools.combinations(range(n), 2):
    pi, pj = dict(path(leaves[i])), dict(path(leaves[j]))
    d = sum(w for k, w in pi.items() if k not in pj) + sum(w for k, w in pj.items() if k not in pi)
    D[i, j] = D[j, i] = d
truth = {frozenset("AB"), frozenset("CDE"), frozenset("FG")}
est_splits, _ = splits_from_newick_text(neighbor_joining_newick(leaves, D))
score = score_split_recovery(truth, est_splits, set(leaves), set(leaves))
fn = len(truth - est_splits); fp = len(est_splits - truth)
print("estimated splits:", sorted(sorted(s) for s in est_splits)); print("truth splits recovered:", len(truth & est_splits), "/", len(truth), " FN", fn, " FP", fp, "(one extra split resolves the polytomy)")
print("project scorer:", {k: score[k] for k in ("rf", "false_negative", "false_positive", "clade_recovery", "polytomy_aware_exact_recovery") if k in score})
ok = fn == 0 and fp == 1 and score.get("false_negative") == 0 and score.get("rf") == 1 and score.get("clade_recovery") == 1.0
print("PASS verify_nj_known_answer" if ok else "FAIL verify_nj_known_answer"); sys.exit(0 if ok else 1)
