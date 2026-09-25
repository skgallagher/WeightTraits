#!/usr/bin/env python
"""Independent structural check of the 50 frozen paper trees.

Reads the topology manifests with plain Python (no WeightTraits code), verifies
the structure the paper describes, recomputes the eligibility rules, and prints
a per-tree fingerprint table.  Then cross-checks the split counts against the
WeightTraits truth-tree code path.

Exit 0 = every expectation met; the last line is PASS/FAIL.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

FROZEN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/training/confirm_paper_numbers")
TREES = FROZEN / "trees"
SUMMARY = json.load(open(FROZEN / "tree_set_summary.json"))

EXPECT = dict(n_trees=50, n_nodes=14, max_depth=4, min_leaves=4,
              excluded={"confirm_paper_tree_015", "confirm_paper_tree_020",
                        "confirm_paper_tree_037", "confirm_paper_tree_047"},
              n_topology=46, n_ordering=26, total_nonroot=641, total_internal=276,
              total_leaves=365, total_truth_splits=117)

fails = []
def expect(cond, msg):
    if not cond:
        fails.append(msg)
    return cond

def load(tree_id):
    rows = [json.loads(l) for l in open(TREES / f"{tree_id}.manifest.jsonl") if l.strip()]
    return rows

def leaf_sets(rows):
    """Return (leaves, children map, leafset under each node)."""
    children = defaultdict(list)
    for r in rows:
        children[r["parent_id"]].append(r["node_id"])
    ids = [r["node_id"] for r in rows]
    leaves = [n for n in ids if not children[n]]
    under = {}
    def rec(n):
        if not children[n]:
            under[n] = frozenset([n]); return under[n]
        s = frozenset().union(*(rec(c) for c in children[n]))
        under[n] = s; return s
    for c in children["root"]:
        rec(c)
    return leaves, children, under

def canonical_newick(rows):
    """Deterministic leaf-only Newick with unary chains collapsed (topology fingerprint)."""
    leaves, children, under = leaf_sets(rows)
    def rec(n):
        kids = children[n]
        if not kids:
            return n
        if len(kids) == 1:
            return rec(kids[0])
        return "(" + ",".join(sorted(rec(k) for k in kids)) + ")"
    tops = children["root"]
    return "(" + ",".join(sorted(rec(k) for k in tops)) + ");"

tree_ids = [t["tree_id"] for t in SUMMARY["trees"]]
expect(len(tree_ids) == EXPECT["n_trees"], f"summary lists {len(tree_ids)} trees, expected 50")
expect(sorted(tree_ids) == [f"confirm_paper_tree_{i:03d}" for i in range(1, 51)], "tree ids are not 001..050")

rows_by_tree = {}
fingerprints = {}
informative = {}
ordering_ok = {}
tot_nonroot = tot_internal = tot_leaves = 0
print(f"{'tree':24s} {'n':>2s} {'leaves':>6s} {'depth':>5s} {'poly':>4s} {'root_kids':>9s} {'inf_splits':>10s}  topology-sha   (n = trained nodes, max 14)")
for tid, summ in zip(tree_ids, SUMMARY["trees"]):
    rows = load(tid); rows_by_tree[tid] = rows
    ids = [r["node_id"] for r in rows]
    expect(len(rows) == summ["n_nodes"], f"{tid}: {len(rows)} rows != summary n_nodes {summ['n_nodes']}")
    expect(len(rows) <= EXPECT["n_nodes"], f"{tid}: {len(rows)} rows > 14")
    expect(len(set(ids)) == len(ids), f"{tid}: duplicate node ids")
    known = set(ids) | {"root"}
    for r in rows:
        expect(r["parent_id"] in known, f"{tid}: {r['node_id']} has unknown parent {r['parent_id']}")
        expect(r["path"][0] == "root" and r["path"][-1] == r["node_id"], f"{tid}: bad path for {r['node_id']}")
        expect(r["depth"] == len(r["path"]) - 1, f"{tid}: depth/path mismatch for {r['node_id']}")
        expect(r["depth"] <= EXPECT["max_depth"], f"{tid}: depth {r['depth']} > 4")
        expect(r["grow"] == "train", f"{tid}: {r['node_id']} grow != train")
    leaves, children, under = leaf_sets(rows)
    expect(len(leaves) >= EXPECT["min_leaves"], f"{tid}: only {len(leaves)} leaves")
    expect(len(leaves) == summ["n_leaves"], f"{tid}: leaves {len(leaves)} != summary {summ['n_leaves']}")
    max_depth = max(r["depth"] for r in rows)
    expect(max_depth == summ["max_depth"], f"{tid}: max_depth {max_depth} != summary")
    n_poly = sum(1 for n in ids + ["root"] if len(children[n]) >= 3)
    expect(n_poly == summ["n_polytomies"], f"{tid}: polytomies {n_poly} != summary {summ['n_polytomies']}")
    # reachability: every node's path must chain through existing parents back to root
    for r in rows:
        p = r["path"]
        for a, b in zip(p, p[1:]):
            expect(b in children[a], f"{tid}: path edge {a}->{b} not in manifest")
    n_leaves = len(leaves)
    # informative unrooted splits of the leaf-only truth tree (dedupe unary chains)
    all_leaves = frozenset(leaves)
    def canon(side):
        other = all_leaves - side
        if len(side) <= 1 or len(other) <= 1:
            return None
        if len(side) != len(other):
            return side if len(side) < len(other) else other
        return min(side, other, key=lambda v: tuple(sorted(v)))
    splits = {c for n, s in under.items() if (c := canon(s)) is not None}
    informative[tid] = len(splits)
    # ordering eligibility: the depth-1 node's children define "root branches";
    # need >=2 branches each with >=1 leaf so same- and cross-branch pairs both exist
    branches = children["root"]                      # root branches = the base model's children
    branch_leaf_counts = [len(under[b]) for b in branches]
    same_pairs = sum(c * (c - 1) // 2 for c in branch_leaf_counts)
    cross_pairs = (n_leaves * (n_leaves - 1) // 2) - same_pairs
    ordering_ok[tid] = (same_pairs > 0 and cross_pairs > 0)
    tot_nonroot += len(rows); tot_leaves += n_leaves; tot_internal += len(rows) - n_leaves
    nw = canonical_newick(rows)
    fingerprints[tid] = hashlib.sha256(nw.encode()).hexdigest()
    print(f"{tid:24s} {len(rows):>2d} {n_leaves:>6d} {max_depth:>5d} {n_poly:>4d} {len(branches):>9d} {len(splits):>10d}  {fingerprints[tid][:12]}")

excluded = {t for t, k in informative.items() if k == 0}
n_topology = 50 - len(excluded)
n_ordering = sum(ordering_ok.values())
dups = [h for h, c in Counter(fingerprints.values()).items() if c > 1]

print()
print(f"non-root nodes total      {tot_nonroot:>5d}   expected {EXPECT['total_nonroot']}")
print(f"internal nodes total      {tot_internal:>5d}   expected {EXPECT['total_internal']}")
print(f"leaves total              {tot_leaves:>5d}   expected {EXPECT['total_leaves']}")
print(f"trees with 0 informative  {sorted(t[-3:] for t in excluded)}   expected {sorted(t[-3:] for t in EXPECT['excluded'])}")
print(f"topology-eligible trees   {n_topology:>5d}   expected {EXPECT['n_topology']}")
print(f"informative truth splits  {sum(informative.values()):>5d}   expected {EXPECT['total_truth_splits']} (over the 46)")
print(f"ordering-eligible trees   {n_ordering:>5d}   expected {EXPECT['n_ordering']}")
print(f"distinct topologies       {50-len(dups):>5d}   duplicate fingerprints: {len(dups)}")
leaf_hist = Counter(len(leaf_sets(r)[0]) for r in rows_by_tree.values())
print(f"leaf-count histogram      {dict(sorted(leaf_hist.items()))}   summary {SUMMARY['leaf_counts']}")

expect(tot_nonroot == EXPECT["total_nonroot"], "non-root total != 641")
expect(tot_internal == EXPECT["total_internal"], "internal total != 276")
expect(tot_leaves == EXPECT["total_leaves"], "leaf total != 365")
expect(excluded == EXPECT["excluded"], f"excluded set differs: {sorted(excluded)}")
expect(n_topology == EXPECT["n_topology"], "topology-eligible != 46")
expect(sum(informative.values()) == EXPECT["total_truth_splits"], "informative split total != 117")
expect(n_ordering == EXPECT["n_ordering"], f"ordering-eligible {n_ordering} != 26")
expect(not dups, "duplicate topologies found")
expect({int(k): v for k, v in SUMMARY["leaf_counts"].items()} == dict(leaf_hist), "leaf histogram != summary")

# ---- cross-check against the WeightTraits truth-tree code path -------------------
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from weighttraits.analysis.direct import truth_newick_from_manifest
    from weighttraits.phylo.splits import splits_from_newick_text
    lib_counts = {}
    for tid in tree_ids:
        nw = truth_newick_from_manifest(FROZEN / "assigned_manifests" / f"{tid}.manifest.jsonl")
        lib_splits, lib_leaves = splits_from_newick_text(nw)
        lib_counts[tid] = len(lib_splits)
    agree = sum(1 for t in tree_ids if lib_counts[t] == informative[t])
    print(f"library split counts agree with independent counts on {agree}/50 trees")
    expect(agree == 50, "library vs independent informative-split counts disagree")
except Exception as exc:  # report but do not hide the independent result
    print(f"library cross-check skipped: {type(exc).__name__}: {exc}")
    fails.append("library cross-check could not run")

print()
if fails:
    for f in fails[:25]:
        print("  FAIL:", f)
    print(f"FAIL verify_trees ({len(fails)} problems)"); sys.exit(1)
print("PASS verify_trees"); sys.exit(0)
