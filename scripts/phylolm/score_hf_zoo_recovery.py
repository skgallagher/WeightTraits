#!/usr/bin/env python3
"""Polytomy-aware recovery scoring for hf_zoo (organic HuggingFace) runs.

`scripts/rescore_recovery.py` and `scripts/fourpoint_additivity.py` do this same
scoring for the standard pipeline, but both are hardcoded to glob
`results/runs_*/run_*/...` / `outputs/runs_*/run_*/tree.newick`. hf_zoo runs live
at `results/hf_zoo/<tier>/weight_*` with a truth tree at
`outputs/hf_zoo/<tier>_resolved/tree.newick`, so every hf_zoo write-up so far
(Llama nodilip, twoclade, etc.) has recomputed this logic ad hoc and by hand.
This script generalizes the same two analyses behind explicit file paths instead
of directory globbing, so it works for any hf_zoo tier (or any tree pair, really).

1. Clade recovery (FN/FP/RF vs a polytomy truth), same definitions as
   `scripts/rescore_recovery.py`:
     FN = true clades the reconstruction MISSED           <- real error
     FP = extra binary-forced edges not in the true tree   <- resolution artifact
     clade_recovery = 1 - FN / |truth splits|
     refinement      = (FN == 0)   (recon contains the whole true tree)
   Plus a label-shuffle permutation p-value for FN (p_FN_vs_label).

2. Four-point additivity (`scripts/fourpoint_additivity.py`'s A formula) split
   into TRUTH-INFORMATIVE quartets (the true tree resolves that quartet, i.e.
   induces a non-star topology on those 4 leaves) vs uninformative ones (the
   true tree is a star on that quartet -- usually because all 4 leaves sit in
   the same unresolved polytomy). Reports mean_A, frac_clean (A>tau), and
   quartet_split_accuracy (does the argmin pair-sum match the true split) for
   each group.

Usage
-----
    # Clade recovery for one reconstruction newick against the truth tree:
    python codex/score_hf_zoo_recovery.py clades \\
        --reference outputs/hf_zoo/d_mistral_7b_resolved/tree.newick \\
        --recon results/hf_zoo/d_mistral_7b/weight_frechet/avg_reconstructed_cosine.newick \\
        --nperm 2000

    # Same, but score every {consensus,avg_reconstructed,frechet}_<metric>.newick
    # found in a directory in one pass:
    python codex/score_hf_zoo_recovery.py clades \\
        --reference outputs/hf_zoo/d_mistral_7b_resolved/tree.newick \\
        --recon-dir results/hf_zoo/d_mistral_7b/weight_frechet \\
        --nperm 2000 \\
        --out results/hf_zoo/d_mistral_7b/weight_frechet/recovery_rescore

    # Four-point additivity from a saved distance cube:
    python codex/score_hf_zoo_recovery.py quartets \\
        --reference outputs/hf_zoo/d_mistral_7b_resolved/tree.newick \\
        --distances results/hf_zoo/d_mistral_7b/weight_common/avg_distance_cosine.npy \\
        --metadata  results/hf_zoo/d_mistral_7b/weight_common/distance_metadata.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import random
from pathlib import Path

import numpy as np
import dendropy

EPS = 1e-12


# ---------------------------------------------------------------------------
# Clade recovery (ported from scripts/rescore_recovery.py, generalized to
# take explicit file paths rather than globbing results/runs_*)
# ---------------------------------------------------------------------------

def load_tree(path: Path) -> dendropy.Tree:
    return dendropy.Tree.get(path=str(path), schema="newick", preserve_underscores=True)


def leaf_labels(t: dendropy.Tree) -> frozenset:
    return frozenset(l.taxon.label for l in t.leaf_node_iter())


def canon(side, common_fs, ref):
    side = frozenset(side)
    return side if ref not in side else common_fs - side


def splits_from_tree(t: dendropy.Tree, common: frozenset, ref: str) -> set:
    """Unrooted nontrivial split set of a tree, induced on `common`, canonicalized
    so the side WITHOUT `ref` represents each split (both sides must have >=2
    leaves -- matches rescore_recovery.py's `_get_splits`)."""
    t = t.clone(depth=1)
    keep = [tx for tx in t.taxon_namespace if tx.label in common]
    t.retain_taxa(keep)
    n = len(common)
    cfs = frozenset(common)
    S = set()
    for nd in t.postorder_node_iter():
        if nd.is_leaf() or nd.parent_node is None:
            continue
        side = frozenset(l.taxon.label for l in nd.leaf_iter())
        if len(side) < 2 or n - len(side) < 2:
            continue
        S.add(canon(side, cfs, ref))
    return S


def perm_pvalues(recon_splits, truth_splits, common, ref, nperm, rng):
    # Sorted, not list(set): set order follows Python's per-process string hashing, which made
    # these p-values differ between runs.
    common = sorted(common)
    common_fs = frozenset(common)
    fn_obs = len(truth_splits - recon_splits)
    rf_obs = len(truth_splits ^ recon_splits)
    base = [list(s) for s in recon_splits]
    fn_le = rf_le = 0
    for _ in range(nperm):
        perm = dict(zip(common, rng.sample(common, len(common))))
        ps = {canon((perm[x] for x in s), common_fs, ref) for s in base}
        fn_le += len(truth_splits - ps) <= fn_obs
        rf_le += len(truth_splits ^ ps) <= rf_obs
    nb = nperm + 1
    return {
        "p_FN_vs_label": (fn_le + 1) / nb,
        "p_RF_vs_label": (rf_le + 1) / nb,
        "FN": fn_obs,
        "RF": rf_obs,
    }


def score_one_recon(truth_tree: dendropy.Tree, recon_path: Path, nperm: int, rng: random.Random) -> dict | None:
    recon_tree = load_tree(recon_path)
    common = leaf_labels(truth_tree) & leaf_labels(recon_tree)
    if len(common) < 4:
        return None
    ref = sorted(common)[0]
    truth_splits = splits_from_tree(truth_tree, common, ref)
    recon_splits = splits_from_tree(recon_tree, common, ref)
    if not truth_splits:
        return None
    fn = len(truth_splits - recon_splits)
    fp = len(recon_splits - truth_splits)
    rf = len(truth_splits ^ recon_splits)
    pv = perm_pvalues(recon_splits, truth_splits, common, ref, nperm, rng)
    recovered = sorted(truth_splits & recon_splits, key=len)
    missed = sorted(truth_splits - recon_splits, key=len)
    return {
        "recon": recon_path.name,
        "leaves": len(common),
        "truth_splits": len(truth_splits),
        "recon_splits": len(recon_splits),
        "FN": fn,
        "FP": fp,
        "RF": rf,
        "clade_recovery": round(1 - fn / len(truth_splits), 4),
        "refinement": int(fn == 0),
        "p_FN_vs_label": round(pv["p_FN_vs_label"], 4),
        "p_RF_vs_label": round(pv["p_RF_vs_label"], 4),
        "recovered_clades": [sorted(s) for s in recovered],
        "missed_clades": [sorted(s) for s in missed],
    }


def cmd_clades(args: argparse.Namespace) -> None:
    truth_tree = load_tree(args.reference)
    rng = random.Random(args.seed)

    recon_paths: list[Path] = []
    if args.recon:
        recon_paths.extend(Path(p) for p in args.recon)
    if args.recon_dir:
        d = Path(args.recon_dir)
        stems = args.stems or ["consensus", "avg_reconstructed", "frechet"]
        for stem in stems:
            recon_paths.extend(sorted(d.glob(f"{stem}_*.newick")))
    if not recon_paths:
        raise SystemExit("Provide --recon <file>... and/or --recon-dir.")

    rows = []
    for rp in recon_paths:
        if not rp.exists():
            print(f"  (skip, missing) {rp}")
            continue
        row = score_one_recon(truth_tree, rp, args.nperm, rng)
        if row is None:
            print(f"  (skip, <4 common leaves or no truth splits) {rp}")
            continue
        rows.append(row)

    print(f"\n{'recon':40}{'leaves':>7}{'truth':>7}{'FN':>5}{'FP':>5}{'RF':>5}"
          f"{'recov%':>8}{'refine':>8}{'p_FNvL':>9}")
    for r in rows:
        print(f"{r['recon']:40}{r['leaves']:>7}{r['truth_splits']:>7}"
              f"{r['FN']:>5}{r['FP']:>5}{r['RF']:>5}"
              f"{100*r['clade_recovery']:>7.1f}{r['refinement']:>8}"
              f"{r['p_FN_vs_label']:>9.4f}")
    for r in rows:
        if r["missed_clades"]:
            print(f"\n{r['recon']}: MISSED clades:")
            for c in r["missed_clades"]:
                print(f"    {c}")
        else:
            print(f"\n{r['recon']}: all {r['truth_splits']} true clade(s) recovered (FN=0).")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        fields = [k for k in rows[0].keys() if k not in ("recovered_clades", "missed_clades")]
        with (out / "clade_recovery.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fields})
        (out / "clade_recovery_detail.json").write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {out}/clade_recovery.csv and clade_recovery_detail.json")


# ---------------------------------------------------------------------------
# Four-point additivity, split by whether the true tree resolves the quartet
# ---------------------------------------------------------------------------

def fourpoint_A(s_ij_kl: float, s_ik_jl: float, s_il_jk: float) -> tuple[float, int]:
    """Return (A, best_pairing_index) for one quartet's three pair-sums, where
    best_pairing_index in {0,1,2} indicates which pairing (ij|kl, ik|jl, il|jk)
    had the SMALLEST sum (= the pairing the additive metric implies as the true
    split, since the split-defining pair-sum is the smallest of the three)."""
    s = np.array([s_ij_kl, s_ik_jl, s_il_jk])
    order = np.argsort(s)          # ascending
    s3, s2, s1 = s[order[0]], s[order[1]], s[order[2]]
    A = (s2 - s3) / (s1 - s3 + EPS)
    return float(A), int(order[0])


def quartet_truth_pairing(truth_tree: dendropy.Tree, quartet: tuple[str, str, str, str]) -> int | None:
    """Return which pairing (0=ab|cd, 1=ac|bd, 2=ad|bc) the true tree resolves
    this quartet to, or None if the true tree is a star on these 4 leaves
    (uninformative quartet)."""
    a, b, c, d = quartet
    t = truth_tree.clone(depth=1)
    keep = [tx for tx in t.taxon_namespace if tx.label in quartet]
    t.retain_taxa(keep)
    t.suppress_unifurcations()
    for nd in t.postorder_node_iter():
        if nd.is_leaf() or nd.parent_node is None:
            continue
        side = frozenset(l.taxon.label for l in nd.leaf_iter())
        if len(side) != 2:
            continue
        # side is a resolved pair -- map it to a pairing index
        if side == {a, b} or side == {c, d}:
            return 0
        if side == {a, c} or side == {b, d}:
            return 1
        if side == {a, d} or side == {b, c}:
            return 2
    return None  # star: no resolved 2-2 split among these 4 leaves


def cmd_quartets(args: argparse.Namespace) -> None:
    truth_tree = load_tree(args.reference)
    D = np.abs(np.load(args.distances))
    meta = json.loads(Path(args.metadata).read_text())
    names = meta["model_names"]
    truth_leaves = leaf_labels(truth_tree)
    idx = [i for i, n in enumerate(names) if n in truth_leaves]
    if len(idx) < 4:
        raise SystemExit(f"Only {len(idx)} of {len(names)} distance-matrix names are in the truth tree's leaf set.")
    names = [names[i] for i in idx]
    D = D[np.ix_(idx, idx)]
    if D.ndim == 3:  # full cube (layers, n, n) -- average like avg_distance would
        D = D.mean(axis=0)

    m = len(names)
    quartets = list(itertools.combinations(range(m), 4))
    if args.max_quartets and len(quartets) > args.max_quartets:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(quartets), args.max_quartets, replace=False)
        quartets = [quartets[t] for t in sel]

    info_A, uninfo_A = [], []
    info_correct = 0
    info_total = 0
    for (i, j, k, l) in quartets:
        a, b, c, d = names[i], names[j], names[k], names[l]
        s_ij_kl = D[i, j] + D[k, l]
        s_ik_jl = D[i, k] + D[j, l]
        s_il_jk = D[i, l] + D[j, k]
        A, best = fourpoint_A(s_ij_kl, s_ik_jl, s_il_jk)
        truth_pairing = quartet_truth_pairing(truth_tree, (a, b, c, d))
        if truth_pairing is None:
            uninfo_A.append(A)
        else:
            info_A.append(A)
            info_total += 1
            info_correct += int(best == truth_pairing)

    def summarize(arr, tau):
        arr = np.array(arr)
        if arr.size == 0:
            return None
        return {
            "n": int(arr.size),
            "mean_A": round(float(arr.mean()), 4),
            "median_A": round(float(np.median(arr)), 4),
            "frac_clean_A_gt_tau": round(float((arr > tau).mean()), 4),
        }

    print(f"\nAll quartets: n={len(quartets)}")
    s_all = summarize(info_A + uninfo_A, args.tau)
    print(f"  {s_all}")
    print(f"\nTruth-informative quartets (true tree resolves a 2-2 split):")
    s_info = summarize(info_A, args.tau)
    print(f"  {s_info}")
    if info_total:
        print(f"  truth_split_quartet_accuracy = {100*info_correct/info_total:.1f}%  ({info_correct}/{info_total})")
    print(f"\nUninformative quartets (true tree is a star on these 4 leaves):")
    s_uninfo = summarize(uninfo_A, args.tau)
    print(f"  {s_uninfo}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "all": s_all,
            "informative": s_info,
            "informative_quartet_split_accuracy_pct": round(100*info_correct/info_total, 2) if info_total else None,
            "uninformative": s_uninfo,
        }
        (out / "fourpoint_additivity.json").write_text(json.dumps(result, indent=2))
        print(f"\nWrote {out}/fourpoint_additivity.json")


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("clades", help="Polytomy-aware clade recovery (FN/FP/RF) vs a truth tree.")
    pc.add_argument("--reference", required=True, type=Path, help="Truth newick (e.g. outputs/hf_zoo/<tier>_resolved/tree.newick).")
    pc.add_argument("--recon", nargs="*", default=None, help="One or more reconstructed newick files to score.")
    pc.add_argument("--recon-dir", default=None, help="Directory to auto-glob {stem}_{metric}.newick from.")
    pc.add_argument("--stems", nargs="*", default=None, help="Stems to glob in --recon-dir (default: consensus avg_reconstructed frechet).")
    pc.add_argument("--nperm", type=int, default=2000)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--out", default=None, help="Optional directory to write clade_recovery.csv/.json.")
    pc.set_defaults(func=cmd_clades)

    pq = sub.add_parser("quartets", help="Four-point additivity split by truth-informative vs uninformative quartets.")
    pq.add_argument("--reference", required=True, type=Path)
    pq.add_argument("--distances", required=True, help="avg_distance_<metric>.npy (2D) or full_distance_<metric>.npy (3D, will be layer-averaged).")
    pq.add_argument("--metadata", required=True, help="distance_metadata.json (for model_names / leaf order).")
    pq.add_argument("--tau", type=float, default=0.5)
    pq.add_argument("--max-quartets", type=int, default=20000)
    pq.add_argument("--seed", type=int, default=0)
    pq.add_argument("--out", default=None)
    pq.set_defaults(func=cmd_quartets)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
