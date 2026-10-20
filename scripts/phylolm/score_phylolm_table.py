#!/usr/bin/env python3
"""Score PhyloLM controlled-tree recovery with the five Table-2 metrics.

The standard PhyloLM post-step (codex/submit_phylolm_training_tree.sh ->
scripts/rescore_recovery.py) only aggregates clade recovery / FN / p-vs-label.
This adds the two remaining columns needed to line PhyloLM up against the
weight-based Table 2 (tab:variants) + Fig (fig:coherence_recovery):

  RF, FN          polytomy-aware, cosine  (score_hf_zoo_recovery.score_one_recon)
  clade recovery  1 - FN / |truth splits|
  exact recovery  % of runs with FN == 0  (binomial SE)
  additivity A    four-point additivity of the PhyloLM distance matrix.
                    - A_info : mean over TRUTH-INFORMATIVE quartets (the abstract's
                      A = 0.98 number; needs the truth tree only to *select*
                      informative quartets, the score itself is label-free)
                    - A_all  : mean over ALL quartets (Fig coherence_recovery's mean_A)
  oracle margin   bottleneck Atteson margin = min fitted internal edge length over
                    2 * max non-additivity error (ORACLE: uses the truth topology)

Every quantity reuses the exact math already in the repo, so the PhyloLM numbers
are apples-to-apples with the whitebox rows.

Reads, per run, the layout that run_phylolm_training_tree.py writes:
  results/<group>/run_*/weight/avg_distance_cosine.npy
  results/<group>/run_*/weight/distance_metadata.json      (model_names, D order)
  results/<group>/run_*/weight/avg_reconstructed_cosine.newick
  outputs/<group>/run_*/tree.newick                         (truth)

Usage
-----
    # one or more explicit _phylolm groups:
    python codex/phylolm/score_phylolm_table.py \
        --groups runs_llama1b_base_qkv_r64_phylolm runs_llama1b_base_full_ft_phylolm

    # or auto-discover every results/*_phylolm group:
    python codex/phylolm/score_phylolm_table.py --all

Writes results/aggregate/recovery_rescore_phylolm/table_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("PHYLOLM_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path(__file__).resolve().parent))  # score_hf_zoo_recovery, atteson_margin

# Clade recovery (RF/FN/clade/exact) + four-point helpers -- explicit-path scorer.
import score_hf_zoo_recovery as hz  # load_tree, score_one_recon, fourpoint_A, quartet_truth_pairing
# Oracle Atteson margin (bottleneck) in distance space.
import atteson_margin as am          # parse_newick, run_margin


def _load_D(weight_dir: Path) -> np.ndarray | None:
    f = weight_dir / "avg_distance_cosine.npy"
    if not f.exists():
        return None
    D = np.abs(np.load(f))
    if D.ndim == 3:            # (layers, n, n) -> layer-average (phylolm writes 1 layer)
        D = D.mean(axis=0)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        return None
    return D


def _additivity(D: np.ndarray, names: list[str], truth_tree, max_quartets: int,
                tau: float, seed: int) -> dict:
    """Mean four-point A over informative and all quartets, + informative split accuracy."""
    m = len(names)
    quartets = list(itertools.combinations(range(m), 4))
    if max_quartets and len(quartets) > max_quartets:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(quartets), max_quartets, replace=False)
        quartets = [quartets[t] for t in sel]
    info_A, all_A = [], []
    info_correct = info_total = 0
    for (i, j, k, l) in quartets:
        s_ij_kl = D[i, j] + D[k, l]
        s_ik_jl = D[i, k] + D[j, l]
        s_il_jk = D[i, l] + D[j, k]
        A, best = hz.fourpoint_A(s_ij_kl, s_ik_jl, s_il_jk)
        all_A.append(A)
        tp = hz.quartet_truth_pairing(truth_tree, (names[i], names[j], names[k], names[l]))
        if tp is not None:
            info_A.append(A)
            info_total += 1
            info_correct += int(best == tp)
    return {
        "A_all": float(np.mean(all_A)) if all_A else float("nan"),
        "A_info": float(np.mean(info_A)) if info_A else float("nan"),
        "frac_clean_info": float(np.mean(np.array(info_A) > tau)) if info_A else float("nan"),
        "quartet_split_acc": (info_correct / info_total) if info_total else float("nan"),
        "n_info_quartets": info_total,
    }


def score_run(group: str, run_dir: Path, nperm: int, max_quartets: int, tau: float,
              seed: int, min_leaves: int) -> dict | None:
    run = run_dir.name
    wdir = run_dir / "weight"
    truth_path = REPO / "outputs" / group / run / "tree.newick"
    recon_path = wdir / "avg_reconstructed_cosine.newick"
    meta_path = wdir / "distance_metadata.json"
    if not (truth_path.exists() and recon_path.exists() and meta_path.exists()):
        return None

    D = _load_D(wdir)
    if D is None:
        return None
    meta = json.loads(meta_path.read_text())
    names = [n.split("/")[-1] for n in meta["model_names"]]
    if len(names) != D.shape[0] or len(names) < min_leaves:
        return None

    truth_tree = hz.load_tree(truth_path)
    # Align D to the truth leaf set (defensive; phylolm D is already leaf-only).
    truth_leaves = set(hz.leaf_labels(truth_tree))
    idx = [i for i, n in enumerate(names) if n in truth_leaves]
    if len(idx) < min_leaves:
        return None
    names = [names[i] for i in idx]
    D = D[np.ix_(idx, idx)]

    rng = random.Random(seed)
    clade = hz.score_one_recon(truth_tree, recon_path, nperm, rng)
    if clade is None:
        return None

    add = _additivity(D, names, truth_tree, max_quartets, tau, seed)

    # Oracle Atteson margin: parse truth via atteson's own parser, fit edges to D.
    margin = am.run_margin(D, names, am.parse_newick(truth_path.read_text()))
    if margin is None:
        margin = {"bottleneck": float("nan"), "mean_margin": float("nan"),
                  "frac_resolvable": float("nan")}

    return {
        "group": group, "run": run, "leaves": clade["leaves"],
        "truth_splits": clade["truth_splits"],
        "FN": clade["FN"], "FP": clade["FP"], "RF": clade["RF"],
        "clade_recovery": clade["clade_recovery"], "exact": clade["refinement"],
        "p_FN_vs_label": clade["p_FN_vs_label"],
        "A_info": add["A_info"], "A_all": add["A_all"],
        "frac_clean_info": add["frac_clean_info"],
        "quartet_split_acc": add["quartet_split_acc"],
        "oracle_margin": margin["bottleneck"], "mean_margin": margin["mean_margin"],
        "frac_resolvable": margin["frac_resolvable"],
    }


def _mean_se(vals: list[float]) -> tuple[float, float]:
    a = np.array([v for v in vals if v == v], dtype=float)  # drop NaN
    if a.size == 0:
        return float("nan"), float("nan")
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), se


def aggregate(group: str, per_run: list[dict]) -> dict:
    n = len(per_run)
    clade_m, clade_se = _mean_se([r["clade_recovery"] for r in per_run])
    rf_m, rf_se = _mean_se([r["RF"] for r in per_run])
    fn_m, fn_se = _mean_se([r["FN"] for r in per_run])
    exact_p = np.mean([r["exact"] for r in per_run]) if n else float("nan")
    exact_se = float(np.sqrt(exact_p * (1 - exact_p) / n)) if n else float("nan")
    ainfo_m, ainfo_se = _mean_se([r["A_info"] for r in per_run])
    aall_m, aall_se = _mean_se([r["A_all"] for r in per_run])
    marg_m, marg_se = _mean_se([r["oracle_margin"] for r in per_run])
    return {
        "group": group, "n": n,
        "clade_recovery": clade_m, "clade_se": clade_se,
        "exact": exact_p, "exact_se": exact_se,
        "RF": rf_m, "RF_se": rf_se, "FN": fn_m, "FN_se": fn_se,
        "A_info": ainfo_m, "A_info_se": ainfo_se,
        "A_all": aall_m, "A_all_se": aall_se,
        "oracle_margin": marg_m, "oracle_margin_se": marg_se,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--groups", nargs="+", help="e.g. runs_llama1b_base_qkv_r64_phylolm")
    g.add_argument("--all", action="store_true", help="auto-discover results/*_phylolm groups")
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--max-quartets", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min-leaves", type=int, default=4, help="Table 2 uses >=4 leaves.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/aggregate/recovery_rescore_phylolm")
    args = ap.parse_args()

    if args.all:
        groups = sorted({Path(p).parts[-3] for p in
                         glob.glob(str(REPO / "results" / "*_phylolm" / "run_*" / "weight"))})
    else:
        groups = args.groups
    if not groups:
        sys.exit("No _phylolm groups found. Run codex/submit_phylolm_training_tree.sh first.")

    args.out.mkdir(parents=True, exist_ok=True)
    per_run_all: list[dict] = []
    summary: list[dict] = []
    for group in groups:
        run_dirs = sorted(glob.glob(str(REPO / "results" / group / "run_*")))
        rows = []
        for rd in run_dirs:
            r = score_run(group, Path(rd), args.nperm, args.max_quartets, args.tau,
                          args.seed, args.min_leaves)
            if r:
                rows.append(r)
        if not rows:
            print(f"[warn] {group}: no scorable runs (>= {args.min_leaves} leaves).", file=sys.stderr)
            continue
        per_run_all.extend(rows)
        summary.append(aggregate(group, rows))

    if not summary:
        sys.exit("Nothing scored.")

    # Per-run CSV.
    pr_fields = list(per_run_all[0].keys())
    with (args.out / "per_run_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pr_fields)
        w.writeheader(); w.writerows(per_run_all)

    # Group summary CSV.
    sm_fields = list(summary[0].keys())
    with (args.out / "table_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sm_fields)
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()})

    # Console table matching the Table 2 columns + A / oracle margin.
    print(f"\nPhyloLM controlled-tree recovery (cosine, >= {args.min_leaves} leaves)\n")
    hdr = (f"{'group':44}{'n':>4}{'clade%':>9}{'exact%':>9}"
           f"{'RF':>7}{'FN':>7}{'A_info':>8}{'A_all':>8}{'margin':>8}")
    print(hdr); print("-" * len(hdr))
    for s in sorted(summary, key=lambda x: x["clade_recovery"]):
        print(f"{s['group']:44}{s['n']:>4}"
              f"{100*s['clade_recovery']:>8.1f}{100*s['exact']:>9.1f}"
              f"{s['RF']:>7.2f}{s['FN']:>7.2f}"
              f"{s['A_info']:>8.3f}{s['A_all']:>8.3f}{s['oracle_margin']:>8.2f}")
    print(f"\nWrote {args.out}/table_metrics.csv and per_run_metrics.csv")


if __name__ == "__main__":
    main()
