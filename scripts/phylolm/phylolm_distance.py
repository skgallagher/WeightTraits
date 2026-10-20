#!/usr/bin/env python3
"""Turn PhyloLM allele populations into (a) a neighbor-joining tree and (b) a
standard distance cube, for polytomy-aware scoring against the Mistral truth tree.

Reads every population JSON written by collect_phylolm.py, computes the PhyloLM
Nei similarity matrix (vendored verbatim in phylolm_core.nei_similarity), converts
to PhyloLM's genetic distance

    dist = -log(max(sim, 1e-3))          # colab_notebook cell 16

and emits:

  <out>/phylolm_sim.npy, phylolm_dist.npy, phylolm_matrix.csv   (raw matrices)
  <out>/phylolm_nj.newick                                       (PhyloLM NJ tree)
  <out>/avg_distance_<metric>.npy, full_distance_<metric>.npy   (distance cube)
  <out>/distance_metadata.json                                  (model_names = leaf ids)

The distance cube writes the SAME PhyloLM distance under all five metric names so
that `scripts/run_analysis.py --from_distances` (which loads cosine/l1/l2/
correlation/threshold unconditionally) reconstructs from the identical PhyloLM
distance -- this reuses the exact same NJ/Frechet + polytomy scorer already
validated on the whitebox hf_zoo cubes, giving a genuine apples-to-apples
blackbox-vs-whitebox number on the SAME models and SAME truth.

Scoring (after this + optionally run_analysis --from_distances):
    # direct on the PhyloLM NJ tree:
    python codex/score_hf_zoo_recovery.py clades \
        --reference outputs/hf_zoo/d_mistral_7b_resolved/tree.newick \
        --recon <out>/phylolm_nj.newick
    # four-point additivity on the PhyloLM distance:
    python codex/score_hf_zoo_recovery.py quartets \
        --reference outputs/hf_zoo/d_mistral_7b_resolved/tree.newick \
        --distances <out>/avg_distance_cosine.npy \
        --metadata  <out>/distance_metadata.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import phylolm_core as pl

_METRICS = ("cosine", "l1", "l2", "correlation", "threshold")


# ---------------------------------------------------------------------------
# Neighbor joining -> newick (topology-faithful to PhyloLM's biopython NJ;
# self-contained so the distance step needs no old-biopython install).
# ---------------------------------------------------------------------------

def nj_newick(labels: list[str], D: np.ndarray) -> str:
    n = len(labels)
    if n < 3:
        raise ValueError("NJ needs >=3 leaves")
    # node id -> newick string; distances kept in a dict keyed by active node ids.
    nodes = {i: labels[i] for i in range(n)}
    dist = {(i, j): float(D[i, j]) for i in range(n) for j in range(n)}
    active = list(range(n))
    nxt = n

    def d(a, b):
        return dist[(a, b)] if (a, b) in dist else dist[(b, a)]

    while len(active) > 2:
        m = len(active)
        r = {a: sum(d(a, b) for b in active if b != a) for a in active}
        best = None
        bi = bj = None
        for x in range(m):
            for y in range(x + 1, m):
                a, b = active[x], active[y]
                q = (m - 2) * d(a, b) - r[a] - r[b]
                if best is None or q < best:
                    best, bi, bj = q, a, b
        # branch lengths (kept non-negative for a valid newick)
        dij = d(bi, bj)
        li = 0.5 * dij + (r[bi] - r[bj]) / (2 * (m - 2))
        lj = dij - li
        li = max(li, 0.0)
        lj = max(lj, 0.0)
        u = nxt
        nxt += 1
        nodes[u] = f"({nodes[bi]}:{li:.6f},{nodes[bj]}:{lj:.6f})"
        for a in active:
            if a in (bi, bj):
                continue
            dist[(u, a)] = 0.5 * (d(bi, a) + d(bj, a) - dij)
        active = [a for a in active if a not in (bi, bj)] + [u]

    # join the final two (unrooted): a--b
    a, b = active
    dab = d(a, b)
    return f"({nodes[a]}:{dab:.6f},{nodes[b]}:0.000000);"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--populations", required=True, type=Path,
                    help="Directory of <model_id>.json population files (from collect_phylolm.py).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="Floor for sim before -log (PhyloLM colab uses 1e-3).")
    ap.add_argument("--rename", default=None,
                    help="Optional JSON {population_id: truth_leaf_name} if ids differ from the truth newick.")
    args = ap.parse_args()

    rename = json.loads(Path(args.rename).read_text()) if args.rename else {}

    pop_files = sorted(Path(args.populations).glob("*.json"))
    if len(pop_files) < 3:
        sys.exit(f"Need >=3 population files in {args.populations}, found {len(pop_files)}")

    all_P: dict[str, list] = {}
    metas: dict[str, dict] = {}
    for f in pop_files:
        rec = pl.load_population(f)
        mid = rename.get(rec["model_id"], rec["model_id"])
        all_P[mid] = rec["P"]
        metas[mid] = rec.get("meta", {})

    model_names = sorted(all_P)
    sim = pl.compute_sim_matrix(model_names, all_P)
    dist = -np.log(np.maximum(sim, args.eps))
    np.fill_diagonal(dist, 0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "phylolm_sim.npy", sim)
    np.save(args.out / "phylolm_dist.npy", dist)

    with (args.out / "phylolm_matrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + model_names)
        for i, m in enumerate(model_names):
            w.writerow([m] + [f"{dist[i, j]:.6f}" for j in range(len(model_names))])

    newick = nj_newick(model_names, dist)
    (args.out / "phylolm_nj.newick").write_text(newick + "\n")

    # distance cube (same PhyloLM distance under every metric name)
    for metric in _METRICS:
        np.save(args.out / f"avg_distance_{metric}.npy", dist)
        np.save(args.out / f"full_distance_{metric}.npy", dist[None, :, :])

    (args.out / "distance_metadata.json").write_text(json.dumps({
        "layer_names": ["phylolm_nei_similarity"],
        "model_names": model_names,
        "nodes_mode": "leaves",
        "granularity": "phylolm_blackbox",
        "source": "codex/phylolm/phylolm_distance.py",
        "phylolm_commit": pl.PHYLOLM_COMMIT,
        "distance": "-log(max(nei_similarity, %g))" % args.eps,
        "n_models": len(model_names),
        "per_model_meta": metas,
    }, indent=2) + "\n")

    print(f"Models ({len(model_names)}): {model_names}")
    print("Nei similarity (min/median/max off-diagonal):")
    off = sim[~np.eye(len(model_names), dtype=bool)]
    print(f"  {off.min():.4f} / {np.median(off):.4f} / {off.max():.4f}")
    if (sim <= args.eps).any():
        print(f"  WARNING: {(sim <= args.eps).sum()} sim entries hit the {args.eps} floor "
              f"(clamped in -log). Consider larger G/N (see PhyloLM colab cell 16 note).")
    print(f"NJ tree -> {args.out / 'phylolm_nj.newick'}")
    print(f"Cube + metadata -> {args.out}")


if __name__ == "__main__":
    main()
