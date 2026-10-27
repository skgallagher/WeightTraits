#!/usr/bin/env python3
"""Weight-based trees for one documented family, scored against its documented tree (App. D).

The steps reproduce the paper's App. D settings:

1. Build one neighbor-joining tree per layer from the cosine cube (Biopython NJ, five-decimal
   Newick).
2. Read each tree into BHV space, clipping negative pendant lengths to zero.
3. Take the Fréchet mean with 25 epochs and seed 0 (tolerance 1e-9).
4. Also build an NJ tree on the layer-averaged matrix.

``weighttraits.paper.hf_families`` then scores each method, giving the documented clades
estimated and the all-quartet tree-likeness score A. It also reports how many per-layer trees
contain each documented clade. Pass ``--phylolm DIR`` (with ``phylolm_nj.newick``,
``phylolm_dist.npy``, and ``distance_metadata.json``) to score PhyloLM alongside the weights.

Writes ``frechet_cosine.newick``, ``avg_reconstructed_cosine.newick``, and ``scores.json`` into
the cube directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from weighttraits.paper.hf_families import (  # noqa: E402
    MethodInputs,
    per_layer_clade_support,
    score_family,
)
from weighttraits.phylo import bhv  # noqa: E402
from weighttraits.phylo.reconstruct import neighbor_joining_newick  # noqa: E402

FRECHET_EPOCHS = 25


def frechet_tree(cube: np.ndarray, labels: list[str]) -> str:
    import dendropy

    namespace = dendropy.TaxonNamespace([dendropy.Taxon(label) for label in labels])
    trees = []
    for layer in cube:
        tree = bhv.from_newick(neighbor_joining_newick(labels, layer), taxon_namespace=namespace)
        tree.pendant = {leaf: max(0.0, length) for leaf, length in tree.pendant.items()}
        trees.append(tree)
    return bhv.to_newick(bhv.frechet_mean(trees, n_epochs=FRECHET_EPOCHS, seed=0, tol=1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True, help="documented tree (Newick)")
    parser.add_argument("--phylolm", type=Path, help="PhyloLM output directory to score as well")
    args = parser.parse_args()
    meta_path = args.cube_dir / "distance_metadata.json"
    labels = json.loads(meta_path.read_text())["model_names"]
    cube = np.abs(np.load(args.cube_dir / "full_distance_cosine.npy"))
    frechet = args.cube_dir / "frechet_cosine.newick"
    frechet.write_text(frechet_tree(cube, labels) + "\n")
    average = args.cube_dir / "avg_reconstructed_cosine.newick"
    average.write_text(
        neighbor_joining_newick(labels, np.load(args.cube_dir / "avg_distance_cosine.npy")) + "\n"
    )
    avg_matrix = args.cube_dir / "avg_distance_cosine.npy"
    methods = {
        "weights_bhv": MethodInputs(frechet, avg_matrix, meta_path),
        "weights_nj_mean": MethodInputs(average, avg_matrix, meta_path),
    }
    if args.phylolm:
        methods["phylolm"] = MethodInputs(
            args.phylolm / "phylolm_nj.newick",
            args.phylolm / "phylolm_dist.npy",
            args.phylolm / "distance_metadata.json",
        )
    scores = score_family(args.truth, methods)
    scores["per_layer_clade_support"] = per_layer_clade_support(cube, labels, args.truth)
    scores["frechet_epochs"] = FRECHET_EPOCHS
    (args.cube_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    for name, method in scores["methods"].items():
        print(f"{name}: {method['clades_estimated']}/{method['n_documented_clades']} clades, "
              f"A={method['tree_likeness_all_quartets']:.3f}")


if __name__ == "__main__":
    main()
