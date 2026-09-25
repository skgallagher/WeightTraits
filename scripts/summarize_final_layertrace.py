#!/usr/bin/env python3
"""Recompute the final fixed-2000 Flan LayerTrace table and profile.

The input is the receipt-validated full-fine-tuning distance-cube suite used by
the final Table 2.  The output JSON records the six documented subset scores;
the CSV is compatible with the paper's ggplot LayerTrace figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

from weighttraits.paper.analysis_contracts import TOPOLOGY_TREE_IDS
from weighttraits.paper.layer_subset_table import (
    SUBSET_DEFINITIONS,
    resolve_layer_subsets,
)
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import score_split_recovery
from weighttraits.phylo.splits import splits_from_newick_text


def _family(name: str) -> str:
    if "relative_attention_bias" in name:
        return "relbias"
    if "SelfAttention" in name:
        return "SA." + name.split("SelfAttention.", 1)[1].split(".", 1)[0]
    if "EncDecAttention" in name:
        return "CA." + name.split("EncDecAttention.", 1)[1].split(".", 1)[0]
    if "DenseReluDense" in name:
        return "FFN"
    if "layer_norm" in name:
        return "LN"
    return "other"


def _block(name: str) -> str:
    if name.startswith("encoder"):
        return "enc"
    if name.startswith("decoder"):
        return "dec"
    return "other"


def _highlight(family: str) -> str:
    return {
        "SA.k": "Self-attention key",
        "CA.k": "Cross-attention key",
        "LN": "LayerNorm",
    }.get(family, "Other layer")


def _se(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values))


def _score(matrix: np.ndarray, labels: list[str], truth_newick: str) -> dict[str, object]:
    truth_splits, truth_leaves = splits_from_newick_text(truth_newick)
    estimated_newick = neighbor_joining_newick(labels, matrix)
    estimated_splits, estimated_leaves = splits_from_newick_text(estimated_newick)
    return score_split_recovery(
        truth_splits, estimated_splits, truth_leaves, estimated_leaves
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    layer_names: list[str] | None = None
    subset_indices: dict[str, tuple[int, ...]] | None = None
    subset_observations: dict[str, list[dict[str, object]]] = {
        subset_id: [] for subset_id, _description, _predicate in SUBSET_DEFINITIONS
    }
    layer_fn: list[list[float]] = []

    for tree_id in TOPOLOGY_TREE_IDS:
        analysis = args.analysis_root / tree_id / "model_leaf_analysis"
        names = [str(value) for value in json.loads((analysis / "layers.json").read_text())]
        labels = [str(value) for value in json.loads((analysis / "models.json").read_text())]
        truth_newick = (analysis / "truth_manifest.newick").read_text().strip()
        with np.load(analysis / "direct_distance_layers.npz", allow_pickle=False) as archive:
            cube = np.asarray(archive["cosine"], dtype=np.float64)

        if layer_names is None:
            layer_names = names
            subset_indices = resolve_layer_subsets(names)
            layer_fn = [[] for _ in names]
        elif names != layer_names:
            raise ValueError(f"layer inventory differs for {tree_id}")
        assert subset_indices is not None

        for subset_id, _description, _predicate in SUBSET_DEFINITIONS:
            matrix = np.mean(cube[np.asarray(subset_indices[subset_id])], axis=0)
            score = _score(matrix, labels, truth_newick)
            subset_observations[subset_id].append(score)

        for index in range(len(names)):
            score = _score(cube[index], labels, truth_newick)
            layer_fn[index].append(100.0 * float(score["false_negative"]) / float(score["n_truth_splits"]))

    assert layer_names is not None
    descriptions = {key: description for key, description, _ in SUBSET_DEFINITIONS}
    subset_rows: list[dict[str, object]] = []
    for subset_id, values in subset_observations.items():
        recovery = [float(value["clade_recovery"]) for value in values]
        paer = [1.0 if value["polytomy_aware_exact_recovery"] else 0.0 for value in values]
        false_negative = [float(value["false_negative"]) for value in values]
        p = statistics.fmean(paer)
        subset_rows.append({
            "subset_id": subset_id,
            "description": descriptions[subset_id],
            "n_tensors": len(subset_indices[subset_id]),
            "n_trees": len(values),
            "clade_recovery": statistics.fmean(recovery),
            "clade_recovery_se": _se(recovery),
            "paer": p,
            "paer_se": math.sqrt(p * (1.0 - p) / len(paer)),
            "false_negative": statistics.fmean(false_negative),
            "false_negative_se": _se(false_negative),
        })

    profile_rows: list[dict[str, object]] = []
    for position, (name, values) in enumerate(zip(layer_names, layer_fn, strict=True)):
        family = _family(name)
        profile_rows.append({
            "group": "fixed2000_flan_full_ft",
            "model": "Flan-T5-base",
            "architecture": "encoder-decoder, d = 768",
            "position": position,
            "layer_name": name,
            "family": family,
            "block": _block(name),
            "highlight": _highlight(family),
            "mean_fn": statistics.fmean(values),
            "se_fn": _se(values),
            "n_runs": len(values),
        })

    key_values = [float(row["mean_fn"]) for row in profile_rows if row["family"] in {"SA.k", "CA.k"}]
    other_values = [float(row["mean_fn"]) for row in profile_rows if row["family"] not in {"SA.k", "CA.k"}]
    layernorm_values = [float(row["mean_fn"]) for row in profile_rows if row["family"] == "LN"]
    payload = {
        "schema": "weighttraits.final_fixed2000_layertrace.v1",
        "valid": True,
        "analysis_root": str(args.analysis_root.resolve()),
        "tree_ids": list(TOPOLOGY_TREE_IDS),
        "n_trees": len(TOPOLOGY_TREE_IDS),
        "n_layers": len(layer_names),
        "subset_rows": subset_rows,
        "individual_layer_summary": {
            "mean_fn_all_layers": statistics.fmean(float(row["mean_fn"]) for row in profile_rows),
            "mean_fn_key_matrices": statistics.fmean(key_values),
            "mean_fn_other_layers": statistics.fmean(other_values),
            "mean_fn_layernorm": statistics.fmean(layernorm_values),
            "min_mean_fn": min(float(row["mean_fn"]) for row in profile_rows),
            "max_mean_fn": max(float(row["mean_fn"]) for row in profile_rows),
            "n_perfect_mean_layers": sum(abs(float(row["mean_fn"])) < 1e-12 for row in profile_rows),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    print(json.dumps(payload["individual_layer_summary"], indent=2, sort_keys=True))
    for row in subset_rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
