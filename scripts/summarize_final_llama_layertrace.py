#!/usr/bin/env python3
"""Build the matched fixed-2000 Llama per-tensor recovery profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from weighttraits.paper.analysis_contracts import TOPOLOGY_TREE_IDS
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import score_split_recovery
from weighttraits.phylo.splits import splits_from_newick_text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(name: str) -> str:
    if ".self_attn." in name:
        projection = name.split(".self_attn.", 1)[1].split("_proj.", 1)[0]
        return f"SA.{projection}"
    if ".mlp." in name:
        return "FFN"
    if "layernorm" in name or name.endswith(".norm.weight"):
        return "LN"
    return "other"


def block(name: str) -> str:
    return "dec" if name.startswith("model.layers.") else "other"


def highlight(value: str) -> str:
    return {
        "SA.k": "Self-attention key",
        "LN": "LayerNorm",
    }.get(value, "Other layer")


def sample_se(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values))


def analysis_dir(root: Path, tree_id: str) -> Path:
    current = root / tree_id / "model_leaf_analysis"
    if not current.is_dir():
        raise FileNotFoundError(current)
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--zero-tolerance", type=float, default=1e-12)
    parser.add_argument("--expected-active", type=int, default=122)
    args = parser.parse_args()

    canonical_names: list[str] | None = None
    canonical_active: list[int] | None = None
    tensor_fn: list[list[float]] = []
    input_files: list[dict[str, str]] = []

    for tree_id in TOPOLOGY_TREE_IDS:
        current = analysis_dir(args.analysis_root.expanduser().resolve(), tree_id)
        layers_path = current / "layers.json"
        models_path = current / "models.json"
        cube_path = current / "direct_distance_layers.npz"
        truth_path = current / "truth_manifest.newick"
        for path in (layers_path, models_path, cube_path, truth_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            input_files.append({"path": str(path), "sha256": sha256(path)})

        names = [str(value) for value in json.loads(layers_path.read_text())]
        labels = [str(value) for value in json.loads(models_path.read_text())]
        truth_splits, truth_leaves = splits_from_newick_text(truth_path.read_text())
        with np.load(cube_path, allow_pickle=False) as archive:
            cube = np.asarray(archive["cosine"], dtype=np.float64)
        if cube.shape != (len(names), len(labels), len(labels)):
            raise ValueError(f"unexpected cube shape for {tree_id}: {cube.shape}")
        maxima = np.max(np.abs(cube), axis=(1, 2))
        active = [int(index) for index in np.flatnonzero(maxima > args.zero_tolerance)]

        if canonical_names is None:
            canonical_names = names
            canonical_active = active
            tensor_fn = [[] for _ in names]
            if len(active) != args.expected_active:
                raise ValueError(
                    f"expected {args.expected_active} active tensors, found {len(active)}"
                )
        elif names != canonical_names or active != canonical_active:
            raise ValueError(f"tensor inventory differs for {tree_id}")

        assert canonical_active is not None
        active_set = set(canonical_active)
        for tensor_index in range(len(names)):
            if tensor_index not in active_set:
                # An all-zero cosine matrix contains no split information. Treat it
                # as recovering no true clades instead of accepting NJ tie-breaking.
                tensor_fn[tensor_index].append(100.0)
                continue
            estimated = neighbor_joining_newick(labels, cube[tensor_index])
            estimated_splits, estimated_leaves = splits_from_newick_text(estimated)
            score = score_split_recovery(
                truth_splits, estimated_splits, truth_leaves, estimated_leaves
            )
            tensor_fn[tensor_index].append(
                100.0 * float(score["false_negative"]) / len(truth_splits)
            )

    assert canonical_names is not None and canonical_active is not None
    rows: list[dict[str, object]] = []
    for position, (name, values) in enumerate(zip(canonical_names, tensor_fn, strict=True)):
        tensor_family = family(name)
        rows.append(
            {
                "group": "fixed2000_llama_full_ft",
                "model": "Llama-3.2-1B",
                "architecture": "decoder-only, d = 2048",
                "position": position,
                "layer_name": name,
                "family": tensor_family,
                "block": block(name),
                "highlight": highlight(tensor_family),
                "mean_fn": statistics.fmean(values),
                "se_fn": sample_se(values),
                "n_runs": len(values),
            }
        )

    key_values = [float(row["mean_fn"]) for row in rows if row["family"] == "SA.k"]
    other_values = [float(row["mean_fn"]) for row in rows if row["family"] != "SA.k"]
    active_names = {canonical_names[index] for index in canonical_active}
    active_values = [
        float(row["mean_fn"]) for row in rows if row["layer_name"] in active_names
    ]
    active_other_values = [
        float(row["mean_fn"])
        for row in rows
        if row["layer_name"] in active_names and row["family"] != "SA.k"
    ]
    payload = {
        "schema": "weighttraits.final_fixed2000_llama_layertrace.v1",
        "valid": True,
        "analysis_root": str(args.analysis_root.expanduser().resolve()),
        "tree_ids": list(TOPOLOGY_TREE_IDS),
        "n_trees": len(TOPOLOGY_TREE_IDS),
        "n_tensors_total": len(canonical_names),
        "n_tensors_active": len(canonical_active),
        "zero_tolerance": args.zero_tolerance,
        "active_tensor_names": [canonical_names[index] for index in canonical_active],
        "individual_tensor_summary": {
            "mean_fn_all_tensors": statistics.fmean(float(row["mean_fn"]) for row in rows),
            "mean_fn_active_tensors": statistics.fmean(active_values),
            "mean_fn_key_matrices": statistics.fmean(key_values),
            "mean_fn_other_tensors": statistics.fmean(other_values),
            "mean_fn_active_non_key_tensors": statistics.fmean(active_other_values),
            "mean_fn_layernorm": statistics.fmean(
                float(row["mean_fn"]) for row in rows if row["family"] == "LN"
            ),
            "min_mean_fn": min(float(row["mean_fn"]) for row in rows),
            "max_mean_fn": max(float(row["mean_fn"]) for row in rows),
        },
        "input_files": input_files,
    }

    json_out = args.json_out.expanduser().resolve()
    csv_out = args.csv_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["individual_tensor_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
