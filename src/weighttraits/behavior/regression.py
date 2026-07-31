"""Auditable pair tables joining whitebox and behavioral distances."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from weighttraits.phylo.reconstruct import load_distance_cube, matrix_from_distance_cube


def paired_distance_rows(
    *,
    run_id: str,
    weight_cube: str | Path,
    behavior_cube: str | Path,
    weight_metric: str,
    behavior_metric: str = "semantic_paired",
    weight_layer: str | int | None = None,
    behavior_layer: str | int | None = None,
    weight_aggregate: str = "mean",
    behavior_aggregate: str = "mean",
    allow_model_subset: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weight = load_distance_cube(weight_cube)
    behavior = load_distance_cube(behavior_cube)
    weight_matrix, weight_selection = matrix_from_distance_cube(
        weight, metric=weight_metric, layer=weight_layer, aggregate=weight_aggregate
    )
    behavior_matrix, behavior_selection = matrix_from_distance_cube(
        behavior, metric=behavior_metric, layer=behavior_layer, aggregate=behavior_aggregate
    )
    weight_models = set(weight.model_ids)
    behavior_models = set(behavior.model_ids)
    if weight_models != behavior_models and not allow_model_subset:
        raise ValueError(
            "weight/behavior cubes must contain identical model IDs; "
            f"weight-only={sorted(weight_models - behavior_models)}, "
            f"behavior-only={sorted(behavior_models - weight_models)}"
        )
    common = sorted(weight_models & behavior_models)
    if len(common) < 2:
        raise ValueError("weight/behavior cubes must share at least two model IDs")
    weight_index = {model_id: index for index, model_id in enumerate(weight.model_ids)}
    behavior_index = {model_id: index for index, model_id in enumerate(behavior.model_ids)}
    observation_counts_path = Path(behavior_cube) / "observation_counts.npy"
    if not observation_counts_path.is_file():
        raise ValueError(
            f"behavior cube is missing pairwise observation counts: {observation_counts_path}"
        )
    observation_counts = np.load(observation_counts_path, allow_pickle=False)
    expected_shape = (len(behavior.model_ids), len(behavior.model_ids))
    if observation_counts.shape != expected_shape:
        raise ValueError(
            "behavior observation-count matrix shape does not match its model axis: "
            f"{observation_counts.shape} != {expected_shape}"
        )
    rows = []
    for left_index, left in enumerate(common):
        for right in common[left_index + 1 :]:
            rows.append(
                {
                    "run_id": run_id,
                    "pair_id": f"{left}::{right}",
                    "model_a": left,
                    "model_b": right,
                    "weight_distance": float(
                        weight_matrix[weight_index[left], weight_index[right]]
                    ),
                    "behavior_distance": float(
                        behavior_matrix[behavior_index[left], behavior_index[right]]
                    ),
                    "behavior_similarity": float(
                        1.0 - behavior_matrix[behavior_index[left], behavior_index[right]]
                    ),
                    "behavior_observations": int(
                        observation_counts[behavior_index[left], behavior_index[right]]
                    ),
                }
            )
    audit = {
        "schema_version": 1,
        "run_id": run_id,
        "weight_cube": str(weight_cube),
        "behavior_cube": str(behavior_cube),
        "weight_metric": weight_metric,
        "behavior_metric": behavior_metric,
        "weight_selection": weight_selection,
        "behavior_selection": behavior_selection,
        "n_weight_models": len(weight.model_ids),
        "n_behavior_models": len(behavior.model_ids),
        "n_common_models": len(common),
        "common_model_ids": common,
        "weight_only_model_ids": sorted(set(weight.model_ids) - set(common)),
        "behavior_only_model_ids": sorted(set(behavior.model_ids) - set(common)),
        "allow_model_subset": allow_model_subset,
        "behavior_observation_counts": str(observation_counts_path),
        "n_pairs": len(rows),
    }
    return rows, audit


def write_paired_distance_rows(
    rows: Sequence[dict[str, Any]], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".json":
        output.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n")
        return
    if suffix != ".csv":
        raise ValueError("paired distance table output must end in .csv or .json")
    columns = [
        "run_id",
        "pair_id",
        "model_a",
        "model_b",
        "weight_distance",
        "behavior_distance",
        "behavior_similarity",
        "behavior_observations",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
