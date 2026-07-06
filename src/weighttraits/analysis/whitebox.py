"""Whitebox analysis workflows from training ledgers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weighttraits.distances.manifest import (
    distance_input_rows_from_training_ledger,
    readers_from_distance_manifest,
    write_distance_input_manifest,
)
from weighttraits.distances.streaming import build_distance_cube, write_distance_cube
from weighttraits.phylo.reconstruct import reconstruct_tree_from_cube
from weighttraits.phylo.recovery import aggregate_recovery, score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text


def analyze_training_ledger(
    ledger: str | Path,
    *,
    truth_manifest: str | Path,
    out_dir: str | Path,
    artifact: str,
    metrics: list[str],
    representation: str | None = None,
    node_ids: list[str] | None = None,
    path_base: str | Path = ".",
    chunk_size: int = 1_000_000,
    eps: float = 1e-3,
    layer: str | int | None = None,
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Run the ledger -> cube -> reconstruction -> recovery path for one artifact mode."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    representation = representation or _default_representation(artifact)
    input_manifest = out / f"{artifact}_distance_inputs.yaml"
    cube_dir = out / "distance_cube"

    rows = distance_input_rows_from_training_ledger(
        ledger,
        truth_manifest=truth_manifest,
        artifact=artifact,
        node_ids=node_ids,
    )
    write_distance_input_manifest(rows, input_manifest, path_base=path_base)

    cube = build_distance_cube(
        readers_from_distance_manifest(input_manifest),
        metrics=metrics,
        chunk_size=chunk_size,
        eps=eps,
        representation=representation,
    )
    write_distance_cube(cube, cube_dir)

    truth_splits, truth_leaves = splits_from_manifest_path(str(truth_manifest))
    score_records = []
    result_rows = []
    for metric in sorted(cube.distances):
        reconstruction = reconstruct_tree_from_cube(
            cube_dir,
            metric=metric,
            layer=layer,
            aggregate=aggregate,
        )
        tree_path = out / f"tree_{metric}.newick"
        audit_path = out / f"tree_{metric}.audit.json"
        score_path = out / f"score_{metric}.json"

        tree_path.write_text(reconstruction.newick + "\n")
        audit_path.write_text(json.dumps(reconstruction.audit, indent=2, sort_keys=True) + "\n")

        estimate_splits, estimate_leaves = splits_from_newick_text(reconstruction.newick)
        score = score_split_recovery(
            truth_splits=truth_splits,
            estimate_splits=estimate_splits,
            truth_leaves=truth_leaves,
            estimate_leaves=estimate_leaves,
        )
        score["truth_manifest"] = str(truth_manifest)
        score["estimate"] = str(tree_path)
        score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")

        score_records.append(score)
        result_rows.append(
            {
                "metric": metric,
                "tree": str(tree_path),
                "tree_audit": str(audit_path),
                "score": str(score_path),
                "rf": score["rf"],
                "normalized_rf": score["normalized_rf"],
                "exact_tree_recovery": score["exact_tree_recovery"],
                "clade_recovery": score["clade_recovery"],
                "split_precision": score["split_precision"],
                "distance_min": reconstruction.audit["distance_min"],
                "distance_max": reconstruction.audit["distance_max"],
                "distance_mean": reconstruction.audit["distance_mean"],
            }
        )

    recovery_aggregate = aggregate_recovery(score_records)
    aggregate_path = out / "aggregate_recovery.json"
    aggregate_path.write_text(json.dumps(recovery_aggregate, indent=2, sort_keys=True) + "\n")

    summary = {
        "artifact": artifact,
        "representation": representation,
        "ledger": str(ledger),
        "truth_manifest": str(truth_manifest),
        "distance_inputs": str(input_manifest),
        "cube": str(cube_dir),
        "n_models": len(cube.model_ids),
        "n_layers": len(cube.layer_names),
        "model_ids": cube.model_ids,
        "metrics": sorted(cube.distances),
        "results": result_rows,
        "aggregate_recovery": recovery_aggregate,
        "aggregate_recovery_path": str(aggregate_path),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _default_representation(artifact: str) -> str:
    if artifact == "adapter_chain":
        return "lora_cumulative_delta"
    if artifact in {"model", "merged"}:
        return "full_weight"
    raise ValueError(f"unsupported ledger artifact mode: {artifact}")
