"""Whitebox analysis workflows from training ledgers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weighttraits.analysis.completion import audit_training_run_set_completion
from weighttraits.analysis.direct import (
    analyze_training_ledger_direct,
    truth_newick_from_manifest,
)
from weighttraits.analysis.tree_diagnostics import TREE_DIAGNOSTICS, write_tree_diagnostics
from weighttraits.distances.manifest import (
    distance_input_rows_from_training_ledger,
    readers_from_distance_manifest,
    write_distance_input_manifest,
)
from weighttraits.distances.streaming import build_distance_cube, write_distance_cube
from weighttraits.phylo.reconstruct import matrix_from_distance_cube, reconstruct_tree_from_cube
from weighttraits.phylo.recovery import aggregate_recovery, score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text


def analyze_training_run_set(
    summary: str | Path,
    *,
    artifact: str,
    metrics: list[str],
    out_dir: str | Path,
    path_base: str | Path = ".",
    optional_artifacts: set[str] | None = None,
    tree_ids: list[str] | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    representation: str | None = None,
    node_ids: list[str] | None = None,
    chunk_size: int = 1_000_000,
    eps: float = 1e-3,
    layer: str | int | None = None,
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Audit a training run set and analyze each ready tree."""
    direct_representation = _default_representation(artifact)
    if representation is not None and representation != direct_representation:
        raise ValueError(
            "analyze-training-run-set uses the direct engine; representation is derived "
            f"from artifact={artifact!r} as {direct_representation!r}"
        )
    base = Path(path_base)
    out_root = Path(out_dir)
    requested_tree_ids = set(tree_ids or [])
    completion = audit_training_run_set_completion(
        summary,
        path_base=path_base,
        optional_artifacts=optional_artifacts,
        only_ready=False,
    )
    known_tree_ids = {row.tree_id for row in completion.rows}
    ready_rows = [row for row in completion.rows if row.ready_for_analysis]
    selected_rows = [
        row for row in ready_rows if not requested_tree_ids or row.tree_id in requested_tree_ids
    ]
    requested_unavailable = sorted(requested_tree_ids - {row.tree_id for row in selected_rows})
    requested_missing = sorted(requested_tree_ids - known_tree_ids)
    requested_not_ready = sorted(set(requested_unavailable) - set(requested_missing))

    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    analyses = []
    n_analyzed = 0
    n_skipped = 0
    n_existing = 0
    for row in selected_rows:
        ledger = _resolve_path(row.ledger, base)
        truth_manifest = _resolve_optional_path(row.manifest, base)
        analysis_dir = out_root / row.tree_id / _run_set_analysis_dir_name(artifact)
        summary_path = analysis_dir / "summary.json"
        planned = {
            "tree_id": row.tree_id,
            "analysis_engine": "direct",
            "artifact": artifact,
            "metrics": list(metrics),
            "ledger": str(ledger),
            "truth_manifest": None if truth_manifest is None else str(truth_manifest),
            "out": str(analysis_dir),
        }
        existing = (
            _compatible_existing_analysis(summary_path, artifact=artifact, metrics=metrics)
            if skip_existing
            else None
        )
        if existing is not None:
            n_existing += 1
            analyses.append(
                {
                    **planned,
                    "status": "already_exists" if dry_run else "skipped_existing",
                    "summary": str(summary_path),
                    "n_models": existing.get("n_models"),
                    "n_layers": existing.get("n_layers"),
                    "aggregate_recovery": existing.get("aggregate_recovery"),
                }
            )
            continue
        if truth_manifest is None:
            n_skipped += 1
            analyses.append(
                {
                    **planned,
                    "status": "skipped",
                    "reason": "missing_truth_manifest",
                }
            )
            continue
        if dry_run:
            analyses.append({**planned, "status": "planned"})
            continue
        result = analyze_training_ledger_direct(
            ledger,
            truth_manifest=truth_manifest,
            out_dir=analysis_dir,
            artifact=artifact,
            metrics=metrics,
            node_ids=node_ids,
            path_base=path_base,
            chunk_size=chunk_size,
            eps=eps,
            layer=layer,
            aggregate=aggregate,
        )
        n_analyzed += 1
        analyses.append(
            {
                **planned,
                "status": "completed",
                "summary": str(summary_path),
                "n_models": result["n_models"],
                "n_layers": result["n_layers"],
                "aggregate_recovery": result["aggregate_recovery"],
            }
        )

    return {
        "valid": not requested_missing and not requested_not_ready and n_skipped == 0,
        "status": _run_set_analysis_status(
            dry_run=dry_run,
            n_selected=len(selected_rows),
            n_existing=n_existing,
            n_skipped=n_skipped,
            requested_missing=requested_missing,
            requested_not_ready=requested_not_ready,
        ),
        "dry_run": dry_run,
        "analysis_engine": "direct",
        "distance_engine": "direct_streaming_sufficient_stats",
        "tree_builder": "biopython_neighbor_joining",
        "rf_engine": "dendropy_treecompare",
        "summary": str(summary),
        "path_base": str(path_base),
        "artifact": artifact,
        "representation": direct_representation,
        "metrics": list(metrics),
        "out_dir": str(out_root),
        "skip_existing": skip_existing,
        "completion": {
            "valid": completion.valid,
            "n_trees": completion.n_trees,
            "n_ready": completion.n_ready,
            "n_failed": completion.n_failed,
            "n_in_progress": completion.n_in_progress,
            "n_not_started": completion.n_not_started,
            "n_total_runs": completion.n_total_runs,
            "n_terminal_nodes": completion.n_terminal_nodes,
            "n_ok_nodes": completion.n_ok_nodes,
            "n_failed_nodes": completion.n_failed_nodes,
            "n_missing_nodes": completion.n_missing_nodes,
            "ready_tree_ids": list(completion.ready_tree_ids),
        },
        "selected_tree_ids": [row.tree_id for row in selected_rows],
        "requested_missing_tree_ids": requested_missing,
        "requested_not_ready_tree_ids": requested_not_ready,
        "n_selected": len(selected_rows),
        "n_analyzed": n_analyzed,
        "n_existing": n_existing,
        "n_skipped": n_skipped,
        "analyses": analyses,
    }


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
    truth_newick = truth_newick_from_manifest(truth_manifest)
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
        matrix, _ = matrix_from_distance_cube(
            cube,
            metric=metric,
            layer=layer,
            aggregate=aggregate,
        )
        diagnostics = write_tree_diagnostics(
            out,
            metric=metric,
            labels=cube.model_ids,
            distances=matrix,
            truth_newick=truth_newick,
            truth_splits=truth_splits,
        )

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
                "polytomy_aware_exact_recovery": score[
                    "polytomy_aware_exact_recovery"
                ],
                "clade_recovery": score["clade_recovery"],
                "split_precision": score["split_precision"],
                "distance_min": reconstruction.audit["distance_min"],
                "distance_max": reconstruction.audit["distance_max"],
                "distance_mean": reconstruction.audit["distance_mean"],
                **diagnostics,
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
        "diagnostics": list(TREE_DIAGNOSTICS),
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


def _resolve_path(path: str | Path, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return base / value


def _resolve_optional_path(path: str | Path | None, base: Path) -> Path | None:
    if path is None:
        return None
    return _resolve_path(path, base)


def _run_set_analysis_dir_name(artifact: str) -> str:
    if artifact == "adapter_chain":
        return "cumulative_leaf_analysis"
    if artifact in {"model", "merged"}:
        return f"{artifact}_leaf_analysis"
    raise ValueError(f"unsupported ledger artifact mode: {artifact}")


def _run_set_analysis_status(
    *,
    dry_run: bool,
    n_selected: int,
    n_existing: int,
    n_skipped: int,
    requested_missing: list[str],
    requested_not_ready: list[str],
) -> str:
    if requested_missing:
        return "requested_trees_missing"
    if requested_not_ready:
        return "requested_trees_not_ready"
    if n_skipped:
        return "selected_trees_skipped"
    if n_selected == 0:
        return "no_ready_trees"
    if n_existing == n_selected:
        return "up_to_date"
    return "planned" if dry_run else "completed"


def _compatible_existing_analysis(
    summary_path: Path,
    *,
    artifact: str,
    metrics: list[str],
) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(summary, dict):
        return None
    if summary.get("analysis_engine") != "direct":
        return None
    if summary.get("artifact") != artifact:
        return None
    existing_metrics = summary.get("metrics")
    if not isinstance(existing_metrics, list):
        return None
    if not set(metrics).issubset({str(metric) for metric in existing_metrics}):
        return None
    existing_diagnostics = summary.get("diagnostics")
    if not isinstance(existing_diagnostics, list):
        return None
    if not set(TREE_DIAGNOSTICS).issubset(
        {str(diagnostic) for diagnostic in existing_diagnostics}
    ):
        return None
    return summary
