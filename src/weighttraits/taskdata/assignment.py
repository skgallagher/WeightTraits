"""Task/dataset assignment decoupled from topology generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_manifest_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def assign_task_data(
    topology_rows: list[dict[str, Any]],
    task_families_config: dict[str, Any],
    *,
    seed: int,
    policy: str = "per_node",
    task_families: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Assign task family and dataset id to each train row in a topology manifest."""

    if policy not in {"per_node", "per_node_without_replacement", "per_edge", "per_depth"}:
        raise ValueError(f"unsupported assignment policy: {policy}")
    candidates = _candidate_families(task_families_config, task_families)
    rng = np.random.default_rng(seed)
    assigned: list[dict[str, Any]] = []
    depth_cache: dict[int, tuple[str, str]] = {}
    train_rows = [row for row in topology_rows if row.get("grow", "train") == "train"]
    no_replacement_pool = _candidate_pairs(candidates) if policy == "per_node_without_replacement" else []
    no_replacement_order = []
    if no_replacement_pool:
        if len(train_rows) > len(no_replacement_pool):
            raise ValueError(
                f"cannot assign {len(train_rows)} train rows without replacement "
                f"from {len(no_replacement_pool)} candidate datasets"
            )
        no_replacement_order = list(rng.permutation(len(no_replacement_pool)))

    for row in topology_rows:
        out = dict(row)
        if out.get("grow", "train") != "train":
            assigned.append(out)
            continue
        if policy == "per_node_without_replacement":
            family, dataset = no_replacement_pool[int(no_replacement_order.pop(0))]
        elif policy == "per_depth":
            depth = int(out.get("depth", 0))
            if depth not in depth_cache:
                depth_cache[depth] = _draw_family_dataset(candidates, rng)
            family, dataset = depth_cache[depth]
        else:
            family, dataset = _draw_family_dataset(candidates, rng)
        out.update(
            {
                "task_family": family,
                "dataset_id": dataset,
                "task_assignment_policy": policy,
                "task_assignment_seed": seed,
            }
        )
        assigned.append(out)
    return assigned


def audit_paired_assignment_set(
    tree_set_path: str | Path,
    reference_summary_path: str | Path,
    candidate_summary_path: str | Path,
    *,
    forbidden_task_families: list[str] | None = None,
    path_base: str | Path = ".",
) -> dict[str, Any]:
    """Audit a reassigned condition against the original, fixed topology set.

    Paths recorded in WeightTraits summaries are repository-relative, so ``path_base``
    is explicit rather than inferred from the summary location.
    """

    base = Path(path_base)
    tree_set = json.loads(Path(tree_set_path).read_text())
    reference_summary = json.loads(Path(reference_summary_path).read_text())
    candidate_summary = json.loads(Path(candidate_summary_path).read_text())
    forbidden = set(forbidden_task_families or [])

    topology_manifests = {
        str(tree["tree_id"]): _resolve_recorded_path(base, tree["manifest"])
        for tree in tree_set.get("trees", [])
    }
    reference_manifests = {
        str(item["tree_id"]): _resolve_recorded_path(base, item["assigned_manifest"])
        for item in reference_summary.get("assignments", [])
    }
    candidate_manifests = {
        str(item["tree_id"]): _resolve_recorded_path(base, item["assigned_manifest"])
        for item in candidate_summary.get("assignments", [])
    }

    expected_tree_ids = set(topology_manifests)
    issues: list[dict[str, Any]] = []
    for label, observed in (
        ("reference", set(reference_manifests)),
        ("candidate", set(candidate_manifests)),
    ):
        if observed != expected_tree_ids:
            issues.append(
                {
                    "issue": "tree_id_set_mismatch",
                    "condition": label,
                    "missing": sorted(expected_tree_ids - observed),
                    "unexpected": sorted(observed - expected_tree_ids),
                }
            )

    tree_reports: list[dict[str, Any]] = []
    for tree_id in sorted(expected_tree_ids & set(reference_manifests) & set(candidate_manifests)):
        topology_rows = load_manifest_rows(topology_manifests[tree_id])
        reference_rows = load_manifest_rows(reference_manifests[tree_id])
        candidate_rows = load_manifest_rows(candidate_manifests[tree_id])
        topology_by_node = _rows_by_node(topology_rows)
        reference_by_node = _rows_by_node(reference_rows)
        candidate_by_node = _rows_by_node(candidate_rows)
        node_ids = set(topology_by_node)

        if set(reference_by_node) != node_ids or set(candidate_by_node) != node_ids:
            issues.append(
                {
                    "issue": "node_id_set_mismatch",
                    "tree_id": tree_id,
                    "topology_nodes": sorted(node_ids),
                    "reference_nodes": sorted(reference_by_node),
                    "candidate_nodes": sorted(candidate_by_node),
                }
            )
            continue

        topology_mismatches = []
        changed_assignments = 0
        candidate_datasets = []
        candidate_families = []
        for node_id in sorted(node_ids):
            topology = _topology_projection(topology_by_node[node_id])
            reference = reference_by_node[node_id]
            candidate = candidate_by_node[node_id]
            if _topology_projection(reference) != topology or _topology_projection(candidate) != topology:
                topology_mismatches.append(node_id)
            if (
                reference.get("task_family"),
                reference.get("dataset_id"),
            ) != (
                candidate.get("task_family"),
                candidate.get("dataset_id"),
            ):
                changed_assignments += 1
            if candidate.get("grow", "train") == "train":
                candidate_datasets.append(str(candidate.get("dataset_id")))
                candidate_families.append(str(candidate.get("task_family")))

        duplicate_datasets = sorted(
            dataset for dataset in set(candidate_datasets) if candidate_datasets.count(dataset) > 1
        )
        forbidden_observed = sorted(forbidden & set(candidate_families))
        if topology_mismatches:
            issues.append(
                {
                    "issue": "topology_mismatch",
                    "tree_id": tree_id,
                    "node_ids": topology_mismatches,
                }
            )
        if duplicate_datasets:
            issues.append(
                {
                    "issue": "duplicate_dataset_within_tree",
                    "tree_id": tree_id,
                    "dataset_ids": duplicate_datasets,
                }
            )
        if forbidden_observed:
            issues.append(
                {
                    "issue": "forbidden_task_family",
                    "tree_id": tree_id,
                    "task_families": forbidden_observed,
                }
            )
        if changed_assignments == 0:
            issues.append({"issue": "assignments_not_regenerated", "tree_id": tree_id})

        tree_reports.append(
            {
                "tree_id": tree_id,
                "n_nodes": len(node_ids),
                "n_changed_assignments": changed_assignments,
                "n_unique_datasets": len(set(candidate_datasets)),
                "task_families": {
                    family: candidate_families.count(family)
                    for family in sorted(set(candidate_families))
                },
            }
        )

    return {
        "ok": not issues,
        "tree_set": str(tree_set_path),
        "reference_assignment_summary": str(reference_summary_path),
        "candidate_assignment_summary": str(candidate_summary_path),
        "forbidden_task_families": sorted(forbidden),
        "n_expected_trees": len(expected_tree_ids),
        "n_audited_trees": len(tree_reports),
        "n_changed_assignments": sum(item["n_changed_assignments"] for item in tree_reports),
        "issues": issues,
        "trees": tree_reports,
    }


def _resolve_recorded_path(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _rows_by_node(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_node = {str(row["node_id"]): row for row in rows}
    if len(by_node) != len(rows):
        raise ValueError("manifest contains duplicate node ids")
    return by_node


def _topology_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in ("node_id", "parent_id", "depth", "path", "grow")
    }


def _candidate_families(
    task_families_config: dict[str, Any],
    task_families: list[str] | None,
) -> dict[str, list[str]]:
    names = task_families or list(task_families_config)
    candidates: dict[str, list[str]] = {}
    for name in names:
        if name not in task_families_config:
            raise ValueError(f"unknown task family: {name}")
        datasets = [
            item["id"]
            for item in task_families_config[name].get("datasets", [])
            if item.get("role") != "evaluation_only"
        ]
        if len(datasets) < 1:
            raise ValueError(f"task family has no assignable datasets: {name}")
        candidates[name] = datasets
    if not candidates:
        raise ValueError("no task families selected")
    return candidates


def _candidate_pairs(candidates: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [
        (family, dataset)
        for family in sorted(candidates)
        for dataset in sorted(candidates[family])
    ]


def _draw_family_dataset(
    candidates: dict[str, list[str]],
    rng: np.random.Generator,
) -> tuple[str, str]:
    family_names = sorted(candidates)
    family = str(rng.choice(family_names))
    dataset = str(rng.choice(candidates[family]))
    return family, dataset
