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
