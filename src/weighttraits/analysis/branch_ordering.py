"""Paper-matched same-branch versus cross-branch distance summaries."""

from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np


BRANCH_ORDERING_FIELDS = (
    "branch_ordering_valid",
    "branch_ordering_status",
    "n_branch_pairs",
    "n_same_branch_pairs",
    "n_cross_branch_pairs",
    "branch_rank_biserial",
    "branch_within_run_r",
)


def branch_ordering_stats(
    manifest_path: str | Path,
    *,
    labels: Sequence[str],
    distances: np.ndarray,
    min_leaves: int = 4,
) -> dict[str, Any]:
    """Summarize root-branch discrimination for one tree and distance matrix.

    The definition matches the active paper and the legacy ELLMTrees analysis:
    leaves are on the same branch when their first ancestor below the root is
    identical. Rank-biserial is oriented so positive values mean cross-branch
    distances are larger. The point-biserial correlation uses ``same_branch``
    as the binary variable, so stronger lineage signal is negative.
    """

    model_ids = [str(label) for label in labels]
    matrix = np.asarray(distances, dtype=np.float64)
    _validate_distance_matrix(matrix, model_ids)
    if min_leaves < 2:
        raise ValueError("min_leaves must be at least 2")

    rows = _load_manifest_rows(manifest_path)
    leaf_info = _manifest_leaf_info(rows)
    foreign = sorted(set(model_ids) - set(leaf_info))
    if foreign:
        return {
            **_empty_stats(
                "nonleaf_or_unknown_labels",
                n_branch_pairs=len(model_ids) * (len(model_ids) - 1) // 2,
            ),
            "branch_ordering_foreign_labels": foreign,
        }

    if len(model_ids) < min_leaves:
        return _empty_stats(
            "too_few_leaves",
            n_branch_pairs=len(model_ids) * (len(model_ids) - 1) // 2,
        )

    same: list[float] = []
    cross: list[float] = []
    binary_labels: list[float] = []
    pair_distances: list[float] = []
    for left_index, right_index in combinations(range(len(model_ids)), 2):
        left = model_ids[left_index]
        right = model_ids[right_index]
        distance = float(matrix[left_index, right_index])
        same_branch = leaf_info[left]["branch"] == leaf_info[right]["branch"]
        binary_labels.append(1.0 if same_branch else 0.0)
        pair_distances.append(distance)
        (same if same_branch else cross).append(distance)

    if not same or not cross:
        return _empty_stats(
            "missing_branch_class",
            n_branch_pairs=len(pair_distances),
            n_same_branch_pairs=len(same),
            n_cross_branch_pairs=len(cross),
        )

    within_run_r = _pearson(binary_labels, pair_distances)
    if within_run_r is None or not math.isfinite(within_run_r):
        return _empty_stats(
            "undefined_within_run_r",
            n_branch_pairs=len(pair_distances),
            n_same_branch_pairs=len(same),
            n_cross_branch_pairs=len(cross),
            branch_rank_biserial=_rank_biserial_cross_greater(cross, same),
        )

    return {
        "branch_ordering_valid": True,
        "branch_ordering_status": "ok",
        "n_branch_pairs": len(pair_distances),
        "n_same_branch_pairs": len(same),
        "n_cross_branch_pairs": len(cross),
        "branch_rank_biserial": _rank_biserial_cross_greater(cross, same),
        "branch_within_run_r": max(-0.999, min(0.999, within_run_r)),
    }


def aggregate_branch_ordering(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-tree ordering values using the paper's run-level estimands."""

    status_counts: dict[str, int] = {}
    valid_rows = []
    pair_rows = []
    for row in rows:
        status = str(row.get("branch_ordering_status") or "missing")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"ok", "missing_branch_class", "undefined_within_run_r"}:
            pair_rows.append(row)
        if row.get("branch_ordering_valid") is True:
            valid_rows.append(row)

    rank_biserials = _finite_values(valid_rows, "branch_rank_biserial")
    within_run_r = [
        max(-0.999, min(0.999, value))
        for value in _finite_values(valid_rows, "branch_within_run_r")
    ]
    return {
        "n_ordering_trees": len(rank_biserials),
        "n_branch_pairs": sum(int(row.get("n_branch_pairs") or 0) for row in pair_rows),
        "n_same_branch_pairs": sum(
            int(row.get("n_same_branch_pairs") or 0) for row in pair_rows
        ),
        "n_cross_branch_pairs": sum(
            int(row.get("n_cross_branch_pairs") or 0) for row in pair_rows
        ),
        "branch_ordering_status_counts": {
            key: status_counts[key] for key in sorted(status_counts)
        },
        "branch_rank_biserial_mean": _mean_or_none(rank_biserials),
        "branch_rank_biserial_se": _standard_error(rank_biserials),
        "branch_within_run_r_fisher_z_mean": _fisher_z_mean(within_run_r),
        "branch_within_run_r_se": _standard_error(within_run_r),
    }


def _load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    rows = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{manifest}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _manifest_leaf_info(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    train_rows = [row for row in rows if row.get("grow", "train") == "train"]
    internal_ids: set[str] = set()
    normalized_paths: dict[str, list[str]] = {}
    for row in train_rows:
        node_id = str(row["node_id"])
        path = [str(part) for part in row.get("path", [node_id])]
        if not path or path[-1] != node_id:
            path.append(node_id)
        normalized_paths[node_id] = path
        internal_ids.update(path[:-1])

    leaves = {}
    for row in train_rows:
        node_id = str(row["node_id"])
        is_leaf = bool(row["is_leaf"]) if "is_leaf" in row else node_id not in internal_ids
        path = normalized_paths[node_id]
        if not is_leaf or len(path) < 2:
            continue
        leaves[node_id] = {
            "branch": path[1],
            "depth": len(path) - 1,
            "path": path,
        }
    return leaves


def _validate_distance_matrix(matrix: np.ndarray, labels: Sequence[str]) -> None:
    expected = (len(labels), len(labels))
    if matrix.shape != expected:
        raise ValueError(f"distance matrix shape {matrix.shape} does not match labels {expected}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance matrix contains non-finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError("distance matrix must be symmetric")


def _empty_stats(
    status: str,
    *,
    n_branch_pairs: int,
    n_same_branch_pairs: int = 0,
    n_cross_branch_pairs: int = 0,
    branch_rank_biserial: float | None = None,
) -> dict[str, Any]:
    return {
        "branch_ordering_valid": False,
        "branch_ordering_status": status,
        "n_branch_pairs": n_branch_pairs,
        "n_same_branch_pairs": n_same_branch_pairs,
        "n_cross_branch_pairs": n_cross_branch_pairs,
        "branch_rank_biserial": branch_rank_biserial,
        "branch_within_run_r": None,
    }


def _rank_biserial_cross_greater(cross: Sequence[float], same: Sequence[float]) -> float:
    concordant = 0.0
    for cross_distance in cross:
        for same_distance in same:
            if cross_distance > same_distance:
                concordant += 1.0
            elif cross_distance == same_distance:
                concordant += 0.5
    return (2.0 * concordant) / (len(cross) * len(same)) - 1.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / math.sqrt(left_ss * right_ss)


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return values


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _standard_error(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _fisher_z_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return math.tanh(statistics.fmean(math.atanh(value) for value in values))
