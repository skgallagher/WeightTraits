"""Paired same-tree comparisons for independent paper analyses."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import fmean, median
from typing import Any

import yaml


WEIGHTTRAITS_PAIRED_COMPARISONS_SCHEMA = "weighttraits.paired_comparisons.v1"

PAIRED_COMPARISON_COLUMNS = [
    "comparison_id",
    "group",
    "label",
    "left_condition",
    "left_metric",
    "right_condition",
    "right_metric",
    "outcome",
    "outcome_label",
    "units",
    "n_pairs",
    "mean_effect",
    "median_effect",
    "ci_low",
    "ci_high",
    "wins",
    "ties",
    "losses",
    "win_rate",
    "sign_test_p",
]


def weighttraits_paired_comparison_rows(
    registry_path: str | Path,
    *,
    base_dir: str | Path = ".",
) -> list[dict[str, Any]]:
    """Compute deterministic paired bootstrap effects on shared topology IDs."""

    registry_file = Path(registry_path)
    registry = yaml.safe_load(registry_file.read_text())
    if not isinstance(registry, dict):
        raise ValueError(f"paired-comparison registry must be a mapping: {registry_file}")
    conditions = _load_conditions(registry, base_dir=Path(base_dir))
    outcomes = _load_outcomes(registry)
    comparisons = registry.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("paired-comparison registry requires non-empty comparisons")
    n_bootstrap = int(registry.get("n_bootstrap", 10_000))
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    seed = int(registry.get("seed", 20260713))

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, comparison in enumerate(comparisons, start=1):
        if not isinstance(comparison, dict):
            raise ValueError(f"comparison {index} must be a mapping")
        comparison_id = _required_text(comparison, "id", context=f"comparison {index}")
        if comparison_id in seen_ids:
            raise ValueError(f"duplicate paired comparison id {comparison_id!r}")
        seen_ids.add(comparison_id)
        left = _load_side(comparison, "left", conditions, comparison_id)
        right = _load_side(comparison, "right", conditions, comparison_id)
        left_rows = left["rows"]
        right_rows = right["rows"]
        left_tree_ids = set(left_rows)
        right_tree_ids = set(right_rows)
        if left_tree_ids != right_tree_ids:
            raise ValueError(
                f"paired comparison {comparison_id!r} has mismatched topology IDs: "
                f"left-only={sorted(left_tree_ids - right_tree_ids)}, "
                f"right-only={sorted(right_tree_ids - left_tree_ids)}"
            )
        tree_ids = sorted(left_tree_ids)
        for outcome in outcomes:
            effects = [
                outcome["direction"]
                * (
                    _required_number(left_rows[tree_id], outcome["field"], context=tree_id)
                    - _required_number(right_rows[tree_id], outcome["field"], context=tree_id)
                )
                * outcome["scale"]
                for tree_id in tree_ids
            ]
            ci_low, ci_high = _bootstrap_mean_ci(
                effects,
                n_bootstrap=n_bootstrap,
                seed=f"{seed}:{comparison_id}:{outcome['id']}",
            )
            wins = sum(value > 1e-12 for value in effects)
            losses = sum(value < -1e-12 for value in effects)
            ties = len(effects) - wins - losses
            non_ties = wins + losses
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "group": str(comparison.get("group", "")),
                    "label": _required_text(comparison, "label", context=comparison_id),
                    "left_condition": left["condition_id"],
                    "left_metric": left["metric"],
                    "right_condition": right["condition_id"],
                    "right_metric": right["metric"],
                    "outcome": outcome["id"],
                    "outcome_label": outcome["label"],
                    "units": outcome["units"],
                    "n_pairs": len(effects),
                    "mean_effect": fmean(effects),
                    "median_effect": median(effects),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": wins / non_ties if non_ties else 0.5,
                    "sign_test_p": _two_sided_sign_test(wins, losses),
                }
            )
    return rows


def write_weighttraits_paired_comparisons_json(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    registry: str | Path | None = None,
) -> None:
    """Write a provenance-bearing paired-comparison artifact."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": WEIGHTTRAITS_PAIRED_COMPARISONS_SCHEMA,
        "producer": "weighttraits",
        "registry": None if registry is None else str(registry),
        "n_rows": len(rows),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_weighttraits_paired_comparisons_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write paired effects in stable long-form columns."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRED_COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _load_conditions(registry: dict[str, Any], *, base_dir: Path) -> dict[str, dict[str, Any]]:
    raw_conditions = registry.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("paired-comparison registry requires non-empty conditions")
    conditions: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_conditions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"condition {index} must be a mapping")
        condition_id = _required_text(raw, "id", context=f"condition {index}")
        if condition_id in conditions:
            raise ValueError(f"duplicate paired condition id {condition_id!r}")
        summary_source = _required_text(raw, "summary", context=condition_id)
        summary_path = Path(summary_source)
        if not summary_path.is_absolute():
            summary_path = base_dir / summary_path
        summary = json.loads(summary_path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"run-set summary must be a mapping: {summary_path}")
        expected_artifact = _required_text(raw, "artifact", context=condition_id)
        expected_representation = _required_text(raw, "representation", context=condition_id)
        if summary.get("artifact") != expected_artifact:
            raise ValueError(f"artifact mismatch for paired condition {condition_id!r}")
        rows = summary.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"run-set summary requires non-empty rows: {summary_path}")
        if {str(row.get("analysis_engine")) for row in rows if isinstance(row, dict)} != {
            "direct"
        }:
            raise ValueError(f"run-set summary is not native direct analysis: {summary_path}")
        if {str(row.get("representation")) for row in rows if isinstance(row, dict)} != {
            expected_representation
        }:
            raise ValueError(f"representation mismatch for paired condition {condition_id!r}")
        expected_trees = raw.get("expected_trees")
        if expected_trees is not None and int(summary.get("n_tree_summaries", -1)) != int(
            expected_trees
        ):
            raise ValueError(f"tree-count mismatch for paired condition {condition_id!r}")
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid row in run-set summary: {summary_path}")
            key = (
                _required_text(row, "tree_id", context=condition_id),
                _required_text(row, "metric", context=condition_id),
            )
            if key in indexed:
                raise ValueError(f"duplicate tree/metric row {key!r} in {summary_path}")
            indexed[key] = row
        conditions[condition_id] = {"rows": indexed, "summary": summary_source}
    return conditions


def _load_outcomes(registry: dict[str, Any]) -> list[dict[str, Any]]:
    raw_outcomes = registry.get("outcomes")
    if not isinstance(raw_outcomes, list) or not raw_outcomes:
        raise ValueError("paired-comparison registry requires non-empty outcomes")
    outcomes = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_outcomes, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"outcome {index} must be a mapping")
        outcome_id = _required_text(raw, "id", context=f"outcome {index}")
        if outcome_id in seen:
            raise ValueError(f"duplicate paired outcome id {outcome_id!r}")
        seen.add(outcome_id)
        direction = raw.get("direction", "higher")
        if direction not in {"higher", "lower"}:
            raise ValueError(f"outcome {outcome_id!r} direction must be higher or lower")
        scale = float(raw.get("scale", 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"outcome {outcome_id!r} requires a positive finite scale")
        outcomes.append(
            {
                "id": outcome_id,
                "field": _required_text(raw, "field", context=outcome_id),
                "label": _required_text(raw, "label", context=outcome_id),
                "units": str(raw.get("units", "")),
                "direction": 1.0 if direction == "higher" else -1.0,
                "scale": scale,
            }
        )
    return outcomes


def _load_side(
    comparison: dict[str, Any],
    key: str,
    conditions: dict[str, dict[str, Any]],
    comparison_id: str,
) -> dict[str, Any]:
    raw = comparison.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"comparison {comparison_id!r} requires mapping {key!r}")
    condition_id = _required_text(raw, "condition", context=f"{comparison_id}/{key}")
    metric = _required_text(raw, "metric", context=f"{comparison_id}/{key}")
    if condition_id not in conditions:
        raise ValueError(f"comparison {comparison_id!r} references unknown condition {condition_id!r}")
    indexed = conditions[condition_id]["rows"]
    rows = {tree_id: row for (tree_id, row_metric), row in indexed.items() if row_metric == metric}
    if not rows:
        raise ValueError(f"condition {condition_id!r} has no rows for metric {metric!r}")
    return {"condition_id": condition_id, "metric": metric, "rows": rows}


def _bootstrap_mean_ci(
    values: list[float],
    *,
    n_bootstrap: int,
    seed: str,
) -> tuple[float, float]:
    if not values:
        raise ValueError("paired bootstrap requires at least one value")
    rng = random.Random(seed)
    n_values = len(values)
    estimates = sorted(
        fmean(values[rng.randrange(n_values)] for _ in range(n_values))
        for _ in range(n_bootstrap)
    )
    return (_quantile(estimates, 0.025), _quantile(estimates, 0.975))


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _two_sided_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _required_text(mapping: dict[str, Any], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty string field {key!r}")
    return value


def _required_number(mapping: dict[str, Any], key: str, *, context: str) -> float:
    raw = mapping.get(key)
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"{context} requires numeric field {key!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} requires numeric field {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} requires finite field {key!r}")
    return value
