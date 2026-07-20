"""Native run-set diagnostics for independent paper analyses."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA = "weighttraits.runset_diagnostics.v1"

RUNSET_DIAGNOSTIC_COLUMNS = [
    "condition_id",
    "label",
    "section",
    "artifact",
    "representation",
    "summary",
    "metric",
    "n_trees",
    "clade_recovery_pct",
    "clade_recovery_se_pct",
    "exact_recovery_pct",
    "exact_recovery_se_pct",
    "rf_mean",
    "rf_se",
    "fn_mean",
    "fn_se",
    "four_point_additivity",
    "four_point_additivity_se",
    "atteson_bottleneck_margin",
    "atteson_bottleneck_margin_se",
    "atteson_certified_pct",
    "atteson_certified_se_pct",
]


def weighttraits_runset_diagnostic_rows(
    registry_path: str | Path,
    *,
    base_dir: str | Path = ".",
) -> list[dict[str, Any]]:
    """Build long-form diagnostics from native WeightTraits run-set summaries."""

    registry_file = Path(registry_path)
    registry = yaml.safe_load(registry_file.read_text())
    if not isinstance(registry, dict):
        raise ValueError(f"WeightTraits diagnostics registry must be a mapping: {registry_file}")
    conditions = registry.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(
            f"WeightTraits diagnostics registry requires non-empty conditions: {registry_file}"
        )
    metrics = registry.get("metrics", ["l2", "cosine", "correlation"])
    if not isinstance(metrics, list) or not metrics or not all(isinstance(item, str) for item in metrics):
        raise ValueError(f"WeightTraits diagnostics registry requires string metrics: {registry_file}")

    base = Path(base_dir)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for condition_index, condition in enumerate(conditions, start=1):
        if not isinstance(condition, dict):
            raise ValueError(f"diagnostics condition {condition_index} must be a mapping")
        condition_id = _required_text(condition, "id", context=f"condition {condition_index}")
        if condition_id in seen_ids:
            raise ValueError(f"duplicate diagnostics condition id {condition_id!r}")
        seen_ids.add(condition_id)
        label = _required_text(condition, "label", context=condition_id)
        summary_source = _required_text(condition, "summary", context=condition_id)
        summary_path = _resolve_path(base, summary_source)
        summary = json.loads(summary_path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"run-set summary must be a mapping: {summary_path}")

        expected_artifact = _required_text(condition, "artifact", context=condition_id)
        if summary.get("artifact") != expected_artifact:
            raise ValueError(
                f"run-set artifact mismatch for {condition_id}: expected {expected_artifact!r}, "
                f"got {summary.get('artifact')!r}"
            )
        expected_representation = _required_text(
            condition,
            "representation",
            context=condition_id,
        )
        _validate_native_summary(
            summary,
            summary_path=summary_path,
            expected_representation=expected_representation,
            expected_trees=condition.get("expected_trees"),
        )
        aggregate_by_metric = summary.get("aggregate_by_metric")
        if not isinstance(aggregate_by_metric, dict):
            raise ValueError(f"run-set summary lacks aggregate_by_metric: {summary_path}")

        for metric in metrics:
            aggregate = aggregate_by_metric.get(metric)
            if not isinstance(aggregate, dict):
                raise ValueError(f"run-set summary {summary_path} lacks metric {metric!r}")
            rows.append(
                {
                    "condition_id": condition_id,
                    "label": label,
                    "section": str(condition.get("section", "")),
                    "artifact": expected_artifact,
                    "representation": expected_representation,
                    "summary": summary_source,
                    "metric": metric,
                    "n_trees": _required_int(aggregate, "n_trees", context=f"{condition_id}/{metric}"),
                    "clade_recovery_pct": _percent(
                        _required_number(
                            aggregate,
                            "clade_recovery_mean",
                            context=f"{condition_id}/{metric}",
                        )
                    ),
                    "clade_recovery_se_pct": _percent(
                        _required_number(
                            aggregate,
                            "clade_recovery_se",
                            context=f"{condition_id}/{metric}",
                        )
                    ),
                    "exact_recovery_pct": _percent(
                        _required_number(
                            aggregate,
                            "polytomy_aware_exact_recovery_rate",
                            context=f"{condition_id}/{metric}",
                        )
                    ),
                    "exact_recovery_se_pct": _percent(
                        _required_number(
                            aggregate,
                            "polytomy_aware_exact_recovery_rate_se",
                            context=f"{condition_id}/{metric}",
                        )
                    ),
                    "rf_mean": _required_number(
                        aggregate,
                        "rf_mean",
                        context=f"{condition_id}/{metric}",
                    ),
                    "rf_se": _required_number(
                        aggregate,
                        "rf_se",
                        context=f"{condition_id}/{metric}",
                    ),
                    "fn_mean": _required_number(
                        aggregate,
                        "false_negative_mean",
                        context=f"{condition_id}/{metric}",
                    ),
                    "fn_se": _required_number(
                        aggregate,
                        "false_negative_se",
                        context=f"{condition_id}/{metric}",
                    ),
                    "four_point_additivity": _required_number(
                        aggregate,
                        "four_point_mean_additivity_mean",
                        context=f"{condition_id}/{metric}",
                    ),
                    "four_point_additivity_se": _required_number(
                        aggregate,
                        "four_point_mean_additivity_se",
                        context=f"{condition_id}/{metric}",
                    ),
                    "atteson_bottleneck_margin": _required_number(
                        aggregate,
                        "atteson_bottleneck_margin_mean",
                        context=f"{condition_id}/{metric}",
                    ),
                    "atteson_bottleneck_margin_se": _required_number(
                        aggregate,
                        "atteson_bottleneck_margin_se",
                        context=f"{condition_id}/{metric}",
                    ),
                    "atteson_certified_pct": _percent(
                        _required_number(
                            aggregate,
                            "atteson_theorem_certified_rate",
                            context=f"{condition_id}/{metric}",
                        )
                    ),
                    "atteson_certified_se_pct": _percent(
                        _required_number(
                            aggregate,
                            "atteson_theorem_certified_rate_se",
                            context=f"{condition_id}/{metric}",
                        )
                    ),
                }
            )
    return rows


def write_weighttraits_runset_diagnostics_json(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    registry: str | Path | None = None,
) -> None:
    """Write a provenance-bearing native diagnostics artifact."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA,
        "producer": "weighttraits",
        "registry": None if registry is None else str(registry),
        "n_rows": len(rows),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_weighttraits_runset_diagnostics_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write native diagnostics in stable long-form columns."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNSET_DIAGNOSTIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in RUNSET_DIAGNOSTIC_COLUMNS} for row in rows)


def _validate_native_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    expected_representation: str,
    expected_trees: object,
) -> None:
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"run-set summary requires non-empty rows: {summary_path}")
    engines = {str(row.get("analysis_engine")) for row in rows if isinstance(row, dict)}
    if engines != {"direct"}:
        raise ValueError(f"run-set summary is not native direct analysis: {summary_path}")
    representations = {
        str(row.get("representation")) for row in rows if isinstance(row, dict)
    }
    if representations != {expected_representation}:
        raise ValueError(
            f"run-set representation mismatch in {summary_path}: {sorted(representations)}"
        )
    if expected_trees is not None:
        expected = int(expected_trees)
        observed = int(summary.get("n_tree_summaries", -1))
        if observed != expected:
            raise ValueError(
                f"run-set tree-count mismatch in {summary_path}: expected {expected}, got {observed}"
            )


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


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


def _required_int(mapping: dict[str, Any], key: str, *, context: str) -> int:
    value = _required_number(mapping, key, context=context)
    if not value.is_integer() or value < 1:
        raise ValueError(f"{context} requires positive integer field {key!r}")
    return int(value)


def _percent(value: float) -> float:
    return 100.0 * value
