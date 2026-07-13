"""Run-set result aggregation for direct whitebox analyses."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_SET_RESULT_COLUMNS = [
    "tree_id",
    "artifact",
    "representation",
    "metric",
    "analysis_engine",
    "tree_builder",
    "rf_engine",
    "n_models",
    "n_layers",
    "n_truth_leaves",
    "n_truth_splits",
    "n_estimate_splits",
    "rf",
    "normalized_rf",
    "exact_tree_recovery",
    "polytomy_aware_exact_recovery",
    "clade_recovery",
    "split_precision",
    "false_negative",
    "false_positive",
    "distance_mean",
    "distance_min",
    "distance_max",
    "four_point_mean_additivity",
    "four_point_informative_mean_additivity",
    "four_point_fraction_clean",
    "four_point_mean_additivity_magnitude",
    "four_point_informative_split_accuracy",
    "atteson_error_linf",
    "atteson_bottleneck_margin",
    "atteson_internal_bottleneck_margin",
    "atteson_theorem_certified",
    "truth_manifest",
    "ledger",
    "summary",
    "score",
    "four_point_additivity",
    "atteson_margin",
]


def summarize_training_run_set_analysis(
    analysis_root: str | Path,
    *,
    artifact: str,
    path_base: str | Path = ".",
    tree_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize every available direct-analysis summary under a run-set root."""
    base = Path(path_base)
    root = _resolve_path(analysis_root, base)
    requested = set(tree_ids or [])
    summaries = _discover_summary_paths(root, artifact=artifact, tree_ids=tree_ids)
    rows = []
    for summary_path in summaries:
        rows.extend(_rows_from_tree_summary(summary_path, base=base, root=root))

    found_tree_ids = {row["tree_id"] for row in rows}
    return {
        "valid": not requested - found_tree_ids,
        "analysis_root": str(root),
        "path_base": str(path_base),
        "artifact": artifact,
        "n_tree_summaries": len(summaries),
        "n_rows": len(rows),
        "tree_ids": sorted(found_tree_ids),
        "missing_tree_ids": sorted(requested - found_tree_ids),
        "metrics": sorted({str(row["metric"]) for row in rows}),
        "aggregate_by_metric": _aggregate_by_metric(rows),
        "rows": rows,
    }


def write_run_set_analysis_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write per-tree, per-metric direct analysis rows as CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_SET_RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in RUN_SET_RESULT_COLUMNS})


def _discover_summary_paths(
    root: Path,
    *,
    artifact: str,
    tree_ids: Sequence[str] | None,
) -> list[Path]:
    analysis_dir_name = _analysis_dir_name(artifact)
    if tree_ids:
        candidates = [root / tree_id / analysis_dir_name / "summary.json" for tree_id in tree_ids]
    else:
        candidates = sorted(root.glob(f"*/{analysis_dir_name}/summary.json"))
    return [path for path in candidates if path.exists()]


def _rows_from_tree_summary(summary_path: Path, *, base: Path, root: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"summary results must be a list: {summary_path}")
    tree_id = _tree_id_from_summary_path(summary_path, root)
    rows = []
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError(f"summary result rows must be mappings: {summary_path}")
        score = _load_score(result.get("score"), base=base, summary_dir=summary_path.parent)
        polytomy_aware_exact = _polytomy_aware_exact_recovery(result, score)
        rows.append(
            {
                "tree_id": tree_id,
                "artifact": summary.get("artifact"),
                "representation": summary.get("representation"),
                "metric": result.get("metric"),
                "analysis_engine": summary.get("analysis_engine"),
                "tree_builder": summary.get("tree_builder"),
                "rf_engine": summary.get("rf_engine"),
                "n_models": summary.get("n_models"),
                "n_layers": summary.get("n_layers"),
                "n_truth_leaves": score.get("n_truth_leaves"),
                "n_truth_splits": score.get("n_truth_splits"),
                "n_estimate_splits": score.get("n_estimate_splits"),
                "rf": result.get("rf"),
                "normalized_rf": result.get("normalized_rf"),
                "exact_tree_recovery": result.get("exact_tree_recovery"),
                "exact_tree_recovery_numeric": _bool_numeric(result.get("exact_tree_recovery")),
                "polytomy_aware_exact_recovery": polytomy_aware_exact,
                "polytomy_aware_exact_recovery_numeric": _bool_numeric(
                    polytomy_aware_exact
                ),
                "clade_recovery": result.get("clade_recovery"),
                "split_precision": result.get("split_precision"),
                "true_positive": score.get("true_positive"),
                "false_negative": score.get("false_negative"),
                "false_positive": score.get("false_positive"),
                "distance_mean": result.get("distance_mean"),
                "distance_min": result.get("distance_min"),
                "distance_max": result.get("distance_max"),
                "four_point_mean_additivity": result.get(
                    "four_point_mean_additivity"
                ),
                "four_point_informative_mean_additivity": result.get(
                    "four_point_informative_mean_additivity"
                ),
                "four_point_fraction_clean": result.get("four_point_fraction_clean"),
                "four_point_mean_additivity_magnitude": result.get(
                    "four_point_mean_additivity_magnitude"
                ),
                "four_point_informative_split_accuracy": result.get(
                    "four_point_informative_split_accuracy"
                ),
                "atteson_error_linf": result.get("atteson_error_linf"),
                "atteson_bottleneck_margin": result.get("atteson_bottleneck_margin"),
                "atteson_internal_bottleneck_margin": result.get(
                    "atteson_internal_bottleneck_margin"
                ),
                "atteson_theorem_certified": result.get("atteson_theorem_certified"),
                "truth_manifest": summary.get("truth_manifest"),
                "ledger": summary.get("ledger"),
                "summary": str(summary_path),
                "score": result.get("score"),
                "four_point_additivity": result.get("four_point_additivity"),
                "atteson_margin": result.get("atteson_margin"),
            }
        )
    return rows


def _load_score(raw_path: object, *, base: Path, summary_dir: Path) -> dict[str, Any]:
    if raw_path is None:
        return {}
    path = Path(str(raw_path))
    candidates = [path] if path.is_absolute() else [base / path, summary_dir / path]
    for candidate in candidates:
        if candidate.exists():
            loaded = json.loads(candidate.read_text())
            if not isinstance(loaded, dict):
                raise ValueError(f"score file must contain a mapping: {candidate}")
            return loaded
    return {}


def _aggregate_by_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    metrics = sorted({str(row["metric"]) for row in rows if row.get("metric") is not None})
    for metric in metrics:
        group = [dict(row) for row in rows if row.get("metric") == metric]
        out[metric] = _aggregate_metric_rows(group)
    return out


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_fields = [
        "rf",
        "normalized_rf",
        "clade_recovery",
        "split_precision",
        "false_negative",
        "false_positive",
        "distance_mean",
        "distance_min",
        "distance_max",
        "four_point_mean_additivity",
        "four_point_informative_mean_additivity",
        "four_point_fraction_clean",
        "four_point_mean_additivity_magnitude",
        "four_point_informative_split_accuracy",
        "atteson_error_linf",
        "atteson_bottleneck_margin",
        "atteson_internal_bottleneck_margin",
    ]
    out: dict[str, Any] = {"n_trees": len(rows)}
    for field in numeric_fields:
        values = [_optional_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        out[f"{field}_mean"] = _mean(values)
        out[f"{field}_se"] = _sample_standard_error(values)

    exact_values = [
        _bool_numeric(row.get("exact_tree_recovery"))
        for row in rows
        if row.get("exact_tree_recovery") is not None
    ]
    out["exact_tree_recovery_rate"] = _mean(exact_values)
    out["exact_tree_recovery_rate_se"] = _binomial_standard_error(
        int(sum(exact_values)),
        len(exact_values),
    )

    polytomy_aware_exact_values = [
        _bool_numeric(row.get("polytomy_aware_exact_recovery"))
        for row in rows
        if row.get("polytomy_aware_exact_recovery") is not None
    ]
    out["polytomy_aware_exact_recovery_rate"] = _mean(polytomy_aware_exact_values)
    out["polytomy_aware_exact_recovery_rate_se"] = _binomial_standard_error(
        int(sum(polytomy_aware_exact_values)),
        len(polytomy_aware_exact_values),
    )

    atteson_certified_values = [
        _bool_numeric(row.get("atteson_theorem_certified"))
        for row in rows
        if row.get("atteson_theorem_certified") is not None
    ]
    out["atteson_theorem_certified_rate"] = _mean(atteson_certified_values)
    out["atteson_theorem_certified_rate_se"] = _binomial_standard_error(
        int(sum(atteson_certified_values)),
        len(atteson_certified_values),
    )

    tp = _sum_optional_int(rows, "true_positive")
    fn = _sum_optional_int(rows, "false_negative")
    fp = _sum_optional_int(rows, "false_positive")
    n_truth = _sum_optional_int(rows, "n_truth_splits")
    n_estimate = sum((row.get("n_estimate_splits") or 0) for row in rows)
    if n_estimate == 0:
        n_estimate = tp + fp
    out.update(
        {
            "pooled_true_positive": tp,
            "pooled_false_negative": fn,
            "pooled_false_positive": fp,
            "pooled_truth_splits": n_truth,
            "pooled_estimate_splits": n_estimate,
            "pooled_clade_recovery": _safe_divide(tp, n_truth, default=1.0),
            "pooled_clade_recovery_se": _binomial_standard_error(tp, n_truth),
            "pooled_split_precision": _safe_divide(
                tp,
                n_estimate,
                default=1.0 if n_truth == 0 else 0.0,
            ),
            "pooled_split_precision_se": _binomial_standard_error(tp, n_estimate),
            "pooled_false_negative_rate": _safe_divide(fn, n_truth, default=0.0),
            "pooled_false_negative_rate_se": _binomial_standard_error(fn, n_truth),
            "pooled_false_discovery_rate": _safe_divide(fp, n_estimate, default=0.0),
            "pooled_false_discovery_rate_se": _binomial_standard_error(fp, n_estimate),
        }
    )
    return out


def _tree_id_from_summary_path(summary_path: Path, root: Path) -> str:
    try:
        return summary_path.relative_to(root).parts[0]
    except ValueError:
        return summary_path.parent.parent.name


def _polytomy_aware_exact_recovery(
    result: Mapping[str, Any], score: Mapping[str, Any]
) -> bool | None:
    value = result.get("polytomy_aware_exact_recovery")
    if value is None:
        value = score.get("polytomy_aware_exact_recovery")
    if value is not None:
        return bool(value)
    false_negative = score.get("false_negative")
    missing_leaves = score.get("n_missing_truth_leaves")
    extra_leaves = score.get("n_extra_estimate_leaves")
    if false_negative is None or missing_leaves is None or extra_leaves is None:
        return None
    return int(false_negative) == 0 and int(missing_leaves) == 0 and int(extra_leaves) == 0


def _analysis_dir_name(artifact: str) -> str:
    if artifact == "adapter_chain":
        return "cumulative_leaf_analysis"
    if artifact in {"model", "merged"}:
        return f"{artifact}_leaf_analysis"
    raise ValueError(f"unsupported artifact: {artifact}")


def _resolve_path(path: str | Path, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return base / value


def _sum_optional_int(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    total = 0
    for row in rows:
        value = row.get(field)
        if value is not None:
            total += int(value)
    return total


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    return float(value)


def _bool_numeric(value: Any) -> float:
    return 1.0 if value is True or str(value).lower() == "true" else 0.0


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _sample_standard_error(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _binomial_standard_error(successes: int, trials: int) -> float | None:
    if trials <= 0:
        return None
    p = successes / trials
    return math.sqrt(p * (1.0 - p) / trials)


def _safe_divide(numerator: float, denominator: float, *, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
