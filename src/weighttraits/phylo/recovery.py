"""Split, RF, clade-recovery, and exact-tree scoring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

from weighttraits.phylo.splits import Split, canonical_split, serialize_split


def project_splits(splits: Iterable[Split], common_leaves: set[str]) -> set[Split]:
    """Project splits to a shared leaf set and drop newly trivial splits."""

    projected: set[Split] = set()
    for split in splits:
        key = canonical_split(split, common_leaves)
        if key is not None:
            projected.add(key)
    return projected


def score_split_recovery(
    truth_splits: Iterable[Split],
    estimate_splits: Iterable[Split],
    truth_leaves: Iterable[str],
    estimate_leaves: Iterable[str],
) -> dict[str, Any]:
    """Score estimated splits against truth with explicit TP/FP/FN accounting."""

    truth_leaf_set = set(truth_leaves)
    estimate_leaf_set = set(estimate_leaves)
    common_leaves = truth_leaf_set & estimate_leaf_set

    truth_projected = project_splits(truth_splits, common_leaves)
    estimate_projected = project_splits(estimate_splits, common_leaves)

    true_positive = truth_projected & estimate_projected
    false_negative = truth_projected - estimate_projected
    false_positive = estimate_projected - truth_projected

    n_truth = len(truth_projected)
    n_estimate = len(estimate_projected)
    tp = len(true_positive)
    fn = len(false_negative)
    fp = len(false_positive)
    rf = fn + fp
    rf_denominator = n_truth + n_estimate

    exact_tree_recovery = (
        truth_leaf_set == estimate_leaf_set
        and truth_projected == estimate_projected
    )

    return {
        "n_truth_leaves": len(truth_leaf_set),
        "n_estimate_leaves": len(estimate_leaf_set),
        "n_common_leaves": len(common_leaves),
        "n_missing_truth_leaves": len(truth_leaf_set - estimate_leaf_set),
        "n_extra_estimate_leaves": len(estimate_leaf_set - truth_leaf_set),
        "n_truth_splits": n_truth,
        "n_estimate_splits": n_estimate,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "rf": rf,
        "normalized_rf": _safe_divide(rf, rf_denominator, default=0.0),
        "clade_recovery": _safe_divide(tp, n_truth, default=1.0),
        "false_negative_rate": _safe_divide(fn, n_truth, default=0.0),
        "split_precision": _safe_divide(tp, n_estimate, default=1.0 if n_truth == 0 else 0.0),
        "false_discovery_rate": _safe_divide(fp, n_estimate, default=0.0),
        "exact_tree_recovery": exact_tree_recovery,
        "exact_tree_recovery_numeric": 1.0 if exact_tree_recovery else 0.0,
        "false_negative_splits": [serialize_split(split) for split in sorted(false_negative, key=sorted)],
        "false_positive_splits": [serialize_split(split) for split in sorted(false_positive, key=sorted)],
    }


def aggregate_recovery(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate recovery records and attach standard errors.

    For every numeric metric, this reports a mean and a sample standard error
    across records. It also reports pooled split-level proportions with binomial
    standard errors for clade recovery, split precision, FN rate, and FP rate.
    """

    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("cannot aggregate zero recovery records")

    numeric_keys = sorted(
        key
        for key in set().union(*(row.keys() for row in rows))
        if all(_is_number(row.get(key)) for row in rows)
    )

    out: dict[str, Any] = {"n_records": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = sum(values) / len(values)
        out[f"{key}_se"] = sample_standard_error(values)

    tp = sum(int(row.get("true_positive", 0)) for row in rows)
    fn = sum(int(row.get("false_negative", 0)) for row in rows)
    fp = sum(int(row.get("false_positive", 0)) for row in rows)
    n_truth = sum(int(row.get("n_truth_splits", 0)) for row in rows)
    n_estimate = sum(int(row.get("n_estimate_splits", 0)) for row in rows)
    exact = sum(float(row.get("exact_tree_recovery_numeric", 0.0)) for row in rows)

    out.update(
        {
            "pooled_true_positive": tp,
            "pooled_false_negative": fn,
            "pooled_false_positive": fp,
            "pooled_truth_splits": n_truth,
            "pooled_estimate_splits": n_estimate,
            "pooled_clade_recovery": _safe_divide(tp, n_truth, default=1.0),
            "pooled_clade_recovery_se": binomial_standard_error(tp, n_truth),
            "pooled_false_negative_rate": _safe_divide(fn, n_truth, default=0.0),
            "pooled_false_negative_rate_se": binomial_standard_error(fn, n_truth),
            "pooled_split_precision": _safe_divide(tp, n_estimate, default=1.0 if n_truth == 0 else 0.0),
            "pooled_split_precision_se": binomial_standard_error(tp, n_estimate),
            "pooled_false_discovery_rate": _safe_divide(fp, n_estimate, default=0.0),
            "pooled_false_discovery_rate_se": binomial_standard_error(fp, n_estimate),
            "exact_tree_recovery_rate": exact / len(rows),
            "exact_tree_recovery_rate_se": binomial_standard_error(int(exact), len(rows)),
        }
    )
    return out


def sample_standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def binomial_standard_error(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    p = successes / trials
    return math.sqrt(p * (1.0 - p) / trials)


def _safe_divide(numerator: float, denominator: float, *, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

