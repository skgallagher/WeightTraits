"""Paper figures generated only from native WeightTraits result artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from weighttraits.paper.diagnostics import WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA
from weighttraits.paper.paired import WEIGHTTRAITS_PAIRED_COMPARISONS_SCHEMA
from weighttraits.paper.results import WEIGHTTRAITS_VARIANTS_SCHEMA


VARIANT_DIAGNOSTIC_METRICS = (
    ("rank_biserial", "rank_biserial_se", "Branch rank-biserial", (-1.0, 1.0)),
    ("within_run_r", "within_run_r_se", "Within-run correlation", (-1.0, 1.0)),
    (
        "clade_recovery_pct",
        "clade_recovery_se_pct",
        "Clade recovery (%)",
        (0.0, 100.0),
    ),
    (
        "exact_recovery_pct",
        "exact_recovery_se_pct",
        "Polytomy-aware exact recovery (%)",
        (0.0, 100.0),
    ),
    ("rf_mean", "rf_se", "Robinson-Foulds distance", None),
    ("fn_mean", "fn_se", "False-negative splits", None),
)


def load_weighttraits_variants_artifact(path: str | Path) -> list[dict[str, Any]]:
    """Load a native variants artifact and reject unverified/legacy table inputs."""

    artifact = Path(path)
    if artifact.suffix.lower() != ".json":
        raise ValueError("WeightTraits variants figures require the provenance-bearing JSON artifact")
    payload = json.loads(artifact.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"WeightTraits variants artifact must be a JSON object: {artifact}")
    if payload.get("schema") != WEIGHTTRAITS_VARIANTS_SCHEMA:
        raise ValueError(
            f"WeightTraits variants artifact has missing or invalid schema: {artifact}"
        )
    if payload.get("producer") != "weighttraits":
        raise ValueError(f"WeightTraits variants artifact has invalid producer: {artifact}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"WeightTraits variants artifact requires non-empty rows: {artifact}")
    if payload.get("n_rows") != len(rows):
        raise ValueError(f"WeightTraits variants artifact row-count mismatch: {artifact}")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"WeightTraits variants row {index} must be an object: {artifact}")
        row = dict(raw)
        variant_id = _required_text(row, "variant_id", index=index)
        if variant_id in seen_ids:
            raise ValueError(f"duplicate WeightTraits variant_id {variant_id!r}: {artifact}")
        seen_ids.add(variant_id)
        _required_text(row, "label", index=index)
        for value_key, se_key, _, bounds in VARIANT_DIAGNOSTIC_METRICS:
            value = _required_finite(row, value_key, index=index)
            se = _optional_finite(row, se_key, index=index)
            if se is not None and se < 0:
                raise ValueError(f"row {index} field {se_key!r} must be non-negative")
            if bounds is not None and not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"row {index} field {value_key!r}={value} is outside {bounds}"
                )
            if bounds is None and value < 0:
                raise ValueError(f"row {index} field {value_key!r} must be non-negative")
            row[value_key] = value
            row[se_key] = se
        normalized.append(row)
    return normalized


def plot_weighttraits_variants_diagnostics(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    title: str = "Independent WeightTraits variant diagnostics",
) -> dict[str, Any]:
    """Plot ordering and recovery estimands from fresh WeightTraits table rows."""

    if not rows:
        raise ValueError("at least one WeightTraits variant row is required")

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "weighttraits"
    import matplotlib.pyplot as plt

    out = Path(path)
    if out.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("figure output must use .png, .pdf, or .svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["label"]) for row in rows]
    positions = list(range(len(rows)))
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharey=True)
    figure.set_facecolor("white")
    for metric_index, (value_key, se_key, metric_title, bounds) in enumerate(
        VARIANT_DIAGNOSTIC_METRICS
    ):
        axis = axes.flat[metric_index]
        axis.set_facecolor("white")
        values = [float(row[value_key]) for row in rows]
        errors = [0.0 if row.get(se_key) is None else float(row[se_key]) for row in rows]
        axis.errorbar(
            values,
            positions,
            xerr=errors,
            fmt="o",
            color="#2f6f8f",
            ecolor="#7294a6",
            elinewidth=1.4,
            capsize=3,
            markersize=5.5,
        )
        axis.set_title(metric_title)
        axis.grid(axis="x", color="#d9dee2", linewidth=0.8)
        axis.set_axisbelow(True)
        if bounds is not None:
            axis.set_xlim(*bounds)
        else:
            upper = max(value + error for value, error in zip(values, errors, strict=True))
            axis.set_xlim(0.0, max(1.0, upper * 1.12))
        if value_key in {"rank_biserial", "within_run_r"}:
            axis.axvline(0.0, color="#737b80", linewidth=0.9)
        axis.set_yticks(positions, labels=labels)

    axes.flat[0].invert_yaxis()

    figure.suptitle(title)
    figure.text(
        0.5,
        0.015,
        "Points are run-level means; horizontal intervals are ±1 standard error.",
        ha="center",
        fontsize=9,
        color="#4f5960",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    metadata = _figure_metadata(out.suffix.lower())
    figure.savefig(out, dpi=200, metadata=metadata, facecolor="white", transparent=False)
    plt.close(figure)

    return {
        "out": str(out),
        "schema": WEIGHTTRAITS_VARIANTS_SCHEMA,
        "n_variants": len(rows),
        "variant_ids": [str(row["variant_id"]) for row in rows],
        "metrics": [spec[0] for spec in VARIANT_DIAGNOSTIC_METRICS],
    }


def load_weighttraits_runset_diagnostics_artifact(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load provenance-bearing run-set diagnostics and reject legacy inputs."""

    artifact = Path(path)
    if artifact.suffix.lower() != ".json":
        raise ValueError("WeightTraits run-set figures require the provenance-bearing JSON artifact")
    payload = json.loads(artifact.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"WeightTraits run-set diagnostics must be a JSON object: {artifact}")
    if payload.get("schema") != WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA:
        raise ValueError(f"WeightTraits run-set diagnostics have invalid schema: {artifact}")
    if payload.get("producer") != "weighttraits":
        raise ValueError(f"WeightTraits run-set diagnostics have invalid producer: {artifact}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"WeightTraits run-set diagnostics require non-empty rows: {artifact}")
    if payload.get("n_rows") != len(rows):
        raise ValueError(f"WeightTraits run-set diagnostics row-count mismatch: {artifact}")

    required_numbers = (
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
    )
    normalized = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"run-set diagnostics row {index} must be an object")
        row = dict(raw)
        condition_id = _required_text(row, "condition_id", index=index)
        _required_text(row, "label", index=index)
        metric = _required_text(row, "metric", index=index)
        key = (condition_id, metric)
        if key in seen:
            raise ValueError(f"duplicate run-set diagnostics row {key!r}")
        seen.add(key)
        for field in required_numbers:
            row[field] = _required_finite(row, field, index=index)
        normalized.append(row)
    return normalized


def plot_weighttraits_metric_robustness(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    title: str = "WeightTraits distance-metric robustness",
) -> dict[str, Any]:
    """Plot recovery sensitivity across L2, cosine, and correlation distances."""

    if not rows:
        raise ValueError("at least one WeightTraits run-set diagnostics row is required")
    plt, out = _figure_context(path)
    condition_ids = list(dict.fromkeys(str(row["condition_id"]) for row in rows))
    metric_order = [
        metric
        for metric in ("l2", "cosine", "correlation")
        if any(str(row["metric"]) == metric for row in rows)
    ]
    if not metric_order:
        raise ValueError("run-set diagnostics do not contain supported metrics")
    indexed = {(str(row["condition_id"]), str(row["metric"])): row for row in rows}
    for condition_id in condition_ids:
        missing = [metric for metric in metric_order if (condition_id, metric) not in indexed]
        if missing:
            raise ValueError(f"condition {condition_id!r} is missing metrics {missing}")

    panel_specs = (
        ("clade_recovery_pct", "clade_recovery_se_pct", "Clade recovery (%)", (0, 100)),
        ("exact_recovery_pct", "exact_recovery_se_pct", "Exact recovery (%)", (0, 100)),
        ("rf_mean", "rf_se", "Robinson-Foulds distance", None),
        ("fn_mean", "fn_se", "False-negative splits", None),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    figure.set_facecolor("white")
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "P", "X")
    positions = list(range(len(metric_order)))
    for panel_index, (value_key, se_key, panel_title, bounds) in enumerate(panel_specs):
        axis = axes.flat[panel_index]
        axis.set_facecolor("white")
        for condition_index, condition_id in enumerate(condition_ids):
            condition_rows = [indexed[(condition_id, metric)] for metric in metric_order]
            axis.errorbar(
                positions,
                [float(row[value_key]) for row in condition_rows],
                yerr=[float(row[se_key]) for row in condition_rows],
                marker=markers[condition_index % len(markers)],
                color=colors[condition_index % len(colors)],
                linewidth=1.5,
                capsize=3,
                label=str(condition_rows[0]["label"]),
            )
        axis.set_title(panel_title)
        axis.grid(axis="y", color="#d9dee2", linewidth=0.8)
        axis.set_axisbelow(True)
        if bounds is not None:
            axis.set_ylim(*bounds)
        else:
            axis.set_ylim(bottom=0)
        axis.set_xticks(positions, labels=[_metric_label(metric) for metric in metric_order])

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.11, 1.0, 0.95))
    _save_figure(figure, out)
    plt.close(figure)
    return {
        "out": str(out),
        "schema": WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA,
        "n_conditions": len(condition_ids),
        "metrics": metric_order,
    }


def plot_weighttraits_additivity_recovery(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    metric: str = "cosine",
    title: str = "WeightTraits additivity and recovery",
) -> dict[str, Any]:
    """Plot fresh four-point and Atteson summaries against clade recovery."""

    selected = [row for row in rows if str(row.get("metric")) == metric]
    if not selected:
        raise ValueError(f"run-set diagnostics do not contain metric {metric!r}")
    condition_ids = [str(row["condition_id"]) for row in selected]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError(f"run-set diagnostics contain duplicate conditions for {metric!r}")

    plt, out = _figure_context(path)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), sharey=True)
    figure.set_facecolor("white")
    colors = plt.get_cmap("tab10").colors
    x_specs = (
        (
            "four_point_additivity",
            "four_point_additivity_se",
            "Four-point additivity",
        ),
        (
            "atteson_bottleneck_margin",
            "atteson_bottleneck_margin_se",
            "Oracle Atteson bottleneck margin",
        ),
    )
    y_values = [float(row["clade_recovery_pct"]) for row in selected]
    y_errors = [float(row["clade_recovery_se_pct"]) for row in selected]
    lower_y = max(0.0, min(value - error for value, error in zip(y_values, y_errors, strict=True)) - 5)
    for axis_index, (x_key, x_se_key, x_label) in enumerate(x_specs):
        axis = axes[axis_index]
        axis.set_facecolor("white")
        x_values = [float(row[x_key]) for row in selected]
        x_errors = [float(row[x_se_key]) for row in selected]
        rho = _spearman(x_values, y_values)
        for row_index, row in enumerate(selected):
            axis.errorbar(
                x_values[row_index],
                y_values[row_index],
                xerr=x_errors[row_index],
                yerr=y_errors[row_index],
                fmt="o",
                color=colors[row_index % len(colors)],
                capsize=3,
                markersize=6,
            )
            offset_y = 8 if row_index % 2 == 0 else -13
            axis.annotate(
                str(row["label"]),
                (x_values[row_index], y_values[row_index]),
                textcoords="offset points",
                xytext=(7, offset_y),
                fontsize=8.5,
            )
        axis.set_xlabel(x_label)
        axis.set_title(f"{x_label} (Spearman ρ={rho:+.2f})")
        axis.grid(color="#d9dee2", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_ylim(lower_y, 100.5)
    axes[0].set_ylabel("Clade recovery (%)")
    figure.suptitle(f"{title} — {_metric_label(metric)}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save_figure(figure, out)
    plt.close(figure)
    return {
        "out": str(out),
        "schema": WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA,
        "metric": metric,
        "n_conditions": len(selected),
        "condition_ids": condition_ids,
    }


def load_weighttraits_paired_comparisons_artifact(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load provenance-bearing paired comparisons and reject arbitrary inputs."""

    artifact = Path(path)
    if artifact.suffix.lower() != ".json":
        raise ValueError("WeightTraits paired figures require the provenance-bearing JSON artifact")
    payload = json.loads(artifact.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"WeightTraits paired comparisons must be a JSON object: {artifact}")
    if payload.get("schema") != WEIGHTTRAITS_PAIRED_COMPARISONS_SCHEMA:
        raise ValueError(f"WeightTraits paired comparisons have invalid schema: {artifact}")
    if payload.get("producer") != "weighttraits":
        raise ValueError(f"WeightTraits paired comparisons have invalid producer: {artifact}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"WeightTraits paired comparisons require non-empty rows: {artifact}")
    if payload.get("n_rows") != len(rows):
        raise ValueError(f"WeightTraits paired comparisons row-count mismatch: {artifact}")
    required_numbers = (
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
    )
    normalized = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"paired-comparison row {index} must be an object")
        row = dict(raw)
        comparison_id = _required_text(row, "comparison_id", index=index)
        outcome = _required_text(row, "outcome", index=index)
        _required_text(row, "label", index=index)
        _required_text(row, "outcome_label", index=index)
        key = (comparison_id, outcome)
        if key in seen:
            raise ValueError(f"duplicate paired-comparison row {key!r}")
        seen.add(key)
        for field in required_numbers:
            row[field] = _required_finite(row, field, index=index)
        if row["ci_low"] > row["ci_high"]:
            raise ValueError(f"paired-comparison row {index} has reversed confidence interval")
        normalized.append(row)
    return normalized


def plot_weighttraits_paired_effects(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    group: str,
    title: str = "WeightTraits paired same-tree effects",
) -> dict[str, Any]:
    """Plot paired mean effects and bootstrap intervals for one comparison group."""

    selected = [row for row in rows if str(row.get("group")) == group]
    if not selected:
        raise ValueError(f"paired comparisons do not contain group {group!r}")
    comparison_ids = list(dict.fromkeys(str(row["comparison_id"]) for row in selected))
    outcomes = list(dict.fromkeys(str(row["outcome"]) for row in selected))
    indexed = {(str(row["comparison_id"]), str(row["outcome"])): row for row in selected}
    for comparison_id in comparison_ids:
        missing = [outcome for outcome in outcomes if (comparison_id, outcome) not in indexed]
        if missing:
            raise ValueError(f"paired comparison {comparison_id!r} is missing outcomes {missing}")

    plt, out = _figure_context(path)
    n_columns = 2
    n_rows = math.ceil(len(outcomes) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(11.5, max(4.5, 3.2 * n_rows)),
        squeeze=False,
    )
    figure.set_facecolor("white")
    y_positions = list(range(len(comparison_ids)))
    for outcome_index, outcome in enumerate(outcomes):
        axis = axes.flat[outcome_index]
        axis.set_facecolor("white")
        outcome_rows = [indexed[(comparison_id, outcome)] for comparison_id in comparison_ids]
        means = [float(row["mean_effect"]) for row in outcome_rows]
        lower_errors = [
            mean - float(row["ci_low"])
            for mean, row in zip(means, outcome_rows, strict=True)
        ]
        upper_errors = [
            float(row["ci_high"]) - mean
            for mean, row in zip(means, outcome_rows, strict=True)
        ]
        axis.errorbar(
            means,
            y_positions,
            xerr=[lower_errors, upper_errors],
            fmt="o",
            color="#2f7699",
            capsize=3,
        )
        axis.axvline(0.0, color="#7a858c", linewidth=1.0)
        axis.grid(axis="x", color="#d9dee2", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_yticks(
            y_positions,
            labels=[str(row["label"]) for row in outcome_rows],
        )
        axis.invert_yaxis()
        units = str(outcome_rows[0].get("units", ""))
        axis.set_xlabel(f"Paired effect ({units}); positive favors first" if units else "Paired effect; positive favors first")
        axis.set_title(str(outcome_rows[0]["outcome_label"]))
    for empty_index in range(len(outcomes), len(axes.flat)):
        axes.flat[empty_index].set_visible(False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_figure(figure, out)
    plt.close(figure)
    return {
        "out": str(out),
        "schema": WEIGHTTRAITS_PAIRED_COMPARISONS_SCHEMA,
        "group": group,
        "n_comparisons": len(comparison_ids),
        "outcomes": outcomes,
    }


def _required_text(row: dict[str, Any], key: str, *, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {index} requires non-empty string field {key!r}")
    return value


def _required_finite(row: dict[str, Any], key: str, *, index: int) -> float:
    value = _optional_finite(row, key, index=index)
    if value is None:
        raise ValueError(f"row {index} requires numeric field {key!r}")
    return value


def _optional_finite(row: dict[str, Any], key: str, *, index: int) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        raise ValueError(f"row {index} field {key!r} must be numeric")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {index} field {key!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {index} field {key!r} must be finite")
    return value


def _figure_metadata(suffix: str) -> dict[str, Any]:
    if suffix == ".svg":
        return {"Creator": "WeightTraits", "Date": None}
    if suffix == ".pdf":
        return {"Creator": "WeightTraits", "CreationDate": None, "ModDate": None}
    return {"Software": "WeightTraits"}


def _figure_context(path: str | Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "weighttraits"
    import matplotlib.pyplot as plt

    out = Path(path)
    if out.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("figure output must use .png, .pdf, or .svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    return plt, out


def _save_figure(figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=200,
        metadata=_figure_metadata(path.suffix.lower()),
        facecolor="white",
        transparent=False,
    )


def _metric_label(metric: str) -> str:
    return {"l2": "L2", "cosine": "Cosine", "correlation": "Correlation"}.get(
        metric,
        metric,
    )


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires paired values")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left_ranks)
    right_ss = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_ss * right_ss)
    return 0.0 if denominator == 0.0 else numerator / denominator


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index, _ in ordered[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks
