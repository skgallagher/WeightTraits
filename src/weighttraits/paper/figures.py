"""Paper figures generated only from native WeightTraits result artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

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
