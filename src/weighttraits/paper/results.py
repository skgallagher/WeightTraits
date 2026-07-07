"""Paper-facing result registry helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


RECOVERY_TABLE_COLUMNS = [
    "result_set_id",
    "result_set_title",
    "summary_id",
    "artifact",
    "representation",
    "metric",
    "n_models",
    "n_layers",
    "n_truth_leaves",
    "n_truth_splits",
    "rf",
    "normalized_rf",
    "exact_tree_recovery",
    "exact_tree_recovery_rate",
    "clade_recovery",
    "split_precision",
    "distance_mean",
    "distance_min",
    "distance_max",
    "environment",
    "status",
    "truth_manifest",
    "ledger",
    "summary",
]


def load_recovery_registry(path: str | Path) -> dict[str, Any]:
    """Load a recovery-result registry YAML file."""

    registry_path = Path(path)
    data = yaml.safe_load(registry_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"recovery registry must be a mapping: {registry_path}")
    result_sets = data.get("result_sets")
    if not isinstance(result_sets, list) or not result_sets:
        raise ValueError(f"recovery registry requires a non-empty result_sets list: {registry_path}")
    return data


def recovery_table_rows(
    registry_path: str | Path,
    *,
    base_dir: str | Path = ".",
) -> list[dict[str, Any]]:
    """Build one table row per registered summary metric result."""

    registry = load_recovery_registry(registry_path)
    base = Path(base_dir)
    rows: list[dict[str, Any]] = []
    for result_set in registry["result_sets"]:
        if not isinstance(result_set, dict):
            raise ValueError("recovery registry result_sets entries must be mappings")
        result_set_id = _required_str(result_set, "id")
        summaries = result_set.get("summaries")
        if not isinstance(summaries, list) or not summaries:
            raise ValueError(f"result set {result_set_id} requires a non-empty summaries list")
        for summary_entry in summaries:
            rows.extend(
                _rows_from_summary_entry(
                    result_set,
                    summary_entry,
                    base_dir=base,
                )
            )
    return rows


def write_recovery_table_json(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    registry: str | Path | None = None,
) -> None:
    """Write recovery table rows as a small JSON artifact."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry": str(registry) if registry is not None else None,
        "n_rows": len(rows),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_recovery_table_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write recovery table rows as CSV with stable columns."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOVERY_TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in RECOVERY_TABLE_COLUMNS})


def _rows_from_summary_entry(
    result_set: dict[str, Any],
    summary_entry: Any,
    *,
    base_dir: Path,
) -> list[dict[str, Any]]:
    if not isinstance(summary_entry, dict):
        raise ValueError(f"summary entry in {result_set['id']} must be a mapping")
    summary_path = _resolve_registered_path(base_dir, _required_str(summary_entry, "summary"))
    summary = _load_summary(summary_path)

    expected_artifact = summary_entry.get("artifact")
    if expected_artifact is not None and summary.get("artifact") != expected_artifact:
        raise ValueError(
            f"summary artifact mismatch for {summary_path}: "
            f"registry has {expected_artifact!r}, summary has {summary.get('artifact')!r}"
        )

    aggregate = summary.get("aggregate_recovery", {})
    if not isinstance(aggregate, dict):
        raise ValueError(f"summary aggregate_recovery must be a mapping: {summary_path}")
    results = summary.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"summary requires a non-empty results list: {summary_path}")

    truth_manifest = str(result_set.get("truth_manifest") or summary.get("truth_manifest") or "")
    status = str(summary_entry.get("status") or result_set.get("status") or "")
    environment = str(summary_entry.get("environment") or result_set.get("environment") or "")
    rows = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"summary result rows must be mappings: {summary_path}")
        rows.append(
            {
                "result_set_id": result_set["id"],
                "result_set_title": str(result_set.get("title", "")),
                "summary_id": str(summary_entry.get("id") or summary_entry.get("label") or ""),
                "artifact": summary["artifact"],
                "representation": summary.get("representation"),
                "metric": result.get("metric"),
                "n_models": summary.get("n_models"),
                "n_layers": summary.get("n_layers"),
                "n_truth_leaves": aggregate.get("n_truth_leaves_mean"),
                "n_truth_splits": aggregate.get("n_truth_splits_mean"),
                "rf": result.get("rf"),
                "normalized_rf": result.get("normalized_rf"),
                "exact_tree_recovery": result.get("exact_tree_recovery"),
                "exact_tree_recovery_rate": aggregate.get("exact_tree_recovery_rate"),
                "clade_recovery": result.get("clade_recovery"),
                "split_precision": result.get("split_precision"),
                "distance_mean": result.get("distance_mean"),
                "distance_min": result.get("distance_min"),
                "distance_max": result.get("distance_max"),
                "environment": environment,
                "status": status,
                "truth_manifest": truth_manifest,
                "ledger": summary.get("ledger"),
                "summary": str(summary_path),
            }
        )
    return rows


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"registered summary does not exist: {path}")
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    for key in ("artifact", "results"):
        if key not in summary:
            raise ValueError(f"summary is missing required key {key!r}: {path}")
    return summary


def _resolve_registered_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"registry entry requires non-empty string key {key!r}")
    return value


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)
