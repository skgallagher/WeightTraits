"""Paper-facing result registry helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
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


ELLMTREES_VARIANTS_TABLE_COLUMNS = [
    "variant_id",
    "section",
    "model",
    "label",
    "recovery_group",
    "n_runs",
    "n_ordering_runs",
    "n_pairs",
    "rank_biserial",
    "rank_biserial_se",
    "within_run_r",
    "within_run_r_se",
    "clade_recovery_pct",
    "clade_recovery_se_pct",
    "exact_recovery_pct",
    "exact_recovery_se_pct",
    "rf_mean",
    "rf_se",
    "fn_mean",
    "fn_se",
    "recovery_source",
    "per_run_recovery_source",
    "branch_source",
]


_DRAFT_LABEL_RE = re.compile(r"\\label\{((?:fig|tab):[^}]+)\}")


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


def load_table_registry(path: str | Path) -> dict[str, Any]:
    """Load a paper table registry YAML file."""

    registry_path = Path(path)
    data = yaml.safe_load(registry_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"table registry must be a mapping: {registry_path}")
    tables = data.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError(f"table registry requires a non-empty tables list: {registry_path}")
    return data


def load_reference_registry(path: str | Path) -> dict[str, Any]:
    """Load a paper reference-surface registry YAML file."""

    registry_path = Path(path)
    data = yaml.safe_load(registry_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"reference registry must be a mapping: {registry_path}")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"reference registry requires a non-empty entries list: {registry_path}")
    return data


def load_ellmtrees_variants_registry(path: str | Path) -> dict[str, Any]:
    """Load the registry for the legacy ELLMTrees variants reference table."""

    registry_path = Path(path)
    data = yaml.safe_load(registry_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"ELLMTrees variants registry must be a mapping: {registry_path}")
    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"ELLMTrees variants registry requires a non-empty variants list: {registry_path}")
    for key in ("recovery_source", "per_run_recovery_source"):
        if not isinstance(data.get(key), str) or not data.get(key):
            raise ValueError(f"ELLMTrees variants registry requires {key}: {registry_path}")
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


def ellmtrees_variants_table_rows(
    registry_path: str | Path,
    *,
    base_dir: str | Path = ".",
) -> list[dict[str, Any]]:
    """Build a reference table for the old ELLMTrees `tab:variants` result."""

    registry = load_ellmtrees_variants_registry(registry_path)
    base = Path(base_dir)
    recovery_source = _required_str(registry, "recovery_source")
    per_run_recovery_source = _required_str(registry, "per_run_recovery_source")
    recovery_by_group = _load_csv_by_key(
        _resolve_registered_path(base, recovery_source),
        key="group",
    )
    per_run_by_group = _load_csv_groups(
        _resolve_registered_path(base, per_run_recovery_source),
        key="group",
    )

    rows = []
    for variant in registry["variants"]:
        if not isinstance(variant, dict):
            raise ValueError("ELLMTrees variants entries must be mappings")
        variant_id = _required_str(variant, "id")
        recovery_group = _required_str(variant, "recovery_group")
        branch_source = _required_str(variant, "branch_source")
        if recovery_group not in recovery_by_group:
            raise ValueError(f"recovery group {recovery_group!r} not found for variant {variant_id}")
        branch_stats = _branch_ordering_stats(_resolve_registered_path(base, branch_source))
        recovery_stats = _recovery_group_stats(
            recovery_by_group[recovery_group],
            per_run_by_group.get(recovery_group, []),
        )
        rows.append(
            {
                "variant_id": variant_id,
                "section": str(variant.get("section", "")),
                "model": str(variant.get("model", "")),
                "label": str(variant.get("label", "")),
                "recovery_group": recovery_group,
                "n_runs": recovery_stats["n_runs"],
                "n_ordering_runs": branch_stats["n_runs"],
                "n_pairs": branch_stats["n_pairs"],
                "rank_biserial": branch_stats["rank_biserial"],
                "rank_biserial_se": branch_stats["rank_biserial_se"],
                "within_run_r": branch_stats["within_run_r"],
                "within_run_r_se": branch_stats["within_run_r_se"],
                "clade_recovery_pct": recovery_stats["clade_recovery_pct"],
                "clade_recovery_se_pct": recovery_stats["clade_recovery_se_pct"],
                "exact_recovery_pct": recovery_stats["exact_recovery_pct"],
                "exact_recovery_se_pct": recovery_stats["exact_recovery_se_pct"],
                "rf_mean": recovery_stats["rf_mean"],
                "rf_se": recovery_stats["rf_se"],
                "fn_mean": recovery_stats["fn_mean"],
                "fn_se": recovery_stats["fn_se"],
                "recovery_source": recovery_source,
                "per_run_recovery_source": per_run_recovery_source,
                "branch_source": branch_source,
            }
        )
    return rows


def compare_table_artifacts(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    key_columns: list[str],
    compare_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    ignore_columns: list[str] | None = None,
    base_dir: str | Path = ".",
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> dict[str, Any]:
    """Compare two paper table artifacts row-by-row."""

    if not key_columns:
        raise ValueError("table comparison requires at least one key column")
    base = Path(base_dir)
    reference_resolved = _resolve_registered_path(base, str(reference_path))
    candidate_resolved = _resolve_registered_path(base, str(candidate_path))
    reference_rows = _load_table_rows(reference_resolved)
    candidate_rows = _load_table_rows(candidate_resolved)
    numeric_set = set(numeric_columns or [])
    ignore_set = set(ignore_columns or [])
    key_set = set(key_columns)
    issues: list[dict[str, Any]] = []
    reference_index = _index_rows(reference_rows, key_columns, "reference", issues)
    candidate_index = _index_rows(candidate_rows, key_columns, "candidate", issues)
    reference_keys = set(reference_index)
    candidate_keys = set(candidate_index)
    missing_keys = sorted(reference_keys - candidate_keys)
    extra_keys = sorted(candidate_keys - reference_keys)
    for key in missing_keys:
        issues.append(_comparison_issue("missing_row", key, "reference row is missing from candidate"))
    for key in extra_keys:
        issues.append(_comparison_issue("extra_row", key, "candidate row is not present in reference"))

    if compare_columns is None:
        all_columns = set()
        for row in reference_rows + candidate_rows:
            all_columns.update(str(column) for column in row)
        columns = sorted(all_columns - key_set - ignore_set)
    else:
        columns = [column for column in compare_columns if column not in key_set and column not in ignore_set]
    for column in numeric_set:
        if column not in key_set and column not in ignore_set and column not in columns:
            columns.append(column)

    n_compared_cells = 0
    for key in sorted(reference_keys & candidate_keys):
        reference_row = reference_index[key]
        candidate_row = candidate_index[key]
        for column in columns:
            reference_has = column in reference_row
            candidate_has = column in candidate_row
            if not reference_has or not candidate_has:
                issues.append(
                    _comparison_issue(
                        "missing_column",
                        key,
                        f"column {column!r} missing in "
                        f"{'reference' if not reference_has else 'candidate'}",
                        column=column,
                    )
                )
                continue
            n_compared_cells += 1
            reference_value = reference_row[column]
            candidate_value = candidate_row[column]
            if column in numeric_set:
                if not _numeric_close(reference_value, candidate_value, atol=atol, rtol=rtol):
                    issues.append(
                        _comparison_issue(
                            "numeric_mismatch",
                            key,
                            f"column {column!r} differs: {reference_value!r} vs {candidate_value!r}",
                            column=column,
                            reference_value=reference_value,
                            candidate_value=candidate_value,
                        )
                    )
            elif _comparable_value(reference_value) != _comparable_value(candidate_value):
                issues.append(
                    _comparison_issue(
                        "value_mismatch",
                        key,
                        f"column {column!r} differs: {reference_value!r} vs {candidate_value!r}",
                        column=column,
                        reference_value=reference_value,
                        candidate_value=candidate_value,
                    )
                )

    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "base_dir": str(base_dir),
        "key_columns": key_columns,
        "compare_columns": columns,
        "numeric_columns": sorted(numeric_set),
        "ignore_columns": sorted(ignore_set),
        "atol": atol,
        "rtol": rtol,
        "n_reference_rows": len(reference_rows),
        "n_candidate_rows": len(candidate_rows),
        "n_matched_rows": len(reference_keys & candidate_keys),
        "n_compared_cells": n_compared_cells,
        "n_issues": len(issues),
        "valid": not issues,
        "issues": issues,
    }


def run_table_registry_comparisons(
    registry_path: str | Path,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Run declared table comparisons from a paper table registry."""

    registry = load_table_registry(registry_path)
    base = Path(base_dir)
    comparisons = []
    issues = []
    for table in registry["tables"]:
        if not isinstance(table, dict):
            issues.append(_comparison_registry_issue(None, None, "invalid_table_entry", "table entries must be mappings"))
            continue
        table_id = table.get("id")
        if not isinstance(table_id, str) or not table_id:
            issues.append(_comparison_registry_issue(None, None, "missing_table_id", "table entry requires id"))
            continue
        raw_comparisons = table.get("comparisons", [])
        if raw_comparisons is None:
            raw_comparisons = []
        if not isinstance(raw_comparisons, list):
            issues.append(
                _comparison_registry_issue(table_id, None, "invalid_comparisons", "comparisons must be a list")
            )
            continue
        for raw_comparison in raw_comparisons:
            comparison_report = _run_table_comparison_spec(raw_comparison, table_id, base)
            comparisons.append(comparison_report)
            if not comparison_report["valid"]:
                issues.append(
                    _comparison_registry_issue(
                        table_id,
                        comparison_report["id"],
                        "comparison_failed",
                        f"comparison {comparison_report['id']} failed with {comparison_report['n_issues']} issues",
                    )
                )

    return {
        "registry": str(registry_path),
        "base_dir": str(base_dir),
        "n_comparisons": len(comparisons),
        "n_issues": len(issues),
        "valid": not issues,
        "comparisons": comparisons,
        "issues": issues,
    }


def _run_table_comparison_spec(raw_spec: Any, table_id: str, base_dir: Path) -> dict[str, Any]:
    if not isinstance(raw_spec, dict):
        issue = _comparison_registry_issue(table_id, None, "invalid_comparison", "comparison entries must be mappings")
        return {
            "id": None,
            "table_id": table_id,
            "valid": False,
            "n_issues": 1,
            "issues": [issue],
        }

    comparison_id = raw_spec.get("id")
    if not isinstance(comparison_id, str) or not comparison_id:
        comparison_id = None
    issues: list[dict[str, Any]] = []
    if comparison_id is None:
        issues.append(_comparison_registry_issue(table_id, None, "missing_comparison_id", "comparison requires id"))

    reference = _comparison_path(raw_spec, "reference", table_id, comparison_id, issues)
    candidate = _comparison_path(raw_spec, "candidate", table_id, comparison_id, issues)
    key_columns = _comparison_str_list(raw_spec, "key_columns", table_id, comparison_id, issues, required=True)
    compare_columns = _comparison_str_list(raw_spec, "compare_columns", table_id, comparison_id, issues)
    numeric_columns = _comparison_str_list(raw_spec, "numeric_columns", table_id, comparison_id, issues)
    ignore_columns = _comparison_str_list(raw_spec, "ignore_columns", table_id, comparison_id, issues)
    atol = _comparison_float(raw_spec, "atol", table_id, comparison_id, issues, default=1e-9)
    rtol = _comparison_float(raw_spec, "rtol", table_id, comparison_id, issues, default=1e-9)
    out = _comparison_path(raw_spec, "out", table_id, comparison_id, issues, required=False)

    if issues:
        return {
            "id": comparison_id,
            "table_id": table_id,
            "reference": reference,
            "candidate": candidate,
            "out": out,
            "valid": False,
            "n_issues": len(issues),
            "issues": issues,
        }

    try:
        report = compare_table_artifacts(
            reference,
            candidate,
            key_columns=key_columns or [],
            compare_columns=compare_columns,
            numeric_columns=numeric_columns,
            ignore_columns=ignore_columns,
            base_dir=base_dir,
            atol=atol,
            rtol=rtol,
        )
    except Exception as exc:  # pragma: no cover - exact exception type depends on artifact parser
        issue = _comparison_registry_issue(
            table_id,
            comparison_id,
            "comparison_error",
            f"{type(exc).__name__}: {exc}",
        )
        return {
            "id": comparison_id,
            "table_id": table_id,
            "reference": reference,
            "candidate": candidate,
            "out": out,
            "valid": False,
            "n_issues": 1,
            "issues": [issue],
        }

    report.update({"id": comparison_id, "table_id": table_id, "out": out})
    if out:
        out_path = _resolve_registered_path(base_dir, out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _comparison_path(
    spec: dict[str, Any],
    key: str,
    table_id: str,
    comparison_id: str | None,
    issues: list[dict[str, Any]],
    *,
    required: bool = True,
) -> str | None:
    value = spec.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        issues.append(_comparison_registry_issue(table_id, comparison_id, f"missing_{key}", f"comparison requires {key}"))
        return None
    return value


def _comparison_str_list(
    spec: dict[str, Any],
    key: str,
    table_id: str,
    comparison_id: str | None,
    issues: list[dict[str, Any]],
    *,
    required: bool = False,
) -> list[str] | None:
    value = spec.get(key)
    if value is None:
        if required:
            issues.append(
                _comparison_registry_issue(table_id, comparison_id, f"missing_{key}", f"comparison requires {key}")
            )
        return None
    if not isinstance(value, list) or not value:
        issues.append(_comparison_registry_issue(table_id, comparison_id, f"invalid_{key}", f"{key} must be a list"))
        return None
    result = []
    for item in value:
        if not isinstance(item, str) or not item:
            issues.append(
                _comparison_registry_issue(
                    table_id,
                    comparison_id,
                    f"invalid_{key}",
                    f"{key} entries must be non-empty strings",
                )
            )
            continue
        result.append(item)
    return result


def _comparison_float(
    spec: dict[str, Any],
    key: str,
    table_id: str,
    comparison_id: str | None,
    issues: list[dict[str, Any]],
    *,
    default: float,
) -> float:
    value = spec.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        issues.append(_comparison_registry_issue(table_id, comparison_id, f"invalid_{key}", f"{key} must be numeric"))
        return default


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


def write_ellmtrees_variants_table_json(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    registry: str | Path | None = None,
) -> None:
    """Write old ELLMTrees variants reference rows as JSON."""

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


def write_ellmtrees_variants_table_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write old ELLMTrees variants reference rows as CSV with stable columns."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ELLMTREES_VARIANTS_TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in ELLMTREES_VARIANTS_TABLE_COLUMNS})


def validate_table_registry(
    registry_path: str | Path,
    *,
    base_dir: str | Path = ".",
    require_outputs: bool = False,
) -> dict[str, Any]:
    """Validate paper table registry inputs and optional generated artifacts."""

    registry = load_table_registry(registry_path)
    base = Path(base_dir)
    entries = []
    issues = []
    for table in registry["tables"]:
        if not isinstance(table, dict):
            issues.append(_issue(None, "invalid_table_entry", "table entries must be mappings"))
            continue
        table_id = table.get("id")
        if not isinstance(table_id, str) or not table_id:
            issues.append(_issue(None, "missing_table_id", "table entry requires a non-empty id"))
            continue
        table_report = _validate_table_entry(table, base, require_outputs=require_outputs)
        table_issues = table_report["issues"]
        entries.append(
            {
                "id": table_id,
                "status": table.get("status"),
                "expected_rows": table.get("expected_rows"),
                "source_inputs": table_report["source_inputs"],
                "outputs": table_report["outputs"],
                "n_issues": len(table_issues),
                "valid": not table_issues,
                "issues": table_issues,
            }
        )
        issues.extend(table_issues)

    return {
        "registry": str(registry_path),
        "base_dir": str(base_dir),
        "require_outputs": require_outputs,
        "n_tables": len(entries),
        "n_issues": len(issues),
        "valid": not issues,
        "tables": entries,
        "issues": issues,
    }


def validate_reference_registry(
    registry_path: str | Path,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Validate pinned paper reference paths, draft labels, and optional SHA-256 digests."""

    registry = load_reference_registry(registry_path)
    base = Path(base_dir)
    entries = []
    issues = []
    for entry in registry["entries"]:
        if not isinstance(entry, dict):
            issue = _reference_issue(None, "invalid_entry", "reference entries must be mappings")
            issues.append(issue)
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            issue = _reference_issue(None, "missing_entry_id", "reference entry requires a non-empty id")
            issues.append(issue)
            continue
        entry_report = _validate_reference_entry(entry, base)
        entry_issues = entry_report["issues"]
        entries.append(
            {
                "id": entry_id,
                "kind": entry.get("kind"),
                "classification": entry.get("classification"),
                "status": entry.get("status"),
                "path": entry_report["path"],
                "source_inputs": entry_report["source_inputs"],
                "n_issues": len(entry_issues),
                "valid": not entry_issues,
                "issues": entry_issues,
            }
        )
        issues.extend(entry_issues)

    draft_label_coverage = _validate_draft_label_coverage(registry, entries, base, issues)

    return {
        "registry": str(registry_path),
        "base_dir": str(base_dir),
        "n_entries": len(entries),
        "n_issues": len(issues),
        "valid": not issues,
        "draft_label_coverage": draft_label_coverage,
        "entries": entries,
        "issues": issues,
    }


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


def _validate_table_entry(
    table: dict[str, Any],
    base_dir: Path,
    *,
    require_outputs: bool,
) -> dict[str, Any]:
    table_id = str(table["id"])
    issues = []
    source_input_report = []
    output_report = []
    for key in ("title", "source_command", "paper_location", "verification_status"):
        if not isinstance(table.get(key), str) or not table.get(key):
            issues.append(_issue(table_id, f"missing_{key}", f"table {table_id} requires {key}"))
    if not isinstance(table.get("expected_rows"), int):
        issues.append(_issue(table_id, "missing_expected_rows", "expected_rows must be an integer"))

    for path in _path_list(table, "source_inputs", table_id, issues):
        exists = _resolve_registered_path(base_dir, path).exists()
        source_input_report.append({"path": path, "exists": exists})
        if not exists:
            issues.append(_issue(table_id, "missing_source_input", f"source input does not exist: {path}"))

    outputs = _output_specs(table, table_id, issues, required=require_outputs)
    for spec in outputs:
        path = spec["path"]
        resolved = _resolve_registered_path(base_dir, path)
        exists = resolved.exists()
        observed_rows = _observed_table_rows(resolved) if exists else None
        observed_sha256 = _sha256_file(resolved) if exists else None
        output_report.append(
            {
                "path": path,
                "exists": exists,
                "observed_rows": observed_rows,
                "expected_sha256": spec.get("sha256"),
                "observed_sha256": observed_sha256,
            }
        )
        if not exists:
            if require_outputs:
                issues.append(_issue(table_id, "missing_output", f"output does not exist: {path}"))
            continue
        expected_rows = table.get("expected_rows")
        if isinstance(expected_rows, int):
            if observed_rows is not None and observed_rows != expected_rows:
                issues.append(
                    _issue(
                        table_id,
                        "row_count_mismatch",
                        f"{path} has {observed_rows} rows, expected {expected_rows}",
                    )
                )
        expected_sha256 = spec.get("sha256")
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            issues.append(
                _issue(
                    table_id,
                    "sha256_mismatch",
                    f"{path} has sha256 {observed_sha256}, expected {expected_sha256}",
                )
            )
    return {"issues": issues, "source_inputs": source_input_report, "outputs": output_report}


def _load_csv_by_key(path: Path, *, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row[key]: row for row in rows if row.get(key)}


def _load_csv_groups(path: Path, *, key: str) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    groups: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            group_key = row.get(key)
            if group_key:
                groups.setdefault(group_key, []).append(row)
    return groups


def _branch_ordering_stats(path: Path) -> dict[str, Any]:
    rows_by_run = _load_csv_groups(path, key="run_id")
    rank_biserials = []
    within_r = []
    n_pairs = 0
    for rows in rows_by_run.values():
        same = []
        cross = []
        labels = []
        distances = []
        for row in rows:
            if "cosine_dist" not in row:
                continue
            label = int(row["same_branch"])
            distance = float(row["cosine_dist"])
            labels.append(float(label))
            distances.append(distance)
            if label:
                same.append(distance)
            else:
                cross.append(distance)
        n_pairs += len(distances)
        if not same or not cross:
            continue
        rank_biserials.append(_rank_biserial_cross_greater(cross, same))
        r = _pearson(labels, distances)
        if r is not None and math.isfinite(r):
            within_r.append(max(-0.999, min(0.999, r)))

    return {
        "n_runs": len(rank_biserials),
        "n_pairs": n_pairs,
        "rank_biserial": _mean_or_none(rank_biserials),
        "rank_biserial_se": _se_or_none(rank_biserials),
        "within_run_r": _fisher_z_mean_or_none(within_r),
        "within_run_r_se": _se_or_none(within_r),
    }


def _recovery_group_stats(
    aggregate_row: dict[str, str],
    per_run_rows: list[dict[str, str]],
) -> dict[str, Any]:
    n_runs = int(float(aggregate_row["n_runs"]))
    if per_run_rows:
        clade_values = [float(row["clade_recovery"]) * 100.0 for row in per_run_rows]
        exact_values = [float(row["refinement"]) for row in per_run_rows]
        rf_values = [float(row["RF"]) for row in per_run_rows]
        fn_values = [float(row["FN"]) for row in per_run_rows]
        exact_rate = _mean_or_none(exact_values)
        return {
            "n_runs": len(per_run_rows),
            "clade_recovery_pct": _mean_or_none(clade_values),
            "clade_recovery_se_pct": _se_or_none(clade_values),
            "exact_recovery_pct": exact_rate * 100.0 if exact_rate is not None else None,
            "exact_recovery_se_pct": _binomial_se_pct(exact_rate, len(per_run_rows)),
            "rf_mean": _mean_or_none(rf_values),
            "rf_se": _se_or_none(rf_values),
            "fn_mean": _mean_or_none(fn_values),
            "fn_se": _se_or_none(fn_values),
        }
    exact_rate = float(aggregate_row["pct_refinement_FN0"]) / 100.0
    return {
        "n_runs": n_runs,
        "clade_recovery_pct": float(aggregate_row["mean_clade_recovery"]),
        "clade_recovery_se_pct": None,
        "exact_recovery_pct": float(aggregate_row["pct_refinement_FN0"]),
        "exact_recovery_se_pct": _binomial_se_pct(exact_rate, n_runs),
        "rf_mean": float(aggregate_row["mean_RF"]),
        "rf_se": None,
        "fn_mean": float(aggregate_row["mean_FN"]),
        "fn_se": None,
    }


def _rank_biserial_cross_greater(cross: list[float], same: list[float]) -> float:
    concordant = 0.0
    for cross_distance in cross:
        for same_distance in same:
            if cross_distance > same_distance:
                concordant += 1.0
            elif cross_distance == same_distance:
                concordant += 0.5
    return (2.0 * concordant) / (len(cross) * len(same)) - 1.0


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    x_ss = sum(value * value for value in x_centered)
    y_ss = sum(value * value for value in y_centered)
    if x_ss == 0.0 or y_ss == 0.0:
        return None
    return sum(a * b for a, b in zip(x_centered, y_centered)) / math.sqrt(x_ss * y_ss)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _se_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _fisher_z_mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return math.tanh(statistics.fmean(math.atanh(value) for value in values))


def _binomial_se_pct(rate: float | None, n: int) -> float | None:
    if rate is None or n <= 0:
        return None
    return math.sqrt(rate * (1.0 - rate) / n) * 100.0


def _validate_reference_entry(entry: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    entry_id = str(entry["id"])
    issues = []
    for key in ("kind", "classification", "status"):
        if not isinstance(entry.get(key), str) or not entry.get(key):
            issues.append(_reference_issue(entry_id, f"missing_{key}", f"entry {entry_id} requires {key}"))

    path_report = None
    if "path" in entry or "sha256" in entry:
        path_report = _validate_reference_path_spec(
            {"path": entry.get("path"), "sha256": entry.get("sha256")},
            base_dir,
            entry_id,
            issues,
            relation="path",
        )

    source_input_report = []
    raw_inputs = entry.get("source_inputs", [])
    if raw_inputs is None:
        raw_inputs = []
    if not isinstance(raw_inputs, list):
        issues.append(_reference_issue(entry_id, "invalid_source_inputs", "source_inputs must be a list"))
    else:
        for raw_input in raw_inputs:
            report = _validate_reference_path_spec(
                raw_input,
                base_dir,
                entry_id,
                issues,
                relation="source_input",
            )
            if report is not None:
                source_input_report.append(report)

    return {"issues": issues, "path": path_report, "source_inputs": source_input_report}


def _validate_reference_path_spec(
    raw_spec: Any,
    base_dir: Path,
    entry_id: str,
    issues: list[dict[str, Any]],
    *,
    relation: str,
) -> dict[str, Any] | None:
    if isinstance(raw_spec, str):
        spec: dict[str, Any] = {"path": raw_spec}
    elif isinstance(raw_spec, dict):
        spec = raw_spec
    else:
        issues.append(
            _reference_issue(entry_id, f"invalid_{relation}", f"{relation} must be a path or mapping")
        )
        return None

    path_value = spec.get("path")
    if not isinstance(path_value, str) or not path_value:
        issues.append(_reference_issue(entry_id, f"missing_{relation}_path", f"{relation} requires path"))
        return None
    sha256 = spec.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
        issues.append(_reference_issue(entry_id, "invalid_sha256", f"invalid sha256 for {path_value}"))
        sha256 = None

    resolved = _resolve_registered_path(base_dir, path_value)
    exists = resolved.exists()
    observed_sha256 = _sha256_file(resolved) if exists else None
    report = {
        "path": path_value,
        "role": spec.get("role"),
        "exists": exists,
        "expected_sha256": sha256,
        "observed_sha256": observed_sha256,
    }
    if not exists:
        issues.append(_reference_issue(entry_id, f"missing_{relation}", f"{relation} does not exist: {path_value}"))
    elif sha256 is not None and observed_sha256 != sha256:
        issues.append(
            _reference_issue(
                entry_id,
                "sha256_mismatch",
                f"{relation} {path_value} has sha256 {observed_sha256}, expected {sha256}",
            )
        )
    return report


def _validate_draft_label_coverage(
    registry: dict[str, Any],
    entries: list[dict[str, Any]],
    base_dir: Path,
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    active_draft = registry.get("active_draft")
    if active_draft is None:
        return None
    if not isinstance(active_draft, str) or not active_draft:
        issues.append(_reference_issue(None, "invalid_active_draft", "active_draft must be a path string"))
        return {
            "active_draft": active_draft,
            "exists": False,
            "labels": [],
            "registered_labels": [],
            "missing_labels": [],
            "stale_registered_labels": [],
        }

    draft_path = _resolve_registered_path(base_dir, active_draft)
    exists = draft_path.exists()
    label_ids = sorted(
        str(entry["id"])
        for entry in entries
        if isinstance(entry.get("id"), str) and re.match(r"^(?:fig|tab):", str(entry["id"]))
    )
    report = {
        "active_draft": active_draft,
        "exists": exists,
        "labels": [],
        "registered_labels": label_ids,
        "missing_labels": [],
        "stale_registered_labels": [],
    }
    if not exists:
        issues.append(_reference_issue(None, "missing_active_draft", f"active_draft does not exist: {active_draft}"))
        return report

    labels = sorted(set(_DRAFT_LABEL_RE.findall(draft_path.read_text())))
    label_set = set(labels)
    registered_set = set(label_ids)
    missing = [label for label in labels if label not in registered_set]
    stale = [label for label in label_ids if label not in label_set]
    report.update(
        {
            "labels": labels,
            "registered_labels": label_ids,
            "missing_labels": missing,
            "stale_registered_labels": stale,
        }
    )
    for label in missing:
        issues.append(
            _reference_issue(label, "missing_draft_label_entry", f"draft label {label} has no registry entry")
        )
    for label in stale:
        issues.append(
            _reference_issue(label, "stale_draft_label_entry", f"registry entry {label} is not in active_draft")
        )
    return report


def _path_list(
    table: dict[str, Any],
    key: str,
    table_id: str,
    issues: list[dict[str, Any]],
    *,
    required: bool = True,
) -> list[str]:
    raw = table.get(key)
    if raw is None and not required:
        return []
    if not isinstance(raw, list) or not raw:
        issues.append(_issue(table_id, f"missing_{key}", f"table {table_id} requires {key}"))
        return []
    paths = []
    for value in raw:
        if not isinstance(value, str) or not value:
            issues.append(_issue(table_id, f"invalid_{key}", f"{key} entries must be paths"))
            continue
        paths.append(value)
    return paths


def _output_specs(
    table: dict[str, Any],
    table_id: str,
    issues: list[dict[str, Any]],
    *,
    required: bool,
) -> list[dict[str, str | None]]:
    raw = table.get("outputs")
    if raw is None and not required:
        return []
    if not isinstance(raw, list) or not raw:
        issues.append(_issue(table_id, "missing_outputs", f"table {table_id} requires outputs"))
        return []
    specs = []
    for value in raw:
        if isinstance(value, str) and value:
            specs.append({"path": value, "sha256": None})
            continue
        if isinstance(value, dict):
            path = value.get("path")
            sha256 = value.get("sha256")
            if not isinstance(path, str) or not path:
                issues.append(_issue(table_id, "invalid_outputs", "output mapping requires path"))
                continue
            if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
                issues.append(_issue(table_id, "invalid_sha256", f"invalid sha256 for output {path}"))
                continue
            specs.append({"path": path, "sha256": sha256})
            continue
        issues.append(_issue(table_id, "invalid_outputs", "outputs entries must be paths or mappings"))
    return specs


def _observed_table_rows(path: Path) -> int | None:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("n_rows"), int):
            return int(payload["n_rows"])
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return len(payload["rows"])
        if isinstance(payload, list):
            return len(payload)
        return None
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    return None


def _load_table_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"table artifact does not exist: {path}")
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"JSON table artifact must contain rows: {path}")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"JSON table rows must be objects: {path}")
        return [dict(row) for row in rows]
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"unsupported table artifact suffix: {path}")


def _index_rows(
    rows: list[dict[str, Any]],
    key_columns: list[str],
    side: str,
    issues: list[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        missing = [column for column in key_columns if column not in row]
        if missing:
            issues.append(
                {
                    "code": "missing_key_column",
                    "side": side,
                    "row_number": row_number,
                    "key": None,
                    "message": f"{side} row {row_number} is missing key columns: {missing}",
                }
            )
            continue
        key = tuple(_comparable_value(row[column]) for column in key_columns)
        if key in indexed:
            issues.append(
                {
                    "code": "duplicate_key",
                    "side": side,
                    "row_number": row_number,
                    "key": list(key),
                    "message": f"{side} has duplicate key {key}",
                }
            )
            continue
        indexed[key] = row
    return indexed


def _comparison_issue(
    code: str,
    key: tuple[str, ...],
    message: str,
    *,
    column: str | None = None,
    reference_value: Any = None,
    candidate_value: Any = None,
) -> dict[str, Any]:
    issue = {"code": code, "key": list(key), "message": message}
    if column is not None:
        issue["column"] = column
    if reference_value is not None:
        issue["reference_value"] = reference_value
    if candidate_value is not None:
        issue["candidate_value"] = candidate_value
    return issue


def _numeric_close(reference_value: Any, candidate_value: Any, *, atol: float, rtol: float) -> bool:
    try:
        reference_float = float(reference_value)
        candidate_float = float(candidate_value)
    except (TypeError, ValueError):
        return False
    if math.isnan(reference_float) and math.isnan(candidate_float):
        return True
    return math.isclose(reference_float, candidate_float, abs_tol=atol, rel_tol=rtol)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(table_id: str | None, code: str, message: str) -> dict[str, Any]:
    return {"table_id": table_id, "code": code, "message": message}


def _reference_issue(entry_id: str | None, code: str, message: str) -> dict[str, Any]:
    return {"entry_id": entry_id, "code": code, "message": message}


def _comparison_registry_issue(
    table_id: str | None,
    comparison_id: str | None,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {"table_id": table_id, "comparison_id": comparison_id, "code": code, "message": message}


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


def _comparable_value(value: Any) -> str:
    return _csv_value(value)
