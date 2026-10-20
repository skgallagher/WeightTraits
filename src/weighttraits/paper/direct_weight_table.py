"""Fail-closed aggregation for the corrected direct-weight Table 2 results.

This module deliberately recomputes every estimator from the per-tree rows in a
native WeightTraits run-set rollup.  It does not trust the rollup's precomputed
``aggregate_by_metric`` block.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

import yaml

from weighttraits.paper.analysis_contracts import validate_strict_training_completion


DIRECT_WEIGHT_TABLE_SCHEMA = "weighttraits.direct_weight_table2.v1"
TREE_PREFIX = "confirm_paper_tree_"
ALL_TREE_IDS = tuple(f"{TREE_PREFIX}{index:03d}" for index in range(1, 51))
TOPOLOGY_EXCLUDED_TREE_IDS = frozenset(
    f"{TREE_PREFIX}{index:03d}" for index in (15, 20, 37, 47)
)
TOPOLOGY_TREE_IDS = tuple(
    tree_id for tree_id in ALL_TREE_IDS if tree_id not in TOPOLOGY_EXCLUDED_TREE_IDS
)
DIRECT_WEIGHT_TABLE_COLUMNS = (
    "cohort_id",
    "architecture",
    "training",
    "artifact",
    "representation",
    "metric",
    "cohort_contract",
    "cohort_contract_sha256",
    "completion_receipt",
    "completion_receipt_sha256",
    "source",
    "source_sha256",
    "truth_hashes_sha256",
    "n_recovery",
    "n_ordering",
    "clade_recovery",
    "clade_recovery_se",
    "paer",
    "paer_se",
    "rf",
    "rf_se",
    "false_negative",
    "false_negative_se",
    "rank_biserial",
    "rank_biserial_se",
    "within_run_r",
    "within_run_r_se",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def build_direct_weight_table(
    config_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate pinned inputs and build the corrected direct-weight table.

    Relative paths are resolved against ``base_dir`` when supplied, otherwise
    against the directory containing ``config_path``.  All referenced files
    must exist and match a declared SHA-256 digest.
    """

    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, Mapping):
        raise ValueError(f"direct-weight config must be a mapping: {config_file}")
    if config.get("version") != 1:
        raise ValueError("direct-weight config requires version: 1")

    metric = _required_text(config, "metric", context="direct-weight config")
    declared_ordering_ids = _load_ordering_tree_ids(config)
    base = (
        Path(base_dir).resolve()
        if base_dir is not None
        else config_file.parent
    )
    truth_specs = _load_truth_specs(config, base=base)
    truth_hashes = {
        tree_id: spec["sha256"] for tree_id, spec in sorted(truth_specs.items())
    }
    truth_hashes_sha256 = _canonical_sha256(truth_hashes)

    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("direct-weight config requires a non-empty conditions list")
    required_cohort_ids = _load_required_cohort_ids(config)

    rows: list[dict[str, Any]] = []
    seen_cohorts: set[str] = set()
    observed_ordering_sets: dict[str, frozenset[str]] = {}
    for index, condition in enumerate(conditions, start=1):
        if not isinstance(condition, Mapping):
            raise ValueError(f"condition {index} must be a mapping")
        cohort_id = _required_text(condition, "cohort_id", context=f"condition {index}")
        if cohort_id in seen_cohorts:
            raise ValueError(f"duplicate cohort_id {cohort_id!r}")
        seen_cohorts.add(cohort_id)

        row, ordering_ids = _aggregate_condition(
            condition,
            cohort_id=cohort_id,
            metric=metric,
            base=base,
            truth_specs=truth_specs,
            truth_hashes_sha256=truth_hashes_sha256,
        )
        rows.append(row)
        observed_ordering_sets[cohort_id] = ordering_ids

        if ordering_ids != declared_ordering_ids:
            raise ValueError(
                f"{cohort_id} ordering-valid tree IDs differ from declared "
                f"ordering_tree_ids: missing="
                f"{sorted(declared_ordering_ids - ordering_ids)}, unexpected="
                f"{sorted(ordering_ids - declared_ordering_ids)}"
            )

    if seen_cohorts != required_cohort_ids:
        raise ValueError(
            "direct-weight conditions differ from required_cohort_ids: "
            f"missing={sorted(required_cohort_ids - seen_cohorts)}, "
            f"unexpected={sorted(seen_cohorts - required_cohort_ids)}"
        )

    ordering_sets = set(observed_ordering_sets.values())
    if len(ordering_sets) != 1:
        rendered = {
            cohort_id: sorted(tree_ids)
            for cohort_id, tree_ids in observed_ordering_sets.items()
        }
        raise ValueError(f"ordering-valid tree IDs differ across conditions: {rendered}")
    common_ordering_ids = next(iter(ordering_sets))

    return {
        "schema": DIRECT_WEIGHT_TABLE_SCHEMA,
        "producer": "weighttraits",
        "config": str(config_file),
        "config_sha256": _sha256(config_file),
        "metric": metric,
        "estimator_contract": {
            "clade_recovery": "arithmetic mean with sample SE across topology trees",
            "rf": "arithmetic mean with sample SE across topology trees",
            "false_negative": "arithmetic mean with sample SE across topology trees",
            "paer": "binary mean with binomial sqrt(p*(1-p)/n) SE",
            "rank_biserial": "arithmetic mean with sample SE across ordering trees",
            "within_run_r": (
                "tanh(mean(atanh(raw per-tree r))) with sample SE of raw per-tree r"
            ),
        },
        "topology_tree_ids": list(TOPOLOGY_TREE_IDS),
        "topology_excluded_tree_ids": sorted(TOPOLOGY_EXCLUDED_TREE_IDS),
        "ordering_tree_ids": sorted(common_ordering_ids),
        "required_cohort_ids": sorted(required_cohort_ids),
        "truth_manifests": {
            tree_id: {
                "path": str(truth_specs[tree_id]["path"]),
                "sha256": truth_specs[tree_id]["sha256"],
            }
            for tree_id in TOPOLOGY_TREE_IDS
        },
        "truth_manifest_sha256_by_tree": truth_hashes,
        "truth_hashes_sha256": truth_hashes_sha256,
        "n_rows": len(rows),
        "rows": rows,
    }


def write_direct_weight_table_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write a provenance-bearing direct-weight table JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_direct_weight_table_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> None:
    """Write the flat Table 2 rows in a stable column order."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DIRECT_WEIGHT_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in DIRECT_WEIGHT_TABLE_COLUMNS})


def _load_truth_specs(
    config: Mapping[str, Any],
    *,
    base: Path,
) -> dict[str, dict[str, Any]]:
    raw_specs = config.get("truth_manifests")
    if not isinstance(raw_specs, Mapping):
        raise ValueError("direct-weight config requires a truth_manifests mapping")
    expected = set(TOPOLOGY_TREE_IDS)
    observed = {str(tree_id) for tree_id in raw_specs}
    if observed != expected:
        raise ValueError(
            "truth_manifests must declare exactly the 46 topology trees; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    specs: dict[str, dict[str, Any]] = {}
    for tree_id in TOPOLOGY_TREE_IDS:
        raw = raw_specs[tree_id]
        path, digest = _pinned_file(raw, label=f"truth manifest {tree_id}", base=base)
        _validate_truth_manifest(path, expected_tree_id=tree_id)
        specs[tree_id] = {"path": path, "sha256": digest}
    return specs


def _load_ordering_tree_ids(config: Mapping[str, Any]) -> frozenset[str]:
    raw_ids = config.get("ordering_tree_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("direct-weight config requires an ordering_tree_ids list")
    if any(not isinstance(tree_id, str) for tree_id in raw_ids):
        raise ValueError("ordering_tree_ids entries must be strings")
    observed = frozenset(raw_ids)
    if len(raw_ids) != len(observed):
        raise ValueError("ordering_tree_ids must not contain duplicates")
    if len(observed) != 26:
        raise ValueError(
            f"ordering_tree_ids must declare exactly 26 trees, got {len(observed)}"
        )
    unexpected = observed - set(TOPOLOGY_TREE_IDS)
    if unexpected:
        raise ValueError(
            "ordering_tree_ids must be a subset of the fixed 46 topology trees; "
            f"unexpected={sorted(unexpected)}"
        )
    return observed


def _load_required_cohort_ids(config: Mapping[str, Any]) -> set[str]:
    raw = config.get("required_cohort_ids")
    if not isinstance(raw, list) or not raw:
        raise ValueError("direct-weight config requires non-empty required_cohort_ids")
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ValueError("required_cohort_ids entries must be nonempty strings")
    observed = {value.strip() for value in raw}
    if len(observed) != len(raw):
        raise ValueError("required_cohort_ids must not contain duplicates")
    return observed


def _aggregate_condition(
    condition: Mapping[str, Any],
    *,
    cohort_id: str,
    metric: str,
    base: Path,
    truth_specs: Mapping[str, Mapping[str, Any]],
    truth_hashes_sha256: str,
) -> tuple[dict[str, Any], frozenset[str]]:
    architecture = _required_text(condition, "architecture", context=cohort_id)
    training = _required_text(condition, "training", context=cohort_id)
    artifact = _required_text(condition, "artifact", context=cohort_id)
    representation = _required_text(condition, "representation", context=cohort_id)
    cohort_path, cohort_sha256 = _pinned_file(
        condition.get("cohort_contract"),
        label=f"{cohort_id} cohort_contract",
        base=base,
    )
    _validate_cohort_contract(
        cohort_path,
        cohort_id=cohort_id,
        truth_specs=truth_specs,
    )
    completion_path, completion_sha256 = _pinned_file(
        condition.get("completion_receipt"),
        label=f"{cohort_id} completion_receipt",
        base=base,
    )
    _validate_completion_receipt(completion_path, cohort_id=cohort_id)
    source_path, source_sha256 = _pinned_file(
        condition.get("rollup"),
        label=f"{cohort_id} rollup",
        base=base,
    )

    payload = json.loads(source_path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{cohort_id} rollup must be a JSON object: {source_path}")
    if payload.get("valid") is not True:
        raise ValueError(f"{cohort_id} rollup is not explicitly valid: {source_path}")
    if payload.get("artifact") != artifact:
        raise ValueError(
            f"{cohort_id} artifact mismatch: declared {artifact!r}, "
            f"rollup has {payload.get('artifact')!r}"
        )
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{cohort_id} rollup rows must be a list: {source_path}")

    metric_rows = [row for row in raw_rows if _row_metric(row) == metric]
    by_tree: dict[str, Mapping[str, Any]] = {}
    for row_index, row in enumerate(metric_rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{cohort_id} metric row {row_index} must be a mapping")
        tree_id = _required_text(row, "tree_id", context=f"{cohort_id} metric row")
        if tree_id in by_tree:
            raise ValueError(
                f"{cohort_id} has duplicate {metric!r} row for tree {tree_id}"
            )
        by_tree[tree_id] = row

    all_ids = set(ALL_TREE_IDS)
    observed_ids = set(by_tree)
    if observed_ids != all_ids:
        raise ValueError(
            f"{cohort_id} must contain exactly one {metric!r} row for each of 50 trees; "
            f"missing={sorted(all_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - all_ids)}"
        )

    topology_rows: list[Mapping[str, Any]] = []
    for tree_id in ALL_TREE_IDS:
        row = by_tree[tree_id]
        if row.get("artifact") != artifact:
            raise ValueError(
                f"{cohort_id}/{tree_id} artifact mismatch: declared {artifact!r}, "
                f"row has {row.get('artifact')!r}"
            )
        if row.get("representation") != representation:
            raise ValueError(
                f"{cohort_id}/{tree_id} representation mismatch: "
                f"declared {representation!r}, row has {row.get('representation')!r}"
            )
        n_truth_splits = _nonnegative_int(
            row.get("n_truth_splits"),
            label=f"{cohort_id}/{tree_id} n_truth_splits",
        )
        if tree_id in TOPOLOGY_EXCLUDED_TREE_IDS:
            if n_truth_splits != 0:
                raise ValueError(
                    f"{cohort_id}/{tree_id} is topology-excluded but has "
                    f"n_truth_splits={n_truth_splits}"
                )
            continue
        if n_truth_splits <= 0:
            raise ValueError(
                f"{cohort_id}/{tree_id} is topology-eligible but has "
                f"n_truth_splits={n_truth_splits}"
            )
        _verify_row_truth(
            row,
            cohort_id=cohort_id,
            tree_id=tree_id,
            expected=truth_specs[tree_id],
            base=base,
        )
        topology_rows.append(row)

    topology_ids = {str(row["tree_id"]) for row in topology_rows}
    if topology_ids != set(TOPOLOGY_TREE_IDS):
        raise ValueError(
            f"{cohort_id} topology IDs differ from the fixed 46-tree set"
        )

    ordering_rows = []
    for row in topology_rows:
        valid = row.get("branch_ordering_valid")
        if not isinstance(valid, bool):
            raise ValueError(
                f"{cohort_id}/{row['tree_id']} branch_ordering_valid must be boolean"
            )
        if valid:
            if row.get("branch_ordering_status") != "ok":
                raise ValueError(
                    f"{cohort_id}/{row['tree_id']} is ordering-valid but status is "
                    f"{row.get('branch_ordering_status')!r}"
                )
            ordering_rows.append(row)

    ordering_ids = frozenset(str(row["tree_id"]) for row in ordering_rows)
    if len(ordering_ids) != 26:
        raise ValueError(
            f"{cohort_id} must have exactly 26 ordering-valid trees, got "
            f"{len(ordering_ids)}: {sorted(ordering_ids)}"
        )

    clade = [
        _bounded_float(
            row.get("clade_recovery"),
            label=f"{cohort_id}/{row['tree_id']} clade_recovery",
            minimum=0.0,
            maximum=1.0,
        )
        for row in topology_rows
    ]
    paer = [
        _strict_bool(
            row.get("polytomy_aware_exact_recovery"),
            label=f"{cohort_id}/{row['tree_id']} polytomy_aware_exact_recovery",
        )
        for row in topology_rows
    ]
    rf = [
        _nonnegative_float(row.get("rf"), label=f"{cohort_id}/{row['tree_id']} rf")
        for row in topology_rows
    ]
    false_negative = [
        _nonnegative_float(
            row.get("false_negative"),
            label=f"{cohort_id}/{row['tree_id']} false_negative",
        )
        for row in topology_rows
    ]
    rank_biserial = [
        _bounded_float(
            row.get("branch_rank_biserial"),
            label=f"{cohort_id}/{row['tree_id']} branch_rank_biserial",
            minimum=-1.0,
            maximum=1.0,
        )
        for row in ordering_rows
    ]
    within_run_r = [
        _open_unit_float(
            row.get("branch_within_run_r"),
            label=f"{cohort_id}/{row['tree_id']} branch_within_run_r",
        )
        for row in ordering_rows
    ]

    clade_mean, clade_se = _mean_sample_se(clade)
    rf_mean, rf_se = _mean_sample_se(rf)
    fn_mean, fn_se = _mean_sample_se(false_negative)
    rank_mean, rank_se = _mean_sample_se(rank_biserial)
    raw_r_se = _sample_se(within_run_r)
    paer_rate = statistics.fmean(float(value) for value in paer)
    paer_se = math.sqrt(paer_rate * (1.0 - paer_rate) / len(paer))
    fisher_r_mean = math.tanh(
        statistics.fmean(math.atanh(value) for value in within_run_r)
    )

    return (
        {
            "cohort_id": cohort_id,
            "architecture": architecture,
            "training": training,
            "artifact": artifact,
            "representation": representation,
            "metric": metric,
            "cohort_contract": str(cohort_path),
            "cohort_contract_sha256": cohort_sha256,
            "completion_receipt": str(completion_path),
            "completion_receipt_sha256": completion_sha256,
            "source": str(source_path),
            "source_sha256": source_sha256,
            "truth_hashes_sha256": truth_hashes_sha256,
            "n_recovery": len(topology_rows),
            "n_ordering": len(ordering_rows),
            "clade_recovery": clade_mean,
            "clade_recovery_se": clade_se,
            "paer": paer_rate,
            "paer_se": paer_se,
            "rf": rf_mean,
            "rf_se": rf_se,
            "false_negative": fn_mean,
            "false_negative_se": fn_se,
            "rank_biserial": rank_mean,
            "rank_biserial_se": rank_se,
            "within_run_r": fisher_r_mean,
            "within_run_r_se": raw_r_se,
        },
        ordering_ids,
    )


def _verify_row_truth(
    row: Mapping[str, Any],
    *,
    cohort_id: str,
    tree_id: str,
    expected: Mapping[str, Any],
    base: Path,
) -> None:
    raw_path = row.get("truth_manifest")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{cohort_id}/{tree_id} requires a truth_manifest path")
    path = _resolve_path(raw_path, base=base)
    expected_path = Path(expected["path"]).resolve()
    if path != expected_path:
        raise ValueError(
            f"{cohort_id}/{tree_id} truth_manifest path mismatch: "
            f"expected {expected_path}, got {path}"
        )
    if not path.is_file():
        raise ValueError(
            f"{cohort_id}/{tree_id} truth_manifest does not exist: {path}"
        )
    observed = _sha256(path)
    if observed != expected["sha256"]:
        raise ValueError(
            f"{cohort_id}/{tree_id} truth_manifest sha256 mismatch: "
            f"expected {expected['sha256']}, got {observed} ({path})"
        )
    _validate_truth_manifest(path, expected_tree_id=tree_id)


def _validate_completion_receipt(path: Path, *, cohort_id: str) -> None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{cohort_id} cohort receipt is invalid JSON: {path}") from error
    try:
        validate_strict_training_completion(payload, cohort_id=cohort_id)
    except ValueError as error:
        raise ValueError(
            f"{cohort_id} completion receipt is not a strict training-completion "
            f"receipt: {path}: {error}"
        ) from error


def _validate_cohort_contract(
    path: Path,
    *,
    cohort_id: str,
    truth_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{cohort_id} cohort contract is invalid JSON: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("valid") is not True:
        raise ValueError(f"{cohort_id} cohort contract is not explicitly valid: {path}")
    expected = {"n_trees": 50, "n_runs": 641, "n_errors": 0, "n_warnings": 0}
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"{cohort_id} cohort contract requires {field}={value}, "
                f"got {payload.get(field)!r}: {path}"
            )
    config_name = Path(str(payload.get("config", ""))).name
    if config_name != f"{cohort_id}.yaml":
        raise ValueError(
            f"{cohort_id} cohort contract config basename must be "
            f"{cohort_id}.yaml, got {config_name!r}: {path}"
        )
    trees = payload.get("trees")
    if not isinstance(trees, list):
        raise ValueError(f"{cohort_id} cohort contract trees must be a list: {path}")
    by_tree: dict[str, Mapping[str, Any]] = {}
    for row in trees:
        if not isinstance(row, Mapping):
            raise ValueError(f"{cohort_id} cohort contract tree rows must be mappings")
        tree_id = str(row.get("tree_id", ""))
        if tree_id in by_tree:
            raise ValueError(f"{cohort_id} cohort contract duplicates {tree_id!r}")
        by_tree[tree_id] = row
    if set(by_tree) != set(ALL_TREE_IDS):
        raise ValueError(
            f"{cohort_id} cohort contract must contain exact tree IDs 001..050"
        )
    if sum(int(row.get("n_runs", -1)) for row in by_tree.values()) != 641:
        raise ValueError(f"{cohort_id} cohort contract per-tree runs do not total 641")
    for tree_id, row in by_tree.items():
        if row.get("valid") is not True or row.get("n_errors") != 0 or row.get(
            "n_warnings"
        ) != 0:
            raise ValueError(f"{cohort_id}/{tree_id} cohort contract is not clean")
        if tree_id in truth_specs and row.get("manifest_sha256") != truth_specs[tree_id][
            "sha256"
        ]:
            raise ValueError(f"{cohort_id}/{tree_id} cohort truth hash mismatch")


def _validate_truth_manifest(path: Path, *, expected_tree_id: str) -> None:
    saw_row = False
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        saw_row = True
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(row, Mapping):
            raise ValueError(f"truth manifest row must be an object: {path}:{line_number}")
        if row.get("tree_id") != expected_tree_id:
            raise ValueError(
                f"truth manifest tree_id mismatch at {path}:{line_number}: "
                f"expected {expected_tree_id!r}, got {row.get('tree_id')!r}"
            )
    if not saw_row:
        raise ValueError(f"truth manifest is empty: {path}")


def _pinned_file(
    raw: Any,
    *,
    label: str,
    base: Path,
) -> tuple[Path, str]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must declare path and sha256")
    path_text = _required_text(raw, "path", context=label)
    digest = _required_sha256(raw, "sha256", context=label)
    path = _resolve_path(path_text, base=base)
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    observed = _sha256(path)
    if observed != digest:
        raise ValueError(
            f"{label} sha256 mismatch: expected {digest}, got {observed} ({path})"
        )
    return path, digest


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _row_metric(row: Any) -> str | None:
    if not isinstance(row, Mapping):
        return None
    value = row.get("metric")
    return value if isinstance(value, str) else None


def _required_text(value: Mapping[str, Any], key: str, *, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context} requires non-empty {key}")
    return raw.strip()


def _required_sha256(value: Mapping[str, Any], key: str, *, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or _SHA256_PATTERN.fullmatch(raw) is None:
        raise ValueError(f"{context} requires lowercase 64-hex {key}")
    return raw


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean, got {value!r}")
    return value


def _bounded_float(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    number = _finite_float(value, label=label)
    if not minimum <= number <= maximum:
        raise ValueError(
            f"{label} must be in [{minimum}, {maximum}], got {number!r}"
        )
    return number


def _open_unit_float(value: Any, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if not -1.0 < number < 1.0:
        raise ValueError(f"{label} must be strictly between -1 and 1, got {number!r}")
    return number


def _nonnegative_float(value: Any, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative, got {number!r}")
    return number


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _mean_sample_se(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), _sample_se(values)


def _sample_se(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample SE requires at least two values")
    return statistics.stdev(values) / math.sqrt(len(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
