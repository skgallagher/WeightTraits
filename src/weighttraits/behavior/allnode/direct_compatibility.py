"""Leaf-only NJ/recovery compatibility for the all-node direct branch.

This module is deliberately downstream of the primary all-node matrices.  It
may only reorder and slice the already-derived trained-leaf matrices; trained
internal ancestors are never accepted as neighbor-joining tips.  A normalized
preflight baseline binds every old matrix/model order/summary/truth artifact,
the source rollup, and the old table source before any comparison is made.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from .checkpoints import ArtifactPin, PRODUCTION_N_TREES
from .direct_execution import DIRECT_METRICS
from .manifest import canonical_sha256
from .production import PRODUCTION_TREE_IDS


BASELINE_SCHEMA = "weighttraits.all_node_leaf_compatibility_baseline.v1"
RECEIPT_SCHEMA = "weighttraits.all_node_leaf_compatibility_receipt.v1"
SPEC_SCHEMA = "weighttraits.all_node_leaf_compatibility_spec.v1"
POPULATION = "trained_leaves"
TREATMENT = "compatibility_only_leaf_nj"
MATRIX_TOLERANCE = 1e-12
RECEIPT_BASENAME = "leaf_compatibility_receipt.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECOVERY_FIELDS = (
    "n_truth_leaves", "n_estimate_leaves", "n_truth_splits",
    "n_estimate_splits", "true_positive", "false_negative", "false_positive",
    "rf", "normalized_rf", "clade_recovery", "split_precision",
    "exact_tree_recovery", "polytomy_aware_exact_recovery",
)
_DELTA_FIELDS = (
    "n_truth_splits", "n_estimate_splits", "true_positive", "false_negative",
    "false_positive", "rf", "normalized_rf", "clade_recovery",
    "split_precision", "exact_tree_recovery_numeric",
    "polytomy_aware_exact_recovery_numeric",
)


@dataclass(frozen=True)
class LeafCompatibilitySpec:
    comparison_available: bool
    baseline_manifest: ArtifactPin | None
    source_direct_rollup: ArtifactPin | None
    source_table_input: ArtifactPin | None
    unavailable_reason: ArtifactPin | None
    reconstruct_source: ArtifactPin | None
    recovery_source: ArtifactPin | None
    splits_source: ArtifactPin | None

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": SPEC_SCHEMA,
            "comparison_available": self.comparison_available,
            "baseline_manifest": _pin_dict(self.baseline_manifest),
            "source_direct_rollup": _pin_dict(self.source_direct_rollup),
            "source_table_input": _pin_dict(self.source_table_input),
            "unavailable_reason": _pin_dict(self.unavailable_reason),
            "reconstruct_source": _pin_dict(self.reconstruct_source),
            "recovery_source": _pin_dict(self.recovery_source),
            "splits_source": _pin_dict(self.splits_source),
        }
        return {**core, "spec_sha256": canonical_sha256(core)}


@dataclass(frozen=True)
class LeafCompatibilityAPI:
    load_npy: Callable[[Path], object]
    neighbor_joining_newick: Callable[[Sequence[str], object], str]
    splits_from_manifest_path: Callable[[str], tuple[object, object]]
    splits_from_newick_text: Callable[[str], tuple[object, object]]
    score_split_recovery: Callable[[object, object, object, object], Mapping[str, Any]]


def build_leaf_compatibility_spec(
    *,
    comparison_available: bool,
    baseline_manifest: ArtifactPin | Mapping[str, Any] | None = None,
    source_direct_rollup: ArtifactPin | Mapping[str, Any] | None = None,
    source_table_input: ArtifactPin | Mapping[str, Any] | None = None,
    unavailable_reason: ArtifactPin | Mapping[str, Any] | None = None,
    reconstruct_source: ArtifactPin | Mapping[str, Any] | None = None,
    recovery_source: ArtifactPin | Mapping[str, Any] | None = None,
    splits_source: ArtifactPin | Mapping[str, Any] | None = None,
) -> LeafCompatibilitySpec:
    if not isinstance(comparison_available, bool):
        raise ValueError("comparison_available must be boolean")
    values = {
        "baseline_manifest": _optional_pin(baseline_manifest, "baseline_manifest"),
        "source_direct_rollup": _optional_pin(source_direct_rollup, "source_direct_rollup"),
        "source_table_input": _optional_pin(source_table_input, "source_table_input"),
        "unavailable_reason": _optional_pin(unavailable_reason, "unavailable_reason"),
        "reconstruct_source": _optional_pin(reconstruct_source, "reconstruct_source"),
        "recovery_source": _optional_pin(recovery_source, "recovery_source"),
        "splits_source": _optional_pin(splits_source, "splits_source"),
    }
    if comparison_available:
        required = (
            "baseline_manifest", "source_direct_rollup", "source_table_input",
            "reconstruct_source", "recovery_source", "splits_source",
        )
        if any(values[name] is None for name in required) or values["unavailable_reason"] is not None:
            raise ValueError("available leaf compatibility requires all baseline/source pins and no reason")
    else:
        if values["unavailable_reason"] is None:
            raise ValueError("unavailable leaf compatibility requires a pinned reason")
        if any(values[name] is not None for name in values if name != "unavailable_reason"):
            raise ValueError("unavailable leaf compatibility cannot carry synthetic baseline/source pins")
    return LeafCompatibilitySpec(comparison_available=comparison_available, **values)


def read_leaf_compatibility_spec(raw: object) -> LeafCompatibilitySpec:
    required = {
        "schema", "comparison_available", "baseline_manifest",
        "source_direct_rollup", "source_table_input", "unavailable_reason",
        "reconstruct_source", "recovery_source", "splits_source", "spec_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != SPEC_SCHEMA:
        raise ValueError("leaf compatibility spec schema/keys drift")
    rebuilt = build_leaf_compatibility_spec(
        comparison_available=raw.get("comparison_available"),
        baseline_manifest=raw.get("baseline_manifest"),
        source_direct_rollup=raw.get("source_direct_rollup"),
        source_table_input=raw.get("source_table_input"),
        unavailable_reason=raw.get("unavailable_reason"),
        reconstruct_source=raw.get("reconstruct_source"),
        recovery_source=raw.get("recovery_source"),
        splits_source=raw.get("splits_source"),
    )
    if canonical_sha256(dict(raw)) != canonical_sha256(rebuilt.to_dict()):
        raise ValueError("leaf compatibility spec differs from recomputed contract")
    return rebuilt


def execute_leaf_compatibility(
    spec: LeafCompatibilitySpec,
    *,
    cohort_key: str,
    direct_plan_sha256: str,
    matrix_loader: Callable[[str, str], Mapping[str, Any]],
    snapshot_root: str,
    api: LeafCompatibilityAPI | None = None,
) -> dict[str, Any]:
    """Build the deterministic compatibility receipt from primary leaf slices."""

    verified = read_leaf_compatibility_spec(spec.to_dict())
    plan_sha = _sha(direct_plan_sha256, "direct_plan_sha256")
    if not callable(matrix_loader):
        raise ValueError("matrix_loader must be callable")
    if not verified.comparison_available:
        reason = _verified_pin_bytes(verified.unavailable_reason)
        core = {
            "schema": RECEIPT_SCHEMA,
            "valid": True,
            "status": "comparison_unavailable",
            "comparison_available": False,
            "population": POPULATION,
            "internal_nodes_as_tips": False,
            "internal_node_treatment": TREATMENT,
            "compatibility_only": True,
            "cohort_key": cohort_key,
            "direct_plan_sha256": plan_sha,
            "spec": verified.to_dict(),
            "unavailable_reason_sha256": hashlib.sha256(reason).hexdigest(),
            "n_rows": 0,
            "rows": [],
            "aggregate_matched_eligibility": [],
        }
        return {**core, "leaf_compatibility_sha256": canonical_sha256(core)}

    assert verified.baseline_manifest is not None
    assert verified.source_direct_rollup is not None
    assert verified.source_table_input is not None
    source_root = _root(snapshot_root, "snapshot_root")
    for label, pin in (
        ("reconstruct_source", verified.reconstruct_source),
        ("recovery_source", verified.recovery_source),
        ("splits_source", verified.splits_source),
    ):
        assert pin is not None
        if not _strictly_below(pin.path, source_root):
            raise ValueError(f"leaf compatibility {label} escapes snapshot_root")
        _verified_pin_bytes(pin)
    _verified_pin_bytes(verified.source_direct_rollup)
    _verified_pin_bytes(verified.source_table_input)
    baseline = _load_baseline(verified, cohort_key=cohort_key)
    selected_api = api if api is not None else _load_api(verified, snapshot_root=snapshot_root)
    rows: list[dict[str, Any]] = []
    for baseline_row in baseline["rows"]:
        tree_id = baseline_row["tree_id"]
        metric = baseline_row["metric"]
        primary = matrix_loader(tree_id, metric)
        leaf_ids, leaf_matrix = _primary_leaf_matrix(primary, tree_id=tree_id, metric=metric)
        legacy_ids = _load_model_ids(_pin(baseline_row["legacy_model_ids"], "legacy_model_ids"))
        if set(legacy_ids) != set(leaf_ids) or len(legacy_ids) != len(leaf_ids):
            raise ValueError(f"{tree_id}/{metric} legacy IDs differ from exact trained leaves")
        index = {node_id: position for position, node_id in enumerate(leaf_ids)}
        reordered = [
            [leaf_matrix[index[left]][index[right]] for right in legacy_ids]
            for left in legacy_ids
        ]
        legacy_matrix_pin = _pin(baseline_row["legacy_matrix"], "legacy_matrix")
        _verified_pin_bytes(legacy_matrix_pin)
        legacy_matrix = _matrix(selected_api.load_npy(Path(legacy_matrix_pin.path)), len(legacy_ids), "legacy matrix")
        new_matrix = _matrix(reordered, len(legacy_ids), "derived leaf matrix")
        matrix_delta = _matrix_delta(new_matrix, legacy_matrix)
        truth_pin = _pin(baseline_row["truth_manifest"], "truth_manifest")
        _verified_pin_bytes(truth_pin)
        _verified_pin_bytes(_pin(baseline_row["source_summary"], "source_summary"))
        newick = selected_api.neighbor_joining_newick(legacy_ids, new_matrix)
        estimate_splits, estimate_leaves = selected_api.splits_from_newick_text(newick)
        truth_splits, truth_leaves = selected_api.splits_from_manifest_path(truth_pin.path)
        score = _recovery(
            selected_api.score_split_recovery(
                truth_splits, estimate_splits, truth_leaves, estimate_leaves
            ),
            "new recovery",
        )
        old = _recovery(
            baseline_row["legacy_recovery"], "legacy recovery", exact=True
        )
        delta = {
            field: _numeric_recovery(score, field) - _numeric_recovery(old, field)
            for field in _DELTA_FIELDS
        }
        rows.append(
            {
                "tree_id": tree_id,
                "metric": metric,
                "population": POPULATION,
                "internal_nodes_as_tips": False,
                "leaf_ids": legacy_ids,
                "leaf_ids_sha256": canonical_sha256(legacy_ids),
                "n_leaves": len(legacy_ids),
                "topology_eligible": baseline_row["topology_eligible"],
                "matched_eligible": tree_id in baseline["matched_eligible_tree_ids"],
                "primary_leaf_matrix_sha256": canonical_sha256(new_matrix),
                "legacy_matrix": legacy_matrix_pin.to_dict(),
                "legacy_model_ids": baseline_row["legacy_model_ids"],
                "source_summary": baseline_row["source_summary"],
                "truth_manifest": truth_pin.to_dict(),
                "matrix_delta": matrix_delta,
                "newick": newick,
                "newick_sha256": hashlib.sha256(newick.encode("utf-8")).hexdigest(),
                "new_recovery": score,
                "legacy_recovery": old,
                "recovery_delta": delta,
            }
        )
    aggregates = _aggregate_matched(rows, baseline["matched_eligible_tree_ids"])
    core = {
        "schema": RECEIPT_SCHEMA,
        "valid": True,
        "status": "completed",
        "comparison_available": True,
        "population": POPULATION,
        "internal_nodes_as_tips": False,
        "internal_node_treatment": TREATMENT,
        "compatibility_only": True,
        "cohort_key": cohort_key,
        "direct_plan_sha256": plan_sha,
        "spec": verified.to_dict(),
        "baseline_sha256": baseline["baseline_sha256"],
        "matched_eligible_tree_ids": baseline["matched_eligible_tree_ids"],
        "n_rows": len(rows),
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
        "aggregate_matched_eligibility": aggregates,
        "aggregate_matched_eligibility_sha256": canonical_sha256(aggregates),
        "matrix_tolerance": MATRIX_TOLERANCE,
    }
    return {**core, "leaf_compatibility_sha256": canonical_sha256(core)}


def validate_leaf_compatibility_receipt(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("leaf compatibility receipt schema drift")
    if canonical_sha256(dict(payload)) != canonical_sha256(dict(expected)):
        raise ValueError("leaf compatibility receipt differs from exact recomputation")
    core = {key: value for key, value in payload.items() if key != "leaf_compatibility_sha256"}
    if payload.get("leaf_compatibility_sha256") != canonical_sha256(core):
        raise ValueError("leaf compatibility receipt hash drift")
    return expected


def _load_baseline(spec: LeafCompatibilitySpec, *, cohort_key: str) -> Mapping[str, Any]:
    assert spec.baseline_manifest is not None
    payload = _json_bytes(_verified_pin_bytes(spec.baseline_manifest), "leaf baseline")
    required = {
        "schema", "valid", "comparison_available", "cohort_key", "population",
        "internal_nodes_as_tips", "source_direct_rollup", "source_table_input",
        "n_trees", "n_metrics", "n_rows", "tree_ids", "metrics",
        "matched_eligible_tree_ids", "rows", "rows_sha256", "baseline_sha256",
    }
    if set(payload) != required or payload.get("schema") != BASELINE_SCHEMA or payload.get("valid") is not True or payload.get("comparison_available") is not True:
        raise ValueError("leaf baseline schema/keys/validity drift")
    if payload.get("cohort_key") != cohort_key or payload.get("population") != POPULATION or payload.get("internal_nodes_as_tips") is not False:
        raise ValueError("leaf baseline cohort/population drift")
    if payload.get("source_direct_rollup") != spec.source_direct_rollup.to_dict() or payload.get("source_table_input") != spec.source_table_input.to_dict():
        raise ValueError("leaf baseline source pins differ from plan")
    if payload.get("n_trees") != PRODUCTION_N_TREES or payload.get("n_metrics") != len(DIRECT_METRICS) or payload.get("n_rows") != PRODUCTION_N_TREES * len(DIRECT_METRICS):
        raise ValueError("leaf baseline exact 50x3 cardinality drift")
    if payload.get("tree_ids") != list(PRODUCTION_TREE_IDS) or payload.get("metrics") != list(DIRECT_METRICS):
        raise ValueError("leaf baseline tree/metric scope drift")
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("rows_sha256") != canonical_sha256(rows):
        raise ValueError("leaf baseline row list/hash drift")
    expected_coordinates = [
        (tree_id, metric) for tree_id in PRODUCTION_TREE_IDS for metric in DIRECT_METRICS
    ]
    coordinates: list[tuple[str, str]] = []
    eligible_by_tree: dict[str, bool] = {}
    row_keys = {
        "tree_id", "metric", "topology_eligible", "legacy_matrix",
        "legacy_model_ids", "source_summary", "truth_manifest", "legacy_recovery",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_keys or not isinstance(row["topology_eligible"], bool):
            raise ValueError("leaf baseline row keys/types drift")
        coordinates.append((row["tree_id"], row["metric"]))
        _pin(row["legacy_matrix"], "legacy_matrix")
        _pin(row["legacy_model_ids"], "legacy_model_ids")
        _pin(row["source_summary"], "source_summary")
        _pin(row["truth_manifest"], "truth_manifest")
        _recovery(row["legacy_recovery"], "legacy recovery", exact=True)
        previous = eligible_by_tree.setdefault(row["tree_id"], row["topology_eligible"])
        if previous != row["topology_eligible"]:
            raise ValueError("leaf baseline topology eligibility differs across metrics")
    if coordinates != expected_coordinates:
        raise ValueError("leaf baseline rows must be exact tree-major 50x3 coordinates")
    matched = payload.get("matched_eligible_tree_ids")
    if not isinstance(matched, list) or len(matched) != len(set(matched)) or any(tree not in PRODUCTION_TREE_IDS for tree in matched):
        raise ValueError("leaf baseline matched eligibility set drift")
    if matched != [tree for tree in PRODUCTION_TREE_IDS if eligible_by_tree[tree]]:
        raise ValueError("leaf baseline matched IDs differ from topology eligibility")
    core = {key: value for key, value in payload.items() if key != "baseline_sha256"}
    if payload.get("baseline_sha256") != canonical_sha256(core):
        raise ValueError("leaf baseline hash drift")
    return payload


def _load_api(spec: LeafCompatibilitySpec, *, snapshot_root: str) -> LeafCompatibilityAPI:
    root = _root(snapshot_root, "snapshot_root")
    modules = {
        "reconstruct": ("weighttraits.phylo.reconstruct", spec.reconstruct_source),
        "recovery": ("weighttraits.phylo.recovery", spec.recovery_source),
        "splits": ("weighttraits.phylo.splits", spec.splits_source),
    }
    loaded = {}
    for label, (name, pin) in modules.items():
        assert pin is not None
        if not _strictly_below(pin.path, root):
            raise ValueError(f"leaf compatibility {label} source escapes snapshot_root")
        _verified_pin_bytes(pin)
        module = importlib.import_module(name)
        source_path = Path(getattr(module, "__file__", ""))
        if source_path.is_symlink() or str(source_path.resolve(strict=True)) != pin.path:
            raise ValueError(f"imported leaf compatibility {label} module path drift")
        loaded[label] = module
    import numpy as np

    return LeafCompatibilityAPI(
        load_npy=lambda path: np.load(path, allow_pickle=False),
        neighbor_joining_newick=loaded["reconstruct"].neighbor_joining_newick,
        splits_from_manifest_path=loaded["splits"].splits_from_manifest_path,
        splits_from_newick_text=loaded["splits"].splits_from_newick_text,
        score_split_recovery=loaded["recovery"].score_split_recovery,
    )


def _primary_leaf_matrix(raw: Mapping[str, Any], *, tree_id: str, metric: str) -> tuple[list[str], list[list[float]]]:
    if raw.get("schema") != "weighttraits.all_node_distance_values.v1" or raw.get("tree_id") != tree_id or raw.get("metric") != metric:
        raise ValueError("primary leaf compatibility matrix scope drift")
    ids = raw.get("leaf_sensitivity_ids")
    matrix = raw.get("leaf_sensitivity_matrix")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("primary leaf sensitivity IDs drift")
    return [_text(value, "leaf ID") for value in ids], _matrix(matrix, len(ids), "primary leaf matrix")


def _load_model_ids(pin: ArtifactPin) -> list[str]:
    raw = _json_bytes(_verified_pin_bytes(pin), "legacy model IDs")
    if isinstance(raw, Mapping):
        raw = raw.get("model_ids")
    if not isinstance(raw, list) or not raw or len(raw) != len(set(raw)):
        raise ValueError("legacy model IDs must be a nonempty unique JSON list")
    return [_text(value, "legacy model ID") for value in raw]


def _matrix(raw: object, size: int, label: str) -> list[list[float]]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list) or len(raw) != size:
        raise ValueError(f"{label} shape drift")
    out = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{label} row shape drift")
        converted = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{label} contains an invalid distance")
            converted.append(0.0 if float(value) == 0.0 else float(value))
        out.append(converted)
    for left in range(size):
        if out[left][left] != 0.0:
            raise ValueError(f"{label} diagonal drift")
        for right in range(left + 1, size):
            if abs(out[left][right] - out[right][left]) > MATRIX_TOLERANCE:
                raise ValueError(f"{label} symmetry drift")
            mean = (out[left][right] + out[right][left]) / 2.0
            out[left][right] = out[right][left] = mean
    return out


def _matrix_delta(new: Sequence[Sequence[float]], old: Sequence[Sequence[float]]) -> dict[str, Any]:
    deltas = [abs(float(left) - float(right)) for new_row, old_row in zip(new, old) for left, right in zip(new_row, old_row)]
    squared = [(float(left) - float(right)) ** 2 for new_row, old_row in zip(new, old) for left, right in zip(new_row, old_row)]
    return {
        "max_abs": max(deltas, default=0.0),
        "mean_abs": sum(deltas) / len(deltas) if deltas else 0.0,
        "rmse": math.sqrt(sum(squared) / len(squared)) if squared else 0.0,
        "n_cells": len(deltas),
        "n_cells_with_abs_delta_gt_tolerance": sum(value > MATRIX_TOLERANCE for value in deltas),
    }


def _recovery(raw: object, label: str, *, exact: bool = False) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not set(_RECOVERY_FIELDS).issubset(raw):
        raise ValueError(f"{label} fields drift")
    if exact and set(raw) != set(_RECOVERY_FIELDS):
        raise ValueError(f"{label} fields drift")
    out = {field: raw[field] for field in _RECOVERY_FIELDS}
    integer_fields = {
        "n_truth_leaves", "n_estimate_leaves", "n_truth_splits", "n_estimate_splits",
        "true_positive", "false_negative", "false_positive", "rf",
    }
    float_fields = {"normalized_rf", "clade_recovery", "split_precision"}
    for field in integer_fields:
        if isinstance(out[field], bool) or not isinstance(out[field], int) or out[field] < 0:
            raise ValueError(f"{label} {field} drift")
    for field in float_fields:
        if isinstance(out[field], bool) or not isinstance(out[field], (int, float)) or not math.isfinite(float(out[field])):
            raise ValueError(f"{label} {field} drift")
        out[field] = float(out[field])
    for field in ("exact_tree_recovery", "polytomy_aware_exact_recovery"):
        if not isinstance(out[field], bool):
            raise ValueError(f"{label} {field} drift")
    return out


def _numeric_recovery(row: Mapping[str, Any], field: str) -> float:
    if field == "exact_tree_recovery_numeric":
        return 1.0 if row["exact_tree_recovery"] else 0.0
    if field == "polytomy_aware_exact_recovery_numeric":
        return 1.0 if row["polytomy_aware_exact_recovery"] else 0.0
    return float(row[field])


def _aggregate_matched(rows: Sequence[Mapping[str, Any]], matched: Sequence[str]) -> list[dict[str, Any]]:
    matched_set = set(matched)
    out = []
    for metric in DIRECT_METRICS:
        selected = [row for row in rows if row["metric"] == metric and row["tree_id"] in matched_set]
        if len(selected) != len(matched):
            raise ValueError("leaf compatibility matched eligibility row count drift")
        field_rows = {}
        for field in _DELTA_FIELDS:
            old = [_numeric_recovery(row["legacy_recovery"], field) for row in selected]
            new = [_numeric_recovery(row["new_recovery"], field) for row in selected]
            old_mean = sum(old) / len(old) if old else 0.0
            new_mean = sum(new) / len(new) if new else 0.0
            field_rows[field] = {
                "legacy_mean": old_mean,
                "new_mean": new_mean,
                "delta": new_mean - old_mean,
            }
        out.append(
            {
                "metric": metric,
                "matched_eligible_tree_ids": list(matched),
                "n_matched_eligible": len(matched),
                "recovery": field_rows,
                "matrix_max_abs_delta": max(
                    (row["matrix_delta"]["max_abs"] for row in selected), default=0.0
                ),
            }
        )
    return out


def _verified_pin_bytes(pin: ArtifactPin | None) -> bytes:
    if pin is None:
        raise ValueError("required leaf compatibility pin is missing")
    path = Path(pin.path)
    if path.is_symlink() or not path.is_file() or str(path.resolve(strict=True)) != pin.path:
        raise ValueError(f"leaf compatibility pin is missing/noncanonical/symlink: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise ValueError(f"leaf compatibility pin SHA-256 drift: {path}")
    return raw


def _json_bytes(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _optional_pin(raw: object, label: str) -> ArtifactPin | None:
    return None if raw is None else _pin(raw, label)


def _pin(raw: object, label: str) -> ArtifactPin:
    if isinstance(raw, ArtifactPin):
        value = raw
    elif isinstance(raw, Mapping) and set(raw) == {"path", "sha256"}:
        value = ArtifactPin(path=raw["path"], sha256=raw["sha256"])
    else:
        raise ValueError(f"{label} must be an exact path+sha256 pin")
    _root(value.path, f"{label}.path")
    _sha(value.sha256, f"{label}.sha256")
    return value


def _pin_dict(value: ArtifactPin | None) -> dict[str, str] | None:
    return None if value is None else value.to_dict()


def _root(raw: object, label: str) -> str:
    if not isinstance(raw, (str, os.PathLike)):
        raise ValueError(f"{label} must be an absolute path")
    text = os.fspath(raw)
    normalized = os.path.normpath(text)
    if not os.path.isabs(text) or text != normalized or text == "/":
        raise ValueError(f"{label} must be a normalized bounded absolute path")
    return text


def _strictly_below(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root and path != root
    except ValueError:
        return False


def _sha(raw: object, label: str) -> str:
    if not isinstance(raw, str) or _SHA256.fullmatch(raw) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return raw


def _text(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ValueError(f"{label} must be nonempty normalized text")
    return raw


__all__ = [
    "BASELINE_SCHEMA", "LeafCompatibilityAPI", "LeafCompatibilitySpec",
    "MATRIX_TOLERANCE", "POPULATION", "RECEIPT_BASENAME", "RECEIPT_SCHEMA",
    "SPEC_SCHEMA", "build_leaf_compatibility_spec", "execute_leaf_compatibility",
    "read_leaf_compatibility_spec", "validate_leaf_compatibility_receipt",
]
