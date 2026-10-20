"""Fail-closed planning for the leaf-only direct compatibility baseline.

The production all-node endpoint treats trained ancestors as observed nodes.
Neighbor joining, however, remains a compatibility-only analysis over the
trained leaves.  This module converts an *existing* legacy leaf-direct output
tree into a normalized, fully pinned baseline without writing during the
planning phase.  A caller can therefore include the returned bytes in a
larger atomic deployment transaction.

No legacy artifact is copied or rewritten.  The only copied input is the old
table source, whose exact bytes are placed inside the isolated deployment so
the comparison is bound to the table that it is intended to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .checkpoints import ArtifactPin, PRODUCTION_N_TREES
from .direct_compatibility import (
    BASELINE_SCHEMA,
    LeafCompatibilitySpec,
    build_leaf_compatibility_spec,
)
from .direct_execution import DIRECT_METRICS
from .manifest import canonical_sha256
from .production import PRODUCTION_TREE_IDS


PREFLIGHT_SCHEMA = "weighttraits.all_node_leaf_compatibility_preflight.v1"
POPULATION = "trained_leaves"
_ARTIFACT_DIRS = {
    "adapter_chain": "cumulative_leaf_analysis",
    "merged": "merged_leaf_analysis",
    "model": "model_leaf_analysis",
}
_RECOVERY_FIELDS = (
    "n_truth_leaves",
    "n_estimate_leaves",
    "n_truth_splits",
    "n_estimate_splits",
    "true_positive",
    "false_negative",
    "false_positive",
    "rf",
    "normalized_rf",
    "clade_recovery",
    "split_precision",
    "exact_tree_recovery",
    "polytomy_aware_exact_recovery",
)


@dataclass(frozen=True)
class PlannedCompatibilityFile:
    """One exact byte payload to include in the deployment transaction."""

    pin: ArtifactPin
    content: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "pin": self.pin.to_dict(),
            "n_bytes": len(self.content),
        }


@dataclass(frozen=True)
class LeafCompatibilityPreflightPlan:
    """In-memory result of an available or unavailable compatibility plan."""

    cohort_key: str
    destination_dir: str
    comparison_available: bool
    spec: LeafCompatibilitySpec
    planned_files: tuple[PlannedCompatibilityFile, ...]
    source_direct_rollup: ArtifactPin | None
    source_table_origin: ArtifactPin | None
    baseline_manifest: ArtifactPin | None
    source_table_input: ArtifactPin | None
    unavailable_reason: ArtifactPin | None

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": PREFLIGHT_SCHEMA,
            "cohort_key": self.cohort_key,
            "destination_dir": self.destination_dir,
            "comparison_available": self.comparison_available,
            "spec": self.spec.to_dict(),
            "planned_files": [row.to_dict() for row in self.planned_files],
            "source_direct_rollup": _pin_dict(self.source_direct_rollup),
            "source_table_origin": _pin_dict(self.source_table_origin),
            "baseline_manifest": _pin_dict(self.baseline_manifest),
            "source_table_input": _pin_dict(self.source_table_input),
            "unavailable_reason": _pin_dict(self.unavailable_reason),
        }
        return {**core, "preflight_sha256": canonical_sha256(core)}


def plan_available_leaf_compatibility(
    *,
    cohort_key: str,
    legacy_output_root: str | Path,
    old_table_source: str | Path,
    destination_dir: str | Path,
    snapshot_root: str | Path,
) -> LeafCompatibilityPreflightPlan:
    """Plan an exact 50-tree x 3-metric legacy leaf comparison.

    This function performs no writes.  It reads and validates the legacy
    output, returns the exact bytes for the copied table source and canonical
    baseline manifest, and constructs the final runtime spec from their future
    absolute paths and SHA-256 digests.
    """

    cohort = _text(cohort_key, "cohort_key")
    root = _canonical_directory(legacy_output_root, "legacy_output_root")
    destination = _future_directory(destination_dir, "destination_dir")
    table_origin, table_bytes = _read_pinned_file(old_table_source, "old_table_source")
    if not table_bytes:
        raise ValueError("old_table_source must not be empty")

    rollup_path = root / "direct_run_set_summary.json"
    rollup_pin, rollup_bytes = _read_pinned_file(rollup_path, "source direct rollup")
    rollup = _json_object(rollup_bytes, "source direct rollup")
    artifact = _validate_rollup_header(rollup)
    analysis_root = root / "analysis"
    if not analysis_root.is_dir() or analysis_root.is_symlink():
        raise ValueError("legacy output is missing its canonical analysis directory")

    by_coordinate: dict[tuple[str, str], Mapping[str, Any]] = {}
    rollup_rows = rollup.get("rows")
    assert isinstance(rollup_rows, list)
    for raw in rollup_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("source rollup rows must be mappings")
        coordinate = (raw.get("tree_id"), raw.get("metric"))
        if coordinate in by_coordinate:
            raise ValueError(f"duplicate source rollup coordinate: {coordinate!r}")
        by_coordinate[coordinate] = raw
    expected_coordinates = {
        (tree_id, metric)
        for tree_id in PRODUCTION_TREE_IDS
        for metric in DIRECT_METRICS
    }
    if set(by_coordinate) != expected_coordinates:
        raise ValueError("source rollup does not have the exact 50x3 coordinate set")

    rows: list[dict[str, Any]] = []
    source_pins: dict[str, ArtifactPin] = {rollup_pin.path: rollup_pin}
    eligibility: dict[str, bool] = {}
    for tree_id in PRODUCTION_TREE_IDS:
        tree_root = analysis_root / tree_id / _ARTIFACT_DIRS[artifact]
        tree_root = _canonical_directory(tree_root, f"{tree_id} legacy analysis")
        models_pin, models_bytes = _read_pinned_file(
            tree_root / "models.json", f"{tree_id} models.json"
        )
        models = _json_list(models_bytes, f"{tree_id} models.json")
        model_ids = [_text(value, f"{tree_id} model ID") for value in models]
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError(f"{tree_id} models.json is empty or contains duplicate IDs")
        summary_pin, summary_bytes = _read_pinned_file(
            tree_root / "summary.json", f"{tree_id} summary"
        )
        summary = _json_object(summary_bytes, f"{tree_id} summary")
        truth_pin = _validate_tree_summary(
            summary,
            tree_id=tree_id,
            artifact=artifact,
            model_ids=model_ids,
            tree_root=tree_root,
        )
        source_pins[models_pin.path] = models_pin
        source_pins[summary_pin.path] = summary_pin
        source_pins[truth_pin.path] = truth_pin

        score_by_metric = _summary_score_paths(summary, tree_root=tree_root)
        tree_n_truth_splits: int | None = None
        for metric in DIRECT_METRICS:
            source_row = by_coordinate[(tree_id, metric)]
            _validate_rollup_row_scope(
                source_row,
                tree_id=tree_id,
                metric=metric,
                artifact=artifact,
                summary_pin=summary_pin,
                truth_pin=truth_pin,
                n_models=len(model_ids),
            )
            matrix_pin, matrix_bytes = _read_pinned_file(
                tree_root / f"distance_matrix_{metric}.npy",
                f"{tree_id}/{metric} legacy matrix",
            )
            _validate_npy_matrix(matrix_bytes, n_models=len(model_ids), label=f"{tree_id}/{metric}")
            score_pin, score_bytes = _read_pinned_file(
                score_by_metric[metric], f"{tree_id}/{metric} score"
            )
            score = _recovery(_json_object(score_bytes, f"{tree_id}/{metric} score"))
            _crosscheck_rollup_recovery(source_row, score, tree_id=tree_id, metric=metric)
            if tree_n_truth_splits is None:
                tree_n_truth_splits = score["n_truth_splits"]
            elif tree_n_truth_splits != score["n_truth_splits"]:
                raise ValueError(f"{tree_id} n_truth_splits differs across metrics")
            source_pins[matrix_pin.path] = matrix_pin
            source_pins[score_pin.path] = score_pin
            rows.append(
                {
                    "tree_id": tree_id,
                    "metric": metric,
                    "topology_eligible": score["n_truth_splits"] > 0,
                    "legacy_matrix": matrix_pin.to_dict(),
                    "legacy_model_ids": models_pin.to_dict(),
                    "source_summary": summary_pin.to_dict(),
                    "truth_manifest": truth_pin.to_dict(),
                    "legacy_recovery": score,
                }
            )
        assert tree_n_truth_splits is not None
        eligibility[tree_id] = tree_n_truth_splits > 0

    table_target = destination / "old_table_source.tex"
    table_target_pin = ArtifactPin(str(table_target), hashlib.sha256(table_bytes).hexdigest())
    baseline_target = destination / "baseline.json"
    baseline_core = {
        "schema": BASELINE_SCHEMA,
        "valid": True,
        "comparison_available": True,
        "cohort_key": cohort,
        "population": POPULATION,
        "internal_nodes_as_tips": False,
        "source_direct_rollup": rollup_pin.to_dict(),
        "source_table_input": table_target_pin.to_dict(),
        "n_trees": PRODUCTION_N_TREES,
        "n_metrics": len(DIRECT_METRICS),
        "n_rows": PRODUCTION_N_TREES * len(DIRECT_METRICS),
        "tree_ids": list(PRODUCTION_TREE_IDS),
        "metrics": list(DIRECT_METRICS),
        "matched_eligible_tree_ids": [
            tree_id for tree_id in PRODUCTION_TREE_IDS if eligibility[tree_id]
        ],
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }
    baseline = {
        **baseline_core,
        "baseline_sha256": canonical_sha256(baseline_core),
    }
    baseline_bytes = _canonical_json_bytes(baseline)
    baseline_pin = ArtifactPin(
        str(baseline_target), hashlib.sha256(baseline_bytes).hexdigest()
    )
    reconstruct, recovery, splits = _engine_pins(snapshot_root)
    spec = build_leaf_compatibility_spec(
        comparison_available=True,
        baseline_manifest=baseline_pin,
        source_direct_rollup=rollup_pin,
        source_table_input=table_target_pin,
        reconstruct_source=reconstruct,
        recovery_source=recovery,
        splits_source=splits,
    )

    # Close the read/check race as far as possible: every consumed legacy
    # artifact is re-hashed after the complete in-memory plan is assembled.
    source_pins[table_origin.path] = table_origin
    for pin in source_pins.values():
        _verify_pin(pin)
    for pin in (reconstruct, recovery, splits):
        _verify_pin(pin)

    return LeafCompatibilityPreflightPlan(
        cohort_key=cohort,
        destination_dir=str(destination),
        comparison_available=True,
        spec=spec,
        planned_files=(
            PlannedCompatibilityFile(table_target_pin, table_bytes),
            PlannedCompatibilityFile(baseline_pin, baseline_bytes),
        ),
        source_direct_rollup=rollup_pin,
        source_table_origin=table_origin,
        baseline_manifest=baseline_pin,
        source_table_input=table_target_pin,
        unavailable_reason=None,
    )


def plan_unavailable_leaf_compatibility(
    *,
    cohort_key: str,
    reason: str,
    destination_dir: str | Path,
) -> LeafCompatibilityPreflightPlan:
    """Plan an explicit, pinned ``comparison_unavailable`` outcome."""

    cohort = _text(cohort_key, "cohort_key")
    explanation = _text(reason, "reason")
    destination = _future_directory(destination_dir, "destination_dir")
    reason_bytes = (
        f"cohort={cohort}\n"
        "status=comparison_unavailable\n"
        f"reason={explanation}\n"
    ).encode("utf-8")
    reason_pin = ArtifactPin(
        str(destination / "comparison_unavailable.txt"),
        hashlib.sha256(reason_bytes).hexdigest(),
    )
    spec = build_leaf_compatibility_spec(
        comparison_available=False,
        unavailable_reason=reason_pin,
    )
    return LeafCompatibilityPreflightPlan(
        cohort_key=cohort,
        destination_dir=str(destination),
        comparison_available=False,
        spec=spec,
        planned_files=(PlannedCompatibilityFile(reason_pin, reason_bytes),),
        source_direct_rollup=None,
        source_table_origin=None,
        baseline_manifest=None,
        source_table_input=None,
        unavailable_reason=reason_pin,
    )


def materialize_leaf_compatibility_preflight(
    plan: LeafCompatibilityPreflightPlan,
) -> LeafCompatibilityPreflightPlan:
    """Atomically and exclusively materialize one already-validated plan."""

    if not isinstance(plan, LeafCompatibilityPreflightPlan):
        raise ValueError("plan must be a LeafCompatibilityPreflightPlan")
    destination = Path(plan.destination_dir)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("compatibility destination parent must already be a canonical directory")
    if os.path.lexists(destination):
        raise FileExistsError(f"compatibility destination already exists: {destination}")
    for row in plan.planned_files:
        path = Path(row.pin.path)
        if path.parent != destination:
            raise ValueError("planned compatibility file escapes its destination")
        if hashlib.sha256(row.content).hexdigest() != row.pin.sha256:
            raise ValueError("planned compatibility file digest drift")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        for row in plan.planned_files:
            target = temporary / Path(row.pin.path).name
            with target.open("xb") as handle:
                handle.write(row.content)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(temporary, destination)
        for row in plan.planned_files:
            _verify_pin(row.pin)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                if child.is_file() and not child.is_symlink():
                    child.unlink()
            temporary.rmdir()
        raise
    return plan


def _validate_rollup_header(raw: Mapping[str, Any]) -> str:
    if raw.get("valid") is not True:
        raise ValueError("source direct rollup is not valid")
    if raw.get("n_tree_summaries") != PRODUCTION_N_TREES:
        raise ValueError("source direct rollup does not contain exactly 50 tree summaries")
    if raw.get("n_rows") != PRODUCTION_N_TREES * len(DIRECT_METRICS):
        raise ValueError("source direct rollup does not contain exactly 150 rows")
    if raw.get("tree_ids") != list(PRODUCTION_TREE_IDS):
        raise ValueError("source direct rollup tree order/scope drift")
    if raw.get("metrics") != sorted(DIRECT_METRICS):
        raise ValueError("source direct rollup metric scope drift")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != PRODUCTION_N_TREES * len(DIRECT_METRICS):
        raise ValueError("source direct rollup rows drift")
    artifact = raw.get("artifact")
    if artifact not in _ARTIFACT_DIRS:
        raise ValueError("source direct rollup artifact is unsupported")
    return artifact


def _validate_tree_summary(
    raw: Mapping[str, Any],
    *,
    tree_id: str,
    artifact: str,
    model_ids: Sequence[str],
    tree_root: Path,
) -> ArtifactPin:
    if raw.get("analysis_engine") != "direct" or raw.get("artifact") != artifact:
        raise ValueError(f"{tree_id} summary engine/artifact drift")
    if raw.get("n_models") != len(model_ids) or raw.get("model_ids") != list(model_ids):
        raise ValueError(f"{tree_id} summary model ordering drift")
    if raw.get("metrics") != sorted(DIRECT_METRICS):
        raise ValueError(f"{tree_id} summary metric scope drift")
    results = raw.get("results")
    if (
        not isinstance(results, list)
        or len(results) != len(DIRECT_METRICS)
        or [row.get("metric") for row in results if isinstance(row, Mapping)]
        != sorted(DIRECT_METRICS)
    ):
        raise ValueError(f"{tree_id} summary result scope/order drift")
    truth_path = _producer_path(raw.get("truth_manifest"), tree_root, f"{tree_id} truth manifest")
    pin, _ = _read_pinned_file(truth_path, f"{tree_id} truth manifest")
    return pin


def _summary_score_paths(raw: Mapping[str, Any], *, tree_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for result in raw["results"]:
        assert isinstance(result, Mapping)
        metric = result["metric"]
        out[metric] = _producer_path(
            result.get("score"), tree_root, f"{metric} source score"
        )
    if set(out) != set(DIRECT_METRICS):
        raise ValueError("tree summary does not bind one score per exact metric")
    return out


def _validate_rollup_row_scope(
    raw: Mapping[str, Any],
    *,
    tree_id: str,
    metric: str,
    artifact: str,
    summary_pin: ArtifactPin,
    truth_pin: ArtifactPin,
    n_models: int,
) -> None:
    expected = {
        "tree_id": tree_id,
        "metric": metric,
        "artifact": artifact,
        "n_models": n_models,
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            raise ValueError(f"{tree_id}/{metric} rollup {field} drift")
    if _canonical_existing_path(raw.get("summary"), f"{tree_id}/{metric} rollup summary") != summary_pin.path:
        raise ValueError(f"{tree_id}/{metric} rollup summary pin drift")
    if _canonical_existing_path(raw.get("truth_manifest"), f"{tree_id}/{metric} rollup truth") != truth_pin.path:
        raise ValueError(f"{tree_id}/{metric} rollup truth pin drift")


def _crosscheck_rollup_recovery(
    row: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    tree_id: str,
    metric: str,
) -> None:
    # The run-set rollup intentionally omits a few detailed score fields.  For
    # every field it does expose, demand exact agreement with the pinned tree
    # score before normalizing the full score into the baseline.
    for field in _RECOVERY_FIELDS:
        if field not in row:
            continue
        left = row[field]
        right = score[field]
        if isinstance(right, float):
            if isinstance(left, bool) or not isinstance(left, (int, float)) or float(left) != right:
                raise ValueError(f"{tree_id}/{metric} rollup/score {field} drift")
        elif left != right:
            raise ValueError(f"{tree_id}/{metric} rollup/score {field} drift")


def _recovery(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not set(_RECOVERY_FIELDS).issubset(raw):
        raise ValueError("legacy score is missing required recovery fields")
    out = {field: raw[field] for field in _RECOVERY_FIELDS}
    integer_fields = {
        "n_truth_leaves",
        "n_estimate_leaves",
        "n_truth_splits",
        "n_estimate_splits",
        "true_positive",
        "false_negative",
        "false_positive",
        "rf",
    }
    for field in integer_fields:
        if isinstance(out[field], bool) or not isinstance(out[field], int) or out[field] < 0:
            raise ValueError(f"legacy score {field} drift")
    for field in ("normalized_rf", "clade_recovery", "split_precision"):
        value = out[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"legacy score {field} drift")
        out[field] = float(value)
    for field in ("exact_tree_recovery", "polytomy_aware_exact_recovery"):
        if not isinstance(out[field], bool):
            raise ValueError(f"legacy score {field} drift")
    return out


def _validate_npy_matrix(raw: bytes, *, n_models: int, label: str) -> None:
    # Loading from bytes avoids reopening a path between pinning and validation.
    import io
    import numpy as np

    try:
        matrix = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{label} legacy matrix is not a safe NPY array") from exc
    if matrix.shape != (n_models, n_models):
        raise ValueError(f"{label} legacy matrix shape drift")
    if not np.issubdtype(matrix.dtype, np.number) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} legacy matrix contains invalid values")
    numeric = np.asarray(matrix, dtype=np.float64)
    if np.any(numeric < 0.0) or not np.array_equal(numeric, numeric.T):
        raise ValueError(f"{label} legacy matrix is negative or not exactly symmetric")
    if not np.array_equal(np.diag(numeric), np.zeros(n_models, dtype=np.float64)):
        raise ValueError(f"{label} legacy matrix diagonal drift")


def _engine_pins(snapshot_root: str | Path) -> tuple[ArtifactPin, ArtifactPin, ArtifactPin]:
    root = _canonical_directory(snapshot_root, "snapshot_root")
    candidates = (
        root / "phylo",
        root / "weighttraits" / "phylo",
        root / "src" / "weighttraits" / "phylo",
    )
    matches = [
        candidate
        for candidate in candidates
        if all((candidate / name).is_file() for name in ("reconstruct.py", "recovery.py", "splits.py"))
    ]
    unique = {str(candidate.resolve(strict=True)): candidate for candidate in matches}
    if len(unique) != 1:
        raise ValueError("snapshot_root does not resolve exactly one pinned phylo engine")
    engine = next(iter(unique.values()))
    pins = []
    for filename in ("reconstruct.py", "recovery.py", "splits.py"):
        pin, _ = _read_pinned_file(engine / filename, f"pinned {filename}")
        pins.append(pin)
    return tuple(pins)  # type: ignore[return-value]


def _producer_path(raw: object, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ValueError(f"{label} must be a nonempty path")
    value = Path(raw)
    return Path(_canonical_existing_path(value if value.is_absolute() else base / value, label))


def _canonical_existing_path(raw: object, label: str) -> str:
    if not isinstance(raw, (str, os.PathLike)):
        raise ValueError(f"{label} must be a path")
    path = Path(raw)
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise ValueError(f"{label} must be normalized and absolute")
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be a canonical regular file")
    return str(path)


def _canonical_directory(raw: object, label: str) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise ValueError(f"{label} must be a path")
    path = Path(raw)
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise ValueError(f"{label} must be normalized and absolute")
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be a canonical directory")
    return path


def _future_directory(raw: object, label: str) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise ValueError(f"{label} must be a path")
    path = Path(raw)
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path) or path == Path("/"):
        raise ValueError(f"{label} must be a bounded normalized absolute path")
    if os.path.lexists(path):
        raise FileExistsError(f"{label} must not already exist: {path}")
    return path


def _read_pinned_file(raw: object, label: str) -> tuple[ArtifactPin, bytes]:
    path = Path(_canonical_existing_path(raw, label))
    content = path.read_bytes()
    return ArtifactPin(str(path), hashlib.sha256(content).hexdigest()), content


def _verify_pin(pin: ArtifactPin) -> None:
    observed, _ = _read_pinned_file(pin.path, "pinned source artifact")
    if observed != pin:
        raise ValueError(f"source artifact changed during compatibility planning: {pin.path}")


def _json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _json_list(raw: bytes, label: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"{label} must contain a JSON list")
    return value


def _canonical_json_bytes(raw: Mapping[str, Any]) -> bytes:
    return (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _pin_dict(pin: ArtifactPin | None) -> dict[str, str] | None:
    return None if pin is None else pin.to_dict()


def _text(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw or raw.strip() != raw or "\n" in raw or "\r" in raw:
        raise ValueError(f"{label} must be nonempty normalized single-line text")
    return raw


__all__ = [
    "LeafCompatibilityPreflightPlan",
    "PREFLIGHT_SCHEMA",
    "PlannedCompatibilityFile",
    "materialize_leaf_compatibility_preflight",
    "plan_available_leaf_compatibility",
    "plan_unavailable_leaf_compatibility",
]
