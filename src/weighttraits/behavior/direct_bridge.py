"""Fail-closed bridge from native direct analyses to behavior analyses.

The bridge is intentionally narrower than a generic NumPy importer. A source
matrix is accepted only when an externally pinned source attestation matches the
native ``analyze_training_ledger_direct`` summary, its ledger, truth manifest,
completion receipt, ordered model IDs, and exact matrix bytes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from weighttraits.verification.digests import sha256_file


DIRECT_DISTANCE_SCHEMA = "weighttraits.behavior.direct_distance.v2"
DIRECT_DISTANCE_METADATA = "direct_distance.json"
DIRECT_DISTANCE_MATRIX = "distance_matrix.npy"
DIRECT_DISTANCE_ENGINE = "direct_streaming_sufficient_stats"
DIRECT_METRICS = frozenset({"cosine", "l1", "l2", "correlation", "threshold", "cka", "linear_cka"})
DIRECT_ARTIFACT_REPRESENTATIONS = {
    "model": "full_weight",
    "merged": "full_weight",
    "adapter_chain": "lora_cumulative_delta",
}


@dataclass(frozen=True)
class DirectDistanceSourceAttestation:
    """Externally pinned provenance required for one native distance matrix."""

    cohort_id: str
    tree_id: str
    metric: str
    artifact: str
    representation: str
    summary: str
    summary_sha256: str
    matrix_sha256: str
    model_ids_sha256: str
    ledger: str
    ledger_sha256: str
    truth_manifest: str
    truth_manifest_sha256: str
    completion_receipt: str
    completion_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "cohort_id",
            "tree_id",
            "metric",
            "artifact",
            "representation",
            "summary",
            "ledger",
            "truth_manifest",
            "completion_receipt",
        ):
            _required_text(getattr(self, field), context=field)
        if self.metric not in DIRECT_METRICS:
            raise ValueError(f"unsupported native direct metric: {self.metric!r}")
        expected_representation = DIRECT_ARTIFACT_REPRESENTATIONS.get(self.artifact)
        if expected_representation is None:
            raise ValueError(f"unsupported native direct artifact: {self.artifact!r}")
        if self.representation != expected_representation:
            raise ValueError(
                f"artifact {self.artifact!r} requires representation "
                f"{expected_representation!r}, got {self.representation!r}"
            )
        for field in (
            "summary_sha256",
            "matrix_sha256",
            "model_ids_sha256",
            "ledger_sha256",
            "truth_manifest_sha256",
            "completion_receipt_sha256",
        ):
            _validated_sha256(getattr(self, field), context=field)

    def to_dict(self) -> dict[str, str]:
        """Return the exact JSON contract used by artifact readers."""

        return {
            field: getattr(self, field)
            for field in (
                "cohort_id",
                "tree_id",
                "metric",
                "artifact",
                "representation",
                "summary",
                "summary_sha256",
                "matrix_sha256",
                "model_ids_sha256",
                "ledger",
                "ledger_sha256",
                "truth_manifest",
                "truth_manifest_sha256",
                "completion_receipt",
                "completion_receipt_sha256",
            )
        }


@dataclass(frozen=True)
class DirectDistancePair:
    """One upper-triangle distance with its exact matrix coordinates."""

    left_index: int
    right_index: int
    left_model_id: str
    right_model_id: str
    distance: float


@dataclass(frozen=True)
class DirectDistanceArtifact:
    """Verified native distance matrix and its immutable ordering contract."""

    metadata_path: Path
    matrix_path: Path
    metadata: Mapping[str, Any]
    attestation: DirectDistanceSourceAttestation
    model_ids: tuple[str, ...]
    matrix: np.ndarray

    def upper_triangle_pairs(self) -> tuple[DirectDistancePair, ...]:
        """Return pairs in deterministic row-major upper-triangle order."""

        return tuple(
            DirectDistancePair(
                left_index=left,
                right_index=right,
                left_model_id=self.model_ids[left],
                right_model_id=self.model_ids[right],
                distance=float(self.matrix[left, right]),
            )
            for left in range(len(self.model_ids))
            for right in range(left + 1, len(self.model_ids))
        )


def export_native_direct_distance(
    summary_path: str | Path,
    *,
    metric: str,
    out_dir: str | Path,
    attestation: DirectDistanceSourceAttestation,
    overwrite: bool = False,
) -> DirectDistanceArtifact:
    """Export one externally attested native direct-analysis matrix.

    Publication is serialized and metadata is replaced last, so an interrupted
    overwrite can be detected rather than consumed as a mixed artifact.
    """

    if not isinstance(attestation, DirectDistanceSourceAttestation):
        raise ValueError("attestation must be a DirectDistanceSourceAttestation")
    summary_file = Path(summary_path)
    _require_same_path(summary_file, Path(attestation.summary), context="summary")
    _verify_file_hash(summary_file, attestation.summary_sha256, context="summary")
    metric_name = _required_text(metric, context="metric")
    if metric_name != attestation.metric:
        raise ValueError(
            f"requested metric {metric_name!r} does not match attested metric "
            f"{attestation.metric!r}"
        )

    summary = _load_json_mapping(summary_file)
    _validate_native_summary(summary, summary_file=summary_file, attestation=attestation)
    results = summary["results"]
    matching_results = [
        row for row in results if isinstance(row, Mapping) and row.get("metric") == metric_name
    ]
    if len(matching_results) != 1:
        raise ValueError(
            f"direct summary must contain exactly one result for metric {metric_name!r}: "
            f"found {len(matching_results)}"
        )

    model_ids = _validated_model_ids(summary.get("model_ids"))
    if ordered_strings_sha256(model_ids) != attestation.model_ids_sha256:
        raise ValueError("native direct ordered model IDs do not match the attestation")
    declared_n = summary.get("n_models")
    if isinstance(declared_n, bool) or not isinstance(declared_n, int):
        raise ValueError(f"direct summary n_models must be an integer: {summary_file}")
    if declared_n != len(model_ids):
        raise ValueError(
            f"direct summary n_models={declared_n} does not match "
            f"{len(model_ids)} ordered model IDs"
        )

    source_matrix = summary_file.parent / f"distance_matrix_{metric_name}.npy"
    _verify_file_hash(source_matrix, attestation.matrix_sha256, context="native matrix")
    matrix = _load_validated_matrix(source_matrix, model_ids=model_ids)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    matrix_out = out / DIRECT_DISTANCE_MATRIX
    metadata_out = out / DIRECT_DISTANCE_METADATA
    with _artifact_write_lock(out, name="direct-distance"):
        if not overwrite and (matrix_out.exists() or metadata_out.exists()):
            raise FileExistsError(
                f"direct-distance artifact already exists under {out}; "
                "pass overwrite=True explicitly"
            )
        _atomic_copy(source_matrix, matrix_out)
        matrix_copy = _load_validated_matrix(matrix_out, model_ids=model_ids)
        if matrix.dtype != matrix_copy.dtype or not np.array_equal(matrix, matrix_copy):
            raise RuntimeError(
                "exported direct matrix does not exactly preserve source numeric values"
            )

        metadata: dict[str, Any] = {
            "schema": DIRECT_DISTANCE_SCHEMA,
            "producer": "weighttraits",
            "analysis_engine": "direct",
            "distance_engine": DIRECT_DISTANCE_ENGINE,
            "tree_id": attestation.tree_id,
            "cohort_id": attestation.cohort_id,
            "metric": metric_name,
            "artifact": attestation.artifact,
            "representation": attestation.representation,
            "n_models": len(model_ids),
            "model_ids": list(model_ids),
            "model_ids_sha256": ordered_strings_sha256(model_ids),
            "matrix": {
                "file": DIRECT_DISTANCE_MATRIX,
                "sha256": sha256_file(matrix_out),
                "values_sha256": array_values_sha256(matrix_copy),
                "dtype": matrix_copy.dtype.str,
                "shape": list(matrix_copy.shape),
                "order": "C",
            },
            "source_attestation": attestation.to_dict(),
        }
        _atomic_write_json(metadata_out, metadata)
    return read_direct_distance_artifact(out, expected_attestation=attestation)


def read_direct_distance_artifact(
    path: str | Path,
    *,
    expected_attestation: DirectDistanceSourceAttestation,
) -> DirectDistanceArtifact:
    """Load an artifact only after matrix and external provenance verification."""

    if not isinstance(expected_attestation, DirectDistanceSourceAttestation):
        raise ValueError("expected_attestation must be a DirectDistanceSourceAttestation")
    raw_path = Path(path)
    metadata_path = raw_path / DIRECT_DISTANCE_METADATA if raw_path.is_dir() else raw_path
    metadata = _load_json_mapping(metadata_path)
    if metadata.get("schema") != DIRECT_DISTANCE_SCHEMA:
        raise ValueError(f"unsupported direct-distance schema in {metadata_path}")
    if metadata.get("producer") != "weighttraits":
        raise ValueError(f"direct-distance artifact producer mismatch: {metadata_path}")
    if metadata.get("analysis_engine") != "direct":
        raise ValueError(f"direct-distance artifact is not native direct analysis: {metadata_path}")
    if metadata.get("distance_engine") != DIRECT_DISTANCE_ENGINE:
        raise ValueError("direct-distance engine contract mismatch")

    expected_values = {
        "tree_id": expected_attestation.tree_id,
        "cohort_id": expected_attestation.cohort_id,
        "metric": expected_attestation.metric,
        "artifact": expected_attestation.artifact,
        "representation": expected_attestation.representation,
        "source_attestation": expected_attestation.to_dict(),
    }
    for field, expected in expected_values.items():
        if metadata.get(field) != expected:
            raise ValueError(f"direct-distance {field} does not match expected provenance")

    model_ids = _validated_model_ids(metadata.get("model_ids"))
    n_models = metadata.get("n_models")
    if isinstance(n_models, bool) or not isinstance(n_models, int) or n_models != len(model_ids):
        raise ValueError("direct-distance n_models does not match ordered model IDs")
    model_ids_hash = ordered_strings_sha256(model_ids)
    if metadata.get("model_ids_sha256") != model_ids_hash:
        raise ValueError("direct-distance ordered model ID hash mismatch")
    if model_ids_hash != expected_attestation.model_ids_sha256:
        raise ValueError("direct-distance ordered model IDs do not match expected provenance")

    matrix_metadata = metadata.get("matrix")
    if not isinstance(matrix_metadata, Mapping):
        raise ValueError("direct-distance matrix metadata must be a mapping")
    if matrix_metadata.get("file") != DIRECT_DISTANCE_MATRIX:
        raise ValueError("direct-distance matrix file must use the canonical artifact name")
    matrix_path = metadata_path.parent / DIRECT_DISTANCE_MATRIX
    if not matrix_path.is_file():
        raise FileNotFoundError(f"direct-distance matrix is missing: {matrix_path}")
    matrix_file_hash = sha256_file(matrix_path)
    if matrix_metadata.get("sha256") != matrix_file_hash:
        raise ValueError("direct-distance matrix file hash mismatch")
    if matrix_file_hash != expected_attestation.matrix_sha256:
        raise ValueError("direct-distance matrix does not match expected provenance")

    matrix = _load_validated_matrix(matrix_path, model_ids=model_ids)
    if matrix_metadata.get("values_sha256") != array_values_sha256(matrix):
        raise ValueError("direct-distance numeric matrix hash mismatch")
    if matrix_metadata.get("dtype") != matrix.dtype.str:
        raise ValueError("direct-distance matrix dtype mismatch")
    if matrix_metadata.get("shape") != list(matrix.shape):
        raise ValueError("direct-distance matrix shape metadata mismatch")
    if matrix_metadata.get("order") != "C":
        raise ValueError("direct-distance matrix order contract must be C")

    matrix.setflags(write=False)
    return DirectDistanceArtifact(
        metadata_path=metadata_path,
        matrix_path=matrix_path,
        metadata=metadata,
        attestation=expected_attestation,
        model_ids=model_ids,
        matrix=matrix,
    )


def array_values_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and exact C-order numeric bytes."""

    values = np.asarray(array)
    descriptor = {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(descriptor))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def ordered_strings_sha256(values: Sequence[str]) -> str:
    """Hash an ordered string list using an unambiguous canonical JSON encoding."""

    return hashlib.sha256(_canonical_json_bytes(list(values))).hexdigest()


def _validate_native_summary(
    summary: Mapping[str, Any],
    *,
    summary_file: Path,
    attestation: DirectDistanceSourceAttestation,
) -> None:
    if summary.get("analysis_engine") != "direct":
        raise ValueError(f"analysis summary is not native direct analysis: {summary_file}")
    if summary.get("distance_engine") != DIRECT_DISTANCE_ENGINE:
        raise ValueError(f"analysis summary has the wrong direct distance engine: {summary_file}")
    if summary.get("artifact") != attestation.artifact:
        raise ValueError("direct summary artifact does not match the attestation")
    if summary.get("representation") != attestation.representation:
        raise ValueError("direct summary representation does not match the attestation")
    metrics = summary.get("metrics")
    if not isinstance(metrics, list) or attestation.metric not in {str(value) for value in metrics}:
        raise ValueError(f"metric {attestation.metric!r} is not declared by {summary_file}")
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"direct summary results must be a list: {summary_file}")
    if summary.get("ledger") != attestation.ledger:
        raise ValueError("direct summary ledger path does not match the attestation")
    if summary.get("truth_manifest") != attestation.truth_manifest:
        raise ValueError("direct summary truth-manifest path does not match the attestation")
    _verify_file_hash(Path(attestation.ledger), attestation.ledger_sha256, context="ledger")
    _verify_file_hash(
        Path(attestation.truth_manifest),
        attestation.truth_manifest_sha256,
        context="truth manifest",
    )
    _verify_file_hash(
        Path(attestation.completion_receipt),
        attestation.completion_receipt_sha256,
        context="completion receipt",
    )
    inferred_tree_id = summary_file.parent.parent.name
    if inferred_tree_id != attestation.tree_id:
        raise ValueError(
            f"direct summary directory identifies tree {inferred_tree_id!r}, "
            f"not attested tree {attestation.tree_id!r}"
        )


def _validated_model_ids(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("direct-distance artifacts require at least two ordered model IDs")
    model_ids = tuple(_required_text(value, context="model_id") for value in raw)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("direct-distance model IDs must be unique")
    return model_ids


def _load_validated_matrix(path: Path, *, model_ids: Sequence[str]) -> np.ndarray:
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid direct-distance NumPy matrix: {path}") from exc
    if not isinstance(matrix, np.ndarray) or not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("direct-distance matrix must use a floating-point dtype")
    if matrix.shape != (len(model_ids), len(model_ids)):
        raise ValueError(
            f"direct-distance matrix shape {matrix.shape} does not match "
            f"{len(model_ids)} ordered model IDs"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("direct-distance matrix contains non-finite values")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("direct-distance matrix must be exactly symmetric")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("direct-distance matrix diagonal must be exactly zero")
    if np.any(matrix < 0):
        raise ValueError("direct-distance matrix contains negative values")
    return matrix


def _load_json_mapping(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out

    try:
        loaded = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return loaded


def _required_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty string without surrounding whitespace")
    return value


def _validated_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase 64-character SHA-256")
    return value


def _require_same_path(observed: Path, expected: Path, *, context: str) -> None:
    try:
        observed_resolved = observed.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{context} source path is unavailable") from exc
    if observed_resolved != expected_resolved:
        raise ValueError(
            f"{context} path does not match attestation: {observed_resolved} != {expected_resolved}"
        )


def _verify_file_hash(path: Path, expected: str, *, context: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{context} SHA-256 mismatch: expected {expected}, got {observed}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@contextmanager
def _artifact_write_lock(directory: Path, *, name: str) -> Iterator[None]:
    lock_path = directory / f".{name}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"another {name} writer holds {lock_path}") from exc
    os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise RuntimeError(f"byte-preserving copy failed for {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
