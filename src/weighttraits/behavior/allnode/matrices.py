"""Fail-closed distance-matrix contracts for the all-trained-node endpoint.

The full matrix is always ordered by ``TreeTopology.ordered_node_ids``.  The
leaf-only sensitivity matrix is not accepted as an independent input: it is
derived exclusively by slicing the validated full matrix at the topology's
frozen leaf indices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

from weighttraits.behavior.allnode.manifest import TreeTopology, canonical_sha256


POPULATION = "all_trained_nodes"
INTERNAL_NODE_TREATMENT = "observed_ancestors"
MATRIX_RECEIPT_SCHEMA = "weighttraits.all_node_distance_matrix.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MatrixReceipt:
    """Immutable provenance projection for one validated all-node matrix."""

    schema: str
    valid: bool
    tree_id: str
    metric: str
    population: str
    internal_node_treatment: str
    n_nodes: int
    n_internal: int
    n_leaves: int
    topology_sha256: str
    pair_index_sha256: str
    ordered_node_ids: tuple[str, ...]
    ordered_node_ids_sha256: str
    matrix_shape: tuple[int, int]
    matrix_sha256: str
    leaf_sensitivity_indices: tuple[int, ...]
    leaf_sensitivity_ids: tuple[str, ...]
    leaf_subset_sha256: str
    leaf_matrix_shape: tuple[int, int]
    leaf_sensitivity_matrix_sha256: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for field in (
            "ordered_node_ids",
            "matrix_shape",
            "leaf_sensitivity_indices",
            "leaf_sensitivity_ids",
            "leaf_matrix_shape",
        ):
            row[field] = list(row[field])
        return row


@dataclass(frozen=True)
class ValidatedDistanceMatrix:
    """Validated full matrix, its exact leaf slice, and their receipt."""

    matrix: tuple[tuple[float, ...], ...]
    leaf_sensitivity_matrix: tuple[tuple[float, ...], ...]
    receipt: MatrixReceipt


def validate_distance_matrix(
    topology: TreeTopology,
    *,
    metric: str,
    ordered_node_ids: Sequence[str],
    matrix: Sequence[Sequence[object]],
) -> ValidatedDistanceMatrix:
    """Validate and bind one matrix to an exact ancestor-aware topology."""

    if not isinstance(topology, TreeTopology):
        raise ValueError("topology must be a validated TreeTopology")
    metric_name = _text(metric, "metric")
    observed_ids = tuple(_text(value, "ordered_node_id") for value in ordered_node_ids)
    expected_ids = topology.ordered_node_ids
    if observed_ids != expected_ids:
        raise ValueError("matrix ordered node IDs differ from topology.ordered_node_ids")
    _sha256(topology.topology_sha256, "topology_sha256")
    _sha256(topology.pair_index_sha256, "pair_index_sha256")
    _sha256(topology.leaf_subset_sha256, "leaf_subset_sha256")

    values = _matrix(matrix, expected_size=len(expected_ids))
    leaf_indices = tuple(topology.leaf_sensitivity_indices)
    _validate_leaf_indices(leaf_indices, size=len(expected_ids))
    leaf_values = tuple(
        tuple(values[left][right] for right in leaf_indices) for left in leaf_indices
    )
    leaf_ids = tuple(expected_ids[index] for index in leaf_indices)
    n_leaves = len(leaf_indices)
    receipt = MatrixReceipt(
        schema=MATRIX_RECEIPT_SCHEMA,
        valid=True,
        tree_id=topology.tree_id,
        metric=metric_name,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        n_nodes=len(expected_ids),
        n_internal=len(expected_ids) - n_leaves,
        n_leaves=n_leaves,
        topology_sha256=topology.topology_sha256,
        pair_index_sha256=topology.pair_index_sha256,
        ordered_node_ids=expected_ids,
        ordered_node_ids_sha256=canonical_sha256(list(expected_ids)),
        matrix_shape=(len(expected_ids), len(expected_ids)),
        matrix_sha256=canonical_sha256([list(row) for row in values]),
        leaf_sensitivity_indices=leaf_indices,
        leaf_sensitivity_ids=leaf_ids,
        leaf_subset_sha256=topology.leaf_subset_sha256,
        leaf_matrix_shape=(n_leaves, n_leaves),
        leaf_sensitivity_matrix_sha256=canonical_sha256(
            [list(row) for row in leaf_values]
        ),
    )
    return ValidatedDistanceMatrix(
        matrix=values,
        leaf_sensitivity_matrix=leaf_values,
        receipt=receipt,
    )


def validate_matrix_receipt(
    payload: Mapping[str, Any],
    topology: TreeTopology,
    *,
    metric: str,
    ordered_node_ids: Sequence[str],
    matrix: Sequence[Sequence[object]],
) -> ValidatedDistanceMatrix:
    """Recompute the complete receipt and reject any saved-field drift."""

    if not isinstance(payload, Mapping):
        raise ValueError("matrix receipt must be a mapping")
    artifact = validate_distance_matrix(
        topology,
        metric=metric,
        ordered_node_ids=ordered_node_ids,
        matrix=matrix,
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(artifact.receipt.to_dict()):
        raise ValueError("matrix receipt differs from its recomputed contract")
    return artifact


def _matrix(
    raw: Sequence[Sequence[object]], *, expected_size: int
) -> tuple[tuple[float, ...], ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("distance matrix must be a sequence of rows")
    if len(raw) != expected_size:
        raise ValueError(
            f"distance matrix row count {len(raw)} differs from topology size {expected_size}"
        )
    rows: list[tuple[float, ...]] = []
    for row_index, raw_row in enumerate(raw):
        if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
            raise ValueError(f"distance matrix row {row_index} is not a sequence")
        if len(raw_row) != expected_size:
            raise ValueError(
                f"distance matrix row {row_index} has length {len(raw_row)}, "
                f"expected {expected_size}"
            )
        row: list[float] = []
        for column_index, raw_value in enumerate(raw_row):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(
                    f"distance matrix value [{row_index},{column_index}] is not numeric"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"distance matrix value [{row_index},{column_index}] is not finite"
                )
            if value < 0.0:
                raise ValueError(
                    f"distance matrix value [{row_index},{column_index}] is negative"
                )
            row.append(0.0 if value == 0.0 else value)
        rows.append(tuple(row))
    values = tuple(rows)
    for left in range(expected_size):
        if values[left][left] != 0.0:
            raise ValueError(f"distance matrix diagonal [{left},{left}] must be zero")
        for right in range(left + 1, expected_size):
            if values[left][right] != values[right][left]:
                raise ValueError(
                    f"distance matrix is not exactly symmetric at [{left},{right}]"
                )
    return values


def _validate_leaf_indices(indices: tuple[int, ...], *, size: int) -> None:
    if not indices:
        raise ValueError("topology has no leaf sensitivity indices")
    previous = -1
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("leaf sensitivity indices must be integers")
        if index <= previous or index < 0 or index >= size:
            raise ValueError("leaf sensitivity indices must be unique, increasing, and in range")
        previous = index


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{context} must be nonempty normalized text")
    return value


def _sha256(value: object, context: str) -> str:
    text = _text(value, context)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return text
