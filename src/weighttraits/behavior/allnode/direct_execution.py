"""Versioned direct-distance execution for observed trained ancestors.

This boundary deliberately stops at distance matrices.  It never invokes
neighbor joining, split recovery, or any routine that interprets trained
internal checkpoints as terminal taxa.  Checkpoint readers and the pinned
streaming distance implementation are dependency-injected through an explicit
source-pinned adapter, then their results are revalidated against the isolated
all-node contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    NodeCheckpointBinding,
    TreeCheckpointManifest,
    read_tree_checkpoint_manifest,
)
from .manifest import TreeTopology, canonical_sha256, read_tree_topology
from .matrices import ValidatedDistanceMatrix, validate_distance_matrix
from .covariates import TreeNodeCovariates, validate_tree_node_covariates
from .resolver import ResolvedAllNodeTree


DIRECT_METRICS = ("l2", "cosine", "correlation")
DIRECT_EXECUTION_SCHEMA = "weighttraits.all_node_direct_execution.v1"
DIRECT_VALUES_SCHEMA = "weighttraits.all_node_distance_values.v1"
DIRECT_ARTIFACT_RECEIPT_SCHEMA = (
    "weighttraits.all_node_distance_artifact_receipt.v1"
)
DIRECT_ARTIFACT_PLAN_SCHEMA = "weighttraits.all_node_direct_artifact_plan.v1"
DISTANCE_ENGINE = "pinned_streaming_sufficient_stats"
LAYER_AGGREGATION = "unweighted_mean"
REPRESENTATION = "full_weight"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MISSING = object()


class TensorReader(Protocol):
    """Structural subset of the frozen WeightTraits tensor-reader contract."""

    model_id: str

    def close(self) -> None:
        """Release reader resources; implementations must be idempotent."""


@dataclass(frozen=True)
class LoadedTensorReader:
    """Reader plus the checkpoint identity the adapter says it opened."""

    node_id: str
    checkpoint: ArtifactPin
    reader: TensorReader


@dataclass(frozen=True)
class PinnedStreamingAdapter:
    """Explicit bridge to externally hash-verified frozen source functions."""

    adapter_id: str
    adapter_version: str
    snapshot_root: str
    readers_source: ArtifactPin
    streaming_source: ArtifactPin
    source_pins_verified: bool
    backend_closes_readers: bool
    open_reader: Callable[[NodeCheckpointBinding], LoadedTensorReader]
    build_distance_cube: Callable[..., object]

    def provenance_dict(self) -> dict[str, Any]:
        core = {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "snapshot_root": self.snapshot_root,
            "readers_source": self.readers_source.to_dict(),
            "streaming_source": self.streaming_source.to_dict(),
            "source_pins_verified": self.source_pins_verified,
            "backend_closes_readers": self.backend_closes_readers,
            "reader_protocol": "weighttraits.distances.readers.TensorReader",
            "distance_backend": "weighttraits.distances.streaming.build_distance_cube",
        }
        return {**core, "adapter_sha256": canonical_sha256(core)}


def pinned_snapshot_adapter(
    *,
    snapshot_root: str,
    readers_source: ArtifactPin,
    streaming_source: ArtifactPin,
    reader_from_path: Callable[..., TensorReader],
    build_distance_cube: Callable[..., object],
    source_pins_verified: bool,
    adapter_id: str = "weighttraits_snapshot_streaming",
    adapter_version: str = "v1",
) -> PinnedStreamingAdapter:
    """Adapt the two safe public frozen helpers without importing the snapshot.

    The caller must hash the source files before construction and supply those
    observed pins.  The adapter opens exactly each already-validated binding's
    checkpoint path with ``model_id=node_id``.
    """

    def open_binding(binding: NodeCheckpointBinding) -> LoadedTensorReader:
        reader = reader_from_path(
            binding.checkpoint.path,
            model_id=binding.node_id,
        )
        # Frozen reader_from_path prefers PEFT adapter files when a directory
        # contains both adapter and merged/full artifacts.  That representation
        # is wrong for bindings explicitly resolved as model/merged weights.
        if type(reader).__name__ in {"LoraFactorReader", "CumulativeLoraReader"}:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
            raise ValueError(
                "model/merged checkpoint resolved to a LoRA-delta reader; "
                "bind the concrete full-weight artifact path"
            )
        return LoadedTensorReader(
            node_id=binding.node_id,
            checkpoint=binding.checkpoint,
            reader=reader,
        )

    adapter = PinnedStreamingAdapter(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        snapshot_root=snapshot_root,
        readers_source=readers_source,
        streaming_source=streaming_source,
        source_pins_verified=source_pins_verified,
        # Frozen build_distance_cube closes readers on successful return.  The
        # execution boundary still closes every opened reader if it raises.
        backend_closes_readers=True,
        open_reader=open_binding,
        build_distance_cube=build_distance_cube,
    )
    _validate_adapter(adapter)
    return adapter


@dataclass(frozen=True)
class DirectMetricResult:
    metric: str
    layer_cube: tuple[tuple[tuple[float, ...], ...], ...]
    layer_cube_sha256: str
    matrix: ValidatedDistanceMatrix


@dataclass(frozen=True)
class DirectExecutionReceipt:
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class DirectExecution:
    metric_results: tuple[DirectMetricResult, ...]
    receipt: DirectExecutionReceipt

    def result(self, metric: str) -> DirectMetricResult:
        for result in self.metric_results:
            if result.metric == metric:
                return result
        raise KeyError(metric)


@dataclass(frozen=True)
class ImmutableArtifactWrite:
    path: str
    content: bytes
    content_sha256: str
    size_bytes: int
    exclusive_create: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "exclusive_create": self.exclusive_create,
        }


@dataclass(frozen=True)
class DirectArtifactPlan:
    schema: str
    valid: bool
    execution_sha256: str
    node_covariates_sha256: str
    overwrite: bool
    atomic_set_required: bool
    scheduler_actions: int
    writes: tuple[ImmutableArtifactWrite, ...]
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "valid": self.valid,
            "execution_sha256": self.execution_sha256,
            "node_covariates_sha256": self.node_covariates_sha256,
            "overwrite": self.overwrite,
            "atomic_set_required": self.atomic_set_required,
            "scheduler_actions": self.scheduler_actions,
            "writes": [write.to_dict() for write in self.writes],
            "plan_sha256": self.plan_sha256,
        }


def execute_direct_distances(
    resolved: ResolvedAllNodeTree,
    *,
    adapter: PinnedStreamingAdapter,
    metrics: Sequence[str] = DIRECT_METRICS,
    chunk_size: int = 1_000_000,
) -> DirectExecution:
    """Compute one tree's topology-ordered all-node direct matrices."""

    verified = _verify_resolved_tree(resolved)
    return execute_manifest_direct_distances(
        topology=verified.topology,
        checkpoint_manifest=verified.checkpoint_manifest,
        node_covariates=verified.node_covariates,
        cohort_key=verified.cohort_key,
        training_cohort_id=verified.training_cohort_id,
        resolution_sha256=verified.resolution_sha256,
        adapter=adapter,
        metrics=metrics,
        chunk_size=chunk_size,
    )


def execute_manifest_direct_distances(
    *,
    topology: TreeTopology,
    checkpoint_manifest: TreeCheckpointManifest,
    node_covariates: TreeNodeCovariates,
    cohort_key: str,
    training_cohort_id: str,
    resolution_sha256: str,
    adapter: PinnedStreamingAdapter,
    metrics: Sequence[str] = DIRECT_METRICS,
    chunk_size: int = 1_000_000,
) -> DirectExecution:
    """Execute from separately supplied, fully validated topology/manifest data."""

    tree, manifest = _verify_manifest_scope(
        topology,
        checkpoint_manifest,
        training_cohort_id=training_cohort_id,
    )
    covariates = validate_tree_node_covariates(
        node_covariates, tree, manifest
    )
    key = _text(cohort_key, "cohort_key")
    training_cohort = _text(training_cohort_id, "training_cohort_id")
    resolution_digest = _sha256(resolution_sha256, "resolution_sha256")
    expected_resolution = canonical_sha256(
        {
            "cohort_key": key,
            "training_cohort_id": training_cohort,
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "topology_sha256": tree.topology_sha256,
            "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
            "node_covariates_sha256": covariates.covariates_sha256,
        }
    )
    if resolution_digest != expected_resolution:
        raise ValueError("resolution_sha256 differs from recomputed manifest scope")
    metric_names = _metrics(metrics)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    adapter_row = _validate_adapter(adapter)

    loaded: list[LoadedTensorReader] = []
    cube_returned = False
    try:
        for binding in manifest.bindings:
            opened = adapter.open_reader(binding)
            try:
                verified_reader = _validate_loaded_reader(opened, binding)
            except Exception:
                # A structurally invalid adapter return may still own a live
                # handle, so close it before failing the boundary.
                reader = getattr(opened, "reader", None)
                close = getattr(reader, "close", None)
                if callable(close):
                    close()
                raise
            loaded.append(verified_reader)
        readers = [item.reader for item in loaded]
        cube = adapter.build_distance_cube(
            readers,
            metrics=list(metric_names),
            chunk_size=chunk_size,
            representation=REPRESENTATION,
        )
        cube_returned = True
        normalized = _normalize_cube(
            cube,
            topology=tree,
            metrics=metric_names,
        )
    finally:
        # The pinned backend closes only after a successful cube return.  Other
        # adapters declare their behavior explicitly.
        if not cube_returned or not adapter.backend_closes_readers:
            _close_readers(loaded)

    results: list[DirectMetricResult] = []
    metric_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        layers = normalized["distances"][metric]
        mean_matrix = _mean_layers(layers)
        validated = validate_distance_matrix(
            tree,
            metric=metric,
            ordered_node_ids=tree.ordered_node_ids,
            matrix=mean_matrix,
        )
        cube_sha = canonical_sha256(
            [[list(row) for row in layer] for layer in layers]
        )
        results.append(
            DirectMetricResult(
                metric=metric,
                layer_cube=layers,
                layer_cube_sha256=cube_sha,
                matrix=validated,
            )
        )
        metric_rows.append(
            {
                "metric": metric,
                "layer_cube_sha256": cube_sha,
                "matrix_receipt": validated.receipt.to_dict(),
                "matrix_receipt_sha256": canonical_sha256(
                    validated.receipt.to_dict()
                ),
            }
        )

    checkpoint_scope = [
        {
            "node_ordinal": binding.node_ordinal,
            "node_id": binding.node_id,
            "node_role": binding.node_role,
            "checkpoint": binding.checkpoint.to_dict(),
        }
        for binding in manifest.bindings
    ]
    core = {
        "distance_engine": DISTANCE_ENGINE,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "direct_matrix_only": True,
        "neighbor_joining": False,
        "tree_builder": None,
        "internal_nodes_as_tips": False,
        "cohort_key": key,
        "training_cohort_id": training_cohort,
        "tree_id": tree.tree_id,
        "resolution_sha256": resolution_digest,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "node_covariates": covariates.to_dict(),
        "node_covariates_sha256": covariates.covariates_sha256,
        "topology_sha256": tree.topology_sha256,
        "pair_index_sha256": tree.pair_index_sha256,
        "leaf_subset_sha256": tree.leaf_subset_sha256,
        "ordered_node_ids": list(tree.ordered_node_ids),
        "ordered_node_ids_sha256": canonical_sha256(list(tree.ordered_node_ids)),
        "n_nodes": tree.n_nodes,
        "n_internal": tree.n_internal,
        "n_leaves": tree.n_leaves,
        "metrics": list(metric_names),
        "chunk_size": chunk_size,
        "representation": REPRESENTATION,
        "layer_aggregation": LAYER_AGGREGATION,
        "layer_names": list(normalized["layer_names"]),
        "layer_scope_sha256": normalized["layer_scope_sha256"],
        "backend_audit_sha256": normalized["backend_audit_sha256"],
        "adapter": adapter_row,
        "checkpoint_scope": checkpoint_scope,
        "checkpoint_scope_sha256": canonical_sha256(checkpoint_scope),
        "metric_results": metric_rows,
    }
    payload = {
        "schema": DIRECT_EXECUTION_SCHEMA,
        "valid": True,
        **core,
        "execution_sha256": canonical_sha256(core),
    }
    return DirectExecution(
        metric_results=tuple(results),
        receipt=DirectExecutionReceipt(payload=payload),
    )


def plan_direct_artifacts(
    execution: DirectExecution,
    *,
    output_dir: str,
) -> DirectArtifactPlan:
    """Plan an exclusive, atomic set of matrix and receipt JSON artifacts."""

    verified = _verify_execution(execution)
    destination = _output_dir(output_dir)
    tree_id = _filename_component(verified.receipt.payload["tree_id"], "tree_id")
    writes: list[ImmutableArtifactWrite] = []
    for result in verified.metric_results:
        matrix_path = os.path.join(
            destination,
            f"{tree_id}.{result.metric}.all_nodes.matrix.json",
        )
        values = {
            "schema": DIRECT_VALUES_SCHEMA,
            "tree_id": tree_id,
            "metric": result.metric,
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "node_covariates_sha256": verified.receipt.payload[
                "node_covariates_sha256"
            ],
            "topology_sha256": result.matrix.receipt.topology_sha256,
            "pair_index_sha256": result.matrix.receipt.pair_index_sha256,
            "ordered_node_ids": list(result.matrix.receipt.ordered_node_ids),
            "matrix": [list(row) for row in result.matrix.matrix],
            "leaf_sensitivity_indices": list(
                result.matrix.receipt.leaf_sensitivity_indices
            ),
            "leaf_sensitivity_ids": list(
                result.matrix.receipt.leaf_sensitivity_ids
            ),
            "leaf_sensitivity_matrix": [
                list(row) for row in result.matrix.leaf_sensitivity_matrix
            ],
        }
        matrix_write = _write(matrix_path, values)
        writes.append(matrix_write)

        receipt_path = os.path.join(
            destination,
            f"{tree_id}.{result.metric}.all_nodes.receipt.json",
        )
        receipt_core = {
            "matrix_receipt": result.matrix.receipt.to_dict(),
            "matrix_receipt_sha256": canonical_sha256(
                result.matrix.receipt.to_dict()
            ),
            "matrix_artifact": matrix_write.to_dict(),
            "execution_sha256": verified.receipt.payload["execution_sha256"],
            "node_covariates_sha256": verified.receipt.payload[
                "node_covariates_sha256"
            ],
        }
        receipt_payload = {
            "schema": DIRECT_ARTIFACT_RECEIPT_SCHEMA,
            "valid": True,
            **receipt_core,
            "artifact_receipt_sha256": canonical_sha256(receipt_core),
        }
        writes.append(_write(receipt_path, receipt_payload))

    execution_path = os.path.join(
        destination,
        f"{tree_id}.direct_execution.receipt.json",
    )
    writes.append(_write(execution_path, verified.receipt.to_dict()))
    write_rows = [write.to_dict() for write in writes]
    core = {
        "execution_sha256": verified.receipt.payload["execution_sha256"],
        "node_covariates_sha256": verified.receipt.payload[
            "node_covariates_sha256"
        ],
        "overwrite": False,
        "atomic_set_required": True,
        "scheduler_actions": 0,
        "writes": write_rows,
    }
    plan = DirectArtifactPlan(
        schema=DIRECT_ARTIFACT_PLAN_SCHEMA,
        valid=True,
        execution_sha256=core["execution_sha256"],
        node_covariates_sha256=core["node_covariates_sha256"],
        overwrite=False,
        atomic_set_required=True,
        scheduler_actions=0,
        writes=tuple(writes),
        plan_sha256=canonical_sha256(core),
    )
    return verify_direct_artifact_plan(plan)


def verify_direct_artifact_plan(plan: DirectArtifactPlan) -> DirectArtifactPlan:
    """Re-hash all planned bytes and reject mutable/overwrite semantics."""

    if not isinstance(plan, DirectArtifactPlan):
        raise ValueError("artifact plan must be a DirectArtifactPlan")
    if (
        plan.schema != DIRECT_ARTIFACT_PLAN_SCHEMA
        or plan.valid is not True
        or plan.overwrite is not False
        or plan.atomic_set_required is not True
        or plan.scheduler_actions != 0
    ):
        raise ValueError("direct artifact plan control fields drift")
    _sha256(plan.execution_sha256, "execution_sha256")
    _sha256(plan.node_covariates_sha256, "node_covariates_sha256")
    if not plan.writes:
        raise ValueError("direct artifact plan has no writes")
    paths: list[str] = []
    for write in plan.writes:
        if not isinstance(write, ImmutableArtifactWrite):
            raise ValueError("direct artifact plan contains an invalid write")
        if write.exclusive_create is not True:
            raise ValueError("all direct artifacts must use exclusive creation")
        path = _absolute_normalized_path(write.path, "artifact path")
        if path == "/":
            raise ValueError("artifact path must not be filesystem root")
        if not isinstance(write.content, bytes):
            raise ValueError("artifact content must be bytes")
        digest = hashlib.sha256(write.content).hexdigest()
        if digest != write.content_sha256 or len(write.content) != write.size_bytes:
            raise ValueError("planned artifact bytes differ from their digest or size")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("direct artifact plan reuses an output path")
    core = {
        "execution_sha256": plan.execution_sha256,
        "node_covariates_sha256": plan.node_covariates_sha256,
        "overwrite": plan.overwrite,
        "atomic_set_required": plan.atomic_set_required,
        "scheduler_actions": plan.scheduler_actions,
        "writes": [write.to_dict() for write in plan.writes],
    }
    if canonical_sha256(core) != plan.plan_sha256:
        raise ValueError("direct artifact plan hash differs from recomputed contract")
    return plan


def _verify_resolved_tree(value: ResolvedAllNodeTree) -> ResolvedAllNodeTree:
    if not isinstance(value, ResolvedAllNodeTree):
        raise ValueError("resolved must be a ResolvedAllNodeTree")
    tree, manifest = _verify_manifest_scope(
        value.topology,
        value.checkpoint_manifest,
        training_cohort_id=value.training_cohort_id,
    )
    covariates = validate_tree_node_covariates(
        value.node_covariates, tree, manifest
    )
    if (
        value.population != POPULATION
        or value.internal_node_treatment != INTERNAL_NODE_TREATMENT
    ):
        raise ValueError("resolved population or internal-node treatment drift")
    core = {
        "cohort_key": _text(value.cohort_key, "cohort_key"),
        "training_cohort_id": _text(
            value.training_cohort_id, "training_cohort_id"
        ),
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "topology_sha256": tree.topology_sha256,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "node_covariates_sha256": covariates.covariates_sha256,
    }
    if _sha256(value.resolution_sha256, "resolution_sha256") != canonical_sha256(core):
        raise ValueError("resolved tree hash differs from recomputed contract")
    return value


def _verify_manifest_scope(
    topology: TreeTopology,
    manifest: TreeCheckpointManifest,
    *,
    training_cohort_id: str,
) -> tuple[TreeTopology, TreeCheckpointManifest]:
    if not isinstance(topology, TreeTopology):
        raise ValueError("topology must be a TreeTopology")
    if not isinstance(manifest, TreeCheckpointManifest):
        raise ValueError("checkpoint_manifest must be a TreeCheckpointManifest")
    tree = read_tree_topology(topology.to_dict())
    checkpoints = read_tree_checkpoint_manifest(manifest.to_dict())
    cohort = _text(training_cohort_id, "training_cohort_id")
    if checkpoints.cohort_id != cohort:
        raise ValueError("checkpoint manifest training cohort identity drift")
    if checkpoints.tree_id != tree.tree_id:
        raise ValueError("checkpoint manifest tree identity differs from topology")
    for field in (
        "topology_sha256",
        "nodes_sha256",
        "leaf_subset_sha256",
        "pair_index_sha256",
    ):
        if getattr(checkpoints, field) != getattr(tree, field):
            raise ValueError(f"checkpoint manifest {field} differs from topology")
    if tuple(binding.node_id for binding in checkpoints.bindings) != tree.ordered_node_ids:
        raise ValueError("checkpoint bindings differ from topology order")
    if tree.root_id in tree.ordered_node_ids or any(
        binding.node_id == tree.root_id for binding in checkpoints.bindings
    ):
        raise ValueError("untrained root appears in direct-distance population")
    if any(binding.artifact_name not in {"model", "merged"} for binding in checkpoints.bindings):
        raise ValueError("direct full-weight endpoint requires model or merged artifacts")
    return tree, checkpoints


def _validate_adapter(adapter: PinnedStreamingAdapter) -> dict[str, Any]:
    if not isinstance(adapter, PinnedStreamingAdapter):
        raise ValueError("adapter must be a PinnedStreamingAdapter")
    _text(adapter.adapter_id, "adapter_id")
    _text(adapter.adapter_version, "adapter_version")
    root = _absolute_normalized_path(adapter.snapshot_root, "snapshot_root")
    if root == "/":
        raise ValueError("snapshot_root must be bounded")
    for label, pin in (
        ("readers_source", adapter.readers_source),
        ("streaming_source", adapter.streaming_source),
    ):
        _artifact_pin(pin, label)
        try:
            inside = os.path.commonpath((root, pin.path)) == root
        except ValueError as exc:
            raise ValueError(f"{label} is not under snapshot_root") from exc
        if not inside or pin.path == root:
            raise ValueError(f"{label} is not under snapshot_root")
    if adapter.source_pins_verified is not True:
        raise ValueError("snapshot source pins must be externally verified")
    if not isinstance(adapter.backend_closes_readers, bool):
        raise ValueError("backend_closes_readers must be boolean")
    if not callable(adapter.open_reader) or not callable(adapter.build_distance_cube):
        raise ValueError("adapter callables are missing")
    row = adapter.provenance_dict()
    expected = canonical_sha256({key: value for key, value in row.items() if key != "adapter_sha256"})
    if row["adapter_sha256"] != expected:
        raise ValueError("adapter provenance hash drift")
    return row


def _validate_loaded_reader(
    loaded: LoadedTensorReader,
    binding: NodeCheckpointBinding,
) -> LoadedTensorReader:
    if not isinstance(loaded, LoadedTensorReader):
        raise ValueError("adapter must return LoadedTensorReader")
    if loaded.node_id != binding.node_id:
        raise ValueError("adapter loaded reader node identity drift")
    _artifact_pin(loaded.checkpoint, "loaded checkpoint")
    if loaded.checkpoint != binding.checkpoint:
        raise ValueError("adapter loaded checkpoint pin drift")
    if not hasattr(loaded.reader, "close") or not callable(loaded.reader.close):
        raise ValueError("loaded reader lacks close()")
    if getattr(loaded.reader, "model_id", None) != binding.node_id:
        raise ValueError("loaded reader model_id differs from topology node ID")
    return loaded


def _normalize_cube(
    cube: object,
    *,
    topology: TreeTopology,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    model_ids = tuple(
        _text(item, "cube model_id") for item in _field(cube, "model_ids")
    )
    if model_ids != topology.ordered_node_ids:
        raise ValueError("distance cube model IDs differ from topology order")
    layer_names = tuple(
        _text(item, "cube layer name") for item in _field(cube, "layer_names")
    )
    if not layer_names or len(layer_names) != len(set(layer_names)):
        raise ValueError("distance cube layer names must be nonempty and unique")
    distances = _field(cube, "distances")
    if not isinstance(distances, Mapping) or set(distances) != set(metrics):
        raise ValueError("distance cube metrics differ from exact direct contract")
    normalized: dict[str, tuple[tuple[tuple[float, ...], ...], ...]] = {}
    for metric in metrics:
        raw_layers = distances[metric]
        if not _sequence(raw_layers) or len(raw_layers) != len(layer_names):
            raise ValueError(f"distance cube {metric!r} layer count drift")
        normalized[metric] = tuple(
            _normalize_layer_matrix(layer, size=topology.n_nodes)
            for layer in raw_layers
        )

    audit = _field(cube, "audit")
    if not isinstance(audit, Mapping):
        raise ValueError("distance cube audit must be a mapping")
    required = {
        "representation": REPRESENTATION,
        "n_models": topology.n_nodes,
        "n_layers": len(layer_names),
    }
    for field, expected in required.items():
        if audit.get(field) != expected:
            raise ValueError(f"distance cube audit {field} drift")
    audit_ids = tuple(audit.get("model_ids", ()))
    if audit_ids != topology.ordered_node_ids:
        raise ValueError("distance cube audit model IDs drift")
    audit_metrics = tuple(audit.get("metrics", ()))
    if set(audit_metrics) != set(metrics) or len(audit_metrics) != len(metrics):
        raise ValueError("distance cube audit metrics drift")
    layers = audit.get("layers")
    if not isinstance(layers, list) or len(layers) != len(layer_names):
        raise ValueError("distance cube audit layer scope drift")
    if tuple(row.get("name") for row in layers if isinstance(row, Mapping)) != layer_names:
        raise ValueError("distance cube audit layer names drift")
    try:
        audit_sha = canonical_sha256(dict(audit))
        layer_sha = canonical_sha256(layers)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance cube audit is not canonical JSON") from exc
    return {
        "model_ids": model_ids,
        "layer_names": layer_names,
        "distances": normalized,
        "backend_audit_sha256": audit_sha,
        "layer_scope_sha256": layer_sha,
    }


def _normalize_layer_matrix(
    raw: object,
    *,
    size: int,
    tolerance: float = 1e-9,
) -> tuple[tuple[float, ...], ...]:
    if not _sequence(raw) or len(raw) != size:
        raise ValueError("distance cube layer matrix row count drift")
    rows: list[list[float]] = []
    for row_index, raw_row in enumerate(raw):
        if not _sequence(raw_row) or len(raw_row) != size:
            raise ValueError(f"distance cube layer row {row_index} shape drift")
        row: list[float] = []
        for value in raw_row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                # numpy scalar floats support float() but not necessarily the
                # numeric ABCs; accept them only if conversion is exact/finite.
                try:
                    converted = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("distance cube contains a nonnumeric value") from exc
            else:
                converted = float(value)
            if not math.isfinite(converted):
                raise ValueError("distance cube contains a nonfinite value")
            if converted < -tolerance:
                raise ValueError("distance cube contains a materially negative value")
            row.append(0.0 if abs(converted) <= tolerance else converted)
        rows.append(row)
    for left in range(size):
        if abs(rows[left][left]) > tolerance:
            raise ValueError("distance cube layer diagonal is not zero")
        rows[left][left] = 0.0
        for right in range(left + 1, size):
            if abs(rows[left][right] - rows[right][left]) > tolerance:
                raise ValueError("distance cube layer is not symmetric")
            mean = (rows[left][right] + rows[right][left]) / 2.0
            value = 0.0 if abs(mean) <= tolerance else mean
            rows[left][right] = rows[right][left] = value
    return tuple(tuple(row) for row in rows)


def _mean_layers(
    layers: tuple[tuple[tuple[float, ...], ...], ...],
) -> tuple[tuple[float, ...], ...]:
    if not layers:
        raise ValueError("cannot aggregate an empty layer cube")
    size = len(layers[0])
    count = float(len(layers))
    return tuple(
        tuple(sum(layer[left][right] for layer in layers) / count for right in range(size))
        for left in range(size)
    )


def _verify_execution(value: DirectExecution) -> DirectExecution:
    if not isinstance(value, DirectExecution):
        raise ValueError("execution must be a DirectExecution")
    payload = value.receipt.to_dict()
    if payload.get("schema") != DIRECT_EXECUTION_SCHEMA or payload.get("valid") is not True:
        raise ValueError("direct execution receipt schema or validity drift")
    metrics = tuple(payload.get("metrics", ()))
    if metrics != DIRECT_METRICS:
        raise ValueError("direct execution receipt metric scope drift")
    results = {result.metric: result for result in value.metric_results}
    if tuple(result.metric for result in value.metric_results) != DIRECT_METRICS:
        raise ValueError("direct execution metric results drift")
    receipt_rows = payload.get("metric_results")
    if not isinstance(receipt_rows, list) or len(receipt_rows) != len(DIRECT_METRICS):
        raise ValueError("direct execution metric receipt scope drift")
    for row in receipt_rows:
        if not isinstance(row, Mapping) or row.get("metric") not in results:
            raise ValueError("direct execution metric receipt is invalid")
        result = results[row["metric"]]
        cube_sha = canonical_sha256(
            [[list(values) for values in layer] for layer in result.layer_cube]
        )
        if result.layer_cube_sha256 != cube_sha or row.get("layer_cube_sha256") != cube_sha:
            raise ValueError("direct execution layer cube hash drift")
        _verify_metric_result(result)
        matrix_receipt = result.matrix.receipt.to_dict()
        if row.get("matrix_receipt") != matrix_receipt or row.get(
            "matrix_receipt_sha256"
        ) != canonical_sha256(matrix_receipt):
            raise ValueError("direct execution matrix receipt drift")
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"schema", "valid", "execution_sha256"}
    }
    if payload.get("execution_sha256") != canonical_sha256(core):
        raise ValueError("direct execution hash differs from recomputed receipt")
    return value


def _verify_metric_result(result: DirectMetricResult) -> None:
    if not isinstance(result, DirectMetricResult) or result.metric not in DIRECT_METRICS:
        raise ValueError("direct metric result is invalid")
    receipt = result.matrix.receipt
    if receipt.metric != result.metric or receipt.population != POPULATION or (
        receipt.internal_node_treatment != INTERNAL_NODE_TREATMENT
    ):
        raise ValueError("direct metric matrix receipt scope drift")
    size = len(receipt.ordered_node_ids)
    matrix = _normalize_layer_matrix(result.matrix.matrix, size=size, tolerance=0.0)
    if matrix != result.matrix.matrix:
        raise ValueError("direct metric matrix differs from its normalized values")
    if receipt.matrix_shape != (size, size) or receipt.matrix_sha256 != canonical_sha256(
        [list(row) for row in matrix]
    ):
        raise ValueError("direct metric matrix hash or shape drift")
    indices = receipt.leaf_sensitivity_indices
    if (
        not indices
        or tuple(sorted(set(indices))) != indices
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= size for index in indices)
    ):
        raise ValueError("direct metric leaf sensitivity indices drift")
    leaf = tuple(tuple(matrix[left][right] for right in indices) for left in indices)
    if leaf != result.matrix.leaf_sensitivity_matrix:
        raise ValueError("direct metric leaf sensitivity slice drift")
    if receipt.leaf_matrix_shape != (len(indices), len(indices)) or (
        receipt.leaf_sensitivity_matrix_sha256
        != canonical_sha256([list(row) for row in leaf])
    ):
        raise ValueError("direct metric leaf matrix hash or shape drift")


def _metrics(values: Sequence[str]) -> tuple[str, ...]:
    if not _sequence(values):
        raise ValueError("metrics must be a sequence")
    observed = tuple(_text(value, "metric") for value in values)
    if len(observed) != len(set(observed)) or set(observed) != set(DIRECT_METRICS):
        raise ValueError(f"metrics must be exactly {DIRECT_METRICS!r}")
    # Freeze canonical order independently of caller ordering.
    return DIRECT_METRICS


def _close_readers(values: Sequence[LoadedTensorReader]) -> None:
    first_error: Exception | None = None
    for loaded in values:
        try:
            loaded.reader.close()
        except Exception as exc:  # pragma: no cover - adapter cleanup boundary
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _write(path: str, payload: Mapping[str, Any]) -> ImmutableArtifactWrite:
    content = _canonical_bytes(payload)
    return ImmutableArtifactWrite(
        path=_absolute_normalized_path(path, "artifact path"),
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        exclusive_create=True,
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact payload is not canonical JSON") from exc


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        observed = value.get(name, _MISSING)
    else:
        observed = getattr(value, name, _MISSING)
    if observed is _MISSING:
        raise ValueError(f"distance cube lacks {name!r}")
    return observed


def _artifact_pin(value: ArtifactPin, label: str) -> ArtifactPin:
    if not isinstance(value, ArtifactPin):
        raise ValueError(f"{label} must be an ArtifactPin")
    _absolute_normalized_path(value.path, f"{label}.path")
    _sha256(value.sha256, f"{label}.sha256")
    return value


def _output_dir(value: object) -> str:
    path = _absolute_normalized_path(value, "output_dir")
    if path == "/":
        raise ValueError("output_dir must be bounded")
    return path


def _absolute_normalized_path(value: object, label: str) -> str:
    text = _text(value, label)
    if not os.path.isabs(text) or os.path.normpath(text) != text:
        raise ValueError(f"{label} must be an absolute normalized path")
    return text


def _filename_component(value: object, label: str) -> str:
    text = _text(value, label)
    if _COMPONENT.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError(f"{label} is not a safe filename component")
    return text


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty normalized text")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text


def _sequence(value: object) -> bool:
    return not isinstance(value, (str, bytes)) and hasattr(value, "__len__") and hasattr(
        value, "__iter__"
    )
