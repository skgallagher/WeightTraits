"""Fail-closed checkpoint bindings for the ancestor-aware all-node endpoint.

This module deliberately does not import the legacy leaf resolver.  Every
checkpoint is bound to one validated :class:`TreeTopology` node in canonical
order, and trained internal nodes remain observed ancestors throughout.

The contracts validate supplied paths and digests but do not inspect or hash
the filesystem.  Deployment code must verify the pinned bytes separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Any, Mapping, Sequence

from .manifest import NodeSpec, TreeTopology, build_tree_topology, canonical_sha256


POPULATION = "all_trained_nodes"
INTERNAL_NODE_TREATMENT = "observed_ancestors"
COMPLETED_STATUS = "completed"
PRODUCTION_N_TREES = 50
PRODUCTION_N_NODES = 641
PRODUCTION_N_INTERNAL = 276
PRODUCTION_N_LEAVES = 365
PRODUCTION_COHORT_IDS = (
    "llama_full",
    "llama_notrans",
    "flan_full",
    "flan_notrans",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_METHOD_ARTIFACT = {"full": "model", "lora": "merged"}


@dataclass(frozen=True)
class ArtifactPin:
    """One absolute path and its externally verified SHA-256."""

    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class NodeCheckpointBinding:
    """One completed checkpoint bound to one observed topology node."""

    population: str
    internal_node_treatment: str
    cohort_id: str
    tree_id: str
    node_ordinal: int
    array_index: int
    node_id: str
    parent_id: str
    depth: int
    is_leaf: bool
    node_role: str
    status: str
    method: str
    artifact_name: str
    checkpoint: ArtifactPin
    base_model_id: str
    base_model_revision: str
    run_list: ArtifactPin
    ledger: ArtifactPin
    truth_manifest: ArtifactPin
    top_config: ArtifactPin
    training_summary: ArtifactPin
    completion_receipt: ArtifactPin

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for field in (
            "checkpoint",
            "run_list",
            "ledger",
            "truth_manifest",
            "top_config",
            "training_summary",
            "completion_receipt",
        ):
            row[field] = getattr(self, field).to_dict()
        return row


@dataclass(frozen=True)
class TreeCheckpointManifest:
    """Exact ordered checkpoint population for one cohort/tree."""

    cohort_id: str
    tree_id: str
    population: str
    internal_node_treatment: str
    root_id: str
    n_nodes: int
    n_internal: int
    n_leaves: int
    topology_sha256: str
    nodes_sha256: str
    leaf_subset_sha256: str
    pair_index_sha256: str
    bindings: tuple[NodeCheckpointBinding, ...]
    bindings_sha256: str
    checkpoint_set_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "weighttraits.all_node_checkpoint_manifest.v1",
            "population": self.population,
            "internal_node_treatment": self.internal_node_treatment,
            "cohort_id": self.cohort_id,
            "tree_id": self.tree_id,
            "root_id": self.root_id,
            "n_nodes": self.n_nodes,
            "n_internal": self.n_internal,
            "n_leaves": self.n_leaves,
            "topology_sha256": self.topology_sha256,
            "nodes_sha256": self.nodes_sha256,
            "leaf_subset_sha256": self.leaf_subset_sha256,
            "pair_index_sha256": self.pair_index_sha256,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "bindings_sha256": self.bindings_sha256,
            "checkpoint_set_sha256": self.checkpoint_set_sha256,
        }


def build_tree_checkpoint_manifest(
    topology: TreeTopology,
    raw_bindings: Sequence[NodeCheckpointBinding | Mapping[str, Any]],
    *,
    cohort_id: str,
) -> TreeCheckpointManifest:
    """Bind every topology member, exactly once and in canonical order."""

    if not isinstance(topology, TreeTopology):
        raise ValueError("topology must be a TreeTopology")
    # Rebuild from primitive node rows so forged dataclass instances are not trusted.
    verified_topology = build_tree_topology(
        topology.tree_id,
        [node.to_dict() for node in topology.nodes],
        root_id=topology.root_id,
    )
    if canonical_sha256(topology.to_dict()) != canonical_sha256(verified_topology.to_dict()):
        raise ValueError("topology differs from its recomputed contract")

    cohort = _text(cohort_id, "cohort_id")
    bindings = tuple(_binding(value) for value in raw_bindings)
    if len(bindings) != verified_topology.n_nodes:
        raise ValueError(
            f"tree {verified_topology.tree_id!r} checkpoint count differs from topology: "
            f"observed={len(bindings)}, expected={verified_topology.n_nodes}"
        )
    for ordinal, (binding, node) in enumerate(zip(bindings, verified_topology.nodes)):
        expected = {
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "cohort_id": cohort,
            "tree_id": verified_topology.tree_id,
            "node_ordinal": ordinal,
            "array_index": node.array_index,
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "is_leaf": node.is_leaf,
            "node_role": node.node_role,
            "status": COMPLETED_STATUS,
        }
        for field, value in expected.items():
            if getattr(binding, field) != value:
                raise ValueError(
                    f"{cohort}/{verified_topology.tree_id}/{node.node_id} {field} drift: "
                    f"expected={value!r}, observed={getattr(binding, field)!r}"
                )

    _require_shared(bindings, "method")
    _require_shared(bindings, "artifact_name")
    _require_shared(bindings, "base_model_id")
    _require_shared(bindings, "base_model_revision")
    for field in (
        "run_list",
        "ledger",
        "truth_manifest",
        "top_config",
        "training_summary",
        "completion_receipt",
    ):
        _require_shared(bindings, field)
    checkpoint_paths = [binding.checkpoint.path for binding in bindings]
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise ValueError("checkpoint paths must be unique within a tree")

    binding_rows = [binding.to_dict() for binding in bindings]
    bindings_sha = canonical_sha256(binding_rows)
    core = {
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "cohort_id": cohort,
        "tree_id": verified_topology.tree_id,
        "root_id": verified_topology.root_id,
        "n_nodes": verified_topology.n_nodes,
        "n_internal": verified_topology.n_internal,
        "n_leaves": verified_topology.n_leaves,
        "topology_sha256": verified_topology.topology_sha256,
        "nodes_sha256": verified_topology.nodes_sha256,
        "leaf_subset_sha256": verified_topology.leaf_subset_sha256,
        "pair_index_sha256": verified_topology.pair_index_sha256,
        "bindings_sha256": bindings_sha,
    }
    return TreeCheckpointManifest(
        cohort_id=cohort,
        tree_id=verified_topology.tree_id,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        root_id=verified_topology.root_id,
        n_nodes=verified_topology.n_nodes,
        n_internal=verified_topology.n_internal,
        n_leaves=verified_topology.n_leaves,
        topology_sha256=verified_topology.topology_sha256,
        nodes_sha256=verified_topology.nodes_sha256,
        leaf_subset_sha256=verified_topology.leaf_subset_sha256,
        pair_index_sha256=verified_topology.pair_index_sha256,
        bindings=bindings,
        bindings_sha256=bindings_sha,
        checkpoint_set_sha256=canonical_sha256(core),
    )


def read_tree_checkpoint_manifest(payload: Mapping[str, Any]) -> TreeCheckpointManifest:
    """Recompute topology, ordering, and all self-hashes from a saved manifest."""

    if not isinstance(payload, Mapping):
        raise ValueError("saved checkpoint manifest must be a mapping")
    if payload.get("schema") != "weighttraits.all_node_checkpoint_manifest.v1":
        raise ValueError("saved checkpoint manifest schema drift")
    rows = payload.get("bindings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("saved checkpoint manifest bindings must be a nonempty list")
    bindings = tuple(_binding(row) for row in rows)
    topology = build_tree_topology(
        payload.get("tree_id"),
        [
            NodeSpec(
                binding.node_ordinal,
                binding.array_index,
                binding.node_id,
                binding.parent_id,
                binding.depth,
                binding.is_leaf,
            )
            for binding in bindings
        ],
        root_id=payload.get("root_id"),
    )
    rebuilt = build_tree_checkpoint_manifest(
        topology,
        bindings,
        cohort_id=payload.get("cohort_id"),
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(rebuilt.to_dict()):
        raise ValueError("saved checkpoint manifest differs from its recomputed contract")
    return rebuilt


def validate_production_checkpoint_manifests(
    cohorts: Mapping[str, Sequence[TreeCheckpointManifest]],
    *,
    expected_cohort_ids: Sequence[str],
    expected_tree_ids: Sequence[str],
) -> dict[str, Any]:
    """Require the exact 50-tree/641-node production checkpoint population."""

    cohort_ids = tuple(_text(value, "cohort_id") for value in expected_cohort_ids)
    if cohort_ids != PRODUCTION_COHORT_IDS:
        raise ValueError(
            f"production cohort IDs must be exactly {PRODUCTION_COHORT_IDS!r}"
        )
    if set(cohorts) != set(cohort_ids):
        raise ValueError("checkpoint cohort IDs differ from the production contract")
    tree_ids = tuple(_text(value, "tree_id") for value in expected_tree_ids)
    if len(tree_ids) != PRODUCTION_N_TREES or len(set(tree_ids)) != PRODUCTION_N_TREES:
        raise ValueError(f"production tree IDs must contain exactly {PRODUCTION_N_TREES} entries")

    receipts: dict[str, Any] = {}
    reference_topology_scope: list[dict[str, str]] | None = None
    for cohort_id in cohort_ids:
        manifests = tuple(cohorts[cohort_id])
        observed_tree_ids = tuple(manifest.tree_id for manifest in manifests)
        if observed_tree_ids != tree_ids:
            raise ValueError(f"cohort {cohort_id!r} tree order differs from production contract")
        verified = tuple(read_tree_checkpoint_manifest(manifest.to_dict()) for manifest in manifests)
        totals = {
            "n_nodes": sum(manifest.n_nodes for manifest in verified),
            "n_internal": sum(manifest.n_internal for manifest in verified),
            "n_leaves": sum(manifest.n_leaves for manifest in verified),
        }
        expected_totals = {
            "n_nodes": PRODUCTION_N_NODES,
            "n_internal": PRODUCTION_N_INTERNAL,
            "n_leaves": PRODUCTION_N_LEAVES,
        }
        if totals != expected_totals:
            raise ValueError(
                f"cohort {cohort_id!r} production checkpoint totals differ: "
                f"observed={totals}, expected={expected_totals}"
            )
        checkpoint_paths = [
            binding.checkpoint.path
            for manifest in verified
            for binding in manifest.bindings
        ]
        if len(checkpoint_paths) != len(set(checkpoint_paths)):
            raise ValueError(
                f"cohort {cohort_id!r} reuses a checkpoint path across trained nodes"
            )
        first = verified[0].bindings[0]
        for field in (
            "method",
            "artifact_name",
            "base_model_id",
            "base_model_revision",
            "top_config",
            "training_summary",
            "completion_receipt",
        ):
            expected = getattr(first, field)
            for manifest in verified:
                if getattr(manifest.bindings[0], field) != expected:
                    raise ValueError(f"cohort {cohort_id!r} {field} differs across trees")
        topology_scope = [
            {
                "tree_id": manifest.tree_id,
                "topology_sha256": manifest.topology_sha256,
                "nodes_sha256": manifest.nodes_sha256,
                "leaf_subset_sha256": manifest.leaf_subset_sha256,
                "pair_index_sha256": manifest.pair_index_sha256,
            }
            for manifest in verified
        ]
        if reference_topology_scope is None:
            reference_topology_scope = topology_scope
        elif topology_scope != reference_topology_scope:
            raise ValueError(f"cohort {cohort_id!r} topology differs from the reference cohort")
        manifest_scope = [
            {"tree_id": manifest.tree_id, "checkpoint_set_sha256": manifest.checkpoint_set_sha256}
            for manifest in verified
        ]
        receipts[cohort_id] = {
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "n_trees": PRODUCTION_N_TREES,
            **totals,
            "tree_ids": list(tree_ids),
            "manifest_scope": manifest_scope,
            "manifest_scope_sha256": canonical_sha256(manifest_scope),
        }
    assert reference_topology_scope is not None
    return {
        "schema": "weighttraits.all_node_production_checkpoints.v1",
        "valid": True,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "exclude_root": True,
        "n_cohorts": len(cohort_ids),
        "cohort_ids": list(cohort_ids),
        "n_trees_per_cohort": PRODUCTION_N_TREES,
        "n_nodes_per_cohort": PRODUCTION_N_NODES,
        "n_internal_per_cohort": PRODUCTION_N_INTERNAL,
        "n_leaves_per_cohort": PRODUCTION_N_LEAVES,
        "topology_scope_sha256": canonical_sha256(reference_topology_scope),
        "cohorts": receipts,
    }


def _binding(value: NodeCheckpointBinding | Mapping[str, Any]) -> NodeCheckpointBinding:
    if isinstance(value, NodeCheckpointBinding):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint binding must be a mapping")
    required = {
        "population",
        "internal_node_treatment",
        "cohort_id",
        "tree_id",
        "node_ordinal",
        "array_index",
        "node_id",
        "parent_id",
        "depth",
        "is_leaf",
        "node_role",
        "status",
        "method",
        "artifact_name",
        "checkpoint",
        "base_model_id",
        "base_model_revision",
        "run_list",
        "ledger",
        "truth_manifest",
        "top_config",
        "training_summary",
        "completion_receipt",
    }
    if set(value) != required:
        raise ValueError(f"checkpoint binding keys must be exactly {sorted(required)!r}")
    population = _text(value["population"], "population")
    treatment = _text(value["internal_node_treatment"], "internal_node_treatment")
    if population != POPULATION:
        raise ValueError(f"population must be exactly {POPULATION!r}")
    if treatment != INTERNAL_NODE_TREATMENT:
        raise ValueError(
            f"internal_node_treatment must be exactly {INTERNAL_NODE_TREATMENT!r}"
        )
    method = _text(value["method"], "method")
    artifact_name = _text(value["artifact_name"], "artifact_name")
    if _METHOD_ARTIFACT.get(method) != artifact_name:
        raise ValueError(
            f"artifact_name {artifact_name!r} is invalid for training method {method!r}"
        )
    status = _text(value["status"], "status")
    if status != COMPLETED_STATUS:
        raise ValueError(f"checkpoint status must be exactly {COMPLETED_STATUS!r}")
    role = _text(value["node_role"], "node_role")
    is_leaf = _boolean(value["is_leaf"], "is_leaf")
    if role != ("leaf" if is_leaf else "internal"):
        raise ValueError("node_role is inconsistent with is_leaf")
    revision = _text(value["base_model_revision"], "base_model_revision")
    if not _REVISION.fullmatch(revision):
        raise ValueError("base_model_revision must be a pinned lowercase 40- or 64-hex digest")
    return NodeCheckpointBinding(
        population=population,
        internal_node_treatment=treatment,
        cohort_id=_text(value["cohort_id"], "cohort_id"),
        tree_id=_text(value["tree_id"], "tree_id"),
        node_ordinal=_integer(value["node_ordinal"], "node_ordinal", minimum=0),
        array_index=_integer(value["array_index"], "array_index", minimum=0),
        node_id=_text(value["node_id"], "node_id"),
        parent_id=_text(value["parent_id"], "parent_id"),
        depth=_integer(value["depth"], "depth", minimum=1),
        is_leaf=is_leaf,
        node_role=role,
        status=status,
        method=method,
        artifact_name=artifact_name,
        checkpoint=_pin(value["checkpoint"], "checkpoint"),
        base_model_id=_text(value["base_model_id"], "base_model_id"),
        base_model_revision=revision,
        run_list=_pin(value["run_list"], "run_list"),
        ledger=_pin(value["ledger"], "ledger"),
        truth_manifest=_pin(value["truth_manifest"], "truth_manifest"),
        top_config=_pin(value["top_config"], "top_config"),
        training_summary=_pin(value["training_summary"], "training_summary"),
        completion_receipt=_pin(value["completion_receipt"], "completion_receipt"),
    )


def _pin(value: object, label: str) -> ArtifactPin:
    if isinstance(value, ArtifactPin):
        value = value.to_dict()
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} pin must contain exactly path and sha256")
    path = _text(value["path"], f"{label}.path")
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"{label}.path must be an absolute normalized path")
    digest = _text(value["sha256"], f"{label}.sha256")
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256")
    return ArtifactPin(path=path, sha256=digest)


def _require_shared(bindings: Sequence[NodeCheckpointBinding], field: str) -> None:
    values = [getattr(binding, field) for binding in bindings]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"checkpoint binding {field} must be identical within one tree")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value
