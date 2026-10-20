"""Versioned node-design contract for the observed all-node population.

Training task labels are not topology metadata and therefore cannot be
reconstructed from checkpoint directories or distance matrices.  This module
binds the exact truth-manifest path, per-node dataset/task labels, and ancestry
paths to one validated topology/checkpoint set in canonical node order.

The inherited ``task_jaccard`` control used a set of manifest ``task`` labels
along each root-to-node path, silently skipping the untrained root because it
has no manifest row.  In the current training records that task identity is
``dataset_id``.  :func:`task_path_jaccard` preserves those exact semantics;
``task_family`` is exposed only as an explicit sensitivity label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Any, Mapping, Sequence

from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    TreeCheckpointManifest,
    read_tree_checkpoint_manifest,
)
from .manifest import TreeTopology, canonical_sha256, read_tree_topology


NODE_COVARIATE_SCHEMA = "weighttraits.all_node_node_covariates.v1"
TASK_LABEL_FIELDS = ("dataset_id", "task_family")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class NodeCovariate:
    """One trained node's topology identity and pinned training-task labels."""

    node_id: str
    parent_id: str
    depth: int
    path: tuple[str, ...]
    dataset_id: str
    task_family: str
    role: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["path"] = list(self.path)
        return row


@dataclass(frozen=True)
class TreeNodeCovariates:
    """Canonical, source-pinned node design for one all-node tree."""

    tree_id: str
    population: str
    internal_node_treatment: str
    topology_sha256: str
    checkpoint_set_sha256: str
    source_manifest: ArtifactPin
    nodes: tuple[NodeCovariate, ...]
    nodes_sha256: str
    covariates_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NODE_COVARIATE_SCHEMA,
            "tree_id": self.tree_id,
            "population": self.population,
            "internal_node_treatment": self.internal_node_treatment,
            "topology_sha256": self.topology_sha256,
            "checkpoint_set_sha256": self.checkpoint_set_sha256,
            "source_manifest": self.source_manifest.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "nodes_sha256": self.nodes_sha256,
            "covariates_sha256": self.covariates_sha256,
        }

    def node(self, node_id: str) -> NodeCovariate:
        identity = _text(node_id, "node_id")
        matches = [node for node in self.nodes if node.node_id == identity]
        if len(matches) != 1:
            raise ValueError(
                f"node covariate contract has {len(matches)} rows for {identity!r}"
            )
        return matches[0]


def build_tree_node_covariates(
    topology: TreeTopology,
    checkpoint_manifest: TreeCheckpointManifest,
    rows: Sequence[NodeCovariate | Mapping[str, Any]],
    *,
    source_manifest: ArtifactPin | Mapping[str, Any],
) -> TreeNodeCovariates:
    """Bind exact ordered node labels and paths to topology/checkpoint hashes."""

    tree, manifest = _verified_scope(topology, checkpoint_manifest)
    source = _pin(source_manifest, "source_manifest")
    declared_sources = {binding.truth_manifest for binding in manifest.bindings}
    if declared_sources != {source}:
        raise ValueError(
            "node covariate source manifest differs from checkpoint bindings"
        )
    normalized = tuple(_node_covariate(value) for value in rows)
    if len(normalized) != tree.n_nodes:
        raise ValueError(
            "node covariate count differs from exact topology: "
            f"observed={len(normalized)}, expected={tree.n_nodes}"
        )
    expected_paths = _topology_paths(tree)
    for ordinal, (row, node) in enumerate(zip(normalized, tree.nodes)):
        expected = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "path": expected_paths[node.node_id],
            "role": node.node_role,
        }
        for field, expected_value in expected.items():
            if getattr(row, field) != expected_value:
                raise ValueError(
                    f"node covariate row {ordinal} {field} differs from topology: "
                    f"expected={expected_value!r}, observed={getattr(row, field)!r}"
                )
    node_rows = [node.to_dict() for node in normalized]
    nodes_sha = canonical_sha256(node_rows)
    core = {
        "tree_id": tree.tree_id,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "topology_sha256": tree.topology_sha256,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "source_manifest": source.to_dict(),
        "nodes_sha256": nodes_sha,
    }
    return TreeNodeCovariates(
        tree_id=tree.tree_id,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        topology_sha256=tree.topology_sha256,
        checkpoint_set_sha256=manifest.checkpoint_set_sha256,
        source_manifest=source,
        nodes=normalized,
        nodes_sha256=nodes_sha,
        covariates_sha256=canonical_sha256(core),
    )


def read_tree_node_covariates(
    payload: Mapping[str, Any],
    topology: TreeTopology,
    checkpoint_manifest: TreeCheckpointManifest,
) -> TreeNodeCovariates:
    """Recompute a saved node-design artifact and reject any field/hash drift."""

    declared = read_node_covariates_payload(payload)
    rebuilt = build_tree_node_covariates(
        topology,
        checkpoint_manifest,
        declared.nodes,
        source_manifest=declared.source_manifest,
    )
    if declared != rebuilt:
        raise ValueError("saved node covariates differ from their recomputed contract")
    return rebuilt


def read_node_covariates_payload(
    payload: Mapping[str, Any],
) -> TreeNodeCovariates:
    """Validate a serialized contract without trusting an external topology.

    This boundary is used by terminal inference receipts, which embed the full
    already-resolved node design but not a second copy of the checkpoint
    manifest.  It validates exact keys, node rows, and both canonical hashes;
    callers that possess topology/checkpoint objects must additionally call
    :func:`read_tree_node_covariates` or
    :func:`validate_tree_node_covariates`.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("saved node covariates must be a mapping")
    required = {
        "schema",
        "tree_id",
        "population",
        "internal_node_treatment",
        "topology_sha256",
        "checkpoint_set_sha256",
        "source_manifest",
        "nodes",
        "nodes_sha256",
        "covariates_sha256",
    }
    if set(payload) != required:
        raise ValueError(
            f"saved node covariate keys must be exactly {sorted(required)!r}"
        )
    if payload.get("schema") != NODE_COVARIATE_SCHEMA:
        raise ValueError("saved node covariate schema drift")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("saved node covariate nodes must be a list")
    nodes = tuple(_node_covariate(value) for value in raw_nodes)
    if not nodes:
        raise ValueError("saved node covariates must contain at least one node")
    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("saved node covariates contain duplicate node IDs")
    node_rows = [node.to_dict() for node in nodes]
    nodes_sha = canonical_sha256(node_rows)
    if _sha256(payload.get("nodes_sha256"), "nodes_sha256") != nodes_sha:
        raise ValueError("saved node covariate node hash drift")
    core = {
        "tree_id": _text(payload.get("tree_id"), "tree_id"),
        "population": _text(payload.get("population"), "population"),
        "internal_node_treatment": _text(
            payload.get("internal_node_treatment"), "internal_node_treatment"
        ),
        "topology_sha256": _sha256(
            payload.get("topology_sha256"), "topology_sha256"
        ),
        "checkpoint_set_sha256": _sha256(
            payload.get("checkpoint_set_sha256"), "checkpoint_set_sha256"
        ),
        "source_manifest": _pin(
            payload.get("source_manifest"), "source_manifest"
        ).to_dict(),
        "nodes_sha256": nodes_sha,
    }
    if core["population"] != POPULATION:
        raise ValueError("saved node covariate population drift")
    if core["internal_node_treatment"] != INTERNAL_NODE_TREATMENT:
        raise ValueError("saved node covariate internal-node treatment drift")
    covariates_sha = canonical_sha256(core)
    if _sha256(
        payload.get("covariates_sha256"), "covariates_sha256"
    ) != covariates_sha:
        raise ValueError("saved node covariate contract hash drift")
    rebuilt = TreeNodeCovariates(
        tree_id=core["tree_id"],
        population=core["population"],
        internal_node_treatment=core["internal_node_treatment"],
        topology_sha256=core["topology_sha256"],
        checkpoint_set_sha256=core["checkpoint_set_sha256"],
        source_manifest=_pin(payload.get("source_manifest"), "source_manifest"),
        nodes=nodes,
        nodes_sha256=nodes_sha,
        covariates_sha256=covariates_sha,
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(rebuilt.to_dict()):
        raise ValueError("saved node covariates differ from their recomputed contract")
    return rebuilt


def validate_tree_node_covariates(
    value: TreeNodeCovariates,
    topology: TreeTopology,
    checkpoint_manifest: TreeCheckpointManifest,
) -> TreeNodeCovariates:
    """Validate an in-memory contract through its serialized boundary."""

    if not isinstance(value, TreeNodeCovariates):
        raise ValueError("node_covariates must be a TreeNodeCovariates contract")
    rebuilt = read_tree_node_covariates(
        value.to_dict(), topology, checkpoint_manifest
    )
    if rebuilt != value:
        raise ValueError("node covariates differ from their validated form")
    return rebuilt


def task_path_labels(
    covariates: TreeNodeCovariates,
    node_id: str,
    *,
    label: str = "dataset_id",
) -> frozenset[str]:
    """Return the legacy set of task labels on one root-to-node path.

    The untrained root is absent from ``covariates.nodes`` and is therefore
    skipped, exactly matching ``if n in node_info`` in the legacy scripts.
    """

    if label not in TASK_LABEL_FIELDS:
        raise ValueError(f"task path label must be one of {TASK_LABEL_FIELDS!r}")
    if not isinstance(covariates, TreeNodeCovariates):
        raise ValueError("covariates must be a TreeNodeCovariates contract")
    row = covariates.node(node_id)
    by_id = {node.node_id: node for node in covariates.nodes}
    return frozenset(
        getattr(by_id[path_node_id], label)
        for path_node_id in row.path
        if path_node_id in by_id
    )


def task_path_jaccard(
    covariates: TreeNodeCovariates,
    left_node_id: str,
    right_node_id: str,
    *,
    label: str = "dataset_id",
) -> float:
    """Compute exact legacy path-task Jaccard from the pinned node design."""

    left = task_path_labels(covariates, left_node_id, label=label)
    right = task_path_labels(covariates, right_node_id, label=label)
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _verified_scope(
    topology: TreeTopology,
    checkpoint_manifest: TreeCheckpointManifest,
) -> tuple[TreeTopology, TreeCheckpointManifest]:
    if not isinstance(topology, TreeTopology):
        raise ValueError("topology must be a TreeTopology")
    if not isinstance(checkpoint_manifest, TreeCheckpointManifest):
        raise ValueError("checkpoint_manifest must be a TreeCheckpointManifest")
    tree = read_tree_topology(topology.to_dict())
    manifest = read_tree_checkpoint_manifest(checkpoint_manifest.to_dict())
    if manifest.tree_id != tree.tree_id:
        raise ValueError("checkpoint manifest tree differs from topology")
    for field in (
        "topology_sha256",
        "nodes_sha256",
        "pair_index_sha256",
        "leaf_subset_sha256",
    ):
        if getattr(manifest, field) != getattr(tree, field):
            raise ValueError(f"checkpoint manifest {field} differs from topology")
    if tuple(binding.node_id for binding in manifest.bindings) != tree.ordered_node_ids:
        raise ValueError("checkpoint bindings differ from topology order")
    return tree, manifest


def _topology_paths(topology: TreeTopology) -> dict[str, tuple[str, ...]]:
    by_id = {node.node_id: node for node in topology.nodes}
    paths: dict[str, tuple[str, ...]] = {}

    def build(node_id: str) -> tuple[str, ...]:
        if node_id in paths:
            return paths[node_id]
        node = by_id[node_id]
        if node.parent_id == topology.root_id:
            path = (topology.root_id, node.node_id)
        else:
            if node.parent_id not in by_id:
                raise ValueError(f"node {node_id!r} has no trained parent covariate")
            path = (*build(node.parent_id), node.node_id)
        paths[node_id] = path
        return path

    for node in topology.nodes:
        build(node.node_id)
    return paths


def _node_covariate(value: NodeCovariate | Mapping[str, Any]) -> NodeCovariate:
    if isinstance(value, NodeCovariate):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("node covariate row must be a mapping")
    required = {
        "node_id",
        "parent_id",
        "depth",
        "path",
        "dataset_id",
        "task_family",
        "role",
    }
    if set(value) != required:
        raise ValueError(f"node covariate keys must be exactly {sorted(required)!r}")
    raw_path = value.get("path")
    if not isinstance(raw_path, (list, tuple)) or not raw_path:
        raise ValueError("node covariate path must be a nonempty sequence")
    path = tuple(_text(item, "node covariate path member") for item in raw_path)
    if len(path) != len(set(path)):
        raise ValueError("node covariate path contains a repeated node")
    role = _text(value.get("role"), "node covariate role")
    if role not in {"internal", "leaf"}:
        raise ValueError("node covariate role must be internal or leaf")
    row = NodeCovariate(
        node_id=_text(value.get("node_id"), "node covariate node_id"),
        parent_id=_text(value.get("parent_id"), "node covariate parent_id"),
        depth=_integer(value.get("depth"), "node covariate depth", minimum=1),
        path=path,
        dataset_id=_text(value.get("dataset_id"), "node covariate dataset_id"),
        task_family=_text(value.get("task_family"), "node covariate task_family"),
        role=role,
    )
    if (
        len(row.path) != row.depth + 1
        or row.path[-1] != row.node_id
        or row.path[-2] != row.parent_id
    ):
        raise ValueError(
            "node covariate path is inconsistent with node_id, parent_id, or depth"
        )
    return row


def _pin(value: ArtifactPin | Mapping[str, Any], label: str) -> ArtifactPin:
    if isinstance(value, ArtifactPin):
        value = value.to_dict()
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _text(value.get("path"), f"{label}.path")
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"{label}.path must be canonical absolute")
    return ArtifactPin(path, _sha256(value.get("sha256"), f"{label}.sha256"))


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty normalized text")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value
