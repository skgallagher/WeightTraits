"""Deterministic topology contracts for trained internal and leaf checkpoints.

The production all-node endpoint treats trained internal checkpoints as observed
ancestors.  It never turns them into terminal taxa.  Canonical ordering is the
per-tree run-list order: contiguous array indices with every trained parent
appearing before its children.  The untrained root is represented only as a
parent sentinel and is never part of the checkpoint population.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


ROOT_ID = "root"
RELATIONSHIPS = frozenset({"ancestor_descendant", "collateral"})


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value with one stable byte representation."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NodeSpec:
    """One trained checkpoint in canonical per-tree order."""

    node_ordinal: int
    array_index: int
    node_id: str
    parent_id: str
    depth: int
    is_leaf: bool

    @property
    def node_role(self) -> str:
        return "leaf" if self.is_leaf else "internal"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["node_role"] = self.node_role
        return row


@dataclass(frozen=True)
class PairIndexRow:
    """One topology-derived unordered pair, independent of any matrix."""

    pair_ordinal: int
    left_ordinal: int
    right_ordinal: int
    left_node_id: str
    right_node_id: str
    left_role: str
    right_role: str
    relationship: str
    ancestor_node_id: str | None
    descendant_node_id: str | None
    is_parent_child: bool
    mrca_node_id: str
    topological_distance: int
    depth_difference: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TreeTopology:
    """Validated all-trained-node topology and its leaf sensitivity subset."""

    tree_id: str
    root_id: str
    nodes: tuple[NodeSpec, ...]
    leaf_sensitivity_indices: tuple[int, ...]
    pair_index: tuple[PairIndexRow, ...]
    nodes_sha256: str
    topology_sha256: str
    leaf_subset_sha256: str
    pair_index_sha256: str

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_sensitivity_indices)

    @property
    def n_internal(self) -> int:
        return self.n_nodes - self.n_leaves

    @property
    def ordered_node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    @property
    def leaf_sensitivity_ids(self) -> tuple[str, ...]:
        return tuple(self.nodes[index].node_id for index in self.leaf_sensitivity_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "weighttraits.all_node_topology.v1",
            "tree_id": self.tree_id,
            "population": "all_trained_nodes",
            "internal_node_treatment": "observed_ancestors",
            "root_id": self.root_id,
            "exclude_root": True,
            "n_nodes": self.n_nodes,
            "n_internal": self.n_internal,
            "n_leaves": self.n_leaves,
            "nodes": [node.to_dict() for node in self.nodes],
            "nodes_sha256": self.nodes_sha256,
            "topology_sha256": self.topology_sha256,
            "leaf_sensitivity_indices": list(self.leaf_sensitivity_indices),
            "leaf_sensitivity_ids": list(self.leaf_sensitivity_ids),
            "leaf_subset_sha256": self.leaf_subset_sha256,
            "pair_index": [row.to_dict() for row in self.pair_index],
            "pair_index_sha256": self.pair_index_sha256,
        }


def build_tree_topology(
    tree_id: str,
    raw_nodes: Sequence[NodeSpec | Mapping[str, Any]],
    *,
    root_id: str = ROOT_ID,
) -> TreeTopology:
    """Validate and freeze one tree's exact trained-node population."""

    tree = _text(tree_id, "tree_id")
    root = _text(root_id, "root_id")
    nodes = tuple(_node(value) for value in raw_nodes)
    if not nodes:
        raise ValueError(f"tree {tree!r} has no trained nodes")
    node_ids = [node.node_id for node in nodes]
    if root in node_ids:
        raise ValueError(f"untrained root {root!r} must not be a checkpoint node")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"tree {tree!r} contains duplicate node IDs")

    by_id: dict[str, NodeSpec] = {}
    children: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    for ordinal, node in enumerate(nodes):
        if node.node_ordinal != ordinal:
            raise ValueError(
                f"tree {tree!r} node ordinal must be contiguous: "
                f"position={ordinal}, declared={node.node_ordinal}"
            )
        if node.array_index != ordinal:
            raise ValueError(
                f"tree {tree!r} array_index must equal canonical ordinal: "
                f"node={node.node_id!r}, array_index={node.array_index}, ordinal={ordinal}"
            )
        if node.depth <= 0:
            raise ValueError(f"tree {tree!r} node {node.node_id!r} has nonpositive depth")
        if node.parent_id == root:
            if node.depth != 1:
                raise ValueError(
                    f"tree {tree!r} root child {node.node_id!r} must have depth 1"
                )
        else:
            parent = by_id.get(node.parent_id)
            if parent is None:
                raise ValueError(
                    f"tree {tree!r} node {node.node_id!r} has missing or later parent "
                    f"{node.parent_id!r}"
                )
            if node.depth != parent.depth + 1:
                raise ValueError(
                    f"tree {tree!r} node {node.node_id!r} depth is inconsistent with parent"
                )
            children[parent.node_id].append(node.node_id)
        by_id[node.node_id] = node

    # Parent-before-child plus a single root sentinel implies acyclicity and
    # connectivity, but require at least one trained child of the root too.
    if not any(node.parent_id == root for node in nodes):
        raise ValueError(f"tree {tree!r} is disconnected from root {root!r}")
    for node in nodes:
        computed_leaf = not children[node.node_id]
        if node.is_leaf is not computed_leaf:
            raise ValueError(
                f"tree {tree!r} node {node.node_id!r} leaf flag is inconsistent with topology"
            )

    leaf_indices = tuple(node.node_ordinal for node in nodes if node.is_leaf)
    if not leaf_indices:
        raise ValueError(f"tree {tree!r} has no trained leaves")
    pair_index = _build_pair_index(nodes, root_id=root)
    node_rows = [node.to_dict() for node in nodes]
    topology_rows = [
        {
            "node_ordinal": node.node_ordinal,
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "node_role": node.node_role,
        }
        for node in nodes
    ]
    leaf_payload = {
        "indices": list(leaf_indices),
        "ids": [nodes[index].node_id for index in leaf_indices],
    }
    pair_rows = [row.to_dict() for row in pair_index]
    return TreeTopology(
        tree_id=tree,
        root_id=root,
        nodes=nodes,
        leaf_sensitivity_indices=leaf_indices,
        pair_index=pair_index,
        nodes_sha256=canonical_sha256(node_rows),
        topology_sha256=canonical_sha256(topology_rows),
        leaf_subset_sha256=canonical_sha256(leaf_payload),
        pair_index_sha256=canonical_sha256(pair_rows),
    )


def read_tree_topology(payload: Mapping[str, Any]) -> TreeTopology:
    """Recompute every saved topology field instead of trusting its hashes.

    This checks internal consistency; callers must also verify the external
    artifact pin to detect a complete, internally consistent replacement.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("saved tree topology must be a mapping")
    if not isinstance(payload.get("nodes"), list):
        raise ValueError("saved tree topology nodes must be a list")
    tree = build_tree_topology(
        payload.get("tree_id"), payload["nodes"], root_id=payload.get("root_id")
    )
    # Canonical JSON distinguishes booleans from numbers, unlike dict equality.
    if canonical_sha256(dict(payload)) != canonical_sha256(tree.to_dict()):
        raise ValueError("saved tree topology differs from its recomputed contract")
    return tree


def validate_cohort_topologies(
    topologies: Sequence[TreeTopology],
    *,
    expected_tree_ids: Sequence[str] | None = None,
    expected_n_nodes: int | None = None,
    expected_n_internal: int | None = None,
    expected_n_leaves: int | None = None,
) -> dict[str, Any]:
    """Validate one cohort and return its immutable all-node scope summary."""

    trees = tuple(topologies)
    if not trees:
        raise ValueError("cohort topology list is empty")
    for tree in trees:
        if not isinstance(tree, TreeTopology):
            raise ValueError("cohort entries must be TreeTopology objects")
        read_tree_topology(tree.to_dict())
    tree_ids = [tree.tree_id for tree in trees]
    if len(tree_ids) != len(set(tree_ids)):
        raise ValueError("cohort contains duplicate tree IDs")
    if expected_tree_ids is not None and tree_ids != list(expected_tree_ids):
        raise ValueError("cohort tree order differs from the expected contract")
    n_nodes = sum(tree.n_nodes for tree in trees)
    n_internal = sum(tree.n_internal for tree in trees)
    n_leaves = sum(tree.n_leaves for tree in trees)
    for label, observed, expected in (
        ("nodes", n_nodes, expected_n_nodes),
        ("internal", n_internal, expected_n_internal),
        ("leaves", n_leaves, expected_n_leaves),
    ):
        if expected is not None:
            _integer(expected, f"expected_n_{label}", minimum=0)
        if expected is not None and observed != expected:
            raise ValueError(
                f"cohort {label} count differs from contract: observed={observed}, expected={expected}"
            )
    tree_scope = [
        {
            "tree_id": tree.tree_id,
            "topology_sha256": tree.topology_sha256,
            "nodes_sha256": tree.nodes_sha256,
            "leaf_subset_sha256": tree.leaf_subset_sha256,
            "pair_index_sha256": tree.pair_index_sha256,
        }
        for tree in trees
    ]
    return {
        "schema": "weighttraits.all_node_cohort_topology.v1",
        "population": "all_trained_nodes",
        "internal_node_treatment": "observed_ancestors",
        "exclude_root": True,
        "n_trees": len(trees),
        "n_nodes": n_nodes,
        "n_internal": n_internal,
        "n_leaves": n_leaves,
        "tree_ids": tree_ids,
        "tree_scope": tree_scope,
        "scope_sha256": canonical_sha256(tree_scope),
    }


def validate_matching_cohort_topologies(
    cohorts: Mapping[str, Sequence[TreeTopology]],
    *,
    expected_cohort_ids: Sequence[str],
    expected_tree_ids: Sequence[str],
    expected_n_nodes: int,
    expected_n_internal: int,
    expected_n_leaves: int,
) -> dict[str, Any]:
    """Require identical ordered ancestors, leaves and pairs across cohorts."""

    cohort_ids = [_text(value, "cohort_id") for value in expected_cohort_ids]
    if not cohort_ids or len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("expected cohort IDs must be nonempty and unique")
    if set(cohorts) != set(cohort_ids):
        raise ValueError("cohort IDs differ from the expected contract")
    receipts = {
        cohort_id: validate_cohort_topologies(
            cohorts[cohort_id],
            expected_tree_ids=expected_tree_ids,
            expected_n_nodes=expected_n_nodes,
            expected_n_internal=expected_n_internal,
            expected_n_leaves=expected_n_leaves,
        )
        for cohort_id in cohort_ids
    }
    reference = receipts[cohort_ids[0]]
    for cohort_id, receipt in receipts.items():
        if receipt["tree_scope"] != reference["tree_scope"]:
            raise ValueError(f"cohort {cohort_id!r} topology differs from reference cohort")
    return {
        "schema": "weighttraits.all_node_matching_cohorts.v1",
        "population": "all_trained_nodes",
        "internal_node_treatment": "observed_ancestors",
        "exclude_root": True,
        "n_cohorts": len(cohort_ids),
        "cohort_ids": cohort_ids,
        "n_trees_per_cohort": reference["n_trees"],
        "n_nodes_per_cohort": reference["n_nodes"],
        "n_internal_per_cohort": reference["n_internal"],
        "n_leaves_per_cohort": reference["n_leaves"],
        "shared_scope_sha256": reference["scope_sha256"],
        "cohorts": receipts,
    }


def _build_pair_index(nodes: Sequence[NodeSpec], *, root_id: str) -> tuple[PairIndexRow, ...]:
    parent = {node.node_id: node.parent_id for node in nodes}

    def lineage(node_id: str) -> list[str]:
        path = [node_id]
        cursor = node_id
        seen = {node_id}
        while cursor != root_id:
            cursor = parent.get(cursor, root_id)
            if cursor in seen:
                raise ValueError(f"cycle encountered while indexing node {node_id!r}")
            seen.add(cursor)
            path.append(cursor)
        return path

    lineages = {node.node_id: lineage(node.node_id) for node in nodes}
    rows: list[PairIndexRow] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            left_node = nodes[left]
            right_node = nodes[right]
            left_path = lineages[left_node.node_id]
            right_path = lineages[right_node.node_id]
            left_ancestors = set(left_path[1:])
            right_ancestors = set(right_path[1:])
            ancestor: str | None = None
            descendant: str | None = None
            if left_node.node_id in right_ancestors:
                ancestor, descendant = left_node.node_id, right_node.node_id
            elif right_node.node_id in left_ancestors:
                ancestor, descendant = right_node.node_id, left_node.node_id
            relationship = "ancestor_descendant" if ancestor else "collateral"

            right_positions = {node_id: index for index, node_id in enumerate(right_path)}
            mrca = next(node_id for node_id in left_path if node_id in right_positions)
            left_distance = left_path.index(mrca)
            right_distance = right_positions[mrca]
            rows.append(
                PairIndexRow(
                    pair_ordinal=len(rows),
                    left_ordinal=left,
                    right_ordinal=right,
                    left_node_id=left_node.node_id,
                    right_node_id=right_node.node_id,
                    left_role=left_node.node_role,
                    right_role=right_node.node_role,
                    relationship=relationship,
                    ancestor_node_id=ancestor,
                    descendant_node_id=descendant,
                    is_parent_child=(
                        ancestor is not None
                        and parent[descendant] == ancestor
                    ),
                    mrca_node_id=mrca,
                    topological_distance=left_distance + right_distance,
                    depth_difference=abs(left_node.depth - right_node.depth),
                )
            )
    expected = len(nodes) * (len(nodes) - 1) // 2
    if len(rows) != expected:
        raise AssertionError("pair-index cardinality corruption")
    if any(row.relationship not in RELATIONSHIPS for row in rows):
        raise AssertionError("pair-index relationship corruption")
    return tuple(rows)


def _node(value: NodeSpec | Mapping[str, Any]) -> NodeSpec:
    if isinstance(value, NodeSpec):
        # Dataclass annotations are not runtime validation.  Apply the same
        # strict checks to constructed instances as to JSON-derived mappings.
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError("node must be a NodeSpec or mapping")
    required = {"node_ordinal", "array_index", "node_id", "parent_id", "depth", "is_leaf"}
    if set(value) not in (required, required | {"node_role"}):
        raise ValueError(f"node keys must be {sorted(required)!r}, optionally with node_role")
    node = NodeSpec(
        node_ordinal=_integer(value["node_ordinal"], "node_ordinal", minimum=0),
        array_index=_integer(value["array_index"], "array_index", minimum=0),
        node_id=_text(value["node_id"], "node_id"),
        parent_id=_text(value["parent_id"], "parent_id"),
        depth=_integer(value["depth"], "depth", minimum=1),
        is_leaf=_boolean(value["is_leaf"], "is_leaf"),
    )
    if "node_role" in value and value["node_role"] != node.node_role:
        raise ValueError("node_role is inconsistent with is_leaf")
    return node


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value
