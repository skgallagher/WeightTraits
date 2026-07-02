"""Flexible topology generators for training-lineage experiments.

This module deliberately knows nothing about task datasets or model training.
It emits lineage topology only. Downstream phases can assign tasks/datasets,
train checkpoints, and analyze weights or behaviors from the same tree.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class TreeNode:
    node_id: str
    children: list["TreeNode"] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)


def iter_nodes(root: TreeNode, include_root: bool = False) -> Iterable[tuple[TreeNode, int, tuple[str, ...]]]:
    """Yield nodes in breadth-first order as ``(node, depth, path)``."""

    queue: deque[tuple[TreeNode, int, tuple[str, ...]]] = deque([(root, 0, (root.node_id,))])
    while queue:
        node, depth, path = queue.popleft()
        if include_root or node is not root:
            yield node, depth, path
        for child in node.children:
            queue.append((child, depth + 1, (*path, child.node_id)))


def leaf_ids(root: TreeNode) -> list[str]:
    return sorted(node.node_id for node, _depth, _path in iter_nodes(root) if not node.children)


def manifest_rows(root: TreeNode) -> list[dict[str, Any]]:
    """Return ELLMTrees-compatible manifest rows with topology fields only."""

    rows: list[dict[str, Any]] = []
    parent_by_child: dict[str, str] = {}
    for parent, _depth, _path in iter_nodes(root, include_root=True):
        for child in parent.children:
            parent_by_child[child.node_id] = parent.node_id

    for node, depth, path in iter_nodes(root):
        row = {
            "node_id": node.node_id,
            "parent_id": parent_by_child.get(node.node_id),
            "depth": depth,
            "path": list(path),
            "grow": node.attrs.get("grow", "train"),
        }
        row.update({f"tree_{key}": value for key, value in node.attrs.items() if key != "grow"})
        rows.append(row)
    return rows


def write_manifest_jsonl(root: TreeNode, path: str | Path) -> list[dict[str, Any]]:
    rows = manifest_rows(root)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def tree_stats(root: TreeNode) -> dict[str, Any]:
    nodes = list(iter_nodes(root))
    leaves = leaf_ids(root)
    depth_counts: dict[int, int] = {}
    max_out_degree = 0
    n_polytomies = 0
    for _node, depth, _path in nodes:
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    for node, _depth, _path in iter_nodes(root, include_root=True):
        max_out_degree = max(max_out_degree, len(node.children))
        if len(node.children) > 2:
            n_polytomies += 1
    return {
        "n_nodes": len(nodes),
        "n_leaves": len(leaves),
        "max_depth": max(depth_counts, default=0),
        "max_out_degree": max_out_degree,
        "n_polytomies": n_polytomies,
        "leaves": leaves,
        "nodes_by_depth": {str(k): depth_counts[k] for k in sorted(depth_counts)},
    }


def generate_tree(
    generator: str,
    *,
    n_nodes: int | None = None,
    n_leaves: int | None = None,
    target_leaves: int | None = None,
    max_depth: int | None = None,
    min_depth: int | None = None,
    min_leaves: int | None = None,
    branch_lambda: float | list[float] = 2.0,
    branch_factor: int = 2,
    seed: int = 0,
    fixed_root: dict[str, Any] | None = None,
    max_children: int | None = None,
    prune_lambda: float | None = None,
    prune_probability: float | None = None,
    contract_probability: float = 0.0,
    backbone_leaves: int = 32,
    backbone_depth: int | None = None,
    max_attempts: int = 200,
) -> TreeNode:
    """Generate a virtual-rooted training lineage tree."""

    constraints = _TreeConstraints(min_depth=min_depth, min_leaves=min_leaves)

    def once(attempt_seed: int) -> TreeNode:
        if generator == "fixed":
            if fixed_root is None:
                raise ValueError("fixed generator requires fixed_root")
            return _from_fixed(fixed_root)
        if generator == "chain":
            return _chain(n_nodes=_require_positive(n_nodes, "n_nodes"))
        if generator in {"balanced", "ellmtrees_balanced"}:
            if generator == "ellmtrees_balanced":
                return _ellmtrees_balanced(n_leaves=_require_positive(n_leaves, "n_leaves"))
            return _balanced(
                n_nodes=n_nodes,
                n_leaves=n_leaves,
                branch_factor=branch_factor,
            )
        if generator == "poisson_branching":
            return _poisson_branching(
                n_nodes=_require_positive(n_nodes, "n_nodes"),
                max_depth=_require_positive(max_depth, "max_depth"),
                branch_lambda=branch_lambda,
                seed=attempt_seed,
                max_children=max_children,
            )
        if generator in {"pruned_binary_backbone", "binary_prune"}:
            return _pruned_binary_backbone(
                backbone_leaves=backbone_leaves,
                backbone_depth=backbone_depth,
                target_leaves=target_leaves,
                prune_lambda=prune_lambda,
                prune_probability=prune_probability,
                contract_probability=contract_probability,
                seed=attempt_seed,
            )
        raise ValueError(f"unknown tree generator: {generator!r}")

    last_stats: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        root = once(seed + attempt)
        last_stats = tree_stats(root)
        if constraints.satisfied_by(last_stats):
            root.attrs["generator"] = generator
            root.attrs["seed"] = seed + attempt
            root.attrs["attempt"] = attempt
            return root

    raise ValueError(
        "could not generate a tree satisfying constraints "
        f"{constraints.as_dict()} after {max_attempts} attempts; last_stats={last_stats}"
    )


def generate_tree_from_config(config: dict[str, Any]) -> TreeNode:
    """Generate a tree from a YAML/JSON-decoded config dictionary."""

    generator = config.get("generator") or config.get("topology")
    if not generator:
        raise ValueError("tree config requires 'generator' or 'topology'")
    kwargs = dict(config)
    kwargs.pop("generator", None)
    kwargs.pop("topology", None)
    if generator == "fixed":
        kwargs["fixed_root"] = kwargs.pop("root")
    return generate_tree(generator, **kwargs)


@dataclass(frozen=True)
class _TreeConstraints:
    min_depth: int | None = None
    min_leaves: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {"min_depth": self.min_depth, "min_leaves": self.min_leaves}

    def satisfied_by(self, stats: dict[str, Any]) -> bool:
        if self.min_depth is not None and stats["max_depth"] < self.min_depth:
            return False
        if self.min_leaves is not None and stats["n_leaves"] < self.min_leaves:
            return False
        return True


def _require_positive(value: int | None, name: str) -> int:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _next_id(counter: list[int]) -> str:
    node_id = f"n{counter[0]}"
    counter[0] += 1
    return node_id


def _from_fixed(raw_root: dict[str, Any]) -> TreeNode:
    def build(raw: dict[str, Any]) -> TreeNode:
        node_id = raw.get("id") or raw.get("node_id")
        if not node_id:
            raise ValueError("fixed tree nodes require id or node_id")
        attrs = {key: value for key, value in raw.items() if key not in {"id", "node_id", "children"}}
        return TreeNode(
            node_id=node_id,
            attrs=attrs,
            children=[build(child) for child in raw.get("children", [])],
        )

    if raw_root.get("id", raw_root.get("node_id")) == "root":
        return build(raw_root)
    return TreeNode("root", children=[build(child) for child in raw_root.get("children", [])])


def _chain(n_nodes: int) -> TreeNode:
    root = TreeNode("root")
    counter = [0]
    parent = root
    for _ in range(n_nodes):
        child = TreeNode(_next_id(counter))
        parent.add_child(child)
        parent = child
    return root


def _ellmtrees_balanced(n_leaves: int) -> TreeNode:
    """Reproduce the old ELLMTrees depth-2, fairly balanced default topology."""

    if n_leaves < 2:
        raise ValueError("n_leaves must be at least 2")
    n_branches = max(2, math.ceil(n_leaves / 3))
    base_children = n_leaves // n_branches
    extra = n_leaves % n_branches

    root = TreeNode("root")
    counter = [0]
    leaf_counter = 0
    for branch_idx in range(n_branches):
        branch = TreeNode(_next_id(counter), attrs={"role": "ellmtrees_balanced_branch"})
        root.add_child(branch)
        n_kids = base_children + (1 if branch_idx < extra else 0)
        for _ in range(n_kids):
            leaf = TreeNode(f"n_leaf_{leaf_counter}", attrs={"role": "ellmtrees_balanced_leaf"})
            leaf_counter += 1
            branch.add_child(leaf)
    return _renumber_breadth_first(root)


def _balanced(
    *,
    n_nodes: int | None,
    n_leaves: int | None,
    branch_factor: int,
) -> TreeNode:
    if branch_factor < 2:
        raise ValueError("branch_factor must be at least 2")
    if n_nodes is None and n_leaves is None:
        raise ValueError("balanced generator requires n_nodes or n_leaves")

    root = TreeNode("root")
    counter = [0]
    queue: deque[TreeNode] = deque([root])
    created = 0
    while queue:
        parent = queue.popleft()
        if n_nodes is not None and created >= n_nodes:
            break
        if n_leaves is not None and len(leaf_ids(root)) >= n_leaves and created > 0:
            break
        remaining = n_nodes - created if n_nodes is not None else branch_factor
        n_children = min(branch_factor, remaining)
        for _ in range(n_children):
            child = TreeNode(_next_id(counter))
            parent.add_child(child)
            queue.append(child)
            created += 1
            if n_nodes is not None and created >= n_nodes:
                break
    return root


def _lambda_for_depth(branch_lambda: float | list[float], child_depth: int) -> float:
    if isinstance(branch_lambda, list):
        if not branch_lambda:
            raise ValueError("branch_lambda list cannot be empty")
        idx = min(child_depth - 1, len(branch_lambda) - 1)
        return float(branch_lambda[idx])
    return float(branch_lambda)


def _poisson_branching(
    *,
    n_nodes: int,
    max_depth: int,
    branch_lambda: float | list[float],
    seed: int,
    max_children: int | None,
) -> TreeNode:
    rng = np.random.default_rng(seed)
    root = TreeNode("root")
    counter = [0]
    queue: deque[tuple[TreeNode, int]] = deque([(root, 0)])
    created = 0

    while queue and created < n_nodes:
        parent, parent_depth = queue.popleft()
        child_depth = parent_depth + 1
        if child_depth > max_depth:
            continue
        lam = _lambda_for_depth(branch_lambda, child_depth)
        n_children = int(rng.poisson(lam))
        if parent is root and n_children < 1:
            n_children = 1
        if max_children is not None:
            n_children = min(n_children, max_children)
        n_children = min(n_children, n_nodes - created)
        for _ in range(n_children):
            child = TreeNode(_next_id(counter))
            parent.add_child(child)
            queue.append((child, child_depth))
            created += 1
            if created >= n_nodes:
                break
    return root


def _pruned_binary_backbone(
    *,
    backbone_leaves: int,
    backbone_depth: int | None,
    target_leaves: int | None,
    prune_lambda: float | None,
    prune_probability: float | None,
    contract_probability: float,
    seed: int,
) -> TreeNode:
    """Grow a full binary backbone, prune subtrees, then optionally contract edges.

    Pruning creates early-terminating lineages and deeper chains. Edge contraction
    turns selected binary internal nodes into polytomies by splicing their children
    into the parent.
    """

    if backbone_depth is None:
        if backbone_leaves < 2:
            raise ValueError("backbone_leaves must be at least 2")
        backbone_depth = math.ceil(math.log2(backbone_leaves))
    if backbone_depth < 1:
        raise ValueError("backbone_depth must be at least 1")
    if not 0.0 <= contract_probability <= 1.0:
        raise ValueError("contract_probability must be in [0, 1]")
    if prune_probability is not None and not 0.0 <= prune_probability <= 1.0:
        raise ValueError("prune_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    root = TreeNode("root")
    counter = [0]

    def build(parent: TreeNode, depth: int) -> None:
        if depth > backbone_depth:
            return
        for _ in range(2):
            child = TreeNode(_next_id(counter), attrs={"backbone_depth": depth})
            parent.add_child(child)
            build(child, depth + 1)

    build(root, 1)

    internal = [
        (node, depth)
        for node, depth, _path in iter_nodes(root)
        if node.children and depth >= 1
    ]
    chosen: set[str] = set()
    if prune_lambda is not None and prune_lambda > 0 and internal:
        k = min(int(rng.poisson(prune_lambda)), len(internal))
        if k > 0:
            chosen_idx = rng.choice(len(internal), size=k, replace=False)
            chosen.update(internal[int(idx)][0].node_id for idx in chosen_idx)
    if prune_probability is not None and prune_probability > 0:
        for node, _depth in internal:
            if rng.random() < prune_probability:
                chosen.add(node.node_id)

    if target_leaves is not None:
        # Keep pruning shallow internal nodes until the leaf count is near the target.
        # This makes target_leaves a soft upper bound without forcing binary shape.
        ordered = sorted(internal, key=lambda item: item[1])
        rng.shuffle(ordered)
        for node, _depth in ordered:
            if len(leaf_ids(root)) <= target_leaves:
                break
            chosen.add(node.node_id)
            node.children = []

    for node, _depth in sorted(internal, key=lambda item: item[1]):
        if node.node_id in chosen:
            node.children = []

    if contract_probability > 0:
        _contract_random_internal_edges(root, rng, contract_probability)

    return _renumber_breadth_first(root)


def _contract_random_internal_edges(root: TreeNode, rng: np.random.Generator, probability: float) -> None:
    """Randomly contract internal child edges, producing polytomies."""

    def visit(node: TreeNode) -> None:
        new_children: list[TreeNode] = []
        for child in node.children:
            visit(child)
            if child.children and rng.random() < probability:
                new_children.extend(child.children)
            else:
                new_children.append(child)
        node.children = new_children

    visit(root)


def _renumber_breadth_first(root: TreeNode) -> TreeNode:
    """Normalize non-root node ids to n0, n1, ... in breadth-first order."""

    for idx, (node, _depth, _path) in enumerate(iter_nodes(root)):
        node.node_id = f"n{idx}"
    return root
