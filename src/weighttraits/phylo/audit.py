"""Topology audit summaries for generated or reference manifests."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

from weighttraits.manifests.reference import load_manifest, trained_child_index
from weighttraits.phylo.splits import splits_from_manifest


def audit_manifest_topology(path: str | Path) -> dict[str, Any]:
    """Return a compact topology audit for a manifest JSONL file."""

    records = load_manifest(path)
    children = {node: sorted(kids) for node, kids in trained_child_index(records).items()}
    nodes = sorted({record.node_id for record in records if record.is_trained})
    splits, leaves = splits_from_manifest(records)

    depth_by_node: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque(("root", 0) for _ in [0])
    seen = set()
    while queue:
        node, depth = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for child in children.get(node, []):
            depth_by_node[child] = depth + 1
            queue.append((child, depth + 1))

    out_degrees = {node: len(children.get(node, [])) for node in ["root", *nodes]}
    leaf_child_counts = {
        node: sum(1 for child in children.get(node, []) if not children.get(child))
        for node in ["root", *nodes]
    }
    cherry_nodes = [node for node, count in leaf_child_counts.items() if count >= 2]
    exact_binary_cherries = [node for node, count in leaf_child_counts.items() if count == 2]

    out_degree_hist = Counter(out_degrees.values())
    depth_hist = Counter(depth_by_node.values())
    n_leaves = len(leaves)
    n_binary_ref_splits = max(n_leaves - 3, 0)

    return {
        "manifest": str(path),
        "n_trained_nodes": len(nodes),
        "n_leaves": n_leaves,
        "n_truth_splits": len(splits),
        "n_binary_reference_splits_if_fully_resolved": n_binary_ref_splits,
        "max_depth": max(depth_by_node.values(), default=0),
        "max_out_degree": max(out_degrees.values(), default=0),
        "n_polytomies": sum(1 for degree in out_degrees.values() if degree > 2),
        "n_internal_nodes": sum(1 for node in nodes if children.get(node)),
        "n_cherry_nodes": len(cherry_nodes),
        "n_exact_binary_cherry_nodes": len(exact_binary_cherries),
        "n_leaf_child_pairs_under_same_parent": sum(count * (count - 1) // 2 for count in leaf_child_counts.values()),
        "out_degree_histogram": {str(key): out_degree_hist[key] for key in sorted(out_degree_hist)},
        "nodes_by_depth": {str(key): depth_hist[key] for key in sorted(depth_hist)},
        "leaves": sorted(leaves),
        "rf_resolution_notes": {
            "has_minimum_rf_leaf_count": n_leaves >= 4,
            "truth_is_polytomous": any(degree > 2 for degree in out_degrees.values()),
            "binary_nj_can_add_false_positive_splits": any(degree > 2 for degree in out_degrees.values()),
        },
    }

