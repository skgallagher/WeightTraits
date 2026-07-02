"""Flexible tree generation for training-lineage experiments."""

from weighttraits.trees.generate import (
    TreeNode,
    generate_tree,
    generate_tree_from_config,
    leaf_ids,
    manifest_rows,
    tree_stats,
)

__all__ = [
    "TreeNode",
    "generate_tree",
    "generate_tree_from_config",
    "leaf_ids",
    "manifest_rows",
    "tree_stats",
]

