"""Flexible tree generation for training-lineage experiments."""

from weighttraits.trees.generate import (
    GeneratedTree,
    TreeNode,
    generate_tree,
    generate_tree_from_config,
    generate_tree_set,
    leaf_ids,
    manifest_rows,
    tree_stats,
)

__all__ = [
    "TreeNode",
    "GeneratedTree",
    "generate_tree",
    "generate_tree_from_config",
    "generate_tree_set",
    "leaf_ids",
    "manifest_rows",
    "tree_stats",
]
