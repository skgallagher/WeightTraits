"""Root-invariant split extraction for manifest and Newick trees."""

from __future__ import annotations

from collections.abc import Iterable

from weighttraits.manifests.reference import (
    ManifestRecord,
    descendant_leaf_sets,
    leaf_ids,
    load_manifest,
)
from weighttraits.phylo.newick import NewickNode, leaf_names, parse_newick


Split = frozenset[str]


def canonical_split(side: Iterable[str], all_leaves: Iterable[str]) -> Split | None:
    """Return the canonical nontrivial side of a bipartition.

    The smaller side is canonical; ties are broken lexicographically. Singleton
    and all-leaf sides are excluded because they are not informative internal
    splits for RF/clade-recovery scoring.
    """

    leaves = frozenset(all_leaves)
    side_set = frozenset(side) & leaves
    other = leaves - side_set
    if len(side_set) <= 1 or len(other) <= 1:
        return None
    if len(side_set) < len(other):
        return side_set
    if len(other) < len(side_set):
        return frozenset(other)
    return min(side_set, other, key=lambda values: tuple(sorted(values)))


def splits_from_manifest(records: Iterable[ManifestRecord]) -> tuple[set[Split], set[str]]:
    """Extract canonical nontrivial splits from an ELLMTrees-style manifest."""

    rows = list(records)
    leaves = set(leaf_ids(rows))
    splits: set[Split] = set()
    for node, descendants in descendant_leaf_sets(rows).items():
        if node == "root":
            continue
        split = canonical_split(descendants, leaves)
        if split is not None:
            splits.add(split)
    return splits, leaves


def splits_from_manifest_path(path: str) -> tuple[set[Split], set[str]]:
    return splits_from_manifest(load_manifest(path))


def splits_from_newick_text(text: str) -> tuple[set[Split], set[str]]:
    """Extract canonical nontrivial splits from Newick text."""

    root = parse_newick(text)
    leaves = set(leaf_names(root))
    splits: set[Split] = set()

    def descend(node: NewickNode, is_root: bool = False) -> frozenset[str]:
        if node.is_leaf:
            if node.name is None:
                raise ValueError("leaf without label")
            return frozenset([node.name])
        descendants: set[str] = set()
        for child in node.children:
            descendants.update(descend(child))
        if not is_root:
            split = canonical_split(descendants, leaves)
            if split is not None:
                splits.add(split)
        return frozenset(descendants)

    descend(root, is_root=True)
    return splits, leaves


def serialize_split(split: Split) -> list[str]:
    return sorted(split)

