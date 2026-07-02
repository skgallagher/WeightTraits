"""Reference topology helpers for ELLMTrees-style JSONL manifests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestRecord:
    node_id: str
    path: tuple[str, ...]
    grow: str = "train"

    @property
    def is_trained(self) -> bool:
        return self.grow == "train"


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                node_id = raw["node_id"]
                lineage = tuple(raw["path"])
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing required field {exc}") from exc
            records.append(ManifestRecord(node_id=node_id, path=lineage, grow=raw.get("grow", "train")))
    return records


def trained_child_index(records: Iterable[ManifestRecord]) -> dict[str, set[str]]:
    """Build a child index after removing non-training nodes from each lineage."""

    rows = list(records)
    trained_ids = {record.node_id for record in rows if record.is_trained}
    children: dict[str, set[str]] = defaultdict(set)
    for record in rows:
        if not record.is_trained:
            continue
        filtered_path = tuple(part for part in record.path if part == "root" or part in trained_ids)
        if not filtered_path or filtered_path[-1] != record.node_id:
            filtered_path = (*filtered_path, record.node_id)
        for parent, child in zip(filtered_path, filtered_path[1:], strict=False):
            children[parent].add(child)
        children.setdefault(record.node_id, set())
    return dict(children)


def leaf_ids(records: Iterable[ManifestRecord]) -> list[str]:
    children = trained_child_index(records)
    all_nodes = {node for node in children if node != "root"}
    child_nodes = {child for descendants in children.values() for child in descendants}
    all_nodes |= child_nodes - {"root"}
    leaves = sorted(node for node in all_nodes if not children.get(node))
    return leaves


def manifest_leaf_ids(path: str | Path) -> list[str]:
    return leaf_ids(load_manifest(path))


def descendant_leaf_sets(records: Iterable[ManifestRecord]) -> dict[str, frozenset[str]]:
    """Return descendant trained leaves for every node in the manifest tree."""

    children = trained_child_index(records)

    def descend(node: str) -> frozenset[str]:
        node_children = children.get(node, set())
        if not node_children and node != "root":
            return frozenset([node])
        leaves: set[str] = set()
        for child in node_children:
            leaves.update(descend(child))
        return frozenset(leaves)

    nodes = set(children)
    for node_children in children.values():
        nodes.update(node_children)
    return {node: descend(node) for node in sorted(nodes)}


def nontrivial_reference_splits(records: Iterable[ManifestRecord]) -> set[frozenset[str]]:
    """Return rooted nontrivial descendant leaf sets for audit comparisons."""

    rows = list(records)
    all_leaves = set(leaf_ids(rows))
    n_leaves = len(all_leaves)
    splits = set()
    for node, leaves in descendant_leaf_sets(rows).items():
        if node == "root":
            continue
        if 1 < len(leaves) < n_leaves:
            splits.add(leaves)
    return splits

