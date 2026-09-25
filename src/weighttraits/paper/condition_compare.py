"""Compare two conditions' estimated trees tree by tree (App. A, q/k/v vs q/k/v/o LoRA).

Two separately trained conditions can produce Table 2 rows that coincide at display precision
without having identical trees. For each tree this reports whether the estimated split sets
are identical and, if not, whether the differing splits only resolve polytomies of the true
tree (compatible with every true split) or contradict it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from weighttraits.phylo.splits import Split, splits_from_newick_text


def _compatible(split: Split, other: Split, leaves: frozenset[str]) -> bool:
    rest, other_rest = leaves - split, leaves - other
    return not (split & other and split & other_rest and rest & other and rest & other_rest)


def compare_tree_pair(
    left_newick: str, right_newick: str, truth_newick: str | None = None
) -> dict[str, Any]:
    left, left_leaves = splits_from_newick_text(left_newick)
    right, right_leaves = splits_from_newick_text(right_newick)
    if left_leaves != right_leaves:
        raise ValueError("the two estimates have different leaf sets")
    row: dict[str, Any] = {
        "identical": left == right,
        "n_left_only": len(left - right),
        "n_right_only": len(right - left),
        "n_left_splits": len(left),
        "n_right_splits": len(right),
    }
    if truth_newick is not None:
        truth, truth_leaves = splits_from_newick_text(truth_newick)
        if truth_leaves != left_leaves:
            raise ValueError("truth and estimates have different leaf sets")
        leaves = frozenset(left_leaves)
        differing = (left - right) | (right - left)
        row.update(
            {
                "n_true_splits": len(truth),
                "left_false_negatives": len(truth - left),
                "right_false_negatives": len(truth - right),
                "left_false_positives": len(left - truth),
                "right_false_positives": len(right - truth),
                "differences_only_resolve_polytomies": all(
                    _compatible(split, true_split, leaves)
                    for split in differing
                    for true_split in truth
                ),
            }
        )
    return row


def compare_conditions(
    pairs: Iterable[tuple[str, Path, Path, Path | None]],
) -> dict[str, Any]:
    """Compare ``(tree_id, left_newick, right_newick, truth_newick_or_None)`` tuples."""

    rows = []
    for tree_id, left, right, truth in pairs:
        row = compare_tree_pair(
            Path(left).read_text(),
            Path(right).read_text(),
            Path(truth).read_text() if truth is not None else None,
        )
        rows.append({"tree_id": tree_id, **row})
    differing = [row for row in rows if not row["identical"]]
    summary: dict[str, Any] = {
        "n_trees": len(rows),
        "n_identical": len(rows) - len(differing),
        "n_differ": len(differing),
        "differing_tree_ids": [row["tree_id"] for row in differing],
    }
    if rows and "n_true_splits" in rows[0]:
        summary["all_differences_only_resolve_polytomies"] = all(
            row["differences_only_resolve_polytomies"] for row in differing
        )
        summary["differing_trees_have_equal_fn_and_fp"] = all(
            row["left_false_negatives"] == row["right_false_negatives"]
            and row["left_false_positives"] == row["right_false_positives"]
            for row in differing
        )
    return {"summary": summary, "trees": rows}


def pairs_from_dirs(
    left: Path,
    right: Path,
    *,
    pattern: str,
    truth_pattern: str | None = None,
    tree_ids: Iterable[str] | None = None,
) -> list[tuple[str, Path, Path, Path | None]]:
    """Build comparison tuples from two directory trees sharing ``{tree_id}`` path patterns.

    ``pattern`` and ``truth_pattern`` are formatted with ``tree_id``, relative to ``left`` and
    ``right`` (the truth is read from under ``left``).
    """

    if tree_ids is None:
        tree_ids = sorted(path.name for path in Path(left).glob("confirm_paper_tree_*"))
    pairs = []
    for tree_id in tree_ids:
        mapping: Mapping[str, str] = {"tree_id": tree_id}
        truth = Path(left) / truth_pattern.format_map(mapping) if truth_pattern else None
        pairs.append(
            (
                tree_id,
                Path(left) / pattern.format_map(mapping),
                Path(right) / pattern.format_map(mapping),
                truth,
            )
        )
    return pairs
