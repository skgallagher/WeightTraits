#!/usr/bin/env python3
"""Summarize same- versus cross-root-branch behavioral distances."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


TREE_PATTERN = re.compile(r"tree[_-]?(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--endpoint", choices=("surface", "semantic"), required=True)
    parser.add_argument("--pairs-glob", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-template",
        default="confirm_paper_tree_{tree_number:03d}.manifest.jsonl",
    )
    parser.add_argument("--expected-trees", type=int)
    parser.add_argument("--min-leaves", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_manifest_branches(path: Path) -> dict[str, str]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)

    train_rows = [row for row in rows if row.get("grow", "train") == "train"]
    internal_ids: set[str] = set()
    paths: dict[str, list[str]] = {}
    for row in train_rows:
        node_id = str(row["node_id"])
        node_path = [str(part) for part in row.get("path", [node_id])]
        if not node_path or node_path[-1] != node_id:
            node_path.append(node_id)
        paths[node_id] = node_path
        internal_ids.update(node_path[:-1])

    branches = {}
    for row in train_rows:
        node_id = str(row["node_id"])
        is_leaf = bool(row["is_leaf"]) if "is_leaf" in row else node_id not in internal_ids
        node_path = paths[node_id]
        if is_leaf and len(node_path) >= 2:
            branches[node_id] = node_path[1]
    return branches


def read_pair_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model_a", "model_b", "behavior_distance"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def summarize_pair_table(
    path: Path,
    *,
    tree_number: int,
    branches: Mapping[str, str],
    min_leaves: int = 4,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = read_pair_table(path)
    labels: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    labeled_rows = []
    same = []
    cross = []

    for row in rows:
        left = str(row["model_a"])
        right = str(row["model_b"])
        if left == right:
            raise ValueError(f"{path} contains a self-pair for {left}")
        pair = tuple(sorted((left, right)))
        if pair in seen_pairs:
            raise ValueError(f"{path} contains duplicate pair {pair}")
        seen_pairs.add(pair)
        labels.update(pair)
        foreign = sorted(set(pair) - set(branches))
        if foreign:
            raise ValueError(f"{path} contains nonleaf or unknown models: {foreign}")
        distance = float(row["behavior_distance"])
        if not math.isfinite(distance):
            raise ValueError(f"{path} contains non-finite distance for {pair}")
        same_branch = branches[left] == branches[right]
        (same if same_branch else cross).append(distance)
        labeled_rows.append(
            {
                "tree_number": tree_number,
                "model_a": left,
                "model_b": right,
                "same_branch": int(same_branch),
                "behavior_distance": distance,
            }
        )

    expected_pairs = len(labels) * (len(labels) - 1) // 2
    if len(labels) < min_leaves:
        raise ValueError(f"{path} has {len(labels)} leaves; expected at least {min_leaves}")
    if len(rows) != expected_pairs:
        raise ValueError(f"{path} has {len(rows)} pairs; expected choose({len(labels)}, 2)={expected_pairs}")

    summary: dict[str, object] = {
        "tree_number": tree_number,
        "pair_file": str(path),
        "n_leaves": len(labels),
        "n_pairs": len(rows),
        "n_same_branch_pairs": len(same),
        "n_cross_branch_pairs": len(cross),
        "branch_ordering_valid": bool(same and cross),
        "branch_ordering_status": "ok" if same and cross else "missing_branch_class",
        "rank_biserial": None,
        "within_tree_point_biserial": None,
    }
    if same and cross:
        _, _, rank_biserial = mann_whitney(cross, same)
        binary = [1.0] * len(same) + [0.0] * len(cross)
        distances = same + cross
        point_biserial = float(np.corrcoef(binary, distances)[0, 1])
        summary["rank_biserial"] = rank_biserial
        summary["within_tree_point_biserial"] = float(
            np.clip(point_biserial, -0.999, 0.999)
        )
    return summary, labeled_rows


def aggregate(
    tree_summaries: Sequence[Mapping[str, object]],
    labeled_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    same = [float(row["behavior_distance"]) for row in labeled_rows if row["same_branch"] == 1]
    cross = [float(row["behavior_distance"]) for row in labeled_rows if row["same_branch"] == 0]
    if not same or not cross:
        raise ValueError("pooled analysis requires both same- and cross-branch pairs")
    _, pooled_p, pooled_rank_biserial = mann_whitney(cross, same)

    valid = [row for row in tree_summaries if row["branch_ordering_valid"] is True]
    rank_biserials = [float(row["rank_biserial"]) for row in valid]
    within_r = [float(row["within_tree_point_biserial"]) for row in valid]
    if not rank_biserials:
        raise ValueError("no tree contains both same- and cross-branch pairs")

    nonzero = [value for value in rank_biserials if value != 0.0]
    wilcoxon_p = (
        float(stats.wilcoxon(rank_biserials, alternative="greater", zero_method="wilcox").pvalue)
        if nonzero
        else 1.0
    )
    positive = sum(value > 0.0 for value in rank_biserials)
    negative = sum(value < 0.0 for value in rank_biserials)
    sign_n = positive + negative
    sign_p = (
        float(stats.binomtest(positive, sign_n, p=0.5, alternative="greater").pvalue)
        if sign_n
        else 1.0
    )
    return {
        "n_trees": len(tree_summaries),
        "n_ordering_trees": len(valid),
        "branch_ordering_status_counts": _counts(
            str(row["branch_ordering_status"]) for row in tree_summaries
        ),
        "n_pairs": len(labeled_rows),
        "n_same_branch_pairs": len(same),
        "n_cross_branch_pairs": len(cross),
        "same_branch_mean": statistics.fmean(same),
        "cross_branch_mean": statistics.fmean(cross),
        "cross_minus_same": statistics.fmean(cross) - statistics.fmean(same),
        "pooled_mann_whitney_p_one_sided": pooled_p,
        "pooled_rank_biserial": pooled_rank_biserial,
        "tree_mean_rank_biserial": statistics.fmean(rank_biserials),
        "tree_rank_biserial_se": _standard_error(rank_biserials),
        "tree_wilcoxon_p_one_sided": wilcoxon_p,
        "tree_positive_rank_biserial": positive,
        "tree_negative_rank_biserial": negative,
        "tree_sign_p_one_sided": sign_p,
        "within_tree_point_biserial_mean": statistics.fmean(within_r),
        "within_tree_point_biserial_se": _standard_error(within_r),
        "within_tree_fisher_z_r": math.tanh(
            statistics.fmean(math.atanh(value) for value in within_r)
        ),
        "pooled_pair_inference": "descriptive_shared_leaf_pairs",
        "tree_level_inference": "one_sided_wilcoxon_on_per_tree_rank_biserial",
    }


def run_analysis(
    pair_paths: Sequence[Path],
    *,
    manifest_for_tree: Callable[[int], Path],
    min_leaves: int = 4,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    tree_summaries = []
    labeled_rows = []
    seen_trees: set[int] = set()
    for path in sorted(pair_paths, key=tree_number_from_path):
        tree_number = tree_number_from_path(path)
        if tree_number in seen_trees:
            raise ValueError(f"multiple pair files supplied for tree {tree_number:03d}")
        seen_trees.add(tree_number)
        summary, rows = summarize_pair_table(
            path,
            tree_number=tree_number,
            branches=read_manifest_branches(manifest_for_tree(tree_number)),
            min_leaves=min_leaves,
        )
        tree_summaries.append(summary)
        labeled_rows.extend(rows)
    return aggregate(tree_summaries, labeled_rows), tree_summaries, labeled_rows


def tree_number_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = TREE_PATTERN.search(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot derive tree number from {path}")


def mann_whitney(cross: Sequence[float], same: Sequence[float]) -> tuple[float, float, float]:
    result = stats.mannwhitneyu(cross, same, alternative="greater", method="asymptotic")
    u_statistic = float(result.statistic)
    rank_biserial = (2.0 * u_statistic) / (len(cross) * len(same)) - 1.0
    return u_statistic, float(result.pvalue), rank_biserial


def _standard_error(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return {key: result[key] for key in sorted(result)}


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pair_paths = [Path(value) for value in sorted(glob.glob(args.pairs_glob))]
    if not pair_paths:
        raise SystemExit(f"no files match --pairs-glob {args.pairs_glob!r}")
    if args.expected_trees is not None and len(pair_paths) != args.expected_trees:
        raise SystemExit(
            f"matched {len(pair_paths)} pair tables; expected {args.expected_trees}"
        )

    def manifest_for_tree(tree_number: int) -> Path:
        return args.manifest_dir / args.manifest_template.format(tree_number=tree_number)

    summary, tree_summaries, labeled_rows = run_analysis(
        pair_paths,
        manifest_for_tree=manifest_for_tree,
        min_leaves=args.min_leaves,
    )
    summary = {
        "condition": args.condition,
        "probe": args.probe,
        "endpoint": args.endpoint,
        **summary,
        "pairs_glob": args.pairs_glob,
        "manifest_dir": str(args.manifest_dir),
        "manifest_template": args.manifest_template,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(args.out_dir / "summary.csv", [summary])
    write_csv(args.out_dir / "per_tree.csv", tree_summaries)
    write_csv(args.out_dir / "labeled_pairs.csv", labeled_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
