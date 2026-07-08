#!/usr/bin/env python3
"""Compare WeightTraits confirm-paper trees with local paper tree artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import re
from statistics import mean
from typing import Iterable
from xml.sax.saxutils import escape


DEFAULT_DRAFT_SPEC = {
    "n_trees": 50,
    "training_node_budget": 14,
    "branch_lambda": 1.5,
    "max_depth": 4,
    "min_leaves": 4,
    "stated_leaf_min": 4,
    "stated_leaf_max": 8,
}

SOURCE_ORDER = ("ELLMTrees runs_branching_v3", "WeightTraits confirm_paper_numbers")
COLORS = {
    "ELLMTrees runs_branching_v3": "#6a7f8f",
    "WeightTraits confirm_paper_numbers": "#2f9e8f",
}


@dataclass(frozen=True)
class TreeRecord:
    source: str
    tree_id: str
    manifest: str
    n_training_nodes: int
    n_leaves: int
    max_depth: int
    max_out_degree: int
    n_polytomies: int
    seed: int | None = None
    candidate_index: int | None = None

    def as_row(self, spec: dict[str, int | float]) -> dict[str, object]:
        return {
            "source": self.source,
            "tree_id": self.tree_id,
            "manifest": self.manifest,
            "seed": "" if self.seed is None else self.seed,
            "candidate_index": "" if self.candidate_index is None else self.candidate_index,
            "n_training_nodes": self.n_training_nodes,
            "n_leaves": self.n_leaves,
            "max_depth": self.max_depth,
            "max_out_degree": self.max_out_degree,
            "n_polytomies": self.n_polytomies,
            "within_stated_leaf_range": int(
                int(spec["stated_leaf_min"]) <= self.n_leaves <= int(spec["stated_leaf_max"])
            ),
            "within_max_depth": int(self.max_depth <= int(spec["max_depth"])),
            "uses_full_training_budget": int(
                self.n_training_nodes == int(spec["training_node_budget"])
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weighttraits-summary",
        type=Path,
        default=Path("examples/training/confirm_paper_numbers/tree_set_summary.json"),
    )
    parser.add_argument(
        "--ellmtrees-runs",
        type=Path,
        default=Path("../ELLMTrees/outputs/runs_branching_v3"),
    )
    parser.add_argument(
        "--paper-tex",
        type=Path,
        default=Path("../ELLMTrees-paper/iclr_draft_v2.tex"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/confirm_paper_tree_comparison"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parent_id(row: dict[str, object]) -> str | None:
    explicit = row.get("parent_id")
    if explicit:
        return str(explicit)
    path = row.get("path")
    if isinstance(path, list) and len(path) >= 2:
        return str(path[-2])
    return None


def row_depth(row: dict[str, object]) -> int:
    if "depth" in row:
        return int(row["depth"])
    path = row.get("path")
    if isinstance(path, list):
        return max(len(path) - 1, 0)
    return 0


def summarize_manifest(
    rows: list[dict[str, object]],
    *,
    source: str,
    tree_id: str,
    manifest: Path,
    seed: int | None = None,
    candidate_index: int | None = None,
) -> TreeRecord:
    train_rows = [row for row in rows if row.get("grow", "train") == "train"]
    node_ids = {str(row["node_id"]) for row in train_rows}
    parent_counts: Counter[str] = Counter()
    for row in train_rows:
        parent = parent_id(row)
        if parent:
            parent_counts[parent] += 1

    if any("is_leaf" in row for row in train_rows):
        leaves = [row for row in train_rows if bool(row.get("is_leaf"))]
        n_leaves = len(leaves)
    else:
        n_leaves = len(node_ids - {node for node in parent_counts if node != "root"})

    max_out_degree = max(parent_counts.values(), default=0)
    n_polytomies = sum(1 for degree in parent_counts.values() if degree > 2)

    return TreeRecord(
        source=source,
        tree_id=tree_id,
        manifest=str(manifest),
        seed=seed,
        candidate_index=candidate_index,
        n_training_nodes=len(train_rows),
        n_leaves=n_leaves,
        max_depth=max((row_depth(row) for row in train_rows), default=0),
        max_out_degree=max_out_degree,
        n_polytomies=n_polytomies,
    )


def load_weighttraits(summary_path: Path) -> list[TreeRecord]:
    summary = json.loads(summary_path.read_text())
    root = summary_path.resolve().parents[3]
    records: list[TreeRecord] = []
    for item in summary["trees"]:
        manifest = Path(item["manifest"])
        if not manifest.is_absolute():
            manifest = root / manifest
        records.append(
            summarize_manifest(
                load_jsonl(manifest),
                source="WeightTraits confirm_paper_numbers",
                tree_id=str(item["tree_id"]),
                manifest=manifest,
                seed=int(item["seed"]),
                candidate_index=int(item["candidate_index"]),
            )
        )
    return records


def load_ellmtrees(run_dir: Path) -> list[TreeRecord]:
    records: list[TreeRecord] = []
    for run in sorted(run_dir.glob("run_*")):
        manifest = run / "manifest.jsonl"
        if not manifest.exists():
            continue
        records.append(
            summarize_manifest(
                load_jsonl(manifest),
                source="ELLMTrees runs_branching_v3",
                tree_id=run.name,
                manifest=manifest,
            )
        )
    return records


def parse_draft_spec(tex_path: Path) -> dict[str, int | float | str]:
    spec: dict[str, int | float | str] = dict(DEFAULT_DRAFT_SPEC)
    if not tex_path.exists():
        spec["paper_tex"] = str(tex_path)
        spec["paper_tex_found"] = 0
        return spec

    text = tex_path.read_text()
    spec["paper_tex"] = str(tex_path)
    spec["paper_tex_found"] = 1

    match = re.search(r"Poisson\}\(\\lambda\s*=\s*([0-9.]+)\)", text)
    if match:
        spec["branch_lambda"] = float(match.group(1))
    match = re.search(r"n_\\mathrm\{nodes\}\s*=\s*([0-9]+)", text)
    if match:
        spec["training_node_budget"] = int(match.group(1))
    match = re.search(r"n_\\mathrm\{layers\}\s*=\s*([0-9]+)", text)
    if match:
        spec["max_depth"] = int(match.group(1))
    match = re.search(r"fewer than\s+([0-9]+)\s+leaves", text)
    if match:
        spec["min_leaves"] = int(match.group(1))
    match = re.search(r"trees with\s+([0-9]+)--([0-9]+)\s+observed leaves", text)
    if match:
        spec["stated_leaf_min"] = int(match.group(1))
        spec["stated_leaf_max"] = int(match.group(2))
    return spec


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def sample_sd(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def source_summary(
    records: list[TreeRecord],
    spec: dict[str, int | float | str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        subset = [record for record in records if record.source == source]
        if not subset:
            continue
        leaves = [record.n_leaves for record in subset]
        depths = [record.max_depth for record in subset]
        nodes = [record.n_training_nodes for record in subset]
        rows.append(
            {
                "source": source,
                "n_trees": len(subset),
                "training_nodes_min": min(nodes),
                "training_nodes_max": max(nodes),
                "training_nodes_mean": fmt(mean(nodes)),
                "leaves_min": min(leaves),
                "leaves_max": max(leaves),
                "leaves_mean": fmt(mean(leaves)),
                "leaves_sd": fmt(sample_sd(leaves)),
                "max_depth_min": min(depths),
                "max_depth_max": max(depths),
                "max_depth_mean": fmt(mean(depths)),
                "outside_stated_leaf_range": sum(
                    not (
                        int(spec["stated_leaf_min"])
                        <= record.n_leaves
                        <= int(spec["stated_leaf_max"])
                    )
                    for record in subset
                ),
                "over_draft_max_depth": sum(
                    record.max_depth > int(spec["max_depth"]) for record in subset
                ),
                "full_draft_budget": sum(
                    record.n_training_nodes == int(spec["training_node_budget"])
                    for record in subset
                ),
            }
        )
    return rows


def distribution_rows(records: list[TreeRecord], attr: str, label: str) -> list[dict[str, object]]:
    values = sorted({getattr(record, attr) for record in records})
    by_source = {
        source: Counter(getattr(record, attr) for record in records if record.source == source)
        for source in SOURCE_ORDER
    }
    rows: list[dict[str, object]] = []
    for value in values:
        row = {label: value}
        for source in SOURCE_ORDER:
            row[source] = by_source[source][value]
        rows.append(row)
    return rows


def distribution_fields(key: str) -> list[str]:
    return [key, *SOURCE_ORDER]


def svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 13,
    anchor: str = "middle",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="#1f2a33">'
        f"{escape(str(text))}</text>"
    )


def write_grouped_bar_svg(
    path: Path,
    rows: list[dict[str, object]],
    *,
    value_key: str,
    title: str,
    subtitle: str,
    target_min: int | None = None,
    target_max: int | None = None,
) -> None:
    width, height = 940, 560
    left, right, top, bottom = 84, 34, 78, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = max((int(row[source]) for row in rows for source in SOURCE_ORDER), default=1)
    y_top = max_count + max(1, int(max_count * 0.16))
    group_w = plot_w / max(len(rows), 1)
    bar_w = min(34, group_w / 3.2)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(24, 32, title, size=20, anchor="start", weight="700"),
        svg_text(24, 55, subtitle, size=13, anchor="start"),
        (
            f'<line x1="{left}" y1="{top + plot_h}" '
            f'x2="{left + plot_w}" y2="{top + plot_h}" '
            'stroke="#6c757d" stroke-width="1"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
            'stroke="#6c757d" stroke-width="1"/>'
        ),
    ]

    if target_min is not None and target_max is not None and rows:
        values = [int(row[value_key]) for row in rows]
        first = min(
            (idx for idx, value in enumerate(values) if target_min <= value <= target_max),
            default=None,
        )
        last = max(
            (idx for idx, value in enumerate(values) if target_min <= value <= target_max),
            default=None,
        )
        if first is not None and last is not None:
            x0 = left + first * group_w + 4
            x1 = left + (last + 1) * group_w - 4
            parts.append(
                f'<rect x="{x0:.1f}" y="{top}" width="{x1 - x0:.1f}" height="{plot_h}" '
                'fill="#f1e6b6" opacity="0.35"/>'
            )
            parts.append(
                svg_text(
                    (x0 + x1) / 2,
                    top + 18,
                    f"draft stated range {target_min}-{target_max}",
                    size=12,
                )
            )

    tick_step = max(1, round(y_top / 5))
    for tick in range(0, y_top + 1, tick_step):
        y = top + plot_h - (tick / y_top) * plot_h
        parts.append(
            f'<line x1="{left - 5}" y1="{y:.1f}" '
            f'x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#d5dadd" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, y + 4, tick, size=12, anchor="end"))

    for idx, row in enumerate(rows):
        center = left + idx * group_w + group_w / 2
        for source_idx, source in enumerate(SOURCE_ORDER):
            count = int(row[source])
            x = center + (source_idx - 0.5) * (bar_w + 5) - bar_w / 2
            h = (count / y_top) * plot_h if y_top else 0
            y = top + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'rx="2" fill="{COLORS[source]}"/>'
            )
            if count:
                parts.append(svg_text(x + bar_w / 2, y - 6, count, size=12))
        parts.append(svg_text(center, top + plot_h + 24, row[value_key], size=12))

    parts.append(svg_text(left + plot_w / 2, height - 22, value_key.replace("_", " "), size=13))
    parts.append(svg_text(18, top + plot_h / 2, "count", size=13, anchor="middle"))
    legend_x = width - 325
    legend_y = 24
    for idx, source in enumerate(SOURCE_ORDER):
        y = legend_y + idx * 22
        parts.append(
            f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" '
            f'fill="{COLORS[source]}"/>'
        )
        parts.append(svg_text(legend_x + 22, y, source, size=12, anchor="start"))

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_scatter_svg(
    path: Path,
    records: list[TreeRecord],
    spec: dict[str, int | float | str],
) -> None:
    width, height = 940, 560
    left, right, top, bottom = 84, 36, 78, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    leaf_min = min(record.n_leaves for record in records) - 0.8
    leaf_max = max(record.n_leaves for record in records) + 0.8
    node_min = min(record.n_training_nodes for record in records) - 0.8
    node_max = max(record.n_training_nodes for record in records) + 1.2

    def sx(leaf: float) -> float:
        return left + ((leaf - leaf_min) / (leaf_max - leaf_min)) * plot_w

    def sy(nodes: float) -> float:
        return top + plot_h - ((nodes - node_min) / (node_max - node_min)) * plot_h

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(24, 32, "Training nodes vs leaves", size=20, anchor="start", weight="700"),
        svg_text(
            24,
            55,
            f"horizontal line marks current draft node budget ({spec['training_node_budget']})",
            size=13,
            anchor="start",
        ),
        (
            f'<line x1="{left}" y1="{top + plot_h}" '
            f'x2="{left + plot_w}" y2="{top + plot_h}" '
            'stroke="#6c757d" stroke-width="1"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
            'stroke="#6c757d" stroke-width="1"/>'
        ),
    ]

    for leaf in range(int(leaf_min) + 1, int(leaf_max) + 1):
        x = sx(leaf)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{top + plot_h}" stroke="#e1e5e8" stroke-width="1"/>'
        )
        parts.append(svg_text(x, top + plot_h + 23, leaf, size=12))
    for nodes in range(int(node_min) + 1, int(node_max) + 1, 2):
        y = sy(nodes)
        parts.append(
            f'<line x1="{left - 5}" y1="{y:.1f}" '
            f'x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#e1e5e8" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, y + 4, nodes, size=12, anchor="end"))

    budget = float(spec["training_node_budget"])
    parts.append(
        f'<line x1="{left}" y1="{sy(budget):.1f}" x2="{left + plot_w}" y2="{sy(budget):.1f}" '
        'stroke="#c94837" stroke-width="2" stroke-dasharray="7 5"/>'
    )

    source_counts: defaultdict[tuple[str, int, int], int] = defaultdict(int)
    for record in records:
        key = (record.source, record.n_leaves, record.n_training_nodes)
        offset_index = source_counts[key]
        source_counts[key] += 1
        jitter_x = ((offset_index % 5) - 2) * 0.055
        jitter_y = ((offset_index // 5) % 5 - 2) * 0.055
        x = sx(record.n_leaves + jitter_x)
        y = sy(record.n_training_nodes + jitter_y)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{COLORS[record.source]}" '
            'opacity="0.78" stroke="#ffffff" stroke-width="1"/>'
        )

    parts.append(svg_text(left + plot_w / 2, height - 22, "n_leaves", size=13))
    parts.append(svg_text(18, top + plot_h / 2, "n_training_nodes", size=13))
    legend_x = width - 325
    legend_y = 24
    for idx, source in enumerate(SOURCE_ORDER):
        y = legend_y + idx * 22
        parts.append(f'<circle cx="{legend_x + 7}" cy="{y - 6}" r="6" fill="{COLORS[source]}"/>')
        parts.append(svg_text(legend_x + 22, y, source, size=12, anchor="start"))

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_report(
    path: Path,
    *,
    spec: dict[str, int | float | str],
    summary_rows: list[dict[str, object]],
    leaf_rows: list[dict[str, object]],
) -> None:
    columns = [
        "source",
        "n_trees",
        "training_nodes_min",
        "training_nodes_max",
        "training_nodes_mean",
        "leaves_min",
        "leaves_max",
        "leaves_mean",
        "leaves_sd",
        "max_depth_min",
        "max_depth_max",
        "max_depth_mean",
        "outside_stated_leaf_range",
        "over_draft_max_depth",
        "full_draft_budget",
    ]
    leaf_columns = ["n_leaves", *SOURCE_ORDER]

    lines = [
        "# Confirm Paper Tree Comparison",
        "",
        "## Draft spec target",
        "",
        f"- paper tex: `{spec['paper_tex']}`",
        f"- n_trees: {spec['n_trees']}",
        f"- branch_lambda: {spec['branch_lambda']}",
        f"- training_node_budget: {spec['training_node_budget']}",
        f"- max_depth: {spec['max_depth']}",
        f"- min_leaves: {spec['min_leaves']}",
        f"- stated observed leaf range: {spec['stated_leaf_min']}-{spec['stated_leaf_max']}",
        "",
        "## Source summary",
        "",
        markdown_table(summary_rows, columns),
        "",
        "## Leaf count distribution",
        "",
        markdown_table(leaf_rows, leaf_columns),
        "",
        "## Graphs",
        "",
        "![Leaf count distribution](leaf_count_distribution.svg)",
        "",
        "![Max-depth distribution](depth_distribution.svg)",
        "",
        "![Training nodes vs leaves](training_nodes_vs_leaves.svg)",
        "",
        "## Files",
        "",
        "- `per_tree_stats.csv`",
        "- `source_summary.csv`",
        "- `leaf_count_distribution.csv`",
        "- `depth_distribution.csv`",
        "- `training_node_distribution.csv`",
        "- `draft_spec.json`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec = parse_draft_spec(args.paper_tex)
    weighttraits = load_weighttraits(args.weighttraits_summary)
    ellmtrees = load_ellmtrees(args.ellmtrees_runs)
    records = [*ellmtrees, *weighttraits]

    per_tree_rows = [record.as_row(spec) for record in records]
    write_csv(
        args.out_dir / "per_tree_stats.csv",
        per_tree_rows,
        [
            "source",
            "tree_id",
            "manifest",
            "seed",
            "candidate_index",
            "n_training_nodes",
            "n_leaves",
            "max_depth",
            "max_out_degree",
            "n_polytomies",
            "within_stated_leaf_range",
            "within_max_depth",
            "uses_full_training_budget",
        ],
    )

    summary_rows = source_summary(records, spec)
    write_csv(args.out_dir / "source_summary.csv", summary_rows, list(summary_rows[0]))

    leaf_rows = distribution_rows(records, "n_leaves", "n_leaves")
    depth_rows = distribution_rows(records, "max_depth", "max_depth")
    node_rows = distribution_rows(records, "n_training_nodes", "n_training_nodes")
    write_csv(
        args.out_dir / "leaf_count_distribution.csv",
        leaf_rows,
        distribution_fields("n_leaves"),
    )
    write_csv(args.out_dir / "depth_distribution.csv", depth_rows, distribution_fields("max_depth"))
    write_csv(
        args.out_dir / "training_node_distribution.csv",
        node_rows,
        distribution_fields("n_training_nodes"),
    )
    (args.out_dir / "draft_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    write_grouped_bar_svg(
        args.out_dir / "leaf_count_distribution.svg",
        leaf_rows,
        value_key="n_leaves",
        title="Leaf count distribution",
        subtitle=(
            "WeightTraits has the accepted heavier leaf tail; shaded range is stated "
            "in the current draft."
        ),
        target_min=int(spec["stated_leaf_min"]),
        target_max=int(spec["stated_leaf_max"]),
    )
    write_grouped_bar_svg(
        args.out_dir / "depth_distribution.svg",
        depth_rows,
        value_key="max_depth",
        title="Max-depth distribution",
        subtitle=f"Current draft max depth target is {spec['max_depth']}.",
        target_min=0,
        target_max=int(spec["max_depth"]),
    )
    write_scatter_svg(args.out_dir / "training_nodes_vs_leaves.svg", records, spec)
    write_report(
        args.out_dir / "README.md",
        spec=spec,
        summary_rows=summary_rows,
        leaf_rows=leaf_rows,
    )

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "n_records": len(records),
                "sources": {
                    source: sum(1 for record in records if record.source == source)
                    for source in SOURCE_ORDER
                },
                "summary": summary_rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
