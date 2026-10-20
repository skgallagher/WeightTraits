"""Figure data for the documented-family three-panel trees (App. D, Figs 5 and 6).

Writes ``<stem>_tree_segments.csv`` and ``<stem>_tree_tips.csv`` (true tree | weight estimate |
PhyloLM estimate on a shared tip set) for ``R/hf_family_trees.R``. Panel headers carry the
clades estimated and the all-quartet tree-likeness score from :mod:`weighttraits.paper.hf_families`.

Ported from ELLMTrees ``scripts/export_ggplot_data.py`` (``export_tree_panels``) and the aligned-tip
cladogram layout in ``codex/plot_hf_zoo_tree_vs_truth.py``; output is unchanged.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

MISTRAL_NAMES = {
    "Mistral-7B-v0.1": "Mistral-7B",
    "Mistral-7B-Instruct-v0.1": "Mistral-Instruct",
    "zephyr-7b-sft-full": "Zephyr-SFT",
    "zephyr-7b-beta": "Zephyr-beta",
    "openchat_3.5": "OpenChat-3.5",
    "Starling-LM-7B-alpha": "Starling-alpha",
    "openchat-3.5-0106": "OpenChat-0106",
    "Starling-LM-7B-beta": "Starling-beta",
    "OpenHermes-2-Mistral-7B": "OpenHermes-2",
    "OpenHermes-2.5-Mistral-7B": "OpenHermes-2.5",
}
LLAMA2_BASE_GROUP = {
    "meta-llama__Llama-2-7b-hf",
    "NousResearch__Nous-Hermes-llama-2-7b",
    "lmsys__vicuna-7b-v1.5",
}
LLAMA2_NAMES = {
    "meta-llama__Llama-2-7b-hf": "Llama-2-7B (base)",
    "NousResearch__Nous-Hermes-llama-2-7b": "Nous-Hermes",
    "lmsys__vicuna-7b-v1.5": "Vicuna-7B-v1.5",
    "meta-llama__Llama-2-7b-chat-hf": "Llama-2-7B-chat",
    "Atulit23__meta-llama-indian-constitution-chat": "Indian-Constitution-chat",
    "BLACKBUN__llama-2-7b-pubmed-qa-211k": "PubMedQA-211k",
    "QIAIUNCC__EYE-Llama_gqa": "EYE-Llama-gqa",
    "QIAIUNCC__EYE-Llama_qa": "EYE-Llama-qa",
    "bineric__NorskGPT-Llama-7B-v0.1": "NorskGPT-Llama",
    "whatdhack__Llama-2-7b-chat-hf-oasst1-ft-sg": "chat-oasst1-ft",
}


def mistral_lineage(label: str) -> str:
    if "zephyr" in label.lower():
        return "Zephyr fork"
    if "openchat_3.5" in label or "Starling-LM-7B-alpha" in label:
        return "OpenChat-Starling alpha"
    if "openchat-3.5-0106" in label or "Starling-LM-7B-beta" in label:
        return "OpenChat-Starling beta"
    return "Other checkpoints"


def mistral_display(label: str) -> str:
    name = label.split("__")[-1]
    return MISTRAL_NAMES.get(name, name)


def llama2_lineage(label: str) -> str:
    return "Base / Nous-Hermes / Vicuna" if label in LLAMA2_BASE_GROUP else (
        "Chat clade (chat + 6 descendants)"
    )


def llama2_display(label: str) -> str:
    return LLAMA2_NAMES.get(label, label)


def load_tree(path: Path):
    import dendropy

    return dendropy.Tree.get(path=str(path), schema="newick", preserve_underscores=True)


def layout(tree) -> tuple[dict, int, int]:
    """Aligned-tip cladogram layout: ``positions[node] = (x, y)``, leaf count, max depth.

    Leaves sit at their topological depth (a dotted connector runs to the shared tip at x=1);
    an internal node's x is the minimum of its children's; y is the mean of its children's.
    """

    leaves = list(tree.leaf_node_iter())
    order = {leaf: i for i, leaf in enumerate(tree.leaf_node_iter())}
    y = {leaf: order[leaf] for leaf in leaves}
    depth = {}
    for node in tree.preorder_node_iter():
        depth[node] = 0 if node.parent_node is None else depth[node.parent_node] + 1
    max_depth = max(depth.values()) if depth else 1
    x = {}
    for node in tree.postorder_node_iter():
        x[node] = depth[node] if node.is_leaf() else min(x[c] for c in node.child_nodes())
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children_y = [y[c] for c in node.child_nodes()]
            y[node] = sum(children_y) / len(children_y)
    return {node: (x[node], y[node]) for node in tree.preorder_node_iter()}, len(leaves), max_depth


def tree_panel_rows(
    tree_specs: Sequence[tuple[str, Path]],
    lineage: Callable[[str], str],
    display: Callable[[str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Segments and tips for several trees laid out on their common tip set."""

    trees = [(panel, load_tree(path)) for panel, path in tree_specs]
    common = set.intersection(
        *({leaf.taxon.label for leaf in tree.leaf_node_iter()} for _, tree in trees)
    )
    for _, tree in trees:
        tree.retain_taxa_with_labels(sorted(common))
    segments: list[dict[str, Any]] = []
    tips: list[dict[str, Any]] = []
    for panel_order, (panel, tree) in enumerate(trees, start=1):
        positions, _, max_depth = layout(tree)
        denominator = max(max_depth, 1)
        base = {"panel": panel, "panel_order": panel_order}
        for node in tree.preorder_node_iter():
            x0, y0 = positions[node][0] / denominator, positions[node][1]
            if node.parent_node is not None:
                px = positions[node.parent_node][0] / denominator
                py = positions[node.parent_node][1]
                segments.append({**base, "kind": "tree", "x": px, "xend": px, "y": py, "yend": y0})
                segments.append({**base, "kind": "tree", "x": px, "xend": x0, "y": y0, "yend": y0})
            if node.is_leaf():
                label = node.taxon.label
                segments.append(
                    {**base, "kind": "connector", "x": x0, "xend": 1.0, "y": y0, "yend": y0}
                )
                tips.append(
                    {
                        **base,
                        "label": label,
                        "display": display(label),
                        "lineage": lineage(label),
                        "x": 1.0,
                        "y": y0,
                    }
                )
    return segments, tips


def panel_header(method: str, scored: dict[str, Any]) -> str:
    clades = scored["n_documented_clades"]
    noun = "clade" if clades == 1 else "clades"
    return (
        f"{method}\n{scored['clades_estimated']}/{clades} {noun} | "
        f"A = {scored['tree_likeness_all_quartets']:.2f}"
    )


def write_family_figure_data(
    stem: str,
    *,
    truth_newick: Path,
    weights_newick: Path,
    phylolm_newick: Path,
    scores: dict[str, Any],
    lineage: Callable[[str], str],
    display: Callable[[str], str],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write ``<stem>_tree_{segments,tips}.csv``; ``scores`` is ``score_family(...)["methods"]``."""

    specs = [
        ("True tree", truth_newick),
        (panel_header("Weights", scores["weights"]), weights_newick),
        (panel_header("PhyloLM", scores["phylolm"]), phylolm_newick),
    ]
    segments, tips = tree_panel_rows(specs, lineage, display)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = (out_dir / f"{stem}_tree_segments.csv", out_dir / f"{stem}_tree_tips.csv")
    for path, rows, fields in (
        (paths[0], segments, ["panel", "panel_order", "kind", "x", "xend", "y", "yend"]),
        (paths[1], tips, ["panel", "panel_order", "label", "display", "lineage", "x", "y"]),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return paths
