#!/usr/bin/env python3
"""Build appendix diagnostics for collapsed Flan behavioral outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROBES = ("hellaswag", "arc_challenge", "mmlu", "truthfulqa", "dolly_open_ended")
CLASSES = ("empty", "label_only", "fragment", "repetition_loop", "natural_language")
CLASS_LABELS = {
    "empty": "Empty",
    "label_only": "Label only",
    "fragment": "Fragment",
    "repetition_loop": "Repetition",
    "natural_language": "Natural language",
}
PROBE_LABELS = {
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-C",
    "mmlu": "MMLU",
    "truthfulqa": "TruthfulQA",
    "dolly_open_ended": "Dolly",
}
FAMILY_ORDER = ("classification", "qa", "summarization", "translation")
FAMILY_LABELS = {
    "classification": "Classification",
    "qa": "QA",
    "summarization": "Summarization",
    "translation": "Translation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-root", type=Path, required=True)
    parser.add_argument("--dolly-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def embedding_coverage(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as payload:
        models = model_strings(payload["model_ids"])
        valid = payload["valid_mask"]
        return {model: float(valid[idx].mean()) for idx, model in enumerate(models)}


def terminal_metadata(manifest_path: Path) -> dict[str, dict]:
    rows = read_jsonl(manifest_path)
    return {str(row["node_id"]): row for row in rows}


def response_summaries(response_dir: Path) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted(response_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            grouped[(str(row["probe_id"]), str(row["model_id"]))].append(
                str(row.get("response", ""))
            )
    summaries = {}
    for key, values in grouped.items():
        counts = Counter(values)
        dominant, dominant_count = counts.most_common(1)[0]
        summaries[key] = {
            "n_responses": len(values),
            "n_unique_responses": len(counts),
            "unique_response_fraction": len(counts) / len(values),
            "dominant_response": dominant if dominant else "<EMPTY>",
            "dominant_response_count": dominant_count,
            "dominant_response_fraction": dominant_count / len(values),
            "mean_response_characters": float(np.mean([len(value) for value in values])),
        }
    return summaries


def collect_rows(root: Path, probes: tuple[str, ...], manifest_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for tree_number in range(1, 51):
        tree_dir = root / f"tree{tree_number:03d}"
        tree_id = f"confirm_paper_tree_{tree_number:03d}"
        metadata = terminal_metadata(manifest_dir / f"{tree_id}.manifest.jsonl")
        response_stats = response_summaries(tree_dir / "responses")
        for probe in probes:
            traits = json.loads(
                (tree_dir / f"{probe}_surface_distances" / "output_surface_traits.json").read_text()
            )["traits_by_model"]
            coverage = embedding_coverage(tree_dir / f"{probe}_embeddings.npz")
            for model_id, model_traits in sorted(traits.items()):
                terminal = metadata[model_id]
                response = response_stats[(probe, model_id)]
                fractions = model_traits["class_fractions"]
                row = {
                    "tree_id": tree_id,
                    "tree_number": tree_number,
                    "model_id": model_id,
                    "probe": probe,
                    "terminal_task_family": terminal["task_family"],
                    "terminal_dataset_id": terminal["dataset_id"],
                    "depth": terminal["depth"],
                    "n_observations": model_traits["n_observations"],
                    "embedding_coverage": coverage[model_id],
                    "natural_language_fraction": fractions["natural_language"],
                    "non_natural_output_fraction": 1.0 - fractions["natural_language"],
                    **{f"{name}_fraction": fractions[name] for name in CLASSES},
                    **response,
                }
                rows.append(row)
    return rows


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    return float(np.average(group[column], weights=group["n_observations"]))


def aggregate_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, probe), group in data.groupby(["terminal_task_family", "probe"], sort=False):
        row = {
            "terminal_task_family": family,
            "probe": probe,
            "n_leaf_models": len(group),
            "n_observations": int(group["n_observations"].sum()),
            "mean_embedding_coverage": weighted_mean(group, "embedding_coverage"),
            "median_natural_language_fraction": float(group["natural_language_fraction"].median()),
            "median_unique_response_fraction": float(group["unique_response_fraction"].median()),
            "median_dominant_response_fraction": float(group["dominant_response_fraction"].median()),
        }
        row.update({f"mean_{name}_fraction": weighted_mean(group, f"{name}_fraction") for name in CLASSES})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["terminal_task_family", "probe"])


def example_table(data: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, group in data.groupby(["terminal_task_family", "probe"], sort=False):
        ranked = group.sort_values(
            ["natural_language_fraction", "dominant_response_fraction", "embedding_coverage"],
            ascending=[True, False, True],
        )
        selected.append(ranked.head(2))
    columns = [
        "tree_id",
        "model_id",
        "terminal_task_family",
        "terminal_dataset_id",
        "probe",
        "natural_language_fraction",
        "embedding_coverage",
        "dominant_response_fraction",
        "dominant_response",
    ]
    return pd.concat(selected, ignore_index=True)[columns]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    title = stem.replace("-", " ").title()
    description = "Appendix diagnostic for Flan-T5 behavioral output collapse"
    fig.savefig(
        out_dir / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Title": title, "Description": description},
    )
    fig.savefig(
        out_dir / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"Title": title, "Author": "WeightTraits", "Subject": description},
    )
    plt.close(fig)


def plot_composition(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(13.2, 3.7), sharey=True)
    colors = {
        "empty": "#6c757d",
        "label_only": "#d55e00",
        "fragment": "#e69f00",
        "repetition_loop": "#cc79a7",
        "natural_language": "#0072b2",
    }
    for axis, family in zip(axes, FAMILY_ORDER):
        family_data = summary[summary["terminal_task_family"] == family].set_index("probe")
        bottom = np.zeros(len(PROBES))
        x = np.arange(len(PROBES))
        for name in CLASSES:
            values = np.array(
                [family_data.loc[probe, f"mean_{name}_fraction"] for probe in PROBES]
            )
            axis.bar(x, values, bottom=bottom, color=colors[name], width=0.78, label=CLASS_LABELS[name])
            bottom += values
        axis.set_title(FAMILY_LABELS[family])
        axis.set_xticks(x, [PROBE_LABELS[probe] for probe in PROBES], rotation=55, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="0.88", linewidth=0.6)
    axes[0].set_ylabel("Fraction of generated outputs")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(top=0.80, bottom=0.28, wspace=0.10)
    save_figure(fig, out_dir, "flan-output-composition")


def plot_coverage(data: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(PROBES), figsize=(13.2, 2.9), sharex=True, sharey=True)
    family_colors = {
        "classification": "#d55e00",
        "qa": "#0072b2",
        "summarization": "#009e73",
        "translation": "#cc79a7",
    }
    for axis, probe in zip(axes, PROBES):
        subset = data[data["probe"] == probe]
        for family in FAMILY_ORDER:
            group = subset[subset["terminal_task_family"] == family]
            axis.scatter(
                group["natural_language_fraction"],
                group["embedding_coverage"],
                s=9,
                alpha=0.42,
                color=family_colors[family],
                label=FAMILY_LABELS[family],
                rasterized=True,
            )
        axis.plot([0, 1], [0, 1], color="0.45", linewidth=0.8, linestyle="--")
        axis.set_title(PROBE_LABELS[probe])
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(color="0.90", linewidth=0.6)
    axes[0].set_ylabel("Nonempty embedding coverage")
    axes[len(axes) // 2].set_xlabel("Natural-language output fraction")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.79, bottom=0.20, wspace=0.10)
    save_figure(fig, out_dir, "flan-semantic-coverage-diagnostic")


def plot_dominance(data: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(PROBES), figsize=(13.2, 3.0), sharey=True)
    for axis, probe in zip(axes, PROBES):
        subset = data[data["probe"] == probe]
        groups = [
            subset[subset["terminal_task_family"] == family]["dominant_response_fraction"].values
            for family in FAMILY_ORDER
        ]
        axis.boxplot(groups, showfliers=False, widths=0.65, patch_artist=False)
        axis.set_title(PROBE_LABELS[probe])
        axis.set_xticks(range(1, len(FAMILY_ORDER) + 1), [FAMILY_LABELS[x] for x in FAMILY_ORDER], rotation=55, ha="right")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", color="0.90", linewidth=0.6)
    axes[0].set_ylabel("Dominant-response fraction")
    fig.subplots_adjust(bottom=0.32, wspace=0.10)
    save_figure(fig, out_dir, "flan-dominant-response")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.mc_root, PROBES[:4], args.manifest_dir)
    rows.extend(collect_rows(args.dolly_root, PROBES[4:], args.manifest_dir))
    data = pd.DataFrame(rows)
    summary = aggregate_table(data)
    examples = example_table(data)
    data.to_csv(args.out_dir / "leaf_probe_diagnostics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(args.out_dir / "task_family_probe_summary.csv", index=False)
    examples.to_csv(args.out_dir / "collapse_examples.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    plot_composition(summary, args.out_dir)
    plot_coverage(data, args.out_dir)
    plot_dominance(data, args.out_dir)
    audit = {
        "schema_version": 1,
        "n_trees": int(data["tree_id"].nunique()),
        "n_leaf_probe_rows": len(data),
        "n_unique_leaf_models": int(data[["tree_id", "model_id"]].drop_duplicates().shape[0]),
        "probes": list(PROBES),
        "terminal_task_families": list(FAMILY_ORDER),
        "output_classes": list(CLASSES),
        "definitions": {
            "embedding_coverage": "fraction of prompt-draw outputs that were nonempty and embedded",
            "natural_language_fraction": "fraction classified as natural-language output by the surface endpoint",
            "dominant_response_fraction": "frequency of the most common exact response, including empty strings",
        },
        "issues": [],
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
