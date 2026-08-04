#!/usr/bin/env python3
"""Build Figure 2 summary points from full-FT and merged-weight LoRA analyses.

Every output row is a summary across the same topology-eligible trees used by
the paper table.  Whole-simulation rows come from the run-set rollups.  Subset
and single-matrix rows are rescored from the saved per-layer cosine-distance
cubes; model checkpoints are not reloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from weighttraits.analysis.direct import _neighbor_joining_newick  # noqa: E402
from weighttraits.phylo.additivity import four_point_additivity  # noqa: E402
from weighttraits.phylo.atteson import atteson_margin  # noqa: E402
from weighttraits.phylo.recovery import score_split_recovery  # noqa: E402
from weighttraits.phylo.splits import splits_from_newick_text  # noqa: E402


SOURCES = (
    {
        "condition_id": "flan_full_ft",
        "architecture": "Flan-T5",
        "condition": "Full fine-tuning (Flan)",
        "root": "outputs/analysis_v20260713/full_finetune",
        "rollup": "outputs/analysis_v20260713/full_finetune_summary.json",
        "artifact": "model",
        "analysis_dir": "model_leaf_analysis",
        "detail": True,
    },
    {
        "condition_id": "flan_qv",
        "architecture": "Flan-T5",
        "condition": "LoRA q/v",
        "root": "outputs/analysis_flan_merged_weights_20260804/lora_finetune",
        "rollup": "outputs/analysis_flan_merged_weights_20260804/aggregate/lora_finetune_rollup.json",
        "detail": False,
    },
    {
        "condition_id": "flan_key",
        "architecture": "Flan-T5",
        "condition": "LoRA key only",
        "root": "outputs/analysis_flan_merged_weights_20260804/lora_k_only",
        "rollup": "outputs/analysis_flan_merged_weights_20260804/aggregate/lora_k_only_rollup.json",
        "detail": False,
    },
    {
        "condition_id": "flan_qkv",
        "architecture": "Flan-T5",
        "condition": "LoRA q/k/v",
        "root": "outputs/analysis_flan_merged_weights_20260804/lora_qkv",
        "rollup": "outputs/analysis_flan_merged_weights_20260804/aggregate/lora_qkv_rollup.json",
        "detail": True,
    },
    {
        "condition_id": "flan_attention",
        "architecture": "Flan-T5",
        "condition": "LoRA full attention",
        "root": "outputs/analysis_flan_merged_weights_20260804/lora_full_attn",
        "rollup": "outputs/analysis_flan_merged_weights_20260804/aggregate/lora_full_attn_rollup.json",
        "detail": False,
    },
    {
        "condition_id": "llama_full_ft",
        "architecture": "Llama-3.2-1B",
        "condition": "Full fine-tuning (Llama)",
        "root": "outputs/analysis_llama32_full_ft_20260722/production",
        "rollup": "outputs/analysis_llama32_full_ft_20260722/production_rollup.json",
        "artifact": "model",
        "analysis_dir": "model_leaf_analysis",
        "detail": True,
    },
    {
        "condition_id": "llama_r8",
        "architecture": "Llama-3.2-1B",
        "condition": "LoRA q/k/v (r=8)",
        "root": "outputs/analysis_llama32_merged_weights_20260804/llama32_1b_lora_qkv_r8",
        "rollup": "outputs/analysis_llama32_merged_weights_20260804/aggregate/llama32_1b_lora_qkv_r8_rollup.json",
        "detail": True,
    },
    {
        "condition_id": "llama_r64",
        "architecture": "Llama-3.2-1B",
        "condition": "LoRA q/k/v (r=64)",
        "root": "outputs/analysis_llama32_merged_weights_20260804/llama32_1b_lora_qkv_r64",
        "rollup": "outputs/analysis_llama32_merged_weights_20260804/aggregate/llama32_1b_lora_qkv_r64_rollup.json",
        "detail": True,
    },
)

PHYLOLM_GROUPS = {
    "runs_weighttraits_llama32_1b_lora_qkv_r8_fresh_phylolm": "llama_r8",
    "runs_weighttraits_llama32_1b_lora_qkv_r64_fresh_phylolm": "llama_r64",
    "runs_weighttraits_llama32_1b_full_finetune_fresh_phylolm": "llama_full_ft",
}


def _mean_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def _summarize(
    records: list[dict[str, float]],
    *,
    source: dict[str, Any],
    level: str,
    label: str,
    family: str,
    projection: str = "aggregate",
) -> dict[str, Any]:
    if len(records) != 46:
        raise ValueError(
            f"expected 46 eligible trees for {source['condition']} / {label}, got {len(records)}"
        )
    row: dict[str, Any] = {
        "condition_id": source["condition_id"],
        "architecture": source["architecture"],
        "condition": source["condition"],
        "level": level,
        "label": label,
        "family": family,
        "projection": projection,
        "n_runs": len(records),
        "artifact": source.get("artifact", "merged"),
        "representation": source.get("representation", "full_weight"),
        "method": source.get("method", "Weights"),
    }
    for output, field in (
        ("clade_recovery_pct", "clade_recovery"),
        ("atteson_bottleneck_margin", "atteson_bottleneck_margin"),
        ("four_point_mean_additivity", "four_point_mean_additivity"),
    ):
        values = [float(record[field]) for record in records]
        mean, se = _mean_se(values)
        if output == "clade_recovery_pct":
            mean *= 100.0
            se *= 100.0
        row[output] = mean
        row[f"{output}_se"] = se
        row[f"{output}_median"] = statistics.median(values) * (
            100.0 if output == "clade_recovery_pct" else 1.0
        )
    return row


def _simulation_row(source: dict[str, Any], root: Path) -> tuple[dict[str, Any], list[str]]:
    path = root / source["rollup"]
    payload = json.loads(path.read_text())
    rows = [
        row
        for row in payload["rows"]
        if row["metric"] == "cosine" and int(row["n_truth_splits"]) > 0
    ]
    expected_artifact = source.get("artifact", "merged")
    if payload.get("artifact") != expected_artifact:
        raise ValueError(f"expected {expected_artifact} artifact: {path}")
    if {str(row.get("representation")) for row in rows} != {"full_weight"}:
        raise ValueError(f"expected full_weight rows: {path}")
    summary = _summarize(
        rows,
        source=source,
        level="simulation",
        label=source["condition"],
        family="whole simulation",
    )
    return summary, sorted(str(row["tree_id"]) for row in rows)


def _phylolm_rows(path: Path) -> list[dict[str, Any]]:
    sources_by_id = {source["condition_id"]: source for source in SOURCES}
    records_by_group: dict[str, list[dict[str, float]]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            group = str(raw["group"])
            if group not in PHYLOLM_GROUPS:
                continue
            records_by_group.setdefault(group, []).append(
                {
                    "clade_recovery": float(raw["clade_recovery"]),
                    "atteson_bottleneck_margin": float(raw["oracle_margin"]),
                    "four_point_mean_additivity": float(raw["A_all"]),
                }
            )

    if set(records_by_group) != set(PHYLOLM_GROUPS):
        missing = sorted(set(PHYLOLM_GROUPS) - set(records_by_group))
        raise ValueError(f"missing fresh PhyloLM groups in {path}: {missing}")

    out = []
    for group, condition_id in PHYLOLM_GROUPS.items():
        source = {
            **sources_by_id[condition_id],
            "artifact": "behavior",
            "representation": "phylolm_nei_distance",
            "method": "PhyloLM",
        }
        out.append(
            _summarize(
                records_by_group[group],
                source=source,
                level="phylolm",
                label="PhyloLM",
                family="behavior-only simulation",
            )
        )
    return out


def _projection(name: str) -> str | None:
    for projection in ("q", "k", "v"):
        if name.endswith(f".{projection}.weight") or name.endswith(f".{projection}_proj.weight"):
            return projection
    return None


def _flan_subsets() -> list[tuple[str, str, Callable[[str], bool]]]:
    def qkv(name: str) -> bool:
        return _projection(name) is not None

    return [
        ("flan_k", "All key projections", lambda name: _projection(name) == "k"),
        ("flan_q", "All query projections", lambda name: _projection(name) == "q"),
        ("flan_v", "All value projections", lambda name: _projection(name) == "v"),
        (
            "flan_encoder_self",
            "Encoder self-attention QKV",
            lambda name: name.startswith("encoder.") and "SelfAttention" in name and qkv(name),
        ),
        (
            "flan_decoder_self",
            "Decoder self-attention QKV",
            lambda name: name.startswith("decoder.") and "SelfAttention" in name and qkv(name),
        ),
        (
            "flan_cross",
            "Decoder cross-attention QKV",
            lambda name: "EncDecAttention" in name and qkv(name),
        ),
        (
            "flan_self",
            "All self-attention QKV",
            lambda name: "SelfAttention" in name and qkv(name),
        ),
    ]


def _llama_block(name: str) -> int | None:
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix) :].split(".", 1)[0])
    except ValueError:
        return None


def _llama_subsets() -> list[tuple[str, str, Callable[[str], bool]]]:
    return [
        ("llama_k", "Key projections", lambda name: _projection(name) == "k"),
        ("llama_q", "Query projections", lambda name: _projection(name) == "q"),
        ("llama_v", "Value projections", lambda name: _projection(name) == "v"),
        (
            "llama_early",
            "Early-block QKV (0-7)",
            lambda name: _projection(name) is not None
            and _llama_block(name) is not None
            and _llama_block(name) < 8,
        ),
        (
            "llama_late",
            "Late-block QKV (8-15)",
            lambda name: _projection(name) is not None
            and _llama_block(name) is not None
            and _llama_block(name) >= 8,
        ),
    ]


def _score_matrix(
    matrix: np.ndarray,
    labels: list[str],
    truth_newick: str,
) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    if not np.isfinite(values).all() or float(np.max(values)) <= 1e-14:
        raise ValueError("distance matrix has no usable variation")
    estimated = _neighbor_joining_newick(labels, values)
    truth_splits, truth_leaves = splits_from_newick_text(truth_newick)
    estimated_splits, estimated_leaves = splits_from_newick_text(estimated)
    score = score_split_recovery(
        truth_splits,
        estimated_splits,
        truth_leaves,
        estimated_leaves,
    )
    additivity = four_point_additivity(labels, values, truth_splits=truth_splits)
    margin = atteson_margin(labels, values, truth_newick)
    if additivity.all is None:
        raise ValueError("distance matrix has no quartets")
    return {
        "clade_recovery": float(score["clade_recovery"]),
        "atteson_bottleneck_margin": float(margin.bottleneck_margin),
        "four_point_mean_additivity": float(additivity.all.mean_additivity),
    }


def _detail_rows(
    source: dict[str, Any],
    root: Path,
    tree_ids: list[str],
) -> list[dict[str, Any]]:
    records_by_layer: dict[str, list[dict[str, float]]] = {}
    records_by_subset: dict[str, list[dict[str, float]]] = {}
    subset_specs = _flan_subsets() if source["architecture"] == "Flan-T5" else _llama_subsets()
    subset_labels = {subset_id: label for subset_id, label, _ in subset_specs}

    for tree_id in tree_ids:
        analysis = (
            root
            / source["root"]
            / tree_id
            / source.get("analysis_dir", "merged_leaf_analysis")
        )
        summary = json.loads((analysis / "summary.json").read_text())
        expected_artifact = source.get("artifact", "merged")
        if (
            summary.get("artifact") != expected_artifact
            or summary.get("representation") != "full_weight"
        ):
            raise ValueError(f"not an expected full-weight analysis: {analysis}")
        layer_names = [str(name) for name in json.loads((analysis / "layers.json").read_text())]
        labels = [str(name) for name in json.loads((analysis / "models.json").read_text())]
        truth_newick = (analysis / "truth_manifest.newick").read_text().strip()
        with np.load(analysis / "direct_distance_layers.npz") as archive:
            cube = np.asarray(archive["cosine"], dtype=np.float64)
        if cube.shape[0] != len(layer_names):
            raise ValueError(f"layer cube mismatch: {analysis}")

        qkv_indices = [index for index, name in enumerate(layer_names) if _projection(name)]
        for index in qkv_indices:
            name = layer_names[index]
            records_by_layer.setdefault(name, []).append(
                _score_matrix(cube[index], labels, truth_newick)
            )

        for subset_id, _, predicate in subset_specs:
            indices = [index for index, name in enumerate(layer_names) if predicate(name)]
            if not indices:
                raise ValueError(f"empty subset {subset_id}: {analysis}")
            matrix = np.mean(cube[indices], axis=0)
            records_by_subset.setdefault(subset_id, []).append(
                _score_matrix(matrix, labels, truth_newick)
            )

    out = []
    for subset_id, records in sorted(records_by_subset.items()):
        out.append(
            _summarize(
                records,
                source=source,
                level="subset",
                label=subset_labels[subset_id],
                family=subset_id,
            )
        )
    for layer_name, records in sorted(records_by_layer.items()):
        projection = _projection(layer_name)
        if projection is None:
            continue
        out.append(
            _summarize(
                records,
                source=source,
                level="layer",
                label=layer_name,
                family="single projection matrix",
                projection=projection,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighttraits-root", type=Path, default=REPO)
    parser.add_argument(
        "--phylolm-metrics",
        type=Path,
        default=REPO.parent
        / "ELLMTrees"
        / "results"
        / "aggregate"
        / "weighttraits_fresh_phylolm"
        / "per_run_metrics.csv",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    root = args.weighttraits_root.resolve()
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        simulation, tree_ids = _simulation_row(source, root)
        rows.append(simulation)
        if source["detail"]:
            rows.extend(_detail_rows(source, root, tree_ids))
    rows.extend(_phylolm_rows(args.phylolm_metrics.resolve()))

    payload = {
        "schema": "weighttraits.merged_weight_figure2.v1",
        "producer": "weighttraits",
        "metric": "cosine",
        "n_rows": len(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} and {args.csv_out} ({len(rows)} summary points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
