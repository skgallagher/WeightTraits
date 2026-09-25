#!/usr/bin/env python3
"""Build the submission Figure 2 panel from the fixed-2000 primary suites.

This is a narrow adapter around ``build_merged_weight_figure2.py``.  It swaps
the archived July/August sources for the receipt-validated September sources
used by Table 2.  The cumulative-update all-projections ablation is excluded
because it is not a complete merged-weight checkpoint suite.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT = Path(__file__).with_name("build_merged_weight_figure2.py")
SPEC = importlib.util.spec_from_file_location("weighttraits_figure2_base", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Figure 2 builder: {SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

from weighttraits.analysis.direct import truth_newick_from_manifest


FINAL_SOURCES = (
    {
        "condition_id": "flan_full_ft",
        "architecture": "Flan-T5",
        "condition": "Full fine-tuning (Flan)",
        "root": "direct/flan_full/analysis",
        "rollup": "direct/flan_full/direct_run_set_summary.json",
        "artifact": "model",
        "analysis_dir": "model_leaf_analysis",
        "detail": True,
    },
    {
        "condition_id": "flan_key",
        "architecture": "Flan-T5",
        "condition": "LoRA key only",
        "root": "direct/flan_k/analysis",
        "rollup": "direct/flan_k/direct_run_set_summary.json",
        "detail": False,
    },
    {
        "condition_id": "flan_qkv",
        "architecture": "Flan-T5",
        "condition": "LoRA q/k/v",
        "root": "direct/flan_qkv/analysis",
        "rollup": "direct/flan_qkv/direct_run_set_summary.json",
        "detail": False,
    },
    {
        "condition_id": "flan_attention",
        "architecture": "Flan-T5",
        "condition": "LoRA full attention",
        "root": "direct/flan_qkvo/analysis",
        "rollup": "direct/flan_qkvo/direct_run_set_summary.json",
        "detail": False,
    },
    {
        "condition_id": "llama_full_ft",
        "architecture": "Llama-3.2-1B",
        "condition": "Full fine-tuning (Llama)",
        "artifact": "model",
        "representation": "full_weight",
        "detail": False,
    },
    {
        "condition_id": "llama_r8",
        "architecture": "Llama-3.2-1B",
        "condition": "LoRA q/k/v (r=8)",
        "root": "direct/llama_r8/analysis",
        "rollup": "direct/llama_r8/direct_run_set_summary.json",
        "detail": True,
    },
    {
        "condition_id": "llama_r64",
        "architecture": "Llama-3.2-1B",
        "condition": "LoRA q/k/v (r=64)",
        "root": "direct/llama_r64/analysis",
        "rollup": "direct/llama_r64/direct_run_set_summary.json",
        "detail": True,
    },
)


def _llama_full_simulation(
    source: dict[str, Any], root: Path
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    final_path = root / "direct_core/llama_full/llama_full_table2_final.json"
    payload = json.loads(final_path.read_text())
    rows = [row for row in payload["rows"] if int(row["n_truth_splits"]) > 0]
    records: list[dict[str, Any]] = []
    source_ids: list[str] = []
    manifest_root = (
        Path(__file__).resolve().parents[1]
        / "examples/training/confirm_paper_numbers/assigned_manifests"
    )
    for row in rows:
        tree_id = str(row["tree_id"])
        matrix_path = (
            root
            / "direct/llama_full_matrices"
            / tree_id
            / f"{tree_id}.cosine.all_nodes.matrix.json"
        )
        matrix_payload = json.loads(matrix_path.read_text())
        labels = [str(value) for value in matrix_payload["leaf_sensitivity_ids"]]
        matrix = np.asarray(matrix_payload["leaf_sensitivity_matrix"], dtype=np.float64)
        truth_newick = truth_newick_from_manifest(
            manifest_root / f"{tree_id}.manifest.jsonl"
        )
        record = base._score_matrix(matrix, labels, truth_newick)
        if abs(record["clade_recovery"] - float(row["clade_recovery"])) > 1e-12:
            raise ValueError(f"Llama full Table 2 recovery mismatch for {tree_id}")
        record["tree_id"] = base._canonical_tree_id(tree_id)
        records.append(record)
        source_ids.append(tree_id)
    summary = base._summarize(
        records,
        source=source,
        level="simulation",
        label=source["condition"],
        family="whole simulation",
    )
    observations = base._observation_rows(
        records,
        source=source,
        level="simulation",
        label=source["condition"],
        family="whole simulation",
    )
    return summary, sorted(source_ids), observations


_archived_simulation_row = base._simulation_row


def _simulation_row(source: dict[str, Any], root: Path):
    if source["condition_id"] == "llama_full_ft":
        return _llama_full_simulation(source, root)
    return _archived_simulation_row(source, root)


def _validate_observations(observations: list[dict[str, Any]]) -> None:
    trees_by_estimator: dict[str, set[str]] = {}
    counts_by_estimator: dict[str, int] = {}
    for row in observations:
        estimator_id = str(row["estimator_id"])
        trees_by_estimator.setdefault(estimator_id, set()).add(str(row["tree_id"]))
        counts_by_estimator[estimator_id] = counts_by_estimator.get(estimator_id, 0) + 1
    if len(trees_by_estimator) < 100:
        raise ValueError(f"unexpectedly small Figure 2 panel: {len(trees_by_estimator)} estimators")
    reference = next(iter(trees_by_estimator.values()))
    if len(reference) != 46:
        raise ValueError(f"expected 46 shared tree IDs, got {len(reference)}")
    for estimator_id, tree_ids in trees_by_estimator.items():
        if counts_by_estimator[estimator_id] != 46 or tree_ids != reference:
            raise ValueError(f"paired panel mismatch for {estimator_id}")


base.SOURCES = FINAL_SOURCES
base._simulation_row = _simulation_row
base._validate_observations = _validate_observations


if __name__ == "__main__":
    raise SystemExit(base.main())
