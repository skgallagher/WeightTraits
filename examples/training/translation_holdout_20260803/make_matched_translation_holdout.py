#!/usr/bin/env python3
"""Derive matched manifests by replacing only translation-assigned nodes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-summary", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    original_summary = json.loads(args.original_summary.read_text())
    family_config = yaml.safe_load(args.task_config.read_text())["task_families"]
    eligible_families = ["summarization", "classification", "qa"]
    candidate_pairs = sorted(
        (family, item["id"])
        for family in eligible_families
        for item in family_config[family]["datasets"]
        if item.get("role") != "evaluation_only"
    )

    new_assignments = []
    audit_trees = []
    total_preserved = 0
    total_replaced = 0
    old_translation_datasets = collections.Counter()
    replacement_families = collections.Counter()

    for entry in original_summary["assignments"]:
        source_path = Path(entry["assigned_manifest"])
        original_rows = read_jsonl(source_path)
        used_nontranslation = {
            (row["task_family"], row["dataset_id"])
            for row in original_rows
            if row.get("task_family") != "translation"
        }
        available = [pair for pair in candidate_pairs if pair not in used_nontranslation]
        translation_indices = [
            index for index, row in enumerate(original_rows)
            if row.get("task_family") == "translation"
        ]
        if len(available) < len(translation_indices):
            raise RuntimeError(
                f"{entry['tree_id']}: need {len(translation_indices)} replacements, "
                f"have only {len(available)} unused non-translation datasets"
            )

        rng = np.random.default_rng(int(entry["assignment_seed"]))
        replacement_order = rng.permutation(len(available))[: len(translation_indices)]
        replacements = []
        new_rows = [dict(row) for row in original_rows]
        for row_index, replacement_index in zip(translation_indices, replacement_order):
            old_row = original_rows[row_index]
            family, dataset_id = available[int(replacement_index)]
            new_rows[row_index]["task_family"] = family
            new_rows[row_index]["dataset_id"] = dataset_id
            new_rows[row_index]["matched_holdout_original_task_family"] = "translation"
            new_rows[row_index]["matched_holdout_original_dataset_id"] = old_row["dataset_id"]
            replacements.append(
                {
                    "node_id": old_row["node_id"],
                    "original_dataset_id": old_row["dataset_id"],
                    "replacement_task_family": family,
                    "replacement_dataset_id": dataset_id,
                }
            )
            old_translation_datasets[old_row["dataset_id"]] += 1
            replacement_families[family] += 1

        for index, (old_row, new_row) in enumerate(zip(original_rows, new_rows)):
            if index in translation_indices:
                continue
            if old_row != new_row:
                raise RuntimeError(f"{entry['tree_id']} row {index}: non-translation row changed")

        datasets = [row["dataset_id"] for row in new_rows if row.get("grow", "train") == "train"]
        if len(datasets) != len(set(datasets)):
            raise RuntimeError(f"{entry['tree_id']}: replacement introduced a duplicate dataset")
        if any(row.get("task_family") == "translation" for row in new_rows):
            raise RuntimeError(f"{entry['tree_id']}: translation row remains")

        out_path = args.out_dir / source_path.name
        write_jsonl(out_path, new_rows)
        family_counts = collections.Counter(
            row["task_family"] for row in new_rows if row.get("grow", "train") == "train"
        )
        new_entry = dict(entry)
        new_entry["assigned_manifest"] = str(out_path)
        new_entry["n_datasets"] = len(datasets)
        new_entry["n_rows"] = len(new_rows)
        new_entry["n_unique_datasets"] = len(set(datasets))
        new_entry["task_families"] = dict(sorted(family_counts.items()))
        new_assignments.append(new_entry)

        preserved = len(original_rows) - len(translation_indices)
        total_preserved += preserved
        total_replaced += len(translation_indices)
        audit_trees.append(
            {
                "tree_id": entry["tree_id"],
                "assignment_seed": entry["assignment_seed"],
                "original_manifest": str(source_path),
                "original_sha256": sha256(source_path),
                "matched_manifest": str(out_path),
                "matched_sha256": sha256(out_path),
                "n_rows": len(new_rows),
                "n_preserved_rows": preserved,
                "n_replaced_rows": len(translation_indices),
                "replacements": replacements,
            }
        )

    matched_summary = dict(original_summary)
    matched_summary["assignments"] = new_assignments
    matched_summary["out_dir"] = str(args.out_dir)
    matched_summary["policy"] = "matched_translation_holdout_per_node_without_replacement"
    matched_summary["matched_holdout"] = {
        "intervention": "translation withheld",
        "original_assignment_summary": str(args.original_summary),
        "eligible_replacement_families": eligible_families,
        "replacement_rng": "numpy.default_rng(original assignment_seed)",
        "preserve_nontranslation_rows": True,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(matched_summary, indent=2, sort_keys=True) + "\n")

    audit = {
        "original_assignment_summary": str(args.original_summary),
        "task_config": str(args.task_config),
        "matched_summary": str(args.summary_out),
        "n_trees": len(audit_trees),
        "n_rows": total_preserved + total_replaced,
        "n_preserved_nontranslation_rows": total_preserved,
        "n_replaced_translation_rows": total_replaced,
        "old_translation_datasets": dict(sorted(old_translation_datasets.items())),
        "replacement_task_families": dict(sorted(replacement_families.items())),
        "trees": audit_trees,
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: audit[key] for key in [
        "n_trees", "n_rows", "n_preserved_nontranslation_rows",
        "n_replaced_translation_rows", "replacement_task_families"
    ]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
