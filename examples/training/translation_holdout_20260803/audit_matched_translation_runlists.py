#!/usr/bin/env python3
"""Audit that matched holdout run lists differ only at translation assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_REPLACEMENTS = 162
PROTOCOL_FIELDS = (
    "base_model",
    "base_model_revision",
    "method",
    "protocol_id",
    "trainer",
    "stopping",
    "lora",
)
PRESERVED_PROMPT_FIELDS = (
    "prompt_template",
    "prompt_source",
    "prompt_fields",
)
REQUIRED_RUNNER_OPTIONS = (
    "registry_path",
    "formats_path",
    "data_cache_root",
    "require_data_cache",
    "max_train_samples",
    "max_eval_samples",
    "expected_cache_recipe",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _runner_contract(row: dict[str, Any]) -> dict[str, Any]:
    runner = dict(row.get("runner", {}))
    return {
        "entrypoint": runner.get("entrypoint"),
        "status": runner.get("status"),
        "options": dict(runner.get("options", {})),
    }


def _runner_contract_issues(
    contract: dict[str, Any],
    *,
    cohort: str,
    index: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if contract["entrypoint"] != "weighttraits.cli run-training-row":
        issues.append(
            {
                "issue": "runner_entrypoint_mismatch",
                "index": index,
                "cohort": cohort,
                "actual": contract["entrypoint"],
            }
        )
    if contract["status"] != "planned":
        issues.append(
            {
                "issue": "runner_status_mismatch",
                "index": index,
                "cohort": cohort,
                "actual": contract["status"],
            }
        )
    options = contract["options"]
    missing = [key for key in REQUIRED_RUNNER_OPTIONS if key not in options]
    if missing:
        issues.append(
            {
                "issue": "runner_options_missing",
                "index": index,
                "cohort": cohort,
                "missing": missing,
            }
        )
    if options.get("require_data_cache") is not True:
        issues.append(
            {
                "issue": "data_cache_not_required",
                "index": index,
                "cohort": cohort,
                "actual": options.get("require_data_cache"),
            }
        )
    for key in ("registry_path", "formats_path", "data_cache_root"):
        if not isinstance(options.get(key), str) or not options[key].strip():
            issues.append(
                {
                    "issue": "runner_path_option_invalid",
                    "index": index,
                    "cohort": cohort,
                    "field": key,
                    "actual": options.get(key),
                }
            )
    expected_caps = {"max_train_samples": 10_000, "max_eval_samples": 1_000}
    for key, expected in expected_caps.items():
        if options.get(key) != expected:
            issues.append(
                {
                    "issue": "runner_sample_cap_mismatch",
                    "index": index,
                    "cohort": cohort,
                    "field": key,
                    "actual": options.get(key),
                    "expected": expected,
                }
            )
    expected_recipe = {
        "sample_strategy": "legacy_subsample",
        "sample_seed": 42,
        "train_limit": 10_000,
        "eval_limit": 1_000,
    }
    if options.get("expected_cache_recipe") != expected_recipe:
        issues.append(
            {
                "issue": "runner_cache_recipe_mismatch",
                "index": index,
                "cohort": cohort,
                "actual": options.get("expected_cache_recipe"),
                "expected": expected_recipe,
            }
        )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-runlists", type=Path, required=True)
    parser.add_argument("--holdout-runlists", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expected-replacements",
        type=int,
        default=EXPECTED_REPLACEMENTS,
        help="Exact number of ordinary translation rows that the matched cohort must replace",
    )
    args = parser.parse_args()
    if args.expected_replacements < 0:
        parser.error("--expected-replacements must be non-negative")

    ordinary = {path.name: path for path in args.ordinary_runlists.glob("*.jsonl")}
    holdout = {path.name: path for path in args.holdout_runlists.glob("*.jsonl")}
    issues: list[dict[str, Any]] = []
    tree_audits: list[dict[str, Any]] = []
    preserved = 0
    replaced = 0
    ordinary_translation_rows = 0
    holdout_translation_rows = 0

    if set(ordinary) != set(holdout):
        issues.append(
            {
                "issue": "tree_file_set_mismatch",
                "ordinary_only": sorted(set(ordinary) - set(holdout)),
                "holdout_only": sorted(set(holdout) - set(ordinary)),
            }
        )

    for name in sorted(set(ordinary) & set(holdout)):
        ordinary_rows = read_jsonl(ordinary[name])
        holdout_rows = read_jsonl(holdout[name])
        tree_preserved = 0
        tree_replaced = 0
        tree_ordinary_translation = 0
        tree_holdout_translation = 0
        tree_issues: list[dict[str, Any]] = []
        if len(ordinary_rows) != len(holdout_rows):
            tree_issues.append(
                {
                    "issue": "row_count_mismatch",
                    "ordinary": len(ordinary_rows),
                    "holdout": len(holdout_rows),
                }
            )

        for index, (old, new) in enumerate(zip(ordinary_rows, holdout_rows)):
            old_job = dict(old.get("job", {}))
            new_job = dict(new.get("job", {}))
            topology_old = (
                old["node_id"],
                old["parent_id"],
                old["depth"],
                old_job["path"],
            )
            topology_new = (
                new["node_id"],
                new["parent_id"],
                new["depth"],
                new_job["path"],
            )
            if topology_old != topology_new:
                tree_issues.append({"issue": "topology_changed", "index": index})

            for field in PROTOCOL_FIELDS:
                if old_job.get(field) != new_job.get(field):
                    tree_issues.append(
                        {"issue": "training_protocol_changed", "index": index, "field": field}
                    )

            ordinary_runner = _runner_contract(old)
            holdout_runner = _runner_contract(new)
            if ordinary_runner != holdout_runner:
                tree_issues.append(
                    {
                        "issue": "runner_contract_changed",
                        "index": index,
                        "ordinary": ordinary_runner,
                        "holdout": holdout_runner,
                    }
                )
            tree_issues.extend(
                _runner_contract_issues(ordinary_runner, cohort="ordinary", index=index)
            )
            tree_issues.extend(
                _runner_contract_issues(holdout_runner, cohort="holdout", index=index)
            )

            for cohort, row, job in (
                ("ordinary", old, old_job),
                ("holdout", new, new_job),
            ):
                if row.get("dataset_id") != job.get("dataset_id"):
                    tree_issues.append(
                        {
                            "issue": "row_job_dataset_mismatch",
                            "index": index,
                            "cohort": cohort,
                        }
                    )
                if row.get("task_family") != job.get("task_family"):
                    tree_issues.append(
                        {
                            "issue": "row_job_task_family_mismatch",
                            "index": index,
                            "cohort": cohort,
                        }
                    )

            old_assignment = (old["task_family"], old["dataset_id"])
            new_assignment = (new["task_family"], new["dataset_id"])
            if old["task_family"] == "translation":
                tree_ordinary_translation += 1
            if new["task_family"] == "translation":
                tree_holdout_translation += 1
            if old_assignment == new_assignment:
                tree_preserved += 1
                for field in PRESERVED_PROMPT_FIELDS:
                    if old_job.get(field) != new_job.get(field):
                        tree_issues.append(
                            {
                                "issue": "preserved_prompt_changed",
                                "index": index,
                                "field": field,
                            }
                        )
            else:
                tree_replaced += 1
                if old["task_family"] != "translation":
                    tree_issues.append(
                        {
                            "issue": "nontranslation_assignment_changed",
                            "index": index,
                            "old": old_assignment,
                            "new": new_assignment,
                        }
                    )
            if new["task_family"] == "translation":
                tree_issues.append({"issue": "translation_remains", "index": index})

        ordinary_config_shas = {
            row.get("job", {}).get("training_config_sha256") for row in ordinary_rows
        }
        holdout_config_shas = {
            row.get("job", {}).get("training_config_sha256") for row in holdout_rows
        }
        for cohort, config_shas in (
            ("ordinary", ordinary_config_shas),
            ("holdout", holdout_config_shas),
        ):
            if len(config_shas) != 1 or not all(_is_sha256(value) for value in config_shas):
                tree_issues.append(
                    {
                        "issue": "training_config_fingerprint_inconsistent",
                        "cohort": cohort,
                        "values": sorted(str(value) for value in config_shas),
                    }
                )

        ordinary_datasets = [row["dataset_id"] for row in ordinary_rows]
        if len(ordinary_datasets) != len(set(ordinary_datasets)):
            tree_issues.append({"issue": "duplicate_ordinary_dataset"})
        new_datasets = [row["dataset_id"] for row in holdout_rows]
        if len(new_datasets) != len(set(new_datasets)):
            tree_issues.append({"issue": "duplicate_holdout_dataset"})

        preserved += tree_preserved
        replaced += tree_replaced
        ordinary_translation_rows += tree_ordinary_translation
        holdout_translation_rows += tree_holdout_translation
        issues.extend({"tree": name, **issue} for issue in tree_issues)
        tree_audits.append(
            {
                "tree": name,
                "ordinary_sha256": sha256(ordinary[name]),
                "holdout_sha256": sha256(holdout[name]),
                "n_rows": len(ordinary_rows),
                "n_preserved": tree_preserved,
                "n_replaced": tree_replaced,
                "n_ordinary_translation": tree_ordinary_translation,
                "n_holdout_translation": tree_holdout_translation,
                "ordinary_training_config_sha256": (
                    next(iter(ordinary_config_shas)) if len(ordinary_config_shas) == 1 else None
                ),
                "holdout_training_config_sha256": (
                    next(iter(holdout_config_shas)) if len(holdout_config_shas) == 1 else None
                ),
                "n_issues": len(tree_issues),
            }
        )

    if replaced != args.expected_replacements:
        issues.append(
            {
                "issue": "replacement_count_mismatch",
                "actual": replaced,
                "expected": args.expected_replacements,
            }
        )
    if ordinary_translation_rows != args.expected_replacements:
        issues.append(
            {
                "issue": "ordinary_translation_count_mismatch",
                "actual": ordinary_translation_rows,
                "expected": args.expected_replacements,
            }
        )
    if holdout_translation_rows != 0:
        issues.append(
            {
                "issue": "holdout_translation_count_nonzero",
                "actual": holdout_translation_rows,
                "expected": 0,
            }
        )

    report = {
        "valid": not issues,
        "ordinary_runlists": str(args.ordinary_runlists),
        "holdout_runlists": str(args.holdout_runlists),
        "n_trees": len(tree_audits),
        "n_rows": preserved + replaced,
        "n_preserved": preserved,
        "n_replaced": replaced,
        "expected_replacements": args.expected_replacements,
        "n_ordinary_translation": ordinary_translation_rows,
        "n_holdout_translation": holdout_translation_rows,
        "n_issues": len(issues),
        "issues": issues,
        "trees": tree_audits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "valid",
                    "n_trees",
                    "n_rows",
                    "n_preserved",
                    "n_replaced",
                    "n_issues",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
