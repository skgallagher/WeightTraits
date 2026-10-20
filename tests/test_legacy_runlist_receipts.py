import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from weighttraits.training.executor import _weighttraits_source_receipt


ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "examples" / "training" / "confirm_paper_numbers"
HOLDOUT = ROOT / "examples" / "training" / "translation_holdout_20260803"
EXPECTED_RUNTIME = {
    "python": "3.11.15",
    "torch": "2.5.1+cu121",
    "transformers": "5.8.0",
    "datasets": "4.8.5",
    "accelerate": "1.13.0",
    "tokenizers": "0.22.2",
    "peft": "0.19.1",
}
EXPECTED_SOURCE_CODE = {
    "algorithm": "sha256-relative-path-and-content-v1",
    "file_count": 60,
    "sha256": "1445e45de825615dba929a63f8812d66e6105170002ff8c1ed35ee4a338e796b",
}
EXPECTED_SOURCE_RECEIPT = {
    "path": "examples/training/confirm_paper_numbers/legacy_causal_seed42_cache_source_receipt.json",
    "sha256": "f3f185801afd6c7f01fb015d69544574d3d6ef48da375e1eadcd0eff32d00818",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_paths() -> list[Path]:
    ordinary = sorted(ASSETS.glob("*_legacy_causal_2000_training_run_list_summary.json"))
    ordinary += sorted(ASSETS.glob("*_legacy_seq2seq_2000_training_run_list_summary.json"))
    return ordinary + [HOLDOUT / "matched_legacy_causal_2000_training_run_list_summary.json"]


FROZEN_TRAINING_TAG = "training-code-frozen-20260810"


def _tagged_package_receipt(tmp_path: Path) -> dict | None:
    """Receipt of the package at the frozen tag, or None when git or the tag is unavailable."""

    try:
        archive = subprocess.run(
            ["git", "archive", FROZEN_TRAINING_TAG, "src/weighttraits"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp_path, filter="data")
    return _weighttraits_source_receipt(tmp_path / "src" / "weighttraits")


def test_frozen_source_receipt_matches_the_tagged_training_package(tmp_path):
    # The run lists pin the training code as it stood when they were built. The package has
    # grown since (analysis and paper modules), so the pin is checked against the tag that
    # exact retraining must check out, not against the current working tree.
    actual = _tagged_package_receipt(tmp_path)
    if actual is None:
        pytest.skip(f"git tag {FROZEN_TRAINING_TAG} is not available in this checkout")

    assert {key: actual[key] for key in EXPECTED_SOURCE_CODE} == EXPECTED_SOURCE_CODE


def test_all_eleven_frozen_cohorts_have_self_consistent_executable_receipts():
    summaries = _summary_paths()
    assert len(summaries) == 11

    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        assert summary["n_trees"] == 50
        assert summary["n_runs"] == 641
        assert summary["n_errors"] == 0
        assert summary["n_warnings"] == 0
        config_path = ROOT / summary["config"]
        assert _sha256(config_path) == summary["config_sha256"]
        validation_path = config_path.with_name(f"{config_path.stem}_data_format_validation.json")
        validation = json.loads(validation_path.read_text())
        assert validation["valid"] is True
        assert validation["n_jobs"] == 641
        assert validation["n_valid"] == 641
        assert validation["n_issues"] == 0

        total_rows = 0
        for tree in summary["trees"]:
            assert tree["valid"] is True
            assert tree["n_errors"] == 0
            assert tree["n_warnings"] == 0
            run_list_path = ROOT / tree["run_list"]
            report_path = ROOT / tree["report"]
            manifest_path = ROOT / tree["manifest"]
            assert _sha256(run_list_path) == tree["run_list_sha256"]
            assert _sha256(report_path) == tree["report_sha256"]
            assert _sha256(manifest_path) == tree["manifest_sha256"]
            rows = [json.loads(line) for line in run_list_path.read_text().splitlines() if line]
            assert len(rows) == tree["n_runs"]
            total_rows += len(rows)
            for row in rows:
                trainer = row["job"]["trainer"]
                options = row["runner"]["options"]
                assert trainer["expected_runtime"] == EXPECTED_RUNTIME
                assert trainer["expected_source_code"] == EXPECTED_SOURCE_CODE
                assert trainer["expected_cache_source_receipt"] == EXPECTED_SOURCE_RECEIPT
                assert trainer["max_steps"] == 2000
                assert trainer["require_max_steps"] is True
                assert options["require_data_cache"] is True
                assert options["max_train_samples"] == 10000
                assert options["max_eval_samples"] == 1000
                assert options["expected_cache_recipe"] == {
                    "sample_strategy": "legacy_subsample",
                    "sample_seed": 42,
                    "train_limit": 10000,
                    "eval_limit": 1000,
                }
        assert total_rows == 641


def test_executor_smoke_inherits_the_same_frozen_runtime_and_source_receipts():
    run_list = ASSETS / "llama32_1b_legacy_causal_executor_smoke.runs.jsonl"
    rows = [json.loads(line) for line in run_list.read_text().splitlines() if line]

    assert len(rows) == 14
    for row in rows:
        trainer = row["job"]["trainer"]
        assert trainer["expected_runtime"] == EXPECTED_RUNTIME
        assert trainer["expected_source_code"] == EXPECTED_SOURCE_CODE
        assert trainer["expected_cache_source_receipt"] == EXPECTED_SOURCE_RECEIPT
        assert trainer["max_steps"] == 2
        assert trainer["require_max_steps"] is True
