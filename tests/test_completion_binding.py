from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from weighttraits.paper.analysis_contracts import validate_strict_training_completion


ROOT = Path(__file__).resolve().parents[1]


def _receipt(*, factor: int, cohort_id: str | None = None) -> dict:
    rows = []
    remaining = 641
    for index in range(1, 51):
        count = 13 if index <= 41 else 12
        remaining -= count
        rows.append(
            {
                "tree_id": f"confirm_paper_tree_{index:03d}",
                "valid": True,
                "ready_for_analysis": True,
                "n_runs": count,
                "n_ledger_nodes": count,
                "n_terminal_nodes": count,
                "n_ok_nodes": count,
                "n_failed_nodes": 0,
                "n_missing_nodes": 0,
                "n_errors": 0,
                "status_counts": {"completed": count},
                "n_expected_artifacts": factor * count,
                "n_existing_artifacts": factor * count,
            }
        )
    assert remaining == 0
    payload = {
        "valid": True,
        "n_trees": 50,
        "n_ready": 50,
        "n_failed": 0,
        "n_in_progress": 0,
        "n_not_started": 0,
        "n_total_runs": 641,
        "n_terminal_nodes": 641,
        "n_ok_nodes": 641,
        "n_failed_nodes": 0,
        "n_missing_nodes": 0,
        "n_errors": 0,
        "n_warnings": 0,
        "rows": rows,
    }
    if cohort_id is not None:
        payload["cohort_id"] = cohort_id
        payload["artifact_factor"] = factor
    return payload


def test_strict_completion_accepts_explicit_four_artifact_contract() -> None:
    validate_strict_training_completion(
        _receipt(factor=4, cohort_id="llama_lora"),
        cohort_id="llama_lora",
    )


def test_strict_completion_rejects_wrong_explicit_factor() -> None:
    with pytest.raises(ValueError, match="artifact counts"):
        validate_strict_training_completion(
            _receipt(factor=2),
            artifact_factor=4,
        )


def test_completion_binding_preserves_source_and_adds_exact_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_receipt(factor=4), sort_keys=True) + "\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "binding.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_completion_binding.py"),
            "--source",
            str(source),
            "--source-sha256",
            digest,
            "--cohort-id",
            "llama32_1b_lora_qkv_r8_legacy_causal_2000",
            "--artifact-factor",
            "4",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text())
    assert payload["source_completion_receipt"] == {
        "path": str(source),
        "sha256": digest,
    }
    validate_strict_training_completion(
        payload,
        cohort_id="llama32_1b_lora_qkv_r8_legacy_causal_2000",
    )
