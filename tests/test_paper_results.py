import csv
import hashlib
import json
from pathlib import Path

import pytest

from weighttraits.paper.results import (
    recovery_table_rows,
    validate_table_registry,
    write_recovery_table_csv,
    write_recovery_table_json,
)


def _write_summary(path: Path, *, artifact: str = "model") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "artifact": artifact,
                "representation": "full_weight",
                "ledger": "outputs/example/training_ledger.jsonl",
                "truth_manifest": "examples/training/truth.jsonl",
                "n_models": 4,
                "n_layers": 12,
                "results": [
                    {
                        "metric": "l2",
                        "rf": 0,
                        "normalized_rf": 0.0,
                        "exact_tree_recovery": True,
                        "clade_recovery": 1.0,
                        "split_precision": 1.0,
                        "distance_mean": 0.125,
                        "distance_min": 0.01,
                        "distance_max": 0.2,
                    },
                    {
                        "metric": "cosine",
                        "rf": 2,
                        "normalized_rf": 0.5,
                        "exact_tree_recovery": False,
                        "clade_recovery": 0.5,
                        "split_precision": 0.5,
                        "distance_mean": 0.25,
                        "distance_min": 0.02,
                        "distance_max": 0.4,
                    },
                ],
                "aggregate_recovery": {
                    "exact_tree_recovery_rate": 0.5,
                    "n_truth_leaves_mean": 4.0,
                    "n_truth_splits_mean": 1.0,
                },
            }
        )
        + "\n"
    )


def test_recovery_table_rows_load_registered_summary(tmp_path):
    summary = tmp_path / "outputs/example/summary.json"
    registry = tmp_path / "paper/recovery_registry.yaml"
    _write_summary(summary)
    registry.parent.mkdir()
    registry.write_text(
        """
result_sets:
  - id: example_whitebox
    title: Example whitebox result
    status: smoke_verified
    environment: local
    truth_manifest: examples/training/truth.jsonl
    summaries:
      - id: full_model
        artifact: model
        summary: outputs/example/summary.json
"""
    )

    rows = recovery_table_rows(registry, base_dir=tmp_path)

    assert len(rows) == 2
    assert rows[0]["result_set_id"] == "example_whitebox"
    assert rows[0]["summary_id"] == "full_model"
    assert rows[0]["metric"] == "l2"
    assert rows[0]["exact_tree_recovery"] is True
    assert rows[0]["exact_tree_recovery_rate"] == 0.5
    assert rows[0]["environment"] == "local"
    assert rows[1]["metric"] == "cosine"


def test_recovery_table_writers_emit_json_and_csv(tmp_path):
    rows = [
        {
            "result_set_id": "example",
            "result_set_title": "Example",
            "summary_id": "full_model",
            "artifact": "model",
            "representation": "full_weight",
            "metric": "l2",
            "n_models": 4,
            "n_layers": 12,
            "n_truth_leaves": 4.0,
            "n_truth_splits": 1.0,
            "rf": 0,
            "normalized_rf": 0.0,
            "exact_tree_recovery": True,
            "exact_tree_recovery_rate": 1.0,
            "clade_recovery": 1.0,
            "split_precision": 1.0,
            "distance_mean": 0.125,
            "distance_min": 0.01,
            "distance_max": 0.2,
            "environment": "local",
            "status": "smoke_verified",
            "truth_manifest": "truth.jsonl",
            "ledger": "ledger.jsonl",
            "summary": "summary.json",
        }
    ]
    json_out = tmp_path / "table.json"
    csv_out = tmp_path / "table.csv"

    write_recovery_table_json(rows, json_out, registry="paper/recovery_registry.yaml")
    write_recovery_table_csv(rows, csv_out)

    payload = json.loads(json_out.read_text())
    assert payload["n_rows"] == 1
    assert payload["rows"][0]["metric"] == "l2"
    with csv_out.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["exact_tree_recovery"] == "true"
    assert csv_rows[0]["distance_mean"] == "0.125"


def test_recovery_table_rows_reject_artifact_mismatch(tmp_path):
    summary = tmp_path / "outputs/example/summary.json"
    registry = tmp_path / "recovery_registry.yaml"
    _write_summary(summary, artifact="merged")
    registry.write_text(
        """
result_sets:
  - id: example_whitebox
    summaries:
      - id: full_model
        artifact: model
        summary: outputs/example/summary.json
"""
    )

    with pytest.raises(ValueError, match="summary artifact mismatch"):
        recovery_table_rows(registry, base_dir=tmp_path)


def test_validate_table_registry_checks_inputs_outputs_and_rows(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    csv_out = tmp_path / "reports/paper/recovery.csv"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    rows = [
        {
            "result_set_id": "example",
            "result_set_title": "Example",
            "summary_id": "full_model",
            "artifact": "model",
            "representation": "full_weight",
            "metric": "l2",
            "n_models": 4,
            "n_layers": 12,
            "n_truth_leaves": 4.0,
            "n_truth_splits": 1.0,
            "rf": 0,
            "normalized_rf": 0.0,
            "exact_tree_recovery": True,
            "exact_tree_recovery_rate": 1.0,
            "clade_recovery": 1.0,
            "split_precision": 1.0,
            "distance_mean": 0.125,
            "distance_min": 0.01,
            "distance_max": 0.2,
            "environment": "local",
            "status": "smoke_verified",
            "truth_manifest": "truth.jsonl",
            "ledger": "ledger.jsonl",
            "summary": "summary.json",
        }
    ]
    write_recovery_table_json(rows, json_out)
    write_recovery_table_csv(rows, csv_out)
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    status: smoke_verified
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - reports/paper/recovery.json
      - reports/paper/recovery.csv
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert report["valid"]
    assert report["n_tables"] == 1
    assert report["tables"][0]["id"] == "whitebox_smoke_recovery"
    assert report["tables"][0]["expected_rows"] == 1
    assert [item["observed_rows"] for item in report["tables"][0]["outputs"]] == [1, 1]
    assert [item["expected_sha256"] for item in report["tables"][0]["outputs"]] == [None, None]
    assert all(item["observed_sha256"] for item in report["tables"][0]["outputs"])


def test_validate_table_registry_reports_missing_required_output(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - reports/paper/missing.json
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "missing_output"


def test_validate_table_registry_reports_row_count_mismatch(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    json_out.parent.mkdir(parents=True)
    json_out.write_text(json.dumps({"n_rows": 2, "rows": [{}, {}]}) + "\n")
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - reports/paper/recovery.json
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "row_count_mismatch"


def test_validate_table_registry_accepts_pinned_output_sha256(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    json_out.parent.mkdir(parents=True)
    json_out.write_text(json.dumps({"n_rows": 1, "rows": [{}]}) + "\n")
    digest = hashlib.sha256(json_out.read_bytes()).hexdigest()
    registry.write_text(
        f"""
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - path: reports/paper/recovery.json
        sha256: {digest}
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert report["valid"]
    assert report["tables"][0]["outputs"][0]["expected_sha256"] == digest
    assert report["tables"][0]["outputs"][0]["observed_sha256"] == digest


def test_validate_table_registry_reports_output_sha256_mismatch(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    json_out.parent.mkdir(parents=True)
    json_out.write_text(json.dumps({"n_rows": 1, "rows": [{}]}) + "\n")
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - path: reports/paper/recovery.json
        sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "sha256_mismatch"
