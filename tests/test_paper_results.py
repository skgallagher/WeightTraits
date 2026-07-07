import csv
import json
from pathlib import Path

import pytest

from weighttraits.paper.results import (
    recovery_table_rows,
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
