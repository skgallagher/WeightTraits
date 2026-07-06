import json

import numpy as np
import pytest

pytest.importorskip("Bio")
safetensors_np = pytest.importorskip("safetensors.numpy")

from weighttraits.cli import main
from weighttraits.training.ledger import TrainingLedgerEvent, append_ledger_event


def test_analyze_training_ledger_cli_writes_summary_scores_and_aggregate(tmp_path):
    ledger = tmp_path / "training_ledger.jsonl"
    truth = tmp_path / "truth_manifest.jsonl"
    checkpoints = tmp_path / "checkpoints"
    out = tmp_path / "whitebox"

    for node_id, value in {"n2": 0.0, "n3": 1.0, "n4": 10.0, "n5": 11.0}.items():
        model_dir = checkpoints / node_id / "model"
        model_dir.mkdir(parents=True)
        safetensors_np.save_file(
            {"layer.weight": np.array([value], dtype=np.float32)},
            str(model_dir / "model.safetensors"),
        )
        append_ledger_event(
            ledger,
            TrainingLedgerEvent(
                node_id=node_id,
                status="completed",
                extra={"artifacts": {"model": str(model_dir)}},
            ),
        )
    _write_truth_manifest(truth)

    assert main(
        [
            "analyze-training-ledger",
            "--ledger",
            str(ledger),
            "--truth-manifest",
            str(truth),
            "--artifact",
            "model",
            "--metric",
            "l2",
            "--out",
            str(out),
        ]
    ) == 0

    summary = json.loads((out / "summary.json").read_text())
    score = json.loads((out / "score_l2.json").read_text())
    aggregate = json.loads((out / "aggregate_recovery.json").read_text())

    assert (out / "model_distance_inputs.yaml").exists()
    assert (out / "distance_cube" / "distance_cube.npz").exists()
    assert (out / "tree_l2.newick").exists()
    assert (out / "tree_l2.audit.json").exists()
    assert summary["artifact"] == "model"
    assert summary["representation"] == "full_weight"
    assert summary["n_models"] == 4
    assert summary["metrics"] == ["l2"]
    assert summary["results"][0]["exact_tree_recovery"] is True
    assert summary["results"][0]["rf"] == 0
    assert summary["aggregate_recovery"]["exact_tree_recovery_rate"] == 1.0
    assert score["clade_recovery"] == 1.0
    assert aggregate["n_records"] == 1


def _write_truth_manifest(path) -> None:
    rows = [
        {"node_id": "n0", "parent_id": "root", "path": ["root", "n0"], "grow": "train"},
        {"node_id": "n1", "parent_id": "root", "path": ["root", "n1"], "grow": "train"},
        {"node_id": "n2", "parent_id": "n0", "path": ["root", "n0", "n2"], "grow": "train"},
        {"node_id": "n3", "parent_id": "n0", "path": ["root", "n0", "n3"], "grow": "train"},
        {"node_id": "n4", "parent_id": "n1", "path": ["root", "n1", "n4"], "grow": "train"},
        {"node_id": "n5", "parent_id": "n1", "path": ["root", "n1", "n5"], "grow": "train"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
