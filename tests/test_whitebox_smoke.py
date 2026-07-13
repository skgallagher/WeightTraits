import json

import numpy as np
import pytest

pytest.importorskip("Bio")
safetensors_np = pytest.importorskip("safetensors.numpy")

from weighttraits.cli import main  # noqa: E402


def test_tiny_whitebox_cli_smoke_builds_reconstructs_scores_and_aggregates(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    cube = tmp_path / "distance_cube"
    tree = tmp_path / "tree.newick"
    tree_audit = tmp_path / "tree.audit.json"
    score = tmp_path / "score.json"
    aggregate = tmp_path / "aggregate.json"
    truth = tmp_path / "truth_manifest.jsonl"

    checkpoints.mkdir()
    _write_checkpoint(checkpoints / "a.safetensors", 0.0)
    _write_checkpoint(checkpoints / "b.safetensors", 1.0)
    _write_checkpoint(checkpoints / "c.safetensors", 10.0)
    _write_checkpoint(checkpoints / "d.safetensors", 11.0)
    _write_quartet_truth_manifest(truth)

    assert main(
        [
            "build-distance-cube",
            "--checkpoint",
            f"a:{checkpoints / 'a.safetensors'}",
            "--checkpoint",
            f"b:{checkpoints / 'b.safetensors'}",
            "--checkpoint",
            f"c:{checkpoints / 'c.safetensors'}",
            "--checkpoint",
            f"d:{checkpoints / 'd.safetensors'}",
            "--metric",
            "l2",
            "--out",
            str(cube),
        ]
    ) == 0

    assert main(
        [
            "reconstruct-tree",
            "--cube",
            str(cube),
            "--metric",
            "l2",
            "--out",
            str(tree),
            "--audit-out",
            str(tree_audit),
        ]
    ) == 0

    assert main(
        [
            "score-tree",
            "--truth-manifest",
            str(truth),
            "--estimate-newick",
            str(tree),
            "--out",
            str(score),
        ]
    ) == 0

    assert main(["aggregate-recovery", "--scores", str(score), "--out", str(aggregate)]) == 0

    score_report = json.loads(score.read_text())
    aggregate_report = json.loads(aggregate.read_text())
    audit_report = json.loads(tree_audit.read_text())

    assert score_report["rf"] == 0
    assert score_report["false_negative"] == 0
    assert score_report["false_positive"] == 0
    assert score_report["clade_recovery"] == 1.0
    assert score_report["exact_tree_recovery"] is True
    assert aggregate_report["n_records"] == 1
    assert aggregate_report["exact_tree_recovery_rate"] == 1.0
    assert audit_report["metric"] == "l2"
    assert audit_report["aggregate"] == "mean"
    assert audit_report["model_ids"] == ["a", "b", "c", "d"]


def _write_checkpoint(path, value: float) -> None:
    safetensors_np.save_file(
        {"layer.weight": np.array([value], dtype=np.float32)},
        str(path),
    )


def _write_quartet_truth_manifest(path) -> None:
    rows = [
        {"node_id": "ab", "path": ["root", "ab"], "grow": "train"},
        {"node_id": "a", "path": ["root", "ab", "a"], "grow": "train"},
        {"node_id": "b", "path": ["root", "ab", "b"], "grow": "train"},
        {"node_id": "cd", "path": ["root", "cd"], "grow": "train"},
        {"node_id": "c", "path": ["root", "cd", "c"], "grow": "train"},
        {"node_id": "d", "path": ["root", "cd", "d"], "grow": "train"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
