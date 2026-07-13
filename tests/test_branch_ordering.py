import json
import math

import numpy as np
import pytest

from weighttraits.analysis.branch_ordering import (
    aggregate_branch_ordering,
    branch_ordering_stats,
)


def test_branch_ordering_matches_paper_orientation(tmp_path):
    manifest = tmp_path / "truth.manifest.jsonl"
    _write_manifest(manifest)
    labels = ["a", "b", "c", "d"]
    distances = np.array(
        [
            [0.0, 0.1, 0.5, 0.6],
            [0.1, 0.0, 0.4, 0.7],
            [0.5, 0.4, 0.0, 0.2],
            [0.6, 0.7, 0.2, 0.0],
        ]
    )

    stats = branch_ordering_stats(manifest, labels=labels, distances=distances)

    assert stats["branch_ordering_valid"] is True
    assert stats["branch_ordering_status"] == "ok"
    assert stats["n_branch_pairs"] == 6
    assert stats["n_same_branch_pairs"] == 2
    assert stats["n_cross_branch_pairs"] == 4
    assert stats["branch_rank_biserial"] == 1.0
    assert stats["branch_within_run_r"] < 0.0


def test_branch_ordering_reports_missing_cross_branch_class(tmp_path):
    manifest = tmp_path / "truth.manifest.jsonl"
    rows = [
        {"node_id": "root_child", "path": ["root", "root_child"]},
        *[
            {
                "node_id": leaf,
                "path": ["root", "root_child", leaf],
                "is_leaf": True,
            }
            for leaf in ("a", "b", "c", "d")
        ],
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    stats = branch_ordering_stats(
        manifest,
        labels=["a", "b", "c", "d"],
        distances=np.ones((4, 4)) - np.eye(4),
    )

    assert stats["branch_ordering_valid"] is False
    assert stats["branch_ordering_status"] == "missing_branch_class"
    assert stats["n_same_branch_pairs"] == 6
    assert stats["n_cross_branch_pairs"] == 0


def test_branch_ordering_marks_nonleaf_selection_without_breaking_analysis(tmp_path):
    manifest = tmp_path / "truth.manifest.jsonl"
    _write_manifest(manifest)

    stats = branch_ordering_stats(
        manifest,
        labels=["left", "a", "b", "c"],
        distances=np.ones((4, 4)) - np.eye(4),
    )

    assert stats["branch_ordering_valid"] is False
    assert stats["branch_ordering_status"] == "nonleaf_or_unknown_labels"
    assert stats["branch_ordering_foreign_labels"] == ["left"]


def test_aggregate_branch_ordering_uses_run_level_effects_and_fisher_z_mean():
    rows = [
        {
            "branch_ordering_valid": True,
            "branch_ordering_status": "ok",
            "n_branch_pairs": 6,
            "n_same_branch_pairs": 2,
            "n_cross_branch_pairs": 4,
            "branch_rank_biserial": 1.0,
            "branch_within_run_r": -0.8,
        },
        {
            "branch_ordering_valid": True,
            "branch_ordering_status": "ok",
            "n_branch_pairs": 10,
            "n_same_branch_pairs": 4,
            "n_cross_branch_pairs": 6,
            "branch_rank_biserial": 0.5,
            "branch_within_run_r": -0.4,
        },
        {
            "branch_ordering_valid": False,
            "branch_ordering_status": "missing_branch_class",
            "n_branch_pairs": 6,
        },
    ]

    aggregate = aggregate_branch_ordering(rows)

    expected_r = math.tanh((math.atanh(-0.8) + math.atanh(-0.4)) / 2.0)
    assert aggregate["n_ordering_trees"] == 2
    assert aggregate["n_branch_pairs"] == 22
    assert aggregate["branch_ordering_status_counts"] == {
        "missing_branch_class": 1,
        "ok": 2,
    }
    assert aggregate["branch_rank_biserial_mean"] == 0.75
    assert aggregate["branch_rank_biserial_se"] == 0.25
    assert aggregate["branch_within_run_r_fisher_z_mean"] == pytest.approx(expected_r)
    assert aggregate["branch_within_run_r_se"] == pytest.approx(0.2)


def _write_manifest(path):
    rows = [
        {"node_id": "left", "path": ["root", "left"]},
        {"node_id": "right", "path": ["root", "right"]},
        {"node_id": "a", "path": ["root", "left", "a"], "is_leaf": True},
        {"node_id": "b", "path": ["root", "left", "b"], "is_leaf": True},
        {"node_id": "c", "path": ["root", "right", "c"], "is_leaf": True},
        {"node_id": "d", "path": ["root", "right", "d"], "is_leaf": True},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
