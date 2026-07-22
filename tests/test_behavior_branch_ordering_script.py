import importlib.util
import json
from itertools import combinations
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "behavior_branch_ordering.py"
SPEC = importlib.util.spec_from_file_location("behavior_branch_ordering", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_behavior_branch_ordering_reports_pooled_and_tree_level_results(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    pair_paths = []
    for tree_number in (1, 2):
        manifest = manifest_dir / f"tree_{tree_number:03d}.manifest.jsonl"
        _write_manifest(manifest, unary=tree_number == 2)
        pair_path = tmp_path / f"tree{tree_number:03d}" / "pairs" / "probe.csv"
        pair_path.parent.mkdir(parents=True)
        _write_pairs(pair_path)
        pair_paths.append(pair_path)

    summary, per_tree, labeled = MODULE.run_analysis(
        pair_paths,
        manifest_for_tree=lambda number: manifest_dir / f"tree_{number:03d}.manifest.jsonl",
    )

    assert summary["n_trees"] == 2
    assert summary["n_ordering_trees"] == 1
    assert summary["branch_ordering_status_counts"] == {
        "missing_branch_class": 1,
        "ok": 1,
    }
    assert summary["n_pairs"] == 12
    assert summary["n_same_branch_pairs"] == 8
    assert summary["n_cross_branch_pairs"] == 4
    assert summary["pooled_rank_biserial"] > 0.0
    assert summary["tree_mean_rank_biserial"] == 1.0
    assert summary["within_tree_fisher_z_r"] < 0.0
    assert len(per_tree) == 2
    assert len(labeled) == 12


def test_behavior_branch_ordering_rejects_incomplete_pair_grid(tmp_path):
    manifest = tmp_path / "truth.manifest.jsonl"
    _write_manifest(manifest)
    pair_path = tmp_path / "tree001" / "pairs" / "probe.csv"
    pair_path.parent.mkdir(parents=True)
    _write_pairs(pair_path, omit_last=True)

    with pytest.raises(ValueError, match=r"expected choose\(4, 2\)=6"):
        MODULE.summarize_pair_table(
            pair_path,
            tree_number=1,
            branches=MODULE.read_manifest_branches(manifest),
        )


def _write_manifest(path, unary=False):
    rows = [
        {"node_id": "left", "path": ["root", "left"]},
        {"node_id": "right", "path": ["root", "left" if unary else "right"]},
        {"node_id": "a", "path": ["root", "left", "a"], "is_leaf": True},
        {"node_id": "b", "path": ["root", "left", "b"], "is_leaf": True},
        {
            "node_id": "c",
            "path": ["root", "left" if unary else "right", "c"],
            "is_leaf": True,
        },
        {
            "node_id": "d",
            "path": ["root", "left" if unary else "right", "d"],
            "is_leaf": True,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _write_pairs(path, omit_last=False):
    lines = ["model_a,model_b,behavior_distance"]
    pairs = list(combinations(("a", "b", "c", "d"), 2))
    if omit_last:
        pairs = pairs[:-1]
    for left, right in pairs:
        same_group = {left, right} <= {"a", "b"} or {left, right} <= {"c", "d"}
        distance = 0.1 if same_group else 0.8
        lines.append(f"{left},{right},{distance}")
    path.write_text("\n".join(lines) + "\n")
