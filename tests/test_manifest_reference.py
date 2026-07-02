import json
from pathlib import Path

from weighttraits.manifests.reference import (
    load_manifest,
    manifest_leaf_ids,
    nontrivial_reference_splits,
)


def _write_manifest(tmp_path: Path, rows) -> Path:
    path = tmp_path / "manifest.jsonl"
    with path.open("w") as handle:
        for row in rows:
            row = dict(row)
            row.setdefault("grow", "train")
            handle.write(json.dumps(row) + "\n")
    return path


def test_multifurcating_root_first_child_is_leaf(tmp_path):
    rows = [
        {"node_id": "n0", "path": ["root", "n0"]},
        {"node_id": "n1", "path": ["root", "n1"]},
        {"node_id": "n2", "path": ["root", "n1", "n2"]},
        {"node_id": "n3", "path": ["root", "n1", "n2", "n3"]},
        {"node_id": "n8", "path": ["root", "n1", "n2", "n3", "n8"]},
        {"node_id": "n9", "path": ["root", "n1", "n2", "n3", "n9"]},
        {"node_id": "n4", "path": ["root", "n1", "n2", "n4"]},
        {"node_id": "n5", "path": ["root", "n1", "n2", "n5"]},
        {"node_id": "n6", "path": ["root", "n1", "n2", "n6"]},
        {"node_id": "n7", "path": ["root", "n1", "n2", "n7"]},
    ]
    assert manifest_leaf_ids(_write_manifest(tmp_path, rows)) == [
        "n0",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
    ]


def test_multifurcating_root_two_subtrees(tmp_path):
    rows = [
        {"node_id": "n0", "path": ["root", "n0"]},
        {"node_id": "n2", "path": ["root", "n0", "n2"]},
        {"node_id": "n7", "path": ["root", "n0", "n2", "n7"]},
        {"node_id": "n8", "path": ["root", "n0", "n2", "n8"]},
        {"node_id": "n3", "path": ["root", "n0", "n3"]},
        {"node_id": "n9", "path": ["root", "n0", "n3", "n9"]},
        {"node_id": "n1", "path": ["root", "n1"]},
        {"node_id": "n4", "path": ["root", "n1", "n4"]},
        {"node_id": "n5", "path": ["root", "n1", "n5"]},
        {"node_id": "n6", "path": ["root", "n1", "n6"]},
    ]
    leaves = manifest_leaf_ids(_write_manifest(tmp_path, rows))
    assert leaves == ["n4", "n5", "n6", "n7", "n8", "n9"]


def test_identity_nodes_are_excluded(tmp_path):
    rows = [
        {"node_id": "n0", "path": ["root", "n0"]},
        {"node_id": "n1", "path": ["root", "n0", "n1"]},
        {"node_id": "n2", "path": ["root", "n0", "n2"]},
        {"node_id": "nX", "path": ["root", "n0", "nX"], "grow": "identity"},
    ]
    assert manifest_leaf_ids(_write_manifest(tmp_path, rows)) == ["n1", "n2"]


def test_nontrivial_reference_splits(tmp_path):
    rows = [
        {"node_id": "n0", "path": ["root", "n0"]},
        {"node_id": "n1", "path": ["root", "n0", "n1"]},
        {"node_id": "n2", "path": ["root", "n0", "n2"]},
        {"node_id": "n3", "path": ["root", "n0", "n2", "n3"]},
        {"node_id": "n4", "path": ["root", "n0", "n2", "n4"]},
    ]
    manifest = _write_manifest(tmp_path, rows)
    splits = nontrivial_reference_splits(load_manifest(manifest))
    assert frozenset({"n3", "n4"}) in splits

