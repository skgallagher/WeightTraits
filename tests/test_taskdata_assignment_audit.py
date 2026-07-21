import json

from weighttraits.taskdata.assignment import audit_paired_assignment_set, write_manifest_rows


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _condition(tmp_path, name, rows):
    manifest = tmp_path / f"{name}.jsonl"
    write_manifest_rows(rows, manifest)
    summary = tmp_path / f"{name}_summary.json"
    _write_json(
        summary,
        {"assignments": [{"tree_id": "tree_001", "assigned_manifest": str(manifest)}]},
    )
    return summary


def test_paired_assignment_audit_accepts_changed_fixed_topology(tmp_path):
    topology = [
        {"node_id": "n0", "parent_id": "root", "depth": 1, "path": ["root", "n0"], "grow": "train"},
        {"node_id": "n1", "parent_id": "n0", "depth": 2, "path": ["root", "n0", "n1"], "grow": "train"},
    ]
    topology_manifest = tmp_path / "topology.jsonl"
    write_manifest_rows(topology, topology_manifest)
    tree_set = tmp_path / "tree_set.json"
    _write_json(tree_set, {"trees": [{"tree_id": "tree_001", "manifest": str(topology_manifest)}]})

    reference = _condition(
        tmp_path,
        "reference",
        [
            {**topology[0], "task_family": "translation", "dataset_id": "wmt"},
            {**topology[1], "task_family": "qa", "dataset_id": "squad"},
        ],
    )
    candidate = _condition(
        tmp_path,
        "candidate",
        [
            {**topology[0], "task_family": "classification", "dataset_id": "sst2"},
            {**topology[1], "task_family": "qa", "dataset_id": "boolq"},
        ],
    )

    report = audit_paired_assignment_set(
        tree_set,
        reference,
        candidate,
        forbidden_task_families=["translation"],
    )

    assert report["ok"] is True
    assert report["n_audited_trees"] == 1
    assert report["n_changed_assignments"] == 2


def test_paired_assignment_audit_rejects_topology_and_forbidden_family(tmp_path):
    topology = [
        {"node_id": "n0", "parent_id": "root", "depth": 1, "path": ["root", "n0"], "grow": "train"},
    ]
    topology_manifest = tmp_path / "topology.jsonl"
    write_manifest_rows(topology, topology_manifest)
    tree_set = tmp_path / "tree_set.json"
    _write_json(tree_set, {"trees": [{"tree_id": "tree_001", "manifest": str(topology_manifest)}]})
    reference = _condition(
        tmp_path,
        "reference",
        [{**topology[0], "task_family": "qa", "dataset_id": "squad"}],
    )
    candidate = _condition(
        tmp_path,
        "candidate",
        [
            {
                **topology[0],
                "parent_id": "wrong",
                "task_family": "translation",
                "dataset_id": "wmt",
            }
        ],
    )

    report = audit_paired_assignment_set(
        tree_set,
        reference,
        candidate,
        forbidden_task_families=["translation"],
    )

    assert report["ok"] is False
    assert {issue["issue"] for issue in report["issues"]} == {
        "forbidden_task_family",
        "topology_mismatch",
    }
