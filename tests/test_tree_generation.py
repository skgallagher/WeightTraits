import json

from weighttraits.trees.generate import (
    generate_tree,
    generate_tree_from_config,
    generate_tree_set,
    leaf_ids,
    manifest_rows,
    tree_stats,
    write_manifest_jsonl,
)


def test_chain_tree_has_one_leaf_and_ordered_paths():
    root = generate_tree("chain", n_nodes=4)
    rows = manifest_rows(root)

    assert leaf_ids(root) == ["n3"]
    assert rows[-1]["path"] == ["root", "n0", "n1", "n2", "n3"]
    assert rows[-1]["depth"] == 4


def test_poisson_tree_is_deterministic_and_respects_limits():
    a = generate_tree(
        "poisson_branching",
        n_nodes=14,
        max_depth=4,
        branch_lambda=[1.5, 2.0, 1.0],
        seed=87,
        max_children=3,
    )
    b = generate_tree(
        "poisson_branching",
        n_nodes=14,
        max_depth=4,
        branch_lambda=[1.5, 2.0, 1.0],
        seed=87,
        max_children=3,
    )

    assert manifest_rows(a) == manifest_rows(b)
    stats = tree_stats(a)
    assert 1 <= stats["n_nodes"] <= 14
    assert stats["max_depth"] <= 4
    assert stats["n_leaves"] >= 1


def test_poisson_constraints_resample_until_minimums():
    root = generate_tree(
        "poisson_branching",
        n_nodes=20,
        max_depth=5,
        min_depth=3,
        min_leaves=4,
        branch_lambda=1.8,
        seed=11,
        max_children=4,
    )
    stats = tree_stats(root)

    assert stats["max_depth"] >= 3
    assert stats["n_leaves"] >= 4


def test_generate_tree_set_rejection_samples_minimum_leaves():
    generated = generate_tree_set(
        {
            "generator": "poisson_branching",
            "n_nodes": 14,
            "max_depth": 4,
            "branch_lambda": 1.5,
        },
        n_trees=10,
        seed_start=20260707,
        min_leaves=4,
    )

    assert len(generated) == 10
    assert generated[0].tree_id == "tree_001"
    assert len({item.seed for item in generated}) == 10
    assert all(item.stats["n_leaves"] >= 4 for item in generated)
    assert all(item.stats["max_depth"] <= 4 for item in generated)


def test_ellmtrees_balanced_default_has_polytomy_and_requested_leaves():
    root = generate_tree("ellmtrees_balanced", n_leaves=10)
    stats = tree_stats(root)

    assert stats["n_leaves"] == 10
    assert stats["max_depth"] == 2
    assert stats["max_out_degree"] > 2
    assert stats["n_polytomies"] >= 1


def test_pruned_binary_backbone_can_make_deep_polytomous_tree():
    root = generate_tree(
        "pruned_binary_backbone",
        backbone_leaves=32,
        target_leaves=12,
        min_depth=3,
        min_leaves=8,
        prune_lambda=3.0,
        contract_probability=0.6,
        seed=5,
    )
    stats = tree_stats(root)

    assert stats["max_depth"] >= 3
    assert stats["n_leaves"] >= 8
    assert stats["n_polytomies"] >= 1


def test_fixed_tree_from_config_preserves_shape():
    root = generate_tree_from_config(
        {
            "generator": "fixed",
            "root": {
                "id": "root",
                "children": [
                    {"id": "n0"},
                    {"id": "n1", "children": [{"id": "n2"}, {"id": "n3"}]},
                ],
            },
        }
    )

    assert leaf_ids(root) == ["n0", "n2", "n3"]
    assert [row["node_id"] for row in manifest_rows(root)] == ["n0", "n1", "n2", "n3"]


def test_write_manifest_jsonl(tmp_path):
    root = generate_tree("balanced", n_nodes=5, branch_factor=2)
    out = tmp_path / "manifest.jsonl"
    rows = write_manifest_jsonl(root, out)

    loaded = [json.loads(line) for line in out.read_text().splitlines()]
    assert loaded == rows
    assert len(rows) == 5
