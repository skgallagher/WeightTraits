from __future__ import annotations

from dataclasses import replace
import json
import unittest

from weighttraits.behavior.allnode.manifest import (
    NodeSpec,
    build_tree_topology,
    read_tree_topology,
    validate_cohort_topologies,
    validate_matching_cohort_topologies,
)


def toy_nodes() -> list[NodeSpec]:
    # root -> a -> {b, c -> d}; a/c are observed ancestors, b/d are leaves.
    return [
        NodeSpec(0, 0, "a", "root", 1, False),
        NodeSpec(1, 1, "b", "a", 2, True),
        NodeSpec(2, 2, "c", "a", 2, False),
        NodeSpec(3, 3, "d", "c", 3, True),
    ]


class TreeTopologyTests(unittest.TestCase):
    def test_builds_ancestor_aware_scope(self) -> None:
        tree = build_tree_topology("tree001", toy_nodes())
        self.assertEqual(tree.ordered_node_ids, ("a", "b", "c", "d"))
        self.assertEqual(tree.leaf_sensitivity_indices, (1, 3))
        self.assertEqual(tree.leaf_sensitivity_ids, ("b", "d"))
        self.assertEqual((tree.n_nodes, tree.n_internal, tree.n_leaves), (4, 2, 2))
        self.assertEqual(len(tree.pair_index), 6)
        pairs = {(row.left_node_id, row.right_node_id): row for row in tree.pair_index}
        self.assertEqual(pairs[("a", "b")].relationship, "ancestor_descendant")
        self.assertTrue(pairs[("a", "b")].is_parent_child)
        self.assertEqual(pairs[("a", "d")].ancestor_node_id, "a")
        self.assertFalse(pairs[("a", "d")].is_parent_child)
        self.assertEqual(pairs[("b", "d")].relationship, "collateral")
        self.assertEqual(pairs[("b", "d")].mrca_node_id, "a")
        self.assertEqual(pairs[("b", "d")].topological_distance, 3)
        self.assertEqual(pairs[("c", "d")].depth_difference, 1)
        self.assertNotIn("root", tree.ordered_node_ids)

    def test_hashes_are_deterministic_and_role_sensitive(self) -> None:
        first = build_tree_topology("tree001", toy_nodes())
        second = build_tree_topology("tree001", toy_nodes())
        self.assertEqual(first.to_dict(), second.to_dict())
        changed = toy_nodes()
        changed[3] = replace(changed[3], node_id="e")
        third = build_tree_topology("tree001", changed)
        self.assertNotEqual(first.nodes_sha256, third.nodes_sha256)
        self.assertNotEqual(first.pair_index_sha256, third.pair_index_sha256)

    def test_saved_topology_recomputes_observed_ancestor_contract(self) -> None:
        tree = build_tree_topology("tree001", toy_nodes())
        payload = json.loads(json.dumps(tree.to_dict()))
        self.assertEqual(payload["internal_node_treatment"], "observed_ancestors")
        self.assertEqual(read_tree_topology(payload), tree)

    def test_rejects_saved_topology_tampering(self) -> None:
        tree = build_tree_topology("tree001", toy_nodes())
        for field, value in (
            ("internal_node_treatment", "terminal_taxa"),
            ("population", "leaves_only"),
            ("exclude_root", 1),
            ("n_nodes", 4.0),
            ("leaf_sensitivity_indices", [0, 1, 2, 3]),
            ("nodes_sha256", "0" * 64),
            ("pair_index_sha256", "0" * 64),
            ("extra_field", True),
        ):
            with self.subTest(field=field):
                payload = tree.to_dict()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, "recomputed contract"):
                    read_tree_topology(payload)
        payload = tree.to_dict()
        payload["pair_index"][0]["relationship"] = "collateral"
        with self.assertRaisesRegex(ValueError, "recomputed contract"):
            read_tree_topology(payload)

    def test_dataclass_and_mapping_inputs_have_same_strict_types(self) -> None:
        for field, value in (
            ("node_ordinal", False),
            ("array_index", 0.0),
            ("depth", True),
            ("is_leaf", 0),
            ("node_id", " a "),
            ("parent_id", 12),
        ):
            invalid = replace(toy_nodes()[0], **{field: value})
            for first in (invalid, invalid.to_dict()):
                with self.subTest(field=field, input_type=type(first).__name__):
                    with self.assertRaises(ValueError):
                        build_tree_topology("tree001", [first, *toy_nodes()[1:]])

    def test_rejects_conflicting_explicit_node_role(self) -> None:
        first = toy_nodes()[0].to_dict()
        first["node_role"] = "leaf"
        with self.assertRaisesRegex(ValueError, "node_role is inconsistent"):
            build_tree_topology("tree001", [first, *toy_nodes()[1:]])

    def test_collateral_nodes_can_share_only_the_untrained_root(self) -> None:
        tree = build_tree_topology(
            "tree001",
            [NodeSpec(0, 0, "a", "root", 1, True), NodeSpec(1, 1, "b", "root", 1, True)],
        )
        pair = tree.pair_index[0]
        self.assertEqual(pair.relationship, "collateral")
        self.assertEqual(pair.mrca_node_id, "root")
        self.assertEqual(pair.topological_distance, 2)
        self.assertIsNone(pair.ancestor_node_id)
        self.assertFalse(pair.is_parent_child)

    def test_rejects_untrained_root_in_population(self) -> None:
        nodes = toy_nodes()
        nodes[0] = replace(nodes[0], node_id="root", parent_id="outside")
        with self.assertRaisesRegex(ValueError, "must not be a checkpoint node"):
            build_tree_topology("tree001", nodes)

    def test_rejects_duplicate_node(self) -> None:
        nodes = toy_nodes()
        nodes[3] = replace(nodes[3], node_id="b")
        with self.assertRaisesRegex(ValueError, "duplicate node IDs"):
            build_tree_topology("tree001", nodes)

    def test_rejects_noncontiguous_or_lexicographic_reorder(self) -> None:
        nodes = toy_nodes()
        nodes[1] = replace(nodes[1], node_ordinal=4)
        with self.assertRaisesRegex(ValueError, "ordinal must be contiguous"):
            build_tree_topology("tree001", nodes)
        reordered = [toy_nodes()[1], toy_nodes()[0], *toy_nodes()[2:]]
        with self.assertRaises(ValueError):
            build_tree_topology("tree001", reordered)

    def test_rejects_bad_parent_depth_cycle_and_leaf_flag(self) -> None:
        for nodes, message in (
            ([toy_nodes()[0], replace(toy_nodes()[1], parent_id="missing"), *toy_nodes()[2:]], "missing or later parent"),
            ([toy_nodes()[0], replace(toy_nodes()[1], depth=3), *toy_nodes()[2:]], "depth is inconsistent"),
            ([replace(toy_nodes()[0], parent_id="d"), *toy_nodes()[1:]], "missing or later parent"),
            ([replace(toy_nodes()[0], is_leaf=True), *toy_nodes()[1:]], "leaf flag is inconsistent"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_tree_topology("tree001", nodes)

    def test_validates_cohort_totals_and_order(self) -> None:
        first = build_tree_topology("tree001", toy_nodes())
        second = build_tree_topology("tree002", toy_nodes())
        receipt = validate_cohort_topologies(
            [first, second],
            expected_tree_ids=["tree001", "tree002"],
            expected_n_nodes=8,
            expected_n_internal=4,
            expected_n_leaves=4,
        )
        self.assertEqual(receipt["population"], "all_trained_nodes")
        self.assertTrue(receipt["exclude_root"])
        self.assertEqual(receipt["n_trees"], 2)
        with self.assertRaisesRegex(ValueError, "tree order"):
            validate_cohort_topologies(
                [first, second], expected_tree_ids=["tree002", "tree001"]
            )
        with self.assertRaisesRegex(ValueError, "nodes count"):
            validate_cohort_topologies([first, second], expected_n_nodes=7)

    def test_cohort_validation_rejects_manually_corrupted_tree(self) -> None:
        tree = build_tree_topology("tree001", toy_nodes())
        for invalid in (
            replace(tree, pair_index_sha256="0" * 64),
            replace(tree, leaf_sensitivity_indices=(0, 1, 2, 3)),
        ):
            with self.assertRaisesRegex(ValueError, "recomputed contract"):
                validate_cohort_topologies([invalid])

    def test_matching_cohorts_require_same_ancestry_not_just_counts(self) -> None:
        tree = build_tree_topology("tree001", toy_nodes())
        cohort_ids = ["llama_full", "llama_notrans", "flan_full", "flan_notrans"]
        cohorts = {label: [tree] for label in cohort_ids}
        expectations = dict(
            expected_cohort_ids=cohort_ids,
            expected_tree_ids=["tree001"],
            expected_n_nodes=4,
            expected_n_internal=2,
            expected_n_leaves=2,
        )
        receipt = validate_matching_cohort_topologies(cohorts, **expectations)
        self.assertEqual(receipt["n_cohorts"], 4)
        self.assertEqual(receipt["n_internal_per_cohort"], 2)

        changed = toy_nodes()
        changed[1] = replace(changed[1], parent_id="root", depth=1)
        cohorts["flan_full"] = [build_tree_topology("tree001", changed)]
        with self.assertRaisesRegex(ValueError, "topology differs"):
            validate_matching_cohort_topologies(cohorts, **expectations)
        del cohorts["flan_full"]
        with self.assertRaisesRegex(ValueError, "cohort IDs differ"):
            validate_matching_cohort_topologies(cohorts, **expectations)


if __name__ == "__main__":
    unittest.main()
