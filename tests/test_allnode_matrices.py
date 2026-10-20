from __future__ import annotations

import copy
import unittest

from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology
from weighttraits.behavior.allnode.matrices import (
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    validate_distance_matrix,
    validate_matrix_receipt,
)


def topology():
    # root -> a -> {b, c -> d}; a/c are observed ancestors, b/d are leaves.
    return build_tree_topology(
        "tree001",
        [
            NodeSpec(0, 0, "a", "root", 1, False),
            NodeSpec(1, 1, "b", "a", 2, True),
            NodeSpec(2, 2, "c", "a", 2, False),
            NodeSpec(3, 3, "d", "c", 3, True),
        ],
    )


def full_matrix():
    return [
        [0, 1, 2, 3],
        [1, 0, 4, 5],
        [2, 4, 0, 6],
        [3, 5, 6, 0],
    ]


class MatrixContractTests(unittest.TestCase):
    def test_validates_all_nodes_and_derives_exact_leaf_slice(self) -> None:
        tree = topology()
        artifact = validate_distance_matrix(
            tree,
            metric="cosine",
            ordered_node_ids=tree.ordered_node_ids,
            matrix=full_matrix(),
        )
        self.assertEqual(artifact.matrix[1][3], 5.0)
        self.assertEqual(artifact.leaf_sensitivity_matrix, ((0.0, 5.0), (5.0, 0.0)))
        receipt = artifact.receipt
        self.assertEqual(receipt.population, POPULATION)
        self.assertEqual(receipt.internal_node_treatment, INTERNAL_NODE_TREATMENT)
        self.assertEqual(receipt.ordered_node_ids, ("a", "b", "c", "d"))
        self.assertEqual(receipt.leaf_sensitivity_indices, (1, 3))
        self.assertEqual(receipt.leaf_sensitivity_ids, ("b", "d"))
        self.assertEqual(receipt.topology_sha256, tree.topology_sha256)
        self.assertEqual(receipt.pair_index_sha256, tree.pair_index_sha256)
        self.assertEqual(receipt.leaf_subset_sha256, tree.leaf_subset_sha256)
        self.assertEqual(receipt.matrix_shape, (4, 4))
        self.assertEqual(receipt.leaf_matrix_shape, (2, 2))

    def test_requires_exact_topology_order_not_merely_same_ids(self) -> None:
        tree = topology()
        with self.assertRaisesRegex(ValueError, "ordered node IDs"):
            validate_distance_matrix(
                tree,
                metric="cosine",
                ordered_node_ids=("a", "c", "b", "d"),
                matrix=full_matrix(),
            )

    def test_rejects_shape_finiteness_symmetry_diagonal_and_negative(self) -> None:
        tree = topology()
        cases = []
        short = full_matrix()[:-1]
        cases.append((short, "row count"))
        ragged = full_matrix()
        ragged[1] = ragged[1][:-1]
        cases.append((ragged, "row 1 has length"))
        infinite = full_matrix()
        infinite[0][1] = float("inf")
        cases.append((infinite, "not finite"))
        asymmetric = full_matrix()
        asymmetric[0][1] = 1.5
        cases.append((asymmetric, "not exactly symmetric"))
        diagonal = full_matrix()
        diagonal[2][2] = 0.01
        cases.append((diagonal, "diagonal"))
        negative = full_matrix()
        negative[0][1] = negative[1][0] = -1
        cases.append((negative, "negative"))
        boolean = full_matrix()
        boolean[0][1] = boolean[1][0] = True
        cases.append((boolean, "not numeric"))
        for matrix, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_distance_matrix(
                        tree,
                        metric="cosine",
                        ordered_node_ids=tree.ordered_node_ids,
                        matrix=matrix,
                    )

    def test_receipt_recomputation_detects_every_bound_hash_drift(self) -> None:
        tree = topology()
        artifact = validate_distance_matrix(
            tree,
            metric="cosine",
            ordered_node_ids=tree.ordered_node_ids,
            matrix=full_matrix(),
        )
        payload = artifact.receipt.to_dict()
        validate_matrix_receipt(
            payload,
            tree,
            metric="cosine",
            ordered_node_ids=tree.ordered_node_ids,
            matrix=full_matrix(),
        )
        for field in (
            "topology_sha256",
            "pair_index_sha256",
            "ordered_node_ids_sha256",
            "matrix_sha256",
            "leaf_subset_sha256",
            "leaf_sensitivity_matrix_sha256",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(payload)
                changed[field] = "0" * 64
                with self.assertRaisesRegex(ValueError, "recomputed contract"):
                    validate_matrix_receipt(
                        changed,
                        tree,
                        metric="cosine",
                        ordered_node_ids=tree.ordered_node_ids,
                        matrix=full_matrix(),
                    )

    def test_matrix_hash_changes_when_values_change(self) -> None:
        tree = topology()
        first = validate_distance_matrix(
            tree,
            metric="cosine",
            ordered_node_ids=tree.ordered_node_ids,
            matrix=full_matrix(),
        )
        changed = full_matrix()
        changed[1][3] = changed[3][1] = 5.25
        second = validate_distance_matrix(
            tree,
            metric="cosine",
            ordered_node_ids=tree.ordered_node_ids,
            matrix=changed,
        )
        self.assertNotEqual(first.receipt.matrix_sha256, second.receipt.matrix_sha256)
        self.assertNotEqual(
            first.receipt.leaf_sensitivity_matrix_sha256,
            second.receipt.leaf_sensitivity_matrix_sha256,
        )

    def test_receipt_binds_ancestry_even_for_same_ordered_ids(self) -> None:
        first_tree = topology()
        other_tree = build_tree_topology(
            "tree001",
            [
                NodeSpec(0, 0, "a", "root", 1, False),
                NodeSpec(1, 1, "b", "a", 2, True),
                NodeSpec(2, 2, "c", "a", 2, True),
                NodeSpec(3, 3, "d", "a", 2, True),
            ],
        )
        first = validate_distance_matrix(
            first_tree,
            metric="cosine",
            ordered_node_ids=first_tree.ordered_node_ids,
            matrix=full_matrix(),
        )
        second = validate_distance_matrix(
            other_tree,
            metric="cosine",
            ordered_node_ids=other_tree.ordered_node_ids,
            matrix=full_matrix(),
        )
        self.assertEqual(first.receipt.ordered_node_ids_sha256, second.receipt.ordered_node_ids_sha256)
        self.assertNotEqual(first.receipt.topology_sha256, second.receipt.topology_sha256)
        self.assertNotEqual(first.receipt.pair_index_sha256, second.receipt.pair_index_sha256)


if __name__ == "__main__":
    unittest.main()
