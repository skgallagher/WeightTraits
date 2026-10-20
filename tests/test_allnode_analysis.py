from __future__ import annotations

from dataclasses import replace
import math
import unittest

from weighttraits.behavior.allnode.analysis import (
    PRESPECIFIED_STRATA,
    aggregate_tree_pair_analyses,
    analyze_tree_pair_distances,
)
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology
from weighttraits.behavior.allnode.matrices import validate_distance_matrix


def topology(tree_id="tree001"):
    return build_tree_topology(
        tree_id,
        [
            NodeSpec(0, 0, "a", "root", 1, False),
            NodeSpec(1, 1, "b", "a", 2, True),
            NodeSpec(2, 2, "c", "a", 2, False),
            NodeSpec(3, 3, "d", "c", 3, True),
        ],
    )


def matrix_from_pairs(values):
    matrix = [[0.0] * 4 for _ in range(4)]
    coordinates = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    if len(values) != len(coordinates):
        raise ValueError("pair vector has the wrong length")
    for (left, right), value in zip(coordinates, values):
        matrix[left][right] = matrix[right][left] = value
    return matrix


def artifact(tree, metric, pair_values):
    return validate_distance_matrix(
        tree,
        metric=metric,
        ordered_node_ids=tree.ordered_node_ids,
        matrix=matrix_from_pairs(pair_values),
    )


class TreePairAnalysisTests(unittest.TestCase):
    def test_prespecified_ancestor_aware_strata(self):
        tree = topology()
        direct = artifact(tree, "direct_cosine", [1, 2, 3, 4, 5, 6])
        semantic = artifact(tree, "semantic_cosine", [2, 4, 6, 8, 10, 12])
        result = analyze_tree_pair_distances(tree, direct, semantic).receipt
        self.assertEqual(result.inference_unit, "tree")
        self.assertEqual(result.prespecified_strata, PRESPECIFIED_STRATA)
        rows = {row.stratum: row for row in result.strata}
        self.assertEqual(rows["all_trained_node_pairs"].n_pairs, 6)
        self.assertEqual(rows["ancestor_descendant"].n_pairs, 4)
        self.assertEqual(rows["parent_child"].n_pairs, 3)
        self.assertEqual(rows["collateral"].n_pairs, 2)
        self.assertEqual(rows["trained_leaf_pairs"].n_pairs, 1)
        for name in PRESPECIFIED_STRATA[:-1]:
            self.assertAlmostEqual(rows[name].pearson_r, 1.0)
        self.assertFalse(rows["trained_leaf_pairs"].correlation_defined)
        self.assertEqual(
            rows["trained_leaf_pairs"].undefined_reason, "fewer_than_two_pairs"
        )
        self.assertEqual(result.direct_matrix_sha256, direct.receipt.matrix_sha256)
        self.assertEqual(result.semantic_matrix_sha256, semantic.receipt.matrix_sha256)
        self.assertEqual(result.topology_sha256, tree.topology_sha256)
        self.assertEqual(result.pair_index_sha256, tree.pair_index_sha256)

    def test_constant_strata_are_explicitly_undefined(self):
        tree = topology()
        direct = artifact(tree, "direct", [1, 2, 3, 4, 5, 6])
        semantic = artifact(tree, "semantic", [1, 1, 1, 1, 1, 1])
        rows = {
            row.stratum: row
            for row in analyze_tree_pair_distances(tree, direct, semantic).receipt.strata
        }
        self.assertEqual(rows["all_trained_node_pairs"].undefined_reason, "constant_semantic")
        self.assertEqual(rows["ancestor_descendant"].undefined_reason, "constant_semantic")
        self.assertEqual(rows["parent_child"].undefined_reason, "constant_semantic")
        self.assertEqual(rows["collateral"].undefined_reason, "constant_semantic")
        self.assertEqual(rows["trained_leaf_pairs"].undefined_reason, "fewer_than_two_pairs")

    def test_rejects_forged_or_differently_bound_matrix(self):
        tree = topology()
        direct = artifact(tree, "direct", [1, 2, 3, 4, 5, 6])
        semantic = artifact(tree, "semantic", [2, 4, 6, 8, 10, 12])
        forged_receipt = replace(semantic.receipt, pair_index_sha256="0" * 64)
        forged = replace(semantic, receipt=forged_receipt)
        with self.assertRaises(ValueError):
            analyze_tree_pair_distances(tree, direct, forged)

        other = topology("tree002")
        other_semantic = artifact(other, "semantic", [2, 4, 6, 8, 10, 12])
        with self.assertRaises(ValueError):
            analyze_tree_pair_distances(tree, direct, other_semantic)


class RunSetAnalysisTests(unittest.TestCase):
    def analysis(self, tree_id, semantic_pairs):
        tree = topology(tree_id)
        direct = artifact(tree, "direct", [1, 2, 3, 4, 5, 6])
        semantic = artifact(tree, "semantic", semantic_pairs)
        return analyze_tree_pair_distances(tree, direct, semantic)

    def test_equal_tree_weight_mean_sd_and_se(self):
        positive = self.analysis("tree001", [1, 2, 3, 4, 5, 6])
        negative = self.analysis("tree002", [6, 5, 4, 3, 2, 1])
        receipt = aggregate_tree_pair_analyses([positive, negative])
        self.assertEqual(receipt.inference_unit, "tree")
        self.assertEqual(
            receipt.aggregation, "equal_weight_mean_of_per_tree_pearson_correlations"
        )
        self.assertFalse(receipt.pair_count_used_as_independent_sample_size)
        self.assertFalse(receipt.fisher_z_pair_level_standard_error)
        rows = {row.stratum: row for row in receipt.strata}
        primary = rows["all_trained_node_pairs"]
        self.assertAlmostEqual(primary.mean_pearson_r, 0.0)
        self.assertAlmostEqual(primary.sample_sd, math.sqrt(2.0))
        self.assertAlmostEqual(primary.standard_error, 1.0)
        self.assertEqual(primary.n_valid_trees, 2)
        self.assertEqual(primary.n_undefined_trees, 0)
        leaves = rows["trained_leaf_pairs"]
        self.assertEqual(leaves.n_valid_trees, 0)
        self.assertEqual(leaves.n_undefined_trees, 2)
        self.assertIsNone(leaves.mean_pearson_r)

    def test_undefined_tree_is_counted_not_imputed(self):
        positive = self.analysis("tree001", [1, 2, 3, 4, 5, 6])
        constant = self.analysis("tree002", [1, 1, 1, 1, 1, 1])
        primary = aggregate_tree_pair_analyses([positive, constant]).strata[0]
        self.assertEqual(primary.n_valid_trees, 1)
        self.assertEqual(primary.n_undefined_trees, 1)
        self.assertAlmostEqual(primary.mean_pearson_r, 1.0)
        self.assertIsNone(primary.sample_sd)
        self.assertIsNone(primary.standard_error)

    def test_rejects_duplicate_trees_and_drifted_tree_receipts(self):
        first = self.analysis("tree001", [1, 2, 3, 4, 5, 6])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_tree_pair_analyses([first, first])
        drifted = replace(
            first,
            receipt=replace(first.receipt, strata_sha256="0" * 64),
        )
        with self.assertRaisesRegex(ValueError, "drifted"):
            aggregate_tree_pair_analyses([drifted])


if __name__ == "__main__":
    unittest.main()
