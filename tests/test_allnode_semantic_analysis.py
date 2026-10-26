from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import unittest

from weighttraits.behavior.allnode.checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    build_tree_checkpoint_manifest,
)
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology
from weighttraits.behavior.allnode.matrices import validate_distance_matrix
from weighttraits.behavior.allnode.covariates import (
    NodeCovariate,
    build_tree_node_covariates,
)
from weighttraits.behavior.allnode.semantic_analysis import (
    ADJUSTED_COVARIATES,
    CONTROLLED_COVARIATES,
    DIRECT_ARTIFACT_KIND,
    LEAF_SENSITIVITY_STRATUM,
    SEMANTIC_ARTIFACT_KIND,
    SEMANTIC_METRIC,
    STRATA,
    aggregate_production_semantic_runset,
    aggregate_semantic_runset,
    aggregate_semantic_runset_streaming,
    analyze_tree_semantic_bridge,
    bind_distance_artifact,
)


REVISION = "a" * 40


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path: str, label: str) -> ArtifactPin:
    return ArtifactPin(path=path, sha256=digest(label))


def topology(tree_id: str):
    g_parent = "a" if int(tree_id[-1]) % 2 == 0 else "e"
    g_depth = 2 if g_parent == "a" else 4
    return build_tree_topology(
        tree_id,
        [
            NodeSpec(0, 0, "a", "root", 1, False),
            NodeSpec(1, 1, "b", "a", 2, False),
            NodeSpec(2, 2, "c", "a", 2, True),
            NodeSpec(3, 3, "d", "b", 3, True),
            NodeSpec(4, 4, "e", "b", 3, False),
            NodeSpec(5, 5, "f", "e", 4, True),
            NodeSpec(6, 6, "g", g_parent, g_depth, True),
        ],
    )


def manifest(tree):
    shared = {
        "run_list": pin(f"/frozen/{tree.tree_id}.runs.jsonl", f"runs/{tree.tree_id}"),
        "ledger": pin(f"/frozen/{tree.tree_id}.ledger.jsonl", f"ledger/{tree.tree_id}"),
        "truth_manifest": pin(f"/frozen/{tree.tree_id}.manifest.jsonl", f"truth/{tree.tree_id}"),
        "top_config": pin("/frozen/config.json", "config"),
        "training_summary": pin("/frozen/summary.json", "summary"),
        "completion_receipt": pin("/frozen/completion.json", "completion"),
    }
    bindings = [
        NodeCheckpointBinding(
            population=POPULATION,
            internal_node_treatment=INTERNAL_NODE_TREATMENT,
            cohort_id="training-cohort",
            tree_id=tree.tree_id,
            node_ordinal=node.node_ordinal,
            array_index=node.array_index,
            node_id=node.node_id,
            parent_id=node.parent_id,
            depth=node.depth,
            is_leaf=node.is_leaf,
            node_role=node.node_role,
            status="completed",
            method="full",
            artifact_name="model",
            checkpoint=pin(f"/frozen/{tree.tree_id}/{node.node_id}/model", f"checkpoint/{tree.tree_id}/{node.node_id}"),
            base_model_id="pinned/base",
            base_model_revision=REVISION,
            **shared,
        )
        for node in tree.nodes
    ]
    return build_tree_checkpoint_manifest(tree, bindings, cohort_id="training-cohort")


def covariates(tree, checkpoint_manifest):
    paths = {}
    for node in tree.nodes:
        paths[node.node_id] = (
            (tree.root_id, node.node_id)
            if node.parent_id == tree.root_id
            else (*paths[node.parent_id], node.node_id)
        )
    datasets = {"a": "ds1", "b": "ds2", "c": "ds3", "d": "ds4", "e": "ds7", "f": "ds5", "g": "ds2"}
    families = {"a": "qa", "b": "classification", "c": "generation", "d": "qa", "e": "classification", "f": "generation", "g": "qa"}
    rows = [
        NodeCovariate(
            node_id=node.node_id,
            parent_id=node.parent_id,
            depth=node.depth,
            path=paths[node.node_id],
            dataset_id=datasets[node.node_id],
            task_family=families[node.node_id],
            role=node.node_role,
        )
        for node in tree.nodes
    ]
    return build_tree_node_covariates(
        tree,
        checkpoint_manifest,
        rows,
        source_manifest=checkpoint_manifest.bindings[0].truth_manifest,
    )


def matrix_from_pairs(tree, values):
    matrix = [[0.0] * tree.n_nodes for _ in range(tree.n_nodes)]
    assert len(values) == len(tree.pair_index)
    for pair, value in zip(tree.pair_index, values):
        matrix[pair.left_ordinal][pair.right_ordinal] = value
        matrix[pair.right_ordinal][pair.left_ordinal] = value
    return matrix


NOISE = (
    0.010,
    -0.006,
    0.004,
    -0.011,
    0.008,
    -0.003,
    0.012,
    -0.009,
    0.002,
    0.006,
    -0.007,
    0.009,
    -0.004,
    0.005,
    -0.010,
)


def analysis(tree_id: str, *, slope: float = -0.55, noise_shift: int = 0):
    tree = topology(tree_id)
    checkpoints = manifest(tree)
    node_covariates = covariates(tree, checkpoints)
    direct_values = [
        (((index * 7) % 23) + 1) / 25.0 for index in range(len(tree.pair_index))
    ]
    semantic_distances = []
    for index, direct in enumerate(direct_values):
        noise = NOISE[(index + noise_shift) % len(NOISE)]
        similarity = 0.82 + slope * direct + noise
        semantic_distances.append(1.0 - similarity)
    direct_matrix = validate_distance_matrix(
        tree,
        metric="cosine",
        ordered_node_ids=tree.ordered_node_ids,
        matrix=matrix_from_pairs(tree, direct_values),
    )
    semantic_matrix = validate_distance_matrix(
        tree,
        metric=SEMANTIC_METRIC,
        ordered_node_ids=tree.ordered_node_ids,
        matrix=matrix_from_pairs(tree, semantic_distances),
    )
    direct = bind_distance_artifact(
        direct_matrix,
        artifact_kind=DIRECT_ARTIFACT_KIND,
        endpoint_id="weight_cosine",
        checkpoint_set_sha256=checkpoints.checkpoint_set_sha256,
        source_receipt_sha256=digest(f"direct/{tree_id}"),
    )
    semantic = bind_distance_artifact(
        semantic_matrix,
        artifact_kind=SEMANTIC_ARTIFACT_KIND,
        endpoint_id="translation",
        checkpoint_set_sha256=checkpoints.checkpoint_set_sha256,
        source_receipt_sha256=digest(f"semantic/{tree_id}"),
    )
    return analyze_tree_semantic_bridge(
        tree,
        checkpoints,
        cohort_key="llama_full",
        endpoint_id="translation",
        node_covariates=node_covariates,
        direct=direct,
        semantic=semantic,
    )


class AllNodeTreeSemanticTests(unittest.TestCase):
    def test_full_population_primary_and_leaf_sensitivity_is_derived(self):
        result = analysis("tree001")
        receipt = result.receipt
        self.assertEqual(receipt.n_nodes, 7)
        self.assertEqual(receipt.n_internal, 3)
        self.assertEqual(receipt.n_leaves, 4)
        self.assertEqual(receipt.n_pairs, math.comb(7, 2))
        self.assertEqual(receipt.n_leaf_sensitivity_pairs, math.comb(4, 2))
        self.assertEqual(dict(receipt.pair_role_counts), {
            "internal_internal": 3,
            "internal_leaf": 12,
            "leaf_leaf": 6,
        })
        rows = {row.stratum: row for row in receipt.strata}
        self.assertEqual(tuple(rows), STRATA)
        self.assertEqual(rows["all_trained_node_pairs"].n_pairs, 21)
        self.assertEqual(rows[LEAF_SENSITIVITY_STRATUM].n_pairs, 6)
        self.assertAlmostEqual(
            rows["all_trained_node_pairs"].standardized_direct_slope,
            rows["all_trained_node_pairs"].pearson_r,
        )
        self.assertEqual(receipt.controlled_regression.covariates, CONTROLLED_COVARIATES)
        self.assertEqual(receipt.adjusted_regression.covariates, ADJUSTED_COVARIATES)
        self.assertIn("task_jaccard", receipt.controlled_regression.formula)
        self.assertEqual(receipt.internal_vs_leaf_indicator,
                         "pair_has_internal=(pair_role_class!=leaf_leaf)")
        first = result.pair_rows[0]
        self.assertAlmostEqual(first.semantic_similarity, 1.0 - first.semantic_distance)
        self.assertGreaterEqual(first.dataset_path_jaccard, 0.0)
        self.assertLessEqual(first.dataset_path_jaccard, 1.0)
        self.assertFalse(receipt.neighbor_joining)
        self.assertIsNone(receipt.tree_builder)
        self.assertFalse(receipt.internal_nodes_as_tips)

    def test_exact_covariate_checkpoint_and_matrix_bindings_fail_closed(self):
        result = analysis("tree001")
        bad_covariate = replace(
            result.node_covariates.nodes[-1],
            path=("root", "a", "b", "f"),
        )
        with self.assertRaisesRegex(ValueError, "path"):
            build_tree_node_covariates(
                result.topology,
                result.checkpoint_manifest,
                (*result.node_covariates.nodes[:-1], bad_covariate),
                source_manifest=result.node_covariates.source_manifest,
            )
        wrong_checkpoint = replace(result.semantic, checkpoint_set_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint set"):
            analyze_tree_semantic_bridge(
                result.topology,
                result.checkpoint_manifest,
                cohort_key="llama_full",
                endpoint_id="translation",
                node_covariates=result.node_covariates,
                direct=result.direct,
                semantic=wrong_checkpoint,
            )
        forged_matrix = replace(
            result.semantic.matrix,
            receipt=replace(result.semantic.matrix.receipt, pair_index_sha256="0" * 64),
        )
        with self.assertRaises(ValueError):
            analyze_tree_semantic_bridge(
                result.topology,
                result.checkpoint_manifest,
                cohort_key="llama_full",
                endpoint_id="translation",
                node_covariates=result.node_covariates,
                direct=result.direct,
                semantic=replace(result.semantic, matrix=forged_matrix),
            )
        wrong_semantic_metric = validate_distance_matrix(
            result.topology,
            metric="cosine",
            ordered_node_ids=result.topology.ordered_node_ids,
            matrix=result.semantic.matrix.matrix,
        )
        with self.assertRaisesRegex(ValueError, "metric"):
            analyze_tree_semantic_bridge(
                result.topology,
                result.checkpoint_manifest,
                cohort_key="llama_full",
                endpoint_id="translation",
                node_covariates=result.node_covariates,
                direct=result.direct,
                semantic=replace(result.semantic, matrix=wrong_semantic_metric),
            )


class TreeUnitRunSetTests(unittest.TestCase):
    def setUp(self):
        self.trees = (
            analysis("tree001", slope=-0.45, noise_shift=0),
            analysis("tree002", slope=-0.60, noise_shift=3),
            analysis("tree003", slope=-0.75, noise_shift=6),
            analysis("tree004", slope=-0.35, noise_shift=9),
        )

    def aggregate(self):
        return aggregate_semantic_runset(
            self.trees,
            expected_tree_ids=("tree001", "tree002", "tree003", "tree004"),
            expected_n_nodes=28,
            expected_n_internal=12,
            expected_n_leaves=16,
        )

    def test_equal_tree_primary_safe_dl_and_legacy_comparability_dl(self):
        receipt = self.aggregate()
        self.assertEqual(receipt.inference_unit, "tree")
        self.assertEqual(receipt.n_nodes, 28)
        self.assertEqual(receipt.per_tree_node_counts, (7, 7, 7, 7))
        self.assertFalse(receipt.pair_count_used_as_independent_sample_size)
        self.assertTrue(receipt.leaf_sensitivity_derived_from_full_matrix)
        primary = {row.stratum: row for row in receipt.strata}["all_trained_node_pairs"]
        self.assertEqual(primary.n_trees, 4)
        self.assertEqual(primary.n_defined_trees, 4)
        self.assertIsNotNone(primary.equal_tree_fisher_r)
        self.assertTrue(primary.dl.defined)
        self.assertFalse(primary.dl.pair_count_variance_used)
        self.assertEqual(
            primary.dl.within_tree_variance,
            "delete_one_trained_node_jackknife_on_fisher_z",
        )
        self.assertTrue(primary.legacy_comparability_dl.defined)
        self.assertTrue(primary.legacy_comparability_dl.pair_count_variance_used)
        self.assertEqual(
            primary.legacy_comparability_dl.within_tree_variance,
            "legacy_1_over_n_pairs_minus_3",
        )
        # Independent implementation of the inherited Fisher-z/DL equations
        # on this fixture (v=1/(n_pairs-3)).
        self.assertAlmostEqual(
            primary.legacy_comparability_dl.correlation,
            -0.9984147519866949,
            places=12,
        )
        self.assertAlmostEqual(
            primary.legacy_comparability_dl.se_z,
            0.12548634281730645,
            places=12,
        )
        self.assertAlmostEqual(
            primary.legacy_comparability_dl.tau2,
            0.007431733379094661,
            places=12,
        )

    def test_streaming_factory_is_exactly_two_pass_and_must_be_replayable(self):
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            yield from self.trees

        streamed = aggregate_semantic_runset_streaming(
            factory,
            expected_tree_ids=("tree001", "tree002", "tree003", "tree004"),
            expected_n_nodes=28,
            expected_n_internal=12,
            expected_n_leaves=16,
        )
        self.assertEqual(calls, 2)
        self.assertEqual(streamed, self.aggregate())

        one_shot = iter(self.trees)
        with self.assertRaisesRegex(ValueError, "replayable"):
            aggregate_semantic_runset_streaming(
                lambda: one_shot,
                expected_tree_ids=("tree001", "tree002", "tree003", "tree004"),
                expected_n_nodes=28,
                expected_n_internal=12,
                expected_n_leaves=16,
            )
        with self.assertRaisesRegex(ValueError, "per-tree node cardinalities"):
            aggregate_semantic_runset_streaming(
                factory,
                expected_tree_ids=("tree001", "tree002", "tree003", "tree004"),
                expected_n_nodes=28,
                expected_n_internal=12,
                expected_n_leaves=16,
                expected_nodes_per_tree=(8, 6, 7, 7),
            )

    def test_regression_inference_is_across_trees_or_tree_clustered(self):
        receipt = self.aggregate()
        slopes = {row.model_id: row for row in receipt.raw_slope_summaries}
        self.assertIn("all_nodes_legacy_controlled", slopes)
        primary = slopes["all_trained_node_pairs"]
        self.assertEqual(primary.inference_unit, "tree")
        self.assertFalse(primary.pair_level_standard_error)
        self.assertIsNotNone(primary.standard_error_across_trees)
        self.assertIsNotNone(primary.ci_low)
        self.assertIsNotNone(primary.p_value)
        clustered = {row.model_id: row for row in receipt.clustered_regressions}
        practical = clustered["legacy_practical_all_nodes"]
        controlled = clustered["legacy_controlled_all_nodes"]
        self.assertTrue(practical.defined)
        self.assertTrue(controlled.defined)
        self.assertEqual(practical.covariance_estimator, "CR1_tree_clustered_sandwich")
        self.assertFalse(practical.pair_level_standard_error)
        self.assertIn("C(run)", controlled.formula)
        self.assertIn("task_jaccard", controlled.formula)
        self.assertIsNotNone(controlled.tree_clustered_se)
        self.assertIsNotNone(controlled.p_value_normal_approximation)

    def test_exact_legacy_hc3_is_separate_from_robust_tree_inference(self):
        receipt = self.aggregate()
        hc3 = {row.model_id: row for row in receipt.legacy_hc3_regressions}
        clustered = {row.model_id: row for row in receipt.clustered_regressions}
        self.assertEqual(tuple(hc3), (
            "legacy_practical_all_nodes_hc3",
            "legacy_controlled_all_nodes_hc3",
            "legacy_practical_trained_leaf_sensitivity_hc3",
            "legacy_controlled_trained_leaf_sensitivity_hc3",
        ))
        for row in hc3.values():
            self.assertTrue(row.defined)
            self.assertTrue(row.legacy_comparability_only)
            self.assertTrue(row.pair_count_used_as_independent_sample_size)
            self.assertEqual(
                row.covariance_estimator, "HC3_pair_row_legacy_comparability"
            )
            self.assertIsNotNone(row.hc3_standard_error)
            self.assertIsNotNone(row.ci_low)
            self.assertIsNotNone(row.p_value_normal_approximation)
            self.assertIsNotNone(row.r_squared)
            self.assertIsNotNone(row.adjusted_r_squared)
        practical_hc3 = hc3["legacy_practical_all_nodes_hc3"]
        practical_clustered = clustered["legacy_practical_all_nodes"]
        self.assertEqual(
            practical_hc3.formula, "semantic_sim ~ cosine_dist + C(run)"
        )
        self.assertAlmostEqual(
            practical_hc3.direct_slope, practical_clustered.direct_slope, places=12
        )
        self.assertAlmostEqual(practical_hc3.direct_slope, -0.5365631360860719)
        self.assertAlmostEqual(practical_hc3.hc3_standard_error, 0.023299638029159256)
        controlled = hc3["legacy_controlled_all_nodes_hc3"]
        self.assertEqual(
            controlled.formula,
            "semantic_sim ~ depth_i + depth_j + lca_depth + task_jaccard + "
            "cosine_dist + C(run)",
        )
        # Independent statsmodels OLS(...).fit(cov_type="HC3") references for
        # the same canonical pair table.  These lock all-node primary and the
        # leaf-derived sensitivity to the inherited numerical estimator.
        references = {
            "legacy_practical_all_nodes_hc3": (
                -0.5365631360860714,
                0.02329963802915925,
            ),
            "legacy_controlled_all_nodes_hc3": (
                -0.5422043301354422,
                0.026096207115996365,
            ),
            "legacy_practical_trained_leaf_sensitivity_hc3": (
                -0.5407282110091742,
                0.05064232616734946,
            ),
            "legacy_controlled_trained_leaf_sensitivity_hc3": (
                -0.5793012628461194,
                0.0798647228315508,
            ),
        }
        for model_id, (slope, standard_error) in references.items():
            self.assertAlmostEqual(hc3[model_id].direct_slope, slope, places=12)
            self.assertAlmostEqual(
                hc3[model_id].hc3_standard_error, standard_error, places=12
            )

    def test_tree_order_counts_and_forged_tree_receipt_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "tree order"):
            aggregate_semantic_runset(
                self.trees,
                expected_tree_ids=("tree004", "tree003", "tree002", "tree001"),
                expected_n_nodes=28,
                expected_n_internal=12,
                expected_n_leaves=16,
            )
        with self.assertRaisesRegex(ValueError, "node totals"):
            aggregate_semantic_runset(
                self.trees,
                expected_tree_ids=("tree001", "tree002", "tree003", "tree004"),
                expected_n_nodes=27,
                expected_n_internal=12,
                expected_n_leaves=15,
            )
        forged = replace(
            self.trees[0],
            receipt=replace(self.trees[0].receipt, pair_rows_sha256="0" * 64),
        )
        with self.assertRaisesRegex(ValueError, "recomputed contract"):
            aggregate_semantic_runset(
                (forged, *self.trees[1:]),
                expected_tree_ids=("tree001", "tree002", "tree003", "tree004"),
                expected_n_nodes=28,
                expected_n_internal=12,
                expected_n_leaves=16,
            )

    def test_production_entrypoint_refuses_nonproduction_scope(self):
        with self.assertRaises(ValueError):
            aggregate_production_semantic_runset(
                self.trees,
                expected_nodes_per_tree=tuple(tree.receipt.n_nodes for tree in self.trees),
            )


if __name__ == "__main__":
    unittest.main()
