from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from weighttraits.behavior.allnode.checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    PRODUCTION_COHORT_IDS,
    build_tree_checkpoint_manifest,
)
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology, canonical_sha256
from weighttraits.behavior.allnode.covariates import NodeCovariate, build_tree_node_covariates
from weighttraits.behavior.allnode.production import (
    PRODUCTION_TREE_IDS,
    validate_production_resolutions,
)
from weighttraits.behavior.allnode.resolver import ResolvedAllNodeTree


REVISION = "a" * 40


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path, label):
    return ArtifactPin(path, digest(label))


def shaped_topology(tree_id, n_nodes, n_internal, changed=False):
    nodes = []
    for index in range(n_internal):
        parent = "root" if index == 0 else f"i{index - 1}"
        nodes.append(NodeSpec(index, index, f"i{index}", parent, index + 1, False))
    for offset in range(n_nodes - n_internal):
        ordinal = n_internal + offset
        parent_index = n_internal - 1
        depth = n_internal + 1
        if changed and offset == n_nodes - n_internal - 1:
            parent_index = n_internal - 2
            depth = n_internal
        nodes.append(
            NodeSpec(ordinal, ordinal, f"l{offset}", f"i{parent_index}", depth, True)
        )
    return build_tree_topology(tree_id, nodes)


def resolved_tree(cohort_key, training_id, tree_index, changed=False, n_nodes_override=None):
    tree_id = PRODUCTION_TREE_IDS[tree_index]
    n_nodes = 13 if tree_index < 41 else 12
    if n_nodes_override is not None:
        n_nodes = n_nodes_override
    n_internal = 6 if tree_index < 26 else 5
    topology = shaped_topology(tree_id, n_nodes, n_internal, changed=changed)
    shared = {
        "run_list": pin(f"/frozen/{training_id}/{tree_id}.runs.jsonl", f"runs/{training_id}/{tree_id}"),
        "ledger": pin(f"/frozen/{training_id}/{tree_id}.ledger.jsonl", f"ledger/{training_id}/{tree_id}"),
        "truth_manifest": pin(f"/frozen/{tree_id}.truth.jsonl", f"truth/{tree_id}"),
        "top_config": pin(f"/frozen/{cohort_key}/config.json", f"config/{cohort_key}"),
        "training_summary": pin(f"/frozen/{cohort_key}/summary.json", f"summary/{cohort_key}"),
        "completion_receipt": pin(f"/frozen/{cohort_key}/completion.json", f"completion/{cohort_key}"),
    }
    bindings = []
    for node in topology.nodes:
        bindings.append(
            NodeCheckpointBinding(
                population=POPULATION,
                internal_node_treatment=INTERNAL_NODE_TREATMENT,
                cohort_id=training_id,
                tree_id=tree_id,
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
                checkpoint=pin(
                    f"/frozen/{cohort_key}/{tree_id}/{node.node_id}/model",
                    f"checkpoint/{cohort_key}/{tree_id}/{node.node_id}",
                ),
                base_model_id="pinned/base-model",
                base_model_revision=REVISION,
                **shared,
            )
        )
    manifest = build_tree_checkpoint_manifest(topology, bindings, cohort_id=training_id)
    paths = {}
    for node in topology.nodes:
        paths[node.node_id] = (
            (topology.root_id, node.node_id)
            if node.parent_id == topology.root_id
            else (*paths[node.parent_id], node.node_id)
        )
    covariates = build_tree_node_covariates(
        topology,
        manifest,
        [
            NodeCovariate(
                node.node_id,
                node.parent_id,
                node.depth,
                paths[node.node_id],
                f"dataset-{node.node_ordinal % 5}",
                f"family-{node.node_ordinal % 3}",
                node.node_role,
            )
            for node in topology.nodes
        ],
        source_manifest=shared["truth_manifest"],
    )
    core = {
        "cohort_key": cohort_key,
        "training_cohort_id": training_id,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "topology_sha256": topology.topology_sha256,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "node_covariates_sha256": covariates.covariates_sha256,
    }
    return ResolvedAllNodeTree(
        cohort_key=cohort_key,
        training_cohort_id=training_id,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        topology=topology,
        checkpoint_manifest=manifest,
        node_covariates=covariates,
        resolution_sha256=canonical_sha256(core),
    )


def production_cohorts():
    training_ids = {
        "llama_full": "shared-llama-training",
        "llama_notrans": "shared-llama-training",
        "flan_full": "shared-flan-training",
        "flan_notrans": "shared-flan-training",
    }
    return {
        cohort_key: [
            resolved_tree(cohort_key, training_ids[cohort_key], index)
            for index in range(50)
        ]
        for cohort_key in PRODUCTION_COHORT_IDS
    }


class ProductionResolutionTests(unittest.TestCase):
    def test_exact_four_cohort_scope_and_shared_training_ids(self):
        receipt = validate_production_resolutions(production_cohorts()).to_dict()
        self.assertTrue(receipt["valid"])
        self.assertTrue(receipt["dry_run_only"])
        self.assertEqual(receipt["scheduler_actions"], 0)
        self.assertEqual(receipt["cohort_keys"], list(PRODUCTION_COHORT_IDS))
        self.assertEqual(receipt["n_trees_per_cohort"], 50)
        self.assertEqual(receipt["n_nodes_per_cohort"], 641)
        self.assertEqual(receipt["n_internal_per_cohort"], 276)
        self.assertEqual(receipt["n_leaves_per_cohort"], 365)
        training_ids = [row["training_cohort_id"] for row in receipt["cohorts"]]
        self.assertEqual(training_ids[0], training_ids[1])
        self.assertEqual(training_ids[2], training_ids[3])

    def test_rejects_missing_cohort_tree_reorder_and_640_nodes(self):
        cohorts = production_cohorts()
        del cohorts["flan_notrans"]
        with self.assertRaisesRegex(ValueError, "external cohort keys"):
            validate_production_resolutions(cohorts)

        cohorts = production_cohorts()
        cohorts["llama_full"][0], cohorts["llama_full"][1] = (
            cohorts["llama_full"][1], cohorts["llama_full"][0]
        )
        with self.assertRaisesRegex(ValueError, "tree order"):
            validate_production_resolutions(cohorts)

        cohorts = production_cohorts()
        cohorts["flan_notrans"][-1] = resolved_tree(
            "flan_notrans", "shared-flan-training", 49, n_nodes_override=11
        )
        with self.assertRaisesRegex(ValueError, "all-node totals"):
            validate_production_resolutions(cohorts)

    def test_rejects_cross_cohort_topology_and_resolution_hash_drift(self):
        cohorts = production_cohorts()
        cohorts["flan_full"][0] = resolved_tree(
            "flan_full", "shared-flan-training", 0, changed=True
        )
        with self.assertRaisesRegex(ValueError, "topology differs"):
            validate_production_resolutions(cohorts)

        cohorts = production_cohorts()
        cohorts["llama_full"][0] = replace(
            cohorts["llama_full"][0], resolution_sha256="0" * 64
        )
        with self.assertRaisesRegex(ValueError, "resolved tree hash"):
            validate_production_resolutions(cohorts)


if __name__ == "__main__":
    unittest.main()
