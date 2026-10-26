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
    read_tree_checkpoint_manifest,
    validate_production_checkpoint_manifests,
)
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology


REVISION = "a" * 40


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path: str, label: str) -> ArtifactPin:
    return ArtifactPin(path, digest(label))


def toy_topology(tree_id: str = "tree001"):
    # root -> a -> {b, c -> d}; a/c are observed ancestors.
    return build_tree_topology(
        tree_id,
        [
            NodeSpec(0, 0, "a", "root", 1, False),
            NodeSpec(1, 1, "b", "a", 2, True),
            NodeSpec(2, 2, "c", "a", 2, False),
            NodeSpec(3, 3, "d", "c", 3, True),
        ],
    )


def binding(node: NodeSpec, *, cohort_id: str, tree_id: str) -> NodeCheckpointBinding:
    return NodeCheckpointBinding(
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        cohort_id=cohort_id,
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
            f"/frozen/{cohort_id}/{tree_id}/{node.node_id}/model",
            f"checkpoint/{cohort_id}/{tree_id}/{node.node_id}",
        ),
        base_model_id="pinned/base-model",
        base_model_revision=REVISION,
        run_list=pin(f"/frozen/{tree_id}.runs.jsonl", f"runs/{tree_id}"),
        ledger=pin(f"/frozen/{tree_id}.ledger.jsonl", f"ledger/{tree_id}"),
        truth_manifest=pin(f"/frozen/{tree_id}.truth.jsonl", f"truth/{tree_id}"),
        top_config=pin(f"/frozen/{cohort_id}/config.json", f"config/{cohort_id}"),
        training_summary=pin(
            f"/frozen/{cohort_id}/summary.json", f"summary/{cohort_id}"
        ),
        completion_receipt=pin(
            f"/frozen/{cohort_id}/completion.json", f"completion/{cohort_id}"
        ),
    )


def toy_bindings(cohort_id: str = "cohort-a", tree_id: str = "tree001"):
    topology = toy_topology(tree_id)
    return topology, [
        binding(node, cohort_id=cohort_id, tree_id=tree_id) for node in topology.nodes
    ]


class TreeCheckpointManifestTests(unittest.TestCase):
    def test_binds_observed_ancestors_in_exact_topology_order(self) -> None:
        topology, bindings = toy_bindings()
        manifest = build_tree_checkpoint_manifest(
            topology, bindings, cohort_id="cohort-a"
        )

        self.assertEqual(manifest.population, "all_trained_nodes")
        self.assertEqual(manifest.internal_node_treatment, "observed_ancestors")
        self.assertEqual((manifest.n_nodes, manifest.n_internal, manifest.n_leaves), (4, 2, 2))
        self.assertEqual([row.node_id for row in manifest.bindings], ["a", "b", "c", "d"])
        self.assertEqual([row.node_role for row in manifest.bindings], ["internal", "leaf", "internal", "leaf"])
        self.assertEqual(manifest.bindings[2].parent_id, "a")
        self.assertEqual(manifest.bindings[2].depth, 2)
        self.assertEqual(manifest.bindings[2].status, "completed")
        self.assertEqual(
            read_tree_checkpoint_manifest(manifest.to_dict()).to_dict(),
            manifest.to_dict(),
        )

    def test_rejects_reorder_identity_drift_and_fake_leaf_ordinal(self) -> None:
        topology, bindings = toy_bindings()
        with self.assertRaisesRegex(ValueError, "node_ordinal drift"):
            build_tree_checkpoint_manifest(
                topology, [bindings[1], bindings[0], *bindings[2:]], cohort_id="cohort-a"
            )

        changed = list(bindings)
        changed[2] = replace(changed[2], parent_id="root")
        with self.assertRaisesRegex(ValueError, "parent_id drift"):
            build_tree_checkpoint_manifest(topology, changed, cohort_id="cohort-a")

        row = bindings[0].to_dict()
        row["leaf_ordinal"] = 0
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            build_tree_checkpoint_manifest(
                topology, [row, *bindings[1:]], cohort_id="cohort-a"
            )

    def test_requires_completed_status_method_artifact_and_strict_pins(self) -> None:
        topology, bindings = toy_bindings()
        bad_status = [replace(bindings[0], status="running"), *bindings[1:]]
        with self.assertRaisesRegex(ValueError, "status must be exactly"):
            build_tree_checkpoint_manifest(topology, bad_status, cohort_id="cohort-a")

        bad_artifact = [replace(bindings[0], artifact_name="adapter"), *bindings[1:]]
        with self.assertRaisesRegex(ValueError, "invalid for training method"):
            build_tree_checkpoint_manifest(topology, bad_artifact, cohort_id="cohort-a")

        bad_pin = bindings[0].to_dict()
        bad_pin["checkpoint"]["sha256"] = "ABC"
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            build_tree_checkpoint_manifest(
                topology, [bad_pin, *bindings[1:]], cohort_id="cohort-a"
            )

        bad_path = bindings[0].to_dict()
        bad_path["checkpoint"]["path"] = "relative/model"
        with self.assertRaisesRegex(ValueError, "absolute normalized path"):
            build_tree_checkpoint_manifest(
                topology, [bad_path, *bindings[1:]], cohort_id="cohort-a"
            )

    def test_rejects_shared_source_pin_drift_and_saved_manifest_tampering(self) -> None:
        topology, bindings = toy_bindings()
        changed = list(bindings)
        changed[3] = replace(
            changed[3], run_list=pin("/frozen/other.runs.jsonl", "other-runs")
        )
        with self.assertRaisesRegex(ValueError, "run_list must be identical"):
            build_tree_checkpoint_manifest(topology, changed, cohort_id="cohort-a")

        manifest = build_tree_checkpoint_manifest(topology, bindings, cohort_id="cohort-a")
        saved = manifest.to_dict()
        saved["bindings_sha256"] = digest("forged")
        with self.assertRaisesRegex(ValueError, "differs from its recomputed contract"):
            read_tree_checkpoint_manifest(saved)


def shaped_topology(tree_id: str, *, n_nodes: int, n_internal: int):
    nodes: list[NodeSpec] = []
    for index in range(n_internal):
        parent = "root" if index == 0 else f"i{index - 1}"
        nodes.append(NodeSpec(index, index, f"i{index}", parent, index + 1, False))
    for offset in range(n_nodes - n_internal):
        ordinal = n_internal + offset
        nodes.append(
            NodeSpec(
                ordinal,
                ordinal,
                f"l{offset}",
                f"i{n_internal - 1}",
                n_internal + 1,
                True,
            )
        )
    return build_tree_topology(tree_id, nodes)


def production_manifests(cohort_id: str, *, final_tree_nodes: int = 12):
    tree_ids = [f"tree{index:03d}" for index in range(1, 51)]
    manifests = []
    for index, tree_id in enumerate(tree_ids):
        n_nodes = 13 if index < 41 else 12
        if index == 49:
            n_nodes = final_tree_nodes
        n_internal = 6 if index < 26 else 5
        topology = shaped_topology(tree_id, n_nodes=n_nodes, n_internal=n_internal)
        rows = [
            binding(node, cohort_id=cohort_id, tree_id=tree_id)
            for node in topology.nodes
        ]
        manifests.append(
            build_tree_checkpoint_manifest(topology, rows, cohort_id=cohort_id)
        )
    return tree_ids, manifests


def production_cohorts(*, changed_cohort: str | None = None, final_tree_nodes: int = 12):
    cohorts = {}
    tree_ids = None
    for cohort_id in PRODUCTION_COHORT_IDS:
        ids, manifests = production_manifests(
            cohort_id,
            final_tree_nodes=(
                final_tree_nodes if cohort_id == changed_cohort else 12
            ),
        )
        tree_ids = ids
        cohorts[cohort_id] = manifests
    assert tree_ids is not None
    return tree_ids, cohorts


class ProductionCheckpointTests(unittest.TestCase):
    def test_requires_exact_fifty_tree_641_node_observed_ancestor_scope(self) -> None:
        tree_ids, cohorts = production_cohorts()
        receipt = validate_production_checkpoint_manifests(
            cohorts,
            expected_cohort_ids=PRODUCTION_COHORT_IDS,
            expected_tree_ids=tree_ids,
        )
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["population"], "all_trained_nodes")
        self.assertEqual(receipt["internal_node_treatment"], "observed_ancestors")
        self.assertEqual(receipt["n_nodes_per_cohort"], 641)
        self.assertEqual(receipt["n_internal_per_cohort"], 276)
        self.assertEqual(receipt["n_leaves_per_cohort"], 365)

    def test_rejects_640_member_scope_at_production_validation_level(self) -> None:
        tree_ids, cohorts = production_cohorts(
            changed_cohort="flan_notrans", final_tree_nodes=11
        )
        with self.assertRaisesRegex(ValueError, "production checkpoint totals differ"):
            validate_production_checkpoint_manifests(
                cohorts,
                expected_cohort_ids=PRODUCTION_COHORT_IDS,
                expected_tree_ids=tree_ids,
            )

    def test_requires_identical_topology_across_cohorts(self) -> None:
        tree_ids, cohorts = production_cohorts()
        receipt = validate_production_checkpoint_manifests(
            cohorts,
            expected_cohort_ids=PRODUCTION_COHORT_IDS,
            expected_tree_ids=tree_ids,
        )
        self.assertEqual(receipt["n_cohorts"], 4)
        with self.assertRaisesRegex(ValueError, "production cohort IDs must be exactly"):
            validate_production_checkpoint_manifests(
                cohorts,
                expected_cohort_ids=PRODUCTION_COHORT_IDS[:-1],
                expected_tree_ids=tree_ids,
            )

    def test_rejects_checkpoint_path_reuse_across_trees(self) -> None:
        tree_ids, cohorts = production_cohorts()
        manifests = cohorts["llama_full"]
        second = manifests[1]
        changed = list(second.bindings)
        changed[0] = replace(
            changed[0], checkpoint=manifests[0].bindings[0].checkpoint
        )
        topology = build_tree_topology(
            second.tree_id,
            [
                NodeSpec(
                    row.node_ordinal,
                    row.array_index,
                    row.node_id,
                    row.parent_id,
                    row.depth,
                    row.is_leaf,
                )
                for row in changed
            ],
        )
        manifests[1] = build_tree_checkpoint_manifest(
            topology, changed, cohort_id="llama_full"
        )
        with self.assertRaisesRegex(ValueError, "reuses a checkpoint path"):
            validate_production_checkpoint_manifests(
                cohorts,
                expected_cohort_ids=PRODUCTION_COHORT_IDS,
                expected_tree_ids=tree_ids,
            )


if __name__ == "__main__":
    unittest.main()
