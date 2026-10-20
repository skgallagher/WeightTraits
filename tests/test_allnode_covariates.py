from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from weighttraits.behavior.allnode.checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    build_tree_checkpoint_manifest,
)
from weighttraits.behavior.allnode.covariates import (
    NodeCovariate,
    build_tree_node_covariates,
    read_tree_node_covariates,
    task_path_jaccard,
    task_path_labels,
)
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path: str, label: str) -> ArtifactPin:
    return ArtifactPin(path, digest(label))


def fixture():
    topology = build_tree_topology(
        "tree001",
        [
            NodeSpec(0, 0, "a", "root", 1, False),
            NodeSpec(1, 1, "b", "a", 2, True),
            NodeSpec(2, 2, "c", "a", 2, False),
            NodeSpec(3, 3, "d", "c", 3, True),
        ],
    )
    truth = pin("/frozen/tree001.truth.jsonl", "truth")
    shared = {
        "run_list": pin("/frozen/tree001.runs.jsonl", "runs"),
        "ledger": pin("/frozen/tree001.ledger.jsonl", "ledger"),
        "truth_manifest": truth,
        "top_config": pin("/frozen/config.json", "config"),
        "training_summary": pin("/frozen/summary.json", "summary"),
        "completion_receipt": pin("/frozen/completion.json", "completion"),
    }
    bindings = [
        NodeCheckpointBinding(
            population=POPULATION,
            internal_node_treatment=INTERNAL_NODE_TREATMENT,
            cohort_id="training",
            tree_id=topology.tree_id,
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
            checkpoint=pin(f"/frozen/{node.node_id}/model", node.node_id),
            base_model_id="base/model",
            base_model_revision="a" * 40,
            **shared,
        )
        for node in topology.nodes
    ]
    manifest = build_tree_checkpoint_manifest(
        topology, bindings, cohort_id="training"
    )
    labels = {
        "a": ("boolq", "qa", ("root", "a")),
        "b": ("rte", "nli", ("root", "a", "b")),
        "c": ("xsum", "summarization", ("root", "a", "c")),
        "d": ("rte", "nli", ("root", "a", "c", "d")),
    }
    rows = [
        NodeCovariate(
            node_id=node.node_id,
            parent_id=node.parent_id,
            depth=node.depth,
            path=labels[node.node_id][2],
            dataset_id=labels[node.node_id][0],
            task_family=labels[node.node_id][1],
            role=node.node_role,
        )
        for node in topology.nodes
    ]
    contract = build_tree_node_covariates(
        topology, manifest, rows, source_manifest=truth
    )
    return topology, manifest, rows, contract


class NodeCovariateTests(unittest.TestCase):
    def test_exact_order_hash_and_legacy_path_task_jaccard(self):
        topology, manifest, _rows, contract = fixture()
        self.assertEqual([row.node_id for row in contract.nodes], ["a", "b", "c", "d"])
        self.assertEqual(task_path_labels(contract, "b"), frozenset({"boolq", "rte"}))
        self.assertEqual(
            task_path_labels(contract, "d"),
            frozenset({"boolq", "xsum", "rte"}),
        )
        self.assertAlmostEqual(task_path_jaccard(contract, "b", "d"), 2 / 3)
        self.assertAlmostEqual(
            task_path_jaccard(contract, "b", "d", label="task_family"),
            2 / 3,
        )
        self.assertEqual(
            read_tree_node_covariates(contract.to_dict(), topology, manifest),
            contract,
        )

    def test_missing_reordered_and_topology_drift_fail_closed(self):
        topology, manifest, rows, _contract = fixture()
        with self.assertRaisesRegex(ValueError, "count differs"):
            build_tree_node_covariates(
                topology, manifest, rows[:-1], source_manifest=manifest.bindings[0].truth_manifest
            )
        with self.assertRaisesRegex(ValueError, "row 0 node_id differs"):
            build_tree_node_covariates(
                topology,
                manifest,
                [rows[1], rows[0], *rows[2:]],
                source_manifest=manifest.bindings[0].truth_manifest,
            )
        changed = list(rows)
        changed[3] = replace(changed[3], path=("root", "x", "c", "d"))
        with self.assertRaisesRegex(ValueError, "path differs"):
            build_tree_node_covariates(
                topology,
                manifest,
                changed,
                source_manifest=manifest.bindings[0].truth_manifest,
            )

    def test_source_label_and_saved_hash_drift_are_detected(self):
        topology, manifest, rows, contract = fixture()
        with self.assertRaisesRegex(ValueError, "source manifest differs"):
            build_tree_node_covariates(
                topology,
                manifest,
                rows,
                source_manifest=pin("/frozen/other.truth.jsonl", "other"),
            )
        changed = list(rows)
        changed[1] = replace(changed[1], dataset_id="mnli")
        changed_contract = build_tree_node_covariates(
            topology,
            manifest,
            changed,
            source_manifest=manifest.bindings[0].truth_manifest,
        )
        self.assertNotEqual(contract.covariates_sha256, changed_contract.covariates_sha256)

        payload = contract.to_dict()
        payload["nodes"][0]["task_family"] = "changed"
        with self.assertRaisesRegex(ValueError, "node hash drift"):
            read_tree_node_covariates(payload, topology, manifest)


if __name__ == "__main__":
    unittest.main()
