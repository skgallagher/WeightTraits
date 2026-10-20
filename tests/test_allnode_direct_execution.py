from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import unittest

from weighttraits.behavior.allnode.checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    build_tree_checkpoint_manifest,
)
from weighttraits.behavior.allnode.direct_execution import (
    DIRECT_METRICS,
    LoadedTensorReader,
    execute_direct_distances,
    execute_manifest_direct_distances,
    pinned_snapshot_adapter,
    plan_direct_artifacts,
    verify_direct_artifact_plan,
)
from weighttraits.behavior.allnode.covariates import NodeCovariate, build_tree_node_covariates
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology, canonical_sha256
from weighttraits.behavior.allnode.resolver import ResolvedAllNodeTree


REVISION = "a" * 40


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path, label):
    return ArtifactPin(path=path, sha256=digest(label))


def resolved_tree():
    topology = build_tree_topology(
        "tree001",
        [
            NodeSpec(0, 0, "a", "root", 1, False),
            NodeSpec(1, 1, "b", "a", 2, True),
            NodeSpec(2, 2, "c", "a", 2, False),
            NodeSpec(3, 3, "d", "c", 3, True),
        ],
    )
    shared = {
        "run_list": pin("/frozen/tree001.runs.jsonl", "runs"),
        "ledger": pin("/frozen/tree001.ledger.jsonl", "ledger"),
        "truth_manifest": pin("/frozen/tree001.truth.jsonl", "truth"),
        "top_config": pin("/frozen/config.json", "config"),
        "training_summary": pin("/frozen/summary.json", "summary"),
        "completion_receipt": pin("/frozen/completion.json", "completion"),
    }
    bindings = [
        NodeCheckpointBinding(
            population=POPULATION,
            internal_node_treatment=INTERNAL_NODE_TREATMENT,
            cohort_id="shared-llama-training",
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
            checkpoint=pin(f"/frozen/{node.node_id}/model", f"checkpoint/{node.node_id}"),
            base_model_id="pinned/base-model",
            base_model_revision=REVISION,
            **shared,
        )
        for node in topology.nodes
    ]
    manifest = build_tree_checkpoint_manifest(
        topology,
        bindings,
        cohort_id="shared-llama-training",
    )
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
                f"dataset-{node.node_ordinal}",
                f"family-{node.node_ordinal % 2}",
                node.node_role,
            )
            for node in topology.nodes
        ],
        source_manifest=shared["truth_manifest"],
    )
    core = {
        "cohort_key": "llama_full",
        "training_cohort_id": "shared-llama-training",
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "topology_sha256": topology.topology_sha256,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "node_covariates_sha256": covariates.covariates_sha256,
    }
    return ResolvedAllNodeTree(
        cohort_key="llama_full",
        training_cohort_id="shared-llama-training",
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        topology=topology,
        checkpoint_manifest=manifest,
        node_covariates=covariates,
        resolution_sha256=canonical_sha256(core),
    )


VECTORS = {
    "a": {"bias": [0.0, 0.0, 0.0], "weight": [1.0, 0.0, 0.0]},
    "b": {"bias": [0.0, 0.0, 0.0], "weight": [0.0, 1.0, 0.0]},
    "c": {"bias": [0.0, 0.0, 0.0], "weight": [1.0, 1.0, 0.0]},
    "d": {"bias": [0.0, 0.0, 0.0], "weight": [1.0, 0.0, 1.0]},
}


class FakeReader:
    def __init__(self, model_id, tensors):
        self.model_id = model_id
        self.tensors = tensors
        self.close_count = 0

    def close(self):
        self.close_count += 1


class LoraFactorReader(FakeReader):
    """Name intentionally matches the frozen snapshot's delta reader."""


class FakeCube:
    def __init__(self, distances, layer_names, model_ids, audit):
        self.distances = distances
        self.layer_names = layer_names
        self.model_ids = model_ids
        self.audit = audit


def metric_value(left, right, metric):
    if metric == "l2":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    if metric == "cosine":
        dot = sum(a * b for a, b in zip(left, right))
        denom = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    else:
        lmean = sum(left) / len(left)
        rmean = sum(right) / len(right)
        lc = [value - lmean for value in left]
        rc = [value - rmean for value in right]
        dot = sum(a * b for a, b in zip(lc, rc))
        denom = math.sqrt(sum(a * a for a in lc) * sum(b * b for b in rc))
    if denom < 1e-12:
        return 0.0
    return 1.0 - dot / denom


def fake_cube_builder(readers, *, metrics, chunk_size, representation):
    assert tuple(metrics) == DIRECT_METRICS
    assert chunk_size > 0
    keys = sorted(readers[0].tensors)
    distances = {}
    for metric in metrics:
        layers = []
        for key in keys:
            matrix = [[0.0 for _ in readers] for _ in readers]
            for left in range(len(readers)):
                for right in range(left + 1, len(readers)):
                    value = metric_value(
                        readers[left].tensors[key],
                        readers[right].tensors[key],
                        metric,
                    )
                    matrix[left][right] = matrix[right][left] = value
            layers.append(matrix)
        distances[metric] = layers
    audit = {
        "representation": representation,
        "metrics": list(metrics),
        "chunk_size": chunk_size,
        "n_models": len(readers),
        "n_layers": len(keys),
        "model_ids": [reader.model_id for reader in readers],
        "layers": [
            {
                "name": key,
                "shape": [len(readers[0].tensors[key])],
                "numel": len(readers[0].tensors[key]),
                "dtype_by_model": {reader.model_id: "float64" for reader in readers},
            }
            for key in keys
        ],
    }
    for reader in readers:
        reader.close()
    return FakeCube(
        distances=distances,
        layer_names=keys,
        model_ids=[reader.model_id for reader in readers],
        audit=audit,
    )


def adapter_fixture():
    opened = []

    def reader_from_path(path, model_id=None):
        reader = FakeReader(model_id, VECTORS[model_id])
        opened.append((path, reader))
        return reader

    adapter = pinned_snapshot_adapter(
        snapshot_root="/snapshot",
        readers_source=pin("/snapshot/src/readers.py", "snapshot-readers"),
        streaming_source=pin("/snapshot/src/streaming.py", "snapshot-streaming"),
        reader_from_path=reader_from_path,
        build_distance_cube=fake_cube_builder,
        source_pins_verified=True,
    )
    return adapter, opened


class DirectExecutionTests(unittest.TestCase):
    def test_exact_all_node_metrics_order_and_leaf_slice(self):
        adapter, opened = adapter_fixture()
        execution = execute_direct_distances(
            resolved_tree(),
            adapter=adapter,
            chunk_size=2,
        )
        self.assertEqual([path for path, _ in opened], [
            "/frozen/a/model",
            "/frozen/b/model",
            "/frozen/c/model",
            "/frozen/d/model",
        ])
        self.assertTrue(all(reader.close_count == 1 for _, reader in opened))
        # The frozen direct contract uses an unweighted mean over tensor layers;
        # the all-zero bias layer has legacy distance zero.
        self.assertAlmostEqual(
            execution.result("l2").matrix.matrix[0][1], math.sqrt(2.0) / 2.0
        )
        self.assertAlmostEqual(execution.result("cosine").matrix.matrix[0][1], 0.5)
        self.assertAlmostEqual(execution.result("correlation").matrix.matrix[0][1], 0.75)
        self.assertAlmostEqual(
            execution.result("l2").matrix.leaf_sensitivity_matrix[0][1],
            math.sqrt(3.0) / 2.0,
        )
        receipt = execution.receipt.to_dict()
        self.assertEqual(receipt["ordered_node_ids"], ["a", "b", "c", "d"])
        self.assertEqual(receipt["n_internal"], 2)
        self.assertEqual(receipt["n_leaves"], 2)
        self.assertFalse(receipt["neighbor_joining"])
        self.assertIsNone(receipt["tree_builder"])
        self.assertFalse(receipt["internal_nodes_as_tips"])
        self.assertEqual(receipt["layer_aggregation"], "unweighted_mean")
        self.assertEqual([row["node_role"] for row in receipt["checkpoint_scope"]], [
            "internal", "leaf", "internal", "leaf"
        ])

    def test_separate_manifest_entrypoint_revalidates_training_identity(self):
        resolved = resolved_tree()
        execution = execute_manifest_direct_distances(
            topology=resolved.topology,
            checkpoint_manifest=resolved.checkpoint_manifest,
            node_covariates=resolved.node_covariates,
            cohort_key=resolved.cohort_key,
            training_cohort_id=resolved.training_cohort_id,
            resolution_sha256=resolved.resolution_sha256,
            adapter=adapter_fixture()[0],
        )
        self.assertEqual(execution.receipt.payload["checkpoint_set_sha256"],
                         resolved.checkpoint_manifest.checkpoint_set_sha256)
        with self.assertRaisesRegex(ValueError, "training cohort identity drift"):
            execute_manifest_direct_distances(
                topology=resolved.topology,
                checkpoint_manifest=resolved.checkpoint_manifest,
                node_covariates=resolved.node_covariates,
                cohort_key=resolved.cohort_key,
                training_cohort_id="wrong-training-cohort",
                resolution_sha256=resolved.resolution_sha256,
                adapter=adapter_fixture()[0],
            )
        with self.assertRaisesRegex(ValueError, "recomputed manifest scope"):
            execute_manifest_direct_distances(
                topology=resolved.topology,
                checkpoint_manifest=resolved.checkpoint_manifest,
                node_covariates=resolved.node_covariates,
                cohort_key=resolved.cohort_key,
                training_cohort_id=resolved.training_cohort_id,
                resolution_sha256="0" * 64,
                adapter=adapter_fixture()[0],
            )

    def test_fail_closed_adapter_metric_and_cube_identity(self):
        adapter, opened = adapter_fixture()
        with self.assertRaisesRegex(ValueError, "metrics must be exactly"):
            execute_direct_distances(
                resolved_tree(), adapter=adapter, metrics=("l2", "cosine")
            )

        unverified = replace(adapter, source_pins_verified=False)
        with self.assertRaisesRegex(ValueError, "externally verified"):
            execute_direct_distances(resolved_tree(), adapter=unverified)

        def wrong_builder(readers, **kwargs):
            cube = fake_cube_builder(readers, **kwargs)
            cube.model_ids[0], cube.model_ids[1] = cube.model_ids[1], cube.model_ids[0]
            return cube

        wrong = replace(adapter, build_distance_cube=wrong_builder)
        with self.assertRaisesRegex(ValueError, "model IDs differ"):
            execute_direct_distances(resolved_tree(), adapter=wrong)

        invalid_readers = []

        def wrong_open(binding):
            reader = FakeReader("wrong", VECTORS[binding.node_id])
            invalid_readers.append(reader)
            return LoadedTensorReader(binding.node_id, binding.checkpoint, reader)

        wrong = replace(adapter, open_reader=wrong_open)
        with self.assertRaisesRegex(ValueError, "model_id differs"):
            execute_direct_distances(resolved_tree(), adapter=wrong)
        self.assertEqual(invalid_readers[0].close_count, 1)

        delta_readers = []

        def lora_reader_from_path(path, model_id=None):
            reader = LoraFactorReader(model_id, VECTORS[model_id])
            delta_readers.append(reader)
            return reader

        delta_adapter = pinned_snapshot_adapter(
            snapshot_root="/snapshot",
            readers_source=pin("/snapshot/src/readers.py", "snapshot-readers"),
            streaming_source=pin("/snapshot/src/streaming.py", "snapshot-streaming"),
            reader_from_path=lora_reader_from_path,
            build_distance_cube=fake_cube_builder,
            source_pins_verified=True,
        )
        with self.assertRaisesRegex(ValueError, "LoRA-delta reader"):
            execute_direct_distances(resolved_tree(), adapter=delta_adapter)
        self.assertEqual(delta_readers[0].close_count, 1)

    def test_backend_failure_closes_every_opened_reader(self):
        adapter, opened = adapter_fixture()

        def explode(readers, **kwargs):
            raise RuntimeError("backend exploded")

        adapter = replace(adapter, build_distance_cube=explode)
        with self.assertRaisesRegex(RuntimeError, "backend exploded"):
            execute_direct_distances(resolved_tree(), adapter=adapter)
        self.assertTrue(all(reader.close_count == 1 for _, reader in opened))

    def test_plans_seven_exclusive_immutable_artifacts(self):
        execution = execute_direct_distances(
            resolved_tree(), adapter=adapter_fixture()[0]
        )
        plan = plan_direct_artifacts(execution, output_dir="/isolated/direct/tree001")
        self.assertEqual(len(plan.writes), 7)
        self.assertFalse(plan.overwrite)
        self.assertTrue(plan.atomic_set_required)
        self.assertEqual(plan.scheduler_actions, 0)
        self.assertTrue(all(write.exclusive_create for write in plan.writes))
        self.assertEqual(len({write.path for write in plan.writes}), 7)
        l2_values = next(
            write for write in plan.writes if write.path.endswith("l2.all_nodes.matrix.json")
        )
        payload = json.loads(l2_values.content)
        self.assertEqual(payload["ordered_node_ids"], ["a", "b", "c", "d"])
        self.assertEqual(payload["leaf_sensitivity_ids"], ["b", "d"])
        self.assertEqual(len(payload["matrix"]), 4)
        self.assertEqual(len(payload["leaf_sensitivity_matrix"]), 2)
        self.assertEqual(
            l2_values.content_sha256,
            hashlib.sha256(l2_values.content).hexdigest(),
        )

        damaged = replace(l2_values, content=b"{}\n")
        tampered = replace(plan, writes=(damaged, *plan.writes[1:]))
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            verify_direct_artifact_plan(tampered)
        with self.assertRaisesRegex(ValueError, "absolute normalized"):
            plan_direct_artifacts(execution, output_dir="relative/out")


if __name__ == "__main__":
    unittest.main()
