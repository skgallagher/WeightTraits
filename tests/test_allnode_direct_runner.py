from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weighttraits.behavior.allnode.checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    build_tree_checkpoint_manifest,
)
from weighttraits.behavior.allnode.covariates import NodeCovariate, build_tree_node_covariates
from weighttraits.behavior.allnode.direct_runner import (
    COHORT_TERMINAL_SCHEMA,
    WeightTraitsDirectAPI,
    build_direct_cohort_plan,
    build_direct_job_config,
    execute_filesystem_direct,
    _reattest_tree_checkpoint_identities,
    read_direct_cohort_plan,
    validate_direct_cohort_terminal,
)
from weighttraits.behavior.allnode.direct_compatibility import build_leaf_compatibility_spec
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology, canonical_sha256
from weighttraits.behavior.allnode.production import PRODUCTION_TREE_IDS
from weighttraits.behavior.allnode.resolver import ResolvedAllNodeTree
from weighttraits.behavior.allnode.symlink_inventory import (
    SourcePathAuthoritySpec,
    SourcePathMapping,
    load_source_path_authority,
)


REVISION = "a" * 40


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def pin(path, label):
    return ArtifactPin(str(path), digest(label))


class FakeReader:
    def __init__(self, model_id):
        self.model_id = model_id
        number = int(model_id[1:]) + (0 if model_id.startswith("i") else 17)
        self.vector = [1.0 + number, 2.0 + number % 3, 4.0 + number % 5]
        self.close_count = 0

    def close(self):
        self.close_count += 1


class FakeCube:
    def __init__(self, readers, metrics, chunk_size):
        self.model_ids = [reader.model_id for reader in readers]
        self.layer_names = ["weight"]
        self.distances = {
            metric: [[
                [_distance(left.vector, right.vector, metric) for right in readers]
                for left in readers
            ]]
            for metric in metrics
        }
        self.audit = {
            "representation": "full_weight",
            "metrics": list(metrics),
            "chunk_size": chunk_size,
            "n_models": len(readers),
            "n_layers": 1,
            "model_ids": self.model_ids,
            "layers": [
                {
                    "name": "weight",
                    "shape": [3],
                    "numel": 3,
                    "dtype_by_model": {
                        reader.model_id: "float64" for reader in readers
                    },
                }
            ],
        }


def _distance(left, right, metric):
    if left == right:
        return 0.0
    if metric == "l2":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    if metric == "cosine":
        numerator = sum(a * b for a, b in zip(left, right))
        denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    else:
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        left_centered = [value - left_mean for value in left]
        right_centered = [value - right_mean for value in right]
        numerator = sum(a * b for a, b in zip(left_centered, right_centered))
        denominator = math.sqrt(
            sum(value * value for value in left_centered)
            * sum(value * value for value in right_centered)
        )
    return 0.0 if denominator < 1e-12 else 1.0 - numerator / denominator


class DirectRunnerFixture:
    def __init__(self, root: Path, *, external_authority: bool = False):
        self.root = root.resolve()
        self.stage = self.root / "stage"
        self.frozen = self.root / "frozen"
        self.frozen.mkdir()
        self.source_authority = None
        self.checkpoint_root = self.frozen / "llama_full"
        if external_authority:
            external = self.root / "external/llama_full"
            external.mkdir(parents=True)
            self.checkpoint_root.symlink_to(external, target_is_directory=True)
            inventory = self.frozen / ".stage_provenance/SYMLINKS.txt"
            inventory.parent.mkdir()
            inventory.write_text(
                f"llama_full -> {external}\n",
                encoding="utf-8",
            )
            manifest = self.frozen / ".stage_provenance/STAGE_SHA256SUMS"
            manifest.write_text(
                f"{_sha_file(inventory)}  .stage_provenance/SYMLINKS.txt\n",
                encoding="utf-8",
            )
            self.source_authority = load_source_path_authority(
                SourcePathAuthoritySpec(
                    stage_root=str(self.frozen),
                    stage_manifest=ArtifactPin(str(manifest), _sha_file(manifest)),
                    symlink_inventory=ArtifactPin(
                        str(inventory), _sha_file(inventory)
                    ),
                    mappings=(
                        SourcePathMapping(str(self.checkpoint_root), str(external)),
                    ),
                )
            )
            self.checkpoint_root = external
        self.snapshot = self.stage / "src" / "weighttraits" / "distances"
        self.deployment = self.stage / "deployments" / "allnode_v1"
        self.jobs = self.deployment / "configs" / "jobs"
        self.snapshot.mkdir(parents=True)
        self.jobs.mkdir(parents=True)
        self.readers = self.snapshot / "readers.py"
        self.streaming = self.snapshot / "streaming.py"
        self.readers.write_text("# pinned readers\n", encoding="utf-8")
        self.streaming.write_text("# pinned streaming\n", encoding="utf-8")
        self.identity = self.stage / "runtime_receipts" / "checkpoint_index.json"
        self.identity.parent.mkdir()
        self.identity.write_text("{}", encoding="utf-8")
        self.reason = self.stage / "runtime_receipts" / "no_legacy_baseline.txt"
        self.reason.write_text("test fixture has no defensible prior baseline\n", encoding="utf-8")
        self.trees = tuple(self._tree(index) for index in range(50))
        self.lookup = {
            binding.checkpoint.path: binding.checkpoint.sha256
            for tree in self.trees
            for binding in tree.checkpoint_manifest.bindings
        }
        self.plan = build_direct_cohort_plan(
            cohort_key="llama_full",
            source_stage_root=str(self.frozen),
            output_dir=str(
                self.stage / "outputs" / "allnode_v1" / "direct" / "llama_full"
            ),
            snapshot_root=str(self.stage / "src" / "weighttraits"),
            readers_source=ArtifactPin(str(self.readers), _sha_file(self.readers)),
            streaming_source=ArtifactPin(str(self.streaming), _sha_file(self.streaming)),
            checkpoint_identity_index=ArtifactPin(
                str(self.identity), _sha_file(self.identity)
            ),
            source_path_authority=self.source_authority,
            leaf_compatibility=build_leaf_compatibility_spec(
                comparison_available=False,
                unavailable_reason=ArtifactPin(
                    str(self.reason), _sha_file(self.reason)
                ),
            ),
            trees=self.trees,
            chunk_size=7,
        )
        self.plan_path = (
            self.deployment
            / "runtime_inputs"
            / "llama_full"
            / "direct_plan.json"
        )
        self.plan_path.parent.mkdir(parents=True)
        self.plan_path.write_text(
            json.dumps(
                self.plan.to_dict(), sort_keys=True, separators=(",", ":")
            ) + "\n",
            encoding="utf-8",
        )
        self.job_config = build_direct_job_config(
            cohort_key="llama_full",
            output_dir=self.plan.output_dir,
            direct_plan=ArtifactPin(str(self.plan_path), _sha_file(self.plan_path)),
        )
        self.config = self.jobs / "direct_llama_full.json"
        self.config.write_text(
            json.dumps(
                self.job_config.to_dict(), sort_keys=True, separators=(",", ":")
            ) + "\n",
            encoding="utf-8",
        )
        self.build_calls = 0
        self.opened = []

    def _tree(self, tree_index):
        tree_id = PRODUCTION_TREE_IDS[tree_index]
        n_nodes = 13 if tree_index < 41 else 12
        n_internal = 6 if tree_index < 26 else 5
        nodes = []
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
        topology = build_tree_topology(tree_id, nodes)
        if self.source_authority is not None:
            for node in topology.nodes:
                (
                    self.checkpoint_root / tree_id / node.node_id / "model"
                ).mkdir(parents=True)
        shared = {
            "run_list": pin(self.frozen / tree_id / "runs.jsonl", f"runs/{tree_id}"),
            "ledger": pin(self.frozen / tree_id / "ledger.jsonl", f"ledger/{tree_id}"),
            "truth_manifest": pin(self.frozen / tree_id / "truth.jsonl", f"truth/{tree_id}"),
            "top_config": pin(self.frozen / "config.json", "config"),
            "training_summary": pin(self.frozen / "summary.json", "summary"),
            "completion_receipt": pin(self.stage / "completion.json", "completion"),
        }
        bindings = [
            NodeCheckpointBinding(
                population=POPULATION,
                internal_node_treatment=INTERNAL_NODE_TREATMENT,
                cohort_id="shared-llama-training",
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
                    self.checkpoint_root / tree_id / node.node_id / "model",
                    f"checkpoint/{tree_id}/{node.node_id}",
                ),
                base_model_id="pinned/base-model",
                base_model_revision=REVISION,
                **shared,
            )
            for node in topology.nodes
        ]
        manifest = build_tree_checkpoint_manifest(
            topology, bindings, cohort_id="shared-llama-training"
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
                    f"dataset-{node.node_ordinal % 4}",
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

    def api(self):
        def reader_from_path(path, model_id=None):
            reader = FakeReader(model_id)
            self.opened.append(reader)
            return reader

        def build_distance_cube(readers, *, metrics, chunk_size, representation):
            self.build_calls += 1
            self.assertions = (tuple(metrics), chunk_size, representation)
            cube = FakeCube(readers, metrics, chunk_size)
            for reader in readers:
                reader.close()
            return cube

        return WeightTraitsDirectAPI(reader_from_path, build_distance_cube)

    def identity_reader(self, payload):
        self.identity_payload = payload
        return self.lookup


def _sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DirectRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = DirectRunnerFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_fresh_atomic_cohort_and_exact_terminal_cache(self):
        with patch(
            "weighttraits.behavior.allnode.direct_runner._reattest_tree_checkpoint_identities",
            wraps=_reattest_tree_checkpoint_identities,
        ) as reattest:
            result = execute_filesystem_direct(
                self.fixture.config,
                api=self.fixture.api(),
                identity_index_reader=self.fixture.identity_reader,
            )
        self.assertFalse(result.cache_hit)
        self.assertEqual(reattest.call_count, 100)
        self.assertEqual(self.fixture.build_calls, 50)
        self.assertEqual(self.fixture.assertions, (("l2", "cosine", "correlation"), 7, "full_weight"))
        self.assertTrue(all(reader.close_count == 1 for reader in self.fixture.opened))
        self.assertEqual(result.terminal_receipt["schema"], COHORT_TERMINAL_SCHEMA)
        self.assertEqual(
            (result.terminal_receipt["n_trees"], result.terminal_receipt["n_nodes"]),
            (50, 641),
        )
        first = Path(self.fixture.plan.output_dir) / PRODUCTION_TREE_IDS[0]
        values = json.loads(
            (first / f"{PRODUCTION_TREE_IDS[0]}.cosine.all_nodes.matrix.json").read_text()
        )
        indices = values["leaf_sensitivity_indices"]
        expected_leaf = [
            [values["matrix"][left][right] for right in indices] for left in indices
        ]
        self.assertEqual(values["leaf_sensitivity_matrix"], expected_leaf)

        cached = execute_filesystem_direct(
            self.fixture.config,
            api=None,
            identity_index_reader=self.fixture.identity_reader,
        )
        self.assertTrue(cached.cache_hit)
        self.assertEqual(self.fixture.build_calls, 50)
        self.assertEqual(
            cached.terminal_receipt["cohort_terminal_sha256"],
            result.terminal_receipt["cohort_terminal_sha256"],
        )
        validated = validate_direct_cohort_terminal(self.fixture.config)
        self.assertEqual(
            validated.terminal_receipt["cohort_terminal_sha256"],
            result.terminal_receipt["cohort_terminal_sha256"],
        )

        matrix = first / f"{PRODUCTION_TREE_IDS[0]}.l2.all_nodes.matrix.json"
        matrix.write_bytes(matrix.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "digest/size drift"):
            execute_filesystem_direct(
                self.fixture.config,
                identity_index_reader=self.fixture.identity_reader,
            )

    def test_plan_round_trip_tamper_and_identity_mismatch(self):
        self.assertEqual(
            read_direct_cohort_plan(self.fixture.plan.to_dict()), self.fixture.plan
        )
        tampered = json.loads(json.dumps(self.fixture.plan.to_dict()))
        tampered["metrics"] = ["cosine"]
        with self.assertRaises(ValueError):
            read_direct_cohort_plan(tampered)

        bad = dict(self.fixture.lookup)
        first_path = next(iter(bad))
        bad[first_path] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity index mismatch"):
            execute_filesystem_direct(
                self.fixture.config,
                api=self.fixture.api(),
                identity_index_reader=lambda payload: bad,
            )
        self.assertFalse(Path(self.fixture.plan.output_dir).exists())

    def test_external_physical_checkpoints_require_and_bind_exact_authority(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = DirectRunnerFixture(
                Path(raw), external_authority=True
            )
            self.assertIsNotNone(fixture.plan.source_path_authority)
            self.assertEqual(
                read_direct_cohort_plan(fixture.plan.to_dict()), fixture.plan
            )
            with self.assertRaisesRegex(ValueError, "without authority"):
                build_direct_cohort_plan(
                    cohort_key="llama_full",
                    source_stage_root=str(fixture.frozen),
                    output_dir=fixture.plan.output_dir,
                    snapshot_root=str(fixture.stage / "src" / "weighttraits"),
                    readers_source=fixture.plan.readers_source,
                    streaming_source=fixture.plan.streaming_source,
                    checkpoint_identity_index=fixture.plan.checkpoint_identity_index,
                    leaf_compatibility=fixture.plan.leaf_compatibility,
                    trees=fixture.trees,
                    chunk_size=7,
                )

    def test_post_tree_identity_failure_prevents_first_tree_publication(self):
        calls = 0

        def fail_second_boundary(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("post-tree checkpoint drift")
            return "a" * 64

        with patch(
            "weighttraits.behavior.allnode.direct_runner._reattest_tree_checkpoint_identities",
            side_effect=fail_second_boundary,
        ):
            with self.assertRaisesRegex(ValueError, "post-tree checkpoint drift"):
                execute_filesystem_direct(
                    self.fixture.config,
                    api=self.fixture.api(),
                    identity_index_reader=self.fixture.identity_reader,
                )
        self.assertEqual(calls, 2)
        self.assertEqual(self.fixture.build_calls, 1)
        first_tree = Path(self.fixture.plan.output_dir) / PRODUCTION_TREE_IDS[0]
        self.assertFalse(first_tree.exists())


if __name__ == "__main__":
    unittest.main()
