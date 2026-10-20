from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_allnode_inference import checkpoint_fixture, covariates_fixture
from test_allnode_inference_runner import RunnerFixture
from weighttraits.behavior.allnode.audit import (
    MINILM_RECEIPT_SHA256,
    MINILM_SNAPSHOT_SHA256,
)
from weighttraits.behavior.allnode.checkpoints import ArtifactPin
from weighttraits.behavior.allnode.manifest import canonical_sha256
from weighttraits.behavior.allnode.matrices import validate_distance_matrix
from weighttraits.behavior.allnode.semantic_analysis import SEMANTIC_METRIC
from weighttraits.behavior.allnode.semantic_runner import (
    DirectCosineSource,
    NodeEndpointSource,
    SemanticEndpointPlan,
    build_semantic_cohort_plan,
    build_semantic_job_config,
    execute_tree_endpoint_streaming,
    read_semantic_cohort_plan,
    read_semantic_job_config,
    topology_from_manifest,
    validate_direct_cohort_terminal,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def source(node, *, prefix: str, texts: tuple[str, ...]) -> NodeEndpointSource:
    return NodeEndpointSource(
        node_id=node.node_id,
        node_ordinal=node.node_ordinal,
        node_role=node.node_role,
        responses=ArtifactPin(f"/inference/{prefix}/{node.node_id}/responses.jsonl", digest(f"responses/{prefix}/{node.node_id}")),
        terminal_receipt=ArtifactPin(f"/inference/{prefix}/{node.node_id}/terminal_receipt.json", digest(f"receipt/{prefix}/{node.node_id}")),
        request_sha256=digest(f"request/{prefix}/{node.node_id}"),
        endpoint_rows_sha256=digest(f"endpoint/{prefix}/{node.node_id}"),
        texts=texts,
    )


class StreamingTreeEndpointSmokeTests(unittest.TestCase):
    def _run(self, *, cohort_key: str, model_task: str):
        manifest, _ = checkpoint_fixture(cohort_id=f"training-{cohort_key}")
        topology = topology_from_manifest(manifest)
        covariates = covariates_fixture(manifest)
        direct_values = (
            (0.0, 0.2, 0.6),
            (0.2, 0.0, 0.4),
            (0.6, 0.4, 0.0),
        )
        direct_matrix = validate_distance_matrix(
            topology,
            metric="cosine",
            ordered_node_ids=topology.ordered_node_ids,
            matrix=direct_values,
        )
        direct = DirectCosineSource(
            matrix=direct_matrix,
            matrix_artifact=ArtifactPin(f"/direct/{cohort_key}/matrix.json", digest(f"direct-matrix/{cohort_key}")),
            artifact_receipt=ArtifactPin(f"/direct/{cohort_key}/receipt.json", digest(f"direct-receipt/{cohort_key}")),
        )
        endpoint = SemanticEndpointPlan(
            endpoint_id="translation",
            protocol_id="legacy_translation_200x1_greedy",
            probe_id="translation",
            prompt_ids=("p0", "p1"),
            samples_per_prompt=1,
            n_paired_coordinates=2,
            coordinates_sha256=canonical_sha256([
                {"protocol_id": "legacy_translation_200x1_greedy", "probe_id": "translation", "prompt_id": prompt, "sample_id": 0}
                for prompt in ("p0", "p1")
            ]),
        )
        texts = {
            0: (f"{model_task} ancestor zero", f"{model_task} ancestor one"),
            1: (f"{model_task} leaf-left zero", f"{model_task} leaf-left one"),
            2: (f"{model_task} leaf-right zero", f"{model_task} leaf-right one"),
        }

        def encoder(rows):
            # Deterministic nonzero, non-collinear test embeddings.
            return [
                [
                    float(len(value) + 1),
                    float(sum(value.encode()) % 37 + 1),
                    float(value.count("leaf") + 1),
                ]
                for value in rows
            ]

        with tempfile.TemporaryDirectory() as directory:
            endpoint_root = Path(directory).resolve() / "endpoint"
            failed_output = endpoint_root / f"{topology.tree_id}-failed"

            def failing_encoder(_):
                raise RuntimeError("synthetic encoder failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
                execute_tree_endpoint_streaming(
                    cohort_key=cohort_key,
                    endpoint=endpoint,
                    topology=topology,
                    manifest=manifest,
                    node_covariates=covariates,
                    direct=direct,
                    response_loader=lambda ordinal: source(
                        topology.nodes[ordinal], prefix=cohort_key, texts=texts[ordinal]
                    ),
                    encoder=failing_encoder,
                    output_dir=failed_output,
                    encoder_receipt_sha256=MINILM_RECEIPT_SHA256,
                )
            self.assertFalse(failed_output.exists())
            self.assertEqual(
                list(endpoint_root.glob(f".{failed_output.name}.*.tmp")), []
            )

            output = endpoint_root / topology.tree_id
            result = execute_tree_endpoint_streaming(
                cohort_key=cohort_key,
                endpoint=endpoint,
                topology=topology,
                manifest=manifest,
                node_covariates=covariates,
                direct=direct,
                response_loader=lambda ordinal: source(
                    topology.nodes[ordinal], prefix=cohort_key, texts=texts[ordinal]
                ),
                encoder=encoder,
                output_dir=output,
                encoder_receipt_sha256=MINILM_RECEIPT_SHA256,
            )
            self.assertFalse(result.cache_hit)
            self.assertEqual(result.analysis.receipt.n_nodes, 3)
            self.assertEqual(result.analysis.receipt.n_internal, 1)
            self.assertEqual(result.analysis.receipt.n_leaves, 2)
            self.assertEqual(result.analysis.receipt.semantic_metric, SEMANTIC_METRIC)
            self.assertEqual(result.analysis.receipt.n_leaf_sensitivity_pairs, 1)
            self.assertTrue((output / "semantic_matrix.json").is_file())
            self.assertTrue((output / "tree_analysis.json").is_file())
            self.assertTrue((output / "terminal_receipt.json").is_file())
            self.assertEqual(result.terminal_payload["n_internal"], 1)
            self.assertEqual(result.terminal_payload["n_leaves"], 2)

            def forbidden(_):
                raise AssertionError("valid semantic cache must not re-embed text")

            cached = execute_tree_endpoint_streaming(
                cohort_key=cohort_key,
                endpoint=endpoint,
                topology=topology,
                manifest=manifest,
                node_covariates=covariates,
                direct=direct,
                response_loader=lambda ordinal: source(
                    topology.nodes[ordinal], prefix=cohort_key, texts=texts[ordinal]
                ),
                encoder=forbidden,
                output_dir=output,
                encoder_receipt_sha256=MINILM_RECEIPT_SHA256,
            )
            self.assertTrue(cached.cache_hit)
            self.assertEqual(cached.terminal_payload, result.terminal_payload)

    def test_causal_internal_and_leaf_smoke(self):
        self._run(cohort_key="llama_notrans", model_task="causal_lm")

    def test_seq2seq_internal_and_leaf_smoke(self):
        self._run(cohort_key="flan_notrans", model_task="seq2seq")


class SemanticPlanTests(unittest.TestCase):
    def test_production_plan_and_generic_job_config_are_self_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RunnerFixture(Path(directory))
            inference_bytes = fixture.plan_path.read_bytes()
            direct_config_path = (
                fixture.stage
                / "deployments"
                / "allnode_v1"
                / "configs"
                / "jobs"
                / "direct_flan_notrans.json"
            )
            direct_config_path.parent.mkdir(parents=True, exist_ok=True)
            direct_config_path.write_bytes(b'{"direct":"fixture"}\n')
            direct_config_sha = hashlib.sha256(direct_config_path.read_bytes()).hexdigest()
            plan = build_semantic_cohort_plan(
                fixture.plan,
                stage_root=str(fixture.stage),
                inference_plan=ArtifactPin(
                    str(fixture.plan_path), hashlib.sha256(inference_bytes).hexdigest()
                ),
                direct_job_config=ArtifactPin(
                    str(direct_config_path),
                    direct_config_sha,
                ),
                encoder_receipt=ArtifactPin(
                    str(fixture.stage / "contracts" / "minilm.receipt.json"),
                    MINILM_RECEIPT_SHA256,
                ),
                encoder_snapshot=ArtifactPin(
                    str(fixture.stage / "models" / "all-MiniLM-L6-v2"),
                    MINILM_SNAPSHOT_SHA256,
                ),
            )
            self.assertEqual(plan.n_trees if hasattr(plan, "n_trees") else len(plan.tree_scope), 50)
            self.assertEqual(plan.endpoints[0].endpoint_id, "translation")
            self.assertEqual(
                plan.direct_job_config.path,
                str(direct_config_path),
            )
            self.assertEqual(read_semantic_cohort_plan(plan.to_dict(), inference=fixture.plan), plan)
            tampered = plan.to_dict()
            tampered["analysis_contract"]["streaming_unit"] = "all_cohort"
            with self.assertRaises(ValueError):
                read_semantic_cohort_plan(tampered, inference=fixture.plan)

            config = build_semantic_job_config(
                plan, runtime_plan_file_sha256=digest("semantic-plan-file")
            )
            self.assertEqual(read_semantic_job_config(config.to_dict()), config)
            self.assertEqual(config.kind, "semantic_allnode_gpu")
            self.assertEqual(config.logical_id, "semantic_flan_notrans")
            self.assertEqual(config.output_root, plan.output_root)

            direct_result = SimpleNamespace(
                job_config_path=direct_config_path,
                job_config_file_sha256=direct_config_sha,
                plan=SimpleNamespace(
                    cohort_key=plan.cohort_key,
                    training_cohort_id=plan.training_cohort_id,
                ),
                output_dir=Path(plan.direct_output_root),
            )
            with patch(
                "weighttraits.behavior.allnode.semantic_runner.validate_direct_production_terminal",
                return_value=direct_result,
            ) as delegated:
                self.assertIs(validate_direct_cohort_terminal(plan), direct_result)
                delegated.assert_called_once_with(direct_config_path)

            direct_config_path.write_bytes(b'{"direct":"drift"}\n')
            with patch(
                "weighttraits.behavior.allnode.semantic_runner.validate_direct_production_terminal"
            ) as forbidden:
                with self.assertRaisesRegex(ValueError, "job-config file hash drift"):
                    validate_direct_cohort_terminal(plan)
                forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
