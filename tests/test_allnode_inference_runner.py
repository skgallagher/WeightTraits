from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_allnode_production import production_cohorts
from weighttraits.behavior.allnode.checkpoints import ArtifactPin, build_tree_checkpoint_manifest
from weighttraits.behavior.allnode.checkpoint_identity import (
    build_checkpoint_identity,
    build_identity_index,
    validate_checkpoint_identity,
)
from weighttraits.behavior.allnode.inference import PromptPanel
from weighttraits.behavior.allnode.covariates import build_tree_node_covariates
from weighttraits.behavior.allnode.manifest import canonical_sha256
from weighttraits.behavior.allnode.inference_runner import (
    CHECKPOINT_PATH_SENTINEL,
    JOB_CONFIG_SCHEMA,
    PLAN_SCHEMA,
    WeightTraitsInferenceAPI,
    _validate_launcher_environment,
    build_inference_cohort_plan,
    build_inference_job_config,
    execute_filesystem_inference,
    load_inference_job_config,
    read_inference_cohort_plan,
    read_inference_job_config,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_pin(path: Path, payload: bytes) -> ArtifactPin:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ArtifactPin(str(path), hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True)
class FakePrompt:
    probe_id: str
    prompt_id: str
    prompt: str
    prompt_sha256: str
    metadata: dict


class FakeProtocol:
    protocol_id = "legacy_translation_200x1_greedy"
    probe_ids = ("translation",)
    prompt_counts = {"translation": 200}
    samples_per_prompt = 1

    def generation_options(self, *, model_task, sample_id):
        if (model_task, sample_id) != ("seq2seq", 0):
            raise ValueError("unexpected fake protocol coordinate")
        return {
            "sample_id": 0,
            "seed": 42,
            "generation_batch_size": 16,
            "seed_scope": "probe_draw",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "min_new_tokens": 0,
            "max_new_tokens": 128,
            "empty_policy": "preserve",
        }


class FakeRegistry:
    def __init__(self, path: Path, sha256: str):
        self.path = path
        self.sha256 = sha256
        self._protocol = FakeProtocol()

    def validate_model_binding(self, *, model_task, model_id, model_revision):
        if model_task != "seq2seq":
            raise ValueError("model task drift")
        if model_id != "pinned/base-model" or model_revision != "a" * 40:
            raise ValueError("model pin drift")

    def protocol(self, protocol_id):
        if protocol_id != self._protocol.protocol_id:
            raise ValueError("protocol drift")
        return self._protocol


class RunnerFixture:
    def __init__(self, root: Path):
        self.stage = root.resolve() / "downstream"
        self.source_stage = root.resolve() / "frozen"
        self.plan_dir = (
            self.stage
            / "deployments"
            / "allnode_v1"
            / "runtime_inputs"
            / "flan_notrans"
        )
        self.plan_dir.mkdir(parents=True)
        self.prompts = tuple(
            FakePrompt(
                probe_id="translation",
                prompt_id=f"translation-prompt-{index:03d}",
                prompt=f"Translate sentence {index}",
                prompt_sha256=hashlib.sha256(
                    f"Translate sentence {index}".encode()
                ).hexdigest(),
                metadata={"model_task": "seq2seq"},
            )
            for index in range(200)
        )
        self.registry_pin = file_pin(
            self.stage / "contracts" / "behavior_registry.json",
            b'{"fixture":"registry"}\n',
        )
        self.prompt_pin = file_pin(
            self.stage / "prompts" / "translation.jsonl",
            b'{"fixture":"rendered-prompts"}\n',
        )
        self.materialization_pin = file_pin(
            self.stage / "prompts" / "translation.receipt.json",
            b'{"valid":true}\n',
        )
        self.panels = (
            PromptPanel(
                protocol_id="legacy_translation_200x1_greedy",
                probe_id="translation",
                prompt_ids=tuple(prompt.prompt_id for prompt in self.prompts),
                samples_per_prompt=1,
                prompt_artifact=self.prompt_pin,
                materialization_receipt=self.materialization_pin,
            ),
        )
        self.template = {
            "backend": "huggingface",
            "device": "cuda",
            "torch_dtype": "auto",
            "local_files_only": True,
            "tokenizer_model_id": "pinned/base-model",
            "tokenizer_revision": "a" * 40,
            "checkpoint": CHECKPOINT_PATH_SENTINEL,
            "transformers_version": "4.55.0",
            "torch_version": "2.7.0",
            "decode_skip_special_tokens": True,
            "decode_clean_up_tokenization_spaces": False,
            "generation_api": "ordered_probe_draw_batches",
            "generation_seed_api": "transformers.set_seed_once_per_probe_draw",
            "tokenizer_padding_side": "right",
            "tokenizer_truncation": True,
            "tokenizer_max_length": 512,
            "generation_pad_token_id": "tokenizer.pad_token_id",
        }
        self.manifests, self.node_covariates = self._manifests()
        identity_payload = build_identity_index(
            {"flan_notrans": [self.first_checkpoint_identity]},
            cohort_order=("flan_notrans",),
        )
        identity_bytes = (
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        self.identity_index_pin = file_pin(
            self.stage
            / "deployments"
            / "allnode_v1"
            / "runtime_receipts"
            / "ALLNODE_V1_CHECKPOINT_IDENTITIES.json",
            identity_bytes,
        )
        self.plan = build_inference_cohort_plan(
            cohort_key="flan_notrans",
            source_stage_root=str(self.source_stage),
            checkpoint_identity_index=self.identity_index_pin,
            checkpoint_manifests=self.manifests,
            node_covariates=self.node_covariates,
            registry=self.registry_pin,
            panels=self.panels,
            planned_backend_template=self.template,
        )
        self.plan_path = self.plan_dir / "inference_plan.json"
        self.plan_path.write_text(
            json.dumps(
                self.plan.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        self.config = build_inference_job_config(
            self.plan,
            runtime_plan_path=self.plan_path,
            runtime_plan_file_sha256=hashlib.sha256(
                self.plan_path.read_bytes()
            ).hexdigest(),
        )
        self.config_path = (
            self.stage
            / "deployments"
            / "allnode_v1"
            / "configs"
            / "jobs"
            / "infer_flan_notrans.json"
        )
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(
            json.dumps(
                self.config.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    def _manifests(self):
        source = production_cohorts()["flan_notrans"]
        manifests = []
        node_covariates = []
        for tree_index, resolved in enumerate(source):
            tree_id = resolved.topology.tree_id
            shared = {}
            for field in (
                "run_list",
                "ledger",
                "truth_manifest",
                "top_config",
                "training_summary",
                "completion_receipt",
            ):
                pin_root = self.stage if field == "completion_receipt" else self.source_stage
                path = pin_root / "training" / tree_id / f"{field}.json"
                if tree_index == 0:
                    shared[field] = file_pin(path, f"{tree_id}/{field}\n".encode())
                else:
                    shared[field] = ArtifactPin(str(path), digest(f"{tree_id}/{field}"))
            bindings = []
            for binding in resolved.checkpoint_manifest.bindings:
                checkpoint = (
                    self.source_stage
                    / "training"
                    / tree_id
                    / "checkpoints"
                    / binding.node_id
                )
                if tree_index == 0 and binding.node_ordinal == 0:
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "model.safetensors").write_bytes(b"pinned-weights")
                    self.first_checkpoint_identity = build_checkpoint_identity(
                        checkpoint,
                        frozen_stage_root=self.source_stage,
                        stage_manifest_sha256="f" * 64,
                    )
                    checkpoint_pin = ArtifactPin(
                        str(checkpoint),
                        self.first_checkpoint_identity.identity_sha256,
                    )
                else:
                    checkpoint_pin = ArtifactPin(
                        str(checkpoint), digest(f"checkpoint/{tree_id}/{binding.node_id}")
                    )
                bindings.append(
                    replace(
                        binding,
                        checkpoint=checkpoint_pin,
                        **shared,
                    )
                )
            manifest = build_tree_checkpoint_manifest(
                resolved.topology,
                bindings,
                cohort_id=resolved.training_cohort_id,
            )
            manifests.append(manifest)
            node_covariates.append(
                build_tree_node_covariates(
                    resolved.topology,
                    manifest,
                    resolved.node_covariates.nodes,
                    source_manifest=shared["truth_manifest"],
                )
            )
        return tuple(manifests), tuple(node_covariates)

    def api(
        self,
        *,
        planned_drift=False,
        observed_drift=False,
        mutate_checkpoint_after_generate=False,
    ):
        fixture = self

        class FakeBackend:
            load_count = 0
            generation_calls = []

            def __init__(self, binding):
                self.binding = binding

            @classmethod
            def _provenance(cls, binding):
                provenance = dict(fixture.template)
                provenance["checkpoint"] = str(binding.checkpoint)
                return provenance

            @classmethod
            def planned_provenance(cls, binding, *, device, torch_dtype):
                if (device, torch_dtype) != ("cuda", "auto"):
                    raise ValueError("runtime override")
                value = cls._provenance(binding)
                if planned_drift:
                    value["torch_version"] = "different"
                return value

            @classmethod
            def load(cls, binding, *, device, torch_dtype):
                cls.load_count += 1
                return cls(binding)

            def provenance(self):
                value = self._provenance(self.binding)
                if observed_drift:
                    value["transformers_version"] = "different"
                return value

            def generate_batch(self, prompts, *, model_task, generation_options):
                type(self).generation_calls.append(
                    (len(prompts), model_task, dict(generation_options))
                )
                if mutate_checkpoint_after_generate:
                    (
                        Path(fixture.first_checkpoint_identity.path)
                        / "model.safetensors"
                    ).write_bytes(b"mutated-after-backend-open")
                return [f"decoded:{prompt}" for prompt in prompts]

        return (
            WeightTraitsInferenceAPI(
                load_registry=lambda path: FakeRegistry(path, self.registry_pin.sha256),
                load_prompts=lambda path: self.prompts,
                backend_class=FakeBackend,
            ),
            FakeBackend,
        )


class InferenceCohortPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = RunnerFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_641_mapping_and_self_hash(self):
        plan = self.fixture.plan
        self.assertEqual(plan.to_dict()["schema"], PLAN_SCHEMA)
        self.assertEqual(len(plan.checkpoint_manifests), 50)
        self.assertEqual(len(plan.flattened_mapping), 641)
        first, first_manifest, first_covariates, first_binding = plan.select(0)
        last, last_manifest, last_covariates, last_binding = plan.select(640)
        self.assertEqual(
            (first.array_index, first.tree_id, first.node_ordinal, first.node_id),
            (0, "confirm_paper_tree_001", 0, first_binding.node_id),
        )
        self.assertEqual(first_manifest.tree_id, "confirm_paper_tree_001")
        self.assertEqual(first_covariates.tree_id, first_manifest.tree_id)
        self.assertEqual(last.array_index, 640)
        self.assertEqual(last_manifest.tree_id, "confirm_paper_tree_050")
        self.assertEqual(last_covariates.tree_id, last_manifest.tree_id)
        self.assertEqual(last.node_id, last_binding.node_id)
        payload_text = json.dumps(plan.to_dict(), sort_keys=True)
        self.assertNotIn("leaf_ordinal", payload_text)
        self.assertEqual(read_inference_cohort_plan(plan.to_dict()), plan)

        changed = json.loads(json.dumps(plan.to_dict()))
        changed["flattened_mapping"][0]["node_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "flattened mapping differs"):
            read_inference_cohort_plan(changed)

    def test_job_config_pins_raw_plan_and_generic_launcher_environment(self):
        self.assertEqual(self.fixture.config.to_dict()["schema"], JOB_CONFIG_SCHEMA)
        self.assertEqual(
            read_inference_job_config(self.fixture.config.to_dict()),
            self.fixture.config,
        )
        changed = json.loads(json.dumps(self.fixture.config.to_dict()))
        changed["logical_id"] = "infer_wrong"
        with self.assertRaisesRegex(ValueError, "contract drift"):
            read_inference_job_config(changed)
        config, plan, plan_path, config_path, config_file_sha = (
            load_inference_job_config(self.fixture.config_path)
        )
        self.assertEqual(config, self.fixture.config)
        self.assertEqual(plan, self.fixture.plan)
        self.assertEqual(plan_path, self.fixture.plan_path)
        self.assertEqual(config_path, self.fixture.config_path)
        expected_environment = {
            "WT_ALLNODE_COHORT_KEY": config.cohort_key,
            "WT_ALLNODE_JOB_CONFIG": str(config_path),
            "WT_ALLNODE_JOB_CONFIG_SHA256": config_file_sha,
            "WT_ALLNODE_KIND": config.kind,
            "WT_ALLNODE_LOGICAL_ID": config.logical_id,
            "WT_ALLNODE_OUTPUT_DIR": config.output_root,
        }
        with patch.dict(os.environ, expected_environment, clear=False):
            _validate_launcher_environment(
                config,
                config_path=config_path,
                config_file_sha256=config_file_sha,
            )
            os.environ["WT_ALLNODE_KIND"] = "wrong"
            with self.assertRaisesRegex(ValueError, "WT_ALLNODE_KIND drift"):
                _validate_launcher_environment(
                    config,
                    config_path=config_path,
                    config_file_sha256=config_file_sha,
                )

        original_config = self.fixture.config_path.read_bytes()
        wrong_stage = json.loads(json.dumps(self.fixture.config.to_dict()))
        wrong_stage["runtime_plan"]["path"] = (
            "/tmp/other/deployments/allnode_v1/runtime_inputs/"
            "flan_notrans/inference_plan.json"
        )
        wrong_stage["config_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in wrong_stage.items()
                if key != "config_sha256"
            }
        )
        self.fixture.config_path.write_text(
            json.dumps(wrong_stage, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with self.assertRaisesRegex(ValueError, "differs from its job config stage"):
            load_inference_job_config(self.fixture.config_path)
        self.fixture.config_path.write_bytes(original_config)

        original = self.fixture.plan_path.read_bytes()
        self.fixture.plan_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "runtime plan file hash drift"):
            load_inference_job_config(self.fixture.config_path)

    def test_selected_checkpoint_uses_sampled_identity_only(self):
        api, backend = self.fixture.api()
        with (
            patch(
                "weighttraits.behavior.allnode.inference_runner.validate_checkpoint_identity",
                wraps=validate_checkpoint_identity,
            ) as reattest,
            patch(
                "weighttraits.behavior.allnode.inference_runner._sha256_path",
                side_effect=AssertionError("checkpoint directory full hash forbidden"),
            ),
        ):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=api,
            )
        self.assertEqual(backend.load_count, 1)
        self.assertEqual(reattest.call_count, 2)
        for call in reattest.call_args_list:
            self.assertTrue(call.kwargs["reattest_filesystem"])
            self.assertIn("source_authority", call.kwargs)
            self.assertEqual(
                call.args[0].path,
                self.fixture.first_checkpoint_identity.path,
            )

    def test_checkpoint_identity_absence_and_filesystem_drift_fail_before_load(self):
        api, backend = self.fixture.api()
        index_path = Path(self.fixture.identity_index_pin.path)
        index_bytes = index_path.read_bytes()
        index_path.write_bytes(index_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "identity index sha256 mismatch"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=api,
            )
        self.assertEqual(backend.load_count, 0)
        index_path.write_bytes(index_bytes)

        with self.assertRaisesRegex(ValueError, "absent from the identity index"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=1,
                api=api,
            )
        self.assertEqual(backend.load_count, 0)

        checkpoint_file = (
            Path(self.fixture.first_checkpoint_identity.path) / "model.safetensors"
        )
        checkpoint_file.write_bytes(b"tampered-weights")
        with self.assertRaisesRegex(ValueError, "filesystem differs"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=api,
            )
        self.assertEqual(backend.load_count, 0)

    def test_fresh_execution_atomic_publication_then_exact_cache(self):
        api, backend = self.fixture.api()
        fresh = execute_filesystem_inference(
            self.fixture.plan_path,
            array_index=0,
            api=api,
        )
        self.assertFalse(fresh.cache_hit)
        self.assertEqual(backend.load_count, 1)
        self.assertEqual(len(backend.generation_calls), 1)
        self.assertEqual(backend.generation_calls[0][0], 200)
        self.assertTrue(fresh.output_path.is_file())
        self.assertTrue(fresh.receipt_path.is_file())
        self.assertEqual(
            fresh.output_path,
            self.fixture.stage
            / "outputs"
            / "allnode_v1"
            / "inference"
            / "flan_notrans"
            / "confirm_paper_tree_001"
            / fresh.mapping.node_id
            / "responses.jsonl",
        )
        self.assertEqual(len(fresh.output_path.read_text().splitlines()), 200)
        self.assertNotIn("leaf_ordinal", json.dumps(fresh.status_dict()))

        cached = execute_filesystem_inference(
            self.fixture.plan_path,
            array_index=0,
            api=api,
        )
        self.assertTrue(cached.cache_hit)
        self.assertEqual(backend.load_count, 1)
        self.assertEqual(len(backend.generation_calls), 1)
        self.assertEqual(cached.runtime_result.writes, ())

    def test_post_backend_checkpoint_drift_fails_before_any_publication(self):
        api, backend = self.fixture.api(mutate_checkpoint_after_generate=True)
        with self.assertRaisesRegex(ValueError, "filesystem differs"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=api,
            )
        self.assertEqual(backend.load_count, 1)
        mapping, _manifest, _covariates, _binding = self.fixture.plan.select(0)
        output_dir = (
            self.fixture.stage
            / "outputs/allnode_v1/inference/flan_notrans"
            / mapping.tree_id
            / mapping.node_id
        )
        self.assertFalse((output_dir / "responses.jsonl").exists())
        self.assertFalse((output_dir / "terminal_receipt.json").exists())

    def test_partial_cache_is_refused_without_backend_load(self):
        api, backend = self.fixture.api()
        mapping, _manifest, _covariates, _binding = self.fixture.plan.select(0)
        output = (
            self.fixture.stage
            / "outputs/allnode_v1/inference/flan_notrans"
            / mapping.tree_id
            / mapping.node_id
            / "responses.jsonl"
        )
        output.parent.mkdir(parents=True)
        output.write_text("occupied\n")
        with self.assertRaisesRegex(FileExistsError, "partial inference cache"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=api,
            )
        self.assertEqual(backend.load_count, 0)

    def test_tampered_complete_cache_is_refused_without_regeneration(self):
        api, backend = self.fixture.api()
        fresh = execute_filesystem_inference(
            self.fixture.plan_path,
            array_index=0,
            api=api,
        )
        self.assertEqual(backend.load_count, 1)
        payload = fresh.output_path.read_bytes()
        fresh.output_path.write_bytes(payload.replace(b"decoded:", b"changed:", 1))
        with self.assertRaisesRegex(ValueError, "artifact pin differs"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=api,
            )
        self.assertEqual(backend.load_count, 1)
        self.assertEqual(len(backend.generation_calls), 1)

    def test_prompt_pin_and_backend_provenance_drift_fail_before_publish(self):
        api, backend = self.fixture.api()
        self.fixture.prompt_pin_path = Path(self.fixture.prompt_pin.path)
        self.fixture.prompt_pin_path.write_bytes(b"tampered\n")
        with self.assertRaisesRegex(ValueError, "prompt artifact sha256 mismatch"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=api,
            )
        self.assertEqual(backend.load_count, 0)

        # Restore exact prompt bytes, then prove runtime-version drift fails
        # before checkpoint loading or output publication.
        self.fixture.prompt_pin_path.write_bytes(b'{"fixture":"rendered-prompts"}\n')
        drift_api, drift_backend = self.fixture.api(planned_drift=True)
        with self.assertRaisesRegex(ValueError, "runtime provenance differs"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=drift_api,
            )
        self.assertEqual(drift_backend.load_count, 0)

    def test_observed_backend_drift_and_array_bounds_fail_closed(self):
        drift_api, drift_backend = self.fixture.api(observed_drift=True)
        with self.assertRaisesRegex(ValueError, "loaded HuggingFace backend provenance"):
            execute_filesystem_inference(
                self.fixture.plan_path,
                array_index=0,
                api=drift_api,
            )
        self.assertEqual(drift_backend.load_count, 1)
        output_root = self.fixture.stage / "outputs"
        self.assertFalse(output_root.exists())

        api, backend = self.fixture.api()
        for index in (-1, 641):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "array_index"):
                    execute_filesystem_inference(
                        self.fixture.plan_path,
                        array_index=index,
                        api=api,
                    )
        self.assertEqual(backend.load_count, 0)

    def test_slurm_template_is_inert_and_exact_route(self):
        wrapper = (
            Path(__file__).parents[1]
            / "scripts"
            / "slurm"
            / "allnode_inference_gpu.sbatch"
        ).read_text()
        for line in (
            "#SBATCH --account=statds",
            "#SBATCH --partition=all",
            "#SBATCH --qos=normal",
            "#SBATCH --nodelist=n03",
            "#SBATCH --exclude=n01",
            "#SBATCH --no-requeue",
            "#SBATCH --gres=gpu:nvidia-l40:1",
            "#SBATCH --array=0-640%4",
        ):
            self.assertIn(line, wrapper)
        self.assertIn("HF_HUB_OFFLINE=1", wrapper)
        self.assertIn("weighttraits_allnode.inference_runner", wrapper)  # deployed wrapper, verbatim
        self.assertIn(
            "STAGE_ROOT=/home/export/sgallagh/WeightTraits-downstream-20260831-native",
            wrapper,
        )
        self.assertIn("#SBATCH --time=12:00:00", wrapper)
        for required in (
            "WT_ALLNODE_JOB_CONFIG",
            "WT_ALLNODE_JOB_CONFIG_SHA256",
            "WT_ALLNODE_KIND",
            "WT_ALLNODE_LOGICAL_ID",
            "WT_ALLNODE_OUTPUT_DIR",
            '--config "${WT_ALLNODE_JOB_CONFIG}"',
        ):
            self.assertIn(required, wrapper)
        self.assertNotIn("WT_ALLNODE_INFERENCE_PLAN_SHA256", wrapper)
        self.assertNotIn("sbatch ", wrapper)
        for forbidden in (
            "OVERRIDE_MAX_STEPS",
            "MAX_TRAIN_SAMPLES",
            "MAX_EVAL_SAMPLES",
            "ROW_END",
            "leaf_ordinal",
        ):
            self.assertNotIn(forbidden, wrapper)


if __name__ == "__main__":
    unittest.main()
