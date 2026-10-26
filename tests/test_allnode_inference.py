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
from weighttraits.behavior.allnode.inference import (
    FROZEN_COHORT_ENDPOINTS,
    FROZEN_COHORT_RESPONSE_COUNTS,
    PromptPanel,
    ResponseRow,
    build_inference_request,
    build_terminal_response_receipt,
    read_inference_request,
    validate_terminal_response_receipt,
)
from weighttraits.behavior.allnode.covariates import NodeCovariate, build_tree_node_covariates
from weighttraits.behavior.allnode.manifest import NodeSpec, build_tree_topology


REVISION = "a" * 40


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path: str, label: str) -> ArtifactPin:
    return ArtifactPin(path, digest(label))


def checkpoint_fixture(cohort_id: str = "ordinary-llama-full"):
    topology = build_tree_topology(
        "tree001",
        [
            NodeSpec(0, 0, "a", "root", 1, False),
            NodeSpec(1, 1, "b", "a", 2, True),
            NodeSpec(2, 2, "c", "a", 2, True),
        ],
    )
    bindings = []
    for node in topology.nodes:
        bindings.append(
            NodeCheckpointBinding(
                population=POPULATION,
                internal_node_treatment=INTERNAL_NODE_TREATMENT,
                cohort_id=cohort_id,
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
                checkpoint=pin(
                    f"/frozen/{cohort_id}/{topology.tree_id}/{node.node_id}/model",
                    f"checkpoint/{cohort_id}/{node.node_id}",
                ),
                base_model_id="pinned/base-model",
                base_model_revision=REVISION,
                run_list=pin("/frozen/tree001.runs.jsonl", "runs"),
                ledger=pin("/frozen/tree001.ledger.jsonl", "ledger"),
                truth_manifest=pin("/frozen/tree001.truth.jsonl", "truth"),
                top_config=pin("/frozen/config.json", "config"),
                training_summary=pin("/frozen/summary.json", "summary"),
                completion_receipt=pin("/frozen/completion.json", "completion"),
            )
        )
    manifest = build_tree_checkpoint_manifest(
        topology, bindings, cohort_id=cohort_id
    )
    return manifest, bindings


def covariates_fixture(manifest):
    topology = build_tree_topology(
        manifest.tree_id,
        [
            NodeSpec(
                binding.node_ordinal,
                binding.array_index,
                binding.node_id,
                binding.parent_id,
                binding.depth,
                binding.node_role == "leaf",
            )
            for binding in manifest.bindings
        ],
        root_id=manifest.root_id,
    )
    paths = {}
    for node in topology.nodes:
        paths[node.node_id] = (
            (topology.root_id, node.node_id)
            if node.parent_id == topology.root_id
            else (*paths[node.parent_id], node.node_id)
        )
    return build_tree_node_covariates(
        topology,
        manifest,
        [
            NodeCovariate(
                node_id=node.node_id,
                parent_id=node.parent_id,
                depth=node.depth,
                path=paths[node.node_id],
                dataset_id=f"dataset-{node.node_ordinal}",
                task_family=f"family-{node.node_ordinal % 2}",
                role=node.node_role,
            )
            for node in topology.nodes
        ],
        source_manifest=manifest.bindings[0].truth_manifest,
    )


def panels(cohort_key: str):
    result = []
    for endpoint in FROZEN_COHORT_ENDPOINTS[cohort_key]:
        result.append(
            PromptPanel(
                protocol_id=endpoint.protocol_id,
                probe_id=endpoint.probe_id,
                prompt_ids=tuple(
                    f"{endpoint.probe_id}-prompt-{index:03d}"
                    for index in range(endpoint.n_prompts)
                ),
                samples_per_prompt=endpoint.samples_per_prompt,
                prompt_artifact=pin(
                    f"/prompts/{cohort_key}/{endpoint.protocol_id}.jsonl",
                    f"prompts/{cohort_key}/{endpoint.protocol_id}",
                ),
                materialization_receipt=pin(
                    f"/prompts/{cohort_key}/{endpoint.protocol_id}.receipt.json",
                    f"materialization/{cohort_key}/{endpoint.protocol_id}",
                ),
            )
        )
    return result


def backend_provenance(binding, model_task: str):
    return {
        "backend": "huggingface",
        "device": "cuda",
        "torch_dtype": "auto",
        "local_files_only": True,
        "tokenizer_model_id": binding.base_model_id,
        "tokenizer_revision": binding.base_model_revision,
        "checkpoint": binding.checkpoint.path,
        "transformers_version": "4.57.1",
        "torch_version": "2.8.0+cu128",
        "decode_skip_special_tokens": True,
        "decode_clean_up_tokenization_spaces": False,
        "generation_api": "ordered_probe_draw_batches",
        "generation_seed_api": "transformers.set_seed_once_per_probe_draw",
        "tokenizer_padding_side": "left" if model_task == "causal_lm" else "right",
        "tokenizer_truncation": True,
        "tokenizer_max_length": 512,
        "generation_pad_token_id": "tokenizer.pad_token_id",
    }


def request(cohort_key: str = "llama_full", *, node_ordinal: int = 0):
    manifest, bindings = checkpoint_fixture()
    model_task = "causal_lm" if cohort_key.startswith("llama_") else "seq2seq"
    return build_inference_request(
        bindings[node_ordinal],
        manifest,
        cohort_key=cohort_key,
        registry=pin("/contracts/behavior-registry.json", "registry"),
        panels=panels(cohort_key),
        planned_backend_provenance=backend_provenance(
            bindings[node_ordinal], model_task
        ),
        node_covariates=covariates_fixture(manifest),
    )


def response_rows(active_request):
    return [
        ResponseRow(
            request_sha256=active_request.request_sha256,
            cohort_id=active_request.cohort_id,
            tree_id=active_request.tree_id,
            node_id=active_request.node_id,
            protocol_id=coordinate.protocol_id,
            probe_id=coordinate.probe_id,
            prompt_id=coordinate.prompt_id,
            sample_id=coordinate.sample_id,
            response_sha256=digest(
                "/".join(
                    (
                        coordinate.protocol_id,
                        coordinate.probe_id,
                        coordinate.prompt_id,
                        str(coordinate.sample_id),
                    )
                )
            ),
            completed=True,
        )
        for coordinate in active_request.coordinates()
    ]


class InferenceRequestTests(unittest.TestCase):
    def test_internal_node_request_is_ancestor_aware_and_never_fakes_leaf_ordinal(self) -> None:
        active = request("llama_full", node_ordinal=0)
        payload = active.to_dict()
        self.assertEqual(active.node_role, "internal")
        self.assertEqual(active.parent_id, "root")
        self.assertEqual(active.population, "all_trained_nodes")
        self.assertEqual(active.internal_node_treatment, "observed_ancestors")
        self.assertEqual(active.n_expected_responses, 1490)
        self.assertEqual((active.device, active.torch_dtype), ("cuda", "auto"))
        self.assertEqual(active.execution_overrides, {})
        self.assertNotIn("leaf_ordinal", payload)
        self.assertNotIn("leaf_ordinal", payload["checkpoint_binding"])
        self.assertEqual(read_inference_request(payload).to_dict(), payload)

    def test_exact_four_cohort_grids_and_deterministic_counts(self) -> None:
        self.assertEqual(
            dict(FROZEN_COHORT_RESPONSE_COUNTS),
            {
                "llama_full": 1490,
                "llama_notrans": 200,
                "flan_full": 1400,
                "flan_notrans": 200,
            },
        )
        for cohort_key, expected in FROZEN_COHORT_RESPONSE_COUNTS.items():
            with self.subTest(cohort_key=cohort_key):
                active = request(cohort_key)
                coordinates = active.coordinates()
                self.assertEqual(len(coordinates), expected)
                self.assertEqual(len(coordinates), len(set(coordinates)))
                self.assertEqual(
                    coordinates[0].protocol_id,
                    "legacy_translation_200x1_greedy",
                )
                self.assertEqual(coordinates[0].sample_id, 0)
                self.assertEqual(
                    active.coordinates_sha256,
                    read_inference_request(active.to_dict()).coordinates_sha256,
                )

    def test_exact_registry_identifiers_and_seed_once_generation_groups(self) -> None:
        active = request("llama_full")
        self.assertEqual(
            [
                (endpoint.protocol_id, endpoint.probe_id)
                for endpoint in FROZEN_COHORT_ENDPOINTS["llama_full"]
            ],
            [
                ("legacy_translation_200x1_greedy", "translation"),
                ("modern_mc_100x3_sampled", "hellaswag"),
                ("modern_mc_100x3_sampled", "arc_challenge"),
                ("modern_mc_100x3_sampled", "mmlu"),
                ("modern_mc_100x3_sampled", "truthfulqa"),
                ("dolly_open_ended_30x3_sampled", "dolly_open_ended"),
            ],
        )
        self.assertEqual(active.model_task, "causal_lm")
        self.assertEqual(len(active.generation_groups), 16)
        translation = active.generation_groups[0]
        self.assertEqual(len(translation.prompt_ids), 200)
        self.assertEqual(
            translation.generation_options,
            {
                "sample_id": 0,
                "seed": 42,
                "generation_batch_size": 16,
                "seed_scope": "probe_draw",
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "min_new_tokens": 0,
                "max_new_tokens": 64,
                "empty_policy": "preserve",
            },
        )
        self.assertEqual(
            [group.sample_id for group in active.generation_groups[1:4]],
            [0, 1, 2],
        )
        self.assertEqual(
            [group.generation_options["seed"] for group in active.generation_groups[1:4]],
            [42, 43, 44],
        )
        self.assertTrue(active.generation_groups[1].generation_options["do_sample"])
        self.assertEqual(active.generation_groups[-1].generation_options["max_new_tokens"], 96)

        flan = request("flan_full")
        self.assertEqual(flan.model_task, "seq2seq")
        self.assertEqual(
            flan.generation_groups[0].generation_options["max_new_tokens"], 128
        )
        self.assertEqual(
            flan.planned_backend_provenance["tokenizer_padding_side"], "right"
        )

    def test_rejects_panel_count_pin_and_manifest_binding_drift(self) -> None:
        manifest, bindings = checkpoint_fixture()
        wrong_panels = panels("llama_full")
        wrong_panels[0] = replace(
            wrong_panels[0], prompt_ids=wrong_panels[0].prompt_ids[:-1]
        )
        with self.assertRaisesRegex(ValueError, "frozen endpoint contract"):
            build_inference_request(
                bindings[0],
                manifest,
                cohort_key="llama_full",
                registry=pin("/contracts/registry.json", "registry"),
                panels=wrong_panels,
                planned_backend_provenance=backend_provenance(
                    bindings[0], "causal_lm"
                ),
                node_covariates=covariates_fixture(manifest),
            )

        changed_pins = panels("llama_full")
        changed_pins[2] = replace(
            changed_pins[2], prompt_artifact=pin("/prompts/other.jsonl", "other")
        )
        with self.assertRaisesRegex(ValueError, "pins differ across probes"):
            build_inference_request(
                bindings[0],
                manifest,
                cohort_key="llama_full",
                registry=pin("/contracts/registry.json", "registry"),
                panels=changed_pins,
                planned_backend_provenance=backend_provenance(
                    bindings[0], "causal_lm"
                ),
                node_covariates=covariates_fixture(manifest),
            )

        with self.assertRaisesRegex(ValueError, "differs from its checkpoint manifest"):
            build_inference_request(
                replace(bindings[0], checkpoint=pin("/frozen/other/model", "other")),
                manifest,
                cohort_key="llama_full",
                registry=pin("/contracts/registry.json", "registry"),
                panels=panels("llama_full"),
                planned_backend_provenance=backend_provenance(
                    bindings[0], "causal_lm"
                ),
                node_covariates=covariates_fixture(manifest),
            )

    def test_saved_request_rejects_override_leaf_and_runtime_drift(self) -> None:
        active = request("flan_notrans")
        for field, value in (
            ("execution_overrides", {"max_rows": 1}),
            ("device", "cpu"),
            ("torch_dtype", "float16"),
        ):
            payload = active.to_dict()
            payload[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "differs from its recomputed contract"):
                    read_inference_request(payload)
        payload = active.to_dict()
        payload["leaf_ordinal"] = 0
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            read_inference_request(payload)

        payload = active.to_dict()
        payload["generation_groups"][0]["generation_options"]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "seed-once"):
            read_inference_request(payload)

        manifest, bindings = checkpoint_fixture()
        backend = backend_provenance(bindings[0], "seq2seq")
        backend["local_files_only"] = False
        with self.assertRaisesRegex(ValueError, "local_files_only"):
            build_inference_request(
                bindings[0],
                manifest,
                cohort_key="flan_notrans",
                registry=pin("/contracts/registry.json", "registry"),
                panels=panels("flan_notrans"),
                planned_backend_provenance=backend,
                node_covariates=covariates_fixture(manifest),
            )


class ResponseReceiptTests(unittest.TestCase):
    def test_terminal_receipt_requires_exact_grid_and_hash_pins(self) -> None:
        active = request("flan_notrans")
        rows = response_rows(active)
        response_pin = pin("/responses/tree001/a.jsonl", "response-artifact")
        receipt = build_terminal_response_receipt(
            active,
            rows,
            responses=response_pin,
            observed_backend_provenance=active.planned_backend_provenance,
        )
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["audit"]["n_expected"], 200)
        self.assertEqual(receipt["audit"]["n_missing"], 0)
        self.assertEqual(receipt["audit"]["n_extra"], 0)
        self.assertEqual(receipt["audit"]["n_duplicates"], 0)
        self.assertEqual(
            validate_terminal_response_receipt(
                receipt,
                expected_request=active,
                observed_responses=response_pin,
            ),
            receipt,
        )

    def test_rejects_missing_extra_duplicate_reordered_and_contract_drift(self) -> None:
        active = request("flan_notrans")
        rows = response_rows(active)
        cases = {
            "missing": rows[:-1],
            "extra": [
                *rows,
                replace(rows[-1], prompt_id="unexpected-prompt"),
            ],
            "duplicate": [rows[0], rows[0], *rows[2:]],
            "reordered": [rows[1], rows[0], *rows[2:]],
            "wrong node": [replace(rows[0], node_id="other"), *rows[1:]],
            "incomplete": [replace(rows[0], completed=False), *rows[1:]],
        }
        for label, changed in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "response grid is invalid"):
                    build_terminal_response_receipt(
                        active,
                        changed,
                        responses=pin("/responses/tree001/a.jsonl", "responses"),
                        observed_backend_provenance=active.planned_backend_provenance,
                    )

    def test_receipt_rejects_request_response_and_row_hash_tampering(self) -> None:
        active = request("flan_notrans")
        receipt = build_terminal_response_receipt(
            active,
            response_rows(active),
            responses=pin("/responses/tree001/a.jsonl", "responses"),
            observed_backend_provenance=active.planned_backend_provenance,
        )
        observed_pin = pin("/responses/tree001/a.jsonl", "responses")
        for mutator in (
            lambda value: value["responses"].__setitem__("sha256", digest("tampered")),
            lambda value: value.__setitem__("request_sha256", digest("tampered")),
            lambda value: value.__setitem__("response_rows_sha256", digest("tampered")),
        ):
            changed = {
                **receipt,
                "responses": dict(receipt["responses"]),
                "request": dict(receipt["request"]),
                "response_rows": [dict(row) for row in receipt["response_rows"]],
                "audit": dict(receipt["audit"]),
            }
            mutator(changed)
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_terminal_response_receipt(
                    changed,
                    expected_request=active,
                    observed_responses=observed_pin,
                )

    def test_receipt_rejects_observed_backend_runtime_drift(self) -> None:
        active = request("flan_notrans")
        observed = dict(active.planned_backend_provenance)
        observed["torch_version"] = "different-runtime"
        with self.assertRaisesRegex(ValueError, "differs from the planned backend"):
            build_terminal_response_receipt(
                active,
                response_rows(active),
                responses=pin("/responses/tree001/a.jsonl", "responses"),
                observed_backend_provenance=observed,
            )


if __name__ == "__main__":
    unittest.main()
