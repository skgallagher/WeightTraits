from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from weighttraits.behavior.allnode.audit import (
    EXECUTION_CONTRACT_SCHEMA,
    INFERENCE_PLAN_SCHEMA,
    MATRIX_PLAN_SCHEMA,
    MINILM_MODEL,
    MINILM_CONFIG_SHA256,
    MINILM_RECEIPT_SHA256,
    MINILM_REVISION,
    MINILM_RUNTIME,
    MINILM_SNAPSHOT_SHA256,
    PROTOCOLS,
    MATERIALIZATION_RECEIPT_SHA256,
    PROMPT_SHA256,
    REGISTRY_SHA256,
    RESPONSES_PER_NODE,
    STAGE_MANIFEST_SHA256,
    TEST_SCHEMA,
    TREE_IDS,
    build_dry_run_audit_bundle,
    serialize_audit_core,
    validate_audit_core,
    validate_dry_run_audit_bundle,
    expected_analysis_contract,
    semantic_endpoint_rows,
)
from weighttraits.behavior.allnode.gate import COHORTS, Pin, canonical_sha256
from weighttraits.behavior.allnode.provenance import (
    DOWNSTREAM_CODE_MANIFEST_BASENAME,
    DOWNSTREAM_CODE_MANIFEST_SHA256,
    PRODUCTION_INTERPRETER,
)


STAGE = "/wright/WeightTraits-downstream-20260831-native"
FROZEN = "/wright/WeightTraits-all-results-20260817-clean"
DEPLOY = STAGE + "/deployments/allnode_v1"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def pin(path: str, label: str) -> Pin:
    return Pin(path, digest(label))


def topology_scope():
    rows = []
    for index, tree_id in enumerate(TREE_IDS):
        n_nodes = 13 if index < 41 else 12
        n_internal = 6 if index < 26 else 5
        rows.append(
            {
                "tree_id": tree_id,
                "n_nodes": n_nodes,
                "n_internal": n_internal,
                "n_leaves": n_nodes - n_internal,
                "topology_sha256": digest(f"topology/{tree_id}"),
                "nodes_sha256": digest(f"nodes/{tree_id}"),
                "leaf_subset_sha256": digest(f"leaves/{tree_id}"),
                "pair_index_sha256": digest(f"pairs/{tree_id}"),
                "ordered_node_ids_sha256": digest(f"order/{tree_id}"),
            }
        )
    return rows


def topology_projection(rows):
    return [
        {
            key: row[key]
            for key in (
                "tree_id",
                "topology_sha256",
                "nodes_sha256",
                "leaf_subset_sha256",
                "pair_index_sha256",
            )
        }
        for row in rows
    ]


def checkpoint_payload(rows):
    scope_sha = canonical_sha256(topology_projection(rows))
    cohorts = {}
    for cohort in COHORTS:
        manifests = [
            {"tree_id": tree_id, "checkpoint_set_sha256": digest(f"{cohort}/{tree_id}")}
            for tree_id in TREE_IDS
        ]
        cohorts[cohort] = {
            "population": "all_trained_nodes",
            "internal_node_treatment": "observed_ancestors",
            "n_trees": 50,
            "n_nodes": 641,
            "n_internal": 276,
            "n_leaves": 365,
            "tree_ids": list(TREE_IDS),
            "manifest_scope": manifests,
            "manifest_scope_sha256": canonical_sha256(manifests),
        }
    return {
        "schema": "weighttraits.all_node_production_checkpoints.v1",
        "valid": True,
        "population": "all_trained_nodes",
        "internal_node_treatment": "observed_ancestors",
        "exclude_root": True,
        "n_cohorts": 4,
        "cohort_ids": list(COHORTS),
        "n_trees_per_cohort": 50,
        "n_nodes_per_cohort": 641,
        "n_internal_per_cohort": 276,
        "n_leaves_per_cohort": 365,
        "topology_scope_sha256": scope_sha,
        "cohorts": cohorts,
    }


def matrix_plan(rows):
    scope_sha = canonical_sha256(topology_projection(rows))
    cohorts = {}
    for cohort in COHORTS:
        output = f"{STAGE}/outputs/allnode_v1/direct/{cohort}"
        core = {
            "cohort_id": cohort,
            "direct_job_logical_id": f"direct_{cohort}",
            "output_root": output,
            "tree_ids": list(TREE_IDS),
            "topology_scope_sha256": scope_sha,
            "matrix_order_scope_sha256": canonical_sha256(rows),
        }
        cohorts[cohort] = {
            "direct_job_logical_id": f"direct_{cohort}",
            "output_root": output,
            "tree_ids": list(TREE_IDS),
            "plan_sha256": canonical_sha256(core),
        }
    return {
        "schema": MATRIX_PLAN_SCHEMA,
        "valid": True,
        "population": "all_trained_nodes",
        "internal_node_treatment": "observed_ancestors",
        "metric": "cosine",
        "cohort_ids": list(COHORTS),
        "n_trees": 50,
        "n_nodes": 641,
        "n_internal": 276,
        "n_leaves": 365,
        "topology_scope": rows,
        "topology_scope_sha256": scope_sha,
        "matrix_order_scope_sha256": canonical_sha256(rows),
        "cohorts": cohorts,
    }


def inference_plan():
    return {
        "schema": INFERENCE_PLAN_SCHEMA,
        "valid": True,
        "cohort_ids": list(COHORTS),
        "n_nodes_per_cohort": 641,
        "cohorts": {
            cohort: {
                "array": {"start": 0, "end": 640, "max_concurrent": 4},
                "n_tasks": 641,
                "protocols": list(PROTOCOLS[cohort]),
                "responses_per_node": RESPONSES_PER_NODE[cohort],
                "expected_response_rows": 641 * RESPONSES_PER_NODE[cohort],
                "device": "cuda",
                "torch_dtype": "auto",
                "execution_overrides": {},
            }
            for cohort in COHORTS
        },
    }


def job_ids():
    result = [f"direct_{cohort}" for cohort in COHORTS]
    for cohort in COHORTS:
        result.extend((f"infer_{cohort}", f"semantic_{cohort}"))
    return sorted(result)


def execution_contract(rows, checkpoints, configs, code, source_resolution, downstream):
    jobs = {}
    for logical_id in job_ids():
        prefix, cohort = logical_id.split("_", 1)
        kind = {
            "direct": "direct_allnode_cpu",
            "infer": "inference_allnode_gpu",
            "semantic": "semantic_allnode_gpu",
        }[prefix]
        plan_name = {
            "direct": "direct_plan.json",
            "infer": "inference_plan.json",
            "semantic": "semantic_plan.json",
        }[prefix]
        plan_schema = {
            "direct": "weighttraits.all_node_direct_cohort_plan.v2",
            "infer": "weighttraits.all_node_inference_cohort_plan.v1",
            "semantic": "weighttraits.all_node_semantic_cohort_plan.v1",
        }[prefix]
        output_prefix = "inference" if prefix == "infer" else prefix
        jobs[logical_id] = {
            "kind": kind,
            "cohort_key": cohort,
            "config": configs[logical_id].to_dict(),
            "runtime_plan": pin(
                f"{DEPLOY}/runtime_inputs/{cohort}/{plan_name}",
                f"plan/{logical_id}",
            ).to_dict(),
            "runtime_plan_schema": plan_schema,
            "output_root": f"{STAGE}/outputs/allnode_v1/{output_prefix}/{cohort}",
            "n_trees": 50,
            "n_nodes": 641,
            "n_internal": 276,
            "n_leaves": 365,
            "topology_scope_sha256": checkpoints["topology_scope_sha256"],
            "checkpoint_scope_sha256": checkpoints["cohorts"][cohort]["manifest_scope_sha256"],
            "node_covariate_scope_sha256": digest(f"covariates/{cohort}"),
            "metrics": ["l2", "cosine", "correlation"] if prefix == "direct" else [],
            "expected_response_rows": 641 * RESPONSES_PER_NODE[cohort] if prefix == "infer" else None,
            "semantic_endpoints": semantic_endpoint_rows(cohort) if prefix == "semantic" else [],
            "execution_overrides": {},
        }
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "valid": True,
        "population": "all_trained_nodes",
        "sensitivity_population": "trained_leaves",
        "internal_node_treatment": "observed_ancestors",
        "exclude_untrained_root": True,
        "cohort_ids": list(COHORTS),
        "tree_ids": list(TREE_IDS),
        "n_nodes_per_cohort": 641,
        "n_internal_per_cohort": 276,
        "n_leaves_per_cohort": 365,
        "checkpoint_identity": {
            "scheme": "frozen_fs_full_small_fixed5x1m_large_v1",
            "index": pin(
                f"{DEPLOY}/runtime_receipts/ALLNODE_V1_CHECKPOINT_IDENTITIES.json",
                "checkpoint identities",
            ).to_dict(),
            "n_checkpoints": 4 * 641,
            "reattest_at_runtime": True,
        },
        "topology_scope_sha256": checkpoints["topology_scope_sha256"],
        "source_scope_sha256": digest("source scope"),
        "runtime_provenance": {
            "production_interpreter": PRODUCTION_INTERPRETER,
            "python_version": "3.11.9",
            "python_implementation": "CPython",
            "source_resolution": source_resolution.to_dict(),
            "allnode_code_manifest": code.to_dict(),
            "downstream_code_manifest": downstream.to_dict(),
            "reattest_before_after_each_job": True,
        },
        "analysis": expected_analysis_contract(),
        "jobs": jobs,
        "execution_overrides": {},
        "no_automatic_recovery": True,
    }


def kwargs():
    rows = topology_scope()
    code = pin(f"{DEPLOY}/runtime_receipts/ALLNODE_V1_CODE_MANIFEST.json", "code")
    source_resolution = pin(
        f"{DEPLOY}/runtime_receipts/ALLNODE_V1_SOURCE_RESOLUTION.json",
        "source resolution",
    )
    downstream = Pin(
        f"{STAGE}/runtime_receipts/{DOWNSTREAM_CODE_MANIFEST_BASENAME}",
        DOWNSTREAM_CODE_MANIFEST_SHA256,
    )
    prompts = {
        cohort: {
            protocol: Pin(
                f"{STAGE}/outputs/prompts/{'causal_lm' if cohort.startswith('llama_') else 'seq2seq'}/{protocol}.{'causal_lm' if cohort.startswith('llama_') else 'seq2seq'}.prompts.jsonl",
                PROMPT_SHA256[cohort][protocol],
            )
            for protocol in PROTOCOLS[cohort]
        }
        for cohort in COHORTS
    }
    materials = {
        cohort: {
            protocol: Pin(
                f"{STAGE}/outputs/prompts/{'causal_lm' if cohort.startswith('llama_') else 'seq2seq'}/prompt_materialization_receipt.json",
                MATERIALIZATION_RECEIPT_SHA256[
                    "causal_lm" if cohort.startswith("llama_") else "seq2seq"
                ],
            )
            for protocol in PROTOCOLS[cohort]
        }
        for cohort in COHORTS
    }
    configs = {
        logical_id: pin(
            f"{DEPLOY}/configs/jobs/{logical_id}.json", f"config/{logical_id}"
        )
        for logical_id in job_ids()
    }
    checkpoints = checkpoint_payload(rows)
    execution = execution_contract(
        rows, checkpoints, configs, code, source_resolution, downstream
    )
    return {
        "audit_path": f"{DEPLOY}/runtime_receipts/ALLNODE_V1_DRY_RUN_AUDIT.json",
        "stage_root": STAGE,
        "frozen_stage": FROZEN,
        "frozen_stage_manifest": Pin(
            f"{FROZEN}/.stage_provenance/STAGE_SHA256SUMS", STAGE_MANIFEST_SHA256
        ),
        "checkpoint_receipt": Pin(
            f"{DEPLOY}/runtime_receipts/ALLNODE_V1_CHECKPOINTS.json",
            canonical_sha256(checkpoints),
        ),
        "checkpoint_payload": checkpoints,
        "registry": Pin(f"{STAGE}/examples/behavior/corrected_behavior_protocols_v1.json", REGISTRY_SHA256),
        "prompt_artifacts": prompts,
        "materialization_receipts": materials,
        "minilm": {
            "model": MINILM_MODEL,
            "revision": MINILM_REVISION,
            "runtime": MINILM_RUNTIME,
            "receipt": {
                "path": f"{STAGE}/runtime_receipts/MINILM_ENCODER.json",
                "sha256": MINILM_RECEIPT_SHA256,
            },
            "config": {
                "path": f"{STAGE}/examples/downstream/20260831/minilm_encoder.json",
                "sha256": MINILM_CONFIG_SHA256,
            },
            "snapshot_sha256": MINILM_SNAPSHOT_SHA256,
        },
        "job_configs": configs,
        "code_manifest": code,
        "source_resolution": source_resolution,
        "downstream_code_manifest": downstream,
        "matrix_plan": matrix_plan(rows),
        "inference_plan": inference_plan(),
        "execution_contract_receipt": Pin(
            f"{DEPLOY}/runtime_receipts/ALLNODE_V1_EXECUTION_CONTRACT.json",
            canonical_sha256(execution),
        ),
        "execution_contract_payload": execution,
        "test_attestation": {
            "schema": TEST_SCHEMA,
            "valid": True,
            "runner": "unittest",
            "interpreter": PRODUCTION_INTERPRETER,
            "python_version": "3.11.9",
            "python_implementation": "CPython",
            "command": [PRODUCTION_INTERPRETER, "-m", "unittest", "discover", "-s", "tests", "-v"],
            "tests_run": 43,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "code_manifest_sha256": code.sha256,
        },
        "state_absence": {
            "outputs_exist": False,
            "intent_exists": False,
            "jobs_receipt_exists": False,
        },
    }


def rebind_execution_contract(values: dict) -> None:
    """Keep an intentionally changed execution payload bound to its receipt.

    Contract-semantic tests need to reach the deeper validator; receipt drift
    itself is exercised separately below.
    """

    receipt = values["execution_contract_receipt"]
    values["execution_contract_receipt"] = Pin(
        receipt.path,
        canonical_sha256(values["execution_contract_payload"]),
    )


class DryRunAuditTests(unittest.TestCase):
    def test_builds_valid_non_circular_boundary_and_exact_gate(self) -> None:
        bundle = build_dry_run_audit_bundle(**kwargs())
        self.assertTrue(bundle.valid)
        self.assertTrue(bundle.audit_core["valid"])
        self.assertNotIn("gate_contract", bundle.audit_core)
        self.assertEqual(bundle.audit_boundary.sha256, canonical_sha256(bundle.audit_core))
        self.assertEqual(bundle.gate_contract.dry_run_audit, bundle.audit_boundary)
        self.assertEqual(len(bundle.gate_contract.jobs), 12)
        self.assertTrue(bundle.gate_contract.to_dict()["no_automatic_recovery"])
        self.assertEqual(
            hashlib.sha256(serialize_audit_core(bundle.audit_core).encode()).hexdigest(),
            bundle.audit_boundary.sha256,
        )
        self.assertEqual(validate_dry_run_audit_bundle(bundle.to_dict()), bundle)

    def test_rejects_checkpoint_scope_or_payload_hash_drift(self) -> None:
        bundle = build_dry_run_audit_bundle(**kwargs()).to_dict()
        bundle["audit_core"]["checkpoint_production"]["payload"]["n_nodes_per_cohort"] = 640
        with self.assertRaises(ValueError):
            validate_dry_run_audit_bundle(bundle)
        core = build_dry_run_audit_bundle(**kwargs()).audit_core
        changed = deepcopy(core)
        changed["checkpoint_production"]["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "payload hash"):
            validate_audit_core(changed)

    def test_rejects_prompt_registry_materialization_or_minilm_pin_drift(self) -> None:
        cases = []
        registry = kwargs()
        registry["registry"] = Pin(registry["registry"].path, "0" * 64)
        cases.append(registry)
        prompt = kwargs()
        old = prompt["prompt_artifacts"]["llama_full"][PROTOCOLS["llama_full"][0]]
        prompt["prompt_artifacts"]["llama_full"][PROTOCOLS["llama_full"][0]] = Pin(old.path, "0" * 64)
        cases.append(prompt)
        material = kwargs()
        del material["materialization_receipts"]["flan_full"][PROTOCOLS["flan_full"][0]]
        cases.append(material)
        minilm = kwargs()
        minilm["minilm"]["revision"] = "0" * 40
        cases.append(minilm)
        for values in cases:
            with self.assertRaises(ValueError):
                build_dry_run_audit_bundle(**values)

    def test_rejects_matrix_topology_order_and_checkpoint_binding_drift(self) -> None:
        order = kwargs()
        order["matrix_plan"]["topology_scope"][0]["ordered_node_ids_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            build_dry_run_audit_bundle(**order)
        topology = kwargs()
        topology["matrix_plan"]["topology_scope"][0]["topology_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            build_dry_run_audit_bundle(**topology)

    def test_rejects_inference_count_protocol_or_override_drift(self) -> None:
        for field, value in (
            ("n_tasks", 640),
            ("protocols", ["wrong"]),
            ("expected_response_rows", 1),
            ("execution_overrides", {"ROW_END": 1}),
        ):
            values = kwargs()
            values["inference_plan"]["cohorts"]["llama_full"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "inference"):
                    build_dry_run_audit_bundle(**values)

    def test_rejects_unbound_or_statistically_drifted_execution_contract(self) -> None:
        formula = kwargs()
        formula["execution_contract_payload"]["analysis"]["regressions"][1]["formula"] = "semantic_similarity ~ direct_distance"
        rebind_execution_contract(formula)
        with self.assertRaisesRegex(ValueError, "regression/DL plan"):
            build_dry_run_audit_bundle(**formula)

        plan = kwargs()
        plan["execution_contract_payload"]["jobs"]["infer_llama_full"]["runtime_plan"]["path"] += ".wrong"
        rebind_execution_contract(plan)
        with self.assertRaisesRegex(ValueError, "parsed job drift"):
            build_dry_run_audit_bundle(**plan)

        config = kwargs()
        config["execution_contract_payload"]["jobs"]["direct_flan_full"]["config"]["sha256"] = "0" * 64
        rebind_execution_contract(config)
        with self.assertRaisesRegex(ValueError, "parsed job drift"):
            build_dry_run_audit_bundle(**config)

        receipt = kwargs()
        receipt["execution_contract_receipt"] = Pin(
            receipt["execution_contract_receipt"].path, "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "receipt/payload hash"):
            build_dry_run_audit_bundle(**receipt)

    def test_rejects_failed_tests_existing_state_or_core_override(self) -> None:
        failed = kwargs()
        failed["test_attestation"]["failures"] = 1
        with self.assertRaisesRegex(ValueError, "test attestation"):
            build_dry_run_audit_bundle(**failed)
        existing = kwargs()
        existing["state_absence"]["outputs_exist"] = True
        with self.assertRaisesRegex(ValueError, "all be false"):
            build_dry_run_audit_bundle(**existing)
        bundle = build_dry_run_audit_bundle(**kwargs()).to_dict()
        bundle["audit_core"]["execution_overrides"] = {"MAX_EVAL_SAMPLES": 1}
        with self.assertRaisesRegex(ValueError, "overrides"):
            validate_dry_run_audit_bundle(bundle)

    def test_rejects_config_code_paths_outside_isolated_deployment(self) -> None:
        config = kwargs()
        logical_id = job_ids()[0]
        config["job_configs"][logical_id] = pin(
            f"{STAGE}/configs/{logical_id}.json", "outside-config"
        )
        with self.assertRaises(ValueError):
            build_dry_run_audit_bundle(**config)
        code = kwargs()
        code["code_manifest"] = pin(f"{STAGE}/CODE_SHA256", "outside-code")
        code["test_attestation"]["code_manifest_sha256"] = code["code_manifest"].sha256
        with self.assertRaises(ValueError):
            build_dry_run_audit_bundle(**code)
        audit = kwargs()
        audit["audit_path"] = f"{DEPLOY}/configs/jobs/direct_llama_full.json"
        with self.assertRaisesRegex(ValueError, "unique isolated audit"):
            build_dry_run_audit_bundle(**audit)

    def test_rejects_runtime_provenance_pin_interpreter_or_version_drift(self) -> None:
        for field in ("source_resolution", "downstream_code_manifest"):
            values = kwargs()
            values[field] = Pin(values[field].path, "0" * 64)
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_dry_run_audit_bundle(**values)
        values = kwargs()
        values["execution_contract_payload"]["runtime_provenance"][
            "production_interpreter"
        ] = "/usr/bin/python3"
        rebind_execution_contract(values)
        with self.assertRaisesRegex(ValueError, "runtime provenance"):
            build_dry_run_audit_bundle(**values)
        values = kwargs()
        values["test_attestation"]["python_version"] = "3.12.1"
        with self.assertRaisesRegex(ValueError, "runtime provenance"):
            build_dry_run_audit_bundle(**values)

    def test_rejects_gate_boundary_or_gate_hash_tampering(self) -> None:
        payload = build_dry_run_audit_bundle(**kwargs()).to_dict()
        payload["audit_boundary"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "boundary hash"):
            validate_dry_run_audit_bundle(payload)
        payload = build_dry_run_audit_bundle(**kwargs()).to_dict()
        payload["gate_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "gate_contract_sha256"):
            validate_dry_run_audit_bundle(payload)


if __name__ == "__main__":
    unittest.main()
