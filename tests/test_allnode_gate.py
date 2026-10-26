from __future__ import annotations

import json
import unittest

from weighttraits.behavior.allnode.gate import (
    COHORTS,
    DIRECT_KIND,
    INFERENCE_KIND,
    Pin,
    SEMANTIC_KIND,
    build_gate_contract,
    serialize_gate_contract,
    validate_gate_contract,
)
from weighttraits.behavior.allnode.provenance import (
    DOWNSTREAM_CODE_MANIFEST_BASENAME,
    DOWNSTREAM_CODE_MANIFEST_SHA256,
)


STAGE = "/wright/WeightTraits-downstream-20260831-native"
FROZEN = "/wright/WeightTraits-all-results-20260817-clean"
DEPLOYMENT = STAGE + "/deployments/allnode_v1"


def contract():
    return build_gate_contract(
        stage_root=STAGE,
        frozen_stage=FROZEN,
        code_manifest=Pin(
            DEPLOYMENT + "/runtime_receipts/ALLNODE_V1_CODE_MANIFEST.json",
            "a" * 64,
        ),
        source_resolution=Pin(
            DEPLOYMENT + "/runtime_receipts/ALLNODE_V1_SOURCE_RESOLUTION.json",
            "c" * 64,
        ),
        source_scope_sha256="d" * 64,
        downstream_code_manifest=Pin(
            STAGE + "/runtime_receipts/" + DOWNSTREAM_CODE_MANIFEST_BASENAME,
            DOWNSTREAM_CODE_MANIFEST_SHA256,
        ),
        python_version="3.11.9",
        dry_run_audit=Pin(
            STAGE + "/runtime_receipts/ALLNODE_V1_DRY_RUN_AUDIT.json", "b" * 64
        ),
    )


class GateContractTests(unittest.TestCase):
    def test_freezes_exact_twelve_job_dag_and_global_gpu_cap(self) -> None:
        payload = contract().to_dict()
        self.assertEqual(payload["population_contract"]["cohort_ids"], list(COHORTS))
        self.assertEqual(payload["population_contract"]["n_trained_nodes_per_cohort"], 641)
        self.assertEqual(payload["population_contract"]["n_internal_per_cohort"], 276)
        self.assertEqual(payload["population_contract"]["n_leaves_per_cohort"], 365)
        self.assertEqual(payload["scheduler_policy"]["max_global_gpu_tasks"], 4)
        self.assertFalse(payload["scheduler_policy"]["requeue"])
        self.assertTrue(payload["no_automatic_recovery"])
        self.assertEqual(len(payload["jobs"]), 12)

        by_id = {job["logical_id"]: job for job in payload["jobs"]}
        self.assertEqual(sum(job["kind"] == DIRECT_KIND for job in payload["jobs"]), 4)
        self.assertEqual(sum(job["kind"] == INFERENCE_KIND for job in payload["jobs"]), 4)
        self.assertEqual(sum(job["kind"] == SEMANTIC_KIND for job in payload["jobs"]), 4)
        for cohort in COHORTS:
            self.assertEqual(by_id[f"direct_{cohort}"]["dependencies"], [])
            self.assertEqual(
                by_id[f"semantic_{cohort}"]["dependencies"],
                [f"direct_{cohort}", f"infer_{cohort}"],
            )
        self.assertEqual(by_id["infer_llama_full"]["dependencies"], [])
        self.assertEqual(
            by_id["infer_llama_notrans"]["dependencies"], ["semantic_llama_full"]
        )
        self.assertEqual(by_id["infer_flan_full"]["dependencies"], ["semantic_llama_notrans"])
        self.assertEqual(by_id["infer_flan_notrans"]["dependencies"], ["semantic_flan_full"])

        for job in payload["jobs"]:
            route = job["route"]
            self.assertEqual(
                (route["account"], route["partition"], route["qos"]),
                ("statds", "all", "normal"),
            )
            self.assertEqual((route["required_node"], route["excluded_node"]), ("n03", "n01"))
            self.assertFalse(route["requeue"])
            if job["kind"] == DIRECT_KIND:
                self.assertEqual(job["resources"]["gpu_count"], 0)
                self.assertIsNone(job["resources"]["gpu_type"])
                self.assertIsNone(job["resources"]["array"])
            elif job["kind"] == INFERENCE_KIND:
                self.assertEqual(job["resources"]["gpu_count"], 1)
                self.assertEqual(job["resources"]["gpu_type"], "nvidia-l40")
                self.assertEqual(
                    job["resources"]["array"],
                    {"start": 0, "end": 640, "max_concurrent": 4},
                )
                self.assertEqual(
                    job["output_dir"],
                    f"{STAGE}/outputs/allnode_v1/inference/{job['cohort_id']}",
                )
            else:
                self.assertEqual(job["resources"]["gpu_count"], 1)
                self.assertEqual(job["resources"]["gpu_type"], "nvidia-l40")
                self.assertIsNone(job["resources"]["array"])

    def test_serialization_is_canonical_and_round_trips(self) -> None:
        plan = contract()
        serialized = serialize_gate_contract(plan)
        self.assertTrue(serialized.endswith("\n"))
        restored = validate_gate_contract(json.loads(serialized))
        self.assertEqual(restored, plan)
        self.assertEqual(restored.sha256, plan.sha256)
        self.assertEqual(serialized, serialize_gate_contract(contract()))

    def test_rejects_route_resource_array_and_dependency_drift(self) -> None:
        mutations = []
        route = contract().to_dict()
        route["jobs"][0]["route"]["qos"] = "priority"
        mutations.append(route)
        gpu = contract().to_dict()
        gpu["jobs"][0]["resources"]["gpu_count"] = 1
        mutations.append(gpu)
        array = contract().to_dict()
        array["jobs"][4]["resources"]["array"]["max_concurrent"] = 5
        mutations.append(array)
        dependency = contract().to_dict()
        dependency["jobs"][4]["dependencies"] = ["direct_llama_full"]
        mutations.append(dependency)
        for payload in mutations:
            with self.subTest(payload=payload["jobs"][0]["logical_id"]):
                with self.assertRaises(ValueError):
                    validate_gate_contract(payload)

    def test_rejects_cycle_unknown_or_duplicate_dependency(self) -> None:
        cycle = contract().to_dict()
        cycle["jobs"][4]["dependencies"] = ["semantic_llama_full"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_gate_contract(cycle)
        unknown = contract().to_dict()
        unknown["jobs"][4]["dependencies"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown dependency"):
            validate_gate_contract(unknown)
        duplicate = contract().to_dict()
        duplicate["jobs"][5]["dependencies"] = ["direct_llama_full", "direct_llama_full"]
        with self.assertRaisesRegex(ValueError, "duplicate dependencies"):
            validate_gate_contract(duplicate)

    def test_rejects_duplicate_names_and_reused_config_or_output_paths(self) -> None:
        cases = []
        duplicate_name = contract().to_dict()
        duplicate_name["jobs"][1]["name"] = duplicate_name["jobs"][0]["name"]
        cases.append((duplicate_name, "duplicate scheduler job names"))
        config_reuse = contract().to_dict()
        config_reuse["jobs"][1]["config_path"] = config_reuse["jobs"][0]["config_path"]
        cases.append((config_reuse, "config paths cannot be reused"))
        output_reuse = contract().to_dict()
        output_reuse["jobs"][1]["output_dir"] = output_reuse["jobs"][0]["output_dir"]
        cases.append((output_reuse, "output paths cannot be reused"))
        reserved_reuse = contract().to_dict()
        reserved_reuse["jobs"][0]["output_dir"] = reserved_reuse["log_dir"]
        cases.append((reserved_reuse, "inside downstream outputs/allnode_v1"))
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_gate_contract(payload)

    def test_rejects_any_execution_override_or_recovery_permission(self) -> None:
        top = contract().to_dict()
        top["execution_overrides"] = {"ROW_END": 1}
        with self.assertRaisesRegex(ValueError, "overrides are forbidden"):
            validate_gate_contract(top)
        job = contract().to_dict()
        job["jobs"][4]["execution_overrides"] = {"MAX_EVAL_SAMPLES": 2}
        with self.assertRaisesRegex(ValueError, "overrides are forbidden"):
            validate_gate_contract(job)
        recovery = contract().to_dict()
        recovery["no_automatic_recovery"] = False
        with self.assertRaisesRegex(ValueError, "must be true"):
            validate_gate_contract(recovery)

    def test_rejects_reused_reserved_paths_and_nonisolated_pins(self) -> None:
        reused = contract().to_dict()
        reused["jobs_receipt_path"] = reused["intent_path"]
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            validate_gate_contract(reused)
        with self.assertRaisesRegex(ValueError, "exact isolated receipt"):
            build_gate_contract(
                stage_root=STAGE,
                frozen_stage=FROZEN,
                code_manifest=Pin(STAGE + "/runtime_receipts/CODE", "a" * 64),
                source_resolution=Pin(
                    DEPLOYMENT + "/runtime_receipts/ALLNODE_V1_SOURCE_RESOLUTION.json",
                    "c" * 64,
                ),
                source_scope_sha256="d" * 64,
                downstream_code_manifest=Pin(
                    STAGE + "/runtime_receipts/" + DOWNSTREAM_CODE_MANIFEST_BASENAME,
                    DOWNSTREAM_CODE_MANIFEST_SHA256,
                ),
                python_version="3.11.9",
                dry_run_audit=Pin(STAGE + "/runtime_receipts/AUDIT", "b" * 64),
            )
        with self.assertRaisesRegex(ValueError, "exact isolated receipt"):
            build_gate_contract(
                stage_root=STAGE,
                frozen_stage=FROZEN,
                code_manifest=Pin(DEPLOYMENT + "/SAME", "a" * 64),
                source_resolution=Pin(
                    DEPLOYMENT + "/runtime_receipts/ALLNODE_V1_SOURCE_RESOLUTION.json",
                    "c" * 64,
                ),
                source_scope_sha256="d" * 64,
                downstream_code_manifest=Pin(
                    STAGE + "/runtime_receipts/" + DOWNSTREAM_CODE_MANIFEST_BASENAME,
                    DOWNSTREAM_CODE_MANIFEST_SHA256,
                ),
                python_version="3.11.9",
                dry_run_audit=Pin(DEPLOYMENT + "/SAME", "b" * 64),
            )

    def test_exact_keys_reject_hidden_scheduler_or_override_fields(self) -> None:
        payload = contract().to_dict()
        payload["jobs"][4]["submit_line"] = "sbatch --qos=priority"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_gate_contract(payload)

    def test_rejects_runtime_provenance_and_publication_root_drift(self) -> None:
        for field, value in (
            ("production_interpreter", "/usr/bin/python3"),
            ("python_version", "3.12"),
            ("source_scope_sha256", "0" * 63),
            ("reattest_before_after_each_job", False),
        ):
            payload = contract().to_dict()
            payload["runtime_provenance"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_gate_contract(payload)
        for field, value in (
            ("config_path", f"{STAGE}/configs/outside.json"),
            ("wrapper_path", f"{STAGE}/scripts/outside.sbatch"),
            ("output_dir", f"{STAGE}/other/output"),
        ):
            payload = contract().to_dict()
            payload["jobs"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_gate_contract(payload)


if __name__ == "__main__":
    unittest.main()
