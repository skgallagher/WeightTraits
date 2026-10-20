from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weighttraits.behavior.allnode.jsonl import decode_lf_json_objects
from weighttraits.behavior.allnode.runtime import _decoded_records
from weighttraits.behavior.allnode.semantic_runner import _jsonl_objects
import weighttraits.behavior.allnode.continuation as continuation
from weighttraits.behavior.allnode.continuation import (
    CommandResult,
    build_plan,
    query_original_scheduler,
    query_v1c1_scheduler,
    sbatch_argv,
)


def row(value: str) -> bytes:
    return (
        json.dumps({"generated_response": value}, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def scheduler_evidence() -> dict:
    observations = []
    for index in range(4):
        argv = ["scheduler-query", str(index)]
        observations.append(
            {
                "argv": argv,
                "argv_sha256": hashlib.sha256("\0".join(argv).encode()).hexdigest(),
                "returncode": 0,
                "stdout_sha256": hashlib.sha256(f"out-{index}".encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "stdout_n_lines": 1,
                "stderr_n_lines": 0,
            }
        )
    core = {
        "schema": "weighttraits.allnode_v1c5_scheduler_evidence.v1",
        "valid": True,
        "capture_host": "headnode",
        "captured_at_utc": "2026-09-02T18:00:00Z",
        "observations": observations,
        "state_contract": {"original": {}, "v1c1": {}},
        "scheduler_queries": 4,
        "scheduler_actions": 0,
        "predecessor_jobs_mutated": False,
    }
    return {
        **core,
        "scheduler_evidence_sha256": continuation.canonical_sha256(core),
    }


def predecessor_evidence() -> dict:
    return {
        "deployment": str(continuation.V1C1_DEPLOYMENT),
        "pins": {
            label: {"path": str(path), "sha256": digest}
            for label, (path, digest) in continuation.V1C1_PINS.items()
        },
        "acceptance_receipt": {
            "path": str(continuation.V1C1_ACCEPTANCE_RECEIPT),
            "absent": True,
        },
        "job_id": "184529",
        "state": "FAILED",
        "exit_code": "1:0",
        "failure_signature": "compute_node_sacct_localhost_6819_connection_refused",
        "failed_before_artifact_validation": True,
        "artifact_inventory_passes_completed": 0,
        "downstream_outputs_created": False,
        "predecessor_jobs_mutated": False,
    }


class LiteralLfTests(unittest.TestCase):
    def test_unicode_line_separators_remain_json_string_data(self):
        payload = row("before\u2028middle\u2029next\u0085after")
        expected = [{"generated_response": "before\u2028middle\u2029next\u0085after"}]
        self.assertEqual(decode_lf_json_objects(payload, "test"), expected)
        self.assertEqual(_decoded_records(payload), expected)
        self.assertEqual(_jsonl_objects(payload, "test"), expected)
        self.assertEqual(len(payload.decode("utf-8").splitlines()), 4)

    def test_actual_like_1490_record_boundary(self):
        payload = b"".join(row("contains \u2028 data" if index == 701 else str(index)) for index in range(1490))
        self.assertEqual(len(decode_lf_json_objects(payload, "test")), 1490)
        self.assertEqual(len(_decoded_records(payload)), 1490)
        self.assertEqual(len(_jsonl_objects(payload, "test")), 1490)

    def test_malformed_framing_fails_closed(self):
        cases = (
            row("ok")[:-1],
            row("ok") + b"\n",
            b"[]\n",
            b"{not-json}\n",
            b"\xff\n",
        )
        for payload in cases:
            with self.subTest(payload=payload[:20]):
                with self.assertRaises(ValueError):
                    decode_lf_json_objects(payload, "test")


class ContinuationPlanTests(unittest.TestCase):
    def test_exact_v1c5_eight_job_serial_chain_and_routes(self):
        plan = build_plan()
        self.assertEqual(len(plan), 8)
        self.assertEqual(
            [job.scheduler_name for job in plan],
            [
                "wt-an-v1c5-adopt-llama-full",
                "wt-an-v1c5-semantic-llama-full",
                "wt-an-v1c5-infer-llama-notrans",
                "wt-an-v1c5-semantic-llama-notrans",
                "wt-an-v1c5-infer-flan-full",
                "wt-an-v1c5-semantic-flan-full",
                "wt-an-v1c5-infer-flan-notrans",
                "wt-an-v1c5-semantic-flan-notrans",
            ],
        )
        self.assertIsNone(plan[0].dependency)
        for previous, current in zip(plan, plan[1:]):
            self.assertEqual(current.dependency, previous.logical_id)
        self.assertFalse(plan[0].gpu)
        self.assertTrue(all(job.gpu for job in plan[1:]))
        for job in plan:
            argv = sbatch_argv(
                job,
                dependency_job_id=None if job.dependency is None else "9000",
                code_manifest_sha256="1" * 64,
                acceptance_sha256="2" * 64,
                scheduler_evidence_sha256="3" * 64,
            )
            self.assertIn("--account=statds", argv)
            self.assertIn("--partition=all", argv)
            self.assertIn("--qos=normal", argv)
            self.assertIn("--nodelist=n03", argv)
            self.assertIn("--exclude=n01", argv)
            self.assertIn("--no-requeue", argv)
            joined = "\0".join(argv)
            for forbidden in (
                "OVERRIDE_MAX_STEPS",
                "MAX_TRAIN_SAMPLES",
                "MAX_EVAL_SAMPLES",
                "ROW_END",
            ):
                self.assertNotIn(forbidden, joined)
            if job.kind == "inference_allnode_gpu":
                self.assertIn("--array=0-640%4", argv)
            else:
                self.assertFalse(any(value.startswith("--array=") for value in argv))
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "slurm"
            / "allnode_v1c5_bound_runner.sbatch"
        ).read_text(encoding="utf-8")
        self.assertEqual(wrapper.count("OLD_DEPLOYMENT_ROOT="), 1)
        self.assertIn('[[ "${WT_ALLNODE_DEPLOYMENT_ROOT}" == "${OLD_DEPLOYMENT_ROOT}" ]]', wrapper)
        for name in (
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_OFFLINE",
            "TOKENIZERS_PARALLELISM",
        ):
            self.assertIn(f"export {name}=", wrapper)

    def test_plan_contains_no_llama_full_inference(self):
        plan = build_plan()
        self.assertNotIn("infer_llama_full", [job.runner_logical_id for job in plan])
        self.assertEqual([job.kind for job in plan].count("artifact_adoption_cpu"), 1)
        self.assertEqual([job.kind for job in plan].count("inference_allnode_gpu"), 3)
        output_targets = continuation.output_targets()
        self.assertNotIn(
            continuation.STAGE / "outputs" / "allnode_v1" / "semantic" / "llama_full",
            output_targets,
        )
        self.assertEqual(
            continuation.preserved_llama_semantic_completion_targets(),
            (
                continuation.STAGE
                / "outputs"
                / "allnode_v1"
                / "semantic"
                / "llama_full"
                / "translation"
                / "runset_receipt.json",
                continuation.STAGE
                / "outputs"
                / "allnode_v1"
                / "semantic"
                / "llama_full"
                / "terminal_receipt.json",
            ),
        )

    def test_v1c5_wrappers_are_isolated_and_adoption_has_no_scheduler_commands(self):
        root = Path(__file__).resolve().parents[1] / "scripts" / "slurm"
        adoption = (root / "allnode_v1c5_adopt_cpu.sbatch").read_text(encoding="utf-8")
        bound = (root / "allnode_v1c5_bound_runner.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("WT_V1C1_", adoption + bound)
        self.assertNotIn("sacct", adoption)
        self.assertNotIn("squeue", adoption)
        self.assertIn("WT_V1C5_SCHEDULER_EVIDENCE", adoption)
        self.assertIn("ALLNODE_V1C5_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json", adoption)
        self.assertIn("ALLNODE_V1C5_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json", bound)

    def test_wright_scheduler_snapshot_without_synthetic_array_parent(self):
        def accounting_row(*values: str) -> str:
            self.assertEqual(len(values), 12)
            return "|".join(values) + "|\n"

        accounting: list[str] = []
        for job_id in range(183511, 183515):
            accounting.append(
                accounting_row(
                    str(job_id),
                    str(job_id),
                    "COMPLETED",
                    "0:0",
                    "00:01:00",
                    "2026-09-01T00:00:00",
                    "2026-09-01T00:01:00",
                    "n03",
                    "all",
                    "normal",
                    "statds",
                    "sbatch direct",
                )
            )
        for array_index in range(641):
            failed = array_index == 438
            accounting.append(
                accounting_row(
                    "184002" if failed else str(190000 + array_index),
                    f"183515_{array_index}",
                    "FAILED" if failed else "COMPLETED",
                    "1:0" if failed else "0:0",
                    "00:02:09" if failed else "01:00:00",
                    "2026-09-01T22:47:42" if failed else "2026-09-01T00:00:00",
                    "2026-09-01T22:49:51" if failed else "2026-09-01T01:00:00",
                    "n03",
                    "all",
                    "normal",
                    "statds",
                    "sbatch array",
                )
            )
        queue = ""

        def runner(argv):
            if argv[0] == "sacct":
                return CommandResult(0, "".join(accounting), "")
            if argv[0] == "squeue":
                return CommandResult(0, queue, "")
            raise AssertionError(argv)

        evidence = query_original_scheduler(runner)
        array = evidence["original_llama_full_array"]
        self.assertEqual(array["parent_state"], "FAILED")
        self.assertEqual(array["n_completed_0_0"], 640)
        self.assertEqual(array["sole_failure"]["job_id"], "183515_438")

    def test_v1c1_predecessor_scheduler_snapshot_is_exact(self):
        names = (
            "wt-an-v1c1-adopt-llama-full",
            "wt-an-v1c1-semantic-llama-full",
            "wt-an-v1c1-infer-llama-notrans",
            "wt-an-v1c1-semantic-llama-notrans",
            "wt-an-v1c1-infer-flan-full",
            "wt-an-v1c1-semantic-flan-full",
            "wt-an-v1c1-infer-flan-notrans",
            "wt-an-v1c1-semantic-flan-notrans",
        )
        arrays = {184531, 184533, 184535}
        accounting = []
        for offset, job_id in enumerate(range(184529, 184537)):
            failed = job_id == 184529
            displayed = f"{job_id}_[0-640%4]" if job_id in arrays else str(job_id)
            submit = (
                f"sbatch --job-name={names[offset]} --account=statds "
                "--partition=all --qos=normal --nodelist=n03 --exclude=n01 "
                "--no-requeue"
            )
            if not failed:
                submit += f" --gres=gpu:nvidia-l40:1 --dependency=afterok:{job_id - 1}"
            if job_id in arrays:
                submit += " --array=0-640%4"
            values = (
                str(job_id),
                displayed,
                names[offset],
                "FAILED" if failed else "CANCELLED by 2208",
                "1:0" if failed else "0:0",
                "00:01:38" if failed else "00:00:00",
                "2026-09-02T13:42:46" if failed else "Unknown",
                "2026-09-02T13:44:24" if failed else "Unknown",
                "n03" if failed else "None assigned",
                "all",
                "normal",
                "statds",
                submit,
            )
            accounting.append("|".join(values) + "|\n")
        queue = ""

        def runner(argv):
            if argv[0] == "sacct":
                return CommandResult(0, "".join(accounting), "")
            if argv[0] == "squeue":
                return CommandResult(0, queue, "")
            raise AssertionError(argv)

        evidence = query_v1c1_scheduler(runner)
        self.assertTrue(evidence["adoption_failed_before_artifact_validation"])
        self.assertEqual(evidence["jobs"][0]["state"], "FAILED")
        self.assertEqual(evidence["successor_queue"], [])
        self.assertTrue(evidence["successors_absent_from_live_queue"])

    def test_v1c1_failure_file_pins_and_acceptance_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            files = {
                label: root / f"{label}.dat" for label in continuation.V1C1_PINS
            }
            for label in ("code_manifest", "test_attestation", "dry_run", "expected_acceptance"):
                files[label].write_bytes(b"{}\n")
            intent = {
                "schema": "weighttraits.allnode_v1c1_submit_intent.v1",
                "valid": True,
                "n_planned": 8,
            }
            files["intent"].write_bytes(continuation.canonical_bytes(intent))
            jobs = {
                "schema": "weighttraits.allnode_v1c1_submit_jobs.v1",
                "valid": True,
                "terminal": True,
                "n_planned": 8,
                "n_submitted": 8,
                "partial": False,
                "failure": None,
                "retry_allowed": False,
                "resubmission_allowed": False,
                "submitted_prefix": [
                    {"job_id": str(value)} for value in range(184529, 184537)
                ],
            }
            files["jobs"].write_bytes(continuation.canonical_bytes(jobs))
            files["adoption_stdout"].write_text('{"valid":true}\n', encoding="utf-8")
            files["adoption_stderr"].write_text(
                "query_original_scheduler localhost:6819 Connection refused\n",
                encoding="utf-8",
            )
            pins = {
                label: (path, hashlib.sha256(path.read_bytes()).hexdigest())
                for label, path in files.items()
            }
            with (
                patch.object(continuation, "V1C1_PINS", pins),
                patch.object(
                    continuation,
                    "V1C1_ACCEPTANCE_RECEIPT",
                    root / "acceptance-does-not-exist.json",
                ),
            ):
                evidence = continuation.attest_v1c1_failure_files()
            self.assertTrue(evidence["failed_before_artifact_validation"])
            self.assertEqual(evidence["artifact_inventory_passes_completed"], 0)


class ContinuationAdoptionTests(unittest.TestCase):
    def _adopt(self, absence_side_effect=None):
        scheduler = scheduler_evidence()
        scheduler_raw = continuation.canonical_bytes(scheduler)
        expected = {
            "scheduler_evidence": scheduler,
            "scheduler_evidence_file": {
                "path": str(continuation.SCHEDULER_EVIDENCE),
                "sha256": hashlib.sha256(scheduler_raw).hexdigest(),
            },
        }
        expected_raw = continuation.canonical_bytes(expected)
        inventory = [{"array_index": value} for value in range(641)]
        absence = patch.object(
            continuation,
            "assert_output_targets_absent",
            side_effect=absence_side_effect,
        )
        with (
            patch.object(continuation.os.path, "lexists", return_value=False),
            patch.object(continuation, "verify_code_manifest"),
            patch.object(continuation, "read_canonical", return_value=(expected, expected_raw)),
            patch.object(continuation, "validate_acceptance_payload"),
            patch.object(continuation, "attest_original_runtime"),
            patch.object(continuation, "attest_v1c1_failure_files"),
            patch.object(
                continuation,
                "load_scheduler_evidence",
                return_value=(scheduler, scheduler_raw),
            ),
            absence as absence_mock,
            patch.object(
                continuation,
                "artifact_inventory_pass",
                side_effect=[inventory, copy.deepcopy(inventory)],
            ) as inventory_mock,
            patch.object(continuation, "validate_direct_terminals", return_value=[]),
            patch.object(continuation, "build_acceptance", return_value=expected),
            patch.object(continuation, "publish_exclusive") as publish_mock,
            patch.object(
                continuation,
                "run_command",
                side_effect=AssertionError("compute adoption queried the scheduler"),
            ) as scheduler_mock,
        ):
            caught = None
            try:
                result = continuation.adopt(
                    code_manifest_sha256="1" * 64,
                    expected_acceptance_path=continuation.EXPECTED_ACCEPTANCE,
                    expected_acceptance_sha256=hashlib.sha256(expected_raw).hexdigest(),
                    scheduler_evidence_path=continuation.SCHEDULER_EVIDENCE,
                    scheduler_evidence_sha256=hashlib.sha256(scheduler_raw).hexdigest(),
                    acceptance_receipt_path=continuation.ACCEPTANCE_RECEIPT,
                )
            except BaseException as exc:  # exercised by the late-appearance test
                result = None
                caught = exc
        return (
            result,
            absence_mock,
            inventory_mock,
            publish_mock,
            scheduler_mock,
            caught,
        )

    def test_adopt_never_queries_scheduler_and_runs_exactly_two_inventory_passes(self):
        result, absence, inventory, publish, scheduler, caught = self._adopt()
        self.assertIsNone(caught)
        self.assertEqual(result["scheduler_evidence"]["capture_host"], "headnode")
        scheduler.assert_not_called()
        self.assertEqual(inventory.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in absence.call_args_list],
            [
                "before_adoption_validation",
                "immediately_before_acceptance_publication",
            ],
        )
        publish.assert_called_once()

    def test_adopt_does_not_publish_when_second_absence_check_fails(self):
        _, absence, inventory, publish, scheduler, caught = self._adopt(
            absence_side_effect=[{}, FileExistsError("injected late output")]
        )
        self.assertIsInstance(caught, FileExistsError)
        self.assertIn("injected late output", str(caught))
        scheduler.assert_not_called()
        self.assertEqual(inventory.call_count, 2)
        self.assertEqual(absence.call_count, 2)
        publish.assert_not_called()

    def test_acceptance_rejects_scheduler_origin_or_compute_query_count_drift(self):
        scheduler = scheduler_evidence()
        inventory = [{"array_index": value} for value in range(641)]
        with (
            patch.object(
                continuation,
                "attest_v1c1_failure_files",
                return_value=predecessor_evidence(),
            ),
            patch.object(continuation, "validate_task438_again", return_value={}),
        ):
            payload = continuation.build_acceptance(
                inventory,
                [{"cohort": str(value)} for value in range(4)],
                scheduler,
            )
        continuation.validate_acceptance_payload(payload)
        for key, value in (
            ("scheduler_evidence_origin", "compute_node_query"),
            ("compute_node_scheduler_queries", 1),
        ):
            changed = copy.deepcopy(payload)
            changed[key] = value
            core = {name: item for name, item in changed.items() if name != "acceptance_sha256"}
            changed["acceptance_sha256"] = continuation.canonical_sha256(core)
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_acceptance_payload(changed)

    def test_nonzero_sbatch_writes_invalid_terminal_receipt_and_raises(self):
        scheduler = scheduler_evidence()
        scheduler_raw = continuation.canonical_bytes(scheduler)
        scheduler_sha = hashlib.sha256(scheduler_raw).hexdigest()
        expected = {"scheduler_evidence": scheduler}
        expected_raw = continuation.canonical_bytes(expected)
        manifest_sha = "1" * 64
        acceptance_sha = hashlib.sha256(expected_raw).hexdigest()
        audit = {
            "code_manifest": {
                "path": str(continuation.CODE_MANIFEST),
                "sha256": manifest_sha,
            },
            "expected_acceptance": {
                "path": str(continuation.EXPECTED_ACCEPTANCE),
                "sha256": acceptance_sha,
            },
            "scheduler_evidence": scheduler,
            "scheduler_evidence_file": {
                "path": str(continuation.SCHEDULER_EVIDENCE),
                "sha256": scheduler_sha,
            },
            "execution_contract_sha256": "2" * 64,
        }
        published: list[tuple[Path, dict]] = []

        def capture(path, payload):
            published.append((Path(path), json.loads(payload)))

        def sbatch_failure(argv):
            self.assertEqual(argv[0], "sbatch")
            return CommandResult(1, "", "injected sbatch failure")

        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "logs"
            with (
                patch.object(continuation, "LOG_DIR", log_dir),
                patch.object(continuation, "load_dry_run", return_value=(audit, b"")),
                patch.object(continuation, "verify_code_manifest"),
                patch.object(continuation, "attest_original_runtime"),
                patch.object(continuation, "attest_v1c1_failure_files"),
                patch.object(continuation, "check_absence", return_value={}),
                patch.object(continuation, "query_duplicate_names", return_value=()),
                patch.object(
                    continuation,
                    "load_scheduler_evidence",
                    return_value=(scheduler, scheduler_raw),
                ),
                patch.object(
                    continuation, "capture_scheduler_evidence", return_value=scheduler
                ),
                patch.object(
                    continuation, "read_canonical", return_value=(expected, expected_raw)
                ),
                patch.object(continuation, "artifact_inventory_pass", return_value=[]),
                patch.object(continuation, "validate_direct_terminals", return_value=[]),
                patch.object(continuation, "build_acceptance", return_value=expected),
                patch.object(continuation, "publish_exclusive", side_effect=capture),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "terminal invalid jobs receipt was written"
                ):
                    continuation.launch(
                        dry_run_path=continuation.DRY_RUN_AUDIT,
                        dry_run_sha256="3" * 64,
                        runner=sbatch_failure,
                    )

        terminal = [payload for path, payload in published if path == continuation.JOBS_RECEIPT]
        self.assertEqual(len(terminal), 1)
        self.assertFalse(terminal[0]["valid"])
        self.assertEqual(terminal[0]["n_submitted"], 0)
        self.assertEqual(terminal[0]["failure"]["error_type"], "SbatchNonzero")


if __name__ == "__main__":
    unittest.main()
