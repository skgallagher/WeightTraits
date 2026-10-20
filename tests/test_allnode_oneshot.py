from __future__ import annotations

import hashlib
import json
import unittest

try:
    from test_allnode_audit import kwargs as audit_kwargs
except ModuleNotFoundError:  # pragma: no cover - package-style invocation
    from test_allnode_audit import kwargs as audit_kwargs

from weighttraits.behavior.allnode.audit import build_dry_run_audit_bundle
from weighttraits.behavior.allnode.oneshot import (
    OneShotRefusal,
    execute_one_shot,
    read_receipt_bytes,
)


class Harness:
    def __init__(self, *, fail_index: int | None = None) -> None:
        self.paths: dict[str, bytes] = {}
        self.fail_index = fail_index
        self.attested: list[str] = []
        self.submissions: list[tuple[str, tuple[str, ...]]] = []
        self.intent_published = False

    def exists(self, path: str) -> bool:
        return path in self.paths

    def create_intent(self, path: str, payload: bytes):
        if path in self.paths:
            raise FileExistsError(path)
        self.paths[path] = payload
        self.intent_published = True

    def attest(self, pin) -> bool:
        self.attested.append(pin.path)
        return True

    def submit(self, job, dependency_ids):
        self.assert_intent()
        index = len(self.submissions)
        self.submissions.append((job.logical_id, dependency_ids))
        if self.fail_index == index:
            raise RuntimeError(f"planned failure {index}")
        return str(9001 + index)

    def assert_intent(self):
        if not self.intent_published:
            raise AssertionError("submission occurred before immutable intent")


def bundle():
    return build_dry_run_audit_bundle(**audit_kwargs())


def run(harness: Harness, *, names=()):
    return execute_one_shot(
        bundle(),
        exists=harness.exists,
        create_exclusive_intent=harness.create_intent,
        attest_pin=harness.attest,
        submit=harness.submit,
        existing_scheduler_names=names,
    )


class OneShotTests(unittest.TestCase):
    def test_success_publishes_intent_first_and_records_exact_12_job_dag(self) -> None:
        harness = Harness()
        raw = run(harness)
        receipt = read_receipt_bytes(raw)
        self.assertTrue(receipt["valid"])
        self.assertTrue(receipt["terminal"])
        self.assertEqual((receipt["n_planned"], receipt["n_submitted"]), (12, 12))
        self.assertFalse(receipt["partial"])
        self.assertIsNone(receipt["failure"])
        self.assertEqual(receipt["unsubmitted_suffix"], [])
        self.assertEqual(len(receipt["planned_jobs"]), 12)
        self.assertEqual(len(receipt["submitted_prefix"]), 12)
        self.assertTrue(receipt["no_automatic_recovery"])
        self.assertFalse(receipt["retry_allowed"])
        self.assertFalse(receipt["cancel_on_failure"])
        self.assertFalse(receipt["resubmission_allowed"])
        self.assertGreater(len(harness.attested), 12)

        expected = [
            ("direct_llama_full", ()),
            ("direct_llama_notrans", ()),
            ("direct_flan_full", ()),
            ("direct_flan_notrans", ()),
            ("infer_llama_full", ()),
            ("semantic_llama_full", ("9001", "9005")),
            ("infer_llama_notrans", ("9006",)),
            ("semantic_llama_notrans", ("9002", "9007")),
            ("infer_flan_full", ("9008",)),
            ("semantic_flan_full", ("9003", "9009")),
            ("infer_flan_notrans", ("9010",)),
            ("semantic_flan_notrans", ("9004", "9011")),
        ]
        self.assertEqual(harness.submissions, expected)
        submitted = receipt["submitted_prefix"]
        self.assertEqual(submitted[5]["dependency_expression"], "afterok:9001:9005")
        self.assertEqual(submitted[11]["dependency_expression"], "afterok:9004:9011")
        self.assertEqual(submitted[4]["resources"]["array"], {"start": 0, "end": 640, "max_concurrent": 4})
        self.assertEqual(submitted[4]["route"]["account"], "statds")
        self.assertIn("sha256", submitted[4]["config"])
        self.assertEqual(receipt["code_manifest"], bundle().gate_contract.code_manifest.to_dict())
        self.assertEqual(receipt["audit_boundary"], bundle().audit_boundary.to_dict())

        intent_bytes = harness.paths[bundle().gate_contract.intent_path]
        self.assertEqual(hashlib.sha256(intent_bytes).hexdigest(), receipt["intent"]["sha256"])
        intent = json.loads(intent_bytes)
        self.assertEqual(intent["n_planned"], 12)
        self.assertEqual(intent["planned_jobs"], receipt["planned_jobs"])

    def test_refuses_existing_state_and_duplicate_scheduler_name_before_intent(self) -> None:
        for path_getter in (
            lambda gate: gate.intent_path,
            lambda gate: gate.jobs_receipt_path,
            lambda gate: gate.log_dir,
            lambda gate: gate.jobs[0].output_dir,
        ):
            harness = Harness()
            target = path_getter(bundle().gate_contract)
            harness.paths[target] = b"existing"
            with self.subTest(target=target):
                with self.assertRaisesRegex(OneShotRefusal, "pre-existing"):
                    run(harness)
                self.assertFalse(harness.intent_published)
                self.assertEqual(harness.submissions, [])

        harness = Harness()
        collision = bundle().gate_contract.jobs[3].name
        with self.assertRaisesRegex(OneShotRefusal, "duplicate scheduler"):
            run(harness, names=("unrelated", collision))
        self.assertFalse(harness.intent_published)
        self.assertEqual(harness.submissions, [])

    def test_pin_failure_is_pre_intent_and_never_submits(self) -> None:
        harness = Harness()

        def reject(pin):
            harness.attested.append(pin.path)
            return False

        with self.assertRaisesRegex(OneShotRefusal, "pin attestation failed"):
            execute_one_shot(
                bundle(),
                exists=harness.exists,
                create_exclusive_intent=harness.create_intent,
                attest_pin=reject,
                submit=harness.submit,
                existing_scheduler_names=(),
            )
        self.assertFalse(harness.intent_published)
        self.assertEqual(harness.submissions, [])

    def test_failure_at_first_is_terminal_invalid_but_not_partial(self) -> None:
        receipt = read_receipt_bytes(run(Harness(fail_index=0)))
        self.assertFalse(receipt["valid"])
        self.assertTrue(receipt["terminal"])
        self.assertEqual(receipt["n_submitted"], 0)
        self.assertFalse(receipt["partial"])
        self.assertEqual(receipt["failure"]["plan_index"], 0)
        self.assertEqual(len(receipt["unsubmitted_suffix"]), 12)
        self.assertTrue(receipt["no_automatic_recovery"])

    def test_failure_at_middle_records_submitted_prefix_and_unsubmitted_suffix(self) -> None:
        harness = Harness(fail_index=6)
        receipt = read_receipt_bytes(run(harness))
        self.assertFalse(receipt["valid"])
        self.assertTrue(receipt["partial"])
        self.assertEqual(receipt["n_submitted"], 6)
        self.assertEqual(len(receipt["submitted_prefix"]), 6)
        self.assertEqual(receipt["failure"]["logical_id"], "infer_llama_notrans")
        self.assertEqual(receipt["unsubmitted_suffix"][0]["logical_id"], "infer_llama_notrans")
        self.assertEqual(len(receipt["unsubmitted_suffix"]), 6)
        self.assertEqual(len(harness.submissions), 7)  # failed callback called exactly once

    def test_failure_at_last_keeps_exact_eleven_job_prefix(self) -> None:
        harness = Harness(fail_index=11)
        receipt = read_receipt_bytes(run(harness))
        self.assertFalse(receipt["valid"])
        self.assertTrue(receipt["partial"])
        self.assertEqual(receipt["n_submitted"], 11)
        self.assertEqual(len(receipt["submitted_prefix"]), 11)
        self.assertEqual(
            [row["logical_id"] for row in receipt["unsubmitted_suffix"]],
            ["semantic_flan_notrans"],
        )
        self.assertEqual(len(harness.submissions), 12)

    def test_second_run_is_refused_by_exclusive_intent_state(self) -> None:
        harness = Harness()
        first = read_receipt_bytes(run(harness))
        self.assertTrue(first["valid"])
        calls = len(harness.submissions)
        with self.assertRaisesRegex(OneShotRefusal, "pre-existing"):
            run(harness)
        self.assertEqual(len(harness.submissions), calls)

    def test_invalid_or_duplicate_job_id_becomes_terminal_failure_without_retry(self) -> None:
        harness = Harness()

        def invalid(job, dependency_ids):
            harness.assert_intent()
            harness.submissions.append((job.logical_id, dependency_ids))
            return "not-a-job-id"

        raw = execute_one_shot(
            bundle(),
            exists=harness.exists,
            create_exclusive_intent=harness.create_intent,
            attest_pin=harness.attest,
            submit=invalid,
            existing_scheduler_names=(),
        )
        receipt = read_receipt_bytes(raw)
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["n_submitted"], 0)
        self.assertEqual(receipt["failure"]["error_type"], "ValueError")
        self.assertEqual(len(harness.submissions), 1)


if __name__ == "__main__":
    unittest.main()
