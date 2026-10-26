from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    from test_allnode_audit import (
        FROZEN,
        STAGE,
        kwargs as audit_kwargs,
        rebind_execution_contract,
    )
except ModuleNotFoundError:  # pragma: no cover
    from test_allnode_audit import (
        FROZEN,
        STAGE,
        kwargs as audit_kwargs,
        rebind_execution_contract,
    )

from weighttraits.behavior.allnode.audit import (
    build_dry_run_audit_bundle,
    serialize_audit_core,
)
from weighttraits.behavior.allnode.gate import Pin, canonical_json, canonical_sha256
from weighttraits.behavior.allnode.launch import (
    CommandResult,
    SACCT_COMMAND,
    SCHEDULER_JOB_NAMES,
    SQUEUE_COMMAND,
    launch_from_persisted_audit,
    load_persisted_dry_run,
    query_scheduler_names,
    sbatch_argv,
)
from weighttraits.behavior.allnode.oneshot import OneShotRefusal, read_receipt_bytes


def _rewrite(value, replacements):
    if isinstance(value, Pin):
        return replace(value, path=_rewrite(value.path, replacements))
    if isinstance(value, dict):
        return {key: _rewrite(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite(child, replacements) for child in value]
    if isinstance(value, tuple):
        return tuple(_rewrite(child, replacements) for child in value)
    if isinstance(value, str):
        result = value
        for old, new in replacements:
            result = result.replace(old, new)
        return result
    return value


class FakeRunner:
    def __init__(self, *, squeue="", sacct="", fail_sbatch=None, interrupt_sbatch=None):
        self.squeue = squeue
        self.sacct = sacct
        self.fail_sbatch = fail_sbatch
        self.interrupt_sbatch = interrupt_sbatch
        self.calls = []
        self.n_sbatch = 0

    def __call__(self, argv):
        command = tuple(argv)
        self.calls.append(command)
        if command == SQUEUE_COMMAND:
            return CommandResult(0, self.squeue, "")
        if command == SACCT_COMMAND:
            return CommandResult(0, self.sacct, "")
        if command[0] != "sbatch":
            raise AssertionError(command)
        index = self.n_sbatch
        self.n_sbatch += 1
        if self.interrupt_sbatch == index:
            raise KeyboardInterrupt(f"interrupt {index}")
        if self.fail_sbatch == index:
            return CommandResult(17, "", f"failed {index}")
        return CommandResult(0, f"{9100 + index};wright\n", "")


class LaunchFixture:
    def __init__(self, root: Path):
        self.stage = root / "stage"
        self.frozen = root / "frozen"
        self.stage.mkdir()
        self.frozen.mkdir()
        values = _rewrite(
            audit_kwargs(),
            ((STAGE, self.stage.as_posix()), (FROZEN, self.frozen.as_posix())),
        )
        matrix = values["matrix_plan"]
        for cohort, row in matrix["cohorts"].items():
            row["plan_sha256"] = canonical_sha256(
                {
                    "cohort_id": cohort,
                    "direct_job_logical_id": f"direct_{cohort}",
                    "output_root": row["output_root"],
                    "tree_ids": row["tree_ids"],
                    "topology_scope_sha256": matrix["topology_scope_sha256"],
                    "matrix_order_scope_sha256": matrix["matrix_order_scope_sha256"],
                }
            )
        # Rewriting the fixture roots changes paths inside the compact parsed
        # execution contract.  Rebind its immutable receipt only after every
        # path rewrite, just as production preflight does before validation.
        rebind_execution_contract(values)
        self.bundle = build_dry_run_audit_bundle(**values)
        self.audit_path = Path(self.bundle.audit_boundary.path)
        self.audit_path.parent.mkdir(parents=True)
        self.audit_path.write_bytes(serialize_audit_core(self.bundle.audit_core).encode())
        (self.stage / "runtime_receipts").mkdir(exist_ok=True)
        (self.stage / "slurm_logs").mkdir(exist_ok=True)
        for job in self.bundle.gate_contract.jobs:
            config = Path(job.config_path)
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("{}", encoding="utf-8")
            wrapper = Path(job.wrapper_path)
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.touch(exist_ok=True)
        self.digests = {}
        self._collect_pins(self.bundle.audit_core)
        self.digests[self.bundle.audit_boundary.path] = self.bundle.audit_boundary.sha256
        self.digests[
            self.bundle.gate_contract.code_manifest.path
        ] = self.bundle.gate_contract.code_manifest.sha256

    def _collect_pins(self, value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                self.digests[value["path"]] = value["sha256"]
            else:
                for child in value.values():
                    self._collect_pins(child)
        elif isinstance(value, list):
            for child in value:
                self._collect_pins(child)

    def digest(self, path):
        return self.digests[str(path)]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = LaunchFixture(Path(self.temporary.name))

    def tearDown(self):
        # Immutable receipts are deliberately mode 0444; restore test cleanup.
        for path in Path(self.temporary.name).rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        self.temporary.cleanup()

    def test_loads_exact_persisted_core_and_canonical_bundle(self):
        loaded = load_persisted_dry_run(self.fixture.audit_path)
        self.assertEqual(loaded.gate_contract, self.fixture.bundle.gate_contract)
        bundle_path = Path(self.temporary.name) / "bundle.json"
        bundle_path.write_text(canonical_json(self.fixture.bundle.to_dict()), encoding="utf-8")
        self.assertEqual(
            load_persisted_dry_run(bundle_path).audit_boundary,
            self.fixture.bundle.audit_boundary,
        )
        bundle_path.write_text(
            canonical_json(self.fixture.bundle.to_dict()) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(OneShotRefusal, "canonical"):
            load_persisted_dry_run(bundle_path)

    def test_rejects_noncanonical_or_wrong_location_core(self):
        core = self.fixture.audit_path.read_bytes()
        self.fixture.audit_path.write_bytes(core + b"\n")
        with self.assertRaisesRegex(OneShotRefusal, "canonical"):
            load_persisted_dry_run(self.fixture.audit_path)
        self.fixture.audit_path.write_bytes(core)
        other = Path(self.temporary.name) / "core.json"
        other.write_bytes(core)
        with self.assertRaisesRegex(OneShotRefusal, "unique pinned path"):
            load_persisted_dry_run(other)

    def test_queries_both_scheduler_sources_and_rejects_either_collision(self):
        expected_name = self.fixture.bundle.gate_contract.jobs[0].name
        for source in ("squeue", "sacct"):
            runner = FakeRunner(
                squeue=expected_name + "\n" if source == "squeue" else "",
                sacct=expected_name + "|\n" if source == "sacct" else "",
            )
            with self.subTest(source=source):
                with self.assertRaisesRegex(OneShotRefusal, "duplicate scheduler"):
                    launch_from_persisted_audit(
                        self.fixture.audit_path,
                        run_command=runner,
                        digest_path=self.fixture.digest,
                        runtime_attestor=lambda bundle: True,
                    )
                self.assertEqual(runner.calls, [SQUEUE_COMMAND, SACCT_COMMAND])
                self.assertFalse(Path(self.fixture.bundle.gate_contract.intent_path).exists())

    def test_query_failure_still_calls_both_and_never_publishes_intent(self):
        calls = []

        def failed(argv):
            calls.append(tuple(argv))
            code = 3 if tuple(argv) == SQUEUE_COMMAND else 4
            return CommandResult(code, "", "unavailable")

        with self.assertRaisesRegex(OneShotRefusal, "squeue.*sacct"):
            launch_from_persisted_audit(
                self.fixture.audit_path,
                run_command=failed,
                digest_path=self.fixture.digest,
                runtime_attestor=lambda bundle: True,
            )
        self.assertEqual(calls, [SQUEUE_COMMAND, SACCT_COMMAND])
        self.assertFalse(Path(self.fixture.bundle.gate_contract.intent_path).exists())

    def test_runtime_provenance_failure_precedes_scheduler_and_intent(self):
        runner = FakeRunner()

        def fail(_bundle):
            raise ValueError("manifest entry drift")

        with self.assertRaisesRegex(OneShotRefusal, "runtime provenance"):
            launch_from_persisted_audit(
                self.fixture.audit_path,
                run_command=runner,
                digest_path=self.fixture.digest,
                runtime_attestor=fail,
            )
        self.assertEqual(runner.calls, [])
        self.assertFalse(Path(self.fixture.bundle.gate_contract.intent_path).exists())

    def test_success_submits_exact_route_resources_environment_logs_and_dag(self):
        runner = FakeRunner()
        raw = launch_from_persisted_audit(
            self.fixture.audit_path,
            run_command=runner,
            digest_path=self.fixture.digest,
            runtime_attestor=lambda bundle: True,
        )
        receipt = read_receipt_bytes(raw)
        self.assertTrue(receipt["valid"])
        self.assertEqual((receipt["n_planned"], receipt["n_submitted"]), (12, 12))
        self.assertEqual(runner.calls[:2], [SQUEUE_COMMAND, SACCT_COMMAND])
        self.assertEqual(len(runner.calls), 14)
        gate = self.fixture.bundle.gate_contract
        ids = {}
        for index, (job, argv) in enumerate(zip(gate.jobs, runner.calls[2:])):
            deps = tuple(ids[value] for value in job.dependencies)
            expected = sbatch_argv(
                job,
                deps,
                config=Pin(job.config_path, self.fixture.digests[job.config_path]),
                log_dir=gate.log_dir,
                gate=gate,
            )
            self.assertEqual(argv, expected)
            self.assertEqual(argv.count("--parsable"), 1)
            self.assertIn("--account=statds", argv)
            self.assertIn("--partition=all", argv)
            self.assertIn("--qos=normal", argv)
            self.assertIn("--nodelist=n03", argv)
            self.assertIn("--exclude=n01", argv)
            self.assertIn("--no-requeue", argv)
            self.assertTrue(any(value.startswith("--output=") for value in argv))
            self.assertTrue(any(value.startswith("--error=") for value in argv))
            exports = [value for value in argv if value.startswith("--export=")]
            self.assertEqual(len(exports), 1)
            self.assertNotIn("OVERRIDE_MAX_STEPS", exports[0])
            for key in (
                "WT_ALLNODE_SOURCE_RESOLUTION",
                "WT_ALLNODE_CODE_MANIFEST",
                "WT_DOWNSTREAM_CODE_MANIFEST",
                "WT_ALLNODE_PRODUCTION_INTERPRETER",
                "WT_ALLNODE_PYTHON_VERSION",
            ):
                self.assertIn(key + "=", exports[0])
            if job.resources.gpu_count:
                self.assertIn("--gres=gpu:nvidia-l40:1", argv)
            else:
                self.assertFalse(any(value.startswith("--gres=") for value in argv))
            if job.resources.array is not None:
                self.assertIn("--array=0-640%4", argv)
            ids[job.logical_id] = str(9100 + index)
        self.assertTrue(Path(gate.intent_path).is_file())
        self.assertTrue(Path(gate.jobs_receipt_path).is_file())
        events = Path(gate.log_dir) / "launch_events"
        self.assertEqual(len(list(events.glob("*_attempt.json"))), 12)
        self.assertEqual(len(list(events.glob("*_result.json"))), 12)

    def test_pin_drift_is_pre_intent_and_no_log_or_submit(self):
        runner = FakeRunner()

        def drift(path):
            return "0" * 64

        with self.assertRaisesRegex(OneShotRefusal, "pin attestation"):
            launch_from_persisted_audit(
                self.fixture.audit_path,
                run_command=runner,
                digest_path=drift,
                runtime_attestor=lambda bundle: True,
            )
        gate = self.fixture.bundle.gate_contract
        self.assertEqual(runner.calls, [SQUEUE_COMMAND, SACCT_COMMAND])
        self.assertFalse(Path(gate.intent_path).exists())
        self.assertFalse(Path(gate.log_dir).exists())

    def test_sbatch_failure_is_terminal_partial_and_never_retried(self):
        runner = FakeRunner(fail_sbatch=6)
        raw = launch_from_persisted_audit(
            self.fixture.audit_path,
            run_command=runner,
            digest_path=self.fixture.digest,
            runtime_attestor=lambda bundle: True,
        )
        receipt = read_receipt_bytes(raw)
        gate = self.fixture.bundle.gate_contract
        self.assertFalse(receipt["valid"])
        self.assertTrue(receipt["terminal"])
        self.assertTrue(receipt["partial"])
        self.assertEqual(receipt["n_submitted"], 6)
        self.assertEqual(runner.n_sbatch, 7)
        self.assertTrue(Path(gate.jobs_receipt_path).exists())
        events = Path(gate.log_dir) / "launch_events"
        self.assertTrue((events / "06_attempt.json").exists())
        self.assertTrue((events / "06_failure.json").exists())
        self.assertFalse((events / "06_result.json").exists())

    def test_keyboard_interrupt_persists_crash_adjacent_unknown_state_receipt(self):
        runner = FakeRunner(interrupt_sbatch=2)
        with self.assertRaises(KeyboardInterrupt):
            launch_from_persisted_audit(
                self.fixture.audit_path,
                run_command=runner,
                digest_path=self.fixture.digest,
                runtime_attestor=lambda bundle: True,
            )
        gate = self.fixture.bundle.gate_contract
        receipt = read_receipt_bytes(Path(gate.jobs_receipt_path).read_bytes())
        self.assertFalse(receipt["valid"])
        self.assertTrue(receipt["terminal"])
        self.assertEqual(receipt["n_submitted"], 2)
        self.assertTrue(receipt["failure"]["crash_adjacent"])
        self.assertTrue(receipt["failure"]["scheduler_state_may_be_unknown"])
        self.assertEqual(runner.n_sbatch, 3)

    def test_second_invocation_is_refused_without_scheduler_resubmission(self):
        first = FakeRunner()
        launch_from_persisted_audit(
            self.fixture.audit_path,
            run_command=first,
            digest_path=self.fixture.digest,
            runtime_attestor=lambda bundle: True,
        )
        second = FakeRunner()
        with self.assertRaisesRegex(OneShotRefusal, "pre-existing"):
            launch_from_persisted_audit(
                self.fixture.audit_path,
                run_command=second,
                digest_path=self.fixture.digest,
                runtime_attestor=lambda bundle: True,
            )
        self.assertEqual(second.calls, [SQUEUE_COMMAND, SACCT_COMMAND])
        self.assertEqual(second.n_sbatch, 0)


class SchedulerParsingTests(unittest.TestCase):
    def test_scheduler_queries_are_bounded_to_exact_twelve_names(self):
        expected_filter = "--name=" + ",".join(SCHEDULER_JOB_NAMES)
        self.assertEqual(len(SCHEDULER_JOB_NAMES), 12)
        self.assertEqual(len(set(SCHEDULER_JOB_NAMES)), 12)
        self.assertIn(expected_filter, SQUEUE_COMMAND)
        self.assertIn(expected_filter, SACCT_COMMAND)
        self.assertIn("--starttime=1970-01-01", SACCT_COMMAND)
        self.assertFalse(any("*" in value for value in SQUEUE_COMMAND + SACCT_COMMAND))

    def test_scheduler_query_rejects_scope_drift_before_running(self):
        runner = FakeRunner()
        for names in (
            SCHEDULER_JOB_NAMES[:-1],
            (*SCHEDULER_JOB_NAMES[:-1], SCHEDULER_JOB_NAMES[0]),
            "not-a-sequence",
        ):
            with self.subTest(names=names):
                with self.assertRaisesRegex(
                    OneShotRefusal, "sequence|exact launch scope"
                ):
                    query_scheduler_names(runner, names)
        self.assertEqual(runner.calls, [])

    def test_scheduler_query_rejects_unexpected_filtered_result(self):
        runner = FakeRunner(squeue="outside-name\n")
        with self.assertRaisesRegex(OneShotRefusal, "outside exact launch scope"):
            query_scheduler_names(runner)

    def test_scheduler_query_retains_bounded_output_guard(self):
        repeated = (SCHEDULER_JOB_NAMES[0] + "\n") * 3000
        runner = FakeRunner(squeue=repeated)
        with self.assertRaisesRegex(OneShotRefusal, "bounded output limit"):
            query_scheduler_names(runner)
        self.assertEqual(runner.calls, [SQUEUE_COMMAND, SACCT_COMMAND])

    def test_scheduler_query_rejects_truncated_or_malformed_names(self):
        runner = FakeRunner(sacct="name|extra|\n")
        with self.assertRaisesRegex(OneShotRefusal, "invalid scheduler"):
            query_scheduler_names(runner)


if __name__ == "__main__":
    unittest.main()
