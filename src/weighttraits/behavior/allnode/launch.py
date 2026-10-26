"""Concrete, irreversible Slurm boundary for the isolated all-node study.

The validation and state-machine modules in this package are deliberately
pure.  This module is the one narrow adapter that is allowed to touch the
filesystem and invoke ``squeue``, ``sacct``, and ``sbatch``.  It has no retry,
cancel, requeue, recovery, or resubmission entry point.

The persisted dry-run artifact is normally the canonical audit-core JSON at
the path that the core itself implies.  A canonical bundle JSON is also
accepted, but its separately pinned audit-core file must still exist and must
contain the exact canonical bytes embedded in the bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence, Union
import uuid

from .audit import (
    BUNDLE_SCHEMA,
    CORE_SCHEMA,
    DryRunAuditBundle,
    serialize_audit_core,
    validate_audit_core,
    validate_dry_run_audit_bundle,
)
from .gate import (
    COHORTS,
    GateContract,
    JobSpec,
    Pin,
    build_gate_contract,
    canonical_json,
)
from .loader import sha256_path
from .oneshot import (
    INTENT_SCHEMA,
    RECEIPT_SCHEMA,
    OneShotRefusal,
    execute_one_shot,
    read_receipt_bytes,
)
from .provenance import attest_runtime_provenance


MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
COMMAND_TIMEOUT_SECONDS = 30.0
ATTEMPT_SCHEMA = "weighttraits.allnode_submit_attempt.v1"
RESULT_SCHEMA = "weighttraits.allnode_submit_result.v1"
FAILURE_SCHEMA = "weighttraits.allnode_submit_failure.v1"
_JOB_ID = re.compile(r"[0-9]+")

SCHEDULER_JOB_NAMES = tuple(
    [f"wt-an-v1-direct-{cohort.replace('_', '-')}" for cohort in COHORTS]
    + [
        f"wt-an-v1-{prefix}-{cohort.replace('_', '-')}"
        for cohort in COHORTS
        for prefix in ("infer", "semantic")
    ]
)
_SCHEDULER_NAME_FILTER = ",".join(SCHEDULER_JOB_NAMES)
SQUEUE_COMMAND = (
    "squeue",
    "--noheader",
    f"--name={_SCHEDULER_NAME_FILTER}",
    "--format=%j",
)
SACCT_COMMAND = (
    "sacct",
    "-X",
    "--noheader",
    "--starttime=1970-01-01",
    f"--name={_SCHEDULER_NAME_FILTER}",
    "--format=JobName%128",
    "--parsable2",
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], object]
DigestPath = Callable[[Union[str, Path]], str]
RuntimeAttestor = Callable[[DryRunAuditBundle], object]


def launch_from_persisted_audit(
    artifact_path: str | Path,
    *,
    run_command: CommandRunner | None = None,
    digest_path: DigestPath | None = None,
    runtime_attestor: RuntimeAttestor | None = None,
) -> bytes:
    """Validate, lock, and submit the exact twelve-job all-node DAG once.

    All failures before immutable intent publication raise
    :class:`OneShotRefusal`.  After intent publication, ordinary submission
    failures produce and persist a terminal invalid jobs receipt.  An
    interruption such as ``KeyboardInterrupt`` is re-raised only after a
    best-effort terminal crash receipt has been published.
    """

    bundle = load_persisted_dry_run(artifact_path)
    gate = bundle.gate_contract
    runner = _subprocess_runner if run_command is None else run_command
    digest = sha256_path if digest_path is None else digest_path
    if not callable(runner):
        raise ValueError("run_command must be callable")
    if not callable(digest):
        raise ValueError("digest_path must be callable")
    attestor = _attest_bundle_runtime if runtime_attestor is None else runtime_attestor
    if not callable(attestor):
        raise ValueError("runtime_attestor must be callable")
    try:
        attestor(bundle)
    except Exception as exc:
        raise OneShotRefusal(f"runtime provenance attestation failed: {exc}") from exc

    # Both scheduler sources are queried exactly once.  We intentionally do
    # not short-circuit after the first result so a pre-intent audit always has
    # evidence from both the live queue and accounting history.
    existing_names = query_scheduler_names(
        runner, tuple(job.name for job in gate.jobs)
    )
    context = _SubmissionContext(bundle=bundle, runner=runner)
    intent_was_published = False

    def create_intent(path: str, payload: bytes) -> bool:
        nonlocal intent_was_published
        _publish_exclusive(Path(path), payload)
        context.intent_bytes = payload
        intent_was_published = True
        return True

    def attest(pin: Pin) -> bool:
        try:
            observed = digest(pin.path)
        except Exception:
            return False
        return observed == pin.sha256

    try:
        receipt = execute_one_shot(
            bundle,
            exists=lambda path: _lexists(Path(path)),
            create_exclusive_intent=create_intent,
            attest_pin=attest,
            submit=context.submit,
            existing_scheduler_names=existing_names,
        )
    except BaseException as exc:
        if intent_was_published:
            # This is deliberately best effort: if the disk itself is failing,
            # the immutable intent and per-attempt journal remain the durable
            # evidence.  There is no second scheduler call and no recovery.
            try:
                crash = context.crash_receipt(exc)
                _publish_exclusive(Path(gate.jobs_receipt_path), crash)
            except BaseException:
                pass
        raise

    # Validate the state machine's terminal bytes again at the concrete
    # boundary before making them visible.
    read_receipt_bytes(receipt)
    _publish_exclusive(Path(gate.jobs_receipt_path), receipt)
    return receipt


def load_persisted_dry_run(artifact_path: str | Path) -> DryRunAuditBundle:
    """Load canonical persisted core or bundle bytes and rederive the gate."""

    path = _canonical_existing_file(artifact_path, "dry-run artifact")
    raw = _read_bounded(path)
    payload = _json_mapping(raw, "dry-run artifact")
    schema = payload.get("schema")
    if schema == CORE_SCHEMA:
        core = validate_audit_core(payload)
        expected = serialize_audit_core(core).encode("utf-8")
        if raw != expected:
            raise OneShotRefusal("persisted audit core is not exact canonical bytes")
        expected_path = (
            Path(core["deployment_root"])
            / "runtime_receipts"
            / "ALLNODE_V1_DRY_RUN_AUDIT.json"
        )
        if path != expected_path:
            raise OneShotRefusal("persisted audit core is not at its unique pinned path")
        boundary = Pin(str(path), hashlib.sha256(raw).hexdigest())
        code = _pin(core["code_manifest"], "code_manifest")
        gate = build_gate_contract(
            stage_root=core["stage_root"],
            frozen_stage=core["frozen_stage"],
            code_manifest=code,
            source_resolution=_pin(core["source_resolution"], "source_resolution"),
            source_scope_sha256=core["execution_contract"]["payload"][
                "source_scope_sha256"
            ],
            downstream_code_manifest=_pin(
                core["downstream_code_manifest"], "downstream_code_manifest"
            ),
            python_version=core["test_attestation"]["python_version"],
            dry_run_audit=boundary,
        )
        return validate_dry_run_audit_bundle(
            {
                "schema": BUNDLE_SCHEMA,
                "valid": True,
                "audit_core": core,
                "audit_boundary": boundary.to_dict(),
                "gate_contract": gate.to_dict(),
                "gate_contract_sha256": gate.sha256,
            }
        )
    if schema == BUNDLE_SCHEMA:
        canonical = canonical_json(payload).encode("utf-8")
        if raw != canonical:
            raise OneShotRefusal("persisted audit bundle is not exact canonical bytes")
        try:
            bundle = validate_dry_run_audit_bundle(payload)
        except Exception as exc:
            raise OneShotRefusal(f"persisted audit bundle validation failed: {exc}") from exc
        core_path = _canonical_existing_file(
            bundle.audit_boundary.path, "pinned audit core"
        )
        core_bytes = _read_bounded(core_path)
        expected_core = serialize_audit_core(bundle.audit_core).encode("utf-8")
        if core_bytes != expected_core:
            raise OneShotRefusal("pinned audit core differs from bundle canonical bytes")
        if hashlib.sha256(core_bytes).hexdigest() != bundle.audit_boundary.sha256:
            raise OneShotRefusal("pinned audit core SHA-256 differs from bundle boundary")
        return bundle
    raise OneShotRefusal("persisted dry-run artifact has an unsupported schema")


def query_scheduler_names(
    run_command: CommandRunner,
    planned_names: Sequence[str] = SCHEDULER_JOB_NAMES,
) -> tuple[str, ...]:
    """Return planned names observed in live and all-history Slurm state."""

    if isinstance(planned_names, (str, bytes)):
        raise OneShotRefusal("planned scheduler names must be a sequence")
    names_to_query = tuple(planned_names)
    if names_to_query != SCHEDULER_JOB_NAMES:
        raise OneShotRefusal("planned scheduler names differ from exact launch scope")

    results: list[tuple[str, CommandResult]] = []
    for label, command in (("squeue", SQUEUE_COMMAND), ("sacct", SACCT_COMMAND)):
        try:
            result = _normalize_result(run_command(command), label)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            result = CommandResult(255, "", f"{type(exc).__name__}: {exc}")
        results.append((label, result))
    failures = [
        f"{label} rc={result.returncode} stderr={result.stderr!r}"
        for label, result in results
        if result.returncode != 0
    ]
    if failures:
        raise OneShotRefusal("scheduler duplicate-name queries failed: " + "; ".join(failures))
    names: set[str] = set()
    for label, result in results:
        names.update(_parse_scheduler_names(result.stdout, label))
    unexpected = names.difference(names_to_query)
    if unexpected:
        raise OneShotRefusal(
            f"scheduler query returned names outside exact launch scope: {sorted(unexpected)}"
        )
    return tuple(sorted(names))


def _attest_bundle_runtime(bundle: DryRunAuditBundle) -> Mapping[str, Any]:
    """Physically replay all runtime provenance before any scheduler query."""

    gate = bundle.gate_contract
    return attest_runtime_provenance(
        phase="launch",
        stage_root=gate.stage_root,
        deployment_root=gate.deployment_root,
        source_resolution_path=gate.source_resolution.path,
        source_resolution_sha256=gate.source_resolution.sha256,
        source_scope_sha256=gate.source_scope_sha256,
        allnode_code_manifest_path=gate.code_manifest.path,
        allnode_code_manifest_sha256=gate.code_manifest.sha256,
        downstream_code_manifest_path=gate.downstream_code_manifest.path,
        downstream_code_manifest_sha256=gate.downstream_code_manifest.sha256,
        expected_python_version=gate.python_version,
    )


def sbatch_argv(
    job: JobSpec,
    dependency_ids: Sequence[str],
    *,
    config: Pin,
    log_dir: str,
    gate: GateContract,
) -> tuple[str, ...]:
    """Translate one already-validated ``JobSpec`` without implicit defaults."""

    if config.path != job.config_path:
        raise ValueError("job config pin path differs from JobSpec")
    if job not in gate.jobs:
        raise ValueError("job is not a member of the exact GateContract")
    dependencies = tuple(_numeric_job_id(value) for value in dependency_ids)
    if len(dependencies) != len(job.dependencies):
        raise ValueError("dependency job-ID count differs from JobSpec dependencies")
    route = job.route
    resources = job.resources
    log_root = Path(log_dir)
    if not log_root.is_absolute() or str(log_root) != log_dir:
        raise ValueError("log_dir must be a canonical absolute path")
    token = "%A_%a" if resources.array is not None else "%j"
    exported = {
        "WT_ALLNODE_COHORT_KEY": job.cohort_id,
        "WT_ALLNODE_JOB_CONFIG": config.path,
        "WT_ALLNODE_JOB_CONFIG_SHA256": config.sha256,
        "WT_ALLNODE_KIND": job.kind,
        "WT_ALLNODE_LOGICAL_ID": job.logical_id,
        "WT_ALLNODE_OUTPUT_DIR": job.output_dir,
        "WT_ALLNODE_STAGE_ROOT": gate.stage_root,
        "WT_ALLNODE_DEPLOYMENT_ROOT": gate.deployment_root,
        "WT_ALLNODE_SOURCE_RESOLUTION": gate.source_resolution.path,
        "WT_ALLNODE_SOURCE_RESOLUTION_SHA256": gate.source_resolution.sha256,
        "WT_ALLNODE_SOURCE_SCOPE_SHA256": gate.source_scope_sha256,
        "WT_ALLNODE_CODE_MANIFEST": gate.code_manifest.path,
        "WT_ALLNODE_CODE_MANIFEST_SHA256": gate.code_manifest.sha256,
        "WT_DOWNSTREAM_CODE_MANIFEST": gate.downstream_code_manifest.path,
        "WT_DOWNSTREAM_CODE_MANIFEST_SHA256": gate.downstream_code_manifest.sha256,
        "WT_ALLNODE_PRODUCTION_INTERPRETER": gate.production_interpreter,
        "WT_ALLNODE_PYTHON_VERSION": gate.python_version,
    }
    export_arg = ",".join(f"{key}={_export_value(value)}" for key, value in exported.items())
    argv = [
        "sbatch",
        "--parsable",
        f"--job-name={job.name}",
        f"--account={route.account}",
        f"--partition={route.partition}",
        f"--qos={route.qos}",
        f"--nodelist={route.required_node}",
        f"--exclude={route.excluded_node}",
        "--no-requeue",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.memory_gib}G",
        f"--time={resources.time_limit}",
    ]
    if resources.gpu_count:
        if resources.gpu_type is None:
            raise ValueError("GPU job lacks an exact GPU type")
        argv.append(f"--gres=gpu:{resources.gpu_type}:{resources.gpu_count}")
    elif resources.gpu_type is not None:
        raise ValueError("CPU job unexpectedly declares a GPU type")
    if resources.array is not None:
        array = resources.array
        argv.append(f"--array={array.start}-{array.end}%{array.max_concurrent}")
    if dependencies:
        argv.append("--dependency=afterok:" + ":".join(dependencies))
    argv.extend(
        (
            f"--output={log_dir}/{job.logical_id}_{token}.out",
            f"--error={log_dir}/{job.logical_id}_{token}.err",
            f"--export={export_arg}",
            job.wrapper_path,
        )
    )
    return tuple(argv)


class _SubmissionContext:
    def __init__(self, *, bundle: DryRunAuditBundle, runner: CommandRunner) -> None:
        self.bundle = bundle
        self.runner = runner
        self.intent_bytes: bytes | None = None
        self._log_ready = False
        self._config_by_id = {
            logical_id: _pin(value, f"job_configs.{logical_id}")
            for logical_id, value in bundle.audit_core["job_configs"].items()
        }

    def submit(self, job: JobSpec, dependency_ids: tuple[str, ...]) -> str:
        self._ensure_log_dir()
        index = len(self._result_events())
        expected = self.bundle.gate_contract.jobs[index]
        if job != expected:
            raise ValueError("submission callback job differs from exact gate order")
        config = self._config_by_id[job.logical_id]
        argv = sbatch_argv(
            job,
            dependency_ids,
            config=config,
            log_dir=self.bundle.gate_contract.log_dir,
            gate=self.bundle.gate_contract,
        )
        attempt = {
            "schema": ATTEMPT_SCHEMA,
            "valid": True,
            "submission_id": self.bundle.gate_contract.submission_id,
            "gate_contract_sha256": self.bundle.gate_contract.sha256,
            "plan_index": index,
            "logical_id": job.logical_id,
            "argv": list(argv),
            "argv_sha256": hashlib.sha256("\0".join(argv).encode("utf-8")).hexdigest(),
            "dependency_job_ids": list(dependency_ids),
            "scheduler_invocations": 1,
            "retry_allowed": False,
        }
        self._event(index, "attempt", attempt)
        try:
            result = _normalize_result(self.runner(argv), "sbatch")
        except BaseException:
            # The durable attempt record intentionally remains without a
            # result.  It denotes scheduler state that may be unknown.
            raise
        if result.returncode != 0:
            failure = {
                "schema": FAILURE_SCHEMA,
                "valid": False,
                "terminal": True,
                "plan_index": index,
                "logical_id": job.logical_id,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "retry_allowed": False,
            }
            self._event(index, "failure", failure)
            raise RuntimeError(
                f"sbatch failed once for {job.logical_id}: rc={result.returncode}; "
                f"stderr={result.stderr!r}"
            )
        job_id = _parse_sbatch_job_id(result.stdout)
        event = {
            "schema": RESULT_SCHEMA,
            "valid": True,
            "terminal": True,
            "plan_index": index,
            "logical_id": job.logical_id,
            "job_id": job_id,
            "dependency_job_ids": list(dependency_ids),
            "returncode": 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "retry_allowed": False,
        }
        try:
            self._event(index, "result", event)
        except BaseException as exc:
            raise RuntimeError(
                f"sbatch returned job {job_id} for {job.logical_id}, but durable "
                f"result publication failed: {type(exc).__name__}: {exc}"
            ) from exc
        return job_id

    def crash_receipt(self, exc: BaseException) -> bytes:
        if self.intent_bytes is None:
            raise RuntimeError("cannot build crash receipt before intent")
        intent = _json_mapping(self.intent_bytes, "published intent")
        if intent.get("schema") != INTENT_SCHEMA:
            raise RuntimeError("published intent schema drift")
        planned = intent["planned_jobs"]
        results = self._result_events()
        submitted: list[dict[str, Any]] = []
        for index, event in enumerate(results):
            if event["plan_index"] != index or event["logical_id"] != planned[index]["logical_id"]:
                raise RuntimeError("durable result journal is not an exact prefix")
            dependencies = event["dependency_job_ids"]
            submitted.append(
                {
                    **planned[index],
                    "job_id": event["job_id"],
                    "dependency_job_ids": dependencies,
                    "dependency_expression": (
                        None if not dependencies else "afterok:" + ":".join(dependencies)
                    ),
                }
            )
        failed_index = len(results)
        failed_row = (
            planned[failed_index]
            if failed_index < len(planned)
            else {"logical_id": "terminal_publication", "name": "terminal_publication"}
        )
        failure = {
            "plan_index": failed_index,
            "logical_id": failed_row["logical_id"],
            "name": failed_row["name"],
            "error_type": type(exc).__name__,
            "message": str(exc),
            "crash_adjacent": True,
            "scheduler_state_may_be_unknown": self._attempt_exists_without_terminal(failed_index),
        }
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "valid": False,
            "terminal": True,
            "submission_id": self.bundle.gate_contract.submission_id,
            "intent": {
                "path": self.bundle.gate_contract.intent_path,
                "sha256": hashlib.sha256(self.intent_bytes).hexdigest(),
            },
            "gate_contract_sha256": self.bundle.gate_contract.sha256,
            "audit_boundary": self.bundle.audit_boundary.to_dict(),
            "audit_core_sha256": self.bundle.audit_boundary.sha256,
            "code_manifest": self.bundle.gate_contract.code_manifest.to_dict(),
            "scheduler_policy": self.bundle.gate_contract.to_dict()["scheduler_policy"],
            "n_planned": len(planned),
            "n_submitted": len(submitted),
            "partial": bool(submitted),
            "failure": failure,
            "planned_jobs": planned,
            "submitted_prefix": submitted,
            "unsubmitted_suffix": planned[failed_index:],
            "execution_overrides": {},
            "no_automatic_recovery": True,
            "retry_allowed": False,
            "cancel_on_failure": False,
            "resubmission_allowed": False,
        }
        return _canonical_bytes(receipt)

    def _ensure_log_dir(self) -> None:
        if self._log_ready:
            return
        if self.intent_bytes is None:
            raise AssertionError("log directory creation attempted before intent")
        root = Path(self.bundle.gate_contract.log_dir)
        parent = _canonical_existing_dir(root.parent, "log parent")
        if _lexists(root):
            raise FileExistsError(f"log directory unexpectedly exists: {root}")
        root.mkdir(mode=0o700)
        if root.parent != parent:
            raise AssertionError("log directory parent drift")
        events = root / "launch_events"
        events.mkdir(mode=0o700)
        self._log_ready = True

    def _event(self, index: int, kind: str, value: Mapping[str, Any]) -> None:
        if not self._log_ready:
            raise AssertionError("launch event attempted before log directory")
        if kind not in {"attempt", "result", "failure"}:
            raise ValueError("unsupported launch event kind")
        path = Path(self.bundle.gate_contract.log_dir) / "launch_events" / f"{index:02d}_{kind}.json"
        _publish_exclusive(path, _canonical_bytes(value))

    def _result_events(self) -> list[dict[str, Any]]:
        event_dir = Path(self.bundle.gate_contract.log_dir) / "launch_events"
        if not event_dir.is_dir() or event_dir.is_symlink():
            return []
        rows: list[dict[str, Any]] = []
        for index in range(len(self.bundle.gate_contract.jobs)):
            path = event_dir / f"{index:02d}_result.json"
            if not path.exists():
                break
            payload = _json_mapping(_read_bounded(path), f"result event {index}")
            if payload.get("schema") != RESULT_SCHEMA or payload.get("valid") is not True:
                raise RuntimeError(f"result event {index} schema drift")
            rows.append(dict(payload))
        return rows

    def _attempt_exists_without_terminal(self, index: int) -> bool:
        root = Path(self.bundle.gate_contract.log_dir) / "launch_events"
        return (
            (root / f"{index:02d}_attempt.json").exists()
            and not (root / f"{index:02d}_result.json").exists()
            and not (root / f"{index:02d}_failure.json").exists()
        )


def _subprocess_runner(argv: Sequence[str]) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        env={"LC_ALL": "C", "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _normalize_result(value: object, label: str) -> CommandResult:
    if isinstance(value, CommandResult):
        result = value
    elif all(hasattr(value, field) for field in ("returncode", "stdout", "stderr")):
        result = CommandResult(
            getattr(value, "returncode"), getattr(value, "stdout"), getattr(value, "stderr")
        )
    else:
        raise ValueError(f"{label} runner returned an unsupported result")
    if isinstance(result.returncode, bool) or not isinstance(result.returncode, int):
        raise ValueError(f"{label} returncode must be an integer")
    stdout = _command_text(result.stdout, f"{label} stdout")
    stderr = _command_text(result.stderr, f"{label} stderr")
    return CommandResult(result.returncode, stdout, stderr)


def _command_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    if "\x00" in value:
        raise ValueError(f"{label} contains NUL")
    if len(value.encode("utf-8")) > MAX_COMMAND_OUTPUT_BYTES:
        raise ValueError(f"{label} exceeds the bounded output limit")
    return value


def _parse_scheduler_names(stdout: str, label: str) -> set[str]:
    names: set[str] = set()
    for line in stdout.splitlines():
        value = line[:-1] if label == "sacct" and line.endswith("|") else line
        if not value:
            continue
        if not value.strip() == value or "|" in value or "\x00" in value:
            raise OneShotRefusal(f"{label} returned an invalid scheduler job name")
        names.add(value)
    return names


def _parse_sbatch_job_id(stdout: str) -> str:
    value = stdout.strip()
    if "\n" in value or "\r" in value:
        raise ValueError("sbatch --parsable returned multiple lines")
    job_id = value.split(";", 1)[0]
    return _numeric_job_id(job_id)


def _numeric_job_id(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError("scheduler job ID must be decimal")
    text = str(value)
    if _JOB_ID.fullmatch(text) is None:
        raise ValueError(f"scheduler job ID must be decimal: {value!r}")
    return text


def _export_value(value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Slurm export value must be nonempty trimmed text")
    if any(character in value for character in (",", "\n", "\r", "\x00")):
        raise ValueError("Slurm export value contains an unsafe character")
    return value


def _publish_exclusive(path: Path, payload: bytes) -> None:
    """Publish bounded bytes atomically without ever replacing a target."""

    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_JSON_BYTES:
        raise ValueError("exclusive payload must be nonempty bounded bytes")
    if not path.is_absolute() or str(path) != path.as_posix():
        raise ValueError("exclusive target must be a canonical absolute POSIX path")
    parent = _canonical_existing_dir(path.parent, "exclusive target parent")
    if _lexists(path):
        raise FileExistsError(path)
    temp = parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    linked = False
    try:
        fd = os.open(temp, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while publishing immutable artifact")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.close(fd)
        fd = None
        os.link(temp, path, follow_symlinks=False)
        linked = True
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if _lexists(temp):
            os.unlink(temp)
    if not linked or hashlib.sha256(_read_bounded(path)).digest() != hashlib.sha256(payload).digest():
        raise OSError(f"exclusive publication verification failed: {path}")


def _read_bounded(path: Path) -> bytes:
    canonical = _canonical_existing_file(path, "bounded input")
    size = canonical.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise OneShotRefusal(f"bounded input has invalid size: {canonical}")
    raw = canonical.read_bytes()
    if len(raw) != size:
        raise OneShotRefusal(f"bounded input changed while being read: {canonical}")
    return raw


def _canonical_existing_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or str(value) != value.as_posix():
        raise OneShotRefusal(f"{label} must be a canonical absolute POSIX path")
    if value.is_symlink() or not value.is_file():
        raise OneShotRefusal(f"{label} must be an existing non-symlink file: {value}")
    return value


def _canonical_existing_dir(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or str(value) != value.as_posix():
        raise OneShotRefusal(f"{label} must be a canonical absolute POSIX path")
    if value.is_symlink() or not value.is_dir():
        raise OneShotRefusal(f"{label} must be an existing non-symlink directory: {value}")
    return value


def _json_mapping(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OneShotRefusal(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise OneShotRefusal(f"{label} must be a JSON object")
    return value


def _pin(value: object, label: str) -> Pin:
    if isinstance(value, Pin):
        return value
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path+sha256 mapping")
    return Pin(value["path"], value["sha256"])


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(value)).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Irreversibly submit the validated allnode_v1 DAG exactly once."
    )
    parser.add_argument(
        "--audit",
        required=True,
        help="Canonical persisted allnode_v1 dry-run audit core or bundle JSON",
    )
    args = parser.parse_args(argv)
    receipt = launch_from_persisted_audit(args.audit)
    parsed = read_receipt_bytes(receipt)
    print(
        canonical_json(
            {
                "valid": parsed["valid"],
                "n_planned": parsed["n_planned"],
                "n_submitted": parsed["n_submitted"],
                "jobs_receipt": load_persisted_dry_run(args.audit).gate_contract.jobs_receipt_path,
            }
        )
    )
    return 0 if parsed["valid"] is True else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    sys.exit(main())


__all__ = [
    "CommandResult",
    "SACCT_COMMAND",
    "SCHEDULER_JOB_NAMES",
    "SQUEUE_COMMAND",
    "launch_from_persisted_audit",
    "load_persisted_dry_run",
    "main",
    "query_scheduler_names",
    "sbatch_argv",
]
