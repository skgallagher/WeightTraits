"""Pure, dependency-injected one-shot submission state machine.

This module never touches the filesystem and never imports a scheduler client.
It serializes an immutable intent, invokes caller-supplied callbacks in a
strict order, and returns immutable receipt bytes.  There is intentionally no
cancel, retry, recovery, or resubmission API.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

from .audit import DryRunAuditBundle, validate_dry_run_audit_bundle
from .gate import GateContract, JobSpec, Pin


INTENT_SCHEMA = "weighttraits.allnode_oneshot_intent.v1"
RECEIPT_SCHEMA = "weighttraits.allnode_oneshot_jobs.v1"
_JOB_ID = re.compile(r"[0-9]+")

Exists = Callable[[str], bool]
CreateExclusiveIntent = Callable[[str, bytes], object]
AttestPin = Callable[[Pin], bool]
Submit = Callable[[JobSpec, tuple[str, ...]], object]


class OneShotRefusal(RuntimeError):
    """Raised only before exclusive intent publication."""


def execute_one_shot(
    bundle: DryRunAuditBundle | Mapping[str, Any],
    *,
    exists: Exists,
    create_exclusive_intent: CreateExclusiveIntent,
    attest_pin: AttestPin,
    submit: Submit,
    existing_scheduler_names: Sequence[str],
) -> bytes:
    """Publish one intent, submit one exact DAG, and return receipt bytes.

    Pre-intent problems raise :class:`OneShotRefusal` and no submit callback is
    called.  Once intent is published, every submission failure is converted
    into a terminal invalid receipt.  No callback is ever retried.
    """

    validated = _bundle(bundle)
    gate = validated.gate_contract
    _validate_topological_contract_order(gate)
    _refuse_existing_state(gate, exists=exists)
    _refuse_scheduler_duplicates(gate, existing_scheduler_names)

    # This is deliberately the final validation step before intent creation.
    # Every unique pin is externally re-attested against live bytes once.
    for pin in _collect_pins(validated):
        try:
            attested = attest_pin(pin)
        except Exception as exc:
            raise OneShotRefusal(
                f"pin attestation callback failed before intent for {pin.path}: {exc}"
            ) from exc
        if attested is not True:
            raise OneShotRefusal(f"pin attestation failed before intent: {pin.path}")

    planned = _planned_jobs(validated)
    intent = {
        "schema": INTENT_SCHEMA,
        "valid": True,
        "terminal_lock": True,
        "submission_id": gate.submission_id,
        "gate_contract_sha256": gate.sha256,
        "audit_boundary": validated.audit_boundary.to_dict(),
        "audit_core_sha256": validated.audit_boundary.sha256,
        "code_manifest": gate.code_manifest.to_dict(),
        "scheduler_policy": gate.to_dict()["scheduler_policy"],
        "n_planned": len(planned),
        "planned_jobs": planned,
        "execution_overrides": {},
        "no_automatic_recovery": True,
        "retry_allowed": False,
        "cancel_on_failure": False,
        "resubmission_allowed": False,
    }
    intent_bytes = _json_bytes(intent)
    try:
        result = create_exclusive_intent(gate.intent_path, intent_bytes)
    except Exception as exc:
        raise OneShotRefusal(f"exclusive intent publication failed: {exc}") from exc
    if result is False:
        raise OneShotRefusal("exclusive intent publication callback returned false")

    submitted: list[dict[str, Any]] = []
    job_ids: dict[str, str] = {}
    failure: dict[str, Any] | None = None
    failed_index: int | None = None
    for index, (job, plan_row) in enumerate(zip(gate.jobs, planned)):
        dependency_ids = tuple(job_ids[dependency] for dependency in job.dependencies)
        try:
            raw_job_id = submit(job, dependency_ids)
            job_id = _validated_job_id(raw_job_id)
            if job_id in job_ids.values():
                raise ValueError(f"duplicate scheduler job id returned: {job_id}")
            dependency_expression = (
                None if not dependency_ids else "afterok:" + ":".join(dependency_ids)
            )
            record = {
                **plan_row,
                "job_id": job_id,
                "dependency_job_ids": list(dependency_ids),
                "dependency_expression": dependency_expression,
            }
            submitted.append(record)
            job_ids[job.logical_id] = job_id
        except Exception as exc:
            failed_index = index
            failure = {
                "plan_index": index,
                "logical_id": job.logical_id,
                "name": job.name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            break

    n_submitted = len(submitted)
    success = failure is None and n_submitted == len(planned)
    suffix_start = len(planned) if success else int(failed_index)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "valid": success,
        "terminal": True,
        "submission_id": gate.submission_id,
        "intent": {
            "path": gate.intent_path,
            "sha256": hashlib.sha256(intent_bytes).hexdigest(),
        },
        "gate_contract_sha256": gate.sha256,
        "audit_boundary": validated.audit_boundary.to_dict(),
        "audit_core_sha256": validated.audit_boundary.sha256,
        "code_manifest": gate.code_manifest.to_dict(),
        "scheduler_policy": gate.to_dict()["scheduler_policy"],
        "n_planned": len(planned),
        "n_submitted": n_submitted,
        "partial": failure is not None and n_submitted > 0,
        "failure": failure,
        "planned_jobs": planned,
        "submitted_prefix": submitted,
        "unsubmitted_suffix": planned[suffix_start:],
        "execution_overrides": {},
        "no_automatic_recovery": True,
        "retry_allowed": False,
        "cancel_on_failure": False,
        "resubmission_allowed": False,
    }
    if success and (receipt["n_planned"], receipt["n_submitted"]) != (12, 12):
        raise AssertionError("successful all-node submission must be exactly 12/12")
    return _json_bytes(receipt)


def read_receipt_bytes(raw: bytes) -> dict[str, Any]:
    """Strictly parse returned bytes without changing their immutable form."""

    if not isinstance(raw, bytes):
        raise ValueError("receipt must be bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid one-shot receipt bytes") from exc
    if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("one-shot receipt schema drift")
    if _json_bytes(payload) != raw:
        raise ValueError("one-shot receipt bytes are not canonical")
    return payload


def _bundle(value: DryRunAuditBundle | Mapping[str, Any]) -> DryRunAuditBundle:
    payload = value.to_dict() if isinstance(value, DryRunAuditBundle) else value
    try:
        return validate_dry_run_audit_bundle(payload)
    except Exception as exc:
        raise OneShotRefusal(f"dry-run audit/gate validation failed: {exc}") from exc


def _validate_topological_contract_order(gate: GateContract) -> None:
    seen: set[str] = set()
    if len(gate.jobs) != 12:
        raise OneShotRefusal("validated gate does not contain exactly 12 jobs")
    for job in gate.jobs:
        missing = [dependency for dependency in job.dependencies if dependency not in seen]
        if missing:
            raise OneShotRefusal(
                f"job {job.logical_id} appears before dependencies {missing}"
            )
        seen.add(job.logical_id)


def _refuse_existing_state(gate: GateContract, *, exists: Exists) -> None:
    paths = [
        gate.intent_path,
        gate.jobs_receipt_path,
        gate.log_dir,
        *(job.output_dir for job in gate.jobs),
    ]
    for path in paths:
        try:
            observed = exists(path)
        except Exception as exc:
            raise OneShotRefusal(f"existence callback failed for {path}: {exc}") from exc
        if observed is not False:
            raise OneShotRefusal(f"refusing pre-existing one-shot state: {path}")


def _refuse_scheduler_duplicates(
    gate: GateContract, existing_scheduler_names: Sequence[str]
) -> None:
    if isinstance(existing_scheduler_names, (str, bytes)):
        raise OneShotRefusal("existing_scheduler_names must be a sequence of names")
    observed: set[str] = set()
    for raw in existing_scheduler_names:
        if not isinstance(raw, str) or not raw or raw.strip() != raw:
            raise OneShotRefusal("existing scheduler names must be nonempty trimmed strings")
        observed.add(raw)
    collisions = sorted({job.name for job in gate.jobs} & observed)
    if collisions:
        raise OneShotRefusal(f"refusing duplicate scheduler names: {collisions}")


def _collect_pins(bundle: DryRunAuditBundle) -> tuple[Pin, ...]:
    pins: list[Pin] = [bundle.audit_boundary, bundle.gate_contract.code_manifest]

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            if set(value) == {"path", "sha256"}:
                pins.append(Pin(value["path"], value["sha256"]))
                return
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(bundle.audit_core)
    by_path: dict[str, Pin] = {}
    for pin in pins:
        previous = by_path.get(pin.path)
        if previous is not None and previous.sha256 != pin.sha256:
            raise OneShotRefusal(f"conflicting SHA-256 pins for {pin.path}")
        by_path[pin.path] = pin
    return tuple(by_path[path] for path in sorted(by_path))


def _planned_jobs(bundle: DryRunAuditBundle) -> list[dict[str, Any]]:
    gate = bundle.gate_contract
    configs = bundle.audit_core["job_configs"]
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(gate.jobs):
        config = configs[job.logical_id]
        rows.append(
            {
                "plan_index": index,
                "logical_id": job.logical_id,
                "name": job.name,
                "kind": job.kind,
                "cohort_id": job.cohort_id,
                "dependency_logical_ids": list(job.dependencies),
                "config": dict(config),
                "wrapper_path": job.wrapper_path,
                "output_dir": job.output_dir,
                "route": job.route.to_dict(),
                "resources": job.resources.to_dict(),
                "execution_overrides": {},
            }
        )
    return rows


def _validated_job_id(raw: object) -> str:
    if isinstance(raw, bool):
        raise ValueError("submit callback did not return a numeric job ID")
    value = str(raw)
    if _JOB_ID.fullmatch(value) is None:
        raise ValueError(f"submit callback did not return a numeric job ID: {raw!r}")
    return value


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "INTENT_SCHEMA",
    "OneShotRefusal",
    "RECEIPT_SCHEMA",
    "execute_one_shot",
    "read_receipt_bytes",
]
