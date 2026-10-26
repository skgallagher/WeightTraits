"""Immutable contract model for the all-node one-shot submission gate.

This module deliberately does not know how to invoke Slurm.  It freezes and
validates the plan that a separate, later submission boundary may consume.
The contract is intentionally more restrictive than a general scheduler
model: there are exactly four cohorts and exactly twelve jobs, with a fixed
dependency graph and fixed Wright resource route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .provenance import (
    ALLNODE_CODE_MANIFEST_BASENAME,
    DOWNSTREAM_CODE_MANIFEST_BASENAME,
    DOWNSTREAM_CODE_MANIFEST_SHA256,
    PRODUCTION_INTERPRETER,
    SOURCE_RESOLUTION_BASENAME,
)


SCHEMA = "weighttraits.allnode_submission_plan.v1"
SUBMISSION_ID = "weighttraits-allnode-v1"
COHORTS = ("llama_full", "llama_notrans", "flan_full", "flan_notrans")
N_TREES = 50
N_NODES = 641
N_INTERNAL = 276
N_LEAVES = 365
NO_AUTOMATIC_RECOVERY = True

DIRECT_KIND = "direct_allnode_cpu"
INFERENCE_KIND = "inference_allnode_gpu"
SEMANTIC_KIND = "semantic_allnode_gpu"
KINDS = frozenset({DIRECT_KIND, INFERENCE_KIND, SEMANTIC_KIND})


def canonical_json(value: object) -> str:
    """Return the unique serialized representation used for plan hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Pin:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _absolute_path(self.path, "pin.path")
        _sha256(self.sha256, "pin.sha256")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Route:
    account: str = "statds"
    partition: str = "all"
    qos: str = "normal"
    required_node: str = "n03"
    excluded_node: str = "n01"
    requeue: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArraySpec:
    start: int
    end: int
    max_concurrent: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Resources:
    cpus_per_task: int
    memory_gib: int
    time_limit: str
    gpu_count: int
    gpu_type: str | None
    array: ArraySpec | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpus_per_task": self.cpus_per_task,
            "memory_gib": self.memory_gib,
            "time_limit": self.time_limit,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "array": None if self.array is None else self.array.to_dict(),
        }


@dataclass(frozen=True)
class JobSpec:
    logical_id: str
    name: str
    kind: str
    cohort_id: str
    dependencies: tuple[str, ...]
    config_path: str
    wrapper_path: str
    output_dir: str
    route: Route
    resources: Resources
    execution_overrides: Mapping[str, object]

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "name": self.name,
            "kind": self.kind,
            "cohort_id": self.cohort_id,
            "dependencies": list(self.dependencies),
            "config_path": self.config_path,
            "wrapper_path": self.wrapper_path,
            "output_dir": self.output_dir,
            "route": self.route.to_dict(),
            "resources": self.resources.to_dict(),
            "execution_overrides": dict(self.execution_overrides),
        }


@dataclass(frozen=True)
class GateContract:
    submission_id: str
    stage_root: str
    frozen_stage: str
    deployment_root: str
    code_manifest: Pin
    source_resolution: Pin
    source_scope_sha256: str
    downstream_code_manifest: Pin
    production_interpreter: str
    python_version: str
    dry_run_audit: Pin
    intent_path: str
    jobs_receipt_path: str
    log_dir: str
    jobs: tuple[JobSpec, ...]

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "submission_id": self.submission_id,
            "stage_root": self.stage_root,
            "frozen_stage": self.frozen_stage,
            "deployment_root": self.deployment_root,
            "runtime_provenance": {
                "production_interpreter": self.production_interpreter,
                "python_version": self.python_version,
                "source_resolution": self.source_resolution.to_dict(),
                "source_scope_sha256": self.source_scope_sha256,
                "allnode_code_manifest": self.code_manifest.to_dict(),
                "downstream_code_manifest": self.downstream_code_manifest.to_dict(),
                "reattest_before_after_each_job": True,
            },
            "population_contract": {
                "cohort_ids": list(COHORTS),
                "n_cohorts": len(COHORTS),
                "n_trees_per_cohort": N_TREES,
                "n_trained_nodes_per_cohort": N_NODES,
                "n_internal_per_cohort": N_INTERNAL,
                "n_leaves_per_cohort": N_LEAVES,
                "primary_population": "all_trained_nodes",
                "sensitivity_population": "trained_leaves",
                "exclude_untrained_root": True,
            },
            "scheduler_policy": {
                **Route().to_dict(),
                "gpu_type": "nvidia-l40",
                "gpus_per_gpu_task": 1,
                "max_global_gpu_tasks": 4,
                "gpu_jobs_dependency_serialized": True,
            },
            "code_manifest": self.code_manifest.to_dict(),
            "dry_run_audit": self.dry_run_audit.to_dict(),
            "intent_path": self.intent_path,
            "jobs_receipt_path": self.jobs_receipt_path,
            "log_dir": self.log_dir,
            "execution_overrides": {},
            "no_automatic_recovery": NO_AUTOMATIC_RECOVERY,
            "jobs": [job.to_dict() for job in self.jobs],
        }


def build_gate_contract(
    *,
    stage_root: str,
    frozen_stage: str,
    code_manifest: Pin,
    source_resolution: Pin,
    source_scope_sha256: str,
    downstream_code_manifest: Pin,
    python_version: str,
    dry_run_audit: Pin,
    submission_id: str = SUBMISSION_ID,
) -> GateContract:
    """Build the sole valid all-node plan for the supplied immutable pins."""

    stage = _absolute_path(stage_root, "stage_root")
    frozen = _absolute_path(frozen_stage, "frozen_stage")
    if stage == frozen or stage in frozen.parents or frozen in stage.parents:
        raise ValueError("stage_root and frozen_stage must be disjoint")
    if submission_id != SUBMISSION_ID:
        raise ValueError(f"submission_id must be exactly {SUBMISSION_ID!r}")
    deployment = stage / "deployments" / "allnode_v1"
    manifest_path = _absolute_path(code_manifest.path, "code_manifest.path")
    if manifest_path != deployment / "runtime_receipts" / ALLNODE_CODE_MANIFEST_BASENAME:
        raise ValueError("code manifest path differs from the exact isolated receipt")
    audit_path = _absolute_path(dry_run_audit.path, "dry_run_audit.path")
    if stage not in audit_path.parents or frozen in audit_path.parents:
        raise ValueError("dry-run audit must be strictly inside only the downstream stage")
    if manifest_path == audit_path:
        raise ValueError("code manifest and dry-run audit cannot reuse a path")
    source_path = _absolute_path(source_resolution.path, "source_resolution.path")
    expected_source = deployment / "runtime_receipts" / SOURCE_RESOLUTION_BASENAME
    if source_path != expected_source:
        raise ValueError("source resolution pin path drift")
    downstream_path = _absolute_path(
        downstream_code_manifest.path, "downstream_code_manifest.path"
    )
    expected_downstream = stage / "runtime_receipts" / DOWNSTREAM_CODE_MANIFEST_BASENAME
    if (
        downstream_path != expected_downstream
        or downstream_code_manifest.sha256 != DOWNSTREAM_CODE_MANIFEST_SHA256
    ):
        raise ValueError("downstream code manifest pin drift")
    version = _version(python_version, "python_version")

    contract = GateContract(
        submission_id=submission_id,
        stage_root=str(stage),
        frozen_stage=str(frozen),
        deployment_root=str(deployment),
        code_manifest=code_manifest,
        source_resolution=source_resolution,
        source_scope_sha256=_sha256(source_scope_sha256, "source_scope_sha256"),
        downstream_code_manifest=downstream_code_manifest,
        production_interpreter=PRODUCTION_INTERPRETER,
        python_version=version,
        dry_run_audit=dry_run_audit,
        intent_path=str(stage / "runtime_receipts" / "ALLNODE_V1_SUBMIT_INTENT.json"),
        jobs_receipt_path=str(stage / "runtime_receipts" / "ALLNODE_V1_SUBMIT_JOBS.json"),
        log_dir=str(stage / "slurm_logs" / "allnode_v1"),
        jobs=_expected_jobs(stage=stage, deployment=deployment),
    )
    _validate_payload_structure(contract.to_dict())
    return contract


def serialize_gate_contract(contract: GateContract) -> str:
    """Serialize a validated contract; no filesystem or scheduler action occurs."""

    validated = validate_gate_contract(contract.to_dict())
    return canonical_json(validated.to_dict()) + "\n"


def validate_gate_contract(payload: GateContract | Mapping[str, Any]) -> GateContract:
    """Validate a serialized plan and return its canonical typed form."""

    raw = payload.to_dict() if isinstance(payload, GateContract) else payload
    if not isinstance(raw, Mapping):
        raise ValueError("gate contract must be a mapping")
    _validate_payload_structure(raw)
    manifest = _pin(raw["code_manifest"], "code_manifest")
    audit = _pin(raw["dry_run_audit"], "dry_run_audit")
    expected = build_gate_contract(
        stage_root=raw["stage_root"],
        frozen_stage=raw["frozen_stage"],
        code_manifest=manifest,
        source_resolution=_pin(
            raw["runtime_provenance"]["source_resolution"], "source_resolution"
        ),
        source_scope_sha256=raw["runtime_provenance"]["source_scope_sha256"],
        downstream_code_manifest=_pin(
            raw["runtime_provenance"]["downstream_code_manifest"],
            "downstream_code_manifest",
        ),
        python_version=raw["runtime_provenance"]["python_version"],
        dry_run_audit=audit,
        submission_id=raw["submission_id"],
    )
    if canonical_sha256(dict(raw)) != canonical_sha256(expected.to_dict()):
        raise ValueError("gate contract differs from the frozen all-node plan")
    return expected


def _expected_jobs(*, stage: PurePosixPath, deployment: PurePosixPath) -> tuple[JobSpec, ...]:
    route = Route()
    resources = {
        DIRECT_KIND: Resources(8, 96, "24:00:00", 0, None, None),
        INFERENCE_KIND: Resources(
            8, 64, "12:00:00", 1, "nvidia-l40", ArraySpec(0, 640, 4)
        ),
        SEMANTIC_KIND: Resources(8, 64, "24:00:00", 1, "nvidia-l40", None),
    }
    wrapper_names = {
        DIRECT_KIND: "allnode_direct_cpu.sbatch",
        INFERENCE_KIND: "allnode_inference_gpu.sbatch",
        SEMANTIC_KIND: "allnode_semantic_gpu.sbatch",
    }

    def job(kind: str, cohort: str, dependencies: Sequence[str]) -> JobSpec:
        prefix = {
            DIRECT_KIND: "direct",
            INFERENCE_KIND: "infer",
            SEMANTIC_KIND: "semantic",
        }[kind]
        logical_id = f"{prefix}_{cohort}"
        output_prefix = "inference" if kind == INFERENCE_KIND else prefix
        return JobSpec(
            logical_id=logical_id,
            name=f"wt-an-v1-{prefix}-{cohort.replace('_', '-')}",
            kind=kind,
            cohort_id=cohort,
            dependencies=tuple(dependencies),
            config_path=str(deployment / "configs" / "jobs" / f"{logical_id}.json"),
            wrapper_path=str(deployment / "scripts" / "slurm" / wrapper_names[kind]),
            output_dir=str(stage / "outputs" / "allnode_v1" / output_prefix / cohort),
            route=route,
            resources=resources[kind],
            execution_overrides={},
        )

    planned: list[JobSpec] = [job(DIRECT_KIND, cohort, ()) for cohort in COHORTS]
    previous_semantic: str | None = None
    for cohort in COHORTS:
        inference_id = f"infer_{cohort}"
        semantic_id = f"semantic_{cohort}"
        planned.append(
            job(
                INFERENCE_KIND,
                cohort,
                () if previous_semantic is None else (previous_semantic,),
            )
        )
        planned.append(
            job(
                SEMANTIC_KIND,
                cohort,
                (f"direct_{cohort}", inference_id),
            )
        )
        previous_semantic = semantic_id
    return tuple(planned)


def _validate_payload_structure(raw: Mapping[str, Any]) -> None:
    _exact_keys(
        raw,
        {
            "schema",
            "submission_id",
            "stage_root",
            "frozen_stage",
            "deployment_root",
            "runtime_provenance",
            "population_contract",
            "scheduler_policy",
            "code_manifest",
            "dry_run_audit",
            "intent_path",
            "jobs_receipt_path",
            "log_dir",
            "execution_overrides",
            "no_automatic_recovery",
            "jobs",
        },
        "gate contract",
    )
    if raw["schema"] != SCHEMA:
        raise ValueError(f"schema must be exactly {SCHEMA!r}")
    if raw["execution_overrides"] != {}:
        raise ValueError("execution overrides are forbidden")
    if raw["no_automatic_recovery"] is not True:
        raise ValueError("no_automatic_recovery must be true")
    _validate_population(raw["population_contract"])
    _validate_scheduler_policy(raw["scheduler_policy"])
    _pin(raw["code_manifest"], "code_manifest")
    _pin(raw["dry_run_audit"], "dry_run_audit")
    _validate_runtime_provenance(raw["runtime_provenance"], raw["code_manifest"])

    stage = _absolute_path(raw["stage_root"], "stage_root")
    frozen = _absolute_path(raw["frozen_stage"], "frozen_stage")
    deployment = _absolute_path(raw["deployment_root"], "deployment_root")
    intent = _absolute_path(raw["intent_path"], "intent_path")
    receipt = _absolute_path(raw["jobs_receipt_path"], "jobs_receipt_path")
    log_dir = _absolute_path(raw["log_dir"], "log_dir")
    if stage == frozen or stage in frozen.parents or frozen in stage.parents:
        raise ValueError("stage_root and frozen_stage must be disjoint")
    for label, path in (
        ("deployment_root", deployment),
        ("intent_path", intent),
        ("jobs_receipt_path", receipt),
        ("log_dir", log_dir),
    ):
        if stage not in path.parents:
            raise ValueError(f"{label} must be strictly inside stage_root")
    if len({intent, receipt, log_dir}) != 3:
        raise ValueError("intent, jobs receipt, and log paths cannot be reused")

    jobs = raw["jobs"]
    if not isinstance(jobs, list) or len(jobs) != 12:
        raise ValueError("all-node gate must contain exactly 12 jobs")
    _validate_jobs(
        jobs,
        reserved=(intent, receipt, log_dir),
        stage=stage,
        deployment=deployment,
    )


def _validate_runtime_provenance(raw: object, code_manifest: object) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("runtime_provenance must be a mapping")
    _exact_keys(
        raw,
        {
            "production_interpreter", "python_version", "source_resolution",
            "source_scope_sha256",
            "allnode_code_manifest", "downstream_code_manifest",
            "reattest_before_after_each_job",
        },
        "runtime_provenance",
    )
    if (
        raw["production_interpreter"] != PRODUCTION_INTERPRETER
        or raw["reattest_before_after_each_job"] is not True
        or _pin(raw["allnode_code_manifest"], "allnode_code_manifest")
        != _pin(code_manifest, "code_manifest")
    ):
        raise ValueError("runtime provenance interpreter/allnode binding drift")
    _version(raw["python_version"], "runtime_provenance.python_version")
    _pin(raw["source_resolution"], "source_resolution")
    _sha256(raw["source_scope_sha256"], "source_scope_sha256")
    downstream = _pin(raw["downstream_code_manifest"], "downstream_code_manifest")
    if downstream.sha256 != DOWNSTREAM_CODE_MANIFEST_SHA256:
        raise ValueError("runtime provenance downstream manifest SHA-256 drift")


def _validate_population(raw: object) -> None:
    expected = {
        "cohort_ids": list(COHORTS),
        "n_cohorts": 4,
        "n_trees_per_cohort": N_TREES,
        "n_trained_nodes_per_cohort": N_NODES,
        "n_internal_per_cohort": N_INTERNAL,
        "n_leaves_per_cohort": N_LEAVES,
        "primary_population": "all_trained_nodes",
        "sensitivity_population": "trained_leaves",
        "exclude_untrained_root": True,
    }
    if not isinstance(raw, Mapping) or canonical_sha256(dict(raw)) != canonical_sha256(expected):
        raise ValueError("population contract differs from exact 4-cohort/641-node scope")


def _validate_scheduler_policy(raw: object) -> None:
    expected = {
        **Route().to_dict(),
        "gpu_type": "nvidia-l40",
        "gpus_per_gpu_task": 1,
        "max_global_gpu_tasks": 4,
        "gpu_jobs_dependency_serialized": True,
    }
    if not isinstance(raw, Mapping) or canonical_sha256(dict(raw)) != canonical_sha256(expected):
        raise ValueError("scheduler policy differs from the frozen Wright route")


def _validate_jobs(
    jobs: Sequence[object], *, reserved: Sequence[PurePosixPath],
    stage: PurePosixPath, deployment: PurePosixPath,
) -> None:
    required = {
        "logical_id",
        "name",
        "kind",
        "cohort_id",
        "dependencies",
        "config_path",
        "wrapper_path",
        "output_dir",
        "route",
        "resources",
        "execution_overrides",
    }
    parsed: list[Mapping[str, Any]] = []
    for index, raw in enumerate(jobs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"jobs[{index}] must be a mapping")
        _exact_keys(raw, required, f"jobs[{index}]")
        if raw["kind"] not in KINDS or raw["cohort_id"] not in COHORTS:
            raise ValueError(f"jobs[{index}] has unsupported kind or cohort")
        if raw["execution_overrides"] != {}:
            raise ValueError(f"jobs[{index}] execution overrides are forbidden")
        if raw["route"] != Route().to_dict():
            raise ValueError(f"jobs[{index}] route drift")
        if not isinstance(raw["dependencies"], list):
            raise ValueError(f"jobs[{index}] dependencies must be a list")
        for field in ("logical_id", "name"):
            if not isinstance(raw[field], str) or not raw[field] or raw[field].strip() != raw[field]:
                raise ValueError(f"jobs[{index}].{field} must be a trimmed string")
        for field in ("config_path", "wrapper_path", "output_dir"):
            _absolute_path(raw[field], f"jobs[{index}].{field}")
        config = _absolute_path(raw["config_path"], f"jobs[{index}].config_path")
        wrapper = _absolute_path(raw["wrapper_path"], f"jobs[{index}].wrapper_path")
        output = _absolute_path(raw["output_dir"], f"jobs[{index}].output_dir")
        if deployment not in config.parents or deployment not in wrapper.parents:
            raise ValueError("job config and wrapper must be inside the isolated deployment")
        output_boundary = stage / "outputs" / "allnode_v1"
        if output_boundary not in output.parents:
            raise ValueError("job output must be inside downstream outputs/allnode_v1")
        parsed.append(raw)

    logical_ids = [row["logical_id"] for row in parsed]
    names = [row["name"] for row in parsed]
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("duplicate logical job IDs are forbidden")
    if len(names) != len(set(names)):
        raise ValueError("duplicate scheduler job names are forbidden")
    configs = [_absolute_path(row["config_path"], "config_path") for row in parsed]
    outputs = [_absolute_path(row["output_dir"], "output_dir") for row in parsed]
    if len(configs) != len(set(configs)):
        raise ValueError("job config paths cannot be reused")
    if len(outputs) != len(set(outputs)):
        raise ValueError("job output paths cannot be reused")
    for index, left in enumerate(outputs):
        if left in reserved:
            raise ValueError("job output overlaps a reserved gate path")
        for right in outputs[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError("job output paths cannot contain one another")
        for reserved_path in reserved:
            if left in reserved_path.parents or reserved_path in left.parents:
                raise ValueError("job output overlaps reserved gate state")

    by_id = {row["logical_id"]: row for row in parsed}
    for row in parsed:
        dependencies = row["dependencies"]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("duplicate dependencies are forbidden")
        for dependency in dependencies:
            if dependency not in by_id:
                raise ValueError(f"unknown dependency {dependency!r}")
    _assert_acyclic(by_id)


def _assert_acyclic(by_id: Mapping[str, Mapping[str, Any]]) -> None:
    state: dict[str, int] = {}

    def visit(logical_id: str) -> None:
        marker = state.get(logical_id, 0)
        if marker == 1:
            raise ValueError("job dependency graph contains a cycle")
        if marker == 2:
            return
        state[logical_id] = 1
        for dependency in by_id[logical_id]["dependencies"]:
            visit(dependency)
        state[logical_id] = 2

    for logical_id in by_id:
        visit(logical_id)


def _pin(raw: object, label: str) -> Pin:
    if isinstance(raw, Pin):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a path+sha256 mapping")
    _exact_keys(raw, {"path", "sha256"}, label)
    return Pin(raw["path"], raw["sha256"])


def _absolute_path(raw: object, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ValueError(f"{label} must be a non-empty trimmed path")
    if any(character in raw for character in (",", "\n", "\r")):
        raise ValueError(f"{label} contains an unsafe character")
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts or str(path) != raw:
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    return path


def _sha256(raw: object, label: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return raw


def _version(raw: object, label: str) -> str:
    if (
        not isinstance(raw, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", raw) is None
    ):
        raise ValueError(f"{label} must be an exact semantic Python version")
    return raw


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(raw)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


__all__ = [
    "COHORTS",
    "DIRECT_KIND",
    "GateContract",
    "INFERENCE_KIND",
    "NO_AUTOMATIC_RECOVERY",
    "Pin",
    "SEMANTIC_KIND",
    "build_gate_contract",
    "canonical_sha256",
    "serialize_gate_contract",
    "validate_gate_contract",
]
