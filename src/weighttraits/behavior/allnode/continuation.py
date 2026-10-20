"""Audited one-shot continuation for the all-trained-node study.

The original llama-full inference array produced all 641 terminal artifact
pairs, but task 438 exited nonzero because the original cache decoder treated
a literal U+2028 inside a JSON string as a record boundary.  This isolated
continuation fixes JSONL framing without changing any original deployment,
artifact, job, config, or receipt.  A CPU adoption gate validates the complete
cache twice before the seven remaining GPU jobs may begin.

There is deliberately no retry, recovery, cancellation, requeue, overwrite,
or resubmission operation in this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
import uuid


STAGE = Path("/home/export/sgallagh/WeightTraits-downstream-20260831-native")
FROZEN = Path("/home/export/sgallagh/WeightTraits-all-results-20260817-clean")
OLD_DEPLOYMENT = STAGE / "deployments" / "allnode_v1"
V1C1_DEPLOYMENT = STAGE / "deployments" / "allnode_v1_continuation_20260902"
DEPLOYMENT = STAGE / "deployments" / "allnode_v1_continuation_20260908_v1c5"
PYTHON_BIN = "/home/export/sgallagh/.conda/envs/ellmtrees/bin/python"
SUBMISSION_ID = "weighttraits-allnode-v1c5-20260908"
SCHEMA_PREFIX = "weighttraits.allnode_v1c5"

RUNTIME_RECEIPTS = DEPLOYMENT / "runtime_receipts"
CODE_MANIFEST = RUNTIME_RECEIPTS / "ALLNODE_V1C5_CODE_MANIFEST.json"
EXPECTED_ACCEPTANCE = (
    RUNTIME_RECEIPTS / "ALLNODE_V1C5_EXPECTED_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json"
)
ACCEPTANCE_RECEIPT = (
    RUNTIME_RECEIPTS / "ALLNODE_V1C5_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json"
)
SCHEDULER_EVIDENCE = RUNTIME_RECEIPTS / "ALLNODE_V1C5_SCHEDULER_EVIDENCE.json"
DRY_RUN_AUDIT = RUNTIME_RECEIPTS / "ALLNODE_V1C5_DRY_RUN_AUDIT.json"
TEST_ATTESTATION = RUNTIME_RECEIPTS / "ALLNODE_V1C5_TEST_ATTESTATION.json"
INTENT = STAGE / "runtime_receipts" / "ALLNODE_V1C5_SUBMIT_INTENT.json"
JOBS_RECEIPT = STAGE / "runtime_receipts" / "ALLNODE_V1C5_SUBMIT_JOBS.json"
LOG_DIR = STAGE / "slurm_logs" / "allnode_v1c5_20260908"

OLD_DRY_RUN = OLD_DEPLOYMENT / "runtime_receipts" / "ALLNODE_V1_DRY_RUN_AUDIT.json"
OLD_INTENT = STAGE / "runtime_receipts" / "ALLNODE_V1_SUBMIT_INTENT.json"
OLD_JOBS = STAGE / "runtime_receipts" / "ALLNODE_V1_SUBMIT_JOBS.json"

V1C1_PINS: dict[str, tuple[Path, str]] = {
    "code_manifest": (
        V1C1_DEPLOYMENT / "runtime_receipts" / "ALLNODE_V1C1_CODE_MANIFEST.json",
        "8653ef19f7d4ff6d94363443b8e503e6d10abba5e221a9b7586be55ff00a8588",
    ),
    "test_attestation": (
        V1C1_DEPLOYMENT / "runtime_receipts" / "ALLNODE_V1C1_TEST_ATTESTATION.json",
        "563fd21c8dec912c3868826ce0cb898d708fc10a9bc87bac9376b4b1a4efabc9",
    ),
    "dry_run": (
        V1C1_DEPLOYMENT / "runtime_receipts" / "ALLNODE_V1C1_DRY_RUN_AUDIT.json",
        "04bae41eed4f496a92b71980efe180041b5eb4eaa0bbe998620ac1b25ccedac0",
    ),
    "expected_acceptance": (
        V1C1_DEPLOYMENT
        / "runtime_receipts"
        / "EXPECTED_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json",
        "4ee2209984ca13b74de9c4c4065eeb55fe3dac01de38a20bd15d50931899c616",
    ),
    "intent": (
        STAGE / "runtime_receipts" / "ALLNODE_V1C1_SUBMIT_INTENT.json",
        "c9f8f91e0321addfd27df1175efc3449623969af417744f0866a479d9dd76c3c",
    ),
    "jobs": (
        STAGE / "runtime_receipts" / "ALLNODE_V1C1_SUBMIT_JOBS.json",
        "dc2e1373aa19c3b5b13e32d30e5f2d557f759eecde0c5d689a187fd5fea26ea3",
    ),
    "adoption_stdout": (
        STAGE / "slurm_logs" / "allnode_v1c1_20260902" / "adopt_llama_full_184529.out",
        "57ffed9b0bc18b7e6cd37335b8aed04aebfc5e6fa111eeecfa6cc8e7e30103c5",
    ),
    "adoption_stderr": (
        STAGE / "slurm_logs" / "allnode_v1c1_20260902" / "adopt_llama_full_184529.err",
        "ce02b0eb9ea9fcc4f2386d520391fa4b12400a2f2c6463f662be3c9946d1c781",
    ),
}
V1C1_ACCEPTANCE_RECEIPT = (
    V1C1_DEPLOYMENT / "runtime_receipts" / "LLAMA_FULL_ARTIFACT_ACCEPTANCE.json"
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
JOB_ID_RE = re.compile(r"[0-9]+")
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_COMMAND_BYTES = 16 * 1024 * 1024
COMMAND_TIMEOUT = 90.0
EXPECTED_TEST_COUNT = 206

OLD_PINS: dict[str, tuple[Path, str]] = {
    "dry_run": (OLD_DRY_RUN, "82e46fdc682b5a98ef159aedfbb1a96dd924942b1f0c59f1e070e126ad36c03b"),
    "intent": (OLD_INTENT, "17e7063ad337d88bd94cfa409037ff0185c2490de19816775d25275ea2e78857"),
    "jobs": (OLD_JOBS, "a754b6e2445735ddbd19ba2cc27b042ce964ce2efde5dfa10182d28016fe1add"),
    "code_manifest": (
        OLD_DEPLOYMENT / "runtime_receipts" / "ALLNODE_V1_CODE_MANIFEST.json",
        "3847d76b2013f713fc4e19ae290987d44a29636e317e52881e2234395bfb79ad",
    ),
    "source_resolution": (
        OLD_DEPLOYMENT / "runtime_receipts" / "ALLNODE_V1_SOURCE_RESOLUTION.json",
        "9c3c0f6001f42f893913523e8e6074ceaee9be6dc4d92b622b002e03b6c1cc72",
    ),
    "downstream_code_manifest": (
        STAGE / "runtime_receipts" / "DEPLOYED_CODE_SHA256_SUBMISSION_20260831_V2",
        "11a95522b1e046731666d4f678a39d39103961bc0b83ae9fe75cfbd4f1a433cb",
    ),
    "frozen_stage_manifest": (
        FROZEN / ".stage_provenance" / "STAGE_SHA256SUMS",
        "642912687e67d0a48162a8d5cc0a95164ad92350ea629cd396fcfdd7917cd5d1",
    ),
}
OLD_GATE_SHA256 = "5f477b879c2c3ed07d1bb29d23f492e50f1131fc93bbba3cfa9e2460d60e9769"
SOURCE_SCOPE_SHA256 = "d1feb8a6e520710681d8d5b515ca63b91d50ca0008cd3535de6a812252943cea"

CONFIG_SHA256 = {
    "infer_llama_full": "3d66f9228510b7846a31fcaeebd10319f440205d0279797662c7ef87d58d250c",
    "semantic_llama_full": "6b57bf2f6d56aebfd7d520bdf7bb6fc40136ed1bf452def6c81df44d199210e3",
    "infer_llama_notrans": "8e74b4e4004b27861c49c235525ef00851418b28b1f8df709b851eaaaf17a43f",
    "semantic_llama_notrans": "6902c7bcd66212ff96eb26371eeb63090cefd3f07f1c4184f7a4e2efdb225dc1",
    "infer_flan_full": "02136fed35c1dce5c7b77bf0d3011b9735f5611ac137b8d6857bea9c17e632e1",
    "semantic_flan_full": "a2952071869ad2970f694a963210ed955b138a9b2c0d9f4c89f9cfdd5e3e3342",
    "infer_flan_notrans": "2c5d1f33b22eba2e3a1d76d7ff540c73de8aa26978c3daa2923d1002ac278f8d",
    "semantic_flan_notrans": "9886cbf84841cddfd1fe56ccd1932f01cbf7a821c2e86ae2c3a193432a31ae3d",
}
DIRECT_TERMINAL_SHA256 = {
    "llama_full": "852139f838e7800b901b4b828db51d64e70f2864f6b6312e12539806cfb0e6c0",
    "llama_notrans": "6daeaa919459bf1db6d093f397566aa43e922871fe522d363c320bbd208a4c47",
    "flan_full": "f3c2b97440709aca8177ab91b779022d62a47a842d3088902707cd711e9d5e6a",
    "flan_notrans": "7f8cba724b3efa0a86835944808731d733c492eb04d45a3d54f2551a0f2f98c6",
}


@dataclass(frozen=True)
class Job:
    logical_id: str
    scheduler_name: str
    runner_logical_id: str | None
    cohort: str
    kind: str
    dependency: str | None
    config_path: str | None
    config_sha256: str | None
    output_path: str
    cpus: int
    memory_gib: int
    time_limit: str
    gpu: bool
    array: str | None
    wrapper: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "scheduler_name": self.scheduler_name,
            "runner_logical_id": self.runner_logical_id,
            "cohort": self.cohort,
            "kind": self.kind,
            "dependencies": [] if self.dependency is None else [self.dependency],
            "config": None
            if self.config_path is None
            else {"path": self.config_path, "sha256": self.config_sha256},
            "output_path": self.output_path,
            "route": {
                "account": "statds",
                "partition": "all",
                "qos": "normal",
                "required_node": "n03",
                "excluded_node": "n01",
                "requeue": False,
            },
            "resources": {
                "cpus_per_task": self.cpus,
                "memory_gib": self.memory_gib,
                "time_limit": self.time_limit,
                "gpu_count": 1 if self.gpu else 0,
                "gpu_type": "nvidia-l40" if self.gpu else None,
                "array": None
                if self.array is None
                else {"start": 0, "end": 640, "max_concurrent": 4},
            },
            "wrapper": self.wrapper,
            "execution_overrides": {},
        }


def _config(logical_id: str) -> str:
    return str(OLD_DEPLOYMENT / "configs" / "jobs" / f"{logical_id}.json")


def build_plan() -> tuple[Job, ...]:
    adopt_wrapper = str(DEPLOYMENT / "scripts" / "slurm" / "allnode_v1c5_adopt_cpu.sbatch")
    runner_wrapper = str(
        DEPLOYMENT / "scripts" / "slurm" / "allnode_v1c5_bound_runner.sbatch"
    )
    rows: list[Job] = [
        Job(
            "adopt_llama_full",
            "wt-an-v1c5-adopt-llama-full",
            None,
            "llama_full",
            "artifact_adoption_cpu",
            None,
            None,
            None,
            str(ACCEPTANCE_RECEIPT),
            8,
            96,
            "24:00:00",
            False,
            None,
            adopt_wrapper,
        )
    ]
    sequence = (
        ("semantic_llama_full", "semantic_allnode_gpu", "llama_full"),
        ("infer_llama_notrans", "inference_allnode_gpu", "llama_notrans"),
        ("semantic_llama_notrans", "semantic_allnode_gpu", "llama_notrans"),
        ("infer_flan_full", "inference_allnode_gpu", "flan_full"),
        ("semantic_flan_full", "semantic_allnode_gpu", "flan_full"),
        ("infer_flan_notrans", "inference_allnode_gpu", "flan_notrans"),
        ("semantic_flan_notrans", "semantic_allnode_gpu", "flan_notrans"),
    )
    previous = rows[0].logical_id
    for runner_id, kind, cohort in sequence:
        prefix = "infer" if kind.startswith("inference") else "semantic"
        output_prefix = "inference" if prefix == "infer" else "semantic"
        rows.append(
            Job(
                f"v1c5_{runner_id}",
                f"wt-an-v1c5-{prefix}-{cohort.replace('_', '-')}",
                runner_id,
                cohort,
                kind,
                previous,
                _config(runner_id),
                CONFIG_SHA256[runner_id],
                str(STAGE / "outputs" / "allnode_v1" / output_prefix / cohort),
                8,
                64,
                "12:00:00" if prefix == "infer" else "24:00:00",
                True,
                "0-640%4" if prefix == "infer" else None,
                runner_wrapper,
            )
        )
        previous = rows[-1].logical_id
    validate_plan(rows)
    return tuple(rows)


def validate_plan(plan: Sequence[Job]) -> None:
    if len(plan) != 8:
        raise ValueError("continuation must contain exactly eight jobs")
    if len({job.logical_id for job in plan}) != 8 or len(
        {job.scheduler_name for job in plan}
    ) != 8:
        raise ValueError("continuation job identities must be unique")
    for index, job in enumerate(plan):
        if job.runner_logical_id == "infer_llama_full":
            raise ValueError("llama-full inference must never be rerun")
        expected_dependency = None if index == 0 else plan[index - 1].logical_id
        if job.dependency != expected_dependency:
            raise ValueError("continuation must be one exact serialized chain")
        if index == 0:
            if job.gpu or job.array is not None or job.kind != "artifact_adoption_cpu":
                raise ValueError("adoption gate resource drift")
        elif not job.gpu:
            raise ValueError("every workload continuation job must request one GPU")
        if job.kind == "inference_allnode_gpu" and job.array != "0-640%4":
            raise ValueError("inference array must be exactly 0-640%4")
        if job.kind == "semantic_allnode_gpu" and job.array is not None:
            raise ValueError("semantic jobs cannot be arrays")
        if job.config_path is not None and job.config_sha256 != CONFIG_SHA256.get(
            job.runner_logical_id or ""
        ):
            raise ValueError("continuation config pin drift")


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = canonical_file(path, "SHA-256 input")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    if (
        not value.is_absolute()
        or ".." in value.parts
        or value.is_symlink()
        or not value.is_file()
        or value.resolve(strict=True) != value
    ):
        raise ValueError(f"{label} must be a canonical non-symlink file: {value}")
    return value


def canonical_dir(path: str | Path, label: str) -> Path:
    value = Path(path)
    if (
        not value.is_absolute()
        or ".." in value.parts
        or value.is_symlink()
        or not value.is_dir()
        or value.resolve(strict=True) != value
    ):
        raise ValueError(f"{label} must be a canonical non-symlink directory: {value}")
    return value


def read_canonical(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = canonical_file(path, label)
    raw = source.read_bytes()
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{label} is empty or exceeds the size limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(payload, dict) or raw != canonical_bytes(payload):
        raise ValueError(f"{label} is not one canonical JSON object")
    return payload, raw


def verify_code_manifest(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    expected = require_sha(expected_sha256, "code manifest SHA-256")
    manifest_path = canonical_file(path, "continuation code manifest")
    if manifest_path != CODE_MANIFEST:
        raise ValueError("continuation code manifest path drift")
    payload, raw = read_canonical(manifest_path, "continuation code manifest")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("continuation code manifest full-file hash drift")
    required = {
        "schema",
        "valid",
        "deployment_root",
        "n_files",
        "files",
        "files_sha256",
        "code_manifest_sha256",
    }
    if set(payload) != required:
        raise ValueError("continuation code manifest keys drift")
    files = payload["files"]
    core = {key: payload[key] for key in payload if key != "code_manifest_sha256"}
    if (
        payload["schema"] != f"{SCHEMA_PREFIX}_code_manifest.v1"
        or payload["valid"] is not True
        or payload["deployment_root"] != str(DEPLOYMENT)
        or not isinstance(files, list)
        or not files
        or payload["n_files"] != len(files)
        or payload["files_sha256"] != canonical_sha256(files)
        or payload["code_manifest_sha256"] != canonical_sha256(core)
    ):
        raise ValueError("continuation code manifest contract drift")
    previous: str | None = None
    seen: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"continuation manifest row {index} drift")
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or relative in seen
            or (previous is not None and relative <= previous)
        ):
            raise ValueError("continuation manifest paths are unsafe or unsorted")
        seen.add(relative)
        previous = relative
        target = canonical_file(DEPLOYMENT / relative, "continuation manifest entry")
        if (
            not isinstance(row["size_bytes"], int)
            or isinstance(row["size_bytes"], bool)
            or target.stat().st_size != row["size_bytes"]
            or sha256_file(target) != require_sha(row["sha256"], "entry SHA-256")
        ):
            raise ValueError(f"continuation manifest entry drift: {relative}")
    return payload


def build_code_manifest(source_root: str | Path, output: str | Path) -> bytes:
    source = canonical_dir(source_root, "manifest source root")
    output_path = Path(output)
    allowed_roots = {
        source / "weighttraits_allnode",
        source / "weighttraits_allnode_continuation",
        source / "scripts" / "slurm",
        source / "tests",
    }
    files: list[dict[str, Any]] = []
    for root in sorted(allowed_roots, key=lambda value: value.as_posix()):
        canonical_dir(root, "manifest source subtree")
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file() or path.suffix not in {".py", ".sbatch"}:
                raise ValueError(f"unexpected manifest source entry: {path}")
            relative = path.relative_to(source).as_posix()
            files.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    files.sort(key=lambda row: row["path"])
    core = {
        "schema": f"{SCHEMA_PREFIX}_code_manifest.v1",
        "valid": True,
        "deployment_root": str(DEPLOYMENT),
        "n_files": len(files),
        "files": files,
        "files_sha256": canonical_sha256(files),
    }
    payload = {**core, "code_manifest_sha256": canonical_sha256(core)}
    raw = canonical_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    publish_exclusive(output_path, raw)
    return raw


def require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def publish_exclusive(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("exclusive target must be an absolute canonical path")
    parent = canonical_dir(destination.parent, "exclusive target parent")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_JSON_BYTES:
        raise ValueError("exclusive payload is empty or exceeds the size limit")
    temporary = parent / f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    linked = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        linked = True
        os.chmod(destination, 0o444)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    if not linked or destination.read_bytes() != payload:
        raise RuntimeError("exclusive publication verification failed")


def attest_original_runtime(phase: str) -> dict[str, Any]:
    """Revalidate every original source/code pin with the production runtime."""

    from weighttraits.behavior.allnode.launch import load_persisted_dry_run
    from weighttraits.behavior.allnode.provenance import attest_runtime_provenance

    for label, (path, expected) in OLD_PINS.items():
        if sha256_file(path) != expected:
            raise ValueError(f"original immutable pin drift: {label}")
    bundle = load_persisted_dry_run(OLD_DRY_RUN)
    gate = bundle.gate_contract
    if (
        gate.sha256 != OLD_GATE_SHA256
        or gate.stage_root != str(STAGE)
        or gate.frozen_stage != str(FROZEN)
        or gate.deployment_root != str(OLD_DEPLOYMENT)
        or gate.source_scope_sha256 != SOURCE_SCOPE_SHA256
        or gate.production_interpreter != PYTHON_BIN
        or gate.python_version != "3.11.15"
        or gate.code_manifest.path != str(OLD_PINS["code_manifest"][0])
        or gate.code_manifest.sha256 != OLD_PINS["code_manifest"][1]
        or gate.source_resolution.path != str(OLD_PINS["source_resolution"][0])
        or gate.source_resolution.sha256 != OLD_PINS["source_resolution"][1]
        or gate.downstream_code_manifest.path
        != str(OLD_PINS["downstream_code_manifest"][0])
        or gate.downstream_code_manifest.sha256
        != OLD_PINS["downstream_code_manifest"][1]
    ):
        raise ValueError("original all-node gate contract drift")
    config_pins = bundle.audit_core.get("job_configs")
    if not isinstance(config_pins, dict):
        raise ValueError("original dry-run lacks job config pins")
    for logical_id, expected in CONFIG_SHA256.items():
        row = config_pins.get(logical_id)
        if row != {"path": _config(logical_id), "sha256": expected}:
            raise ValueError(f"original config boundary drift: {logical_id}")
        if sha256_file(_config(logical_id)) != expected:
            raise ValueError(f"original config file drift: {logical_id}")
    return attest_runtime_provenance(
        phase=phase,
        stage_root=str(STAGE),
        deployment_root=str(OLD_DEPLOYMENT),
        source_resolution_path=str(OLD_PINS["source_resolution"][0]),
        source_resolution_sha256=OLD_PINS["source_resolution"][1],
        source_scope_sha256=SOURCE_SCOPE_SHA256,
        allnode_code_manifest_path=str(OLD_PINS["code_manifest"][0]),
        allnode_code_manifest_sha256=OLD_PINS["code_manifest"][1],
        downstream_code_manifest_path=str(OLD_PINS["downstream_code_manifest"][0]),
        downstream_code_manifest_sha256=OLD_PINS["downstream_code_manifest"][1],
        expected_python_version="3.11.15",
    )


def attest_v1c1_failure_files() -> dict[str, Any]:
    """Pin the failed predecessor without invoking or changing it."""

    for label, (path, expected) in V1C1_PINS.items():
        if sha256_file(path) != expected:
            raise ValueError(f"v1c1 immutable pin drift: {label}")
    if os.path.lexists(V1C1_ACCEPTANCE_RECEIPT):
        raise FileExistsError("v1c1 acceptance receipt unexpectedly exists")

    intent, _ = read_canonical(V1C1_PINS["intent"][0], "v1c1 submit intent")
    jobs, _ = read_canonical(V1C1_PINS["jobs"][0], "v1c1 submit jobs")
    submitted = jobs.get("submitted_prefix")
    expected_ids = [str(value) for value in range(184529, 184537)]
    if (
        intent.get("schema") != "weighttraits.allnode_v1c1_submit_intent.v1"
        or intent.get("valid") is not True
        or intent.get("n_planned") != 8
        or jobs.get("schema") != "weighttraits.allnode_v1c1_submit_jobs.v1"
        or jobs.get("valid") is not True
        or jobs.get("terminal") is not True
        or jobs.get("n_planned") != 8
        or jobs.get("n_submitted") != 8
        or jobs.get("partial") is not False
        or jobs.get("failure") is not None
        or jobs.get("retry_allowed") is not False
        or jobs.get("resubmission_allowed") is not False
        or not isinstance(submitted, list)
        or [row.get("job_id") for row in submitted] != expected_ids
    ):
        raise ValueError("v1c1 immutable submission receipt drift")

    stdout = canonical_file(
        V1C1_PINS["adoption_stdout"][0], "v1c1 adoption stdout"
    ).read_text(encoding="utf-8")
    stderr = canonical_file(
        V1C1_PINS["adoption_stderr"][0], "v1c1 adoption stderr"
    ).read_text(encoding="utf-8")
    if (
        "\"valid\":true" not in stdout
        or "localhost:6819" not in stderr
        or "Connection refused" not in stderr
        or "query_original_scheduler" not in stderr
    ):
        raise ValueError("v1c1 failure signature drift")

    return {
        "deployment": str(V1C1_DEPLOYMENT),
        "pins": {
            label: {"path": str(path), "sha256": digest}
            for label, (path, digest) in V1C1_PINS.items()
        },
        "acceptance_receipt": {
            "path": str(V1C1_ACCEPTANCE_RECEIPT),
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


def artifact_inventory_pass() -> list[dict[str, Any]]:
    """Validate all 641 llama-full terminal pairs without a generation path."""

    from weighttraits.behavior.allnode.checkpoints import (
        PRODUCTION_N_INTERNAL,
        PRODUCTION_N_LEAVES,
        PRODUCTION_N_NODES,
        PRODUCTION_N_TREES,
    )
    from weighttraits.behavior.allnode.semantic_runner import (
        load_node_endpoint_source,
        load_semantic_job_config,
    )

    _, semantic, inference, _, _ = load_semantic_job_config(
        _config("semantic_llama_full")
    )
    if (
        inference.cohort_key != "llama_full"
        or semantic.cohort_key != "llama_full"
        or len(inference.checkpoint_manifests) != PRODUCTION_N_TREES
        or len(inference.flattened_mapping) != PRODUCTION_N_NODES
        or not semantic.endpoints
    ):
        raise ValueError("llama-full inference/semantic plan population drift")
    observed_counts = (
        sum(value.n_nodes for value in inference.checkpoint_manifests),
        sum(value.n_internal for value in inference.checkpoint_manifests),
        sum(value.n_leaves for value in inference.checkpoint_manifests),
    )
    if observed_counts != (
        PRODUCTION_N_NODES,
        PRODUCTION_N_INTERNAL,
        PRODUCTION_N_LEAVES,
    ):
        raise ValueError("llama-full checkpoint role totals drift")

    endpoint = semantic.endpoints[0]
    rows: list[dict[str, Any]] = []
    for array_index in range(PRODUCTION_N_NODES):
        mapping, _, _, binding = inference.select(array_index)
        tree_position = next(
            index
            for index, manifest in enumerate(inference.checkpoint_manifests)
            if manifest.tree_id == mapping.tree_id
        )
        source = load_node_endpoint_source(
            inference,
            stage_root=semantic.stage_root,
            tree_position=tree_position,
            node_ordinal=mapping.node_ordinal,
            endpoint=endpoint,
        )
        if (
            source.node_id != mapping.node_id
            or source.node_ordinal != mapping.node_ordinal
            or source.node_role != binding.node_role
            or not source.texts
        ):
            raise ValueError(f"llama-full mapping/source drift at array index {array_index}")
        response_path = canonical_file(source.responses.path, "llama-full responses")
        terminal_path = canonical_file(
            source.terminal_receipt.path, "llama-full terminal receipt"
        )
        response_bytes = response_path.read_bytes()
        if not response_bytes.endswith(b"\n"):
            raise ValueError(f"responses lack terminal LF at array index {array_index}")
        n_lf_records = response_bytes.count(b"\n")
        if n_lf_records != 1490:
            raise ValueError(
                f"llama-full response cardinality drift at {array_index}: {n_lf_records}"
            )
        if hashlib.sha256(response_bytes).hexdigest() != source.responses.sha256:
            raise ValueError(f"response changed after cache validation at {array_index}")
        if sha256_file(terminal_path) != source.terminal_receipt.sha256:
            raise ValueError(f"terminal changed after cache validation at {array_index}")
        rows.append(
            {
                "array_index": array_index,
                "tree_id": mapping.tree_id,
                "node_ordinal": mapping.node_ordinal,
                "node_id": mapping.node_id,
                "node_role": source.node_role,
                "checkpoint_identity_sha256": binding.checkpoint.sha256,
                "request_sha256": source.request_sha256,
                "responses": {
                    "path": str(response_path),
                    "sha256": source.responses.sha256,
                    "size_bytes": len(response_bytes),
                    "n_lf_records": n_lf_records,
                },
                "terminal_receipt": {
                    "path": str(terminal_path),
                    "sha256": source.terminal_receipt.sha256,
                },
            }
        )
    if [row["array_index"] for row in rows] != list(range(641)):
        raise AssertionError("llama-full artifact inventory order corruption")
    roles = {role: sum(row["node_role"] == role for row in rows) for role in {"internal", "leaf"}}
    if roles != {"internal": 276, "leaf": 365}:
        raise ValueError(f"llama-full inventory role totals drift: {roles}")
    return rows


def validate_task438_again(expected_row: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen the exceptional terminal pair immediately before adoption."""

    from weighttraits.behavior.allnode.semantic_runner import (
        load_node_endpoint_source,
        load_semantic_job_config,
    )

    _, semantic, inference, _, _ = load_semantic_job_config(
        _config("semantic_llama_full")
    )
    mapping, _, _, binding = inference.select(438)
    tree_position = next(
        index
        for index, manifest in enumerate(inference.checkpoint_manifests)
        if manifest.tree_id == mapping.tree_id
    )
    endpoints: list[dict[str, Any]] = []
    response_pin: dict[str, Any] | None = None
    terminal_pin: dict[str, Any] | None = None
    request_sha: str | None = None
    for endpoint in semantic.endpoints:
        source = load_node_endpoint_source(
            inference,
            stage_root=semantic.stage_root,
            tree_position=tree_position,
            node_ordinal=mapping.node_ordinal,
            endpoint=endpoint,
        )
        response_pin = source.responses.to_dict()
        terminal_pin = source.terminal_receipt.to_dict()
        request_sha = source.request_sha256
        endpoints.append(
            {
                "endpoint_id": endpoint.endpoint_id,
                "n_rows": len(source.texts),
                "endpoint_rows_sha256": source.endpoint_rows_sha256,
            }
        )
    evidence = {
        "array_index": 438,
        "tree_id": mapping.tree_id,
        "node_ordinal": mapping.node_ordinal,
        "node_id": mapping.node_id,
        "node_role": binding.node_role,
        "checkpoint_identity_sha256": binding.checkpoint.sha256,
        "request_sha256": request_sha,
        "responses": response_pin,
        "terminal_receipt": terminal_pin,
        "endpoints": endpoints,
        "literal_lf_validation": True,
        "accepted_artifact_not_scheduler_success": True,
    }
    for key in (
        "array_index",
        "tree_id",
        "node_ordinal",
        "node_id",
        "node_role",
        "checkpoint_identity_sha256",
        "request_sha256",
    ):
        if evidence[key] != expected_row[key]:
            raise ValueError(f"task438 evidence differs from inventory: {key}")
    if response_pin != {
        "path": expected_row["responses"]["path"],
        "sha256": expected_row["responses"]["sha256"],
    } or terminal_pin != expected_row["terminal_receipt"]:
        raise ValueError("task438 artifact pins differ from inventory")
    if (
        mapping.tree_id != "confirm_paper_tree_034"
        or mapping.node_id != "n6"
        or mapping.node_ordinal != 6
        or binding.node_role != "leaf"
        or binding.checkpoint.sha256
        != "b6fc9bcc40efd1dd72d82ca03dd227b27f838c0310406b43044574d5fd999bcf"
        or response_pin["sha256"]
        != "415a94fa3ae55ba472b43a6ce434a339bb4c2465f0a0d784435749db337dc607"
        or terminal_pin["sha256"]
        != "97e1dc8b53f9cb1f13e75e5b7f53022cd69db9783bdb15f2fb60bde6837f0193"
        or request_sha
        != "169717c7f9718bcdeffa097cc0ab3ac2a049ca06028b5f08fd129da07a747ed2"
    ):
        raise ValueError("task438 exact exception evidence drift")
    return evidence


def validate_direct_terminals() -> list[dict[str, Any]]:
    from weighttraits.behavior.allnode.semantic_runner import (
        load_semantic_job_config,
        validate_direct_cohort_terminal,
    )

    rows: list[dict[str, Any]] = []
    for cohort in ("llama_full", "llama_notrans", "flan_full", "flan_notrans"):
        _, plan, _, _, _ = load_semantic_job_config(_config(f"semantic_{cohort}"))
        result = validate_direct_cohort_terminal(plan)
        terminal = canonical_file(
            result.output_dir / "terminal_receipt.json", "direct cohort terminal"
        )
        digest = sha256_file(terminal)
        if digest != DIRECT_TERMINAL_SHA256[cohort]:
            raise ValueError(f"adopted direct terminal hash drift: {cohort}")
        if (
            result.terminal_receipt.get("n_trees") != 50
            or result.terminal_receipt.get("n_nodes") != 641
            or result.terminal_receipt.get("n_internal") != 276
            or result.terminal_receipt.get("n_leaves") != 365
        ):
            raise ValueError(f"adopted direct terminal population drift: {cohort}")
        rows.append(
            {
                "cohort": cohort,
                "path": str(terminal),
                "sha256": digest,
                "n_trees": 50,
                "n_nodes": 641,
                "n_internal": 276,
                "n_leaves": 365,
            }
        )
    return rows


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], CommandResult]


def run_command(argv: Sequence[str]) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        env={"LC_ALL": "C", "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    for value, label in ((completed.stdout, "stdout"), (completed.stderr, "stderr")):
        if len(value.encode("utf-8")) > MAX_COMMAND_BYTES or "\x00" in value:
            raise ValueError(f"scheduler {label} exceeds the safe output boundary")
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def require_success(result: CommandResult, label: str) -> str:
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed: rc={result.returncode}; stderr={result.stderr!r}"
        )
    return result.stdout


def query_duplicate_names(
    runner: CommandRunner = run_command,
) -> tuple[str, ...]:
    observed: set[str] = set()
    for job in build_plan():
        commands = (
            (
                "squeue",
                "--noheader",
                f"--name={job.scheduler_name}",
                "--format=%j",
            ),
            (
                "sacct",
                "-X",
                "--noheader",
                "--starttime=2026-08-31",
                f"--name={job.scheduler_name}",
                "--format=JobName%128",
                "--parsable2",
            ),
        )
        for command in commands:
            stdout = require_success(runner(command), command[0])
            for raw in stdout.splitlines():
                value = raw[:-1] if command[0] == "sacct" and raw.endswith("|") else raw
                if not value:
                    continue
                if value.strip() != value or value != job.scheduler_name:
                    raise ValueError(
                        f"{command[0]} returned a name outside the exact query: {value!r}"
                    )
                observed.add(value)
    return tuple(sorted(observed))


def _normalized_state(value: str) -> str:
    return value.split("+", 1)[0].split(" ", 1)[0]


def query_original_scheduler(
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    job_ids = ",".join(str(value) for value in range(183511, 183523))
    accounting_command = (
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        f"--jobs={job_ids}",
        "--format=JobIDRaw,JobID,State,ExitCode,Elapsed,Start,End,NodeList,Partition,QOS,Account,SubmitLine%4096",
    )
    stdout = require_success(runner(accounting_command), "original sacct")
    rows: list[dict[str, str]] = []
    fields = (
        "job_id_raw",
        "job_id",
        "state",
        "exit_code",
        "elapsed",
        "start",
        "end",
        "node_list",
        "partition",
        "qos",
        "account",
        "submit_line",
    )
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line:
            continue
        parts = line.split("|")
        if parts and parts[-1] == "":
            parts.pop()
        if len(parts) != len(fields):
            raise ValueError(f"unexpected original sacct row width at line {line_number}")
        row = dict(zip(fields, parts))
        if any(
            token in row["submit_line"]
            for token in (
                "OVERRIDE_MAX_STEPS",
                "MAX_TRAIN_SAMPLES",
                "MAX_EVAL_SAMPLES",
                "ROW_END",
            )
        ):
            raise ValueError(f"production override found in old job {row['job_id']}")
        rows.append(row)
    by_id = {row["job_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate original accounting job IDs")

    directs = []
    for job_id in range(183511, 183515):
        row = by_id.get(str(job_id))
        if row is None or (
            _normalized_state(row["state"]), row["exit_code"]
        ) != ("COMPLETED", "0:0"):
            raise ValueError(f"original direct job state drift: {job_id}")
        if (row["partition"], row["qos"], row["account"], row["node_list"]) != (
            "all",
            "normal",
            "statds",
            "n03",
        ):
            raise ValueError(f"original direct route drift: {job_id}")
        directs.append(
            {
                "job_id": str(job_id),
                "state": "COMPLETED",
                "exit_code": "0:0",
                "node": "n03",
                "partition": "all",
                "qos": "normal",
                "account": "statds",
            }
        )

    # Wright's ``sacct -X`` currently returns every array task but may omit the
    # synthetic array-parent row once accounting has settled.  When present it
    # must agree; the exact 640/1 task distribution below is the authoritative
    # evidence from which the logical parent failure is derived.
    parent = by_id.get("183515")
    if parent is not None and _normalized_state(parent["state"]) != "FAILED":
        raise ValueError("original llama-full array parent state drift")
    tasks = [
        row
        for job_id, row in by_id.items()
        if re.fullmatch(r"183515_[0-9]+", job_id) is not None
    ]
    tasks.sort(key=lambda row: int(row["job_id"].split("_", 1)[1]))
    if [int(row["job_id"].split("_", 1)[1]) for row in tasks] != list(range(641)):
        raise ValueError("original llama-full accounting lacks exact tasks 0..640")
    completed = [
        row
        for row in tasks
        if (_normalized_state(row["state"]), row["exit_code"])
        == ("COMPLETED", "0:0")
    ]
    failed = [
        row
        for row in tasks
        if (_normalized_state(row["state"]), row["exit_code"]) == ("FAILED", "1:0")
    ]
    if len(completed) != 640 or [row["job_id"] for row in failed] != ["183515_438"]:
        raise ValueError("original llama-full task state distribution drift")
    for row in tasks:
        if (row["partition"], row["qos"], row["account"], row["node_list"]) != (
            "all",
            "normal",
            "statds",
            "n03",
        ):
            raise ValueError(f"original llama-full task route drift: {row['job_id']}")
    failure = failed[0]
    if (
        failure["job_id_raw"] != "184002"
        or failure["elapsed"] != "00:02:09"
        or failure["start"] != "2026-09-01T22:47:42"
        or failure["end"] != "2026-09-01T22:49:51"
    ):
        raise ValueError("task438 exact scheduler evidence drift")

    queue_command = (
        "squeue",
        "--noheader",
        "--jobs=183516,183517,183518,183519,183520,183521,183522",
        "--format=%i|%T|%R|%P|%q|%a|%N",
    )
    queue_stdout = require_success(runner(queue_command), "blocked successor squeue")
    if queue_stdout.strip():
        raise ValueError("cancelled original successors unexpectedly remain in the live queue")

    return {
        "original_direct_jobs": directs,
        "original_llama_full_array": {
            "job_id": "183515",
            "parent_state": "FAILED",
            "n_tasks": 641,
            "n_completed_0_0": 640,
            "n_failed_1_0": 1,
            "sole_failure": {
                "job_id_raw": failure["job_id_raw"],
                "job_id": failure["job_id"],
                "state": "FAILED",
                "exit_code": "1:0",
                "elapsed": failure["elapsed"],
                "start": failure["start"],
                "end": failure["end"],
                "node": failure["node_list"],
                "partition": failure["partition"],
                "qos": failure["qos"],
                "account": failure["account"],
            },
        },
        "old_successors_absent_from_live_queue": True,
    }


def query_v1c1_scheduler(
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    """Validate the failed v1c1 chain from the Wright login node."""

    job_ids = ",".join(str(value) for value in range(184529, 184537))
    accounting_command = (
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        f"--jobs={job_ids}",
        "--format=JobIDRaw,JobID,JobName%128,State,ExitCode,Elapsed,Start,End,NodeList,Partition,QOS,Account,SubmitLine%4096",
    )
    stdout = require_success(runner(accounting_command), "v1c1 sacct")
    fields = (
        "job_id_raw",
        "job_id",
        "job_name",
        "state",
        "exit_code",
        "elapsed",
        "start",
        "end",
        "node_list",
        "partition",
        "qos",
        "account",
        "submit_line",
    )
    rows: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line:
            continue
        parts = line.split("|")
        if parts and parts[-1] == "":
            parts.pop()
        if len(parts) != len(fields):
            raise ValueError(f"unexpected v1c1 sacct row width at line {line_number}")
        row = dict(zip(fields, parts))
        base_id = row["job_id_raw"]
        if base_id in rows or base_id not in {str(value) for value in range(184529, 184537)}:
            raise ValueError(f"unexpected or duplicate v1c1 accounting row: {base_id}")
        rows[base_id] = row
    if sorted(rows, key=int) != [str(value) for value in range(184529, 184537)]:
        raise ValueError("v1c1 accounting lacks the exact eight parent rows")

    expected_names = (
        "wt-an-v1c1-adopt-llama-full",
        "wt-an-v1c1-semantic-llama-full",
        "wt-an-v1c1-infer-llama-notrans",
        "wt-an-v1c1-semantic-llama-notrans",
        "wt-an-v1c1-infer-flan-full",
        "wt-an-v1c1-semantic-flan-full",
        "wt-an-v1c1-infer-flan-notrans",
        "wt-an-v1c1-semantic-flan-notrans",
    )
    expected_dependencies: dict[str, str | None] = {
        "184529": None,
        **{
            str(job_id): f"afterok:{job_id - 1}"
            for job_id in range(184530, 184537)
        },
    }
    array_jobs = {"184531", "184533", "184535"}
    normalized_rows: list[dict[str, Any]] = []
    for offset, job_id in enumerate(range(184529, 184537)):
        key = str(job_id)
        row = rows[key]
        expected_display = f"{key}_[0-640%4]" if key in array_jobs else key
        dependency = expected_dependencies[key]
        submit = row["submit_line"]
        required_tokens = (
            f"--job-name={expected_names[offset]}",
            "--account=statds",
            "--partition=all",
            "--qos=normal",
            "--nodelist=n03",
            "--exclude=n01",
            "--no-requeue",
        )
        if (
            row["job_id"] != expected_display
            or row["job_name"] != expected_names[offset]
            or (row["partition"], row["qos"], row["account"])
            != ("all", "normal", "statds")
            or any(token not in submit for token in required_tokens)
            or any(
                token in submit
                for token in (
                    "OVERRIDE_MAX_STEPS",
                    "MAX_TRAIN_SAMPLES",
                    "MAX_EVAL_SAMPLES",
                    "ROW_END",
                )
            )
        ):
            raise ValueError(f"v1c1 route/name/override drift: {key}")
        if dependency is None:
            if "--dependency=" in submit or "--gres=gpu:" in submit or "--array=" in submit:
                raise ValueError("v1c1 adoption resource/dependency drift")
        else:
            if (
                f"--dependency={dependency}" not in submit
                or "--gres=gpu:nvidia-l40:1" not in submit
            ):
                raise ValueError(f"v1c1 dependency/GPU drift: {key}")
            if (key in array_jobs) != ("--array=0-640%4" in submit):
                raise ValueError(f"v1c1 array shape drift: {key}")
        if key == "184529":
            if (
                (_normalized_state(row["state"]), row["exit_code"])
                != ("FAILED", "1:0")
                or row["elapsed"] != "00:01:38"
                or row["start"] != "2026-09-02T13:42:46"
                or row["end"] != "2026-09-02T13:44:24"
                or row["node_list"] != "n03"
            ):
                raise ValueError("v1c1 adoption failure evidence drift")
        elif (
            _normalized_state(row["state"]),
            row["exit_code"],
            row["node_list"],
        ) != ("CANCELLED", "0:0", "None assigned"):
            raise ValueError(f"v1c1 successor accounting state drift: {key}")
        normalized_rows.append(
            {
                "job_id_raw": row["job_id_raw"],
                "job_id": row["job_id"],
                "job_name": row["job_name"],
                "state": _normalized_state(row["state"]),
                "exit_code": row["exit_code"],
                "elapsed": row["elapsed"],
                "start": row["start"],
                "end": row["end"],
                "node_list": row["node_list"],
                "partition": row["partition"],
                "qos": row["qos"],
                "account": row["account"],
                "dependency": dependency,
            }
        )

    queue_command = (
        "squeue",
        "--noheader",
        "--jobs=184530,184531,184532,184533,184534,184535,184536",
        "--format=%i|%j|%T|%R|%P|%q|%a|%N|%E",
    )
    queue_stdout = require_success(runner(queue_command), "v1c1 successor squeue")
    if queue_stdout.strip():
        raise ValueError("cancelled v1c1 successors unexpectedly remain in the live queue")
    return {
        "jobs": normalized_rows,
        "successor_queue": [],
        "successors_absent_from_live_queue": True,
        "adoption_failed_before_artifact_validation": True,
        "predecessor_jobs_mutated": False,
    }


def _command_observation(
    argv: Sequence[str], result: CommandResult
) -> dict[str, Any]:
    rendered = "\0".join(argv).encode("utf-8")
    return {
        "argv": list(argv),
        "argv_sha256": hashlib.sha256(rendered).hexdigest(),
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        "stdout_n_lines": len(result.stdout.splitlines()),
        "stderr_n_lines": len(result.stderr.splitlines()),
    }


def capture_scheduler_evidence(
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    """Capture immutable login-node evidence for compute-side consumption."""

    if socket.gethostname() != "headnode":
        raise ValueError("scheduler evidence must be captured on Wright headnode")
    observations: list[dict[str, Any]] = []

    def recording_runner(argv: Sequence[str]) -> CommandResult:
        result = runner(argv)
        observations.append(_command_observation(argv, result))
        return result

    original = query_original_scheduler(recording_runner)
    v1c1 = query_v1c1_scheduler(recording_runner)
    if len(observations) != 4 or any(row["returncode"] != 0 for row in observations):
        raise ValueError("scheduler evidence must contain four successful queries")
    core = {
        "schema": f"{SCHEMA_PREFIX}_scheduler_evidence.v1",
        "valid": True,
        "capture_host": "headnode",
        "captured_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "observations": observations,
        "state_contract": {"original": original, "v1c1": v1c1},
        "scheduler_queries": 4,
        "scheduler_actions": 0,
        "predecessor_jobs_mutated": False,
    }
    payload = {**core, "scheduler_evidence_sha256": canonical_sha256(core)}
    validate_scheduler_evidence(payload)
    return payload


def validate_scheduler_evidence(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "valid",
        "capture_host",
        "captured_at_utc",
        "observations",
        "state_contract",
        "scheduler_queries",
        "scheduler_actions",
        "predecessor_jobs_mutated",
        "scheduler_evidence_sha256",
    }
    core = {key: payload[key] for key in payload if key != "scheduler_evidence_sha256"}
    observations = payload.get("observations")
    state_contract = payload.get("state_contract")
    if (
        set(payload) != required
        or payload["schema"] != f"{SCHEMA_PREFIX}_scheduler_evidence.v1"
        or payload["valid"] is not True
        or payload["capture_host"] != "headnode"
        or not isinstance(payload["captured_at_utc"], str)
        or not payload["captured_at_utc"].endswith("Z")
        or not isinstance(observations, list)
        or len(observations) != 4
        or not isinstance(state_contract, dict)
        or set(state_contract) != {"original", "v1c1"}
        or payload["scheduler_queries"] != 4
        or payload["scheduler_actions"] != 0
        or payload["predecessor_jobs_mutated"] is not False
        or payload["scheduler_evidence_sha256"] != canonical_sha256(core)
    ):
        raise ValueError("scheduler evidence contract drift")
    observation_keys = {
        "argv",
        "argv_sha256",
        "returncode",
        "stdout_sha256",
        "stderr_sha256",
        "stdout_n_lines",
        "stderr_n_lines",
    }
    for observation in observations:
        if (
            not isinstance(observation, dict)
            or set(observation) != observation_keys
            or not isinstance(observation["argv"], list)
            or observation["returncode"] != 0
            or observation["argv_sha256"]
            != hashlib.sha256("\0".join(observation["argv"]).encode("utf-8")).hexdigest()
        ):
            raise ValueError("scheduler query observation drift")
        for key in ("argv_sha256", "stdout_sha256", "stderr_sha256"):
            require_sha(observation[key], f"scheduler observation {key}")


def scheduler_stable_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the revalidation fields whose bytes must remain unchanged."""

    validate_scheduler_evidence(payload)
    return {
        "capture_host": payload["capture_host"],
        "observations": payload["observations"],
        "state_contract": payload["state_contract"],
        "scheduler_queries": payload["scheduler_queries"],
        "scheduler_actions": payload["scheduler_actions"],
        "predecessor_jobs_mutated": payload["predecessor_jobs_mutated"],
    }


def build_acceptance(
    inventory: Sequence[Mapping[str, Any]],
    direct_terminals: Sequence[Mapping[str, Any]],
    scheduler_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if len(inventory) != 641:
        raise ValueError("acceptance inventory must contain exactly 641 ordered rows")
    validate_scheduler_evidence(scheduler_evidence)
    predecessor = attest_v1c1_failure_files()
    inventory_rows = [dict(row) for row in inventory]
    direct_rows = [dict(row) for row in direct_terminals]
    task438 = validate_task438_again(inventory_rows[438])
    core = {
        "schema": f"{SCHEMA_PREFIX}_llama_full_artifact_acceptance.v1",
        "valid": True,
        "status": "completed",
        "submission_id": SUBMISSION_ID,
        "accepted_cohort": "llama_full",
        "accepted_without_inference_rerun": True,
        "accepted_artifacts_not_scheduler_success": True,
        "read_only_validation": True,
        "jsonl_framing": {
            "record_separator": "literal_byte_0a",
            "unicode_line_separators_are_string_data": True,
            "terminal_lf_required": True,
        },
        "population": {
            "n_trees": 50,
            "n_nodes": 641,
            "n_internal": 276,
            "n_leaves": 365,
            "n_response_records": 955090,
        },
        "original_pins": {
            label: {"path": str(path), "sha256": digest}
            for label, (path, digest) in OLD_PINS.items()
        },
        "original_gate_contract_sha256": OLD_GATE_SHA256,
        "source_scope_sha256": SOURCE_SCOPE_SHA256,
        "scheduler_evidence": dict(scheduler_evidence),
        "scheduler_evidence_file": {
            "path": str(SCHEDULER_EVIDENCE),
            "sha256": hashlib.sha256(
                canonical_bytes(dict(scheduler_evidence))
            ).hexdigest(),
        },
        "scheduler_evidence_origin": "wright_login_preflight_revalidated_at_launch",
        "compute_node_scheduler_queries": 0,
        "predecessor_v1c1_files": predecessor["pins"],
        "predecessor_v1c1_failure": {
            key: value for key, value in predecessor.items() if key != "pins"
        },
        "predecessor_jobs_mutated": False,
        "artifact_inventory": inventory_rows,
        "artifact_inventory_sha256": canonical_sha256(inventory_rows),
        "direct_terminals": direct_rows,
        "direct_terminals_sha256": canonical_sha256(direct_rows),
        "task438_exception_evidence": task438,
        "execution_overrides": {},
        "scheduler_actions_during_validation": 0,
        "inference_actions_during_validation": 0,
        "output_absence_checks_during_adoption": 2,
        "no_automatic_recovery": True,
        "retry_allowed": False,
        "resubmission_allowed": False,
    }
    return {**core, "acceptance_sha256": canonical_sha256(core)}


def validate_acceptance_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "valid",
        "status",
        "submission_id",
        "accepted_cohort",
        "accepted_without_inference_rerun",
        "accepted_artifacts_not_scheduler_success",
        "read_only_validation",
        "jsonl_framing",
        "population",
        "original_pins",
        "original_gate_contract_sha256",
        "source_scope_sha256",
        "scheduler_evidence",
        "scheduler_evidence_file",
        "scheduler_evidence_origin",
        "compute_node_scheduler_queries",
        "predecessor_v1c1_files",
        "predecessor_v1c1_failure",
        "predecessor_jobs_mutated",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "direct_terminals",
        "direct_terminals_sha256",
        "task438_exception_evidence",
        "execution_overrides",
        "scheduler_actions_during_validation",
        "inference_actions_during_validation",
        "output_absence_checks_during_adoption",
        "no_automatic_recovery",
        "retry_allowed",
        "resubmission_allowed",
        "acceptance_sha256",
    }
    if set(payload) != required:
        raise ValueError("acceptance receipt keys drift")
    scheduler = payload.get("scheduler_evidence")
    if not isinstance(scheduler, dict):
        raise ValueError("acceptance scheduler evidence drift")
    validate_scheduler_evidence(scheduler)
    core = {key: payload[key] for key in payload if key != "acceptance_sha256"}
    if (
        payload["schema"] != f"{SCHEMA_PREFIX}_llama_full_artifact_acceptance.v1"
        or payload["valid"] is not True
        or payload["status"] != "completed"
        or payload["submission_id"] != SUBMISSION_ID
        or payload["accepted_cohort"] != "llama_full"
        or payload["accepted_without_inference_rerun"] is not True
        or payload["accepted_artifacts_not_scheduler_success"] is not True
        or payload["read_only_validation"] is not True
        or payload["scheduler_evidence_file"]
        != {
            "path": str(SCHEDULER_EVIDENCE),
            "sha256": hashlib.sha256(canonical_bytes(scheduler)).hexdigest(),
        }
        or payload["scheduler_evidence_origin"]
        != "wright_login_preflight_revalidated_at_launch"
        or payload["compute_node_scheduler_queries"] != 0
        or payload["predecessor_v1c1_files"]
        != {
            label: {"path": str(path), "sha256": digest}
            for label, (path, digest) in V1C1_PINS.items()
        }
        or payload["predecessor_v1c1_failure"]
        != {
            "deployment": str(V1C1_DEPLOYMENT),
            "acceptance_receipt": {
                "path": str(V1C1_ACCEPTANCE_RECEIPT),
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
        or payload["predecessor_jobs_mutated"] is not False
        or payload["population"]
        != {
            "n_trees": 50,
            "n_nodes": 641,
            "n_internal": 276,
            "n_leaves": 365,
            "n_response_records": 955090,
        }
        or not isinstance(payload["artifact_inventory"], list)
        or len(payload["artifact_inventory"]) != 641
        or payload["artifact_inventory_sha256"]
        != canonical_sha256(payload["artifact_inventory"])
        or not isinstance(payload["direct_terminals"], list)
        or len(payload["direct_terminals"]) != 4
        or payload["direct_terminals_sha256"]
        != canonical_sha256(payload["direct_terminals"])
        or payload["execution_overrides"] != {}
        or payload["scheduler_actions_during_validation"] != 0
        or payload["inference_actions_during_validation"] != 0
        or payload["output_absence_checks_during_adoption"] != 2
        or payload["no_automatic_recovery"] is not True
        or payload["retry_allowed"] is not False
        or payload["resubmission_allowed"] is not False
        or payload["acceptance_sha256"] != canonical_sha256(core)
    ):
        raise ValueError("acceptance receipt contract drift")


def output_targets() -> tuple[Path, ...]:
    return tuple(
        Path(job.output_path)
        for job in build_plan()[1:]
        if not (job.kind == "semantic_allnode_gpu" and job.cohort == "llama_full")
    )


def preserved_llama_semantic_completion_targets() -> tuple[Path, ...]:
    """Files that must remain absent while the 50-tree cache is preserved."""

    root = STAGE / "outputs" / "allnode_v1" / "semantic" / "llama_full"
    return (
        root / "translation" / "runset_receipt.json",
        root / "terminal_receipt.json",
    )


def assert_output_targets_absent(phase: str) -> dict[str, bool]:
    if not isinstance(phase, str) or not phase or phase.strip() != phase:
        raise ValueError("output-absence phase must be nonempty trimmed text")
    targets = (*output_targets(), *preserved_llama_semantic_completion_targets())
    observed = {str(path): os.path.lexists(path) for path in targets}
    existing = [path for path, exists in observed.items() if exists]
    if existing:
        raise FileExistsError(f"downstream output appeared during {phase}: {existing}")
    return observed


def check_absence(*, before_preflight: bool) -> dict[str, bool]:
    targets: list[Path] = [INTENT, JOBS_RECEIPT, LOG_DIR, ACCEPTANCE_RECEIPT]
    targets.extend(output_targets())
    targets.extend(preserved_llama_semantic_completion_targets())
    if before_preflight:
        targets.extend((SCHEDULER_EVIDENCE, EXPECTED_ACCEPTANCE, DRY_RUN_AUDIT))
    observed = {str(path): os.path.lexists(path) for path in targets}
    existing = [path for path, exists in observed.items() if exists]
    if existing:
        raise FileExistsError(f"pre-existing continuation state: {existing}")
    return observed


def execution_contract(code_manifest_sha256: str, acceptance_sha256: str) -> dict[str, Any]:
    plan = [job.to_dict() for job in build_plan()]
    return {
        "schema": f"{SCHEMA_PREFIX}_execution_contract.v1",
        "valid": True,
        "submission_id": SUBMISSION_ID,
        "stage_root": str(STAGE),
        "frozen_stage": str(FROZEN),
        "original_deployment": str(OLD_DEPLOYMENT),
        "continuation_deployment": str(DEPLOYMENT),
        "code_manifest": {
            "path": str(CODE_MANIFEST),
            "sha256": require_sha(code_manifest_sha256, "code manifest SHA-256"),
        },
        "expected_acceptance": {
            "path": str(EXPECTED_ACCEPTANCE),
            "sha256": require_sha(acceptance_sha256, "acceptance SHA-256"),
        },
        "acceptance_receipt_path": str(ACCEPTANCE_RECEIPT),
        "scheduler_evidence_path": str(SCHEDULER_EVIDENCE),
        "intent_path": str(INTENT),
        "jobs_receipt_path": str(JOBS_RECEIPT),
        "log_dir": str(LOG_DIR),
        "population": {
            "n_trees": 50,
            "n_nodes": 641,
            "n_internal": 276,
            "n_leaves": 365,
            "primary_population": "all_trained_nodes",
            "sensitivity_population": "trained_leaves",
            "exclude_untrained_root": True,
        },
        "scheduler_policy": {
            "account": "statds",
            "partition": "all",
            "qos": "normal",
            "required_node": "n03",
            "excluded_node": "n01",
            "requeue": False,
            "gpu_type": "nvidia-l40",
            "gpus_per_gpu_job": 1,
            "maximum_concurrent_gpu_tasks": 4,
            "gpu_jobs_dependency_serialized": True,
            "adoption_gate_scheduler_queries": 0,
            "scheduler_evidence_origin": "wright_login_preflight_revalidated_at_launch",
        },
        "jobs": plan,
        "jobs_sha256": canonical_sha256(plan),
        "accepted_without_llama_full_inference_rerun": True,
        "execution_overrides": {},
        "no_automatic_recovery": True,
        "retry_allowed": False,
        "cancel_on_failure": False,
        "resubmission_allowed": False,
    }


def run_test_attestation(*, code_manifest_sha256: str) -> dict[str, Any]:
    manifest_sha = require_sha(code_manifest_sha256, "code manifest SHA-256")
    verify_code_manifest(CODE_MANIFEST, manifest_sha)
    if os.path.lexists(TEST_ATTESTATION):
        raise FileExistsError("continuation test attestation already exists")
    command = (
        PYTHON_BIN,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(DEPLOYMENT / "tests"),
        "-p",
        "test_*.py",
    )
    environment = {
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(DEPLOYMENT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    completed = subprocess.run(
        list(command),
        cwd=DEPLOYMENT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    combined = completed.stdout + completed.stderr
    match = re.search(r"Ran ([0-9]+) tests? in ", combined)
    n_tests = int(match.group(1)) if match is not None else None
    if (
        completed.returncode != 0
        or n_tests != EXPECTED_TEST_COUNT
        or not re.search(r"\nOK\s*$", combined)
    ):
        raise RuntimeError(
            f"continuation test suite failed: rc={completed.returncode}; "
            f"n_tests={n_tests}; stdout={completed.stdout[-2000:]!r}; "
            f"stderr={completed.stderr[-4000:]!r}"
        )
    core = {
        "schema": f"{SCHEMA_PREFIX}_test_attestation.v1",
        "valid": True,
        "code_manifest": {"path": str(CODE_MANIFEST), "sha256": manifest_sha},
        "interpreter": sys.executable,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "command": list(command),
        "returncode": 0,
        "n_tests": n_tests,
        "status": "OK",
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "scheduler_actions": 0,
    }
    payload = {**core, "test_attestation_sha256": canonical_sha256(core)}
    publish_exclusive(TEST_ATTESTATION, canonical_bytes(payload))
    return payload


def load_test_attestation(code_manifest_sha256: str) -> tuple[dict[str, Any], str]:
    payload, raw = read_canonical(TEST_ATTESTATION, "continuation test attestation")
    required = {
        "schema",
        "valid",
        "code_manifest",
        "interpreter",
        "python_version",
        "command",
        "returncode",
        "n_tests",
        "status",
        "stdout_sha256",
        "stderr_sha256",
        "scheduler_actions",
        "test_attestation_sha256",
    }
    core = {key: payload[key] for key in payload if key != "test_attestation_sha256"}
    if (
        set(payload) != required
        or payload["schema"] != f"{SCHEMA_PREFIX}_test_attestation.v1"
        or payload["valid"] is not True
        or payload["code_manifest"]
        != {"path": str(CODE_MANIFEST), "sha256": code_manifest_sha256}
        or payload["interpreter"] != PYTHON_BIN
        or payload["python_version"] != "3.11.15"
        or payload["returncode"] != 0
        or payload["n_tests"] != EXPECTED_TEST_COUNT
        or payload["status"] != "OK"
        or payload["scheduler_actions"] != 0
        or payload["test_attestation_sha256"] != canonical_sha256(core)
    ):
        raise ValueError("continuation test attestation contract drift")
    return payload, hashlib.sha256(raw).hexdigest()


def load_scheduler_evidence(expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    payload, raw = read_canonical(SCHEDULER_EVIDENCE, "v1c5 scheduler evidence")
    validate_scheduler_evidence(payload)
    if hashlib.sha256(raw).hexdigest() != require_sha(
        expected_sha256, "scheduler evidence SHA-256"
    ):
        raise ValueError("scheduler evidence full-file hash drift")
    return payload, raw


def make_preflight(
    *, code_manifest_sha256: str, runner: CommandRunner = run_command
) -> dict[str, Any]:
    manifest_sha = require_sha(code_manifest_sha256, "code manifest SHA-256")
    verify_code_manifest(CODE_MANIFEST, manifest_sha)
    test_attestation, test_attestation_file_sha = load_test_attestation(manifest_sha)
    before_attestation = attest_original_runtime("preflight")
    predecessor = attest_v1c1_failure_files()
    absence = check_absence(before_preflight=True)
    collisions = query_duplicate_names(runner)
    if collisions:
        raise ValueError(f"continuation scheduler names already exist: {collisions}")
    scheduler = capture_scheduler_evidence(runner)
    scheduler_raw = canonical_bytes(scheduler)
    scheduler_file_sha = hashlib.sha256(scheduler_raw).hexdigest()
    inventory = artifact_inventory_pass()
    direct = validate_direct_terminals()
    acceptance = build_acceptance(inventory, direct, scheduler)
    validate_acceptance_payload(acceptance)
    acceptance_raw = canonical_bytes(acceptance)
    acceptance_file_sha = hashlib.sha256(acceptance_raw).hexdigest()
    contract = execution_contract(manifest_sha, acceptance_file_sha)
    audit_core = {
        "schema": f"{SCHEMA_PREFIX}_dry_run_audit.v1",
        "valid": True,
        "submission_id": SUBMISSION_ID,
        "code_manifest": {"path": str(CODE_MANIFEST), "sha256": manifest_sha},
        "original_pins": {
            label: {"path": str(path), "sha256": digest}
            for label, (path, digest) in OLD_PINS.items()
        },
        "original_gate_contract_sha256": OLD_GATE_SHA256,
        "source_scope_sha256": SOURCE_SCOPE_SHA256,
        "runtime_attestation": before_attestation,
        "test_attestation": {
            "path": str(TEST_ATTESTATION),
            "sha256": test_attestation_file_sha,
            "payload": test_attestation,
        },
        "scheduler_evidence": scheduler,
        "scheduler_evidence_file": {
            "path": str(SCHEDULER_EVIDENCE),
            "sha256": scheduler_file_sha,
        },
        "scheduler_queries": 4,
        "scheduler_name_collisions": [],
        "predecessor_v1c1": predecessor,
        "initial_state_absence": absence,
        "llama_full_inventory_sha256": acceptance["artifact_inventory_sha256"],
        "task438_exception_evidence": acceptance["task438_exception_evidence"],
        "direct_terminals": direct,
        "expected_acceptance": {
            "path": str(EXPECTED_ACCEPTANCE),
            "sha256": acceptance_file_sha,
            "acceptance_sha256": acceptance["acceptance_sha256"],
        },
        "execution_contract": contract,
        "execution_contract_sha256": canonical_sha256(contract),
        "decoder_repair": {
            "isolated_copy_only": True,
            "original_deployment_modified": False,
            "runtime_decoder_uses_literal_lf": True,
            "semantic_decoder_uses_literal_lf": True,
            "unicode_u2028_preserved_as_json_string_data": True,
        },
        "execution_overrides": {},
        "scheduler_actions": 0,
        "inference_actions": 0,
        "no_automatic_recovery": True,
    }
    audit = {**audit_core, "dry_run_audit_sha256": canonical_sha256(audit_core)}
    audit_raw = canonical_bytes(audit)
    publish_exclusive(SCHEDULER_EVIDENCE, scheduler_raw)
    publish_exclusive(EXPECTED_ACCEPTANCE, acceptance_raw)
    publish_exclusive(DRY_RUN_AUDIT, audit_raw)
    verify_code_manifest(CODE_MANIFEST, manifest_sha)
    attest_original_runtime("preflight")
    attest_v1c1_failure_files()
    return audit


def load_dry_run(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    if Path(path) != DRY_RUN_AUDIT:
        raise ValueError("dry-run audit path drift")
    payload, raw = read_canonical(path, "continuation dry-run audit")
    if hashlib.sha256(raw).hexdigest() != require_sha(expected_sha256, "audit SHA-256"):
        raise ValueError("continuation dry-run full-file hash drift")
    if set(payload) != {
        "schema",
        "valid",
        "submission_id",
        "code_manifest",
        "original_pins",
        "original_gate_contract_sha256",
        "source_scope_sha256",
        "runtime_attestation",
        "test_attestation",
        "scheduler_evidence",
        "scheduler_evidence_file",
        "scheduler_queries",
        "scheduler_name_collisions",
        "predecessor_v1c1",
        "initial_state_absence",
        "llama_full_inventory_sha256",
        "task438_exception_evidence",
        "direct_terminals",
        "expected_acceptance",
        "execution_contract",
        "execution_contract_sha256",
        "decoder_repair",
        "execution_overrides",
        "scheduler_actions",
        "inference_actions",
        "no_automatic_recovery",
        "dry_run_audit_sha256",
    }:
        raise ValueError("continuation dry-run keys drift")
    core = {key: payload[key] for key in payload if key != "dry_run_audit_sha256"}
    if (
        payload["schema"] != f"{SCHEMA_PREFIX}_dry_run_audit.v1"
        or payload["valid"] is not True
        or payload["submission_id"] != SUBMISSION_ID
        or payload["scheduler_name_collisions"] != []
        or payload["scheduler_queries"] != 4
        or payload["execution_overrides"] != {}
        or payload["scheduler_actions"] != 0
        or payload["inference_actions"] != 0
        or payload["no_automatic_recovery"] is not True
        or payload["dry_run_audit_sha256"] != canonical_sha256(core)
        or payload["execution_contract_sha256"]
        != canonical_sha256(payload["execution_contract"])
    ):
        raise ValueError("continuation dry-run contract drift")
    manifest = payload["code_manifest"]
    acceptance = payload["expected_acceptance"]
    if not isinstance(manifest, dict) or set(manifest) != {"path", "sha256"}:
        raise ValueError("dry-run code manifest pin drift")
    if not isinstance(acceptance, dict) or set(acceptance) != {
        "path",
        "sha256",
        "acceptance_sha256",
    }:
        raise ValueError("dry-run acceptance pin drift")
    expected_contract = execution_contract(manifest["sha256"], acceptance["sha256"])
    if payload["execution_contract"] != expected_contract:
        raise ValueError("dry-run execution contract differs from exact v1c5 plan")
    verify_code_manifest(manifest["path"], manifest["sha256"])
    test_payload, test_file_sha = load_test_attestation(manifest["sha256"])
    if payload["test_attestation"] != {
        "path": str(TEST_ATTESTATION),
        "sha256": test_file_sha,
        "payload": test_payload,
    }:
        raise ValueError("dry-run test attestation pin drift")
    if payload["predecessor_v1c1"] != attest_v1c1_failure_files():
        raise ValueError("dry-run v1c1 predecessor evidence drift")
    scheduler_file = payload["scheduler_evidence_file"]
    if not isinstance(scheduler_file, dict) or scheduler_file != {
        "path": str(SCHEDULER_EVIDENCE),
        "sha256": scheduler_file.get("sha256")
        if isinstance(scheduler_file, dict)
        else None,
    }:
        raise ValueError("dry-run scheduler evidence pin drift")
    pinned_scheduler, scheduler_raw = load_scheduler_evidence(
        scheduler_file["sha256"]
    )
    if (
        pinned_scheduler != payload["scheduler_evidence"]
        or hashlib.sha256(scheduler_raw).hexdigest() != scheduler_file["sha256"]
    ):
        raise ValueError("dry-run scheduler evidence differs from pinned file")
    expected_payload, expected_raw = read_canonical(
        acceptance["path"], "expected artifact acceptance"
    )
    validate_acceptance_payload(expected_payload)
    if (
        hashlib.sha256(expected_raw).hexdigest() != acceptance["sha256"]
        or expected_payload["acceptance_sha256"] != acceptance["acceptance_sha256"]
    ):
        raise ValueError("expected acceptance pin drift")
    return payload, raw


def adopt(
    *,
    code_manifest_sha256: str,
    expected_acceptance_path: str | Path,
    expected_acceptance_sha256: str,
    scheduler_evidence_path: str | Path,
    scheduler_evidence_sha256: str,
    acceptance_receipt_path: str | Path,
) -> dict[str, Any]:
    if Path(expected_acceptance_path) != EXPECTED_ACCEPTANCE:
        raise ValueError("expected acceptance path drift")
    if Path(acceptance_receipt_path) != ACCEPTANCE_RECEIPT:
        raise ValueError("acceptance receipt path drift")
    if Path(scheduler_evidence_path) != SCHEDULER_EVIDENCE:
        raise ValueError("scheduler evidence path drift")
    if os.path.lexists(ACCEPTANCE_RECEIPT):
        raise FileExistsError("acceptance receipt already exists")
    verify_code_manifest(CODE_MANIFEST, code_manifest_sha256)
    expected, expected_raw = read_canonical(EXPECTED_ACCEPTANCE, "expected acceptance")
    validate_acceptance_payload(expected)
    if hashlib.sha256(expected_raw).hexdigest() != require_sha(
        expected_acceptance_sha256, "expected acceptance SHA-256"
    ):
        raise ValueError("expected acceptance full-file hash drift")
    attest_original_runtime("before-runner")
    attest_v1c1_failure_files()
    scheduler, scheduler_raw = load_scheduler_evidence(scheduler_evidence_sha256)
    if (
        scheduler != expected["scheduler_evidence"]
        or expected["scheduler_evidence_file"]
        != {
            "path": str(SCHEDULER_EVIDENCE),
            "sha256": hashlib.sha256(scheduler_raw).hexdigest(),
        }
    ):
        raise ValueError("pinned scheduler evidence differs from expected acceptance")
    assert_output_targets_absent("before_adoption_validation")
    first = artifact_inventory_pass()
    second = artifact_inventory_pass()
    if first != second or canonical_sha256(first) != canonical_sha256(second):
        raise ValueError("two independent llama-full cache passes differ")
    direct = validate_direct_terminals()
    rebuilt = build_acceptance(second, direct, scheduler)
    if rebuilt != expected or canonical_bytes(rebuilt) != expected_raw:
        raise ValueError("compute-node acceptance differs from precomputed bytes")
    attest_original_runtime("after-runner")
    verify_code_manifest(CODE_MANIFEST, code_manifest_sha256)
    assert_output_targets_absent("immediately_before_acceptance_publication")
    publish_exclusive(ACCEPTANCE_RECEIPT, expected_raw)
    return rebuilt


def attest_continuation(
    *,
    code_manifest_path: str | Path,
    code_manifest_sha256: str,
    acceptance_path: str | Path | None = None,
    acceptance_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = verify_code_manifest(code_manifest_path, code_manifest_sha256)
    result: dict[str, Any] = {
        "schema": f"{SCHEMA_PREFIX}_attestation.v1",
        "valid": True,
        "code_manifest_sha256": require_sha(
            code_manifest_sha256, "code manifest SHA-256"
        ),
        "n_code_files": manifest["n_files"],
    }
    if acceptance_path is not None or acceptance_sha256 is not None:
        if Path(acceptance_path or "") != ACCEPTANCE_RECEIPT:
            raise ValueError("acceptance receipt path drift")
        payload, raw = read_canonical(ACCEPTANCE_RECEIPT, "artifact acceptance receipt")
        validate_acceptance_payload(payload)
        expected = require_sha(acceptance_sha256, "acceptance receipt SHA-256")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("artifact acceptance receipt full-file hash drift")
        result["acceptance_receipt_sha256"] = expected
        result["artifact_inventory_sha256"] = payload["artifact_inventory_sha256"]
    return result


def _expected_job_for_runner(logical_id: str) -> Job:
    matches = [job for job in build_plan() if job.runner_logical_id == logical_id]
    if len(matches) != 1:
        raise ValueError(f"runner logical ID is not in the exact continuation: {logical_id}")
    return matches[0]


def run_bound_workload(
    *,
    code_manifest_sha256: str,
    acceptance_sha256: str,
    config_path: str | Path,
    array_index: int | None,
) -> Mapping[str, Any]:
    """Run one pinned original workload through the patched isolated package."""

    config = canonical_file(config_path, "bound original job config")
    logical_id = config.stem
    job = _expected_job_for_runner(logical_id)
    if str(config) != job.config_path or sha256_file(config) != job.config_sha256:
        raise ValueError("bound original job config pin drift")
    expected_environment = {
        "WT_ALLNODE_CONTINUATION_LOGICAL_ID": job.logical_id,
        "WT_ALLNODE_COHORT_KEY": job.cohort,
        "WT_ALLNODE_JOB_CONFIG": job.config_path,
        "WT_ALLNODE_JOB_CONFIG_SHA256": job.config_sha256,
        "WT_ALLNODE_KIND": job.kind,
        "WT_ALLNODE_LOGICAL_ID": job.runner_logical_id,
        "WT_ALLNODE_OUTPUT_DIR": job.output_path,
        "WT_ALLNODE_STAGE_ROOT": str(STAGE),
        "WT_ALLNODE_DEPLOYMENT_ROOT": str(OLD_DEPLOYMENT),
        "WT_ALLNODE_SOURCE_RESOLUTION": str(OLD_PINS["source_resolution"][0]),
        "WT_ALLNODE_SOURCE_RESOLUTION_SHA256": OLD_PINS["source_resolution"][1],
        "WT_ALLNODE_SOURCE_SCOPE_SHA256": SOURCE_SCOPE_SHA256,
        "WT_ALLNODE_CODE_MANIFEST": str(OLD_PINS["code_manifest"][0]),
        "WT_ALLNODE_CODE_MANIFEST_SHA256": OLD_PINS["code_manifest"][1],
        "WT_DOWNSTREAM_CODE_MANIFEST": str(OLD_PINS["downstream_code_manifest"][0]),
        "WT_DOWNSTREAM_CODE_MANIFEST_SHA256": OLD_PINS["downstream_code_manifest"][1],
        "WT_ALLNODE_PRODUCTION_INTERPRETER": PYTHON_BIN,
        "WT_ALLNODE_PYTHON_VERSION": "3.11.15",
        "WT_V1C5_CODE_MANIFEST": str(CODE_MANIFEST),
        "WT_V1C5_CODE_MANIFEST_SHA256": code_manifest_sha256,
        "WT_V1C5_ACCEPTANCE_RECEIPT": str(ACCEPTANCE_RECEIPT),
        "WT_V1C5_ACCEPTANCE_RECEIPT_SHA256": acceptance_sha256,
    }
    for name, expected in expected_environment.items():
        if os.environ.get(name) != expected:
            raise ValueError(f"continuation launcher environment drift: {name}")

    verify_code_manifest(CODE_MANIFEST, code_manifest_sha256)
    attest_continuation(
        code_manifest_path=CODE_MANIFEST,
        code_manifest_sha256=code_manifest_sha256,
        acceptance_path=ACCEPTANCE_RECEIPT,
        acceptance_sha256=acceptance_sha256,
    )
    attest_original_runtime("before-runner")
    result: object | None = None
    try:
        if job.kind == "inference_allnode_gpu":
            if (
                isinstance(array_index, bool)
                or not isinstance(array_index, int)
                or not 0 <= array_index <= 640
            ):
                raise ValueError("inference continuation requires array index 0..640")
            from weighttraits.behavior.allnode.inference_runner import (
                execute_filesystem_inference,
                load_inference_job_config,
            )

            loaded, _, plan_path, loaded_config_path, loaded_sha = (
                load_inference_job_config(config)
            )
            if (
                loaded.logical_id != logical_id
                or loaded_config_path != config
                or loaded_sha != job.config_sha256
            ):
                raise ValueError("loaded inference config differs from continuation binding")
            result = execute_filesystem_inference(plan_path, array_index=array_index)
        elif job.kind == "semantic_allnode_gpu":
            if array_index is not None:
                raise ValueError("semantic continuation does not accept an array index")
            from weighttraits.behavior.allnode.semantic_runner import execute_filesystem_semantic

            result = execute_filesystem_semantic(config)
        else:
            raise AssertionError("unsupported bound workload kind")
    finally:
        attest_original_runtime("after-runner")
        attest_continuation(
            code_manifest_path=CODE_MANIFEST,
            code_manifest_sha256=code_manifest_sha256,
            acceptance_path=ACCEPTANCE_RECEIPT,
            acceptance_sha256=acceptance_sha256,
        )
    if result is None or not hasattr(result, "status_dict"):
        raise AssertionError("bound workload returned no status contract")
    status = result.status_dict()
    if not isinstance(status, Mapping) or status.get("valid") is not True:
        raise ValueError("bound workload returned an invalid status contract")
    return status


def _safe_export(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Slurm export values must be nonempty trimmed text")
    if any(character in value for character in (",", "\n", "\r", "\x00")):
        raise ValueError("Slurm export value contains an unsafe character")
    return value


def sbatch_argv(
    job: Job,
    *,
    dependency_job_id: str | None,
    code_manifest_sha256: str,
    acceptance_sha256: str,
    scheduler_evidence_sha256: str | None = None,
) -> tuple[str, ...]:
    if job not in build_plan():
        raise ValueError("job is outside the exact continuation plan")
    if (job.dependency is None) != (dependency_job_id is None):
        raise ValueError("scheduler dependency cardinality drift")
    if dependency_job_id is not None and JOB_ID_RE.fullmatch(dependency_job_id) is None:
        raise ValueError("scheduler dependency job ID must be decimal")
    manifest_sha = require_sha(code_manifest_sha256, "code manifest SHA-256")
    accepted_sha = require_sha(acceptance_sha256, "acceptance SHA-256")
    scheduler_sha = (
        None
        if scheduler_evidence_sha256 is None
        else require_sha(scheduler_evidence_sha256, "scheduler evidence SHA-256")
    )
    exported: dict[str, str] = {
        "WT_V1C5_CODE_MANIFEST": str(CODE_MANIFEST),
        "WT_V1C5_CODE_MANIFEST_SHA256": manifest_sha,
    }
    if job.kind == "artifact_adoption_cpu":
        if scheduler_sha is None:
            raise ValueError("adoption requires pinned scheduler evidence")
        exported.update(
            {
                "WT_V1C5_EXPECTED_ACCEPTANCE": str(EXPECTED_ACCEPTANCE),
                "WT_V1C5_EXPECTED_ACCEPTANCE_SHA256": accepted_sha,
                "WT_V1C5_ACCEPTANCE_RECEIPT": str(ACCEPTANCE_RECEIPT),
                "WT_V1C5_SCHEDULER_EVIDENCE": str(SCHEDULER_EVIDENCE),
                "WT_V1C5_SCHEDULER_EVIDENCE_SHA256": scheduler_sha,
            }
        )
    else:
        assert job.runner_logical_id is not None
        assert job.config_path is not None and job.config_sha256 is not None
        exported.update(
            {
                "WT_ALLNODE_CONTINUATION_LOGICAL_ID": job.logical_id,
                "WT_ALLNODE_COHORT_KEY": job.cohort,
                "WT_ALLNODE_JOB_CONFIG": job.config_path,
                "WT_ALLNODE_JOB_CONFIG_SHA256": job.config_sha256,
                "WT_ALLNODE_KIND": job.kind,
                "WT_ALLNODE_LOGICAL_ID": job.runner_logical_id,
                "WT_ALLNODE_OUTPUT_DIR": job.output_path,
                "WT_ALLNODE_STAGE_ROOT": str(STAGE),
                "WT_ALLNODE_DEPLOYMENT_ROOT": str(OLD_DEPLOYMENT),
                "WT_ALLNODE_SOURCE_RESOLUTION": str(OLD_PINS["source_resolution"][0]),
                "WT_ALLNODE_SOURCE_RESOLUTION_SHA256": OLD_PINS["source_resolution"][1],
                "WT_ALLNODE_SOURCE_SCOPE_SHA256": SOURCE_SCOPE_SHA256,
                "WT_ALLNODE_CODE_MANIFEST": str(OLD_PINS["code_manifest"][0]),
                "WT_ALLNODE_CODE_MANIFEST_SHA256": OLD_PINS["code_manifest"][1],
                "WT_DOWNSTREAM_CODE_MANIFEST": str(
                    OLD_PINS["downstream_code_manifest"][0]
                ),
                "WT_DOWNSTREAM_CODE_MANIFEST_SHA256": OLD_PINS[
                    "downstream_code_manifest"
                ][1],
                "WT_ALLNODE_PRODUCTION_INTERPRETER": PYTHON_BIN,
                "WT_ALLNODE_PYTHON_VERSION": "3.11.15",
                "WT_V1C5_ACCEPTANCE_RECEIPT": str(ACCEPTANCE_RECEIPT),
                "WT_V1C5_ACCEPTANCE_RECEIPT_SHA256": accepted_sha,
            }
        )
    export_arg = ",".join(
        f"{key}={_safe_export(value)}" for key, value in exported.items()
    )
    argv = [
        "sbatch",
        "--parsable",
        f"--job-name={job.scheduler_name}",
        "--account=statds",
        "--partition=all",
        "--qos=normal",
        "--nodelist=n03",
        "--exclude=n01",
        "--no-requeue",
        f"--cpus-per-task={job.cpus}",
        f"--mem={job.memory_gib}G",
        f"--time={job.time_limit}",
    ]
    if job.gpu:
        argv.append("--gres=gpu:nvidia-l40:1")
    if job.array is not None:
        argv.append(f"--array={job.array}")
    if dependency_job_id is not None:
        argv.append(f"--dependency=afterok:{dependency_job_id}")
    token = "%A_%a" if job.array is not None else "%j"
    argv.extend(
        (
            f"--output={LOG_DIR}/{job.logical_id}_{token}.out",
            f"--error={LOG_DIR}/{job.logical_id}_{token}.err",
            f"--export={export_arg}",
            job.wrapper,
        )
    )
    rendered = "\0".join(argv)
    for forbidden in (
        "OVERRIDE_MAX_STEPS",
        "MAX_TRAIN_SAMPLES",
        "MAX_EVAL_SAMPLES",
        "ROW_END",
    ):
        if forbidden in rendered:
            raise ValueError(f"forbidden production override in sbatch argv: {forbidden}")
    return tuple(argv)


def _parse_job_id(stdout: str) -> str:
    value = stdout.strip()
    if "\n" in value or "\r" in value:
        raise ValueError("sbatch --parsable returned multiple lines")
    job_id = value.split(";", 1)[0]
    if JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(f"sbatch returned an invalid job ID: {value!r}")
    return job_id


def _event_path(index: int, kind: str) -> Path:
    if kind not in {"attempt", "result", "failure"}:
        raise ValueError("invalid launch event kind")
    return LOG_DIR / "launch_events" / f"{index:02d}_{kind}.json"


def _terminal_jobs_receipt(
    *,
    valid: bool,
    audit_sha256: str,
    intent_raw: bytes,
    plan: Sequence[Job],
    submitted: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_submit_jobs.v1",
        "valid": valid,
        "terminal": True,
        "submission_id": SUBMISSION_ID,
        "dry_run_audit": {"path": str(DRY_RUN_AUDIT), "sha256": audit_sha256},
        "intent": {
            "path": str(INTENT),
            "sha256": hashlib.sha256(intent_raw).hexdigest(),
        },
        "n_planned": len(plan),
        "n_submitted": len(submitted),
        "partial": bool(submitted) and len(submitted) != len(plan),
        "failure": None if failure is None else dict(failure),
        "planned_jobs": [job.to_dict() for job in plan],
        "submitted_prefix": [dict(row) for row in submitted],
        "unsubmitted_suffix": [job.to_dict() for job in plan[len(submitted) :]],
        "execution_overrides": {},
        "no_automatic_recovery": True,
        "retry_allowed": False,
        "cancel_on_failure": False,
        "resubmission_allowed": False,
    }


def launch(
    *,
    dry_run_path: str | Path,
    dry_run_sha256: str,
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    """Submit the exact eight-job continuation once, with immutable journaling."""

    audit_sha = require_sha(dry_run_sha256, "dry-run audit SHA-256")
    audit, _ = load_dry_run(dry_run_path, audit_sha)
    manifest_pin = audit["code_manifest"]
    acceptance_pin = audit["expected_acceptance"]
    scheduler_pin = audit["scheduler_evidence_file"]
    verify_code_manifest(manifest_pin["path"], manifest_pin["sha256"])
    attest_original_runtime("launch")
    attest_v1c1_failure_files()
    check_absence(before_preflight=False)
    collisions = query_duplicate_names(runner)
    if collisions:
        raise ValueError(f"continuation scheduler names already exist: {collisions}")
    pinned_scheduler, _ = load_scheduler_evidence(scheduler_pin["sha256"])
    current_scheduler = capture_scheduler_evidence(runner)
    if scheduler_stable_view(current_scheduler) != scheduler_stable_view(
        pinned_scheduler
    ):
        raise ValueError("scheduler state/output bytes changed since v1c5 preflight")

    expected, expected_raw = read_canonical(
        EXPECTED_ACCEPTANCE, "expected artifact acceptance"
    )
    current_inventory = artifact_inventory_pass()
    current_direct = validate_direct_terminals()
    current_acceptance = build_acceptance(
        current_inventory, current_direct, pinned_scheduler
    )
    if current_acceptance != expected or canonical_bytes(current_acceptance) != expected_raw:
        raise ValueError("artifact acceptance changed since continuation preflight")
    if hashlib.sha256(expected_raw).hexdigest() != acceptance_pin["sha256"]:
        raise ValueError("expected acceptance hash differs from dry-run boundary")
    verify_code_manifest(manifest_pin["path"], manifest_pin["sha256"])
    attest_original_runtime("launch")

    plan = build_plan()
    intent_core = {
        "schema": f"{SCHEMA_PREFIX}_submit_intent.v1",
        "valid": True,
        "terminal": False,
        "submission_id": SUBMISSION_ID,
        "dry_run_audit": {"path": str(DRY_RUN_AUDIT), "sha256": audit_sha},
        "code_manifest": dict(manifest_pin),
        "expected_acceptance": dict(acceptance_pin),
        "scheduler_evidence": dict(scheduler_pin),
        "acceptance_receipt_path": str(ACCEPTANCE_RECEIPT),
        "n_planned": 8,
        "planned_jobs": [job.to_dict() for job in plan],
        "execution_contract_sha256": audit["execution_contract_sha256"],
        "execution_overrides": {},
        "no_automatic_recovery": True,
        "retry_allowed": False,
        "cancel_on_failure": False,
        "resubmission_allowed": False,
    }
    intent = {**intent_core, "intent_sha256": canonical_sha256(intent_core)}
    intent_raw = canonical_bytes(intent)
    publish_exclusive(INTENT, intent_raw)

    submitted: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    try:
        if os.path.lexists(LOG_DIR):
            raise FileExistsError("continuation log directory appeared after intent")
        LOG_DIR.mkdir(mode=0o700)
        (LOG_DIR / "launch_events").mkdir(mode=0o700)
        previous_job_id: str | None = None
        for index, job in enumerate(plan):
            argv = sbatch_argv(
                job,
                dependency_job_id=previous_job_id,
                code_manifest_sha256=manifest_pin["sha256"],
                acceptance_sha256=acceptance_pin["sha256"],
                scheduler_evidence_sha256=scheduler_pin["sha256"],
            )
            attempt = {
                "schema": f"{SCHEMA_PREFIX}_submit_attempt.v1",
                "valid": True,
                "submission_id": SUBMISSION_ID,
                "plan_index": index,
                "logical_id": job.logical_id,
                "argv": list(argv),
                "argv_sha256": hashlib.sha256("\0".join(argv).encode("utf-8")).hexdigest(),
                "dependency_job_ids": []
                if previous_job_id is None
                else [previous_job_id],
                "scheduler_invocations": 1,
                "retry_allowed": False,
            }
            publish_exclusive(_event_path(index, "attempt"), canonical_bytes(attempt))
            result = runner(argv)
            if result.returncode != 0:
                failure = {
                    "plan_index": index,
                    "logical_id": job.logical_id,
                    "error_type": "SbatchNonzero",
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "scheduler_state_may_be_unknown": False,
                }
                publish_exclusive(
                    _event_path(index, "failure"), canonical_bytes(failure)
                )
                break
            job_id = _parse_job_id(result.stdout)
            event = {
                "schema": f"{SCHEMA_PREFIX}_submit_result.v1",
                "valid": True,
                "terminal": True,
                "plan_index": index,
                "logical_id": job.logical_id,
                "job_id": job_id,
                "dependency_job_ids": []
                if previous_job_id is None
                else [previous_job_id],
                "dependency_expression": None
                if previous_job_id is None
                else f"afterok:{previous_job_id}",
                "returncode": 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "retry_allowed": False,
            }
            publish_exclusive(_event_path(index, "result"), canonical_bytes(event))
            submitted.append({**job.to_dict(), **event})
            previous_job_id = job_id
    except BaseException as exc:
        failure = {
            "plan_index": len(submitted),
            "logical_id": plan[len(submitted)].logical_id
            if len(submitted) < len(plan)
            else "terminal_publication",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "scheduler_state_may_be_unknown": True,
        }
        receipt = _terminal_jobs_receipt(
            valid=False,
            audit_sha256=audit_sha,
            intent_raw=intent_raw,
            plan=plan,
            submitted=submitted,
            failure=failure,
        )
        try:
            publish_exclusive(JOBS_RECEIPT, canonical_bytes(receipt))
        finally:
            raise

    valid = failure is None and len(submitted) == len(plan)
    receipt = _terminal_jobs_receipt(
        valid=valid,
        audit_sha256=audit_sha,
        intent_raw=intent_raw,
        plan=plan,
        submitted=submitted,
        failure=failure,
    )
    publish_exclusive(JOBS_RECEIPT, canonical_bytes(receipt))
    if not valid:
        raise RuntimeError(
            "continuation submission stopped after "
            f"{len(submitted)} of {len(plan)} jobs; "
            "a terminal invalid jobs receipt was written and no retry was attempted"
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--source-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    attest = subparsers.add_parser("attest")
    attest.add_argument("--code-manifest", type=Path, required=True)
    attest.add_argument("--code-manifest-sha256", required=True)
    attest.add_argument("--acceptance", type=Path)
    attest.add_argument("--acceptance-sha256")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--code-manifest-sha256", required=True)

    tester = subparsers.add_parser("test")
    tester.add_argument("--code-manifest-sha256", required=True)

    adopter = subparsers.add_parser("adopt")
    adopter.add_argument("--code-manifest", type=Path, required=True)
    adopter.add_argument("--code-manifest-sha256", required=True)
    adopter.add_argument("--expected-acceptance", type=Path, required=True)
    adopter.add_argument("--expected-acceptance-sha256", required=True)
    adopter.add_argument("--scheduler-evidence", type=Path, required=True)
    adopter.add_argument("--scheduler-evidence-sha256", required=True)
    adopter.add_argument("--acceptance-receipt", type=Path, required=True)

    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--dry-run", type=Path, required=True)
    launcher.add_argument("--dry-run-sha256", required=True)

    bound = subparsers.add_parser("run")
    bound.add_argument("--code-manifest", type=Path, required=True)
    bound.add_argument("--code-manifest-sha256", required=True)
    bound.add_argument("--acceptance", type=Path, required=True)
    bound.add_argument("--acceptance-sha256", required=True)
    bound.add_argument("--config", type=Path, required=True)
    bound.add_argument("--array-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        raw = build_code_manifest(args.source_root, args.output)
        value: Mapping[str, Any] = {
            "valid": True,
            "path": str(args.output),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    elif args.command == "attest":
        value = attest_continuation(
            code_manifest_path=args.code_manifest,
            code_manifest_sha256=args.code_manifest_sha256,
            acceptance_path=args.acceptance,
            acceptance_sha256=args.acceptance_sha256,
        )
    elif args.command == "preflight":
        value = make_preflight(code_manifest_sha256=args.code_manifest_sha256)
    elif args.command == "test":
        value = run_test_attestation(code_manifest_sha256=args.code_manifest_sha256)
    elif args.command == "adopt":
        if args.code_manifest != CODE_MANIFEST:
            raise ValueError("adoption code manifest path drift")
        value = adopt(
            code_manifest_sha256=args.code_manifest_sha256,
            expected_acceptance_path=args.expected_acceptance,
            expected_acceptance_sha256=args.expected_acceptance_sha256,
            scheduler_evidence_path=args.scheduler_evidence,
            scheduler_evidence_sha256=args.scheduler_evidence_sha256,
            acceptance_receipt_path=args.acceptance_receipt,
        )
    elif args.command == "launch":
        value = launch(
            dry_run_path=args.dry_run,
            dry_run_sha256=args.dry_run_sha256,
        )
    elif args.command == "run":
        if args.code_manifest != CODE_MANIFEST or args.acceptance != ACCEPTANCE_RECEIPT:
            raise ValueError("bound workload continuation path drift")
        value = run_bound_workload(
            code_manifest_sha256=args.code_manifest_sha256,
            acceptance_sha256=args.acceptance_sha256,
            config_path=args.config,
            array_index=args.array_index,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


__all__ = [
    "ACCEPTANCE_RECEIPT",
    "CODE_MANIFEST",
    "DRY_RUN_AUDIT",
    "EXPECTED_ACCEPTANCE",
    "JOBS_RECEIPT",
    "Job",
    "build_code_manifest",
    "build_plan",
    "canonical_bytes",
    "canonical_sha256",
    "launch",
    "main",
    "make_preflight",
    "run_bound_workload",
    "sbatch_argv",
    "validate_acceptance_payload",
]
