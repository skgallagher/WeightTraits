"""One-shot Slurm submission gate for isolated downstream analyses.

The gate deliberately owns only three job kinds.  It cannot submit training,
recovery, cancellation, requeue, or scheduler-mutation commands.  Every job is
bound to a hashed config, wrapper, explicit input set, and a fully replayed code
manifest before the first ``sbatch`` call is made.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping


SUBMISSION_SCHEMA = "weighttraits.downstream_submission.v1"
INTENT_SCHEMA = "weighttraits.downstream_submission_intent.v1"
JOBS_SCHEMA = "weighttraits.downstream_submission_jobs.v1"

DIRECT_KIND = "direct_runset_cpu"
BEHAVIOR_KIND = "behavior_smoke"
PHYLOLM_KIND = "phylolm_smoke"
JOB_KINDS = frozenset({DIRECT_KIND, BEHAVIOR_KIND, PHYLOLM_KIND})
GPU_KINDS = frozenset({BEHAVIOR_KIND, PHYLOLM_KIND})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_WRAPPER_NAMES = {
    DIRECT_KIND: "downstream_direct_runset_cpu.sbatch",
    BEHAVIOR_KIND: "downstream_behavior_smoke.sbatch",
    PHYLOLM_KIND: "downstream_phylolm_smoke.sbatch",
}
_JOB_CONFIG_KEYS = {
    DIRECT_KIND: {
        "schema",
        "kind",
        "stage_root",
        "output_dir",
        "cohort_id",
        "training_summary",
        "completion_receipt",
        "stage_manifest",
        "path_base",
        "artifact_kind",
        "metrics",
        "chunk_size",
        "eps",
        "aggregate",
        "expected_n_trees",
        "expected_n_rows",
    },
    BEHAVIOR_KIND: {
        "schema",
        "kind",
        "stage_root",
        "output_dir",
        "cohort_id",
        "tree_id",
        "model_task",
        "path_base",
        "registry",
        "training_summary",
        "completion_receipt",
        "prompt_artifacts",
        "expected_leaf_ids",
        "leaf_id",
        "device",
        "torch_dtype",
    },
    PHYLOLM_KIND: {
        "schema",
        "kind",
        "stage_root",
        "output_dir",
        "cohort_id",
        "tree_id",
        "training_summary",
        "completion_receipt",
        "stage_manifest",
        "genome_receipt",
        "truth_manifest",
        "base_model_id",
        "base_model_revision",
        "path_base",
        "batch_size",
        "torch_dtype",
    },
}


@dataclass(frozen=True)
class Pin:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class CodeManifest:
    root: Path
    pin: Pin

    def to_dict(self) -> dict[str, str]:
        return {"root": str(self.root), **self.pin.to_dict()}


@dataclass(frozen=True)
class JobPlan:
    kind: str
    name: str
    config: Pin
    wrapper: Pin
    output_dir: Path
    input_pins: tuple[Pin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "config": self.config.to_dict(),
            "wrapper": self.wrapper.to_dict(),
            "output_dir": str(self.output_dir),
            "input_pins": [pin.to_dict() for pin in self.input_pins],
        }


@dataclass(frozen=True)
class SubmissionPlan:
    config: Pin
    submission_id: str
    stage_root: Path
    frozen_stage: Path
    code_manifest: CodeManifest
    intent_path: Path
    jobs_receipt_path: Path
    log_dir: Path
    jobs: tuple[JobPlan, ...]


Runner = Callable[..., subprocess.CompletedProcess[str]]


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_submission_plan(
    config_path: str | Path,
    *,
    expected_sha256: str,
) -> SubmissionPlan:
    """Parse and fully validate a submission config without scheduler mutation."""

    config = Path(config_path).resolve(strict=True)
    expected = _digest(expected_sha256, "submission config sha256")
    observed = sha256_file(config)
    if observed != expected:
        raise ValueError(
            f"submission config hash mismatch: expected {expected}, got {observed}"
        )
    raw = _json_object(config)
    _exact_keys(
        raw,
        {
            "schema",
            "submission_id",
            "stage_root",
            "frozen_stage",
            "code_manifest",
            "intent_path",
            "jobs_receipt_path",
            "log_dir",
            "jobs",
        },
        "submission config",
    )
    if raw["schema"] != SUBMISSION_SCHEMA:
        raise ValueError(f"submission schema must be {SUBMISSION_SCHEMA!r}")
    submission_id = _identifier(raw["submission_id"], "submission_id")
    stage_root = _existing_directory(raw["stage_root"], "stage_root")
    frozen_stage = _existing_directory(raw["frozen_stage"], "frozen_stage")
    if stage_root == frozen_stage or stage_root in frozen_stage.parents or frozen_stage in stage_root.parents:
        raise ValueError("downstream and frozen stages must be disjoint directories")
    if _is_within(config, frozen_stage):
        raise ValueError("submission config must not live in the frozen training stage")

    code_manifest = _code_manifest(raw["code_manifest"])
    if not _is_within(code_manifest.root, stage_root):
        raise ValueError("code root must be inside the downstream stage")
    if not _is_within(code_manifest.pin.path, stage_root):
        raise ValueError("code manifest must be inside the downstream stage")
    code_rows = verify_code_manifest(code_manifest)

    intent_path = _new_path(raw["intent_path"], "intent_path")
    jobs_receipt_path = _new_path(raw["jobs_receipt_path"], "jobs_receipt_path")
    log_dir = Path(_text(raw["log_dir"], "log_dir")).resolve()
    for label, path in (
        ("intent_path", intent_path),
        ("jobs_receipt_path", jobs_receipt_path),
        ("log_dir", log_dir),
    ):
        if not _is_within(path, stage_root):
            raise ValueError(f"{label} must be inside the downstream stage")
        if _is_within(path, frozen_stage):
            raise ValueError(f"{label} must not be inside the frozen training stage")
    if intent_path == jobs_receipt_path:
        raise ValueError("intent and jobs receipt paths must differ")

    raw_jobs = raw["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("submission config requires a non-empty jobs list")
    jobs = tuple(
        _job_plan(value, index=index, stage_root=stage_root, frozen_stage=frozen_stage, code_root=code_manifest.root)
        for index, value in enumerate(raw_jobs)
    )
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("submission plan contains duplicate job names")
    outputs = [job.output_dir for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("submission plan contains duplicate output directories")
    for index, left in enumerate(outputs):
        for right in outputs[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError("job output directories must not contain one another")
    for output in outputs:
        for reserved in (intent_path, jobs_receipt_path, log_dir):
            if output == reserved or output in reserved.parents or reserved in output.parents:
                raise ValueError(
                    f"job output directory overlaps submission receipt/log state: {output} vs {reserved}"
                )
    manifest_paths = {row["path"] for row in code_rows}
    required_code_paths = {
        "scripts/run_downstream_job.py",
        "scripts/submit_downstream_jobs.py",
        "src/weighttraits/downstream/jobs.py",
        "src/weighttraits/downstream/submission.py",
        *(str(job.wrapper.path.relative_to(code_manifest.root)) for job in jobs),
    }
    if not required_code_paths.issubset(manifest_paths):
        raise ValueError(
            "code manifest is missing downstream gate/runner/wrapper files: "
            f"{sorted(required_code_paths - manifest_paths)}"
        )

    return SubmissionPlan(
        config=Pin(config, observed),
        submission_id=submission_id,
        stage_root=stage_root,
        frozen_stage=frozen_stage,
        code_manifest=code_manifest,
        intent_path=intent_path,
        jobs_receipt_path=jobs_receipt_path,
        log_dir=log_dir,
        jobs=jobs,
    )


def verify_code_manifest(manifest: CodeManifest) -> list[dict[str, str]]:
    """Verify the manifest file pin and every relative file entry."""

    _verify_pin(manifest.pin, "code manifest")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest.pin.path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})[ \t]+[* ]?(.+)", line)
        if match is None:
            raise ValueError(
                f"invalid code-manifest line {line_number}: {raw_line!r}"
            )
        digest, relative_text = match.groups()
        relative = Path(relative_text)
        normalized_relative = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or normalized_relative in seen:
            raise ValueError(f"unsafe or duplicate code-manifest path: {relative_text!r}")
        seen.add(normalized_relative)
        target = (manifest.root / relative).resolve(strict=True)
        if not _is_within(target, manifest.root) or not target.is_file():
            raise ValueError(f"code-manifest target is not a regular in-root file: {target}")
        observed = sha256_file(target)
        if observed != digest:
            raise ValueError(
                f"code-manifest hash mismatch for {relative_text}: expected {digest}, got {observed}"
            )
        rows.append({"path": normalized_relative, "sha256": digest})
    if not rows:
        raise ValueError("code manifest is empty")
    return rows


def submit_downstream_plan(
    config_path: str | Path,
    *,
    expected_sha256: str,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Validate, lock, and submit the exact job plan once.

    A failed or partial submission is also terminal: the immutable intent and a
    ``valid=false`` jobs receipt remain as duplicate locks.  No cleanup,
    cancellation, recovery, or resubmission is attempted.
    """

    plan = load_submission_plan(config_path, expected_sha256=expected_sha256)
    for job in plan.jobs:
        _assert_no_duplicate(job.name, runner=runner)

    code_rows = verify_code_manifest(plan.code_manifest)
    intent = {
        "schema": INTENT_SCHEMA,
        "valid": True,
        "submission_id": plan.submission_id,
        "submission_config": plan.config.to_dict(),
        "stage_root": str(plan.stage_root),
        "frozen_stage": str(plan.frozen_stage),
        "scheduler_policy": {
            "account": "statds",
            "partition": "all",
            "qos": "normal",
            "required_node": "n03",
            "excluded_node": "n01",
            "requeue": False,
            "gpu_resource": "gpu:nvidia-l40:1",
            "gpu_array_concurrency": 1,
            "gpu_jobs_serialized": True,
        },
        "code_manifest": plan.code_manifest.to_dict(),
        "code_manifest_entries": code_rows,
        "jobs": [job.to_dict() for job in plan.jobs],
    }
    _write_immutable_json(plan.intent_path, intent)
    intent_pin = Pin(plan.intent_path, sha256_file(plan.intent_path))
    plan.log_dir.mkdir(parents=True, exist_ok=True)

    submitted: list[dict[str, Any]] = []
    failure: dict[str, str] | None = None
    previous_gpu_job_id: str | None = None
    for job in plan.jobs:
        try:
            command = _sbatch_command(
                plan,
                job,
                previous_gpu_job_id=previous_gpu_job_id,
            )
            result = runner(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = _job_id(result.stdout)
            submitted.append(
                {
                    "kind": job.kind,
                    "name": job.name,
                    "job_id": job_id,
                    "dependency": (
                        None
                        if job.kind not in GPU_KINDS or previous_gpu_job_id is None
                        else f"afterok:{previous_gpu_job_id}"
                    ),
                    "output_dir": str(job.output_dir),
                    "config": job.config.to_dict(),
                    "wrapper": job.wrapper.to_dict(),
                    "submit_argv": command,
                }
            )
            if job.kind in GPU_KINDS:
                previous_gpu_job_id = job_id
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            failure = {
                "job": job.name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            break

    receipt = {
        "schema": JOBS_SCHEMA,
        "valid": failure is None and len(submitted) == len(plan.jobs),
        "submission_id": plan.submission_id,
        "intent": intent_pin.to_dict(),
        "submission_config": plan.config.to_dict(),
        "n_planned": len(plan.jobs),
        "n_submitted": len(submitted),
        "partial": len(submitted) != len(plan.jobs),
        "failure": failure,
        "jobs": submitted,
        "no_automatic_recovery": True,
    }
    _write_immutable_json(plan.jobs_receipt_path, receipt)
    if failure is not None:
        raise RuntimeError(
            "downstream submission stopped fail-closed after a partial/failed sbatch; "
            f"inspect immutable receipt {plan.jobs_receipt_path}: {failure}"
        )
    return receipt


def _job_plan(
    raw: object,
    *,
    index: int,
    stage_root: Path,
    frozen_stage: Path,
    code_root: Path,
) -> JobPlan:
    if not isinstance(raw, Mapping):
        raise ValueError(f"jobs[{index}] must be an object")
    _exact_keys(
        raw,
        {"kind", "name", "config", "wrapper", "output_dir", "input_pins"},
        f"jobs[{index}]",
    )
    kind = _text(raw["kind"], f"jobs[{index}].kind")
    if kind not in JOB_KINDS:
        raise ValueError(f"jobs[{index}] unsupported kind: {kind!r}")
    name = _identifier(raw["name"], f"jobs[{index}].name")
    config = _pin(raw["config"], f"jobs[{index}].config")
    wrapper = _pin(raw["wrapper"], f"jobs[{index}].wrapper")
    expected_wrapper = _WRAPPER_NAMES[kind]
    if wrapper.path.name != expected_wrapper or not _is_within(wrapper.path, code_root):
        raise ValueError(
            f"jobs[{index}] {kind} wrapper must be in the code root and named {expected_wrapper}"
        )
    if not _is_within(config.path, stage_root) or _is_within(config.path, frozen_stage):
        raise ValueError(f"jobs[{index}] config must be inside only the downstream stage")
    output_dir = _new_path(raw["output_dir"], f"jobs[{index}].output_dir")
    if not _is_within(output_dir, stage_root) or _is_within(output_dir, frozen_stage):
        raise ValueError(f"jobs[{index}] output must be inside only the downstream stage")
    raw_pins = raw["input_pins"]
    if not isinstance(raw_pins, list) or not raw_pins:
        raise ValueError(f"jobs[{index}].input_pins must be non-empty")
    input_pins = tuple(
        _pin(value, f"jobs[{index}].input_pins[{pin_index}]")
        for pin_index, value in enumerate(raw_pins)
    )
    all_pins = (config, wrapper, *input_pins)
    pin_paths = [pin.path for pin in all_pins]
    if len(pin_paths) != len(set(pin_paths)):
        raise ValueError(f"jobs[{index}] contains duplicate pinned files")
    for pin in all_pins:
        _verify_pin(pin, f"jobs[{index}] pin")
    declared_pins = _validate_job_config(
        config.path,
        kind=kind,
        output_dir=output_dir,
        stage_root=stage_root,
        frozen_stage=frozen_stage,
    )
    input_set = {(str(pin.path), pin.sha256) for pin in input_pins}
    declared_set = {(str(pin.path), pin.sha256) for pin in declared_pins}
    if input_set != declared_set:
        raise ValueError(
            f"jobs[{index}] input_pins must exactly equal all path+sha256 pins in its config: "
            f"missing={sorted(declared_set - input_set)}, unexpected={sorted(input_set - declared_set)}"
        )
    return JobPlan(kind, name, config, wrapper, output_dir, input_pins)


def _validate_job_config(
    path: Path,
    *,
    kind: str,
    output_dir: Path,
    stage_root: Path,
    frozen_stage: Path,
) -> tuple[Pin, ...]:
    payload = _json_object(path)
    _exact_keys(payload, _JOB_CONFIG_KEYS[kind], f"{kind} job config")
    expected_schema = f"weighttraits.downstream_job.{kind}.v1"
    if payload.get("schema") != expected_schema:
        raise ValueError(f"{kind} job config schema must be {expected_schema!r}")
    if payload.get("kind") != kind:
        raise ValueError(f"{kind} job config kind mismatch")
    declared_output = Path(_text(payload.get("output_dir"), "job output_dir")).resolve()
    if declared_output != output_dir:
        raise ValueError(f"{kind} job config output_dir differs from submission plan")
    if Path(_text(payload.get("stage_root"), "job stage_root")).resolve() != stage_root:
        raise ValueError(f"{kind} job config stage_root differs from submission plan")
    if Path(_text(payload.get("path_base"), "job path_base")).resolve() != frozen_stage:
        raise ValueError(f"{kind} job config path_base must be the exact frozen stage")
    _validate_fixed_job_fields(payload, kind=kind)
    pins = tuple(_nested_pins(payload))
    if not pins:
        raise ValueError(f"{kind} job config requires at least one path+sha256 input pin")
    paths = [pin.path for pin in pins]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{kind} job config contains duplicate pinned input paths")
    return pins


def _validate_fixed_job_fields(payload: Mapping[str, Any], *, kind: str) -> None:
    if kind == DIRECT_KIND:
        if payload.get("artifact_kind") not in {"model", "merged", "adapter_chain"}:
            raise ValueError("direct artifact_kind must be model, merged, or adapter_chain")
        if payload.get("metrics") != ["l2", "cosine", "correlation"]:
            raise ValueError("direct metrics must be exactly l2, cosine, correlation")
        for field, expected in (
            ("chunk_size", 1_000_000),
            ("eps", 1e-3),
            ("aggregate", "mean"),
            ("expected_n_trees", 50),
            ("expected_n_rows", 150),
        ):
            if payload.get(field) != expected:
                raise ValueError(f"direct {field} must be exactly {expected!r}")
    elif kind == BEHAVIOR_KIND:
        if payload.get("tree_id") != "confirm_paper_tree_001":
            raise ValueError("behavior smoke tree_id must be confirm_paper_tree_001")
        if payload.get("expected_leaf_ids") != [
            "n3", "n7", "n8", "n9", "n10", "n11", "n12", "n13"
        ]:
            raise ValueError("behavior smoke expected_leaf_ids differ from exact Tree001 truth")
        if payload.get("leaf_id") != "n3":
            raise ValueError("behavior smoke must run only leaf_id n3")
        if payload.get("model_task") not in {"causal_lm", "seq2seq"}:
            raise ValueError("behavior smoke model_task must be causal_lm or seq2seq")
        if payload.get("device") != "cuda" or payload.get("torch_dtype") != "auto":
            raise ValueError("behavior smoke requires cuda/auto")
        if not isinstance(payload.get("prompt_artifacts"), Mapping) or not payload["prompt_artifacts"]:
            raise ValueError("behavior smoke requires non-empty prompt_artifacts")
    else:
        if payload.get("tree_id") != "confirm_paper_tree_001":
            raise ValueError("PhyloLM smoke tree_id must be confirm_paper_tree_001")
        if payload.get("batch_size") != 64 or payload.get("torch_dtype") != "auto":
            raise ValueError("PhyloLM smoke requires batch_size=64 and torch_dtype=auto")
        revision = payload.get("base_model_revision")
        if (
            not isinstance(revision, str)
            or len(revision) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError("PhyloLM smoke requires a pinned base-model revision")


def _nested_pins(value: object):
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256"}:
            yield _pin(value, "job config input")
            return
        for child in value.values():
            yield from _nested_pins(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_pins(child)


def _sbatch_command(
    plan: SubmissionPlan,
    job: JobPlan,
    *,
    previous_gpu_job_id: str | None,
) -> list[str]:
    export_values = {
        "WT_CODE_ROOT": str(plan.code_manifest.root),
        "WT_CODE_MANIFEST": str(plan.code_manifest.pin.path),
        "WT_CODE_MANIFEST_SHA256": plan.code_manifest.pin.sha256,
        "WT_JOB_CONFIG": str(job.config.path),
        "WT_JOB_CONFIG_SHA256": job.config.sha256,
    }
    for key, value in export_values.items():
        if any(character in value for character in (",", "\n", "\r")):
            raise ValueError(f"unsafe sbatch export value for {key}")
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job.name}",
        "--account=statds",
        "--partition=all",
        "--qos=normal",
        "--nodelist=n03",
        "--exclude=n01",
        "--no-requeue",
        f"--output={plan.log_dir / (job.name + '_%A_%a.out')}",
        f"--error={plan.log_dir / (job.name + '_%A_%a.err')}",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in export_values.items()),
    ]
    if job.kind == DIRECT_KIND:
        command.extend(["--cpus-per-task=8", "--mem=96G", "--time=24:00:00"])
    elif job.kind == BEHAVIOR_KIND:
        command.extend(
            [
                "--cpus-per-task=8",
                "--mem=64G",
                "--time=12:00:00",
                "--gres=gpu:nvidia-l40:1",
                "--array=0-0%1",
            ]
        )
    else:
        command.extend(
            [
                "--cpus-per-task=4",
                "--mem=48G",
                "--time=12:00:00",
                "--gres=gpu:nvidia-l40:1",
                "--array=0-0%1",
            ]
        )
    if job.kind in GPU_KINDS and previous_gpu_job_id is not None:
        command.append(f"--dependency=afterok:{previous_gpu_job_id}")
    command.append(str(job.wrapper.path))
    return command


def _assert_no_duplicate(name: str, *, runner: Runner) -> None:
    commands = (
        ["squeue", "--noheader", f"--name={name}", "--format=%i|%j|%T"],
        [
            "sacct",
            "-X",
            "--noheader",
            "--starttime=1970-01-01",
            f"--name={name}",
            "--format=JobIDRaw,JobName,State,ExitCode",
        ],
    )
    for command in commands:
        result = runner(command, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            raise FileExistsError(
                f"refusing duplicate Slurm job name {name!r}; evidence: {result.stdout.strip()}"
            )


def _job_id(stdout: str) -> str:
    value = stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"sbatch did not return a numeric job id: {stdout!r}")
    return value


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _pin(raw: object, label: str) -> Pin:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a path+sha256 object")
    _exact_keys(raw, {"path", "sha256"}, label)
    path = Path(_text(raw["path"], f"{label}.path")).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must pin a regular file: {path}")
    return Pin(path, _digest(raw["sha256"], f"{label}.sha256"))


def _code_manifest(raw: object) -> CodeManifest:
    if not isinstance(raw, Mapping):
        raise ValueError("code_manifest must be an object")
    _exact_keys(raw, {"root", "path", "sha256"}, "code_manifest")
    root = _existing_directory(raw["root"], "code_manifest.root")
    pin = _pin(
        {"path": raw["path"], "sha256": raw["sha256"]},
        "code_manifest",
    )
    return CodeManifest(root, pin)


def _verify_pin(pin: Pin, label: str) -> None:
    observed = sha256_file(pin.path)
    if observed != pin.sha256:
        raise ValueError(
            f"{label} hash mismatch for {pin.path}: expected {pin.sha256}, got {observed}"
        )


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    keys = set(raw)
    if keys != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - keys)}, "
            f"unexpected={sorted(keys - expected)}"
        )


def _identifier(raw: object, label: str) -> str:
    value = _text(raw, label)
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_IDENTIFIER_RE.pattern!r}")
    return value


def _text(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return raw


def _digest(raw: object, label: str) -> str:
    value = _text(raw, label)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _existing_directory(raw: object, label: str) -> Path:
    path = Path(_text(raw, label)).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _new_path(raw: object, label: str) -> Path:
    path = Path(_text(raw, label)).resolve()
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


__all__ = [
    "BEHAVIOR_KIND",
    "CodeManifest",
    "DIRECT_KIND",
    "GPU_KINDS",
    "INTENT_SCHEMA",
    "JOBS_SCHEMA",
    "PHYLOLM_KIND",
    "Pin",
    "SUBMISSION_SCHEMA",
    "load_submission_plan",
    "sha256_file",
    "submit_downstream_plan",
    "verify_code_manifest",
]
