"""Training run-list and scheduler dry-run helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


PathExists = Callable[[str], bool]


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    scheduler: str = "local"
    account: str | None = None
    partition: str | None = None
    default_max_concurrent: int | None = None
    require_run_list: bool = True
    require_dry_run_first: bool = True
    allow_broad_arrays: bool = False
    paths: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["notes"] = list(self.notes)
        return out


@dataclass(frozen=True)
class RunPreflightIssue:
    severity: str
    issue: str
    node_id: str | None
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunSpec:
    array_index: int
    run_id: str
    node_id: str
    parent_id: str
    depth: int
    method: str
    dataset_id: str | None
    task_family: str | None
    init_from: str
    output_dir: str
    expected_artifacts: dict[str, str]
    ledger_path: str
    runner: dict[str, Any]
    job: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunList:
    runs: tuple[TrainingRunSpec, ...]
    profile: ExecutionProfile | None = None
    issues: tuple[RunPreflightIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "n_runs": len(self.runs),
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "profile": None if self.profile is None else self.profile.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_execution_profile(path: str | Path) -> ExecutionProfile:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"execution profile must be a mapping: {path}")
    return ExecutionProfile(
        name=str(raw.get("cluster") or raw.get("profile") or Path(path).stem),
        scheduler=str(raw.get("scheduler", "local")),
        account=None if raw.get("account") is None else str(raw.get("account")),
        partition=None if raw.get("partition") is None else str(raw.get("partition")),
        default_max_concurrent=_optional_int(raw.get("default_max_concurrent")),
        require_run_list=bool(raw.get("require_run_list", True)),
        require_dry_run_first=bool(raw.get("require_dry_run_first", True)),
        allow_broad_arrays=bool(raw.get("allow_broad_arrays", False)),
        paths={str(key): str(value) for key, value in dict(raw.get("paths", {})).items()},
        notes=tuple(str(note) for note in raw.get("notes", [])),
    )


def build_training_run_list(
    jobs: Sequence[Any],
    *,
    profile: ExecutionProfile | None = None,
    run_list_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    runner_entrypoint: str = "pending_hf_peft_executor",
    allow_existing_artifacts: bool = False,
    check_filesystem: bool = True,
    path_exists: PathExists | None = None,
) -> TrainingRunList:
    """Build run specs and preflight issues from planned training jobs."""
    exists = path_exists or (lambda value: Path(value).exists())
    runs: list[TrainingRunSpec] = []
    issues: list[RunPreflightIssue] = []
    all_node_ids = {str(getattr(job, "node_id", "")) for job in jobs}
    seen_node_ids: set[str] = set()
    ledger = str(ledger_path or _default_ledger_path(jobs))
    run_list = None if run_list_path is None else str(run_list_path)

    for array_index, job in enumerate(jobs):
        run = _run_spec_from_job(
            job,
            array_index=array_index,
            ledger_path=ledger,
            run_list_path=run_list,
            runner_entrypoint=runner_entrypoint,
        )
        runs.append(run)
        issues.extend(
            _preflight_job(
                job,
                all_node_ids=all_node_ids,
                seen_node_ids=seen_node_ids,
                allow_existing_artifacts=allow_existing_artifacts,
                check_filesystem=check_filesystem,
                exists=exists,
            )
        )
        seen_node_ids.add(run.node_id)

    issues.extend(_preflight_profile(profile, len(runs)))
    return TrainingRunList(runs=tuple(runs), profile=profile, issues=tuple(issues))


def write_training_run_list(run_list: TrainingRunList, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for run in run_list.runs:
            handle.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")


def load_training_run_specs(path: str | Path) -> list[TrainingRunSpec]:
    runs = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                runs.append(_run_spec_from_dict(json.loads(line)))
    return runs


def write_training_run_report(run_list: TrainingRunList, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run_list.to_dict(), indent=2, sort_keys=True) + "\n")


def select_training_run(
    runs: Sequence[TrainingRunSpec],
    *,
    index: int | None = None,
    node_id: str | None = None,
) -> TrainingRunSpec:
    if index is None and node_id is None:
        raise ValueError("select by index or node_id")
    if index is not None:
        if index < 0 or index >= len(runs):
            raise IndexError(f"run index out of range: {index}")
        return runs[index]
    matches = [run for run in runs if run.node_id == node_id]
    if not matches:
        raise ValueError(f"node id not found in run list: {node_id}")
    if len(matches) > 1:
        raise ValueError(f"multiple runs found for node id: {node_id}")
    return matches[0]


def render_slurm_array_script(
    run_list: TrainingRunList,
    *,
    run_list_path: str | Path,
    profile: ExecutionProfile,
    job_name: str = "weighttraits-train",
    max_concurrent: int | None = None,
    python: str = "python",
) -> str:
    if profile.scheduler != "slurm":
        raise ValueError(f"SLURM script requires scheduler=slurm, got {profile.scheduler}")
    if not run_list.runs:
        raise ValueError("cannot render a SLURM array script with zero runs")
    throttle = max_concurrent or profile.default_max_concurrent
    array = f"0-{len(run_list.runs) - 1}"
    if throttle:
        array = f"{array}%{throttle}"
    repo = profile.paths.get("repo", ".")
    logs = profile.paths.get("logs", "slurm_logs")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array}",
        f"#SBATCH --output={logs}/{job_name}_%A_%a.out",
        f"#SBATCH --error={logs}/{job_name}_%A_%a.err",
    ]
    if profile.account:
        lines.append(f"#SBATCH --account={profile.account}")
    if profile.partition:
        lines.append(f"#SBATCH --partition={profile.partition}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"RUN_LIST=\"${{RUN_LIST:-{run_list_path}}}\"",
            f"cd \"{repo}\"",
            f"mkdir -p \"{logs}\"",
            (
                f"PYTHONPATH=\"${{PYTHONPATH:-src}}\" {python} -m weighttraits.cli "
                "describe-training-run "
                "--run-list \"${RUN_LIST}\" "
                "--index \"${SLURM_ARRAY_TASK_ID}\""
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_slurm_array_script(
    run_list: TrainingRunList,
    path: str | Path,
    *,
    run_list_path: str | Path,
    profile: ExecutionProfile,
    job_name: str = "weighttraits-train",
    max_concurrent: int | None = None,
    python: str = "python",
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_slurm_array_script(
            run_list,
            run_list_path=run_list_path,
            profile=profile,
            job_name=job_name,
            max_concurrent=max_concurrent,
            python=python,
        )
    )


def _run_spec_from_job(
    job: Any,
    *,
    array_index: int,
    ledger_path: str,
    run_list_path: str | None,
    runner_entrypoint: str,
) -> TrainingRunSpec:
    job_dict = job.to_dict() if hasattr(job, "to_dict") else dict(job)
    node_id = str(job_dict["node_id"])
    return TrainingRunSpec(
        array_index=array_index,
        run_id=node_id,
        node_id=node_id,
        parent_id=str(job_dict.get("parent_id", "root")),
        depth=int(job_dict.get("depth", 0)),
        method=str(job_dict.get("method", "full")),
        dataset_id=job_dict.get("dataset_id"),
        task_family=job_dict.get("task_family"),
        init_from=str(job_dict.get("init_from", "")),
        output_dir=str(job_dict.get("output_dir", "")),
        expected_artifacts={
            str(k): str(v) for k, v in job_dict.get("expected_artifacts", {}).items()
        },
        ledger_path=ledger_path,
        runner={
            "entrypoint": runner_entrypoint,
            "run_list_path": run_list_path,
            "status": "planned",
        },
        job=job_dict,
    )


def _preflight_job(
    job: Any,
    *,
    all_node_ids: set[str],
    seen_node_ids: set[str],
    allow_existing_artifacts: bool,
    check_filesystem: bool,
    exists: PathExists,
) -> list[RunPreflightIssue]:
    job_dict = job.to_dict() if hasattr(job, "to_dict") else dict(job)
    node_id = str(job_dict["node_id"])
    parent_id = str(job_dict.get("parent_id", "root"))
    issues: list[RunPreflightIssue] = []

    if not job_dict.get("dataset_id"):
        issues.append(
            _issue(
                "error",
                "missing_dataset_id",
                node_id,
                "training job does not declare dataset_id",
            )
        )
    if parent_id != "root" and parent_id not in seen_node_ids:
        init_from = str(job_dict.get("init_from", ""))
        if parent_id in all_node_ids:
            issues.append(
                _issue(
                    "error",
                    "parent_after_child",
                    node_id,
                    f"parent {parent_id} appears after child in the run list",
                    path=init_from,
                )
            )
        elif not check_filesystem or exists(init_from):
            issues.append(
                _issue(
                    "warning",
                    "external_parent_artifact",
                    node_id,
                    f"parent {parent_id} is not in the run list but init artifact is external",
                    path=init_from,
                )
            )
        else:
            issues.append(
                _issue(
                    "error",
                    "missing_parent_artifact",
                    node_id,
                    f"parent {parent_id} is not in the run list and init artifact is missing",
                    path=init_from,
                )
            )

    if check_filesystem and not allow_existing_artifacts:
        for artifact_name, artifact_path in dict(job_dict.get("expected_artifacts", {})).items():
            if exists(str(artifact_path)):
                issues.append(
                    _issue(
                        "error",
                        "artifact_exists",
                        node_id,
                        f"expected artifact already exists: {artifact_name}",
                        path=str(artifact_path),
                    )
                )

    issues.extend(_preflight_stopping(job_dict, node_id))
    if job_dict.get("method") == "lora":
        lora = dict(job_dict.get("lora") or {})
        if lora.get("merge_after_train") is False:
            issues.append(
                _issue(
                    "error",
                    "lora_merge_disabled",
                    node_id,
                    (
                        "LoRA children initialize from merged parent weights; "
                        "merge_after_train is required"
                    ),
                )
            )
    return issues


def _preflight_stopping(job_dict: Mapping[str, Any], node_id: str) -> list[RunPreflightIssue]:
    stopping = dict(job_dict.get("stopping", {}))
    issues = []
    if stopping.get("patience") is None:
        issues.append(
            _issue(
                "warning",
                "no_early_stopping_patience",
                node_id,
                "job has no early-stopping patience",
            )
        )
    if stopping.get("plateau_window") is None:
        issues.append(
            _issue(
                "warning",
                "no_plateau_stopping",
                node_id,
                "job has no plateau window for small loss-diff stopping",
            )
        )
    if stopping.get("loss_increase_relative") is None:
        issues.append(
            _issue(
                "warning",
                "no_loss_increase_warning",
                node_id,
                "job has no relative loss-increase warning threshold",
            )
        )
    return issues


def _preflight_profile(
    profile: ExecutionProfile | None,
    n_runs: int,
) -> list[RunPreflightIssue]:
    if profile is None:
        return []
    issues = []
    if profile.require_run_list and n_runs == 0:
        issues.append(
            _issue(
                "error",
                "empty_run_list",
                None,
                "profile requires an explicit non-empty run list",
            )
        )
    if (
        profile.default_max_concurrent is not None
        and n_runs > profile.default_max_concurrent
        and not profile.allow_broad_arrays
    ):
        issues.append(
            _issue(
                "warning",
                "array_exceeds_default_concurrency",
                None,
                f"{n_runs} runs exceed default concurrency {profile.default_max_concurrent}",
            )
        )
    return issues


def _default_ledger_path(jobs: Sequence[Any]) -> str:
    if not jobs:
        return "outputs/training_ledger.jsonl"
    first = jobs[0].to_dict() if hasattr(jobs[0], "to_dict") else dict(jobs[0])
    output_dir = Path(str(first.get("output_dir", "outputs")))
    return str(output_dir.parent / "training_ledger.jsonl")


def _run_spec_from_dict(row: dict[str, Any]) -> TrainingRunSpec:
    return TrainingRunSpec(
        array_index=int(row["array_index"]),
        run_id=str(row["run_id"]),
        node_id=str(row["node_id"]),
        parent_id=str(row["parent_id"]),
        depth=int(row["depth"]),
        method=str(row["method"]),
        dataset_id=row.get("dataset_id"),
        task_family=row.get("task_family"),
        init_from=str(row["init_from"]),
        output_dir=str(row["output_dir"]),
        expected_artifacts={str(k): str(v) for k, v in row.get("expected_artifacts", {}).items()},
        ledger_path=str(row["ledger_path"]),
        runner=dict(row.get("runner", {})),
        job=dict(row.get("job", {})),
    )


def _issue(
    severity: str,
    issue: str,
    node_id: str | None,
    message: str,
    *,
    path: str | None = None,
) -> RunPreflightIssue:
    return RunPreflightIssue(
        severity=severity,
        issue=issue,
        node_id=node_id,
        message=message,
        path=path,
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
