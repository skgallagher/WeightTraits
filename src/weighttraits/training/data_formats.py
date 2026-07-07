"""Offline dataset format contracts for training prompt validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from weighttraits.training.planner import TrainingJob


@dataclass(frozen=True)
class DatasetFormatSpec:
    dataset_id: str
    task_family: str | None
    prompt_fields: tuple[str, ...]
    raw_fields: tuple[str, ...] = ()
    field_map: dict[str, str] | None = None
    train_split: str | None = None
    eval_split: str | None = None
    role: str = "training"
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["prompt_fields"] = list(self.prompt_fields)
        out["raw_fields"] = list(self.raw_fields)
        return out


@dataclass(frozen=True)
class JobFormatIssue:
    node_id: str
    dataset_id: str | None
    task_family: str | None
    prompt_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    issue: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["prompt_fields"] = list(self.prompt_fields)
        out["missing_fields"] = list(self.missing_fields)
        return out


@dataclass(frozen=True)
class DataFormatValidationReport:
    n_jobs: int
    n_valid: int
    issues: tuple[JobFormatIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "n_jobs": self.n_jobs,
            "n_valid": self.n_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_dataset_format_specs(path: str | Path) -> dict[str, DatasetFormatSpec]:
    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict):
        rows = raw.get("datasets", raw)
        if isinstance(rows, dict):
            rows = [dict({"dataset_id": key}, **value) for key, value in rows.items()]
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError(f"dataset format specs must be a list or mapping: {path}")
    specs = [_normalize_spec(row) for row in rows]
    return {spec.dataset_id: spec for spec in specs}


def validate_training_jobs_against_formats(
    jobs: list[TrainingJob],
    specs: dict[str, DatasetFormatSpec],
) -> DataFormatValidationReport:
    issues: list[JobFormatIssue] = []
    valid_count = 0
    for job in jobs:
        if job.dataset_id is None:
            issues.append(_issue(job, "missing_dataset_id", job.prompt_fields))
            continue
        spec = specs.get(job.dataset_id)
        if spec is None:
            issues.append(_issue(job, "missing_dataset_format_spec", job.prompt_fields))
            continue
        if (
            spec.task_family is not None
            and job.task_family is not None
            and spec.task_family != job.task_family
        ):
            issues.append(
                _issue(
                    job,
                    "task_family_mismatch",
                    (),
                    details={
                        "job_task_family": job.task_family,
                        "spec_task_family": spec.task_family,
                    },
                )
            )
            continue
        available = set(spec.prompt_fields)
        missing = tuple(field for field in job.prompt_fields if field not in available)
        if missing:
            issues.append(_issue(job, "missing_prompt_fields", missing))
            continue
        valid_count += 1
    return DataFormatValidationReport(
        n_jobs=len(jobs),
        n_valid=valid_count,
        issues=tuple(issues),
    )


def write_data_format_report(report: DataFormatValidationReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def lookup_field(row: Mapping[str, Any], raw_field: str) -> tuple[bool, Any]:
    current: Any = row
    for part in str(raw_field).split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


def _normalize_spec(row: Any) -> DatasetFormatSpec:
    if not isinstance(row, dict):
        raise ValueError("dataset format spec rows must be mappings")
    dataset_id = row.get("dataset_id") or row.get("id")
    if not dataset_id:
        raise ValueError("dataset format spec requires dataset_id")
    field_map = row.get("field_map")
    if field_map is not None and not isinstance(field_map, dict):
        raise ValueError(f"field_map must be a mapping for dataset {dataset_id}")
    prompt_fields = row.get("prompt_fields")
    if prompt_fields is None and field_map is not None:
        prompt_fields = sorted(field_map)
    if prompt_fields is None:
        raise ValueError(f"dataset format spec requires prompt_fields or field_map: {dataset_id}")
    return DatasetFormatSpec(
        dataset_id=str(dataset_id),
        task_family=None if row.get("task_family") is None else str(row.get("task_family")),
        prompt_fields=tuple(sorted(_as_str_list(prompt_fields, "prompt_fields", str(dataset_id)))),
        raw_fields=tuple(sorted(_as_str_list(row.get("raw_fields", []), "raw_fields", str(dataset_id)))),
        field_map={str(key): str(value) for key, value in field_map.items()} if field_map else None,
        train_split=None if row.get("train_split") is None else str(row.get("train_split")),
        eval_split=None if row.get("eval_split") is None else str(row.get("eval_split")),
        role=str(row.get("role", "training")),
        note=None if row.get("note") is None else str(row.get("note")),
    )


def _as_str_list(value: Any, key: str, dataset_id: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list for dataset {dataset_id}")
    return [str(item) for item in value]


def _issue(
    job: TrainingJob,
    issue: str,
    missing_fields: tuple[str, ...],
    *,
    details: dict[str, Any] | None = None,
) -> JobFormatIssue:
    return JobFormatIssue(
        node_id=job.node_id,
        dataset_id=job.dataset_id,
        task_family=job.task_family,
        prompt_fields=job.prompt_fields,
        missing_fields=missing_fields,
        issue=issue,
        details=details or {},
    )
