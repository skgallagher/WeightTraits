"""Dataset registry and split audits for training execution dry runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from weighttraits.training.data_formats import DatasetFormatSpec


DatasetLoader = Callable[..., Any]


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_id: str
    task_family: str | None
    hf_args: tuple[str, ...]
    role: str = "training"
    status: str | None = None
    filter: dict[str, Any] | None = None
    note: str | None = None
    train_split: str | None = None
    eval_split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hf_args"] = list(self.hf_args)
        return out


@dataclass(frozen=True)
class DatasetSplitAudit:
    dataset_id: str
    task_family: str | None
    hf_args: tuple[str, ...]
    role: str
    requested_splits: tuple[str, ...]
    available_splits: tuple[str, ...]
    row_counts: dict[str, int | None]
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hf_args"] = list(self.hf_args)
        out["requested_splits"] = list(self.requested_splits)
        out["available_splits"] = list(self.available_splits)
        return out


@dataclass(frozen=True)
class DatasetAuditReport:
    n_datasets: int
    n_ok: int
    audits: tuple[DatasetSplitAudit, ...]

    @property
    def valid(self) -> bool:
        return all(audit.status in {"ok", "not_loaded"} for audit in self.audits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "n_datasets": self.n_datasets,
            "n_ok": self.n_ok,
            "audits": [audit.to_dict() for audit in self.audits],
        }


def load_dataset_registry(path: str | Path) -> dict[str, DatasetRegistryEntry]:
    """Load task/data candidate registry entries keyed by dataset id."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"dataset registry must be a mapping: {path}")
    entries = _entries_from_task_families(raw)
    if not entries:
        entries = _entries_from_dataset_rows(raw.get("datasets", []), task_family=None, family_status=None)
    if not entries:
        raise ValueError(f"dataset registry has no datasets: {path}")
    out: dict[str, DatasetRegistryEntry] = {}
    for entry in entries:
        if entry.dataset_id in out:
            raise ValueError(f"duplicate dataset id in registry: {entry.dataset_id}")
        out[entry.dataset_id] = entry
    return out


def audit_dataset_registry(
    registry: Mapping[str, DatasetRegistryEntry],
    *,
    dataset_ids: list[str] | None = None,
    format_specs: Mapping[str, DatasetFormatSpec] | None = None,
    load: bool = True,
    loader: DatasetLoader | None = None,
) -> DatasetAuditReport:
    """Audit registered Hugging Face datasets and requested split names.

    With ``load=False`` this validates registry shape and requested splits without importing
    or downloading Hugging Face datasets. With ``load=True`` it calls ``loader`` if provided,
    otherwise ``datasets.load_dataset``.
    """
    ids = dataset_ids or sorted(registry)
    audits: list[DatasetSplitAudit] = []
    for dataset_id in ids:
        entry = registry.get(dataset_id)
        if entry is None:
            audits.append(
                DatasetSplitAudit(
                    dataset_id=dataset_id,
                    task_family=None,
                    hf_args=(),
                    role="unknown",
                    requested_splits=(),
                    available_splits=(),
                    row_counts={},
                    status="unknown_dataset_id",
                    error=f"dataset id not found in registry: {dataset_id}",
                )
            )
            continue
        requested = _requested_splits(entry, None if format_specs is None else format_specs.get(dataset_id))
        if not load:
            audits.append(
                DatasetSplitAudit(
                    dataset_id=entry.dataset_id,
                    task_family=entry.task_family,
                    hf_args=entry.hf_args,
                    role=entry.role,
                    requested_splits=requested,
                    available_splits=(),
                    row_counts={},
                    status="not_loaded",
                )
            )
            continue
        audits.append(_load_and_audit(entry, requested, loader))
    n_ok = sum(1 for audit in audits if audit.status in {"ok", "not_loaded"})
    return DatasetAuditReport(n_datasets=len(audits), n_ok=n_ok, audits=tuple(audits))


def write_dataset_audit_report(report: DatasetAuditReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def _entries_from_task_families(raw: dict[str, Any]) -> list[DatasetRegistryEntry]:
    families = raw.get("task_families", {})
    if not isinstance(families, dict):
        return []
    entries: list[DatasetRegistryEntry] = []
    for family_id, family in families.items():
        if not isinstance(family, dict):
            raise ValueError(f"task family must be a mapping: {family_id}")
        entries.extend(
            _entries_from_dataset_rows(
                family.get("datasets", []),
                task_family=str(family_id),
                family_status=None if family.get("status") is None else str(family.get("status")),
            )
        )
    return entries


def _entries_from_dataset_rows(
    rows: Any,
    *,
    task_family: str | None,
    family_status: str | None,
) -> list[DatasetRegistryEntry]:
    if not isinstance(rows, list):
        raise ValueError("dataset registry rows must be a list")
    entries = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("dataset registry row must be a mapping")
        dataset_id = row.get("dataset_id") or row.get("id")
        if not dataset_id:
            raise ValueError("dataset registry row requires id or dataset_id")
        hf_args = row.get("hf_args")
        if not isinstance(hf_args, list) or not hf_args:
            raise ValueError(f"dataset {dataset_id} requires non-empty hf_args")
        filter_spec = row.get("filter")
        if filter_spec is not None and not isinstance(filter_spec, dict):
            raise ValueError(f"dataset {dataset_id} filter must be a mapping")
        entries.append(
            DatasetRegistryEntry(
                dataset_id=str(dataset_id),
                task_family=task_family,
                hf_args=tuple(str(item) for item in hf_args),
                role=str(row.get("role", "training")),
                status=None if row.get("status") is None else str(row.get("status")),
                filter={str(k): v for k, v in filter_spec.items()} if filter_spec else None,
                note=None if row.get("note") is None else str(row.get("note")),
                train_split=None if row.get("train_split") is None else str(row.get("train_split")),
                eval_split=None if row.get("eval_split") is None else str(row.get("eval_split")),
            )
        )
    return entries


def _requested_splits(
    entry: DatasetRegistryEntry,
    spec: DatasetFormatSpec | None,
) -> tuple[str, ...]:
    splits = []
    for value in (entry.train_split, entry.eval_split):
        if value and value not in splits:
            splits.append(value)
    if spec is not None:
        for value in (spec.train_split, spec.eval_split):
            if value and value not in splits:
                splits.append(value)
    return tuple(splits)


def _load_and_audit(
    entry: DatasetRegistryEntry,
    requested_splits: tuple[str, ...],
    loader: DatasetLoader | None,
) -> DatasetSplitAudit:
    try:
        dataset_loader = loader or _default_hf_loader()
        dataset = dataset_loader(*entry.hf_args)
        available_splits, row_counts = _dataset_split_summary(dataset)
        missing = sorted(set(requested_splits) - set(available_splits))
        status = "missing_splits" if missing else "ok"
        error = f"missing requested splits: {', '.join(missing)}" if missing else None
    except Exception as exc:  # pragma: no cover - exact HF exceptions vary by environment.
        available_splits = ()
        row_counts = {}
        status = "load_failed"
        error = str(exc)
    return DatasetSplitAudit(
        dataset_id=entry.dataset_id,
        task_family=entry.task_family,
        hf_args=entry.hf_args,
        role=entry.role,
        requested_splits=requested_splits,
        available_splits=available_splits,
        row_counts=row_counts,
        status=status,
        error=error,
    )


def _default_hf_loader() -> DatasetLoader:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError(
            "datasets is not installed; install WeightTraits with the training extra or use --no-load"
        ) from exc
    return load_dataset


def _dataset_split_summary(dataset: Any) -> tuple[tuple[str, ...], dict[str, int | None]]:
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        names = tuple(str(name) for name in dataset.keys())
        counts = {name: _safe_len(dataset[name]) for name in names}
        return names, counts
    return ("<single>",), {"<single>": _safe_len(dataset)}


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None
