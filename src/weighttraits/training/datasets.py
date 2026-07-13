"""Dataset registry and split audits for training execution dry runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import yaml

from weighttraits.training.data_formats import DatasetFormatSpec, lookup_field
from weighttraits.training.prompts import render_prompt


DatasetLoader = Callable[..., Any]


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_id: str
    task_family: str | None
    hf_args: tuple[str, ...]
    hf_kwargs: dict[str, Any] | None = None
    role: str = "training"
    status: str | None = None
    filter: dict[str, Any] | None = None
    note: str | None = None
    train_split: str | None = None
    eval_split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hf_args"] = list(self.hf_args)
        if self.hf_kwargs is None:
            out["hf_kwargs"] = {}
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
    hf_kwargs: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hf_args"] = list(self.hf_args)
        if self.hf_kwargs is None:
            out["hf_kwargs"] = {}
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


@dataclass(frozen=True)
class SampleRenderIssue:
    sample_index: int | None
    issue: str
    missing_fields: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["missing_fields"] = list(self.missing_fields)
        return out


@dataclass(frozen=True)
class TrainingSampleRenderAudit:
    node_id: str
    dataset_id: str | None
    task_family: str | None
    split: str | None
    prompt_source: str
    prompt_fields: tuple[str, ...]
    n_seen: int
    n_rendered: int
    n_missing_field_rows: int
    n_empty_render_rows: int
    status: str
    issues: tuple[SampleRenderIssue, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["prompt_fields"] = list(self.prompt_fields)
        out["issues"] = [issue.to_dict() for issue in self.issues]
        return out


@dataclass(frozen=True)
class TrainingSampleRenderAuditReport:
    n_jobs: int
    n_ok: int
    audits: tuple[TrainingSampleRenderAudit, ...]

    @property
    def valid(self) -> bool:
        return all(audit.status == "ok" for audit in self.audits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "n_jobs": self.n_jobs,
            "n_ok": self.n_ok,
            "audits": [audit.to_dict() for audit in self.audits],
        }


@dataclass(frozen=True)
class DatasetCacheSplitReport:
    dataset_id: str
    split: str
    path: str | None
    limit: int | None
    min_required: int
    n_scanned: int
    n_cached: int
    n_filtered: int
    n_dropped: int
    status: str
    error: str | None = None
    sample_strategy: str = "first"
    sample_seed: int | None = None
    shuffle_buffer_size: int | None = None

    @property
    def valid(self) -> bool:
        return self.status in {"ok", "cached", "skipped"} and self.n_cached >= self.min_required

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetCacheReport:
    out_dir: str
    n_datasets: int
    n_ok: int
    splits: tuple[DatasetCacheSplitReport, ...]
    sample_strategy: str = "first"
    sample_seed: int | None = None
    shuffle_buffer_size: int | None = None

    @property
    def valid(self) -> bool:
        return all(split.valid for split in self.splits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "out_dir": self.out_dir,
            "n_datasets": self.n_datasets,
            "n_ok": self.n_ok,
            "sample_strategy": self.sample_strategy,
            "sample_seed": self.sample_seed,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "splits": [split.to_dict() for split in self.splits],
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
                    hf_kwargs=entry.hf_kwargs,
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


def audit_training_sample_rendering(
    jobs: Sequence[Any],
    registry: Mapping[str, DatasetRegistryEntry],
    format_specs: Mapping[str, DatasetFormatSpec],
    *,
    dataset_ids: list[str] | None = None,
    max_samples: int = 8,
    split: str | None = None,
    loader: DatasetLoader | None = None,
) -> TrainingSampleRenderAuditReport:
    """Render a small dataset sample for each planned training job.

    The audit applies each dataset's field map before rendering the planned prompt
    template. Reports intentionally include only counts, field names, indices, and
    errors; they do not serialize raw examples or rendered prompts.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    selected_ids = set(dataset_ids or [])
    selected_jobs = [
        job
        for job in jobs
        if not selected_ids or getattr(job, "dataset_id", None) in selected_ids
    ]
    dataset_cache: dict[str, Any] = {}
    audits = [
        _audit_job_sample_rendering(
            job,
            registry=registry,
            format_specs=format_specs,
            dataset_cache=dataset_cache,
            max_samples=max_samples,
            split_override=split,
            loader=loader,
        )
        for job in selected_jobs
    ]
    n_ok = sum(1 for audit in audits if audit.status == "ok")
    return TrainingSampleRenderAuditReport(n_jobs=len(audits), n_ok=n_ok, audits=tuple(audits))


def select_training_sample_jobs(
    jobs: Sequence[Any],
    *,
    dataset_ids: list[str] | None = None,
    selection: str = "all-jobs",
) -> list[Any]:
    """Select planned jobs for sample-render audits."""
    selected_ids = set(dataset_ids or [])
    filtered = [
        job
        for job in jobs
        if not selected_ids or getattr(job, "dataset_id", None) in selected_ids
    ]
    if selection == "all-jobs":
        return filtered
    if selection != "one-per-dataset":
        raise ValueError(f"unsupported sample job selection: {selection}")

    out = []
    seen: set[str | None] = set()
    for job in filtered:
        dataset_id = getattr(job, "dataset_id", None)
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        out.append(job)
    return out


def write_training_sample_render_audit_report(
    report: TrainingSampleRenderAuditReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def cache_training_datasets(
    registry: Mapping[str, DatasetRegistryEntry],
    format_specs: Mapping[str, DatasetFormatSpec],
    *,
    out_dir: str | Path,
    dataset_ids: list[str] | None = None,
    train_limit: int | None = 10_000,
    eval_limit: int | None = 1_000,
    max_scan: int | None = None,
    min_train_rows: int = 10_000,
    min_eval_rows: int = 0,
    loader: DatasetLoader | None = None,
    overwrite: bool = False,
    sample_strategy: str = "first",
    sample_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
) -> DatasetCacheReport:
    """Write bounded canonical-row dataset caches for training runs.

    ``first`` preserves registry order. ``seeded_shuffle`` performs a deterministic
    shuffle before filtering and truncation. For streaming iterables the shuffle is
    bounded by ``shuffle_buffer_size``.
    """
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be positive or None")
    if eval_limit is not None and eval_limit < 1:
        raise ValueError("eval_limit must be positive or None")
    if max_scan is not None and max_scan < 1:
        raise ValueError("max_scan must be positive or None")
    if sample_strategy not in {"first", "seeded_shuffle"}:
        raise ValueError("sample_strategy must be 'first' or 'seeded_shuffle'")
    if shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive")
    effective_seed = sample_seed if sample_strategy == "seeded_shuffle" else None
    effective_buffer_size = shuffle_buffer_size if sample_strategy == "seeded_shuffle" else None
    out_root = Path(out_dir)
    ids = dataset_ids or sorted(registry)
    split_reports: list[DatasetCacheSplitReport] = []
    for dataset_id in ids:
        entry = registry.get(dataset_id)
        spec = format_specs.get(dataset_id)
        if entry is None:
            split_reports.append(
                _cache_error_report(
                    dataset_id=dataset_id,
                    split="train",
                    min_required=min_train_rows,
                    error=f"dataset id not found in registry: {dataset_id}",
                )
            )
            continue
        if spec is None:
            split_reports.append(
                _cache_error_report(
                    dataset_id=dataset_id,
                    split="train",
                    min_required=min_train_rows,
                    error=f"dataset id not found in format specs: {dataset_id}",
                )
            )
            continue
        train_split = spec.train_split or entry.train_split or "train"
        eval_split = spec.eval_split or entry.eval_split
        try:
            dataset = _load_uncached_dataset(entry, loader)
        except Exception as exc:  # pragma: no cover - exact HF exceptions vary by environment.
            split_reports.append(
                _cache_error_report(
                    dataset_id=dataset_id,
                    split=train_split,
                    min_required=min_train_rows,
                    error=str(exc),
                )
            )
            continue

        train_dataset, available_splits = _select_split_dataset(dataset, train_split)
        if train_dataset is None:
            split_reports.append(
                _cache_error_report(
                    dataset_id=dataset_id,
                    split=train_split,
                    min_required=min_train_rows,
                    error=f"missing train split: {train_split}; available: {', '.join(available_splits)}",
                )
            )
        else:
            split_reports.append(
                _cache_dataset_split(
                    train_dataset,
                    entry=entry,
                    spec=spec,
                    out_dir=out_root,
                    split_name=train_split,
                    limit=train_limit,
                    max_scan=max_scan,
                    min_required=min_train_rows,
                    overwrite=overwrite,
                    sample_strategy=sample_strategy,
                    sample_seed=effective_seed,
                    shuffle_buffer_size=effective_buffer_size,
                )
            )

        if eval_split:
            eval_dataset, available_splits = _select_split_dataset(dataset, eval_split)
            if eval_dataset is None:
                split_reports.append(
                    _cache_error_report(
                        dataset_id=dataset_id,
                        split=eval_split,
                        min_required=min_eval_rows,
                        error=f"missing eval split: {eval_split}; available: {', '.join(available_splits)}",
                    )
                )
            else:
                split_reports.append(
                    _cache_dataset_split(
                        eval_dataset,
                        entry=entry,
                        spec=spec,
                        out_dir=out_root,
                        split_name=eval_split,
                        limit=eval_limit,
                        max_scan=max_scan,
                        min_required=min_eval_rows,
                        overwrite=overwrite,
                        sample_strategy=sample_strategy,
                        sample_seed=effective_seed,
                        shuffle_buffer_size=effective_buffer_size,
                    )
                )
    ids_with_reports = {split.dataset_id for split in split_reports}
    n_ok = sum(
        1
        for dataset_id in ids_with_reports
        if all(split.valid for split in split_reports if split.dataset_id == dataset_id)
    )
    return DatasetCacheReport(
        out_dir=str(out_root),
        n_datasets=len(ids),
        n_ok=n_ok,
        splits=tuple(split_reports),
        sample_strategy=sample_strategy,
        sample_seed=effective_seed,
        shuffle_buffer_size=effective_buffer_size,
    )


def write_dataset_cache_report(report: DatasetCacheReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def dataset_cache_split_path(cache_root: str | Path, dataset_id: str, split_name: str) -> Path:
    safe_split = str(split_name).replace("/", "_")
    return Path(cache_root) / str(dataset_id) / f"{safe_split}.jsonl"


def load_cached_dataset_splits(
    cache_root: str | Path,
    *,
    dataset_id: str,
    train_split: str,
    eval_split: str | None = None,
    require: bool = False,
) -> dict[str, Any] | None:
    train_path = dataset_cache_split_path(cache_root, dataset_id, train_split)
    if not train_path.exists():
        if require:
            raise FileNotFoundError(f"missing cached train split: {train_path}")
        return None
    splits: dict[str, Any] = {train_split: _iter_jsonl_rows(train_path)}
    if eval_split:
        eval_path = dataset_cache_split_path(cache_root, dataset_id, eval_split)
        if eval_path.exists():
            splits[eval_split] = _iter_jsonl_rows(eval_path)
        elif require:
            raise FileNotFoundError(f"missing cached eval split: {eval_path}")
    return splits


def canonical_dataset_example(row: Any, spec: DatasetFormatSpec) -> dict[str, Any]:
    row_map = {str(key): value for key, value in row.items()}
    canonical = dict(row_map)
    for prompt_field, raw_field in (spec.field_map or {}).items():
        found, value = lookup_field(row_map, raw_field)
        if found:
            canonical[prompt_field] = value
    return canonical


def dataset_example_passes_filter(
    example: Mapping[str, Any],
    filter_spec: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
    if not filter_spec:
        return True, None
    for field, limit in _filter_field_limits(filter_spec.get("max_chars")).items():
        if field not in example:
            return False, f"missing filter field: {field}"
        if len(str(example[field])) > limit:
            return False, f"{field} exceeds max_chars={limit}"
    for field, limit in _filter_field_limits(filter_spec.get("min_chars")).items():
        if field not in example:
            return False, f"missing filter field: {field}"
        if len(str(example[field])) < limit:
            return False, f"{field} below min_chars={limit}"
    for field in _filter_field_list(filter_spec.get("require_nonempty")):
        if field not in example or not str(example[field]).strip():
            return False, f"{field} is empty"
    return True, None


def _cache_dataset_split(
    dataset_split: Any,
    *,
    entry: DatasetRegistryEntry,
    spec: DatasetFormatSpec,
    out_dir: Path,
    split_name: str,
    limit: int | None,
    max_scan: int | None,
    min_required: int,
    overwrite: bool,
    sample_strategy: str,
    sample_seed: int | None,
    shuffle_buffer_size: int | None,
) -> DatasetCacheSplitReport:
    path = dataset_cache_split_path(out_dir, entry.dataset_id, split_name)
    metadata_path = path.with_suffix(".metadata.json")
    if path.exists() and not overwrite:
        n_cached = _count_jsonl_rows(path)
        metadata = _read_cache_metadata(metadata_path)
        requested = _cache_sampling_metadata(
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        existing = metadata or _cache_sampling_metadata(
            sample_strategy="first",
            sample_seed=None,
            shuffle_buffer_size=None,
        )
        sampling_matches = all(existing.get(key) == value for key, value in requested.items())
        if not sampling_matches:
            return DatasetCacheSplitReport(
                dataset_id=entry.dataset_id,
                split=split_name,
                path=str(path),
                limit=limit,
                min_required=min_required,
                n_scanned=0,
                n_cached=n_cached,
                n_filtered=0,
                n_dropped=0,
                status="sampling_mismatch",
                error="existing cache uses different sampling; rerun with --overwrite",
                sample_strategy=str(existing.get("sample_strategy", "unknown")),
                sample_seed=existing.get("sample_seed"),
                shuffle_buffer_size=existing.get("shuffle_buffer_size"),
            )
        return DatasetCacheSplitReport(
            dataset_id=entry.dataset_id,
            split=split_name,
            path=str(path),
            limit=limit,
            min_required=min_required,
            n_scanned=0,
            n_cached=n_cached,
            n_filtered=0,
            n_dropped=0,
            status="cached" if n_cached >= min_required else "short",
            error=None if n_cached >= min_required else "existing cache is shorter than required",
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    n_scanned = 0
    n_cached = 0
    n_filtered = 0
    n_dropped = 0
    with path.open("w") as handle:
        sampled_rows = _sample_dataset_rows(
            dataset_split,
            strategy=sample_strategy,
            seed=sample_seed,
            buffer_size=shuffle_buffer_size,
        )
        for row in sampled_rows:
            if limit is not None and n_cached >= limit:
                break
            if max_scan is not None and n_scanned >= max_scan:
                break
            n_scanned += 1
            if not _is_mapping_like(row):
                n_dropped += 1
                continue
            try:
                canonical = canonical_dataset_example(row, spec)
                missing = [field for field in _cache_required_fields(spec) if field not in canonical]
                if missing:
                    n_dropped += 1
                    continue
                keep, _ = dataset_example_passes_filter(canonical, entry.filter)
            except Exception:
                n_dropped += 1
                continue
            if not keep:
                n_filtered += 1
                continue
            handle.write(
                json.dumps(_cache_row(canonical, spec), default=str, sort_keys=True) + "\n"
            )
            n_cached += 1
    metadata_path.write_text(
        json.dumps(
            _cache_sampling_metadata(
                sample_strategy=sample_strategy,
                sample_seed=sample_seed,
                shuffle_buffer_size=shuffle_buffer_size,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if n_cached >= min_required:
        status = "ok"
    elif max_scan is not None and n_scanned >= max_scan:
        status = "scan_limit"
    else:
        status = "short"
    return DatasetCacheSplitReport(
        dataset_id=entry.dataset_id,
        split=split_name,
        path=str(path),
        limit=limit,
        min_required=min_required,
        n_scanned=n_scanned,
        n_cached=n_cached,
        n_filtered=n_filtered,
        n_dropped=n_dropped,
        status=status,
        error=None if status == "ok" else "not enough rows after filtering",
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )


def _sample_dataset_rows(
    rows: Any,
    *,
    strategy: str,
    seed: int | None,
    buffer_size: int | None,
):
    if strategy == "first":
        return rows
    if seed is None or buffer_size is None:
        raise ValueError("seeded_shuffle requires a seed and shuffle buffer size")
    shuffle = getattr(rows, "shuffle", None)
    if callable(shuffle):
        try:
            return shuffle(seed=seed, buffer_size=buffer_size)
        except TypeError:
            return shuffle(seed=seed)
    return _buffer_shuffle(rows, seed=seed, buffer_size=buffer_size)


def _buffer_shuffle(rows: Any, *, seed: int, buffer_size: int):
    rng = random.Random(seed)
    buffer: list[Any] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer


def _cache_sampling_metadata(
    *,
    sample_strategy: str,
    sample_seed: int | None,
    shuffle_buffer_size: int | None,
) -> dict[str, Any]:
    return {
        "sample_strategy": sample_strategy,
        "sample_seed": sample_seed,
        "shuffle_buffer_size": shuffle_buffer_size,
    }


def _read_cache_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else None


def _cache_error_report(
    *,
    dataset_id: str,
    split: str,
    min_required: int,
    error: str,
) -> DatasetCacheSplitReport:
    return DatasetCacheSplitReport(
        dataset_id=dataset_id,
        split=split,
        path=None,
        limit=None,
        min_required=min_required,
        n_scanned=0,
        n_cached=0,
        n_filtered=0,
        n_dropped=0,
        status="error",
        error=error,
    )


def _cache_row(canonical: Mapping[str, Any], spec: DatasetFormatSpec) -> dict[str, Any]:
    fields = _cache_required_fields(spec)
    fields.add("target")
    return {field: canonical[field] for field in sorted(fields) if field in canonical}


def _cache_required_fields(spec: DatasetFormatSpec) -> set[str]:
    fields = set(spec.prompt_fields)
    fields.update((spec.field_map or {}).keys())
    return fields


def _iter_jsonl_rows(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _count_jsonl_rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def _filter_field_limits(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("filter field limits must be a mapping")
    out: dict[str, int] = {}
    for field, limit in value.items():
        out[str(field)] = int(limit)
    return out


def _filter_field_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("filter field lists must be lists")
    return tuple(str(item) for item in value)


def _audit_job_sample_rendering(
    job: Any,
    *,
    registry: Mapping[str, DatasetRegistryEntry],
    format_specs: Mapping[str, DatasetFormatSpec],
    dataset_cache: dict[str, Any],
    max_samples: int,
    split_override: str | None,
    loader: DatasetLoader | None,
) -> TrainingSampleRenderAudit:
    node_id = str(getattr(job, "node_id", ""))
    dataset_id = getattr(job, "dataset_id", None)
    task_family = getattr(job, "task_family", None)
    prompt_source = str(getattr(job, "prompt_source", "unknown"))
    prompt_fields = tuple(str(field) for field in getattr(job, "prompt_fields", ()))
    prompt_template = str(getattr(job, "prompt_template", ""))

    if not dataset_id:
        return _sample_render_audit(
            node_id=node_id,
            dataset_id=None,
            task_family=task_family,
            prompt_source=prompt_source,
            prompt_fields=prompt_fields,
            status="missing_dataset_id",
            error="training job does not declare dataset_id",
        )
    dataset_key = str(dataset_id)
    entry = registry.get(dataset_key)
    if entry is None:
        return _sample_render_audit(
            node_id=node_id,
            dataset_id=dataset_key,
            task_family=task_family,
            prompt_source=prompt_source,
            prompt_fields=prompt_fields,
            status="unknown_dataset_id",
            error=f"dataset id not found in registry: {dataset_key}",
        )
    spec = format_specs.get(dataset_key)
    if spec is None:
        return _sample_render_audit(
            node_id=node_id,
            dataset_id=dataset_key,
            task_family=task_family,
            prompt_source=prompt_source,
            prompt_fields=prompt_fields,
            status="missing_dataset_format_spec",
            error=f"dataset id not found in format specs: {dataset_key}",
        )

    split_name = _sample_split_name(entry, spec, split_override)
    try:
        dataset = _load_cached_dataset(entry, dataset_cache, loader)
    except Exception as exc:  # pragma: no cover - exact HF exceptions vary by environment.
        return _sample_render_audit(
            node_id=node_id,
            dataset_id=dataset_key,
            task_family=task_family,
            prompt_source=prompt_source,
            prompt_fields=prompt_fields,
            split=split_name,
            status="load_failed",
            error=str(exc),
        )

    split_dataset, available_splits = _select_split_dataset(dataset, split_name)
    if split_dataset is None:
        return _sample_render_audit(
            node_id=node_id,
            dataset_id=dataset_key,
            task_family=task_family,
            prompt_source=prompt_source,
            prompt_fields=prompt_fields,
            split=split_name,
            status="missing_split",
            issues=(
                SampleRenderIssue(
                    sample_index=None,
                    issue="missing_split",
                    error=f"available splits: {', '.join(available_splits)}",
                ),
            ),
            error=f"missing split: {split_name}",
        )

    n_seen = 0
    n_rendered = 0
    n_missing_field_rows = 0
    n_empty_render_rows = 0
    issues: list[SampleRenderIssue] = []
    for sample_index, row in _iter_sample_rows(split_dataset, spec, entry.filter, max_samples):
        n_seen += 1
        if not _is_mapping_like(row):
            issues.append(SampleRenderIssue(sample_index=sample_index, issue="non_mapping_row"))
            continue
        canonical, missing = _canonical_prompt_example(row, spec, prompt_fields)
        if missing:
            n_missing_field_rows += 1
            issues.append(
                SampleRenderIssue(
                    sample_index=sample_index,
                    issue="missing_prompt_fields",
                    missing_fields=missing,
                )
            )
            continue
        try:
            rendered = render_prompt(prompt_template, canonical)
        except Exception as exc:
            issues.append(
                SampleRenderIssue(
                    sample_index=sample_index,
                    issue="render_failed",
                    error=str(exc),
                )
            )
            continue
        if not rendered.strip():
            n_empty_render_rows += 1
            issues.append(
                SampleRenderIssue(sample_index=sample_index, issue="empty_rendered_prompt")
            )
            continue
        n_rendered += 1

    if n_seen == 0:
        status = "empty_split"
        issues.append(SampleRenderIssue(sample_index=None, issue="empty_split"))
    elif issues:
        status = "row_issues"
    else:
        status = "ok"
    return TrainingSampleRenderAudit(
        node_id=node_id,
        dataset_id=dataset_key,
        task_family=task_family,
        split=split_name,
        prompt_source=prompt_source,
        prompt_fields=prompt_fields,
        n_seen=n_seen,
        n_rendered=n_rendered,
        n_missing_field_rows=n_missing_field_rows,
        n_empty_render_rows=n_empty_render_rows,
        status=status,
        issues=tuple(issues),
    )


def _sample_render_audit(
    *,
    node_id: str,
    dataset_id: str | None,
    task_family: str | None,
    prompt_source: str,
    prompt_fields: tuple[str, ...],
    status: str,
    split: str | None = None,
    issues: tuple[SampleRenderIssue, ...] = (),
    error: str | None = None,
) -> TrainingSampleRenderAudit:
    return TrainingSampleRenderAudit(
        node_id=node_id,
        dataset_id=dataset_id,
        task_family=task_family,
        split=split,
        prompt_source=prompt_source,
        prompt_fields=prompt_fields,
        n_seen=0,
        n_rendered=0,
        n_missing_field_rows=0,
        n_empty_render_rows=0,
        status=status,
        issues=issues,
        error=error,
    )


def _sample_split_name(
    entry: DatasetRegistryEntry,
    spec: DatasetFormatSpec,
    split_override: str | None,
) -> str:
    return (
        split_override
        or spec.train_split
        or entry.train_split
        or spec.eval_split
        or entry.eval_split
        or "train"
    )


def _load_cached_dataset(
    entry: DatasetRegistryEntry,
    dataset_cache: dict[str, Any],
    loader: DatasetLoader | None,
) -> Any:
    if entry.dataset_id not in dataset_cache:
        dataset_cache[entry.dataset_id] = _load_uncached_dataset(entry, loader)
    return dataset_cache[entry.dataset_id]


def _load_uncached_dataset(
    entry: DatasetRegistryEntry,
    loader: DatasetLoader | None,
) -> Any:
    dataset_loader = loader or _default_hf_loader()
    return dataset_loader(*entry.hf_args, **(entry.hf_kwargs or {}))


def _select_split_dataset(dataset: Any, split_name: str) -> tuple[Any | None, tuple[str, ...]]:
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        available = tuple(str(name) for name in dataset.keys())
        if split_name in dataset:
            return dataset[split_name], available
        return None, available
    return dataset, ("<single>",)


def _iter_sample_rows(
    dataset_split: Any,
    spec: DatasetFormatSpec,
    filter_spec: Mapping[str, Any] | None,
    max_samples: int,
):
    n_yielded = 0
    for sample_index, row in enumerate(dataset_split):
        if n_yielded >= max_samples:
            break
        if _is_mapping_like(row):
            canonical = canonical_dataset_example(row, spec)
            keep, _ = dataset_example_passes_filter(canonical, filter_spec)
            if not keep:
                continue
        n_yielded += 1
        yield sample_index, row


def _canonical_prompt_example(
    row: Any,
    spec: DatasetFormatSpec,
    prompt_fields: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    canonical = canonical_dataset_example(row, spec)
    missing = tuple(field for field in prompt_fields if field not in canonical)
    return canonical, missing


def _is_mapping_like(value: Any) -> bool:
    return isinstance(value, Mapping) or hasattr(value, "items")


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
        hf_kwargs = row.get("hf_kwargs", {})
        if hf_kwargs is None:
            hf_kwargs = {}
        if not isinstance(hf_kwargs, dict):
            raise ValueError(f"dataset {dataset_id} hf_kwargs must be a mapping")
        filter_spec = row.get("filter")
        if filter_spec is not None and not isinstance(filter_spec, dict):
            raise ValueError(f"dataset {dataset_id} filter must be a mapping")
        entries.append(
            DatasetRegistryEntry(
                dataset_id=str(dataset_id),
                task_family=task_family,
                hf_args=tuple(str(item) for item in hf_args),
                hf_kwargs={str(k): v for k, v in hf_kwargs.items()} if hf_kwargs else None,
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
        dataset = dataset_loader(*entry.hf_args, **(entry.hf_kwargs or {}))
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
        hf_kwargs=entry.hf_kwargs,
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
