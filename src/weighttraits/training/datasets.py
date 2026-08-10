"""Dataset registry and split audits for training execution dry runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import operator
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import yaml

from weighttraits.training.data_formats import DatasetFormatSpec, lookup_field
from weighttraits.training.prompts import render_prompt


DatasetLoader = Callable[..., Any]


_CACHE_METADATA_VERSION = 3
_CACHE_SAMPLE_STRATEGIES = frozenset({"first", "seeded_shuffle", "legacy_subsample"})
_CACHE_PROVENANCE_KEYS = (
    "cache_metadata_version",
    "dataset_id",
    "split",
    "dataset_registry_entry_sha256",
    "dataset_format_spec_sha256",
    "source_split_fingerprint",
    "source_split_num_rows",
    "cache_jsonl_sha256",
    "cache_row_count",
    "cache_limit",
    "cache_n_filtered",
    "cache_n_dropped",
)


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
class DatasetCacheRecipe:
    """Expected sampling recipe for required train and evaluation caches."""

    sample_strategy: str
    sample_seed: int | None
    train_limit: int | None
    eval_limit: int | None

    def __post_init__(self) -> None:
        if self.sample_strategy not in _CACHE_SAMPLE_STRATEGIES:
            supported = ", ".join(sorted(_CACHE_SAMPLE_STRATEGIES))
            raise ValueError(f"unsupported cache sample strategy; expected one of: {supported}")
        if self.sample_strategy == "first":
            if self.sample_seed is not None:
                raise ValueError("cache sample seed must be None when strategy is 'first'")
        elif isinstance(self.sample_seed, bool) or not isinstance(self.sample_seed, int):
            raise ValueError(f"cache sample seed is required for {self.sample_strategy}")
        for field_name in ("train_limit", "eval_limit"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"cache {field_name} must be a positive integer or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataset_registry_entry_fingerprint(entry: DatasetRegistryEntry) -> str:
    """Return a deterministic SHA-256 fingerprint for one registry entry."""

    return _stable_json_sha256(entry.to_dict())


def dataset_format_spec_fingerprint(spec: DatasetFormatSpec) -> str:
    """Return a deterministic SHA-256 fingerprint for one canonical format spec."""

    return _stable_json_sha256(spec.to_dict())


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
    target_field: str | None
    n_seen: int
    n_rendered: int
    n_missing_field_rows: int
    n_empty_render_rows: int
    n_valid_target_rows: int
    n_missing_target_rows: int
    n_non_string_target_rows: int
    n_empty_target_rows: int
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
        if self.status not in {"ok", "cached", "skipped"}:
            return False
        return self.sample_strategy == "legacy_subsample" or self.n_cached >= self.min_required

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
        entries = _entries_from_dataset_rows(
            raw.get("datasets", []), task_family=None, family_status=None
        )
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
        requested = _requested_splits(
            entry, None if format_specs is None else format_specs.get(dataset_id)
        )
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
        job for job in jobs if not selected_ids or getattr(job, "dataset_id", None) in selected_ids
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
        job for job in jobs if not selected_ids or getattr(job, "dataset_id", None) in selected_ids
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
    bounded by ``shuffle_buffer_size``. ``legacy_subsample`` exactly mirrors the old
    non-streaming selection rule: shuffle and select only when the sized source split
    is longer than that split's limit.
    """
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be positive or None")
    if eval_limit is not None and eval_limit < 1:
        raise ValueError("eval_limit must be positive or None")
    if max_scan is not None and max_scan < 1:
        raise ValueError("max_scan must be positive or None")
    if sample_strategy not in _CACHE_SAMPLE_STRATEGIES:
        raise ValueError("sample_strategy must be 'first', 'seeded_shuffle', or 'legacy_subsample'")
    if shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive")
    effective_seed = (
        sample_seed if sample_strategy in {"seeded_shuffle", "legacy_subsample"} else None
    )
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
    registry_entry: DatasetRegistryEntry | None = None,
    format_spec: DatasetFormatSpec | None = None,
    expected_recipe: DatasetCacheRecipe | None = None,
) -> dict[str, Any] | None:
    if (registry_entry is None) != (format_spec is None):
        raise ValueError("registry_entry and format_spec must be provided together")
    if expected_recipe is not None and registry_entry is None:
        raise ValueError(
            "strict cache loading requires registry_entry and format_spec with expected_recipe"
        )
    train_path = dataset_cache_split_path(cache_root, dataset_id, train_split)
    if not train_path.exists():
        if require:
            raise FileNotFoundError(f"missing cached train split: {train_path}")
        return None

    paths = [
        (
            train_split,
            train_path,
            None if expected_recipe is None else expected_recipe.train_limit,
        )
    ]
    if eval_split:
        eval_path = dataset_cache_split_path(cache_root, dataset_id, eval_split)
        if eval_path.exists():
            paths.append(
                (
                    eval_split,
                    eval_path,
                    None if expected_recipe is None else expected_recipe.eval_limit,
                )
            )
        elif require:
            raise FileNotFoundError(f"missing cached eval split: {eval_path}")

    if expected_recipe is None:
        return {split_name: _iter_jsonl_rows(path) for split_name, path, _ in paths}

    splits: dict[str, Any] = {}
    for split_name, path, expected_limit in paths:
        try:
            metadata = _read_cache_metadata(path.with_suffix(".metadata.json"))
            issue = _cache_metadata_issue(
                metadata,
                dataset_id=dataset_id,
                split_name=split_name,
                entry=registry_entry,
                spec=format_spec,
                expected_recipe=expected_recipe,
                expected_limit=expected_limit,
            )
            rows = None if issue else _read_and_validate_cache_rows(path, metadata)
        except ValueError as exc:
            issue = str(exc)
            rows = None
        if issue:
            message = f"incompatible cached dataset split {path}: {issue}"
            if require:
                raise ValueError(message)
            return None
        splits[split_name] = rows
    return splits


def canonical_dataset_example(row: Any, spec: DatasetFormatSpec) -> dict[str, Any]:
    row_map = {str(key): value for key, value in row.items()}
    canonical = dict(row_map)
    for prompt_field, raw_field in (spec.field_map or {}).items():
        found, value = lookup_field(row_map, raw_field)
        if found:
            canonical[prompt_field] = value
    if spec.transforms:
        # Resolve every transform against the same post-field-map snapshot. This lets one
        # specification derive both a display-friendly choices string and the correct answer
        # text from the original choices mapping without transform ordering affecting results.
        source_example = dict(canonical)
        updates = {
            output_field: _apply_canonical_transform(
                output_field,
                transform,
                source_example,
                dataset_id=spec.dataset_id,
            )
            for output_field, transform in spec.transforms.items()
        }
        canonical.update(updates)
    return canonical


def _apply_canonical_transform(
    output_field: str,
    transform: Mapping[str, Any],
    example: Mapping[str, Any],
    *,
    dataset_id: str,
) -> Any:
    op = str(transform.get("op", ""))
    if op == "label_index":
        value = _required_transform_source(
            example,
            transform.get("source"),
            dataset_id=dataset_id,
            output_field=output_field,
        )
        labels = transform.get("labels")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError(_transform_error(dataset_id, output_field, "labels must be a sequence"))
        if isinstance(value, str) and value in labels:
            return value
        if isinstance(value, bool):
            raise TypeError(_transform_error(dataset_id, output_field, "label index is boolean"))
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise TypeError(
                _transform_error(
                    dataset_id, output_field, f"label index is not an integer: {value!r}"
                )
            ) from exc
        if index < 0 or index >= len(labels):
            raise IndexError(
                _transform_error(
                    dataset_id,
                    output_field,
                    f"label index {index} is outside configured labels of length {len(labels)}",
                )
            )
        return str(labels[index])

    if op == "first_item":
        source = transform.get("source")
        found, value = _transform_source(example, source)
        if not found and output_field in example and str(source).startswith(f"{output_field}."):
            return example[output_field]
        if not found:
            raise KeyError(
                _transform_error(
                    dataset_id,
                    output_field,
                    f"transform source field is missing: {source}",
                )
            )
        if isinstance(value, str) and source == output_field:
            return value
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(
                _transform_error(dataset_id, output_field, "first_item source must be a sequence")
            )
        return value[0] if value else transform.get("default", "")

    if op == "bool_map":
        value = _required_transform_source(
            example,
            transform.get("source"),
            dataset_id=dataset_id,
            output_field=output_field,
        )
        if value in (transform["true_value"], transform["false_value"]):
            return value
        if not isinstance(value, bool):
            raise TypeError(
                _transform_error(
                    dataset_id, output_field, f"bool_map source is not boolean: {value!r}"
                )
            )
        return transform["true_value"] if value else transform["false_value"]

    if op == "format_choices":
        choices = _required_transform_source(
            example,
            transform.get("source"),
            dataset_id=dataset_id,
            output_field=output_field,
        )
        if isinstance(choices, str):
            return choices
        labels, texts = _choice_parts(
            choices,
            transform,
            dataset_id=dataset_id,
            output_field=output_field,
        )
        separator = str(transform.get("separator", " "))
        return separator.join(f"{label}) {text}" for label, text in zip(labels, texts))

    if op == "choice_text":
        answer_key = _required_transform_source(
            example,
            transform.get("key_source"),
            dataset_id=dataset_id,
            output_field=output_field,
        )
        choices = _required_transform_source(
            example,
            transform.get("choices_source"),
            dataset_id=dataset_id,
            output_field=output_field,
        )
        if isinstance(choices, str) and output_field in example:
            return example[output_field]
        labels, texts = _choice_parts(
            choices,
            transform,
            dataset_id=dataset_id,
            output_field=output_field,
        )
        for label, text in zip(labels, texts):
            if label == answer_key or str(label) == str(answer_key):
                return str(text)
        if transform.get("fallback_to_key", False):
            return str(answer_key)
        raise ValueError(
            _transform_error(
                dataset_id,
                output_field,
                f"answer key {answer_key!r} is absent from choice labels",
            )
        )

    if op == "compose_qa_input":
        existing = example.get(output_field)
        if isinstance(existing, str):
            return existing
        question = _required_transform_source(
            example,
            transform.get("question_source"),
            dataset_id=dataset_id,
            output_field=output_field,
        )
        qa_input = f"question: {question}"
        if "choices_source" in transform:
            choices = _required_transform_source(
                example,
                transform["choices_source"],
                dataset_id=dataset_id,
                output_field=output_field,
            )
            if isinstance(choices, str):
                formatted_choices = choices
            else:
                labels, texts = _choice_parts(
                    choices,
                    transform,
                    dataset_id=dataset_id,
                    output_field=output_field,
                )
                separator = str(transform.get("separator", " "))
                formatted_choices = separator.join(
                    f"{label}) {text}" for label, text in zip(labels, texts)
                )
            qa_input += f" choices: {formatted_choices}"
        if "context_source" in transform:
            found, context = _transform_source(example, transform["context_source"])
            if found and context:
                qa_input += f" context: {context}"
        return qa_input

    raise ValueError(
        _transform_error(dataset_id, output_field, f"unsupported transform op: {op!r}")
    )


def _required_transform_source(
    example: Mapping[str, Any],
    source: Any,
    *,
    dataset_id: str,
    output_field: str,
) -> Any:
    if not isinstance(source, str) or not source:
        raise ValueError(
            _transform_error(dataset_id, output_field, f"invalid transform source: {source!r}")
        )
    found, value = _transform_source(example, source)
    if not found:
        raise KeyError(
            _transform_error(
                dataset_id, output_field, f"transform source field is missing: {source}"
            )
        )
    return value


def _transform_source(example: Mapping[str, Any], source: Any) -> tuple[bool, Any]:
    if not isinstance(source, str) or not source:
        return False, None
    return lookup_field(example, source)


def _choice_parts(
    choices: Any,
    transform: Mapping[str, Any],
    *,
    dataset_id: str,
    output_field: str,
) -> tuple[Sequence[Any], Sequence[Any]]:
    if not isinstance(choices, Mapping):
        raise TypeError(
            _transform_error(dataset_id, output_field, "choices source must be a mapping")
        )
    label_field = str(transform.get("label_field", "label"))
    text_field = str(transform.get("text_field", "text"))
    labels = choices.get(label_field)
    texts = choices.get(text_field)
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or not isinstance(texts, Sequence)
        or isinstance(texts, (str, bytes))
    ):
        raise TypeError(
            _transform_error(
                dataset_id,
                output_field,
                f"choices fields {label_field!r} and {text_field!r} must be sequences",
            )
        )
    if len(labels) != len(texts):
        raise ValueError(
            _transform_error(dataset_id, output_field, "choice label/text lengths do not match")
        )
    return labels, texts


def _transform_error(dataset_id: str, output_field: str, detail: str) -> str:
    return f"dataset {dataset_id} transform for {output_field!r}: {detail}"


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
    requested_metadata = _cache_metadata(
        entry=entry,
        spec=spec,
        split_name=split_name,
        source_split=dataset_split,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        shuffle_buffer_size=shuffle_buffer_size,
        cache_limit=limit,
    )
    if path.exists() and not overwrite:
        n_cached = _count_jsonl_rows(path)
        try:
            metadata = _read_cache_metadata(metadata_path)
        except ValueError as exc:
            return _cache_provenance_mismatch_report(
                entry=entry,
                split_name=split_name,
                path=path,
                limit=limit,
                min_required=min_required,
                n_cached=n_cached,
                metadata=None,
                error=str(exc),
            )
        provenance_issue = _cache_metadata_issue(
            metadata,
            dataset_id=entry.dataset_id,
            split_name=split_name,
            entry=entry,
            spec=spec,
            source_split=dataset_split,
            cache_path=path,
        )
        if provenance_issue:
            return _cache_provenance_mismatch_report(
                entry=entry,
                split_name=split_name,
                path=path,
                limit=limit,
                min_required=min_required,
                n_cached=n_cached,
                metadata=metadata,
                error=provenance_issue,
            )
        requested = _cache_sampling_metadata(
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            shuffle_buffer_size=shuffle_buffer_size,
            cache_limit=limit,
        )
        existing = metadata or {}
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
        cache_is_sufficient = sample_strategy == "legacy_subsample" or n_cached >= min_required
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
            status="cached" if cache_is_sufficient else "short",
            error=None if cache_is_sufficient else "existing cache is shorter than required",
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )

    n_scanned = 0
    n_cached = 0
    n_filtered = 0
    n_dropped = 0
    sampled_rows = _sample_dataset_rows(
        dataset_split,
        strategy=sample_strategy,
        seed=sample_seed,
        buffer_size=shuffle_buffer_size,
        limit=limit,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
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
                missing = [
                    field for field in _cache_required_fields(spec) if field not in canonical
                ]
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
    requested_metadata.update(_cache_content_metadata(path, row_count=n_cached))
    requested_metadata.update(
        {
            "cache_n_filtered": n_filtered,
            "cache_n_dropped": n_dropped,
        }
    )
    exactness_issue = _legacy_cache_exactness_issue(requested_metadata)
    metadata_path.write_text(
        json.dumps(
            requested_metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if exactness_issue:
        status = "exactness_mismatch"
    elif sample_strategy == "legacy_subsample":
        status = "ok"
    elif n_cached >= min_required:
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
        error=(None if status == "ok" else exactness_issue or "not enough rows after filtering"),
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
    limit: int | None,
):
    if strategy == "first":
        return rows
    if strategy == "legacy_subsample":
        if seed is None:
            raise ValueError("legacy_subsample requires a seed")
        try:
            source_size = len(rows)
        except (TypeError, AttributeError) as exc:
            raise ValueError(
                "legacy_subsample requires a sized, non-streaming dataset split"
            ) from exc
        if limit is None or source_size <= limit:
            return rows
        shuffle = getattr(rows, "shuffle", None)
        select = getattr(rows, "select", None)
        if not callable(shuffle) or not callable(select):
            raise ValueError(
                "legacy_subsample requires a non-streaming dataset with shuffle() and select()"
            )
        shuffled = shuffle(seed=seed)
        shuffled_select = getattr(shuffled, "select", None)
        if not callable(shuffled_select):
            raise ValueError("legacy_subsample shuffle() result does not support select()")
        return shuffled_select(range(limit))
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
    cache_limit: int | None,
) -> dict[str, Any]:
    return {
        "sample_strategy": sample_strategy,
        "sample_seed": sample_seed,
        "shuffle_buffer_size": shuffle_buffer_size,
        "cache_limit": cache_limit,
    }


def _cache_metadata(
    *,
    entry: DatasetRegistryEntry,
    spec: DatasetFormatSpec,
    split_name: str,
    source_split: Any,
    sample_strategy: str,
    sample_seed: int | None,
    shuffle_buffer_size: int | None,
    cache_limit: int | None,
) -> dict[str, Any]:
    metadata = _cache_sampling_metadata(
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        shuffle_buffer_size=shuffle_buffer_size,
        cache_limit=cache_limit,
    )
    metadata.update(_cache_provenance_metadata(entry, spec, split_name))
    metadata.update(_source_split_metadata(source_split))
    return metadata


def _cache_provenance_metadata(
    entry: DatasetRegistryEntry,
    spec: DatasetFormatSpec,
    split_name: str,
) -> dict[str, Any]:
    if entry.dataset_id != spec.dataset_id:
        raise ValueError(
            "cache provenance requires matching registry and format dataset ids: "
            f"{entry.dataset_id!r} != {spec.dataset_id!r}"
        )
    return {
        "cache_metadata_version": _CACHE_METADATA_VERSION,
        "dataset_id": entry.dataset_id,
        "split": str(split_name),
        "dataset_registry_entry_sha256": dataset_registry_entry_fingerprint(entry),
        "dataset_format_spec_sha256": dataset_format_spec_fingerprint(spec),
    }


def _cache_metadata_issue(
    metadata: Mapping[str, Any] | None,
    *,
    dataset_id: str,
    split_name: str,
    entry: DatasetRegistryEntry | None,
    spec: DatasetFormatSpec | None,
    source_split: Any | None = None,
    cache_path: Path | None = None,
    expected_recipe: DatasetCacheRecipe | None = None,
    expected_limit: int | None = None,
) -> str | None:
    if metadata is None:
        return "cache metadata is missing, so provenance cannot be verified"
    missing = [key for key in _CACHE_PROVENANCE_KEYS if key not in metadata]
    if missing:
        return "cache metadata lacks provenance fields: " + ", ".join(missing)
    if metadata.get("cache_metadata_version") != _CACHE_METADATA_VERSION:
        return (
            "cache metadata version differs: "
            f"{metadata.get('cache_metadata_version')!r} != {_CACHE_METADATA_VERSION}"
        )
    if metadata.get("dataset_id") != dataset_id:
        return f"cache dataset id differs: {metadata.get('dataset_id')!r} != {dataset_id!r}"
    if metadata.get("split") != split_name:
        return f"cache split differs: {metadata.get('split')!r} != {split_name!r}"
    exactness_issue = _legacy_cache_exactness_issue(metadata)
    if exactness_issue:
        return exactness_issue
    if expected_recipe is not None:
        expected_sampling = {
            "sample_strategy": expected_recipe.sample_strategy,
            "sample_seed": expected_recipe.sample_seed,
            "cache_limit": expected_limit,
        }
        mismatched_recipe = [
            key for key, value in expected_sampling.items() if metadata.get(key) != value
        ]
        if mismatched_recipe:
            return "cache sampling recipe differs: " + ", ".join(mismatched_recipe)
    if cache_path is not None:
        content_issue = _cache_content_issue(metadata, cache_path)
        if content_issue:
            return content_issue
    if source_split is not None:
        expected_source = _source_split_metadata(source_split)
        mismatched_source = [
            key
            for key, value in expected_source.items()
            if value is not None and metadata.get(key) != value
        ]
        if mismatched_source:
            return "cache source split provenance differs: " + ", ".join(mismatched_source)
    if entry is None or spec is None:
        return None
    if entry.dataset_id != dataset_id or spec.dataset_id != dataset_id:
        return (
            "requested registry/format dataset id differs from cache path: "
            f"entry={entry.dataset_id!r}, spec={spec.dataset_id!r}, path={dataset_id!r}"
        )
    expected = _cache_provenance_metadata(entry, spec, split_name)
    mismatched = [
        key
        for key in (
            "dataset_registry_entry_sha256",
            "dataset_format_spec_sha256",
        )
        if metadata.get(key) != expected[key]
    ]
    if mismatched:
        return "cache provenance fingerprint differs: " + ", ".join(mismatched)
    return None


def _legacy_cache_exactness_issue(metadata: Mapping[str, Any]) -> str | None:
    if metadata.get("sample_strategy") != "legacy_subsample":
        return None
    n_filtered = metadata.get("cache_n_filtered")
    n_dropped = metadata.get("cache_n_dropped")
    if n_filtered != 0 or n_dropped != 0:
        return (
            "legacy cache is not exact: filtered/dropped rows must both be zero "
            f"(filtered={n_filtered!r}, dropped={n_dropped!r})"
        )
    source_size = metadata.get("source_split_num_rows")
    if isinstance(source_size, bool) or not isinstance(source_size, int) or source_size < 0:
        return "legacy cache is not exact: source split row count is unavailable"
    cache_limit = metadata.get("cache_limit")
    if cache_limit is not None and (
        isinstance(cache_limit, bool) or not isinstance(cache_limit, int) or cache_limit < 1
    ):
        return "legacy cache is not exact: cache_limit is invalid"
    expected_count = source_size if cache_limit is None else min(source_size, cache_limit)
    if metadata.get("cache_row_count") != expected_count:
        return (
            "legacy cache is not exact: cache_row_count differs from "
            f"min(cache_limit, source_split_num_rows): "
            f"{metadata.get('cache_row_count')!r} != {expected_count!r}"
        )
    return None


def _cache_provenance_mismatch_report(
    *,
    entry: DatasetRegistryEntry,
    split_name: str,
    path: Path,
    limit: int | None,
    min_required: int,
    n_cached: int,
    metadata: Mapping[str, Any] | None,
    error: str,
) -> DatasetCacheSplitReport:
    existing = metadata or {}
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
        status="provenance_mismatch",
        error=f"existing cache provenance is incompatible: {error}; rerun with --overwrite",
        sample_strategy=str(existing.get("sample_strategy", "unknown")),
        sample_seed=existing.get("sample_seed"),
        shuffle_buffer_size=existing.get("shuffle_buffer_size"),
    )


def _source_split_metadata(dataset_split: Any) -> dict[str, Any]:
    fingerprint = getattr(dataset_split, "_fingerprint", None)
    if fingerprint is not None:
        fingerprint = str(fingerprint)
    try:
        row_count = len(dataset_split)
    except (TypeError, AttributeError):
        row_count = None
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        row_count = None
    return {
        "source_split_fingerprint": fingerprint,
        "source_split_num_rows": row_count,
    }


def _cache_content_metadata(path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        "cache_jsonl_sha256": _file_sha256(path),
        "cache_row_count": row_count,
    }


def _cache_content_issue(metadata: Mapping[str, Any], path: Path) -> str | None:
    expected_count = metadata.get("cache_row_count")
    actual_count, actual_sha256 = _cache_file_stats(path)
    if expected_count != actual_count:
        return f"cache row count differs: {actual_count!r} != {expected_count!r}"
    expected_sha256 = metadata.get("cache_jsonl_sha256")
    if expected_sha256 != actual_sha256:
        return "cache JSONL fingerprint differs: cache_jsonl_sha256"
    return None


def _read_and_validate_cache_rows(path: Path, metadata: Mapping[str, Any]) -> list[Any]:
    digest = hashlib.sha256()
    rows: list[Any] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid cached JSONL {path} at line {line_number}") from exc
    if metadata.get("cache_row_count") != len(rows):
        expected_count = metadata.get("cache_row_count")
        raise ValueError(f"cache row count differs: {len(rows)!r} != {expected_count!r}")
    if metadata.get("cache_jsonl_sha256") != digest.hexdigest():
        raise ValueError("cache JSONL fingerprint differs: cache_jsonl_sha256")
    return rows


def _cache_file_stats(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    row_count = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if raw_line.strip():
                row_count += 1
    return row_count, digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_cache_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cache metadata {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"invalid cache metadata {path}: expected a JSON object")
    return raw


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
    fields.update((spec.transforms or {}).keys())
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
    target_field = _configured_target_field(job)

    if not dataset_id:
        return _sample_render_audit(
            node_id=node_id,
            dataset_id=None,
            task_family=task_family,
            prompt_source=prompt_source,
            prompt_fields=prompt_fields,
            target_field=target_field,
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
            target_field=target_field,
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
            target_field=target_field,
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
            target_field=target_field,
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
            target_field=target_field,
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
    n_valid_target_rows = 0
    n_missing_target_rows = 0
    n_non_string_target_rows = 0
    n_empty_target_rows = 0
    issues: list[SampleRenderIssue] = []
    for sample_index, row in _iter_sample_rows(split_dataset, spec, entry.filter, max_samples):
        n_seen += 1
        if not _is_mapping_like(row):
            issues.append(SampleRenderIssue(sample_index=sample_index, issue="non_mapping_row"))
            continue
        canonical, missing = _canonical_prompt_example(row, spec, prompt_fields)
        if target_field is not None:
            if target_field not in canonical:
                n_missing_target_rows += 1
                issues.append(
                    SampleRenderIssue(
                        sample_index=sample_index,
                        issue="missing_target_field",
                        missing_fields=(target_field,),
                    )
                )
            elif not isinstance(canonical[target_field], str):
                n_non_string_target_rows += 1
                issues.append(
                    SampleRenderIssue(
                        sample_index=sample_index,
                        issue="non_string_target",
                        error=(
                            "expected a canonical string target; got "
                            f"{type(canonical[target_field]).__name__}"
                        ),
                    )
                )
            elif not canonical[target_field].strip():
                n_empty_target_rows += 1
                issues.append(SampleRenderIssue(sample_index=sample_index, issue="empty_target"))
            else:
                n_valid_target_rows += 1
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
        target_field=target_field,
        n_seen=n_seen,
        n_rendered=n_rendered,
        n_missing_field_rows=n_missing_field_rows,
        n_empty_render_rows=n_empty_render_rows,
        n_valid_target_rows=n_valid_target_rows,
        n_missing_target_rows=n_missing_target_rows,
        n_non_string_target_rows=n_non_string_target_rows,
        n_empty_target_rows=n_empty_target_rows,
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
    target_field: str | None = None,
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
        target_field=target_field,
        n_seen=0,
        n_rendered=0,
        n_missing_field_rows=0,
        n_empty_render_rows=0,
        n_valid_target_rows=0,
        n_missing_target_rows=0,
        n_non_string_target_rows=0,
        n_empty_target_rows=0,
        status=status,
        issues=issues,
        error=error,
    )


def _configured_target_field(job: Any) -> str | None:
    trainer = getattr(job, "trainer", None)
    if not isinstance(trainer, Mapping):
        return None
    target_field = trainer.get("target_field") or trainer.get("label_field")
    if target_field is None:
        return None
    normalized = str(target_field).strip()
    return normalized or None


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
