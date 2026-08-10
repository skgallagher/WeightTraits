import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from weighttraits.cli import build_parser
from weighttraits.training.data_formats import DatasetFormatSpec, load_dataset_format_specs
from weighttraits.training.datasets import (
    audit_dataset_registry,
    audit_training_sample_rendering,
    cache_training_datasets,
    canonical_dataset_example,
    dataset_cache_split_path,
    dataset_format_spec_fingerprint,
    dataset_registry_entry_fingerprint,
    DatasetCacheRecipe,
    DatasetRegistryEntry,
    load_cached_dataset_splits,
    load_dataset_registry,
    select_training_sample_jobs,
    write_training_sample_render_audit_report,
)


def _registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
version: 1
task_families:
  qa_reasoning:
    status: inherited
    datasets:
      - id: boolq
        hf_args: [google/boolq]
      - id: ai2_arc_easy
        hf_args: [allenai/ai2_arc, ARC-Easy]
        role: eval_or_behavior_probe
        note: small audit fixture
"""
    )
    return path


def _formats_path(tmp_path: Path) -> Path:
    path = tmp_path / "formats.yaml"
    path.write_text(
        """
datasets:
  - dataset_id: boolq
    task_family: qa_reasoning
    train_split: train
    eval_split: validation
    prompt_fields: [question, passage, answer]
  - dataset_id: ai2_arc_easy
    task_family: qa_reasoning
    train_split: train
    eval_split: test
    prompt_fields: [question, choices, answer]
"""
    )
    return path


def test_load_dataset_registry_from_task_families(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))

    assert sorted(registry) == ["ai2_arc_easy", "boolq"]
    assert registry["boolq"].task_family == "qa_reasoning"
    assert registry["boolq"].hf_args == ("google/boolq",)
    assert registry["ai2_arc_easy"].hf_args == ("allenai/ai2_arc", "ARC-Easy")
    assert registry["ai2_arc_easy"].role == "eval_or_behavior_probe"


def test_dataset_registry_supports_hf_kwargs_for_local_json(tmp_path):
    registry_path = tmp_path / "registry_kwargs.yaml"
    registry_path.write_text(
        """
datasets:
  - id: tiny_json
    hf_args: [json]
    hf_kwargs:
      data_files:
        train: examples/training/tiny_train.jsonl
        validation: examples/training/tiny_validation.jsonl
    train_split: train
    eval_split: validation
"""
    )
    registry = load_dataset_registry(registry_path)

    def fake_loader(*args, **kwargs):
        assert args == ("json",)
        assert kwargs == {
            "data_files": {
                "train": "examples/training/tiny_train.jsonl",
                "validation": "examples/training/tiny_validation.jsonl",
            }
        }
        return {"train": [1], "validation": [2]}

    report = audit_dataset_registry(registry, loader=fake_loader)
    audit = report.audits[0]

    assert registry["tiny_json"].hf_kwargs == {
        "data_files": {
            "train": "examples/training/tiny_train.jsonl",
            "validation": "examples/training/tiny_validation.jsonl",
        }
    }
    assert report.valid
    assert audit.hf_kwargs == registry["tiny_json"].hf_kwargs
    assert audit.row_counts == {"train": 1, "validation": 1}


def test_audit_dataset_registry_no_load_records_requested_splits(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_formats_path(tmp_path))

    report = audit_dataset_registry(
        registry,
        dataset_ids=["boolq"],
        format_specs=formats,
        load=False,
    )
    audit = report.audits[0]

    assert report.valid
    assert audit.status == "not_loaded"
    assert audit.requested_splits == ("train", "validation")
    assert audit.available_splits == ()


def test_audit_dataset_registry_with_loader_counts_rows(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_formats_path(tmp_path))

    def fake_loader(*args):
        assert args == ("google/boolq",)
        return {"train": [1, 2, 3], "validation": [4]}

    report = audit_dataset_registry(
        registry,
        dataset_ids=["boolq"],
        format_specs=formats,
        loader=fake_loader,
    )
    audit = report.audits[0]

    assert report.valid
    assert audit.status == "ok"
    assert audit.available_splits == ("train", "validation")
    assert audit.row_counts == {"train": 3, "validation": 1}


def test_audit_dataset_registry_reports_missing_requested_splits(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_formats_path(tmp_path))

    report = audit_dataset_registry(
        registry,
        dataset_ids=["boolq"],
        format_specs=formats,
        loader=lambda *args: {"train": [1]},
    )
    audit = report.audits[0]

    assert not report.valid
    assert audit.status == "missing_splits"
    assert audit.error == "missing requested splits: validation"


def test_audit_dataset_registry_reports_unknown_dataset_id(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))

    report = audit_dataset_registry(registry, dataset_ids=["missing"], load=False)

    assert not report.valid
    assert report.audits[0].status == "unknown_dataset_id"


def test_audit_dataset_registry_reports_loader_failure(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))

    def failing_loader(*args):
        raise RuntimeError("offline")

    report = audit_dataset_registry(registry, dataset_ids=["boolq"], loader=failing_loader)

    assert not report.valid
    assert report.audits[0].status == "load_failed"
    assert report.audits[0].error == "offline"


def _sample_render_formats_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample_formats.yaml"
    path.write_text(
        """
datasets:
  - dataset_id: boolq
    task_family: qa_reasoning
    train_split: train
    field_map:
      question: question
      context: passage
      answer: answer
"""
    )
    return path


def _sample_render_job():
    return SimpleNamespace(
        node_id="n0",
        dataset_id="boolq",
        task_family="qa_reasoning",
        prompt_source="task:qa_reasoning",
        prompt_fields=("question", "context", "answer"),
        prompt_template="Question: {question}\nContext: {context}\nAnswer: {answer}",
    )


def test_audit_training_sample_rendering_applies_field_map(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))

    def fake_loader(*args):
        assert args == ("google/boolq",)
        return {
            "train": [
                {"question": "Q1", "passage": "P1", "answer": True},
                {"question": "Q2", "passage": "P2", "answer": False},
            ]
        }

    report = audit_training_sample_rendering(
        [_sample_render_job()],
        registry,
        formats,
        max_samples=2,
        loader=fake_loader,
    )
    audit = report.audits[0]

    assert report.valid
    assert audit.status == "ok"
    assert audit.split == "train"
    assert audit.n_seen == 2
    assert audit.n_rendered == 2
    assert audit.issues == ()


def test_audit_training_sample_rendering_reports_row_issues(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    job = SimpleNamespace(
        node_id="n0",
        dataset_id="boolq",
        task_family="qa_reasoning",
        prompt_source="custom",
        prompt_fields=("context",),
        prompt_template="{context}",
    )

    report = audit_training_sample_rendering(
        [job],
        registry,
        formats,
        loader=lambda *args: {"train": [{"question": "Q"}, {"question": "Q", "passage": ""}]},
    )
    audit = report.audits[0]

    assert not report.valid
    assert audit.status == "row_issues"
    assert audit.n_seen == 2
    assert audit.n_missing_field_rows == 1
    assert audit.n_empty_render_rows == 1
    assert [issue.issue for issue in audit.issues] == [
        "missing_prompt_fields",
        "empty_rendered_prompt",
    ]
    assert audit.issues[0].missing_fields == ("context",)


def test_audit_training_sample_rendering_reports_missing_split(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))

    report = audit_training_sample_rendering(
        [_sample_render_job()],
        registry,
        formats,
        loader=lambda *args: {"validation": [{"question": "Q", "passage": "P", "answer": True}]},
    )
    audit = report.audits[0]

    assert not report.valid
    assert audit.status == "missing_split"
    assert audit.error == "missing split: train"


def test_audit_training_sample_rendering_validates_configured_target_after_transforms(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    job = SimpleNamespace(
        **vars(_sample_render_job()),
        trainer={"target_field": "answer"},
    )
    raw_report = audit_training_sample_rendering(
        [job],
        registry,
        formats,
        max_samples=5,
        loader=lambda *args: {
            "train": [
                {"question": "Q1", "passage": "P1", "answer": True},
                {"question": "Q2", "passage": "P2", "answer": 1},
                {"question": "Q3", "passage": "P3", "answer": {"private": "value"}},
                {"question": "Q4", "passage": "P4", "answer": ""},
                {"question": "Q5", "passage": "P5"},
            ]
        },
    )
    raw_audit = raw_report.audits[0]

    assert not raw_report.valid
    assert raw_audit.status == "row_issues"
    assert raw_audit.target_field == "answer"
    assert raw_audit.n_valid_target_rows == 0
    assert raw_audit.n_non_string_target_rows == 3
    assert raw_audit.n_empty_target_rows == 1
    assert raw_audit.n_missing_target_rows == 1
    assert {issue.issue for issue in raw_audit.issues} >= {
        "non_string_target",
        "empty_target",
        "missing_target_field",
    }
    assert "private" not in json.dumps(raw_report.to_dict())

    transformed_spec = replace(
        formats["boolq"],
        transforms={
            "answer": {
                "op": "bool_map",
                "source": "answer",
                "true_value": "yes",
                "false_value": "no",
            }
        },
    )
    transformed_report = audit_training_sample_rendering(
        [job],
        registry,
        {"boolq": transformed_spec},
        max_samples=2,
        loader=lambda *args: {
            "train": [
                {"question": "Q1", "passage": "P1", "answer": True},
                {"question": "Q2", "passage": "P2", "answer": False},
            ]
        },
    )
    transformed_audit = transformed_report.audits[0]

    assert transformed_report.valid
    assert transformed_audit.n_valid_target_rows == 2
    assert transformed_audit.n_non_string_target_rows == 0
    assert transformed_audit.issues == ()


def test_choice_text_transform_has_validated_opt_in_answer_key_fallback(tmp_path):
    base_spec = DatasetFormatSpec(
        dataset_id="arc",
        task_family="qa_reasoning",
        prompt_fields=("answer", "choices", "question"),
        field_map={"answer": "answerKey", "choices": "choices", "question": "question"},
        transforms={
            "answer": {
                "op": "choice_text",
                "key_source": "answer",
                "choices_source": "choices",
            }
        },
    )
    row = {
        "question": "Q",
        "answerKey": "Z",
        "choices": {"label": ["A", "B"], "text": ["one", "two"]},
    }

    with pytest.raises(ValueError, match="answer key 'Z' is absent"):
        canonical_dataset_example(row, base_spec)

    fallback_spec = replace(
        base_spec,
        transforms={
            "answer": {
                "op": "choice_text",
                "key_source": "answer",
                "choices_source": "choices",
                "fallback_to_key": True,
            }
        },
    )
    assert canonical_dataset_example(row, fallback_spec)["answer"] == "Z"

    invalid_formats = tmp_path / "invalid_fallback.yaml"
    invalid_formats.write_text(
        """
datasets:
  - dataset_id: arc
    prompt_fields: [answer, choices, question]
    transforms:
      answer:
        op: choice_text
        key_source: answer
        choices_source: choices
        fallback_to_key: "true"
"""
    )
    with pytest.raises(ValueError, match="fallback_to_key.*must be a boolean"):
        load_dataset_format_specs(invalid_formats)


def test_select_training_sample_jobs_can_cover_one_per_dataset():
    jobs = [
        SimpleNamespace(node_id="a", dataset_id="boolq"),
        SimpleNamespace(node_id="b", dataset_id="boolq"),
        SimpleNamespace(node_id="c", dataset_id="squad"),
        SimpleNamespace(node_id="d", dataset_id="squad"),
    ]

    selected = select_training_sample_jobs(jobs, selection="one-per-dataset")

    assert [job.node_id for job in selected] == ["a", "c"]
    assert select_training_sample_jobs(
        jobs,
        dataset_ids=["squad"],
        selection="one-per-dataset",
    ) == [jobs[2]]


def test_cache_training_datasets_filters_after_field_map_and_stops_at_limit(tmp_path):
    registry = {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            filter={"max_chars": {"context": 2}},
            train_split="train",
            eval_split="validation",
        )
    }
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))

    def fake_loader(*args):
        assert args == ("google/boolq",)
        return {
            "train": [
                {"question": "Q long", "passage": "long", "answer": True},
                {"question": "Q1", "passage": "P1", "answer": True},
                {"question": "Q2", "passage": "P2", "answer": False},
                {"question": "Q3", "passage": "P3", "answer": False},
            ],
            "validation": [{"question": "QV", "passage": "PV", "answer": True}],
        }

    report = cache_training_datasets(
        registry,
        formats,
        out_dir=tmp_path / "cache",
        train_limit=2,
        eval_limit=1,
        min_train_rows=2,
        loader=fake_loader,
    )
    train_report = next(split for split in report.splits if split.split == "train")
    cached = load_cached_dataset_splits(
        tmp_path / "cache",
        dataset_id="boolq",
        train_split="train",
        eval_split="validation",
        require=True,
    )

    assert report.valid
    assert train_report.n_scanned == 3
    assert train_report.n_filtered == 1
    assert train_report.n_cached == 2
    assert dataset_cache_split_path(tmp_path / "cache", "boolq", "train").exists()
    assert [row["question"] for row in cached["train"]] == ["Q1", "Q2"]


def test_cache_training_datasets_supports_seeded_shuffle(tmp_path):
    registry = {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            train_split="train",
        )
    }
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    rows = [
        {"question": f"Q{index}", "passage": f"P{index}", "answer": True} for index in range(20)
    ]

    def cached_questions(cache_name, seed):
        report = cache_training_datasets(
            registry,
            formats,
            out_dir=tmp_path / cache_name,
            train_limit=5,
            min_train_rows=5,
            loader=lambda *args: {"train": rows},
            sample_strategy="seeded_shuffle",
            sample_seed=seed,
            shuffle_buffer_size=20,
        )
        cached = load_cached_dataset_splits(
            tmp_path / cache_name,
            dataset_id="boolq",
            train_split="train",
            require=True,
        )
        return report, [row["question"] for row in cached["train"]]

    first_report, first = cached_questions("cache_a", 42)
    second_report, second = cached_questions("cache_b", 42)
    _, different_seed = cached_questions("cache_c", 43)

    assert first == second
    assert first != different_seed
    assert first != [f"Q{index}" for index in range(5)]
    assert first_report.sample_strategy == "seeded_shuffle"
    assert first_report.sample_seed == 42
    assert first_report.splits[0].shuffle_buffer_size == 20
    assert second_report.valid


def test_cache_training_datasets_legacy_subsample_matches_old_sized_split_rule(tmp_path):
    class FakeDataset:
        def __init__(self, rows, calls=None):
            self.rows = list(rows)
            self.calls = [] if calls is None else calls

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return len(self.rows)

        def shuffle(self, *, seed):
            self.calls.append(("shuffle", seed))
            return FakeDataset(reversed(self.rows), self.calls)

        def select(self, indices):
            selected = tuple(indices)
            self.calls.append(("select", selected))
            return FakeDataset((self.rows[index] for index in selected), self.calls)

    registry = {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            train_split="train",
        )
    }
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    source_rows = [
        {"question": f"Q{index}", "passage": f"P{index}", "answer": True} for index in range(4)
    ]
    long_split = FakeDataset(source_rows)
    long_cache = tmp_path / "long_cache"

    long_report = cache_training_datasets(
        registry,
        formats,
        out_dir=long_cache,
        train_limit=2,
        min_train_rows=2,
        loader=lambda *args: {"train": long_split},
        sample_strategy="legacy_subsample",
        sample_seed=17,
    )
    long_rows = load_cached_dataset_splits(
        long_cache,
        dataset_id="boolq",
        train_split="train",
        require=True,
        registry_entry=registry["boolq"],
        format_spec=formats["boolq"],
        expected_recipe=DatasetCacheRecipe("legacy_subsample", 17, 2, None),
    )
    metadata_path = dataset_cache_split_path(long_cache, "boolq", "train").with_suffix(
        ".metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())

    assert long_report.valid
    assert long_split.calls == [("shuffle", 17), ("select", (0, 1))]
    assert [row["question"] for row in long_rows["train"]] == ["Q3", "Q2"]
    assert metadata["sample_strategy"] == "legacy_subsample"
    assert metadata["sample_seed"] == 17
    assert metadata["shuffle_buffer_size"] is None

    short_split = FakeDataset(source_rows[:2])
    short_cache = tmp_path / "short_cache"
    short_report = cache_training_datasets(
        registry,
        formats,
        out_dir=short_cache,
        train_limit=3,
        min_train_rows=3,
        loader=lambda *args: {"train": short_split},
        sample_strategy="legacy_subsample",
        sample_seed=17,
    )
    short_rows = load_cached_dataset_splits(
        short_cache,
        dataset_id="boolq",
        train_split="train",
        require=True,
        registry_entry=registry["boolq"],
        format_spec=formats["boolq"],
        expected_recipe=DatasetCacheRecipe("legacy_subsample", 17, 3, None),
    )

    assert short_report.valid
    assert short_split.calls == []
    assert [row["question"] for row in short_rows["train"]] == ["Q0", "Q1"]


def test_cache_training_datasets_legacy_subsample_rejects_unsized_split(tmp_path):
    registry = {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            train_split="train",
        )
    }
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))

    def unsized_rows():
        yield {"question": "Q", "passage": "P", "answer": True}

    cache_root = tmp_path / "cache"
    with pytest.raises(ValueError, match="sized, non-streaming"):
        cache_training_datasets(
            registry,
            formats,
            out_dir=cache_root,
            train_limit=1,
            min_train_rows=1,
            loader=lambda *args: {"train": unsized_rows()},
            sample_strategy="legacy_subsample",
            sample_seed=17,
        )

    assert not dataset_cache_split_path(cache_root, "boolq", "train").exists()


def test_cache_training_datasets_rejects_sampling_change_without_overwrite(tmp_path):
    registry = {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            train_split="train",
        )
    }
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    rows = [{"question": "Q", "passage": "P", "answer": True}]
    cache_root = tmp_path / "cache"
    cache_training_datasets(
        registry,
        formats,
        out_dir=cache_root,
        train_limit=1,
        min_train_rows=1,
        loader=lambda *args: {"train": rows},
    )

    report = cache_training_datasets(
        registry,
        formats,
        out_dir=cache_root,
        train_limit=1,
        min_train_rows=1,
        loader=lambda *args: {"train": rows},
        sample_strategy="seeded_shuffle",
        sample_seed=42,
    )

    assert not report.valid
    assert report.splits[0].status == "sampling_mismatch"
    assert "--overwrite" in report.splits[0].error


def test_cache_provenance_fingerprints_are_stable_across_mapping_order():
    entry_a = DatasetRegistryEntry(
        dataset_id="arc",
        task_family="qa_reasoning",
        hf_args=("allenai/ai2_arc", "ARC-Easy"),
        hf_kwargs={
            "data_files": {"train": "train.jsonl", "validation": "validation.jsonl"},
            "revision": "main",
        },
        filter={"equals": {"language": "en"}, "max_chars": {"question": 1000}},
        train_split="train",
        eval_split="validation",
    )
    entry_b = replace(
        entry_a,
        hf_kwargs={
            "revision": "main",
            "data_files": {"validation": "validation.jsonl", "train": "train.jsonl"},
        },
        filter={"max_chars": {"question": 1000}, "equals": {"language": "en"}},
    )
    spec_a = DatasetFormatSpec(
        dataset_id="arc",
        task_family="qa_reasoning",
        prompt_fields=("answer", "choices", "question"),
        field_map={"question": "question", "choices": "choices", "answer": "answerKey"},
        transforms={
            "answer": {
                "op": "choice_text",
                "key_source": "answer",
                "choices_source": "choices",
            },
            "choices": {"op": "format_choices", "source": "choices"},
        },
        train_split="train",
        eval_split="validation",
    )
    spec_b = replace(
        spec_a,
        field_map={"answer": "answerKey", "choices": "choices", "question": "question"},
        transforms={
            "choices": {"source": "choices", "op": "format_choices"},
            "answer": {
                "choices_source": "choices",
                "key_source": "answer",
                "op": "choice_text",
            },
        },
    )

    entry_fingerprint = dataset_registry_entry_fingerprint(entry_a)
    spec_fingerprint = dataset_format_spec_fingerprint(spec_a)

    assert entry_fingerprint == dataset_registry_entry_fingerprint(entry_b)
    assert spec_fingerprint == dataset_format_spec_fingerprint(spec_b)
    assert len(entry_fingerprint) == 64
    assert len(spec_fingerprint) == 64


def test_cache_metadata_records_and_validates_dataset_provenance(tmp_path):
    class FingerprintedRows(list):
        pass

    registry = {
        "boolq": DatasetRegistryEntry(
            dataset_id="boolq",
            task_family="qa_reasoning",
            hf_args=("google/boolq",),
            train_split="train",
            eval_split="validation",
        )
    }
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    train_rows = FingerprintedRows([{"question": "Q", "passage": "P", "answer": True}])
    train_rows._fingerprint = "train-source-v1"
    validation_rows = FingerprintedRows([{"question": "QV", "passage": "PV", "answer": False}])
    validation_rows._fingerprint = "validation-source-v1"
    rows = {"train": train_rows, "validation": validation_rows}
    cache_root = tmp_path / "cache"

    report = cache_training_datasets(
        registry,
        formats,
        out_dir=cache_root,
        train_limit=1,
        eval_limit=1,
        min_train_rows=1,
        loader=lambda *args: rows,
    )
    train_path = dataset_cache_split_path(cache_root, "boolq", "train")
    metadata = json.loads(train_path.with_suffix(".metadata.json").read_text())
    cached = load_cached_dataset_splits(
        cache_root,
        dataset_id="boolq",
        train_split="train",
        eval_split="validation",
        require=True,
        registry_entry=registry["boolq"],
        format_spec=formats["boolq"],
        expected_recipe=DatasetCacheRecipe("first", None, 1, 1),
    )
    repeated = cache_training_datasets(
        registry,
        formats,
        out_dir=cache_root,
        train_limit=1,
        eval_limit=1,
        min_train_rows=1,
        loader=lambda *args: rows,
    )
    train_rows._fingerprint = "train-source-v2"
    source_changed = cache_training_datasets(
        registry,
        formats,
        out_dir=cache_root,
        train_limit=1,
        eval_limit=1,
        min_train_rows=1,
        loader=lambda *args: rows,
    )

    assert report.valid
    assert metadata["cache_metadata_version"] == 3
    assert metadata["dataset_id"] == "boolq"
    assert metadata["split"] == "train"
    assert metadata["dataset_registry_entry_sha256"] == dataset_registry_entry_fingerprint(
        registry["boolq"]
    )
    assert metadata["dataset_format_spec_sha256"] == dataset_format_spec_fingerprint(
        formats["boolq"]
    )
    assert metadata["source_split_fingerprint"] == "train-source-v1"
    assert metadata["source_split_num_rows"] == 1
    assert metadata["cache_row_count"] == 1
    assert metadata["cache_limit"] == 1
    assert len(metadata["cache_jsonl_sha256"]) == 64
    assert list(cached) == ["train", "validation"]
    assert all(split.status == "cached" for split in repeated.splits)
    changed_train = next(split for split in source_changed.splits if split.split == "train")
    assert not source_changed.valid
    assert changed_train.status == "provenance_mismatch"
    assert "source_split_fingerprint" in changed_train.error


def test_cache_rejects_registry_or_format_fingerprint_mismatch(tmp_path):
    entry = DatasetRegistryEntry(
        dataset_id="boolq",
        task_family="qa_reasoning",
        hf_args=("google/boolq",),
        train_split="train",
    )
    registry = {"boolq": entry}
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    spec = formats["boolq"]
    rows = [{"question": "Q", "passage": "P", "answer": True}]
    cache_root = tmp_path / "cache"
    cache_training_datasets(
        registry,
        formats,
        out_dir=cache_root,
        train_limit=1,
        min_train_rows=1,
        loader=lambda *args: {"train": rows},
    )

    changed_spec = replace(spec, prompt_fields=("answer", "context", "question", "schema"))
    format_report = cache_training_datasets(
        registry,
        {"boolq": changed_spec},
        out_dir=cache_root,
        train_limit=1,
        min_train_rows=1,
        loader=lambda *args: {"train": rows},
    )
    changed_entry = replace(entry, filter={"max_chars": {"context": 100}})
    registry_report = cache_training_datasets(
        {"boolq": changed_entry},
        formats,
        out_dir=cache_root,
        train_limit=1,
        min_train_rows=1,
        loader=lambda *args: {"train": rows},
    )

    assert not format_report.valid
    assert format_report.splits[0].status == "provenance_mismatch"
    assert "dataset_format_spec_sha256" in format_report.splits[0].error
    assert "--overwrite" in format_report.splits[0].error
    assert not registry_report.valid
    assert registry_report.splits[0].status == "provenance_mismatch"
    assert "dataset_registry_entry_sha256" in registry_report.splits[0].error

    with pytest.raises(ValueError, match="dataset_format_spec_sha256"):
        load_cached_dataset_splits(
            cache_root,
            dataset_id="boolq",
            train_split="train",
            require=True,
            registry_entry=entry,
            format_spec=changed_spec,
            expected_recipe=DatasetCacheRecipe("first", None, 1, None),
        )
    assert (
        load_cached_dataset_splits(
            cache_root,
            dataset_id="boolq",
            train_split="train",
            registry_entry=entry,
            format_spec=changed_spec,
            expected_recipe=DatasetCacheRecipe("first", None, 1, None),
        )
        is None
    )


def test_required_cache_recipe_rejects_wrong_strategy_seed_or_split_limit(tmp_path):
    entry = DatasetRegistryEntry(
        dataset_id="boolq",
        task_family="qa_reasoning",
        hf_args=("google/boolq",),
        train_split="train",
        eval_split="validation",
    )
    registry = {"boolq": entry}
    spec = load_dataset_format_specs(_sample_render_formats_path(tmp_path))["boolq"]
    cache_root = tmp_path / "cache"
    report = cache_training_datasets(
        registry,
        {"boolq": spec},
        out_dir=cache_root,
        train_limit=2,
        eval_limit=1,
        min_train_rows=2,
        loader=lambda *args: {
            "train": [
                {"question": "Q1", "passage": "P1", "answer": True},
                {"question": "Q2", "passage": "P2", "answer": False},
            ],
            "validation": [{"question": "QV", "passage": "PV", "answer": True}],
        },
        sample_strategy="legacy_subsample",
        sample_seed=42,
    )
    assert report.valid

    correct = DatasetCacheRecipe("legacy_subsample", 42, 2, 1)
    loaded = load_cached_dataset_splits(
        cache_root,
        dataset_id="boolq",
        train_split="train",
        eval_split="validation",
        require=True,
        registry_entry=entry,
        format_spec=spec,
        expected_recipe=correct,
    )
    assert len(loaded["train"]) == 2
    assert len(loaded["validation"]) == 1

    wrong_recipes = [
        ("sample_strategy", DatasetCacheRecipe("first", None, 2, 1)),
        ("sample_seed", DatasetCacheRecipe("legacy_subsample", 43, 2, 1)),
        ("cache_limit", DatasetCacheRecipe("legacy_subsample", 42, 3, 1)),
        ("cache_limit", DatasetCacheRecipe("legacy_subsample", 42, 2, 2)),
    ]
    for mismatch_field, recipe in wrong_recipes:
        with pytest.raises(ValueError, match=mismatch_field):
            load_cached_dataset_splits(
                cache_root,
                dataset_id="boolq",
                train_split="train",
                eval_split="validation",
                require=True,
                registry_entry=entry,
                format_spec=spec,
                expected_recipe=recipe,
            )


def test_strict_cache_load_reads_and_validates_each_jsonl_once(tmp_path, monkeypatch):
    entry = DatasetRegistryEntry(
        dataset_id="boolq",
        task_family="qa_reasoning",
        hf_args=("google/boolq",),
        train_split="train",
    )
    spec = load_dataset_format_specs(_sample_render_formats_path(tmp_path))["boolq"]
    cache_root = tmp_path / "cache"
    cache_training_datasets(
        {"boolq": entry},
        {"boolq": spec},
        out_dir=cache_root,
        train_limit=1,
        min_train_rows=1,
        loader=lambda *args: {"train": [{"question": "Q", "passage": "P", "answer": True}]},
    )
    train_path = dataset_cache_split_path(cache_root, "boolq", "train")
    original_open = Path.open
    jsonl_opens = []

    def tracked_open(path, *args, **kwargs):
        if path == train_path:
            jsonl_opens.append(args[0] if args else kwargs.get("mode", "r"))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    loaded = load_cached_dataset_splits(
        cache_root,
        dataset_id="boolq",
        train_split="train",
        require=True,
        registry_entry=entry,
        format_spec=spec,
        expected_recipe=DatasetCacheRecipe("first", None, 1, None),
    )

    assert isinstance(loaded["train"], list)
    assert len(loaded["train"]) == 1
    assert jsonl_opens == ["rb"]


def test_legacy_subsample_cache_rejects_any_filtered_or_dropped_rows(tmp_path):
    entry = DatasetRegistryEntry(
        dataset_id="boolq",
        task_family="qa_reasoning",
        hf_args=("google/boolq",),
        filter={"max_chars": {"context": 1}},
        train_split="train",
    )
    spec = load_dataset_format_specs(_sample_render_formats_path(tmp_path))["boolq"]
    cache_root = tmp_path / "cache"
    report = cache_training_datasets(
        {"boolq": entry},
        {"boolq": spec},
        out_dir=cache_root,
        train_limit=2,
        min_train_rows=0,
        loader=lambda *args: {
            "train": [
                {"question": "Q1", "passage": "too long", "answer": True},
                {"passage": "P", "answer": False},
            ]
        },
        sample_strategy="legacy_subsample",
        sample_seed=42,
    )
    split = report.splits[0]

    assert not report.valid
    assert split.status == "exactness_mismatch"
    assert split.n_filtered == 1
    assert split.n_dropped == 1
    assert "filtered/dropped rows must both be zero" in split.error
    with pytest.raises(ValueError, match="legacy cache is not exact"):
        load_cached_dataset_splits(
            cache_root,
            dataset_id="boolq",
            train_split="train",
            require=True,
            registry_entry=entry,
            format_spec=spec,
            expected_recipe=DatasetCacheRecipe("legacy_subsample", 42, 2, None),
        )


def test_legacy_subsample_cache_requires_complete_expected_row_count(tmp_path):
    entry = DatasetRegistryEntry(
        dataset_id="boolq",
        task_family="qa_reasoning",
        hf_args=("google/boolq",),
        train_split="train",
    )
    spec = load_dataset_format_specs(_sample_render_formats_path(tmp_path))["boolq"]
    cache_root = tmp_path / "cache"
    rows = [{"question": f"Q{index}", "passage": f"P{index}", "answer": True} for index in range(3)]
    report = cache_training_datasets(
        {"boolq": entry},
        {"boolq": spec},
        out_dir=cache_root,
        train_limit=3,
        max_scan=2,
        min_train_rows=0,
        loader=lambda *args: {"train": rows},
        sample_strategy="legacy_subsample",
        sample_seed=42,
    )
    split = report.splits[0]

    assert not report.valid
    assert split.status == "exactness_mismatch"
    assert split.n_cached == 2
    assert "min(cache_limit, source_split_num_rows)" in split.error
    with pytest.raises(ValueError, match="cache_row_count differs"):
        load_cached_dataset_splits(
            cache_root,
            dataset_id="boolq",
            train_split="train",
            require=True,
            registry_entry=entry,
            format_spec=spec,
            expected_recipe=DatasetCacheRecipe("legacy_subsample", 42, 3, None),
        )


def test_legacy_cache_metadata_is_visibly_incompatible(tmp_path):
    entry = DatasetRegistryEntry(
        dataset_id="boolq",
        task_family="qa_reasoning",
        hf_args=("google/boolq",),
        train_split="train",
    )
    spec = load_dataset_format_specs(_sample_render_formats_path(tmp_path))["boolq"]
    train_path = dataset_cache_split_path(tmp_path / "cache", "boolq", "train")
    train_path.parent.mkdir(parents=True)
    train_path.write_text('{"answer": true, "context": "P", "question": "Q"}\n')
    train_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "sample_strategy": "first",
                "sample_seed": None,
                "shuffle_buffer_size": None,
            }
        )
    )

    legacy_loaded = load_cached_dataset_splits(
        tmp_path / "cache",
        dataset_id="boolq",
        train_split="train",
        require=True,
    )
    assert list(legacy_loaded["train"])[0]["question"] == "Q"

    with pytest.raises(ValueError, match="lacks provenance fields"):
        load_cached_dataset_splits(
            tmp_path / "cache",
            dataset_id="boolq",
            train_split="train",
            require=True,
            registry_entry=entry,
            format_spec=spec,
            expected_recipe=DatasetCacheRecipe("first", None, 1, None),
        )
    assert (
        load_cached_dataset_splits(
            tmp_path / "cache",
            dataset_id="boolq",
            train_split="train",
            registry_entry=entry,
            format_spec=spec,
            expected_recipe=DatasetCacheRecipe("first", None, 1, None),
        )
        is None
    )


def test_required_cache_load_rejects_tampered_jsonl_content(tmp_path):
    entry = DatasetRegistryEntry(
        dataset_id="boolq",
        task_family="qa_reasoning",
        hf_args=("google/boolq",),
        train_split="train",
    )
    registry = {"boolq": entry}
    spec = load_dataset_format_specs(_sample_render_formats_path(tmp_path))["boolq"]
    cache_root = tmp_path / "cache"
    cache_training_datasets(
        registry,
        {"boolq": spec},
        out_dir=cache_root,
        train_limit=1,
        min_train_rows=1,
        loader=lambda *args: {"train": [{"question": "Q", "passage": "P", "answer": True}]},
    )
    train_path = dataset_cache_split_path(cache_root, "boolq", "train")
    train_path.write_text(train_path.read_text().replace('"question": "Q"', '"question": "X"'))

    with pytest.raises(ValueError, match="cache_jsonl_sha256"):
        load_cached_dataset_splits(
            cache_root,
            dataset_id="boolq",
            train_split="train",
            require=True,
            registry_entry=entry,
            format_spec=spec,
            expected_recipe=DatasetCacheRecipe("first", None, 1, None),
        )
    assert (
        load_cached_dataset_splits(
            cache_root,
            dataset_id="boolq",
            train_split="train",
            registry_entry=entry,
            format_spec=spec,
            expected_recipe=DatasetCacheRecipe("first", None, 1, None),
        )
        is None
    )


def test_write_training_sample_render_audit_report_does_not_store_raw_samples(tmp_path):
    registry = load_dataset_registry(_registry_path(tmp_path))
    formats = load_dataset_format_specs(_sample_render_formats_path(tmp_path))
    report = audit_training_sample_rendering(
        [_sample_render_job()],
        registry,
        formats,
        loader=lambda *args: {
            "train": [{"question": "secret question", "passage": "secret passage", "answer": True}]
        },
    )
    out = tmp_path / "sample_render_report.json"

    write_training_sample_render_audit_report(report, out)
    text = out.read_text()

    assert "secret question" not in text
    assert "secret passage" not in text
    assert '"n_rendered": 1' in text


def test_audit_datasets_parser_accepts_registry_formats_and_no_load():
    args = build_parser().parse_args(
        [
            "audit-datasets",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--dataset-id",
            "boolq",
            "--out",
            "/tmp/datasets.json",
            "--no-load",
            "--streaming",
            "--allow-issues",
        ]
    )

    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.dataset_id == ["boolq"]
    assert args.out == Path("/tmp/datasets.json")
    assert args.no_load
    assert args.streaming
    assert args.allow_issues


def test_audit_training_samples_parser_accepts_plan_registry_and_formats():
    args = build_parser().parse_args(
        [
            "audit-training-samples",
            "--manifest",
            "/tmp/manifest.jsonl",
            "--config",
            "/tmp/training.yaml",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--dataset-id",
            "boolq",
            "--max-samples",
            "3",
            "--split",
            "validation",
            "--out",
            "/tmp/samples.json",
            "--streaming",
            "--allow-issues",
        ]
    )

    assert args.manifest == Path("/tmp/manifest.jsonl")
    assert args.config == Path("/tmp/training.yaml")
    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.dataset_id == ["boolq"]
    assert args.max_samples == 3
    assert args.split == "validation"
    assert args.out == Path("/tmp/samples.json")
    assert args.streaming
    assert args.allow_issues


def test_audit_training_sample_set_parser_accepts_assignment_summary_selection():
    args = build_parser().parse_args(
        [
            "audit-training-sample-set",
            "--assignment-summary",
            "/tmp/assignment_summary.json",
            "--config",
            "/tmp/training.yaml",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--dataset-id",
            "boolq",
            "--selection",
            "all-jobs",
            "--max-samples",
            "2",
            "--split",
            "validation",
            "--out",
            "/tmp/sample_set.json",
            "--streaming",
            "--allow-issues",
        ]
    )

    assert args.assignment_summary == Path("/tmp/assignment_summary.json")
    assert args.config == Path("/tmp/training.yaml")
    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.dataset_id == ["boolq"]
    assert args.selection == "all-jobs"
    assert args.max_samples == 2
    assert args.split == "validation"
    assert args.out == Path("/tmp/sample_set.json")
    assert args.streaming
    assert args.allow_issues


def test_cache_training_datasets_parser_accepts_bounded_cache_options():
    args = build_parser().parse_args(
        [
            "cache-training-datasets",
            "--registry",
            "/tmp/task_data.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--dataset-id",
            "boolq",
            "--out-dir",
            "/tmp/cache",
            "--summary-out",
            "/tmp/cache_summary.json",
            "--train-limit",
            "10000",
            "--eval-limit",
            "1000",
            "--max-scan",
            "50000",
            "--min-train-rows",
            "10000",
            "--streaming",
            "--sample-strategy",
            "legacy_subsample",
            "--sample-seed",
            "17",
            "--shuffle-buffer-size",
            "5000",
            "--overwrite",
            "--allow-issues",
        ]
    )

    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.dataset_id == ["boolq"]
    assert args.out_dir == Path("/tmp/cache")
    assert args.summary_out == Path("/tmp/cache_summary.json")
    assert args.train_limit == 10000
    assert args.eval_limit == 1000
    assert args.max_scan == 50000
    assert args.min_train_rows == 10000
    assert args.streaming
    assert args.sample_strategy == "legacy_subsample"
    assert args.sample_seed == 17
    assert args.shuffle_buffer_size == 5000
    assert args.overwrite
    assert args.allow_issues


def test_cache_training_datasets_rejects_legacy_subsample_with_streaming():
    args = build_parser().parse_args(
        [
            "cache-training-datasets",
            "--registry",
            "/tmp/unused_registry.yaml",
            "--formats",
            "/tmp/unused_formats.yaml",
            "--out-dir",
            "/tmp/unused_cache",
            "--sample-strategy",
            "legacy_subsample",
            "--streaming",
        ]
    )

    with pytest.raises(ValueError, match="incompatible with --streaming"):
        args.func(args)
