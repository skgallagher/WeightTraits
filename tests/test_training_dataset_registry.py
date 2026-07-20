from pathlib import Path
from types import SimpleNamespace

from weighttraits.cli import build_parser
from weighttraits.training.data_formats import load_dataset_format_specs
from weighttraits.training.datasets import (
    audit_dataset_registry,
    audit_training_sample_rendering,
    cache_training_datasets,
    dataset_cache_split_path,
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
        {"question": f"Q{index}", "passage": f"P{index}", "answer": True}
        for index in range(20)
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
            "seeded_shuffle",
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
    assert args.sample_strategy == "seeded_shuffle"
    assert args.sample_seed == 17
    assert args.shuffle_buffer_size == 5000
    assert args.overwrite
    assert args.allow_issues
