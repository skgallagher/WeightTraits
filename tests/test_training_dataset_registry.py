from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.data_formats import load_dataset_format_specs
from weighttraits.training.datasets import audit_dataset_registry, load_dataset_registry


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
            "--allow-issues",
        ]
    )

    assert args.registry == Path("/tmp/task_data.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.dataset_id == ["boolq"]
    assert args.out == Path("/tmp/datasets.json")
    assert args.no_load
    assert args.allow_issues
