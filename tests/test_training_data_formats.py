import json
from pathlib import Path

from weighttraits.cli import build_parser
from weighttraits.training.data_formats import (
    load_dataset_format_specs,
    validate_training_jobs_against_formats,
    write_data_format_report,
)
from weighttraits.training.planner import build_training_jobs


def _jobs(tmp_path: Path):
    rows = [
        {
            "node_id": "n0",
            "parent_id": "root",
            "depth": 1,
            "path": ["root", "n0"],
            "grow": "train",
            "task_family": "qa_reasoning",
            "dataset_id": "boolq",
        },
        {
            "node_id": "n1",
            "parent_id": "n0",
            "depth": 2,
            "path": ["root", "n0", "n1"],
            "grow": "train",
            "task_family": "math_reasoning",
            "dataset_id": "math",
        },
    ]
    config = {
        "base_model": "base",
        "output_root": str(tmp_path / "outputs"),
        "prompt": {
            "task_templates": {
                "qa_reasoning": "Question: {question}\nContext: {context}\nAnswer: {answer}",
                "math_reasoning": "Problem: {problem}\nSolution: {solution}",
            }
        },
    }
    return build_training_jobs(rows, config)


def test_dataset_format_validation_accepts_matching_specs(tmp_path):
    spec_path = tmp_path / "formats.yaml"
    spec_path.write_text(
        """
datasets:
  - dataset_id: boolq
    task_family: qa_reasoning
    prompt_fields: [question, context, answer]
  - dataset_id: math
    task_family: math_reasoning
    field_map:
      problem: question
      solution: answer
"""
    )

    report = validate_training_jobs_against_formats(_jobs(tmp_path), load_dataset_format_specs(spec_path))

    assert report.valid
    assert report.n_jobs == 2
    assert report.n_valid == 2


def test_dataset_format_validation_reports_missing_spec(tmp_path):
    spec_path = tmp_path / "formats.yaml"
    spec_path.write_text(
        """
datasets:
  - dataset_id: boolq
    prompt_fields: [question, context, answer]
"""
    )

    report = validate_training_jobs_against_formats(_jobs(tmp_path), load_dataset_format_specs(spec_path))

    assert not report.valid
    assert report.issues[0].issue == "missing_dataset_format_spec"
    assert report.issues[0].dataset_id == "math"


def test_dataset_format_validation_reports_missing_prompt_fields(tmp_path):
    spec_path = tmp_path / "formats.yaml"
    spec_path.write_text(
        """
datasets:
  - dataset_id: boolq
    prompt_fields: [question, answer]
  - dataset_id: math
    prompt_fields: [problem, solution]
"""
    )

    report = validate_training_jobs_against_formats(_jobs(tmp_path), load_dataset_format_specs(spec_path))

    assert not report.valid
    assert report.issues[0].issue == "missing_prompt_fields"
    assert report.issues[0].missing_fields == ("context",)


def test_dataset_format_validation_reports_task_family_mismatch(tmp_path):
    spec_path = tmp_path / "formats.yaml"
    spec_path.write_text(
        """
datasets:
  - dataset_id: boolq
    task_family: classification
    prompt_fields: [question, context, answer]
  - dataset_id: math
    task_family: math_reasoning
    prompt_fields: [problem, solution]
"""
    )

    report = validate_training_jobs_against_formats(_jobs(tmp_path), load_dataset_format_specs(spec_path))

    assert not report.valid
    assert report.issues[0].issue == "task_family_mismatch"
    assert report.issues[0].details == {
        "job_task_family": "qa_reasoning",
        "spec_task_family": "classification",
    }


def test_dataset_format_loader_rejects_string_prompt_fields(tmp_path):
    spec_path = tmp_path / "formats.yaml"
    spec_path.write_text(
        """
datasets:
  - dataset_id: boolq
    prompt_fields: question
"""
    )

    try:
        load_dataset_format_specs(spec_path)
    except ValueError as exc:
        assert "prompt_fields must be a list" in str(exc)
    else:
        raise AssertionError("expected malformed prompt_fields to fail")


def test_write_data_format_report(tmp_path):
    spec_path = tmp_path / "formats.yaml"
    spec_path.write_text(
        """
datasets:
  - dataset_id: boolq
    prompt_fields: [question, context, answer]
  - dataset_id: math
    prompt_fields: [problem, solution]
"""
    )
    report = validate_training_jobs_against_formats(_jobs(tmp_path), load_dataset_format_specs(spec_path))
    out = tmp_path / "report.json"

    write_data_format_report(report, out)
    loaded = json.loads(out.read_text())

    assert loaded["valid"] is True
    assert loaded["n_valid"] == 2


def test_validate_training_data_parser():
    args = build_parser().parse_args(
        [
            "validate-training-data",
            "--manifest",
            "/tmp/manifest.jsonl",
            "--config",
            "/tmp/training.yaml",
            "--formats",
            "/tmp/formats.yaml",
            "--out",
            "/tmp/report.json",
            "--allow-issues",
        ]
    )

    assert args.manifest == Path("/tmp/manifest.jsonl")
    assert args.config == Path("/tmp/training.yaml")
    assert args.formats == Path("/tmp/formats.yaml")
    assert args.out == Path("/tmp/report.json")
    assert args.allow_issues
