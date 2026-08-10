from pathlib import Path

import pytest

from weighttraits.training.data_formats import load_dataset_format_specs
from weighttraits.training.datasets import canonical_dataset_example


def _transform_specs(tmp_path: Path):
    path = tmp_path / "formats.yaml"
    path.write_text(
        """
datasets:
  - dataset_id: sentiment
    task_family: classification
    prompt_fields: [text]
    field_map:
      target: label
    transforms:
      target:
        op: label_index
        source: target
        labels: [negative, positive]
  - dataset_id: extractive_qa
    task_family: qa
    prompt_fields: [question]
    field_map:
      target: answers
    transforms:
      target:
        op: first_item
        source: target.text
        default: unanswerable
  - dataset_id: boolean_qa
    task_family: qa
    prompt_fields: [question]
    field_map:
      target: answer
    transforms:
      target:
        op: bool_map
        source: target
        true_value: "yes"
        false_value: "no"
  - dataset_id: multiple_choice
    task_family: qa
    prompt_fields: [question, choices]
    field_map:
      target: answerKey
    transforms:
      choices:
        op: format_choices
        source: choices
      target:
        op: choice_text
        key_source: target
        choices_source: choices
  - dataset_id: qa_composition
    task_family: qa
    prompt_fields: [qa_input]
    transforms:
      qa_input:
        op: compose_qa_input
        question_source: question
        choices_source: choices
        context_source: context
  - dataset_id: unchanged
    task_family: summarization
    prompt_fields: [document]
    field_map:
      target: summary
"""
    )
    return load_dataset_format_specs(path)


def test_dataset_format_loader_preserves_declarative_transforms(tmp_path):
    specs = _transform_specs(tmp_path)

    assert specs["sentiment"].transforms == {
        "target": {
            "op": "label_index",
            "source": "target",
            "labels": ["negative", "positive"],
        }
    }
    assert specs["sentiment"].to_dict()["transforms"] == specs["sentiment"].transforms
    assert specs["unchanged"].transforms is None


def test_label_index_transform_maps_configured_label_string(tmp_path):
    spec = _transform_specs(tmp_path)["sentiment"]

    example = canonical_dataset_example({"text": "great", "label": 1}, spec)

    assert example["target"] == "positive"
    assert (
        canonical_dataset_example({"text": "great", "target": "positive"}, spec)["target"]
        == "positive"
    )


def test_first_item_transform_supports_nested_source_and_empty_default(tmp_path):
    spec = _transform_specs(tmp_path)["extractive_qa"]

    answered = canonical_dataset_example(
        {"question": "Who?", "answers": {"text": ["Ada"], "answer_start": [4]}},
        spec,
    )
    unanswered = canonical_dataset_example(
        {"question": "Who?", "answers": {"text": [], "answer_start": []}},
        spec,
    )

    assert answered["target"] == "Ada"
    assert unanswered["target"] == "unanswerable"
    assert (
        canonical_dataset_example(
            {"question": "Who?", "target": "Ada"},
            spec,
        )["target"]
        == "Ada"
    )


@pytest.mark.parametrize(("answer", "expected"), [(True, "yes"), (False, "no")])
def test_bool_map_transform_uses_configured_strings(tmp_path, answer, expected):
    spec = _transform_specs(tmp_path)["boolean_qa"]

    example = canonical_dataset_example({"question": "Is it?", "answer": answer}, spec)

    assert example["target"] == expected
    assert (
        canonical_dataset_example(
            {"question": "Is it?", "target": expected},
            spec,
        )["target"]
        == expected
    )


def test_choice_transforms_format_prompt_choices_and_select_full_answer_text(tmp_path):
    spec = _transform_specs(tmp_path)["multiple_choice"]
    raw_choices = {
        "label": ["A", "B", "C"],
        "text": ["first", "second", "third"],
    }

    example = canonical_dataset_example(
        {"question": "Pick one", "choices": raw_choices, "answerKey": "B"},
        spec,
    )

    assert example["choices"] == "A) first B) second C) third"
    assert example["target"] == "second"
    cached_example = canonical_dataset_example(
        {
            "question": "Pick one",
            "choices": "A) first B) second C) third",
            "target": "second",
        },
        spec,
    )
    assert cached_example["choices"] == "A) first B) second C) third"
    assert cached_example["target"] == "second"


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("", "question: Pick one choices: A) first B) second"),
        (
            "supporting text",
            "question: Pick one choices: A) first B) second context: supporting text",
        ),
    ],
)
def test_compose_qa_input_formats_choices_and_uses_raw_context_truthiness(
    tmp_path,
    context,
    expected,
):
    spec = _transform_specs(tmp_path)["qa_composition"]
    raw = {
        "question": "Pick one",
        "choices": {"label": ["A", "B"], "text": ["first", "second"]},
        "context": context,
    }

    example = canonical_dataset_example(raw, spec)

    assert example["qa_input"] == expected
    assert canonical_dataset_example(example, spec)["qa_input"] == expected


def test_transforms_absent_preserves_existing_canonical_mapping(tmp_path):
    spec = _transform_specs(tmp_path)["unchanged"]
    row = {"document": "doc", "summary": "short"}

    example = canonical_dataset_example(row, spec)

    assert example == {"document": "doc", "summary": "short", "target": "short"}


def test_dataset_format_loader_rejects_unknown_transform(tmp_path):
    path = tmp_path / "formats.yaml"
    path.write_text(
        """
datasets:
  - dataset_id: broken
    prompt_fields: [text]
    transforms:
      target:
        op: dataset_specific_magic
        source: label
"""
    )

    with pytest.raises(ValueError, match="unsupported transform op"):
        load_dataset_format_specs(path)


def test_dataset_format_loader_validates_compose_qa_input_sources(tmp_path):
    path = tmp_path / "formats.yaml"
    path.write_text(
        """
datasets:
  - dataset_id: broken
    prompt_fields: [qa_input]
    transforms:
      qa_input:
        op: compose_qa_input
        question_source: 7
"""
    )

    with pytest.raises(ValueError, match="question_source.*must be a string"):
        load_dataset_format_specs(path)
