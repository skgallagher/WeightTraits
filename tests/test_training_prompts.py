import pytest

from weighttraits.training.prompts import (
    render_prompt,
    template_fields,
    validate_prompt_examples,
)


def test_template_fields_extracts_unique_root_fields():
    fields = template_fields("Question: {question}\nA: {answers[0]}\nAgain: {question}\n{{literal}}")

    assert fields == ["answers", "question"]


def test_render_prompt_fails_loudly_on_missing_fields():
    with pytest.raises(KeyError, match="missing fields"):
        render_prompt("Question: {question}\nAnswer: {answer}", {"question": "Q"})


def test_render_prompt_formats_complete_example():
    rendered = render_prompt(
        "Question: {question}\nAnswer: {answer}",
        {"question": "2+2?", "answer": "4"},
    )

    assert rendered == "Question: 2+2?\nAnswer: 4"


def test_validate_prompt_examples_reports_missing_fields():
    report = validate_prompt_examples(
        "Question: {question}\nContext: {context}\nAnswer: {answer}",
        [
            {"question": "Q", "context": "C", "answer": "A"},
            {"question": "Q", "answer": "A"},
        ],
    )

    assert not report.valid
    assert report.required_fields == ["answer", "context", "question"]
    assert report.missing_by_example == [{"index": 1, "missing_fields": ["context"]}]
