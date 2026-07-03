"""Prompt template rendering and validation."""

from __future__ import annotations

from dataclasses import dataclass
import string
from typing import Any


@dataclass(frozen=True)
class PromptValidationReport:
    required_fields: list[str]
    n_examples: int
    missing_by_example: list[dict[str, Any]]

    @property
    def valid(self) -> bool:
        return not self.missing_by_example


def template_fields(template: str) -> list[str]:
    """Return unique root field names required by a `str.format` template."""

    fields = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root and root not in fields:
            fields.append(root)
    return sorted(fields)


def render_prompt(template: str, example: dict[str, Any]) -> str:
    """Render a prompt template and fail loudly on missing fields."""

    missing = [field for field in template_fields(template) if field not in example]
    if missing:
        raise KeyError(f"prompt example missing fields: {missing}")
    return template.format(**example)


def validate_prompt_examples(
    template: str,
    examples: list[dict[str, Any]],
) -> PromptValidationReport:
    fields = template_fields(template)
    missing_rows = []
    for idx, example in enumerate(examples):
        missing = [field for field in fields if field not in example]
        if missing:
            missing_rows.append({"index": idx, "missing_fields": missing})
    return PromptValidationReport(
        required_fields=fields,
        n_examples=len(examples),
        missing_by_example=missing_rows,
    )
