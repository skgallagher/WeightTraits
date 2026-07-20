"""Stable JSONL schema and audits for behavioral probe responses."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


RESPONSE_STATUSES = {"completed", "dropped", "error"}


@dataclass(frozen=True)
class BehaviorResponse:
    run_id: str
    model_id: str
    probe_id: str
    prompt_id: str
    sample_id: int
    prompt: str
    response: str | None
    status: str = "completed"
    reference: str | None = None
    seed: int | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (self.run_id, self.model_id, self.probe_id, self.prompt_id, self.sample_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BehaviorResponse":
        required = ("run_id", "model_id", "probe_id", "prompt_id", "sample_id", "prompt")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"behavior response is missing required fields: {missing}")
        status = str(row.get("status", "completed"))
        if status not in RESPONSE_STATUSES:
            raise ValueError(f"unsupported behavior response status: {status}")
        sample_id = int(row["sample_id"])
        if sample_id < 0:
            raise ValueError("behavior response sample_id must be non-negative")
        response = row.get("response")
        return cls(
            run_id=str(row["run_id"]),
            model_id=str(row["model_id"]),
            probe_id=str(row["probe_id"]),
            prompt_id=str(row["prompt_id"]),
            sample_id=sample_id,
            prompt=str(row["prompt"]),
            response=None if response is None else str(response),
            status=status,
            reference=None if row.get("reference") is None else str(row["reference"]),
            seed=None if row.get("seed") is None else int(row["seed"]),
            reason=None if row.get("reason") is None else str(row["reason"]),
            metadata=dict(row.get("metadata", {})),
            schema_version=int(row.get("schema_version", 1)),
        )


def load_behavior_responses(path: str | Path) -> list[BehaviorResponse]:
    records: list[BehaviorResponse] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(BehaviorResponse.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid behavior response at line {line_number}: {exc}") from exc
    return records


def write_behavior_responses(
    records: Iterable[BehaviorResponse], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = prediction.lower().split()
    expected = reference.lower().split()
    if not predicted or not expected:
        return 0.0
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, start=1):
            if token == expected_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    overlap = previous[-1]
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def audit_behavior_responses(
    records: Iterable[BehaviorResponse],
    *,
    expected_samples_per_prompt: int | None = None,
    allow_empty_completed: bool = False,
) -> dict[str, Any]:
    rows = list(records)
    if expected_samples_per_prompt is not None and expected_samples_per_prompt <= 0:
        raise ValueError("expected_samples_per_prompt must be positive")

    key_counts = Counter(record.key for record in rows)
    duplicates = sorted(key for key, count in key_counts.items() if count > 1)
    status_counts = Counter(record.status for record in rows)
    reason_counts = Counter(record.reason or "unspecified" for record in rows if record.status != "completed")
    empty_completed = sorted(
        record.key
        for record in rows
        if record.status == "completed" and not (record.response or "").strip()
    )

    prompt_texts: dict[tuple[str, str], set[str]] = defaultdict(set)
    samples: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    models_by_run_probe: dict[tuple[str, str], set[str]] = defaultdict(set)
    prompts_by_run_probe: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in rows:
        prompt_texts[(record.probe_id, record.prompt_id)].add(record.prompt)
        samples[(record.run_id, record.model_id, record.probe_id, record.prompt_id)].add(
            record.sample_id
        )
        models_by_run_probe[(record.run_id, record.probe_id)].add(record.model_id)
        prompts_by_run_probe[(record.run_id, record.probe_id)].add(record.prompt_id)
    inconsistent_prompts = sorted(key for key, texts in prompt_texts.items() if len(texts) > 1)
    incomplete: list[dict[str, Any]] = []
    if expected_samples_per_prompt is not None:
        expected = set(range(expected_samples_per_prompt))
        for run_probe, models in sorted(models_by_run_probe.items()):
            run_id, probe_id = run_probe
            for model_id in sorted(models):
                for prompt_id in sorted(prompts_by_run_probe[run_probe]):
                    key = (run_id, model_id, probe_id, prompt_id)
                    observed = samples.get(key, set())
                    missing = sorted(expected - observed)
                    unexpected = sorted(observed - expected)
                    if missing or unexpected:
                        incomplete.append(
                            {
                                "run_id": run_id,
                                "model_id": model_id,
                                "probe_id": probe_id,
                                "prompt_id": prompt_id,
                                "missing_sample_ids": missing,
                                "unexpected_sample_ids": unexpected,
                            }
                        )

    issues: list[str] = []
    if duplicates:
        issues.append("duplicate response keys")
    if any(status != "completed" and count for status, count in status_counts.items()):
        issues.append("non-completed response records")
    if empty_completed and not allow_empty_completed:
        issues.append("completed responses with empty text")
    if inconsistent_prompts:
        issues.append("prompt IDs mapped to inconsistent prompt text")
    if incomplete:
        issues.append("incomplete prompt/sample grids")
    completed = [
        record for record in rows if record.status == "completed" and (record.response or "").strip()
    ]
    response_texts = [str(record.response).strip() for record in completed]
    reference_scores = [
        _rouge_l_f1(str(record.response), record.reference)
        for record in completed
        if record.reference is not None and record.reference.strip()
    ]
    unique_by_model = {
        model_id: len({str(record.response).strip() for record in completed if record.model_id == model_id})
        for model_id in sorted({record.model_id for record in completed})
    }
    dominant_fraction_by_model = {}
    unique_draws_by_model_prompt: dict[tuple[str, str], int] = {}
    for model_id in sorted({record.model_id for record in completed}):
        model_responses = [
            str(record.response).strip() for record in completed if record.model_id == model_id
        ]
        counts = Counter(model_responses)
        dominant_fraction_by_model[model_id] = (
            max(counts.values()) / len(model_responses) if model_responses else None
        )
        for prompt_id in sorted(
            {record.prompt_id for record in completed if record.model_id == model_id}
        ):
            unique_draws_by_model_prompt[(model_id, prompt_id)] = len(
                {
                    str(record.response).strip()
                    for record in completed
                    if record.model_id == model_id and record.prompt_id == prompt_id
                }
            )
    mean_unique_draws_by_model = {}
    varying_prompt_fraction_by_model = {}
    for model_id in sorted({record.model_id for record in completed}):
        draw_counts = [
            count
            for (observed_model, _), count in unique_draws_by_model_prompt.items()
            if observed_model == model_id
        ]
        mean_unique_draws_by_model[model_id] = (
            sum(draw_counts) / len(draw_counts) if draw_counts else None
        )
        varying_prompt_fraction_by_model[model_id] = (
            sum(count > 1 for count in draw_counts) / len(draw_counts) if draw_counts else None
        )
    return {
        "schema_version": 1,
        "valid": not issues,
        "issues": issues,
        "n_records": len(rows),
        "n_runs": len({record.run_id for record in rows}),
        "n_models": len({record.model_id for record in rows}),
        "n_probes": len({record.probe_id for record in rows}),
        "n_prompts": len({(record.probe_id, record.prompt_id) for record in rows}),
        "status_counts": dict(sorted(status_counts.items())),
        "dropped_reason_counts": dict(sorted(reason_counts.items())),
        "duplicate_keys": [list(key) for key in duplicates],
        "empty_completed_keys": [list(key) for key in empty_completed],
        "inconsistent_prompt_keys": [list(key) for key in inconsistent_prompts],
        "incomplete_grids": incomplete,
        "expected_samples_per_prompt": expected_samples_per_prompt,
        "allow_empty_completed": allow_empty_completed,
        "quality": {
            "n_completed_nonempty": len(completed),
            "mean_response_characters": (
                sum(len(text) for text in response_texts) / len(response_texts)
                if response_texts
                else None
            ),
            "unique_response_fraction": (
                len(set(response_texts)) / len(response_texts) if response_texts else None
            ),
            "unique_responses_by_model": unique_by_model,
            "dominant_response_fraction_by_model": dominant_fraction_by_model,
            "mean_unique_draws_per_prompt_by_model": mean_unique_draws_by_model,
            "varying_prompt_fraction_by_model": varying_prompt_fraction_by_model,
            "mean_reference_rouge_l_f1": (
                sum(reference_scores) / len(reference_scores) if reference_scores else None
            ),
            "n_reference_scored": len(reference_scores),
        },
    }
