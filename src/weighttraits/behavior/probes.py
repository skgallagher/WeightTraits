"""Prompt artifacts for held-out behavioral probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


HELDOUT_PROBE_DATASETS: dict[str, dict[str, Any]] = {
    "hellaswag": {
        "dataset_args": ("Rowan/hellaswag",),
        "split": "validation",
        "max_new_tokens": 64,
    },
    "arc_challenge": {
        "dataset_args": ("allenai/ai2_arc", "ARC-Challenge"),
        "split": "test",
        "max_new_tokens": 64,
    },
    "mmlu": {
        "dataset_args": ("cais/mmlu", "all"),
        "split": "test",
        "max_new_tokens": 64,
    },
    "truthfulqa": {
        "dataset_args": ("truthful_qa", "generation"),
        "split": "validation",
        "max_new_tokens": 64,
    },
    "dolly_open_ended": {
        "dataset_args": ("databricks/databricks-dolly-15k",),
        "split": "train",
        "max_new_tokens": 96,
    },
}

_DOLLY_OPEN_ENDED_CATEGORIES = {
    "open_qa",
    "general_qa",
    "brainstorming",
    "creative_writing",
}


@dataclass(frozen=True)
class BehaviorPrompt:
    probe_id: str
    prompt_id: str
    prompt: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BehaviorPrompt":
        required = ("probe_id", "prompt_id", "prompt")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"behavior prompt is missing required fields: {missing}")
        prompt = str(row["prompt"])
        if not prompt.strip():
            raise ValueError("behavior prompt text must be non-empty")
        return cls(
            probe_id=str(row["probe_id"]),
            prompt_id=str(row["prompt_id"]),
            prompt=prompt,
            reference=None if row.get("reference") is None else str(row["reference"]),
            metadata=dict(row.get("metadata", {})),
            schema_version=int(row.get("schema_version", 1)),
        )


def load_behavior_prompts(path: str | Path) -> list[BehaviorPrompt]:
    prompts: list[BehaviorPrompt] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                prompts.append(BehaviorPrompt.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid behavior prompt at line {line_number}: {exc}") from exc
    if not prompts:
        raise ValueError("behavior prompt artifact is empty")
    keys = [(prompt.probe_id, prompt.prompt_id) for prompt in prompts]
    if len(set(keys)) != len(keys):
        raise ValueError("behavior prompt artifact contains duplicate probe/prompt IDs")
    return prompts


def write_behavior_prompts(prompts: Iterable[BehaviorPrompt], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def hellaswag_prompts_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model_task: str,
    n_prompts: int,
    seed: int,
) -> list[BehaviorPrompt]:
    if model_task not in {"seq2seq", "causal_lm"}:
        raise ValueError("HellaSwag model_task must be seq2seq or causal_lm")
    if n_prompts <= 0 or n_prompts > len(rows):
        raise ValueError("n_prompts must be positive and no larger than the available rows")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(rows), size=n_prompts, replace=False)
    prompts: list[BehaviorPrompt] = []
    for ordinal, source_index in enumerate(selected):
        row = rows[int(source_index)]
        activity = str(row["activity_label"])
        context = str(row["ctx"])
        endings = list(row["endings"])
        label = int(row["label"])
        if label < 0 or label >= len(endings):
            raise ValueError(f"HellaSwag row {source_index} has an invalid label index")
        if model_task == "causal_lm":
            text = f"Complete the following sentence.\n\n{activity}: {context}"
        else:
            text = f"complete: {activity}: {context}"
        prompts.append(
            BehaviorPrompt(
                probe_id="hellaswag",
                prompt_id=f"hellaswag-{ordinal:04d}",
                prompt=text,
                reference=str(endings[label]),
                metadata={
                    "dataset": "Rowan/hellaswag",
                    "split": "validation",
                    "source_index": int(source_index),
                    "model_task": model_task,
                    "seed": seed,
                },
            )
        )
    return prompts


def heldout_probe_prompts_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    probe_id: str,
    model_task: str,
    n_prompts: int,
    seed: int,
    prompt_style: str = "standard",
) -> list[BehaviorPrompt]:
    """Materialize a deterministic free-generation held-out probe artifact.

    Prompt wording matches the generative behavioral panel used by ELLMTrees.
    References are retained only for response-health checks; behavioral distance
    is computed between freely generated model outputs.
    """
    if probe_id not in HELDOUT_PROBE_DATASETS:
        raise ValueError(f"unsupported held-out behavior probe: {probe_id}")
    if model_task not in {"seq2seq", "causal_lm"}:
        raise ValueError("behavior probe model_task must be seq2seq or causal_lm")
    if prompt_style not in {"standard", "explanation_first"}:
        raise ValueError(f"unsupported behavior prompt style: {prompt_style}")
    if prompt_style != "standard" and probe_id not in {"mmlu", "arc_challenge"}:
        raise ValueError("explanation_first is only supported for multiple-choice probes")
    if probe_id == "hellaswag":
        return hellaswag_prompts_from_rows(
            rows, model_task=model_task, n_prompts=n_prompts, seed=seed
        )
    eligible_rows = list(enumerate(rows))
    if probe_id == "dolly_open_ended":
        eligible_rows = [
            (source_index, row)
            for source_index, row in eligible_rows
            if str(row.get("category", "")) in _DOLLY_OPEN_ENDED_CATEGORIES
            and str(row.get("instruction", "")).strip()
        ]
    if n_prompts <= 0 or n_prompts > len(eligible_rows):
        raise ValueError("n_prompts must be positive and no larger than the available rows")

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(eligible_rows), size=n_prompts, replace=False)
    config = HELDOUT_PROBE_DATASETS[probe_id]
    prompts: list[BehaviorPrompt] = []
    for ordinal, selected_index_value in enumerate(selected):
        source_index, row = eligible_rows[int(selected_index_value)]
        if probe_id == "mmlu":
            choices = [str(choice) for choice in row["choices"]]
            labels = list("ABCDEFGH"[: len(choices)])
            answer_index = int(row["answer"])
            if answer_index < 0 or answer_index >= len(choices):
                raise ValueError(f"MMLU row {source_index} has an invalid answer index")
            choices_block = "\n".join(
                f"{label}) {choice}" for label, choice in zip(labels, choices)
            )
            question = str(row["question"])
            reference = choices[answer_index]
        elif probe_id == "arc_challenge":
            choice_record = row["choices"]
            labels = [str(label) for label in choice_record["label"]]
            choices = [str(choice) for choice in choice_record["text"]]
            if len(labels) != len(choices) or not choices:
                raise ValueError(f"ARC-Challenge row {source_index} has invalid choices")
            choices_block = "\n".join(
                f"{label}) {choice}" for label, choice in zip(labels, choices)
            )
            question = str(row["question"])
            answer_key = str(row["answerKey"])
            if answer_key not in labels:
                raise ValueError(f"ARC-Challenge row {source_index} has an unknown answer key")
            reference = choices[labels.index(answer_key)]
        elif probe_id == "truthfulqa":
            question = str(row["question"])
            choices_block = ""
            reference = str(row["best_answer"])
        else:
            instruction = str(row["instruction"]).strip()
            context = str(row.get("context", "")).strip()
            question = instruction
            choices_block = ""
            reference = str(row.get("response", "")) or None

        if probe_id in {"mmlu", "arc_challenge"}:
            if model_task == "causal_lm":
                instruction = (
                    "Explain the reasoning in at least one complete sentence. Then state the "
                    "correct answer in words. Do not reply with only a letter, number, "
                    "true/false, or classification label."
                    if prompt_style == "explanation_first"
                    else "Answer the following multiple-choice question in your own words."
                )
                text = f"{instruction}\n\nQuestion: {question}\n{choices_block}\n\nAnswer:\n"
            else:
                text = f"answer: {question}\n{choices_block}"
        elif probe_id == "truthfulqa" and model_task == "causal_lm":
            text = (
                "Answer the following question truthfully and concisely.\n\n"
                f"Question: {question}\n\nAnswer:\n"
            )
        elif probe_id == "truthfulqa":
            text = f"answer truthfully: {question}"
        elif model_task == "causal_lm":
            context_block = f"\n\nContext: {context}" if context else ""
            text = (
                "Respond naturally to the following request. Use complete sentences when "
                "appropriate.\n\n"
                f"Request: {question}{context_block}\n\nResponse:\n"
            )
        else:
            context_block = f" context: {context}" if context else ""
            text = f"respond naturally: {question}{context_block}"

        metadata = {
            "dataset": config["dataset_args"][0],
            "dataset_config": (
                config["dataset_args"][1] if len(config["dataset_args"]) > 1 else None
            ),
            "split": config["split"],
            "source_index": source_index,
            "model_task": model_task,
            "seed": seed,
            "prompt_style": prompt_style,
        }
        if probe_id == "mmlu" and row.get("subject") is not None:
            metadata["subject"] = str(row["subject"])
        if probe_id == "dolly_open_ended":
            metadata["category"] = str(row["category"])
        prompts.append(
            BehaviorPrompt(
                probe_id=probe_id,
                prompt_id=f"{probe_id}-{ordinal:04d}",
                prompt=text,
                reference=reference,
                metadata=metadata,
            )
        )
    return prompts


def load_heldout_probe_prompts(
    *,
    probe_id: str,
    model_task: str,
    n_prompts: int,
    seed: int,
    prompt_style: str = "standard",
    local_files_only: bool = True,
) -> list[BehaviorPrompt]:
    if probe_id not in HELDOUT_PROBE_DATASETS:
        raise ValueError(f"unsupported held-out behavior probe: {probe_id}")
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise ImportError("behavior prompt preparation requires datasets") from exc
    config = HELDOUT_PROBE_DATASETS[probe_id]
    dataset = load_dataset(
        *config["dataset_args"],
        split=config["split"],
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    return heldout_probe_prompts_from_rows(
        [dict(row) for row in dataset],
        probe_id=probe_id,
        model_task=model_task,
        n_prompts=n_prompts,
        seed=seed,
        prompt_style=prompt_style,
    )


def load_hellaswag_prompts(
    *, model_task: str, n_prompts: int, seed: int, local_files_only: bool = True
) -> list[BehaviorPrompt]:
    return load_heldout_probe_prompts(
        probe_id="hellaswag",
        model_task=model_task,
        n_prompts=n_prompts,
        seed=seed,
        local_files_only=local_files_only,
    )
