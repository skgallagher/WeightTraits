"""Stable response records and strict behavioral response-grid audits.

The contracts in :mod:`weighttraits.behavior.contracts` freeze source rows and prompts.  This
module freezes the other side of that boundary: decoded model responses and the exact Cartesian
grid of protocol, probe, prompt, and draw identifiers that a response artifact must contain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from weighttraits.behavior.contracts import (
    BehaviorProtocolRegistry,
    RenderedBehaviorPrompt,
)


RESPONSE_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ResponseProvenance:
    """Immutable provenance shared by every response in one leaf request."""

    request_sha256: str
    training_summary: str
    training_summary_sha256: str
    run_list: str
    run_list_sha256: str
    ledger: str
    ledger_sha256: str
    truth_manifest: str
    truth_manifest_sha256: str
    checkpoint: str
    checkpoint_artifact: str
    checkpoint_sha256: str
    registry: str
    registry_sha256: str
    prompt_artifacts: Mapping[str, Mapping[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorResponse:
    """One decoded completion at one frozen protocol-grid coordinate."""

    cohort_id: str
    tree_id: str
    model_id: str
    base_model_id: str
    base_model_revision: str
    model_task: str
    protocol_id: str
    probe_id: str
    prompt_id: str
    sample_id: int
    source_index: int
    source_row_sha256: str
    prompt_sha256: str
    draw_seed: int
    generation_batch_size: int
    seed_scope: str
    do_sample: bool
    temperature: float | None
    top_p: float | None
    min_new_tokens: int
    max_new_tokens: int
    empty_policy: str
    text: str
    reference: str | None
    response_sha256: str
    completed: bool
    empty: bool
    fixture_id: str
    fixture_sha256: str
    source_indices_sha256: str
    dataset: str
    dataset_config: str | None
    dataset_split: str
    dataset_revision: str
    provenance: ResponseProvenance
    schema_version: int = RESPONSE_SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.protocol_id, self.probe_id, self.prompt_id, self.sample_id)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["provenance"] = self.provenance.to_dict()
        return row


@dataclass(frozen=True)
class ResponseGridAudit:
    valid: bool
    n_expected: int
    n_observed: int
    n_missing: int
    n_extra: int
    n_duplicates: int
    n_errors: int
    protocol_counts: Mapping[str, int]
    probe_counts: Mapping[str, int]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["protocol_counts"] = dict(self.protocol_counts)
        row["probe_counts"] = dict(self.probe_counts)
        row["errors"] = list(self.errors)
        return row


def response_from_text(
    *,
    cohort_id: str,
    tree_id: str,
    model_id: str,
    base_model_id: str,
    base_model_revision: str,
    model_task: str,
    protocol_id: str,
    prompt: RenderedBehaviorPrompt,
    sample_id: int,
    generation_options: Mapping[str, Any],
    text: str,
    provenance: ResponseProvenance,
) -> BehaviorResponse:
    """Build a stable response row without normalizing the decoded continuation."""

    if not isinstance(text, str):
        raise TypeError("decoded response text must be a string")
    metadata = prompt.metadata
    option_sample_id = _integer(generation_options.get("sample_id"), "sample_id")
    if sample_id != option_sample_id:
        raise ValueError(
            f"sample_id mismatch: explicit {sample_id}, generation options {option_sample_id}"
        )
    return BehaviorResponse(
        cohort_id=_nonempty(cohort_id, "cohort_id"),
        tree_id=_nonempty(tree_id, "tree_id"),
        model_id=_nonempty(model_id, "model_id"),
        base_model_id=_nonempty(base_model_id, "base_model_id"),
        base_model_revision=_sha256_or_revision(base_model_revision, "base_model_revision"),
        model_task=_nonempty(model_task, "model_task"),
        protocol_id=_nonempty(protocol_id, "protocol_id"),
        probe_id=prompt.probe_id,
        prompt_id=prompt.prompt_id,
        sample_id=option_sample_id,
        source_index=prompt.source_index,
        source_row_sha256=_sha256_value(prompt.source_row_sha256, "source_row_sha256"),
        prompt_sha256=_sha256_value(prompt.prompt_sha256, "prompt_sha256"),
        draw_seed=_integer(generation_options.get("seed"), "draw_seed"),
        generation_batch_size=_positive_integer(
            generation_options.get("generation_batch_size"), "generation_batch_size"
        ),
        seed_scope=_exact_text(
            generation_options.get("seed_scope"), "seed_scope", "probe_draw"
        ),
        do_sample=_boolean(generation_options.get("do_sample"), "do_sample"),
        temperature=_optional_number(generation_options.get("temperature"), "temperature"),
        top_p=_optional_number(generation_options.get("top_p"), "top_p"),
        min_new_tokens=_integer(generation_options.get("min_new_tokens"), "min_new_tokens"),
        max_new_tokens=_integer(generation_options.get("max_new_tokens"), "max_new_tokens"),
        empty_policy=_nonempty(generation_options.get("empty_policy"), "empty_policy"),
        text=text,
        reference=prompt.reference,
        response_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        completed=True,
        empty=text == "",
        fixture_id=_metadata_string(metadata, "fixture_id"),
        fixture_sha256=_metadata_sha256(metadata, "fixture_sha256"),
        source_indices_sha256=_metadata_sha256(metadata, "source_indices_sha256"),
        dataset=_metadata_string(metadata, "dataset"),
        dataset_config=_metadata_optional_string(metadata, "dataset_config"),
        dataset_split=_metadata_string(metadata, "split"),
        dataset_revision=_metadata_string(metadata, "dataset_revision"),
        provenance=provenance,
    )


def load_rendered_behavior_prompts(path: str | Path) -> list[RenderedBehaviorPrompt]:
    """Load contract-rendered prompts from JSONL and verify their self-hashes."""

    source = Path(path)
    prompts: list[RenderedBehaviorPrompt] = []
    seen: set[tuple[str, str]] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{source}:{line_number} prompt row must be an object")
            prompt = _rendered_prompt_from_dict(raw, context=f"{source}:{line_number}")
            key = (prompt.probe_id, prompt.prompt_id)
            if key in seen:
                raise ValueError(f"{source}:{line_number} duplicate prompt coordinate: {key}")
            seen.add(key)
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"rendered prompt artifact is empty: {source}")
    return prompts


def load_behavior_responses(path: str | Path) -> list[BehaviorResponse]:
    """Load response JSONL strictly, preserving file order."""

    source = Path(path)
    if not source.exists():
        return []
    responses: list[BehaviorResponse] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{source}:{line_number} response row must be an object")
            responses.append(
                _response_from_dict(raw, context=f"{source}:{line_number}")
            )
    return responses


def expected_response_coordinates(
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
) -> list[tuple[str, RenderedBehaviorPrompt, int]]:
    """Return the exact deterministic requested grid after strict prompt-panel validation."""

    if not prompts_by_protocol:
        raise ValueError("at least one behavior protocol must be requested")
    coordinates: list[tuple[str, RenderedBehaviorPrompt, int]] = []
    for protocol_id, supplied_prompts in prompts_by_protocol.items():
        protocol = registry.protocol(protocol_id)
        prompts = list(supplied_prompts)
        grouped: dict[str, list[RenderedBehaviorPrompt]] = {
            probe_id: [] for probe_id in protocol.probe_ids
        }
        seen: set[tuple[str, str]] = set()
        for prompt in prompts:
            if prompt.probe_id not in grouped:
                raise ValueError(
                    f"protocol {protocol_id!r} received unexpected probe {prompt.probe_id!r}"
                )
            key = (prompt.probe_id, prompt.prompt_id)
            if key in seen:
                raise ValueError(f"protocol {protocol_id!r} has duplicate prompt {key}")
            seen.add(key)
            if prompt.metadata.get("fixture_id") != protocol.fixture_id:
                raise ValueError(
                    f"protocol {protocol_id!r} prompt {prompt.prompt_id!r} fixture mismatch: "
                    f"expected {protocol.fixture_id!r}, "
                    f"got {prompt.metadata.get('fixture_id')!r}"
                )
            grouped[prompt.probe_id].append(prompt)
        for probe_id in protocol.probe_ids:
            expected = protocol.prompt_counts[probe_id]
            observed = len(grouped[probe_id])
            if observed != expected:
                raise ValueError(
                    f"protocol {protocol_id!r} probe {probe_id!r} has {observed} prompts, "
                    f"expected {expected}"
                )
            for prompt in grouped[probe_id]:
                _validate_prompt_contract(prompt, protocol_id=protocol_id, model_task=None)
                for sample_id in range(protocol.samples_per_prompt):
                    coordinates.append((protocol_id, prompt, sample_id))
    return coordinates


def audit_response_grid(
    responses: Sequence[BehaviorResponse],
    *,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    cohort_id: str,
    tree_id: str,
    model_id: str,
    base_model_id: str,
    base_model_revision: str,
    model_task: str,
    request_sha256: str,
    expected_provenance: ResponseProvenance | None = None,
) -> ResponseGridAudit:
    """Require exactly one completed response at every expected coordinate and no others."""

    expected_rows = expected_response_coordinates(registry, prompts_by_protocol)
    expected = {
        (protocol_id, prompt.probe_id, prompt.prompt_id, sample_id): (
            prompt,
            registry.protocol(protocol_id).generation_options(
                model_task=model_task, sample_id=sample_id
            ),
        )
        for protocol_id, prompt, sample_id in expected_rows
    }
    observed: dict[tuple[str, str, str, int], BehaviorResponse] = {}
    duplicates = 0
    errors: list[str] = []
    protocol_counts: dict[str, int] = {}
    probe_counts: dict[str, int] = {}
    for row in responses:
        protocol_counts[row.protocol_id] = protocol_counts.get(row.protocol_id, 0) + 1
        probe_key = f"{row.protocol_id}/{row.probe_id}"
        probe_counts[probe_key] = probe_counts.get(probe_key, 0) + 1
        if row.key in observed:
            duplicates += 1
            errors.append(f"duplicate response coordinate: {row.key}")
            continue
        observed[row.key] = row
        expected_value = expected.get(row.key)
        if expected_value is None:
            continue
        prompt, options = expected_value
        errors.extend(
            _response_contract_errors(
                row,
                prompt=prompt,
                options=options,
                cohort_id=cohort_id,
                tree_id=tree_id,
                model_id=model_id,
                base_model_id=base_model_id,
                base_model_revision=base_model_revision,
                model_task=model_task,
                request_sha256=request_sha256,
                expected_provenance=expected_provenance,
            )
        )
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing:
        errors.append(f"missing {len(missing)} response coordinates; first={missing[0]}")
    if extra:
        errors.append(f"unexpected {len(extra)} response coordinates; first={extra[0]}")
    return ResponseGridAudit(
        valid=not errors,
        n_expected=len(expected),
        n_observed=len(responses),
        n_missing=len(missing),
        n_extra=len(extra),
        n_duplicates=duplicates,
        n_errors=len(errors),
        protocol_counts=dict(sorted(protocol_counts.items())),
        probe_counts=dict(sorted(probe_counts.items())),
        errors=tuple(errors),
    )


def _response_contract_errors(
    row: BehaviorResponse,
    *,
    prompt: RenderedBehaviorPrompt,
    options: Mapping[str, Any],
    cohort_id: str,
    tree_id: str,
    model_id: str,
    base_model_id: str,
    base_model_revision: str,
    model_task: str,
    request_sha256: str,
    expected_provenance: ResponseProvenance | None,
) -> list[str]:
    errors: list[str] = []
    expected_values = {
        "cohort_id": cohort_id,
        "tree_id": tree_id,
        "model_id": model_id,
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "model_task": model_task,
        "source_index": prompt.source_index,
        "source_row_sha256": prompt.source_row_sha256,
        "prompt_sha256": prompt.prompt_sha256,
        "sample_id": options["sample_id"],
        "draw_seed": options["seed"],
        "generation_batch_size": options["generation_batch_size"],
        "seed_scope": options["seed_scope"],
        "do_sample": options["do_sample"],
        "temperature": options["temperature"],
        "top_p": options["top_p"],
        "min_new_tokens": options["min_new_tokens"],
        "max_new_tokens": options["max_new_tokens"],
        "empty_policy": options["empty_policy"],
        "reference": prompt.reference,
        "fixture_id": prompt.metadata.get("fixture_id"),
        "fixture_sha256": prompt.metadata.get("fixture_sha256"),
        "source_indices_sha256": prompt.metadata.get("source_indices_sha256"),
        "dataset": prompt.metadata.get("dataset"),
        "dataset_config": prompt.metadata.get("dataset_config"),
        "dataset_split": prompt.metadata.get("split"),
        "dataset_revision": prompt.metadata.get("dataset_revision"),
    }
    for field, expected in expected_values.items():
        observed = getattr(row, field)
        if observed != expected:
            errors.append(f"{row.key} {field} mismatch: expected {expected!r}, got {observed!r}")
    if row.schema_version != RESPONSE_SCHEMA_VERSION:
        errors.append(f"{row.key} schema_version mismatch")
    if not row.completed:
        errors.append(f"{row.key} is not marked completed")
    if row.empty != (row.text == ""):
        errors.append(f"{row.key} empty flag does not match exact decoded text")
    if row.response_sha256 != hashlib.sha256(row.text.encode("utf-8")).hexdigest():
        errors.append(f"{row.key} response_sha256 mismatch")
    if row.provenance.request_sha256 != request_sha256:
        errors.append(f"{row.key} request provenance drift")
    if expected_provenance is not None and row.provenance != expected_provenance:
        errors.append(f"{row.key} detailed response provenance drift")
    return errors


def _rendered_prompt_from_dict(raw: Mapping[str, Any], *, context: str) -> RenderedBehaviorPrompt:
    required = {
        "probe_id",
        "prompt_id",
        "prompt",
        "reference",
        "source_index",
        "source_row_sha256",
        "prompt_sha256",
        "metadata",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"{context} is missing prompt fields: {missing}")
    metadata = raw["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} metadata must be an object")
    prompt = RenderedBehaviorPrompt(
        probe_id=_nonempty(raw["probe_id"], f"{context}.probe_id"),
        prompt_id=_nonempty(raw["prompt_id"], f"{context}.prompt_id"),
        prompt=_string(raw["prompt"], f"{context}.prompt"),
        reference=_optional_string(raw["reference"], f"{context}.reference"),
        source_index=_integer(raw["source_index"], f"{context}.source_index"),
        source_row_sha256=_sha256_value(
            raw["source_row_sha256"], f"{context}.source_row_sha256"
        ),
        prompt_sha256=_sha256_value(raw["prompt_sha256"], f"{context}.prompt_sha256"),
        metadata=dict(metadata),
    )
    _validate_prompt_contract(prompt, protocol_id=None, model_task=None)
    return prompt


def _validate_prompt_contract(
    prompt: RenderedBehaviorPrompt,
    *,
    protocol_id: str | None,
    model_task: str | None,
) -> None:
    actual_prompt_sha = hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest()
    if prompt.prompt_sha256 != actual_prompt_sha:
        raise ValueError(f"prompt {prompt.prompt_id!r} prompt_sha256 mismatch")
    metadata = prompt.metadata
    for key in ("fixture_id", "fixture_sha256", "source_indices_sha256", "model_task"):
        if key not in metadata:
            raise ValueError(f"prompt {prompt.prompt_id!r} metadata lacks {key!r}")
    if metadata.get("source_row_sha256") != prompt.source_row_sha256:
        raise ValueError(f"prompt {prompt.prompt_id!r} source-row hash metadata mismatch")
    if metadata.get("prompt_sha256") != prompt.prompt_sha256:
        raise ValueError(f"prompt {prompt.prompt_id!r} prompt hash metadata mismatch")
    _metadata_sha256(metadata, "fixture_sha256")
    _metadata_sha256(metadata, "source_indices_sha256")
    if protocol_id is not None and metadata.get("protocol_id") not in (None, protocol_id):
        raise ValueError(f"prompt {prompt.prompt_id!r} protocol metadata mismatch")
    if model_task is not None and metadata.get("model_task") != model_task:
        raise ValueError(f"prompt {prompt.prompt_id!r} model_task metadata mismatch")


def _response_from_dict(raw: Mapping[str, Any], *, context: str) -> BehaviorResponse:
    try:
        provenance_raw = raw["provenance"]
        if not isinstance(provenance_raw, dict):
            raise ValueError(f"{context}.provenance must be an object")
        provenance = ResponseProvenance(**provenance_raw)
        values = dict(raw)
        values["provenance"] = provenance
        response = BehaviorResponse(**values)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid behavior response at {context}: {exc}") from exc
    for name, value in response.provenance.to_dict().items():
        if name.endswith("sha256"):
            _sha256_value(value, f"{context}.provenance.{name}")
    _sha256_value(response.source_row_sha256, f"{context}.source_row_sha256")
    _sha256_value(response.prompt_sha256, f"{context}.prompt_sha256")
    _sha256_value(response.response_sha256, f"{context}.response_sha256")
    return response


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str:
    if key not in metadata:
        raise ValueError(f"prompt metadata lacks {key!r}")
    return _nonempty(metadata[key], f"prompt metadata {key}")


def _metadata_optional_string(metadata: Mapping[str, Any], key: str) -> str | None:
    if key not in metadata:
        raise ValueError(f"prompt metadata lacks {key!r}")
    return _optional_string(metadata[key], f"prompt metadata {key}")


def _metadata_sha256(metadata: Mapping[str, Any], key: str) -> str:
    if key not in metadata:
        raise ValueError(f"prompt metadata lacks {key!r}")
    return _sha256_value(metadata[key], f"prompt metadata {key}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _nonempty(value: Any, context: str) -> str:
    result = _string(value, context)
    if not result:
        raise ValueError(f"{context} must be non-empty")
    return result


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _sha256_value(value: Any, context: str) -> str:
    result = _nonempty(value, context)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return result


def _sha256_or_revision(value: Any, context: str) -> str:
    result = _nonempty(value, context)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", result):
        raise ValueError(f"{context} must be a pinned lowercase commit or SHA256")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _positive_integer(value: Any, context: str) -> int:
    result = _integer(value, context)
    if result <= 0:
        raise ValueError(f"{context} must be positive")
    return result


def _exact_text(value: Any, context: str, expected: str) -> str:
    result = _nonempty(value, context)
    if result != expected:
        raise ValueError(f"{context} must be {expected!r}")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _optional_number(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number or null")
    return float(value)
