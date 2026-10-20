"""Fail-closed contracts for the corrected held-out behavioral panels.

This module deliberately stops at fixture validation and prompt materialization.  It does not
download datasets, load checkpoints, infer an architecture from an adapter, or run generation.
Those operations consume the validated objects produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MODEL_TASKS = ("causal_lm", "seq2seq")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class DatasetPin:
    name: str
    config: str | None
    split: str
    revision: str


@dataclass(frozen=True)
class SourceSelection:
    method: str
    seed: int
    source_row_count: int | None
    n_prompts: int
    source_indices: tuple[int, ...]
    source_indices_sha256: str


@dataclass(frozen=True)
class ProbeFixture:
    probe_id: str
    dataset: DatasetPin
    eligibility: Mapping[str, Any] | None
    selection: SourceSelection


@dataclass(frozen=True)
class SourceArtifactPin:
    location: str
    sha256: str


@dataclass(frozen=True)
class SourceFixture:
    fixture_id: str
    source_prompt_artifact: SourceArtifactPin | None
    probes: Mapping[str, ProbeFixture]
    path: Path
    sha256: str


@dataclass(frozen=True)
class ModelPin:
    model_id: str
    revision: str


@dataclass(frozen=True)
class AnalysisContract:
    primary_empty_policy: str
    sensitivity_empty_policy: str
    natural_language_only: bool


@dataclass(frozen=True)
class ArchitectureContract:
    max_new_tokens: int
    prompt_templates: Mapping[str, str]


@dataclass(frozen=True)
class BehaviorProtocol:
    protocol_id: str
    fixture_id: str
    probe_ids: tuple[str, ...]
    prompt_counts: Mapping[str, int]
    samples_per_prompt: int
    base_seed: int
    draw_seeds: tuple[int, ...]
    generation_batch_size: int
    seed_scope: str
    do_sample: bool
    temperature: float | None
    top_p: float | None
    min_new_tokens: int
    empty_policy: str
    analysis: AnalysisContract
    architectures: Mapping[str, ArchitectureContract]

    def generation_options(self, *, model_task: str, sample_id: int) -> dict[str, Any]:
        """Return one draw's exact, architecture-specific generation declaration."""

        task = _explicit_model_task(model_task)
        if task not in self.architectures:
            raise ValueError(f"protocol {self.protocol_id!r} has no {task!r} architecture")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise ValueError("sample_id must be an integer")
        if sample_id < 0 or sample_id >= self.samples_per_prompt:
            raise ValueError(
                f"sample_id {sample_id} is outside [0, {self.samples_per_prompt})"
            )
        architecture = self.architectures[task]
        return {
            "sample_id": sample_id,
            "seed": self.draw_seeds[sample_id],
            "generation_batch_size": self.generation_batch_size,
            "seed_scope": self.seed_scope,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": architecture.max_new_tokens,
            "empty_policy": self.empty_policy,
        }


@dataclass(frozen=True)
class RenderedBehaviorPrompt:
    probe_id: str
    prompt_id: str
    prompt: str
    reference: str | None
    source_index: int
    source_row_sha256: str
    prompt_sha256: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "probe_id": self.probe_id,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "source_index": self.source_index,
            "source_row_sha256": self.source_row_sha256,
            "prompt_sha256": self.prompt_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BehaviorProtocolRegistry:
    model_pins: Mapping[str, ModelPin]
    embedding_pin: ModelPin
    fixtures: Mapping[str, SourceFixture]
    protocols: Mapping[str, BehaviorProtocol]
    path: Path
    sha256: str

    def protocol(self, protocol_id: str) -> BehaviorProtocol:
        try:
            return self.protocols[protocol_id]
        except KeyError as exc:
            raise ValueError(f"unknown behavior protocol: {protocol_id!r}") from exc

    def validate_model_binding(
        self,
        *,
        model_task: str,
        model_id: str,
        model_revision: str,
    ) -> ModelPin:
        """Validate an explicit model binding; adapter paths are intentionally irrelevant."""

        task = _explicit_model_task(model_task)
        pin = self.model_pins[task]
        if model_id != pin.model_id:
            raise ValueError(
                f"{task} model_id mismatch: expected {pin.model_id!r}, got {model_id!r}"
            )
        if model_revision != pin.revision:
            raise ValueError(
                f"{task} model revision mismatch: expected {pin.revision}, got {model_revision}"
            )
        return pin

    def materialize_probe(
        self,
        *,
        protocol_id: str,
        probe_id: str,
        rows: Sequence[Mapping[str, Any]] | Mapping[int, Mapping[str, Any]],
        model_task: str,
        dataset_name: str,
        dataset_config: str | None,
        dataset_revision: str,
    ) -> list[RenderedBehaviorPrompt]:
        """Render the exact frozen source-index panel from caller-supplied rows."""

        protocol = self.protocol(protocol_id)
        task = _explicit_model_task(model_task)
        if probe_id not in protocol.probe_ids:
            raise ValueError(f"probe {probe_id!r} is not part of protocol {protocol_id!r}")
        fixture = self.fixtures[protocol.fixture_id]
        probe = fixture.probes[probe_id]
        _validate_dataset_binding(
            probe.dataset,
            name=dataset_name,
            config=dataset_config,
            revision=dataset_revision,
        )
        architecture = protocol.architectures[task]
        template = architecture.prompt_templates[probe_id]
        materialized: list[RenderedBehaviorPrompt] = []
        for ordinal, source_index in enumerate(probe.selection.source_indices):
            row = _source_row(rows, source_index, probe_id=probe_id)
            _validate_eligibility(row, probe, source_index=source_index)
            prompt, reference = render_probe_prompt(
                probe_id=probe_id,
                model_task=task,
                source_row=row,
                template=template,
            )
            source_row_sha256 = _canonical_json_sha256(
                row, context=f"{probe_id} source row {source_index}"
            )
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            materialized.append(
                RenderedBehaviorPrompt(
                    probe_id=probe_id,
                    prompt_id=f"{probe_id}-{ordinal:04d}",
                    prompt=prompt,
                    reference=reference,
                    source_index=source_index,
                    source_row_sha256=source_row_sha256,
                    prompt_sha256=prompt_sha256,
                    metadata={
                        "schema_version": SCHEMA_VERSION,
                        "fixture_id": fixture.fixture_id,
                        "dataset": probe.dataset.name,
                        "dataset_config": probe.dataset.config,
                        "split": probe.dataset.split,
                        "dataset_revision": probe.dataset.revision,
                        "source_index": source_index,
                        "source_row_sha256": source_row_sha256,
                        "prompt_sha256": prompt_sha256,
                        "source_indices_sha256": probe.selection.source_indices_sha256,
                        "fixture_sha256": fixture.sha256,
                        "model_task": task,
                        "selection_seed": probe.selection.seed,
                    },
                )
            )
        expected = protocol.prompt_counts[probe_id]
        if len(materialized) != expected:
            raise ValueError(
                f"materialized {len(materialized)} {probe_id} prompts, expected {expected}"
            )
        return materialized


def load_behavior_protocol_registry(path: str | Path) -> BehaviorProtocolRegistry:
    """Load the versioned registry and all SHA-pinned compact source fixtures."""

    registry_path = Path(path).resolve()
    raw_bytes = registry_path.read_bytes()
    raw = _decode_json_object(raw_bytes, context=str(registry_path))
    _exact_keys(
        raw,
        {"schema_version", "model_pins", "embedding_pin", "fixtures", "protocols"},
        context="behavior registry",
    )
    _schema_version(raw.get("schema_version"), context="behavior registry")

    model_pin_rows = _mapping(raw["model_pins"], context="model_pins")
    if set(model_pin_rows) != set(MODEL_TASKS):
        raise ValueError(f"model_pins must contain exactly {list(MODEL_TASKS)}")
    model_pins = {
        model_task: _parse_model_pin(value, context=f"model_pins.{model_task}")
        for model_task, value in model_pin_rows.items()
    }
    embedding_pin = _parse_model_pin(raw["embedding_pin"], context="embedding_pin")

    fixture_specs = _mapping(raw["fixtures"], context="fixtures")
    if not fixture_specs:
        raise ValueError("fixtures must be non-empty")
    fixtures: dict[str, SourceFixture] = {}
    for fixture_id, fixture_spec_value in fixture_specs.items():
        fixture_spec = _mapping(fixture_spec_value, context=f"fixtures.{fixture_id}")
        _exact_keys(fixture_spec, {"path", "sha256"}, context=f"fixtures.{fixture_id}")
        fixture_path = (registry_path.parent / _string(fixture_spec["path"], "fixture path")).resolve()
        fixture = load_source_fixture(
            fixture_path,
            expected_sha256=_sha256(fixture_spec["sha256"], context=f"fixtures.{fixture_id}"),
        )
        if fixture.fixture_id != fixture_id:
            raise ValueError(
                f"fixture key/id mismatch: registry has {fixture_id!r}, file has "
                f"{fixture.fixture_id!r}"
            )
        fixtures[fixture_id] = fixture

    protocol_rows = _mapping(raw["protocols"], context="protocols")
    if not protocol_rows:
        raise ValueError("protocols must be non-empty")
    protocols = {
        protocol_id: _parse_protocol(
            protocol_id,
            value,
            fixtures=fixtures,
            model_tasks=tuple(model_pins),
        )
        for protocol_id, value in protocol_rows.items()
    }
    return BehaviorProtocolRegistry(
        model_pins=model_pins,
        embedding_pin=embedding_pin,
        fixtures=fixtures,
        protocols=protocols,
        path=registry_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_source_fixture(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> SourceFixture:
    fixture_path = Path(path).resolve()
    raw_bytes = fixture_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, context=f"fixture {fixture_path}")
        if observed_sha256 != expected:
            raise ValueError(
                f"fixture sha256 mismatch for {fixture_path}: expected {expected}, "
                f"got {observed_sha256}"
            )
    raw = _decode_json_object(raw_bytes, context=str(fixture_path))
    _exact_keys(
        raw,
        {"schema_version", "fixture_id", "source_prompt_artifact", "probes"},
        context=f"fixture {fixture_path}",
    )
    _schema_version(raw.get("schema_version"), context=f"fixture {fixture_path}")
    fixture_id = _string(raw["fixture_id"], "fixture_id")

    artifact_raw = raw["source_prompt_artifact"]
    source_artifact = None
    if artifact_raw is not None:
        artifact = _mapping(artifact_raw, context=f"fixture {fixture_id}.source_prompt_artifact")
        _exact_keys(
            artifact,
            {"location", "sha256"},
            context=f"fixture {fixture_id}.source_prompt_artifact",
        )
        source_artifact = SourceArtifactPin(
            location=_string(artifact["location"], "source artifact location"),
            sha256=_sha256(artifact["sha256"], context="source prompt artifact"),
        )

    probe_rows = raw["probes"]
    if not isinstance(probe_rows, list) or not probe_rows:
        raise ValueError(f"fixture {fixture_id} probes must be a non-empty list")
    probes: dict[str, ProbeFixture] = {}
    for index, probe_value in enumerate(probe_rows):
        probe = _parse_probe_fixture(probe_value, context=f"fixture {fixture_id}.probes[{index}]")
        if probe.probe_id in probes:
            raise ValueError(f"fixture {fixture_id} has duplicate probe {probe.probe_id!r}")
        probes[probe.probe_id] = probe
    return SourceFixture(
        fixture_id=fixture_id,
        source_prompt_artifact=source_artifact,
        probes=probes,
        path=fixture_path,
        sha256=observed_sha256,
    )


def render_probe_prompt(
    *,
    probe_id: str,
    model_task: str,
    source_row: Mapping[str, Any],
    template: str,
) -> tuple[str, str | None]:
    """Render one audited probe row with no architecture inference."""

    _explicit_model_task(model_task)
    row = _mapping(source_row, context=f"{probe_id} source row")
    values: dict[str, str]
    reference: str | None
    if probe_id == "translation":
        translation = row.get("translation", row)
        translation = _mapping(translation, context="translation row")
        values = {"en": _nonempty_text(translation.get("en"), "translation.en")}
        reference = _nonempty_text(translation.get("fr"), "translation.fr")
    elif probe_id == "hellaswag":
        endings_value = row.get("endings")
        if not isinstance(endings_value, list) or not endings_value:
            raise ValueError("hellaswag.endings must be a non-empty list")
        endings = [_nonempty_text(value, "hellaswag ending") for value in endings_value]
        label = _index_integer(row.get("label"), "hellaswag.label")
        if label < 0 or label >= len(endings):
            raise ValueError("hellaswag.label is outside the endings list")
        values = {
            "activity": _nonempty_text(row.get("activity_label"), "hellaswag.activity_label"),
            "context": _nonempty_text(row.get("ctx"), "hellaswag.ctx"),
        }
        reference = endings[label]
    elif probe_id == "mmlu":
        choices = _text_list(row.get("choices"), context="mmlu.choices")
        answer = _integer(row.get("answer"), "mmlu.answer")
        if answer < 0 or answer >= len(choices):
            raise ValueError("mmlu.answer is outside the choices list")
        labels = list("ABCDEFGH"[: len(choices)])
        if len(labels) != len(choices):
            raise ValueError("mmlu supports at most eight choices")
        values = {
            "question": _nonempty_text(row.get("question"), "mmlu.question"),
            "choices_block": "\n".join(
                f"{label}) {choice}" for label, choice in zip(labels, choices)
            ),
        }
        reference = choices[answer]
    elif probe_id == "arc_challenge":
        choices_row = _mapping(row.get("choices"), context="arc_challenge.choices")
        _exact_keys(choices_row, {"label", "text"}, context="arc_challenge.choices")
        labels = _label_list(choices_row["label"], context="arc_challenge.choices.label")
        choices = _text_list(choices_row["text"], context="arc_challenge.choices.text")
        if len(labels) != len(choices):
            raise ValueError("arc_challenge choice labels/text have different lengths")
        answer_key = _nonempty_text(row.get("answerKey"), "arc_challenge.answerKey")
        if answer_key not in labels:
            raise ValueError("arc_challenge.answerKey is absent from choice labels")
        values = {
            "question": _nonempty_text(row.get("question"), "arc_challenge.question"),
            "choices_block": "\n".join(
                f"{label}) {choice}" for label, choice in zip(labels, choices)
            ),
        }
        reference = choices[labels.index(answer_key)]
    elif probe_id == "truthfulqa":
        values = {"question": _nonempty_text(row.get("question"), "truthfulqa.question")}
        reference = _nonempty_text(row.get("best_answer"), "truthfulqa.best_answer")
    elif probe_id == "dolly_open_ended":
        instruction = _nonempty_text(row.get("instruction"), "dolly.instruction")
        context = str(row.get("context", "")).strip()
        if model_task == "causal_lm":
            context_block = f"\n\nContext: {context}" if context else ""
        else:
            context_block = f" context: {context}" if context else ""
        values = {"instruction": instruction, "context_block": context_block}
        response = row.get("response")
        reference = None if response is None or not str(response).strip() else str(response)
    else:
        raise ValueError(f"unsupported behavior probe: {probe_id!r}")
    try:
        prompt = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid {probe_id}/{model_task} prompt template: {exc}") from exc
    if not prompt.strip():
        raise ValueError(f"{probe_id}/{model_task} rendered an empty prompt")
    return prompt, reference


def response_is_eligible(
    protocol: BehaviorProtocol,
    response: str,
    *,
    analysis: str,
) -> bool:
    """Apply the declared primary or exact-empty sensitivity policy.

    Label-only and fragment outputs remain eligible.  This contract never silently substitutes a
    ``natural_language_only`` filter for the requested exact-empty sensitivity analysis.
    """

    if not isinstance(response, str):
        raise ValueError("behavior response must be a string")
    if protocol.analysis.natural_language_only:
        raise ValueError("corrected behavior protocols must not use natural_language_only")
    if analysis == "primary":
        if protocol.analysis.primary_empty_policy != "preserve":
            raise ValueError("unsupported primary empty policy")
        return True
    if analysis == "sensitivity":
        if protocol.analysis.sensitivity_empty_policy != "exclude_exact_empty":
            raise ValueError("unsupported sensitivity empty policy")
        return bool(response.strip())
    raise ValueError("analysis must be 'primary' or 'sensitivity'")


def _parse_model_pin(value: Any, *, context: str) -> ModelPin:
    raw = _mapping(value, context=context)
    _exact_keys(raw, {"model_id", "revision"}, context=context)
    revision = _string(raw["revision"], f"{context}.revision")
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"{context}.revision must be a pinned 40-character commit")
    return ModelPin(model_id=_string(raw["model_id"], f"{context}.model_id"), revision=revision)


def _parse_probe_fixture(value: Any, *, context: str) -> ProbeFixture:
    raw = _mapping(value, context=context)
    _exact_keys(raw, {"probe_id", "dataset", "eligibility", "selection"}, context=context)
    probe_id = _string(raw["probe_id"], f"{context}.probe_id")
    dataset_raw = _mapping(raw["dataset"], context=f"{context}.dataset")
    _exact_keys(dataset_raw, {"name", "config", "split", "revision"}, context=f"{context}.dataset")
    revision = _string(dataset_raw["revision"], f"{context}.dataset.revision")
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"{context}.dataset.revision must be a pinned 40-character commit")
    config_value = dataset_raw["config"]
    config = None if config_value is None else _string(config_value, f"{context}.dataset.config")
    dataset = DatasetPin(
        name=_string(dataset_raw["name"], f"{context}.dataset.name"),
        config=config,
        split=_string(dataset_raw["split"], f"{context}.dataset.split"),
        revision=revision,
    )

    eligibility_value = raw["eligibility"]
    eligibility = None
    if eligibility_value is not None:
        eligibility_raw = _mapping(eligibility_value, context=f"{context}.eligibility")
        _exact_keys(
            eligibility_raw,
            {"field", "allowed_values", "require_nonempty"},
            context=f"{context}.eligibility",
        )
        eligibility = {
            "field": _string(eligibility_raw["field"], f"{context}.eligibility.field"),
            "allowed_values": _text_list(
                eligibility_raw["allowed_values"], context=f"{context}.eligibility.allowed_values"
            ),
            "require_nonempty": _text_list(
                eligibility_raw["require_nonempty"],
                context=f"{context}.eligibility.require_nonempty",
            ),
        }

    selection_raw = _mapping(raw["selection"], context=f"{context}.selection")
    _exact_keys(
        selection_raw,
        {
            "method",
            "seed",
            "source_row_count",
            "n_prompts",
            "source_indices",
            "source_indices_sha256",
        },
        context=f"{context}.selection",
    )
    indices_value = selection_raw["source_indices"]
    if not isinstance(indices_value, list):
        raise ValueError(f"{context}.selection.source_indices must be a list")
    indices = tuple(
        _integer(index, f"{context}.selection.source_indices[{position}]")
        for position, index in enumerate(indices_value)
    )
    if any(index < 0 for index in indices):
        raise ValueError(f"{context}.selection.source_indices must be non-negative")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{context}.selection.source_indices contains duplicates")
    n_prompts = _integer(selection_raw["n_prompts"], f"{context}.selection.n_prompts")
    if n_prompts <= 0 or len(indices) != n_prompts:
        raise ValueError(
            f"{context}.selection requires exactly n_prompts={n_prompts} source indices"
        )
    source_count_value = selection_raw["source_row_count"]
    source_count = (
        None
        if source_count_value is None
        else _integer(source_count_value, f"{context}.selection.source_row_count")
    )
    if source_count is not None:
        if source_count <= 0 or max(indices) >= source_count:
            raise ValueError(f"{context}.selection source index exceeds source_row_count")
    expected_indices_sha = _sha256(
        selection_raw["source_indices_sha256"],
        context=f"{context}.selection.source_indices_sha256",
    )
    observed_indices_sha = _canonical_indices_sha256(indices)
    if observed_indices_sha != expected_indices_sha:
        raise ValueError(
            f"{context}.selection source_indices sha256 mismatch: expected "
            f"{expected_indices_sha}, got {observed_indices_sha}"
        )
    selection = SourceSelection(
        method=_string(selection_raw["method"], f"{context}.selection.method"),
        seed=_integer(selection_raw["seed"], f"{context}.selection.seed"),
        source_row_count=source_count,
        n_prompts=n_prompts,
        source_indices=indices,
        source_indices_sha256=expected_indices_sha,
    )
    return ProbeFixture(
        probe_id=probe_id,
        dataset=dataset,
        eligibility=eligibility,
        selection=selection,
    )


def _parse_protocol(
    protocol_id: str,
    value: Any,
    *,
    fixtures: Mapping[str, SourceFixture],
    model_tasks: tuple[str, ...],
) -> BehaviorProtocol:
    context = f"protocols.{protocol_id}"
    raw = _mapping(value, context=context)
    _exact_keys(
        raw,
        {
            "fixture_id",
            "probe_ids",
            "prompt_counts",
            "samples_per_prompt",
            "base_seed",
            "draw_seeds",
            "generation_batch_size",
            "seed_scope",
            "do_sample",
            "temperature",
            "top_p",
            "min_new_tokens",
            "empty_policy",
            "analysis",
            "architectures",
        },
        context=context,
    )
    fixture_id = _string(raw["fixture_id"], f"{context}.fixture_id")
    if fixture_id not in fixtures:
        raise ValueError(f"{context} names unknown fixture {fixture_id!r}")
    fixture = fixtures[fixture_id]
    probe_ids = tuple(_text_list(raw["probe_ids"], context=f"{context}.probe_ids"))
    if len(probe_ids) != len(set(probe_ids)) or set(probe_ids) != set(fixture.probes):
        raise ValueError(f"{context}.probe_ids must match its fixture probes exactly")
    counts_raw = _mapping(raw["prompt_counts"], context=f"{context}.prompt_counts")
    if set(counts_raw) != set(probe_ids):
        raise ValueError(f"{context}.prompt_counts must match probe_ids exactly")
    prompt_counts = {
        probe_id: _integer(counts_raw[probe_id], f"{context}.prompt_counts.{probe_id}")
        for probe_id in probe_ids
    }
    for probe_id, count in prompt_counts.items():
        if count <= 0 or count != fixture.probes[probe_id].selection.n_prompts:
            raise ValueError(f"{context} count for {probe_id} does not match its fixture")

    samples_per_prompt = _integer(raw["samples_per_prompt"], f"{context}.samples_per_prompt")
    if samples_per_prompt <= 0:
        raise ValueError(f"{context}.samples_per_prompt must be positive")
    base_seed = _integer(raw["base_seed"], f"{context}.base_seed")
    draw_seeds_value = raw["draw_seeds"]
    if not isinstance(draw_seeds_value, list):
        raise ValueError(f"{context}.draw_seeds must be a list")
    draw_seeds = tuple(
        _integer(seed, f"{context}.draw_seeds[{index}]")
        for index, seed in enumerate(draw_seeds_value)
    )
    expected_seeds = tuple(base_seed + sample_id for sample_id in range(samples_per_prompt))
    if draw_seeds != expected_seeds:
        raise ValueError(
            f"{context}.draw_seeds must be exactly base_seed + sample_id: {expected_seeds}"
        )
    generation_batch_size = _integer(
        raw["generation_batch_size"], f"{context}.generation_batch_size"
    )
    if generation_batch_size <= 0:
        raise ValueError(f"{context}.generation_batch_size must be positive")
    seed_scope = _string(raw["seed_scope"], f"{context}.seed_scope")
    if seed_scope != "probe_draw":
        raise ValueError(
            f"{context}.seed_scope must be 'probe_draw' so each probe/draw is seeded "
            "once before ordered batched generation"
        )
    do_sample = raw["do_sample"]
    if not isinstance(do_sample, bool):
        raise ValueError(f"{context}.do_sample must be boolean")
    temperature = _optional_float(raw["temperature"], f"{context}.temperature")
    top_p = _optional_float(raw["top_p"], f"{context}.top_p")
    if do_sample:
        if temperature is None or temperature <= 0:
            raise ValueError(f"{context} sampled generation requires positive temperature")
        if top_p is None or not 0 < top_p <= 1:
            raise ValueError(f"{context} sampled generation requires top_p in (0, 1]")
    elif temperature is not None or top_p is not None:
        raise ValueError(f"{context} greedy generation requires null temperature and top_p")
    min_new_tokens = _integer(raw["min_new_tokens"], f"{context}.min_new_tokens")
    if min_new_tokens < 0:
        raise ValueError(f"{context}.min_new_tokens must be non-negative")
    empty_policy = _string(raw["empty_policy"], f"{context}.empty_policy")
    if empty_policy != "preserve":
        raise ValueError(f"{context}.empty_policy must be 'preserve'")

    analysis_raw = _mapping(raw["analysis"], context=f"{context}.analysis")
    _exact_keys(
        analysis_raw,
        {"primary_empty_policy", "sensitivity_empty_policy", "natural_language_only"},
        context=f"{context}.analysis",
    )
    natural_language_only = analysis_raw["natural_language_only"]
    if not isinstance(natural_language_only, bool):
        raise ValueError(f"{context}.analysis.natural_language_only must be boolean")
    analysis = AnalysisContract(
        primary_empty_policy=_string(
            analysis_raw["primary_empty_policy"], f"{context}.analysis.primary_empty_policy"
        ),
        sensitivity_empty_policy=_string(
            analysis_raw["sensitivity_empty_policy"],
            f"{context}.analysis.sensitivity_empty_policy",
        ),
        natural_language_only=natural_language_only,
    )
    if analysis != AnalysisContract("preserve", "exclude_exact_empty", False):
        raise ValueError(
            f"{context}.analysis must preserve primary empties, exclude only exact-empty "
            "sensitivity rows, and keep natural_language_only false"
        )

    architecture_rows = _mapping(raw["architectures"], context=f"{context}.architectures")
    if set(architecture_rows) != set(model_tasks):
        raise ValueError(f"{context}.architectures must contain exactly {list(model_tasks)}")
    architectures: dict[str, ArchitectureContract] = {}
    for model_task, architecture_value in architecture_rows.items():
        architecture_raw = _mapping(
            architecture_value, context=f"{context}.architectures.{model_task}"
        )
        _exact_keys(
            architecture_raw,
            {"max_new_tokens", "prompt_templates"},
            context=f"{context}.architectures.{model_task}",
        )
        max_new_tokens = _integer(
            architecture_raw["max_new_tokens"],
            f"{context}.architectures.{model_task}.max_new_tokens",
        )
        if max_new_tokens <= 0 or min_new_tokens > max_new_tokens:
            raise ValueError(f"{context}/{model_task} has invalid new-token bounds")
        templates_raw = _mapping(
            architecture_raw["prompt_templates"],
            context=f"{context}.architectures.{model_task}.prompt_templates",
        )
        if set(templates_raw) != set(probe_ids):
            raise ValueError(f"{context}/{model_task} prompt templates must match probe_ids")
        architectures[model_task] = ArchitectureContract(
            max_new_tokens=max_new_tokens,
            prompt_templates={
                probe_id: _string(
                    templates_raw[probe_id],
                    f"{context}.architectures.{model_task}.prompt_templates.{probe_id}",
                )
                for probe_id in probe_ids
            },
        )
    return BehaviorProtocol(
        protocol_id=protocol_id,
        fixture_id=fixture_id,
        probe_ids=probe_ids,
        prompt_counts=prompt_counts,
        samples_per_prompt=samples_per_prompt,
        base_seed=base_seed,
        draw_seeds=draw_seeds,
        generation_batch_size=generation_batch_size,
        seed_scope=seed_scope,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        min_new_tokens=min_new_tokens,
        empty_policy=empty_policy,
        analysis=analysis,
        architectures=architectures,
    )


def _validate_dataset_binding(
    expected: DatasetPin,
    *,
    name: str,
    config: str | None,
    revision: str,
) -> None:
    if name != expected.name:
        raise ValueError(f"dataset name mismatch: expected {expected.name!r}, got {name!r}")
    if config != expected.config:
        raise ValueError(f"dataset config mismatch: expected {expected.config!r}, got {config!r}")
    if revision != expected.revision:
        raise ValueError(
            f"dataset revision mismatch: expected {expected.revision}, got {revision}"
        )


def _validate_eligibility(
    row: Mapping[str, Any],
    probe: ProbeFixture,
    *,
    source_index: int,
) -> None:
    if probe.eligibility is None:
        return
    field = str(probe.eligibility["field"])
    allowed = set(probe.eligibility["allowed_values"])
    if str(row.get(field, "")) not in allowed:
        raise ValueError(
            f"{probe.probe_id} source row {source_index} fails {field} eligibility"
        )
    for required in probe.eligibility["require_nonempty"]:
        if not str(row.get(required, "")).strip():
            raise ValueError(
                f"{probe.probe_id} source row {source_index} has empty {required!r}"
            )


def _source_row(
    rows: Sequence[Mapping[str, Any]] | Mapping[int, Mapping[str, Any]],
    index: int,
    *,
    probe_id: str,
) -> Mapping[str, Any]:
    try:
        row = rows[index]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"{probe_id} source fixture is missing row index {index}") from exc
    return _mapping(row, context=f"{probe_id} source row {index}")


def _explicit_model_task(model_task: Any) -> str:
    if model_task is None:
        raise ValueError("model_task is required explicitly; it is never inferred from adapters")
    if not isinstance(model_task, str) or model_task not in MODEL_TASKS:
        raise ValueError(f"model_task must be one of {list(MODEL_TASKS)}")
    return model_task


def _decode_json_object(raw: bytes, *, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {context}: {exc}") from exc
    return _mapping(value, context=context)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{context} schema mismatch: missing={missing}, unknown={unknown}")


def _schema_version(value: Any, *, context: str) -> None:
    if value != SCHEMA_VERSION or isinstance(value, bool):
        raise ValueError(f"{context} schema_version must be exactly {SCHEMA_VERSION}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _nonempty_text(value: Any, context: str) -> str:
    text = _string(value, context)
    if not text.strip():
        raise ValueError(f"{context} must be non-empty")
    return text


def _text_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    return [_nonempty_text(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _index_integer(value: Any, context: str) -> int:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return _integer(value, context)


def _label_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    labels: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(f"{context}[{index}] must be a string or integer label")
        text = str(item)
        if not text.strip():
            raise ValueError(f"{context}[{index}] must be non-empty")
        labels.append(text)
    return labels


def _optional_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric or null")
    return float(value)


def _sha256(value: Any, *, context: str) -> str:
    digest = _string(value, f"{context}.sha256")
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{context} must be a lowercase 64-character sha256")
    return digest


def _canonical_indices_sha256(indices: Sequence[int]) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: Any, *, context: str) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not canonical-JSON serializable: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AnalysisContract",
    "ArchitectureContract",
    "BehaviorProtocol",
    "BehaviorProtocolRegistry",
    "DatasetPin",
    "ModelPin",
    "ProbeFixture",
    "RenderedBehaviorPrompt",
    "SourceArtifactPin",
    "SourceFixture",
    "SourceSelection",
    "load_behavior_protocol_registry",
    "load_source_fixture",
    "render_probe_prompt",
    "response_is_eligible",
]
