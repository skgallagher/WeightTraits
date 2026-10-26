"""Exact request and response-grid contracts for all-node behavior inference.

The module is intentionally orchestration-only and pure standard library.  It
does not load models or inspect response files; instead it freezes the exact
node, prompt, runtime, and artifact pins that production execution must verify.
Internal checkpoints remain observed ancestors and never acquire a
``leaf_ordinal`` compatibility alias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Any, Mapping, Sequence

from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    TreeCheckpointManifest,
    read_tree_checkpoint_manifest,
)
from .covariates import (
    TreeNodeCovariates,
    read_node_covariates_payload,
    validate_tree_node_covariates,
)
from .manifest import NodeSpec, build_tree_topology, canonical_sha256


REQUEST_SCHEMA = "weighttraits.all_node_inference_request.v3"
RESPONSE_ROW_SCHEMA = "weighttraits.all_node_response_row.v1"
RECEIPT_SCHEMA = "weighttraits.all_node_response_receipt.v3"
DEVICE = "cuda"
TORCH_DTYPE = "auto"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_METHOD_ARTIFACT = {"full": "model", "lora": "merged"}


@dataclass(frozen=True)
class EndpointShape:
    protocol_id: str
    probe_id: str
    n_prompts: int
    samples_per_prompt: int

    @property
    def n_responses(self) -> int:
        return self.n_prompts * self.samples_per_prompt


TRANSLATION_PROTOCOL = "legacy_translation_200x1_greedy"
MODERN_MC_PROTOCOL = "modern_mc_100x3_sampled"
DOLLY_PROTOCOL = "dolly_open_ended_30x3_sampled"

TRANSLATION = EndpointShape(TRANSLATION_PROTOCOL, "translation", 200, 1)
HELLASWAG = EndpointShape(MODERN_MC_PROTOCOL, "hellaswag", 100, 3)
ARC = EndpointShape(MODERN_MC_PROTOCOL, "arc_challenge", 100, 3)
MMLU = EndpointShape(MODERN_MC_PROTOCOL, "mmlu", 100, 3)
TRUTHFULQA = EndpointShape(MODERN_MC_PROTOCOL, "truthfulqa", 100, 3)
DOLLY = EndpointShape(DOLLY_PROTOCOL, "dolly_open_ended", 30, 3)

FROZEN_COHORT_ENDPOINTS: Mapping[str, tuple[EndpointShape, ...]] = {
    "llama_full": (TRANSLATION, HELLASWAG, ARC, MMLU, TRUTHFULQA, DOLLY),
    "llama_notrans": (TRANSLATION,),
    "flan_full": (TRANSLATION, HELLASWAG, ARC, MMLU, TRUTHFULQA),
    "flan_notrans": (TRANSLATION,),
}
FROZEN_COHORT_MODEL_TASK: Mapping[str, str] = {
    "llama_full": "causal_lm",
    "llama_notrans": "causal_lm",
    "flan_full": "seq2seq",
    "flan_notrans": "seq2seq",
}

BACKEND_PROVENANCE_KEYS = frozenset(
    {
        "backend",
        "device",
        "torch_dtype",
        "local_files_only",
        "tokenizer_model_id",
        "tokenizer_revision",
        "checkpoint",
        "transformers_version",
        "torch_version",
        "decode_skip_special_tokens",
        "decode_clean_up_tokenization_spaces",
        "generation_api",
        "generation_seed_api",
        "tokenizer_padding_side",
        "tokenizer_truncation",
        "tokenizer_max_length",
        "generation_pad_token_id",
    }
)
GENERATION_OPTION_KEYS = frozenset(
    {
        "sample_id",
        "seed",
        "generation_batch_size",
        "seed_scope",
        "do_sample",
        "temperature",
        "top_p",
        "min_new_tokens",
        "max_new_tokens",
        "empty_policy",
    }
)
FROZEN_COHORT_RESPONSE_COUNTS: Mapping[str, int] = {
    cohort: sum(endpoint.n_responses for endpoint in endpoints)
    for cohort, endpoints in FROZEN_COHORT_ENDPOINTS.items()
}


@dataclass(frozen=True, order=True)
class ResponseCoordinate:
    """One deterministic protocol/probe/prompt/draw coordinate."""

    protocol_id: str
    probe_id: str
    prompt_id: str
    sample_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptPanel:
    """One exact prompt panel and its materialization provenance."""

    protocol_id: str
    probe_id: str
    prompt_ids: tuple[str, ...]
    samples_per_prompt: int
    prompt_artifact: ArtifactPin
    materialization_receipt: ArtifactPin

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "probe_id": self.probe_id,
            "prompt_ids": list(self.prompt_ids),
            "samples_per_prompt": self.samples_per_prompt,
            "prompt_artifact": self.prompt_artifact.to_dict(),
            "materialization_receipt": self.materialization_receipt.to_dict(),
        }


@dataclass(frozen=True)
class GenerationGroup:
    """One seed-once protocol/probe/draw batch over its full prompt panel."""

    group_ordinal: int
    protocol_id: str
    probe_id: str
    sample_id: int
    prompt_ids: tuple[str, ...]
    generation_options: Mapping[str, Any]

    @property
    def coordinates(self) -> tuple[ResponseCoordinate, ...]:
        return tuple(
            ResponseCoordinate(
                self.protocol_id,
                self.probe_id,
                prompt_id,
                self.sample_id,
            )
            for prompt_id in self.prompt_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_ordinal": self.group_ordinal,
            "protocol_id": self.protocol_id,
            "probe_id": self.probe_id,
            "sample_id": self.sample_id,
            "prompt_ids": list(self.prompt_ids),
            "generation_options": dict(self.generation_options),
        }


@dataclass(frozen=True)
class AllNodeInferenceRequest:
    """One fully pinned inference request for one trained node."""

    cohort_key: str
    cohort_id: str
    tree_id: str
    node_id: str
    node_ordinal: int
    parent_id: str
    depth: int
    node_role: str
    population: str
    internal_node_treatment: str
    checkpoint_binding: NodeCheckpointBinding
    checkpoint_binding_sha256: str
    checkpoint_manifest_sha256: str
    topology_sha256: str
    pair_index_sha256: str
    node_covariates: TreeNodeCovariates
    node_covariates_sha256: str
    registry: ArtifactPin
    panels: tuple[PromptPanel, ...]
    model_task: str
    generation_groups: tuple[GenerationGroup, ...]
    generation_groups_sha256: str
    planned_backend_provenance: Mapping[str, Any]
    planned_backend_provenance_sha256: str
    n_expected_responses: int
    coordinates_sha256: str
    base_model_id: str
    base_model_revision: str
    device: str
    torch_dtype: str
    execution_overrides: Mapping[str, Any]

    def coordinates(self) -> tuple[ResponseCoordinate, ...]:
        return _coordinates(self.panels)

    def grouped_coordinates(
        self,
    ) -> tuple[tuple[GenerationGroup, tuple[ResponseCoordinate, ...]], ...]:
        return tuple((group, group.coordinates) for group in self.generation_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_SCHEMA,
            "population": self.population,
            "internal_node_treatment": self.internal_node_treatment,
            "cohort_key": self.cohort_key,
            "cohort_id": self.cohort_id,
            "tree_id": self.tree_id,
            "node_id": self.node_id,
            "node_ordinal": self.node_ordinal,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "node_role": self.node_role,
            "checkpoint_binding": self.checkpoint_binding.to_dict(),
            "checkpoint_binding_sha256": self.checkpoint_binding_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "topology_sha256": self.topology_sha256,
            "pair_index_sha256": self.pair_index_sha256,
            "node_covariates": self.node_covariates.to_dict(),
            "node_covariates_sha256": self.node_covariates_sha256,
            "registry": self.registry.to_dict(),
            "panels": [panel.to_dict() for panel in self.panels],
            "model_task": self.model_task,
            "generation_groups": [group.to_dict() for group in self.generation_groups],
            "generation_groups_sha256": self.generation_groups_sha256,
            "planned_backend_provenance": dict(self.planned_backend_provenance),
            "planned_backend_provenance_sha256": self.planned_backend_provenance_sha256,
            "n_expected_responses": self.n_expected_responses,
            "coordinates_sha256": self.coordinates_sha256,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "execution_overrides": dict(self.execution_overrides),
        }

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ResponseRow:
    """Hash-only receipt row for one decoded response artifact coordinate."""

    request_sha256: str
    cohort_id: str
    tree_id: str
    node_id: str
    protocol_id: str
    probe_id: str
    prompt_id: str
    sample_id: int
    response_sha256: str
    completed: bool

    @property
    def coordinate(self) -> ResponseCoordinate:
        return ResponseCoordinate(
            self.protocol_id, self.probe_id, self.prompt_id, self.sample_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RESPONSE_ROW_SCHEMA, **asdict(self)}


def build_inference_request(
    binding: NodeCheckpointBinding,
    manifest: TreeCheckpointManifest,
    *,
    cohort_key: str,
    registry: ArtifactPin | Mapping[str, Any],
    panels: Sequence[PromptPanel | Mapping[str, Any]],
    planned_backend_provenance: Mapping[str, Any],
    node_covariates: TreeNodeCovariates,
) -> AllNodeInferenceRequest:
    """Freeze one internal-or-leaf node's exact production request."""

    key = _text(cohort_key, "cohort_key")
    if key not in FROZEN_COHORT_ENDPOINTS:
        raise ValueError(f"unknown frozen cohort_key {key!r}")
    verified_manifest = read_tree_checkpoint_manifest(manifest.to_dict())
    node = _binding(binding)
    if node.node_ordinal >= len(verified_manifest.bindings):
        raise ValueError("node ordinal is outside its checkpoint manifest")
    declared = verified_manifest.bindings[node.node_ordinal]
    if canonical_sha256(node.to_dict()) != canonical_sha256(declared.to_dict()):
        raise ValueError("node checkpoint binding differs from its checkpoint manifest")
    if node.internal_node_treatment != INTERNAL_NODE_TREATMENT:
        raise ValueError("internal node treatment drift")
    covariates = validate_tree_node_covariates(
        node_covariates,
        _topology_from_manifest(verified_manifest),
        verified_manifest,
    )
    node_covariate = covariates.nodes[node.node_ordinal]
    expected_covariate_identity = (
        node.node_id,
        node.parent_id,
        node.depth,
        node.node_role,
    )
    observed_covariate_identity = (
        node_covariate.node_id,
        node_covariate.parent_id,
        node_covariate.depth,
        node_covariate.role,
    )
    if observed_covariate_identity != expected_covariate_identity:
        raise ValueError("selected node differs from its node covariate contract")

    active_panels = tuple(_panel(value) for value in panels)
    _validate_frozen_panels(key, active_panels)
    coordinates = _coordinates(active_panels)
    model_task = FROZEN_COHORT_MODEL_TASK[key]
    generation_groups = _generation_groups(
        active_panels,
        model_task=model_task,
    )
    _validate_generation_groups(
        generation_groups,
        panels=active_panels,
        model_task=model_task,
    )
    backend = _backend_provenance(
        planned_backend_provenance,
        binding=node,
        model_task=model_task,
        label="planned_backend_provenance",
    )
    expected = FROZEN_COHORT_RESPONSE_COUNTS[key]
    if len(coordinates) != expected:
        raise AssertionError("frozen response-grid cardinality corruption")
    registry_pin = _pin(registry, "registry")
    return AllNodeInferenceRequest(
        cohort_key=key,
        cohort_id=node.cohort_id,
        tree_id=node.tree_id,
        node_id=node.node_id,
        node_ordinal=node.node_ordinal,
        parent_id=node.parent_id,
        depth=node.depth,
        node_role=node.node_role,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        checkpoint_binding=node,
        checkpoint_binding_sha256=canonical_sha256(node.to_dict()),
        checkpoint_manifest_sha256=verified_manifest.checkpoint_set_sha256,
        topology_sha256=verified_manifest.topology_sha256,
        pair_index_sha256=verified_manifest.pair_index_sha256,
        node_covariates=covariates,
        node_covariates_sha256=covariates.covariates_sha256,
        registry=registry_pin,
        panels=active_panels,
        model_task=model_task,
        generation_groups=generation_groups,
        generation_groups_sha256=canonical_sha256(
            [group.to_dict() for group in generation_groups]
        ),
        planned_backend_provenance=backend,
        planned_backend_provenance_sha256=canonical_sha256(backend),
        n_expected_responses=expected,
        coordinates_sha256=canonical_sha256(
            [coordinate.to_dict() for coordinate in coordinates]
        ),
        base_model_id=node.base_model_id,
        base_model_revision=node.base_model_revision,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        execution_overrides={},
    )


def read_inference_request(payload: Mapping[str, Any]) -> AllNodeInferenceRequest:
    """Recompute and exactly compare a saved request."""

    if not isinstance(payload, Mapping):
        raise ValueError("saved inference request must be a mapping")
    expected_keys = {
        "schema",
        "population",
        "internal_node_treatment",
        "cohort_key",
        "cohort_id",
        "tree_id",
        "node_id",
        "node_ordinal",
        "parent_id",
        "depth",
        "node_role",
        "checkpoint_binding",
        "checkpoint_binding_sha256",
        "checkpoint_manifest_sha256",
        "topology_sha256",
        "pair_index_sha256",
        "node_covariates",
        "node_covariates_sha256",
        "registry",
        "panels",
        "model_task",
        "generation_groups",
        "generation_groups_sha256",
        "planned_backend_provenance",
        "planned_backend_provenance_sha256",
        "n_expected_responses",
        "coordinates_sha256",
        "base_model_id",
        "base_model_revision",
        "device",
        "torch_dtype",
        "execution_overrides",
    }
    if set(payload) != expected_keys:
        raise ValueError(f"inference request keys must be exactly {sorted(expected_keys)!r}")
    if payload.get("schema") != REQUEST_SCHEMA:
        raise ValueError("inference request schema drift")
    node = _binding(payload.get("checkpoint_binding"))
    panels_raw = payload.get("panels")
    if not isinstance(panels_raw, list):
        raise ValueError("inference request panels must be a list")
    panels = tuple(_panel(value) for value in panels_raw)
    key = _text(payload.get("cohort_key"), "cohort_key")
    _validate_frozen_panels(key, panels)
    coordinates = _coordinates(panels)
    model_task = FROZEN_COHORT_MODEL_TASK.get(key)
    if model_task is None:
        raise ValueError(f"unknown frozen cohort_key {key!r}")
    raw_groups = payload.get("generation_groups")
    if not isinstance(raw_groups, list):
        raise ValueError("inference request generation_groups must be a list")
    groups = tuple(_generation_group(value) for value in raw_groups)
    _validate_generation_groups(groups, panels=panels, model_task=model_task)
    backend = _backend_provenance(
        payload.get("planned_backend_provenance"),
        binding=node,
        model_task=model_task,
        label="planned_backend_provenance",
    )
    raw_covariates = payload.get("node_covariates")
    if not isinstance(raw_covariates, Mapping):
        raise ValueError("inference request node_covariates must be a mapping")
    covariates = read_node_covariates_payload(raw_covariates)
    declared_covariates_sha = _sha256(
        payload.get("node_covariates_sha256"), "node_covariates_sha256"
    )
    if covariates.covariates_sha256 != declared_covariates_sha:
        raise ValueError("inference request node covariate hash drift")
    checkpoint_manifest_sha = _sha256(
        payload.get("checkpoint_manifest_sha256"), "checkpoint_manifest_sha256"
    )
    topology_sha = _sha256(payload.get("topology_sha256"), "topology_sha256")
    if (
        covariates.tree_id != node.tree_id
        or covariates.checkpoint_set_sha256 != checkpoint_manifest_sha
        or covariates.topology_sha256 != topology_sha
        or covariates.source_manifest != node.truth_manifest
        or node.node_ordinal >= len(covariates.nodes)
    ):
        raise ValueError("inference request node covariate scope drift")
    selected_covariate = covariates.nodes[node.node_ordinal]
    if (
        selected_covariate.node_id,
        selected_covariate.parent_id,
        selected_covariate.depth,
        selected_covariate.role,
    ) != (node.node_id, node.parent_id, node.depth, node.node_role):
        raise ValueError("inference request selected node covariate drift")
    request = AllNodeInferenceRequest(
        cohort_key=key,
        cohort_id=node.cohort_id,
        tree_id=node.tree_id,
        node_id=node.node_id,
        node_ordinal=node.node_ordinal,
        parent_id=node.parent_id,
        depth=node.depth,
        node_role=node.node_role,
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        checkpoint_binding=node,
        checkpoint_binding_sha256=canonical_sha256(node.to_dict()),
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        topology_sha256=topology_sha,
        pair_index_sha256=_sha256(payload.get("pair_index_sha256"), "pair_index_sha256"),
        node_covariates=covariates,
        node_covariates_sha256=declared_covariates_sha,
        registry=_pin(payload.get("registry"), "registry"),
        panels=panels,
        model_task=model_task,
        generation_groups=groups,
        generation_groups_sha256=canonical_sha256(
            [group.to_dict() for group in groups]
        ),
        planned_backend_provenance=backend,
        planned_backend_provenance_sha256=canonical_sha256(backend),
        n_expected_responses=len(coordinates),
        coordinates_sha256=canonical_sha256(
            [coordinate.to_dict() for coordinate in coordinates]
        ),
        base_model_id=node.base_model_id,
        base_model_revision=node.base_model_revision,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        execution_overrides={},
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(request.to_dict()):
        raise ValueError("saved inference request differs from its recomputed contract")
    return request


def build_terminal_response_receipt(
    request: AllNodeInferenceRequest,
    raw_rows: Sequence[ResponseRow | Mapping[str, Any]],
    *,
    responses: ArtifactPin | Mapping[str, Any],
    observed_backend_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact grid and build one terminal hash-bound receipt."""

    verified_request = read_inference_request(request.to_dict())
    rows = tuple(_response_row(value) for value in raw_rows)
    audit = _audit_rows(rows, request=verified_request)
    if not audit["valid"]:
        raise ValueError(
            "response grid is invalid: "
            f"missing={audit['n_missing']}, extra={audit['n_extra']}, "
            f"duplicates={audit['n_duplicates']}, order_errors={audit['n_order_errors']}"
        )
    row_payload = [row.to_dict() for row in rows]
    observed_backend = _backend_provenance(
        observed_backend_provenance,
        binding=verified_request.checkpoint_binding,
        model_task=verified_request.model_task,
        label="observed_backend_provenance",
    )
    if canonical_sha256(observed_backend) != canonical_sha256(
        verified_request.planned_backend_provenance
    ):
        raise ValueError("observed backend provenance differs from the planned backend")
    return {
        "schema": RECEIPT_SCHEMA,
        "valid": True,
        "status": "completed",
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "request": verified_request.to_dict(),
        "request_sha256": verified_request.request_sha256,
        "observed_backend_provenance": observed_backend,
        "observed_backend_provenance_sha256": canonical_sha256(observed_backend),
        "responses": _pin(responses, "responses").to_dict(),
        "response_rows": row_payload,
        "response_rows_sha256": canonical_sha256(row_payload),
        "audit": audit,
    }


def validate_terminal_response_receipt(
    payload: Mapping[str, Any],
    *,
    expected_request: AllNodeInferenceRequest,
    observed_responses: ArtifactPin | Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a terminal receipt and reject any request/grid/hash drift.

    ``observed_responses`` is the independently measured path/digest pin for
    the response JSONL.  This module deliberately does not read or hash files.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("response receipt must be a mapping")
    expected_keys = {
        "schema",
        "valid",
        "status",
        "population",
        "internal_node_treatment",
        "request",
        "request_sha256",
        "observed_backend_provenance",
        "observed_backend_provenance_sha256",
        "responses",
        "response_rows",
        "response_rows_sha256",
        "audit",
    }
    if set(payload) != expected_keys:
        raise ValueError(f"response receipt keys must be exactly {sorted(expected_keys)!r}")
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("response receipt schema drift")
    if payload.get("valid") is not True or payload.get("status") != "completed":
        raise ValueError("response receipt is not terminal completed/valid")
    request_raw = payload.get("request")
    if not isinstance(request_raw, Mapping):
        raise ValueError("response receipt request must be a mapping")
    request = read_inference_request(request_raw)
    expected = read_inference_request(expected_request.to_dict())
    if canonical_sha256(request.to_dict()) != canonical_sha256(expected.to_dict()):
        raise ValueError("response receipt request differs from expected request")
    declared_responses = _pin(payload.get("responses"), "responses")
    if declared_responses != _pin(observed_responses, "observed_responses"):
        raise ValueError("response receipt artifact pin differs from observed responses")
    rows_raw = payload.get("response_rows")
    if not isinstance(rows_raw, list):
        raise ValueError("response receipt response_rows must be a list")
    rebuilt = build_terminal_response_receipt(
        request,
        rows_raw,
        responses=declared_responses,
        observed_backend_provenance=payload.get("observed_backend_provenance"),
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(rebuilt):
        raise ValueError("response receipt differs from its recomputed contract")
    return rebuilt


def _topology_from_manifest(manifest: TreeCheckpointManifest):
    """Rebuild the exact topology needed to validate node-design paths."""

    return build_tree_topology(
        manifest.tree_id,
        [
            NodeSpec(
                node_ordinal=binding.node_ordinal,
                array_index=binding.array_index,
                node_id=binding.node_id,
                parent_id=binding.parent_id,
                depth=binding.depth,
                is_leaf=binding.node_role == "leaf",
            )
            for binding in manifest.bindings
        ],
        root_id=manifest.root_id,
    )


def _validate_frozen_panels(cohort_key: str, panels: Sequence[PromptPanel]) -> None:
    expected = FROZEN_COHORT_ENDPOINTS.get(cohort_key)
    if expected is None:
        raise ValueError(f"unknown frozen cohort_key {cohort_key!r}")
    observed_shapes = tuple(
        EndpointShape(
            panel.protocol_id,
            panel.probe_id,
            len(panel.prompt_ids),
            panel.samples_per_prompt,
        )
        for panel in panels
    )
    if observed_shapes != expected:
        raise ValueError(
            f"cohort {cohort_key!r} prompt panels differ from the frozen endpoint contract"
        )
    protocol_pins: dict[str, tuple[ArtifactPin, ArtifactPin]] = {}
    for panel in panels:
        pins = (panel.prompt_artifact, panel.materialization_receipt)
        previous = protocol_pins.setdefault(panel.protocol_id, pins)
        if previous != pins:
            raise ValueError(
                f"protocol {panel.protocol_id!r} prompt/materialization pins differ across probes"
            )


def _generation_groups(
    panels: Sequence[PromptPanel],
    *,
    model_task: str,
) -> tuple[GenerationGroup, ...]:
    groups: list[GenerationGroup] = []
    for panel in panels:
        for sample_id in range(panel.samples_per_prompt):
            groups.append(
                GenerationGroup(
                    group_ordinal=len(groups),
                    protocol_id=panel.protocol_id,
                    probe_id=panel.probe_id,
                    sample_id=sample_id,
                    prompt_ids=panel.prompt_ids,
                    generation_options=_expected_generation_options(
                        panel.protocol_id,
                        sample_id=sample_id,
                        model_task=model_task,
                    ),
                )
            )
    return tuple(groups)


def _validate_generation_groups(
    groups: Sequence[GenerationGroup],
    *,
    panels: Sequence[PromptPanel],
    model_task: str,
) -> None:
    expected = _generation_groups(panels, model_task=model_task)
    observed = tuple(groups)
    if canonical_sha256([group.to_dict() for group in observed]) != canonical_sha256(
        [group.to_dict() for group in expected]
    ):
        raise ValueError(
            "generation groups differ from the exact seed-once protocol/probe/draw contract"
        )
    flattened = tuple(
        coordinate for group in observed for coordinate in group.coordinates
    )
    expected_coordinates = _coordinates(panels)
    if len(flattened) != len(expected_coordinates) or set(flattened) != set(
        expected_coordinates
    ):
        raise ValueError("generation groups do not cover the exact response grid")


def _generation_group(value: GenerationGroup | Mapping[str, Any]) -> GenerationGroup:
    if isinstance(value, GenerationGroup):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("generation group must be a mapping")
    required = {
        "group_ordinal",
        "protocol_id",
        "probe_id",
        "sample_id",
        "prompt_ids",
        "generation_options",
    }
    if set(value) != required:
        raise ValueError(f"generation group keys must be exactly {sorted(required)!r}")
    raw_prompt_ids = value["prompt_ids"]
    if not isinstance(raw_prompt_ids, (list, tuple)) or not raw_prompt_ids:
        raise ValueError("generation group prompt_ids must be a nonempty sequence")
    prompt_ids = tuple(_text(item, "generation group prompt_id") for item in raw_prompt_ids)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("generation group prompt_ids contain duplicates")
    options = value["generation_options"]
    if not isinstance(options, Mapping) or set(options) != GENERATION_OPTION_KEYS:
        raise ValueError(
            "generation_options keys differ from the frozen registry contract"
        )
    return GenerationGroup(
        group_ordinal=_integer(value["group_ordinal"], "group_ordinal", minimum=0),
        protocol_id=_text(value["protocol_id"], "protocol_id"),
        probe_id=_text(value["probe_id"], "probe_id"),
        sample_id=_integer(value["sample_id"], "sample_id", minimum=0),
        prompt_ids=prompt_ids,
        generation_options=dict(options),
    )


def _expected_generation_options(
    protocol_id: str,
    *,
    sample_id: int,
    model_task: str,
) -> dict[str, Any]:
    if model_task not in {"causal_lm", "seq2seq"}:
        raise ValueError("model_task must be causal_lm or seq2seq")
    if protocol_id == TRANSLATION_PROTOCOL:
        if sample_id != 0:
            raise ValueError("translation has exactly one registered draw")
        return {
            "sample_id": 0,
            "seed": 42,
            "generation_batch_size": 16,
            "seed_scope": "probe_draw",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "min_new_tokens": 0,
            "max_new_tokens": 64 if model_task == "causal_lm" else 128,
            "empty_policy": "preserve",
        }
    if protocol_id in {MODERN_MC_PROTOCOL, DOLLY_PROTOCOL}:
        if sample_id not in {0, 1, 2}:
            raise ValueError(f"{protocol_id} has exactly three registered draws")
        return {
            "sample_id": sample_id,
            "seed": 42 + sample_id,
            "generation_batch_size": 32,
            "seed_scope": "probe_draw",
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "min_new_tokens": 0,
            "max_new_tokens": 96 if protocol_id == DOLLY_PROTOCOL else 64,
            "empty_policy": "preserve",
        }
    raise ValueError(f"unknown frozen behavior protocol {protocol_id!r}")


def _backend_provenance(
    value: object,
    *,
    binding: NodeCheckpointBinding,
    model_task: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != BACKEND_PROVENANCE_KEYS:
        raise ValueError(
            f"{label} keys must be exactly {sorted(BACKEND_PROVENANCE_KEYS)!r}"
        )
    normalized = dict(value)
    expected = {
        "backend": "huggingface",
        "device": DEVICE,
        "torch_dtype": TORCH_DTYPE,
        "local_files_only": True,
        "tokenizer_model_id": binding.base_model_id,
        "tokenizer_revision": binding.base_model_revision,
        "checkpoint": binding.checkpoint.path,
        "decode_skip_special_tokens": True,
        "decode_clean_up_tokenization_spaces": False,
        "generation_api": "ordered_probe_draw_batches",
        "generation_seed_api": "transformers.set_seed_once_per_probe_draw",
        "tokenizer_padding_side": "left" if model_task == "causal_lm" else "right",
        "tokenizer_truncation": True,
        "tokenizer_max_length": 512,
        "generation_pad_token_id": "tokenizer.pad_token_id",
    }
    for field, expected_value in expected.items():
        if normalized.get(field) != expected_value:
            raise ValueError(
                f"{label}.{field} must be exactly {expected_value!r}"
            )
    for field in ("transformers_version", "torch_version"):
        _text(normalized.get(field), f"{label}.{field}")
    return normalized


def _coordinates(panels: Sequence[PromptPanel]) -> tuple[ResponseCoordinate, ...]:
    rows = tuple(
        ResponseCoordinate(panel.protocol_id, panel.probe_id, prompt_id, sample_id)
        for panel in panels
        for prompt_id in panel.prompt_ids
        for sample_id in range(panel.samples_per_prompt)
    )
    if len(rows) != len(set(rows)):
        raise ValueError("deterministic response grid contains duplicate coordinates")
    return rows


def _audit_rows(
    rows: Sequence[ResponseRow], *, request: AllNodeInferenceRequest
) -> dict[str, Any]:
    expected = request.coordinates()
    observed = tuple(row.coordinate for row in rows)
    seen: set[ResponseCoordinate] = set()
    duplicates = 0
    for coordinate in observed:
        if coordinate in seen:
            duplicates += 1
        seen.add(coordinate)
    expected_set = set(expected)
    observed_set = set(observed)
    missing = expected_set - observed_set
    extra = observed_set - expected_set
    order_errors = sum(
        left != right for left, right in zip(observed, expected)
    ) + abs(len(observed) - len(expected))
    contract_errors = 0
    for row in rows:
        if (
            row.request_sha256 != request.request_sha256
            or row.cohort_id != request.cohort_id
            or row.tree_id != request.tree_id
            or row.node_id != request.node_id
            or row.completed is not True
        ):
            contract_errors += 1
    valid = not (missing or extra or duplicates or order_errors or contract_errors)
    return {
        "valid": valid,
        "n_expected": len(expected),
        "n_observed": len(rows),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_duplicates": duplicates,
        "n_order_errors": order_errors,
        "n_contract_errors": contract_errors,
    }


def _panel(value: PromptPanel | Mapping[str, Any]) -> PromptPanel:
    if isinstance(value, PromptPanel):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("prompt panel must be a mapping")
    required = {
        "protocol_id",
        "probe_id",
        "prompt_ids",
        "samples_per_prompt",
        "prompt_artifact",
        "materialization_receipt",
    }
    if set(value) != required:
        raise ValueError(f"prompt panel keys must be exactly {sorted(required)!r}")
    raw_ids = value["prompt_ids"]
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError("prompt_ids must be a nonempty ordered sequence")
    prompt_ids = tuple(_text(prompt_id, "prompt_id") for prompt_id in raw_ids)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt_ids must be unique within a panel")
    return PromptPanel(
        protocol_id=_text(value["protocol_id"], "protocol_id"),
        probe_id=_text(value["probe_id"], "probe_id"),
        prompt_ids=prompt_ids,
        samples_per_prompt=_integer(
            value["samples_per_prompt"], "samples_per_prompt", minimum=1
        ),
        prompt_artifact=_pin(value["prompt_artifact"], "prompt_artifact"),
        materialization_receipt=_pin(
            value["materialization_receipt"], "materialization_receipt"
        ),
    )


def _response_row(value: ResponseRow | Mapping[str, Any]) -> ResponseRow:
    if isinstance(value, ResponseRow):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("response row must be a mapping")
    required = {
        "schema",
        "request_sha256",
        "cohort_id",
        "tree_id",
        "node_id",
        "protocol_id",
        "probe_id",
        "prompt_id",
        "sample_id",
        "response_sha256",
        "completed",
    }
    if set(value) != required:
        raise ValueError(f"response row keys must be exactly {sorted(required)!r}")
    if value.get("schema") != RESPONSE_ROW_SCHEMA:
        raise ValueError("response row schema drift")
    return ResponseRow(
        request_sha256=_sha256(value["request_sha256"], "request_sha256"),
        cohort_id=_text(value["cohort_id"], "cohort_id"),
        tree_id=_text(value["tree_id"], "tree_id"),
        node_id=_text(value["node_id"], "node_id"),
        protocol_id=_text(value["protocol_id"], "protocol_id"),
        probe_id=_text(value["probe_id"], "probe_id"),
        prompt_id=_text(value["prompt_id"], "prompt_id"),
        sample_id=_integer(value["sample_id"], "sample_id", minimum=0),
        response_sha256=_sha256(value["response_sha256"], "response_sha256"),
        completed=_boolean(value["completed"], "completed"),
    )


def _binding(value: NodeCheckpointBinding | Mapping[str, Any]) -> NodeCheckpointBinding:
    if isinstance(value, NodeCheckpointBinding):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("node checkpoint binding must be a mapping")
    required = {
        "population",
        "internal_node_treatment",
        "cohort_id",
        "tree_id",
        "node_ordinal",
        "array_index",
        "node_id",
        "parent_id",
        "depth",
        "is_leaf",
        "node_role",
        "status",
        "method",
        "artifact_name",
        "checkpoint",
        "base_model_id",
        "base_model_revision",
        "run_list",
        "ledger",
        "truth_manifest",
        "top_config",
        "training_summary",
        "completion_receipt",
    }
    if set(value) != required:
        raise ValueError(f"node checkpoint binding keys must be exactly {sorted(required)!r}")
    if value["population"] != POPULATION:
        raise ValueError(f"population must be exactly {POPULATION!r}")
    if value["internal_node_treatment"] != INTERNAL_NODE_TREATMENT:
        raise ValueError(
            f"internal_node_treatment must be exactly {INTERNAL_NODE_TREATMENT!r}"
        )
    status = _text(value["status"], "status")
    if status != "completed":
        raise ValueError("node checkpoint status must be exactly 'completed'")
    method = _text(value["method"], "method")
    artifact_name = _text(value["artifact_name"], "artifact_name")
    if _METHOD_ARTIFACT.get(method) != artifact_name:
        raise ValueError("node checkpoint method/artifact pairing drift")
    is_leaf = _boolean(value["is_leaf"], "is_leaf")
    role = _text(value["node_role"], "node_role")
    if role != ("leaf" if is_leaf else "internal"):
        raise ValueError("node_role is inconsistent with is_leaf")
    revision = _text(value["base_model_revision"], "base_model_revision")
    if not _REVISION.fullmatch(revision):
        raise ValueError("base_model_revision must be a pinned lowercase revision")
    return NodeCheckpointBinding(
        population=POPULATION,
        internal_node_treatment=INTERNAL_NODE_TREATMENT,
        cohort_id=_text(value["cohort_id"], "cohort_id"),
        tree_id=_text(value["tree_id"], "tree_id"),
        node_ordinal=_integer(value["node_ordinal"], "node_ordinal", minimum=0),
        array_index=_integer(value["array_index"], "array_index", minimum=0),
        node_id=_text(value["node_id"], "node_id"),
        parent_id=_text(value["parent_id"], "parent_id"),
        depth=_integer(value["depth"], "depth", minimum=1),
        is_leaf=is_leaf,
        node_role=role,
        status=status,
        method=method,
        artifact_name=artifact_name,
        checkpoint=_pin(value["checkpoint"], "checkpoint"),
        base_model_id=_text(value["base_model_id"], "base_model_id"),
        base_model_revision=revision,
        run_list=_pin(value["run_list"], "run_list"),
        ledger=_pin(value["ledger"], "ledger"),
        truth_manifest=_pin(value["truth_manifest"], "truth_manifest"),
        top_config=_pin(value["top_config"], "top_config"),
        training_summary=_pin(value["training_summary"], "training_summary"),
        completion_receipt=_pin(value["completion_receipt"], "completion_receipt"),
    )


def _pin(value: ArtifactPin | Mapping[str, Any], label: str) -> ArtifactPin:
    if isinstance(value, ArtifactPin):
        value = value.to_dict()
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} pin must contain exactly path and sha256")
    path = _text(value["path"], f"{label}.path")
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"{label}.path must be an absolute normalized path")
    return ArtifactPin(path, _sha256(value["sha256"], f"{label}.sha256"))


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value
