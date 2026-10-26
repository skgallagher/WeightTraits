"""Filesystem-backed production runner for all-node behavior inference.

The runner consumes one exact, self-hashed cohort plan at the fixed deployment
location::

    deployments/allnode_v1/runtime_inputs/<cohort_key>/inference_plan.json

Each plan contains all fifty checkpoint manifests and an explicit flattened
``array_index`` mapping over all 641 trained nodes.  Execution never accepts a
leaf alias, checkpoint/output override, model/runtime override, or remote model
lookup.  The selected checkpoint is authority-aware and re-attested before the
backend opens it and again after generation, before either output is published.
A response JSONL is published first and its terminal receipt second, using
same-filesystem exclusive atomic links.  Existing files are reusable only when
both form an exact validated terminal cache.

Imports from the pinned WeightTraits source are intentionally lazy.  Production
imports must resolve below ``<stage>/src/weighttraits``; tests may inject a
small compatible API without importing torch or transformers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    NodeCheckpointBinding,
    POPULATION,
    PRODUCTION_N_INTERNAL,
    PRODUCTION_N_LEAVES,
    PRODUCTION_N_NODES,
    PRODUCTION_N_TREES,
    TreeCheckpointManifest,
    read_tree_checkpoint_manifest,
)
from .checkpoint_identity import (
    CheckpointIdentity,
    read_identity_index,
    source_authority_from_identity,
    validate_checkpoint_identity,
)
from .inference import (
    BACKEND_PROVENANCE_KEYS,
    DEVICE,
    FROZEN_COHORT_MODEL_TASK,
    GenerationGroup,
    PromptPanel,
    ResponseCoordinate,
    TORCH_DTYPE,
    build_inference_request,
)
from .covariates import (
    TreeNodeCovariates,
    read_tree_node_covariates,
    validate_tree_node_covariates,
)
from .manifest import NodeSpec, build_tree_topology, canonical_sha256
from .production import PRODUCTION_TREE_IDS
from .provenance import attest_runtime_environment
from .runtime import RuntimeExecutionResult, execute_all_node_inference


PLAN_SCHEMA = "weighttraits.all_node_inference_cohort_plan.v1"
JOB_CONFIG_SCHEMA = "weighttraits.all_node_inference_job_config.v1"
CLI_STATUS_SCHEMA = "weighttraits.all_node_inference_cli_status.v1"
CHECKPOINT_PATH_SENTINEL = "$CHECKPOINT_BINDING_PATH"
PLAN_BASENAME = "inference_plan.json"
OUTPUT_RELATIVE_ROOT = Path("outputs/allnode_v1/inference")

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, order=True)
class FlattenedNodeMapping:
    """One canonical Slurm array index mapped to one manifest binding."""

    array_index: int
    tree_id: str
    node_ordinal: int
    node_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "array_index": self.array_index,
            "tree_id": self.tree_id,
            "node_ordinal": self.node_ordinal,
            "node_id": self.node_id,
        }


@dataclass(frozen=True)
class InferenceCohortPlan:
    """The exact four-independent-cohort input contract for one cohort."""

    cohort_key: str
    training_cohort_id: str
    source_stage_root: str
    model_task: str
    checkpoint_identity_index: ArtifactPin
    checkpoint_manifests: tuple[TreeCheckpointManifest, ...]
    node_covariates: tuple[TreeNodeCovariates, ...]
    flattened_mapping: tuple[FlattenedNodeMapping, ...]
    registry: ArtifactPin
    panels: tuple[PromptPanel, ...]
    planned_backend_template: Mapping[str, Any]
    checkpoint_manifests_sha256: str
    node_covariates_sha256: str
    flattened_mapping_sha256: str
    panels_sha256: str
    planned_backend_template_sha256: str
    inference_plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        core = _plan_core(
            cohort_key=self.cohort_key,
            training_cohort_id=self.training_cohort_id,
            source_stage_root=self.source_stage_root,
            model_task=self.model_task,
            checkpoint_identity_index=self.checkpoint_identity_index,
            checkpoint_manifests=self.checkpoint_manifests,
            node_covariates=self.node_covariates,
            flattened_mapping=self.flattened_mapping,
            registry=self.registry,
            panels=self.panels,
            planned_backend_template=self.planned_backend_template,
            checkpoint_manifests_sha256=self.checkpoint_manifests_sha256,
            node_covariates_sha256=self.node_covariates_sha256,
            flattened_mapping_sha256=self.flattened_mapping_sha256,
            panels_sha256=self.panels_sha256,
            planned_backend_template_sha256=self.planned_backend_template_sha256,
        )
        return {**core, "inference_plan_sha256": self.inference_plan_sha256}

    def select(
        self, array_index: int
    ) -> tuple[
        FlattenedNodeMapping,
        TreeCheckpointManifest,
        TreeNodeCovariates,
        NodeCheckpointBinding,
    ]:
        index = _integer(array_index, "array_index", minimum=0)
        if index >= PRODUCTION_N_NODES:
            raise ValueError(
                f"array_index must be in [0, {PRODUCTION_N_NODES - 1}], got {index}"
            )
        mapping = self.flattened_mapping[index]
        if mapping.array_index != index:
            raise AssertionError("flattened inference mapping index corruption")
        tree_position = PRODUCTION_TREE_IDS.index(mapping.tree_id)
        manifest = self.checkpoint_manifests[tree_position]
        covariates = self.node_covariates[tree_position]
        binding = manifest.bindings[mapping.node_ordinal]
        if (binding.node_id, binding.node_ordinal) != (
            mapping.node_id,
            mapping.node_ordinal,
        ):
            raise AssertionError("flattened inference mapping binding corruption")
        return mapping, manifest, covariates, binding


@dataclass(frozen=True)
class InferenceJobConfig:
    """Thin immutable launcher binding for one inference cohort plan."""

    logical_id: str
    cohort_key: str
    kind: str
    runtime_plan: ArtifactPin
    output_root: str
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": JOB_CONFIG_SCHEMA,
            "valid": True,
            "logical_id": self.logical_id,
            "cohort_key": self.cohort_key,
            "kind": self.kind,
            "runtime_plan": self.runtime_plan.to_dict(),
            "output_root": self.output_root,
            "execution_overrides": {},
        }
        return {**core, "config_sha256": self.config_sha256}


@dataclass(frozen=True)
class AllNodeBackendBinding:
    """Duck-typed input for the pinned HuggingFace backend, without leaf aliases."""

    checkpoint: Path
    base_model_id: str
    base_model_revision: str
    model_task: str


@dataclass(frozen=True)
class WeightTraitsInferenceAPI:
    """The three pinned-source operations needed by the filesystem runner."""

    load_registry: Callable[[Path], object]
    load_prompts: Callable[[Path], Sequence[object]]
    backend_class: object


@dataclass(frozen=True)
class FilesystemInferenceResult:
    """One completed fresh publication or exact terminal cache validation."""

    cache_hit: bool
    plan: InferenceCohortPlan
    plan_path: Path
    plan_file_sha256: str
    mapping: FlattenedNodeMapping
    output_path: Path
    receipt_path: Path
    runtime_result: RuntimeExecutionResult

    def status_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CLI_STATUS_SCHEMA,
            "valid": True,
            "status": "cached" if self.cache_hit else "completed",
            "cache_hit": self.cache_hit,
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "cohort_key": self.plan.cohort_key,
            "training_cohort_id": self.plan.training_cohort_id,
            "array_index": self.mapping.array_index,
            "tree_id": self.mapping.tree_id,
            "node_ordinal": self.mapping.node_ordinal,
            "node_id": self.mapping.node_id,
            "inference_plan": str(self.plan_path),
            "inference_plan_file_sha256": self.plan_file_sha256,
            "inference_plan_sha256": self.plan.inference_plan_sha256,
            "node_covariates_sha256": self.runtime_result.request.node_covariates_sha256,
            "request_sha256": self.runtime_result.request.request_sha256,
            "responses": self.runtime_result.receipt["responses"],
            "terminal_receipt": str(self.receipt_path),
            "execution_overrides": {},
        }
        _reject_leaf_alias(payload)
        return payload


def build_inference_cohort_plan(
    *,
    cohort_key: str,
    source_stage_root: str,
    checkpoint_identity_index: ArtifactPin | Mapping[str, Any],
    checkpoint_manifests: Sequence[TreeCheckpointManifest | Mapping[str, Any]],
    node_covariates: Sequence[TreeNodeCovariates | Mapping[str, Any]],
    registry: ArtifactPin | Mapping[str, Any],
    panels: Sequence[PromptPanel | Mapping[str, Any]],
    planned_backend_template: Mapping[str, Any],
) -> InferenceCohortPlan:
    """Build and self-hash the exact fifty-tree/641-node cohort plan."""

    key = _text(cohort_key, "cohort_key")
    if key not in FROZEN_COHORT_MODEL_TASK:
        raise ValueError(f"unknown frozen cohort_key {key!r}")
    source_root = _canonical_absolute_root(source_stage_root, "source_stage_root")
    identity_index_pin = _pin(
        checkpoint_identity_index, "checkpoint_identity_index"
    )
    manifests = tuple(_manifest(value) for value in checkpoint_manifests)
    if len(manifests) != PRODUCTION_N_TREES:
        raise ValueError(
            f"checkpoint_manifests must contain exactly {PRODUCTION_N_TREES} trees"
        )
    if tuple(manifest.tree_id for manifest in manifests) != PRODUCTION_TREE_IDS:
        raise ValueError("checkpoint manifest tree order must be exactly trees 001..050")
    raw_covariates = tuple(node_covariates)
    if len(raw_covariates) != len(manifests):
        raise ValueError("node_covariates must contain exactly one contract per tree")
    covariate_contracts = tuple(
        _covariates(value, manifest)
        for value, manifest in zip(raw_covariates, manifests)
    )
    training_ids = {manifest.cohort_id for manifest in manifests}
    if len(training_ids) != 1:
        raise ValueError("checkpoint manifests mix training cohort identities")
    totals = (
        sum(manifest.n_nodes for manifest in manifests),
        sum(manifest.n_internal for manifest in manifests),
        sum(manifest.n_leaves for manifest in manifests),
    )
    expected_totals = (
        PRODUCTION_N_NODES,
        PRODUCTION_N_INTERNAL,
        PRODUCTION_N_LEAVES,
    )
    if totals != expected_totals:
        raise ValueError(
            "checkpoint manifest all-node totals differ: "
            f"observed={totals}, expected={expected_totals}"
        )
    checkpoint_paths = [
        binding.checkpoint.path
        for manifest in manifests
        for binding in manifest.bindings
    ]
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise ValueError("cohort inference plan reuses a checkpoint path")
    shared_model = {
        (binding.base_model_id, binding.base_model_revision)
        for manifest in manifests
        for binding in manifest.bindings
    }
    if len(shared_model) != 1:
        raise ValueError("cohort inference plan mixes base-model pins")

    mapping = tuple(
        FlattenedNodeMapping(
            array_index=array_index,
            tree_id=manifest.tree_id,
            node_ordinal=binding.node_ordinal,
            node_id=binding.node_id,
        )
        for array_index, (manifest, binding) in enumerate(
            (manifest, binding)
            for manifest in manifests
            for binding in manifest.bindings
        )
    )
    if len(mapping) != PRODUCTION_N_NODES:
        raise AssertionError("flattened inference mapping cardinality corruption")

    registry_pin = _pin(registry, "registry")
    prompt_panels = tuple(_panel(value) for value in panels)
    template = _backend_template(
        planned_backend_template,
        binding=manifests[0].bindings[0],
        model_task=FROZEN_COHORT_MODEL_TASK[key],
    )
    # Reuse the request contract to prove exact panel identifiers/order/counts,
    # generation settings, and every non-checkpoint backend-template field.
    build_inference_request(
        manifests[0].bindings[0],
        manifests[0],
        cohort_key=key,
        registry=registry_pin,
        panels=prompt_panels,
        planned_backend_provenance=_materialize_backend_template(
            template, manifests[0].bindings[0]
        ),
        node_covariates=covariate_contracts[0],
    )

    manifest_rows = [manifest.to_dict() for manifest in manifests]
    covariate_rows = [value.to_dict() for value in covariate_contracts]
    mapping_rows = [row.to_dict() for row in mapping]
    panel_rows = [panel.to_dict() for panel in prompt_panels]
    manifest_sha = canonical_sha256(manifest_rows)
    covariates_sha = canonical_sha256(covariate_rows)
    mapping_sha = canonical_sha256(mapping_rows)
    panels_sha = canonical_sha256(panel_rows)
    template_sha = canonical_sha256(template)
    core = _plan_core(
        cohort_key=key,
        training_cohort_id=next(iter(training_ids)),
        source_stage_root=source_root,
        model_task=FROZEN_COHORT_MODEL_TASK[key],
        checkpoint_identity_index=identity_index_pin,
        checkpoint_manifests=manifests,
        node_covariates=covariate_contracts,
        flattened_mapping=mapping,
        registry=registry_pin,
        panels=prompt_panels,
        planned_backend_template=template,
        checkpoint_manifests_sha256=manifest_sha,
        node_covariates_sha256=covariates_sha,
        flattened_mapping_sha256=mapping_sha,
        panels_sha256=panels_sha,
        planned_backend_template_sha256=template_sha,
    )
    return InferenceCohortPlan(
        cohort_key=key,
        training_cohort_id=next(iter(training_ids)),
        source_stage_root=source_root,
        model_task=FROZEN_COHORT_MODEL_TASK[key],
        checkpoint_identity_index=identity_index_pin,
        checkpoint_manifests=manifests,
        node_covariates=covariate_contracts,
        flattened_mapping=mapping,
        registry=registry_pin,
        panels=prompt_panels,
        planned_backend_template=template,
        checkpoint_manifests_sha256=manifest_sha,
        node_covariates_sha256=covariates_sha,
        flattened_mapping_sha256=mapping_sha,
        panels_sha256=panels_sha,
        planned_backend_template_sha256=template_sha,
        inference_plan_sha256=canonical_sha256(core),
    )


def read_inference_cohort_plan(payload: Mapping[str, Any]) -> InferenceCohortPlan:
    """Rebuild a saved cohort plan and compare every field and self-hash."""

    if not isinstance(payload, Mapping):
        raise ValueError("inference cohort plan must be a mapping")
    required = {
        "schema",
        "valid",
        "population",
        "internal_node_treatment",
        "exclude_root",
        "cohort_key",
        "training_cohort_id",
        "source_stage_root",
        "model_task",
        "checkpoint_identity_index",
        "n_trees",
        "n_nodes",
        "n_internal",
        "n_leaves",
        "tree_ids",
        "checkpoint_manifests",
        "checkpoint_manifests_sha256",
        "node_covariates",
        "node_covariates_sha256",
        "flattened_mapping",
        "flattened_mapping_sha256",
        "registry",
        "panels",
        "panels_sha256",
        "planned_backend_template",
        "planned_backend_template_sha256",
        "execution_overrides",
        "inference_plan_sha256",
    }
    if set(payload) != required:
        raise ValueError(f"inference cohort plan keys must be exactly {sorted(required)!r}")
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("inference cohort plan schema drift")
    raw_manifests = payload.get("checkpoint_manifests")
    raw_covariates = payload.get("node_covariates")
    raw_panels = payload.get("panels")
    raw_mapping = payload.get("flattened_mapping")
    if (
        not isinstance(raw_manifests, list)
        or not isinstance(raw_covariates, list)
        or not isinstance(raw_panels, list)
        or not isinstance(raw_mapping, list)
    ):
        raise ValueError(
            "inference cohort plan manifests/covariates/mapping/panels must be lists"
        )
    declared_mapping = tuple(_mapping(value) for value in raw_mapping)
    rebuilt = build_inference_cohort_plan(
        cohort_key=payload.get("cohort_key"),
        source_stage_root=payload.get("source_stage_root"),
        checkpoint_identity_index=payload.get("checkpoint_identity_index"),
        checkpoint_manifests=raw_manifests,
        node_covariates=raw_covariates,
        registry=payload.get("registry"),
        panels=raw_panels,
        planned_backend_template=payload.get("planned_backend_template"),
    )
    if declared_mapping != rebuilt.flattened_mapping:
        raise ValueError("saved flattened mapping differs from recomputed manifests")
    if canonical_sha256(dict(payload)) != canonical_sha256(rebuilt.to_dict()):
        raise ValueError("saved inference cohort plan differs from its recomputed contract")
    _reject_leaf_alias(payload)
    return rebuilt


def load_inference_cohort_plan(
    path: str | Path,
) -> tuple[InferenceCohortPlan, Path, Path, str]:
    """Load the exact deployment path and return plan, path, stage root, file hash."""

    raw_path = Path(path)
    absolute = Path(os.path.abspath(str(raw_path)))
    resolved = raw_path.resolve(strict=True)
    if absolute != resolved:
        raise ValueError("inference plan path must be canonical and contain no symlinks")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"inference plan is not a regular file: {resolved}")
    if (
        resolved.name != PLAN_BASENAME
        or resolved.parent.parent.name != "runtime_inputs"
        or resolved.parent.parent.parent.name != "allnode_v1"
        or resolved.parent.parent.parent.parent.name != "deployments"
    ):
        raise ValueError(
            "inference plan must be at deployments/allnode_v1/runtime_inputs/"
            "<cohort_key>/inference_plan.json"
        )
    stage_root = resolved.parent.parent.parent.parent.parent
    raw_bytes = resolved.read_bytes()
    payload = _json_object(raw_bytes, label="inference cohort plan")
    plan = read_inference_cohort_plan(payload)
    if resolved.parent.name != plan.cohort_key:
        raise ValueError("inference plan directory and cohort_key differ")
    return plan, resolved, stage_root, hashlib.sha256(raw_bytes).hexdigest()


def build_inference_job_config(
    plan: InferenceCohortPlan,
    *,
    runtime_plan_path: str | Path,
    runtime_plan_file_sha256: str,
) -> InferenceJobConfig:
    """Bind one saved plan to the exact generic launcher environment.

    The config pins the raw plan-file bytes, rather than merely repeating the
    plan's internal self-hash.  Its path determines the unique downstream
    stage and therefore the only permitted output root.
    """

    if not isinstance(plan, InferenceCohortPlan):
        raise ValueError("plan must be an InferenceCohortPlan")
    path = _canonical_planned_inference_path(runtime_plan_path, plan.cohort_key)
    stage_root = path.parent.parent.parent.parent.parent
    output_root = stage_root / OUTPUT_RELATIVE_ROOT / plan.cohort_key
    runtime_plan = ArtifactPin(
        str(path),
        _sha256(runtime_plan_file_sha256, "runtime_plan_file_sha256"),
    )
    core = {
        "schema": JOB_CONFIG_SCHEMA,
        "valid": True,
        "logical_id": f"infer_{plan.cohort_key}",
        "cohort_key": plan.cohort_key,
        "kind": "inference_allnode_gpu",
        "runtime_plan": runtime_plan.to_dict(),
        "output_root": str(output_root),
        "execution_overrides": {},
    }
    return InferenceJobConfig(
        logical_id=core["logical_id"],
        cohort_key=plan.cohort_key,
        kind=core["kind"],
        runtime_plan=runtime_plan,
        output_root=str(output_root),
        config_sha256=canonical_sha256(core),
    )


def read_inference_job_config(payload: Mapping[str, Any]) -> InferenceJobConfig:
    """Validate a serialized launcher config without touching the filesystem."""

    if not isinstance(payload, Mapping):
        raise ValueError("inference job config must be a mapping")
    required = {
        "schema",
        "valid",
        "logical_id",
        "cohort_key",
        "kind",
        "runtime_plan",
        "output_root",
        "execution_overrides",
        "config_sha256",
    }
    if set(payload) != required:
        raise ValueError("inference job config keys drift")
    core = {key: payload[key] for key in required if key != "config_sha256"}
    if (
        core["schema"] != JOB_CONFIG_SCHEMA
        or core["valid"] is not True
        or core["kind"] != "inference_allnode_gpu"
        or core["execution_overrides"] != {}
        or payload["config_sha256"] != canonical_sha256(core)
    ):
        raise ValueError("inference job config contract drift")
    cohort = _safe_component(core["cohort_key"], "cohort_key")
    if cohort not in FROZEN_COHORT_MODEL_TASK:
        raise ValueError(f"unknown frozen cohort_key {cohort!r}")
    logical_id = _safe_component(core["logical_id"], "logical_id")
    if logical_id != f"infer_{cohort}":
        raise ValueError("inference job logical ID drift")
    output_root = _canonical_absolute_root(core["output_root"], "output_root")
    return InferenceJobConfig(
        logical_id=logical_id,
        cohort_key=cohort,
        kind=core["kind"],
        runtime_plan=_pin(core["runtime_plan"], "runtime_plan"),
        output_root=output_root,
        config_sha256=_sha256(payload["config_sha256"], "config_sha256"),
    )


def load_inference_job_config(
    path: str | Path,
) -> tuple[InferenceJobConfig, InferenceCohortPlan, Path, Path, str]:
    """Load the unique config and independently attest its pinned plan bytes."""

    raw_path = Path(path)
    absolute = Path(os.path.abspath(str(raw_path)))
    resolved = raw_path.resolve(strict=True)
    if absolute != resolved or resolved.is_symlink() or not resolved.is_file():
        raise ValueError(
            "inference config path must be a canonical regular file without symlinks"
        )
    if (
        resolved.parent.name != "jobs"
        or resolved.parent.parent.name != "configs"
        or resolved.parent.parent.parent.name != "allnode_v1"
        or resolved.parent.parent.parent.parent.name != "deployments"
    ):
        raise ValueError(
            "inference config must be at deployments/allnode_v1/configs/jobs/"
            "infer_<cohort>.json"
        )
    raw = resolved.read_bytes()
    config = read_inference_job_config(
        _json_object(raw, label="inference job config")
    )
    if resolved.name != f"infer_{config.cohort_key}.json":
        raise ValueError("inference config filename and cohort_key differ")
    stage_from_config = resolved.parent.parent.parent.parent.parent
    expected_plan_path = (
        stage_from_config
        / "deployments"
        / "allnode_v1"
        / "runtime_inputs"
        / config.cohort_key
        / PLAN_BASENAME
    )
    plan_path = _canonical_planned_inference_path(
        config.runtime_plan.path,
        config.cohort_key,
    )
    if plan_path != expected_plan_path:
        raise ValueError("inference runtime plan path differs from its job config stage")
    observed_plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if observed_plan_sha != config.runtime_plan.sha256:
        raise ValueError("inference runtime plan file hash drift")
    plan, loaded_plan_path, stage_root, _ = load_inference_cohort_plan(plan_path)
    if loaded_plan_path != plan_path:
        raise ValueError("inference runtime plan canonical path drift")
    expected_output = stage_root / OUTPUT_RELATIVE_ROOT / plan.cohort_key
    if (
        plan.cohort_key != config.cohort_key
        or config.logical_id != f"infer_{plan.cohort_key}"
        or config.output_root != str(expected_output)
    ):
        raise ValueError("inference job config and runtime plan scope differ")
    return (
        config,
        plan,
        loaded_plan_path,
        resolved,
        hashlib.sha256(raw).hexdigest(),
    )


def execute_filesystem_inference(
    plan_path: str | Path,
    *,
    array_index: int,
    api: WeightTraitsInferenceAPI | None = None,
) -> FilesystemInferenceResult:
    """Validate inputs, execute/cache one node, and publish exclusively in order."""

    plan, saved_plan_path, stage_root, plan_file_sha = load_inference_cohort_plan(
        plan_path
    )
    mapping, manifest, node_covariates, binding = plan.select(array_index)
    output_dir = _fixed_output_dir(stage_root, plan.cohort_key, mapping)
    output_path = output_dir / "responses.jsonl"
    receipt_path = output_dir / "terminal_receipt.json"
    output_exists = os.path.lexists(output_path)
    receipt_exists = os.path.lexists(receipt_path)
    if output_exists != receipt_exists:
        raise FileExistsError(
            "partial inference cache is refused: response and terminal receipt must "
            "either both be absent or both exist"
        )
    if output_path.is_symlink() or receipt_path.is_symlink():
        raise ValueError("inference output cache must not contain symlinks")

    source_stage_root = Path(plan.source_stage_root)
    selected_api = api if api is not None else _load_pinned_weighttraits_api(stage_root)
    pre_backend_identity = _verify_selected_checkpoint_identity(
        plan,
        binding,
        downstream_stage_root=stage_root,
    )
    _verify_selected_binding_files(
        binding,
        source_stage_root=source_stage_root,
        downstream_stage_root=stage_root,
    )
    registry, prompt_payloads = _load_runtime_inputs(
        plan,
        stage_root=stage_root,
        api=selected_api,
    )
    planned_backend = _materialize_backend_template(
        plan.planned_backend_template, binding
    )
    request = build_inference_request(
        binding,
        manifest,
        cohort_key=plan.cohort_key,
        registry=plan.registry,
        panels=plan.panels,
        planned_backend_provenance=planned_backend,
        node_covariates=node_covariates,
    )
    _validate_registry_generation_contract(registry, request.generation_groups, request)

    if output_exists and receipt_exists:
        runtime_result = execute_all_node_inference(
            request,
            prompt_payloads={},
            generate_batch=None,
            observed_backend_provenance=None,
            output_path=str(output_path),
            receipt_path=str(receipt_path),
            existing_output=output_path.read_bytes(),
            existing_receipt=receipt_path.read_bytes(),
        )
        return FilesystemInferenceResult(
            cache_hit=True,
            plan=plan,
            plan_path=saved_plan_path,
            plan_file_sha256=plan_file_sha,
            mapping=mapping,
            output_path=output_path,
            receipt_path=receipt_path,
            runtime_result=runtime_result,
        )

    backend_type = selected_api.backend_class
    adapter = AllNodeBackendBinding(
        checkpoint=Path(pre_backend_identity.path),
        base_model_id=binding.base_model_id,
        base_model_revision=binding.base_model_revision,
        model_task=plan.model_task,
    )
    planned_from_runtime = backend_type.planned_provenance(
        adapter,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
    )
    if canonical_sha256(dict(planned_from_runtime)) != canonical_sha256(planned_backend):
        raise ValueError(
            "installed HuggingFace runtime provenance differs from the planned backend template"
        )
    backend = backend_type.load(
        adapter,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
    )
    observed_backend = dict(backend.provenance())
    if canonical_sha256(observed_backend) != canonical_sha256(planned_backend):
        raise ValueError("loaded HuggingFace backend provenance differs from its plan")

    def generate_batch(
        prompt_batch: Sequence[object],
        group: GenerationGroup,
        request_value: object,
    ) -> Sequence[str]:
        if getattr(request_value, "request_sha256", None) != request.request_sha256:
            raise AssertionError("runtime supplied an unexpected inference request")
        if not all(isinstance(prompt, str) for prompt in prompt_batch):
            raise TypeError("filesystem inference prompt payloads must be text")
        return backend.generate_batch(
            list(prompt_batch),
            model_task=request.model_task,
            generation_options=group.generation_options,
        )

    runtime_result = execute_all_node_inference(
        request,
        prompt_payloads=prompt_payloads,
        generate_batch=generate_batch,
        observed_backend_provenance=observed_backend,
        output_path=str(output_path),
        receipt_path=str(receipt_path),
    )
    if runtime_result.cache_hit or len(runtime_result.writes) != 2:
        raise AssertionError("fresh filesystem inference did not return two writes")
    post_backend_identity = _verify_selected_checkpoint_identity(
        plan,
        binding,
        downstream_stage_root=stage_root,
    )
    if post_backend_identity != pre_backend_identity:
        raise ValueError("selected checkpoint identity changed during inference")
    for expected_order, write in enumerate(runtime_result.writes):
        if write.publish_order != expected_order:
            raise AssertionError("runtime write publication order drift")
        _publish_exclusive_atomic(Path(write.path), write.payload, stage_root=stage_root)
    # Independently re-read and validate the just-published bytes through the
    # exact cache path before reporting terminal success.
    validated = execute_all_node_inference(
        request,
        prompt_payloads={},
        generate_batch=None,
        observed_backend_provenance=None,
        output_path=str(output_path),
        receipt_path=str(receipt_path),
        existing_output=output_path.read_bytes(),
        existing_receipt=receipt_path.read_bytes(),
    )
    if not validated.cache_hit:
        raise AssertionError("published inference artifacts did not validate as cache")
    return FilesystemInferenceResult(
        cache_hit=False,
        plan=plan,
        plan_path=saved_plan_path,
        plan_file_sha256=plan_file_sha,
        mapping=mapping,
        output_path=output_path,
        receipt_path=receipt_path,
        runtime_result=validated,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one exact all-node behavior inference array member"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--array-index",
        type=int,
        help="Exact flattened index 0..640; defaults to SLURM_ARRAY_TASK_ID",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attest_runtime_environment("before-runner")
    try:
        config, _plan, plan_path, config_path, config_file_sha = (
            load_inference_job_config(args.config)
        )
        _validate_launcher_environment(
            config,
            config_path=config_path,
            config_file_sha256=config_file_sha,
        )
        array_index = args.array_index
        if array_index is None:
            raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
            if raw_index is None:
                raise ValueError("--array-index or SLURM_ARRAY_TASK_ID is required")
            if not raw_index.isdecimal():
                raise ValueError("SLURM_ARRAY_TASK_ID must be a decimal integer")
            array_index = int(raw_index)
        result = execute_filesystem_inference(plan_path, array_index=array_index)
    finally:
        attest_runtime_environment("after-runner")
    print(
        json.dumps(
            result.status_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


def _validate_launcher_environment(
    config: InferenceJobConfig,
    *,
    config_path: Path,
    config_file_sha256: str,
) -> None:
    expected = {
        "WT_ALLNODE_COHORT_KEY": config.cohort_key,
        "WT_ALLNODE_JOB_CONFIG": str(config_path),
        "WT_ALLNODE_JOB_CONFIG_SHA256": config_file_sha256,
        "WT_ALLNODE_KIND": config.kind,
        "WT_ALLNODE_LOGICAL_ID": config.logical_id,
        "WT_ALLNODE_OUTPUT_DIR": config.output_root,
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise ValueError(f"launcher environment {name} drift")


def _plan_core(
    *,
    cohort_key: str,
    training_cohort_id: str,
    source_stage_root: str,
    model_task: str,
    checkpoint_identity_index: ArtifactPin,
    checkpoint_manifests: Sequence[TreeCheckpointManifest],
    node_covariates: Sequence[TreeNodeCovariates],
    flattened_mapping: Sequence[FlattenedNodeMapping],
    registry: ArtifactPin,
    panels: Sequence[PromptPanel],
    planned_backend_template: Mapping[str, Any],
    checkpoint_manifests_sha256: str,
    node_covariates_sha256: str,
    flattened_mapping_sha256: str,
    panels_sha256: str,
    planned_backend_template_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "valid": True,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "exclude_root": True,
        "cohort_key": cohort_key,
        "training_cohort_id": training_cohort_id,
        "source_stage_root": source_stage_root,
        "model_task": model_task,
        "checkpoint_identity_index": checkpoint_identity_index.to_dict(),
        "n_trees": PRODUCTION_N_TREES,
        "n_nodes": PRODUCTION_N_NODES,
        "n_internal": PRODUCTION_N_INTERNAL,
        "n_leaves": PRODUCTION_N_LEAVES,
        "tree_ids": list(PRODUCTION_TREE_IDS),
        "checkpoint_manifests": [value.to_dict() for value in checkpoint_manifests],
        "checkpoint_manifests_sha256": checkpoint_manifests_sha256,
        "node_covariates": [value.to_dict() for value in node_covariates],
        "node_covariates_sha256": node_covariates_sha256,
        "flattened_mapping": [value.to_dict() for value in flattened_mapping],
        "flattened_mapping_sha256": flattened_mapping_sha256,
        "registry": registry.to_dict(),
        "panels": [value.to_dict() for value in panels],
        "panels_sha256": panels_sha256,
        "planned_backend_template": dict(planned_backend_template),
        "planned_backend_template_sha256": planned_backend_template_sha256,
        "execution_overrides": {},
    }


def _load_runtime_inputs(
    plan: InferenceCohortPlan,
    *,
    stage_root: Path,
    api: WeightTraitsInferenceAPI,
) -> tuple[object, dict[ResponseCoordinate, str]]:
    _verify_file_pin(plan.registry, stage_root=stage_root, label="registry")
    registry = api.load_registry(Path(plan.registry.path))
    if Path(registry.path).resolve(strict=True) != Path(plan.registry.path):
        raise ValueError("loaded registry path differs from its inference plan pin")
    if registry.sha256 != plan.registry.sha256:
        raise ValueError("loaded registry hash differs from its inference plan pin")
    registry.validate_model_binding(
        model_task=plan.model_task,
        model_id=plan.checkpoint_manifests[0].bindings[0].base_model_id,
        model_revision=plan.checkpoint_manifests[0].bindings[0].base_model_revision,
    )
    fixtures = getattr(registry, "fixtures", {})
    if not isinstance(fixtures, Mapping):
        raise ValueError("loaded registry fixtures must be a mapping")
    for fixture_id, fixture in fixtures.items():
        fixture_path = getattr(fixture, "path", None)
        fixture_sha = getattr(fixture, "sha256", None)
        if not isinstance(fixture_path, Path) or not isinstance(fixture_sha, str):
            raise ValueError(f"loaded registry fixture {fixture_id!r} lacks path/hash")
        fixture_pin = ArtifactPin(str(fixture_path), fixture_sha)
        _verify_file_pin(
            fixture_pin,
            stage_root=stage_root,
            label=f"registry fixture {fixture_id}",
        )

    panels_by_protocol: dict[str, list[PromptPanel]] = {}
    for panel in plan.panels:
        panels_by_protocol.setdefault(panel.protocol_id, []).append(panel)
    prompts_by_coordinate: dict[tuple[str, str, str], str] = {}
    for protocol_id, protocol_panels in panels_by_protocol.items():
        artifact = protocol_panels[0].prompt_artifact
        materialization = protocol_panels[0].materialization_receipt
        _verify_file_pin(
            artifact,
            stage_root=stage_root,
            label=f"{protocol_id} prompt artifact",
        )
        _verify_file_pin(
            materialization,
            stage_root=stage_root,
            label=f"{protocol_id} materialization receipt",
        )
        loaded = list(api.load_prompts(Path(artifact.path)))
        expected_order = [
            (panel.probe_id, prompt_id)
            for panel in protocol_panels
            for prompt_id in panel.prompt_ids
        ]
        observed_order = [
            (
                _text(getattr(prompt, "probe_id", None), "loaded prompt probe_id"),
                _text(getattr(prompt, "prompt_id", None), "loaded prompt prompt_id"),
            )
            for prompt in loaded
        ]
        if observed_order != expected_order:
            raise ValueError(
                f"{protocol_id} rendered prompt order/IDs differ from the cohort plan"
            )
        protocol = registry.protocol(protocol_id)
        if tuple(protocol.probe_ids) != tuple(panel.probe_id for panel in protocol_panels):
            raise ValueError(f"{protocol_id} registry probe order differs from the plan")
        if protocol.samples_per_prompt != protocol_panels[0].samples_per_prompt:
            raise ValueError(f"{protocol_id} registry draw count differs from the plan")
        for panel in protocol_panels:
            if protocol.prompt_counts.get(panel.probe_id) != len(panel.prompt_ids):
                raise ValueError(
                    f"{protocol_id}/{panel.probe_id} registry prompt count differs"
                )
        for prompt in loaded:
            text = getattr(prompt, "prompt", None)
            if not isinstance(text, str):
                raise ValueError("loaded rendered prompt text must be a string")
            metadata = getattr(prompt, "metadata", None)
            if not isinstance(metadata, Mapping) or metadata.get("model_task") != plan.model_task:
                raise ValueError("rendered prompt model_task differs from the cohort plan")
            declared_sha = getattr(prompt, "prompt_sha256", None)
            if declared_sha != hashlib.sha256(text.encode("utf-8")).hexdigest():
                raise ValueError("rendered prompt text differs from its prompt_sha256")
            key = (protocol_id, prompt.probe_id, prompt.prompt_id)
            if key in prompts_by_coordinate:
                raise ValueError(f"duplicate rendered prompt coordinate {key!r}")
            prompts_by_coordinate[key] = text

    payloads: dict[ResponseCoordinate, str] = {}
    for panel in plan.panels:
        for prompt_id in panel.prompt_ids:
            prompt_text = prompts_by_coordinate[
                (panel.protocol_id, panel.probe_id, prompt_id)
            ]
            for sample_id in range(panel.samples_per_prompt):
                coordinate = ResponseCoordinate(
                    panel.protocol_id, panel.probe_id, prompt_id, sample_id
                )
                payloads[coordinate] = prompt_text
    return registry, payloads


def _validate_registry_generation_contract(
    registry: object,
    groups: Sequence[GenerationGroup],
    request: object,
) -> None:
    for group in groups:
        options = registry.protocol(group.protocol_id).generation_options(
            model_task=request.model_task,
            sample_id=group.sample_id,
        )
        if canonical_sha256(dict(options)) != canonical_sha256(
            dict(group.generation_options)
        ):
            raise ValueError(
                "registry generation options differ from the exact inference request for "
                f"{group.protocol_id}/{group.probe_id}/draw{group.sample_id}"
            )


def _verify_selected_binding_files(
    binding: NodeCheckpointBinding,
    *,
    source_stage_root: Path,
    downstream_stage_root: Path,
) -> None:
    for field in (
        "run_list",
        "ledger",
        "truth_manifest",
        "top_config",
        "training_summary",
    ):
        _verify_file_pin(
            getattr(binding, field),
            stage_root=source_stage_root,
            label=f"selected binding {field}",
        )
    _verify_file_pin(
        binding.completion_receipt,
        stage_root=downstream_stage_root,
        label="selected binding completion_receipt",
    )


def _verify_selected_checkpoint_identity(
    plan: InferenceCohortPlan,
    binding: NodeCheckpointBinding,
    *,
    downstream_stage_root: Path,
) -> CheckpointIdentity:
    """Re-attest one selected checkpoint through the sampled identity index.

    Full-model directories can be several GiB.  The explicit versioned
    identity scheme enumerates every file, hashes small files in full, and
    rechecks five fixed windows of each large file.  This avoids silently
    treating the sampled identity digest as a whole-directory SHA-256 while
    still detecting runtime drift before the backend loads the checkpoint.
    """

    _verify_file_pin(
        plan.checkpoint_identity_index,
        stage_root=downstream_stage_root,
        label="checkpoint identity index",
    )
    payload = _json_object(
        Path(plan.checkpoint_identity_index.path).read_bytes(),
        label="checkpoint identity index",
    )
    lookup = read_identity_index(payload)
    identity = lookup.get(binding.checkpoint.path)
    if identity is None:
        raise ValueError("selected checkpoint is absent from the identity index")
    if identity.identity_sha256 != binding.checkpoint.sha256:
        raise ValueError(
            "selected checkpoint binding digest differs from its identity index"
        )
    source_authority = source_authority_from_identity(identity)
    verified = validate_checkpoint_identity(
        identity,
        reattest_filesystem=True,
        source_authority=source_authority,
    )
    if verified.path != binding.checkpoint.path:
        raise ValueError("selected checkpoint identity path drift")
    return verified


def _verify_file_pin(pin: ArtifactPin, *, stage_root: Path, label: str) -> None:
    _verify_path_pin(
        pin,
        stage_root=stage_root,
        label=label,
        allow_directory=False,
    )


def _verify_path_pin(
    pin: ArtifactPin,
    *,
    stage_root: Path,
    label: str,
    allow_directory: bool,
) -> None:
    path = Path(pin.path)
    _require_beneath(path, stage_root, label=label)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if allow_directory:
        observed = _sha256_path(path)
    else:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a regular file: {path}")
        observed = _sha256_file(path)
    if observed != pin.sha256:
        raise ValueError(
            f"{label} sha256 mismatch: expected {pin.sha256}, observed {observed}"
        )


def _load_pinned_weighttraits_api(stage_root: Path) -> WeightTraitsInferenceAPI:
    try:
        from weighttraits.behavior import contracts as contracts_module
        from weighttraits.behavior import probe_inference as backend_module
        from weighttraits.behavior import responses as responses_module
    except ImportError as exc:  # pragma: no cover - production environment only
        raise RuntimeError("pinned WeightTraits behavior source is not importable") from exc
    expected_root = stage_root / "src" / "weighttraits"
    for module in (contracts_module, backend_module, responses_module):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError("pinned WeightTraits module lacks a filesystem source path")
        _require_beneath(
            Path(module_file).resolve(strict=True),
            expected_root,
            label=f"pinned source module {module.__name__}",
        )
    return WeightTraitsInferenceAPI(
        load_registry=contracts_module.load_behavior_protocol_registry,
        load_prompts=responses_module.load_rendered_behavior_prompts,
        backend_class=backend_module.HuggingFaceProbeBackend,
    )


def _publish_exclusive_atomic(path: Path, payload: bytes, *, stage_root: Path) -> None:
    expected_root = stage_root / OUTPUT_RELATIVE_ROOT
    _require_beneath(path, expected_root, label="inference output")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if parent.resolve(strict=True) != parent:
        raise ValueError("inference output parent must be canonical and contain no symlinks")
    if os.path.lexists(path):
        raise FileExistsError(f"exclusive inference output already exists: {path}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"exclusive inference output appeared during publication: {path}"
            ) from exc
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fixed_output_dir(
    stage_root: Path, cohort_key: str, mapping: FlattenedNodeMapping
) -> Path:
    components = (cohort_key, mapping.tree_id, mapping.node_id)
    for component in components:
        if not _SAFE_COMPONENT.fullmatch(component) or component in {".", ".."}:
            raise ValueError(f"unsafe inference output path component {component!r}")
    output = stage_root / OUTPUT_RELATIVE_ROOT / cohort_key / mapping.tree_id / mapping.node_id
    _require_beneath(output, stage_root / OUTPUT_RELATIVE_ROOT, label="inference output")
    return output


def _backend_template(
    value: object,
    *,
    binding: NodeCheckpointBinding,
    model_task: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != BACKEND_PROVENANCE_KEYS:
        raise ValueError(
            "planned_backend_template keys must be exactly the backend provenance keys"
        )
    template = dict(value)
    if template.get("checkpoint") != CHECKPOINT_PATH_SENTINEL:
        raise ValueError(
            f"planned_backend_template.checkpoint must be {CHECKPOINT_PATH_SENTINEL!r}"
        )
    materialized = _materialize_backend_template(template, binding)
    # build_inference_request performs the exact full backend validation; this
    # lightweight check provides a focused error before panel validation.
    expected_padding = "left" if model_task == "causal_lm" else "right"
    expected = {
        "backend": "huggingface",
        "device": DEVICE,
        "torch_dtype": TORCH_DTYPE,
        "local_files_only": True,
        "tokenizer_model_id": binding.base_model_id,
        "tokenizer_revision": binding.base_model_revision,
        "decode_skip_special_tokens": True,
        "decode_clean_up_tokenization_spaces": False,
        "generation_api": "ordered_probe_draw_batches",
        "generation_seed_api": "transformers.set_seed_once_per_probe_draw",
        "tokenizer_padding_side": expected_padding,
        "tokenizer_truncation": True,
        "tokenizer_max_length": 512,
        "generation_pad_token_id": "tokenizer.pad_token_id",
    }
    for field, expected_value in expected.items():
        if materialized.get(field) != expected_value:
            raise ValueError(
                f"planned_backend_template.{field} must be exactly {expected_value!r}"
            )
    for field in ("transformers_version", "torch_version"):
        _text(materialized.get(field), f"planned_backend_template.{field}")
    return template


def _materialize_backend_template(
    template: Mapping[str, Any], binding: NodeCheckpointBinding
) -> dict[str, Any]:
    result = dict(template)
    if result.get("checkpoint") != CHECKPOINT_PATH_SENTINEL:
        raise ValueError("planned backend template checkpoint sentinel drift")
    result["checkpoint"] = binding.checkpoint.path
    return result


def _manifest(
    value: TreeCheckpointManifest | Mapping[str, Any]
) -> TreeCheckpointManifest:
    if isinstance(value, TreeCheckpointManifest):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint manifest must be a mapping")
    return read_tree_checkpoint_manifest(value)


def _covariates(
    value: TreeNodeCovariates | Mapping[str, Any],
    manifest: TreeCheckpointManifest,
) -> TreeNodeCovariates:
    topology = build_tree_topology(
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
    if isinstance(value, TreeNodeCovariates):
        return validate_tree_node_covariates(value, topology, manifest)
    if not isinstance(value, Mapping):
        raise ValueError("node covariate contract must be a mapping")
    return read_tree_node_covariates(value, topology, manifest)


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
    raw_ids = value.get("prompt_ids")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError("prompt panel prompt_ids must be a nonempty sequence")
    return PromptPanel(
        protocol_id=_text(value.get("protocol_id"), "protocol_id"),
        probe_id=_text(value.get("probe_id"), "probe_id"),
        prompt_ids=tuple(_text(item, "prompt_id") for item in raw_ids),
        samples_per_prompt=_integer(
            value.get("samples_per_prompt"), "samples_per_prompt", minimum=1
        ),
        prompt_artifact=_pin(value.get("prompt_artifact"), "prompt_artifact"),
        materialization_receipt=_pin(
            value.get("materialization_receipt"), "materialization_receipt"
        ),
    )


def _mapping(value: object) -> FlattenedNodeMapping:
    if not isinstance(value, Mapping):
        raise ValueError("flattened mapping row must be a mapping")
    required = {"array_index", "tree_id", "node_ordinal", "node_id"}
    if set(value) != required:
        raise ValueError(f"flattened mapping keys must be exactly {sorted(required)!r}")
    return FlattenedNodeMapping(
        array_index=_integer(value.get("array_index"), "array_index", minimum=0),
        tree_id=_text(value.get("tree_id"), "tree_id"),
        node_ordinal=_integer(value.get("node_ordinal"), "node_ordinal", minimum=0),
        node_id=_text(value.get("node_id"), "node_id"),
    )


def _pin(value: ArtifactPin | Mapping[str, Any], label: str) -> ArtifactPin:
    if isinstance(value, ArtifactPin):
        value = value.to_dict()
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} pin must contain exactly path and sha256")
    path = _text(value.get("path"), f"{label}.path")
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"{label}.path must be canonical absolute")
    return ArtifactPin(path, _sha256(value.get("sha256"), f"{label}.sha256"))


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"pinned artifact must not be a symlink: {path}")
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"required pinned artifact is missing: {path}")
    root = path.resolve(strict=True)
    entries = sorted(root.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise ValueError(f"pinned directory contains a symlink: {path}")
    files = [entry for entry in entries if entry.is_file()]
    if not files:
        raise ValueError(f"pinned directory is empty: {path}")
    rows = [
        {
            "path": item.relative_to(root).as_posix(),
            "size": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in files
    ]
    return canonical_sha256(rows)


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_beneath(path: Path, root: Path, *, label: str) -> None:
    if not path.is_absolute() or not root.is_absolute():
        raise ValueError(f"{label} path and root must be absolute")
    normalized = Path(os.path.abspath(str(path)))
    normalized_root = Path(os.path.abspath(str(root)))
    try:
        normalized.relative_to(normalized_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay beneath {normalized_root}: {path}") from exc
    if os.path.lexists(path):
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"{label} resolves outside {resolved_root}: {path}"
            ) from exc
        if resolved_path != normalized:
            raise ValueError(f"{label} must be canonical and contain no symlinks: {path}")


def _reject_leaf_alias(value: object) -> None:
    if isinstance(value, Mapping):
        if "leaf_ordinal" in value:
            raise ValueError("leaf_ordinal aliases are forbidden in all-node inference")
        for nested in value.values():
            _reject_leaf_alias(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_leaf_alias(nested)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty normalized text")
    return value


def _safe_component(value: object, label: str) -> str:
    text = _text(value, label)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"{label} must be a safe path component")
    return text


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return text


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _canonical_absolute_root(value: object, label: str) -> str:
    path = _text(value, label)
    if not os.path.isabs(path) or os.path.normpath(path) != path or path == "/":
        raise ValueError(f"{label} must be a bounded canonical absolute path")
    return path


def _canonical_planned_inference_path(
    value: str | Path,
    cohort_key: str,
) -> Path:
    text = str(value)
    path = Path(text)
    if not path.is_absolute() or os.path.normpath(text) != text:
        raise ValueError("runtime_plan_path must be a canonical absolute path")
    if (
        path.name != PLAN_BASENAME
        or path.parent.name != cohort_key
        or path.parent.parent.name != "runtime_inputs"
        or path.parent.parent.parent.name != "allnode_v1"
        or path.parent.parent.parent.parent.name != "deployments"
    ):
        raise ValueError("runtime_plan_path differs from the unique inference plan path")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
