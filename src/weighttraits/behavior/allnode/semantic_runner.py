"""Filesystem production boundary for the all-trained-node semantic study.

The inference job publishes one terminal response receipt per observed trained
checkpoint.  This runner validates those receipts against the exact inference
plan, embeds one tree/endpoint at a time with the receipt-pinned offline
MiniLM snapshot, and computes the mean same-prompt/same-draw cosine-distance
matrix.  Text is never accumulated across nodes or trees: normalized
embeddings are staged in a temporary memmap and reduced one response
coordinate at a time.

All permanent files are exclusive, same-filesystem atomic publications.  A
tree/endpoint terminal receipt is published last.  Existing artifacts are
accepted only as a complete, hash-valid cache; partial directories fail
closed.  The module has no scheduler, recovery, overwrite, or submission API.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from .audit import (
    MINILM_MODEL,
    MINILM_RECEIPT_SHA256,
    MINILM_REVISION,
    MINILM_RUNTIME,
    MINILM_SNAPSHOT_SHA256,
    expected_analysis_contract,
    semantic_endpoint_rows,
)
from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    PRODUCTION_N_INTERNAL,
    PRODUCTION_N_LEAVES,
    PRODUCTION_N_NODES,
    PRODUCTION_N_TREES,
    TreeCheckpointManifest,
)
from .direct_execution import (
    DIRECT_ARTIFACT_RECEIPT_SCHEMA,
    DIRECT_VALUES_SCHEMA,
)
from .direct_runner import (
    FilesystemDirectResult,
    validate_direct_cohort_terminal as validate_direct_production_terminal,
)
from .inference import (
    PromptPanel,
    build_inference_request,
)
from .jsonl import decode_lf_json_objects
from .inference_runner import (
    CHECKPOINT_PATH_SENTINEL,
    InferenceCohortPlan,
    load_inference_cohort_plan,
)
from .encoder_snapshot import require_encoder_snapshot_sha256
from .manifest import NodeSpec, TreeTopology, build_tree_topology, canonical_sha256
from .matrices import ValidatedDistanceMatrix, validate_distance_matrix
from .production import PRODUCTION_TREE_IDS
from .provenance import attest_runtime_environment
from .runtime import RuntimeExecutionResult, execute_all_node_inference
from .semantic_analysis import (
    DIRECT_ARTIFACT_KIND,
    SEMANTIC_ARTIFACT_KIND,
    SEMANTIC_METRIC,
    TreeSemanticAnalysis,
    aggregate_production_semantic_runset_streaming,
    analyze_tree_semantic_bridge,
    bind_distance_artifact,
)


PLAN_SCHEMA = "weighttraits.all_node_semantic_cohort_plan.v1"
JOB_CONFIG_SCHEMA = "weighttraits.all_node_semantic_job_config.v1"
SEMANTIC_MATRIX_SCHEMA = "weighttraits.all_node_semantic_matrix_artifact.v1"
TREE_TERMINAL_SCHEMA = "weighttraits.all_node_semantic_tree_endpoint.v1"
ENDPOINT_RUNSET_SCHEMA = "weighttraits.all_node_semantic_endpoint_runset.v1"
COHORT_TERMINAL_SCHEMA = "weighttraits.all_node_semantic_cohort_terminal.v1"
CLI_STATUS_SCHEMA = "weighttraits.all_node_semantic_cli_status.v1"

PLAN_BASENAME = "semantic_plan.json"
INFERENCE_PLAN_BASENAME = "inference_plan.json"
DIRECT_RELATIVE_ROOT = Path("outputs/allnode_v1/direct")
INFERENCE_RELATIVE_ROOT = Path("outputs/allnode_v1/inference")
SEMANTIC_RELATIVE_ROOT = Path("outputs/allnode_v1/semantic")
ENCODER_RECEIPT_SCHEMA = "weighttraits.behavior.encoder_receipt.v1"
ENCODER_MODE = "pinned_internal_minilm"
MINILM_PACKAGE = "sentence-transformers"
MINILM_VERSION = MINILM_RUNTIME.split("==", 1)[1]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class TextEncoder(Protocol):
    """Minimal batched encoder used by the streaming executor."""

    def __call__(self, texts: Sequence[str]) -> object:
        """Return a real two-dimensional array, one row per input text."""


@dataclass(frozen=True)
class SemanticEndpointPlan:
    endpoint_id: str
    protocol_id: str
    probe_id: str
    prompt_ids: tuple[str, ...]
    samples_per_prompt: int
    n_paired_coordinates: int
    coordinates_sha256: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["prompt_ids"] = list(self.prompt_ids)
        return row


@dataclass(frozen=True)
class SemanticCohortPlan:
    cohort_key: str
    training_cohort_id: str
    stage_root: str
    deployment_root: str
    inference_plan: ArtifactPin
    direct_job_config: ArtifactPin
    direct_output_root: str
    inference_output_root: str
    output_root: str
    encoder_receipt: ArtifactPin
    encoder_snapshot: ArtifactPin
    encoder_runtime: str
    endpoints: tuple[SemanticEndpointPlan, ...]
    tree_scope: tuple[Mapping[str, Any], ...]
    topology_scope_sha256: str
    checkpoint_scope_sha256: str
    node_covariate_scope_sha256: str
    endpoint_scope_sha256: str
    analysis_contract: Mapping[str, Any]
    analysis_contract_sha256: str
    semantic_plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        core = _semantic_plan_core(
            cohort_key=self.cohort_key,
            training_cohort_id=self.training_cohort_id,
            stage_root=self.stage_root,
            deployment_root=self.deployment_root,
            inference_plan=self.inference_plan,
            direct_job_config=self.direct_job_config,
            direct_output_root=self.direct_output_root,
            inference_output_root=self.inference_output_root,
            output_root=self.output_root,
            encoder_receipt=self.encoder_receipt,
            encoder_snapshot=self.encoder_snapshot,
            encoder_runtime=self.encoder_runtime,
            endpoints=self.endpoints,
            tree_scope=self.tree_scope,
            topology_scope_sha256=self.topology_scope_sha256,
            checkpoint_scope_sha256=self.checkpoint_scope_sha256,
            node_covariate_scope_sha256=self.node_covariate_scope_sha256,
            endpoint_scope_sha256=self.endpoint_scope_sha256,
            analysis_contract=self.analysis_contract,
            analysis_contract_sha256=self.analysis_contract_sha256,
        )
        return {**core, "semantic_plan_sha256": self.semantic_plan_sha256}


@dataclass(frozen=True)
class SemanticJobConfig:
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
class NodeEndpointSource:
    node_id: str
    node_ordinal: int
    node_role: str
    responses: ArtifactPin
    terminal_receipt: ArtifactPin
    request_sha256: str
    endpoint_rows_sha256: str
    texts: tuple[str, ...]

    def scope_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_ordinal": self.node_ordinal,
            "node_role": self.node_role,
            "responses": self.responses.to_dict(),
            "terminal_receipt": self.terminal_receipt.to_dict(),
            "request_sha256": self.request_sha256,
            "endpoint_rows_sha256": self.endpoint_rows_sha256,
        }


@dataclass(frozen=True)
class DirectCosineSource:
    matrix: ValidatedDistanceMatrix
    matrix_artifact: ArtifactPin
    artifact_receipt: ArtifactPin


@dataclass(frozen=True)
class TreeEndpointResult:
    cache_hit: bool
    analysis: TreeSemanticAnalysis
    matrix_artifact: ArtifactPin
    analysis_artifact: ArtifactPin
    terminal_receipt: ArtifactPin
    terminal_payload: Mapping[str, Any]


@dataclass(frozen=True)
class FilesystemSemanticResult:
    cache_hit: bool
    plan: SemanticCohortPlan
    config: SemanticJobConfig
    output_root: Path
    endpoint_runsets: tuple[ArtifactPin, ...]
    terminal_receipt: ArtifactPin
    terminal_payload: Mapping[str, Any]

    def status_dict(self) -> dict[str, Any]:
        return {
            "schema": CLI_STATUS_SCHEMA,
            "valid": True,
            "status": "cached" if self.cache_hit else "completed",
            "cache_hit": self.cache_hit,
            "cohort_key": self.plan.cohort_key,
            "training_cohort_id": self.plan.training_cohort_id,
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "n_trees": PRODUCTION_N_TREES,
            "n_nodes": PRODUCTION_N_NODES,
            "n_internal": PRODUCTION_N_INTERNAL,
            "n_leaves": PRODUCTION_N_LEAVES,
            "endpoint_runsets": [pin.to_dict() for pin in self.endpoint_runsets],
            "terminal_receipt": self.terminal_receipt.to_dict(),
            "execution_overrides": {},
        }


def build_semantic_cohort_plan(
    inference: InferenceCohortPlan,
    *,
    stage_root: str,
    inference_plan: ArtifactPin | Mapping[str, Any],
    direct_job_config: ArtifactPin | Mapping[str, Any],
    encoder_receipt: ArtifactPin | Mapping[str, Any],
    encoder_snapshot: ArtifactPin | Mapping[str, Any],
) -> SemanticCohortPlan:
    """Freeze the exact semantic execution contract for one production cohort."""

    if not isinstance(inference, InferenceCohortPlan):
        raise ValueError("inference must be an InferenceCohortPlan")
    stage = _canonical_root(stage_root, "stage_root")
    deployment = str(Path(stage) / "deployments" / "allnode_v1")
    inference_pin = _pin(inference_plan, "inference_plan")
    expected_inference_path = str(
        Path(deployment)
        / "runtime_inputs"
        / inference.cohort_key
        / INFERENCE_PLAN_BASENAME
    )
    if inference_pin.path != expected_inference_path:
        raise ValueError("semantic inference-plan pin path drift")
    direct_config_pin = _pin(direct_job_config, "direct_job_config")
    expected_direct_config_path = str(
        Path(deployment)
        / "configs"
        / "jobs"
        / f"direct_{inference.cohort_key}.json"
    )
    if direct_config_pin.path != expected_direct_config_path:
        raise ValueError("semantic direct job-config pin path drift")
    receipt_pin = _pin(encoder_receipt, "encoder_receipt")
    if receipt_pin.sha256 != MINILM_RECEIPT_SHA256:
        raise ValueError("semantic MiniLM receipt hash drift")
    snapshot_pin = _pin(encoder_snapshot, "encoder_snapshot")
    if snapshot_pin.sha256 != MINILM_SNAPSHOT_SHA256:
        raise ValueError("semantic MiniLM snapshot hash drift")

    endpoints = tuple(_endpoint_from_panel(panel) for panel in inference.panels)
    expected_endpoints = semantic_endpoint_rows(inference.cohort_key)
    if [
        {
            "endpoint_id": row.endpoint_id,
            "protocol_id": row.protocol_id,
            "probe_id": row.probe_id,
            "n_prompts": len(row.prompt_ids),
            "samples_per_prompt": row.samples_per_prompt,
            "n_paired_coordinates": row.n_paired_coordinates,
        }
        for row in endpoints
    ] != expected_endpoints:
        raise ValueError("semantic endpoints differ from the audited cohort contract")
    tree_scope = tuple(
        {
            "tree_id": manifest.tree_id,
            "topology_sha256": manifest.topology_sha256,
            "pair_index_sha256": manifest.pair_index_sha256,
            "leaf_subset_sha256": manifest.leaf_subset_sha256,
            "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
            "node_covariates_sha256": covariates.covariates_sha256,
        }
        for manifest, covariates in zip(
            inference.checkpoint_manifests, inference.node_covariates
        )
    )
    if tuple(row["tree_id"] for row in tree_scope) != PRODUCTION_TREE_IDS:
        raise ValueError("semantic tree scope must be exactly trees 001..050")
    topology_scope_sha = canonical_sha256(
        [
            {
                key: row[key]
                for key in ("tree_id", "topology_sha256", "pair_index_sha256", "leaf_subset_sha256")
            }
            for row in tree_scope
        ]
    )
    checkpoint_scope_sha = canonical_sha256(
        [
            {"tree_id": row["tree_id"], "checkpoint_set_sha256": row["checkpoint_set_sha256"]}
            for row in tree_scope
        ]
    )
    covariate_scope_sha = canonical_sha256(
        [
            {"tree_id": row["tree_id"], "node_covariates_sha256": row["node_covariates_sha256"]}
            for row in tree_scope
        ]
    )
    endpoint_scope_sha = canonical_sha256([row.to_dict() for row in endpoints])
    analysis = expected_analysis_contract()
    analysis_sha = canonical_sha256(analysis)
    core = _semantic_plan_core(
        cohort_key=inference.cohort_key,
        training_cohort_id=inference.training_cohort_id,
        stage_root=stage,
        deployment_root=deployment,
        inference_plan=inference_pin,
        direct_job_config=direct_config_pin,
        direct_output_root=str(Path(stage) / DIRECT_RELATIVE_ROOT / inference.cohort_key),
        inference_output_root=str(Path(stage) / INFERENCE_RELATIVE_ROOT / inference.cohort_key),
        output_root=str(Path(stage) / SEMANTIC_RELATIVE_ROOT / inference.cohort_key),
        encoder_receipt=receipt_pin,
        encoder_snapshot=snapshot_pin,
        encoder_runtime=MINILM_RUNTIME,
        endpoints=endpoints,
        tree_scope=tree_scope,
        topology_scope_sha256=topology_scope_sha,
        checkpoint_scope_sha256=checkpoint_scope_sha,
        node_covariate_scope_sha256=covariate_scope_sha,
        endpoint_scope_sha256=endpoint_scope_sha,
        analysis_contract=analysis,
        analysis_contract_sha256=analysis_sha,
    )
    return SemanticCohortPlan(
        cohort_key=inference.cohort_key,
        training_cohort_id=inference.training_cohort_id,
        stage_root=stage,
        deployment_root=deployment,
        inference_plan=inference_pin,
        direct_job_config=direct_config_pin,
        direct_output_root=core["direct_output_root"],
        inference_output_root=core["inference_output_root"],
        output_root=core["output_root"],
        encoder_receipt=receipt_pin,
        encoder_snapshot=snapshot_pin,
        encoder_runtime=MINILM_RUNTIME,
        endpoints=endpoints,
        tree_scope=tree_scope,
        topology_scope_sha256=topology_scope_sha,
        checkpoint_scope_sha256=checkpoint_scope_sha,
        node_covariate_scope_sha256=covariate_scope_sha,
        endpoint_scope_sha256=endpoint_scope_sha,
        analysis_contract=analysis,
        analysis_contract_sha256=analysis_sha,
        semantic_plan_sha256=canonical_sha256(core),
    )


def read_semantic_cohort_plan(
    payload: Mapping[str, Any],
    *,
    inference: InferenceCohortPlan,
) -> SemanticCohortPlan:
    """Rebuild a saved plan and reject every field or self-hash drift."""

    if not isinstance(payload, Mapping):
        raise ValueError("semantic cohort plan must be a mapping")
    required = set(_semantic_plan_core_keys()) | {"semantic_plan_sha256"}
    if set(payload) != required:
        raise ValueError(f"semantic cohort plan keys must be exactly {sorted(required)!r}")
    if payload.get("schema") != PLAN_SCHEMA or payload.get("valid") is not True:
        raise ValueError("semantic cohort plan schema or validity drift")
    rebuilt = build_semantic_cohort_plan(
        inference,
        stage_root=payload.get("stage_root"),
        inference_plan=payload.get("inference_plan"),
        direct_job_config=payload.get("direct_job_config"),
        encoder_receipt=payload.get("encoder_receipt"),
        encoder_snapshot=payload.get("encoder_snapshot"),
    )
    if canonical_sha256(dict(payload)) != canonical_sha256(rebuilt.to_dict()):
        raise ValueError("saved semantic cohort plan differs from recomputed contract")
    return rebuilt


def load_semantic_cohort_plan(
    path: str | Path,
) -> tuple[SemanticCohortPlan, InferenceCohortPlan, Path, str]:
    """Load the unique plan path plus its independently pinned inference plan."""

    plan_path = _canonical_file(path, "semantic plan")
    if (
        plan_path.name != PLAN_BASENAME
        or plan_path.parent.parent.name != "runtime_inputs"
        or plan_path.parent.parent.parent.name != "allnode_v1"
        or plan_path.parent.parent.parent.parent.name != "deployments"
    ):
        raise ValueError(
            "semantic plan must be deployments/allnode_v1/runtime_inputs/"
            "<cohort_key>/semantic_plan.json"
        )
    payload_bytes = plan_path.read_bytes()
    payload = _json_object(payload_bytes, "semantic plan")
    if payload_bytes != _canonical_bytes(payload):
        raise ValueError("semantic plan is not canonical JSON bytes")
    raw_inference_pin = _pin(payload.get("inference_plan"), "inference_plan")
    observed_inference_sha = _sha256_file(Path(raw_inference_pin.path))
    if observed_inference_sha != raw_inference_pin.sha256:
        raise ValueError("semantic plan inference-plan file hash drift")
    inference, inference_path, _, _ = load_inference_cohort_plan(raw_inference_pin.path)
    if str(inference_path) != raw_inference_pin.path:
        raise ValueError("semantic plan inference-plan canonical path drift")
    plan = read_semantic_cohort_plan(payload, inference=inference)
    if plan_path.parent.name != plan.cohort_key:
        raise ValueError("semantic plan directory and cohort key differ")
    return plan, inference, plan_path, hashlib.sha256(payload_bytes).hexdigest()


def build_semantic_job_config(
    plan: SemanticCohortPlan,
    *,
    runtime_plan_file_sha256: str,
) -> SemanticJobConfig:
    """Build the launcher-consumed immutable job config for one semantic job."""

    if not isinstance(plan, SemanticCohortPlan):
        raise ValueError("plan must be a SemanticCohortPlan")
    runtime_plan = ArtifactPin(
        str(
            Path(plan.deployment_root)
            / "runtime_inputs"
            / plan.cohort_key
            / PLAN_BASENAME
        ),
        _sha256(runtime_plan_file_sha256, "runtime_plan_file_sha256"),
    )
    core = {
        "schema": JOB_CONFIG_SCHEMA,
        "valid": True,
        "logical_id": f"semantic_{plan.cohort_key}",
        "cohort_key": plan.cohort_key,
        "kind": "semantic_allnode_gpu",
        "runtime_plan": runtime_plan.to_dict(),
        "output_root": plan.output_root,
        "execution_overrides": {},
    }
    return SemanticJobConfig(
        logical_id=core["logical_id"],
        cohort_key=plan.cohort_key,
        kind=core["kind"],
        runtime_plan=runtime_plan,
        output_root=plan.output_root,
        config_sha256=canonical_sha256(core),
    )


def read_semantic_job_config(payload: Mapping[str, Any]) -> SemanticJobConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("semantic job config must be a mapping")
    required = {
        "schema", "valid", "logical_id", "cohort_key", "kind",
        "runtime_plan", "output_root", "execution_overrides", "config_sha256",
    }
    if set(payload) != required:
        raise ValueError("semantic job config keys drift")
    core = {key: payload[key] for key in required if key != "config_sha256"}
    if (
        core["schema"] != JOB_CONFIG_SCHEMA
        or core["valid"] is not True
        or core["kind"] != "semantic_allnode_gpu"
        or core["execution_overrides"] != {}
        or payload["config_sha256"] != canonical_sha256(core)
    ):
        raise ValueError("semantic job config contract drift")
    cohort = _component(core["cohort_key"], "cohort_key")
    logical = _component(core["logical_id"], "logical_id")
    if logical != f"semantic_{cohort}":
        raise ValueError("semantic job logical ID drift")
    return SemanticJobConfig(
        logical_id=logical,
        cohort_key=cohort,
        kind=core["kind"],
        runtime_plan=_pin(core["runtime_plan"], "runtime_plan"),
        output_root=_canonical_root(core["output_root"], "output_root", require_exists=False),
        config_sha256=_sha256(payload["config_sha256"], "config_sha256"),
    )


def load_semantic_job_config(
    path: str | Path,
) -> tuple[SemanticJobConfig, SemanticCohortPlan, InferenceCohortPlan, Path, str]:
    """Load the unique job config and its exact semantic/inference plans."""

    config_path = _canonical_file(path, "semantic job config")
    if (
        config_path.parent.name != "jobs"
        or config_path.parent.parent.name != "configs"
        or config_path.parent.parent.parent.name != "allnode_v1"
        or config_path.parent.parent.parent.parent.name != "deployments"
    ):
        raise ValueError(
            "semantic config must be deployments/allnode_v1/configs/jobs/"
            "semantic_<cohort>.json"
        )
    raw = config_path.read_bytes()
    raw_payload = _json_object(raw, "semantic job config")
    if raw != _canonical_bytes(raw_payload):
        raise ValueError("semantic job config is not canonical JSON bytes")
    config = read_semantic_job_config(raw_payload)
    if config_path.name != f"semantic_{config.cohort_key}.json":
        raise ValueError("semantic config filename and cohort key differ")
    if _sha256_file(Path(config.runtime_plan.path)) != config.runtime_plan.sha256:
        raise ValueError("semantic runtime plan file hash drift")
    plan, inference, plan_path, _ = load_semantic_cohort_plan(config.runtime_plan.path)
    if str(plan_path) != config.runtime_plan.path:
        raise ValueError("semantic runtime plan path drift")
    if (
        plan.cohort_key != config.cohort_key
        or plan.output_root != config.output_root
        or config.logical_id != f"semantic_{plan.cohort_key}"
    ):
        raise ValueError("semantic job config and runtime plan scope differ")
    return config, plan, inference, config_path, hashlib.sha256(raw).hexdigest()


def execute_filesystem_semantic(
    config_path: str | Path,
    *,
    encoder: TextEncoder | None = None,
    allow_test_encoder: bool = False,
) -> FilesystemSemanticResult:
    """Execute/cache every endpoint with replayable one-tree aggregation."""

    config, plan, inference, _, _ = load_semantic_job_config(config_path)
    if encoder is None:
        snapshot = validate_encoder_receipt(plan)
        active_encoder = load_pinned_minilm_encoder(snapshot)
    else:
        if not allow_test_encoder:
            raise ValueError("custom semantic encoders are test-only")
        active_encoder = encoder
    validated_direct = validate_direct_cohort_terminal(plan)
    direct_terminal_path = _canonical_file(
        validated_direct.output_dir / "terminal_receipt.json",
        "validated direct cohort terminal",
    )
    direct_terminal_pin = ArtifactPin(
        str(direct_terminal_path), _sha256_file(direct_terminal_path)
    )
    output_root = Path(plan.output_root)
    if os.path.lexists(output_root) and output_root.is_symlink():
        raise ValueError("semantic output root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.resolve(strict=True) != output_root:
        raise ValueError("semantic output root must be canonical")
    allowed = {endpoint.endpoint_id for endpoint in plan.endpoints} | {"terminal_receipt.json"}
    unknown = sorted(entry.name for entry in output_root.iterdir() if entry.name not in allowed)
    if unknown:
        raise ValueError(f"semantic output root contains unexpected entries: {unknown!r}")
    cohort_terminal_path = output_root / "terminal_receipt.json"
    existed_terminal = cohort_terminal_path.exists()
    if existed_terminal and cohort_terminal_path.is_symlink():
        raise ValueError("semantic cohort terminal must not be a symlink")

    runset_pins: list[ArtifactPin] = []
    for endpoint in plan.endpoints:
        endpoint_root = output_root / endpoint.endpoint_id
        if os.path.lexists(endpoint_root) and endpoint_root.is_symlink():
            raise ValueError("semantic endpoint output root must not be a symlink")
        endpoint_root.mkdir(parents=True, exist_ok=True)
        expected_entries = set(PRODUCTION_TREE_IDS) | {"runset_receipt.json"}
        unexpected = sorted(
            entry.name for entry in endpoint_root.iterdir() if entry.name not in expected_entries
        )
        if unexpected:
            raise ValueError(
                f"semantic endpoint root contains unexpected entries: {unexpected!r}"
            )

        def analysis_factory(endpoint_value: SemanticEndpointPlan = endpoint):
            for tree_position, (manifest, covariates) in enumerate(
                zip(inference.checkpoint_manifests, inference.node_covariates)
            ):
                topology = topology_from_manifest(manifest)
                direct = load_direct_cosine_source(
                    Path(plan.direct_output_root) / topology.tree_id,
                    topology=topology,
                    manifest=manifest,
                    node_covariates_sha256=covariates.covariates_sha256,
                )
                result = execute_tree_endpoint_streaming(
                    cohort_key=plan.cohort_key,
                    endpoint=endpoint_value,
                    topology=topology,
                    manifest=manifest,
                    node_covariates=covariates,
                    direct=direct,
                    response_loader=lambda ordinal, position=tree_position: load_node_endpoint_source(
                        inference,
                        stage_root=plan.stage_root,
                        tree_position=position,
                        node_ordinal=ordinal,
                        endpoint=endpoint_value,
                    ),
                    encoder=active_encoder,
                    output_dir=endpoint_root / topology.tree_id,
                    encoder_receipt_sha256=plan.encoder_receipt.sha256,
                )
                yield result.analysis

        manifest_node_counts = tuple(
            manifest.n_nodes for manifest in inference.checkpoint_manifests
        )
        runset = aggregate_production_semantic_runset_streaming(
            analysis_factory,
            expected_nodes_per_tree=manifest_node_counts,
        )
        tree_terminal_rows = []
        for tree_id in PRODUCTION_TREE_IDS:
            path = _canonical_file(
                endpoint_root / tree_id / "terminal_receipt.json",
                "semantic tree terminal",
            )
            tree_terminal_rows.append(
                {
                    "tree_id": tree_id,
                    "terminal_receipt": str(path),
                    "terminal_receipt_sha256": _sha256_file(path),
                }
            )
        runset_core = {
            "cohort_key": plan.cohort_key,
            "training_cohort_id": plan.training_cohort_id,
            "endpoint": endpoint.to_dict(),
            "population": POPULATION,
            "internal_node_treatment": INTERNAL_NODE_TREATMENT,
            "root_excluded": True,
            "n_trees": PRODUCTION_N_TREES,
            "n_nodes": PRODUCTION_N_NODES,
            "n_internal": PRODUCTION_N_INTERNAL,
            "n_leaves": PRODUCTION_N_LEAVES,
            "semantic_plan_sha256": plan.semantic_plan_sha256,
            "direct_job_config": plan.direct_job_config.to_dict(),
            "direct_cohort_terminal": direct_terminal_pin.to_dict(),
            "tree_terminals": tree_terminal_rows,
            "tree_terminal_scope_sha256": canonical_sha256(tree_terminal_rows),
            "analysis": runset.to_dict(),
            "analysis_sha256": runset.analysis_sha256,
            "analysis_contract_sha256": plan.analysis_contract_sha256,
            "execution_overrides": {},
        }
        runset_payload = {
            "schema": ENDPOINT_RUNSET_SCHEMA,
            "valid": True,
            "status": "completed",
            **runset_core,
            "terminal_sha256": canonical_sha256(runset_core),
        }
        runset_path = endpoint_root / "runset_receipt.json"
        if runset_path.exists():
            if runset_path.is_symlink() or _json_object(
                runset_path.read_bytes(), "semantic runset receipt"
            ) != runset_payload:
                raise ValueError("semantic runset receipt differs from recomputed result")
        else:
            _publish_exclusive_atomic(runset_path, _canonical_bytes(runset_payload))
        runset_pins.append(ArtifactPin(str(runset_path), _sha256_file(runset_path)))

    cohort_core = {
        "cohort_key": plan.cohort_key,
        "training_cohort_id": plan.training_cohort_id,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "root_excluded": True,
        "n_trees": PRODUCTION_N_TREES,
        "n_nodes": PRODUCTION_N_NODES,
        "n_internal": PRODUCTION_N_INTERNAL,
        "n_leaves": PRODUCTION_N_LEAVES,
        "semantic_plan_sha256": plan.semantic_plan_sha256,
        "direct_job_config": plan.direct_job_config.to_dict(),
        "direct_cohort_terminal": direct_terminal_pin.to_dict(),
        "analysis_contract_sha256": plan.analysis_contract_sha256,
        "endpoint_runsets": [pin.to_dict() for pin in runset_pins],
        "endpoint_scope_sha256": plan.endpoint_scope_sha256,
        "execution_overrides": {},
    }
    cohort_payload = {
        "schema": COHORT_TERMINAL_SCHEMA,
        "valid": True,
        "status": "completed",
        **cohort_core,
        "terminal_sha256": canonical_sha256(cohort_core),
    }
    if existed_terminal:
        if _json_object(cohort_terminal_path.read_bytes(), "semantic cohort terminal") != cohort_payload:
            raise ValueError("semantic cohort terminal differs from recomputed result")
    else:
        _publish_exclusive_atomic(cohort_terminal_path, _canonical_bytes(cohort_payload))
    terminal_pin = ArtifactPin(str(cohort_terminal_path), _sha256_file(cohort_terminal_path))
    return FilesystemSemanticResult(
        cache_hit=existed_terminal,
        plan=plan,
        config=config,
        output_root=output_root,
        endpoint_runsets=tuple(runset_pins),
        terminal_receipt=terminal_pin,
        terminal_payload=cohort_payload,
    )


def validate_direct_cohort_terminal(
    plan: SemanticCohortPlan,
) -> FilesystemDirectResult:
    """Validate the exact direct job-config boundary before reading matrices."""

    config_path = _canonical_file(plan.direct_job_config.path, "direct job config")
    if _sha256_file(config_path) != plan.direct_job_config.sha256:
        raise ValueError("semantic direct job-config file hash drift")
    result = validate_direct_production_terminal(config_path)
    if (
        str(result.job_config_path) != plan.direct_job_config.path
        or result.job_config_file_sha256 != plan.direct_job_config.sha256
        or result.plan.cohort_key != plan.cohort_key
        or result.plan.training_cohort_id != plan.training_cohort_id
        or str(result.output_dir) != plan.direct_output_root
    ):
        raise ValueError("validated direct dependency and semantic plan scope differ")
    return result


def execute_tree_endpoint_streaming(
    *,
    cohort_key: str,
    endpoint: SemanticEndpointPlan,
    topology: TreeTopology,
    manifest: TreeCheckpointManifest,
    node_covariates: object,
    direct: DirectCosineSource,
    response_loader: Callable[[int], NodeEndpointSource],
    encoder: TextEncoder,
    output_dir: str | Path,
    encoder_receipt_sha256: str,
) -> TreeEndpointResult:
    """Execute/cache one tree/endpoint without retaining a cohort's texts."""

    if not callable(response_loader) or not callable(encoder):
        raise ValueError("response_loader and encoder must be callable")
    endpoint = _validate_endpoint(endpoint)
    destination = Path(_canonical_root(output_dir, "output_dir", require_exists=False))
    if os.path.lexists(destination) and (
        destination.is_symlink() or not destination.is_dir()
    ):
        raise ValueError("semantic tree/endpoint output must be a real directory")
    matrix_path = destination / "semantic_matrix.json"
    analysis_path = destination / "tree_analysis.json"
    terminal_path = destination / "terminal_receipt.json"
    existence = tuple(os.path.lexists(path) for path in (matrix_path, analysis_path, terminal_path))
    if destination.exists() and (
        not all(existence)
        or {entry.name for entry in destination.iterdir()}
        != {"semantic_matrix.json", "tree_analysis.json", "terminal_receipt.json"}
    ):
        raise FileExistsError("partial semantic tree/endpoint cache is refused")
    if any(path.is_symlink() for path in (matrix_path, analysis_path, terminal_path)):
        raise ValueError("semantic tree/endpoint cache must not contain symlinks")

    response_scope: list[dict[str, Any]] = []
    for ordinal in range(topology.n_nodes):
        source = response_loader(ordinal)
        _validate_node_source(source, topology=topology, endpoint=endpoint, ordinal=ordinal)
        response_scope.append(source.scope_dict())
    response_scope_sha = canonical_sha256(response_scope)

    cache_hit = all(existence)
    if cache_hit:
        semantic_matrix = _read_semantic_matrix_artifact(
            matrix_path,
            topology=topology,
            endpoint=endpoint,
            checkpoint_set_sha256=manifest.checkpoint_set_sha256,
            node_covariates_sha256=getattr(node_covariates, "covariates_sha256", None),
            response_scope_sha256=response_scope_sha,
            encoder_receipt_sha256=encoder_receipt_sha256,
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        semantic_matrix = _stream_semantic_matrix(
            response_loader,
            topology=topology,
            endpoint=endpoint,
            encoder=encoder,
            temp_parent=destination.parent,
            expected_scope=response_scope,
        )
        matrix_payload = _semantic_matrix_payload(
            semantic_matrix,
            endpoint=endpoint,
            checkpoint_set_sha256=manifest.checkpoint_set_sha256,
            node_covariates_sha256=getattr(node_covariates, "covariates_sha256", None),
            response_scope=response_scope,
            response_scope_sha256=response_scope_sha,
            encoder_receipt_sha256=encoder_receipt_sha256,
        )
        matrix_bytes = _canonical_bytes(matrix_payload)

    matrix_pin = ArtifactPin(
        str(matrix_path),
        _sha256_file(matrix_path) if cache_hit else hashlib.sha256(matrix_bytes).hexdigest(),
    )
    semantic_bound = bind_distance_artifact(
        semantic_matrix,
        artifact_kind=SEMANTIC_ARTIFACT_KIND,
        endpoint_id=endpoint.endpoint_id,
        checkpoint_set_sha256=manifest.checkpoint_set_sha256,
        source_receipt_sha256=matrix_pin.sha256,
    )
    direct_bound = bind_distance_artifact(
        direct.matrix,
        artifact_kind=DIRECT_ARTIFACT_KIND,
        endpoint_id="weight_cosine",
        checkpoint_set_sha256=manifest.checkpoint_set_sha256,
        source_receipt_sha256=direct.artifact_receipt.sha256,
    )
    analysis = analyze_tree_semantic_bridge(
        topology,
        manifest,
        cohort_key=cohort_key,
        endpoint_id=endpoint.endpoint_id,
        node_covariates=node_covariates,
        direct=direct_bound,
        semantic=semantic_bound,
    )
    analysis_payload = analysis.receipt.to_dict()
    analysis_bytes = _canonical_bytes(analysis_payload)
    if cache_hit and _json_object(analysis_path.read_bytes(), "tree analysis") != analysis_payload:
        raise ValueError("cached tree analysis differs from recomputed analysis")
    analysis_pin = ArtifactPin(
        str(analysis_path),
        _sha256_file(analysis_path) if cache_hit else hashlib.sha256(analysis_bytes).hexdigest(),
    )

    terminal_core = {
        "cohort_key": cohort_key,
        "training_cohort_id": manifest.cohort_id,
        "tree_id": topology.tree_id,
        "endpoint_id": endpoint.endpoint_id,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "root_excluded": True,
        "n_nodes": topology.n_nodes,
        "n_internal": topology.n_internal,
        "n_leaves": topology.n_leaves,
        "n_paired_coordinates": endpoint.n_paired_coordinates,
        "checkpoint_set_sha256": manifest.checkpoint_set_sha256,
        "node_covariates_sha256": getattr(node_covariates, "covariates_sha256"),
        "response_scope_sha256": response_scope_sha,
        "encoder_receipt_sha256": _sha256(encoder_receipt_sha256, "encoder_receipt_sha256"),
        "direct_matrix_artifact": direct.matrix_artifact.to_dict(),
        "direct_artifact_receipt": direct.artifact_receipt.to_dict(),
        "semantic_matrix_artifact": matrix_pin.to_dict(),
        "tree_analysis_artifact": analysis_pin.to_dict(),
        "tree_analysis_sha256": analysis.receipt.analysis_sha256,
        "streaming_unit": "one_tree_one_endpoint",
        "execution_overrides": {},
    }
    terminal_payload = {
        "schema": TREE_TERMINAL_SCHEMA,
        "valid": True,
        "status": "completed",
        **terminal_core,
        "terminal_sha256": canonical_sha256(terminal_core),
    }
    terminal_bytes = _canonical_bytes(terminal_payload)
    if not cache_hit:
        # Compute every artifact before creating the publication directory.  A
        # model/analysis failure therefore leaves no visible or hidden partial
        # result.  Publish the complete three-file unit with one same-filesystem
        # directory rename; the final path never exists in a partial state.
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
        )
        try:
            _publish_exclusive_atomic(staging / matrix_path.name, matrix_bytes)
            _publish_exclusive_atomic(staging / analysis_path.name, analysis_bytes)
            _publish_exclusive_atomic(staging / terminal_path.name, terminal_bytes)
            if os.path.lexists(destination):
                raise FileExistsError(
                    f"semantic tree/endpoint output appeared concurrently: {destination}"
                )
            os.rename(staging, destination)
            staging = None
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
    elif _json_object(terminal_path.read_bytes(), "tree terminal") != terminal_payload:
        raise ValueError("cached tree terminal differs from recomputed terminal")
    terminal_pin = ArtifactPin(str(terminal_path), _sha256_file(terminal_path))
    return TreeEndpointResult(
        cache_hit=cache_hit,
        analysis=analysis,
        matrix_artifact=matrix_pin,
        analysis_artifact=analysis_pin,
        terminal_receipt=terminal_pin,
        terminal_payload=terminal_payload,
    )


def load_direct_cosine_source(
    root: str | Path,
    *,
    topology: TreeTopology,
    manifest: TreeCheckpointManifest,
    node_covariates_sha256: str,
) -> DirectCosineSource:
    """Read and independently revalidate the direct runner's cosine artifact."""

    base = Path(_canonical_root(root, "direct_output_root"))
    tree_id = _component(topology.tree_id, "tree_id")
    matrix_path = _canonical_file(
        base / f"{tree_id}.cosine.all_nodes.matrix.json", "direct cosine matrix"
    )
    receipt_path = _canonical_file(
        base / f"{tree_id}.cosine.all_nodes.receipt.json", "direct cosine receipt"
    )
    matrix_bytes = matrix_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    matrix_payload = _json_object(matrix_bytes, "direct cosine matrix")
    required_matrix = {
        "schema", "tree_id", "metric", "population", "internal_node_treatment",
        "node_covariates_sha256", "topology_sha256", "pair_index_sha256",
        "ordered_node_ids", "matrix", "leaf_sensitivity_indices",
        "leaf_sensitivity_ids", "leaf_sensitivity_matrix",
    }
    if set(matrix_payload) != required_matrix or matrix_payload.get("schema") != DIRECT_VALUES_SCHEMA:
        raise ValueError("direct cosine matrix payload keys/schema drift")
    if (
        matrix_payload["tree_id"] != topology.tree_id
        or matrix_payload["metric"] != "cosine"
        or matrix_payload["population"] != POPULATION
        or matrix_payload["internal_node_treatment"] != INTERNAL_NODE_TREATMENT
        or matrix_payload["node_covariates_sha256"] != node_covariates_sha256
        or matrix_payload["topology_sha256"] != topology.topology_sha256
        or matrix_payload["pair_index_sha256"] != topology.pair_index_sha256
    ):
        raise ValueError("direct cosine matrix scope drift")
    matrix = validate_distance_matrix(
        topology,
        metric="cosine",
        ordered_node_ids=matrix_payload["ordered_node_ids"],
        matrix=matrix_payload["matrix"],
    )
    if (
        matrix_payload["leaf_sensitivity_indices"] != list(matrix.receipt.leaf_sensitivity_indices)
        or matrix_payload["leaf_sensitivity_ids"] != list(matrix.receipt.leaf_sensitivity_ids)
        or matrix_payload["leaf_sensitivity_matrix"] != [list(row) for row in matrix.leaf_sensitivity_matrix]
    ):
        raise ValueError("direct cosine leaf sensitivity is not the derived full-matrix slice")

    receipt = _json_object(receipt_bytes, "direct cosine receipt")
    required_receipt = {
        "schema", "valid", "matrix_receipt", "matrix_receipt_sha256",
        "matrix_artifact", "execution_sha256", "node_covariates_sha256",
        "artifact_receipt_sha256",
    }
    if set(receipt) != required_receipt or receipt.get("schema") != DIRECT_ARTIFACT_RECEIPT_SCHEMA or receipt.get("valid") is not True:
        raise ValueError("direct cosine artifact receipt keys/schema drift")
    matrix_pin = ArtifactPin(str(matrix_path), hashlib.sha256(matrix_bytes).hexdigest())
    matrix_write = receipt.get("matrix_artifact")
    if not isinstance(matrix_write, Mapping) or matrix_write != {
        "path": matrix_pin.path,
        "content_sha256": matrix_pin.sha256,
        "size_bytes": len(matrix_bytes),
        "exclusive_create": True,
    }:
        raise ValueError("direct cosine matrix artifact pin drift")
    receipt_core = {
        key: receipt[key]
        for key in (
            "matrix_receipt", "matrix_receipt_sha256", "matrix_artifact",
            "execution_sha256", "node_covariates_sha256",
        )
    }
    if (
        receipt["matrix_receipt"] != matrix.receipt.to_dict()
        or receipt["matrix_receipt_sha256"] != canonical_sha256(matrix.receipt.to_dict())
        or receipt["node_covariates_sha256"] != node_covariates_sha256
        or receipt["artifact_receipt_sha256"] != canonical_sha256(receipt_core)
    ):
        raise ValueError("direct cosine artifact receipt hash/scope drift")
    return DirectCosineSource(
        matrix=matrix,
        matrix_artifact=matrix_pin,
        artifact_receipt=ArtifactPin(str(receipt_path), hashlib.sha256(receipt_bytes).hexdigest()),
    )


def load_node_endpoint_source(
    inference_plan: InferenceCohortPlan,
    *,
    stage_root: str,
    tree_position: int,
    node_ordinal: int,
    endpoint: SemanticEndpointPlan,
) -> NodeEndpointSource:
    """Validate one terminal response pair and return exactly one endpoint's text."""

    manifest = inference_plan.checkpoint_manifests[tree_position]
    covariates = inference_plan.node_covariates[tree_position]
    binding = manifest.bindings[node_ordinal]
    backend = dict(inference_plan.planned_backend_template)
    if backend.get("checkpoint") != CHECKPOINT_PATH_SENTINEL:
        raise ValueError("inference backend template checkpoint sentinel drift")
    backend["checkpoint"] = binding.checkpoint.path
    request = build_inference_request(
        binding,
        manifest,
        cohort_key=inference_plan.cohort_key,
        registry=inference_plan.registry,
        panels=inference_plan.panels,
        planned_backend_provenance=backend,
        node_covariates=covariates,
    )
    root = Path(_canonical_root(stage_root, "stage_root"))
    output_dir = (
        root / INFERENCE_RELATIVE_ROOT / inference_plan.cohort_key
        / manifest.tree_id / binding.node_id
    )
    responses_path = _canonical_file(output_dir / "responses.jsonl", "responses")
    receipt_path = _canonical_file(output_dir / "terminal_receipt.json", "terminal response receipt")
    response_bytes = responses_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    cached: RuntimeExecutionResult = execute_all_node_inference(
        request,
        prompt_payloads={},
        generate_batch=None,
        observed_backend_provenance=None,
        output_path=str(responses_path),
        receipt_path=str(receipt_path),
        existing_output=response_bytes,
        existing_receipt=receipt_bytes,
    )
    if not cached.cache_hit:
        raise AssertionError("terminal response validation did not produce a cache hit")
    decoded = _jsonl_objects(response_bytes, "responses")
    selected: list[dict[str, Any]] = []
    expected_coordinates = [
        coordinate
        for coordinate in request.coordinates()
        if coordinate.protocol_id == endpoint.protocol_id
        and coordinate.probe_id == endpoint.probe_id
    ]
    for row in decoded:
        response_row = row.get("response_row")
        if not isinstance(response_row, Mapping):
            raise ValueError("decoded response row is missing response_row")
        if (
            response_row.get("protocol_id") == endpoint.protocol_id
            and response_row.get("probe_id") == endpoint.probe_id
        ):
            selected.append(row)
    observed_coordinates = [
        (
            row["response_row"]["protocol_id"],
            row["response_row"]["probe_id"],
            row["response_row"]["prompt_id"],
            row["response_row"]["sample_id"],
        )
        for row in selected
    ]
    expected_rows = [
        (value.protocol_id, value.probe_id, value.prompt_id, value.sample_id)
        for value in expected_coordinates
    ]
    if observed_coordinates != expected_rows or len(selected) != endpoint.n_paired_coordinates:
        raise ValueError("decoded endpoint response coordinates differ from the semantic plan")
    endpoint_hash_rows = [
        {
            "protocol_id": row["response_row"]["protocol_id"],
            "probe_id": row["response_row"]["probe_id"],
            "prompt_id": row["response_row"]["prompt_id"],
            "sample_id": row["response_row"]["sample_id"],
            "response_sha256": row["response_row"]["response_sha256"],
        }
        for row in selected
    ]
    return NodeEndpointSource(
        node_id=binding.node_id,
        node_ordinal=binding.node_ordinal,
        node_role=binding.node_role,
        responses=ArtifactPin(str(responses_path), hashlib.sha256(response_bytes).hexdigest()),
        terminal_receipt=ArtifactPin(str(receipt_path), hashlib.sha256(receipt_bytes).hexdigest()),
        request_sha256=request.request_sha256,
        endpoint_rows_sha256=canonical_sha256(endpoint_hash_rows),
        texts=tuple(row["text"] for row in selected),
    )


def _stream_semantic_matrix(
    source_loader: Callable[[int], NodeEndpointSource],
    *,
    topology: TreeTopology,
    endpoint: SemanticEndpointPlan,
    encoder: TextEncoder,
    temp_parent: Path,
    expected_scope: Sequence[Mapping[str, Any]],
) -> ValidatedDistanceMatrix:
    """Encode node batches to a memmap and reduce coordinate-wise distances."""

    try:
        import numpy as np
    except ImportError:
        # The audited Wright runtime has numpy through sentence-transformers.
        # This binary-spool implementation keeps the contract testable in a
        # minimal Python environment and retains the same bounded-memory rule.
        return _stream_semantic_matrix_stdlib(
            source_loader,
            topology=topology,
            endpoint=endpoint,
            encoder=encoder,
            temp_parent=temp_parent,
            expected_scope=expected_scope,
        )
    temp_parent.mkdir(parents=True, exist_ok=True)
    memmap_path: Path | None = None
    embeddings = None
    dimension: int | None = None
    try:
        for ordinal in range(topology.n_nodes):
            source = source_loader(ordinal)
            _validate_node_source(
                source,
                topology=topology,
                endpoint=endpoint,
                ordinal=ordinal,
            )
            if source.scope_dict() != dict(expected_scope[ordinal]):
                raise ValueError("semantic response source changed between validation and embedding")
            raw = np.asarray(encoder(source.texts))
            if raw.ndim != 2 or raw.shape[0] != endpoint.n_paired_coordinates:
                raise ValueError(
                    "semantic encoder must return exactly one row per endpoint coordinate"
                )
            if raw.shape[1] < 1 or not np.issubdtype(raw.dtype, np.floating):
                raise ValueError("semantic encoder must return real nonempty embeddings")
            values = np.asarray(raw, dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError("semantic encoder returned non-finite embeddings")
            norms = np.linalg.norm(values.astype(np.float64), axis=1)
            if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
                raise ValueError("semantic encoder returned a zero/nonfinite embedding")
            values = values / norms[:, None].astype(np.float32)
            if dimension is None:
                dimension = int(values.shape[1])
                handle = tempfile.NamedTemporaryFile(
                    prefix=".allnode-semantic-embeddings-",
                    suffix=".mmap",
                    dir=temp_parent,
                    delete=False,
                )
                memmap_path = Path(handle.name)
                handle.close()
                embeddings = np.memmap(
                    memmap_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(topology.n_nodes, endpoint.n_paired_coordinates, dimension),
                )
            elif values.shape[1] != dimension:
                raise ValueError("semantic encoder embedding dimension drift across nodes")
            assert embeddings is not None
            embeddings[ordinal, :, :] = values
        assert embeddings is not None and dimension is not None
        embeddings.flush()
        accumulated = np.zeros((topology.n_nodes, topology.n_nodes), dtype=np.float64)
        for coordinate in range(endpoint.n_paired_coordinates):
            values = np.asarray(embeddings[:, coordinate, :], dtype=np.float64)
            similarities = np.clip(values @ values.T, -1.0, 1.0)
            distances = np.clip(1.0 - similarities, 0.0, 2.0)
            np.fill_diagonal(distances, 0.0)
            accumulated += distances
        accumulated /= float(endpoint.n_paired_coordinates)
        # Floating matrix multiplication may differ by the last bit across the
        # two triangles.  Canonicalize each pair to one exact symmetric value.
        accumulated = (accumulated + accumulated.T) / 2.0
        np.fill_diagonal(accumulated, 0.0)
        return validate_distance_matrix(
            topology,
            metric=SEMANTIC_METRIC,
            ordered_node_ids=topology.ordered_node_ids,
            matrix=accumulated.tolist(),
        )
    finally:
        if embeddings is not None:
            del embeddings
        if memmap_path is not None and memmap_path.exists():
            memmap_path.unlink()


def _stream_semantic_matrix_stdlib(
    source_loader: Callable[[int], NodeEndpointSource],
    *,
    topology: TreeTopology,
    endpoint: SemanticEndpointPlan,
    encoder: TextEncoder,
    temp_parent: Path,
    expected_scope: Sequence[Mapping[str, Any]],
) -> ValidatedDistanceMatrix:
    """Numpy-free binary spool with the same one-node/coordinate memory bound."""

    import array

    temp_parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".allnode-semantic-embeddings-",
        suffix=".bin",
        dir=temp_parent,
        delete=False,
    )
    path = Path(handle.name)
    dimension: int | None = None
    try:
        with handle:
            for ordinal in range(topology.n_nodes):
                source = source_loader(ordinal)
                _validate_node_source(
                    source,
                    topology=topology,
                    endpoint=endpoint,
                    ordinal=ordinal,
                )
                if source.scope_dict() != dict(expected_scope[ordinal]):
                    raise ValueError(
                        "semantic response source changed between validation and embedding"
                    )
                raw = encoder(source.texts)
                if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                    raise ValueError("semantic encoder must return a sequence of rows")
                if len(raw) != endpoint.n_paired_coordinates:
                    raise ValueError(
                        "semantic encoder must return exactly one row per endpoint coordinate"
                    )
                for raw_row in raw:
                    if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
                        raise ValueError("semantic encoder row must be a numeric sequence")
                    row = [float(value) for value in raw_row]
                    if not row or any(not math.isfinite(value) for value in row):
                        raise ValueError("semantic encoder returned empty/nonfinite embeddings")
                    if dimension is None:
                        dimension = len(row)
                    elif len(row) != dimension:
                        raise ValueError("semantic encoder embedding dimension drift")
                    norm = math.sqrt(math.fsum(value * value for value in row))
                    if not math.isfinite(norm) or norm <= 0.0:
                        raise ValueError("semantic encoder returned a zero/nonfinite embedding")
                    array.array("d", (value / norm for value in row)).tofile(handle)
        assert dimension is not None
        size = topology.n_nodes
        coordinates = endpoint.n_paired_coordinates
        item_size = array.array("d").itemsize
        matrix = [[0.0] * size for _ in range(size)]
        with path.open("rb") as reader:
            for coordinate in range(coordinates):
                vectors: list[tuple[float, ...]] = []
                for node in range(size):
                    offset = ((node * coordinates + coordinate) * dimension) * item_size
                    reader.seek(offset)
                    values = array.array("d")
                    values.fromfile(reader, dimension)
                    if len(values) != dimension:
                        raise ValueError("temporary semantic embedding spool is truncated")
                    vectors.append(tuple(values))
                for left in range(size):
                    for right in range(left + 1, size):
                        similarity = math.fsum(
                            a * b for a, b in zip(vectors[left], vectors[right])
                        )
                        distance = min(2.0, max(0.0, 1.0 - similarity))
                        matrix[left][right] += distance
                        matrix[right][left] += distance
        for left in range(size):
            for right in range(size):
                matrix[left][right] /= float(coordinates)
        return validate_distance_matrix(
            topology,
            metric=SEMANTIC_METRIC,
            ordered_node_ids=topology.ordered_node_ids,
            matrix=matrix,
        )
    finally:
        if path.exists():
            path.unlink()


def _semantic_matrix_payload(
    matrix: ValidatedDistanceMatrix,
    *,
    endpoint: SemanticEndpointPlan,
    checkpoint_set_sha256: str,
    node_covariates_sha256: object,
    response_scope: Sequence[Mapping[str, Any]],
    response_scope_sha256: str,
    encoder_receipt_sha256: str,
) -> dict[str, Any]:
    core = {
        "endpoint": endpoint.to_dict(),
        "checkpoint_set_sha256": _sha256(checkpoint_set_sha256, "checkpoint_set_sha256"),
        "node_covariates_sha256": _sha256(node_covariates_sha256, "node_covariates_sha256"),
        "encoder": {
            "model": MINILM_MODEL,
            "revision": MINILM_REVISION,
            "runtime": MINILM_RUNTIME,
            "receipt_sha256": _sha256(encoder_receipt_sha256, "encoder_receipt_sha256"),
        },
        "pairing": "exact_protocol_probe_prompt_id_sample_id",
        "aggregation": "mean_paired_response_embedding_cosine_distance",
        "response_scope": [dict(row) for row in response_scope],
        "response_scope_sha256": _sha256(response_scope_sha256, "response_scope_sha256"),
        "matrix_receipt": matrix.receipt.to_dict(),
        "matrix": [list(row) for row in matrix.matrix],
        "leaf_sensitivity_matrix": [list(row) for row in matrix.leaf_sensitivity_matrix],
    }
    return {
        "schema": SEMANTIC_MATRIX_SCHEMA,
        "valid": True,
        **core,
        "artifact_sha256": canonical_sha256(core),
    }


def _read_semantic_matrix_artifact(
    path: Path,
    *,
    topology: TreeTopology,
    endpoint: SemanticEndpointPlan,
    checkpoint_set_sha256: str,
    node_covariates_sha256: object,
    response_scope_sha256: str,
    encoder_receipt_sha256: str,
) -> ValidatedDistanceMatrix:
    payload = _json_object(path.read_bytes(), "semantic matrix")
    if payload.get("schema") != SEMANTIC_MATRIX_SCHEMA or payload.get("valid") is not True:
        raise ValueError("semantic matrix artifact schema/validity drift")
    matrix = validate_distance_matrix(
        topology,
        metric=SEMANTIC_METRIC,
        ordered_node_ids=topology.ordered_node_ids,
        matrix=payload.get("matrix"),
    )
    expected = _semantic_matrix_payload(
        matrix,
        endpoint=endpoint,
        checkpoint_set_sha256=checkpoint_set_sha256,
        node_covariates_sha256=node_covariates_sha256,
        response_scope=payload.get("response_scope", ()),
        response_scope_sha256=response_scope_sha256,
        encoder_receipt_sha256=encoder_receipt_sha256,
    )
    if payload != expected:
        raise ValueError("cached semantic matrix differs from its recomputed contract")
    return matrix


def _validate_node_source(
    source: NodeEndpointSource,
    *,
    topology: TreeTopology,
    endpoint: SemanticEndpointPlan,
    ordinal: int,
) -> None:
    if not isinstance(source, NodeEndpointSource):
        raise ValueError("response_loader must return NodeEndpointSource")
    node = topology.nodes[ordinal]
    if (source.node_id, source.node_ordinal, source.node_role) != (
        node.node_id, node.node_ordinal, node.node_role
    ):
        raise ValueError("semantic response source node order/identity drift")
    _pin(source.responses, "responses")
    _pin(source.terminal_receipt, "terminal_receipt")
    _sha256(source.request_sha256, "request_sha256")
    _sha256(source.endpoint_rows_sha256, "endpoint_rows_sha256")
    if len(source.texts) != endpoint.n_paired_coordinates or any(
        not isinstance(text, str) for text in source.texts
    ):
        raise ValueError("semantic response source endpoint text grid drift")


def _endpoint_from_panel(panel: PromptPanel) -> SemanticEndpointPlan:
    coordinates = [
        {
            "protocol_id": panel.protocol_id,
            "probe_id": panel.probe_id,
            "prompt_id": prompt_id,
            "sample_id": sample_id,
        }
        for prompt_id in panel.prompt_ids
        for sample_id in range(panel.samples_per_prompt)
    ]
    return _validate_endpoint(SemanticEndpointPlan(
        endpoint_id=panel.probe_id,
        protocol_id=panel.protocol_id,
        probe_id=panel.probe_id,
        prompt_ids=panel.prompt_ids,
        samples_per_prompt=panel.samples_per_prompt,
        n_paired_coordinates=len(coordinates),
        coordinates_sha256=canonical_sha256(coordinates),
    ))


def _validate_endpoint(value: SemanticEndpointPlan) -> SemanticEndpointPlan:
    if not isinstance(value, SemanticEndpointPlan):
        raise ValueError("endpoint must be a SemanticEndpointPlan")
    _component(value.endpoint_id, "endpoint_id")
    _component(value.protocol_id, "protocol_id")
    _component(value.probe_id, "probe_id")
    if value.endpoint_id != value.probe_id:
        raise ValueError("semantic endpoint ID must equal probe ID")
    if (
        not value.prompt_ids
        or len(value.prompt_ids) != len(set(value.prompt_ids))
        or any(not isinstance(prompt, str) or not prompt for prompt in value.prompt_ids)
        or isinstance(value.samples_per_prompt, bool)
        or not isinstance(value.samples_per_prompt, int)
        or value.samples_per_prompt < 1
    ):
        raise ValueError("semantic endpoint prompt/draw contract drift")
    coordinates = [
        {
            "protocol_id": value.protocol_id,
            "probe_id": value.probe_id,
            "prompt_id": prompt_id,
            "sample_id": sample_id,
        }
        for prompt_id in value.prompt_ids
        for sample_id in range(value.samples_per_prompt)
    ]
    if (
        value.n_paired_coordinates != len(coordinates)
        or value.coordinates_sha256 != canonical_sha256(coordinates)
    ):
        raise ValueError("semantic endpoint coordinate count/hash drift")
    return value


def _semantic_plan_core(
    *,
    cohort_key: str,
    training_cohort_id: str,
    stage_root: str,
    deployment_root: str,
    inference_plan: ArtifactPin,
    direct_job_config: ArtifactPin,
    direct_output_root: str,
    inference_output_root: str,
    output_root: str,
    encoder_receipt: ArtifactPin,
    encoder_snapshot: ArtifactPin,
    encoder_runtime: str,
    endpoints: Sequence[SemanticEndpointPlan],
    tree_scope: Sequence[Mapping[str, Any]],
    topology_scope_sha256: str,
    checkpoint_scope_sha256: str,
    node_covariate_scope_sha256: str,
    endpoint_scope_sha256: str,
    analysis_contract: Mapping[str, Any],
    analysis_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "valid": True,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "exclude_root": True,
        "cohort_key": cohort_key,
        "training_cohort_id": training_cohort_id,
        "stage_root": stage_root,
        "deployment_root": deployment_root,
        "inference_plan": inference_plan.to_dict(),
        "direct_job_config": direct_job_config.to_dict(),
        "direct_output_root": direct_output_root,
        "inference_output_root": inference_output_root,
        "output_root": output_root,
        "encoder_receipt": encoder_receipt.to_dict(),
        "encoder_snapshot": encoder_snapshot.to_dict(),
        "encoder_model": MINILM_MODEL,
        "encoder_revision": MINILM_REVISION,
        "encoder_runtime": encoder_runtime,
        "n_trees": PRODUCTION_N_TREES,
        "n_nodes": PRODUCTION_N_NODES,
        "n_internal": PRODUCTION_N_INTERNAL,
        "n_leaves": PRODUCTION_N_LEAVES,
        "tree_ids": list(PRODUCTION_TREE_IDS),
        "endpoints": [row.to_dict() for row in endpoints],
        "tree_scope": [dict(row) for row in tree_scope],
        "topology_scope_sha256": topology_scope_sha256,
        "checkpoint_scope_sha256": checkpoint_scope_sha256,
        "node_covariate_scope_sha256": node_covariate_scope_sha256,
        "endpoint_scope_sha256": endpoint_scope_sha256,
        "analysis_contract": dict(analysis_contract),
        "analysis_contract_sha256": analysis_contract_sha256,
        "streaming": {
            "unit": "one_tree_one_endpoint",
            "text_retention": "one_node_one_endpoint",
            "embedding_store": "temporary_memmap",
            "distance_reduction": "one_response_coordinate_at_a_time",
        },
        "leaf_sensitivity": "derived_from_validated_full_matrix",
        "execution_overrides": {},
    }


def _semantic_plan_core_keys() -> tuple[str, ...]:
    # Use a harmless dummy call only to keep the exact reader key set coupled
    # to the writer.  Values are not validated here.
    return (
        "schema", "valid", "population", "internal_node_treatment", "exclude_root",
        "cohort_key", "training_cohort_id", "stage_root", "deployment_root",
        "inference_plan", "direct_job_config", "direct_output_root",
        "inference_output_root", "output_root",
        "encoder_receipt", "encoder_snapshot", "encoder_model", "encoder_revision",
        "encoder_runtime", "n_trees", "n_nodes", "n_internal", "n_leaves", "tree_ids",
        "endpoints", "tree_scope", "topology_scope_sha256", "checkpoint_scope_sha256",
        "node_covariate_scope_sha256", "endpoint_scope_sha256", "analysis_contract",
        "analysis_contract_sha256", "streaming", "leaf_sensitivity", "execution_overrides",
    )


def topology_from_manifest(manifest: TreeCheckpointManifest) -> TreeTopology:
    return build_tree_topology(
        manifest.tree_id,
        [
            NodeSpec(
                binding.node_ordinal,
                binding.array_index,
                binding.node_id,
                binding.parent_id,
                binding.depth,
                binding.node_role == "leaf",
            )
            for binding in manifest.bindings
        ],
        root_id=manifest.root_id,
    )


def validate_encoder_receipt(plan: SemanticCohortPlan) -> Path:
    """Revalidate the pinned receipt, snapshot bytes, and installed runtime."""

    path = _canonical_file(plan.encoder_receipt.path, "encoder receipt")
    if _sha256_file(path) != plan.encoder_receipt.sha256:
        raise ValueError("MiniLM encoder receipt file hash drift")
    payload = _json_object(path.read_bytes(), "encoder receipt")
    required = {"schema", "valid", "mode", "model", "revision", "snapshot", "runtime", "input_config"}
    if set(payload) != required:
        raise ValueError("MiniLM encoder receipt keys drift")
    if (
        payload["schema"] != ENCODER_RECEIPT_SCHEMA
        or payload["valid"] is not True
        or payload["mode"] != ENCODER_MODE
        or payload["model"] != MINILM_MODEL
        or payload["revision"] != MINILM_REVISION
        or payload["runtime"] != {"package": MINILM_PACKAGE, "version": MINILM_VERSION}
        or payload["snapshot"] != plan.encoder_snapshot.to_dict()
    ):
        raise ValueError("MiniLM encoder receipt contract drift")
    snapshot = Path(plan.encoder_snapshot.path)
    require_encoder_snapshot_sha256(snapshot, plan.encoder_snapshot.sha256)
    if importlib.metadata.version(MINILM_PACKAGE) != MINILM_VERSION:
        raise ValueError("installed sentence-transformers runtime drift")
    return snapshot


def load_pinned_minilm_encoder(snapshot: Path, *, batch_size: int = 128) -> TextEncoder:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sentence-transformers is required for semantic execution") from exc
    model = SentenceTransformer(str(snapshot), local_files_only=True)

    def encode(texts: Sequence[str]) -> object:
        return model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    return encode


def _pin(value: ArtifactPin | Mapping[str, Any] | object, label: str) -> ArtifactPin:
    if isinstance(value, ArtifactPin):
        pin = value
    elif isinstance(value, Mapping) and set(value) == {"path", "sha256"}:
        pin = ArtifactPin(str(value["path"]), str(value["sha256"]))
    else:
        raise ValueError(f"{label} must be an ArtifactPin")
    if not os.path.isabs(pin.path) or os.path.normpath(pin.path) != pin.path:
        raise ValueError(f"{label}.path must be absolute and normalized")
    _sha256(pin.sha256, f"{label}.sha256")
    return pin


def _canonical_root(value: str | Path, label: str, *, require_exists: bool = True) -> str:
    raw = Path(value)
    absolute = Path(os.path.abspath(str(raw)))
    if require_exists:
        resolved = raw.resolve(strict=True)
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValueError(f"{label} must be a real directory")
    else:
        resolved = raw.resolve(strict=False)
    if absolute != resolved or str(resolved) == "/":
        raise ValueError(f"{label} must be a canonical bounded absolute path")
    return str(resolved)


def _canonical_file(value: str | Path, label: str) -> Path:
    raw = Path(value)
    absolute = Path(os.path.abspath(str(raw)))
    resolved = raw.resolve(strict=True)
    if absolute != resolved or not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a canonical regular file without symlinks")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_exclusive_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("semantic output path must not contain a terminal symlink")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"semantic output already exists: {path}")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not one UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _jsonl_objects(payload: bytes, label: str) -> list[dict[str, Any]]:
    return decode_lf_json_objects(payload, label)


def _component(value: object, label: str) -> str:
    if not isinstance(value, str) or _COMPONENT.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{label} is not a safe component")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _validate_launcher_environment(
    config: SemanticJobConfig,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one exact all-trained-node semantic cohort"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attest_runtime_environment("before-runner")
    try:
        config, _, _, config_path, config_file_sha = load_semantic_job_config(args.config)
        _validate_launcher_environment(
            config,
            config_path=config_path,
            config_file_sha256=config_file_sha,
        )
        result = execute_filesystem_semantic(config_path)
    finally:
        attest_runtime_environment("after-runner")
    print(
        json.dumps(
            result.status_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PLAN_SCHEMA",
    "JOB_CONFIG_SCHEMA",
    "SEMANTIC_MATRIX_SCHEMA",
    "TREE_TERMINAL_SCHEMA",
    "SemanticEndpointPlan",
    "SemanticCohortPlan",
    "SemanticJobConfig",
    "NodeEndpointSource",
    "DirectCosineSource",
    "TreeEndpointResult",
    "FilesystemSemanticResult",
    "build_semantic_cohort_plan",
    "read_semantic_cohort_plan",
    "load_semantic_cohort_plan",
    "build_semantic_job_config",
    "read_semantic_job_config",
    "load_semantic_job_config",
    "execute_filesystem_semantic",
    "validate_direct_cohort_terminal",
    "execute_tree_endpoint_streaming",
    "load_direct_cosine_source",
    "load_node_endpoint_source",
    "topology_from_manifest",
    "validate_encoder_receipt",
    "load_pinned_minilm_encoder",
    "main",
]
