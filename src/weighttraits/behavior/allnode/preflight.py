"""Fail-closed production assembler for the isolated ``allnode_v1`` branch.

The module has two deliberately separate phases.  :func:`plan_preflight`
performs every expensive read-only check and returns the exact bytes that may
be published.  :func:`materialize_preflight` revalidates that plan and creates
each file exclusively, with the canonical dry-run audit written last as the
terminal marker.  Neither phase knows how to query or mutate Slurm.

Checkpoint directories are attested lazily through the versioned sampled
identity scheme.  Normal source files are still hashed in full.  Consequently
the frozen-stage loader sees a normal digest callback while multi-gigabyte
checkpoints are never reread in full merely to assemble the run contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .audit import (
    EXECUTION_CONTRACT_SCHEMA,
    INFERENCE_PLAN_SCHEMA,
    MATERIALIZATION_RECEIPT_SHA256,
    MATRIX_PLAN_SCHEMA,
    MINILM_CONFIG_SHA256,
    MINILM_MODEL,
    MINILM_RECEIPT_SHA256,
    MINILM_REVISION,
    MINILM_RUNTIME,
    MINILM_SNAPSHOT_SHA256,
    PROTOCOLS,
    PROMPT_SHA256,
    REGISTRY_SHA256,
    RESPONSES_PER_NODE,
    STAGE_MANIFEST_SHA256,
    TEST_SCHEMA,
    TREE_IDS,
    build_dry_run_audit_bundle,
    expected_analysis_contract,
    semantic_endpoint_rows,
    serialize_audit_core,
    validate_dry_run_audit_bundle,
)
from .checkpoint_identity import (
    SCHEME as CHECKPOINT_IDENTITY_SCHEME,
    CheckpointIdentity,
    build_checkpoint_identity,
    build_identity_index,
    read_identity_index,
)
from .checkpoints import (
    ArtifactPin,
    INTERNAL_NODE_TREATMENT,
    POPULATION,
    PRODUCTION_N_INTERNAL,
    PRODUCTION_N_LEAVES,
    PRODUCTION_N_NODES,
    PRODUCTION_N_TREES,
    validate_production_checkpoint_manifests,
)
from .direct_compatibility import LeafCompatibilitySpec, read_leaf_compatibility_spec
from .direct_runner import (
    PLAN_SCHEMA as DIRECT_PLAN_SCHEMA,
    DirectCohortPlan,
    build_direct_cohort_plan,
    build_direct_job_config,
    read_direct_cohort_plan,
    read_direct_job_config,
)
from .gate import COHORTS, Pin, canonical_sha256
from .inference import (
    BACKEND_PROVENANCE_KEYS,
    FROZEN_COHORT_ENDPOINTS,
    FROZEN_COHORT_MODEL_TASK,
    PromptPanel,
)
from .inference_runner import (
    CHECKPOINT_PATH_SENTINEL,
    PLAN_SCHEMA as INFERENCE_COHORT_PLAN_SCHEMA,
    InferenceCohortPlan,
    build_inference_cohort_plan,
    build_inference_job_config,
    read_inference_cohort_plan,
    read_inference_job_config,
)
from .encoder_snapshot import require_encoder_snapshot_sha256
from .loader import LoadedCohort, load_cohort_from_frozen_stage, sha256_file
from .manifest import canonical_sha256 as manifest_sha256
from .production import ProductionResolutionReceipt, validate_production_resolutions
from .provenance import (
    DOWNSTREAM_CODE_MANIFEST_BASENAME,
    DOWNSTREAM_CODE_MANIFEST_SHA256,
    production_runtime,
    verify_downstream_code_manifest,
)
from .semantic_runner import (
    PLAN_SCHEMA as SEMANTIC_PLAN_SCHEMA,
    SemanticCohortPlan,
    build_semantic_cohort_plan,
    build_semantic_job_config,
    read_semantic_cohort_plan,
    read_semantic_job_config,
)
from .source_specs import (
    DOWNSTREAM_STAGE,
    FROZEN_STAGE,
    SOURCE_PATH_AUTHORITY,
    production_source_specs,
)
from .symlink_inventory import (
    SourcePathAuthority,
    load_source_path_authority,
)


CODE_MANIFEST_SCHEMA = "weighttraits.all_node_code_manifest.v1"
PREFLIGHT_PLAN_SCHEMA = "weighttraits.all_node_preflight_publish_plan.v1"
SOURCE_RESOLUTION_SCHEMA = "weighttraits.all_node_source_resolution.v1"
AUDIT_BASENAME = "ALLNODE_V1_DRY_RUN_AUDIT.json"
IDENTITY_BASENAME = "ALLNODE_V1_CHECKPOINT_IDENTITIES.json"
CHECKPOINT_BASENAME = "ALLNODE_V1_CHECKPOINTS.json"
EXECUTION_BASENAME = "ALLNODE_V1_EXECUTION_CONTRACT.json"
SOURCE_BASENAME = "ALLNODE_V1_SOURCE_RESOLUTION.json"
CODE_BASENAME = "ALLNODE_V1_CODE_MANIFEST.json"
TEST_BASENAME = "ALLNODE_V1_TEST_ATTESTATION.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TEST_COUNT = re.compile(r"Ran ([0-9]+) tests? in ")


@dataclass(frozen=True)
class PlannedFile:
    """One exact exclusive publication target."""

    path: str
    payload: bytes
    terminal: bool = False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def pin(self) -> ArtifactPin:
        return ArtifactPin(self.path, self.sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": len(self.payload),
            "terminal": self.terminal,
            "exclusive_create": True,
        }


@dataclass(frozen=True)
class PreflightPlan:
    """Fully validated bytes; materialization itself remains a separate step."""

    stage_root: str
    frozen_stage: str
    source_authority: SourcePathAuthority
    files: tuple[PlannedFile, ...]
    audit_bundle: object
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": PREFLIGHT_PLAN_SCHEMA,
            "valid": True,
            "read_only_source_validation": True,
            "scheduler_actions": 0,
            "stage_root": self.stage_root,
            "frozen_stage": self.frozen_stage,
            "source_path_authority": self.source_authority.to_dict(),
            "n_files": len(self.files),
            "files": [value.to_dict() for value in self.files],
            "terminal_path": self.files[-1].path,
        }
        return {**core, "plan_sha256": self.plan_sha256}


class LazyCheckpointDigester:
    """Full-hash files and sample-attest frozen checkpoint directories lazily."""

    def __init__(
        self,
        *,
        frozen_stage_root: str,
        stage_manifest_sha256: str,
        source_authority: SourcePathAuthority,
    ) -> None:
        self.frozen_stage_root = _root(frozen_stage_root, "frozen_stage_root")
        self.stage_manifest_sha256 = _sha(stage_manifest_sha256, "stage_manifest_sha256")
        if not isinstance(source_authority, SourcePathAuthority):
            raise ValueError("checkpoint digester requires a source-path authority")
        if source_authority.stage_root != self.frozen_stage_root:
            raise ValueError("checkpoint digester source authority root drift")
        if source_authority.stage_manifest_sha256 != self.stage_manifest_sha256:
            raise ValueError("checkpoint digester stage manifest authority drift")
        self.source_authority = source_authority
        self._identities: dict[str, CheckpointIdentity] = {}

    @property
    def identities(self) -> Mapping[str, CheckpointIdentity]:
        return dict(self._identities)

    def __call__(self, path: str | Path) -> str:
        source = Path(path)
        if source.is_symlink():
            raise ValueError(f"source artifact must not be a symlink: {source}")
        if source.is_file():
            return sha256_file(source)
        if not source.is_dir():
            raise FileNotFoundError(f"required source artifact is missing: {source}")
        physical = str(source)
        logical = self.source_authority.logical_for_physical(physical)
        value = self._identities.get(physical)
        if value is None:
            value = build_checkpoint_identity(
                physical,
                frozen_stage_root=self.frozen_stage_root,
                stage_manifest_sha256=self.stage_manifest_sha256,
                source_authority=self.source_authority,
                logical_path=logical,
            )
            self._identities[physical] = value
        return value.identity_sha256


def checkpoint_production_payload(
    cohorts: Mapping[str, Sequence[object]],
) -> dict[str, Any]:
    """Build the exact audit-compatible 4x50/641 checkpoint receipt."""

    manifests = {
        cohort: tuple(getattr(tree, "checkpoint_manifest") for tree in cohorts[cohort])
        for cohort in COHORTS
    }
    return validate_production_checkpoint_manifests(
        manifests,
        expected_cohort_ids=COHORTS,
        expected_tree_ids=TREE_IDS,
    )


def matrix_plan_payload(
    cohorts: Mapping[str, Sequence[object]],
    *,
    stage_root: str,
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the compact all-node matrix/order plan from validated topologies."""

    stage = _absolute(stage_root, "stage_root")
    reference = tuple(cohorts[COHORTS[0]])
    rows: list[dict[str, Any]] = []
    for tree in reference:
        topology = tree.topology
        rows.append(
            {
                "tree_id": topology.tree_id,
                "n_nodes": topology.n_nodes,
                "n_internal": topology.n_internal,
                "n_leaves": topology.n_leaves,
                "topology_sha256": topology.topology_sha256,
                "nodes_sha256": topology.nodes_sha256,
                "leaf_subset_sha256": topology.leaf_subset_sha256,
                "pair_index_sha256": topology.pair_index_sha256,
                "ordered_node_ids_sha256": manifest_sha256(
                    list(topology.ordered_node_ids)
                ),
            }
        )
    projection = [
        {
            key: row[key]
            for key in (
                "tree_id",
                "topology_sha256",
                "nodes_sha256",
                "leaf_subset_sha256",
                "pair_index_sha256",
            )
        }
        for row in rows
    ]
    topology_sha = canonical_sha256(projection)
    order_sha = canonical_sha256(rows)
    if topology_sha != checkpoint_payload.get("topology_scope_sha256"):
        raise ValueError("matrix topology projection differs from checkpoint production")
    cohort_rows: dict[str, Any] = {}
    for cohort in COHORTS:
        output = f"{stage}/outputs/allnode_v1/direct/{cohort}"
        core = {
            "cohort_id": cohort,
            "direct_job_logical_id": f"direct_{cohort}",
            "output_root": output,
            "tree_ids": list(TREE_IDS),
            "topology_scope_sha256": topology_sha,
            "matrix_order_scope_sha256": order_sha,
        }
        cohort_rows[cohort] = {
            "direct_job_logical_id": f"direct_{cohort}",
            "output_root": output,
            "tree_ids": list(TREE_IDS),
            "plan_sha256": canonical_sha256(core),
        }
    return {
        "schema": MATRIX_PLAN_SCHEMA,
        "valid": True,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "metric": "cosine",
        "cohort_ids": list(COHORTS),
        "n_trees": PRODUCTION_N_TREES,
        "n_nodes": PRODUCTION_N_NODES,
        "n_internal": PRODUCTION_N_INTERNAL,
        "n_leaves": PRODUCTION_N_LEAVES,
        "topology_scope": rows,
        "topology_scope_sha256": topology_sha,
        "matrix_order_scope_sha256": order_sha,
        "cohorts": cohort_rows,
    }


def inference_plan_payload() -> dict[str, Any]:
    """Build the compact exact 4x641 inference-count contract."""

    return {
        "schema": INFERENCE_PLAN_SCHEMA,
        "valid": True,
        "cohort_ids": list(COHORTS),
        "n_nodes_per_cohort": PRODUCTION_N_NODES,
        "cohorts": {
            cohort: {
                "array": {"start": 0, "end": 640, "max_concurrent": 4},
                "n_tasks": PRODUCTION_N_NODES,
                "protocols": list(PROTOCOLS[cohort]),
                "responses_per_node": RESPONSES_PER_NODE[cohort],
                "expected_response_rows": PRODUCTION_N_NODES
                * RESPONSES_PER_NODE[cohort],
                "device": "cuda",
                "torch_dtype": "auto",
                "execution_overrides": {},
            }
            for cohort in COHORTS
        },
    }


def build_prompt_panels(
    *,
    cohort_key: str,
    stage_root: str,
) -> tuple[tuple[PromptPanel, ...], dict[str, Pin], dict[str, Pin]]:
    """Read exact immutable prompt JSONL and construct endpoint panels."""

    if cohort_key not in FROZEN_COHORT_ENDPOINTS:
        raise ValueError(f"unknown cohort {cohort_key!r}")
    stage = Path(_root(stage_root, "stage_root"))
    model_task = FROZEN_COHORT_MODEL_TASK[cohort_key]
    prompt_root = stage / "outputs" / "prompts" / model_task
    receipt_path = prompt_root / "prompt_materialization_receipt.json"
    receipt_sha = sha256_file(receipt_path)
    if receipt_sha != MATERIALIZATION_RECEIPT_SHA256[model_task]:
        raise ValueError("prompt materialization receipt hash drift")
    receipt = _json_file(receipt_path, "prompt materialization receipt")
    if receipt.get("valid") is not True or receipt.get("model_task") != model_task:
        raise ValueError("prompt materialization receipt scope drift")
    expected_registry = {
        "path": str(
            stage / "examples" / "behavior" / "corrected_behavior_protocols_v1.json"
        ),
        "sha256": REGISTRY_SHA256,
    }
    if receipt.get("registry") != expected_registry:
        raise ValueError("prompt materialization registry pin drift")
    receipt_pin = ArtifactPin(str(receipt_path), receipt_sha)
    output_rows = receipt.get("outputs")
    if not isinstance(output_rows, Mapping):
        raise ValueError("prompt materialization receipt outputs must be a mapping")
    by_protocol: dict[str, list[object]] = {}
    prompt_pins: dict[str, Pin] = {}
    receipt_pins: dict[str, Pin] = {}
    for protocol in PROTOCOLS[cohort_key]:
        prompt_path = prompt_root / f"{protocol}.{model_task}.prompts.jsonl"
        observed_sha = sha256_file(prompt_path)
        if observed_sha != PROMPT_SHA256[cohort_key][protocol]:
            raise ValueError(f"prompt artifact hash drift for {cohort_key}/{protocol}")
        raw_output = output_rows.get(protocol)
        if not isinstance(raw_output, Mapping):
            raise ValueError(f"prompt receipt lacks output for {protocol}")
        if (
            raw_output.get("path") != str(prompt_path)
            or raw_output.get("sha256") != observed_sha
        ):
            raise ValueError(f"prompt receipt output pin drift for {protocol}")
        rows = _load_rendered_prompts(prompt_path, stage_root=stage)
        if raw_output.get("n_prompts") != len(rows):
            raise ValueError(f"prompt receipt row count drift for {protocol}")
        by_protocol[protocol] = rows
        prompt_pins[protocol] = Pin(str(prompt_path), observed_sha)
        receipt_pins[protocol] = Pin(str(receipt_path), receipt_sha)

    panels: list[PromptPanel] = []
    for shape in FROZEN_COHORT_ENDPOINTS[cohort_key]:
        rows = [
            row for row in by_protocol[shape.protocol_id]
            if getattr(row, "probe_id", None) == shape.probe_id
        ]
        prompt_ids = tuple(
            _text(getattr(row, "prompt_id", None), "prompt_id") for row in rows
        )
        if len(prompt_ids) != shape.n_prompts or len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError(
                f"prompt panel cardinality/uniqueness drift for {shape.protocol_id}/{shape.probe_id}"
            )
        panels.append(
            PromptPanel(
                protocol_id=shape.protocol_id,
                probe_id=shape.probe_id,
                prompt_ids=prompt_ids,
                samples_per_prompt=shape.samples_per_prompt,
                prompt_artifact=ArtifactPin(
                    prompt_pins[shape.protocol_id].path,
                    prompt_pins[shape.protocol_id].sha256,
                ),
                materialization_receipt=receipt_pin,
            )
        )
    return tuple(panels), prompt_pins, receipt_pins


def backend_template(*, model_task: str, binding: object) -> dict[str, Any]:
    """Freeze the exact installed offline HuggingFace runtime template."""

    if model_task not in {"causal_lm", "seq2seq"}:
        raise ValueError("model_task must be causal_lm or seq2seq")
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    result = {
        "backend": "huggingface",
        "device": "cuda",
        "torch_dtype": "auto",
        "local_files_only": True,
        "tokenizer_model_id": binding.base_model_id,
        "tokenizer_revision": binding.base_model_revision,
        "checkpoint": CHECKPOINT_PATH_SENTINEL,
        "transformers_version": _text(
            getattr(transformers, "__version__", None), "transformers.__version__"
        ),
        "torch_version": _text(getattr(torch, "__version__", None), "torch.__version__"),
        "decode_skip_special_tokens": True,
        "decode_clean_up_tokenization_spaces": False,
        "generation_api": "ordered_probe_draw_batches",
        "generation_seed_api": "transformers.set_seed_once_per_probe_draw",
        "tokenizer_padding_side": "left" if model_task == "causal_lm" else "right",
        "tokenizer_truncation": True,
        "tokenizer_max_length": 512,
        "generation_pad_token_id": "tokenizer.pad_token_id",
    }
    if set(result) != BACKEND_PROVENANCE_KEYS:
        raise AssertionError("backend provenance key set corruption")
    return result


def code_manifest_payload(deployment_root: str) -> dict[str, Any]:
    """Hash the exact Python, Slurm-wrapper, and test source set."""

    root = Path(_root(deployment_root, "deployment_root"))
    candidates: list[Path] = []
    for relative, pattern in (
        ("weighttraits_allnode", "*.py"),
        ("scripts/slurm", "*.sbatch"),
        ("tests", "test_*.py"),
    ):
        folder = root / relative
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"code manifest source directory missing: {folder}")
        candidates.extend(folder.glob(pattern))
    files = sorted(candidates, key=lambda value: value.relative_to(root).as_posix())
    if not files or any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("code manifest contains a missing, symlink, or non-file entry")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    core = {
        "schema": CODE_MANIFEST_SCHEMA,
        "valid": True,
        "deployment_root": str(root),
        "n_files": len(rows),
        "files": rows,
        "files_sha256": canonical_sha256(rows),
    }
    return {**core, "code_manifest_sha256": canonical_sha256(core)}


def run_test_attestation(
    *, deployment_root: str, code_manifest_sha256: str
) -> dict[str, Any]:
    """Run the exact audited unittest command and parse its terminal count."""

    root = _root(deployment_root, "deployment_root")
    runtime = production_runtime()
    command = [runtime["interpreter"], "-m", "unittest", "discover", "-s", "tests", "-v"]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    transcript = completed.stdout + "\n" + completed.stderr
    matches = _TEST_COUNT.findall(transcript)
    if (
        completed.returncode != 0
        or len(matches) != 1
        or "\nOK" not in completed.stderr
        or "skipped=" in completed.stderr
    ):
        tail = "\n".join(transcript.splitlines()[-30:])
        raise RuntimeError(f"allnode production tests did not pass exactly:\n{tail}")
    tests_run = int(matches[0])
    if tests_run <= 0:
        raise RuntimeError("allnode production suite ran no tests")
    return {
        "schema": TEST_SCHEMA,
        "valid": True,
        "runner": "unittest",
        "command": command,
        **runtime,
        "tests_run": tests_run,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "code_manifest_sha256": _sha(code_manifest_sha256, "code_manifest_sha256"),
    }


def execution_contract_payload(
    *,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_identity_index: Pin,
    source_scope_sha256: str,
    source_resolution: Pin,
    allnode_code_manifest: Pin,
    downstream_code_manifest: Pin,
    runtime: Mapping[str, str],
    direct_plans: Mapping[str, DirectCohortPlan],
    inference_plans: Mapping[str, InferenceCohortPlan],
    semantic_plans: Mapping[str, SemanticCohortPlan],
    runtime_plan_pins: Mapping[str, Pin],
    job_config_pins: Mapping[str, Pin],
) -> dict[str, Any]:
    """Bind parsed critical fields from all twelve exact runtime plans."""

    jobs: dict[str, Any] = {}
    for logical_id in sorted(job_config_pins):
        prefix, cohort = logical_id.split("_", 1)
        plan = {
            "direct": direct_plans,
            "infer": inference_plans,
            "semantic": semantic_plans,
        }[prefix][cohort]
        jobs[logical_id] = {
            "kind": {
                "direct": "direct_allnode_cpu",
                "infer": "inference_allnode_gpu",
                "semantic": "semantic_allnode_gpu",
            }[prefix],
            "cohort_key": cohort,
            "config": job_config_pins[logical_id].to_dict(),
            "runtime_plan": runtime_plan_pins[logical_id].to_dict(),
            "runtime_plan_schema": {
                "direct": DIRECT_PLAN_SCHEMA,
                "infer": INFERENCE_COHORT_PLAN_SCHEMA,
                "semantic": SEMANTIC_PLAN_SCHEMA,
            }[prefix],
            "output_root": {
                "direct": direct_plans[cohort].output_dir,
                "infer": f"{_stage_from_plan_pin(runtime_plan_pins[logical_id])}/outputs/allnode_v1/inference/{cohort}",
                "semantic": semantic_plans[cohort].output_root,
            }[prefix],
            "n_trees": PRODUCTION_N_TREES,
            "n_nodes": PRODUCTION_N_NODES,
            "n_internal": PRODUCTION_N_INTERNAL,
            "n_leaves": PRODUCTION_N_LEAVES,
            "topology_scope_sha256": checkpoint_payload["topology_scope_sha256"],
            "checkpoint_scope_sha256": checkpoint_payload["cohorts"][cohort][
                "manifest_scope_sha256"
            ],
            "node_covariate_scope_sha256": _node_covariate_scope(
                direct_plans[cohort]
            ),
            "metrics": ["l2", "cosine", "correlation"]
            if prefix == "direct"
            else [],
            "expected_response_rows": PRODUCTION_N_NODES
            * RESPONSES_PER_NODE[cohort]
            if prefix == "infer"
            else None,
            "semantic_endpoints": semantic_endpoint_rows(cohort)
            if prefix == "semantic"
            else [],
            "execution_overrides": {},
        }
        del plan
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "valid": True,
        "population": POPULATION,
        "sensitivity_population": "trained_leaves",
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "exclude_untrained_root": True,
        "cohort_ids": list(COHORTS),
        "tree_ids": list(TREE_IDS),
        "n_nodes_per_cohort": PRODUCTION_N_NODES,
        "n_internal_per_cohort": PRODUCTION_N_INTERNAL,
        "n_leaves_per_cohort": PRODUCTION_N_LEAVES,
        "checkpoint_identity": {
            "scheme": CHECKPOINT_IDENTITY_SCHEME,
            "index": checkpoint_identity_index.to_dict(),
            "n_checkpoints": len(COHORTS) * PRODUCTION_N_NODES,
            "reattest_at_runtime": True,
        },
        "topology_scope_sha256": checkpoint_payload["topology_scope_sha256"],
        "source_scope_sha256": _sha(source_scope_sha256, "source_scope_sha256"),
        "runtime_provenance": {
            "production_interpreter": runtime["interpreter"],
            "python_version": runtime["python_version"],
            "python_implementation": runtime["python_implementation"],
            "source_resolution": source_resolution.to_dict(),
            "allnode_code_manifest": allnode_code_manifest.to_dict(),
            "downstream_code_manifest": downstream_code_manifest.to_dict(),
            "reattest_before_after_each_job": True,
        },
        "analysis": expected_analysis_contract(),
        "jobs": jobs,
        "execution_overrides": {},
        "no_automatic_recovery": True,
    }


def plan_preflight(
    *,
    compatibility: Mapping[str, object],
    stage_root: str = DOWNSTREAM_STAGE,
    frozen_stage: str = FROZEN_STAGE,
    snapshot_root: str | None = None,
    extra_planned_files: Sequence[object] = (),
    run_tests: bool = True,
) -> PreflightPlan:
    """Assemble and validate every production artifact without publishing it."""

    if stage_root != DOWNSTREAM_STAGE or frozen_stage != FROZEN_STAGE:
        raise ValueError("production preflight roots differ from immutable source specs")
    stage = _root(stage_root, "stage_root")
    frozen = _root(frozen_stage, "frozen_stage")
    deployment = f"{stage}/deployments/allnode_v1"
    snapshot = _root(snapshot_root or f"{stage}/src", "snapshot_root")
    if not isinstance(compatibility, Mapping) or set(compatibility) != set(COHORTS):
        raise ValueError("leaf compatibility plans must cover the exact four cohorts")

    source_authority = load_source_path_authority(SOURCE_PATH_AUTHORITY)
    if source_authority.stage_root != frozen:
        raise ValueError("production source authority root drift")
    if source_authority.stage_manifest_sha256 != STAGE_MANIFEST_SHA256:
        raise ValueError("production source authority stage-manifest drift")
    manifest_path = source_authority.stage_manifest.path
    downstream_manifest = Pin(
        f"{stage}/runtime_receipts/{DOWNSTREAM_CODE_MANIFEST_BASENAME}",
        DOWNSTREAM_CODE_MANIFEST_SHA256,
    )
    verify_downstream_code_manifest(
        stage_root=stage,
        manifest_path=downstream_manifest.path,
        manifest_sha256=downstream_manifest.sha256,
    )
    digester = LazyCheckpointDigester(
        frozen_stage_root=frozen,
        stage_manifest_sha256=STAGE_MANIFEST_SHA256,
        source_authority=source_authority,
    )
    loaded: dict[str, LoadedCohort] = {}
    cohorts: dict[str, tuple[object, ...]] = {}
    source_specs = production_source_specs()
    for spec in source_specs:
        if spec.source_authority != SOURCE_PATH_AUTHORITY:
            raise ValueError("cohort source-path authority spec drift")
        value = load_cohort_from_frozen_stage(
            spec,
            digest_path=digester,
            source_authority=source_authority,
        )
        loaded[spec.cohort_key] = value
        cohorts[spec.cohort_key] = value.trees
    production: ProductionResolutionReceipt = validate_production_resolutions(cohorts)
    checkpoint_payload = checkpoint_production_payload(cohorts)

    identities_by_cohort: dict[str, tuple[CheckpointIdentity, ...]] = {}
    identity_lookup = digester.identities
    for cohort in COHORTS:
        ordered = tuple(
            identity_lookup[binding.checkpoint.path]
            for tree in cohorts[cohort]
            for binding in tree.checkpoint_manifest.bindings
        )
        if len(ordered) != PRODUCTION_N_NODES:
            raise AssertionError("checkpoint identity population cardinality corruption")
        identities_by_cohort[cohort] = ordered
    identity_payload = build_identity_index(
        identities_by_cohort, cohort_order=COHORTS
    )
    if len(read_identity_index(identity_payload)) != len(COHORTS) * PRODUCTION_N_NODES:
        raise AssertionError("checkpoint identity index cardinality corruption")

    runtime_receipts = Path(deployment) / "runtime_receipts"
    identity_file = PlannedFile(
        str(runtime_receipts / IDENTITY_BASENAME), _canonical_bytes(identity_payload)
    )
    checkpoint_file = PlannedFile(
        str(runtime_receipts / CHECKPOINT_BASENAME), _canonical_bytes(checkpoint_payload)
    )

    registry_path = f"{stage}/examples/behavior/corrected_behavior_protocols_v1.json"
    registry_pin = ArtifactPin(registry_path, sha256_file(registry_path))
    if registry_pin.sha256 != REGISTRY_SHA256:
        raise ValueError("behavior registry hash drift")
    encoder_receipt_path = f"{stage}/runtime_receipts/MINILM_ENCODER.json"
    encoder_receipt_pin = ArtifactPin(
        encoder_receipt_path, sha256_file(encoder_receipt_path)
    )
    if encoder_receipt_pin.sha256 != MINILM_RECEIPT_SHA256:
        raise ValueError("MiniLM encoder receipt hash drift")
    encoder_payload = _json_file(Path(encoder_receipt_path), "MiniLM receipt")
    encoder_snapshot = _artifact_pin(encoder_payload.get("snapshot"), "MiniLM snapshot")
    if encoder_snapshot.sha256 != MINILM_SNAPSHOT_SHA256:
        raise ValueError("MiniLM snapshot identity drift")
    require_encoder_snapshot_sha256(encoder_snapshot.path, encoder_snapshot.sha256)
    minilm_config_path = f"{stage}/examples/downstream/20260831/minilm_encoder.json"
    if sha256_file(minilm_config_path) != MINILM_CONFIG_SHA256:
        raise ValueError("MiniLM config hash drift")

    readers_pin = ArtifactPin(
        f"{snapshot}/weighttraits/distances/readers.py",
        sha256_file(f"{snapshot}/weighttraits/distances/readers.py"),
    )
    streaming_pin = ArtifactPin(
        f"{snapshot}/weighttraits/distances/streaming.py",
        sha256_file(f"{snapshot}/weighttraits/distances/streaming.py"),
    )
    identity_pin = identity_file.pin
    compatibility_specs = {
        cohort: _compatibility_spec(compatibility[cohort]) for cohort in COHORTS
    }
    compatibility_file_values: list[object] = list(extra_planned_files)
    for value in compatibility.values():
        planned = getattr(value, "planned_files", ())
        if isinstance(planned, (str, bytes)) or not isinstance(planned, Sequence):
            raise ValueError("compatibility planned_files must be a sequence")
        compatibility_file_values.extend(planned)
    compatibility_files = tuple(
        _planned_file(value) for value in compatibility_file_values
    )

    prompt_artifacts: dict[str, dict[str, Pin]] = {}
    materialization_receipts: dict[str, dict[str, Pin]] = {}
    direct_plans: dict[str, DirectCohortPlan] = {}
    inference_plans: dict[str, InferenceCohortPlan] = {}
    semantic_plans: dict[str, SemanticCohortPlan] = {}
    configs: dict[str, object] = {}
    plan_files: dict[str, PlannedFile] = {}
    config_files: dict[str, PlannedFile] = {}

    for cohort in COHORTS:
        panels, prompts, receipts = build_prompt_panels(
            cohort_key=cohort, stage_root=stage
        )
        prompt_artifacts[cohort] = prompts
        materialization_receipts[cohort] = receipts
        direct = build_direct_cohort_plan(
            cohort_key=cohort,
            source_stage_root=frozen,
            output_dir=f"{stage}/outputs/allnode_v1/direct/{cohort}",
            snapshot_root=snapshot,
            readers_source=readers_pin,
            streaming_source=streaming_pin,
            checkpoint_identity_index=identity_pin,
            source_path_authority=source_authority,
            leaf_compatibility=compatibility_specs[cohort],
            trees=cohorts[cohort],
        )
        read_direct_cohort_plan(direct.to_dict())
        direct_path = f"{deployment}/runtime_inputs/{cohort}/direct_plan.json"
        direct_file = PlannedFile(direct_path, _canonical_line_bytes(direct.to_dict()))
        direct_config = build_direct_job_config(
            cohort_key=cohort,
            output_dir=direct.output_dir,
            direct_plan=direct_file.pin,
        )
        read_direct_job_config(direct_config.to_dict())
        direct_config_file = PlannedFile(
            f"{deployment}/configs/jobs/direct_{cohort}.json",
            _canonical_line_bytes(direct_config.to_dict()),
        )

        manifests = tuple(tree.checkpoint_manifest for tree in cohorts[cohort])
        covariates = tuple(tree.node_covariates for tree in cohorts[cohort])
        inference = build_inference_cohort_plan(
            cohort_key=cohort,
            source_stage_root=frozen,
            checkpoint_identity_index=identity_pin,
            checkpoint_manifests=manifests,
            node_covariates=covariates,
            registry=registry_pin,
            panels=panels,
            planned_backend_template=backend_template(
                model_task=FROZEN_COHORT_MODEL_TASK[cohort],
                binding=manifests[0].bindings[0],
            ),
        )
        read_inference_cohort_plan(inference.to_dict())
        inference_path = f"{deployment}/runtime_inputs/{cohort}/inference_plan.json"
        inference_file = PlannedFile(
            inference_path, _canonical_line_bytes(inference.to_dict())
        )
        inference_config = build_inference_job_config(
            inference,
            runtime_plan_path=inference_path,
            runtime_plan_file_sha256=inference_file.sha256,
        )
        read_inference_job_config(inference_config.to_dict())
        inference_config_file = PlannedFile(
            f"{deployment}/configs/jobs/infer_{cohort}.json",
            _canonical_line_bytes(inference_config.to_dict()),
        )

        semantic = build_semantic_cohort_plan(
            inference,
            stage_root=stage,
            inference_plan=inference_file.pin,
            direct_job_config=direct_config_file.pin,
            encoder_receipt=encoder_receipt_pin,
            encoder_snapshot=encoder_snapshot,
        )
        read_semantic_cohort_plan(semantic.to_dict(), inference=inference)
        semantic_path = f"{deployment}/runtime_inputs/{cohort}/semantic_plan.json"
        semantic_file = PlannedFile(
            semantic_path, _canonical_line_bytes(semantic.to_dict())
        )
        semantic_config = build_semantic_job_config(
            semantic, runtime_plan_file_sha256=semantic_file.sha256
        )
        read_semantic_job_config(semantic_config.to_dict())
        semantic_config_file = PlannedFile(
            f"{deployment}/configs/jobs/semantic_{cohort}.json",
            _canonical_line_bytes(semantic_config.to_dict()),
        )

        direct_plans[cohort] = direct
        inference_plans[cohort] = inference
        semantic_plans[cohort] = semantic
        for logical_id, plan_file, config, config_file in (
            (f"direct_{cohort}", direct_file, direct_config, direct_config_file),
            (f"infer_{cohort}", inference_file, inference_config, inference_config_file),
            (f"semantic_{cohort}", semantic_file, semantic_config, semantic_config_file),
        ):
            plan_files[logical_id] = plan_file
            configs[logical_id] = config
            config_files[logical_id] = config_file

    matrix = matrix_plan_payload(
        cohorts, stage_root=stage, checkpoint_payload=checkpoint_payload
    )
    inference_summary = inference_plan_payload()
    source_core = {
        "schema": SOURCE_RESOLUTION_SCHEMA,
        "valid": True,
        "read_only": True,
        "scheduler_actions": 0,
        "source_path_authority": source_authority.to_dict(),
        "loads": [loaded[cohort].to_dict() for cohort in COHORTS],
        "production_resolution": production.to_dict(),
    }
    source_payload = {
        **source_core,
        "source_scope_sha256": canonical_sha256(source_core),
    }
    source_file = PlannedFile(
        str(runtime_receipts / SOURCE_BASENAME), _canonical_bytes(source_payload)
    )
    runtime_pins = {
        logical_id: Pin(value.path, value.sha256)
        for logical_id, value in sorted(plan_files.items())
    }
    config_pins = {
        logical_id: Pin(value.path, value.sha256)
        for logical_id, value in sorted(config_files.items())
    }
    code_payload = code_manifest_payload(deployment)
    code_file = PlannedFile(
        str(runtime_receipts / CODE_BASENAME), _canonical_bytes(code_payload)
    )
    runtime = production_runtime()
    execution = execution_contract_payload(
        checkpoint_payload=checkpoint_payload,
        checkpoint_identity_index=Pin(identity_file.path, identity_file.sha256),
        source_scope_sha256=source_payload["source_scope_sha256"],
        source_resolution=source_file.pin,
        allnode_code_manifest=code_file.pin,
        downstream_code_manifest=downstream_manifest,
        runtime=runtime,
        direct_plans=direct_plans,
        inference_plans=inference_plans,
        semantic_plans=semantic_plans,
        runtime_plan_pins=runtime_pins,
        job_config_pins=config_pins,
    )
    execution_file = PlannedFile(
        str(runtime_receipts / EXECUTION_BASENAME), _canonical_bytes(execution)
    )

    tests = (
        run_test_attestation(
            deployment_root=deployment, code_manifest_sha256=code_file.sha256
        )
        if run_tests
        else _refuse_skipped_tests()
    )
    if code_manifest_payload(deployment) != code_payload:
        raise ValueError("code source changed while the passing suite was running")
    source_authority.reattest()
    verify_downstream_code_manifest(
        stage_root=stage,
        manifest_path=downstream_manifest.path,
        manifest_sha256=downstream_manifest.sha256,
    )
    test_file = PlannedFile(
        str(runtime_receipts / TEST_BASENAME), _canonical_bytes(tests)
    )
    state_absence = _state_absence(stage, deployment)
    audit_path = str(runtime_receipts / AUDIT_BASENAME)
    bundle = build_dry_run_audit_bundle(
        audit_path=audit_path,
        stage_root=stage,
        frozen_stage=frozen,
        frozen_stage_manifest=Pin(manifest_path, STAGE_MANIFEST_SHA256),
        checkpoint_receipt=Pin(checkpoint_file.path, checkpoint_file.sha256),
        checkpoint_payload=checkpoint_payload,
        registry=Pin(registry_pin.path, registry_pin.sha256),
        prompt_artifacts=prompt_artifacts,
        materialization_receipts=materialization_receipts,
        minilm={
            "model": MINILM_MODEL,
            "revision": MINILM_REVISION,
            "runtime": MINILM_RUNTIME,
            "receipt": Pin(encoder_receipt_pin.path, encoder_receipt_pin.sha256).to_dict(),
            "config": Pin(
                minilm_config_path,
                MINILM_CONFIG_SHA256,
            ).to_dict(),
            "snapshot_sha256": MINILM_SNAPSHOT_SHA256,
        },
        job_configs=config_pins,
        code_manifest=Pin(code_file.path, code_file.sha256),
        source_resolution=source_file.pin,
        downstream_code_manifest=downstream_manifest,
        matrix_plan=matrix,
        inference_plan=inference_summary,
        execution_contract_receipt=Pin(execution_file.path, execution_file.sha256),
        execution_contract_payload=execution,
        test_attestation=tests,
        state_absence=state_absence,
    )
    validate_dry_run_audit_bundle(bundle.to_dict())
    audit_file = PlannedFile(
        audit_path,
        serialize_audit_core(bundle.audit_core).encode("utf-8"),
        terminal=True,
    )
    if audit_file.sha256 != bundle.audit_boundary.sha256:
        raise AssertionError("canonical audit boundary hash corruption")

    ordinary = (
        compatibility_files
        + (identity_file, checkpoint_file, source_file)
        + tuple(plan_files[key] for key in sorted(plan_files))
        + tuple(config_files[key] for key in sorted(config_files))
        + (execution_file, code_file, test_file)
    )
    _validate_planned_targets(
        ordinary + (audit_file,), stage_root=stage, deployment_root=deployment
    )
    core = {
        "schema": PREFLIGHT_PLAN_SCHEMA,
        "valid": True,
        "read_only_source_validation": True,
        "scheduler_actions": 0,
        "stage_root": stage,
        "frozen_stage": frozen,
        "source_path_authority": source_authority.to_dict(),
        "n_files": len(ordinary) + 1,
        "files": [value.to_dict() for value in ordinary + (audit_file,)],
        "terminal_path": audit_file.path,
    }
    return PreflightPlan(
        stage_root=stage,
        frozen_stage=frozen,
        source_authority=source_authority,
        files=ordinary + (audit_file,),
        audit_bundle=bundle,
        plan_sha256=canonical_sha256(core),
    )


def plan_default_compatibility(
    *,
    stage_root: str = DOWNSTREAM_STAGE,
    snapshot_root: str | None = None,
) -> dict[str, object]:
    """Build the only defensible legacy compatibility grid for production.

    Exact legacy rollups exist for the two Flan full-finetune conditions.
    There are no matching ``llama_full`` or ``llama_notrans`` legacy rollups;
    the r8/r64 LoRA cohorts are intentionally not substituted.
    """

    from .compatibility_preflight import (
        plan_available_leaf_compatibility,
        plan_unavailable_leaf_compatibility,
    )

    stage = _root(stage_root, "stage_root")
    snapshot = _root(snapshot_root or f"{stage}/src", "snapshot_root")
    root = Path(stage) / "deployments" / "allnode_v1" / "runtime_inputs" / "compatibility"
    table = (
        Path(stage)
        / "deployments"
        / "allnode_v1"
        / "reference_sources"
        / "iclr_draft_v4.tex"
    )
    result: dict[str, object] = {
        "llama_full": plan_unavailable_leaf_compatibility(
            cohort_key="llama_full",
            reason=(
                "No exact legacy llama_full leaf-direct rollup exists in the frozen or "
                "downstream stages; llama_r8 and llama_r64 are different LoRA cohorts "
                "and must not be substituted."
            ),
            destination_dir=root / "llama_full",
        ),
        "llama_notrans": plan_unavailable_leaf_compatibility(
            cohort_key="llama_notrans",
            reason=(
                "No exact legacy llama_notrans leaf-direct rollup exists in the frozen "
                "or downstream stages; llama_r8 and llama_r64 are different LoRA cohorts "
                "and must not be substituted."
            ),
            destination_dir=root / "llama_notrans",
        ),
    }
    for cohort in ("flan_full", "flan_notrans"):
        result[cohort] = plan_available_leaf_compatibility(
            cohort_key=cohort,
            legacy_output_root=Path(stage) / "outputs" / "direct" / cohort,
            old_table_source=table,
            destination_dir=root / cohort,
            snapshot_root=snapshot,
        )
    return {cohort: result[cohort] for cohort in COHORTS}


def materialize_preflight(plan: PreflightPlan) -> Pin:
    """Publish one validated plan exclusively; the audit is always last."""

    if not isinstance(plan, PreflightPlan):
        raise ValueError("plan must be a PreflightPlan")
    core = {key: value for key, value in plan.to_dict().items() if key != "plan_sha256"}
    if canonical_sha256(core) != plan.plan_sha256:
        raise ValueError("preflight publish-plan hash drift")
    plan.source_authority.reattest()
    bundle = validate_dry_run_audit_bundle(plan.audit_bundle.to_dict())
    if plan.files[-1].path != bundle.audit_boundary.path or not plan.files[-1].terminal:
        raise ValueError("preflight terminal artifact differs from audit boundary")
    _validate_planned_targets(
        plan.files,
        stage_root=plan.stage_root,
        deployment_root=f"{plan.stage_root}/deployments/allnode_v1",
    )
    for value in plan.files:
        if value.terminal:
            plan.source_authority.reattest()
        _publish_exclusive(value)
    plan.source_authority.reattest()
    return Pin(plan.files[-1].path, plan.files[-1].sha256)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan production and optionally publish; never contacts the scheduler."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", default=DOWNSTREAM_STAGE)
    parser.add_argument("--frozen-stage", default=FROZEN_STAGE)
    parser.add_argument("--snapshot-root")
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="exclusively publish validated preflight files with the audit last",
    )
    args = parser.parse_args(argv)
    compatibility = plan_default_compatibility(
        stage_root=args.stage_root,
        snapshot_root=args.snapshot_root,
    )
    plan = plan_preflight(
        compatibility=compatibility,
        stage_root=args.stage_root,
        frozen_stage=args.frozen_stage,
        snapshot_root=args.snapshot_root,
    )
    published = materialize_preflight(plan) if args.materialize else None
    print(
        json.dumps(
            {
                "schema": "weighttraits.all_node_preflight_cli_status.v1",
                "valid": True,
                "materialized": published is not None,
                "plan_sha256": plan.plan_sha256,
                "n_files": len(plan.files),
                "audit": plan.files[-1].pin.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _state_absence(stage: str, deployment: str) -> dict[str, bool]:
    intent = Path(stage) / "runtime_receipts" / "ALLNODE_V1_SUBMIT_INTENT.json"
    jobs = Path(stage) / "runtime_receipts" / "ALLNODE_V1_SUBMIT_JOBS.json"
    output_roots = [
        Path(stage) / "outputs" / "allnode_v1" / kind / cohort
        for kind in ("direct", "inference", "semantic")
        for cohort in COHORTS
    ]
    log = Path(stage) / "slurm_logs" / "allnode_v1"
    if any(os.path.lexists(path) for path in output_roots + [log]):
        raise FileExistsError("allnode_v1 output or log state already exists")
    if os.path.lexists(intent) or os.path.lexists(jobs):
        raise FileExistsError("allnode_v1 intent or jobs receipt already exists")
    return {
        "outputs_exist": False,
        "intent_exists": False,
        "jobs_receipt_exists": False,
    }


def _node_covariate_scope(plan: DirectCohortPlan) -> str:
    return canonical_sha256(
        [tree.node_covariates.covariates_sha256 for tree in plan.trees]
    )


def _stage_from_plan_pin(pin: Pin) -> str:
    path = Path(pin.path)
    try:
        position = path.parts.index("deployments")
    except ValueError as exc:
        raise ValueError("runtime plan pin is outside a deployment") from exc
    return str(Path(*path.parts[:position]))


def _compatibility_spec(value: object) -> LeafCompatibilitySpec:
    if isinstance(value, LeafCompatibilitySpec):
        return read_leaf_compatibility_spec(value.to_dict())
    spec = getattr(value, "spec", None)
    if isinstance(spec, LeafCompatibilitySpec):
        return read_leaf_compatibility_spec(spec.to_dict())
    if isinstance(value, Mapping):
        return read_leaf_compatibility_spec(value)
    raise ValueError("compatibility value must expose an exact LeafCompatibilitySpec")


def _planned_file(value: object) -> PlannedFile:
    if isinstance(value, PlannedFile):
        return value
    target = getattr(value, "target", None)
    payload = getattr(value, "payload", None)
    if payload is None:
        payload = getattr(value, "content", None)
    if target is None:
        target = getattr(value, "pin", None)
    if isinstance(target, ArtifactPin) and isinstance(payload, bytes):
        result = PlannedFile(target.path, payload)
        if result.sha256 != target.sha256:
            raise ValueError("external planned-file pin differs from its bytes")
        return result
    path = getattr(value, "path", None)
    if isinstance(path, str) and isinstance(payload, bytes):
        return PlannedFile(path, payload)
    raise ValueError("external planned file must expose target/pin and exact bytes")


def _validate_planned_targets(
    files: Sequence[PlannedFile], *, stage_root: str, deployment_root: str
) -> None:
    stage = Path(_absolute(stage_root, "stage_root"))
    deployment = Path(_absolute(deployment_root, "deployment_root"))
    if deployment != stage / "deployments" / "allnode_v1":
        raise ValueError("preflight deployment publication root drift")
    paths = [value.path for value in files]
    if len(paths) != len(set(paths)):
        raise ValueError("preflight publish plan reuses an artifact path")
    if not files or sum(value.terminal for value in files) != 1 or not files[-1].terminal:
        raise ValueError("preflight publish plan must contain one terminal file last")
    for value in files:
        path = Path(_absolute(value.path, "planned file path"))
        if deployment not in path.parents:
            raise ValueError(
                "preflight publications must remain strictly inside the isolated deployment"
            )
        if not value.payload:
            raise ValueError("preflight cannot publish an empty artifact")
        if os.path.lexists(value.path):
            raise FileExistsError(f"preflight target already exists: {value.path}")


def _publish_exclusive(value: PlannedFile) -> None:
    destination = Path(value.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or destination.parent.is_symlink():
        raise ValueError("preflight target parent is not canonical")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value.payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"preflight target appeared during publication: {destination}"
            ) from exc
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _canonical_line_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _json_file(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_stage_manifest_entry(
    manifest_path: str,
    *,
    relative_path: str,
    sha256: str,
) -> None:
    """Require one exact sha256sum-style entry in the pinned stage manifest."""

    manifest = Path(_absolute(manifest_path, "stage manifest path"))
    expected_sha = _sha(sha256, "stage manifest entry sha256")
    if (
        not relative_path
        or Path(relative_path).is_absolute()
        or relative_path in {".", ".."}
        or ".." in Path(relative_path).parts
        or os.path.normpath(relative_path) != relative_path
    ):
        raise ValueError("stage manifest entry path must be normalized and relative")
    stage = manifest.parent.parent
    expected = str(stage / relative_path)
    matches: list[str] = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot read frozen stage manifest") from exc
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or _SHA256.fullmatch(parts[0]) is None:
            raise ValueError("frozen stage manifest row is not sha256sum format")
        raw_path = parts[1]
        if raw_path.startswith("*"):
            raw_path = raw_path[1:]
        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        candidate = raw_path if os.path.isabs(raw_path) else str(stage / raw_path)
        if os.path.normpath(candidate) != candidate:
            raise ValueError("frozen stage manifest contains a non-normal path")
        if candidate == expected:
            matches.append(parts[0])
    if matches != [expected_sha]:
        raise ValueError("pinned symlink inventory is absent or duplicated in stage manifest")


def _jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not lines or any(not line for line in lines):
        raise ValueError(f"{label} must be nonempty JSONL without blank rows")
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} has invalid JSON at row {index}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} row {index} must be an object")
        if not isinstance(value.get("prompt"), str):
            raise ValueError(f"{label} row {index} lacks rendered prompt text")
        rows.append(value)
    coordinates = [(row.get("probe_id"), row.get("prompt_id")) for row in rows]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"{label} contains duplicate prompt coordinates")
    return rows


def _load_rendered_prompts(path: Path, *, stage_root: Path) -> list[object]:
    """Use the pinned WeightTraits prompt loader and reject import-path drift."""

    contracts = importlib.import_module("weighttraits.behavior.contracts")
    responses = importlib.import_module("weighttraits.behavior.responses")
    expected_root = stage_root / "src" / "weighttraits" / "behavior"
    for module, filename in (
        (contracts, "contracts.py"),
        (responses, "responses.py"),
    ):
        module_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
        if module_path != expected_root / filename or module_path.is_symlink():
            raise ValueError("imported WeightTraits prompt API path drift")
    registry_path = (
        stage_root / "examples" / "behavior" / "corrected_behavior_protocols_v1.json"
    )
    contracts.load_behavior_protocol_registry(registry_path)
    rows = responses.load_rendered_behavior_prompts(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("pinned WeightTraits prompt loader returned no rows")
    return rows


def _artifact_pin(value: object, label: str) -> ArtifactPin:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path+sha256 mapping")
    return ArtifactPin(_absolute(value["path"], f"{label}.path"), _sha(value["sha256"], f"{label}.sha256"))


def _absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty trimmed string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be a normalized absolute path")
    return value


def _root(value: object, label: str) -> str:
    path = Path(_absolute(value, label))
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    return str(path)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty trimmed text")
    return value


def _refuse_skipped_tests() -> dict[str, Any]:
    raise ValueError("production preflight cannot skip the full passing test suite")


__all__ = [
    "CODE_MANIFEST_SCHEMA",
    "LazyCheckpointDigester",
    "PlannedFile",
    "PreflightPlan",
    "backend_template",
    "build_prompt_panels",
    "checkpoint_production_payload",
    "code_manifest_payload",
    "execution_contract_payload",
    "inference_plan_payload",
    "materialize_preflight",
    "matrix_plan_payload",
    "main",
    "plan_default_compatibility",
    "plan_preflight",
    "run_test_attestation",
]


if __name__ == "__main__":
    raise SystemExit(main())
