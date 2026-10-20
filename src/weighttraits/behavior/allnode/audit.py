"""Pure validation model for the all-node pre-submission dry-run audit.

The persisted hash boundary is the canonical ``audit_core`` JSON only.  The
validated gate plan is derived *after* that core is hashed and points back to
``Pin(audit_path, audit_core_sha256)``.  The in-memory bundle contains both,
but is not itself the pinned artifact.  This two-layer construction avoids a
self-hash cycle while still binding the exact gate to the exact audit bytes.

Nothing in this module reads or writes a file, runs tests, or contacts Slurm.
Every path/hash and every attestation is supplied by the caller and validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .gate import (
    COHORTS,
    GateContract,
    Pin,
    build_gate_contract,
    canonical_sha256,
    validate_gate_contract,
)
from .provenance import (
    DOWNSTREAM_CODE_MANIFEST_BASENAME,
    DOWNSTREAM_CODE_MANIFEST_SHA256,
    PRODUCTION_INTERPRETER,
    SOURCE_RESOLUTION_BASENAME,
)


CORE_SCHEMA = "weighttraits.allnode_dry_run_audit_core.v1"
BUNDLE_SCHEMA = "weighttraits.allnode_dry_run_audit_bundle.v1"
CHECKPOINT_SCHEMA = "weighttraits.all_node_production_checkpoints.v1"
MATRIX_PLAN_SCHEMA = "weighttraits.all_node_matrix_plan.v1"
INFERENCE_PLAN_SCHEMA = "weighttraits.all_node_inference_plan.v1"
TEST_SCHEMA = "weighttraits.all_node_test_attestation.v1"
EXECUTION_CONTRACT_SCHEMA = "weighttraits.all_node_execution_contract.v1"

TREE_IDS = tuple(f"confirm_paper_tree_{index:03d}" for index in range(1, 51))
N_NODES = 641
N_INTERNAL = 276
N_LEAVES = 365

REGISTRY_SHA256 = "c13b025802b7fae7b5c1770626865e2bab7d45e5783137b85fd041e2b4689bab"
STAGE_MANIFEST_SHA256 = "642912687e67d0a48162a8d5cc0a95164ad92350ea629cd396fcfdd7917cd5d1"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_RECEIPT_SHA256 = "8ddd16f39a086917488872ae8d6acdc7226a958f5cf0fea90b65140c3b568493"
MINILM_CONFIG_SHA256 = "be6af486d90db923b51f301ea0e7b8214b5f1c5bf692be17ce4077379a42be90"
MINILM_SNAPSHOT_SHA256 = "152ef3ad82d673e7378b1e53da7ec8bef6b3e2b8b92dea0ac185c2db5dcf6d3a"
MINILM_RUNTIME = "sentence-transformers==5.4.1"

TRANSLATION = "legacy_translation_200x1_greedy"
MODERN_MC = "modern_mc_100x3_sampled"
DOLLY = "dolly_open_ended_30x3_sampled"
PROTOCOLS = {
    "llama_full": (TRANSLATION, MODERN_MC, DOLLY),
    "llama_notrans": (TRANSLATION,),
    "flan_full": (TRANSLATION, MODERN_MC),
    "flan_notrans": (TRANSLATION,),
}
RESPONSES_PER_NODE = {
    "llama_full": 1490,
    "llama_notrans": 200,
    "flan_full": 1400,
    "flan_notrans": 200,
}
SEMANTIC_ENDPOINTS = {
    "llama_full": (
        (TRANSLATION, "translation", 200, 1),
        (MODERN_MC, "hellaswag", 100, 3),
        (MODERN_MC, "arc_challenge", 100, 3),
        (MODERN_MC, "mmlu", 100, 3),
        (MODERN_MC, "truthfulqa", 100, 3),
        (DOLLY, "dolly_open_ended", 30, 3),
    ),
    "llama_notrans": ((TRANSLATION, "translation", 200, 1),),
    "flan_full": (
        (TRANSLATION, "translation", 200, 1),
        (MODERN_MC, "hellaswag", 100, 3),
        (MODERN_MC, "arc_challenge", 100, 3),
        (MODERN_MC, "mmlu", 100, 3),
        (MODERN_MC, "truthfulqa", 100, 3),
    ),
    "flan_notrans": ((TRANSLATION, "translation", 200, 1),),
}
PROMPT_SHA256 = {
    "llama_full": {
        TRANSLATION: "b752ed42f75d7470bd13333a52e7e48c7caf184d7f001ed51881c6bca2244520",
        MODERN_MC: "7d89a05453cd885950d7168dace7e95b6c4005f94d632a10da5262bd830408d2",
        DOLLY: "081c850c0f636c03ee2e1c512e7cbb2c8d1dd0871a51f46c388cbfbb091096e3",
    },
    "llama_notrans": {
        TRANSLATION: "b752ed42f75d7470bd13333a52e7e48c7caf184d7f001ed51881c6bca2244520"
    },
    "flan_full": {
        TRANSLATION: "8e611e4e1ede859d47b7a61baf6fc563576b7de5bf1e231efc37c66d164957fe",
        MODERN_MC: "493a7353f8beb700257e32910e277b2d12ce8931f69db008ec7c7a97197ec05f",
    },
    "flan_notrans": {
        TRANSLATION: "8e611e4e1ede859d47b7a61baf6fc563576b7de5bf1e231efc37c66d164957fe"
    },
}
MATERIALIZATION_RECEIPT_SHA256 = {
    "causal_lm": "8c892c9e04278fb0b3d2f2defcd901b0a341491e175f0e4e689524d3e5ea6fcf",
    "seq2seq": "40a819141e242b7ce0fd19f8c6a242497916c287f49d28c95b25dd93b38235f1",
}


@dataclass(frozen=True)
class DryRunAuditBundle:
    """Derived validation view; only ``audit_core`` is the pinned file."""

    audit_core: Mapping[str, Any]
    audit_boundary: Pin
    gate_contract: GateContract

    @property
    def valid(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_SCHEMA,
            "valid": True,
            "audit_core": dict(self.audit_core),
            "audit_boundary": self.audit_boundary.to_dict(),
            "gate_contract": self.gate_contract.to_dict(),
            "gate_contract_sha256": self.gate_contract.sha256,
        }


def serialize_audit_core(core: Mapping[str, Any]) -> str:
    """Return the exact bytes whose SHA-256 is used by the gate pin."""

    validated = validate_audit_core(core)
    return json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_dry_run_audit_bundle(
    *,
    audit_path: str,
    stage_root: str,
    frozen_stage: str,
    frozen_stage_manifest: Pin,
    checkpoint_receipt: Pin,
    checkpoint_payload: Mapping[str, Any],
    registry: Pin,
    prompt_artifacts: Mapping[str, Mapping[str, Pin]],
    materialization_receipts: Mapping[str, Mapping[str, Pin]],
    minilm: Mapping[str, Any],
    job_configs: Mapping[str, Pin],
    code_manifest: Pin,
    source_resolution: Pin,
    downstream_code_manifest: Pin,
    matrix_plan: Mapping[str, Any],
    inference_plan: Mapping[str, Any],
    execution_contract_receipt: Pin,
    execution_contract_payload: Mapping[str, Any],
    test_attestation: Mapping[str, Any],
    state_absence: Mapping[str, Any],
) -> DryRunAuditBundle:
    """Validate all preflight evidence and bind it to the exact gate plan."""

    stage = _path(stage_root, "stage_root")
    frozen = _path(frozen_stage, "frozen_stage")
    deployment = stage / "deployments" / "allnode_v1"
    audit = _path(audit_path, "audit_path")
    expected_audit = deployment / "runtime_receipts" / "ALLNODE_V1_DRY_RUN_AUDIT.json"
    if audit != expected_audit:
        raise ValueError("audit_path differs from the unique isolated audit boundary")

    core = {
        "schema": CORE_SCHEMA,
        "valid": True,
        "stage_root": str(stage),
        "frozen_stage": str(frozen),
        "deployment_root": str(deployment),
        "scope": _scope(),
        "frozen_stage_manifest": frozen_stage_manifest.to_dict(),
        "checkpoint_production": {
            "receipt": checkpoint_receipt.to_dict(),
            "payload": dict(checkpoint_payload),
            "payload_sha256": canonical_sha256(dict(checkpoint_payload)),
        },
        "registry": registry.to_dict(),
        "prompt_artifacts": _pin_grid_dict(prompt_artifacts),
        "materialization_receipts": _pin_grid_dict(materialization_receipts),
        "minilm": dict(minilm),
        "job_configs": {key: value.to_dict() for key, value in sorted(job_configs.items())},
        "code_manifest": code_manifest.to_dict(),
        "source_resolution": source_resolution.to_dict(),
        "downstream_code_manifest": downstream_code_manifest.to_dict(),
        "matrix_plan": dict(matrix_plan),
        "inference_plan": dict(inference_plan),
        "execution_contract": {
            "receipt": execution_contract_receipt.to_dict(),
            "payload": dict(execution_contract_payload),
            "payload_sha256": canonical_sha256(dict(execution_contract_payload)),
        },
        "test_attestation": dict(test_attestation),
        "state_absence": dict(state_absence),
        "execution_overrides": {},
        "no_automatic_recovery": True,
    }
    validated_core = validate_audit_core(core)
    core_hash = canonical_sha256(validated_core)
    boundary = Pin(str(audit), core_hash)
    gate = build_gate_contract(
        stage_root=str(stage),
        frozen_stage=str(frozen),
        code_manifest=code_manifest,
        source_resolution=source_resolution,
        source_scope_sha256=execution_contract_payload["source_scope_sha256"],
        downstream_code_manifest=downstream_code_manifest,
        python_version=test_attestation["python_version"],
        dry_run_audit=boundary,
    )
    gate = validate_gate_contract(gate)
    _validate_config_gate_binding(validated_core["job_configs"], gate)
    bundle = DryRunAuditBundle(validated_core, boundary, gate)
    validate_dry_run_audit_bundle(bundle.to_dict())
    return bundle


def validate_dry_run_audit_bundle(payload: Mapping[str, Any]) -> DryRunAuditBundle:
    """Revalidate an in-memory bundle and its non-circular hash boundary."""

    if not isinstance(payload, Mapping):
        raise ValueError("audit bundle must be a mapping")
    _keys(
        payload,
        {
            "schema",
            "valid",
            "audit_core",
            "audit_boundary",
            "gate_contract",
            "gate_contract_sha256",
        },
        "audit bundle",
    )
    if payload["schema"] != BUNDLE_SCHEMA or payload["valid"] is not True:
        raise ValueError("audit bundle schema/validity drift")
    core = validate_audit_core(payload["audit_core"])
    boundary = _pin(payload["audit_boundary"], "audit_boundary")
    if boundary.sha256 != canonical_sha256(core):
        raise ValueError("audit boundary hash differs from canonical audit-core bytes")
    stage = _path(core["stage_root"], "stage_root")
    deployment = stage / "deployments" / "allnode_v1"
    expected_audit = deployment / "runtime_receipts" / "ALLNODE_V1_DRY_RUN_AUDIT.json"
    if _path(boundary.path, "audit_boundary.path") != expected_audit:
        raise ValueError("audit boundary path differs from the unique isolated audit path")
    gate = validate_gate_contract(payload["gate_contract"])
    if gate.dry_run_audit != boundary:
        raise ValueError("GateContract dry_run_audit pin differs from audit boundary")
    code_pin = _pin(core["code_manifest"], "code_manifest")
    if gate.code_manifest != code_pin:
        raise ValueError("GateContract code manifest differs from audit core")
    if gate.sha256 != payload["gate_contract_sha256"]:
        raise ValueError("gate_contract_sha256 drift")
    _validate_config_gate_binding(core["job_configs"], gate)
    return DryRunAuditBundle(core, boundary, gate)


def validate_audit_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("audit core must be a mapping")
    _keys(
        payload,
        {
            "schema",
            "valid",
            "stage_root",
            "frozen_stage",
            "deployment_root",
            "scope",
            "frozen_stage_manifest",
            "checkpoint_production",
            "registry",
            "prompt_artifacts",
            "materialization_receipts",
            "minilm",
            "job_configs",
            "code_manifest",
            "source_resolution",
            "downstream_code_manifest",
            "matrix_plan",
            "inference_plan",
            "execution_contract",
            "test_attestation",
            "state_absence",
            "execution_overrides",
            "no_automatic_recovery",
        },
        "audit core",
    )
    if payload["schema"] != CORE_SCHEMA or payload["valid"] is not True:
        raise ValueError("audit core schema/validity drift")
    if payload["scope"] != _scope():
        raise ValueError("audit scope differs from exact 4-cohort/50-tree/641-node contract")
    if payload["execution_overrides"] != {}:
        raise ValueError("execution overrides are forbidden")
    if payload["no_automatic_recovery"] is not True:
        raise ValueError("no_automatic_recovery must be true")

    stage = _path(payload["stage_root"], "stage_root")
    frozen = _path(payload["frozen_stage"], "frozen_stage")
    deployment = _path(payload["deployment_root"], "deployment_root")
    if deployment != stage / "deployments" / "allnode_v1":
        raise ValueError("deployment_root differs from isolated allnode_v1 path")
    if stage == frozen or stage in frozen.parents or frozen in stage.parents:
        raise ValueError("stage and frozen roots must be disjoint")

    stage_manifest = _pin(payload["frozen_stage_manifest"], "frozen_stage_manifest")
    _under(stage_manifest.path, frozen, "frozen_stage_manifest")
    if stage_manifest.sha256 != STAGE_MANIFEST_SHA256:
        raise ValueError("frozen stage manifest hash drift")
    checkpoint = _validate_checkpoint_production(payload["checkpoint_production"])
    _under(checkpoint.path, deployment, "checkpoint receipt")
    registry = _pin(payload["registry"], "registry")
    expected_registry = stage / "examples" / "behavior" / "corrected_behavior_protocols_v1.json"
    if _path(registry.path, "registry.path") != expected_registry or registry.sha256 != REGISTRY_SHA256:
        raise ValueError("registry hash drift")
    _validate_pin_grids(payload["prompt_artifacts"], payload["materialization_receipts"], stage)
    _validate_minilm(payload["minilm"], stage)
    _validate_job_configs(payload["job_configs"], deployment)
    code = _pin(payload["code_manifest"], "code_manifest")
    _under(code.path, deployment, "code_manifest")
    source_resolution = _pin(payload["source_resolution"], "source_resolution")
    if _path(source_resolution.path, "source_resolution.path") != (
        deployment / "runtime_receipts" / SOURCE_RESOLUTION_BASENAME
    ):
        raise ValueError("source resolution receipt path drift")
    downstream_code = _pin(
        payload["downstream_code_manifest"], "downstream_code_manifest"
    )
    if (
        _path(downstream_code.path, "downstream_code_manifest.path")
        != stage / "runtime_receipts" / DOWNSTREAM_CODE_MANIFEST_BASENAME
        or downstream_code.sha256 != DOWNSTREAM_CODE_MANIFEST_SHA256
    ):
        raise ValueError("downstream code manifest pin drift")
    test_runtime = _validate_test_attestation(payload["test_attestation"], code)
    matrix = _validate_matrix_plan(payload["matrix_plan"], stage, checkpoint_payload=payload["checkpoint_production"]["payload"])
    _validate_inference_plan(payload["inference_plan"])
    _validate_execution_contract(
        payload["execution_contract"],
        stage=stage,
        deployment=deployment,
        checkpoint_payload=payload["checkpoint_production"]["payload"],
        job_configs=payload["job_configs"],
        code_manifest=code,
        source_resolution=source_resolution,
        downstream_code_manifest=downstream_code,
        test_runtime=test_runtime,
    )
    _validate_state_absence(payload["state_absence"])

    # Matrix and checkpoint topology are already cross-bound above.  Bind the
    # inference population to the same exact cohort/node totals as well.
    if matrix["topology_scope_sha256"] != payload["checkpoint_production"]["payload"]["topology_scope_sha256"]:
        raise ValueError("matrix plan topology differs from checkpoint production scope")
    return json.loads(json.dumps(dict(payload), sort_keys=True))


def _scope() -> dict[str, Any]:
    return {
        "cohort_ids": list(COHORTS),
        "n_cohorts": 4,
        "tree_ids": list(TREE_IDS),
        "n_trees_per_cohort": 50,
        "n_nodes_per_cohort": N_NODES,
        "n_internal_per_cohort": N_INTERNAL,
        "n_leaves_per_cohort": N_LEAVES,
        "primary_population": "all_trained_nodes",
        "sensitivity_population": "trained_leaves",
        "internal_node_treatment": "observed_ancestors",
        "exclude_untrained_root": True,
    }


def _validate_checkpoint_production(raw: object) -> Pin:
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint_production must be a mapping")
    _keys(raw, {"receipt", "payload", "payload_sha256"}, "checkpoint_production")
    pin = _pin(raw["receipt"], "checkpoint receipt")
    payload = raw["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if raw["payload_sha256"] != canonical_sha256(dict(payload)):
        raise ValueError("checkpoint payload hash drift")
    if pin.sha256 != raw["payload_sha256"]:
        raise ValueError("checkpoint receipt pin is not the canonical checkpoint payload")
    _keys(
        payload,
        {
            "schema",
            "valid",
            "population",
            "internal_node_treatment",
            "exclude_root",
            "n_cohorts",
            "cohort_ids",
            "n_trees_per_cohort",
            "n_nodes_per_cohort",
            "n_internal_per_cohort",
            "n_leaves_per_cohort",
            "topology_scope_sha256",
            "cohorts",
        },
        "checkpoint payload",
    )
    exact = (
        payload["schema"] == CHECKPOINT_SCHEMA
        and payload["valid"] is True
        and payload["population"] == "all_trained_nodes"
        and payload["internal_node_treatment"] == "observed_ancestors"
        and payload["exclude_root"] is True
        and payload["n_cohorts"] == 4
        and payload["cohort_ids"] == list(COHORTS)
        and payload["n_trees_per_cohort"] == 50
        and payload["n_nodes_per_cohort"] == N_NODES
        and payload["n_internal_per_cohort"] == N_INTERNAL
        and payload["n_leaves_per_cohort"] == N_LEAVES
        and _is_sha(payload["topology_scope_sha256"])
    )
    if not exact:
        raise ValueError("checkpoint production receipt differs from exact production scope")
    cohorts = payload["cohorts"]
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(COHORTS):
        raise ValueError("checkpoint cohort map/order drift")
    for cohort in COHORTS:
        row = cohorts[cohort]
        if not isinstance(row, Mapping):
            raise ValueError("checkpoint cohort row must be a mapping")
        _keys(
            row,
            {
                "population",
                "internal_node_treatment",
                "n_trees",
                "n_nodes",
                "n_internal",
                "n_leaves",
                "tree_ids",
                "manifest_scope",
                "manifest_scope_sha256",
            },
            f"checkpoint cohort {cohort}",
        )
        if (
            row["population"] != "all_trained_nodes"
            or row["internal_node_treatment"] != "observed_ancestors"
            or (row["n_trees"], row["n_nodes"], row["n_internal"], row["n_leaves"])
            != (50, N_NODES, N_INTERNAL, N_LEAVES)
            or row["tree_ids"] != list(TREE_IDS)
        ):
            raise ValueError(f"checkpoint cohort {cohort} scope drift")
        manifests = row["manifest_scope"]
        if not isinstance(manifests, list) or len(manifests) != 50:
            raise ValueError(f"checkpoint cohort {cohort} must bind 50 manifests")
        for tree_id, manifest in zip(TREE_IDS, manifests):
            if (
                not isinstance(manifest, Mapping)
                or set(manifest) != {"tree_id", "checkpoint_set_sha256"}
                or manifest["tree_id"] != tree_id
                or not _is_sha(manifest["checkpoint_set_sha256"])
            ):
                raise ValueError(f"checkpoint cohort {cohort} manifest scope drift")
        if row["manifest_scope_sha256"] != canonical_sha256(manifests):
            raise ValueError(f"checkpoint cohort {cohort} manifest-scope hash drift")
    return pin


def _validate_pin_grids(prompts: object, receipts: object, stage: PurePosixPath) -> None:
    for label, grid in (("prompt_artifacts", prompts), ("materialization_receipts", receipts)):
        if not isinstance(grid, Mapping) or set(grid) != set(COHORTS):
            raise ValueError(f"{label} cohort grid/order drift")
        for cohort in COHORTS:
            rows = grid[cohort]
            if not isinstance(rows, Mapping) or set(rows) != set(PROTOCOLS[cohort]):
                raise ValueError(f"{label}.{cohort} protocol grid/order drift")
            for protocol in PROTOCOLS[cohort]:
                pin = _pin(rows[protocol], f"{label}.{cohort}.{protocol}")
                model_task = "causal_lm" if cohort.startswith("llama_") else "seq2seq"
                if label == "prompt_artifacts":
                    filename = f"{protocol}.{model_task}.prompts.jsonl"
                    expected_path = stage / "outputs" / "prompts" / model_task / filename
                    expected_sha = PROMPT_SHA256[cohort][protocol]
                else:
                    expected_path = (
                        stage / "outputs" / "prompts" / model_task
                        / "prompt_materialization_receipt.json"
                    )
                    expected_sha = MATERIALIZATION_RECEIPT_SHA256[model_task]
                if _path(pin.path, f"{label}.{cohort}.{protocol}.path") != expected_path:
                    raise ValueError(f"{label} path drift for {cohort}/{protocol}")
                if pin.sha256 != expected_sha:
                    raise ValueError(f"{label} hash drift for {cohort}/{protocol}")


def _validate_minilm(raw: object, stage: PurePosixPath) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("minilm must be a mapping")
    _keys(raw, {"model", "revision", "runtime", "receipt", "config", "snapshot_sha256"}, "minilm")
    receipt = _pin(raw["receipt"], "minilm.receipt")
    config = _pin(raw["config"], "minilm.config")
    if (
        raw["model"] != MINILM_MODEL
        or raw["revision"] != MINILM_REVISION
        or raw["runtime"] != MINILM_RUNTIME
        or raw["snapshot_sha256"] != MINILM_SNAPSHOT_SHA256
        or receipt.sha256 != MINILM_RECEIPT_SHA256
        or config.sha256 != MINILM_CONFIG_SHA256
        or _path(receipt.path, "minilm.receipt.path")
        != stage / "runtime_receipts" / "MINILM_ENCODER.json"
        or _path(config.path, "minilm.config.path")
        != stage / "examples" / "downstream" / "20260831" / "minilm_encoder.json"
    ):
        raise ValueError("MiniLM pin drift")


def _validate_job_configs(raw: object, deployment: PurePosixPath) -> None:
    expected_ids = [f"direct_{c}" for c in COHORTS]
    for cohort in COHORTS:
        expected_ids.extend((f"infer_{cohort}", f"semantic_{cohort}"))
    if not isinstance(raw, Mapping) or list(raw) != sorted(expected_ids):
        raise ValueError("job config pin set differs from exact 12-job plan")
    seen: set[str] = set()
    for logical_id in sorted(expected_ids):
        pin = _pin(raw[logical_id], f"job_configs.{logical_id}")
        _under(pin.path, deployment, f"job_configs.{logical_id}")
        if pin.path in seen:
            raise ValueError("job config paths cannot be reused")
        seen.add(pin.path)


def _validate_matrix_plan(raw: object, stage: PurePosixPath, *, checkpoint_payload: object) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("matrix_plan must be a mapping")
    _keys(
        raw,
        {
            "schema",
            "valid",
            "population",
            "internal_node_treatment",
            "metric",
            "cohort_ids",
            "n_trees",
            "n_nodes",
            "n_internal",
            "n_leaves",
            "topology_scope",
            "topology_scope_sha256",
            "matrix_order_scope_sha256",
            "cohorts",
        },
        "matrix_plan",
    )
    if (
        raw["schema"] != MATRIX_PLAN_SCHEMA
        or raw["valid"] is not True
        or raw["population"] != "all_trained_nodes"
        or raw["internal_node_treatment"] != "observed_ancestors"
        or raw["metric"] != "cosine"
        or raw["cohort_ids"] != list(COHORTS)
        or (raw["n_trees"], raw["n_nodes"], raw["n_internal"], raw["n_leaves"])
        != (50, N_NODES, N_INTERNAL, N_LEAVES)
    ):
        raise ValueError("matrix plan scope drift")
    scope = raw["topology_scope"]
    if not isinstance(scope, list) or len(scope) != 50:
        raise ValueError("matrix plan must contain 50 topology/order rows")
    projection = []
    totals = [0, 0, 0]
    for tree_id, row in zip(TREE_IDS, scope):
        if not isinstance(row, Mapping):
            raise ValueError("matrix topology row must be a mapping")
        _keys(
            row,
            {
                "tree_id",
                "n_nodes",
                "n_internal",
                "n_leaves",
                "topology_sha256",
                "nodes_sha256",
                "leaf_subset_sha256",
                "pair_index_sha256",
                "ordered_node_ids_sha256",
            },
            "matrix topology row",
        )
        if row["tree_id"] != tree_id or any(not _is_sha(row[key]) for key in (
            "topology_sha256", "nodes_sha256", "leaf_subset_sha256", "pair_index_sha256", "ordered_node_ids_sha256"
        )):
            raise ValueError("matrix topology/order binding drift")
        counts = (row["n_nodes"], row["n_internal"], row["n_leaves"])
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
            raise ValueError("matrix topology counts must be positive integers")
        if counts[1] + counts[2] != counts[0]:
            raise ValueError("matrix topology internal+leaf count mismatch")
        for index, value in enumerate(counts):
            totals[index] += value
        projection.append({key: row[key] for key in (
            "tree_id", "topology_sha256", "nodes_sha256", "leaf_subset_sha256", "pair_index_sha256"
        )})
    if tuple(totals) != (N_NODES, N_INTERNAL, N_LEAVES):
        raise ValueError("matrix topology totals drift")
    if raw["topology_scope_sha256"] != canonical_sha256(projection):
        raise ValueError("matrix topology-scope hash drift")
    if raw["matrix_order_scope_sha256"] != canonical_sha256(scope):
        raise ValueError("matrix topology/order-scope hash drift")
    if not isinstance(checkpoint_payload, Mapping) or raw["topology_scope_sha256"] != checkpoint_payload["topology_scope_sha256"]:
        raise ValueError("matrix plan is not bound to checkpoint topology")
    cohorts = raw["cohorts"]
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(COHORTS):
        raise ValueError("matrix cohort grid/order drift")
    seen: set[str] = set()
    for cohort in COHORTS:
        row = cohorts[cohort]
        if not isinstance(row, Mapping):
            raise ValueError("matrix cohort row must be a mapping")
        _keys(row, {"direct_job_logical_id", "output_root", "tree_ids", "plan_sha256"}, "matrix cohort row")
        root = _path(row["output_root"], "matrix output_root")
        expected_root = stage / "outputs" / "allnode_v1" / "direct" / cohort
        core = {
            "cohort_id": cohort,
            "direct_job_logical_id": f"direct_{cohort}",
            "output_root": str(root),
            "tree_ids": list(TREE_IDS),
            "topology_scope_sha256": raw["topology_scope_sha256"],
            "matrix_order_scope_sha256": raw["matrix_order_scope_sha256"],
        }
        if root != expected_root or row["direct_job_logical_id"] != f"direct_{cohort}" or row["tree_ids"] != list(TREE_IDS) or row["plan_sha256"] != canonical_sha256(core):
            raise ValueError(f"matrix cohort {cohort} plan drift")
        if str(root) in seen:
            raise ValueError("matrix output roots cannot be reused")
        seen.add(str(root))
    return raw


def _validate_inference_plan(raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("inference_plan must be a mapping")
    _keys(raw, {"schema", "valid", "cohort_ids", "n_nodes_per_cohort", "cohorts"}, "inference_plan")
    if raw["schema"] != INFERENCE_PLAN_SCHEMA or raw["valid"] is not True or raw["cohort_ids"] != list(COHORTS) or raw["n_nodes_per_cohort"] != N_NODES:
        raise ValueError("inference plan scope drift")
    cohorts = raw["cohorts"]
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(COHORTS):
        raise ValueError("inference cohort grid/order drift")
    for cohort in COHORTS:
        row = cohorts[cohort]
        if not isinstance(row, Mapping):
            raise ValueError("inference cohort row must be a mapping")
        _keys(row, {"array", "n_tasks", "protocols", "responses_per_node", "expected_response_rows", "device", "torch_dtype", "execution_overrides"}, "inference cohort row")
        if (
            row["array"] != {"start": 0, "end": 640, "max_concurrent": 4}
            or row["n_tasks"] != N_NODES
            or row["protocols"] != list(PROTOCOLS[cohort])
            or row["responses_per_node"] != RESPONSES_PER_NODE[cohort]
            or row["expected_response_rows"] != N_NODES * RESPONSES_PER_NODE[cohort]
            or row["device"] != "cuda"
            or row["torch_dtype"] != "auto"
            or row["execution_overrides"] != {}
        ):
            raise ValueError(f"inference count/protocol contract drift for {cohort}")


def _validate_execution_contract(
    raw: object,
    *,
    stage: PurePosixPath,
    deployment: PurePosixPath,
    checkpoint_payload: object,
    job_configs: object,
    code_manifest: Pin,
    source_resolution: Pin,
    downstream_code_manifest: Pin,
    test_runtime: Mapping[str, str],
) -> None:
    """Bind the parsed runtime plans and requested statistics into the audit.

    The large per-cohort runtime-plan files are independently pinned.  This
    compact receipt records the critical fields parsed from those exact bytes,
    so a dry-run cannot pass merely because twelve arbitrary JSON files exist.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("execution_contract must be a mapping")
    _keys(raw, {"receipt", "payload", "payload_sha256"}, "execution_contract")
    receipt = _pin(raw["receipt"], "execution contract receipt")
    expected_receipt = deployment / "runtime_receipts" / "ALLNODE_V1_EXECUTION_CONTRACT.json"
    if _path(receipt.path, "execution contract receipt path") != expected_receipt:
        raise ValueError("execution contract receipt path drift")
    payload = raw["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("execution contract payload must be a mapping")
    payload_sha = canonical_sha256(dict(payload))
    if raw["payload_sha256"] != payload_sha or receipt.sha256 != payload_sha:
        raise ValueError("execution contract receipt/payload hash drift")
    _keys(
        payload,
        {
            "schema",
            "valid",
            "population",
            "sensitivity_population",
            "internal_node_treatment",
            "exclude_untrained_root",
            "cohort_ids",
            "tree_ids",
            "n_nodes_per_cohort",
            "n_internal_per_cohort",
            "n_leaves_per_cohort",
            "checkpoint_identity",
            "topology_scope_sha256",
            "source_scope_sha256",
            "runtime_provenance",
            "analysis",
            "jobs",
            "execution_overrides",
            "no_automatic_recovery",
        },
        "execution contract payload",
    )
    if (
        payload["schema"] != EXECUTION_CONTRACT_SCHEMA
        or payload["valid"] is not True
        or payload["population"] != "all_trained_nodes"
        or payload["sensitivity_population"] != "trained_leaves"
        or payload["internal_node_treatment"] != "observed_ancestors"
        or payload["exclude_untrained_root"] is not True
        or payload["cohort_ids"] != list(COHORTS)
        or payload["tree_ids"] != list(TREE_IDS)
        or (
            payload["n_nodes_per_cohort"],
            payload["n_internal_per_cohort"],
            payload["n_leaves_per_cohort"],
        )
        != (N_NODES, N_INTERNAL, N_LEAVES)
        or payload["execution_overrides"] != {}
        or payload["no_automatic_recovery"] is not True
    ):
        raise ValueError("execution contract production scope drift")
    if not _is_sha(payload["topology_scope_sha256"]) or not _is_sha(payload["source_scope_sha256"]):
        raise ValueError("execution contract scope hash drift")
    if not isinstance(checkpoint_payload, Mapping) or payload["topology_scope_sha256"] != checkpoint_payload["topology_scope_sha256"]:
        raise ValueError("execution contract topology differs from checkpoint production")

    provenance = payload["runtime_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("execution contract runtime provenance must be a mapping")
    _keys(
        provenance,
        {
            "production_interpreter", "python_version", "python_implementation",
            "source_resolution", "allnode_code_manifest",
            "downstream_code_manifest", "reattest_before_after_each_job",
        },
        "execution contract runtime provenance",
    )
    if (
        provenance["production_interpreter"] != PRODUCTION_INTERPRETER
        or provenance["python_version"] != test_runtime["python_version"]
        or provenance["python_implementation"] != test_runtime["python_implementation"]
        or _pin(provenance["source_resolution"], "runtime source_resolution")
        != source_resolution
        or _pin(provenance["allnode_code_manifest"], "runtime allnode code manifest")
        != code_manifest
        or _pin(
            provenance["downstream_code_manifest"], "runtime downstream code manifest"
        )
        != downstream_code_manifest
        or provenance["reattest_before_after_each_job"] is not True
    ):
        raise ValueError("execution contract runtime provenance binding drift")

    identity = payload["checkpoint_identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("execution contract checkpoint identity must be a mapping")
    _keys(
        identity,
        {"scheme", "index", "n_checkpoints", "reattest_at_runtime"},
        "execution contract checkpoint identity",
    )
    index_pin = _pin(identity["index"], "checkpoint identity index")
    expected_index = deployment / "runtime_receipts" / "ALLNODE_V1_CHECKPOINT_IDENTITIES.json"
    if (
        identity["scheme"] != "frozen_fs_full_small_fixed5x1m_large_v1"
        or _path(index_pin.path, "checkpoint identity index path") != expected_index
        or identity["n_checkpoints"] != len(COHORTS) * N_NODES
        or identity["reattest_at_runtime"] is not True
    ):
        raise ValueError("execution contract checkpoint identity drift")

    if payload["analysis"] != _expected_analysis_contract():
        raise ValueError("execution contract semantic/regression/DL plan drift")
    if not isinstance(job_configs, Mapping):
        raise ValueError("execution contract job config pins must be a mapping")
    jobs = payload["jobs"]
    expected_ids = _expected_job_ids()
    if not isinstance(jobs, Mapping) or list(jobs) != expected_ids:
        raise ValueError("execution contract parsed job set/order drift")
    for logical_id in expected_ids:
        row = jobs[logical_id]
        if not isinstance(row, Mapping):
            raise ValueError("execution contract job row must be a mapping")
        _validate_execution_job(
            logical_id,
            row,
            stage=stage,
            deployment=deployment,
            config_pin=_pin(job_configs[logical_id], f"job_configs.{logical_id}"),
            checkpoint_payload=checkpoint_payload,
        )


def _validate_execution_job(
    logical_id: str,
    row: Mapping[str, Any],
    *,
    stage: PurePosixPath,
    deployment: PurePosixPath,
    config_pin: Pin,
    checkpoint_payload: Mapping[str, Any],
) -> None:
    _keys(
        row,
        {
            "kind",
            "cohort_key",
            "config",
            "runtime_plan",
            "runtime_plan_schema",
            "output_root",
            "n_trees",
            "n_nodes",
            "n_internal",
            "n_leaves",
            "topology_scope_sha256",
            "checkpoint_scope_sha256",
            "node_covariate_scope_sha256",
            "metrics",
            "expected_response_rows",
            "semantic_endpoints",
            "execution_overrides",
        },
        f"execution contract job {logical_id}",
    )
    prefix, cohort = logical_id.split("_", 1)
    kind = {"direct": "direct_allnode_cpu", "infer": "inference_allnode_gpu", "semantic": "semantic_allnode_gpu"}[prefix]
    plan_name = {"direct": "direct_plan.json", "infer": "inference_plan.json", "semantic": "semantic_plan.json"}[prefix]
    plan_schema = {
        "direct": "weighttraits.all_node_direct_cohort_plan.v2",
        "infer": "weighttraits.all_node_inference_cohort_plan.v1",
        "semantic": "weighttraits.all_node_semantic_cohort_plan.v1",
    }[prefix]
    output_prefix = "inference" if prefix == "infer" else prefix
    expected_plan = deployment / "runtime_inputs" / cohort / plan_name
    expected_output = stage / "outputs" / "allnode_v1" / output_prefix / cohort
    parsed_config = _pin(row["config"], f"execution job {logical_id} config")
    runtime_plan = _pin(row["runtime_plan"], f"execution job {logical_id} runtime plan")
    endpoints = _semantic_endpoint_rows(cohort) if prefix == "semantic" else []
    expected_responses = N_NODES * RESPONSES_PER_NODE[cohort] if prefix == "infer" else None
    metrics = ["l2", "cosine", "correlation"] if prefix == "direct" else []
    checkpoint_scope = checkpoint_payload["cohorts"][cohort]["manifest_scope_sha256"]
    if (
        row["kind"] != kind
        or row["cohort_key"] != cohort
        or parsed_config != config_pin
        or _path(runtime_plan.path, "runtime plan path") != expected_plan
        or row["runtime_plan_schema"] != plan_schema
        or _path(row["output_root"], "execution job output root") != expected_output
        or (row["n_trees"], row["n_nodes"], row["n_internal"], row["n_leaves"])
        != (50, N_NODES, N_INTERNAL, N_LEAVES)
        or row["topology_scope_sha256"] != checkpoint_payload["topology_scope_sha256"]
        or row["checkpoint_scope_sha256"] != checkpoint_scope
        or not _is_sha(row["node_covariate_scope_sha256"])
        or row["metrics"] != metrics
        or row["expected_response_rows"] != expected_responses
        or row["semantic_endpoints"] != endpoints
        or row["execution_overrides"] != {}
    ):
        raise ValueError(f"execution contract parsed job drift for {logical_id}")


def _expected_job_ids() -> list[str]:
    values = [f"direct_{cohort}" for cohort in COHORTS]
    for cohort in COHORTS:
        values.extend((f"infer_{cohort}", f"semantic_{cohort}"))
    return sorted(values)


def _semantic_endpoint_rows(cohort: str) -> list[dict[str, Any]]:
    return [
        {
            "endpoint_id": probe,
            "protocol_id": protocol,
            "probe_id": probe,
            "n_prompts": n_prompts,
            "samples_per_prompt": samples,
            "n_paired_coordinates": n_prompts * samples,
        }
        for protocol, probe, n_prompts, samples in SEMANTIC_ENDPOINTS[cohort]
    ]


def _expected_analysis_contract() -> dict[str, Any]:
    return {
        "semantic_encoder": {
            "model": MINILM_MODEL,
            "revision": MINILM_REVISION,
            "runtime": MINILM_RUNTIME,
            "snapshot_sha256": MINILM_SNAPSHOT_SHA256,
        },
        "pairing": "exact_protocol_probe_prompt_id_sample_id",
        "semantic_distance": "mean_paired_response_embedding_cosine_distance",
        "direct_metric": "cosine",
        "outcome": "semantic_similarity=1-semantic_distance",
        "primary_population": "all_trained_nodes",
        "sensitivity_population": "trained_leaves_derived_from_full_matrices",
        "strata": [
            "all_trained_node_pairs",
            "ancestor_descendant",
            "parent_child",
            "collateral",
            "pairs_involving_internal",
            "internal_internal",
            "internal_leaf",
            "trained_leaf_pairs",
        ],
        "regressions": [
            {
                "id": "legacy_practical_all_nodes_hc3",
                "stratum": "all_trained_node_pairs",
                "formula": "semantic_sim ~ cosine_dist + C(run)",
                "covariance": "HC3_pair_row_legacy_comparability",
                "compatibility": True,
            },
            {
                "id": "legacy_controlled_all_nodes_hc3",
                "stratum": "all_trained_node_pairs",
                "formula": "semantic_sim ~ depth_i + depth_j + lca_depth + task_jaccard + cosine_dist + C(run)",
                "covariance": "HC3_pair_row_legacy_comparability",
                "compatibility": True,
            },
            {
                "id": "legacy_practical_all_nodes",
                "stratum": "all_trained_node_pairs",
                "formula": "semantic_sim ~ cosine_dist + C(run)",
                "covariance": "CR1_tree_clustered_sandwich",
                "compatibility": False,
            },
            {
                "id": "legacy_controlled_all_nodes",
                "stratum": "all_trained_node_pairs",
                "formula": "semantic_sim ~ depth_i + depth_j + lca_depth + task_jaccard + cosine_dist + C(run)",
                "covariance": "CR1_tree_clustered_sandwich",
                "compatibility": False,
            },
            {
                "id": "internal_ancestry_adjusted_all_nodes",
                "stratum": "all_trained_node_pairs",
                "formula": "semantic_similarity=1-paired_semantic_cosine_distance ~ direct_weight_cosine_distance + pair_has_internal + ancestor_descendant + parent_child + C(run)",
                "covariance": "CR1_tree_clustered_sandwich",
                "compatibility": False,
            },
            {
                "id": "legacy_practical_trained_leaf_sensitivity_hc3",
                "stratum": "trained_leaf_pairs",
                "formula": "semantic_sim ~ cosine_dist + C(run)",
                "covariance": "HC3_pair_row_legacy_comparability",
                "compatibility": True,
            },
            {
                "id": "legacy_controlled_trained_leaf_sensitivity_hc3",
                "stratum": "trained_leaf_pairs",
                "formula": "semantic_sim ~ depth_i + depth_j + lca_depth + task_jaccard + cosine_dist + C(run)",
                "covariance": "HC3_pair_row_legacy_comparability",
                "compatibility": True,
            },
            {
                "id": "legacy_practical_trained_leaf_sensitivity",
                "stratum": "trained_leaf_pairs",
                "formula": "semantic_sim ~ cosine_dist + C(run)",
                "covariance": "CR1_tree_clustered_sandwich",
                "compatibility": False,
            },
            {
                "id": "legacy_controlled_trained_leaf_sensitivity",
                "stratum": "trained_leaf_pairs",
                "formula": "semantic_sim ~ depth_i + depth_j + lca_depth + task_jaccard + cosine_dist + C(run)",
                "covariance": "CR1_tree_clustered_sandwich",
                "compatibility": False,
            },
        ],
        "meta_analysis": [
            {
                "id": "legacy_pair_count_der_simonian_laird",
                "effect": "per_tree_pearson_r_fisher_z",
                "within_tree_variance": "1/(n_pairs-3)",
                "compatibility": True,
            },
            {
                "id": "robust_node_jackknife_der_simonian_laird",
                "effect": "per_tree_pearson_r_fisher_z",
                "within_tree_variance": "delete_one_trained_node_incident_dyads",
                "compatibility": False,
            },
            {
                "id": "equal_tree_fisher_z",
                "effect": "per_tree_pearson_r_fisher_z",
                "within_tree_variance": "empirical_across_tree",
                "compatibility": False,
            },
        ],
        "inference_unit": "tree",
        "neighbor_joining": False,
        "internal_nodes_as_tips": False,
        "streaming_unit": "one_tree_one_endpoint",
    }


def expected_analysis_contract() -> dict[str, Any]:
    """Return a detached copy of the exact audited semantic/statistical plan."""

    return json.loads(json.dumps(_expected_analysis_contract(), sort_keys=True))


def semantic_endpoint_rows(cohort: str) -> list[dict[str, Any]]:
    """Return the exact ordered paired-response endpoints for one cohort."""

    if cohort not in SEMANTIC_ENDPOINTS:
        raise ValueError(f"unknown semantic cohort {cohort!r}")
    return json.loads(json.dumps(_semantic_endpoint_rows(cohort), sort_keys=True))


def _validate_test_attestation(raw: object, code: Pin) -> Mapping[str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError("test_attestation must be a mapping")
    _keys(raw, {
        "schema", "valid", "runner", "command", "tests_run", "failures",
        "errors", "skipped", "code_manifest_sha256", "interpreter",
        "python_version", "python_implementation",
    }, "test_attestation")
    if (
        raw["schema"] != TEST_SCHEMA
        or raw["valid"] is not True
        or raw["runner"] != "unittest"
        or raw["interpreter"] != PRODUCTION_INTERPRETER
        or raw["command"] != [PRODUCTION_INTERPRETER, "-m", "unittest", "discover", "-s", "tests", "-v"]
        or not isinstance(raw["python_version"], str)
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", raw["python_version"])
        or raw["python_implementation"] != "CPython"
        or isinstance(raw["tests_run"], bool)
        or not isinstance(raw["tests_run"], int)
        or raw["tests_run"] <= 0
        or (raw["failures"], raw["errors"], raw["skipped"]) != (0, 0, 0)
        or raw["code_manifest_sha256"] != code.sha256
    ):
        raise ValueError("test attestation is not an exact passing suite result")
    return {
        "python_version": raw["python_version"],
        "python_implementation": raw["python_implementation"],
    }


def _validate_state_absence(raw: object) -> None:
    expected = {"outputs_exist": False, "intent_exists": False, "jobs_receipt_exists": False}
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        raise ValueError("outputs/intent/jobs-receipt absence flags must all be false")


def _validate_config_gate_binding(raw: object, gate: GateContract) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("job_configs must be a mapping")
    by_id = {job.logical_id: job for job in gate.jobs}
    if set(raw) != set(by_id):
        raise ValueError("job config pins differ from GateContract jobs")
    for logical_id, job in by_id.items():
        pin = _pin(raw[logical_id], f"job_configs.{logical_id}")
        if pin.path != job.config_path:
            raise ValueError(f"job config path differs from GateContract for {logical_id}")


def _pin_grid_dict(raw: Mapping[str, Mapping[str, Pin]]) -> dict[str, dict[str, dict[str, str]]]:
    return {
        cohort: {protocol: pin.to_dict() for protocol, pin in rows.items()}
        for cohort, rows in raw.items()
    }


def _pin(raw: object, label: str) -> Pin:
    if isinstance(raw, Pin):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a path+sha256 mapping")
    _keys(raw, {"path", "sha256"}, label)
    return Pin(raw["path"], raw["sha256"])


def _path(raw: object, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ValueError(f"{label} must be a normalized path string")
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts or str(path) != raw:
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    return path


def _under(raw: str, parent: PurePosixPath, label: str) -> None:
    path = _path(raw, label)
    if parent not in path.parents:
        raise ValueError(f"{label} must be strictly inside {parent}")


def _is_sha(raw: object) -> bool:
    return isinstance(raw, str) and len(raw) == 64 and all(c in "0123456789abcdef" for c in raw)


def _keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(raw)
    if observed != expected:
        raise ValueError(f"{label} keys differ: missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}")


__all__ = [
    "BUNDLE_SCHEMA",
    "CORE_SCHEMA",
    "DryRunAuditBundle",
    "EXECUTION_CONTRACT_SCHEMA",
    "INFERENCE_PLAN_SCHEMA",
    "MATRIX_PLAN_SCHEMA",
    "PROTOCOLS",
    "RESPONSES_PER_NODE",
    "TEST_SCHEMA",
    "TREE_IDS",
    "build_dry_run_audit_bundle",
    "expected_analysis_contract",
    "semantic_endpoint_rows",
    "serialize_audit_core",
    "validate_audit_core",
    "validate_dry_run_audit_bundle",
]
