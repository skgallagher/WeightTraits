"""Fail-closed EOS sensitivity and leaf-depth summaries for behavior responses.

The inference pipeline writes one complete, receipt-backed response grid per trained leaf.  This
module validates those grids against the frozen behavior protocol and prompt artifacts, validates
the exact trained-leaf set and depths against SHA-pinned truth manifests, and then computes the
empty-response diagnostics used by the reproducibility workbook.

Primary counts retain every response.  The paired sensitivity count includes a prompt/draw
identity for a leaf pair only when both raw decoded responses satisfy ``text.strip() != ""``.
No response is normalized, reordered by file position, or silently truncated.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from weighttraits.behavior.contracts import (
    BehaviorProtocolRegistry,
    RenderedBehaviorPrompt,
    load_behavior_protocol_registry,
)
from weighttraits.behavior.responses import (
    BehaviorResponse,
    audit_response_grid,
    expected_response_coordinates,
    load_behavior_responses,
    load_rendered_behavior_prompts,
)
from weighttraits.behavior.probe_inference import validate_inference_receipt_schema
from weighttraits.manifests.reference import leaf_ids, load_manifest


EOS_DEPTH_INPUT_SCHEMA = "weighttraits.behavior.eos_depth_input.v1"
EOS_DEPTH_SCHEMA = "weighttraits.behavior.eos_depth.v1"
EOS_DEPTH_RECEIPT_SCHEMA = "weighttraits.behavior.eos_depth_receipt.v1"
EMPTY_DEFINITION = 'text.strip() == ""'
CONFIDENCE_LEVEL = 0.95
_SHA256 = re.compile(r"[0-9a-f]{64}")

PER_LEAF_COLUMNS = (
    "panel",
    "probe",
    "tree",
    "node",
    "depth",
    "task_family",
    "n_outputs",
    "n_empty",
    "empty_percent",
)
PER_DEPTH_COLUMNS = (
    "panel",
    "probe",
    "depth",
    "n_trees",
    "n_leaves",
    "n_outputs",
    "n_empty",
    "pooled_empty_percent",
    "mean_tree_empty_percent",
    "t95_ci_low_tree_empty_percent",
    "t95_ci_high_tree_empty_percent",
)
SENSITIVITY_PAIR_COLUMNS = (
    "panel",
    "probe",
    "tree",
    "left_node",
    "right_node",
    "n_primary",
    "n_paired_nonempty",
    "n_excluded",
    "paired_nonempty_percent",
)
SENSITIVITY_SUMMARY_COLUMNS = (
    "panel",
    "probe",
    "n_trees",
    "n_leaf_pairs",
    "n_primary",
    "n_paired_nonempty",
    "n_excluded",
    "paired_nonempty_percent",
)


def build_eos_depth_analysis(config_path: str | Path) -> dict[str, Any]:
    """Validate all pinned inputs and build EOS/depth rows.

    The input JSON is intentionally an explicit inventory.  It contains no directory globs and
    therefore cannot silently omit or discover a leaf after the analysis contract was frozen.
    """

    config = Path(config_path).resolve(strict=True)
    raw = _json_object(config)
    _exact_keys(
        raw,
        {
            "schema",
            "panel_id",
            "registry",
            "prompt_artifacts",
            "tree_manifests",
            "response_receipts",
        },
        context="EOS/depth input",
    )
    if raw["schema"] != EOS_DEPTH_INPUT_SCHEMA:
        raise ValueError(
            f"EOS/depth input schema must be {EOS_DEPTH_INPUT_SCHEMA!r}, got {raw['schema']!r}"
        )
    panel_id = _nonempty(raw["panel_id"], "panel_id")
    base = config.parent

    registry_path, registry_sha = _load_pinned_file(
        raw["registry"], base=base, context="registry"
    )
    registry = load_behavior_protocol_registry(registry_path)
    if registry.sha256 != registry_sha:
        raise ValueError("behavior registry loader hash disagrees with pinned registry hash")

    prompt_specs = _mapping(raw["prompt_artifacts"], "prompt_artifacts")
    if not prompt_specs:
        raise ValueError("prompt_artifacts must be non-empty")
    prompts_by_protocol: dict[str, list[RenderedBehaviorPrompt]] = {}
    prompt_inputs: dict[str, dict[str, Any]] = {}
    used_probe_ids: set[str] = set()
    prompt_tasks: set[str] = set()
    for protocol_id, spec in prompt_specs.items():
        protocol = registry.protocol(protocol_id)
        overlap = used_probe_ids & set(protocol.probe_ids)
        if overlap:
            raise ValueError(
                "probe IDs must be unique across one EOS/depth panel; "
                f"duplicates={sorted(overlap)}"
            )
        used_probe_ids.update(protocol.probe_ids)
        prompt_path, prompt_sha = _load_pinned_file(
            spec,
            base=base,
            context=f"prompt_artifacts.{protocol_id}",
        )
        prompts = load_rendered_behavior_prompts(prompt_path)
        prompts_by_protocol[protocol_id] = prompts
        prompt_inputs[protocol_id] = {
            "path": str(prompt_path),
            "sha256": prompt_sha,
        }
        for prompt in prompts:
            task = prompt.metadata.get("model_task")
            if not isinstance(task, str) or not task:
                raise ValueError(f"prompt {prompt.prompt_id!r} lacks explicit model_task")
            prompt_tasks.add(task)
    expected_coordinates = expected_response_coordinates(registry, prompts_by_protocol)
    if len(prompt_tasks) != 1:
        raise ValueError(
            f"all prompt artifacts in one panel must share one model_task, got {sorted(prompt_tasks)}"
        )
    model_task = next(iter(prompt_tasks))

    tree_rows = _list(raw["tree_manifests"], "tree_manifests")
    if not tree_rows:
        raise ValueError("tree_manifests must be non-empty")
    tree_details: dict[str, dict[str, Any]] = {}
    tree_order: list[str] = []
    tree_inputs: list[dict[str, Any]] = []
    for index, spec_value in enumerate(tree_rows):
        spec = _mapping(spec_value, f"tree_manifests[{index}]")
        _exact_keys(spec, {"tree_id", "path", "sha256"}, context=f"tree_manifests[{index}]")
        tree_id = _nonempty(spec["tree_id"], f"tree_manifests[{index}].tree_id")
        if tree_id in tree_details:
            raise ValueError(f"duplicate tree manifest declaration: {tree_id!r}")
        path, digest = _load_pinned_file(
            {"path": spec["path"], "sha256": spec["sha256"]},
            base=base,
            context=f"tree_manifests[{index}]",
        )
        details = _load_tree_details(path, tree_id=tree_id)
        tree_details[tree_id] = details
        tree_order.append(tree_id)
        tree_inputs.append({"tree_id": tree_id, "path": str(path), "sha256": digest})

    receipt_specs = _list(raw["response_receipts"], "response_receipts")
    if not receipt_specs:
        raise ValueError("response_receipts must be non-empty")
    responses_by_tree: dict[str, dict[str, tuple[BehaviorResponse, ...]]] = defaultdict(dict)
    receipt_inputs: list[dict[str, Any]] = []
    response_inputs: list[dict[str, Any]] = []
    panel_binding: dict[str, str] | None = None
    tree_path_by_id = {row["tree_id"]: Path(row["path"]) for row in tree_inputs}
    tree_sha_by_id = {row["tree_id"]: row["sha256"] for row in tree_inputs}
    for index, spec_value in enumerate(receipt_specs):
        receipt_path, receipt_sha = _load_pinned_file(
            spec_value,
            base=base,
            context=f"response_receipts[{index}]",
        )
        loaded = _load_validated_leaf_receipt(
            receipt_path,
            registry=registry,
            prompts_by_protocol=prompts_by_protocol,
            prompt_inputs=prompt_inputs,
            expected_coordinates=expected_coordinates,
            expected_model_task=model_task,
            tree_path_by_id=tree_path_by_id,
            tree_sha_by_id=tree_sha_by_id,
        )
        tree_id = loaded["tree_id"]
        model_id = loaded["model_id"]
        if model_id in responses_by_tree[tree_id]:
            raise ValueError(f"duplicate response receipt for {tree_id}/{model_id}")
        responses_by_tree[tree_id][model_id] = tuple(loaded["responses"])
        binding = {
            "cohort_id": loaded["cohort_id"],
            "base_model_id": loaded["base_model_id"],
            "base_model_revision": loaded["base_model_revision"],
            "model_task": loaded["model_task"],
        }
        if panel_binding is None:
            panel_binding = binding
        elif binding != panel_binding:
            raise ValueError(
                "all leaf receipts in one panel must share cohort/base-model/task binding: "
                f"expected {panel_binding}, got {binding}"
            )
        receipt_inputs.append({"path": str(receipt_path), "sha256": receipt_sha})
        response_inputs.append(
            {
                "tree_id": tree_id,
                "model_id": model_id,
                "path": loaded["response_path"],
                "sha256": loaded["response_sha256"],
                "request_sha256": loaded["request_sha256"],
            }
        )
    assert panel_binding is not None

    observed_trees = set(responses_by_tree)
    expected_trees = set(tree_details)
    if observed_trees != expected_trees:
        raise ValueError(
            "response-receipt/tree-manifest sets differ: "
            f"missing={sorted(expected_trees - observed_trees)}, "
            f"extra={sorted(observed_trees - expected_trees)}"
        )
    for tree_id in tree_order:
        expected_leaves = set(tree_details[tree_id]["leaf_order"])
        observed_leaves = set(responses_by_tree[tree_id])
        if observed_leaves != expected_leaves:
            raise ValueError(
                f"{tree_id} response/model set does not equal truth-manifest leaf set: "
                f"missing={sorted(expected_leaves - observed_leaves)}, "
                f"extra={sorted(observed_leaves - expected_leaves)}"
            )

    identity_panels = _identity_panel_receipts(
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
    )
    per_leaf_rows, per_depth_rows = _build_depth_rows(
        panel_id=panel_id,
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        tree_order=tree_order,
        tree_details=tree_details,
        responses_by_tree=responses_by_tree,
    )
    sensitivity_rows, sensitivity_summary = _build_sensitivity_rows(
        panel_id=panel_id,
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        tree_order=tree_order,
        tree_details=tree_details,
        responses_by_tree=responses_by_tree,
    )
    payload: dict[str, Any] = {
        "schema": EOS_DEPTH_SCHEMA,
        "valid": True,
        "panel": panel_id,
        **panel_binding,
        "empty_definition": EMPTY_DEFINITION,
        "primary_policy": "preserve every exact protocol-declared prompt/draw response",
        "sensitivity_policy": (
            "for each leaf pair, exclude a prompt/draw identity when either raw decoded "
            "response satisfies text.strip() == \"\""
        ),
        "confidence_interval": {
            "level": CONFIDENCE_LEVEL,
            "method": (
                "two-sided Student t interval over per-tree, depth-specific empty percentages"
            ),
            "single_tree_policy": "null bounds because a t interval has zero degrees of freedom",
        },
        "inputs": {
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "registry": {"path": str(registry_path), "sha256": registry_sha},
            "prompt_artifacts": prompt_inputs,
            "tree_manifests": tree_inputs,
            "response_receipts": receipt_inputs,
            "responses": response_inputs,
        },
        "identity_panels": identity_panels,
        "counts": {
            "n_protocols": len(prompts_by_protocol),
            "n_probes": len(used_probe_ids),
            "n_trees": len(tree_order),
            "n_leaves": sum(len(tree_details[tree]["leaf_order"]) for tree in tree_order),
            "n_response_receipts": len(receipt_inputs),
            "n_response_rows": sum(row["n_expected"] for row in identity_panels.values())
            * sum(len(tree_details[tree]["leaf_order"]) for tree in tree_order),
            "n_per_leaf_rows": len(per_leaf_rows),
            "n_per_depth_rows": len(per_depth_rows),
            "n_sensitivity_pair_rows": len(sensitivity_rows),
            "n_sensitivity_summary_rows": len(sensitivity_summary),
        },
        "per_leaf_rows": per_leaf_rows,
        "per_depth_rows": per_depth_rows,
        "sensitivity_pair_rows": sensitivity_rows,
        "sensitivity_summary_rows": sensitivity_summary,
    }
    return payload


def write_eos_depth_outputs(
    payload: Mapping[str, Any],
    *,
    json_path: str | Path,
    per_leaf_csv_path: str | Path,
    per_depth_csv_path: str | Path,
    sensitivity_csv_path: str | Path,
    sensitivity_summary_csv_path: str | Path,
    receipt_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically write JSON/CSV artifacts, then write the completion receipt last."""

    if payload.get("schema") != EOS_DEPTH_SCHEMA or payload.get("valid") is not True:
        raise ValueError("refusing to write an invalid EOS/depth payload")
    _revalidate_payload_input_hashes(payload)
    outputs = {
        "analysis_json": Path(json_path).resolve(),
        "per_leaf_csv": Path(per_leaf_csv_path).resolve(),
        "per_depth_csv": Path(per_depth_csv_path).resolve(),
        "sensitivity_csv": Path(sensitivity_csv_path).resolve(),
        "sensitivity_summary_csv": Path(sensitivity_summary_csv_path).resolve(),
    }
    receipt = Path(receipt_path).resolve()
    all_outputs = [*outputs.values(), receipt]
    if len(set(all_outputs)) != len(all_outputs):
        raise ValueError("EOS/depth output paths must all be distinct")
    input_paths = _payload_input_paths(payload)
    collisions = sorted(str(path) for path in set(all_outputs) & input_paths)
    if collisions:
        raise ValueError(f"outputs must not overwrite pinned inputs: {collisions}")
    existing = [str(path) for path in all_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    json_bytes = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    per_leaf_bytes = _csv_bytes(payload["per_leaf_rows"], PER_LEAF_COLUMNS)
    per_depth_bytes = _csv_bytes(payload["per_depth_rows"], PER_DEPTH_COLUMNS)
    sensitivity_bytes = _csv_bytes(
        payload["sensitivity_pair_rows"], SENSITIVITY_PAIR_COLUMNS
    )
    sensitivity_summary_bytes = _csv_bytes(
        payload["sensitivity_summary_rows"], SENSITIVITY_SUMMARY_COLUMNS
    )
    contents = {
        outputs["analysis_json"]: json_bytes,
        outputs["per_leaf_csv"]: per_leaf_bytes,
        outputs["per_depth_csv"]: per_depth_bytes,
        outputs["sensitivity_csv"]: sensitivity_bytes,
        outputs["sensitivity_summary_csv"]: sensitivity_summary_bytes,
    }
    for path, content in contents.items():
        _write_bytes_atomic(path, content, overwrite=overwrite)

    receipt_payload = {
        "schema": EOS_DEPTH_RECEIPT_SCHEMA,
        "valid": True,
        "panel": payload["panel"],
        "empty_definition": EMPTY_DEFINITION,
        "input_config": dict(payload["inputs"]["config"]),
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "n_rows": _output_row_count(name, payload),
            }
            for name, path in outputs.items()
        },
        "counts": dict(payload["counts"]),
    }
    _write_bytes_atomic(
        receipt,
        (json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        overwrite=overwrite,
    )
    return receipt_payload


def _load_validated_leaf_receipt(
    receipt_path: Path,
    *,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    prompt_inputs: Mapping[str, Mapping[str, Any]],
    expected_coordinates: Sequence[tuple[str, RenderedBehaviorPrompt, int]],
    expected_model_task: str,
    tree_path_by_id: Mapping[str, Path],
    tree_sha_by_id: Mapping[str, str],
) -> dict[str, Any]:
    payload = _json_object(receipt_path)
    required = {
        "schema_version",
        "valid",
        "status",
        "request_sha256",
        "request",
        "provenance",
        "responses",
        "responses_sha256",
        "audit",
        "prompt_artifacts",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise ValueError(
            f"leaf receipt schema mismatch at {receipt_path}: missing={missing}, unknown={unknown}"
        )
    validate_inference_receipt_schema(
        payload,
        context=f"leaf receipt {receipt_path}",
    )
    if payload["valid"] is not True or payload["status"] != "completed":
        raise ValueError(f"leaf receipt is not valid/completed: {receipt_path}")
    request = _mapping(payload["request"], f"{receipt_path}.request")
    request_sha = _sha256(payload["request_sha256"], f"{receipt_path}.request_sha256")
    if canonical_json_sha256(request) != request_sha:
        raise ValueError(f"leaf receipt request SHA256 mismatch: {receipt_path}")
    provenance = _mapping(payload["provenance"], f"{receipt_path}.provenance")
    if provenance.get("request_sha256") != request_sha:
        raise ValueError(f"leaf receipt provenance request hash mismatch: {receipt_path}")

    tree_id = _nonempty(request.get("tree_id"), f"{receipt_path}.request.tree_id")
    if tree_id not in tree_path_by_id:
        raise ValueError(f"leaf receipt names undeclared tree {tree_id!r}: {receipt_path}")
    model_id = _nonempty(request.get("model_id"), f"{receipt_path}.request.model_id")
    model_task = _nonempty(request.get("model_task"), f"{receipt_path}.request.model_task")
    if model_task != expected_model_task:
        raise ValueError(
            f"{tree_id}/{model_id} model_task mismatch: expected {expected_model_task!r}, "
            f"got {model_task!r}"
        )
    request_registry = _declared_path(
        request.get("registry"), receipt_path.parent, f"{receipt_path}.request.registry"
    )
    if request_registry != registry.path or request.get("registry_sha256") != registry.sha256:
        raise ValueError(f"{tree_id}/{model_id} behavior registry provenance drift")
    truth_path = _declared_path(
        request.get("truth_manifest"),
        receipt_path.parent,
        f"{receipt_path}.request.truth_manifest",
    )
    if truth_path != tree_path_by_id[tree_id]:
        raise ValueError(f"{tree_id}/{model_id} truth-manifest path drift")
    if request.get("truth_manifest_sha256") != tree_sha_by_id[tree_id]:
        raise ValueError(f"{tree_id}/{model_id} truth-manifest hash drift")
    if request.get("n_expected_responses") != len(expected_coordinates):
        raise ValueError(
            f"{tree_id}/{model_id} expected-response count drift: expected "
            f"{len(expected_coordinates)}, got {request.get('n_expected_responses')!r}"
        )
    _validate_request_protocols(
        request.get("protocols"),
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        model_task=model_task,
        context=f"{tree_id}/{model_id}",
    )
    if "prompt_artifacts" not in request:
        raise ValueError(f"{tree_id}/{model_id} request lacks pinned prompt_artifacts")
    declared_prompt_artifacts = request["prompt_artifacts"]
    _validate_declared_prompt_artifacts(
        declared_prompt_artifacts,
        prompt_inputs=prompt_inputs,
        base=receipt_path.parent,
        context=f"{tree_id}/{model_id}.prompt_artifacts",
    )
    if payload["prompt_artifacts"] != declared_prompt_artifacts:
        raise ValueError(f"{tree_id}/{model_id} duplicate prompt-artifact declarations drift")
    if provenance.get("prompt_artifacts") != declared_prompt_artifacts:
        raise ValueError(f"{tree_id}/{model_id} prompt-artifact response provenance drift")

    provenance_pairs = {
        "training_summary": "training_summary_sha256",
        "run_list": "run_list_sha256",
        "ledger": "ledger_sha256",
        "truth_manifest": "truth_manifest_sha256",
        "checkpoint": "checkpoint_sha256",
        "registry": "registry_sha256",
    }
    for path_key, hash_key in provenance_pairs.items():
        if provenance.get(path_key) != request.get(path_key):
            raise ValueError(f"{tree_id}/{model_id} {path_key} receipt/request drift")
        if provenance.get(hash_key) != request.get(hash_key):
            raise ValueError(f"{tree_id}/{model_id} {hash_key} receipt/request drift")
    if provenance.get("checkpoint_artifact") != request.get("checkpoint_artifact"):
        raise ValueError(f"{tree_id}/{model_id} checkpoint artifact provenance drift")

    response_path = _declared_path(
        payload["responses"], receipt_path.parent, f"{receipt_path}.responses"
    )
    response_sha = sha256_file(response_path)
    if payload["responses_sha256"] != response_sha:
        raise ValueError(f"response JSONL hash mismatch: {response_path}")
    responses = load_behavior_responses(response_path)
    audit = audit_response_grid(
        responses,
        registry=registry,
        prompts_by_protocol=prompts_by_protocol,
        cohort_id=_nonempty(request.get("cohort_id"), f"{tree_id}/{model_id}.cohort_id"),
        tree_id=tree_id,
        model_id=model_id,
        base_model_id=_nonempty(
            request.get("base_model_id"), f"{tree_id}/{model_id}.base_model_id"
        ),
        base_model_revision=_nonempty(
            request.get("base_model_revision"),
            f"{tree_id}/{model_id}.base_model_revision",
        ),
        model_task=model_task,
        request_sha256=request_sha,
    )
    if not audit.valid or payload["audit"] != audit.to_dict():
        raise ValueError(f"leaf receipt response-grid audit drift: {tree_id}/{model_id}")
    if audit.n_expected != len(expected_coordinates) or audit.n_observed != len(expected_coordinates):
        raise ValueError(f"leaf response grid is not exactly complete: {tree_id}/{model_id}")
    if any(row.provenance.to_dict() != provenance for row in responses):
        raise ValueError(f"leaf response rows do not share exact receipt provenance: {tree_id}/{model_id}")

    return {
        "tree_id": tree_id,
        "model_id": model_id,
        "cohort_id": request["cohort_id"],
        "base_model_id": request["base_model_id"],
        "base_model_revision": request["base_model_revision"],
        "model_task": model_task,
        "request_sha256": request_sha,
        "response_path": str(response_path),
        "response_sha256": response_sha,
        "responses": responses,
    }


def _validate_request_protocols(
    raw_protocols: Any,
    *,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    model_task: str,
    context: str,
) -> None:
    supplied = _list(raw_protocols, f"{context}.protocols")
    if len(supplied) != len(prompts_by_protocol):
        raise ValueError(f"{context} protocol count drift")
    supplied_by_id: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(supplied):
        row = _mapping(value, f"{context}.protocols[{index}]")
        protocol_id = _nonempty(
            row.get("protocol_id"), f"{context}.protocols[{index}].protocol_id"
        )
        if protocol_id in supplied_by_id:
            raise ValueError(f"{context} duplicate request protocol {protocol_id!r}")
        supplied_by_id[protocol_id] = row
    if set(supplied_by_id) != set(prompts_by_protocol):
        raise ValueError(
            f"{context} request protocol set drift: expected {sorted(prompts_by_protocol)}, "
            f"got {sorted(supplied_by_id)}"
        )
    for protocol_id, prompts in prompts_by_protocol.items():
        protocol = registry.protocol(protocol_id)
        expected = {
            "protocol_id": protocol_id,
            "probe_ids": list(protocol.probe_ids),
            "prompt_counts": dict(protocol.prompt_counts),
            "samples_per_prompt": protocol.samples_per_prompt,
            "draws": [
                protocol.generation_options(model_task=model_task, sample_id=sample_id)
                for sample_id in range(protocol.samples_per_prompt)
            ],
            "prompts": [
                {
                    "probe_id": prompt.probe_id,
                    "prompt_id": prompt.prompt_id,
                    "source_index": prompt.source_index,
                    "source_row_sha256": prompt.source_row_sha256,
                    "prompt_sha256": prompt.prompt_sha256,
                    "reference_sha256": None
                    if prompt.reference is None
                    else hashlib.sha256(prompt.reference.encode("utf-8")).hexdigest(),
                    "fixture_id": prompt.metadata.get("fixture_id"),
                    "fixture_sha256": prompt.metadata.get("fixture_sha256"),
                    "source_indices_sha256": prompt.metadata.get("source_indices_sha256"),
                    "dataset_revision": prompt.metadata.get("dataset_revision"),
                }
                for prompt in prompts
            ],
        }
        if dict(supplied_by_id[protocol_id]) != expected:
            raise ValueError(f"{context} protocol/prompt/draw declaration drift: {protocol_id}")


def _validate_declared_prompt_artifacts(
    value: Any,
    *,
    prompt_inputs: Mapping[str, Mapping[str, Any]],
    base: Path,
    context: str,
) -> None:
    raw = _mapping(value, context)
    if set(raw) != set(prompt_inputs):
        raise ValueError(f"{context} protocol set drift")
    for protocol_id, spec_value in raw.items():
        spec = _mapping(spec_value, f"{context}.{protocol_id}")
        _exact_keys(spec, {"path", "sha256"}, context=f"{context}.{protocol_id}")
        path = _declared_path(spec["path"], base, f"{context}.{protocol_id}.path")
        expected = prompt_inputs[protocol_id]
        if path != Path(expected["path"]) or spec["sha256"] != expected["sha256"]:
            raise ValueError(f"{context}.{protocol_id} path/hash drift")


def _load_tree_details(path: Path, *, tree_id: str) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid truth-manifest JSON at {path}:{line_number}") from exc
            row = _mapping(value, f"{path}:{line_number}")
            required = {"node_id", "path", "depth", "grow", "parent_id"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{path}:{line_number} missing truth-manifest fields: {missing}")
            node_id = _nonempty(row["node_id"], f"{path}:{line_number}.node_id")
            lineage = _list(row["path"], f"{path}:{line_number}.path")
            if not all(isinstance(item, str) and item for item in lineage):
                raise ValueError(f"{path}:{line_number}.path must contain non-empty strings")
            if not lineage or lineage[0] != "root" or lineage[-1] != node_id:
                raise ValueError(f"{path}:{line_number} has an invalid root-to-node path")
            depth = _integer(row["depth"], f"{path}:{line_number}.depth")
            if depth < 1 or depth != len(lineage) - 1:
                raise ValueError(f"{path}:{line_number} depth/path mismatch")
            if row["parent_id"] != lineage[-2]:
                raise ValueError(f"{path}:{line_number} parent/path mismatch")
            if row.get("tree_id") not in (None, tree_id):
                raise ValueError(f"{path}:{line_number} tree_id mismatch")
            raw_rows.append(dict(row))
    if not raw_rows:
        raise ValueError(f"truth manifest is empty: {path}")
    node_ids = [row["node_id"] for row in raw_rows]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"truth manifest contains duplicate node IDs: {path}")
    records = load_manifest(path)
    leaf_set = set(leaf_ids(records))
    leaf_order = [row["node_id"] for row in raw_rows if row["node_id"] in leaf_set]
    if not leaf_order or len(leaf_order) != len(leaf_set):
        raise ValueError(f"truth manifest does not define one ordered trained-leaf set: {path}")
    row_by_node = {row["node_id"]: row for row in raw_rows}
    leaves: dict[str, dict[str, Any]] = {}
    for node_id in leaf_order:
        row = row_by_node[node_id]
        if row["grow"] != "train":
            raise ValueError(f"truth-manifest leaf is not trained: {tree_id}/{node_id}")
        task_family = _nonempty(
            row.get("task_family"), f"truth manifest {tree_id}/{node_id}.task_family"
        )
        leaves[node_id] = {"depth": int(row["depth"]), "task_family": task_family}
    return {"leaf_order": tuple(leaf_order), "leaves": leaves}


def _identity_panel_receipts(
    *,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for protocol_id, prompts in prompts_by_protocol.items():
        protocol = registry.protocol(protocol_id)
        for probe_id in protocol.probe_ids:
            probe_prompts = [prompt for prompt in prompts if prompt.probe_id == probe_id]
            identities = [
                {
                    "prompt_id": prompt.prompt_id,
                    "sample_id": sample_id,
                    "prompt_sha256": prompt.prompt_sha256,
                    "source_row_sha256": prompt.source_row_sha256,
                    "draw_seed": protocol.draw_seeds[sample_id],
                }
                for prompt in probe_prompts
                for sample_id in range(protocol.samples_per_prompt)
            ]
            key = f"{protocol_id}/{probe_id}"
            result[key] = {
                "protocol_id": protocol_id,
                "probe_id": probe_id,
                "n_prompts": len(probe_prompts),
                "samples_per_prompt": protocol.samples_per_prompt,
                "n_expected": len(identities),
                "identities_sha256": canonical_json_sha256(identities),
            }
    return result


def _build_depth_rows(
    *,
    panel_id: str,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    tree_order: Sequence[str],
    tree_details: Mapping[str, Mapping[str, Any]],
    responses_by_tree: Mapping[str, Mapping[str, Sequence[BehaviorResponse]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_leaf: list[dict[str, Any]] = []
    for protocol_id, prompts in prompts_by_protocol.items():
        protocol = registry.protocol(protocol_id)
        for probe_id in protocol.probe_ids:
            expected_keys = {
                (prompt.prompt_id, sample_id)
                for prompt in prompts
                if prompt.probe_id == probe_id
                for sample_id in range(protocol.samples_per_prompt)
            }
            for tree_id in tree_order:
                for node_id in tree_details[tree_id]["leaf_order"]:
                    indexed = _probe_response_index(
                        responses_by_tree[tree_id][node_id],
                        protocol_id=protocol_id,
                        probe_id=probe_id,
                    )
                    if set(indexed) != expected_keys:
                        raise ValueError(
                            f"internal prompt/draw grid mismatch at {tree_id}/{node_id}/{probe_id}"
                        )
                    n_outputs = len(expected_keys)
                    n_empty = sum(not indexed[key].text.strip() for key in expected_keys)
                    leaf = tree_details[tree_id]["leaves"][node_id]
                    per_leaf.append(
                        {
                            "panel": panel_id,
                            "probe": probe_id,
                            "tree": tree_id,
                            "node": node_id,
                            "depth": leaf["depth"],
                            "task_family": leaf["task_family"],
                            "n_outputs": n_outputs,
                            "n_empty": n_empty,
                            "empty_percent": _percent(n_empty, n_outputs),
                        }
                    )

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_leaf:
        grouped[(row["panel"], row["probe"], row["depth"])].append(row)
    per_depth: list[dict[str, Any]] = []
    for (panel, probe, depth), leaves in grouped.items():
        tree_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in leaves:
            tree_counts[row["tree"]][0] += int(row["n_empty"])
            tree_counts[row["tree"]][1] += int(row["n_outputs"])
        tree_percentages = np.asarray(
            [_percent(empty, total) for empty, total in tree_counts.values()], dtype=np.float64
        )
        mean, low, high = _mean_t_interval(tree_percentages)
        n_outputs = sum(int(row["n_outputs"]) for row in leaves)
        n_empty = sum(int(row["n_empty"]) for row in leaves)
        per_depth.append(
            {
                "panel": panel,
                "probe": probe,
                "depth": depth,
                "n_trees": len(tree_counts),
                "n_leaves": len(leaves),
                "n_outputs": n_outputs,
                "n_empty": n_empty,
                "pooled_empty_percent": _percent(n_empty, n_outputs),
                "mean_tree_empty_percent": mean,
                "t95_ci_low_tree_empty_percent": low,
                "t95_ci_high_tree_empty_percent": high,
            }
        )
    return per_leaf, per_depth


def _build_sensitivity_rows(
    *,
    panel_id: str,
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    tree_order: Sequence[str],
    tree_details: Mapping[str, Mapping[str, Any]],
    responses_by_tree: Mapping[str, Mapping[str, Sequence[BehaviorResponse]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for protocol_id, prompts in prompts_by_protocol.items():
        protocol = registry.protocol(protocol_id)
        for probe_id in protocol.probe_ids:
            expected_keys = tuple(
                (prompt.prompt_id, sample_id)
                for prompt in prompts
                if prompt.probe_id == probe_id
                for sample_id in range(protocol.samples_per_prompt)
            )
            for tree_id in tree_order:
                leaves = tree_details[tree_id]["leaf_order"]
                indexed_by_leaf = {
                    node_id: _probe_response_index(
                        responses_by_tree[tree_id][node_id],
                        protocol_id=protocol_id,
                        probe_id=probe_id,
                    )
                    for node_id in leaves
                }
                for left_index, left_node in enumerate(leaves):
                    for right_node in leaves[left_index + 1 :]:
                        left = indexed_by_leaf[left_node]
                        right = indexed_by_leaf[right_node]
                        n_primary = len(expected_keys)
                        n_paired = sum(
                            bool(left[key].text.strip()) and bool(right[key].text.strip())
                            for key in expected_keys
                        )
                        rows.append(
                            {
                                "panel": panel_id,
                                "probe": probe_id,
                                "tree": tree_id,
                                "left_node": left_node,
                                "right_node": right_node,
                                "n_primary": n_primary,
                                "n_paired_nonempty": n_paired,
                                "n_excluded": n_primary - n_paired,
                                "paired_nonempty_percent": _percent(n_paired, n_primary),
                            }
                        )

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["probe"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (panel, probe), pair_rows in grouped.items():
        n_primary = sum(int(row["n_primary"]) for row in pair_rows)
        n_paired = sum(int(row["n_paired_nonempty"]) for row in pair_rows)
        summaries.append(
            {
                "panel": panel,
                "probe": probe,
                "n_trees": len({row["tree"] for row in pair_rows}),
                "n_leaf_pairs": len(pair_rows),
                "n_primary": n_primary,
                "n_paired_nonempty": n_paired,
                "n_excluded": n_primary - n_paired,
                "paired_nonempty_percent": _percent(n_paired, n_primary),
            }
        )
    return rows, summaries


def _probe_response_index(
    responses: Sequence[BehaviorResponse], *, protocol_id: str, probe_id: str
) -> dict[tuple[str, int], BehaviorResponse]:
    indexed: dict[tuple[str, int], BehaviorResponse] = {}
    for row in responses:
        if row.protocol_id != protocol_id or row.probe_id != probe_id:
            continue
        key = (row.prompt_id, row.sample_id)
        if key in indexed:
            raise ValueError(f"duplicate prompt/draw response identity: {protocol_id}/{probe_id}/{key}")
        indexed[key] = row
    return indexed


def _mean_t_interval(values: np.ndarray) -> tuple[float, float | None, float | None]:
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("t-interval inputs must be one non-empty finite vector")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, None, None
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    critical = float(student_t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, df=len(values) - 1))
    margin = critical * standard_error
    return mean, mean - margin, mean + margin


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError(f"invalid count ratio: {numerator}/{denominator}")
    return 100.0 * numerator / denominator


def _payload_input_paths(payload: Mapping[str, Any]) -> set[Path]:
    inputs = _mapping(payload.get("inputs"), "payload.inputs")
    paths = {
        Path(inputs["config"]["path"]),
        Path(inputs["registry"]["path"]),
    }
    paths.update(Path(row["path"]) for row in inputs["prompt_artifacts"].values())
    for key in ("tree_manifests", "response_receipts", "responses"):
        paths.update(Path(row["path"]) for row in inputs[key])
    return {path.resolve() for path in paths}


def _revalidate_payload_input_hashes(payload: Mapping[str, Any]) -> None:
    """Prevent a validated-in-memory payload from being written after input drift."""

    inputs = _mapping(payload.get("inputs"), "payload.inputs")
    pinned_rows: list[Mapping[str, Any]] = [inputs["config"], inputs["registry"]]
    pinned_rows.extend(inputs["prompt_artifacts"].values())
    for key in ("tree_manifests", "response_receipts", "responses"):
        pinned_rows.extend(inputs[key])
    for index, row_value in enumerate(pinned_rows):
        row = _mapping(row_value, f"payload input pin {index}")
        path = Path(_nonempty(row.get("path"), f"payload input pin {index}.path")).resolve(
            strict=True
        )
        expected = _sha256(row.get("sha256"), f"payload input pin {index}.sha256")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"pinned input changed after analysis: expected {expected}, got {observed} "
                f"for {path}"
            )


def _output_row_count(name: str, payload: Mapping[str, Any]) -> int | None:
    return {
        "analysis_json": None,
        "per_leaf_csv": len(payload["per_leaf_rows"]),
        "per_depth_csv": len(payload["per_depth_rows"]),
        "sensitivity_csv": len(payload["sensitivity_pair_rows"]),
        "sensitivity_summary_csv": len(payload["sensitivity_summary_rows"]),
    }[name]


def _csv_bytes(rows: Any, columns: Sequence[str]) -> bytes:
    values = _list(rows, "CSV rows")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for index, value in enumerate(values):
        row = _mapping(value, f"CSV row {index}")
        if set(row) != set(columns):
            raise ValueError(
                f"CSV row {index} schema mismatch: expected {list(columns)}, got {sorted(row)}"
            )
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _write_bytes_atomic(path: Path, content: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary output: {temporary}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_pinned_file(value: Any, *, base: Path, context: str) -> tuple[Path, str]:
    spec = _mapping(value, context)
    _exact_keys(spec, {"path", "sha256"}, context=context)
    path = _declared_path(spec["path"], base, f"{context}.path")
    expected = _sha256(spec["sha256"], f"{context}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{context} SHA256 mismatch: expected {expected}, got {observed} for {path}"
        )
    return path, observed


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    try:
        data = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical-JSON serializable: {exc}") from exc
    return hashlib.sha256(data).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _declared_path(value: Any, base: Path, context: str) -> Path:
    raw = _nonempty(value, context)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be an object with string keys")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{context} schema mismatch: missing={missing}, unknown={unknown}")


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, context: str) -> str:
    result = _nonempty(value, context)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


__all__ = [
    "CONFIDENCE_LEVEL",
    "EMPTY_DEFINITION",
    "EOS_DEPTH_INPUT_SCHEMA",
    "EOS_DEPTH_RECEIPT_SCHEMA",
    "EOS_DEPTH_SCHEMA",
    "PER_DEPTH_COLUMNS",
    "PER_LEAF_COLUMNS",
    "SENSITIVITY_PAIR_COLUMNS",
    "SENSITIVITY_SUMMARY_COLUMNS",
    "build_eos_depth_analysis",
    "canonical_json_sha256",
    "sha256_file",
    "write_eos_depth_outputs",
]
