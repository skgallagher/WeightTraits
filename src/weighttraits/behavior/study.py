"""End-to-end corrected behavior studies with fail-closed receipts.

This module is the orchestration boundary between completed WeightTraits training, strict prompt
materialization, native direct-weight distances, held-out response receipts, paired MiniLM
semantic distances, and Fisher-z/DerSimonian-Laird summaries.  It never reads or writes the legacy
ELLMTrees result layout.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from weighttraits.behavior.contracts import (
    BehaviorProtocolRegistry,
    RenderedBehaviorPrompt,
    load_behavior_protocol_registry,
)
from weighttraits.behavior.direct_bridge import (
    DirectDistanceSourceAttestation,
    export_native_direct_distance,
)
from weighttraits.behavior.eos_depth import build_eos_depth_analysis, write_eos_depth_outputs
from weighttraits.behavior.materialize import validate_prompt_materialization_receipt
from weighttraits.behavior.meta import (
    CorrelationStudy,
    MetaAnalysisSummary,
    fisher_z_dersimonian_laird,
    paired_semantic_similarity_study,
    write_meta_summary_json,
)
from weighttraits.behavior.probe_inference import (
    ResolvedCheckpointTree,
    canonical_json_sha256,
    resolve_leaf_checkpoints,
    sha256_file,
    sha256_path,
    validate_inference_receipt_schema,
)
from weighttraits.behavior.responses import (
    ResponseProvenance,
    audit_response_grid,
    expected_response_coordinates,
    load_behavior_responses,
    load_rendered_behavior_prompts,
)
from weighttraits.behavior.semantic import (
    MINILM_MODEL,
    MINILM_REVISION,
    SemanticAnalysisAttestation,
    SemanticPanelSpec,
    SemanticResponse,
    build_paired_semantic_distances,
    read_paired_semantic_artifact,
    semantic_response_rows_sha256,
    write_paired_semantic_artifact,
)
from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    validate_strict_training_completion,
)


CHECKPOINT_SET_INPUT_SCHEMA = "weighttraits.behavior.checkpoint_set_input.v1"
CHECKPOINT_SET_RECEIPT_SCHEMA = "weighttraits.behavior.checkpoint_set_receipt.v1"
ENCODER_INPUT_SCHEMA = "weighttraits.behavior.encoder_input.v1"
ENCODER_RECEIPT_SCHEMA = "weighttraits.behavior.encoder_receipt.v1"
STUDY_INPUT_SCHEMA = "weighttraits.behavior.study_input.v1"
STUDY_OUTPUT_SCHEMA = "weighttraits.behavior.study.v1"
STUDY_RECEIPT_SCHEMA = "weighttraits.behavior.study_receipt.v1"
CELL_RECEIPT_SCHEMA = "weighttraits.behavior.cell_receipt.v1"
RESPONSE_SET_RECEIPT_SCHEMA = "weighttraits.behavior.response_set_receipt.v1"
TABLE_SUMMARY_SCHEMA = "weighttraits.behavior.table3_table7_summary.v1"

ALLOWED_ROUTES = frozenset({"table3", "table7", "workbook", "appendix"})
TABLE_SUMMARY_COLUMNS = (
    "cell_id",
    "study_id",
    "model_family",
    "condition",
    "cohort_id",
    "protocol_id",
    "probe_id",
    "role",
    "routes",
    "n_studies",
    "n_pairs_min",
    "n_pairs_max",
    "n_pairs_total",
    "random_effect_r",
    "random_effect_ci_low_r",
    "random_effect_ci_high_r",
    "random_effect_p_value",
    "fixed_effect_r",
    "equal_weight_r",
    "q",
    "q_df",
    "q_p_value",
    "tau2",
    "i2",
    "i2_percent",
    "study_inputs_sha256",
    "dl_summary_sha256",
    "cell_receipt_sha256",
)


@dataclass(frozen=True)
class CellSpec:
    """One explicit paper/workbook behavioral cell."""

    cell_id: str
    protocol_id: str
    probe_id: str
    role: str
    routes: tuple[str, ...]
    eos_depth: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "protocol_id": self.protocol_id,
            "probe_id": self.probe_id,
            "role": self.role,
            "routes": list(self.routes),
            "eos_depth": self.eos_depth,
        }


def build_checkpoint_set_receipt(
    config_path: str | Path,
    *,
    receipt_path: str | Path,
    _allow_nonpaper: bool = False,
) -> dict[str, Any]:
    """Resolve and hash the exact leaf checkpoint grid for a completed cohort."""

    config = Path(config_path).resolve(strict=True)
    destination = Path(receipt_path).resolve()
    _reject_legacy_output(destination)
    if destination.exists():
        raise FileExistsError(f"checkpoint-set receipt already exists: {destination}")
    raw = _json_object(config)
    _exact_keys(
        raw,
        {
            "schema",
            "cohort_id",
            "model_task",
            "path_base",
            "registry",
            "training_summary",
            "completion_receipt",
            "tree_ids",
        },
        context="checkpoint-set input",
    )
    if raw["schema"] != CHECKPOINT_SET_INPUT_SCHEMA:
        raise ValueError(f"checkpoint-set schema must be {CHECKPOINT_SET_INPUT_SCHEMA!r}")
    base = config.parent
    cohort_id = _identifier(raw["cohort_id"], "cohort_id")
    model_task = _model_task(raw["model_task"])
    path_base = _directory(raw["path_base"], base=base, label="path_base")
    registry_path, registry_sha = _pinned_file(raw["registry"], base=base, label="registry")
    registry = load_behavior_protocol_registry(registry_path)
    if registry.sha256 != registry_sha:
        raise ValueError("registry loader hash disagrees with its pinned hash")
    training_path, training_sha = _pinned_file(
        raw["training_summary"], base=base, label="training_summary"
    )
    completion_path, completion_sha, completion = _completion_receipt(
        raw["completion_receipt"], base=base, cohort_id=cohort_id, strict=not _allow_nonpaper
    )
    tree_ids = _tree_ids(raw["tree_ids"], strict=not _allow_nonpaper)
    pin = registry.model_pins[model_task]
    trees = [
        _resolved_tree_row(
            resolve_leaf_checkpoints(
                training_path,
                cohort_id=cohort_id,
                tree_id=tree_id,
                model_task=model_task,
                base_model_id=pin.model_id,
                base_model_revision=pin.revision,
                path_base=path_base,
            )
        )
        for tree_id in tree_ids
    ]
    core = {
        "cohort_id": cohort_id,
        "model_task": model_task,
        "base_model_id": pin.model_id,
        "base_model_revision": pin.revision,
        "path_base": str(path_base),
        "registry": {"path": str(registry_path), "sha256": registry_sha},
        "training_summary": {"path": str(training_path), "sha256": training_sha},
        "completion_receipt": {"path": str(completion_path), "sha256": completion_sha},
        "trees": trees,
    }
    payload = {
        "schema": CHECKPOINT_SET_RECEIPT_SCHEMA,
        "valid": True,
        "input_config": {"path": str(config), "sha256": sha256_file(config)},
        **core,
        "n_trees": len(trees),
        "n_leaves": sum(len(tree["leaves"]) for tree in trees),
        "checkpoint_set_sha256": canonical_json_sha256(core),
        "completion_valid": completion.get("valid") is True,
    }
    _write_json_atomic(destination, payload)
    return payload


def build_encoder_receipt(
    config_path: str | Path,
    *,
    receipt_path: str | Path,
    _package_version: str | None = None,
) -> dict[str, Any]:
    """Pin the actual local MiniLM snapshot and sentence-transformers runtime."""

    config = Path(config_path).resolve(strict=True)
    destination = Path(receipt_path).resolve()
    _reject_legacy_output(destination)
    if destination.exists():
        raise FileExistsError(f"encoder receipt already exists: {destination}")
    raw = _json_object(config)
    _exact_keys(raw, {"schema", "snapshot"}, context="encoder input")
    if raw["schema"] != ENCODER_INPUT_SCHEMA:
        raise ValueError(f"encoder schema must be {ENCODER_INPUT_SCHEMA!r}")
    snapshot = _directory(raw["snapshot"], base=config.parent, label="encoder snapshot")
    version = _package_version
    if version is None:
        try:
            version = importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers must be installed for the encoder receipt") from exc
    version = _identifier(version, "sentence-transformers version")
    payload = {
        "schema": ENCODER_RECEIPT_SCHEMA,
        "valid": True,
        "mode": "pinned_internal_minilm",
        "model": MINILM_MODEL,
        "revision": MINILM_REVISION,
        "snapshot": {
            "path": str(snapshot),
            "sha256": _encoder_snapshot_sha256(snapshot),
        },
        "runtime": {"package": "sentence-transformers", "version": version},
        "input_config": {"path": str(config), "sha256": sha256_file(config)},
    }
    _write_json_atomic(destination, payload)
    return payload


def build_behavior_study(
    config_path: str | Path,
    *,
    out_dir: str | Path,
    _allow_nonpaper: bool = False,
    _semantic_encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
    _package_version: str | None = None,
) -> dict[str, Any]:
    """Build every explicitly declared cell for one completed cohort.

    Production callers do not pass the underscored test hooks.  The terminal study receipt is
    written last; an interrupted directory is invalid even if some independently verified
    intermediate artifacts are present.
    """

    config = Path(config_path).resolve(strict=True)
    output = Path(out_dir).resolve()
    _reject_legacy_output(output)
    if output.exists():
        raise FileExistsError(f"behavior study output directory already exists: {output}")
    raw = _json_object(config)
    _exact_keys(
        raw,
        {
            "schema",
            "study_id",
            "model_family",
            "condition",
            "cohort_id",
            "model_task",
            "direct_metric",
            "path_base",
            "registry",
            "training_summary",
            "completion_receipt",
            "checkpoint_set_receipt",
            "encoder_receipt",
            "prompt_artifacts",
            "cells",
            "trees",
        },
        context="behavior study input",
    )
    if raw["schema"] != STUDY_INPUT_SCHEMA:
        raise ValueError(f"behavior study schema must be {STUDY_INPUT_SCHEMA!r}")
    study_id = _safe_identifier(raw["study_id"], "study_id")
    model_family = _identifier(raw["model_family"], "model_family")
    condition = _identifier(raw["condition"], "condition")
    cohort_id = _identifier(raw["cohort_id"], "cohort_id")
    model_task = _model_task(raw["model_task"])
    if raw["direct_metric"] != "cosine":
        raise ValueError("corrected Table 3/Table 7 studies require direct_metric='cosine'")
    base = config.parent
    path_base = _directory(raw["path_base"], base=base, label="path_base")
    registry_path, registry_sha = _pinned_file(raw["registry"], base=base, label="registry")
    registry = load_behavior_protocol_registry(registry_path)
    if registry.sha256 != registry_sha:
        raise ValueError("registry loader hash disagrees with its pinned hash")
    registry.validate_model_binding(
        model_task=model_task,
        model_id=registry.model_pins[model_task].model_id,
        model_revision=registry.model_pins[model_task].revision,
    )
    training_path, training_sha = _pinned_file(
        raw["training_summary"], base=base, label="training_summary"
    )
    completion_path, completion_sha, _ = _completion_receipt(
        raw["completion_receipt"], base=base, cohort_id=cohort_id, strict=not _allow_nonpaper
    )
    checkpoint_path, checkpoint_sha = _pinned_file(
        raw["checkpoint_set_receipt"], base=base, label="checkpoint_set_receipt"
    )
    checkpoint_payload = _validate_checkpoint_set(
        checkpoint_path,
        registry=registry,
        registry_path=registry_path,
        registry_sha=registry_sha,
        training_path=training_path,
        training_sha=training_sha,
        completion_path=completion_path,
        completion_sha=completion_sha,
        cohort_id=cohort_id,
        model_task=model_task,
        path_base=path_base,
        strict=not _allow_nonpaper,
    )
    encoder_path, encoder_sha = _pinned_file(
        raw["encoder_receipt"], base=base, label="encoder_receipt"
    )
    encoder_snapshot = _validate_encoder_receipt(
        encoder_path,
        registry=registry,
        package_version=_package_version,
    )
    cells = _cell_specs(raw["cells"], registry=registry)
    prompts_by_protocol, prompt_pins = _prompt_artifacts(
        raw["prompt_artifacts"],
        base=base,
        registry=registry,
        registry_path=registry_path,
        registry_sha=registry_sha,
        model_task=model_task,
        cells=cells,
    )
    expected_coordinates = expected_response_coordinates(registry, prompts_by_protocol)
    tree_specs = _tree_specs(raw["trees"], base=base, strict=not _allow_nonpaper)
    checkpoint_trees = {
        tree["tree_id"]: tree for tree in _list(checkpoint_payload["trees"], "checkpoint trees")
    }
    if set(tree_specs) != set(checkpoint_trees):
        raise ValueError("study tree set differs from the checkpoint-set receipt")

    output.mkdir(parents=True)
    studies_by_cell: dict[str, list[CorrelationStudy]] = {cell.cell_id: [] for cell in cells}
    tree_artifacts: dict[str, dict[str, Any]] = {}
    all_response_receipt_pins: list[dict[str, str]] = []
    for tree_id in tree_specs:
        tree_spec = tree_specs[tree_id]
        checkpoint_tree = _mapping(checkpoint_trees[tree_id], f"checkpoint tree {tree_id}")
        truth = _mapping(checkpoint_tree["truth_manifest"], f"checkpoint {tree_id} truth")
        if tree_spec["truth_manifest"] != truth:
            raise ValueError(f"{tree_id} truth manifest differs from checkpoint-set receipt")
        response_bundle = _validate_tree_response_receipts(
            tree_spec["response_receipts"],
            base=base,
            registry=registry,
            registry_path=registry_path,
            registry_sha=registry_sha,
            prompts_by_protocol=prompts_by_protocol,
            prompt_pins=prompt_pins,
            expected_coordinates=expected_coordinates,
            checkpoint_tree=checkpoint_tree,
            checkpoint_receipt_path=checkpoint_path,
            checkpoint_receipt_sha=checkpoint_sha,
            cohort_id=cohort_id,
            model_task=model_task,
        )
        response_set_path = output / "response_sets" / f"{tree_id}.response_set.json"
        response_set_payload = {
            "schema": RESPONSE_SET_RECEIPT_SCHEMA,
            "valid": True,
            "cohort_id": cohort_id,
            "tree_id": tree_id,
            "model_task": model_task,
            "registry": {"path": str(registry_path), "sha256": registry_sha},
            "checkpoint_set_receipt": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
            },
            "prompt_artifacts": prompt_pins,
            "leaves": response_bundle["receipt_rows"],
            "n_leaves": len(response_bundle["receipt_rows"]),
            "n_response_rows": sum(
                len(rows) for rows in response_bundle["responses_by_model"].values()
            ),
        }
        _write_json_atomic(response_set_path, response_set_payload)
        response_set_sha = sha256_file(response_set_path)
        all_response_receipt_pins.extend(
            {"path": row["receipt"], "sha256": row["receipt_sha256"]}
            for row in response_bundle["receipt_rows"]
        )

        direct_attestation = _direct_attestation(
            tree_spec["direct_attestation"],
            base=base,
            cohort_id=cohort_id,
            tree_id=tree_id,
            completion_path=completion_path,
            completion_sha=completion_sha,
            checkpoint_tree=checkpoint_tree,
        )
        direct = export_native_direct_distance(
            direct_attestation.summary,
            metric="cosine",
            out_dir=output / "direct" / tree_id,
            attestation=direct_attestation,
        )
        model_ids = tuple(leaf["leaf_id"] for leaf in checkpoint_tree["leaves"])
        if direct.model_ids != model_ids:
            raise ValueError(
                f"{tree_id} direct model ordering differs from checkpoint truth order: "
                f"{direct.model_ids} != {model_ids}"
            )
        direct_sha = sha256_file(direct.metadata_path)
        semantic_rows: dict[str, dict[str, str]] = {}
        for cell in cells:
            protocol = registry.protocol(cell.protocol_id)
            probe_prompts = [
                prompt
                for prompt in prompts_by_protocol[cell.protocol_id]
                if prompt.probe_id == cell.probe_id
            ]
            sample_ids = tuple(f"draw-{index}" for index in range(protocol.samples_per_prompt))
            panel = SemanticPanelSpec.from_prompt_samples(
                panel_id=f"{cell.protocol_id}/{cell.probe_id}",
                prompt_ids=[prompt.prompt_id for prompt in probe_prompts],
                sample_ids=sample_ids,
            )
            semantic_responses = _semantic_responses(
                response_bundle["responses_by_model"],
                model_ids=model_ids,
                protocol_id=cell.protocol_id,
                probe_id=cell.probe_id,
                prompts=probe_prompts,
                samples_per_prompt=protocol.samples_per_prompt,
            )
            response_rows_sha = semantic_response_rows_sha256(
                semantic_responses,
                panel=panel,
                model_ids=model_ids,
            )
            attestation = SemanticAnalysisAttestation(
                cohort_id=cohort_id,
                tree_id=tree_id,
                protocol_id=cell.protocol_id,
                probe_id=cell.probe_id,
                panel_id=panel.panel_id,
                expected_prompt_count=len(probe_prompts),
                expected_sample_ids=sample_ids,
                registry_sha256=registry_sha,
                fixture_sha256=registry.fixtures[protocol.fixture_id].sha256,
                prompt_artifact_sha256=prompt_pins[cell.protocol_id]["sha256"],
                response_receipt_sha256=response_set_sha,
                response_rows_sha256=response_rows_sha,
                checkpoint_set_sha256=checkpoint_sha,
                encoder_receipt_sha256=encoder_sha,
            )
            semantic = build_paired_semantic_distances(
                semantic_responses,
                panel=panel,
                model_ids=model_ids,
                attestation=attestation,
                embedding_model=registry.embedding_pin.model_id,
                embedding_revision=registry.embedding_pin.revision,
                encoder_snapshot=encoder_snapshot,
                encoder=_semantic_encoder,
                allow_test_encoder=_semantic_encoder is not None,
            )
            semantic_dir = output / "semantic" / cell.cell_id / tree_id
            semantic_metadata = write_paired_semantic_artifact(
                semantic,
                semantic_dir,
                allow_test_artifact=_semantic_encoder is not None,
            )
            reread = read_paired_semantic_artifact(
                semantic_dir,
                expected_attestation=attestation,
                allow_test_artifact=_semantic_encoder is not None,
            )
            semantic_sha = sha256_file(semantic_metadata)
            study = paired_semantic_similarity_study(
                f"{tree_id}/{cell.cell_id}",
                direct_pairs=direct.upper_triangle_pairs(),
                semantic_pairs=reread.upper_triangle_pairs(),
                direct_artifact_sha256=direct_sha,
                semantic_artifact_sha256=semantic_sha,
            )
            studies_by_cell[cell.cell_id].append(study)
            semantic_rows[cell.cell_id] = {
                "path": str(semantic_metadata),
                "sha256": semantic_sha,
                "response_rows_sha256": response_rows_sha,
            }
        tree_artifacts[tree_id] = {
            "direct": {"path": str(direct.metadata_path), "sha256": direct_sha},
            "response_set": {"path": str(response_set_path), "sha256": response_set_sha},
            "semantic": semantic_rows,
        }

    cell_rows: list[dict[str, Any]] = []
    cell_receipts: dict[str, dict[str, str]] = {}
    for cell in cells:
        summary = fisher_z_dersimonian_laird(
            studies_by_cell[cell.cell_id],
            analysis_id=f"{study_id}/{cell.cell_id}",
            endpoint=f"{cell.protocol_id}/{cell.probe_id}",
        )
        dl_fields = _dl_summary_fields(
            summary,
            cell=cell,
            study_id=study_id,
            model_family=model_family,
            condition=condition,
            cohort_id=cohort_id,
        )
        dl_summary_sha = canonical_json_sha256(summary.to_dict())
        dl_fields["dl_summary_sha256"] = dl_summary_sha
        meta_path = output / "meta" / f"{cell.cell_id}.json"
        write_meta_summary_json(
            summary,
            meta_path,
            provenance={
                "study_config": {"path": str(config), "sha256": sha256_file(config)},
                "registry": {"path": str(registry_path), "sha256": registry_sha},
                "checkpoint_set_receipt": {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha,
                },
                "encoder_receipt": {"path": str(encoder_path), "sha256": encoder_sha},
                "prompt_artifact": prompt_pins[cell.protocol_id],
                "tree_artifacts": {
                    tree_id: {
                        "direct": tree_artifacts[tree_id]["direct"],
                        "semantic": tree_artifacts[tree_id]["semantic"][cell.cell_id],
                    }
                    for tree_id in tree_artifacts
                },
            },
        )
        cell_receipt_path = output / "cells" / f"{cell.cell_id}.receipt.json"
        cell_receipt_payload = {
            "schema": CELL_RECEIPT_SCHEMA,
            "valid": True,
            "production_valid": _semantic_encoder is None,
            "study_id": study_id,
            "model_family": model_family,
            "condition": condition,
            "cohort_id": cohort_id,
            "cell": cell.to_dict(),
            "dl_summary": summary.to_dict(),
            "dl_summary_sha256": dl_summary_sha,
            "table_summary_fields": dl_fields,
            "meta_artifact": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
            "n_tree_studies": len(studies_by_cell[cell.cell_id]),
            "tree_study_inputs": {
                study.study_id: study.input_provenance.to_dict()
                for study in studies_by_cell[cell.cell_id]
            },
        }
        _write_json_atomic(cell_receipt_path, cell_receipt_payload)
        cell_receipt_sha = sha256_file(cell_receipt_path)
        dl_fields["cell_receipt_sha256"] = cell_receipt_sha
        cell_rows.append(dl_fields)
        cell_receipts[cell.cell_id] = {
            "path": str(cell_receipt_path),
            "sha256": cell_receipt_sha,
        }

    summary_json = output / "table3_table7_summary.json"
    summary_csv = output / "table3_table7_summary.csv"
    summary_payload = {
        "schema": TABLE_SUMMARY_SCHEMA,
        "valid": True,
        "production_valid": _semantic_encoder is None,
        "study_id": study_id,
        "model_family": model_family,
        "condition": condition,
        "cohort_id": cohort_id,
        "shared_source_contract": (
            "Table 3 and Table 7 fields are projections of the identical dl_summary_sha256"
        ),
        "rows": cell_rows,
    }
    _write_json_atomic(summary_json, summary_payload)
    _write_csv_atomic(summary_csv, cell_rows, TABLE_SUMMARY_COLUMNS)

    eos_receipt: dict[str, Any] | None = None
    if any(cell.eos_depth for cell in cells):
        eos_dir = output / "eos_depth"
        eos_config = eos_dir / "input.json"
        eos_input = {
            "schema": "weighttraits.behavior.eos_depth_input.v1",
            "panel_id": study_id,
            "registry": {"path": str(registry_path), "sha256": registry_sha},
            "prompt_artifacts": prompt_pins,
            "tree_manifests": [
                {"tree_id": tree_id, **tree_specs[tree_id]["truth_manifest"]}
                for tree_id in tree_specs
            ],
            "response_receipts": all_response_receipt_pins,
        }
        _write_json_atomic(eos_config, eos_input)
        eos_payload = build_eos_depth_analysis(eos_config)
        eos_receipt = write_eos_depth_outputs(
            eos_payload,
            json_path=eos_dir / "analysis.json",
            per_leaf_csv_path=eos_dir / "per_leaf.csv",
            per_depth_csv_path=eos_dir / "per_depth.csv",
            sensitivity_csv_path=eos_dir / "sensitivity_pairs.csv",
            sensitivity_summary_csv_path=eos_dir / "sensitivity_summary.csv",
            receipt_path=eos_dir / "receipt.json",
        )

    output_files = _output_inventory(output, excluded={output / "study_receipt.json"})
    study_receipt = {
        "schema": STUDY_RECEIPT_SCHEMA,
        "valid": True,
        "production_valid": _semantic_encoder is None,
        "study_id": study_id,
        "model_family": model_family,
        "condition": condition,
        "cohort_id": cohort_id,
        "model_task": model_task,
        "input_config": {"path": str(config), "sha256": sha256_file(config)},
        "registry": {"path": str(registry_path), "sha256": registry_sha},
        "training_summary": {"path": str(training_path), "sha256": training_sha},
        "completion_receipt": {"path": str(completion_path), "sha256": completion_sha},
        "checkpoint_set_receipt": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "encoder_receipt": {"path": str(encoder_path), "sha256": encoder_sha},
        "prompt_artifacts": prompt_pins,
        "cells": cell_receipts,
        "table_summary_json": {"path": str(summary_json), "sha256": sha256_file(summary_json)},
        "table_summary_csv": {"path": str(summary_csv), "sha256": sha256_file(summary_csv)},
        "eos_depth_receipt": eos_receipt,
        "n_cells": len(cells),
        "n_trees": len(tree_specs),
        "output_files": output_files,
        "output_files_sha256": canonical_json_sha256(output_files),
    }
    _write_json_atomic(output / "study_receipt.json", study_receipt)
    return study_receipt


def load_cell_receipt(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load one cell receipt and recompute its shared DL summary projection."""

    source = Path(path).resolve(strict=True)
    if sha256_file(source) != _sha256(expected_sha256, "cell receipt sha256"):
        raise ValueError(f"cell receipt hash mismatch: {source}")
    payload = _json_object(source)
    if payload.get("schema") != CELL_RECEIPT_SCHEMA or payload.get("valid") is not True:
        raise ValueError(f"invalid cell receipt: {source}")
    summary = _mapping(payload.get("dl_summary"), "cell dl_summary")
    if canonical_json_sha256(summary) != payload.get("dl_summary_sha256"):
        raise ValueError(f"cell DL summary hash mismatch: {source}")
    fields = _mapping(payload.get("table_summary_fields"), "cell table_summary_fields")
    if fields.get("dl_summary_sha256") != payload.get("dl_summary_sha256"):
        raise ValueError(f"cell Table 3/Table 7 source hash drift: {source}")
    return payload


def _resolved_tree_row(tree: ResolvedCheckpointTree) -> dict[str, Any]:
    first = tree.leaves[0]
    leaves = [
        {
            "leaf_id": leaf.leaf_id,
            "leaf_ordinal": leaf.leaf_ordinal,
            "checkpoint": str(leaf.checkpoint),
            "checkpoint_sha256": sha256_path(leaf.checkpoint),
        }
        for leaf in tree.leaves
    ]
    return {
        "tree_id": tree.tree_id,
        "method": tree.method,
        "artifact_name": tree.artifact_name,
        "truth_manifest": {
            "path": str(first.truth_manifest),
            "sha256": first.truth_manifest_sha256,
        },
        "run_list": {"path": str(first.run_list), "sha256": first.run_list_sha256},
        "ledger": {"path": str(first.ledger), "sha256": first.ledger_sha256},
        "leaves": leaves,
        "leaves_sha256": canonical_json_sha256(leaves),
    }


def _validate_checkpoint_set(
    path: Path,
    *,
    registry: BehaviorProtocolRegistry,
    registry_path: Path,
    registry_sha: str,
    training_path: Path,
    training_sha: str,
    completion_path: Path,
    completion_sha: str,
    cohort_id: str,
    model_task: str,
    path_base: Path,
    strict: bool,
) -> dict[str, Any]:
    payload = _json_object(path)
    if payload.get("schema") != CHECKPOINT_SET_RECEIPT_SCHEMA or payload.get("valid") is not True:
        raise ValueError(f"invalid checkpoint-set receipt: {path}")
    expected_scalars = {
        "cohort_id": cohort_id,
        "model_task": model_task,
        "base_model_id": registry.model_pins[model_task].model_id,
        "base_model_revision": registry.model_pins[model_task].revision,
        "path_base": str(path_base),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint-set {key} drift: expected {expected!r}")
    expected_pins = {
        "registry": {"path": str(registry_path), "sha256": registry_sha},
        "training_summary": {"path": str(training_path), "sha256": training_sha},
        "completion_receipt": {"path": str(completion_path), "sha256": completion_sha},
    }
    for key, expected in expected_pins.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint-set {key} pin drift")
    trees = _list(payload.get("trees"), "checkpoint-set trees")
    tree_ids = [tree.get("tree_id") for tree in trees if isinstance(tree, Mapping)]
    if len(tree_ids) != len(trees) or len(set(tree_ids)) != len(tree_ids):
        raise ValueError("checkpoint-set tree IDs are missing or duplicated")
    if strict and tuple(tree_ids) != ALL_TREE_IDS:
        raise ValueError("production checkpoint-set receipt must contain exact trees 001..050")
    pin = registry.model_pins[model_task]
    expected_trees = [
        _resolved_tree_row(
            resolve_leaf_checkpoints(
                training_path,
                cohort_id=cohort_id,
                tree_id=tree_id,
                model_task=model_task,
                base_model_id=pin.model_id,
                base_model_revision=pin.revision,
                path_base=path_base,
            )
        )
        for tree_id in tree_ids
    ]
    if trees != expected_trees:
        raise ValueError("checkpoint-set receipt differs from freshly resolved checkpoint grid")
    core = {
        "cohort_id": cohort_id,
        "model_task": model_task,
        "base_model_id": pin.model_id,
        "base_model_revision": pin.revision,
        "path_base": str(path_base),
        **expected_pins,
        "trees": expected_trees,
    }
    if payload.get("checkpoint_set_sha256") != canonical_json_sha256(core):
        raise ValueError("checkpoint-set self hash mismatch")
    return payload


def _validate_encoder_receipt(
    path: Path,
    *,
    registry: BehaviorProtocolRegistry,
    package_version: str | None,
) -> Path:
    payload = _json_object(path)
    if payload.get("schema") != ENCODER_RECEIPT_SCHEMA or payload.get("valid") is not True:
        raise ValueError(f"invalid encoder receipt: {path}")
    expected = registry.embedding_pin
    if expected.model_id != MINILM_MODEL or expected.revision != MINILM_REVISION:
        raise ValueError("behavior registry embedding pin differs from the internal MiniLM contract")
    if payload.get("mode") != "pinned_internal_minilm":
        raise ValueError("encoder receipt mode drift")
    if payload.get("model") != expected.model_id or payload.get("revision") != expected.revision:
        raise ValueError("encoder receipt model/revision drift")
    snapshot = _mapping(payload.get("snapshot"), "encoder snapshot")
    snapshot_path = Path(_identifier(snapshot.get("path"), "encoder snapshot path")).resolve(
        strict=True
    )
    if snapshot.get("sha256") != _encoder_snapshot_sha256(snapshot_path):
        raise ValueError("encoder snapshot hash drift")
    runtime = _mapping(payload.get("runtime"), "encoder runtime")
    if runtime.get("package") != "sentence-transformers":
        raise ValueError("encoder runtime package drift")
    observed_version = package_version
    if observed_version is None:
        try:
            observed_version = importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required for semantic analysis") from exc
    if runtime.get("version") != observed_version:
        raise ValueError(
            f"encoder runtime version drift: receipt={runtime.get('version')!r}, "
            f"installed={observed_version!r}"
        )
    return snapshot_path


def _prompt_artifacts(
    raw: Any,
    *,
    base: Path,
    registry: BehaviorProtocolRegistry,
    registry_path: Path,
    registry_sha: str,
    model_task: str,
    cells: Sequence[CellSpec],
) -> tuple[dict[str, list[RenderedBehaviorPrompt]], dict[str, dict[str, str]]]:
    specs = _mapping(raw, "prompt_artifacts")
    required_protocols = {cell.protocol_id for cell in cells}
    if set(specs) != required_protocols:
        raise ValueError(
            "prompt artifact protocols must equal the explicitly declared cell protocols: "
            f"expected={sorted(required_protocols)}, got={sorted(specs)}"
        )
    prompts: dict[str, list[RenderedBehaviorPrompt]] = {}
    pins: dict[str, dict[str, str]] = {}
    used_probes: set[str] = set()
    for protocol_id, value in specs.items():
        spec = _mapping(value, f"prompt_artifacts.{protocol_id}")
        _exact_keys(
            spec,
            {"path", "sha256", "materialization_receipt"},
            context=f"prompt_artifacts.{protocol_id}",
        )
        path, digest = _pinned_file(spec, base=base, label=f"prompt_artifacts.{protocol_id}")
        receipt_path, receipt_sha = _pinned_file(
            spec["materialization_receipt"],
            base=base,
            label=f"prompt_artifacts.{protocol_id}.materialization_receipt",
        )
        validate_prompt_materialization_receipt(
            receipt_path,
            registry_path=registry_path,
            registry_sha256=registry_sha,
            model_task=model_task,
            protocol_id=protocol_id,
            prompt_path=path,
            prompt_sha256=digest,
        )
        loaded = load_rendered_behavior_prompts(path)
        protocol = registry.protocol(protocol_id)
        probe_set = {prompt.probe_id for prompt in loaded}
        if probe_set != set(protocol.probe_ids):
            raise ValueError(f"{protocol_id} prompt probes differ from registry")
        overlap = used_probes & probe_set
        if overlap:
            raise ValueError(f"probe IDs overlap across protocols: {sorted(overlap)}")
        used_probes.update(probe_set)
        for prompt in loaded:
            if prompt.metadata.get("model_task") != model_task:
                raise ValueError(f"prompt {prompt.prompt_id!r} model_task drift")
            fixture = registry.fixtures[protocol.fixture_id]
            if prompt.metadata.get("fixture_sha256") != fixture.sha256:
                raise ValueError(f"prompt {prompt.prompt_id!r} fixture hash drift")
        prompts[protocol_id] = loaded
        pins[protocol_id] = {
            "path": str(path),
            "sha256": digest,
            "materialization_receipt": str(receipt_path),
            "materialization_receipt_sha256": receipt_sha,
        }
    expected_response_coordinates(registry, prompts)
    return prompts, pins


def _validate_tree_response_receipts(
    raw_specs: Any,
    *,
    base: Path,
    registry: BehaviorProtocolRegistry,
    registry_path: Path,
    registry_sha: str,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    prompt_pins: Mapping[str, Mapping[str, str]],
    expected_coordinates: Sequence[tuple[str, RenderedBehaviorPrompt, int]],
    checkpoint_tree: Mapping[str, Any],
    checkpoint_receipt_path: Path,
    checkpoint_receipt_sha: str,
    cohort_id: str,
    model_task: str,
) -> dict[str, Any]:
    specs = _list(raw_specs, "response_receipts")
    expected_leaves = [leaf["leaf_id"] for leaf in checkpoint_tree["leaves"]]
    by_model: dict[str, list[Any]] = {}
    receipt_rows: list[dict[str, Any]] = []
    expected_prompt_pins = {
        protocol_id: {"path": row["path"], "sha256": row["sha256"]}
        for protocol_id, row in prompt_pins.items()
    }
    for index, spec in enumerate(specs):
        receipt_path, receipt_sha = _pinned_file(
            spec, base=base, label=f"response_receipts[{index}]"
        )
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
        _exact_keys(payload, required, context=f"response receipt {receipt_path}")
        validate_inference_receipt_schema(
            payload,
            context=f"response receipt {receipt_path}",
        )
        if payload["valid"] is not True:
            raise ValueError(f"invalid response receipt: {receipt_path}")
        if payload["status"] != "completed":
            raise ValueError(f"response receipt is not completed: {receipt_path}")
        request = _mapping(payload["request"], f"{receipt_path}.request")
        request_sha = _sha256(payload["request_sha256"], f"{receipt_path}.request_sha256")
        if canonical_json_sha256(request) != request_sha:
            raise ValueError(f"response request hash drift: {receipt_path}")
        tree_id = _identifier(request.get("tree_id"), "response tree_id")
        if tree_id != checkpoint_tree["tree_id"]:
            raise ValueError(f"response tree mismatch: {receipt_path}")
        model_id = _identifier(request.get("model_id"), "response model_id")
        if model_id in by_model:
            raise ValueError(f"duplicate response receipt for {tree_id}/{model_id}")
        expected_leaf = next(
            (leaf for leaf in checkpoint_tree["leaves"] if leaf["leaf_id"] == model_id), None
        )
        if expected_leaf is None:
            raise ValueError(f"response receipt names undeclared leaf {tree_id}/{model_id}")
        expected_request = {
            "cohort_id": cohort_id,
            "tree_id": tree_id,
            "model_id": model_id,
            "leaf_ordinal": expected_leaf["leaf_ordinal"],
            "base_model_id": registry.model_pins[model_task].model_id,
            "base_model_revision": registry.model_pins[model_task].revision,
            "model_task": model_task,
            "checkpoint": expected_leaf["checkpoint"],
            "checkpoint_artifact": checkpoint_tree["artifact_name"],
            "checkpoint_sha256": expected_leaf["checkpoint_sha256"],
            "training_summary": None,
            "training_summary_sha256": None,
            "run_list": checkpoint_tree["run_list"]["path"],
            "run_list_sha256": checkpoint_tree["run_list"]["sha256"],
            "ledger": checkpoint_tree["ledger"]["path"],
            "ledger_sha256": checkpoint_tree["ledger"]["sha256"],
            "truth_manifest": checkpoint_tree["truth_manifest"]["path"],
            "truth_manifest_sha256": checkpoint_tree["truth_manifest"]["sha256"],
            "registry": str(registry_path),
            "registry_sha256": registry_sha,
            "n_expected_responses": len(expected_coordinates),
        }
        for key, expected in expected_request.items():
            if expected is None:
                continue
            if request.get(key) != expected:
                raise ValueError(
                    f"{tree_id}/{model_id} response request {key} drift: "
                    f"expected {expected!r}, got {request.get(key)!r}"
                )
        request_training = Path(
            _identifier(request.get("training_summary"), "training_summary")
        ).resolve(strict=True)
        first_request_training_sha = _sha256(
            request.get("training_summary_sha256"), "training_summary_sha256"
        )
        if sha256_file(request_training) != first_request_training_sha:
            raise ValueError(f"{tree_id}/{model_id} training summary hash drift")
        if request.get("prompt_artifacts") != expected_prompt_pins:
            raise ValueError(f"{tree_id}/{model_id} prompt artifact request drift")
        if payload["prompt_artifacts"] != expected_prompt_pins:
            raise ValueError(f"{tree_id}/{model_id} prompt artifact receipt drift")
        if request.get("protocols") != _expected_protocol_request_rows(
            registry, prompts_by_protocol, model_task=model_task
        ):
            raise ValueError(f"{tree_id}/{model_id} protocol request declaration drift")
        provenance_raw = _mapping(payload["provenance"], f"{receipt_path}.provenance")
        try:
            provenance = ResponseProvenance(**dict(provenance_raw))
        except TypeError as exc:
            raise ValueError(f"invalid response provenance: {receipt_path}: {exc}") from exc
        if provenance.request_sha256 != request_sha:
            raise ValueError(f"{tree_id}/{model_id} response provenance request drift")
        for key in (
            "training_summary",
            "training_summary_sha256",
            "run_list",
            "run_list_sha256",
            "ledger",
            "ledger_sha256",
            "truth_manifest",
            "truth_manifest_sha256",
            "checkpoint",
            "checkpoint_artifact",
            "checkpoint_sha256",
            "registry",
            "registry_sha256",
            "prompt_artifacts",
        ):
            if provenance.to_dict()[key] != request.get(key):
                raise ValueError(f"{tree_id}/{model_id} response provenance {key} drift")
        response_path = Path(_identifier(payload["responses"], "responses path")).resolve(
            strict=True
        )
        response_sha = _sha256(payload["responses_sha256"], "responses_sha256")
        if sha256_file(response_path) != response_sha:
            raise ValueError(f"response artifact hash drift: {response_path}")
        responses = load_behavior_responses(response_path)
        audit = audit_response_grid(
            responses,
            registry=registry,
            prompts_by_protocol=prompts_by_protocol,
            cohort_id=cohort_id,
            tree_id=tree_id,
            model_id=model_id,
            base_model_id=registry.model_pins[model_task].model_id,
            base_model_revision=registry.model_pins[model_task].revision,
            model_task=model_task,
            request_sha256=request_sha,
            expected_provenance=provenance,
        )
        if not audit.valid or payload["audit"] != audit.to_dict():
            raise ValueError(f"response-grid audit drift: {tree_id}/{model_id}")
        by_model[model_id] = responses
        receipt_rows.append(
            {
                "model_id": model_id,
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "request_sha256": request_sha,
                "responses": str(response_path),
                "responses_sha256": response_sha,
            }
        )
    if set(by_model) != set(expected_leaves):
        raise ValueError(
            f"response receipt leaf set mismatch for {checkpoint_tree['tree_id']}: "
            f"missing={sorted(set(expected_leaves) - set(by_model))}, "
            f"extra={sorted(set(by_model) - set(expected_leaves))}"
        )
    receipt_by_model = {row["model_id"]: row for row in receipt_rows}
    return {
        "responses_by_model": {model: by_model[model] for model in expected_leaves},
        "receipt_rows": [receipt_by_model[model] for model in expected_leaves],
        "checkpoint_set_receipt": {
            "path": str(checkpoint_receipt_path),
            "sha256": checkpoint_receipt_sha,
        },
    }


def _expected_protocol_request_rows(
    registry: BehaviorProtocolRegistry,
    prompts_by_protocol: Mapping[str, Sequence[RenderedBehaviorPrompt]],
    *,
    model_task: str,
) -> list[dict[str, Any]]:
    rows = []
    for protocol_id, prompts in prompts_by_protocol.items():
        protocol = registry.protocol(protocol_id)
        rows.append(
            {
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
        )
    return rows


def _semantic_responses(
    responses_by_model: Mapping[str, Sequence[Any]],
    *,
    model_ids: Sequence[str],
    protocol_id: str,
    probe_id: str,
    prompts: Sequence[RenderedBehaviorPrompt],
    samples_per_prompt: int,
) -> list[SemanticResponse]:
    result: list[SemanticResponse] = []
    for model_id in model_ids:
        index = {response.key: response for response in responses_by_model[model_id]}
        for prompt in prompts:
            for sample_id in range(samples_per_prompt):
                key = (protocol_id, probe_id, prompt.prompt_id, sample_id)
                if key not in index:
                    raise ValueError(f"missing semantic response coordinate {model_id}/{key}")
                result.append(
                    SemanticResponse(
                        model_id=model_id,
                        prompt_id=prompt.prompt_id,
                        sample_id=f"draw-{sample_id}",
                        text=index[key].text,
                    )
                )
    return result


def _direct_attestation(
    raw: Any,
    *,
    base: Path,
    cohort_id: str,
    tree_id: str,
    completion_path: Path,
    completion_sha: str,
    checkpoint_tree: Mapping[str, Any],
) -> DirectDistanceSourceAttestation:
    value = _mapping(raw, f"{tree_id}.direct_attestation")
    expected_fields = set(DirectDistanceSourceAttestation.__dataclass_fields__)
    _exact_keys(value, expected_fields, context=f"{tree_id}.direct_attestation")
    normalized = dict(value)
    for key in ("summary", "ledger", "truth_manifest", "completion_receipt"):
        path = Path(_identifier(value[key], f"{tree_id}.{key}"))
        normalized[key] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    attestation = DirectDistanceSourceAttestation(**normalized)
    if attestation.cohort_id != cohort_id or attestation.tree_id != tree_id:
        raise ValueError(f"{tree_id} direct cohort/tree binding drift")
    if attestation.metric != "cosine":
        raise ValueError(f"{tree_id} direct metric must be cosine")
    if Path(attestation.completion_receipt).resolve(strict=True) != completion_path:
        raise ValueError(f"{tree_id} direct completion receipt path drift")
    if attestation.completion_receipt_sha256 != completion_sha:
        raise ValueError(f"{tree_id} direct completion receipt hash drift")
    for field in ("ledger", "truth_manifest"):
        expected = checkpoint_tree[field]
        if Path(getattr(attestation, field)).resolve(strict=True) != Path(expected["path"]):
            raise ValueError(f"{tree_id} direct {field} path drift")
        if getattr(attestation, f"{field}_sha256") != expected["sha256"]:
            raise ValueError(f"{tree_id} direct {field} hash drift")
    return attestation


def _dl_summary_fields(
    summary: MetaAnalysisSummary,
    *,
    cell: CellSpec,
    study_id: str,
    model_family: str,
    condition: str,
    cohort_id: str,
) -> dict[str, Any]:
    payload = summary.to_dict()
    random = payload["random_effect"]
    fixed = payload["fixed_effect"]
    heterogeneity = payload["heterogeneity"]
    return {
        "cell_id": cell.cell_id,
        "study_id": study_id,
        "model_family": model_family,
        "condition": condition,
        "cohort_id": cohort_id,
        "protocol_id": cell.protocol_id,
        "probe_id": cell.probe_id,
        "role": cell.role,
        "routes": ",".join(cell.routes),
        "n_studies": payload["n_studies"],
        "n_pairs_min": payload["n_pairs_min"],
        "n_pairs_max": payload["n_pairs_max"],
        "n_pairs_total": payload["n_pairs_total"],
        "random_effect_r": random["correlation"],
        "random_effect_ci_low_r": random["ci_low_r"],
        "random_effect_ci_high_r": random["ci_high_r"],
        "random_effect_p_value": random["p_value"],
        "fixed_effect_r": fixed["correlation"],
        "equal_weight_r": payload["equal_weight_r"],
        "q": heterogeneity["q"],
        "q_df": heterogeneity["df"],
        "q_p_value": heterogeneity["q_p_value"],
        "tau2": heterogeneity["tau2"],
        "i2": heterogeneity["i2"],
        "i2_percent": heterogeneity["i2_percent"],
        "study_inputs_sha256": payload["study_inputs_sha256"],
    }


def _cell_specs(raw: Any, *, registry: BehaviorProtocolRegistry) -> tuple[CellSpec, ...]:
    values = _list(raw, "cells")
    if not values:
        raise ValueError("study must declare at least one cell")
    result: list[CellSpec] = []
    for index, value in enumerate(values):
        row = _mapping(value, f"cells[{index}]")
        _exact_keys(
            row,
            {"cell_id", "protocol_id", "probe_id", "role", "routes", "eos_depth"},
            context=f"cells[{index}]",
        )
        cell_id = _safe_identifier(row["cell_id"], f"cells[{index}].cell_id")
        protocol_id = _identifier(row["protocol_id"], f"cells[{index}].protocol_id")
        probe_id = _identifier(row["probe_id"], f"cells[{index}].probe_id")
        protocol = registry.protocol(protocol_id)
        if probe_id not in protocol.probe_ids:
            raise ValueError(f"cell {cell_id!r} probe is not in protocol {protocol_id!r}")
        role = _identifier(row["role"], f"cells[{index}].role")
        route_values = _list(row["routes"], f"cells[{index}].routes")
        routes = tuple(_identifier(route, f"cells[{index}].routes") for route in route_values)
        if not routes or len(set(routes)) != len(routes) or not set(routes) <= ALLOWED_ROUTES:
            raise ValueError(f"cell {cell_id!r} has invalid routes: {routes}")
        if role == "paper" and ({"table3", "table7"} <= set(routes)) is False:
            raise ValueError(f"paper cell {cell_id!r} must route to both Table 3 and Table 7")
        eos_depth = row["eos_depth"]
        if not isinstance(eos_depth, bool):
            raise ValueError(f"cells[{index}].eos_depth must be boolean")
        result.append(CellSpec(cell_id, protocol_id, probe_id, role, routes, eos_depth))
    ids = [cell.cell_id for cell in result]
    endpoints = [(cell.protocol_id, cell.probe_id) for cell in result]
    if len(set(ids)) != len(ids):
        raise ValueError("study cell IDs must be unique")
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("study protocol/probe endpoints must be unique")
    return tuple(result)


def _tree_specs(raw: Any, *, base: Path, strict: bool) -> dict[str, dict[str, Any]]:
    values = _list(raw, "trees")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        row = _mapping(value, f"trees[{index}]")
        _exact_keys(
            row,
            {"tree_id", "truth_manifest", "response_receipts", "direct_attestation"},
            context=f"trees[{index}]",
        )
        tree_id = _identifier(row["tree_id"], f"trees[{index}].tree_id")
        if tree_id in result:
            raise ValueError(f"duplicate study tree {tree_id!r}")
        truth_path, truth_sha = _pinned_file(
            row["truth_manifest"], base=base, label=f"trees[{index}].truth_manifest"
        )
        result[tree_id] = {
            "truth_manifest": {"path": str(truth_path), "sha256": truth_sha},
            "response_receipts": row["response_receipts"],
            "direct_attestation": row["direct_attestation"],
        }
    if strict and tuple(result) != ALL_TREE_IDS:
        raise ValueError("production study must declare exact ordered trees 001..050")
    if not strict and len(result) < 3:
        raise ValueError("behavior meta-analysis requires at least three declared trees")
    return result


def _completion_receipt(
    raw: Any,
    *,
    base: Path,
    cohort_id: str,
    strict: bool,
) -> tuple[Path, str, dict[str, Any]]:
    path, digest = _pinned_file(raw, base=base, label="completion_receipt")
    payload = _json_object(path)
    if strict:
        validate_strict_training_completion(payload, cohort_id=cohort_id)
    elif payload.get("valid") is not True:
        raise ValueError("non-paper test completion receipt must still be explicitly valid")
    return path, digest, payload


def _tree_ids(raw: Any, *, strict: bool) -> tuple[str, ...]:
    values = tuple(_identifier(value, "tree_id") for value in _list(raw, "tree_ids"))
    if not values or len(set(values)) != len(values):
        raise ValueError("tree_ids must be non-empty and unique")
    if strict and values != ALL_TREE_IDS:
        raise ValueError("production checkpoint set must declare exact ordered trees 001..050")
    return values


def _encoder_snapshot_sha256(path: Path) -> str:
    root = path.resolve(strict=True)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"encoder snapshot is empty: {root}")
    rows = []
    for item in files:
        resolved = item.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"encoder snapshot entry is not a file: {item}")
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "symlink_target": os.readlink(item) if item.is_symlink() else None,
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return canonical_json_sha256(rows)


def _output_inventory(root: Path, *, excluded: set[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path in excluded or path.name.startswith("."):
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return rows


def _pinned_file(raw: Any, *, base: Path, label: str) -> tuple[Path, str]:
    spec = _mapping(raw, label)
    if not {"path", "sha256"} <= set(spec):
        raise ValueError(f"{label} requires path and sha256")
    path = Path(_identifier(spec["path"], f"{label}.path"))
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    digest = _sha256(spec["sha256"], f"{label}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(f"{label} SHA256 mismatch: expected {digest}, observed {observed}")
    return path, digest


def _directory(raw: Any, *, base: Path, label: str) -> Path:
    path = Path(_identifier(raw, label))
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label} directory: {path}")
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    _write_text_atomic(path, buffer.getvalue())


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_object(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            out[key] = value
        return out

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
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


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be non-empty text without surrounding whitespace")
    return value


def _safe_identifier(value: Any, context: str) -> str:
    text = _identifier(value, context)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{context} must be a safe single path component")
    return text


def _sha256(value: Any, context: str) -> str:
    text = _identifier(value, context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return text


def _model_task(value: Any) -> str:
    task = _identifier(value, "model_task")
    if task not in {"causal_lm", "seq2seq"}:
        raise ValueError("model_task must be exactly 'causal_lm' or 'seq2seq'")
    return task


def _reject_legacy_output(path: Path) -> None:
    if any(part.lower() == "ellmtrees" for part in path.parts):
        raise ValueError(f"refusing to write corrected behavior outputs under legacy ELLMTrees: {path}")


__all__ = [
    "CELL_RECEIPT_SCHEMA",
    "CHECKPOINT_SET_INPUT_SCHEMA",
    "CHECKPOINT_SET_RECEIPT_SCHEMA",
    "ENCODER_INPUT_SCHEMA",
    "ENCODER_RECEIPT_SCHEMA",
    "STUDY_INPUT_SCHEMA",
    "STUDY_RECEIPT_SCHEMA",
    "TABLE_SUMMARY_COLUMNS",
    "build_behavior_study",
    "build_checkpoint_set_receipt",
    "build_encoder_receipt",
    "load_cell_receipt",
]
