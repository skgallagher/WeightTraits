"""Fail-closed production attestations for native direct-weight analyses.

The ordinary direct analyzer writes useful scientific artifacts, but a path to
an ``npz`` file is not provenance.  This module adds two deliberately separate
receipts:

* a producer receipt binds the complete training contract, immutable source
  checkpoints, analysis parameters, ordering, and every output byte; and
* a replay receipt records an independent, full recomputation of every layer
  and metric from the pinned checkpoints.

Downstream paper builders should accept only a *pinned replay receipt*.  They
must not accept a bare distance cube, summary, rollup, or producer receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
from typing import Any

import numpy as np

from weighttraits.analysis.completion import audit_training_tree_completion
from weighttraits.analysis import direct as direct_engine
from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    canonical_sha256,
    canonical_tree_id,
    sha256,
    validate_strict_training_completion,
)
from weighttraits.training.ledger import latest_status_by_node, load_ledger_events


DIRECT_ANALYSIS_RECEIPT_SCHEMA = "weighttraits.direct_analysis_receipt.v1"
DIRECT_ANALYSIS_REPLAY_SCHEMA = "weighttraits.direct_analysis_replay_receipt.v1"
DIRECT_ANALYSIS_RECEIPT_NAME = "direct_analysis_receipt.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_CONTRACT_FILES = (
    "assignment_summary",
    "config",
    "formats",
    "registry",
)


def produce_attested_direct_analysis(
    *,
    training_summary_path: str | Path,
    completion_receipt_path: str | Path,
    stage_manifest_path: str | Path,
    cohort_id: str,
    tree_id: str,
    out_dir: str | Path,
    artifact: str,
    metrics: Sequence[str],
    path_base: str | Path = ".",
    chunk_size: int = 1_000_000,
    eps: float = 1e-3,
    layer: str | int | None = None,
    aggregate: str = "mean",
    producer_code_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Run one direct analysis into a fresh directory and publish its receipt last."""

    root = Path(out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"attested direct analysis requires an empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    normalized_metrics = _normalize_metrics(metrics)
    source = _capture_source_contract(
        training_summary_path=training_summary_path,
        completion_receipt_path=completion_receipt_path,
        stage_manifest_path=stage_manifest_path,
        cohort_id=cohort_id,
        tree_id=tree_id,
        artifact=artifact,
        path_base=path_base,
        producer_code_paths=producer_code_paths,
    )
    parameters = _parameters(
        artifact=artifact,
        metrics=normalized_metrics,
        chunk_size=chunk_size,
        eps=eps,
        layer=layer,
        aggregate=aggregate,
    )

    summary = direct_engine.analyze_training_ledger_direct(
        source["ledger"]["path"],
        truth_manifest=source["truth_manifest"]["path"],
        out_dir=root,
        artifact=artifact,
        metrics=list(normalized_metrics),
        path_base=source["path_base"],
        chunk_size=chunk_size,
        eps=eps,
        layer=layer,
        aggregate=aggregate,
    )
    receipt_path = root / DIRECT_ANALYSIS_RECEIPT_NAME
    summary["attestation_receipt"] = str(receipt_path)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    post_source = _capture_source_contract(
        training_summary_path=training_summary_path,
        completion_receipt_path=completion_receipt_path,
        stage_manifest_path=stage_manifest_path,
        cohort_id=cohort_id,
        tree_id=tree_id,
        artifact=artifact,
        path_base=path_base,
        producer_code_paths=producer_code_paths,
    )
    if canonical_sha256(post_source) != canonical_sha256(source):
        raise ValueError("direct-analysis source inputs changed while analysis was running")

    return attest_existing_direct_analysis(
        analysis_root=root,
        source=source,
        parameters=parameters,
        receipt_path=receipt_path,
    )


def attest_existing_direct_analysis(
    *,
    analysis_root: str | Path,
    source: Mapping[str, Any],
    parameters: Mapping[str, Any],
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Attest existing native outputs after validating their complete semantics.

    This entry point exists so expensive already-computed outputs can be brought
    under the contract.  It intentionally accepts a previously captured source
    object only; callers should normally use :func:`build_direct_source_contract`
    rather than constructing that object by hand.  A subsequent replay receipt
    is still required before a paper builder may consume the result.
    """

    root = Path(analysis_root).resolve()
    destination = (
        (root / DIRECT_ANALYSIS_RECEIPT_NAME)
        if receipt_path is None
        else Path(receipt_path).resolve()
    )
    if destination != root / DIRECT_ANALYSIS_RECEIPT_NAME:
        raise ValueError(
            f"producer receipt must be {root / DIRECT_ANALYSIS_RECEIPT_NAME}, got {destination}"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite direct-analysis receipt: {destination}")
    _validate_captured_source(source)
    normalized_parameters = _validate_parameters(parameters)
    semantic = _validate_analysis_outputs(root, source=source, parameters=normalized_parameters)
    outputs = _pin_output_tree(root, excluded={destination})
    expected_relpaths = set(semantic["expected_output_relpaths"])
    observed_relpaths = {str(row["relative_path"]) for row in outputs}
    if observed_relpaths != expected_relpaths:
        raise ValueError(
            "direct-analysis output set differs from the exact native contract: "
            f"missing={sorted(expected_relpaths - observed_relpaths)}, "
            f"unexpected={sorted(observed_relpaths - expected_relpaths)}"
        )

    payload = {
        "schema": DIRECT_ANALYSIS_RECEIPT_SCHEMA,
        "producer": "weighttraits",
        "valid": True,
        "cohort_id": source["cohort_id"],
        "tree_id": source["tree_id"],
        "analysis_root": str(root),
        "parameters": normalized_parameters,
        "model_ids": semantic["model_ids"],
        "model_ids_sha256": canonical_sha256(semantic["model_ids"]),
        "layer_names": semantic["layer_names"],
        "layer_names_sha256": canonical_sha256(semantic["layer_names"]),
        "metric_shapes": semantic["metric_shapes"],
        "source": dict(source),
        "source_sha256": canonical_sha256(source),
        "outputs": outputs,
        "outputs_sha256": canonical_sha256(outputs),
        "runtime": _runtime_versions(),
        "receipt_publish_order": "source reverified, outputs hashed, receipt atomically replaced last",
    }
    _atomic_write_json(destination, payload)
    return payload


def build_direct_source_contract(
    *,
    training_summary_path: str | Path,
    completion_receipt_path: str | Path,
    stage_manifest_path: str | Path,
    cohort_id: str,
    tree_id: str,
    artifact: str,
    path_base: str | Path = ".",
    producer_code_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Capture and validate the immutable source side of an analysis receipt."""

    return _capture_source_contract(
        training_summary_path=training_summary_path,
        completion_receipt_path=completion_receipt_path,
        stage_manifest_path=stage_manifest_path,
        cohort_id=cohort_id,
        tree_id=tree_id,
        artifact=artifact,
        path_base=path_base,
        producer_code_paths=producer_code_paths,
    )


def verify_direct_analysis_receipt(
    receipt_path: str | Path,
    *,
    expected_sha256: str | None = None,
    replay: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Deep-verify a producer receipt and optionally recompute its complete cube."""

    receipt_file = Path(receipt_path).resolve()
    _verify_expected_digest(receipt_file, expected_sha256, label="direct-analysis receipt")
    payload = _load_json_object(receipt_file, label="direct-analysis receipt")
    if payload.get("schema") != DIRECT_ANALYSIS_RECEIPT_SCHEMA or payload.get("valid") is not True:
        raise ValueError(f"not a valid {DIRECT_ANALYSIS_RECEIPT_SCHEMA} receipt: {receipt_file}")
    root = Path(_required_text(payload, "analysis_root", context="direct receipt")).resolve()
    if receipt_file != root / DIRECT_ANALYSIS_RECEIPT_NAME:
        raise ValueError("direct-analysis receipt is not at the canonical analysis-root path")

    source = _required_mapping(payload, "source", context="direct receipt")
    parameters = _validate_parameters(
        _required_mapping(payload, "parameters", context="direct receipt")
    )
    if payload.get("source_sha256") != canonical_sha256(source):
        raise ValueError("direct-analysis source canonical hash mismatch")
    _validate_captured_source(source)

    rebuilt = _capture_source_contract(
        training_summary_path=source["training_summary"]["path"],
        completion_receipt_path=source["completion_receipt"]["path"],
        stage_manifest_path=source["stage_manifest"]["path"],
        cohort_id=str(source["cohort_id"]),
        tree_id=str(source["tree_id"]),
        artifact=str(parameters["artifact"]),
        path_base=source["path_base"],
        producer_code_paths=[row["path"] for row in source["producer_code"]],
    )
    if rebuilt != source:
        raise ValueError("direct-analysis source contract no longer matches pinned inputs")

    declared_outputs = payload.get("outputs")
    if not isinstance(declared_outputs, list) or not declared_outputs:
        raise ValueError("direct-analysis receipt requires a non-empty outputs list")
    if payload.get("outputs_sha256") != canonical_sha256(declared_outputs):
        raise ValueError("direct-analysis output manifest canonical hash mismatch")
    observed_outputs = _pin_output_tree(root, excluded={receipt_file})
    if observed_outputs != declared_outputs:
        raise ValueError("direct-analysis output files differ from the pinned manifest")

    semantic = _validate_analysis_outputs(root, source=source, parameters=parameters)
    if payload.get("model_ids") != semantic["model_ids"]:
        raise ValueError("direct-analysis receipt model ordering mismatch")
    if payload.get("layer_names") != semantic["layer_names"]:
        raise ValueError("direct-analysis receipt layer ordering mismatch")
    if payload.get("metric_shapes") != semantic["metric_shapes"]:
        raise ValueError("direct-analysis receipt metric shape mismatch")
    if payload.get("model_ids_sha256") != canonical_sha256(semantic["model_ids"]):
        raise ValueError("direct-analysis receipt model ordering hash mismatch")
    if payload.get("layer_names_sha256") != canonical_sha256(semantic["layer_names"]):
        raise ValueError("direct-analysis receipt layer ordering hash mismatch")

    result = {
        "valid": True,
        "receipt": str(receipt_file),
        "receipt_sha256": sha256(receipt_file),
        "cohort_id": source["cohort_id"],
        "tree_id": source["tree_id"],
        "artifact": parameters["artifact"],
        "metrics": parameters["metrics"],
        "model_ids": semantic["model_ids"],
        "layer_names": semantic["layer_names"],
        "metric_shapes": semantic["metric_shapes"],
        "source_sha256": payload["source_sha256"],
        "outputs_sha256": payload["outputs_sha256"],
        "replayed": False,
    }
    if replay:
        comparisons = _replay_distance_cube(
            root,
            source=source,
            parameters=parameters,
            rtol=rtol,
            atol=atol,
        )
        result.update(
            {
                "replayed": True,
                "rtol": float(rtol),
                "atol": float(atol),
                "exact": rtol == 0.0 and atol == 0.0,
                "comparisons": comparisons,
            }
        )
    return result


def write_direct_analysis_replay_receipt(
    direct_receipt_path: str | Path,
    output_path: str | Path,
    *,
    expected_direct_sha256: str | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Independently replay a direct cube and atomically publish a second receipt."""

    direct_path = Path(direct_receipt_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite direct replay receipt: {output}")
    direct_payload = _load_json_object(direct_path, label="direct-analysis receipt")
    analysis_root = Path(str(direct_payload.get("analysis_root", ""))).resolve()
    if output == analysis_root or analysis_root in output.parents:
        raise ValueError("replay receipt must be outside the producer analysis root")

    result = verify_direct_analysis_receipt(
        direct_path,
        expected_sha256=expected_direct_sha256,
        replay=True,
        rtol=rtol,
        atol=atol,
    )
    if not result["comparisons"] or not all(
        row.get("matches") is True for row in result["comparisons"]
    ):
        raise ValueError("direct-analysis replay did not match every metric and layer")

    verifier_code = [_artifact_pin(path) for path in _default_code_paths()]
    source = _required_mapping(direct_payload, "source", context="direct receipt")
    payload = {
        "schema": DIRECT_ANALYSIS_REPLAY_SCHEMA,
        "producer": "weighttraits",
        "valid": True,
        "cohort_id": result["cohort_id"],
        "tree_id": result["tree_id"],
        "artifact": result["artifact"],
        "metrics": result["metrics"],
        "model_ids_sha256": canonical_sha256(result["model_ids"]),
        "layer_names_sha256": canonical_sha256(result["layer_names"]),
        "direct_analysis_receipt": _file_pin(direct_path),
        "stage_manifest": dict(source["stage_manifest"]),
        "verifier_code": verifier_code,
        "verification": {
            "replayed": True,
            "all_metrics_and_layers_compared": True,
            "exact": result["exact"],
            "rtol": result["rtol"],
            "atol": result["atol"],
            "comparisons": result["comparisons"],
            "source_sha256": result["source_sha256"],
            "outputs_sha256": result["outputs_sha256"],
        },
        "runtime": _runtime_versions(),
        "receipt_publish_order": "direct receipt reverified, cube replayed, receipt atomically replaced last",
    }
    _atomic_write_json(output, payload)
    return payload


def validate_pinned_direct_replay_receipt(
    raw_pin: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    expected_cohort_id: str | None = None,
    expected_tree_id: str | None = None,
    expected_artifact: str | None = None,
    expected_metric: str | None = None,
    deep: bool = False,
) -> dict[str, Any]:
    """Consumer interface for Table 4 inventory builders.

    The caller supplies the replay receipt as a mandatory ``{path, sha256}``
    pin.  The fast path validates both receipt hashes and every direct output
    hash without re-reading all checkpoint shards.  ``deep=True`` additionally
    re-hashes source checkpoints and replays the cube again.
    """

    base = Path(base_dir).resolve()
    replay_path = _resolve_path(
        _required_text(raw_pin, "path", context="direct replay pin"), base
    )
    replay_digest = _required_digest(raw_pin, "sha256", context="direct replay pin")
    _verify_expected_digest(replay_path, replay_digest, label="direct replay receipt")
    replay_payload = _load_json_object(replay_path, label="direct replay receipt")
    if (
        replay_payload.get("schema") != DIRECT_ANALYSIS_REPLAY_SCHEMA
        or replay_payload.get("valid") is not True
    ):
        raise ValueError(f"not a valid {DIRECT_ANALYSIS_REPLAY_SCHEMA}: {replay_path}")

    verification = _required_mapping(
        replay_payload, "verification", context="direct replay receipt"
    )
    comparisons = verification.get("comparisons")
    if (
        verification.get("replayed") is not True
        or verification.get("all_metrics_and_layers_compared") is not True
        or verification.get("exact") is not True
        or verification.get("rtol") != 0.0
        or verification.get("atol") != 0.0
        or not isinstance(comparisons, list)
        or not comparisons
        or not all(isinstance(row, Mapping) and row.get("matches") is True for row in comparisons)
    ):
        raise ValueError("direct replay receipt is not an exact complete-cube replay")

    direct_pin = _required_mapping(
        replay_payload, "direct_analysis_receipt", context="direct replay receipt"
    )
    direct_path = Path(_required_text(direct_pin, "path", context="direct receipt pin"))
    direct_digest = _required_digest(direct_pin, "sha256", context="direct receipt pin")
    _verify_expected_digest(direct_path, direct_digest, label="direct-analysis receipt")
    direct_payload = _load_json_object(direct_path, label="direct-analysis receipt")
    _validate_shallow_direct_payload(direct_path, direct_payload)

    for pin in replay_payload.get("verifier_code", []):
        _verify_artifact_pin(pin, label="direct replay verifier code")
    _verify_artifact_pin(replay_payload.get("stage_manifest"), label="stage manifest")

    if deep:
        result = verify_direct_analysis_receipt(
            direct_path,
            expected_sha256=direct_digest,
            replay=True,
            rtol=0.0,
            atol=0.0,
        )
        if not all(row["matches"] for row in result["comparisons"]):
            raise ValueError("deep direct replay verification failed")

    _require_expected_identity(
        replay_payload,
        expected_cohort_id=expected_cohort_id,
        expected_tree_id=expected_tree_id,
        expected_artifact=expected_artifact,
        expected_metric=expected_metric,
    )
    return {
        "replay_receipt": str(replay_path),
        "replay_receipt_sha256": replay_digest,
        "direct_receipt": str(direct_path),
        "direct_receipt_sha256": direct_digest,
        "cohort_id": replay_payload["cohort_id"],
        "tree_id": replay_payload["tree_id"],
        "artifact": replay_payload["artifact"],
        "metrics": list(replay_payload["metrics"]),
        "direct_payload": direct_payload,
    }


def _capture_source_contract(
    *,
    training_summary_path: str | Path,
    completion_receipt_path: str | Path,
    stage_manifest_path: str | Path,
    cohort_id: str,
    tree_id: str,
    artifact: str,
    path_base: str | Path,
    producer_code_paths: Sequence[str | Path] | None,
) -> dict[str, Any]:
    base = Path(path_base).resolve()
    summary_path = _resolve_path(training_summary_path, base)
    completion_path = _resolve_path(completion_receipt_path, base)
    stage_path = _resolve_path(stage_manifest_path, base)
    canonical_id = canonical_tree_id(tree_id)
    if not cohort_id.strip():
        raise ValueError("direct source contract requires a nonempty cohort_id")
    if artifact not in {"model", "merged", "adapter_chain"}:
        raise ValueError(f"unsupported direct artifact mode: {artifact!r}")

    summary = _load_json_object(summary_path, label="training summary")
    row = _validate_training_summary(summary, cohort_id=cohort_id, tree_id=canonical_id)
    completion = _load_json_object(completion_path, label="completion receipt")
    if completion.get("cohort_id") != cohort_id:
        raise ValueError(
            f"completion receipt cohort_id must be {cohort_id!r}, "
            f"got {completion.get('cohort_id')!r}"
        )
    validate_strict_training_completion(completion, cohort_id=cohort_id)
    completion_row = next(
        item for item in completion["rows"] if canonical_tree_id(item["tree_id"]) == canonical_id
    )
    if completion_row["n_runs"] != row["n_runs"]:
        raise ValueError("training summary and completion receipt disagree on per-tree n_runs")

    contract_files = {
        field: _declared_summary_pin(summary, field, base=base)
        for field in _TOP_LEVEL_CONTRACT_FILES
    }
    report = _declared_row_pin(row, "report", base=base)
    run_list = _declared_row_pin(row, "run_list", base=base)
    truth_manifest = _declared_row_pin(row, "manifest", base=base)
    ledger_path = _resolve_path(_required_text(row, "ledger", context=canonical_id), base)
    ledger = _file_pin(ledger_path)
    stage_manifest = _file_pin(stage_path)

    audit = audit_training_tree_completion(
        run_list["path"],
        ledger=ledger["path"],
        path_base=base,
    )
    if (
        not audit.valid
        or audit.n_runs != row["n_runs"]
        or audit.n_ok_nodes != audit.n_runs
        or audit.n_terminal_nodes != audit.n_runs
        or audit.n_failed_nodes != 0
        or audit.n_missing_nodes != 0
        or audit.n_existing_artifacts != audit.n_expected_artifacts
        or audit.status_counts != {"completed": audit.n_runs}
        or audit.issues
    ):
        raise ValueError("selected tree does not pass a clean, completed run-list/ledger audit")

    checkpoint_contract = _checkpoint_contract(
        ledger_path=ledger_path,
        truth_manifest_path=Path(truth_manifest["path"]),
        artifact=artifact,
        path_base=base,
    )
    code_paths = list(producer_code_paths or _default_code_paths())
    if not code_paths:
        raise ValueError("direct source contract requires producer code pins")
    producer_code = [_artifact_pin(_resolve_path(path, base)) for path in code_paths]
    if len({row["path"] for row in producer_code}) != len(producer_code):
        raise ValueError("direct source contract contains duplicate producer code paths")

    return {
        "cohort_id": cohort_id,
        "tree_id": canonical_id,
        "path_base": str(base),
        "training_summary": _file_pin(summary_path),
        "completion_receipt": _file_pin(completion_path),
        "stage_manifest": stage_manifest,
        "training_contract_files": contract_files,
        "tree_report": report,
        "run_list": run_list,
        "ledger": ledger,
        "truth_manifest": truth_manifest,
        "checkpoints": checkpoint_contract,
        "producer_code": producer_code,
    }


def _validate_training_summary(
    summary: Mapping[str, Any], *, cohort_id: str, tree_id: str
) -> Mapping[str, Any]:
    expected = {"n_trees": 50, "n_runs": 641, "n_errors": 0, "n_warnings": 0}
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(
                f"training summary requires {field}={value}, got {summary.get(field)!r}"
            )
    config = _required_text(summary, "config", context="training summary")
    if Path(config).stem != cohort_id:
        raise ValueError(
            f"training summary config stem {Path(config).stem!r} differs from cohort_id "
            f"{cohort_id!r}"
        )
    trees = summary.get("trees")
    if not isinstance(trees, list) or len(trees) != 50:
        raise ValueError("training summary must contain exactly 50 tree rows")
    by_tree: dict[str, Mapping[str, Any]] = {}
    n_runs = 0
    for raw in trees:
        if not isinstance(raw, Mapping):
            raise ValueError("training summary tree rows must be objects")
        canonical = canonical_tree_id(raw.get("tree_id"))
        if canonical in by_tree:
            raise ValueError(f"training summary duplicates {canonical}")
        if raw.get("valid") is not True or raw.get("n_errors") != 0 or raw.get("n_warnings") != 0:
            raise ValueError(f"training summary tree row is not clean: {canonical}")
        count = raw.get("n_runs")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"training summary has invalid n_runs for {canonical}")
        n_runs += count
        by_tree[canonical] = raw
    if set(by_tree) != set(ALL_TREE_IDS) or n_runs != 641:
        raise ValueError("training summary must contain exact trees 001..050 totaling 641 runs")
    return by_tree[tree_id]


def _checkpoint_contract(
    *,
    ledger_path: Path,
    truth_manifest_path: Path,
    artifact: str,
    path_base: Path,
) -> dict[str, Any]:
    events = latest_status_by_node(load_ledger_events(ledger_path))
    model_ids = direct_engine._manifest_leaf_ids(truth_manifest_path)
    if not model_ids:
        raise ValueError("truth manifest has no trained leaves")
    artifacts: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    if artifact == "adapter_chain":
        lineages = direct_engine._manifest_lineages(truth_manifest_path)
        for model_id in model_ids:
            artifact_ids = []
            for node_id in lineages[model_id]:
                if node_id == "root":
                    continue
                artifact_id = f"{node_id}:adapter"
                artifact_ids.append(artifact_id)
                if artifact_id not in artifacts:
                    raw = direct_engine._event_artifact(events, node_id, "adapter")
                    artifacts[artifact_id] = _artifact_pin(_resolve_path(raw, path_base))
            if not artifact_ids:
                raise ValueError(f"adapter-chain model {model_id!r} has no source adapters")
            bindings.append({"model_id": model_id, "artifact_ids": artifact_ids})
    else:
        for model_id in model_ids:
            artifact_id = f"{model_id}:{artifact}"
            raw = direct_engine._event_artifact(events, model_id, artifact)
            artifacts[artifact_id] = _artifact_pin(_resolve_path(raw, path_base))
            bindings.append({"model_id": model_id, "artifact_ids": [artifact_id]})
    return {
        "artifact_mode": artifact,
        "model_bindings": bindings,
        "artifacts": [
            {"artifact_id": artifact_id, **artifacts[artifact_id]}
            for artifact_id in sorted(artifacts)
        ],
        "artifacts_sha256": canonical_sha256(
            [
                {"artifact_id": artifact_id, **artifacts[artifact_id]}
                for artifact_id in sorted(artifacts)
            ]
        ),
    }


def _validate_captured_source(source: Mapping[str, Any]) -> None:
    required_pins = (
        "training_summary",
        "completion_receipt",
        "stage_manifest",
        "tree_report",
        "run_list",
        "ledger",
        "truth_manifest",
    )
    for field in required_pins:
        _verify_artifact_pin(source.get(field), label=field)
    contract_files = _required_mapping(
        source, "training_contract_files", context="direct source"
    )
    if set(contract_files) != set(_TOP_LEVEL_CONTRACT_FILES):
        raise ValueError("direct source training contract file set is incomplete")
    for label, pin in contract_files.items():
        _verify_artifact_pin(pin, label=f"training contract {label}")
    code = source.get("producer_code")
    if not isinstance(code, list) or not code:
        raise ValueError("direct source requires producer_code pins")
    for pin in code:
        _verify_artifact_pin(pin, label="producer code")
    checkpoints = _required_mapping(source, "checkpoints", context="direct source")
    artifacts = checkpoints.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("direct source requires checkpoint artifact pins")
    for pin in artifacts:
        _verify_artifact_pin(pin, label=f"checkpoint {pin.get('artifact_id', '?')}")
    if checkpoints.get("artifacts_sha256") != canonical_sha256(artifacts):
        raise ValueError("checkpoint artifact manifest canonical hash mismatch")


def _validate_analysis_outputs(
    root: Path, *, source: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = list(parameters["metrics"])
    expected = {
        "aggregate_recovery.json",
        "direct_distance_layers.npz",
        "distance_audit.json",
        "layers.json",
        "models.json",
        "summary.json",
        "truth_manifest.newick",
    }
    for metric in metrics:
        expected.update(
            {
                f"atteson_margin_{metric}.json",
                f"branch_ordering_{metric}.json",
                f"distance_matrix_{metric}.npy",
                f"four_point_additivity_{metric}.json",
                f"score_{metric}.json",
                f"tree_{metric}.audit.json",
                f"tree_{metric}.newick",
            }
        )
    for relpath in expected:
        path = root / relpath
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe direct-analysis output: {path}")

    layer_names = _load_json_list(root / "layers.json", label="direct layer ordering")
    model_ids = _load_json_list(root / "models.json", label="direct model ordering")
    if model_ids != [row["model_id"] for row in source["checkpoints"]["model_bindings"]]:
        raise ValueError("direct output model ordering differs from checkpoint bindings")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("direct output model ordering contains duplicates")
    if len(set(layer_names)) != len(layer_names) or not layer_names:
        raise ValueError("direct output layer ordering is empty or contains duplicates")

    with np.load(root / "direct_distance_layers.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(metrics):
            raise ValueError("direct distance cube metric set differs from parameters")
        cubes = {metric: np.asarray(archive[metric], dtype=np.float64) for metric in metrics}
    metric_shapes: dict[str, list[int]] = {}
    for metric, cube in cubes.items():
        expected_shape = (len(layer_names), len(model_ids), len(model_ids))
        if cube.shape != expected_shape:
            raise ValueError(
                f"direct cube {metric!r} shape {cube.shape} differs from {expected_shape}"
            )
        if not np.all(np.isfinite(cube)):
            raise ValueError(f"direct cube {metric!r} contains non-finite values")
        if not np.array_equal(cube, np.swapaxes(cube, 1, 2)):
            raise ValueError(f"direct cube {metric!r} is not exactly symmetric")
        diagonal = np.diagonal(cube, axis1=1, axis2=2)
        if not np.array_equal(diagonal, np.zeros_like(diagonal)):
            raise ValueError(f"direct cube {metric!r} has a nonzero diagonal")
        metric_shapes[metric] = list(cube.shape)

    audit = _load_json_object(root / "distance_audit.json", label="distance audit")
    expected_audit = {
        "analysis_engine": "direct",
        "artifact": parameters["artifact"],
        "metrics": sorted(metrics),
        "chunk_size": parameters["chunk_size"],
        "eps": parameters["eps"],
        "n_models": len(model_ids),
        "n_layers": len(layer_names),
        "model_ids": model_ids,
    }
    for field, value in expected_audit.items():
        if audit.get(field) != value:
            raise ValueError(f"distance audit field {field!r} differs from receipt parameters")
    if [row.get("name") for row in audit.get("layers", [])] != layer_names:
        raise ValueError("distance audit layer ordering mismatch")

    summary = _load_json_object(root / "summary.json", label="direct summary")
    if summary.get("schema") != direct_engine.DIRECT_ANALYSIS_SUMMARY_SCHEMA:
        raise ValueError("direct summary schema is missing or obsolete")
    summary_expected = {
        "analysis_engine": "direct",
        "artifact": parameters["artifact"],
        "ledger": source["ledger"]["path"],
        "truth_manifest": source["truth_manifest"]["path"],
        "distance_layers": str(root / "direct_distance_layers.npz"),
        "n_models": len(model_ids),
        "n_layers": len(layer_names),
        "model_ids": model_ids,
        "metrics": sorted(metrics),
        "parameters": dict(parameters),
        "attestation_receipt": str(root / DIRECT_ANALYSIS_RECEIPT_NAME),
    }
    for field, value in summary_expected.items():
        if summary.get(field) != value:
            raise ValueError(f"direct summary field {field!r} differs from receipt contract")
    results = summary.get("results")
    if not isinstance(results, list) or [row.get("metric") for row in results] != sorted(metrics):
        raise ValueError("direct summary results do not have the exact metric ordering")

    expected_truth = direct_engine.truth_newick_from_manifest(source["truth_manifest"]["path"])
    if (root / "truth_manifest.newick").read_text().strip() != expected_truth:
        raise ValueError("direct truth Newick does not match the pinned truth manifest")
    for metric in metrics:
        matrix, _ = direct_engine._select_distance_matrix(
            cubes[metric],
            layer_names=layer_names,
            model_ids=model_ids,
            metric=metric,
            layer=parameters["layer"],
            aggregate=parameters["aggregate"],
        )
        stored_matrix = np.load(root / f"distance_matrix_{metric}.npy", allow_pickle=False)
        if not np.array_equal(matrix, stored_matrix):
            raise ValueError(f"stored {metric!r} matrix is not the selected cube matrix")
        expected_tree = direct_engine._neighbor_joining_newick(model_ids, matrix)
        if (root / f"tree_{metric}.newick").read_text().strip() != expected_tree:
            raise ValueError(f"stored {metric!r} tree is not NJ of the selected matrix")
        tree_audit = _load_json_object(
            root / f"tree_{metric}.audit.json", label=f"{metric} tree audit"
        )
        if (
            tree_audit.get("source_distance_layers")
            != str(root / "direct_distance_layers.npz")
            or tree_audit.get("model_ids") != model_ids
            or tree_audit.get("metric") != metric
        ):
            raise ValueError(f"stored {metric!r} tree audit provenance mismatch")

    return {
        "expected_output_relpaths": sorted(expected),
        "model_ids": model_ids,
        "layer_names": layer_names,
        "metric_shapes": metric_shapes,
    }


def _replay_distance_cube(
    root: Path,
    *,
    source: Mapping[str, Any],
    parameters: Mapping[str, Any],
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("direct replay tolerances must be nonnegative")
    readers = direct_engine._readers_from_ledger(
        source["ledger"]["path"],
        truth_manifest=Path(source["truth_manifest"]["path"]),
        artifact=str(parameters["artifact"]),
        node_ids=None,
        path_base=source["path_base"],
    )
    distances, layer_names, model_ids, _ = direct_engine._direct_distance_layers(
        readers,
        metrics=list(parameters["metrics"]),
        chunk_size=int(parameters["chunk_size"]),
        eps=float(parameters["eps"]),
        artifact=str(parameters["artifact"]),
    )
    stored_layers = _load_json_list(root / "layers.json", label="stored layers")
    stored_models = _load_json_list(root / "models.json", label="stored models")
    if layer_names != stored_layers or model_ids != stored_models:
        raise ValueError("direct replay changed layer or model ordering")

    comparisons = []
    with np.load(root / "direct_distance_layers.npz", allow_pickle=False) as archive:
        for metric in parameters["metrics"]:
            observed = np.asarray(archive[metric], dtype=np.float64)
            expected = np.asarray(distances[metric], dtype=np.float64)
            matches = bool(np.allclose(observed, expected, rtol=rtol, atol=atol, equal_nan=False))
            max_abs = float(np.max(np.abs(observed - expected))) if observed.size else 0.0
            row = {
                "metric": metric,
                "shape": list(observed.shape),
                "n_layers": len(layer_names),
                "n_models": len(model_ids),
                "n_values": int(observed.size),
                "matches": matches,
                "max_abs_error": max_abs,
            }
            comparisons.append(row)
            if not matches:
                raise ValueError(
                    f"direct replay mismatch for {metric}: max_abs_error={max_abs}"
                )
    _validate_captured_source(source)
    return comparisons


def _validate_shallow_direct_payload(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != DIRECT_ANALYSIS_RECEIPT_SCHEMA or payload.get("valid") is not True:
        raise ValueError(f"invalid direct-analysis producer receipt: {path}")
    root = Path(_required_text(payload, "analysis_root", context="direct receipt")).resolve()
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or payload.get("outputs_sha256") != canonical_sha256(outputs):
        raise ValueError("direct-analysis output manifest is invalid")
    if _pin_output_tree(root, excluded={path}) != outputs:
        raise ValueError("direct-analysis outputs no longer match producer receipt")
    source = _required_mapping(payload, "source", context="direct receipt")
    if payload.get("source_sha256") != canonical_sha256(source):
        raise ValueError("direct-analysis source hash is invalid")


def _parameters(
    *,
    artifact: str,
    metrics: Sequence[str],
    chunk_size: int,
    eps: float,
    layer: str | int | None,
    aggregate: str,
) -> dict[str, Any]:
    return _validate_parameters(
        {
            "artifact": artifact,
            "metrics": list(metrics),
            "chunk_size": chunk_size,
            "eps": eps,
            "layer": layer,
            "aggregate": aggregate,
        }
    )


def _validate_parameters(raw: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _required_text(raw, "artifact", context="direct parameters")
    if artifact not in {"model", "merged", "adapter_chain"}:
        raise ValueError(f"unsupported direct artifact mode: {artifact!r}")
    metrics = raw.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("direct parameters metrics must be a list")
    normalized = _normalize_metrics(metrics)
    if metrics != normalized:
        raise ValueError("direct parameters metrics must be normalized and duplicate-free")
    chunk_size = raw.get("chunk_size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("direct parameters chunk_size must be a positive integer")
    eps = raw.get("eps")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not np.isfinite(eps):
        raise ValueError("direct parameters eps must be a finite number")
    if float(eps) < 0.0:
        raise ValueError("direct parameters eps must be nonnegative")
    layer = raw.get("layer")
    if layer is not None and (isinstance(layer, bool) or not isinstance(layer, (str, int))):
        raise ValueError("direct parameters layer must be null, text, or an integer")
    aggregate = _required_text(raw, "aggregate", context="direct parameters")
    if aggregate not in {"mean", "median"}:
        raise ValueError("direct parameters aggregate must be mean or median")
    if layer is not None and aggregate != "mean":
        raise ValueError("selected-layer direct analyses must retain aggregate='mean' canonicality")
    return {
        "artifact": artifact,
        "metrics": normalized,
        "chunk_size": chunk_size,
        "eps": float(eps),
        "layer": layer,
        "aggregate": aggregate,
    }


def _normalize_metrics(metrics: Sequence[Any]) -> list[str]:
    if not metrics:
        raise ValueError("direct attestation requires at least one metric")
    out = []
    allowed = direct_engine.DIRECT_VECTOR_METRICS | direct_engine.DIRECT_MATRIX_METRICS
    for raw in metrics:
        metric = "cka" if raw == "linear_cka" else str(raw)
        if metric not in allowed:
            raise ValueError(f"unsupported direct metric: {raw!r}")
        if metric in out:
            raise ValueError(f"duplicate direct metric: {metric!r}")
        out.append(metric)
    return out


def _declared_summary_pin(
    summary: Mapping[str, Any], field: str, *, base: Path
) -> dict[str, Any]:
    path = _resolve_path(_required_text(summary, field, context="training summary"), base)
    digest = _required_digest(summary, f"{field}_sha256", context="training summary")
    _verify_expected_digest(path, digest, label=f"training summary {field}")
    return _file_pin(path)


def _declared_row_pin(
    row: Mapping[str, Any], field: str, *, base: Path
) -> dict[str, Any]:
    path = _resolve_path(_required_text(row, field, context="training tree row"), base)
    digest = _required_digest(row, f"{field}_sha256", context="training tree row")
    _verify_expected_digest(path, digest, label=f"training tree {field}")
    return _file_pin(path)


def _artifact_pin(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved.is_symlink():
        raise ValueError(f"artifact roots may not be symlinks: {resolved}")
    if resolved.is_file():
        return _file_pin(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(f"artifact does not exist: {resolved}")
    files = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"artifact directory contains a symlink: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"artifact directory contains a non-regular entry: {candidate}")
        files.append(
            {
                "relative_path": candidate.relative_to(resolved).as_posix(),
                "size": candidate.stat().st_size,
                "sha256": sha256(candidate),
            }
        )
    if not files:
        raise ValueError(f"artifact directory is empty: {resolved}")
    return {
        "path": str(resolved),
        "kind": "directory",
        "n_files": len(files),
        "files": files,
        "sha256": canonical_sha256(files),
    }


def _file_pin(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"expected a regular non-symlink file: {resolved}")
    return {
        "path": str(resolved),
        "kind": "file",
        "size": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def _verify_artifact_pin(raw: Any, *, label: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} pin must be an object")
    path = Path(_required_text(raw, "path", context=f"{label} pin")).resolve()
    expected = _required_digest(raw, "sha256", context=f"{label} pin")
    observed = _artifact_pin(path)
    if observed != {key: value for key, value in raw.items() if key != "artifact_id"}:
        raise ValueError(f"{label} no longer matches its pinned artifact manifest")
    if observed["sha256"] != expected:
        raise ValueError(f"{label} sha256 mismatch")


def _pin_output_tree(root: Path, *, excluded: set[Path]) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"direct analysis root must be a regular directory: {root}")
    excluded_resolved = {path.resolve() for path in excluded}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.resolve() in excluded_resolved:
            continue
        if path.is_symlink():
            raise ValueError(f"direct analysis output contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"direct analysis output is not a regular file: {path}")
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def _default_code_paths() -> list[Path]:
    return [Path(direct_engine.__file__).resolve(), Path(__file__).resolve()]


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for distribution in ("biopython", "dendropy", "safetensors"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temp.exists():
        raise FileExistsError(f"temporary receipt path already exists: {temp}")
    try:
        with temp.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing regular {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_json_list(path: Path, *, label: str) -> list[str]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return value


def _required_mapping(
    mapping: Mapping[str, Any], key: str, *, context: str
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} requires mapping field {key!r}")
    return value


def _required_text(mapping: Mapping[str, Any], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires nonempty text field {key!r}")
    return value.strip()


def _required_digest(mapping: Mapping[str, Any], key: str, *, context: str) -> str:
    value = _required_text(mapping, key, context=context)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} field {key!r} must be lowercase SHA-256")
    return value


def _verify_expected_digest(path: Path, expected: str | None, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing regular {label}: {path}")
    if expected is None:
        return
    if _SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"{label} expected sha256 must be lowercase hexadecimal")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected}, observed {observed}: {path}"
        )


def _resolve_path(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _require_expected_identity(
    payload: Mapping[str, Any],
    *,
    expected_cohort_id: str | None,
    expected_tree_id: str | None,
    expected_artifact: str | None,
    expected_metric: str | None,
) -> None:
    if expected_cohort_id is not None and payload.get("cohort_id") != expected_cohort_id:
        raise ValueError("direct replay cohort_id differs from downstream inventory")
    if expected_tree_id is not None and payload.get("tree_id") != canonical_tree_id(
        expected_tree_id
    ):
        raise ValueError("direct replay tree_id differs from downstream inventory")
    if expected_artifact is not None and payload.get("artifact") != expected_artifact:
        raise ValueError("direct replay artifact mode differs from downstream inventory")
    if expected_metric is not None and expected_metric not in payload.get("metrics", []):
        raise ValueError("direct replay does not contain the downstream metric")

