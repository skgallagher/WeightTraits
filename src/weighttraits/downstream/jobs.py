"""Pinned in-job runners for direct, behavior, and PhyloLM downstream work."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from weighttraits.downstream.submission import (
    BEHAVIOR_KIND,
    CodeManifest,
    DIRECT_KIND,
    PHYLOLM_KIND,
    Pin,
    sha256_file,
    verify_code_manifest,
)
from weighttraits.behavior.contracts import load_behavior_protocol_registry
from weighttraits.behavior.probe_inference import resolve_leaf_checkpoints
from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    validate_strict_training_completion,
)


EXACT_DIRECT_METRICS = ("l2", "cosine", "correlation")
TREE001_SMOKE_LEAF_IDS = ("n3", "n7", "n8", "n9", "n10", "n11", "n12", "n13")
_DIRECT_SCHEMA = f"weighttraits.downstream_job.{DIRECT_KIND}.v1"
_BEHAVIOR_SCHEMA = f"weighttraits.downstream_job.{BEHAVIOR_KIND}.v1"
_PHYLOLM_SCHEMA = f"weighttraits.downstream_job.{PHYLOLM_KIND}.v1"


def run_downstream_job(
    *,
    kind: str,
    config_path: str | Path,
    expected_config_sha256: str,
    code_root: str | Path,
    code_manifest_path: str | Path,
    expected_code_manifest_sha256: str,
    runner=subprocess.run,
) -> dict[str, Any]:
    """Run one already-submitted job after replaying every deployment pin."""

    config = Path(config_path).resolve(strict=True)
    _require_sha(config, expected_config_sha256, "job config")
    root = Path(code_root).resolve(strict=True)
    manifest = CodeManifest(
        root=root,
        pin=Pin(
            Path(code_manifest_path).resolve(strict=True),
            _sha(expected_code_manifest_sha256, "code manifest sha256"),
        ),
    )
    verify_code_manifest(manifest)
    raw = _json_object(config)
    if raw.get("kind") != kind:
        raise ValueError(f"job kind mismatch: command={kind!r}, config={raw.get('kind')!r}")
    if kind == DIRECT_KIND:
        receipt = _run_direct(raw, config=config, code_root=root, runner=runner)
    elif kind == BEHAVIOR_KIND:
        receipt = _run_behavior(raw, config=config, code_root=root, runner=runner)
    elif kind == PHYLOLM_KIND:
        receipt = _run_phylolm(raw, config=config, code_root=root, runner=runner)
    else:
        raise ValueError(f"unsupported downstream job kind: {kind!r}")
    _require_sha(config, expected_config_sha256, "job config after execution")
    verify_code_manifest(manifest)
    return receipt


def _run_direct(
    raw: Mapping[str, Any],
    *,
    config: Path,
    code_root: Path,
    runner,
) -> dict[str, Any]:
    _exact_keys(
        raw,
        {
            "schema",
            "kind",
            "stage_root",
            "output_dir",
            "cohort_id",
            "training_summary",
            "completion_receipt",
            "stage_manifest",
            "path_base",
            "artifact_kind",
            "metrics",
            "chunk_size",
            "eps",
            "aggregate",
            "expected_n_trees",
            "expected_n_rows",
        },
        "direct job config",
    )
    if raw["schema"] != _DIRECT_SCHEMA:
        raise ValueError(f"direct job schema must be {_DIRECT_SCHEMA!r}")
    output, _ = _fresh_output(raw, config=config)
    cohort = _text(raw["cohort_id"], "cohort_id")
    summary = _pin(raw["training_summary"], "training_summary")
    completion = _pin(raw["completion_receipt"], "completion_receipt")
    stage_manifest = _pin(raw["stage_manifest"], "stage_manifest")
    path_base = _directory(raw["path_base"], "path_base")
    artifact = _choice(
        raw["artifact_kind"],
        {"model", "merged", "adapter_chain"},
        "artifact_kind",
    )
    metrics = tuple(raw["metrics"]) if isinstance(raw["metrics"], list) else ()
    if metrics != EXACT_DIRECT_METRICS:
        raise ValueError(
            f"direct run-set metrics must be exactly {list(EXACT_DIRECT_METRICS)!r} in order"
        )
    expected_n_trees = _exact_int(raw["expected_n_trees"], 50, "expected_n_trees")
    expected_n_rows = _exact_int(raw["expected_n_rows"], 150, "expected_n_rows")
    chunk_size = _positive_int(raw["chunk_size"], "chunk_size")
    eps = _exact_float(raw["eps"], 1e-3, "eps")
    aggregate = _choice(raw["aggregate"], {"mean"}, "aggregate")
    _validate_completion(completion, cohort_id=cohort)
    _verify_pins((summary, completion, stage_manifest))

    output.mkdir(parents=True, exist_ok=False)
    analysis_root = output / "analysis"
    analysis_report = output / "direct_run_set_analysis_report.json"
    rollup_json = output / "direct_run_set_summary.json"
    rollup_csv = output / "direct_run_set_rows.csv"
    command = [
        sys.executable,
        "-m",
        "weighttraits.cli",
        "analyze-training-run-set",
        "--summary",
        str(summary.path),
        "--path-base",
        str(path_base),
        "--artifact",
        artifact,
        "--out",
        str(analysis_root),
        "--report-out",
        str(analysis_report),
        "--chunk-size",
        str(chunk_size),
        "--eps",
        str(eps),
        "--aggregate",
        aggregate,
    ]
    for metric in metrics:
        command.extend(["--metric", metric])
    _run(command, code_root=code_root, runner=runner)
    analysis = _json_object(analysis_report)
    if (
        analysis.get("valid") is not True
        or analysis.get("status") != "completed"
        or analysis.get("analysis_engine") != "direct"
        or analysis.get("metrics") != list(metrics)
        or analysis.get("n_selected") != expected_n_trees
        or analysis.get("n_analyzed") != expected_n_trees
        or analysis.get("n_existing") != 0
        or analysis.get("n_skipped") != 0
        or analysis.get("selected_tree_ids") != list(ALL_TREE_IDS)
    ):
        raise ValueError("direct run-set analysis failed the exact 50-tree fresh-output contract")

    command = [
        sys.executable,
        "-m",
        "weighttraits.cli",
        "summarize-training-run-set-analysis",
        "--analysis-root",
        str(analysis_root),
        "--artifact",
        artifact,
        "--path-base",
        str(path_base),
        "--out",
        str(rollup_json),
        "--csv-out",
        str(rollup_csv),
    ]
    _run(command, code_root=code_root, runner=runner)
    rollup = _json_object(rollup_json)
    if (
        rollup.get("valid") is not True
        or rollup.get("n_tree_summaries") != expected_n_trees
        or rollup.get("n_rows") != expected_n_rows
        or rollup.get("tree_ids") != list(ALL_TREE_IDS)
        or rollup.get("metrics") != sorted(metrics)
        or len(rollup.get("rows", [])) != expected_n_rows
    ):
        raise ValueError("direct run-set rollup failed the exact 50-tree/3-metric/150-row contract")
    with rollup_csv.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != expected_n_rows:
        raise ValueError("direct run-set CSV row count differs from the JSON rollup")

    _verify_pins((summary, completion, stage_manifest))
    receipt = {
        "schema": "weighttraits.downstream_direct_runset_receipt.v1",
        "valid": True,
        "cohort_id": cohort,
        "artifact": artifact,
        "metrics": list(metrics),
        "n_selected": expected_n_trees,
        "n_tree_summaries": expected_n_trees,
        "n_rows": expected_n_rows,
        "input_config": {"path": str(config), "sha256": sha256_file(config)},
        "training_summary": summary.to_dict(),
        "completion_receipt": completion.to_dict(),
        "stage_manifest": stage_manifest.to_dict(),
        "analysis_report": _file_pin(analysis_report),
        "rollup_json": _file_pin(rollup_json),
        "rollup_csv": _file_pin(rollup_csv),
    }
    _write_json_exclusive(output / "receipt.json", receipt)
    return receipt


def _run_behavior(
    raw: Mapping[str, Any],
    *,
    config: Path,
    code_root: Path,
    runner,
) -> dict[str, Any]:
    _exact_keys(
        raw,
        {
            "schema",
            "kind",
            "stage_root",
            "output_dir",
            "cohort_id",
            "tree_id",
            "model_task",
            "path_base",
            "registry",
            "training_summary",
            "completion_receipt",
            "prompt_artifacts",
            "expected_leaf_ids",
            "leaf_id",
            "device",
            "torch_dtype",
        },
        "behavior smoke config",
    )
    if raw["schema"] != _BEHAVIOR_SCHEMA:
        raise ValueError(f"behavior smoke schema must be {_BEHAVIOR_SCHEMA!r}")
    output, _ = _fresh_output(raw, config=config)
    cohort = _text(raw["cohort_id"], "cohort_id")
    tree_id = _canonical_tree(raw["tree_id"])
    model_task = _choice(raw["model_task"], {"causal_lm", "seq2seq"}, "model_task")
    path_base = _directory(raw["path_base"], "path_base")
    registry = _pin(raw["registry"], "registry")
    summary = _pin(raw["training_summary"], "training_summary")
    completion = _pin(raw["completion_receipt"], "completion_receipt")
    prompts = _pin_mapping(raw["prompt_artifacts"], "prompt_artifacts")
    expected_leaf_ids = tuple(raw["expected_leaf_ids"]) if isinstance(raw["expected_leaf_ids"], list) else ()
    if expected_leaf_ids != TREE001_SMOKE_LEAF_IDS:
        raise ValueError(
            "Tree001 behavior smoke expected_leaf_ids must be exactly "
            f"{list(TREE001_SMOKE_LEAF_IDS)!r} in truth order"
        )
    leaf_id = _text(raw["leaf_id"], "leaf_id")
    if leaf_id != "n3":
        raise ValueError("Tree001 behavior smoke must run exactly leaf_id='n3'")
    if raw["device"] != "cuda" or raw["torch_dtype"] != "auto":
        raise ValueError("native behavior smoke requires device='cuda' and torch_dtype='auto'")
    _validate_completion(completion, cohort_id=cohort)
    _verify_pins((registry, summary, completion, *prompts.values()))

    registry_payload = load_behavior_protocol_registry(registry.path)
    model_pin = registry_payload.model_pins[model_task]
    resolved = resolve_leaf_checkpoints(
        summary.path,
        cohort_id=cohort,
        tree_id=tree_id,
        model_task=model_task,
        base_model_id=model_pin.model_id,
        base_model_revision=model_pin.revision,
        path_base=path_base,
    )
    observed_leaf_ids = tuple(leaf.leaf_id for leaf in resolved.leaves)
    if observed_leaf_ids != expected_leaf_ids:
        raise ValueError(
            "live Tree001 checkpoint resolution differs from the exact expected leaf order: "
            f"observed={list(observed_leaf_ids)!r}"
        )
    resolved.leaf(leaf_id)

    output.mkdir(parents=True, exist_ok=False)
    response_dir = output / "responses"
    receipt_dir = output / "response_receipts"
    response_dir.mkdir()
    receipt_dir.mkdir()
    script = code_root / "scripts" / "run_behavior_probe_inference.py"
    response = response_dir / f"{leaf_id}.jsonl"
    inference_receipt = receipt_dir / f"{leaf_id}.json"
    command = [
        sys.executable,
        str(script),
        "--training-summary",
        str(summary.path),
        "--cohort-id",
        cohort,
        "--tree-id",
        tree_id,
        "--leaf-id",
        leaf_id,
        "--model-task",
        model_task,
        "--registry",
        str(registry.path),
        "--path-base",
        str(path_base),
        "--out",
        str(response),
        "--receipt",
        str(inference_receipt),
        "--device",
        "cuda",
        "--torch-dtype",
        "auto",
    ]
    for protocol_id, pin in sorted(prompts.items()):
        command.extend(["--protocol-prompts", f"{protocol_id}={pin.path}"])
    _run(command, code_root=code_root, runner=runner, offline=True)
    inference = _json_object(inference_receipt)
    request = inference.get("request")
    if (
        inference.get("valid") is not True
        or inference.get("status") != "completed"
        or not isinstance(request, dict)
        or request.get("cohort_id") != cohort
        or request.get("tree_id") != tree_id
        or request.get("model_id") != leaf_id
    ):
        raise ValueError(f"invalid terminal behavior inference receipt for {leaf_id}")

    _verify_pins((registry, summary, completion, *prompts.values()))
    receipt = {
        "schema": "weighttraits.downstream_behavior_smoke_receipt.v1",
        "valid": True,
        "cohort_id": cohort,
        "tree_id": tree_id,
        "model_task": model_task,
        "n_tree_leaves": len(expected_leaf_ids),
        "expected_leaf_ids": list(expected_leaf_ids),
        "leaf_id": leaf_id,
        "input_config": {"path": str(config), "sha256": sha256_file(config)},
        "registry": registry.to_dict(),
        "training_summary": summary.to_dict(),
        "completion_receipt": completion.to_dict(),
        "prompt_artifacts": {key: pin.to_dict() for key, pin in sorted(prompts.items())},
        "responses": _file_pin(response),
        "inference_receipt": _file_pin(inference_receipt),
    }
    _write_json_exclusive(output / "receipt.json", receipt)
    return receipt


def _run_phylolm(
    raw: Mapping[str, Any],
    *,
    config: Path,
    code_root: Path,
    runner,
) -> dict[str, Any]:
    _exact_keys(
        raw,
        {
            "schema",
            "kind",
            "stage_root",
            "output_dir",
            "cohort_id",
            "tree_id",
            "training_summary",
            "completion_receipt",
            "stage_manifest",
            "genome_receipt",
            "truth_manifest",
            "base_model_id",
            "base_model_revision",
            "path_base",
            "batch_size",
            "torch_dtype",
        },
        "PhyloLM smoke config",
    )
    if raw["schema"] != _PHYLOLM_SCHEMA:
        raise ValueError(f"PhyloLM smoke schema must be {_PHYLOLM_SCHEMA!r}")
    output, _ = _fresh_output(raw, config=config)
    cohort = _text(raw["cohort_id"], "cohort_id")
    tree_id = _canonical_tree(raw["tree_id"])
    summary = _pin(raw["training_summary"], "training_summary")
    completion = _pin(raw["completion_receipt"], "completion_receipt")
    stage_manifest = _pin(raw["stage_manifest"], "stage_manifest")
    genome = _pin(raw["genome_receipt"], "genome_receipt")
    truth = _pin(raw["truth_manifest"], "truth_manifest")
    base_model_id = _text(raw["base_model_id"], "base_model_id")
    base_revision = _revision(raw["base_model_revision"], "base_model_revision")
    path_base = _directory(raw["path_base"], "path_base")
    _exact_int(raw["batch_size"], 64, "batch_size")
    if raw["torch_dtype"] != "auto":
        raise ValueError("corrected PhyloLM smoke requires torch_dtype='auto'")
    _validate_completion(completion, cohort_id=cohort)
    _verify_pins((summary, completion, stage_manifest, genome, truth))

    output.mkdir(parents=True, exist_ok=False)
    populations = output / "populations"
    tree_output = output / "tree"
    population_script = code_root / "scripts" / "run_phylolm_tree_population.py"
    analysis_script = code_root / "scripts" / "analyze_phylolm_tree.py"
    command = [
        sys.executable,
        str(population_script),
        "--training-summary",
        str(summary.path),
        "--completion-receipt",
        str(completion.path),
        "--stage-manifest",
        str(stage_manifest.path),
        "--genome-receipt",
        str(genome.path),
        "--cohort-id",
        cohort,
        "--tree-id",
        tree_id,
        "--base-model-id",
        base_model_id,
        "--base-model-revision",
        base_revision,
        "--path-base",
        str(path_base),
        "--out-dir",
        str(populations),
        "--batch-size",
        "64",
        "--torch-dtype",
        "auto",
    ]
    _run(command, code_root=code_root, runner=runner, offline=True)
    population_specs = populations / "population_specs.json"
    command = [
        sys.executable,
        str(analysis_script),
        "--population-specs",
        str(population_specs),
        "--suite-id",
        cohort,
        "--tree-id",
        tree_id,
        "--truth-manifest",
        str(truth.path),
        "--out-dir",
        str(tree_output),
    ]
    _run(command, code_root=code_root, runner=runner, offline=True)
    tree_receipt = tree_output / "receipt.json"
    tree_payload = _json_object(tree_receipt)
    if (
        tree_payload.get("valid") is not True
        or tree_payload.get("suite_id") != cohort
        or tree_payload.get("tree_id") != tree_id
        or tree_payload.get("completion_receipt") != completion.to_dict()
        or tree_payload.get("stage_manifest") != stage_manifest.to_dict()
        or tree_payload.get("genome_receipt") != str(genome.path)
        or tree_payload.get("genome_receipt_sha256") != genome.sha256
        or tree_payload.get("truth_manifest") != str(truth.path)
        or tree_payload.get("truth_sha256") != truth.sha256
        or tree_payload.get("model_ids") != list(TREE001_SMOKE_LEAF_IDS)
        or tree_payload.get("n_truth_splits", 0) <= 0
    ):
        raise ValueError("PhyloLM tree smoke failed its terminal provenance contract")
    _verify_pins((summary, completion, stage_manifest, genome, truth))
    receipt = {
        "schema": "weighttraits.downstream_phylolm_smoke_receipt.v1",
        "valid": True,
        "cohort_id": cohort,
        "tree_id": tree_id,
        "input_config": {"path": str(config), "sha256": sha256_file(config)},
        "training_summary": summary.to_dict(),
        "completion_receipt": completion.to_dict(),
        "stage_manifest": stage_manifest.to_dict(),
        "genome_receipt": genome.to_dict(),
        "truth_manifest": truth.to_dict(),
        "population_specs": _file_pin(population_specs),
        "tree_receipt": _file_pin(tree_receipt),
    }
    _write_json_exclusive(output / "receipt.json", receipt)
    return receipt


def _run(command: Sequence[str], *, code_root: Path, runner, offline: bool = False) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code_root / "src")
    if offline:
        env.update(
            {
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
            }
        )
    runner(
        list(command),
        cwd=code_root,
        env=env,
        check=True,
        text=True,
    )


def _fresh_output(raw: Mapping[str, Any], *, config: Path) -> tuple[Path, Path]:
    stage = _directory(raw["stage_root"], "stage_root")
    output = Path(_text(raw["output_dir"], "output_dir")).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing downstream output directory: {output}")
    if stage not in output.parents:
        raise ValueError("downstream job output must be strictly inside stage_root")
    if stage not in config.parents:
        raise ValueError("downstream job config must be inside stage_root")
    return output, stage


def _validate_completion(pin: Pin, *, cohort_id: str) -> None:
    payload = _json_object(pin.path)
    validate_strict_training_completion(payload, cohort_id=cohort_id)


def _pin(raw: object, label: str) -> Pin:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path+sha256 object")
    path = Path(_text(raw["path"], f"{label}.path")).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must pin a file: {path}")
    pin = Pin(path, _sha(raw["sha256"], f"{label}.sha256"))
    _require_sha(pin.path, pin.sha256, label)
    return pin


def _pin_mapping(raw: object, label: str) -> dict[str, Pin]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{label} must be a non-empty mapping")
    result: dict[str, Pin] = {}
    for key, value in raw.items():
        result[_text(key, f"{label} key")] = _pin(value, f"{label}.{key}")
    return result


def _verify_pins(pins: Sequence[Pin]) -> None:
    for pin in pins:
        _require_sha(pin.path, pin.sha256, str(pin.path))


def _require_sha(path: Path, expected: object, label: str) -> None:
    digest = _sha(expected, f"{label} sha256")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(f"{label} hash mismatch: expected {digest}, got {observed}")


def _file_pin(path: Path) -> dict[str, str]:
    source = path.resolve(strict=True)
    return {"path": str(source), "sha256": sha256_file(source)}


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    keys = set(raw)
    if keys != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - keys)}, "
            f"unexpected={sorted(keys - expected)}"
        )


def _text(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return raw


def _sha(raw: object, label: str) -> str:
    value = _text(raw, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _directory(raw: object, label: str) -> Path:
    path = Path(_text(raw, label)).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _choice(raw: object, allowed: set[str], label: str) -> str:
    value = _text(raw, label)
    if value not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)}")
    return value


def _positive_int(raw: object, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return raw


def _exact_int(raw: object, expected: int, label: str) -> int:
    value = _positive_int(raw, label)
    if value != expected:
        raise ValueError(f"{label} must be exactly {expected}")
    return value


def _exact_float(raw: object, expected: float, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(raw)
    if value != expected:
        raise ValueError(f"{label} must be exactly {expected}")
    return value


def _revision(raw: object, label: str) -> str:
    value = _text(raw, label)
    if len(value) not in {40, 64} or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a pinned 40- or 64-character revision")
    return value


def _canonical_tree(raw: object) -> str:
    value = _text(raw, "tree_id")
    if value != ALL_TREE_IDS[0]:
        raise ValueError("smoke tree_id must be exactly 'confirm_paper_tree_001'")
    return value


__all__ = ["EXACT_DIRECT_METRICS", "TREE001_SMOKE_LEAF_IDS", "run_downstream_job"]
