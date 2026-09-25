from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scipy.stats import t as student_t

from weighttraits.behavior.contracts import load_behavior_protocol_registry
from weighttraits.behavior.eos_depth import (
    EMPTY_DEFINITION,
    build_eos_depth_analysis,
    canonical_json_sha256,
    sha256_file,
    write_eos_depth_outputs,
)
from weighttraits.behavior.responses import (
    ResponseProvenance,
    audit_response_grid,
    expected_response_coordinates,
    response_from_text,
)
from weighttraits.behavior.probe_inference import INFERENCE_SCHEMA_VERSION


HEX = "a" * 64
BASE_REVISION = "b" * 40
OTHER_REVISION = "c" * 40
DATASET_REVISION = "d" * 40


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _protocol_request_row(protocol, prompts, *, model_task: str) -> dict:
    return {
        "protocol_id": protocol.protocol_id,
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
                "reference_sha256": hashlib.sha256(prompt.reference.encode()).hexdigest(),
                "fixture_id": prompt.metadata["fixture_id"],
                "fixture_sha256": prompt.metadata["fixture_sha256"],
                "source_indices_sha256": prompt.metadata["source_indices_sha256"],
                "dataset_revision": prompt.metadata["dataset_revision"],
            }
            for prompt in prompts
        ],
    }


def _make_registry(root: Path):
    indices = [0, 1]
    indices_sha = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    fixture_path = root / "fixture.json"
    _write_json(
        fixture_path,
        {
            "schema_version": 1,
            "fixture_id": "tiny_translation",
            "source_prompt_artifact": None,
            "probes": [
                {
                    "probe_id": "translation",
                    "dataset": {
                        "name": "tiny",
                        "config": None,
                        "split": "test",
                        "revision": DATASET_REVISION,
                    },
                    "eligibility": None,
                    "selection": {
                        "method": "explicit_indices",
                        "seed": 42,
                        "source_row_count": 2,
                        "n_prompts": 2,
                        "source_indices": indices,
                        "source_indices_sha256": indices_sha,
                    },
                }
            ],
        },
    )
    registry_path = root / "registry.json"
    _write_json(
        registry_path,
        {
            "schema_version": 1,
            "model_pins": {
                "causal_lm": {"model_id": "tiny/base", "revision": BASE_REVISION},
                "seq2seq": {"model_id": "tiny/seq", "revision": OTHER_REVISION},
            },
            "embedding_pin": {"model_id": "tiny/embed", "revision": OTHER_REVISION},
            "fixtures": {
                "tiny_translation": {
                    "path": fixture_path.name,
                    "sha256": sha256_file(fixture_path),
                }
            },
            "protocols": {
                "tiny_translation_2x2": {
                    "fixture_id": "tiny_translation",
                    "probe_ids": ["translation"],
                    "prompt_counts": {"translation": 2},
                    "samples_per_prompt": 2,
                    "base_seed": 42,
                    "draw_seeds": [42, 43],
                    "generation_batch_size": 32,
                    "seed_scope": "probe_draw",
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "min_new_tokens": 0,
                    "empty_policy": "preserve",
                    "analysis": {
                        "primary_empty_policy": "preserve",
                        "sensitivity_empty_policy": "exclude_exact_empty",
                        "natural_language_only": False,
                    },
                    "architectures": {
                        "causal_lm": {
                            "max_new_tokens": 8,
                            "prompt_templates": {
                                "translation": "Translate: {en}\nAnswer:"
                            },
                        },
                        "seq2seq": {
                            "max_new_tokens": 8,
                            "prompt_templates": {"translation": "translate: {en}"},
                        },
                    },
                }
            },
        },
    )
    registry = load_behavior_protocol_registry(registry_path)
    prompts = registry.materialize_probe(
        protocol_id="tiny_translation_2x2",
        probe_id="translation",
        rows=[
            {"translation": {"en": "one", "fr": "un"}},
            {"translation": {"en": "two", "fr": "deux"}},
        ],
        model_task="causal_lm",
        dataset_name="tiny",
        dataset_config=None,
        dataset_revision=DATASET_REVISION,
    )
    prompt_path = root / "prompts.jsonl"
    _write_jsonl(prompt_path, [prompt.to_dict() for prompt in prompts])
    return registry_path, registry, prompt_path, prompts


def _tree_manifest(root: Path, tree_number: int) -> tuple[str, Path]:
    tree_id = f"tree_{tree_number:03d}"
    path = root / f"{tree_id}.manifest.jsonl"
    _write_jsonl(
        path,
        [
            {
                "tree_id": tree_id,
                "node_id": "n0",
                "parent_id": "root",
                "path": ["root", "n0"],
                "depth": 1,
                "grow": "train",
                "task_family": "classification",
            },
            {
                "tree_id": tree_id,
                "node_id": "n1",
                "parent_id": "n0",
                "path": ["root", "n0", "n1"],
                "depth": 2,
                "grow": "train",
                "task_family": "generative_qa",
            },
            {
                "tree_id": tree_id,
                "node_id": "n2",
                "parent_id": "n0",
                "path": ["root", "n0", "n2"],
                "depth": 2,
                "grow": "train",
                "task_family": "classification",
            },
        ],
    )
    return tree_id, path


def _make_receipt(
    root: Path,
    *,
    tree_id: str,
    truth_manifest: Path,
    model_id: str,
    texts: list[str],
    registry_path: Path,
    registry,
    prompt_path: Path,
    prompts,
) -> Path:
    protocol_id = "tiny_translation_2x2"
    protocol = registry.protocol(protocol_id)
    coordinates = expected_response_coordinates(registry, {protocol_id: prompts})
    assert len(coordinates) == len(texts) == 4
    prompt_artifacts = {
        protocol_id: {"path": str(prompt_path), "sha256": sha256_file(prompt_path)}
    }
    request = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "cohort_id": "tiny_cohort",
        "tree_id": tree_id,
        "model_id": model_id,
        "leaf_ordinal": int(model_id.removeprefix("n")) - 1,
        "base_model_id": "tiny/base",
        "base_model_revision": BASE_REVISION,
        "model_task": "causal_lm",
        "checkpoint": f"/sealed/{tree_id}/{model_id}/model",
        "checkpoint_artifact": "model",
        "checkpoint_sha256": HEX,
        "training_summary": "/sealed/summary.json",
        "training_summary_sha256": HEX,
        "run_list": f"/sealed/{tree_id}.runs.jsonl",
        "run_list_sha256": HEX,
        "ledger": f"/sealed/{tree_id}.ledger.jsonl",
        "ledger_sha256": HEX,
        "truth_manifest": str(truth_manifest),
        "truth_manifest_sha256": sha256_file(truth_manifest),
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "prompt_artifacts": prompt_artifacts,
        "backend_settings": {"backend": "test"},
        "n_expected_responses": 4,
        "protocols": [_protocol_request_row(protocol, prompts, model_task="causal_lm")],
    }
    request_sha = canonical_json_sha256(request)
    provenance = ResponseProvenance(
        request_sha256=request_sha,
        training_summary=request["training_summary"],
        training_summary_sha256=HEX,
        run_list=request["run_list"],
        run_list_sha256=HEX,
        ledger=request["ledger"],
        ledger_sha256=HEX,
        truth_manifest=str(truth_manifest),
        truth_manifest_sha256=sha256_file(truth_manifest),
        checkpoint=request["checkpoint"],
        checkpoint_artifact="model",
        checkpoint_sha256=HEX,
        registry=str(registry_path),
        registry_sha256=sha256_file(registry_path),
        prompt_artifacts=prompt_artifacts,
    )
    responses = []
    for (coordinate, text) in zip(coordinates, texts, strict=True):
        response_protocol, prompt, sample_id = coordinate
        responses.append(
            response_from_text(
                cohort_id="tiny_cohort",
                tree_id=tree_id,
                model_id=model_id,
                base_model_id="tiny/base",
                base_model_revision=BASE_REVISION,
                model_task="causal_lm",
                protocol_id=response_protocol,
                prompt=prompt,
                sample_id=sample_id,
                generation_options=protocol.generation_options(
                    model_task="causal_lm", sample_id=sample_id
                ),
                text=text,
                provenance=provenance,
            )
        )
    response_path = root / f"{tree_id}_{model_id}.responses.jsonl"
    _write_jsonl(response_path, [row.to_dict() for row in responses])
    audit = audit_response_grid(
        responses,
        registry=registry,
        prompts_by_protocol={protocol_id: prompts},
        cohort_id="tiny_cohort",
        tree_id=tree_id,
        model_id=model_id,
        base_model_id="tiny/base",
        base_model_revision=BASE_REVISION,
        model_task="causal_lm",
        request_sha256=request_sha,
    )
    assert audit.valid
    receipt_path = root / f"{tree_id}_{model_id}.receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "valid": True,
            "status": "completed",
            "request_sha256": request_sha,
            "request": request,
            "provenance": provenance.to_dict(),
            "responses": str(response_path),
            "responses_sha256": sha256_file(response_path),
            "audit": audit.to_dict(),
            "prompt_artifacts": prompt_artifacts,
        },
    )
    return receipt_path


def _case(tmp_path: Path) -> tuple[Path, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry_path, registry, prompt_path, prompts = _make_registry(tmp_path)
    text_grid = {
        (1, "n1"): ["ok", "ok", "ok", "ok"],
        (1, "n2"): ["", "   ", "ok", "ok"],
        (2, "n1"): ["", "ok", "ok", "ok"],
        (2, "n2"): ["ok", "", "  ", "ok"],
    }
    tree_specs = []
    receipts = []
    for tree_number in (1, 2):
        tree_id, tree_path = _tree_manifest(tmp_path, tree_number)
        tree_specs.append(
            {"tree_id": tree_id, "path": str(tree_path), "sha256": sha256_file(tree_path)}
        )
        for node_id in ("n1", "n2"):
            receipt = _make_receipt(
                tmp_path,
                tree_id=tree_id,
                truth_manifest=tree_path,
                model_id=node_id,
                texts=text_grid[(tree_number, node_id)],
                registry_path=registry_path,
                registry=registry,
                prompt_path=prompt_path,
                prompts=prompts,
            )
            receipts.append(receipt)
    config = tmp_path / "eos_depth_input.json"
    _write_json(
        config,
        {
            "schema": "weighttraits.behavior.eos_depth_input.v1",
            "panel_id": "tiny-panel",
            "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
            "prompt_artifacts": {
                "tiny_translation_2x2": {
                    "path": str(prompt_path),
                    "sha256": sha256_file(prompt_path),
                }
            },
            "tree_manifests": tree_specs,
            "response_receipts": [
                {"path": str(path), "sha256": sha256_file(path)} for path in receipts
            ],
        },
    )
    return config, receipts


def test_eos_depth_uses_strip_empty_exact_pairs_and_tree_level_t_interval(
    tmp_path: Path,
) -> None:
    config, _ = _case(tmp_path)
    payload = build_eos_depth_analysis(config)

    assert payload["valid"] is True
    assert payload["empty_definition"] == EMPTY_DEFINITION
    per_leaf = {
        (row["tree"], row["node"]): row for row in payload["per_leaf_rows"]
    }
    assert per_leaf[("tree_001", "n1")]["n_empty"] == 0
    assert per_leaf[("tree_001", "n2")]["n_empty"] == 2
    assert per_leaf[("tree_002", "n1")]["n_empty"] == 1
    assert per_leaf[("tree_002", "n2")]["n_empty"] == 2
    assert per_leaf[("tree_001", "n1")]["task_family"] == "generative_qa"

    depth = payload["per_depth_rows"][0]
    assert depth["n_trees"] == 2
    assert depth["n_leaves"] == 4
    assert depth["n_outputs"] == 16
    assert depth["n_empty"] == 5
    assert depth["pooled_empty_percent"] == pytest.approx(31.25)
    assert depth["mean_tree_empty_percent"] == pytest.approx(31.25)
    margin = float(student_t.ppf(0.975, df=1)) * 6.25
    assert depth["t95_ci_low_tree_empty_percent"] == pytest.approx(31.25 - margin)
    assert depth["t95_ci_high_tree_empty_percent"] == pytest.approx(31.25 + margin)

    pairs = {row["tree"]: row for row in payload["sensitivity_pair_rows"]}
    assert pairs["tree_001"]["n_primary"] == 4
    assert pairs["tree_001"]["n_paired_nonempty"] == 2
    assert pairs["tree_002"]["n_paired_nonempty"] == 1
    summary = payload["sensitivity_summary_rows"][0]
    assert summary["n_primary"] == 8
    assert summary["n_paired_nonempty"] == 3
    assert summary["n_excluded"] == 5


def test_eos_depth_fails_closed_on_missing_leaf_and_response_hash_drift(tmp_path: Path) -> None:
    config, receipts = _case(tmp_path)
    raw = json.loads(config.read_text())
    raw["response_receipts"] = raw["response_receipts"][:-1]
    _write_json(config, raw)
    with pytest.raises(ValueError, match="leaf set"):
        build_eos_depth_analysis(config)

    config, receipts = _case(tmp_path / "hash_drift")
    receipt = json.loads(receipts[0].read_text())
    Path(receipt["responses"]).write_text(
        Path(receipt["responses"]).read_text() + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="response JSONL hash mismatch"):
        build_eos_depth_analysis(config)


def test_eos_depth_fails_closed_on_incomplete_generation_grid(tmp_path: Path) -> None:
    config, receipts = _case(tmp_path)
    receipt_path = receipts[0]
    receipt = json.loads(receipt_path.read_text())
    response_path = Path(receipt["responses"])
    response_path.write_text("".join(response_path.read_text().splitlines(keepends=True)[:-1]))
    receipt["responses_sha256"] = sha256_file(response_path)
    _write_json(receipt_path, receipt)
    config_raw = json.loads(config.read_text())
    for spec in config_raw["response_receipts"]:
        if spec["path"] == str(receipt_path):
            spec["sha256"] = sha256_file(receipt_path)
    _write_json(config, config_raw)

    with pytest.raises(ValueError, match="response-grid audit drift"):
        build_eos_depth_analysis(config)


@pytest.mark.parametrize("schema_version", [1, 3, True])
def test_eos_depth_rejects_noncanonical_inference_receipt_schema(
    tmp_path: Path,
    schema_version: object,
) -> None:
    config, receipts = _case(tmp_path)
    receipt_path = receipts[0]
    receipt = json.loads(receipt_path.read_text())
    receipt["schema_version"] = schema_version
    _write_json(receipt_path, receipt)
    config_raw = json.loads(config.read_text())
    for spec in config_raw["response_receipts"]:
        if spec["path"] == str(receipt_path):
            spec["sha256"] = sha256_file(receipt_path)
    _write_json(config, config_raw)

    with pytest.raises(
        ValueError,
        match=rf"schema_version must be exactly {INFERENCE_SCHEMA_VERSION}",
    ):
        build_eos_depth_analysis(config)


def test_atomic_outputs_are_hashed_and_refuse_implicit_overwrite(tmp_path: Path) -> None:
    config, _ = _case(tmp_path / "inputs")
    payload = build_eos_depth_analysis(config)
    out = tmp_path / "outputs"
    paths = {
        "json_path": out / "analysis.json",
        "per_leaf_csv_path": out / "per_leaf.csv",
        "per_depth_csv_path": out / "per_depth.csv",
        "sensitivity_csv_path": out / "sensitivity.csv",
        "sensitivity_summary_csv_path": out / "sensitivity_summary.csv",
        "receipt_path": out / "receipt.json",
    }
    receipt = write_eos_depth_outputs(payload, **paths)
    assert receipt["valid"] is True
    for row in receipt["outputs"].values():
        assert sha256_file(row["path"]) == row["sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_eos_depth_outputs(payload, **paths)

    config.write_text(config.read_text() + "\n", encoding="utf-8")
    drift_paths = {key: path.with_name(f"drift_{path.name}") for key, path in paths.items()}
    with pytest.raises(ValueError, match="changed after analysis"):
        write_eos_depth_outputs(payload, **drift_paths)
