from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from weighttraits.behavior.materialize import (
    PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2,
    PROMPT_SOURCE_AUDIT_INPUT_SCHEMA,
    PROMPT_SOURCE_AUDIT_RECEIPT_SCHEMA,
    audit_prompt_sources,
    build_prompt_artifacts,
)
from weighttraits.behavior.probe_inference import canonical_json_sha256, sha256_file


pa = pytest.importorskip("pyarrow")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_arrow(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def _fixture(
    *,
    fixture_id: str,
    probe_id: str,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    source_row_count: int | None,
    source_indices: list[int],
    eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "source_prompt_artifact": None,
        "probes": [
            {
                "probe_id": probe_id,
                "dataset": {
                    "name": dataset_name,
                    "config": dataset_config,
                    "split": split,
                    "revision": "1" * 40,
                },
                "eligibility": eligibility,
                "selection": {
                    "method": "frozen-test-selection",
                    "seed": 42,
                    "source_row_count": source_row_count,
                    "n_prompts": len(source_indices),
                    "source_indices": source_indices,
                    "source_indices_sha256": canonical_json_sha256(source_indices),
                },
            }
        ],
    }


def _protocol(
    *,
    fixture_id: str,
    probe_id: str,
    prompt_count: int,
    causal_template: str,
    seq2seq_template: str,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "probe_ids": [probe_id],
        "prompt_counts": {probe_id: prompt_count},
        "samples_per_prompt": 1,
        "base_seed": 42,
        "draw_seeds": [42],
        "generation_batch_size": 2,
        "seed_scope": "probe_draw",
        "do_sample": False,
        "temperature": None,
        "top_p": None,
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
                "prompt_templates": {probe_id: causal_template},
            },
            "seq2seq": {
                "max_new_tokens": 8,
                "prompt_templates": {probe_id: seq2seq_template},
            },
        },
    }


def _workspace(
    tmp_path: Path,
    *,
    translation_fixture_count: int = 3,
    dolly_index: int = 1,
    translation_input_sha: str | None = None,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    translation_rows = [
        {"translation": {"en": "first", "fr": "premier"}},
        {"translation": {"en": "second", "fr": "deuxième"}},
        {"translation": {"en": "third", "fr": "troisième"}},
    ]
    dolly_rows = [
        {
            "instruction": "Classify this.",
            "context": "",
            "response": "No.",
            "category": "classification",
        },
        {
            "instruction": "Explain rain.",
            "context": "For a child.",
            "response": "Clouds release water.",
            "category": "open_qa",
        },
    ]
    translation_arrow = tmp_path / "translation.arrow"
    dolly_arrow = tmp_path / "dolly.arrow"
    _write_arrow(translation_arrow, translation_rows)
    _write_arrow(dolly_arrow, dolly_rows)

    translation_fixture = _fixture(
        fixture_id="tiny_translation",
        probe_id="translation",
        dataset_name="tiny/translation",
        dataset_config="en-fr",
        split="test",
        source_row_count=translation_fixture_count,
        source_indices=[2, 0],
    )
    dolly_fixture = _fixture(
        fixture_id="tiny_dolly",
        probe_id="dolly_open_ended",
        dataset_name="tiny/dolly",
        dataset_config=None,
        split="train",
        source_row_count=None,
        source_indices=[dolly_index],
        eligibility={
            "field": "category",
            "allowed_values": ["open_qa"],
            "require_nonempty": ["instruction"],
        },
    )
    translation_fixture_path = tmp_path / "translation_fixture.json"
    dolly_fixture_path = tmp_path / "dolly_fixture.json"
    _write_json(translation_fixture_path, translation_fixture)
    _write_json(dolly_fixture_path, dolly_fixture)

    registry = {
        "schema_version": 1,
        "model_pins": {
            "causal_lm": {"model_id": "tiny/causal", "revision": "2" * 40},
            "seq2seq": {"model_id": "tiny/seq2seq", "revision": "3" * 40},
        },
        "embedding_pin": {"model_id": "tiny/encoder", "revision": "4" * 40},
        "fixtures": {
            "tiny_translation": {
                "path": translation_fixture_path.name,
                "sha256": sha256_file(translation_fixture_path),
            },
            "tiny_dolly": {
                "path": dolly_fixture_path.name,
                "sha256": sha256_file(dolly_fixture_path),
            },
        },
        "protocols": {
            "tiny_translation_protocol": _protocol(
                fixture_id="tiny_translation",
                probe_id="translation",
                prompt_count=2,
                causal_template="Translate: {en}",
                seq2seq_template="translate: {en}",
            ),
            "tiny_dolly_protocol": _protocol(
                fixture_id="tiny_dolly",
                probe_id="dolly_open_ended",
                prompt_count=1,
                causal_template="Request: {instruction}{context_block}",
                seq2seq_template="request: {instruction}{context_block}",
            ),
        },
    }
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)

    audit_input = {
        "schema": PROMPT_SOURCE_AUDIT_INPUT_SCHEMA,
        "model_task": "causal_lm",
        "registry": {"path": registry_path.name, "sha256": sha256_file(registry_path)},
        "protocols": {
            "tiny_translation_protocol": {
                "probes": {
                    "translation": {
                        "source_artifacts": [
                            {
                                "path": translation_arrow.name,
                                "sha256": translation_input_sha,
                                "format": "arrow",
                            }
                        ]
                    }
                }
            },
            "tiny_dolly_protocol": {
                "probes": {
                    "dolly_open_ended": {
                        "source_artifacts": [
                            {"path": dolly_arrow.name, "sha256": None, "format": "arrow"}
                        ]
                    }
                }
            },
        },
    }
    input_path = tmp_path / "source_audit_input.json"
    _write_json(input_path, audit_input)
    return {
        "input": input_path,
        "registry": registry_path,
        "translation_arrow": translation_arrow,
        "dolly_arrow": dolly_arrow,
        "receipt": tmp_path / "source_audit_receipt.json",
        "materialization": tmp_path / "materialization.json",
    }


def test_source_audit_resolves_null_count_and_builds_verified_config(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    result = audit_prompt_sources(
        paths["input"],
        receipt_path=paths["receipt"],
        materialization_config_path=paths["materialization"],
    )

    receipt = json.loads(paths["receipt"].read_text())
    materialization = json.loads(paths["materialization"].read_text())
    assert receipt["schema"] == PROMPT_SOURCE_AUDIT_RECEIPT_SCHEMA
    assert receipt["valid"] is True
    assert receipt["n_protocols"] == 2
    assert receipt["n_probes"] == 2
    assert receipt["n_prompts"] == 3
    assert receipt["n_resolved_fixture_source_row_counts"] == 1
    dolly = receipt["protocols"]["tiny_dolly_protocol"]["probes"]["dolly_open_ended"]
    assert dolly["fixture_source_row_count"] is None
    assert dolly["source_row_count"] == 2
    assert dolly["resolved_fixture_source_row_count"] is True
    assert dolly["source_indices"] == [1]
    assert dolly["source_indices_sha256"] == canonical_json_sha256([1])
    assert dolly["source_artifacts"][0]["sha256"] == sha256_file(paths["dolly_arrow"])

    translation = receipt["protocols"]["tiny_translation_protocol"]["probes"][
        "translation"
    ]
    selected_translation_rows = [
        {"translation": {"en": "third", "fr": "troisième"}},
        {"translation": {"en": "first", "fr": "premier"}},
    ]
    assert translation["source_row_count"] == 3
    assert translation["resolved_fixture_source_row_count"] is False
    assert translation["selected_rows_sha256"] == canonical_json_sha256(
        selected_translation_rows
    )
    assert translation["rendered_prompts_sha256"] == canonical_json_sha256(
        ["Translate: third", "Translate: first"]
    )

    assert materialization["schema"] == PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2
    assert materialization["source_audit"] == result["receipt_artifact"]
    assert materialization["protocols"]["tiny_dolly_protocol"]["probes"][
        "dolly_open_ended"
    ]["source_row_count"] == 2
    assert receipt["materialization_protocols_sha256"] == canonical_json_sha256(
        materialization["protocols"]
    )

    prompt_receipt = build_prompt_artifacts(
        paths["materialization"], out_dir=tmp_path / "prompts"
    )
    assert prompt_receipt["valid"] is True
    assert prompt_receipt["n_protocols"] == 2
    assert prompt_receipt["n_prompts"] == 3


def test_source_audit_rejects_explicit_sha_and_fixture_count_mismatches(
    tmp_path: Path,
) -> None:
    sha_paths = _workspace(tmp_path / "sha", translation_input_sha="0" * 64)
    with pytest.raises(ValueError, match="source artifact SHA256 mismatch"):
        audit_prompt_sources(
            sha_paths["input"],
            receipt_path=sha_paths["receipt"],
            materialization_config_path=sha_paths["materialization"],
        )
    assert not sha_paths["receipt"].exists()
    assert not sha_paths["materialization"].exists()

    count_paths = _workspace(tmp_path / "count", translation_fixture_count=4)
    with pytest.raises(ValueError, match="source row count mismatch"):
        audit_prompt_sources(
            count_paths["input"],
            receipt_path=count_paths["receipt"],
            materialization_config_path=count_paths["materialization"],
        )
    assert not count_paths["receipt"].exists()
    assert not count_paths["materialization"].exists()


def test_source_audit_rejects_fixture_index_outside_exact_arrow_rows(tmp_path: Path) -> None:
    paths = _workspace(tmp_path, dolly_index=5)
    with pytest.raises(ValueError, match=r"selected source indices are missing: \[5\]"):
        audit_prompt_sources(
            paths["input"],
            receipt_path=paths["receipt"],
            materialization_config_path=paths["materialization"],
        )
    assert not paths["receipt"].exists()
    assert not paths["materialization"].exists()


def test_generated_config_rejects_source_and_hash_drift(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    audit_prompt_sources(
        paths["input"],
        receipt_path=paths["receipt"],
        materialization_config_path=paths["materialization"],
    )
    _write_arrow(
        paths["translation_arrow"],
        [
            {"translation": {"en": "mutated", "fr": "premier"}},
            {"translation": {"en": "second", "fr": "deuxième"}},
            {"translation": {"en": "third", "fr": "troisième"}},
        ],
    )
    with pytest.raises(ValueError, match="source_artifacts.*SHA256 mismatch"):
        build_prompt_artifacts(paths["materialization"], out_dir=tmp_path / "changed-source")
    assert not (tmp_path / "changed-source" / "prompt_materialization_receipt.json").exists()

    clean_paths = _workspace(tmp_path / "config-drift")
    audit_prompt_sources(
        clean_paths["input"],
        receipt_path=clean_paths["receipt"],
        materialization_config_path=clean_paths["materialization"],
    )
    config = json.loads(clean_paths["materialization"].read_text())
    config["protocols"]["tiny_translation_protocol"]["probes"]["translation"][
        "selected_rows_sha256"
    ] = hashlib.sha256(b"tampered").hexdigest()
    _write_json(clean_paths["materialization"], config)
    with pytest.raises(ValueError, match="materialization protocol drift"):
        build_prompt_artifacts(
            clean_paths["materialization"], out_dir=tmp_path / "changed-config"
        )
    assert not (tmp_path / "changed-config").exists()
