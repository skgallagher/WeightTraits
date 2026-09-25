"""Strict, receipt-backed materialization of corrected behavior prompt artifacts.

The source of truth is a SHA-pinned raw dataset artifact, never a legacy prompt JSONL.  The
materializer reads rows without treating Unicode line/paragraph separators as record boundaries,
renders the fixture's explicit source-index panel for one declared model task, and publishes the
receipt only after every prompt artifact has been written and reloaded successfully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from weighttraits.behavior.contracts import load_behavior_protocol_registry
from weighttraits.behavior.probe_inference import canonical_json_sha256, sha256_file
from weighttraits.behavior.responses import load_rendered_behavior_prompts


PROMPT_MATERIALIZATION_INPUT_SCHEMA = "weighttraits.behavior.prompt_materialization_input.v1"
PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2 = "weighttraits.behavior.prompt_materialization_input.v2"
PROMPT_MATERIALIZATION_RECEIPT_SCHEMA = (
    "weighttraits.behavior.prompt_materialization_receipt.v1"
)
PROMPT_SOURCE_AUDIT_INPUT_SCHEMA = "weighttraits.behavior.prompt_source_audit_input.v1"
PROMPT_SOURCE_AUDIT_RECEIPT_SCHEMA = "weighttraits.behavior.prompt_source_audit_receipt.v1"
SUPPORTED_SOURCE_FORMATS = frozenset({"arrow", "parquet", "jsonl"})


def audit_prompt_sources(
    config_path: str | Path,
    *,
    receipt_path: str | Path,
    materialization_config_path: str | Path,
) -> dict[str, Any]:
    """Audit offline dataset shards and emit a receipt-pinned materialization config.

    A source SHA may be supplied in the input to verify a pre-existing pin, or set to ``null``
    for this sealing pass.  In either case the file is hashed before and after it is read, and the
    generated materialization config always contains the observed lowercase SHA256.  A fixture's
    ``source_row_count`` may be null only in the registry; the audit records and authorizes the
    exact observed count without rewriting the registry or fixture.
    """

    config = Path(config_path).resolve(strict=True)
    receipt = Path(receipt_path).resolve()
    materialization_config = Path(materialization_config_path).resolve()
    for output in (receipt, materialization_config):
        _reject_legacy_output(output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite source-audit output: {output}")
    if receipt == materialization_config:
        raise ValueError("source-audit receipt and materialization config must be different files")

    raw = _json_object(config)
    _exact_keys(
        raw,
        {"schema", "model_task", "registry", "protocols"},
        context="prompt source audit input",
    )
    if raw["schema"] != PROMPT_SOURCE_AUDIT_INPUT_SCHEMA:
        raise ValueError(f"prompt source audit schema must be {PROMPT_SOURCE_AUDIT_INPUT_SCHEMA!r}")
    model_task = _model_task(raw["model_task"])
    registry_spec = _mapping(raw["registry"], "registry")
    _exact_keys(registry_spec, {"path", "sha256"}, context="registry")
    registry_path, registry_sha = _pinned_file(
        registry_spec, base=config.parent, label="registry"
    )
    registry = load_behavior_protocol_registry(registry_path)
    if registry.sha256 != registry_sha:
        raise ValueError("registry loader hash disagrees with the pinned registry hash")

    protocol_inputs = _mapping(raw["protocols"], "protocols")
    if not protocol_inputs:
        raise ValueError("prompt source audit must declare at least one protocol")
    materialization_protocols: dict[str, dict[str, Any]] = {}
    audit_protocols: dict[str, dict[str, Any]] = {}
    n_probes = 0
    n_prompts = 0
    n_resolved_fixture_counts = 0
    for protocol_id, raw_protocol in protocol_inputs.items():
        protocol = registry.protocol(protocol_id)
        protocol_input = _mapping(raw_protocol, f"protocols.{protocol_id}")
        _exact_keys(protocol_input, {"probes"}, context=f"protocols.{protocol_id}")
        probe_inputs = _mapping(protocol_input["probes"], f"protocols.{protocol_id}.probes")
        if set(probe_inputs) != set(protocol.probe_ids):
            raise ValueError(
                f"protocol {protocol_id!r} probe declarations differ from the registry: "
                f"expected={sorted(protocol.probe_ids)}, got={sorted(probe_inputs)}"
            )

        fixture = registry.fixtures[protocol.fixture_id]
        materialization_probes: dict[str, dict[str, Any]] = {}
        audited_probes: dict[str, dict[str, Any]] = {}
        for probe_id in protocol.probe_ids:
            probe = fixture.probes[probe_id]
            probe_input = _mapping(
                probe_inputs[probe_id], f"protocols.{protocol_id}.probes.{probe_id}"
            )
            _exact_keys(
                probe_input,
                {"source_artifacts"},
                context=f"protocols.{protocol_id}.probes.{probe_id}",
            )
            artifact_inputs = _source_artifact_inputs(
                probe_input["source_artifacts"],
                base=config.parent,
                context=f"protocols.{protocol_id}.probes.{probe_id}.source_artifacts",
            )
            rows, source_row_count, artifact_pins, artifact_audit = _audit_source_artifacts(
                artifact_inputs,
                indices=probe.selection.source_indices,
            )
            fixture_count = probe.selection.source_row_count
            if fixture_count is not None and fixture_count != source_row_count:
                raise ValueError(
                    f"{protocol_id}/{probe_id} source row count mismatch: "
                    f"fixture={fixture_count}, observed={source_row_count}"
                )
            if fixture_count is None:
                n_resolved_fixture_counts += 1
            indices_sha = canonical_json_sha256(list(probe.selection.source_indices))
            if indices_sha != probe.selection.source_indices_sha256:
                raise ValueError(
                    f"{protocol_id}/{probe_id} fixture source-index hash mismatch: "
                    f"expected {probe.selection.source_indices_sha256}, observed {indices_sha}"
                )
            selected = [rows[index] for index in probe.selection.source_indices]
            selected_sha = canonical_json_sha256(selected)
            prompts = registry.materialize_probe(
                protocol_id=protocol_id,
                probe_id=probe_id,
                rows=rows,
                model_task=model_task,
                dataset_name=probe.dataset.name,
                dataset_config=probe.dataset.config,
                dataset_revision=probe.dataset.revision,
            )
            rendered_sha = canonical_json_sha256([prompt.prompt for prompt in prompts])
            materialization_probes[probe_id] = {
                "dataset_name": probe.dataset.name,
                "dataset_config": probe.dataset.config,
                "dataset_revision": probe.dataset.revision,
                "source_row_count": source_row_count,
                "source_artifacts": artifact_pins,
                "selected_rows_sha256": selected_sha,
                "rendered_prompts_sha256": rendered_sha,
            }
            audited_probes[probe_id] = {
                "dataset_name": probe.dataset.name,
                "dataset_config": probe.dataset.config,
                "dataset_split": probe.dataset.split,
                "dataset_revision": probe.dataset.revision,
                "fixture_source_row_count": fixture_count,
                "source_row_count": source_row_count,
                "resolved_fixture_source_row_count": fixture_count is None,
                "n_prompts": len(prompts),
                "source_indices": list(probe.selection.source_indices),
                "source_indices_sha256": indices_sha,
                "selected_rows_sha256": selected_sha,
                "rendered_prompts_sha256": rendered_sha,
                "source_artifacts": artifact_audit,
            }
            n_probes += 1
            n_prompts += len(prompts)
        materialization_protocols[protocol_id] = {"probes": materialization_probes}
        audit_protocols[protocol_id] = {
            "fixture_id": fixture.fixture_id,
            "fixture_sha256": fixture.sha256,
            "probes": audited_probes,
        }

    protocol_payload_sha = canonical_json_sha256(materialization_protocols)
    receipt_payload = {
        "schema": PROMPT_SOURCE_AUDIT_RECEIPT_SCHEMA,
        "valid": True,
        "model_task": model_task,
        "input_config": {"path": str(config), "sha256": sha256_file(config)},
        "registry": {"path": str(registry_path), "sha256": registry_sha},
        "protocols": audit_protocols,
        "materialization_protocols_sha256": protocol_payload_sha,
        "n_protocols": len(audit_protocols),
        "n_probes": n_probes,
        "n_prompts": n_prompts,
        "n_resolved_fixture_source_row_counts": n_resolved_fixture_counts,
    }
    _write_json_atomic_new(receipt, receipt_payload)
    receipt_pin = {"path": str(receipt), "sha256": sha256_file(receipt)}
    materialization_payload = {
        "schema": PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2,
        "model_task": model_task,
        "registry": {"path": str(registry_path), "sha256": registry_sha},
        "source_audit": receipt_pin,
        "protocols": materialization_protocols,
    }
    _write_json_atomic_new(materialization_config, materialization_payload)
    return {
        "receipt": receipt_payload,
        "receipt_artifact": receipt_pin,
        "materialization_config": {
            "path": str(materialization_config),
            "sha256": sha256_file(materialization_config),
        },
    }


def build_prompt_artifacts(
    config_path: str | Path,
    *,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Materialize every explicitly declared protocol and write one terminal receipt.

    Existing output directories are rejected.  All individual files use same-directory temporary
    files and ``os.replace``; the receipt is written last, so an interrupted directory is never a
    valid prompt materialization.
    """

    config = Path(config_path).resolve(strict=True)
    output = Path(out_dir).resolve()
    _reject_legacy_output(output)
    if output.exists():
        raise FileExistsError(f"prompt output directory already exists: {output}")
    raw = _json_object(config)
    source_audit: Mapping[str, Any] | None = None
    if raw.get("schema") == PROMPT_MATERIALIZATION_INPUT_SCHEMA:
        _exact_keys(
            raw,
            {"schema", "model_task", "registry", "protocols"},
            context="prompt materialization input",
        )
    elif raw.get("schema") == PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2:
        _exact_keys(
            raw,
            {"schema", "model_task", "registry", "source_audit", "protocols"},
            context="prompt materialization input",
        )
    else:
        raise ValueError(
            "prompt materialization schema must be "
            f"{PROMPT_MATERIALIZATION_INPUT_SCHEMA!r} or "
            f"{PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2!r}"
        )
    model_task = _model_task(raw["model_task"])
    base = config.parent
    registry_path, registry_sha = _pinned_file(raw["registry"], base=base, label="registry")
    registry = load_behavior_protocol_registry(registry_path)
    if registry.sha256 != registry_sha:
        raise ValueError("registry loader hash disagrees with the pinned registry hash")

    protocol_specs = _mapping(raw["protocols"], "protocols")
    if not protocol_specs:
        raise ValueError("prompt materialization must declare at least one protocol")
    if raw["schema"] == PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2:
        source_audit = _validate_source_audit_pin(
            raw["source_audit"],
            base=base,
            model_task=model_task,
            registry_path=registry_path,
            registry_sha256=registry_sha,
            registry=registry,
            protocol_specs=protocol_specs,
        )
    output.mkdir(parents=True)
    output_rows: dict[str, dict[str, Any]] = {}
    source_receipts: dict[str, dict[str, Any]] = {}
    try:
        for protocol_id, raw_protocol in protocol_specs.items():
            protocol = registry.protocol(protocol_id)
            protocol_spec = _mapping(raw_protocol, f"protocols.{protocol_id}")
            _exact_keys(
                protocol_spec,
                {"probes"},
                context=f"protocols.{protocol_id}",
            )
            probe_specs = _mapping(protocol_spec["probes"], f"protocols.{protocol_id}.probes")
            if set(probe_specs) != set(protocol.probe_ids):
                raise ValueError(
                    f"protocol {protocol_id!r} probe declarations differ from the registry: "
                    f"expected={sorted(protocol.probe_ids)}, got={sorted(probe_specs)}"
                )

            rendered = []
            selected_rows: list[dict[str, Any]] = []
            protocol_sources: dict[str, Any] = {}
            for probe_id in protocol.probe_ids:
                fixture = registry.fixtures[protocol.fixture_id]
                probe = fixture.probes[probe_id]
                spec = _mapping(
                    probe_specs[probe_id], f"protocols.{protocol_id}.probes.{probe_id}"
                )
                _exact_keys(
                    spec,
                    {
                        "dataset_name",
                        "dataset_config",
                        "dataset_revision",
                        "source_row_count",
                        "source_artifacts",
                        "selected_rows_sha256",
                        "rendered_prompts_sha256",
                    },
                    context=f"protocols.{protocol_id}.probes.{probe_id}",
                )
                expected_count = _positive_integer(
                    spec["source_row_count"],
                    f"protocols.{protocol_id}.probes.{probe_id}.source_row_count",
                )
                if probe.selection.source_row_count is None:
                    if source_audit is None:
                        raise ValueError(
                            f"fixture {fixture.fixture_id!r} probe {probe_id!r} must declare "
                            "selection.source_row_count or use a valid v2 source-audit receipt"
                        )
                    audited_probe = _audit_probe_row(
                        source_audit, protocol_id=protocol_id, probe_id=probe_id
                    )
                    if (
                        audited_probe.get("fixture_source_row_count") is not None
                        or audited_probe.get("resolved_fixture_source_row_count") is not True
                        or audited_probe.get("source_row_count") != expected_count
                    ):
                        raise ValueError(
                            f"{protocol_id}/{probe_id} source audit does not authorize the "
                            "resolved fixture row count"
                        )
                elif probe.selection.source_row_count != expected_count:
                    raise ValueError(
                        f"{protocol_id}/{probe_id} source_row_count differs from fixture: "
                        f"{expected_count} != {probe.selection.source_row_count}"
                    )
                artifacts = _load_source_artifacts(
                    spec["source_artifacts"],
                    base=base,
                    context=f"protocols.{protocol_id}.probes.{probe_id}.source_artifacts",
                )
                rows, observed_count = _selected_source_rows(
                    artifacts,
                    indices=probe.selection.source_indices,
                )
                if observed_count != expected_count:
                    raise ValueError(
                        f"{protocol_id}/{probe_id} source row count mismatch: "
                        f"expected {expected_count}, observed {observed_count}"
                    )
                prompts = registry.materialize_probe(
                    protocol_id=protocol_id,
                    probe_id=probe_id,
                    rows=rows,
                    model_task=model_task,
                    dataset_name=_nonempty(spec["dataset_name"], "dataset_name"),
                    dataset_config=_optional_string(spec["dataset_config"], "dataset_config"),
                    dataset_revision=_nonempty(spec["dataset_revision"], "dataset_revision"),
                )
                # These two aggregate definitions reproduce the independent source audit exactly:
                # the bare ordered raw row list and the bare ordered rendered prompt-text list.
                # There are deliberately no wrapper keys, prompt IDs, references, or newlines.
                probe_selected_rows = [
                    rows[source_index] for source_index in probe.selection.source_indices
                ]
                probe_selected_sha = canonical_json_sha256(probe_selected_rows)
                expected_probe_selected = _sha256(
                    spec["selected_rows_sha256"],
                    f"protocols.{protocol_id}.probes.{probe_id}.selected_rows_sha256",
                )
                if probe_selected_sha != expected_probe_selected:
                    raise ValueError(
                        f"{protocol_id}/{probe_id} selected-row aggregate hash mismatch: "
                        f"expected {expected_probe_selected}, observed {probe_selected_sha}"
                    )
                probe_rendered_rows = [prompt.prompt for prompt in prompts]
                probe_rendered_sha = canonical_json_sha256(probe_rendered_rows)
                expected_probe_rendered = _sha256(
                    spec["rendered_prompts_sha256"],
                    f"protocols.{protocol_id}.probes.{probe_id}.rendered_prompts_sha256",
                )
                if probe_rendered_sha != expected_probe_rendered:
                    raise ValueError(
                        f"{protocol_id}/{probe_id} rendered-prompt aggregate hash mismatch: "
                        f"expected {expected_probe_rendered}, observed {probe_rendered_sha}"
                    )
                rendered.extend(prompts)
                selected_rows.extend(
                    {
                        "probe_id": prompt.probe_id,
                        "source_index": prompt.source_index,
                        "source_row_sha256": prompt.source_row_sha256,
                    }
                    for prompt in prompts
                )
                protocol_sources[probe_id] = {
                    "source_row_count": observed_count,
                    "source_indices_sha256": probe.selection.source_indices_sha256,
                    "selected_rows_sha256": probe_selected_sha,
                    "rendered_prompts_sha256": probe_rendered_sha,
                    "artifacts": artifacts,
                }

            selected_sha = canonical_json_sha256(selected_rows)
            rendered_rows = [prompt.to_dict() for prompt in rendered]
            rendered_sha = canonical_json_sha256(rendered_rows)
            artifact_path = output / f"{protocol_id}.{model_task}.prompts.jsonl"
            _write_jsonl_atomic(artifact_path, rendered_rows)
            reloaded = load_rendered_behavior_prompts(artifact_path)
            if [row.to_dict() for row in reloaded] != rendered_rows:
                raise RuntimeError(f"rendered prompt reload drift: {artifact_path}")
            output_rows[protocol_id] = {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
                "n_prompts": len(rendered_rows),
                "selected_rows_sha256": selected_sha,
                "rendered_prompts_sha256": rendered_sha,
                "fixture_id": protocol.fixture_id,
                "fixture_sha256": registry.fixtures[protocol.fixture_id].sha256,
                "probe_ids": list(protocol.probe_ids),
            }
            source_receipts[protocol_id] = protocol_sources

        receipt = {
            "schema": PROMPT_MATERIALIZATION_RECEIPT_SCHEMA,
            "valid": True,
            "model_task": model_task,
            "input_config": {"path": str(config), "sha256": sha256_file(config)},
            "registry": {"path": str(registry_path), "sha256": registry_sha},
            "sources": source_receipts,
            "outputs": output_rows,
            "n_protocols": len(output_rows),
            "n_prompts": sum(row["n_prompts"] for row in output_rows.values()),
        }
        _write_json_atomic(output / "prompt_materialization_receipt.json", receipt)
        return receipt
    except Exception:
        # Keep any partial directory for diagnostics; absence of the terminal receipt makes it
        # unambiguously invalid and prevents a silent retry over mixed files.
        raise


def validate_prompt_materialization_receipt(
    receipt_path: str | Path,
    *,
    registry_path: Path,
    registry_sha256: str,
    model_task: str,
    protocol_id: str,
    prompt_path: Path,
    prompt_sha256: str,
) -> Mapping[str, Any]:
    """Cross-check one prompt artifact against its actual materialization receipt."""

    receipt_file = Path(receipt_path).resolve(strict=True)
    payload = _json_object(receipt_file)
    if payload.get("schema") != PROMPT_MATERIALIZATION_RECEIPT_SCHEMA:
        raise ValueError(f"invalid prompt materialization receipt schema: {receipt_file}")
    if payload.get("valid") is not True:
        raise ValueError(f"prompt materialization receipt is not valid: {receipt_file}")
    if payload.get("model_task") != _model_task(model_task):
        raise ValueError("prompt materialization receipt model_task drift")
    registry = _mapping(payload.get("registry"), "prompt receipt registry")
    if Path(str(registry.get("path"))).resolve(strict=True) != registry_path.resolve(strict=True):
        raise ValueError("prompt materialization receipt registry path drift")
    if registry.get("sha256") != registry_sha256:
        raise ValueError("prompt materialization receipt registry hash drift")
    outputs = _mapping(payload.get("outputs"), "prompt receipt outputs")
    if protocol_id not in outputs:
        raise ValueError(f"prompt receipt does not contain protocol {protocol_id!r}")
    row = _mapping(outputs[protocol_id], f"prompt receipt outputs.{protocol_id}")
    if Path(str(row.get("path"))).resolve(strict=True) != prompt_path.resolve(strict=True):
        raise ValueError(f"prompt receipt path drift for {protocol_id}")
    if row.get("sha256") != prompt_sha256 or sha256_file(prompt_path) != prompt_sha256:
        raise ValueError(f"prompt receipt artifact hash drift for {protocol_id}")
    prompts = load_rendered_behavior_prompts(prompt_path)
    if canonical_json_sha256([prompt.to_dict() for prompt in prompts]) != row.get(
        "rendered_prompts_sha256"
    ):
        raise ValueError(f"prompt receipt rendered aggregate drift for {protocol_id}")
    return payload


def _source_artifact_inputs(
    raw: Any,
    *,
    base: Path,
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty list")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        item_context = f"{context}[{index}]"
        spec = _mapping(value, item_context)
        _exact_keys(spec, {"path", "sha256", "format"}, context=item_context)
        path_value = _nonempty(spec["path"], f"{item_context}.path")
        path = Path(path_value)
        path = path.resolve(strict=True) if path.is_absolute() else (base / path).resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(f"missing source artifact: {path}")
        source_format = _nonempty(spec["format"], f"{item_context}.format")
        if source_format not in SUPPORTED_SOURCE_FORMATS:
            raise ValueError(f"unsupported source format {source_format!r} at {item_context}")
        expected_sha = spec["sha256"]
        if expected_sha is not None:
            expected_sha = _sha256(expected_sha, f"{item_context}.sha256")
        result.append(
            {
                "path": path,
                "format": source_format,
                "expected_sha256": expected_sha,
            }
        )
    paths = [row["path"] for row in result]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{context} contains duplicate source paths")
    return result


def _audit_source_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    indices: Sequence[int],
) -> tuple[
    dict[int, Mapping[str, Any]],
    int,
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    requested = set(indices)
    selected_rows: dict[int, Mapping[str, Any]] = {}
    materialization_pins: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    offset = 0
    for artifact in artifacts:
        path = Path(artifact["path"])
        source_format = str(artifact["format"])
        before_stat = path.stat()
        before_sha = sha256_file(path)
        expected_sha = artifact.get("expected_sha256")
        if expected_sha is not None and before_sha != expected_sha:
            raise ValueError(
                f"source artifact SHA256 mismatch: expected {expected_sha}, "
                f"observed {before_sha}: {path}"
            )
        shard_rows = _read_source_shard(path, source_format)
        after_sha = sha256_file(path)
        after_stat = path.stat()
        before_identity = (
            before_stat.st_dev,
            before_stat.st_ino,
            before_stat.st_size,
            before_stat.st_mtime_ns,
        )
        after_identity = (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_size,
            after_stat.st_mtime_ns,
        )
        if before_sha != after_sha or before_identity != after_identity:
            raise ValueError(f"source artifact changed while it was audited: {path}")
        for local_index, row in enumerate(shard_rows):
            global_index = offset + local_index
            if global_index in requested:
                if not isinstance(row, Mapping):
                    raise ValueError(f"source row {global_index} is not an object: {path}")
                selected_rows[global_index] = dict(row)
        n_rows = len(shard_rows)
        materialization_pins.append(
            {"path": str(path), "sha256": before_sha, "format": source_format}
        )
        audit_rows.append(
            {
                "path": str(path),
                "input_sha256": expected_sha,
                "sha256": before_sha,
                "format": source_format,
                "size_bytes": before_stat.st_size,
                "n_rows": n_rows,
                "global_start_index": offset,
                "global_stop_index_exclusive": offset + n_rows,
            }
        )
        offset += n_rows
    if offset < 1:
        raise ValueError("source artifacts contain zero rows")
    missing = sorted(requested - set(selected_rows))
    if missing:
        raise ValueError(f"selected source indices are missing: {missing[:20]}")
    return selected_rows, offset, materialization_pins, audit_rows


def _validate_source_audit_pin(
    raw: Any,
    *,
    base: Path,
    model_task: str,
    registry_path: Path,
    registry_sha256: str,
    registry: Any,
    protocol_specs: Mapping[str, Any],
) -> Mapping[str, Any]:
    spec = _mapping(raw, "source_audit")
    _exact_keys(spec, {"path", "sha256"}, context="source_audit")
    receipt_path, _ = _pinned_file(spec, base=base, label="source_audit")
    payload = _json_object(receipt_path)
    _exact_keys(
        payload,
        {
            "schema",
            "valid",
            "model_task",
            "input_config",
            "registry",
            "protocols",
            "materialization_protocols_sha256",
            "n_protocols",
            "n_probes",
            "n_prompts",
            "n_resolved_fixture_source_row_counts",
        },
        context="prompt source audit receipt",
    )
    if payload.get("schema") != PROMPT_SOURCE_AUDIT_RECEIPT_SCHEMA:
        raise ValueError(f"invalid prompt source audit receipt schema: {receipt_path}")
    if payload.get("valid") is not True:
        raise ValueError(f"prompt source audit receipt is not valid: {receipt_path}")
    if payload.get("model_task") != model_task:
        raise ValueError("prompt source audit model_task drift")
    audit_input = _mapping(payload.get("input_config"), "source audit input_config")
    _exact_keys(audit_input, {"path", "sha256"}, context="source audit input_config")
    _pinned_file(audit_input, base=receipt_path.parent, label="source audit input_config")
    receipt_registry = _mapping(payload.get("registry"), "source audit registry")
    _exact_keys(receipt_registry, {"path", "sha256"}, context="source audit registry")
    if Path(str(receipt_registry.get("path"))).resolve(strict=True) != registry_path:
        raise ValueError("prompt source audit registry path drift")
    if receipt_registry.get("sha256") != registry_sha256:
        raise ValueError("prompt source audit registry hash drift")
    observed_protocols_sha = canonical_json_sha256(protocol_specs)
    if payload.get("materialization_protocols_sha256") != observed_protocols_sha:
        raise ValueError("prompt source audit materialization protocol drift")
    audit_protocols = _mapping(payload.get("protocols"), "source audit protocols")
    if set(audit_protocols) != set(protocol_specs):
        raise ValueError("prompt source audit protocol set drift")
    observed_probe_count = 0
    observed_prompt_count = 0
    observed_resolved_count = 0
    for protocol_id in protocol_specs:
        protocol = registry.protocol(protocol_id)
        fixture = registry.fixtures[protocol.fixture_id]
        audit_protocol = _mapping(
            audit_protocols[protocol_id], f"source audit protocols.{protocol_id}"
        )
        _exact_keys(
            audit_protocol,
            {"fixture_id", "fixture_sha256", "probes"},
            context=f"source audit protocols.{protocol_id}",
        )
        if (
            audit_protocol.get("fixture_id") != fixture.fixture_id
            or audit_protocol.get("fixture_sha256") != fixture.sha256
        ):
            raise ValueError(f"prompt source audit fixture drift for {protocol_id}")
        audit_probes = _mapping(
            audit_protocol.get("probes"), f"source audit protocols.{protocol_id}.probes"
        )
        if set(audit_probes) != set(protocol.probe_ids):
            raise ValueError(f"prompt source audit probe set drift for {protocol_id}")
        for probe_id in protocol.probe_ids:
            probe = fixture.probes[probe_id]
            audit_probe = _mapping(
                audit_probes[probe_id],
                f"source audit protocols.{protocol_id}.probes.{probe_id}",
            )
            _exact_keys(
                audit_probe,
                {
                    "dataset_name",
                    "dataset_config",
                    "dataset_split",
                    "dataset_revision",
                    "fixture_source_row_count",
                    "source_row_count",
                    "resolved_fixture_source_row_count",
                    "n_prompts",
                    "source_indices",
                    "source_indices_sha256",
                    "selected_rows_sha256",
                    "rendered_prompts_sha256",
                    "source_artifacts",
                },
                context=f"source audit protocols.{protocol_id}.probes.{probe_id}",
            )
            if audit_probe.get("source_indices") != list(probe.selection.source_indices):
                raise ValueError(f"prompt source audit indices drift for {protocol_id}/{probe_id}")
            if audit_probe.get("source_indices_sha256") != probe.selection.source_indices_sha256:
                raise ValueError(
                    f"prompt source audit source-index hash drift for {protocol_id}/{probe_id}"
                )
            if audit_probe.get("fixture_source_row_count") != probe.selection.source_row_count:
                raise ValueError(
                    f"prompt source audit fixture row-count drift for {protocol_id}/{probe_id}"
                )
            expected_resolved = probe.selection.source_row_count is None
            if audit_probe.get("resolved_fixture_source_row_count") is not expected_resolved:
                raise ValueError(
                    f"prompt source audit row-count resolution drift for {protocol_id}/{probe_id}"
                )
            source_row_count = _positive_integer(
                audit_probe.get("source_row_count"),
                f"source audit protocols.{protocol_id}.probes.{probe_id}.source_row_count",
            )
            config_protocol = _mapping(
                protocol_specs[protocol_id], f"protocols.{protocol_id}"
            )
            config_probes = _mapping(
                config_protocol.get("probes"), f"protocols.{protocol_id}.probes"
            )
            config_probe = _mapping(
                config_probes.get(probe_id), f"protocols.{protocol_id}.probes.{probe_id}"
            )
            if config_probe.get("source_row_count") != source_row_count:
                raise ValueError(
                    f"prompt source audit source row-count drift for {protocol_id}/{probe_id}"
                )
            for key in (
                "dataset_name",
                "dataset_config",
                "dataset_revision",
                "selected_rows_sha256",
                "rendered_prompts_sha256",
            ):
                if config_probe.get(key) != audit_probe.get(key):
                    raise ValueError(
                        f"prompt source audit {key} drift for {protocol_id}/{probe_id}"
                    )
            config_artifacts = config_probe.get("source_artifacts")
            audit_artifacts = audit_probe.get("source_artifacts")
            if not isinstance(config_artifacts, list) or not isinstance(audit_artifacts, list):
                raise ValueError(
                    f"prompt source audit artifacts must be lists for {protocol_id}/{probe_id}"
                )
            audited_pins: list[dict[str, Any]] = []
            artifact_offset = 0
            for index, artifact_value in enumerate(audit_artifacts):
                artifact = _mapping(
                    artifact_value,
                    f"source audit protocols.{protocol_id}.probes.{probe_id}."
                    f"source_artifacts[{index}]",
                )
                _exact_keys(
                    artifact,
                    {
                        "path",
                        "input_sha256",
                        "sha256",
                        "format",
                        "size_bytes",
                        "n_rows",
                        "global_start_index",
                        "global_stop_index_exclusive",
                    },
                    context=f"source audit protocols.{protocol_id}.probes.{probe_id}."
                    f"source_artifacts[{index}]",
                )
                _sha256(
                    artifact.get("sha256"),
                    f"source audit protocols.{protocol_id}.probes.{probe_id}."
                    f"source_artifacts[{index}].sha256",
                )
                input_sha = artifact.get("input_sha256")
                if input_sha is not None:
                    _sha256(
                        input_sha,
                        f"source audit protocols.{protocol_id}.probes.{probe_id}."
                        f"source_artifacts[{index}].input_sha256",
                    )
                if artifact.get("format") not in SUPPORTED_SOURCE_FORMATS:
                    raise ValueError(
                        f"prompt source audit format drift for {protocol_id}/{probe_id}"
                    )
                n_rows = artifact.get("n_rows")
                if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 0:
                    raise ValueError(
                        f"prompt source audit shard row count is invalid for "
                        f"{protocol_id}/{probe_id}"
                    )
                size_bytes = artifact.get("size_bytes")
                if (
                    isinstance(size_bytes, bool)
                    or not isinstance(size_bytes, int)
                    or size_bytes < 1
                ):
                    raise ValueError(
                        f"prompt source audit shard size is invalid for {protocol_id}/{probe_id}"
                    )
                if (
                    artifact.get("global_start_index") != artifact_offset
                    or artifact.get("global_stop_index_exclusive") != artifact_offset + n_rows
                ):
                    raise ValueError(
                        f"prompt source audit shard offsets drift for {protocol_id}/{probe_id}"
                    )
                artifact_offset += n_rows
                audited_pins.append(
                    {
                        "path": artifact.get("path"),
                        "sha256": artifact.get("sha256"),
                        "format": artifact.get("format"),
                    }
                )
            if config_artifacts != audited_pins:
                raise ValueError(
                    f"prompt source audit artifact pins drift for {protocol_id}/{probe_id}"
                )
            if artifact_offset != source_row_count:
                raise ValueError(
                    f"prompt source audit shard rows drift for {protocol_id}/{probe_id}"
                )
            for index, artifact_value in enumerate(config_artifacts):
                artifact = _mapping(
                    artifact_value,
                    f"protocols.{protocol_id}.probes.{probe_id}.source_artifacts[{index}]",
                )
                _exact_keys(
                    artifact,
                    {"path", "sha256", "format"},
                    context=f"protocols.{protocol_id}.probes.{probe_id}."
                    f"source_artifacts[{index}]",
                )
                _pinned_file(
                    artifact,
                    base=base,
                    label=f"protocols.{protocol_id}.probes.{probe_id}."
                    f"source_artifacts[{index}]",
                )
            if audit_probe.get("dataset_name") != probe.dataset.name:
                raise ValueError(f"prompt source audit dataset drift for {protocol_id}/{probe_id}")
            if audit_probe.get("dataset_config") != probe.dataset.config:
                raise ValueError(
                    f"prompt source audit dataset config drift for {protocol_id}/{probe_id}"
                )
            if audit_probe.get("dataset_split") != probe.dataset.split:
                raise ValueError(
                    f"prompt source audit dataset split drift for {protocol_id}/{probe_id}"
                )
            if audit_probe.get("dataset_revision") != probe.dataset.revision:
                raise ValueError(
                    f"prompt source audit dataset revision drift for {protocol_id}/{probe_id}"
                )
            n_probe_prompts = _positive_integer(
                audit_probe.get("n_prompts"),
                f"source audit protocols.{protocol_id}.probes.{probe_id}.n_prompts",
            )
            if n_probe_prompts != probe.selection.n_prompts:
                raise ValueError(f"prompt source audit prompt count drift for {protocol_id}/{probe_id}")
            if expected_resolved:
                observed_resolved_count += 1
            observed_probe_count += 1
            observed_prompt_count += n_probe_prompts
    if payload.get("n_protocols") != len(protocol_specs):
        raise ValueError("prompt source audit n_protocols drift")
    if payload.get("n_probes") != observed_probe_count:
        raise ValueError("prompt source audit n_probes drift")
    if payload.get("n_prompts") != observed_prompt_count:
        raise ValueError("prompt source audit n_prompts drift")
    if payload.get("n_resolved_fixture_source_row_counts") != observed_resolved_count:
        raise ValueError("prompt source audit resolved fixture row-count total drift")
    return payload


def _audit_probe_row(
    source_audit: Mapping[str, Any],
    *,
    protocol_id: str,
    probe_id: str,
) -> Mapping[str, Any]:
    protocols = _mapping(source_audit.get("protocols"), "source audit protocols")
    protocol = _mapping(protocols.get(protocol_id), f"source audit protocols.{protocol_id}")
    probes = _mapping(protocol.get("probes"), f"source audit protocols.{protocol_id}.probes")
    return _mapping(
        probes.get(probe_id), f"source audit protocols.{protocol_id}.probes.{probe_id}"
    )


def _load_source_artifacts(
    raw: Any,
    *,
    base: Path,
    context: str,
) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty list")
    result: list[dict[str, str]] = []
    for index, value in enumerate(raw):
        spec = _mapping(value, f"{context}[{index}]")
        _exact_keys(spec, {"path", "sha256", "format"}, context=f"{context}[{index}]")
        path, digest = _pinned_file(spec, base=base, label=f"{context}[{index}]")
        source_format = _nonempty(spec["format"], f"{context}[{index}].format")
        if source_format not in SUPPORTED_SOURCE_FORMATS:
            raise ValueError(f"unsupported source format {source_format!r} at {context}[{index}]")
        result.append({"path": str(path), "sha256": digest, "format": source_format})
    paths = [row["path"] for row in result]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{context} contains duplicate source paths")
    return result


def _selected_source_rows(
    artifacts: Sequence[Mapping[str, str]],
    *,
    indices: Sequence[int],
) -> tuple[dict[int, Mapping[str, Any]], int]:
    requested = set(indices)
    rows: dict[int, Mapping[str, Any]] = {}
    offset = 0
    for artifact in artifacts:
        path = Path(artifact["path"])
        source_format = artifact["format"]
        shard_rows = _read_source_shard(path, source_format)
        for local_index, row in enumerate(shard_rows):
            global_index = offset + local_index
            if global_index in requested:
                if not isinstance(row, Mapping):
                    raise ValueError(f"source row {global_index} is not an object: {path}")
                rows[global_index] = dict(row)
        offset += len(shard_rows)
    missing = sorted(requested - set(rows))
    if missing:
        raise ValueError(f"selected source indices are missing: {missing[:20]}")
    return rows, offset


def _read_source_shard(path: Path, source_format: str) -> Sequence[Mapping[str, Any]]:
    if source_format == "jsonl":
        rows: list[Mapping[str, Any]] = []
        # File iteration splits only on the actual newline character.  In particular, U+2028
        # embedded in MMLU text remains within the JSON string and is never treated as a row break.
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") and line_number > 0:
                    pass
                if not line.strip():
                    raise ValueError(f"blank JSONL row is forbidden: {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
                rows.append(dict(value))
        return rows
    if source_format == "arrow":
        try:
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("datasets is required to read pinned Arrow sources") from exc
        dataset = Dataset.from_file(str(path))
        return [dict(dataset[index]) for index in range(len(dataset))]
    if source_format == "parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("pyarrow is required to read pinned Parquet sources") from exc
        return [dict(row) for row in parquet.read_table(path).to_pylist()]
    raise AssertionError(f"unreachable source format: {source_format}")


def _pinned_file(raw: Any, *, base: Path, label: str) -> tuple[Path, str]:
    spec = _mapping(raw, label)
    path = Path(_nonempty(spec.get("path"), f"{label}.path"))
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    digest = _sha256(spec.get("sha256"), f"{label}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(f"{label} SHA256 mismatch: expected {digest}, observed {observed}")
    return path, digest


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    _write_text_atomic(path, payload)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_json_atomic_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    _write_text_atomic_new(path, payload)


def _write_text_atomic_new(path: Path, text: str) -> None:
    """Atomically create a file while refusing both pre-existing and racing writers."""

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
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{context} schema mismatch: missing={missing}, unknown={unknown}")


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be non-empty text without surrounding whitespace")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, context)


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _sha256(value: Any, context: str) -> str:
    text = _nonempty(value, context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return text


def _model_task(value: Any) -> str:
    task = _nonempty(value, "model_task")
    if task not in {"causal_lm", "seq2seq"}:
        raise ValueError("model_task must be exactly 'causal_lm' or 'seq2seq'")
    return task


def _reject_legacy_output(path: Path) -> None:
    if any(part.lower() == "ellmtrees" for part in path.parts):
        raise ValueError(f"refusing to write corrected behavior outputs under legacy ELLMTrees: {path}")


__all__ = [
    "PROMPT_MATERIALIZATION_INPUT_SCHEMA",
    "PROMPT_MATERIALIZATION_INPUT_SCHEMA_V2",
    "PROMPT_MATERIALIZATION_RECEIPT_SCHEMA",
    "PROMPT_SOURCE_AUDIT_INPUT_SCHEMA",
    "PROMPT_SOURCE_AUDIT_RECEIPT_SCHEMA",
    "audit_prompt_sources",
    "build_prompt_artifacts",
    "validate_prompt_materialization_receipt",
]
