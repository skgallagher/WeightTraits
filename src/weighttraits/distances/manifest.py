"""Manifest helpers for distance-cube inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import yaml

from weighttraits.distances.readers import (
    CumulativeLoraReader,
    LoraFactorReader,
    TensorReader,
    reader_from_path,
)
from weighttraits.manifests.reference import leaf_ids, load_manifest
from weighttraits.training.ledger import latest_status_by_node, load_ledger_events


DISTANCE_READY_STATUSES = {"completed", "skipped", "stopped_early"}


@dataclass(frozen=True)
class DistanceInputSpec:
    model_id: str
    checkpoint: Path | None = None
    adapter_chain: tuple[Path, ...] = ()


def load_distance_input_manifest(path: str | Path) -> list[DistanceInputSpec]:
    """Load checkpoint or adapter-chain inputs for `build-distance-cube`.

    Supported row fields:
    - `model_id`, `node_id`, or `id`;
    - exactly one of `checkpoint` / `checkpoint_path` / `path` or `adapter_chain`.
    """

    manifest_path = Path(path)
    rows = _load_rows(manifest_path)
    return [_normalize_row(row, base_dir=manifest_path.parent, row_idx=idx) for idx, row in enumerate(rows)]


def readers_from_distance_manifest(path: str | Path) -> list[TensorReader]:
    specs = load_distance_input_manifest(path)
    readers: list[TensorReader] = []
    for spec in specs:
        if spec.checkpoint is not None:
            readers.append(reader_from_path(spec.checkpoint, model_id=spec.model_id))
            continue

        edge_readers = []
        for adapter_path in spec.adapter_chain:
            reader = reader_from_path(adapter_path)
            if not isinstance(reader, LoraFactorReader):
                raise ValueError(
                    f"adapter_chain entries must be PEFT adapter directories: {adapter_path}"
                )
            edge_readers.append(reader)
        readers.append(CumulativeLoraReader(edge_readers, model_id=spec.model_id))
    return readers


def distance_input_rows_from_training_ledger(
    ledger: str | Path,
    *,
    truth_manifest: str | Path | None = None,
    artifact: str = "model",
    node_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Derive distance input rows from completed training ledger artifacts."""

    if artifact not in {"model", "merged", "adapter_chain"}:
        raise ValueError(f"unsupported ledger artifact mode: {artifact}")
    events = latest_status_by_node(load_ledger_events(ledger))
    targets = _target_node_ids(events, truth_manifest=truth_manifest, node_ids=node_ids)
    if artifact == "adapter_chain":
        if truth_manifest is None:
            raise ValueError("adapter_chain distance inputs require --truth-manifest")
        return _adapter_chain_rows(events, truth_manifest=truth_manifest, targets=targets)
    return [_checkpoint_row(events, node_id=node_id, artifact=artifact) for node_id in targets]


def write_distance_input_manifest(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    path_base: str | Path = ".",
) -> None:
    """Write distance input rows as YAML, relativizing paths to the output file."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalized = _relativize_rows(rows, out.parent, Path(path_base))
    out.write_text(yaml.safe_dump({"models": normalized}, sort_keys=False))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            for key in ("models", "checkpoints", "inputs"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        rows = loaded

    if not isinstance(rows, list):
        raise ValueError(f"distance input manifest must contain a list of rows: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"distance input manifest rows must be objects: {path}")
    return rows


def _target_node_ids(
    events: dict[str, Any],
    *,
    truth_manifest: str | Path | None,
    node_ids: list[str] | None,
) -> list[str]:
    if node_ids:
        return [str(node_id) for node_id in node_ids]
    if truth_manifest is not None:
        return leaf_ids(load_manifest(truth_manifest))
    return sorted(
        node_id
        for node_id, event in events.items()
        if event.status in DISTANCE_READY_STATUSES and _event_artifacts(event)
    )


def _checkpoint_row(events: dict[str, Any], *, node_id: str, artifact: str) -> dict[str, Any]:
    artifacts = _terminal_artifacts(events, node_id)
    try:
        checkpoint = artifacts[artifact]
    except KeyError as exc:
        raise ValueError(f"node {node_id!r} has no {artifact!r} artifact in ledger") from exc
    return {"model_id": node_id, "checkpoint": str(checkpoint)}


def _adapter_chain_rows(
    events: dict[str, Any],
    *,
    truth_manifest: str | Path,
    targets: list[str],
) -> list[dict[str, Any]]:
    records = load_manifest(truth_manifest)
    paths = {record.node_id: record.path for record in records}
    rows = []
    for node_id in targets:
        if node_id not in paths:
            raise ValueError(f"node {node_id!r} is not present in truth manifest")
        chain = []
        for part in paths[node_id]:
            if part == "root":
                continue
            artifacts = _terminal_artifacts(events, part)
            try:
                chain.append(str(artifacts["adapter"]))
            except KeyError as exc:
                raise ValueError(f"node {part!r} has no adapter artifact in ledger") from exc
        if not chain:
            raise ValueError(f"node {node_id!r} has an empty adapter chain")
        rows.append({"model_id": node_id, "adapter_chain": chain})
    return rows


def _terminal_artifacts(events: dict[str, Any], node_id: str) -> dict[str, Any]:
    try:
        event = events[node_id]
    except KeyError as exc:
        raise ValueError(f"node {node_id!r} is missing from ledger") from exc
    if event.status not in DISTANCE_READY_STATUSES:
        raise ValueError(f"node {node_id!r} is not distance-ready in ledger: {event.status}")
    artifacts = _event_artifacts(event)
    if not artifacts:
        raise ValueError(f"node {node_id!r} has no artifacts in ledger")
    return artifacts


def _event_artifacts(event: Any) -> dict[str, Any]:
    artifacts = event.extra.get("artifacts", {})
    return artifacts if isinstance(artifacts, dict) else {}


def _relativize_rows(
    rows: list[dict[str, Any]],
    manifest_dir: Path,
    path_base: Path,
) -> list[dict[str, Any]]:
    return [
        _relativize_row(row, manifest_dir=manifest_dir, path_base=path_base)
        for row in rows
    ]


def _relativize_row(
    row: dict[str, Any],
    *,
    manifest_dir: Path,
    path_base: Path,
) -> dict[str, Any]:
    out = dict(row)
    if "checkpoint" in out:
        out["checkpoint"] = _relativize_path(str(out["checkpoint"]), manifest_dir, path_base)
    if "adapter_chain" in out:
        out["adapter_chain"] = [
            _relativize_path(str(path), manifest_dir, path_base)
            for path in out["adapter_chain"]
        ]
    return out


def _relativize_path(value: str, manifest_dir: Path, path_base: Path) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else path_base / path
    return os.path.relpath(
        absolute.resolve(strict=False),
        manifest_dir.resolve(strict=False),
    )


def _normalize_row(row: dict[str, Any], *, base_dir: Path, row_idx: int) -> DistanceInputSpec:
    checkpoint_value = _first_present(row, ("checkpoint", "checkpoint_path", "path"))
    adapter_value = row.get("adapter_chain")
    if checkpoint_value is not None and adapter_value is not None:
        raise ValueError(f"row {row_idx} has both checkpoint and adapter_chain")
    if checkpoint_value is None and adapter_value is None:
        raise ValueError(f"row {row_idx} needs checkpoint or adapter_chain")

    if checkpoint_value is not None:
        checkpoint = _resolve_path(base_dir, str(checkpoint_value))
        model_id = _model_id(row, fallback=checkpoint.stem, row_idx=row_idx)
        return DistanceInputSpec(model_id=model_id, checkpoint=checkpoint)

    adapter_chain = tuple(_resolve_path(base_dir, item) for item in _as_path_list(adapter_value))
    if not adapter_chain:
        raise ValueError(f"row {row_idx} has an empty adapter_chain")
    model_id = _model_id(row, fallback=adapter_chain[-1].stem, row_idx=row_idx)
    return DistanceInputSpec(model_id=model_id, adapter_chain=adapter_chain)


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _model_id(row: dict[str, Any], *, fallback: str, row_idx: int) -> str:
    value = _first_present(row, ("model_id", "node_id", "id"))
    if value is None:
        value = fallback
    text = str(value)
    if not text:
        raise ValueError(f"row {row_idx} has an empty model id")
    return text


def _as_path_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("adapter_chain must be a comma-separated string or list")


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path
