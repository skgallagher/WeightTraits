"""Manifest helpers for distance-cube inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from weighttraits.distances.readers import (
    CumulativeLoraReader,
    LoraFactorReader,
    TensorReader,
    reader_from_path,
)


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
