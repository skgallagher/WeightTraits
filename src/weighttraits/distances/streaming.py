"""Streaming distance-cube construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from weighttraits.distances.metrics import linear_cka_distance, pairwise_distance_matrix
from weighttraits.distances.readers import TensorReader


VECTOR_METRICS = {"cosine", "l1", "l2", "correlation", "threshold"}
MATRIX_METRICS = {"cka", "linear_cka"}


@dataclass
class DistanceCube:
    distances: dict[str, np.ndarray]
    layer_names: list[str]
    model_ids: list[str]
    audit: dict[str, Any]


def build_distance_cube(
    readers: list[TensorReader],
    *,
    metrics: list[str],
    chunk_size: int = 1_000_000,
    eps: float = 1e-3,
    representation: str = "full_weight",
) -> DistanceCube:
    """Build a layer x model x model distance cube without loading full checkpoints."""

    if not readers:
        raise ValueError("at least one reader is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    requested = _normalize_metrics(metrics)
    keys = readers[0].keys()
    _validate_reader_keys(readers, keys)
    model_ids = [reader.model_id for reader in readers]

    distances: dict[str, list[np.ndarray]] = {metric: [] for metric in requested}
    layer_audit: list[dict[str, Any]] = []

    for key in keys:
        infos = [reader.tensor_info(key) for reader in readers]
        shapes = {info.shape for info in infos}
        if len(shapes) != 1:
            raise ValueError(f"shape mismatch for {key}: {sorted(shapes)}")
        shape = next(iter(shapes))
        layer_audit.append(
            {
                "name": key,
                "shape": list(shape),
                "numel": infos[0].numel,
                "dtype_by_model": {reader.model_id: info.dtype for reader, info in zip(readers, infos)},
            }
        )

        vector_metrics = [metric for metric in requested if metric in VECTOR_METRICS]
        if vector_metrics:
            vector_mats = _compute_vector_metrics(
                readers,
                key,
                metrics=vector_metrics,
                chunk_size=chunk_size,
                eps=eps,
            )
            for metric, matrix in vector_mats.items():
                distances[metric].append(matrix)

        matrix_metrics = [metric for metric in requested if metric in MATRIX_METRICS]
        if matrix_metrics:
            matrix_mats = _compute_matrix_metrics(readers, key, metrics=matrix_metrics)
            for metric, matrix in matrix_mats.items():
                distances[metric].append(matrix)

    for reader in readers:
        reader.close()

    stacked = {
        metric: np.stack(mats, axis=0) if mats else np.empty((0, len(readers), len(readers)))
        for metric, mats in distances.items()
    }
    audit = {
        "representation": representation,
        "metrics": requested,
        "chunk_size": chunk_size,
        "eps": eps,
        "n_models": len(readers),
        "n_layers": len(keys),
        "model_ids": model_ids,
        "reader_type_by_model": {reader.model_id: type(reader).__name__ for reader in readers},
        "layers": layer_audit,
        "metric_execution": {
            metric: "chunk_streamed" if metric in VECTOR_METRICS else "tensor_at_a_time"
            for metric in requested
        },
    }
    return DistanceCube(distances=stacked, layer_names=keys, model_ids=model_ids, audit=audit)


def write_distance_cube(cube: DistanceCube, out_dir: str | Path) -> None:
    """Write a distance cube directory."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "distance_cube.npz", **cube.distances)
    (out / "layers.json").write_text(json.dumps(cube.layer_names, indent=2) + "\n")
    (out / "models.json").write_text(json.dumps(cube.model_ids, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(sorted(cube.distances), indent=2) + "\n")
    (out / "audit.json").write_text(json.dumps(cube.audit, indent=2, sort_keys=True) + "\n")


def _normalize_metrics(metrics: list[str]) -> list[str]:
    if not metrics:
        raise ValueError("at least one metric is required")
    normalized = []
    for metric in metrics:
        key = "cka" if metric == "linear_cka" else metric
        if key not in VECTOR_METRICS | MATRIX_METRICS:
            raise ValueError(f"unsupported streaming metric: {metric}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def _validate_reader_keys(readers: list[TensorReader], keys: list[str]) -> None:
    ref = set(keys)
    for reader in readers[1:]:
        other = set(reader.keys())
        if other != ref:
            missing = sorted(ref - other)[:5]
            extra = sorted(other - ref)[:5]
            raise ValueError(
                f"reader {reader.model_id} key mismatch: missing={missing}, extra={extra}"
            )


def _compute_vector_metrics(
    readers: list[TensorReader],
    key: str,
    *,
    metrics: list[str],
    chunk_size: int,
    eps: float,
) -> dict[str, np.ndarray]:
    n_models = len(readers)
    dot = np.zeros((n_models, n_models), dtype=np.float64)
    sums = np.zeros(n_models, dtype=np.float64)
    sums_sq = np.zeros(n_models, dtype=np.float64)
    l1 = np.zeros((n_models, n_models), dtype=np.float64) if "l1" in metrics else None
    threshold = (
        np.zeros((n_models, n_models), dtype=np.float64) if "threshold" in metrics else None
    )
    total = 0

    chunk_iters = [reader.iter_flat_chunks(key, chunk_size) for reader in readers]
    for chunks in zip(*chunk_iters, strict=True):
        x = np.vstack([np.asarray(chunk, dtype=np.float64).reshape(1, -1) for chunk in chunks])
        total += x.shape[1]
        dot += x @ x.T
        sums += x.sum(axis=1)
        sums_sq += np.sum(x * x, axis=1)

        if l1 is not None or threshold is not None:
            for i in range(n_models):
                diff = np.abs(x - x[i])
                if l1 is not None:
                    l1[i, :] += diff.sum(axis=1)
                if threshold is not None:
                    threshold[i, :] += (diff > eps).sum(axis=1)

    out: dict[str, np.ndarray] = {}
    if "cosine" in metrics:
        norms = np.sqrt(np.maximum(np.diag(dot), 0.0))
        denom = np.outer(norms, norms)
        out["cosine"] = _one_minus_similarity(dot, denom)
    if "l2" in metrics:
        norm2 = np.diag(dot)
        sqdist = np.maximum(norm2[:, None] + norm2[None, :] - 2.0 * dot, 0.0)
        out["l2"] = np.sqrt(sqdist)
    if "correlation" in metrics:
        centered_dot = dot - np.outer(sums, sums) / total
        centered_norm2 = np.maximum(sums_sq - (sums * sums) / total, 0.0)
        denom = np.outer(np.sqrt(centered_norm2), np.sqrt(centered_norm2))
        out["correlation"] = _one_minus_similarity(centered_dot, denom)
    if l1 is not None:
        out["l1"] = l1
    if threshold is not None:
        out["threshold"] = threshold

    for matrix in out.values():
        np.fill_diagonal(matrix, 0.0)
    return out


def _compute_matrix_metrics(
    readers: list[TensorReader],
    key: str,
    *,
    metrics: list[str],
) -> dict[str, np.ndarray]:
    tensors = [reader.read_tensor(key) for reader in readers]
    out: dict[str, np.ndarray] = {}
    if "cka" in metrics:
        out["cka"] = pairwise_distance_matrix(tensors, metric=linear_cka_distance)
    return out


def _one_minus_similarity(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    similarity = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator > 1e-12
    similarity[mask] = numerator[mask] / denominator[mask]
    out = 1.0 - similarity
    out[~mask] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
