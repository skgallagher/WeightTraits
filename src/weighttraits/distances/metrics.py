"""Small, dependency-light distance functions used by tests and smoke checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np


ArrayLike = Sequence[float] | np.ndarray
MetricFunction = Callable[[ArrayLike, ArrayLike], float]


def _as_vector(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional vector")
    return arr


def _as_matrix(values: ArrayLike) -> np.ndarray:
    """Convert a tensor-like object to samples x features for matrix metrics."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        raise ValueError("expected an array with at least one dimension")
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def cosine_distance(a: ArrayLike, b: ArrayLike) -> float:
    va = _as_vector(a)
    vb = _as_vector(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(va, vb) / denom)


def l1_distance(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.sum(np.abs(_as_vector(a) - _as_vector(b))))


def l2_distance(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.linalg.norm(_as_vector(a) - _as_vector(b)))


def correlation_distance(a: ArrayLike, b: ArrayLike) -> float:
    va = _as_vector(a)
    vb = _as_vector(b)
    ca = va - va.mean()
    cb = vb - vb.mean()
    denom = np.linalg.norm(ca) * np.linalg.norm(cb)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(ca, cb) / denom)


def threshold_distance(a: ArrayLike, b: ArrayLike, eps: float = 1e-3) -> float:
    return float(np.sum(np.abs(_as_vector(a) - _as_vector(b)) > eps))


def linear_cka_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Centered linear CKA similarity for two tensors.

    Tensors are reshaped to ``samples x features`` using the first axis as the
    sample axis. For a 2-D weight matrix this treats rows as samples. For
    higher-rank tensors, all non-leading dimensions are flattened into features.
    """

    x = _as_matrix(a)
    y = _as_matrix(b)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA requires matching sample axis, got {x.shape[0]} and {y.shape[0]}")

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)

    cross = x_centered.T @ y_centered
    x_auto = x_centered.T @ x_centered
    y_auto = y_centered.T @ y_centered

    numerator = float(np.sum(cross * cross))
    denom = float(np.linalg.norm(x_auto, ord="fro") * np.linalg.norm(y_auto, ord="fro"))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(numerator / denom, 0.0, 1.0))


def linear_cka_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Distance form of centered linear CKA: ``1 - CKA``."""

    return 1.0 - linear_cka_similarity(a, b)


METRICS: dict[str, MetricFunction] = {
    "cosine": cosine_distance,
    "l1": l1_distance,
    "l2": l2_distance,
    "correlation": correlation_distance,
    "threshold": threshold_distance,
    "linear_cka": linear_cka_distance,
    "cka": linear_cka_distance,
}


def available_metrics() -> tuple[str, ...]:
    """Return supported metric names."""

    return tuple(sorted(METRICS))


def get_metric(name: str, **kwargs) -> MetricFunction:
    """Return a metric function by name.

    Keyword arguments are partially applied to metrics that accept them, such as
    ``threshold`` with ``eps``.
    """

    try:
        metric = METRICS[name]
    except KeyError as exc:
        raise ValueError(f"unknown metric {name!r}; available={available_metrics()}") from exc
    return partial(metric, **kwargs) if kwargs else metric


def pairwise_distance_matrix(
    vectors: Sequence[ArrayLike],
    metric: str | MetricFunction = cosine_distance,
    **metric_kwargs,
) -> np.ndarray:
    """Compute a symmetric pairwise distance matrix for a small vector collection."""

    metric_fn = get_metric(metric, **metric_kwargs) if isinstance(metric, str) else metric
    n = len(vectors)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = metric_fn(vectors[i], vectors[j])
            out[i, j] = value
            out[j, i] = value
    return out
