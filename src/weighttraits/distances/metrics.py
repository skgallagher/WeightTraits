"""Small, dependency-light distance functions used by tests and smoke checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _as_vector(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional vector")
    return arr


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


def pairwise_distance_matrix(
    vectors: Sequence[ArrayLike],
    metric: Callable[[ArrayLike, ArrayLike], float] = cosine_distance,
) -> np.ndarray:
    """Compute a symmetric pairwise distance matrix for a small vector collection."""

    n = len(vectors)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = metric(vectors[i], vectors[j])
            out[i, j] = value
            out[j, i] = value
    return out

