"""Behavioral distance matrices with explicit missing-observation audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class BehaviorDistanceResult:
    distances: np.ndarray
    observation_counts: np.ndarray
    model_ids: tuple[str, ...]
    audit: dict[str, Any]


def paired_cosine_distances(
    embeddings: np.ndarray,
    *,
    model_ids: Sequence[str],
    observation_ids: Sequence[str],
    valid_mask: np.ndarray | None = None,
) -> BehaviorDistanceResult:
    """Average cosine distance between aligned output embeddings for each model pair.

    ``embeddings`` has shape ``model x observation x dimension``. An observation is normally a
    prompt/repeat pair. Pairwise means use only observations valid for both models and record the
    exact denominator in ``observation_counts``.
    """

    values = np.asarray(embeddings, dtype=np.float64)
    models = tuple(str(item) for item in model_ids)
    observations = tuple(str(item) for item in observation_ids)
    if values.ndim != 3:
        raise ValueError("behavior embeddings must have shape model x observation x dimension")
    n_models, n_observations, dimension = values.shape
    if len(models) != n_models or len(set(models)) != n_models:
        raise ValueError("model_ids must be unique and match the embedding model axis")
    if len(observations) != n_observations or len(set(observations)) != n_observations:
        raise ValueError("observation_ids must be unique and match the observation axis")
    if dimension <= 0:
        raise ValueError("behavior embeddings must have a non-empty feature dimension")

    finite = np.all(np.isfinite(values), axis=2)
    norms = np.linalg.norm(values, axis=2)
    usable = finite & (norms > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != (n_models, n_observations):
            raise ValueError("valid_mask must have shape model x observation")
        usable &= mask

    normalized = np.zeros_like(values)
    normalized[usable] = values[usable] / norms[usable, None]
    distances = np.zeros((n_models, n_models), dtype=np.float64)
    counts = np.zeros((n_models, n_models), dtype=np.int64)
    missing_pairs: list[dict[str, Any]] = []
    for left in range(n_models):
        counts[left, left] = int(np.sum(usable[left]))
        for right in range(left + 1, n_models):
            shared = usable[left] & usable[right]
            count = int(np.sum(shared))
            counts[left, right] = counts[right, left] = count
            if count == 0:
                distances[left, right] = distances[right, left] = np.nan
                missing_pairs.append({"model_i": models[left], "model_j": models[right]})
                continue
            similarities = np.sum(normalized[left, shared] * normalized[right, shared], axis=1)
            distance = float(np.mean(1.0 - np.clip(similarities, -1.0, 1.0)))
            distances[left, right] = distances[right, left] = distance

    return BehaviorDistanceResult(
        distances=distances,
        observation_counts=counts,
        model_ids=models,
        audit={
            "schema_version": 1,
            "representation": "paired_output_embedding_cosine",
            "n_models": n_models,
            "n_observations": n_observations,
            "embedding_dimension": dimension,
            "usable_observations_by_model": {
                model: int(np.sum(usable[index])) for index, model in enumerate(models)
            },
            "n_pairs_without_shared_observations": len(missing_pairs),
            "pairs_without_shared_observations": missing_pairs,
        },
    )
