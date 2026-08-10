"""Prompt-resampling diagnostics for paired semantic behavior endpoints.

The primary paired endpoint averages output-embedding similarity within prompt
before correlating it with weight distance.  These helpers quantify sensitivity
to the prompt panel without pretending that leaf pairs are independent: the
hierarchical bootstrap samples entire runs/trees and never individual pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PromptPairingRun:
    """One tree's weight distances and prompt-level semantic similarities.

    ``weight_distance`` has shape ``(n_leaf_pairs,)``. ``prompt_similarity``
    has shape ``(n_leaf_pairs, n_prompts)`` in the same pair order.
    """

    tree_id: str
    weight_distance: np.ndarray
    prompt_similarity: np.ndarray


def flatten_prompt_outputs(
    predictions: Mapping[str, Sequence[str]],
    leaves: Sequence[str],
) -> list[str]:
    """Return exact leaf-major, prompt-minor output text without normalization."""

    flat: list[str] = []
    for leaf in leaves:
        if leaf not in predictions:
            raise ValueError(f"predictions are missing leaf {leaf!r}")
        outputs = predictions[leaf]
        if not all(isinstance(text, str) for text in outputs):
            raise ValueError(f"{leaf}: every prompt output must be a string")
        flat.extend(outputs)
    return flat


def validate_saved_matrix_rebuild_error(
    errors: Sequence[float],
    *,
    max_abs_error: float,
) -> float:
    """Return the maximum rebuild error or fail closed above the tolerance."""

    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("saved-matrix rebuild errors must be one non-empty finite vector")
    if not np.isfinite(max_abs_error) or max_abs_error < 0:
        raise ValueError("saved-matrix rebuild tolerance must be finite and non-negative")
    observed = float(np.max(values))
    if observed > max_abs_error:
        raise RuntimeError(
            "saved dist_semantic_paired.npy does not match the exact prompt-paired rebuild: "
            f"max absolute error {observed:.12g} exceeds {max_abs_error:.12g}"
        )
    return observed


def validate_prompt_pairing_runs(runs: Sequence[PromptPairingRun]) -> tuple[int, np.ndarray]:
    """Validate aligned run inputs and return prompt count and pair counts."""
    if len(runs) < 3:
        raise ValueError("prompt-pairing sensitivity requires at least three runs")

    prompt_counts: set[int] = set()
    pair_counts = []
    tree_ids: set[str] = set()
    for run in runs:
        if not run.tree_id or run.tree_id in tree_ids:
            raise ValueError(f"tree IDs must be non-empty and unique: {run.tree_id!r}")
        tree_ids.add(run.tree_id)
        weights = np.asarray(run.weight_distance)
        prompt_similarity = np.asarray(run.prompt_similarity)
        if weights.ndim != 1:
            raise ValueError(f"{run.tree_id}: weight_distance must be one-dimensional")
        if prompt_similarity.ndim != 2:
            raise ValueError(f"{run.tree_id}: prompt_similarity must be two-dimensional")
        if len(weights) != prompt_similarity.shape[0]:
            raise ValueError(f"{run.tree_id}: leaf-pair rows are not aligned")
        if len(weights) < 4:
            raise ValueError(f"{run.tree_id}: at least four leaf pairs are required")
        if not np.isfinite(weights).all() or not np.isfinite(prompt_similarity).all():
            raise ValueError(f"{run.tree_id}: distances and similarities must be finite")
        if float(np.std(weights)) == 0.0:
            raise ValueError(f"{run.tree_id}: weight distances are constant")
        prompt_counts.add(prompt_similarity.shape[1])
        pair_counts.append(len(weights))

    if len(prompt_counts) != 1:
        raise ValueError(f"runs must share one prompt count: {sorted(prompt_counts)}")
    n_prompts = prompt_counts.pop()
    if n_prompts < 2:
        raise ValueError("at least two prompts are required")
    return n_prompts, np.asarray(pair_counts, dtype=np.int64)


def pearson_columns(x: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Pearson correlation between one vector and every column of a matrix."""
    x = np.asarray(x, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if x.ndim != 1 or ys.ndim != 2 or len(x) != ys.shape[0]:
        raise ValueError("x must align with the rows of a two-dimensional ys")
    x_centered = x - np.mean(x)
    ys_centered = ys - np.mean(ys, axis=0, keepdims=True)
    numerator = np.sum(x_centered[:, None] * ys_centered, axis=0)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered) * np.sum(ys_centered * ys_centered, axis=0)
    )
    if np.any(denominator == 0):
        raise ValueError("correlation is undefined for a constant vector")
    return np.clip(numerator / denominator, -0.999, 0.999)


def dersimonian_laird_pool(
    run_correlations: np.ndarray,
    pair_counts: np.ndarray,
) -> np.ndarray:
    """Pool per-run correlations with the legacy Fisher-z DL estimator.

    ``run_correlations`` may be ``(n_runs,)`` or ``(n_runs, n_replicates)``.
    The returned array always has shape ``(n_replicates,)``.
    """
    correlations = np.asarray(run_correlations, dtype=np.float64)
    if correlations.ndim == 1:
        correlations = correlations[:, None]
    counts = np.asarray(pair_counts, dtype=np.float64)
    if correlations.ndim != 2 or counts.ndim != 1 or len(counts) != correlations.shape[0]:
        raise ValueError("one pair count is required for each run")
    if len(counts) < 3 or np.any(counts <= 3):
        raise ValueError(
            "DL pooling requires at least three runs and more than three pairs per run"
        )
    if not np.isfinite(correlations).all():
        raise ValueError("run correlations must be finite")

    z = np.arctanh(np.clip(correlations, -0.999, 0.999))
    variances = 1.0 / (counts - 3.0)
    fixed_weights = 1.0 / variances
    fixed_z = np.sum(fixed_weights[:, None] * z, axis=0) / np.sum(fixed_weights)
    q = np.sum(fixed_weights[:, None] * (z - fixed_z[None, :]) ** 2, axis=0)
    c = np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    tau_squared = np.maximum(0.0, (q - (len(counts) - 1.0)) / c)
    random_weights = 1.0 / (variances[:, None] + tau_squared[None, :])
    random_z = np.sum(random_weights * z, axis=0) / np.sum(random_weights, axis=0)
    return np.tanh(random_z)


def prompt_pairing_sensitivity(
    runs: Sequence[PromptPairingRun],
    *,
    bootstrap_replicates: int = 2_000,
    seed: int = 20_260_810,
) -> dict[str, object]:
    """Compute full, split-half, and prompt/run-cluster bootstrap diagnostics.

    Prompt indices are sampled jointly across runs because the probe panel is a
    shared blocking factor.  The hierarchical interval additionally samples
    whole runs/trees.  Individual leaf pairs are never resampled because pairs
    sharing a model/node are dependent.
    """
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    n_prompts, pair_counts = validate_prompt_pairing_runs(runs)

    full_correlations = np.asarray(
        [
            pearson_columns(
                run.weight_distance,
                np.mean(run.prompt_similarity, axis=1, keepdims=True),
            )[0]
            for run in runs
        ]
    )
    full_dl = float(dersimonian_laird_pool(full_correlations, pair_counts)[0])

    split = n_prompts // 2
    prompt_halves = (np.arange(split), np.arange(split, n_prompts))
    half_correlations = np.empty((len(runs), 2), dtype=np.float64)
    for run_index, run in enumerate(runs):
        for half_index, prompt_indices in enumerate(prompt_halves):
            half_correlations[run_index, half_index] = pearson_columns(
                run.weight_distance,
                np.mean(run.prompt_similarity[:, prompt_indices], axis=1, keepdims=True),
            )[0]
    half_dl = dersimonian_laird_pool(half_correlations, pair_counts)

    rng = np.random.default_rng(seed)
    prompt_draws = rng.integers(
        0,
        n_prompts,
        size=(bootstrap_replicates, n_prompts),
    )
    prompt_weights = np.zeros((n_prompts, bootstrap_replicates), dtype=np.float64)
    replicate_indices = np.repeat(np.arange(bootstrap_replicates), n_prompts)
    np.add.at(
        prompt_weights,
        (prompt_draws.ravel(), replicate_indices),
        1.0 / n_prompts,
    )

    bootstrap_correlations = np.empty(
        (len(runs), bootstrap_replicates),
        dtype=np.float64,
    )
    for run_index, run in enumerate(runs):
        bootstrap_correlations[run_index] = pearson_columns(
            run.weight_distance,
            run.prompt_similarity @ prompt_weights,
        )
    prompt_bootstrap_dl = dersimonian_laird_pool(bootstrap_correlations, pair_counts)

    cluster_bootstrap_dl = np.empty(bootstrap_replicates, dtype=np.float64)
    cluster_bootstrap_equal_weight = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        run_indices = rng.integers(0, len(runs), size=len(runs))
        sampled_correlations = bootstrap_correlations[run_indices, replicate]
        cluster_bootstrap_dl[replicate] = dersimonian_laird_pool(
            sampled_correlations,
            pair_counts[run_indices],
        )[0]
        cluster_bootstrap_equal_weight[replicate] = np.tanh(
            np.mean(np.arctanh(sampled_correlations))
        )

    triplicated_correlations = np.asarray(
        [
            pearson_columns(
                run.weight_distance,
                np.mean(np.repeat(run.prompt_similarity, 3, axis=1), axis=1, keepdims=True),
            )[0]
            for run in runs
        ]
    )
    triplicated_dl = float(dersimonian_laird_pool(triplicated_correlations, pair_counts)[0])

    def quantiles(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]

    return {
        "n_runs": len(runs),
        "n_prompts": n_prompts,
        "pairs_per_run_min": int(np.min(pair_counts)),
        "pairs_per_run_max": int(np.max(pair_counts)),
        "full_dl_r": full_dl,
        "duplicate_each_prompt_3x_dl_r": triplicated_dl,
        "duplicate_3x_abs_difference": abs(full_dl - triplicated_dl),
        "disjoint_halves_dl_r": [float(value) for value in half_dl],
        "prompt_bootstrap_dl_quantiles_2.5_50_97.5": quantiles(prompt_bootstrap_dl),
        "prompt_bootstrap_probability_negative": float(np.mean(prompt_bootstrap_dl < 0)),
        "hierarchical_run_prompt_cluster_dl_quantiles_2.5_50_97.5": quantiles(cluster_bootstrap_dl),
        "hierarchical_run_prompt_cluster_equal_weight_quantiles_2.5_50_97.5": quantiles(
            cluster_bootstrap_equal_weight
        ),
        "hierarchical_probability_negative": float(np.mean(cluster_bootstrap_dl < 0)),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "resampling_unit": "whole run/tree; leaf pairs are never resampled independently",
        "run_r": {
            run.tree_id: float(value) for run, value in zip(runs, full_correlations, strict=True)
        },
    }
