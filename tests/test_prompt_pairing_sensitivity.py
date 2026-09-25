from __future__ import annotations

import numpy as np
import pytest

from weighttraits.analysis.prompt_pairing import (
    PromptPairingRun,
    dersimonian_laird_pool,
    prompt_pairing_sensitivity,
    validate_prompt_pairing_runs,
)


def _synthetic_runs() -> list[PromptPairingRun]:
    weights = np.asarray([0.05, 0.11, 0.19, 0.27, 0.36, 0.48])
    prompt_offsets = np.asarray(
        [
            [0.00, 0.02, -0.01, 0.01],
            [0.01, -0.01, 0.00, 0.02],
            [-0.02, 0.01, 0.02, -0.01],
            [0.01, 0.00, -0.02, 0.01],
            [0.00, -0.02, 0.01, 0.00],
            [-0.01, 0.01, 0.00, -0.02],
        ]
    )
    runs = []
    for index, scale in enumerate([0.7, 0.8, 0.9, 1.0], start=1):
        similarity = 0.9 - scale * weights[:, None] + prompt_offsets
        runs.append(
            PromptPairingRun(
                tree_id=f"run_{index:03d}",
                weight_distance=weights + index * np.asarray([0.0, 0.01, -0.01, 0.0, 0.01, -0.01]),
                prompt_similarity=similarity,
            )
        )
    return runs


def test_exact_prompt_triplication_does_not_change_pooled_point_estimate() -> None:
    report = prompt_pairing_sensitivity(
        _synthetic_runs(),
        bootstrap_replicates=40,
        seed=7,
    )

    assert report["duplicate_each_prompt_3x_dl_r"] == pytest.approx(
        report["full_dl_r"],
        abs=1e-12,
    )
    assert report["duplicate_3x_abs_difference"] < 1e-12
    assert report["resampling_unit"] == (
        "whole run/tree; leaf pairs are never resampled independently"
    )


def test_prompt_and_whole_run_cluster_bootstrap_is_deterministic() -> None:
    first = prompt_pairing_sensitivity(
        _synthetic_runs(),
        bootstrap_replicates=50,
        seed=19,
    )
    second = prompt_pairing_sensitivity(
        _synthetic_runs(),
        bootstrap_replicates=50,
        seed=19,
    )

    assert first == second
    assert len(first["disjoint_halves_dl_r"]) == 2
    assert len(first["prompt_bootstrap_dl_quantiles_2.5_50_97.5"]) == 3
    assert len(first["hierarchical_run_prompt_cluster_dl_quantiles_2.5_50_97.5"]) == 3


def test_dl_pool_vectorizes_replicates() -> None:
    correlations = np.asarray(
        [
            [-0.20, -0.10],
            [-0.30, -0.15],
            [-0.25, -0.12],
            [-0.35, -0.18],
        ]
    )
    counts = np.asarray([6, 10, 15, 21])

    vectorized = dersimonian_laird_pool(correlations, counts)
    scalar = np.asarray(
        [dersimonian_laird_pool(correlations[:, index], counts)[0] for index in range(2)]
    )

    np.testing.assert_allclose(vectorized, scalar, atol=0.0, rtol=0.0)


def test_validation_rejects_pair_level_misalignment() -> None:
    runs = _synthetic_runs()
    runs[0] = PromptPairingRun(
        tree_id=runs[0].tree_id,
        weight_distance=runs[0].weight_distance[:-1],
        prompt_similarity=runs[0].prompt_similarity,
    )

    with pytest.raises(ValueError, match="leaf-pair rows are not aligned"):
        validate_prompt_pairing_runs(runs)
