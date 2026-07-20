import numpy as np

from weighttraits.phylo.additivity import four_point_additivity


LABELS = ("a", "b", "c", "d")


def test_exact_resolved_quartet_has_unit_additivity_and_correct_split():
    distances = np.asarray(
        [
            [0, 2, 4, 4],
            [2, 0, 4, 4],
            [4, 4, 0, 2],
            [4, 4, 2, 0],
        ],
        dtype=float,
    )

    result = four_point_additivity(
        LABELS,
        distances,
        truth_splits={frozenset({"a", "b"})},
    )

    assert result.all is not None
    assert result.informative is not None
    assert result.all.mean_additivity > 0.999999
    assert result.informative_quartet_split_accuracy == 1.0
    assert result.uninformative is None


def test_star_quartet_has_zero_additivity():
    distances = np.ones((4, 4), dtype=float)
    np.fill_diagonal(distances, 0.0)

    result = four_point_additivity(LABELS, distances)

    assert result.all is not None
    assert result.all.mean_additivity == 0.0
    assert result.all.fraction_clean == 0.0


def test_seeded_quartet_subsample_is_deterministic():
    labels = tuple(f"n{index}" for index in range(12))
    coordinates = np.arange(12, dtype=float)
    distances = np.abs(coordinates[:, None] - coordinates[None, :])

    first = four_point_additivity(labels, distances, max_quartets=20, sample_seed=7)
    second = four_point_additivity(labels, distances, max_quartets=20, sample_seed=7)

    assert first.n_quartets_total == 495
    assert first.n_quartets_scored == 20
    assert first.as_dict() == second.as_dict()
