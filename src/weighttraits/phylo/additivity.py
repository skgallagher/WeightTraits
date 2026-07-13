"""Label-free four-point additivity diagnostics for distance matrices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FourPointSubsetSummary:
    n: int
    mean_additivity: float
    median_additivity: float
    fraction_clean: float
    mean_relative_internal_edge: float
    mean_additivity_magnitude: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_additivity": self.mean_additivity,
            "median_additivity": self.median_additivity,
            "fraction_clean": self.fraction_clean,
            "mean_relative_internal_edge": self.mean_relative_internal_edge,
            "mean_additivity_magnitude": self.mean_additivity_magnitude,
        }


@dataclass(frozen=True)
class FourPointAdditivityResult:
    labels: tuple[str, ...]
    n_quartets_total: int
    n_quartets_scored: int
    clean_threshold: float
    max_quartets: int | None
    sample_seed: int
    all: FourPointSubsetSummary | None
    informative: FourPointSubsetSummary | None
    uninformative: FourPointSubsetSummary | None
    informative_quartet_split_accuracy: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "n_quartets_total": self.n_quartets_total,
            "n_quartets_scored": self.n_quartets_scored,
            "clean_threshold": self.clean_threshold,
            "max_quartets": self.max_quartets,
            "sample_seed": self.sample_seed,
            "all": None if self.all is None else self.all.as_dict(),
            "informative": None if self.informative is None else self.informative.as_dict(),
            "uninformative": (
                None if self.uninformative is None else self.uninformative.as_dict()
            ),
            "informative_quartet_split_accuracy": (
                self.informative_quartet_split_accuracy
            ),
        }


def four_point_additivity(
    labels: Sequence[str],
    distances: np.ndarray,
    *,
    truth_splits: Iterable[frozenset[str]] | None = None,
    clean_threshold: float = 0.5,
    max_quartets: int | None = 20_000,
    sample_seed: int = 0,
    eps: float = 1e-12,
) -> FourPointAdditivityResult:
    """Estimate four-point additivity over all or a seeded sample of quartets.

    For sorted pair sums ``s1 >= s2 >= s3``, the quartet score is
    ``(s2 - s3) / (s1 - s3 + eps)``. It is one for a resolved additive
    quartet and zero for a star. Truth splits are optional and are used only to
    report informative-quartet summaries and split accuracy.
    """

    label_tuple = tuple(str(label) for label in labels)
    matrix = _validated_distance_matrix(distances, label_tuple)
    if max_quartets is not None and max_quartets < 1:
        raise ValueError("max_quartets must be positive or None")
    if not 0.0 <= clean_threshold <= 1.0:
        raise ValueError("clean_threshold must be between zero and one")

    total = math.comb(len(label_tuple), 4) if len(label_tuple) >= 4 else 0
    quartet_indices = _sample_quartets(
        len(label_tuple),
        total=total,
        max_quartets=max_quartets,
        seed=sample_seed,
    )
    normalized_truth_splits = (
        tuple(frozenset(split) for split in truth_splits)
        if truth_splits is not None
        else None
    )
    scale_values = matrix[np.triu_indices(len(label_tuple), 1)]
    scale = float(np.median(scale_values)) + eps if scale_values.size else eps

    all_rows: list[tuple[float, float]] = []
    informative_rows: list[tuple[float, float]] = []
    uninformative_rows: list[tuple[float, float]] = []
    informative_correct = 0
    for i, j, k, ell in quartet_indices:
        sums = np.asarray(
            [
                matrix[i, j] + matrix[k, ell],
                matrix[i, k] + matrix[j, ell],
                matrix[i, ell] + matrix[j, k],
            ],
            dtype=float,
        )
        order = np.argsort(sums)
        smallest, middle, largest = sums[order[0]], sums[order[1]], sums[order[2]]
        additivity = float((middle - smallest) / (largest - smallest + eps))
        relative_edge = float((middle - smallest) / (2.0 * scale))
        row = (additivity, relative_edge)
        all_rows.append(row)
        if normalized_truth_splits is None:
            continue
        quartet = (
            label_tuple[i],
            label_tuple[j],
            label_tuple[k],
            label_tuple[ell],
        )
        truth_pairing = _truth_quartet_pairing(quartet, normalized_truth_splits)
        if truth_pairing is None:
            uninformative_rows.append(row)
        else:
            informative_rows.append(row)
            informative_correct += int(int(order[0]) == truth_pairing)

    informative_accuracy = (
        informative_correct / len(informative_rows) if informative_rows else None
    )
    return FourPointAdditivityResult(
        labels=label_tuple,
        n_quartets_total=total,
        n_quartets_scored=len(all_rows),
        clean_threshold=clean_threshold,
        max_quartets=max_quartets,
        sample_seed=sample_seed,
        all=_summarize(all_rows, clean_threshold),
        informative=_summarize(informative_rows, clean_threshold),
        uninformative=_summarize(uninformative_rows, clean_threshold),
        informative_quartet_split_accuracy=informative_accuracy,
    )


def _summarize(
    rows: list[tuple[float, float]], clean_threshold: float
) -> FourPointSubsetSummary | None:
    if not rows:
        return None
    additivity = np.asarray([row[0] for row in rows], dtype=float)
    relative_edges = np.asarray([row[1] for row in rows], dtype=float)
    return FourPointSubsetSummary(
        n=len(rows),
        mean_additivity=float(np.mean(additivity)),
        median_additivity=float(np.median(additivity)),
        fraction_clean=float(np.mean(additivity > clean_threshold)),
        mean_relative_internal_edge=float(np.mean(relative_edges)),
        mean_additivity_magnitude=float(np.mean(additivity * relative_edges)),
    )


def _truth_quartet_pairing(
    quartet: tuple[str, str, str, str],
    truth_splits: Sequence[frozenset[str]],
) -> int | None:
    a, b, c, d = quartet
    quartet_set = frozenset(quartet)
    pairings = (
        frozenset((a, b)),
        frozenset((a, c)),
        frozenset((a, d)),
    )
    for split in truth_splits:
        side = split & quartet_set
        if len(side) != 2:
            continue
        for index, pairing in enumerate(pairings):
            if side == pairing or quartet_set - side == pairing:
                return index
    return None


def _sample_quartets(
    n_labels: int,
    *,
    total: int,
    max_quartets: int | None,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    if max_quartets is None or total <= max_quartets:
        return list(combinations(range(n_labels), 4))
    rng = np.random.default_rng(seed)
    ranks = np.sort(rng.choice(total, max_quartets, replace=False))
    return [_unrank_combination(n_labels, 4, int(rank)) for rank in ranks]


def _unrank_combination(n: int, k: int, rank: int) -> tuple[int, ...]:
    values: list[int] = []
    start = 0
    for position in range(k):
        for value in range(start, n):
            count = math.comb(n - value - 1, k - position - 1)
            if rank < count:
                values.append(value)
                start = value + 1
                break
            rank -= count
    if len(values) != k:
        raise ValueError("quartet rank is out of range")
    return tuple(values)


def _validated_distance_matrix(matrix: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    out = np.asarray(matrix, dtype=float)
    n = len(labels)
    if len(set(labels)) != n:
        raise ValueError("distance matrix labels must be unique")
    if out.shape != (n, n):
        raise ValueError(f"distance matrix shape {out.shape} does not match {n} labels")
    if not np.all(np.isfinite(out)):
        raise ValueError("distance matrix contains non-finite values")
    if n and np.max(np.abs(out - out.T)) > 1e-9:
        raise ValueError("distance matrix must be symmetric")
    if n and np.max(np.abs(np.diag(out))) > 1e-9:
        raise ValueError("distance matrix diagonal must be zero")
    if out.size and np.min(out) < -1e-9:
        raise ValueError("distance matrix contains negative distances")
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out
