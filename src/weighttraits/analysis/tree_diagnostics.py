"""Shared per-tree distance diagnostics for analysis engines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from weighttraits.phylo.additivity import four_point_additivity
from weighttraits.phylo.atteson import atteson_margin


TREE_DIAGNOSTICS = ("four_point_additivity", "atteson_margin")


def write_tree_diagnostics(
    out_dir: str | Path,
    *,
    metric: str,
    labels: Sequence[str],
    distances: np.ndarray,
    truth_newick: str,
    truth_splits: Iterable[frozenset[str]],
) -> dict[str, Any]:
    """Compute and persist label-free additivity and oracle Atteson diagnostics."""

    out = Path(out_dir)
    additivity = four_point_additivity(
        labels,
        distances,
        truth_splits=truth_splits,
    )
    atteson = atteson_margin(labels, distances, truth_newick)
    additivity_path = out / f"four_point_additivity_{metric}.json"
    atteson_path = out / f"atteson_margin_{metric}.json"
    additivity_path.write_text(json.dumps(additivity.as_dict(), indent=2, sort_keys=True) + "\n")
    atteson_path.write_text(json.dumps(atteson.as_dict(), indent=2, sort_keys=True) + "\n")

    all_quartets = additivity.all
    informative = additivity.informative
    return {
        "four_point_additivity": str(additivity_path),
        "four_point_mean_additivity": (
            None if all_quartets is None else all_quartets.mean_additivity
        ),
        "four_point_fraction_clean": (
            None if all_quartets is None else all_quartets.fraction_clean
        ),
        "four_point_mean_additivity_magnitude": (
            None if all_quartets is None else all_quartets.mean_additivity_magnitude
        ),
        "four_point_informative_mean_additivity": (
            None if informative is None else informative.mean_additivity
        ),
        "four_point_informative_split_accuracy": (
            additivity.informative_quartet_split_accuracy
        ),
        "atteson_margin": str(atteson_path),
        "atteson_error_linf": atteson.error_linf,
        "atteson_bottleneck_margin": atteson.bottleneck_margin,
        "atteson_internal_bottleneck_margin": atteson.internal_bottleneck_margin,
        "atteson_theorem_certified": atteson.theorem_certified,
    }
