"""Fractional-logit fits of clade estimation on the tree-likeness diagnostics (Figure 2).

Clade estimation ``p`` (the fraction of true clades an estimator recovers) is modelled with a
quasi-binomial GLM, ``logit E[p] = b0 + b1 x``, where ``x`` is either the tree-likeness score
``A`` or ``log10(max(m~, 0.03))`` for the fitted Atteson margin ``m~``. Odds ratios are
reported per ``+0.10`` in ``A`` and per tenfold increase in ``m~``.

Two fits appear in the paper:

* **estimator means** (Figure 2): one point per estimator, averaged over the 46 shared trees;
  intervals come from a paired bootstrap that resamples the 46 tree IDs jointly for every
  estimator and recomputes all means before refitting;
* **tree level** (the weights-only check, Section 5): one point per estimator-by-tree record.

Point estimates match R's ``glm.fit(..., family = quasibinomial())`` exactly. Bootstrap
intervals use NumPy's generator, so they differ slightly from the R script that produced the
paper's intervals (``R/figure2.R``, seed 20260805; ``R/odds_ratio_tree_level.R``, seed 20260923).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ATTESON_FLOOR = 0.03
A_CONTRAST = 0.10  # odds ratio per +0.10 in A
M_CONTRAST = 1.0  # odds ratio per tenfold increase in m~ (x is log10)


def fit_fractional_logit(
    x: Sequence[float], y: Sequence[float], *, max_iter: int = 100, tol: float = 1e-12
) -> dict[str, Any]:
    """Quasi-binomial logit fit by iteratively reweighted least squares.

    Constant prior weights (R's ``weights = rep(46, n)``) do not change the coefficients, so
    none are taken. Returns the coefficients and the deviance ratio explained.
    """

    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.shape != yv.shape or xv.ndim != 1 or len(xv) < 2:
        raise ValueError("x and y must be equal-length vectors with at least two points")
    if np.any((yv < 0) | (yv > 1)):
        raise ValueError("fractional outcomes must lie in [0, 1]")
    design = np.column_stack([np.ones_like(xv), xv])
    mu = (yv + 0.5) / 2.0  # R's binomial initialisation with unit weights
    eta = np.log(mu / (1 - mu))
    beta = np.zeros(2)
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-eta))
        variance = np.clip(mu * (1 - mu), 1e-300, None)
        z = eta + (yv - mu) / variance
        weights = variance
        wx = design * weights[:, None]
        new_beta = np.linalg.solve(design.T @ wx, wx.T @ z)
        eta = design @ new_beta
        if np.max(np.abs(new_beta - beta)) < tol * (np.max(np.abs(new_beta)) + tol):
            beta = new_beta
            break
        beta = new_beta
    mu = 1.0 / (1.0 + np.exp(-(design @ beta)))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "deviance_explained": 1.0 - _deviance(yv, mu) / _deviance(yv, np.full_like(yv, yv.mean())),
    }


def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(y > 0, y * np.log(y / mu), 0.0)
        b = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
    return float(2.0 * np.sum(a + b))


def atteson_x(margin: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(margin, ATTESON_FLOOR))


def load_figure2_records(path: str | Path) -> dict[str, Any]:
    """Estimator-by-tree matrices from ``merged_weight_figure2_records.csv``."""

    rows = list(csv.DictReader(Path(path).open()))
    estimators = sorted({row["estimator_id"] for row in rows})
    trees = sorted({row["tree_id"] for row in rows})
    index_e = {e: i for i, e in enumerate(estimators)}
    index_t = {t: i for i, t in enumerate(trees)}
    shape = (len(estimators), len(trees))
    matrices = {key: np.full(shape, np.nan) for key in ("recovery", "atteson", "additivity")}
    level = {}
    for row in rows:
        i, j = index_e[row["estimator_id"]], index_t[row["tree_id"]]
        if not np.isnan(matrices["recovery"][i, j]):
            raise ValueError(f"duplicate record for {row['estimator_id']} / {row['tree_id']}")
        matrices["recovery"][i, j] = float(row["clade_recovery"])
        matrices["atteson"][i, j] = float(row["atteson_bottleneck_margin"])
        matrices["additivity"][i, j] = float(row["four_point_mean_additivity"])
        level[row["estimator_id"]] = row.get("level", "")
    if any(np.isnan(matrix).any() for matrix in matrices.values()):
        raise ValueError("every estimator needs exactly one record per tree")
    return {"estimators": estimators, "trees": trees, "levels": level, **matrices}


def _summary_row(name, contrast, scale, fit, boot_slopes) -> dict[str, Any]:
    ratios = np.exp(scale * np.asarray(boot_slopes))
    return {
        "diagnostic": name,
        "contrast": contrast,
        "slope": fit["slope"],
        "odds_ratio": float(np.exp(scale * fit["slope"])),
        "odds_ratio_lower": float(np.quantile(ratios, 0.025)),
        "odds_ratio_upper": float(np.quantile(ratios, 0.975)),
        "deviance_explained": fit["deviance_explained"],
    }


def estimator_mean_odds_ratios(
    records: dict[str, Any], *, reps: int = 2000, seed: int = 20260805
) -> list[dict[str, Any]]:
    """Figure 2 fits on estimator means with a paired-tree bootstrap."""

    recovery, atteson, additivity = records["recovery"], records["atteson"], records["additivity"]
    n_trees = recovery.shape[1]
    y = recovery.mean(axis=1)
    fits = {
        "atteson": fit_fractional_logit(atteson_x(atteson.mean(axis=1)), y),
        "additivity": fit_fractional_logit(additivity.mean(axis=1), y),
    }
    rng = np.random.default_rng(seed)
    slopes = {"atteson": [], "additivity": []}
    for _ in range(reps):
        sample = rng.integers(0, n_trees, n_trees)
        yb = recovery[:, sample].mean(axis=1)
        slopes["atteson"].append(
            fit_fractional_logit(atteson_x(atteson[:, sample].mean(axis=1)), yb)["slope"]
        )
        slopes["additivity"].append(
            fit_fractional_logit(additivity[:, sample].mean(axis=1), yb)["slope"]
        )
    return [
        _summary_row(
            "Fitted Atteson diagnostic",
            "10-fold increase in mean Atteson diagnostic",
            M_CONTRAST,
            fits["atteson"],
            slopes["atteson"],
        ),
        _summary_row(
            "Four-point additivity A",
            "0.10 increase in mean additivity",
            A_CONTRAST,
            fits["additivity"],
            slopes["additivity"],
        ),
    ]


def tree_level_odds_ratios(
    records: dict[str, Any],
    *,
    keep: Callable[[str], bool] = lambda level: True,
    reps: int = 2000,
    seed: int = 20260923,
) -> list[dict[str, Any]]:
    """Pooled estimator-by-tree fits, optionally restricted by estimator level.

    ``keep=lambda level: level != "phylolm"`` gives the paper's weights-only check.
    """

    rows = [i for i, e in enumerate(records["estimators"]) if keep(records["levels"][e])]
    if not rows:
        raise ValueError("no estimators selected")
    recovery = records["recovery"][rows]
    atteson = records["atteson"][rows]
    additivity = records["additivity"][rows]
    n_trees = recovery.shape[1]
    fits = {
        "atteson": fit_fractional_logit(atteson_x(atteson).ravel(), recovery.ravel()),
        "additivity": fit_fractional_logit(additivity.ravel(), recovery.ravel()),
    }
    rng = np.random.default_rng(seed)
    slopes = {"atteson": [], "additivity": []}
    for _ in range(reps):
        sample = rng.integers(0, n_trees, n_trees)
        yb = recovery[:, sample].ravel()
        slopes["atteson"].append(
            fit_fractional_logit(atteson_x(atteson[:, sample]).ravel(), yb)["slope"]
        )
        slopes["additivity"].append(fit_fractional_logit(additivity[:, sample].ravel(), yb)["slope"])
    out = [
        _summary_row(
            "Fitted Atteson diagnostic",
            "10-fold increase in Atteson diagnostic (tree level)",
            M_CONTRAST,
            fits["atteson"],
            slopes["atteson"],
        ),
        _summary_row(
            "Four-point additivity A",
            "0.10 increase in additivity (tree level)",
            A_CONTRAST,
            fits["additivity"],
            slopes["additivity"],
        ),
    ]
    for row in out:
        row["n_estimators"] = len(rows)
        row["n_records"] = int(recovery.size)
    return out
