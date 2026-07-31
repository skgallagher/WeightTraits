"""Auditable random-effects pooling for within-run behavioral correlations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CorrelationEffect:
    run_id: str
    r: float
    n_pairs: int


@dataclass(frozen=True)
class RandomEffectsCorrelation:
    r: float
    ci_low: float
    ci_high: float
    p_value: float
    tau_squared: float
    q: float
    i_squared_pct: float
    n_runs: int
    total_pairs: int
    effects: tuple[CorrelationEffect, ...]
    method: str = "DerSimonian-Laird on Fisher-z correlations"
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["effects"] = [asdict(effect) for effect in self.effects]
        return out


def dersimonian_laird_correlations(
    effects: Sequence[CorrelationEffect],
    *,
    confidence: float = 0.95,
) -> RandomEffectsCorrelation:
    """Pool independent per-run Pearson correlations with DL random effects."""

    rows = tuple(effects)
    if not rows:
        raise ValueError("at least one correlation effect is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if len({row.run_id for row in rows}) != len(rows):
        raise ValueError("correlation effect run IDs must be unique")
    for row in rows:
        if not -1 < row.r < 1:
            raise ValueError("correlations must lie strictly between -1 and 1")
        if row.n_pairs <= 3:
            raise ValueError("each correlation effect requires more than three pairs")

    z = np.asarray([np.arctanh(row.r) for row in rows], dtype=np.float64)
    variance = np.asarray([1.0 / (row.n_pairs - 3) for row in rows], dtype=np.float64)
    fixed_weights = 1.0 / variance
    fixed_mean = float(np.sum(fixed_weights * z) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * np.square(z - fixed_mean)))
    degrees = len(rows) - 1
    c = float(np.sum(fixed_weights) - np.sum(np.square(fixed_weights)) / np.sum(fixed_weights))
    tau_squared = max(0.0, (q - degrees) / c) if c > 0 else 0.0
    random_weights = 1.0 / (variance + tau_squared)
    pooled_z = float(np.sum(random_weights * z) / np.sum(random_weights))
    standard_error = math.sqrt(1.0 / float(np.sum(random_weights)))
    alpha = 1.0 - confidence
    critical = _normal_quantile(1.0 - alpha / 2.0)
    lower_z = pooled_z - critical * standard_error
    upper_z = pooled_z + critical * standard_error
    z_statistic = pooled_z / standard_error
    p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0))
    i_squared = max(0.0, (q - degrees) / q) * 100.0 if q > 0 else 0.0
    return RandomEffectsCorrelation(
        r=float(np.tanh(pooled_z)),
        ci_low=float(np.tanh(lower_z)),
        ci_high=float(np.tanh(upper_z)),
        p_value=p_value,
        tau_squared=tau_squared,
        q=q,
        i_squared_pct=i_squared,
        n_runs=len(rows),
        total_pairs=sum(row.n_pairs for row in rows),
        effects=rows,
    )


def _normal_quantile(probability: float) -> float:
    # Acklam's rational approximation; sufficient for audit confidence intervals without SciPy.
    if not 0 < probability < 1:
        raise ValueError("normal quantile probability must be between zero and one")
    a = (-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269,
         -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197,
         -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742)
    low = 0.02425
    high = 1.0 - low
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        )
    if probability > high:
        return -_normal_quantile(1.0 - probability)
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r) + 1.0
    )
