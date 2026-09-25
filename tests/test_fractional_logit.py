import csv
from pathlib import Path

import numpy as np
import pytest

from weighttraits.stats.fractional_logit import (
    estimator_mean_odds_ratios,
    fit_fractional_logit,
    load_figure2_records,
)

ROOT = Path(__file__).parents[1]
RECORDS = ROOT / "reports" / "paper" / "merged_weight_figure2_records.csv"
R_BOOTSTRAP = ROOT / "reports" / "paper" / "merged_weight_figure2_bootstrap.csv"


def test_noiseless_logistic_curve_is_recovered():
    x = np.linspace(-2, 2, 25)
    y = 1 / (1 + np.exp(-(0.3 + 1.7 * x)))
    fit = fit_fractional_logit(x, y)
    assert fit["intercept"] == pytest.approx(0.3, abs=1e-9)
    assert fit["slope"] == pytest.approx(1.7, abs=1e-9)
    assert fit["deviance_explained"] == pytest.approx(1.0)


def test_fractional_outcomes_at_the_bounds_are_allowed():
    fit = fit_fractional_logit([0.0, 1.0, 2.0, 3.0], [0.0, 0.2, 0.9, 1.0])
    assert fit["slope"] > 0


def test_rejects_outcomes_outside_the_unit_interval():
    with pytest.raises(ValueError):
        fit_fractional_logit([0.0, 1.0], [0.5, 1.5])


@pytest.mark.skipif(
    not (RECORDS.is_file() and R_BOOTSTRAP.is_file()), reason="Figure 2 artifacts not present"
)
def test_point_estimates_match_the_r_figure2_fit():
    rows = {row["diagnostic"]: row for row in csv.DictReader(R_BOOTSTRAP.open())}
    ours = {
        row["diagnostic"]: row
        for row in estimator_mean_odds_ratios(load_figure2_records(RECORDS), reps=20)
    }
    # R's glm.fit stops at a relative deviance change of 1e-8, so agree to ~1e-7.
    for diagnostic, reference in rows.items():
        assert ours[diagnostic]["slope"] == pytest.approx(float(reference["slope"]), rel=1e-7)
        assert ours[diagnostic]["odds_ratio"] == pytest.approx(
            float(reference["odds_ratio"]), rel=1e-7
        )
        assert ours[diagnostic]["deviance_explained"] == pytest.approx(
            float(reference["null_deviance_explained"]), rel=1e-7
        )
