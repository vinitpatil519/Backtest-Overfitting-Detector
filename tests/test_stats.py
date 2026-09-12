"""Closed-form checks on the statistics layer."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from engine.stats import (
    EULER_GAMMA,
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    drawdown_profile,
    effective_number_of_trials,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    psr_denominator,
    sharpe_ratio,
    sharpe_ratio_matrix,
    sharpe_standard_error,
)


def test_moments_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    m = moments(x)
    assert m["n"] == 5000
    assert m["mean"] == pytest.approx(float(x.mean()))
    assert m["std"] == pytest.approx(float(x.std(ddof=1)))
    assert m["skew"] == pytest.approx(float(sps.skew(x, bias=False)))
    # Non-excess convention: a Gaussian sample sits near 3.
    assert m["kurtosis"] == pytest.approx(3.0, abs=0.25)


def test_sharpe_matches_definition():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, 2000)
    expected = float((x.mean()) / x.std(ddof=1))
    assert sharpe_ratio(x) == pytest.approx(expected)
    assert sharpe_ratio_matrix(x[:, None])[0] == pytest.approx(expected)


def test_annualisation_round_trips():
    assert deannualize_sharpe(annualize_sharpe(0.05, 252), 252) == pytest.approx(0.05)
    assert annualize_sharpe(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))


def test_psr_denominator_reduces_to_lo_2002():
    # Gaussian returns: skew 0, kurtosis 3  ->  1 + SR^2 / 2
    assert psr_denominator(0.4, 0.0, 3.0) == pytest.approx(1.0 + 0.4**2 / 2.0)


def test_psr_is_one_half_at_the_benchmark():
    assert probabilistic_sharpe_ratio(0.1, 500, 0.0, 3.0, 0.1) == pytest.approx(0.5)


def test_psr_increases_with_track_record():
    short = probabilistic_sharpe_ratio(0.05, 100, 0.0, 3.0, 0.0)
    long = probabilistic_sharpe_ratio(0.05, 2000, 0.0, 3.0, 0.0)
    assert long > short


def test_psr_penalises_negative_skew_and_fat_tails():
    base = probabilistic_sharpe_ratio(0.08, 500, 0.0, 3.0)
    skewed = probabilistic_sharpe_ratio(0.08, 500, -1.5, 3.0)
    fat = probabilistic_sharpe_ratio(0.08, 500, 0.0, 9.0)
    assert skewed < base
    assert fat < base


def test_sharpe_standard_error_matches_gaussian_formula():
    sr, n = 0.1, 1001
    expected = math.sqrt((1.0 + sr * sr / 2.0) / (n - 1))
    assert sharpe_standard_error(sr, n, 0.0, 3.0) == pytest.approx(expected)


def test_expected_max_sharpe_matches_the_closed_form():
    n, var = 100.0, 0.04
    z1 = sps.norm.ppf(1.0 - 1.0 / n)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n * math.e))
    expected = math.sqrt(var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    assert expected_max_sharpe(n, var) == pytest.approx(expected)


def test_expected_max_sharpe_is_monotone_in_trials():
    values = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000, 10000)]
    assert values == sorted(values)


def test_expected_max_sharpe_degenerate_cases():
    assert expected_max_sharpe(1, 0.04) == 0.0
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_expected_max_sharpe_tracks_monte_carlo():
    """The analytic benchmark should match a simulated maximum within noise."""
    rng = np.random.default_rng(5)
    n, sigma = 200, 0.05
    draws = rng.normal(0.0, sigma, size=(40000, n)).max(axis=1)
    assert expected_max_sharpe(n, sigma**2) == pytest.approx(float(draws.mean()), abs=0.004)


def test_deflation_lowers_the_probability_as_trials_grow():
    args = dict(sr=0.12, n_obs=1000, skew=0.0, kurtosis=3.0, variance_of_sharpes=0.0025)
    few = deflated_sharpe_ratio(n_trials=5, **args)
    many = deflated_sharpe_ratio(n_trials=5000, **args)
    assert many["sr_star"] > few["sr_star"]
    assert many["dsr"] < few["dsr"]


def test_min_trl_is_infinite_below_the_benchmark():
    assert math.isinf(min_track_record_length(0.01, 0.0, 3.0, 0.05))


def test_min_trl_shrinks_as_the_edge_grows():
    weak = min_track_record_length(0.02, 0.0, 3.0, 0.0)
    strong = min_track_record_length(0.20, 0.0, 3.0, 0.0)
    assert strong < weak


def test_min_trl_makes_psr_hit_the_confidence_level():
    sr, skew, kurt, conf = 0.06, -0.3, 4.5, 0.95
    n = min_track_record_length(sr, skew, kurt, 0.0, conf)
    assert probabilistic_sharpe_ratio(sr, int(round(n)), skew, kurt, 0.0) == pytest.approx(
        conf, abs=1e-3
    )


def test_effective_trials_collapses_for_identical_columns():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(500)
    duplicated = np.column_stack([base + 1e-9 * rng.standard_normal(500) for _ in range(20)])
    out = effective_number_of_trials(duplicated)
    assert out["mean_corr"] > 0.99
    assert out["n_eff"] < 2.0


def test_effective_trials_is_full_for_independent_columns():
    rng = np.random.default_rng(3)
    independent = rng.standard_normal((4000, 25))
    out = effective_number_of_trials(independent)
    assert out["n_eff"] > 20.0
    assert out["n_eff"] <= 25.0


def test_drawdown_profile_on_a_known_path():
    returns = np.array([0.10, -0.50, 0.20])
    dd = drawdown_profile(returns)
    # 1.10 -> 0.55 is a 50% drawdown from the peak.
    assert dd["max_drawdown"] == pytest.approx(-0.50)
    assert dd["max_drawdown_length"] == 2.0
    assert dd["total_return"] == pytest.approx(1.10 * 0.50 * 1.20 - 1.0)


def test_moments_requires_enough_data():
    with pytest.raises(ValueError):
        moments([0.01, 0.02])
