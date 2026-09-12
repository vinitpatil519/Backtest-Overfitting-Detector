"""End-to-end pipeline, scoring and Monte Carlo behaviour."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from engine.montecarlo import block_bootstrap_sharpe, null_max_sharpe_distribution
from engine.pipeline import AnalysisConfig, parse_returns_csv, run_analysis
from engine.score import robustness_score
from engine.stats import expected_max_sharpe


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def test_null_distribution_agrees_with_the_analytic_benchmark():
    sigma, n = 0.05, 250
    out = null_max_sharpe_distribution(
        n_trials=n, sigma_sharpe=sigma, observed_sharpe=0.0, n_paths=40_000, seed=1
    )
    analytic = expected_max_sharpe(n, sigma**2) * math.sqrt(252)
    assert out["mean"] == pytest.approx(analytic, rel=0.02)


def test_null_p_value_bounds():
    out_low = null_max_sharpe_distribution(
        n_trials=100, sigma_sharpe=0.05, observed_sharpe=10.0, n_paths=5_000, seed=2
    )
    out_high = null_max_sharpe_distribution(
        n_trials=100, sigma_sharpe=0.05, observed_sharpe=-10.0, n_paths=5_000, seed=2
    )
    assert out_low["p_value"] == 0.0
    assert out_high["p_value"] == 1.0


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0005, 0.01, 2000)
    point = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
    out = block_bootstrap_sharpe(returns, n_samples=2000, block_size=10, seed=4)
    assert out["ci_low"] < point < out["ci_high"]
    assert out["prob_positive"] > 0.5


def test_bootstrap_rejects_short_series():
    with pytest.raises(ValueError):
        block_bootstrap_sharpe(np.zeros(10) + 0.01)


# --------------------------------------------------------------------------- #
# CSV ingestion
# --------------------------------------------------------------------------- #
def test_parse_csv_handles_header_and_multiple_columns():
    rng = np.random.default_rng(5)
    data = rng.normal(0, 0.01, (100, 3))
    text = "a,b,c\n" + "\n".join(",".join(f"{v:.6f}" for v in row) for row in data)
    matrix, labels, info = parse_returns_csv(text)
    assert matrix.shape == (100, 3)
    assert labels == ["a", "b", "c"]
    assert info["as_percent"] is False


def test_parse_csv_detects_percentages():
    rng = np.random.default_rng(6)
    data = rng.normal(0, 1.5, 120)  # values around 1.5 read as percent
    text = "\n".join(f"{v:.4f}" for v in data)
    matrix, _, info = parse_returns_csv(text)
    assert info["as_percent"] is True
    assert float(np.abs(matrix).mean()) < 0.1


def test_parse_csv_accepts_tabs_and_semicolons():
    rng = np.random.default_rng(7)
    data = rng.normal(0, 0.01, (60, 2))
    for delimiter in ("\t", ";"):
        text = "\n".join(delimiter.join(f"{v:.6f}" for v in row) for row in data)
        matrix, _, _ = parse_returns_csv(text)
        assert matrix.shape == (60, 2)


def test_parse_csv_rejects_tiny_inputs():
    with pytest.raises(ValueError):
        parse_returns_csv("0.01\n0.02\n0.03")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_score_is_high_for_a_clean_result():
    out = robustness_score(
        dsr=0.99,
        pbo=0.02,
        prob_oos_loss=0.03,
        mc_p_value=0.001,
        bootstrap_prob_positive=0.99,
        skew=0.1,
        kurtosis=3.1,
        min_trl=200.0,
        n_obs=2000,
    )
    assert out["total"] > 90
    assert out["grade"] == "A"
    assert out["verdict"] == "Survives deflation"


def test_score_is_low_for_an_overfit_result():
    out = robustness_score(
        dsr=0.05,
        pbo=0.80,
        prob_oos_loss=0.70,
        mc_p_value=0.90,
        bootstrap_prob_positive=0.40,
        skew=-1.5,
        kurtosis=12.0,
        min_trl=float("inf"),
        n_obs=500,
    )
    assert out["total"] < 25
    assert out["grade"] == "F"
    assert any(flag["level"] == "fail" for flag in out["flags"])


def test_missing_components_renormalise_the_weights():
    out = robustness_score(
        dsr=1.0,
        pbo=None,
        prob_oos_loss=None,
        mc_p_value=0.0,
        bootstrap_prob_positive=1.0,
        skew=0.0,
        kurtosis=3.0,
        min_trl=10.0,
        n_obs=1000,
    )
    assert out["total"] == pytest.approx(100.0)
    unavailable = [c for c in out["components"] if not c["available"]]
    assert {c["key"] for c in unavailable} == {"pbo", "degradation"}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_default_run_is_json_serialisable_and_shaped():
    result = run_analysis(AnalysisConfig(max_strategies=200, mc_paths=2000, n_bars=800))
    text = json.dumps(result)  # raises on NaN only with allow_nan=False, so check too
    assert "NaN" not in text and "Infinity" not in text
    for key in ("meta", "universe", "best", "deflated", "pbo", "montecarlo", "bootstrap", "score"):
        assert key in result
    assert 0 < result["meta"]["n_strategies"] <= 200
    assert 0.0 <= result["pbo"]["pbo"] <= 1.0
    assert 0.0 <= result["deflated"]["dsr"] <= 1.0


def test_driftless_random_walk_is_flagged_as_overfit():
    result = run_analysis(AnalysisConfig(seed=11, n_bars=1250, max_strategies=600))
    assert result["deflated"]["dsr"] < 0.95
    assert result["score"]["total"] < 60
    # The naive Sharpe looks impressive; only the deflation catches it.
    assert result["best"]["sharpe_annual"] > 0.5


def test_a_genuine_edge_survives_deflation():
    result = run_analysis(
        AnalysisConfig(seed=11, n_bars=2000, autocorr=0.25, max_strategies=600)
    )
    assert result["pbo"]["pbo"] < 0.20
    assert result["score"]["total"] > 65
    assert result["score"]["verdict"] == "Survives deflation"


def test_runs_are_reproducible_for_a_fixed_seed():
    config = AnalysisConfig(max_strategies=150, mc_paths=1500, n_bars=700)
    first = run_analysis(config)
    second = run_analysis(config)
    assert first["best"]["sharpe_annual"] == second["best"]["sharpe_annual"]
    assert first["score"]["total"] == second["score"]["total"]


def test_upload_mode_with_a_single_series_skips_pbo():
    rng = np.random.default_rng(12)
    series = rng.normal(0.0004, 0.01, 900)
    csv = "\n".join(f"{v:.8f}" for v in series)
    result = run_analysis(AnalysisConfig(mode="upload", csv=csv, declared_trials=100))
    assert result["pbo"] is None
    assert result["pbo_error"]
    assert result["meta"]["n_strategies"] == 1
    assert result["deflated"]["n_eff"] == 100.0


def test_more_declared_trials_deflate_harder():
    rng = np.random.default_rng(13)
    series = rng.normal(0.0008, 0.01, 1200)
    csv = "\n".join(f"{v:.8f}" for v in series)
    few = run_analysis(AnalysisConfig(mode="upload", csv=csv, declared_trials=1))
    many = run_analysis(AnalysisConfig(mode="upload", csv=csv, declared_trials=5000))
    assert many["deflated"]["dsr"] < few["deflated"]["dsr"]
    assert many["deflated"]["sr_star_annual"] > few["deflated"]["sr_star_annual"]


def test_equity_curve_is_downsampled_but_ordered():
    result = run_analysis(AnalysisConfig(n_bars=5000, max_strategies=100, mc_paths=1000))
    equity = result["best"]["equity"]
    assert len(equity["t"]) <= 600
    assert equity["t"] == sorted(equity["t"])
    assert len(equity["t"]) == len(equity["v"])
