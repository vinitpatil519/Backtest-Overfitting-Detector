"""Market simulation and the vectorised backtester."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.backtest import (
    FAMILIES,
    build_specs,
    describe_spec,
    run_grid,
    simulate_prices,
)


def test_simulation_is_reproducible():
    a = simulate_prices(n_bars=500, seed=42)
    b = simulate_prices(n_bars=500, seed=42)
    c = simulate_prices(n_bars=500, seed=43)
    assert np.allclose(a["prices"], b["prices"])
    assert not np.allclose(a["prices"], c["prices"])


def test_simulation_hits_the_requested_volatility():
    market = simulate_prices(n_bars=60_000, seed=1, vol_annual=0.25, periods_per_year=252)
    realised = float(np.std(market["log_returns"], ddof=1) * math.sqrt(252))
    assert realised == pytest.approx(0.25, rel=0.03)


def test_autocorrelation_is_injected_and_preserves_volatility():
    rho = 0.35
    market = simulate_prices(n_bars=60_000, seed=2, vol_annual=0.2, autocorr=rho)
    x = market["log_returns"]
    measured = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    assert measured == pytest.approx(rho, abs=0.03)
    assert float(np.std(x, ddof=1) * math.sqrt(252)) == pytest.approx(0.2, rel=0.05)


def test_fat_tails_raise_kurtosis():
    thin = simulate_prices(n_bars=40_000, seed=3, fat_tails=False)
    fat = simulate_prices(n_bars=40_000, seed=3, fat_tails=True)
    from scipy.stats import kurtosis

    assert kurtosis(fat["log_returns"]) > kurtosis(thin["log_returns"]) + 1.0


def test_grid_is_capped_and_keeps_every_family():
    specs = build_specs(FAMILIES, max_strategies=1000)
    assert len(specs) <= 1000
    assert {s["family"] for s in specs} == set(FAMILIES)


def test_grid_can_be_restricted_to_one_family():
    specs = build_specs(["momentum"], max_strategies=100)
    assert {s["family"] for s in specs} == {"momentum"}


def test_ma_cross_never_pairs_a_slow_below_a_fast():
    specs = [s for s in build_specs(["ma_cross"], 5000) if s["family"] == "ma_cross"]
    assert all(int(s["slow"]) > int(s["fast"]) for s in specs)


def test_describe_spec_is_readable():
    label = describe_spec({"family": "ma_cross", "fast": 10, "slow": 100, "hold": 3})
    assert "10/100" in label


def test_backtest_shapes_and_warmup():
    market = simulate_prices(n_bars=800, seed=4)
    specs = build_specs(FAMILIES, max_strategies=120)
    grid = run_grid(market["prices"], market["returns"], specs, cost_bps=1.0)
    returns = grid["returns"]
    # Degenerate columns are dropped, so the grid can only shrink.
    assert 0 < returns.shape[1] <= len(specs)
    assert len(grid["specs"]) == returns.shape[1] == len(grid["labels"])
    assert returns.shape[0] == 800 - grid["warmup"]
    assert np.isfinite(returns).all()


def test_positions_stay_within_minus_one_and_one():
    market = simulate_prices(n_bars=600, seed=5)
    specs = build_specs(FAMILIES, max_strategies=200)
    grid = run_grid(market["prices"], market["returns"], specs)
    assert np.all(np.abs(grid["positions"]) <= 1.0 + 1e-12)


def test_costs_can_only_reduce_returns():
    market = simulate_prices(n_bars=900, seed=6)
    specs = build_specs(FAMILIES, max_strategies=150)
    free = run_grid(market["prices"], market["returns"], specs, cost_bps=0.0)["returns"]
    charged = run_grid(market["prices"], market["returns"], specs, cost_bps=25.0)["returns"]
    assert charged.sum() < free.sum()
    assert np.all(charged <= free + 1e-12)


def test_no_look_ahead_bias():
    """Flipping the final asset return must not change any earlier strategy return."""
    market = simulate_prices(n_bars=500, seed=7)
    specs = build_specs(["momentum"], max_strategies=40)
    base = run_grid(market["prices"], market["returns"], specs)["returns"]

    tampered = market["returns"].copy()
    tampered[-1] = -tampered[-1] * 5.0
    altered = run_grid(market["prices"], tampered, specs)["returns"]

    assert np.allclose(base[:-1], altered[:-1])


def test_driftless_grid_has_a_sharpe_cross_section_centred_near_zero():
    market = simulate_prices(n_bars=3000, seed=8, drift_annual=0.0)
    specs = build_specs(FAMILIES, max_strategies=400)
    returns = run_grid(market["prices"], market["returns"], specs, cost_bps=0.0)["returns"]
    sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1)
    assert abs(float(np.median(sharpes))) < 0.05


def test_empty_grid_is_rejected():
    market = simulate_prices(n_bars=300, seed=9)
    with pytest.raises(ValueError):
        run_grid(market["prices"], market["returns"], [])
