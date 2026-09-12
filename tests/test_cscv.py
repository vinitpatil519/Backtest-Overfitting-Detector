"""CSCV / PBO behaviour under known regimes."""

from __future__ import annotations

import numpy as np
import pytest

from engine.cscv import cscv_pbo


def test_pure_noise_gives_pbo_near_one_half():
    """With no persistent skill the IS winner is a coin flip out-of-sample."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((1200, 60)) * 0.01
    out = cscv_pbo(returns, n_splits=10, seed=0)
    assert 0.30 < out["pbo"] < 0.70
    assert out["n_combos"] == 252
    assert out["n_splits"] == 10


def test_persistent_skill_drives_pbo_down():
    """One strategy with a genuine edge should keep winning out-of-sample."""
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((1500, 40)) * 0.01
    returns[:, 7] += 0.004  # a large, stable edge
    out = cscv_pbo(returns, n_splits=10, seed=0)
    assert out["pbo"] < 0.05
    assert out["oos_sharpe_mean"] > 0
    assert out["most_selected"][0]["index"] == 7


def test_anti_persistent_data_drives_pbo_up():
    """Flip the sign between halves: the IS winner becomes the OOS loser."""
    n_obs, n_strategies = 1200, 30
    rng = np.random.default_rng(2)
    returns = rng.standard_normal((n_obs, n_strategies)) * 0.001
    tilt = np.linspace(-0.01, 0.01, n_strategies)
    half = n_obs // 2
    returns[:half] += tilt
    returns[half:] -= tilt
    out = cscv_pbo(returns, n_splits=8, seed=0)
    assert out["pbo"] > 0.6


def test_combination_count_matches_the_binomial():
    rng = np.random.default_rng(3)
    returns = rng.standard_normal((800, 10)) * 0.01
    out = cscv_pbo(returns, n_splits=8, max_combos=None, seed=0)
    assert out["n_combos_total"] == 70  # C(8, 4)
    assert out["n_combos"] == 70


def test_max_combos_subsamples_reproducibly():
    rng = np.random.default_rng(4)
    returns = rng.standard_normal((1600, 20)) * 0.01
    first = cscv_pbo(returns, n_splits=16, max_combos=500, seed=11)
    second = cscv_pbo(returns, n_splits=16, max_combos=500, seed=11)
    assert first["n_combos"] == 500
    assert first["n_combos_total"] == 12870
    assert first["pbo"] == second["pbo"]


def test_block_sums_reproduce_a_direct_sharpe():
    """The incremental Sharpe reconstruction must equal the direct computation."""
    rng = np.random.default_rng(5)
    returns = rng.standard_normal((600, 8)) * 0.01
    out = cscv_pbo(returns, n_splits=4, max_combos=None, seed=0)
    # With S = 4 the first combination is blocks (0, 1); rebuild it by hand.
    block = out["block_length"]
    used = out["observations_used"]
    trimmed = returns[returns.shape[0] - used :]
    is_slice = trimmed[: 2 * block]
    direct = is_slice.mean(axis=0) / is_slice.std(axis=0, ddof=1)
    assert out["is_sharpe"][0] == pytest.approx(
        float(direct.max()) * np.sqrt(252), rel=1e-9
    )


def test_odd_split_counts_are_rounded_up_to_even():
    rng = np.random.default_rng(6)
    returns = rng.standard_normal((900, 12)) * 0.01
    out = cscv_pbo(returns, n_splits=7, seed=0)
    assert out["n_splits"] % 2 == 0


def test_single_strategy_is_rejected():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        cscv_pbo(rng.standard_normal((500, 1)), n_splits=8)


def test_logits_and_histogram_are_consistent():
    rng = np.random.default_rng(8)
    returns = rng.standard_normal((1000, 25)) * 0.01
    out = cscv_pbo(returns, n_splits=10, seed=0)
    assert len(out["logit_hist_edges"]) == len(out["logit_hist_counts"]) + 1
    assert int(np.sum(out["logit_hist_counts"])) == out["n_combos"]
