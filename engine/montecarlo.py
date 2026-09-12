"""Monte Carlo validation: null distribution of the maximum Sharpe and a
stationary block bootstrap of the selected strategy.

Two complementary questions:

* **Selection test.**  If none of the ``N`` trials had any edge, how good would
  the *best* of them look anyway?  Simulating the maximum of ``N`` draws from
  the null answers that directly, and it is the empirical twin of the analytic
  ``expected_max_sharpe`` in :mod:`engine.stats`.
* **Estimation test.**  How wide is the Sharpe of the selected strategy?  A
  circular block bootstrap preserves short-horizon autocorrelation, which an
  i.i.d. bootstrap would destroy.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["null_max_sharpe_distribution", "block_bootstrap_sharpe"]

_MAX_CELLS = 40_000_000  # draw budget per Monte Carlo batch


def null_max_sharpe_distribution(
    n_trials: float,
    sigma_sharpe: float,
    observed_sharpe: float,
    n_paths: int = 20_000,
    periods_per_year: int = 252,
    seed: int = 0,
) -> Dict[str, object]:
    r"""Simulate :math:`\max_n \widehat{SR}_n` when every trial has zero edge.

    Sharpe estimates are simulated directly — ``SR_n ~ N(0, sigma_sharpe^2)`` —
    rather than re-simulating whole return paths.  Under the null this is exact
    up to the Gaussian approximation of the Sharpe estimator, and it turns an
    ``N x T x paths`` problem into ``N x paths``.

    ``n_trials`` should be the *effective* (independence-adjusted) trial count.
    """
    n = max(int(round(float(n_trials))), 1)
    sigma = float(sigma_sharpe)
    paths = int(max(n_paths, 200))
    scale = math.sqrt(max(periods_per_year, 1))

    if not math.isfinite(sigma) or sigma <= 0.0:
        return {
            "n_trials_effective": n,
            "n_paths": paths,
            "sigma_sharpe_annual": 0.0,
            "max_sharpe": np.zeros(paths, dtype=float),
            "mean": 0.0,
            "median": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "observed_annual": float(observed_sharpe) * scale,
            "p_value": 1.0,
            "hist_counts": np.zeros(1, dtype=int),
            "hist_edges": np.zeros(2, dtype=float),
        }

    rng = np.random.default_rng(int(seed))
    # Chunk the (paths x n) draw so memory stays bounded for large grids.
    per_batch = max(1, min(paths, _MAX_CELLS // max(n, 1)))
    maxima = np.empty(paths, dtype=float)
    filled = 0
    while filled < paths:
        rows = min(per_batch, paths - filled)
        draws = rng.normal(0.0, sigma, size=(rows, n))
        maxima[filled : filled + rows] = draws.max(axis=1)
        filled += rows

    observed = float(observed_sharpe)
    p_value = float(np.mean(maxima >= observed))
    annual = maxima * scale
    counts, edges = np.histogram(annual, bins=45)

    return {
        "n_trials_effective": n,
        "n_paths": paths,
        "sigma_sharpe_annual": float(sigma * scale),
        "max_sharpe": annual,
        "mean": float(annual.mean()),
        "median": float(np.median(annual)),
        "q95": float(np.quantile(annual, 0.95)),
        "q99": float(np.quantile(annual, 0.99)),
        "observed_annual": observed * scale,
        "p_value": p_value,
        "hist_counts": counts.astype(int),
        "hist_edges": edges.astype(float),
    }


def block_bootstrap_sharpe(
    returns: Sequence[float] | np.ndarray,
    n_samples: int = 2000,
    block_size: int = 20,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    seed: int = 0,
    max_cells: Optional[int] = _MAX_CELLS,
) -> Dict[str, object]:
    """Circular block bootstrap of the Sharpe ratio.

    Blocks wrap around the end of the series (Politis-Romano style), so every
    observation has equal resampling weight and no tail is under-represented.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    t = r.size
    if t < 20:
        raise ValueError("need at least 20 observations to bootstrap")

    block = int(min(max(block_size, 1), max(t // 4, 1)))
    n_blocks = int(math.ceil(t / block))
    samples = int(max(n_samples, 100))
    if max_cells is not None:
        samples = int(min(samples, max(100, max_cells // (n_blocks * block))))

    rng = np.random.default_rng(int(seed))
    offsets = np.arange(block)
    scale = math.sqrt(max(periods_per_year, 1))

    sharpes = np.empty(samples, dtype=float)
    batch = max(1, min(samples, 4096))
    filled = 0
    while filled < samples:
        rows = min(batch, samples - filled)
        starts = rng.integers(0, t, size=(rows, n_blocks))
        idx = (starts[:, :, None] + offsets[None, None, :]) % t
        draw = r[idx.reshape(rows, -1)[:, :t]]
        mean = draw.mean(axis=1)
        std = draw.std(axis=1, ddof=1)
        sr = np.zeros(rows, dtype=float)
        good = std > 0
        sr[good] = mean[good] / std[good]
        sharpes[filled : filled + rows] = sr
        filled += rows

    annual = sharpes * scale
    alpha = (1.0 - float(confidence)) / 2.0
    counts, edges = np.histogram(annual, bins=45)

    return {
        "n_samples": int(samples),
        "block_size": block,
        "mean": float(annual.mean()),
        "median": float(np.median(annual)),
        "std": float(annual.std(ddof=1)),
        "ci_low": float(np.quantile(annual, alpha)),
        "ci_high": float(np.quantile(annual, 1.0 - alpha)),
        "prob_positive": float(np.mean(annual > 0.0)),
        "confidence": float(confidence),
        "hist_counts": counts.astype(int),
        "hist_edges": edges.astype(float),
    }
