"""Combinatorially Symmetric Cross-Validation and the Probability of Backtest
Overfitting (PBO).

Reference
---------
Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2017)
    "The Probability of Backtest Overfitting", Journal of Computational
    Finance 20(4).

Algorithm
---------
1. Split the ``T x N`` performance matrix into ``S`` contiguous, equally sized
   submatrices along time.
2. For each of the ``C(S, S/2)`` ways of choosing half the submatrices as the
   in-sample (IS) set, the complement is the out-of-sample (OOS) set.
3. Pick the strategy with the best IS Sharpe, then look up its *rank* among all
   ``N`` OOS Sharpes.  Relative rank ``w = rank / (N + 1)``; logit
   ``lambda = ln(w / (1 - w))``.
4. ``PBO = P[lambda <= 0]`` — the probability that the strategy selected
   in-sample lands in the bottom half out-of-sample.

Implementation note
-------------------
The Sharpe ratio of any union of submatrices is reconstructed from per-block
sums and sums of squares, so each combination costs one matrix multiply instead
of a re-scan of the returns.  Combinations are processed in chunks to keep peak
memory flat even at ``S = 16`` (12 870 combinations).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Optional

import numpy as np

__all__ = ["cscv_pbo"]

_CHUNK = 512


def _sharpe_from_moments(
    count: float, total: np.ndarray, total_sq: np.ndarray
) -> np.ndarray:
    """Sharpe per column from block sums, using the ddof=1 variance."""
    mean = total / count
    var = (total_sq - count * mean * mean) / (count - 1.0)
    std = np.sqrt(np.clip(var, 0.0, None))
    out = np.zeros_like(mean)
    good = std > 0
    out[good] = mean[good] / std[good]
    return out


def cscv_pbo(
    returns: np.ndarray,
    n_splits: int = 10,
    max_combos: Optional[int] = 4000,
    periods_per_year: int = 252,
    seed: int = 0,
    max_pairs_returned: int = 2000,
) -> Dict[str, object]:
    """Run CSCV and return PBO plus the diagnostics the UI plots.

    Parameters
    ----------
    returns
        ``(T, N)`` matrix of per-period strategy returns.
    n_splits
        ``S``, forced to be even and at least 4.
    max_combos
        Cap on the number of combinations actually evaluated.  When the full
        ``C(S, S/2)`` exceeds it, a reproducible random subsample is used.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    n_obs, n_strategies = r.shape
    if n_strategies < 2:
        raise ValueError("CSCV needs at least 2 strategies")

    s = int(n_splits)
    if s % 2 != 0:
        s += 1
    s = max(4, min(s, 16))
    while s > 4 and n_obs // s < 10:
        s -= 2
    block_len = n_obs // s
    if block_len < 2:
        raise ValueError("not enough observations for the requested number of splits")

    used = block_len * s
    trimmed = r[n_obs - used :]  # keep the most recent observations

    blocks = trimmed.reshape(s, block_len, n_strategies)
    block_sum = blocks.sum(axis=1)
    block_sumsq = (blocks * blocks).sum(axis=1)

    all_combos = list(combinations(range(s), s // 2))
    n_total_combos = len(all_combos)
    rng = np.random.default_rng(int(seed))
    if max_combos is not None and n_total_combos > int(max_combos):
        pick = rng.choice(n_total_combos, size=int(max_combos), replace=False)
        pick.sort()
        combos = [all_combos[i] for i in pick]
    else:
        combos = all_combos
    n_combos = len(combos)

    count_half = float(block_len * (s // 2))
    scale = math.sqrt(max(periods_per_year, 1))

    logits = np.empty(n_combos, dtype=float)
    relative_rank = np.empty(n_combos, dtype=float)
    is_best = np.empty(n_combos, dtype=float)
    oos_best = np.empty(n_combos, dtype=float)
    best_index = np.empty(n_combos, dtype=np.int64)

    eps = 1.0 / (2.0 * (n_strategies + 1.0))

    for start in range(0, n_combos, _CHUNK):
        chunk = combos[start : start + _CHUNK]
        mask = np.zeros((len(chunk), s), dtype=float)
        for row, combo in enumerate(chunk):
            mask[row, list(combo)] = 1.0
        anti = 1.0 - mask

        is_sr = _sharpe_from_moments(count_half, mask @ block_sum, mask @ block_sumsq)
        oos_sr = _sharpe_from_moments(count_half, anti @ block_sum, anti @ block_sumsq)

        chosen = np.argmax(is_sr, axis=1)
        rows = np.arange(len(chunk))
        chosen_oos = oos_sr[rows, chosen]

        # Rank 1 = worst OOS, rank N = best. Ties share the lower rank.
        rank = (oos_sr < chosen_oos[:, None]).sum(axis=1) + 1
        omega = rank / (n_strategies + 1.0)
        omega = np.clip(omega, eps, 1.0 - eps)

        sl = slice(start, start + len(chunk))
        logits[sl] = np.log(omega / (1.0 - omega))
        relative_rank[sl] = omega
        is_best[sl] = is_sr[rows, chosen] * scale
        oos_best[sl] = chosen_oos * scale
        best_index[sl] = chosen

    pbo = float(np.mean(logits <= 0.0))

    # Performance degradation: OOS Sharpe regressed on IS Sharpe.
    slope = intercept = r_squared = float("nan")
    if n_combos >= 3 and np.std(is_best) > 0:
        slope, intercept = np.polyfit(is_best, oos_best, 1)
        fitted = slope * is_best + intercept
        ss_res = float(np.sum((oos_best - fitted) ** 2))
        ss_tot = float(np.sum((oos_best - oos_best.mean()) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        slope = float(slope)
        intercept = float(intercept)

    prob_oos_loss = float(np.mean(oos_best <= 0.0))
    degradation = float(np.mean(oos_best - is_best))

    # Keep the payload bounded when S = 16.
    if n_combos > max_pairs_returned:
        take = rng.choice(n_combos, size=max_pairs_returned, replace=False)
        take.sort()
    else:
        take = np.arange(n_combos)

    hist_counts, hist_edges = np.histogram(logits, bins=41)

    unique, counts = np.unique(best_index, return_counts=True)
    top = np.argsort(-counts)[:8]

    return {
        "pbo": pbo,
        "n_splits": s,
        "block_length": int(block_len),
        "n_combos": int(n_combos),
        "n_combos_total": int(n_total_combos),
        "observations_used": int(used),
        "logits": logits[take].astype(float),
        "logit_hist_counts": hist_counts.astype(int),
        "logit_hist_edges": hist_edges.astype(float),
        "median_logit": float(np.median(logits)),
        "median_relative_rank": float(np.median(relative_rank)),
        "is_sharpe": is_best[take].astype(float),
        "oos_sharpe": oos_best[take].astype(float),
        "is_sharpe_mean": float(np.mean(is_best)),
        "oos_sharpe_mean": float(np.mean(oos_best)),
        "degradation": degradation,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "prob_oos_loss": prob_oos_loss,
        "most_selected": [
            {"index": int(unique[i]), "count": int(counts[i])} for i in top
        ],
    }
