"""Core statistics: Sharpe, PSR, Deflated Sharpe, MinTRL, moments, drawdowns.

References
----------
Bailey, D. and Lopez de Prado, M. (2012)
    "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2).
Bailey, D. and Lopez de Prado, M. (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting and Non-Normality", Journal of Portfolio Management 40(5).

All Sharpe ratios inside this module are expressed *per observation* unless a
function name or argument says ``annual``.  Mixing the two is the single most
common implementation error in DSR code, so the conversion helpers are explicit.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
from scipy import stats as sps

# Euler-Mascheroni constant, used by the expected maximum of N Gaussian draws.
EULER_GAMMA: float = 0.577215664901532860606512090082
_EULER_E: float = math.e

__all__ = [
    "EULER_GAMMA",
    "annualize_sharpe",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "drawdown_profile",
    "effective_number_of_trials",
    "expected_max_sharpe",
    "min_track_record_length",
    "moments",
    "probabilistic_sharpe_ratio",
    "psr_denominator",
    "sharpe_ratio",
    "sharpe_ratio_matrix",
    "sharpe_standard_error",
]


def _as_1d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


# --------------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------------- #
def moments(returns: Sequence[float] | np.ndarray) -> Dict[str, float]:
    """Sample moments of a return series.

    ``skew`` is the bias-corrected Fisher-Pearson coefficient (G1) and
    ``kurtosis`` is the bias-corrected *non-excess* kurtosis (G2 + 3), i.e. a
    Gaussian sample has kurtosis 3.  The PSR denominator below expects the
    non-excess convention.
    """
    a = _as_1d(returns)
    n = int(a.size)
    if n < 3:
        raise ValueError("need at least 3 finite observations")

    mean = float(a.mean())
    std = float(a.std(ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return {"n": n, "mean": mean, "std": 0.0, "skew": 0.0, "kurtosis": 3.0}

    skew = float(sps.skew(a, bias=False))
    kurt = float(sps.kurtosis(a, fisher=False, bias=False))
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0
    return {"n": n, "mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


# --------------------------------------------------------------------------- #
# Sharpe ratio
# --------------------------------------------------------------------------- #
def sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    risk_free_per_period: float = 0.0,
) -> float:
    """Per-observation Sharpe ratio ``(mean - rf) / std`` with ``ddof=1``."""
    a = _as_1d(returns)
    if a.size < 2:
        return 0.0
    std = float(a.std(ddof=1))
    if std <= 0.0:
        return 0.0
    return float((a.mean() - risk_free_per_period) / std)


def sharpe_ratio_matrix(
    returns: np.ndarray,
    risk_free_per_period: float = 0.0,
) -> np.ndarray:
    """Column-wise per-observation Sharpe ratios of a ``(T, N)`` matrix."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    mean = r.mean(axis=0)
    std = r.std(axis=0, ddof=1)
    out = np.zeros(r.shape[1], dtype=float)
    good = std > 0
    out[good] = (mean[good] - risk_free_per_period) / std[good]
    return out


def annualize_sharpe(sr_per_period: float, periods_per_year: int) -> float:
    return float(sr_per_period * math.sqrt(max(periods_per_year, 1)))


def deannualize_sharpe(sr_annual: float, periods_per_year: int) -> float:
    return float(sr_annual / math.sqrt(max(periods_per_year, 1)))


# --------------------------------------------------------------------------- #
# Probabilistic / Deflated Sharpe ratio
# --------------------------------------------------------------------------- #
def psr_denominator(sr: float, skew: float, kurtosis: float) -> float:
    r"""Variance factor of the Sharpe estimator under non-IID returns.

    .. math:: 1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2

    ``kurtosis`` is non-excess (3 for a Gaussian), so the term vanishes to
    ``1 + SR^2 / 2`` in the Gaussian case — the classic Lo (2002) result.
    """
    den = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(den) or den <= 0.0:
        # Degenerate moment estimates: fall back on the Gaussian variance factor.
        den = max(1.0 + 0.5 * sr * sr, 1e-12)
    return float(den)


def sharpe_standard_error(
    sr: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """Standard error of the per-observation Sharpe estimator."""
    if n_obs < 3:
        return float("inf")
    return float(math.sqrt(psr_denominator(sr, skew, kurtosis) / (n_obs - 1)))


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
) -> float:
    r"""Probability that the true Sharpe ratio exceeds ``sr_benchmark``.

    .. math::
        \widehat{PSR}(SR^*) = \Phi\!\left(
            \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}
                 {\sqrt{1 - \gamma_3\widehat{SR}
                        + \frac{\gamma_4-1}{4}\widehat{SR}^2}}\right)
    """
    if n_obs < 3:
        return float("nan")
    den = math.sqrt(psr_denominator(sr, skew, kurtosis))
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / den
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: float, variance_of_sharpes: float) -> float:
    r"""Expected maximum of ``N`` independent Sharpe ratios with zero true skill.

    .. math::
        SR^*_0 = \sqrt{V[\widehat{SR}_n]}\left[(1-\gamma)\,
        \Phi^{-1}\!\left(1-\tfrac1N\right)
        + \gamma\,\Phi^{-1}\!\left(1-\tfrac1{Ne}\right)\right]

    This is the benchmark the Deflated Sharpe Ratio must be beaten against: the
    Sharpe a researcher gets *for free* by running ``N`` trials.
    """
    n = float(n_trials)
    v = float(variance_of_sharpes)
    if not math.isfinite(n) or n < 2.0 or not math.isfinite(v) or v <= 0.0:
        return 0.0
    z1 = float(sps.norm.ppf(1.0 - 1.0 / n))
    z2 = float(sps.norm.ppf(1.0 - 1.0 / (n * _EULER_E)))
    return float(math.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: float,
    variance_of_sharpes: float,
) -> Dict[str, float]:
    """Deflated Sharpe Ratio = PSR evaluated against the selection-bias benchmark."""
    sr_star = expected_max_sharpe(n_trials, variance_of_sharpes)
    dsr = probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, sr_star)
    return {"sr_star": float(sr_star), "dsr": float(dsr)}


def min_track_record_length(
    sr: float,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    r"""Observations needed before ``SR > SR^*`` is significant at ``confidence``.

    .. math::
        MinTRL = 1 + \left[1 - \gamma_3 SR + \frac{\gamma_4-1}{4}SR^2\right]
                 \left(\frac{\Phi^{-1}(p)}{SR - SR^*}\right)^2
    """
    gap = sr - sr_benchmark
    if gap <= 0.0:
        return float("inf")
    z = float(sps.norm.ppf(confidence))
    return float(1.0 + psr_denominator(sr, skew, kurtosis) * (z / gap) ** 2)


# --------------------------------------------------------------------------- #
# Effective number of independent trials
# --------------------------------------------------------------------------- #
def effective_number_of_trials(
    returns_matrix: np.ndarray,
    max_columns: int = 400,
    seed: int = 0,
) -> Dict[str, float]:
    r"""Heuristic count of *independent* trials behind a correlated grid.

    A parameter sweep of 1000 highly correlated variants is nowhere near 1000
    independent experiments.  Using the average pairwise correlation
    :math:`\bar\rho`:

    .. math:: N_{eff} = 1 + (N - 1)(1 - \bar\rho)

    The scikit-learn clustering estimator in :mod:`engine.cluster` is a sharper
    (but much heavier) alternative for local research runs.
    """
    r = np.asarray(returns_matrix, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    n_total = int(r.shape[1])
    if n_total < 2:
        return {"n_trials": float(n_total), "mean_corr": 0.0, "n_eff": float(n_total)}

    cols = np.arange(n_total)
    if n_total > max_columns:
        rng = np.random.default_rng(seed)
        cols = np.sort(rng.choice(n_total, size=max_columns, replace=False))
    sample = r[:, cols]

    std = sample.std(axis=0, ddof=1)
    keep = std > 0
    sample = sample[:, keep]
    if sample.shape[1] < 2:
        return {"n_trials": float(n_total), "mean_corr": 0.0, "n_eff": float(n_total)}

    corr = np.corrcoef(sample, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    k = corr.shape[0]
    off = corr[np.triu_indices(k, k=1)]
    # Correlation sign is irrelevant for redundancy: a perfectly inverted
    # variant is the same experiment with the sign flipped.
    mean_corr = float(np.mean(np.abs(off))) if off.size else 0.0
    mean_corr = float(min(max(mean_corr, 0.0), 0.999))

    n_eff = 1.0 + (n_total - 1.0) * (1.0 - mean_corr)
    n_eff = float(min(max(n_eff, 1.0), float(n_total)))
    return {"n_trials": float(n_total), "mean_corr": mean_corr, "n_eff": n_eff}


# --------------------------------------------------------------------------- #
# Drawdowns and path statistics
# --------------------------------------------------------------------------- #
def drawdown_profile(returns: Sequence[float] | np.ndarray) -> Dict[str, float]:
    """Max drawdown, its length, and Calmar-style ratio on a compounded path."""
    a = _as_1d(returns)
    if a.size == 0:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_length": 0.0,
            "total_return": 0.0,
            "calmar": 0.0,
        }
    equity = np.cumprod(1.0 + a)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    # Longest stretch spent below a previous peak.
    under = dd < -1e-12
    longest = 0
    run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    total = float(equity[-1] - 1.0)
    calmar = float(total / abs(max_dd)) if max_dd < -1e-12 else 0.0
    return {
        "max_drawdown": max_dd,
        "max_drawdown_length": float(longest),
        "total_return": total,
        "calmar": calmar,
    }
