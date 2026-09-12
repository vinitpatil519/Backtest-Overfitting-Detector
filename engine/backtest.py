"""Synthetic market generator and a vectorised multi-strategy backtester.

The point of the lab is deliberately adversarial: by default the price path is a
driftless random walk, so *every* strategy in the grid has zero true edge.  Any
impressive Sharpe that comes out is, by construction, selection bias — which is
exactly what the Deflated Sharpe Ratio and PBO are supposed to detect.

Everything is NumPy; a 1000-strategy x 1250-bar sweep runs in well under a
second, which is what makes the whole analysis viable inside a serverless
function.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

FAMILIES: Tuple[str, ...] = ("ma_cross", "momentum", "mean_reversion", "breakout")

_HOLDS: Tuple[int, ...] = (1, 2, 3, 5)
_MA_FAST: Tuple[int, ...] = (2, 3, 5, 8, 10, 12, 15, 20, 25, 30)
_MA_SLOW: Tuple[int, ...] = (20, 30, 40, 50, 60, 80, 100, 120, 150, 200)
_MOM_LOOKBACK: Tuple[int, ...] = (5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150, 200)
_MOM_THRESHOLD: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.03)
_MR_LOOKBACK: Tuple[int, ...] = (5, 10, 15, 20, 30, 40, 60, 80, 100, 120)
_MR_ZSCORE: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
_BO_LOOKBACK: Tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150)
_BO_BAND: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.02)

__all__ = [
    "FAMILIES",
    "build_specs",
    "describe_spec",
    "run_grid",
    "simulate_prices",
]


# --------------------------------------------------------------------------- #
# Rolling helpers (all O(T) via cumulative sums)
# --------------------------------------------------------------------------- #
def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` observations along axis 0; NaN warm-up."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    csum = np.cumsum(x, axis=0)
    out = np.full_like(x, np.nan, dtype=float)
    head = csum[window - 1 :]
    tail = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=float), csum[:-window]], axis=0)
    out[window - 1 :] = (head - tail) / float(window)
    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing sample standard deviation (ddof=1) along axis 0."""
    x = np.asarray(x, dtype=float)
    if window < 2:
        return np.zeros_like(x)
    mean = _rolling_mean(x, window)
    mean_sq = _rolling_mean(x * x, window)
    var = (mean_sq - mean * mean) * (window / (window - 1.0))
    return np.sqrt(np.clip(var, 0.0, None))


def _rolling_extreme(x: np.ndarray, window: int, kind: str) -> np.ndarray:
    """Trailing max/min over ``window`` observations of a 1-D series."""
    x = np.asarray(x, dtype=float).ravel()
    out = np.full(x.shape[0], np.nan, dtype=float)
    if window < 1 or window > x.shape[0]:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1 :] = view.max(axis=-1) if kind == "max" else view.min(axis=-1)
    return out


def _shift(x: np.ndarray, periods: int) -> np.ndarray:
    """Shift forward in time, padding with NaN (no look-ahead)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if periods <= 0:
        return x.copy()
    if periods < x.shape[0]:
        out[periods:] = x[:-periods]
    return out


# --------------------------------------------------------------------------- #
# Market simulation
# --------------------------------------------------------------------------- #
def simulate_prices(
    n_bars: int = 1250,
    seed: int = 7,
    drift_annual: float = 0.0,
    vol_annual: float = 0.20,
    periods_per_year: int = 252,
    fat_tails: bool = False,
    t_df: int = 4,
    autocorr: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Geometric random walk, optionally fat-tailed and serially correlated.

    ``drift_annual = 0`` and ``autocorr = 0`` (the defaults) mean there is no
    edge to find anywhere in the price path, so the whole grid is a pure
    selection-bias experiment.

    Two controls create a *genuine* edge, which is what you use to check that the
    detector does not simply condemn everything:

    * ``drift_annual`` — a directional edge any long-biased rule can capture;
      note that it also widens the cross-section of trial Sharpes, because long
      and short variants split apart.
    * ``autocorr`` — AR(1) serial correlation in returns.  Positive values make
      momentum genuinely predictive, negative values reward mean reversion,
      without handing out free beta.
    """
    n_bars = int(max(n_bars, 50))
    ppy = int(max(periods_per_year, 1))
    rng = np.random.default_rng(int(seed))

    sigma = float(vol_annual) / math.sqrt(ppy)
    mu = float(drift_annual) / ppy

    if fat_tails:
        df = int(max(t_df, 3))
        raw = rng.standard_t(df, size=n_bars)
        raw /= math.sqrt(df / (df - 2.0))  # rescale to unit variance
    else:
        raw = rng.standard_normal(n_bars)

    rho = float(min(max(autocorr, -0.95), 0.95))
    if abs(rho) > 1e-12:
        # AR(1) with the innovation variance rescaled so the unconditional
        # volatility still equals `sigma`.
        shocks = raw * math.sqrt(1.0 - rho * rho)
        filtered = np.empty(n_bars, dtype=float)
        state = 0.0
        for i in range(n_bars):
            state = rho * state + shocks[i]
            filtered[i] = state
        raw = filtered

    log_ret = (mu - 0.5 * sigma * sigma) + sigma * raw
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    simple_ret = np.empty(n_bars, dtype=float)
    simple_ret[0] = 0.0
    simple_ret[1:] = prices[1:] / prices[:-1] - 1.0
    return {"prices": prices, "returns": simple_ret, "log_returns": log_ret}


# --------------------------------------------------------------------------- #
# Strategy grid
# --------------------------------------------------------------------------- #
def build_specs(
    families: Iterable[str] = FAMILIES,
    max_strategies: int = 1000,
) -> List[Dict[str, object]]:
    """Enumerate the parameter grid, then thin it evenly to ``max_strategies``.

    Thinning with ``linspace`` keeps the family mix proportional instead of
    truncating whichever family happens to be enumerated last.
    """
    wanted = {f for f in families if f in FAMILIES}
    if not wanted:
        wanted = set(FAMILIES)

    specs: List[Dict[str, object]] = []

    if "ma_cross" in wanted:
        for hold in _HOLDS:
            for fast in _MA_FAST:
                for slow in _MA_SLOW:
                    if slow > fast:
                        specs.append(
                            {"family": "ma_cross", "fast": fast, "slow": slow, "hold": hold}
                        )
    if "momentum" in wanted:
        for hold in _HOLDS:
            for lookback in _MOM_LOOKBACK:
                for threshold in _MOM_THRESHOLD:
                    specs.append(
                        {
                            "family": "momentum",
                            "lookback": lookback,
                            "threshold": threshold,
                            "hold": hold,
                        }
                    )
    if "mean_reversion" in wanted:
        for hold in _HOLDS:
            for lookback in _MR_LOOKBACK:
                for zscore in _MR_ZSCORE:
                    specs.append(
                        {
                            "family": "mean_reversion",
                            "lookback": lookback,
                            "zscore": zscore,
                            "hold": hold,
                        }
                    )
    if "breakout" in wanted:
        for hold in _HOLDS:
            for lookback in _BO_LOOKBACK:
                for band in _BO_BAND:
                    specs.append(
                        {
                            "family": "breakout",
                            "lookback": lookback,
                            "band": band,
                            "hold": hold,
                        }
                    )

    limit = int(max(max_strategies, 2))
    if len(specs) > limit:
        idx = np.unique(np.rint(np.linspace(0, len(specs) - 1, limit)).astype(int))
        specs = [specs[i] for i in idx]
    return specs


def describe_spec(spec: Dict[str, object]) -> str:
    """Human readable label, e.g. ``MA cross 10/100 hold 3``."""
    family = str(spec.get("family", "?"))
    hold = spec.get("hold", 1)
    if family == "ma_cross":
        return f"MA cross {spec.get('fast')}/{spec.get('slow')} · hold {hold}"
    if family == "momentum":
        return (
            f"Momentum {spec.get('lookback')}b · thr "
            f"{float(spec.get('threshold', 0.0)) * 100:.1f}% · hold {hold}"
        )
    if family == "mean_reversion":
        return (
            f"Mean-revert {spec.get('lookback')}b · z "
            f"{float(spec.get('zscore', 0.0)):.2f} · hold {hold}"
        )
    if family == "breakout":
        return (
            f"Breakout {spec.get('lookback')}b · band "
            f"{float(spec.get('band', 0.0)) * 100:.1f}% · hold {hold}"
        )
    return family


def _spec_window(spec: Dict[str, object]) -> int:
    family = str(spec.get("family"))
    hold = int(spec.get("hold", 1))
    if family == "ma_cross":
        base = int(spec.get("slow", 1))
    elif family in ("momentum", "mean_reversion", "breakout"):
        base = int(spec.get("lookback", 1))
    else:
        base = 1
    return base + hold + 2


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
def _positions_ma_cross(prices: np.ndarray, specs: List[Dict[str, object]]) -> np.ndarray:
    windows = sorted({int(s["fast"]) for s in specs} | {int(s["slow"]) for s in specs})
    index = {w: i for i, w in enumerate(windows)}
    sma = np.column_stack([_rolling_mean(prices, w) for w in windows])
    fast_idx = np.array([index[int(s["fast"])] for s in specs], dtype=int)
    slow_idx = np.array([index[int(s["slow"])] for s in specs], dtype=int)
    spread = sma[:, fast_idx] - sma[:, slow_idx]
    return np.sign(spread)


def _positions_momentum(prices: np.ndarray, specs: List[Dict[str, object]]) -> np.ndarray:
    lookbacks = sorted({int(s["lookback"]) for s in specs})
    index = {w: i for i, w in enumerate(lookbacks)}
    mom = np.column_stack([prices / _shift(prices, w) - 1.0 for w in lookbacks])
    col = np.array([index[int(s["lookback"])] for s in specs], dtype=int)
    thr = np.array([float(s["threshold"]) for s in specs], dtype=float)
    signal = mom[:, col]
    return (signal > thr).astype(float) - (signal < -thr).astype(float)


def _positions_mean_reversion(
    prices: np.ndarray, specs: List[Dict[str, object]]
) -> np.ndarray:
    lookbacks = sorted({int(s["lookback"]) for s in specs})
    index = {w: i for i, w in enumerate(lookbacks)}
    zcols = []
    for w in lookbacks:
        mean = _rolling_mean(prices, w)
        std = _rolling_std(prices, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std > 0, (prices - mean) / std, np.nan)
        zcols.append(z)
    zmat = np.column_stack(zcols)
    col = np.array([index[int(s["lookback"])] for s in specs], dtype=int)
    thr = np.array([float(s["zscore"]) for s in specs], dtype=float)
    z = zmat[:, col]
    # Fade the move: short when stretched high, long when stretched low.
    return (z < -thr).astype(float) - (z > thr).astype(float)


def _positions_breakout(prices: np.ndarray, specs: List[Dict[str, object]]) -> np.ndarray:
    lookbacks = sorted({int(s["lookback"]) for s in specs})
    index = {w: i for i, w in enumerate(lookbacks)}
    highs, lows = [], []
    for w in lookbacks:
        # Shift by one bar so the current bar never sees its own extreme.
        highs.append(_shift(_rolling_extreme(prices, w, "max"), 1))
        lows.append(_shift(_rolling_extreme(prices, w, "min"), 1))
    hmat = np.column_stack(highs)
    lmat = np.column_stack(lows)
    col = np.array([index[int(s["lookback"])] for s in specs], dtype=int)
    band = np.array([float(s["band"]) for s in specs], dtype=float)
    upper = hmat[:, col] * (1.0 + band)
    lower = lmat[:, col] * (1.0 - band)
    px = prices[:, None]
    return (px > upper).astype(float) - (px < lower).astype(float)


def _apply_hold(positions: np.ndarray, holds: np.ndarray) -> np.ndarray:
    """Smooth each column over its own holding window, then re-sign it."""
    out = positions.copy()
    for hold in np.unique(holds):
        h = int(hold)
        if h <= 1:
            continue
        cols = np.flatnonzero(holds == h)
        out[:, cols] = np.sign(_rolling_mean(positions[:, cols], h))
    return out


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def run_grid(
    prices: np.ndarray,
    asset_returns: np.ndarray,
    specs: Sequence[Dict[str, object]],
    cost_bps: float = 1.0,
) -> Dict[str, object]:
    """Backtest every spec on one price path.

    Returns a dict with the ``(T_eff, N)`` matrix of net strategy returns, the
    aligned spec list, the warm-up length that was trimmed, and per-strategy
    turnover.

    Timing convention: the position formed from data up to bar ``t`` earns the
    asset return of bar ``t+1``.  Costs are charged on the change in position at
    the bar where the trade happens.
    """
    prices = np.asarray(prices, dtype=float).ravel()
    asset_returns = np.asarray(asset_returns, dtype=float).ravel()
    specs = list(specs)
    if not specs:
        raise ValueError("empty strategy grid")

    n_bars = prices.shape[0]
    n_specs = len(specs)
    raw = np.zeros((n_bars, n_specs), dtype=float)

    for family, builder in (
        ("ma_cross", _positions_ma_cross),
        ("momentum", _positions_momentum),
        ("mean_reversion", _positions_mean_reversion),
        ("breakout", _positions_breakout),
    ):
        cols = [i for i, s in enumerate(specs) if s.get("family") == family]
        if not cols:
            continue
        raw[:, cols] = builder(prices, [specs[i] for i in cols])

    # Scrub warm-up NaNs *before* the holding filter: its cumulative-sum kernel
    # would otherwise smear a single NaN across the rest of the column.
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    holds = np.array([int(s.get("hold", 1)) for s in specs], dtype=int)
    positions = np.nan_to_num(_apply_hold(raw, holds), nan=0.0, posinf=0.0, neginf=0.0)

    # Lag the position by one bar: no bar trades on its own information.
    lagged = np.vstack([np.zeros((1, n_specs), dtype=float), positions[:-1]])
    previous = np.vstack([np.zeros((1, n_specs), dtype=float), lagged[:-1]])
    turnover = np.abs(lagged - previous)

    cost_rate = float(cost_bps) / 10_000.0
    gross = lagged * asset_returns[:, None]
    net = gross - cost_rate * turnover

    warmup = int(min(max(_spec_window(s) for s in specs), max(n_bars // 2, 1)))
    net = net[warmup:]
    turnover = turnover[warmup:]
    positions = positions[warmup:]

    # A rule whose threshold never triggers produces a flat, zero-variance column.
    # It is not a trial, it is a non-event, and leaving it in would distort both
    # the cross-sectional Sharpe variance and the CSCV ranks.
    alive = net.std(axis=0, ddof=1) > 0
    if not bool(alive.any()):
        raise ValueError("no strategy in the grid ever took a position")
    if not bool(alive.all()):
        keep = np.flatnonzero(alive)
        net = net[:, keep]
        positions = positions[:, keep]
        turnover = turnover[:, keep]
        specs = [specs[i] for i in keep]

    return {
        "returns": np.ascontiguousarray(net),
        "positions": positions,
        "turnover_per_bar": turnover.mean(axis=0),
        "specs": specs,
        "warmup": warmup,
        "labels": [describe_spec(s) for s in specs],
    }
