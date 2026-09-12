"""End-to-end analysis: generate or ingest returns, then run the full battery.

``run_analysis`` returns a JSON-safe dictionary consumed directly by the API and
the UI.  Every non-finite value is converted to ``None`` before it leaves this
module — ``NaN`` and ``Infinity`` are not valid JSON and silently break clients
that parse strictly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.backtest import FAMILIES, build_specs, run_grid, simulate_prices
from engine.cscv import cscv_pbo
from engine.montecarlo import block_bootstrap_sharpe, null_max_sharpe_distribution
from engine.score import robustness_score
from engine.stats import (
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
    sharpe_ratio_matrix,
    sharpe_standard_error,
)

__all__ = ["AnalysisConfig", "run_analysis", "parse_returns_csv"]

_MAX_EQUITY_POINTS = 600
_MAX_SHARPES_RETURNED = 2000


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisConfig:
    mode: str = "synthetic"  # "synthetic" | "upload"

    # Synthetic market
    n_bars: int = 1250
    periods_per_year: int = 252
    seed: int = 7
    drift_annual: float = 0.0
    vol_annual: float = 0.20
    fat_tails: bool = False
    autocorr: float = 0.0
    cost_bps: float = 1.0

    # Strategy grid
    max_strategies: int = 1000
    families: Sequence[str] = field(default_factory=lambda: list(FAMILIES))

    # CSCV
    n_splits: int = 10
    max_combos: int = 4000

    # Monte Carlo
    mc_paths: int = 20000
    bootstrap_samples: int = 2000
    block_size: int = 20

    # Inference
    benchmark_sr_annual: float = 0.0
    confidence: float = 0.95

    # Uploaded data
    csv: str = ""
    declared_trials: int = 0

    # Use the scikit-learn clustering estimate of N_eff instead of the
    # correlation heuristic. Local research only: scikit-learn is not part of
    # the serverless bundle, so this falls back automatically when missing.
    cluster_trials: bool = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _finite(x: Any) -> Optional[float]:
    """JSON-safe float: non-finite becomes ``None``."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _finite_list(x: Sequence[float] | np.ndarray) -> List[Optional[float]]:
    arr = np.asarray(x, dtype=float).ravel()
    return [float(v) if math.isfinite(v) else None for v in arr]


def _clean(obj: Any) -> Any:
    """Recursively make a structure JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _finite_list(obj)
    if isinstance(obj, (np.floating, float)):
        return _finite(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _downsample(values: np.ndarray, limit: int = _MAX_EQUITY_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    if n <= limit:
        idx = np.arange(n)
    else:
        idx = np.unique(np.rint(np.linspace(0, n - 1, limit)).astype(int))
    return idx, values[idx]


def _histogram(values: np.ndarray, bins: int = 45) -> Dict[str, Any]:
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return {"counts": counts.astype(int).tolist(), "edges": _finite_list(edges)}


def parse_returns_csv(text: str) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Parse pasted CSV/TSV returns into a ``(T, N)`` matrix.

    Accepts comma, semicolon, tab or whitespace delimiters, an optional header
    row, and either decimal returns (``0.012``) or percentages (``1.2``).  The
    percentage convention is inferred from the magnitude of the data and is
    reported back so the caller can surface the assumption.
    """
    raw_lines = [ln.strip() for ln in str(text).replace("\r", "\n").split("\n")]
    lines = [ln for ln in raw_lines if ln]
    if not lines:
        raise ValueError("no data found")

    def split(line: str) -> List[str]:
        for delimiter in (",", ";", "\t"):
            if delimiter in line:
                return [c.strip() for c in line.split(delimiter)]
        return [c for c in line.split() if c]

    header: List[str] = []
    first = split(lines[0])
    try:
        [float(c) for c in first]
    except ValueError:
        header = first
        lines = lines[1:]
    if not lines:
        raise ValueError("no numeric rows found")

    rows: List[List[float]] = []
    width = 0
    for line in lines:
        cells = split(line)
        try:
            values = [float(c) for c in cells]
        except ValueError:
            continue  # skip stray text rows (dates, blank separators)
        if not values:
            continue
        width = max(width, len(values))
        rows.append(values)
    if len(rows) < 30:
        raise ValueError("need at least 30 numeric rows")

    matrix = np.full((len(rows), width), np.nan, dtype=float)
    for i, values in enumerate(rows):
        matrix[i, : len(values)] = values

    # Drop columns that are entirely empty or constant.
    keep = []
    for j in range(width):
        col = matrix[:, j]
        if np.isfinite(col).sum() >= 30 and np.nanstd(col) > 0:
            keep.append(j)
    if not keep:
        raise ValueError("no usable numeric columns")
    matrix = matrix[:, keep]
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    as_percent = bool(np.nanmean(np.abs(matrix)) > 1.0)
    if as_percent:
        matrix = matrix / 100.0

    if header and len(header) >= max(keep) + 1:
        labels = [header[j] for j in keep]
    else:
        labels = [f"series {j + 1}" for j in range(matrix.shape[1])]

    return matrix, labels, {"as_percent": as_percent, "n_rows": int(matrix.shape[0])}


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    cfg = config
    ppy = int(max(cfg.periods_per_year, 1))
    notes: List[str] = []

    # ---------------- data ---------------------------------------------- #
    if cfg.mode == "upload":
        matrix, labels, info = parse_returns_csv(cfg.csv)
        returns = matrix
        specs = [{"family": "uploaded", "label": name} for name in labels]
        warmup = 0
        turnover = np.full(returns.shape[1], np.nan)
        market = None
        if info.get("as_percent"):
            notes.append("Input values looked like percentages and were divided by 100.")
    else:
        market = simulate_prices(
            n_bars=cfg.n_bars,
            seed=cfg.seed,
            drift_annual=cfg.drift_annual,
            vol_annual=cfg.vol_annual,
            periods_per_year=ppy,
            fat_tails=cfg.fat_tails,
            autocorr=cfg.autocorr,
        )
        specs = build_specs(cfg.families, cfg.max_strategies)
        grid = run_grid(
            market["prices"], market["returns"], specs, cost_bps=cfg.cost_bps
        )
        returns = grid["returns"]
        specs = grid["specs"]
        labels = grid["labels"]
        warmup = int(grid["warmup"])
        turnover = np.asarray(grid["turnover_per_bar"], dtype=float)
        if abs(cfg.drift_annual) < 1e-9 and abs(cfg.autocorr) < 1e-9:
            notes.append(
                "The price path is a driftless random walk, so no strategy in the grid "
                "has any true edge by construction."
            )

    n_obs, n_strategies = returns.shape
    if n_obs < 30:
        raise ValueError("need at least 30 observations after warm-up")

    # ---------------- cross-section of Sharpes --------------------------- #
    sharpes = sharpe_ratio_matrix(returns)
    sharpes_annual = sharpes * math.sqrt(ppy)
    best = int(np.argmax(sharpes))
    best_returns = returns[:, best]

    var_sharpe = float(np.var(sharpes, ddof=1)) if n_strategies > 1 else float("nan")

    m = moments(best_returns)
    sr_best = float(m["mean"] / m["std"]) if m["std"] > 0 else 0.0
    sr_best_annual = annualize_sharpe(sr_best, ppy)
    benchmark_period = deannualize_sharpe(cfg.benchmark_sr_annual, ppy)

    se = sharpe_standard_error(sr_best, n_obs, m["skew"], m["kurtosis"])
    from scipy import stats as _sps  # local import keeps module import cheap

    z_conf = float(_sps.norm.ppf(0.5 + float(cfg.confidence) / 2.0))

    # ---------------- effective trials ----------------------------------- #
    trials_method = "correlation"
    if n_strategies > 1:
        trials = effective_number_of_trials(returns, seed=cfg.seed)
        if cfg.cluster_trials and n_strategies >= 4:
            try:
                from engine.cluster import effective_trials_by_clustering
            except ImportError:
                notes.append(
                    "scikit-learn is unavailable, so the correlation heuristic was used "
                    "for the effective trial count."
                )
            else:
                clustered = effective_trials_by_clustering(returns, seed=cfg.seed)
                trials["n_eff"] = float(clustered["n_eff"])
                trials_method = f"clustering (k={clustered['k']})"
                notes.append(
                    f"Effective trials estimated by clustering into {clustered['k']} groups "
                    f"(silhouette {clustered['silhouette']:.3f})."
                )
    else:
        declared = int(max(cfg.declared_trials, 1))
        trials = {
            "n_trials": float(declared),
            "mean_corr": 0.0,
            "n_eff": float(declared),
        }
        if declared > 1:
            notes.append(
                f"A single return series was supplied; the deflation uses the {declared} "
                "trials you declared."
            )
        else:
            notes.append(
                "A single return series with one declared trial: the Deflated Sharpe "
                "collapses to the Probabilistic Sharpe Ratio."
            )

    # When only one series exists there is no cross-sectional Sharpe variance, so
    # fall back on the analytic variance of the Sharpe estimator itself.
    if not math.isfinite(var_sharpe) or var_sharpe <= 0.0:
        var_sharpe = psr_denominator(sr_best, m["skew"], m["kurtosis"]) / max(n_obs - 1, 1)
        notes.append(
            "Sharpe variance across trials was unavailable, so the analytic variance of "
            "the Sharpe estimator was used instead."
        )

    n_eff = float(trials["n_eff"])
    deflated = deflated_sharpe_ratio(
        sr_best, n_obs, m["skew"], m["kurtosis"], n_eff, var_sharpe
    )
    psr_benchmark = probabilistic_sharpe_ratio(
        sr_best, n_obs, m["skew"], m["kurtosis"], benchmark_period
    )
    mintrl = min_track_record_length(
        sr_best, m["skew"], m["kurtosis"], benchmark_period, cfg.confidence
    )
    sr_star_naive = expected_max_sharpe(trials["n_trials"], var_sharpe)

    # ---------------- CSCV / PBO ----------------------------------------- #
    pbo_result: Optional[Dict[str, Any]] = None
    pbo_error: Optional[str] = None
    if n_strategies >= 2:
        try:
            pbo_result = cscv_pbo(
                returns,
                n_splits=cfg.n_splits,
                max_combos=cfg.max_combos,
                periods_per_year=ppy,
                seed=cfg.seed,
            )
        except ValueError as exc:
            pbo_error = str(exc)
    else:
        pbo_error = "PBO needs at least two candidate strategies."

    # ---------------- Monte Carlo ---------------------------------------- #
    null = null_max_sharpe_distribution(
        n_trials=n_eff,
        sigma_sharpe=math.sqrt(max(var_sharpe, 0.0)),
        observed_sharpe=sr_best,
        n_paths=cfg.mc_paths,
        periods_per_year=ppy,
        seed=cfg.seed + 1,
    )
    bootstrap = block_bootstrap_sharpe(
        best_returns,
        n_samples=cfg.bootstrap_samples,
        block_size=cfg.block_size,
        confidence=cfg.confidence,
        periods_per_year=ppy,
        seed=cfg.seed + 2,
    )

    # ---------------- score ---------------------------------------------- #
    score = robustness_score(
        dsr=deflated["dsr"],
        pbo=pbo_result["pbo"] if pbo_result else None,
        prob_oos_loss=pbo_result["prob_oos_loss"] if pbo_result else None,
        mc_p_value=null["p_value"],
        bootstrap_prob_positive=bootstrap["prob_positive"],
        skew=m["skew"],
        kurtosis=m["kurtosis"],
        min_trl=mintrl,
        n_obs=n_obs,
    )

    # ---------------- presentation payload ------------------------------- #
    equity = np.cumprod(1.0 + best_returns)
    eq_idx, eq_vals = _downsample(equity)
    dd = drawdown_profile(best_returns)

    order = np.argsort(-sharpes_annual)
    leaderboard = [
        {
            "rank": int(i + 1),
            "index": int(j),
            "label": labels[j],
            "family": str(specs[j].get("family", "uploaded")),
            "sharpe_annual": _finite(sharpes_annual[j]),
        }
        for i, j in enumerate(order[:10])
    ]

    family_counts: Dict[str, int] = {}
    for s in specs:
        key = str(s.get("family", "uploaded"))
        family_counts[key] = family_counts.get(key, 0) + 1

    sharpe_sample = sharpes_annual
    if sharpe_sample.size > _MAX_SHARPES_RETURNED:
        rng = np.random.default_rng(cfg.seed)
        pick = np.sort(rng.choice(sharpe_sample.size, _MAX_SHARPES_RETURNED, replace=False))
        sharpe_sample = sharpe_sample[pick]

    hit_rate = float(np.mean(best_returns > 0.0))
    turnover_annual = (
        _finite(float(turnover[best]) * ppy) if np.isfinite(turnover[best]) else None
    )

    payload: Dict[str, Any] = {
        "meta": {
            "mode": cfg.mode,
            "periods_per_year": ppy,
            "seed": int(cfg.seed),
            "n_observations": int(n_obs),
            "n_strategies": int(n_strategies),
            "warmup_bars": int(warmup),
            "cost_bps": _finite(cfg.cost_bps),
            "drift_annual": _finite(cfg.drift_annual),
            "vol_annual": _finite(cfg.vol_annual),
            "fat_tails": bool(cfg.fat_tails),
            "autocorr": _finite(cfg.autocorr),
            "confidence": _finite(cfg.confidence),
            "benchmark_sr_annual": _finite(cfg.benchmark_sr_annual),
            "notes": notes,
            "runtime_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "engine_version": "1.0.0",
        },
        "universe": {
            "sharpes_annual": _finite_list(sharpe_sample),
            "sharpe_histogram": _histogram(sharpes_annual, bins=45),
            "sharpe_mean": _finite(np.mean(sharpes_annual)),
            "sharpe_std": _finite(np.std(sharpes_annual, ddof=1) if n_strategies > 1 else 0.0),
            "sharpe_max": _finite(np.max(sharpes_annual)),
            "sharpe_min": _finite(np.min(sharpes_annual)),
            "family_counts": family_counts,
            "leaderboard": leaderboard,
        },
        "best": {
            "index": best,
            "label": labels[best],
            "family": str(specs[best].get("family", "uploaded")),
            "config": _clean(specs[best]),
            "sharpe_period": _finite(sr_best),
            "sharpe_annual": _finite(sr_best_annual),
            "sharpe_se_annual": _finite(se * math.sqrt(ppy)),
            "sharpe_ci_low": _finite((sr_best - z_conf * se) * math.sqrt(ppy)),
            "sharpe_ci_high": _finite((sr_best + z_conf * se) * math.sqrt(ppy)),
            "mean_period": _finite(m["mean"]),
            "vol_annual": _finite(m["std"] * math.sqrt(ppy)),
            "skew": _finite(m["skew"]),
            "kurtosis": _finite(m["kurtosis"]),
            "excess_kurtosis": _finite(m["kurtosis"] - 3.0),
            "hit_rate": _finite(hit_rate),
            "turnover_annual": turnover_annual,
            "max_drawdown": _finite(dd["max_drawdown"]),
            "max_drawdown_length": _finite(dd["max_drawdown_length"]),
            "total_return": _finite(dd["total_return"]),
            "calmar": _finite(dd["calmar"]),
            "equity": {
                "t": eq_idx.astype(int).tolist(),
                "v": _finite_list(eq_vals),
            },
        },
        "deflated": {
            "n_trials": _finite(trials["n_trials"]),
            "n_eff": _finite(n_eff),
            "n_eff_method": trials_method,
            "mean_abs_correlation": _finite(trials["mean_corr"]),
            "variance_of_sharpes": _finite(var_sharpe),
            "sr_star_period": _finite(deflated["sr_star"]),
            "sr_star_annual": _finite(deflated["sr_star"] * math.sqrt(ppy)),
            "sr_star_naive_annual": _finite(sr_star_naive * math.sqrt(ppy)),
            "dsr": _finite(deflated["dsr"]),
            "psr_benchmark": _finite(psr_benchmark),
            "min_trl": _finite(mintrl),
            "min_trl_years": _finite(mintrl / ppy) if math.isfinite(mintrl) else None,
            "track_record_sufficient": bool(math.isfinite(mintrl) and mintrl <= n_obs),
        },
        "pbo": None
        if pbo_result is None
        else {
            "pbo": _finite(pbo_result["pbo"]),
            "n_splits": int(pbo_result["n_splits"]),
            "block_length": int(pbo_result["block_length"]),
            "n_combos": int(pbo_result["n_combos"]),
            "n_combos_total": int(pbo_result["n_combos_total"]),
            "observations_used": int(pbo_result["observations_used"]),
            "median_logit": _finite(pbo_result["median_logit"]),
            "median_relative_rank": _finite(pbo_result["median_relative_rank"]),
            "logit_histogram": {
                "counts": np.asarray(pbo_result["logit_hist_counts"]).astype(int).tolist(),
                "edges": _finite_list(pbo_result["logit_hist_edges"]),
            },
            "is_sharpe": _finite_list(pbo_result["is_sharpe"]),
            "oos_sharpe": _finite_list(pbo_result["oos_sharpe"]),
            "is_sharpe_mean": _finite(pbo_result["is_sharpe_mean"]),
            "oos_sharpe_mean": _finite(pbo_result["oos_sharpe_mean"]),
            "degradation": _finite(pbo_result["degradation"]),
            "slope": _finite(pbo_result["slope"]),
            "intercept": _finite(pbo_result["intercept"]),
            "r_squared": _finite(pbo_result["r_squared"]),
            "prob_oos_loss": _finite(pbo_result["prob_oos_loss"]),
            "most_selected": [
                {
                    "index": int(item["index"]),
                    "count": int(item["count"]),
                    "label": labels[int(item["index"])],
                }
                for item in pbo_result["most_selected"]
            ],
        },
        "pbo_error": pbo_error,
        "montecarlo": {
            "n_paths": int(null["n_paths"]),
            "n_trials_effective": int(null["n_trials_effective"]),
            "sigma_sharpe_annual": _finite(null["sigma_sharpe_annual"]),
            "histogram": {
                "counts": np.asarray(null["hist_counts"]).astype(int).tolist(),
                "edges": _finite_list(null["hist_edges"]),
            },
            "mean": _finite(null["mean"]),
            "median": _finite(null["median"]),
            "q95": _finite(null["q95"]),
            "q99": _finite(null["q99"]),
            "observed_annual": _finite(null["observed_annual"]),
            "p_value": _finite(null["p_value"]),
        },
        "bootstrap": {
            "n_samples": int(bootstrap["n_samples"]),
            "block_size": int(bootstrap["block_size"]),
            "histogram": {
                "counts": np.asarray(bootstrap["hist_counts"]).astype(int).tolist(),
                "edges": _finite_list(bootstrap["hist_edges"]),
            },
            "mean": _finite(bootstrap["mean"]),
            "median": _finite(bootstrap["median"]),
            "std": _finite(bootstrap["std"]),
            "ci_low": _finite(bootstrap["ci_low"]),
            "ci_high": _finite(bootstrap["ci_high"]),
            "prob_positive": _finite(bootstrap["prob_positive"]),
            "confidence": _finite(bootstrap["confidence"]),
        },
        "score": score,
    }

    return _clean(payload)
