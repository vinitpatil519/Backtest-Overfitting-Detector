"""pandas views over an analysis result.

Optional module — pandas is *not* installed in the serverless bundle, so nothing
in :mod:`engine` imports this at module scope.  It exists for notebooks, the CLI
report and anyone who wants the numbers as tidy tables.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

__all__ = [
    "universe_frame",
    "metrics_frame",
    "score_frame",
    "cscv_frame",
    "equity_frame",
]


def universe_frame(result: Dict[str, Any]) -> pd.DataFrame:
    """One row per strategy that made it into the returned sample."""
    universe = result["universe"]
    frame = pd.DataFrame({"sharpe_annual": universe["sharpes_annual"]})
    frame.index.name = "sample_index"
    return frame


def metrics_frame(result: Dict[str, Any]) -> pd.DataFrame:
    """The headline metrics as a two-column ``metric / value`` table."""
    best = result["best"]
    deflated = result["deflated"]
    mc = result["montecarlo"]
    boot = result["bootstrap"]
    pbo = result.get("pbo") or {}

    rows = [
        ("Observations", result["meta"]["n_observations"]),
        ("Strategies tested", result["meta"]["n_strategies"]),
        ("Best Sharpe (annual)", best["sharpe_annual"]),
        ("Sharpe 95% CI low", best["sharpe_ci_low"]),
        ("Sharpe 95% CI high", best["sharpe_ci_high"]),
        ("Skew", best["skew"]),
        ("Kurtosis", best["kurtosis"]),
        ("Max drawdown", best["max_drawdown"]),
        ("Effective trials", deflated["n_eff"]),
        ("Selection benchmark SR* (annual)", deflated["sr_star_annual"]),
        ("Probabilistic Sharpe", deflated["psr_benchmark"]),
        ("Deflated Sharpe", deflated["dsr"]),
        ("MinTRL (observations)", deflated["min_trl"]),
        ("PBO", pbo.get("pbo")),
        ("IS to OOS slope", pbo.get("slope")),
        ("P(OOS Sharpe <= 0)", pbo.get("prob_oos_loss")),
        ("Monte Carlo p-value", mc["p_value"]),
        ("Null E[max Sharpe] (annual)", mc["mean"]),
        ("Bootstrap Sharpe CI low", boot["ci_low"]),
        ("Bootstrap Sharpe CI high", boot["ci_high"]),
        ("Robustness score", result["score"]["total"]),
        ("Grade", result["score"]["grade"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def score_frame(result: Dict[str, Any]) -> pd.DataFrame:
    """Score decomposition: weight, component value and points contributed."""
    return pd.DataFrame(result["score"]["components"])


def cscv_frame(result: Dict[str, Any]) -> pd.DataFrame:
    """In-sample / out-of-sample Sharpe pairs from the CSCV combinations."""
    pbo = result.get("pbo")
    if not pbo:
        return pd.DataFrame(columns=["is_sharpe", "oos_sharpe"])
    return pd.DataFrame(
        {"is_sharpe": pbo["is_sharpe"], "oos_sharpe": pbo["oos_sharpe"]}
    )


def equity_frame(result: Dict[str, Any]) -> pd.DataFrame:
    """Compounded equity curve of the selected strategy."""
    equity = result["best"]["equity"]
    return pd.DataFrame({"bar": equity["t"], "equity": equity["v"]}).set_index("bar")


def summary(result: Dict[str, Any]) -> str:
    """Printable text summary built from the frames above."""
    frame = metrics_frame(result)
    with pd.option_context("display.max_rows", None, "display.width", 100):
        body = frame.to_string(index=False, na_rep="n/a")
    score = result["score"]
    header = (
        f"{score['verdict']} — score {score['total']}/100 (grade {score['grade']})\n"
        f"{score['headline']}\n"
    )
    return header + "\n" + body + "\n"


def _quiet_nan(x: Any) -> float:
    return float("nan") if x is None else float(np.asarray(x, dtype=float))
