"""Robustness score: collapse the statistical battery into one 0-100 number.

Each component is already a probability (or is mapped onto ``[0, 1]``), so the
score is a plain weighted average scaled by 100.  Components that cannot be
computed for a given input — PBO needs at least two strategies, for example —
are dropped and the remaining weights are renormalised, so the score stays on
the same scale instead of silently collapsing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

__all__ = ["robustness_score", "COMPONENTS"]

COMPONENTS = (
    ("deflated_sharpe", "Deflated Sharpe", 0.30),
    ("pbo", "Out-of-sample selection (1 - PBO)", 0.25),
    ("degradation", "IS to OOS persistence", 0.15),
    ("monte_carlo", "Beats the luck null", 0.15),
    ("stability", "Estimation stability", 0.15),
)

_GRADES = ((80.0, "A"), (65.0, "B"), (50.0, "C"), (35.0, "D"))


def _clip01(x: float) -> float:
    if x is None or not math.isfinite(x):
        return float("nan")
    return float(min(max(x, 0.0), 1.0))


def _grade(score: float) -> str:
    for cutoff, letter in _GRADES:
        if score >= cutoff:
            return letter
    return "F"


def robustness_score(
    dsr: float,
    pbo: Optional[float],
    prob_oos_loss: Optional[float],
    mc_p_value: float,
    bootstrap_prob_positive: float,
    skew: float,
    kurtosis: float,
    min_trl: float,
    n_obs: int,
) -> Dict[str, object]:
    """Blend the metrics into a score, a grade, a verdict and a flag list."""
    values: Dict[str, float] = {
        "deflated_sharpe": _clip01(dsr),
        "pbo": _clip01(1.0 - pbo) if pbo is not None else float("nan"),
        "degradation": _clip01(1.0 - prob_oos_loss)
        if prob_oos_loss is not None
        else float("nan"),
        "monte_carlo": _clip01(1.0 - mc_p_value),
    }

    # Estimation stability: how reliably the bootstrap keeps the Sharpe positive,
    # penalised for fat left tails and for a track record shorter than MinTRL.
    tail_penalty = 0.0
    if math.isfinite(skew) and skew < 0:
        tail_penalty += min(abs(skew) * 0.15, 0.25)
    if math.isfinite(kurtosis) and kurtosis > 3.0:
        tail_penalty += min((kurtosis - 3.0) * 0.03, 0.25)
    length_ok = 1.0
    if math.isfinite(min_trl) and min_trl > 0:
        length_ok = _clip01(n_obs / min_trl)
    stability = _clip01(
        0.6 * _clip01(bootstrap_prob_positive) + 0.4 * length_ok - tail_penalty
    )
    values["stability"] = stability

    components: List[Dict[str, object]] = []
    weight_sum = 0.0
    weighted = 0.0
    for key, label, weight in COMPONENTS:
        value = values.get(key, float("nan"))
        available = math.isfinite(value)
        if available:
            weight_sum += weight
            weighted += weight * value
        components.append(
            {
                "key": key,
                "label": label,
                "weight": weight,
                "value": value if available else None,
                "available": available,
            }
        )

    total = 100.0 * (weighted / weight_sum) if weight_sum > 0 else 0.0
    for component in components:
        if component["available"]:
            component["contribution"] = round(
                100.0 * component["weight"] * float(component["value"]) / weight_sum, 2
            )
        else:
            component["contribution"] = None

    flags: List[Dict[str, str]] = []
    if math.isfinite(values["deflated_sharpe"]):
        if values["deflated_sharpe"] < 0.50:
            flags.append(
                {
                    "level": "fail",
                    "message": (
                        f"Deflated Sharpe {values['deflated_sharpe']:.1%}: after correcting for "
                        "the number of trials, the true Sharpe is more likely negative than positive."
                    ),
                }
            )
        elif values["deflated_sharpe"] < 0.95:
            flags.append(
                {
                    "level": "warn",
                    "message": (
                        f"Deflated Sharpe {values['deflated_sharpe']:.1%} is below the 95% bar "
                        "normally required to call a result significant."
                    ),
                }
            )
        else:
            flags.append(
                {
                    "level": "pass",
                    "message": f"Deflated Sharpe {values['deflated_sharpe']:.1%} clears the 95% threshold.",
                }
            )

    if pbo is not None:
        if pbo > 0.50:
            flags.append(
                {
                    "level": "fail",
                    "message": (
                        f"PBO {pbo:.1%}: the in-sample winner lands in the bottom half "
                        "out-of-sample more often than not."
                    ),
                }
            )
        elif pbo > 0.25:
            flags.append(
                {"level": "warn", "message": f"PBO {pbo:.1%} shows material selection instability."}
            )
        else:
            flags.append(
                {"level": "pass", "message": f"PBO {pbo:.1%}: selection holds up out-of-sample."}
            )

    if mc_p_value > 0.05:
        flags.append(
            {
                "level": "fail" if mc_p_value > 0.20 else "warn",
                "message": (
                    f"Monte Carlo p-value {mc_p_value:.3f}: a zero-edge grid of the same size "
                    "produces a best Sharpe this good that often."
                ),
            }
        )
    else:
        flags.append(
            {
                "level": "pass",
                "message": f"Monte Carlo p-value {mc_p_value:.3f} against the zero-edge null.",
            }
        )

    if math.isfinite(min_trl) and min_trl > n_obs:
        flags.append(
            {
                "level": "warn",
                "message": (
                    f"Track record too short: {int(n_obs)} observations against a MinTRL of "
                    f"{min_trl:,.0f} needed for significance."
                ),
            }
        )
    if math.isfinite(skew) and skew < -0.5:
        flags.append(
            {"level": "warn", "message": f"Negative skew {skew:.2f}: losses cluster in the left tail."}
        )
    if math.isfinite(kurtosis) and kurtosis > 6.0:
        flags.append(
            {
                "level": "warn",
                "message": f"Excess kurtosis {kurtosis - 3.0:.2f}: fat tails inflate the naive Sharpe.",
            }
        )

    grade = _grade(total)
    if total >= 65.0:
        verdict = "Survives deflation"
        headline = "The edge holds after correcting for selection bias and multiple testing."
    elif total >= 50.0:
        verdict = "Inconclusive"
        headline = "Some evidence of edge, but not enough to separate it from luck."
    elif total >= 35.0:
        verdict = "Likely overfit"
        headline = "Most of the apparent performance is explained by the search itself."
    else:
        verdict = "Overfit"
        headline = "The backtest is statistically indistinguishable from noise mined for a winner."

    return {
        "total": round(float(total), 1),
        "grade": grade,
        "verdict": verdict,
        "headline": headline,
        "components": components,
        "flags": flags,
    }
