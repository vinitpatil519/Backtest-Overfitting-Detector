"""matplotlib report figures for an analysis result.

Optional module — used by :mod:`engine.cli` to write a PNG report next to the
JSON.  The web UI draws its own SVG charts, so matplotlib never ships to Vercel.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend: no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["render_report", "save_report"]

_BG = "#0b0d11"
_PANEL = "#151922"
_GRID = "#2a3142"
_TEXT = "#e6eaf2"
_MUTED = "#8b94a8"
_PASS = "#34d399"
_WARN = "#fbbf24"
_FAIL = "#f87171"
_BEAM = "#60a5fa"


def _style(ax: "plt.Axes", title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=10, pad=8, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=_MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=_MUTED, fontsize=8)


def _bars(ax: "plt.Axes", edges: List[float], counts: List[int], color: str) -> None:
    edges_arr = np.asarray(edges, dtype=float)
    counts_arr = np.asarray(counts, dtype=float)
    if edges_arr.size < 2:
        return
    widths = np.diff(edges_arr)
    ax.bar(
        edges_arr[:-1],
        counts_arr,
        width=widths,
        align="edge",
        color=color,
        alpha=0.75,
        linewidth=0,
    )


def _verdict_color(score: float) -> str:
    if score >= 65.0:
        return _PASS
    if score >= 50.0:
        return _WARN
    return _FAIL


def render_report(result: Dict[str, Any]) -> "plt.Figure":
    """Build the six-panel diagnostic figure."""
    score = result["score"]
    best = result["best"]
    deflated = result["deflated"]
    mc = result["montecarlo"]
    boot = result["bootstrap"]
    pbo = result.get("pbo")

    fig = plt.figure(figsize=(15, 10), facecolor=_BG)
    grid = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22)

    total = float(score["total"])
    fig.suptitle(
        f"Backtest Overfitting Report — {score['verdict']}  |  score {total:.1f}/100"
        f"  (grade {score['grade']})",
        color=_TEXT,
        fontsize=14,
        x=0.02,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.02,
        0.945,
        score["headline"],
        color=_MUTED,
        fontsize=9.5,
        ha="left",
    )

    # 1. Sharpe distribution across the grid ------------------------------- #
    ax = fig.add_subplot(grid[0, 0])
    hist = result["universe"]["sharpe_histogram"]
    _bars(ax, hist["edges"], hist["counts"], _BEAM)
    for value, color, label in (
        (best["sharpe_annual"], _FAIL, "best in grid"),
        (deflated["sr_star_annual"], _WARN, "SR* selection benchmark"),
        (mc["mean"], _PASS, "E[max] under null"),
    ):
        if value is not None and math.isfinite(float(value)):
            ax.axvline(float(value), color=color, linewidth=1.4, linestyle="--", label=label)
    _style(
        ax,
        f"Sharpe distribution across {result['meta']['n_strategies']} strategies",
        "annualised Sharpe",
        "strategies",
    )
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7.5)

    # 2. Deflated Sharpe meter --------------------------------------------- #
    ax = fig.add_subplot(grid[0, 1])
    dsr = float(deflated["dsr"] or 0.0)
    psr = float(deflated["psr_benchmark"] or 0.0)
    ax.barh(["PSR", "DSR"], [psr, dsr], color=[_BEAM, _verdict_color(dsr * 100)], height=0.5)
    ax.axvline(0.95, color=_WARN, linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    for i, value in enumerate([psr, dsr]):
        ax.text(min(value + 0.02, 0.9), i, f"{value:.1%}", color=_TEXT, va="center", fontsize=9)
    _style(ax, "Probabilistic vs Deflated Sharpe (0.95 threshold)", "probability", "")

    # 3. CSCV logit distribution ------------------------------------------- #
    ax = fig.add_subplot(grid[1, 0])
    if pbo:
        logit_hist = pbo["logit_histogram"]
        edges = np.asarray(logit_hist["edges"], dtype=float)
        counts = np.asarray(logit_hist["counts"], dtype=float)
        widths = np.diff(edges)
        colors = [_FAIL if edges[i] < 0 else _PASS for i in range(len(counts))]
        ax.bar(edges[:-1], counts, width=widths, align="edge", color=colors, alpha=0.8, linewidth=0)
        ax.axvline(0.0, color=_TEXT, linewidth=1.0)
        _style(
            ax,
            f"CSCV logits — PBO = {float(pbo['pbo']):.1%} over {pbo['n_combos']} splits",
            "logit of relative OOS rank",
            "combinations",
        )
    else:
        _style(ax, "CSCV unavailable", "", "")
        ax.text(0.5, 0.5, result.get("pbo_error") or "n/a", color=_MUTED, ha="center")

    # 4. IS vs OOS scatter --------------------------------------------------- #
    ax = fig.add_subplot(grid[1, 1])
    if pbo:
        xs = np.asarray(pbo["is_sharpe"], dtype=float)
        ys = np.asarray(pbo["oos_sharpe"], dtype=float)
        ax.scatter(xs, ys, s=6, color=_BEAM, alpha=0.35, linewidths=0)
        slope = pbo.get("slope")
        intercept = pbo.get("intercept")
        if slope is not None and intercept is not None and xs.size:
            line = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(
                line,
                float(slope) * line + float(intercept),
                color=_WARN,
                linewidth=1.6,
                label=f"slope {float(slope):.2f}",
            )
            ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7.5)
        ax.axhline(0.0, color=_MUTED, linewidth=0.8, linestyle=":")
        _style(ax, "Performance degradation", "in-sample Sharpe", "out-of-sample Sharpe")
    else:
        _style(ax, "Degradation unavailable", "", "")

    # 5. Monte Carlo null ----------------------------------------------------- #
    ax = fig.add_subplot(grid[2, 0])
    _bars(ax, mc["histogram"]["edges"], mc["histogram"]["counts"], _MUTED)
    observed = mc.get("observed_annual")
    if observed is not None:
        ax.axvline(
            float(observed),
            color=_FAIL,
            linewidth=1.6,
            label=f"observed {float(observed):.2f}",
        )
    if mc.get("q95") is not None:
        ax.axvline(float(mc["q95"]), color=_WARN, linestyle="--", linewidth=1.2, label="null 95%")
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7.5)
    _style(
        ax,
        f"Null max-Sharpe over {mc['n_trials_effective']} effective trials "
        f"— p = {float(mc['p_value']):.3f}",
        "annualised Sharpe",
        "paths",
    )

    # 6. Equity curve + bootstrap CI ------------------------------------------ #
    ax = fig.add_subplot(grid[2, 1])
    equity = best["equity"]
    ax.plot(equity["t"], equity["v"], color=_PASS, linewidth=1.4)
    ax.axhline(1.0, color=_MUTED, linestyle=":", linewidth=0.8)
    _style(
        ax,
        f"Selected strategy: {best['label']}  |  bootstrap Sharpe CI "
        f"[{float(boot['ci_low']):.2f}, {float(boot['ci_high']):.2f}]",
        "bar",
        "equity (x)",
    )

    return fig


def save_report(result: Dict[str, Any], path: str, dpi: int = 140) -> str:
    """Render and write the report to ``path``; returns the absolute path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig = render_report(result)
    fig.savefig(path, dpi=dpi, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)
