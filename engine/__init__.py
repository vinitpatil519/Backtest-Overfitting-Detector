"""Backtest Overfitting Detector — mathematical engine.

The modules imported here depend only on NumPy and SciPy so that the whole
package can be bundled into a small serverless function.

Heavier, research-oriented helpers are kept in optional modules that are *not*
imported at package level:

* ``engine.frames``  — pandas conveniences (tidy DataFrames of every metric)
* ``engine.cluster`` — scikit-learn estimate of the effective number of trials
* ``engine.report``  — matplotlib figure generation
* ``engine.cli``     — command line entry point tying all of the above together
"""

from engine.stats import (
    EULER_GAMMA,
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    drawdown_profile,
    effective_number_of_trials,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_standard_error,
)
from engine.backtest import (
    FAMILIES,
    build_specs,
    run_grid,
    simulate_prices,
)
from engine.cscv import cscv_pbo
from engine.montecarlo import (
    block_bootstrap_sharpe,
    null_max_sharpe_distribution,
)
from engine.score import robustness_score
from engine.pipeline import AnalysisConfig, run_analysis

__all__ = [
    "EULER_GAMMA",
    "FAMILIES",
    "AnalysisConfig",
    "annualize_sharpe",
    "block_bootstrap_sharpe",
    "build_specs",
    "cscv_pbo",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "drawdown_profile",
    "effective_number_of_trials",
    "expected_max_sharpe",
    "min_track_record_length",
    "moments",
    "null_max_sharpe_distribution",
    "probabilistic_sharpe_ratio",
    "robustness_score",
    "run_analysis",
    "run_grid",
    "sharpe_ratio",
    "sharpe_standard_error",
    "simulate_prices",
]

__version__ = "1.0.0"
