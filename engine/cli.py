"""Command line entry point.

Examples
--------
Run the default synthetic lab and write JSON + PNG into ``reports/``::

    python -m engine.cli --out reports

Check that a genuine edge is *not* flagged as overfitting::

    python -m engine.cli --drift-annual 0.25 --max-strategies 400

Analyse your own returns (one column per strategy)::

    python -m engine.cli --csv my_returns.csv --declared-trials 250
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from engine.backtest import FAMILIES
from engine.pipeline import AnalysisConfig, run_analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine.cli",
        description="Detect whether an impressive Sharpe ratio is real or statistical luck.",
    )
    parser.add_argument("--csv", type=str, default="", help="path to a returns CSV")
    parser.add_argument("--declared-trials", type=int, default=0)
    parser.add_argument("--n-bars", type=int, default=1250)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drift-annual", type=float, default=0.0)
    parser.add_argument("--vol-annual", type=float, default=0.20)
    parser.add_argument("--fat-tails", action="store_true")
    parser.add_argument("--autocorr", type=float, default=0.0,
                        help="AR(1) serial correlation of returns; creates a real edge")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--max-strategies", type=int, default=1000)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--max-combos", type=int, default=4000)
    parser.add_argument("--mc-paths", type=int, default=20000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--benchmark-sr-annual", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out", type=str, default="", help="directory for report.json / report.png")
    parser.add_argument("--no-figure", action="store_true", help="skip the matplotlib report")
    parser.add_argument("--cluster-trials", action="store_true",
                        help="use the scikit-learn clustering estimate of effective trials")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    # Windows consoles default to a legacy code page that mangles the em dashes
    # and Greek letters in the report text.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = _build_parser().parse_args(argv)

    csv_text = ""
    mode = "synthetic"
    if args.csv:
        with open(args.csv, "r", encoding="utf-8") as handle:
            csv_text = handle.read()
        mode = "upload"

    config = AnalysisConfig(
        mode=mode,
        n_bars=args.n_bars,
        periods_per_year=args.periods_per_year,
        seed=args.seed,
        drift_annual=args.drift_annual,
        vol_annual=args.vol_annual,
        fat_tails=args.fat_tails,
        autocorr=args.autocorr,
        cost_bps=args.cost_bps,
        max_strategies=args.max_strategies,
        families=args.families,
        n_splits=args.n_splits,
        max_combos=args.max_combos,
        mc_paths=args.mc_paths,
        bootstrap_samples=args.bootstrap_samples,
        block_size=args.block_size,
        benchmark_sr_annual=args.benchmark_sr_annual,
        confidence=args.confidence,
        csv=csv_text,
        declared_trials=args.declared_trials,
        cluster_trials=args.cluster_trials,
    )

    result = run_analysis(config)

    try:
        from engine.frames import summary  # noqa: WPS433

        print(summary(result))
    except ImportError:
        score = result["score"]
        print(f"{score['verdict']} — score {score['total']}/100 (grade {score['grade']})")
        print(score["headline"])

    for flag in result["score"]["flags"]:
        marker = {"pass": "[ok]  ", "warn": "[warn]", "fail": "[fail]"}.get(flag["level"], "[info]")
        print(f"{marker} {flag['message']}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json_path = os.path.join(args.out, "report.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {json_path}")

        if not args.no_figure:
            try:
                from engine.report import save_report  # noqa: WPS433

                png_path = save_report(result, os.path.join(args.out, "report.png"))
                print(f"wrote {png_path}")
            except ImportError:
                print("matplotlib is not installed; skipped the figure.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
