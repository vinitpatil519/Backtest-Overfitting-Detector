"""Vercel Python serverless function exposing the analysis engine.

Vercel detects the module-level ``app`` (an ASGI application) and serves it.
``vercel.json`` rewrites ``/api/py/*`` here and ships the ``engine/`` package
alongside this file via ``includeFiles``.

Locally:  uvicorn api.index:app --reload --port 5328
"""

from __future__ import annotations

import os
import sys
from typing import List, Literal, Optional

# Make the repository root importable when this file is the process entry point.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from engine import __version__ as ENGINE_VERSION  # noqa: E402
from engine.backtest import FAMILIES  # noqa: E402
from engine.pipeline import AnalysisConfig, run_analysis  # noqa: E402

app = FastAPI(
    title="Backtest Overfitting Detector",
    version=ENGINE_VERSION,
    docs_url="/api/py/docs",
    openapi_url="/api/py/openapi.json",
)

# Upper bounds exist so a crafted request cannot exhaust the function's
# memory or wall-clock budget.
MAX_CSV_CHARS = 4_000_000


class AnalyzeRequest(BaseModel):
    mode: Literal["synthetic", "upload"] = "synthetic"

    n_bars: int = Field(1250, ge=120, le=20_000)
    periods_per_year: int = Field(252, ge=1, le=8_760)
    seed: int = Field(7, ge=0, le=2_147_483_647)
    drift_annual: float = Field(0.0, ge=-2.0, le=2.0)
    vol_annual: float = Field(0.20, gt=0.0, le=5.0)
    fat_tails: bool = False
    autocorr: float = Field(0.0, ge=-0.9, le=0.9)
    cost_bps: float = Field(1.0, ge=0.0, le=100.0)

    max_strategies: int = Field(1000, ge=2, le=2_000)
    families: List[str] = Field(default_factory=lambda: list(FAMILIES))

    n_splits: int = Field(10, ge=4, le=16)
    max_combos: int = Field(4000, ge=50, le=12_870)

    mc_paths: int = Field(20_000, ge=500, le=200_000)
    bootstrap_samples: int = Field(2_000, ge=200, le=20_000)
    block_size: int = Field(20, ge=1, le=250)

    benchmark_sr_annual: float = Field(0.0, ge=-5.0, le=5.0)
    confidence: float = Field(0.95, ge=0.5, le=0.9999)

    csv: str = ""
    declared_trials: int = Field(0, ge=0, le=1_000_000)


@app.get("/api/py/health")
def health() -> dict:
    import numpy  # noqa: WPS433
    import scipy  # noqa: WPS433

    return {
        "status": "ok",
        "engine": ENGINE_VERSION,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


@app.get("/api/py/families")
def families() -> dict:
    return {"families": list(FAMILIES)}


@app.post("/api/py/analyze")
def analyze(request: AnalyzeRequest):
    if request.mode == "upload":
        if not request.csv.strip():
            return JSONResponse(
                status_code=400, content={"error": "No data supplied for upload mode."}
            )
        if len(request.csv) > MAX_CSV_CHARS:
            return JSONResponse(
                status_code=413,
                content={"error": "Input is too large; keep it under 4 MB of text."},
            )

    config = AnalysisConfig(
        mode=request.mode,
        n_bars=request.n_bars,
        periods_per_year=request.periods_per_year,
        seed=request.seed,
        drift_annual=request.drift_annual,
        vol_annual=request.vol_annual,
        fat_tails=request.fat_tails,
        autocorr=request.autocorr,
        cost_bps=request.cost_bps,
        max_strategies=request.max_strategies,
        families=request.families or list(FAMILIES),
        n_splits=request.n_splits,
        max_combos=request.max_combos,
        mc_paths=request.mc_paths,
        bootstrap_samples=request.bootstrap_samples,
        block_size=request.block_size,
        benchmark_sr_annual=request.benchmark_sr_annual,
        confidence=request.confidence,
        csv=request.csv,
        declared_trials=request.declared_trials,
    )

    try:
        return run_analysis(config)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={"error": "Run too large for the serverless budget; reduce the grid or paths."},
        )
