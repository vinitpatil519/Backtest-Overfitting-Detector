"""Optional research modules: pandas frames, scikit-learn clustering, matplotlib.

These dependencies are intentionally absent from the serverless bundle, so every
test here skips cleanly when the package is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.pipeline import AnalysisConfig, run_analysis


@pytest.fixture(scope="module")
def result():
    return run_analysis(AnalysisConfig(max_strategies=200, n_bars=900, mc_paths=2000))


# --------------------------------------------------------------------------- #
# scikit-learn
# --------------------------------------------------------------------------- #
def test_correlation_distance_is_a_clean_metric():
    pytest.importorskip("sklearn")
    from engine.cluster import correlation_distance, correlation_to_distance

    rng = np.random.default_rng(0)
    data = rng.standard_normal((400, 12))
    distance = correlation_distance(data)

    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)
    assert distance.min() >= 0.0
    assert correlation_to_distance(1.0) == pytest.approx(0.0)
    assert correlation_to_distance(-1.0) == pytest.approx(1.0)
    assert correlation_to_distance(0.9) == pytest.approx(math.sqrt(0.05))


def test_clustering_merges_duplicates_but_keeps_distinct_strategies():
    pytest.importorskip("sklearn")
    from engine.cluster import effective_trials_by_clustering

    rng = np.random.default_rng(1)
    base = rng.standard_normal((800, 5))
    # Five real signals, each repeated six times with a little noise.
    duplicated = np.column_stack(
        [base[:, j] + 0.02 * rng.standard_normal(800) for j in range(5) for _ in range(6)]
    )
    out = effective_trials_by_clustering(duplicated, corr_threshold=0.9)
    assert out["n_trials"] == 30
    assert out["k"] == 5
    assert out["n_eff"] == pytest.approx(5.0)


def test_clustering_keeps_independent_strategies_separate():
    pytest.importorskip("sklearn")
    from engine.cluster import effective_trials_by_clustering

    rng = np.random.default_rng(2)
    independent = rng.standard_normal((2000, 20))
    out = effective_trials_by_clustering(independent, corr_threshold=0.9)
    assert out["k"] == 20
    assert out["n_eff"] == pytest.approx(20.0)


def test_clustering_never_exceeds_the_trial_count():
    pytest.importorskip("sklearn")
    from engine.cluster import effective_trials_by_clustering

    rng = np.random.default_rng(3)
    out = effective_trials_by_clustering(rng.standard_normal((500, 40)))
    assert 1.0 <= out["n_eff"] <= out["n_trials"]


def test_pipeline_can_use_the_clustering_estimate():
    pytest.importorskip("sklearn")
    out = run_analysis(
        AnalysisConfig(max_strategies=120, n_bars=800, mc_paths=1000, cluster_trials=True)
    )
    assert out["deflated"]["n_eff_method"].startswith("clustering")
    assert 1.0 <= out["deflated"]["n_eff"] <= out["meta"]["n_strategies"]


def test_both_trial_estimators_agree_that_a_random_walk_is_overfit():
    pytest.importorskip("sklearn")
    common = dict(max_strategies=400, n_bars=1250, seed=7)
    heuristic = run_analysis(AnalysisConfig(**common))
    clustered = run_analysis(AnalysisConfig(cluster_trials=True, **common))
    assert heuristic["score"]["total"] < 60
    assert clustered["score"]["total"] < 60


# --------------------------------------------------------------------------- #
# pandas
# --------------------------------------------------------------------------- #
def test_metrics_frame_has_every_headline_number(result):
    pytest.importorskip("pandas")
    from engine.frames import cscv_frame, equity_frame, metrics_frame, score_frame

    metrics = metrics_frame(result)
    assert list(metrics.columns) == ["metric", "value"]
    names = set(metrics["metric"])
    for expected in ("Deflated Sharpe", "PBO", "Monte Carlo p-value", "Robustness score"):
        assert expected in names

    assert len(score_frame(result)) == 5
    assert len(cscv_frame(result)) == len(result["pbo"]["is_sharpe"])
    assert equity_frame(result).index.name == "bar"


def test_summary_text_leads_with_the_verdict(result):
    pytest.importorskip("pandas")
    from engine.frames import summary

    text = summary(result)
    assert result["score"]["verdict"] in text
    assert "Deflated Sharpe" in text


# --------------------------------------------------------------------------- #
# matplotlib
# --------------------------------------------------------------------------- #
def test_report_renders_and_writes_a_png(result, tmp_path):
    pytest.importorskip("matplotlib")
    from engine.report import save_report

    path = save_report(result, str(tmp_path / "report.png"), dpi=60)
    assert (tmp_path / "report.png").stat().st_size > 10_000
    assert path.endswith("report.png")


def test_report_handles_a_result_without_pbo(tmp_path):
    pytest.importorskip("matplotlib")
    from engine.report import save_report

    rng = np.random.default_rng(4)
    series = rng.normal(0.0004, 0.01, 600)
    csv = "\n".join(f"{v:.8f}" for v in series)
    single = run_analysis(AnalysisConfig(mode="upload", csv=csv, declared_trials=50))
    assert single["pbo"] is None
    save_report(single, str(tmp_path / "single.png"), dpi=60)
    assert (tmp_path / "single.png").stat().st_size > 10_000
