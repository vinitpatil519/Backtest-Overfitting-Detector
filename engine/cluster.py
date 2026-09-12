"""scikit-learn estimate of the *effective* number of independent trials.

A grid of 1000 correlated parameter variants is not 1000 independent
experiments, and the Deflated Sharpe Ratio is very sensitive to that count.
This module clusters strategies on correlation distance and uses the number of
clusters as ``N_eff`` — a sharper estimator than the average-correlation
heuristic in :func:`engine.stats.effective_number_of_trials`, at the cost of a
scikit-learn dependency.

Cluster count comes from a *correlation threshold*, not from a silhouette
search. Silhouette almost always prefers the coarsest possible split on
financial correlation matrices, which would report two effective trials for a
thousand-strategy sweep and make the deflation vanish. Cutting the dendrogram at
"strategies correlated above ρ are the same experiment" is both defensible and
stable.

Optional module: nothing in the serverless path imports it.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

__all__ = ["effective_trials_by_clustering", "correlation_distance", "correlation_to_distance"]


def correlation_to_distance(rho: float) -> float:
    r"""Map a correlation cutoff onto the metric :math:`\sqrt{(1-\rho)/2}`."""
    return float(math.sqrt(max((1.0 - float(rho)) / 2.0, 0.0)))


def correlation_distance(returns: np.ndarray) -> np.ndarray:
    r"""Lopez de Prado's correlation distance :math:`\sqrt{(1-\rho)/2}`.

    The transform is a proper metric, which is what lets hierarchical clustering
    behave sensibly on a correlation matrix.
    """
    r = np.asarray(returns, dtype=float)
    corr = np.corrcoef(r, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    # Rounding can leave ~1e-9 on the diagonal, which scikit-learn rejects for a
    # precomputed metric, and can make the matrix marginally asymmetric.
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def _degenerate(n_total: int, reason: str) -> Dict[str, Any]:
    return {
        "n_trials": n_total,
        "n_eff": float(n_total),
        "k": n_total,
        "silhouette": float("nan"),
        "corr_threshold": float("nan"),
        "n_sampled": n_total,
        "reason": reason,
    }


def effective_trials_by_clustering(
    returns: np.ndarray,
    corr_threshold: float = 0.9,
    max_columns: int = 400,
    linkage: str = "average",
    seed: int = 0,
) -> Dict[str, Any]:
    """Cluster the strategies and return the cluster count as ``N_eff``.

    Parameters
    ----------
    returns
        ``(T, N)`` matrix of strategy returns.
    corr_threshold
        Strategies whose average linkage correlation exceeds this value are
        treated as one experiment.  The default of 0.9 merges only near
        duplicates.  Looser cutoffs collapse a 500-strategy sweep to a couple of
        dozen "experiments", which shrinks the selection benchmark far enough to
        let a pure-noise grid pass — the exact failure this tool exists to catch.
        Treat anything below ~0.8 as an exploratory setting, not a default.
    max_columns
        Subsample cap — the distance matrix is ``O(n²)``.  When the grid is
        larger, the cluster count is scaled back up proportionally.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    n_total = int(r.shape[1])
    if n_total < 4:
        return _degenerate(n_total, "fewer than 4 strategies")

    rng = np.random.default_rng(seed)
    cols = np.arange(n_total)
    if n_total > max_columns:
        cols = np.sort(rng.choice(n_total, size=max_columns, replace=False))
    sample = r[:, cols]
    sample = sample[:, sample.std(axis=0, ddof=1) > 0]
    if sample.shape[1] < 4:
        return _degenerate(n_total, "fewer than 4 non-degenerate strategies")

    distance = correlation_distance(sample)
    n_sampled = int(distance.shape[0])

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=correlation_to_distance(corr_threshold),
        metric="precomputed",
        linkage=linkage,
    )
    labels = model.fit_predict(distance)
    k = int(len(np.unique(labels)))

    silhouette = float("nan")
    if 1 < k < n_sampled:
        silhouette = float(silhouette_score(distance, labels, metric="precomputed"))

    # Scale the cluster count from the subsample back up to the full grid: the
    # sample is a random slice of the same population, so cluster density is
    # comparable.
    n_eff = float(k) * (n_total / float(n_sampled))
    n_eff = float(min(max(n_eff, 1.0), float(n_total)))

    return {
        "n_trials": n_total,
        "n_eff": n_eff,
        "k": k,
        "silhouette": silhouette,
        "corr_threshold": float(corr_threshold),
        "n_sampled": n_sampled,
        "reason": "ok",
    }
