"""End-to-end PISC pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from .dtic import (
    group_similar_prototypes,
    initialize_kmeans_prototypes,
    select_representative_prototypes,
)
from .features import (
    build_soft_prototype_features,
    estimate_distance_threshold,
)

DEFAULT_QUANTILE_GRID = np.linspace(0.01, 0.60, 120)


def select_quantile_for_target(
    prototypes: np.ndarray,
    target_selected: int,
    quantile_grid: np.ndarray = DEFAULT_QUANTILE_GRID,
    max_representatives_per_group: int = 2,
    seed: int = 42,
) -> float:
    """Pick the threshold quantile whose representative count is closest to the target.

    DTIC is run for every quantile in the grid and the one yielding a
    representative count closest to ``target_selected`` is returned.
    """
    best_quantile = float(quantile_grid[0])
    best_deviation = float("inf")
    for quantile in quantile_grid:
        threshold = estimate_distance_threshold(prototypes, quantile)
        groups = group_similar_prototypes(prototypes, threshold, seed=seed)
        _, selected = select_representative_prototypes(
            prototypes, groups, max_representatives_per_group
        )
        deviation = abs(len(selected) - target_selected)
        if deviation < best_deviation:
            best_deviation = deviation
            best_quantile = float(quantile)
    return best_quantile


def run_pisc(
    data: np.ndarray,
    n_clusters: int,
    n_initial: int = 200,
    target_selected: int = 100,
    top_k: Optional[int] = None,
    metric: str = "euclidean",
    seed: int = 42,
    quantile_grid: np.ndarray = DEFAULT_QUANTILE_GRID,
    max_representatives_per_group: int = 2,
) -> dict:
    """Run the full PISC pipeline and return predictions plus diagnostics.

    Args:
        data: L2-normalized (and optionally UMAP-reduced) feature matrix.
        n_clusters: number of target clusters for the final KMeans.
        n_initial: number of overcomplete KMeans pseudo-prototypes (M).
        target_selected: target number of retained representatives (T).
        top_k: top-k sparsification (kappa); None disables it.
        metric: soft-feature kernel ("euclidean" or "cosine").
        seed: random seed for prototype KMeans, grouping, and final KMeans.
        quantile_grid: candidate distance-threshold quantiles.
        max_representatives_per_group: DTIC keeps at most this many per group.

    Returns:
        dict with keys: preds, selected_prototypes, n_selected, n_groups,
        quantile, threshold.
    """
    initial = initialize_kmeans_prototypes(data, n_prototypes=n_initial, seed=seed)
    quantile = select_quantile_for_target(
        initial, target_selected, quantile_grid, max_representatives_per_group, seed
    )
    threshold = estimate_distance_threshold(initial, quantile)
    groups = group_similar_prototypes(initial, distance_threshold=threshold, seed=seed)
    selected_prototypes, selected_indices = select_representative_prototypes(
        initial, groups, max_representatives_per_group
    )
    soft = build_soft_prototype_features(data, selected_prototypes, metric=metric, top_k=top_k)
    preds = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=seed
    ).fit_predict(soft)

    return {
        "preds": preds,
        "selected_prototypes": selected_prototypes,
        "n_selected": len(selected_indices),
        "n_groups": len(groups),
        "quantile": quantile,
        "threshold": threshold,
    }
