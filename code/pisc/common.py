"""Shared experiment runner: L2 normalize -> (UMAP) -> PISC -> evaluate."""

from __future__ import annotations

import numpy as np

from pisc import evaluate_clustering, l2_normalize, run_pisc, umap_reduce


def run_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    seed: int = 42,
    reduce_dim: int | None = 3,
    verbose: bool = True,
):
    """Run the PISC pipeline on ``data`` and return metrics plus diagnostics.

    ``reduce_dim=None`` skips UMAP (used when the data already lives in the
    target low-dimensional space, e.g. the synthetic dataset).
    """
    data = l2_normalize(data)
    if reduce_dim is not None:
        data = umap_reduce(data, n_components=reduce_dim, seed=seed)

    result = run_pisc(
        data,
        n_clusters=cfg["n_clusters"],
        n_initial=cfg["n_initial"],
        target_selected=cfg["target_selected"],
        top_k=cfg["top_k"],
        metric=cfg["metric"],
        seed=seed,
    )
    metrics = evaluate_clustering(labels, result["preds"])

    if verbose:
        print(f"  retained {result['n_selected']}/{cfg['n_initial']} prototypes "
              f"in {result['n_groups']} groups (quantile {result['quantile']:.4f})")
        for key, value in metrics.items():
            print(f"  {key:10s}: {value:.2f}")

    return metrics, result
