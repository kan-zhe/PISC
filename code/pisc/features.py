"""Soft prototype-response features and top-k sparsification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def estimate_distance_threshold(prototypes: np.ndarray, quantile: float) -> float:
    """Grouping threshold from a quantile of the pairwise prototype distances."""
    distances = euclidean_distances(prototypes)
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    return float(np.quantile(upper_triangle, quantile))


def build_soft_prototype_features(
    data: np.ndarray,
    prototypes: np.ndarray,
    metric: str = "euclidean",
    top_k: int | None = None,
) -> np.ndarray:
    """Map each sample to exp(sim) with each selected prototype.

    ``metric`` selects the kernel: ``"euclidean"`` uses ``exp(-d)``, ``"cosine"``
    uses ``exp(cos)``. If ``top_k`` is set, only the ``top_k`` strongest
    responses per sample are kept (weak responses zeroed).
    """
    if metric == "cosine":
        soft = np.exp(cosine_similarity(data, prototypes))
    else:
        soft = np.exp(-euclidean_distances(data, prototypes))

    if top_k is not None:
        soft = topk_sparsify(soft, top_k)
    return soft


def topk_sparsify(soft: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k strongest entries in each row, zeroing the rest."""
    if k >= soft.shape[1]:
        return soft
    topk_idx = np.argpartition(-soft, k, axis=1)[:, :k]
    mask = np.zeros_like(soft, dtype=bool)
    np.put_along_axis(mask, topk_idx, True, axis=1)
    return soft * mask
