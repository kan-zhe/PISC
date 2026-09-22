"""Dimensionality reduction helper (UMAP)."""

from __future__ import annotations

import numpy as np


def umap_reduce(
    data: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Project ``data`` to ``n_components`` dimensions with UMAP (deterministic)."""
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(data)


def l2_normalize(data: np.ndarray) -> np.ndarray:
    """Normalize each row to unit L2 norm."""
    return data / np.linalg.norm(data, axis=1, keepdims=True)
