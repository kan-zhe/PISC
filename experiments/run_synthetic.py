"""PISC on the synthetic Gaussian dataset (3-D, 10 clusters, 1000 samples)."""

from __future__ import annotations

import numpy as np

from pisc import _env  # noqa: F401  (deterministic env before sklearn import)
from pisc.config import DATASET_CONFIGS

from .common import run_dataset

N_CLUSTERS = 10
POINTS_PER_CLUSTER = 100
FEATURE_DIM = 3
CLUSTER_STD = 0.5
SEED = 42


def generate_synthetic_gaussian_data(
    n_clusters: int,
    points_per_cluster: int,
    feature_dimension: int,
    cluster_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n_clusters`` Gaussian clusters and their ground-truth labels."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(n_clusters, feature_dimension))
    data_parts, label_parts = [], []
    for cluster_id, center in enumerate(centers):
        points = center + cluster_std * rng.standard_normal(
            size=(points_per_cluster, feature_dimension)
        )
        data_parts.append(points)
        label_parts.extend([cluster_id] * points_per_cluster)
    return np.vstack(data_parts), np.asarray(label_parts)


def main() -> None:
    cfg = DATASET_CONFIGS["synthetic"]
    data, labels = generate_synthetic_gaussian_data(
        N_CLUSTERS, POINTS_PER_CLUSTER, FEATURE_DIM, CLUSTER_STD, SEED
    )
    print(f"Synthetic: {data.shape}, {len(np.unique(labels))} classes "
          f"(already 3-D, no UMAP)")
    run_dataset(data, labels, cfg, seed=SEED, reduce_dim=None)


if __name__ == "__main__":
    main()
