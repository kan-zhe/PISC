"""PISC on MNIST (image, 10 classes).

28x28 grayscale images are vectorized to 784-d pixels, L2-normalized,
UMAP-reduced to 3-D, and clustered with the paper's official configuration.
"""

from __future__ import annotations

import os

import numpy as np

from pisc import _env  # noqa: F401  (deterministic env before sklearn import)
from pisc.config import DATASET_CONFIGS

from .common import run_dataset

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "data")
SEED = 42


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load the MNIST test set as 784-d pixel vectors in [0, 1]."""
    from torchvision import datasets

    ds = datasets.MNIST(root=DATA_DIR, train=False, download=True)
    x = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y = ds.targets.numpy()
    return x, y


def main() -> None:
    cfg = DATASET_CONFIGS["mnist"]
    data, labels = load_mnist()
    print(f"MNIST: {data.shape}, {len(np.unique(labels))} classes")
    run_dataset(data, labels, cfg, seed=SEED, reduce_dim=3)


if __name__ == "__main__":
    main()
