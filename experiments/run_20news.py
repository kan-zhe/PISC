"""PISC on 20 Newsgroups (text, 20 classes).

Documents are encoded with the pretrained ``all-MiniLM-L6-v2`` sentence
encoder (cached under ``data/``), L2-normalized, UMAP-reduced to 3-D, and
clustered with the paper's official configuration.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from pisc import _env  # noqa: F401  (deterministic env before sklearn import)
from pisc.config import DATASET_CONFIGS

from .common import run_dataset

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "data")
CACHE = os.path.join(DATA_DIR, "20news_embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"
SEED = 42


def load_20news() -> tuple[np.ndarray, np.ndarray]:
    """Fetch the test set and encode it (L2-normalized 384-d embeddings)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    newsgroups = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=SEED,
    )
    if os.path.exists(CACHE):
        embeddings = np.load(CACHE)
        print(f"Loaded cached embeddings: {embeddings.shape}")
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(
            newsgroups.data, batch_size=64, normalize_embeddings=True
        )
        np.save(CACHE, embeddings)
        print(f"Encoded and cached embeddings: {embeddings.shape}")
    return embeddings, newsgroups.target


def main() -> None:
    cfg = DATASET_CONFIGS["20news"]
    data, labels = load_20news()
    print(f"20 Newsgroups: {data.shape}, {len(np.unique(labels))} classes")
    run_dataset(data, labels, cfg, seed=SEED, reduce_dim=3)


if __name__ == "__main__":
    main()
