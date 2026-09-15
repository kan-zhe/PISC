"""Per-dataset hyperparameter configurations used in the paper.

``target_selected`` (T) and ``top_k`` (kappa) are selected by a joint grid
search; the soft-feature kernel is chosen per dataset.
"""

DATASET_CONFIGS = {
    "synthetic": {
        "n_clusters": 10,
        "n_initial": 200,
        "target_selected": 160,
        "top_k": 5,
        "metric": "cosine",
    },
    "20news": {
        "n_clusters": 20,
        "n_initial": 200,
        "target_selected": 100,
        "top_k": 8,
        "metric": "euclidean",
    },
    "mnist": {
        "n_clusters": 10,
        "n_initial": 200,
        "target_selected": 160,
        "top_k": 10,
        "metric": "euclidean",
    },
    "fmnist": {
        "n_clusters": 10,
        "n_initial": 200,
        "target_selected": 140,
        "top_k": 20,
        "metric": "cosine",
    },
}
