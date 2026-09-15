"""PISC: Prototype-Induced Similarity Clustering.

An unsupervised clustering method that operates in a sparse
prototype-induced representation space:

  encode -> L2 normalize -> (UMAP-3D) -> KMeans pseudo-prototypes
  -> DTIC distance-threshold grouping -> closest-pair representative
     selection -> exp(sim) soft responses -> top-k sparsification
  -> KMeans.

See README.md for the method description and usage.
"""

from . import _env  # noqa: F401  (deterministic single-threaded execution first)
from .dtic import (
    group_similar_prototypes,
    initialize_kmeans_prototypes,
    select_representative_prototypes,
)
from .features import build_soft_prototype_features, estimate_distance_threshold
from .pipeline import run_pisc, select_quantile_for_target
from .reduce import l2_normalize, umap_reduce
from .evaluate import align_predicted_cluster_labels, evaluate_clustering

__version__ = "1.0.0"
__all__ = [
    "group_similar_prototypes",
    "initialize_kmeans_prototypes",
    "select_representative_prototypes",
    "build_soft_prototype_features",
    "estimate_distance_threshold",
    "run_pisc",
    "select_quantile_for_target",
    "l2_normalize",
    "umap_reduce",
    "align_predicted_cluster_labels",
    "evaluate_clustering",
]
