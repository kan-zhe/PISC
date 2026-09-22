"""Deterministic single-threaded execution.

scikit-learn's KMeans uses OpenMP/BLAS parallel summation, which is not
bit-reproducible across processes even with a fixed ``random_state``: centroids
can differ by ~1e-6, which occasionally flips distance-threshold grouping
decisions. Forcing a single thread makes every run reproducible. This module
must be imported before ``numpy``/``sklearn``/``umap`` are imported.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
