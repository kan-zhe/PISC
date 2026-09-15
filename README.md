# PISC: Prototype-Induced Similarity Clustering

An unsupervised clustering method that performs K-means in a sparse,
prototype-induced similarity representation space.

Given an unlabeled dataset, PISC:

1. **Encodes** the input (pretrained sentence encoder for text, raw pixels for
   images) and **L2-normalizes** the features.
2. **Reduces** to a three-dimensional UMAP embedding.
3. **Builds an overcomplete prototype pool** with $M=200$ KMeans centroids.
4. **Selects representatives** with **DTIC** (Distance-Threshold Iterative
   Clustering): groups nearby pseudo-prototypes under an adaptively chosen
   distance threshold (greedy clique cover), then keeps at most two
   representatives per group, chosen as the closest pair.
5. **Re-encodes each sample** by its soft responses
   $\exp(\operatorname{sim}(\cdot,\cdot))$ to the selected prototypes and
   applies **top-$\kappa$ sparsification** (only the strongest responses per
   sample are kept).
6. **Clusters** with K-means in the sparse response space.

The compact prototype representation preserves cluster structure while
suppressing weak prototype associations.

## Repository structure

```
pisc/
  dtic.py        DTIC: prototype initialization, clique-cover grouping,
                 closest/farthest-pair representative selection
  features.py    soft prototype-response features (exp(-d) / exp(cos)) + top-k
  pipeline.py    end-to-end run_pisc() and quantile-for-target selection
  evaluate.py    Hungarian label alignment + macro P/R/F1, ARI, NMI
  reduce.py      UMAP reduction and L2 normalization
  config.py      per-dataset hyperparameters used in the paper
  _env.py        deterministic single-threaded execution setup
experiments/
  run_synthetic.py   3-D Gaussian clusters (10 classes, 1000 samples)
  run_20news.py       20 Newsgroups (text, 20 classes)
  run_mnist.py        MNIST (images, 10 classes)
  run_fmnist.py       Fashion-MNIST (images, 10 classes)
```

## Installation

```
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Usage

Run from the repository root (the `code/` directory):

```
python -m experiments.run_synthetic
python -m experiments.run_20news
python -m experiments.run_mnist
python -m experiments.run_fmnist
```

- `run_synthetic.py` generates the data on the fly (no download).
- `run_20news.py` downloads the test set via `sklearn` and encodes it with the
  pretrained `all-MiniLM-L6-v2` sentence encoder on first run (cached under
  `data/20news_embeddings.npy`).
- `run_mnist.py` / `run_fmnist.py` download the test sets via `torchvision`
  into `data/` on first run.

Each script prints the retained prototype count, the number of groups, and the
five clustering metrics (macro Precision / Recall / F1, ARI, NMI) as
percentages.

## Per-dataset configuration (as used in the paper)

| Dataset       | $M$ | $T$ (target) | $\kappa$ | Kernel             | F1 (report) |
|---------------|-----|--------------|----------|--------------------|-------------|
| Synthetic     | 200 | 160          | 5        | cosine             | 99.60 |
| 20 Newsgroups | 200 | 100          | 8        | euclidean          | 55.80 |
| MNIST         | 200 | 160          | 10       | euclidean          | 94.09 |
| Fashion-MNIST | 200 | 140          | 20       | cosine             | 61.52 |

$T$ and $\kappa$ are selected by a joint grid search; the soft-feature kernel
is chosen per dataset. These defaults are stored in `pisc/config.py`.

## Reproducibility

- Every run uses a fixed random seed (`seed = 42`) for data shuffling, UMAP,
  prototype KMeans, grouping, and the final KMeans.
- scikit-learn's KMeans uses OpenMP/BLAS parallel summation, which is not
  bit-reproducible across processes even with a fixed `random_state`. The
  package forces `OMP_NUM_THREADS = 1` (see `pisc/_env.py`) so that every
  reported number is reproducible exactly.
- Results are reported after aligning predicted cluster indices to the
  ground-truth labels with the Hungarian algorithm (F1); ARI and NMI are
  permutation-invariant and computed directly.

## License

MIT — see [LICENSE](LICENSE).
