"""Evaluation: Hungarian label alignment and standard clustering metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


def align_predicted_cluster_labels(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Map predicted cluster IDs onto ground-truth IDs via Hungarian matching.

    Alignment is applied only when the number of unique true and predicted
    labels coincide; otherwise the raw predictions are returned unchanged.
    Ground-truth outliers (label -1) are excluded from the alignment.
    """
    non_noise_mask = y_true != -1
    y_true_f = y_true[non_noise_mask]
    y_pred_f = y_pred[non_noise_mask]

    true_labels = np.unique(y_true_f)
    pred_labels = np.unique(y_pred_f)
    if len(true_labels) != len(pred_labels):
        return y_pred

    overlap = np.zeros((len(true_labels), len(pred_labels)))
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            overlap[i, j] = np.sum((y_true_f == t) & (y_pred_f == p))

    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {pred_labels[c]: true_labels[r] for r, c in zip(row_ind, col_ind)}

    aligned = np.array([mapping.get(p, p) for p in y_pred])
    return aligned


def evaluate_clustering(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute macro P/R/F1 (label-aligned), ARI, and NMI.

    Returns values as percentages. ARI/NMI are computed directly (permutation
    invariant); F1 is computed after Hungarian alignment.
    """
    aligned = align_predicted_cluster_labels(y_true, y_pred)
    non_noise_mask = y_true != -1
    y_true_eval = y_true[non_noise_mask]
    y_pred_eval = aligned[non_noise_mask]

    precision = precision_score(y_true_eval, y_pred_eval, average="macro", zero_division=0)
    recall = recall_score(y_true_eval, y_pred_eval, average="macro", zero_division=0)
    f1 = f1_score(y_true_eval, y_pred_eval, average="macro", zero_division=0)
    ari = adjusted_rand_score(y_true_eval, y_pred_eval)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return {
        "Precision": float(precision * 100),
        "Recall": float(recall * 100),
        "F1": float(f1 * 100),
        "ARI": float(ari * 100),
        "NMI": float(nmi * 100),
    }
