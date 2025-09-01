"""
src/eval.py

Evaluation metrics for recommender.
"""

import numpy as np


def precision_at_k(pred_items, true_items, k=10) -> float:
    """
    Precision@k.

    Args:
        pred_items: list of recommended item IDs
        true_items: set of relevant item IDs
        k: cutoff

    Returns:
        Precision@k score
    """
    pred_topk = pred_items[:k]
    return len([i for i in pred_topk if i in true_items]) / k


def ndcg_at_k(pred_items, true_items, k=10) -> float:
    """
    Normalized Discounted Cumulative Gain.

    Args:
        pred_items: list of recommended item IDs
        true_items: set of relevant item IDs
        k: cutoff

    Returns:
        NDCG@k
    """
    dcg = 0.0
    for idx, item in enumerate(pred_items[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(idx + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
    return dcg / idcg if idcg > 0 else 0.0