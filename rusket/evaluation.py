"""Evaluation metrics for recommendation models."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from . import _rusket

MetricName = Literal["ndcg", "hr", "precision", "recall"]


def evaluate(
    model,
    test_interactions,
    k: int = 10,
    metrics: list[MetricName] | None = None,
) -> dict[str, float]:
    """Evaluate a trained recommendation model on a test set.

    Compute metrics like NDCG@k, Hit Rate@k, Precision@k, and Recall@k using
    fast natively-backed Rust evaluation loops.

    Parameters
    ----------
    model : SupportsItemFactors / Recommender
        A trained recommendation model supporting ``recommend_users(user_ids, k)``.
    test_interactions : np.ndarray or pd.DataFrame
        Ground truth test interactions. Must either have columns "user" and "item",
        or be a 2D array format.
    k : int, default=10
        The cutoff rank for evaluation.
    metrics : list of str, optional
        Metrics to compute. Default: ["ndcg", "hr", "precision", "recall"].

    Returns
    -------
    dict[str, float]
        Dictionary of averaged metric values.
    """
    if metrics is None:
        metrics = ["ndcg", "hr", "precision", "recall"]

    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and isinstance(test_interactions, pd.DataFrame):
        if "user" not in test_interactions.columns or "item" not in test_interactions.columns:
            raise ValueError(
                "When passing a DataFrame to evaluate(), it must contain 'user' and 'item' columns."
            )
        users = test_interactions["user"].values.astype(np.int32)
        items = test_interactions["item"].values.astype(np.int32)
    else:
        # Assume it's a 2D array or similar with users in col 0 and items in col 1
        interactions = np.asarray(test_interactions, dtype=np.int32)
        if interactions.ndim != 2 or interactions.shape[1] < 2:
            raise ValueError("Expected test_interactions to have shape (N, 2).")
        users = interactions[:, 0]
        items = interactions[:, 1]

    # Group test items by user
    # Note: A real world implementation can be optimized further in rust.
    # For now we use python dict to aggregate by user, which is fast enough for test lists.
    user_test_items: dict[int, list[int]] = {}
    for u, i in zip(users, items):
        if u not in user_test_items:
            user_test_items[u] = []
        user_test_items[u].append(i)

    unique_users = np.array(list(user_test_items.keys()), dtype=np.int32)

    if hasattr(model, "recommend_users"):
        # Bulk predict top-k for all test users
        try:
            r_users, r_items, r_scores = model.recommend_users(unique_users, n=k, filter_already_liked_items=True)
            
            # Form dict mapping user -> predicted items
            predictions: dict[int, list[int]] = {u: [] for u in unique_users}
            for u, i in zip(r_users, r_items):
                predictions[u].append(i)
                
        except Exception as e:
            warnings.warn(f"Failed to use bulk recommend_users: {e}. Falling back to 0.0 metrics.", stacklevel=2)
            res: dict[str, float] = {m: 0.0 for m in metrics}
            return res
    else:
        # Try a slower fallback (can be implemented later)
        raise TypeError("Model must support rank-based `recommend_users()`.")

    results: dict[str, float] = {m: 0.0 for m in metrics}
    n_users = len(unique_users)

    if n_users == 0:
        return results

    # Compute metrics calling rust backend per user
    for u in unique_users:
        actual = user_test_items[u]
        pred = predictions[u]
        
        if "ndcg" in metrics:
            results["ndcg"] += _rusket.ndcg_at_k(actual, pred, k)  # type: ignore
        if "hr" in metrics:
            results["hr"] += _rusket.hit_rate_at_k(actual, pred, k)  # type: ignore
        if "precision" in metrics:
            results["precision"] += _rusket.precision_at_k(actual, pred, k)  # type: ignore
        if "recall" in metrics:
            results["recall"] += _rusket.recall_at_k(actual, pred, k)  # type: ignore

    for metric in metrics:
        results[metric] /= n_users

    return results
