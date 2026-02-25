"""Evaluation metrics for recommendation models."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

from . import _rusket

MetricName = Literal["ndcg", "hr", "precision", "recall"]


def evaluate(
    model: Any,
    test_interactions: Any,
    k: int = 10,
    metrics: list[MetricName] | None = None,
) -> dict[str, float]:
    """Evaluate a trained recommendation model on a test set.

    Compute metrics like NDCG@k, Hit Rate@k, Precision@k, and Recall@k using
    fast natively-backed Rust evaluation loops.

    Parameters
    ----------
    model : Any
        A trained recommendation model supporting ``recommend_items(user_id, k, exclude_seen)``.
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

    import numpy as np

    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and isinstance(test_interactions, pd.DataFrame):
        if "user" not in test_interactions.columns or "item" not in test_interactions.columns:
            raise ValueError("When passing a DataFrame to evaluate(), it must contain 'user' and 'item' columns.")
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
    user_test_items: dict[int, list[int]] = {}
    for u, i in zip(users, items, strict=False):
        user_test_items.setdefault(u, []).append(i)

    unique_users = list(user_test_items.keys())

    if not hasattr(model, "recommend_items"):
        raise TypeError("Model must support `recommend_items(user_id, n, exclude_seen)`.")

    # Batch: collect predictions for all users first
    all_actual: list[list[int]] = []
    all_pred: list[list[int]] = []

    for u in unique_users:
        try:
            r_items, _r_scores = model.recommend_items(u, n=k, exclude_seen=True)
            all_pred.append(r_items.tolist())
            all_actual.append(user_test_items[u])
        except Exception as e:
            warnings.warn(
                f"Failed to use recommend_items for user {u}: {e}. Falling back to 0.0 metrics.", stacklevel=2
            )
            return dict.fromkeys(metrics, 0.0)

    n_users = len(all_actual)
    if n_users == 0:
        return dict.fromkeys(metrics, 0.0)

    # Batch-compute metrics via Rust backend, one call per metric
    results: dict[str, float] = {}
    metric_fns: dict[str, Any] = {
        "ndcg": _rusket.ndcg_at_k,
        "hr": _rusket.hit_rate_at_k,
        "precision": _rusket.precision_at_k,
        "recall": _rusket.recall_at_k,
    }

    for m in metrics:
        fn = metric_fns.get(m)
        if fn is None:
            results[m] = 0.0
            continue
        total = sum(fn(actual, pred, k) for actual, pred in zip(all_actual, all_pred, strict=False))  # type: ignore
        results[m] = total / n_users

    return results
