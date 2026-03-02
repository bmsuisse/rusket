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

    When a model has ``_user_labels`` / ``_item_labels`` (set by
    ``from_transactions()``), the test IDs are automatically mapped to
    internal 0-based indices so that ``recommend_items()`` receives valid
    indices and the recommended item indices can be compared with the
    ground truth.

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
        from rusket._dependencies import import_optional_dependency

        pd = import_optional_dependency("pandas")
    except ImportError:
        pd = None

    if pd is not None and isinstance(test_interactions, pd.DataFrame):
        if "user" not in test_interactions.columns or "item" not in test_interactions.columns:
            raise ValueError("When passing a DataFrame to evaluate(), it must contain 'user' and 'item' columns.")
        users = test_interactions["user"].values
        items = test_interactions["item"].values
    else:
        interactions = np.asarray(test_interactions)
        if interactions.ndim != 2 or interactions.shape[1] < 2:
            raise ValueError("Expected test_interactions to have shape (N, 2).")
        users = interactions[:, 0]
        items = interactions[:, 1]

    # ── Build label → internal-index lookups ──────────────────────────
    user_labels: list[Any] | None = getattr(model, "_user_labels", None)
    item_labels: list[Any] | None = getattr(model, "_item_labels", None)
    has_label_maps = user_labels is not None and item_labels is not None

    user_to_idx: dict[Any, int] | None = None
    item_to_idx: dict[Any, int] | None = None

    if has_label_maps:
        user_to_idx = {}
        for idx, lbl in enumerate(user_labels):  # type: ignore[arg-type]
            user_to_idx[lbl] = idx
            # Also store coerced int variant for dtype robustness
            try:
                user_to_idx[int(lbl)] = idx
            except (ValueError, TypeError):
                pass

        item_to_idx = {}
        # _item_labels are stored as str, so also index by numeric form
        for idx, lbl in enumerate(item_labels):  # type: ignore[arg-type]
            item_to_idx[lbl] = idx
            try:
                item_to_idx[int(lbl)] = idx
            except (ValueError, TypeError):
                pass

    # ── Group test items by user (mapped to internal indices) ─────────
    user_test_items: dict[int, list[int]] = {}
    skipped_users = 0
    skipped_items = 0

    for u, i in zip(users, items, strict=False):
        # Map user label → internal index
        if user_to_idx is not None:
            u_idx = user_to_idx.get(u)
            if u_idx is None:
                skipped_users += 1
                continue
        else:
            u_idx = int(u)

        # Map item label → internal index
        if item_to_idx is not None:
            i_idx = item_to_idx.get(i)
            if i_idx is None:
                # Also try str(i) since _item_labels stores strings
                i_idx = item_to_idx.get(str(i))
            if i_idx is None:
                skipped_items += 1
                continue
        else:
            i_idx = int(i)

        user_test_items.setdefault(u_idx, []).append(i_idx)

    if skipped_users > 0:
        total = len(users)
        warnings.warn(
            f"evaluate: skipped {skipped_users}/{total} interactions with unknown user labels.",
            stacklevel=2,
        )
    if skipped_items > 0:
        total = len(items)
        warnings.warn(
            f"evaluate: skipped {skipped_items}/{total} interactions with unknown item labels.",
            stacklevel=2,
        )

    unique_users = list(user_test_items.keys())

    if not hasattr(model, "recommend_items"):
        raise TypeError("Model must support `recommend_items(user_id, n, exclude_seen)`.")

    # Batch: collect predictions for all users
    all_actual: list[list[int]] = []
    all_pred: list[list[int]] = []

    for u in unique_users:
        r_items, _r_scores = model.recommend_items(u, n=k, exclude_seen=True)
        all_pred.append(r_items.tolist())
        all_actual.append(user_test_items[u])

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


def coverage_at_k(all_pred: list[list[int]], n_unique_items: int) -> float:
    """Compute the catalog coverage at k.

    Coverage is the proportion of the total item catalog that is recommended
    to at least one user in the top-k list.

    Parameters
    ----------
    all_pred : list of list of int
        The top-k recommended item indices for each user.
    n_unique_items : int
        The total number of unique items in the catalog.

    Returns
    -------
    float
        The coverage at k (between 0.0 and 1.0).
    """
    if not all_pred or n_unique_items <= 0:
        return 0.0

    recommended_items = set()
    for pred in all_pred:
        recommended_items.update(pred)

    return len(recommended_items) / n_unique_items


def novelty_at_k(all_pred: list[list[int]], item_popularity: dict[int, int], total_users: int) -> float:
    """Compute the mean novelty at k.

    Novelty is calculated as the mean self-information of the recommended items.
    Items that are rarely interacted with in the training set have higher self-information
    (-log2(p)), indicating higher novelty.

    Parameters
    ----------
    all_pred : list of list of int
        The top-k recommended item indices for each user.
    item_popularity : dict[int, int]
        A mapping from item index to its frequency in the training set.
    total_users : int
        The total number of users in the training set (used to compute item probability p).

    Returns
    -------
    float
        The mean novelty of the recommendations.
    """
    import math

    if not all_pred or total_users <= 0:
        return 0.0

    total_novelty = 0.0
    valid_users = 0

    for pred in all_pred:
        if not pred:
            continue

        user_novelty = 0.0
        for item in pred:
            freq = item_popularity.get(item, 0)
            # Add smoothing (1) to prevent log2(0) and p > 1
            p = (freq + 1) / (total_users + 1)
            user_novelty += -math.log2(p)

        total_novelty += user_novelty / len(pred)
        valid_users += 1

    if valid_users == 0:
        return 0.0

    return total_novelty / valid_users
