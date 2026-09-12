"""Evaluation metrics for recommendation models."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

from .. import _rusket

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

    # Collect predictions for all users. Prefer a single `batch_recommend()`
    # call (ALS, SVD) over one `recommend_items()` FFI round trip per user --
    # ponytail: only taken when there's no label round-trip to redo (i.e. no
    # _user_labels/_item_labels), keeping this fast path simple and provably
    # index-safe rather than re-deriving label -> index maps from
    # batch_recommend's external-id output.
    all_pred: list[list[int]] | None = None
    if not has_label_maps and hasattr(model, "batch_recommend"):
        try:
            all_pred = _predictions_via_batch_recommend(model, unique_users, k)
        except (TypeError, ValueError, ImportError):
            all_pred = None

    if all_pred is None:
        all_pred = []
        for u in unique_users:
            r_items, _r_scores = model.recommend_items(u, n=k, exclude_seen=True)
            all_pred.append(r_items.tolist())

    all_actual: list[list[int]] = [user_test_items[u] for u in unique_users]

    n_users = len(all_actual)
    if n_users == 0:
        return dict.fromkeys(metrics, 0.0)

    # Batch-compute all metrics for all users in a single FFI call.
    actual_indptr = np.zeros(n_users + 1, dtype=np.int64)
    for idx, actual in enumerate(all_actual):
        actual_indptr[idx + 1] = actual_indptr[idx] + len(actual)
    actual_flat = np.fromiter(
        (item for actual in all_actual for item in actual),
        dtype=np.int32,
        count=int(actual_indptr[-1]),
    )

    pred_arr = np.full((n_users, k), -1, dtype=np.int32)
    for idx, pred in enumerate(all_pred):
        row = pred[:k]
        if row:
            pred_arr[idx, : len(row)] = row

    ndcg, hit_rate, precision, recall = _rusket.metrics_batch(actual_indptr, actual_flat, pred_arr, k)
    metric_arrays: dict[str, Any] = {
        "ndcg": ndcg,
        "hr": hit_rate,
        "precision": precision,
        "recall": recall,
    }

    results: dict[str, float] = {}
    for m in metrics:
        arr = metric_arrays.get(m)
        results[m] = float(arr.mean()) if arr is not None else 0.0

    return results


def _predictions_via_batch_recommend(model: Any, unique_users: list[int], k: int) -> list[list[int]]:
    """Get top-k predictions for `unique_users` via a single `batch_recommend()` call.

    Only called when the model has no label maps, so `user_id`/`item_id`
    columns returned by `batch_recommend()` are already internal indices
    (see `ALS.batch_recommend`/`SVD.batch_recommend`, which only remap
    through `_user_labels`/`_item_labels` when those are set) -- no
    label -> index round trip is needed here.
    """
    df = model.batch_recommend(n=k, exclude_seen=True, format="pandas")

    preds: dict[int, list[int]] = {}
    for u, i in zip(df["user_id"].to_numpy(), df["item_id"].to_numpy(), strict=False):
        preds.setdefault(int(u), []).append(int(i))

    return [preds.get(u, []) for u in unique_users]


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
