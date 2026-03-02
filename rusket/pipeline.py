"""Multi-stage recommendation pipeline: retrieve → rerank → filter.

Inspired by Twitter/X's recommendation architecture, this module provides a
composable ``Pipeline`` that chains multiple recommendation models into a
production-style funnel.

Example
-------
>>> from rusket import ALS, BPR, ItemKNN, Pipeline
>>>
>>> als = ALS(factors=64).from_transactions(df, user_col="user", item_col="item").fit()
>>> bpr = BPR(factors=64).from_transactions(df, user_col="user", item_col="item").fit()
>>> knn = ItemKNN(k=50).from_transactions(df, user_col="user", item_col="item").fit()
>>>
>>> pipeline = Pipeline(
...     retrieve=[als, knn],           # cheap candidate generation
...     rerank=bpr,                    # expensive re-scoring
...     filter=lambda ids, sc: (       # business rules
...         [i for i in ids if i not in blocked],
...         [s for i, s in zip(ids, sc) if i not in blocked],
...     ),
... )
>>> items, scores = pipeline.recommend(user_id=42, n=10)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np


class Pipeline:
    """Multi-stage recommendation pipeline.

    Composes multiple recommendation models into a **retrieve → rerank → filter**
    funnel, following the architecture used by production recommendation systems
    at Twitter/X, YouTube, and Spotify.

    Parameters
    ----------
    retrieve : list or single model
        One or more ``ImplicitRecommender`` instances used for candidate
        generation.  Each model's ``recommend_items()`` is called and results
        are merged.
    rerank : model, optional
        An ``ImplicitRecommender`` used to re-score the merged candidate set.
        Typically a heavier model (e.g. BPR or LightGCN) that produces
        higher-quality rankings on a smaller candidate pool.
    filter : callable, optional
        A function ``(item_ids, scores) -> (filtered_ids, filtered_scores)``
        applied after re-ranking.  Use for block lists, category restrictions,
        recency filters, NSFW removal, etc.
    merge_strategy : {'max', 'mean', 'sum'}, default='max'
        How to combine scores when multiple retrievers return the same item.

    Examples
    --------
    >>> pipeline = Pipeline(
    ...     retrieve=[als, item_knn],
    ...     rerank=bpr,
    ...     filter=lambda ids, sc: (
    ...         [i for i in ids if i not in blocked_set],
    ...         [s for i, s in zip(ids, sc) if i not in blocked_set],
    ...     ),
    ... )
    >>> items, scores = pipeline.recommend(user_id=42, n=10)
    """

    def __init__(
        self,
        retrieve: Any | list[Any] | None = None,
        rerank: Any | None = None,
        filter: Callable[[list[Any], list[float]], tuple[list[Any], list[float]]] | None = None,
        merge_strategy: Literal["max", "mean", "sum"] = "max",
    ) -> None:
        if retrieve is None:
            raise ValueError("At least one retriever model is required.")

        self.retrievers: list[Any] = retrieve if isinstance(retrieve, list) else [retrieve]
        self.reranker: Any | None = rerank
        self.filter_fn: Callable[[list[Any], list[float]], tuple[list[Any], list[float]]] | None = filter
        self.merge_strategy: str = merge_strategy

        if not self.retrievers:
            raise ValueError("At least one retriever model is required.")

    def __repr__(self) -> str:
        retriever_names = [type(r).__name__ for r in self.retrievers]
        reranker_name = type(self.reranker).__name__ if self.reranker else "None"
        filter_name = "custom" if self.filter_fn else "None"
        return (
            f"Pipeline(retrieve={retriever_names}, "
            f"rerank={reranker_name}, filter={filter_name}, "
            f"merge={self.merge_strategy})"
        )

    def recommend(
        self,
        user_id: int | Any,
        n: int = 10,
        exclude_seen: bool = True,
        retrieve_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the full pipeline for a single user.

        Parameters
        ----------
        user_id : int or any
            The user to generate recommendations for.
        n : int, default=10
            Number of final items to return.
        exclude_seen : bool, default=True
            Whether to exclude items the user has already interacted with.
        retrieve_k : int, optional
            Number of candidates per retriever.  Defaults to ``n * 10``
            to produce a wide candidate pool for re-ranking.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, scores)`` arrays, sorted by descending score.
        """
        import numpy as np

        k = retrieve_k or n * 10

        # ── Stage 1: Retrieve ──────────────────────────────────────────────
        merged = self._retrieve(user_id, k, exclude_seen)

        if not merged:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        candidate_ids = np.array(list(merged.keys()), dtype=np.int64)
        candidate_scores = np.array(list(merged.values()), dtype=np.float64)

        # ── Stage 2: Rerank ────────────────────────────────────────────────
        if self.reranker is not None:
            candidate_scores = self._rerank_for_user(user_id, candidate_ids, candidate_scores)

        # ── Stage 3: Filter ────────────────────────────────────────────────
        if self.filter_fn is not None:
            candidate_ids, candidate_scores = self._filter(candidate_ids, candidate_scores)
            if len(candidate_ids) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # ── Final: sort and top-N ──────────────────────────────────────────
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        candidate_scores = np.asarray(candidate_scores, dtype=np.float64)

        if len(candidate_ids) <= n:
            order = np.argsort(-candidate_scores)
            return candidate_ids[order], candidate_scores[order]

        top_idx = np.argpartition(candidate_scores, -n)[-n:]
        top_idx = top_idx[np.argsort(-candidate_scores[top_idx])]
        return candidate_ids[top_idx], candidate_scores[top_idx]

    def recommend_batch(
        self,
        user_ids: list[int | Any] | np.ndarray | None = None,
        n: int = 10,
        exclude_seen: bool = True,
        retrieve_k: int | None = None,
        format: str = "pandas",
    ) -> Any:
        """Batch recommendations for multiple users.

        Uses the **Rust-accelerated** fast path when all models expose
        ``user_factors`` / ``item_factors`` and share the same user indexing.
        Falls back to the Python per-user loop otherwise.

        Parameters
        ----------
        user_ids : list or array, optional
            Users to score.  If None, uses all users from the first retriever.
        n : int, default=10
            Items per user.
        exclude_seen : bool, default=True
            Whether to exclude items users have already interacted with.
        retrieve_k : int, optional
            Candidates per retriever (default: ``n * 10``).
        format : str, default='pandas'
            Output format: ``'pandas'``, ``'polars'``, or ``'records'``.

        Returns
        -------
        DataFrame or list of dicts
            Columns: ``user_id``, ``item_ids``, ``scores``.
        """
        if user_ids is None:
            first = self.retrievers[0]
            n_users = getattr(first, "_n_users", None)
            if n_users is None:
                uf = getattr(first, "user_factors", None)
                if uf is not None:
                    n_users = uf.shape[0]  # type: ignore[union-attr]
            if n_users is None:
                raise ValueError("Cannot infer user_ids — pass them explicitly.")
            user_ids = list(range(n_users))

        k = retrieve_k or n * 10

        # ── Try Rust fast path ─────────────────────────────────────────────
        result = self._try_rust_batch(list(user_ids), n, k, exclude_seen)
        if result is not None:
            return self._format_batch_result(result, format)

        # ── Python fallback ────────────────────────────────────────────────
        records: list[dict[str, Any]] = []
        for uid in user_ids:
            ids, scores = self.recommend(uid, n=n, exclude_seen=exclude_seen, retrieve_k=retrieve_k)
            records.append(
                {
                    "user_id": uid,
                    "item_ids": ids.tolist(),
                    "scores": scores.tolist(),
                }
            )

        return self._format_batch_result(records, format)

    def _try_rust_batch(
        self,
        user_ids: list[Any],
        n: int,
        retrieve_k: int,
        exclude_seen: bool,
    ) -> list[dict[str, Any]] | None:
        """Attempt Rust-accelerated batch scoring.

        Returns None if models don't support the fast path.
        """
        import numpy as np

        # All models must expose user_factors/item_factors numpy arrays
        try:
            retriever_uf = []
            retriever_if = []
            n_items_list = []
            k_factors_list = []

            for model in self.retrievers:
                uf = model.user_factors
                itf = model.item_factors
                if not isinstance(uf, np.ndarray) or not isinstance(itf, np.ndarray):
                    return None
                retriever_uf.append(np.ascontiguousarray(uf, dtype=np.float32))
                retriever_if.append(np.ascontiguousarray(itf, dtype=np.float32))
                n_items_list.append(itf.shape[0])
                k_factors_list.append(itf.shape[1])

            n_users = retriever_uf[0].shape[0]

            # Build exclusion arrays (CSR format)
            if exclude_seen and hasattr(self.retrievers[0], "_fit_indptr"):
                indptr = np.asarray(self.retrievers[0]._fit_indptr, dtype=np.int64)
                indices = np.asarray(self.retrievers[0]._fit_indices, dtype=np.int32)
            else:
                indptr = np.zeros(n_users + 1, dtype=np.int64)
                indices = np.array([], dtype=np.int32)

            # Reranker factors (optional)
            reranker_uf = None
            reranker_if = None
            n_reranker_items = 0
            k_rerank = 0

            if self.reranker is not None:
                try:
                    ruf = self.reranker.user_factors
                    rif = self.reranker.item_factors
                    if isinstance(ruf, np.ndarray) and isinstance(rif, np.ndarray):
                        reranker_uf = np.ascontiguousarray(ruf, dtype=np.float32)
                        reranker_if = np.ascontiguousarray(rif, dtype=np.float32)
                        n_reranker_items = rif.shape[0]
                        k_rerank = rif.shape[1]
                except (AttributeError, NotImplementedError):
                    pass

            merge_map = {"max": 0, "sum": 1, "mean": 2}
            merge_code = merge_map.get(self.merge_strategy, 0)

            from . import _rusket as _rust  # type: ignore[attr-defined]

            all_uids, all_iids, all_scores = _rust.pipeline_batch_recommend(
                retriever_uf,
                retriever_if,
                n_users,
                n_items_list,
                k_factors_list,
                n,
                retrieve_k,
                merge_code,
                indptr,
                indices,
                reranker_uf,
                reranker_if,
                n_reranker_items,
                k_rerank,
            )

            # Reshape flat arrays into per-user records
            all_uids = np.asarray(all_uids)
            all_iids = np.asarray(all_iids)
            all_scores_arr = np.asarray(all_scores)

            # Filter to requested user_ids and group
            records: list[dict[str, Any]] = []
            requested = set(user_ids)

            # Group by user_id
            if len(all_uids) == 0:
                for uid in user_ids:
                    records.append({"user_id": uid, "item_ids": [], "scores": []})
                return records

            # Build per-user dict
            from collections import defaultdict

            user_results: dict[int, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
            for i in range(len(all_uids)):
                uid = int(all_uids[i])
                if uid in requested:
                    user_results[uid][0].append(int(all_iids[i]))
                    user_results[uid][1].append(float(all_scores_arr[i]))

            # Apply filter if present (post Rust, per user)
            for uid in user_ids:
                item_ids, scores = user_results.get(int(uid), ([], []))
                if self.filter_fn is not None and item_ids:
                    item_ids, scores = self.filter_fn(item_ids, scores)
                records.append(
                    {
                        "user_id": uid,
                        "item_ids": list(item_ids),
                        "scores": list(scores),
                    }
                )

            return records

        except Exception:
            return None

    @staticmethod
    def _format_batch_result(records: list[dict[str, Any]], format: str) -> Any:
        """Format batch results into the requested output type."""
        if format == "records":
            return records

        if format == "polars":
            from rusket._dependencies import import_optional_dependency

            pl = import_optional_dependency("polars")

            return pl.DataFrame(records)

        from rusket._dependencies import import_optional_dependency

        pd = import_optional_dependency("pandas")

        return pd.DataFrame(records)

    # ── Internal stages ────────────────────────────────────────────────────

    def _retrieve(
        self,
        user_id: int | Any,
        k: int,
        exclude_seen: bool,
    ) -> dict[int, float]:
        """Merge candidates from all retrievers.

        Returns a dict of ``{item_id: merged_score}``.
        """
        merged: dict[int, list[float]] = {}

        for model in self.retrievers:
            try:
                ids, scores = model.recommend_items(user_id=user_id, n=k, exclude_seen=exclude_seen)
            except Exception:
                continue

            for item_id, score in zip(ids, scores, strict=False):
                item_key = int(item_id)
                if item_key not in merged:
                    merged[item_key] = []
                merged[item_key].append(float(score))

        # Aggregate per merge strategy
        result: dict[int, float] = {}
        for item_id, score_list in merged.items():
            if self.merge_strategy == "max":
                result[item_id] = max(score_list)
            elif self.merge_strategy == "sum":
                result[item_id] = sum(score_list)
            elif self.merge_strategy == "mean":
                result[item_id] = sum(score_list) / len(score_list)
            else:
                result[item_id] = max(score_list)

        return result

    def _rerank_for_user(
        self,
        user_id: int | Any,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> np.ndarray:
        """Re-score candidates using the reranker's user-item dot product."""
        import numpy as np

        reranker = self.reranker
        if reranker is None:
            return candidate_scores

        if not (hasattr(reranker, "item_factors") and hasattr(reranker, "user_factors")):
            return candidate_scores

        item_factors = reranker.item_factors
        user_factors = reranker.user_factors
        n_users = user_factors.shape[0]
        n_items = item_factors.shape[0]

        # Resolve user index in the reranker's space
        uid = int(user_id)
        if uid >= n_users:
            return candidate_scores

        user_vec = user_factors[uid]
        valid_mask = candidate_ids < n_items

        new_scores = np.full_like(candidate_scores, -np.inf)
        valid_ids = candidate_ids[valid_mask]
        if len(valid_ids) > 0:
            new_scores[valid_mask] = item_factors[valid_ids] @ user_vec

        return new_scores

    def _filter(
        self,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the user-supplied filter function."""
        import numpy as np

        if self.filter_fn is None:
            return candidate_ids, candidate_scores

        filtered_ids, filtered_scores = self.filter_fn(candidate_ids.tolist(), [float(s) for s in candidate_scores])
        return np.array(filtered_ids, dtype=np.int64), np.array(filtered_scores, dtype=np.float64)
