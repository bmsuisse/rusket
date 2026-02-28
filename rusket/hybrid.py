"""Hybrid recommender that blends scores from multiple models."""

from __future__ import annotations

from typing import Any

import numpy as np


class HybridRecommender:
    """Weighted ensemble of multiple recommendation models.

    Blends the output of several pre-fitted models by combining their
    ``recommend_items`` scores with configurable weights.

    Parameters
    ----------
    models_and_weights : list[tuple[Any, float]]
        List of ``(model, weight)`` pairs.  Each model must implement
        ``recommend_items(user_id, n, exclude_seen) -> (ids, scores)``.

    Example
    -------
    >>> hybrid = HybridRecommender([
    ...     (als_model, 0.7),
    ...     (bpr_model, 0.3),
    ... ])
    >>> ids, scores = hybrid.recommend_items(user_id=42, n=10)
    """

    def __init__(self, models_and_weights: list[tuple[Any, float]]) -> None:
        if not models_and_weights:
            raise ValueError("At least one (model, weight) pair is required.")

        self.models_and_weights = models_and_weights

        # Normalise weights to sum to 1
        total_weight = sum(w for _, w in models_and_weights)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive.")
        self._normalised_weights = [w / total_weight for _, w in models_and_weights]

    def __repr__(self) -> str:
        models = ", ".join(
            f"{type(m).__name__}(w={w:.2f})"
            for (m, _), w in zip(self.models_and_weights, self._normalised_weights, strict=False)
        )
        return f"HybridRecommender([{models}])"

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend recommendations from all constituent models.

        For each model, requests a large candidate set (``n * 3``), maps item
        scores into a shared score vector, applies the weight, and returns the
        top-*n* from the blended result.

        Parameters
        ----------
        user_id : int
            Internal user index.
        n : int, default=10
            Number of items to return.
        exclude_seen : bool, default=True
            Whether to exclude items already seen.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, blended_scores)`` sorted by descending score.
        """
        # Determine the item space size from the first model that exposes it
        n_items = 0
        for model, _ in self.models_and_weights:
            n_items_candidate = getattr(model, "_n_items", 0)
            if n_items_candidate > n_items:
                n_items = n_items_candidate

        if n_items == 0:
            raise RuntimeError("Cannot determine item space size from constituent models.")

        blended = np.zeros(n_items, dtype=np.float64)

        # Ask each model for a generous candidate pool
        pool_size = min(n * 3, n_items)

        for (model, _), weight in zip(self.models_and_weights, self._normalised_weights, strict=False):
            try:
                ids, scores = model.recommend_items(user_id, n=pool_size, exclude_seen=exclude_seen)
                ids = np.asarray(ids)
                scores = np.asarray(scores, dtype=np.float64)

                # Filter out -inf scores (items excluded by the model)
                valid = np.isfinite(scores)
                ids = ids[valid]
                scores = scores[valid]

                if len(scores) == 0:
                    continue

                # Min-max normalise scores per model to [0, 1]
                s_min, s_max = scores.min(), scores.max()
                if s_max > s_min:
                    scores = (scores - s_min) / (s_max - s_min)
                else:
                    scores = np.ones_like(scores)

                for item_id, score in zip(ids, scores, strict=False):
                    if 0 <= item_id < n_items:
                        blended[item_id] += weight * score
            except Exception:
                # If a model fails for a user, skip it gracefully
                continue

        top_n = np.argsort(blended)[::-1][:n]
        return top_n.astype(np.intp), blended[top_n].astype(np.float32)

    def fit(self) -> HybridRecommender:
        """No-op — constituent models must be pre-fitted."""
        return self
