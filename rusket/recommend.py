"""High-level Business Recommender Workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .als import ALS


class Recommender:
    """Hybrid recommender combining ALS collaborative filtering, semantic similarities, and association rules."""

    def __init__(
        self,
        als_model: ALS | None = None,
        rules_df: pd.DataFrame | None = None,
        item_embeddings: np.ndarray | None = None,
    ):
        self.als_model = als_model
        self.rules_df = rules_df
        self.item_embeddings = item_embeddings

        if self.item_embeddings is not None and self.als_model is not None:
            if self.item_embeddings.shape[0] != self.als_model.item_factors.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: ALS model has {self.als_model.item_factors.shape[0]} items, "
                    f"but embeddings matrix has {self.item_embeddings.shape[0]} items."
                )

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 5,
        alpha: float = 0.5,
        target_item_for_semantic: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-N recommendations for a user via Hybrid ALS + Semantic.

        Parameters
        ----------
        user_id : int
            The user ID to generate recommendations for.
        n : int, default=5
            Number of items to return.
        alpha : float, default=0.5
            Weight blending CF vs Semantic.
            ``alpha=1.0`` is pure CF. ``alpha=0.0`` is pure semantic.
        target_item_for_semantic : int | None, default=None
            If provided, semantic similarity is computed against this item.
            If None, and alpha < 1.0, it computes semantic similarity against
            the user's most recently interacted item (if history is available)
            or falls back to pure CF.
        """
        if self.als_model is None:
            raise ValueError("ALS model is not provided to the Recommender.")

        # 1. Get raw CF scores
        cf_rec_tuple = self.als_model.recommend_items(user_id=user_id, n=n, exclude_seen=True)

        if self.item_embeddings is None or alpha == 1.0:
            return cf_rec_tuple

        # 2. Hybrid scoring
        # Calculate full CF scores for all items
        u_factors = self.als_model.user_factors[user_id]
        cf_raw_scores = np.dot(self.als_model.item_factors, u_factors)

        target_item = target_item_for_semantic
        if target_item is None:
            try:
                # Naive fallback: try to find an item the user bought
                if self.als_model._fit_indptr is not None and self.als_model._fit_indices is not None:
                    start_idx = self.als_model._fit_indptr[user_id]
                    end_idx = self.als_model._fit_indptr[user_id + 1]
                    if end_idx > start_idx:
                        target_item = self.als_model._fit_indices[end_idx - 1]
            except (AttributeError, IndexError):
                pass

        if target_item is None:
            # Cannot do semantic without a target anchor. Fallback to CF.
            return cf_rec_tuple

        # Compute cosine similarity
        target_emb = self.item_embeddings[target_item]
        norms = np.linalg.norm(self.item_embeddings, axis=1) * np.linalg.norm(target_emb)
        norms[norms == 0] = 1e-9  # prevent div zero
        semantic_raw_scores = np.dot(self.item_embeddings, target_emb) / norms

        # Normalize both to [0, 1] for blending
        def _minmax(arr: np.ndarray) -> np.ndarray:
            ptp = np.ptp(arr)
            return (arr - arr.min()) / ptp if ptp > 0 else arr

        cf_norm = _minmax(cf_raw_scores)
        sem_norm = _minmax(semantic_raw_scores)

        hybrid_scores = (alpha * cf_norm) + ((1.0 - alpha) * sem_norm)

        # Mask seen items
        try:
            if self.als_model._fit_indptr is not None and self.als_model._fit_indices is not None:
                start_idx = self.als_model._fit_indptr[user_id]
                end_idx = self.als_model._fit_indptr[user_id + 1]
                seen = self.als_model._fit_indices[start_idx:end_idx]
                hybrid_scores[seen] = -np.inf
        except (AttributeError, IndexError):
            pass

        # Get top N
        top_idx = np.argpartition(hybrid_scores, -n)[-n:]
        top_idx = top_idx[np.argsort(-hybrid_scores[top_idx])]
        top_scores = hybrid_scores[top_idx]

        return top_idx, top_scores

    def predict_next_chunk(self, user_history_df: pd.DataFrame, user_col: str = "user_id", k: int = 5) -> pd.DataFrame:
        """Batch-rank the next best products for every user in *user_history_df*."""
        recs = [
            {
                user_col: u,
                "recommended_items": self.recommend_for_user(int(u), n=k)[0].tolist(),
            }
            for u in user_history_df[user_col].unique()
        ]
        return pd.DataFrame(recs)

    def recommend_for_cart(self, cart_items: list[int], n: int = 5) -> list[int]:
        """Suggest items to add to an active cart using association rules."""
        if self.rules_df is None:
            raise ValueError("Association rules are not provided to the Recommender.")
        if self.rules_df.empty:
            return []

        cart_set = frozenset(cart_items)
        valid_rules = self.rules_df[
            self.rules_df["antecedents"].apply(lambda ant: frozenset(ant).issubset(cart_set))
        ].sort_values(by=["lift", "confidence"], ascending=False)  # type: ignore

        if valid_rules.empty:
            return []

        suggestions: list[int] = []
        for consequents in valid_rules["consequents"]:
            for item in consequents:
                if item not in cart_set and item not in suggestions:
                    suggestions.append(item)
                    if len(suggestions) >= n:
                        return suggestions
        return suggestions


NextBestAction = Recommender


def score_potential(
    user_history: list[list[int]],
    als_model: ALS,
    target_categories: list[int] | None = None,
) -> np.ndarray:
    """Cross-selling potential scores — shape ``(n_users, n_items)`` or ``(n_users, len(target_categories))``.

    Items the user has already interacted with are masked to ``-inf``.
    """
    u_factors = als_model.user_factors
    i_factors = als_model.item_factors

    if target_categories is not None:
        i_factors = i_factors[target_categories]

    potential_scores = np.dot(u_factors, i_factors.T)

    for user_id, history in enumerate(user_history):
        if user_id >= potential_scores.shape[0]:
            break
        for item in history:
            if target_categories is not None:
                if item in target_categories:
                    potential_scores[user_id, target_categories.index(item)] = -np.inf
            elif item < potential_scores.shape[1]:
                potential_scores[user_id, item] = -np.inf

    return potential_scores
