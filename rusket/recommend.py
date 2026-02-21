"""High-level Business Recommender Workflows."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .als import ALS


class Recommender:
    """Hybrid recommender combining ALS collaborative filtering and association rules."""

    def __init__(self, als_model: ALS | None = None, rules_df: pd.DataFrame | None = None):
        self.als_model = als_model
        self.rules_df = rules_df

    def recommend_for_user(self, user_id: int, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Top-N recommendations for a user via ALS."""
        if self.als_model is None:
            raise ValueError("ALS model is not provided to the Recommender.")
        return self.als_model.recommend_items(user_id=user_id, n=n, exclude_seen=True)

    def predict_next_chunk(
        self, user_history_df: pd.DataFrame, user_col: str = "user_id", k: int = 5
    ) -> pd.DataFrame:
        """Batch-rank the next best products for every user in *user_history_df*."""
        recs = [
            {user_col: u, "recommended_items": self.recommend_for_user(int(u), n=k)[0].tolist()}
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
        ].sort_values(by=["lift", "confidence"], ascending=[False, False])

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
