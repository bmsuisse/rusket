"""High-level Business Recommender Workflows."""

from __future__ import annotations

import pandas as pd
import numpy as np
import typing
from typing import Any

from .als import ALS

class Recommender:
    """A high-level Recommender hybrid model combining ALS and Frequent Pattern Mining.
    
    This class wraps both the collaborative filtering (ALS) and association rules
    to provide the ultimate "Next Best Action" for a user or a shopping cart.
    """
    
    def __init__(self, als_model: ALS | None = None, rules_df: pd.DataFrame | None = None):
        """Initialize the Recommender.
        
        Args:
            als_model: A fitted `rusket.ALS` model (provides personalized recommendations).
            rules_df: A DataFrame of association rules from `rusket.association_rules` 
                      (provides strict "bought X -> usually buys Y" rules).
        """
        self.als_model = als_model
        self.rules_df = rules_df
        
    def recommend_for_user(self, user_id: int, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Provides dynamic top-N recommendations for a specific user using ALS.
        
        Args:
            user_id: The integer index of the user.
            n: Number of recommendations to generate.
            
        Returns:
            Tuple of `(item_ids, scores)`.
        """
        if self.als_model is None:
            raise ValueError("ALS model is not provided to the Recommender.")
            
        return self.als_model.recommend_items(user_id=user_id, n=n, exclude_seen=True)
        
    def predict_next_chunk(self, user_history_df: pd.DataFrame, user_col: str = "user_id", k: int = 5) -> pd.DataFrame:
        """Batch ranks the next best products for a chunk of users.
        
        Args:
            user_history_df: DataFrame containing the users to predict for.
            k: Number of recommendations per user.
            
        Returns:
            A DataFrame with `user_id` and `recommended_items`.
        """
        unique_users = user_history_df[user_col].unique()
        recs = []
        for u in unique_users:
            items, _ = self.recommend_for_user(user_id=int(u), n=k)
            recs.append({user_col: u, "recommended_items": items.tolist()})
            
        return pd.DataFrame(recs)
        
    def recommend_for_cart(self, cart_items: list[int], n: int = 5) -> list[int]:
        """Recommends products to add to an active shopping cart using association rules.
        
        This mimics a "Frequently bought together" carousel by searching the 
        association rules for antecedents that match the cart contents.
        
        Args:
            cart_items: A list of item indices currently in the cart.
            n: Maximum number of cross-sell item recommendations to return.
            
        Returns:
            List of suggested item indices, sorted by lift/confidence.
        """
        if self.rules_df is None:
            raise ValueError("Association rules are not provided to the Recommender.")
            
        if self.rules_df.empty:
            return []
            
        cart_set = frozenset(cart_items)
        
        # Find rules where antecedents are a subset of the cart
        valid_rules = self.rules_df[
            self.rules_df["antecedents"].apply(lambda ant: iter(ant).__next__() in cart_set if len(ant) == 1 else frozenset(ant).issubset(cart_set))
        ]
        
        if valid_rules.empty:
            return []
            
        # Prioritize high lift and confidence
        valid_rules = typing.cast(Any, valid_rules).sort_values(
            by=["lift", "confidence"], ascending=[False, False]
        )
        
        suggestions: list[int] = []
        for consequents in valid_rules["consequents"]:
            for item in consequents:
                # Type mapping, assuming consequents contains ints or strings mapped back
                if item not in cart_items and item not in suggestions:
                    suggestions.append(item)
                    if len(suggestions) >= n:
                        return suggestions
                        
        return suggestions

NextBestAction = Recommender

def score_potential(
    user_history: list[list[int]], 
    als_model: ALS,
    target_categories: list[int] | None = None
) -> np.ndarray:
    """Calculates cross-selling potential scores for users.
    
    Generates a score for every user indicating the probability they 
    should have interacted with specific target categories/items by now but haven't.
    
    Args:
        user_history: A list where each element is a list of item IDs the user has bought.
        als_model: A fitted `rusket.ALS` model.
        target_categories: Optional. Only score potential for these specific item IDs.
        
    Returns:
        np.ndarray: Matrix of potential scores of shape (n_users, n_items).
                    If `target_categories` is provided, shape is (n_users, len(target_categories)).
    """
    if als_model.user_factors is None or als_model.item_factors is None:
        raise ValueError("Model has not been fitted yet.")
        
    u_factors = als_model.user_factors
    i_factors = als_model.item_factors
    
    if target_categories is not None:
        i_factors = i_factors[target_categories]
        
    # Raw ALS scores for all users and target items (Dot product U * I.T)
    potential_scores = np.dot(u_factors, i_factors.T)
    
    # Mask out items the user has already interacted with
    for user_id, history in enumerate(user_history):
        if user_id >= potential_scores.shape[0]:
            break
            
        for item in history:
            if target_categories is not None:
                if item in target_categories:
                    col_idx = target_categories.index(item)
                    potential_scores[user_id, col_idx] = -np.inf
            else:
                if item < potential_scores.shape[1]:
                    potential_scores[user_id, item] = -np.inf
                    
    return potential_scores
