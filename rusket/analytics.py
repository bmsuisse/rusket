"""Enterprise Analytics & Reporting."""

from __future__ import annotations
import pandas as pd

def find_substitutes(rules_df: pd.DataFrame, max_lift: float = 0.8) -> pd.DataFrame:
    """Finds substitute or cannibalizing products using negative association rules.
    
    If Item A and Item B have high individual support but extremely low 
    co-occurrence (lift < 1.0), they likely cannibalize each other (substitutes).
    
    Args:
        rules_df: DataFrame output from `rusket.association_rules`.
        max_lift: The upper bound for lift to be considered a substitute combination.
                  Lift < 1.0 implies negative correlation (buying A makes you less likely to buy B).
                  
    Returns:
        pd.DataFrame sorted by most severe cannibalization (lowest lift).
    """
    # Filter for rules with single antecedents and consequents
    substitutes = rules_df[
        (rules_df["antecedents"].apply(len) == 1) & 
        (rules_df["consequents"].apply(len) == 1) &
        (rules_df["lift"] < max_lift)
    ].copy()
    
    return substitutes.sort_values(by=["lift", "confidence"], ascending=[True, True]).reset_index(drop=True)


def customer_saturation(
    df: pd.DataFrame, 
    user_col: str, 
    category_col: str | None = None,
    item_col: str | None = None
) -> pd.DataFrame:
    """Calculates customer saturation and groups users into deciles.
    
    Determines how deeply a customer has penetrated a category or catalog.
    "These 10% of users buy 80% of our products in this category."
    
    Args:
        df: The interaction DataFrame containing users and categories/items.
        user_col: Column name identifying the user.
        category_col: Column name for the category (optional).
        item_col: Column name for the items bought (optional).
                  At least one of `category_col` or `item_col` must be provided.
                  
    Returns:
        pd.DataFrame with columns measuring unique items/categories bought and decile assignment.
    """
    if category_col is None and item_col is None:
        raise ValueError("Must provide either category_col or item_col.")
        
    target_col = category_col if category_col else item_col
    assert target_col is not None
    
    # Calculate unique items/categories bought per user
    user_stats = df.groupby(user_col)[target_col].nunique().reset_index()
    user_stats.rename(columns={target_col: "unique_count"}, inplace=True)
    
    # Calculate total unique items/categories available
    total_unique = df[target_col].nunique()
    
    # Calculate Saturation %
    user_stats["saturation_pct"] = user_stats["unique_count"] / total_unique
    
    # Assign to deciles (1 is the top 10% most saturated customers, 10 is the bottom 10%)
    # Use qcut to divide into 10 quantiles based on unique_count
    # Since many users might have the same count, we use rank(method='first')
    user_stats["rank"] = user_stats["unique_count"].rank(method="first", ascending=False)
    user_stats["decile"] = pd.qcut(user_stats["rank"], q=10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    user_stats.drop(columns=["rank"], inplace=True)
    
    return user_stats.sort_values(by="saturation_pct", ascending=False).reset_index(drop=True)
