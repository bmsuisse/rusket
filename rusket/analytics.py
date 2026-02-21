"""Enterprise Analytics & Reporting."""

from __future__ import annotations
import pandas as pd


def find_substitutes(rules_df: pd.DataFrame, max_lift: float = 0.8) -> pd.DataFrame:
    """Substitute/cannibalizing products via negative association rules.

    Items with high individual support but low co-occurrence (lift < 1.0)
    likely cannibalize each other.

    Parameters
    ----------
    rules_df
        DataFrame output from ``rusket.association_rules``.
    max_lift
        Upper bound for lift; lift < 1.0 implies negative correlation.

    Returns
    -------
    pd.DataFrame sorted ascending by lift (most severe cannibalization first).
    """
    substitutes = rules_df[
        (rules_df["antecedents"].apply(len) == 1)
        & (rules_df["consequents"].apply(len) == 1)
        & (rules_df["lift"] < max_lift)
    ].copy()
    return substitutes.sort_values(by=["lift", "confidence"]).reset_index(drop=True)  # type: ignore


def customer_saturation(
    df: pd.DataFrame,
    user_col: str,
    category_col: str | None = None,
    item_col: str | None = None,
) -> pd.DataFrame:
    """Customer saturation by unique items/categories bought, split into deciles.

    Parameters
    ----------
    df
        Interaction DataFrame.
    user_col
        Column identifying the user.
    category_col
        Category column (optional; at least one of category/item required).
    item_col
        Item column (optional).

    Returns
    -------
    pd.DataFrame with ``unique_count``, ``saturation_pct``, and ``decile`` columns.
    """
    if category_col is None and item_col is None:
        raise ValueError("Must provide either category_col or item_col.")

    target_col = category_col or item_col
    assert target_col is not None  # narrowing for type checker

    user_stats = df.groupby(user_col)[target_col].nunique().reset_index()
    user_stats.rename(columns={target_col: "unique_count"}, inplace=True)

    total_unique = df[target_col].nunique()
    user_stats["saturation_pct"] = user_stats["unique_count"] / total_unique

    user_stats["rank"] = user_stats["unique_count"].rank(method="first", ascending=False)
    user_stats["decile"] = pd.qcut(user_stats["rank"], q=10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    user_stats.drop(columns=["rank"], inplace=True)

    return user_stats.sort_values(by="saturation_pct", ascending=False).reset_index(drop=True)
