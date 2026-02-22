from rusket.hupm import hupm


def test_hupm_basic():
    """Test High-Utility Pattern Mining on a toy dataset."""
    # Transactions:
    # T1: A (profit 5), B (profit 2), C (profit 1)  -> TU = 8
    # T2: A (profit 5), C (profit 1)                -> TU = 6
    # T3: B (profit 2), C (profit 1)                -> TU = 3
    #
    # Let's say:
    # Item A = 1
    # Item B = 2
    # Item C = 3

    transactions = [[1, 2, 3], [1, 3], [2, 3]]

    utilities = [[5.0, 2.0, 1.0], [5.0, 1.0], [2.0, 1.0]]

    # Utilities:
    # A = 5 + 5 = 10
    # B = 2 + 2 = 4
    # C = 1 + 1 + 1 = 3
    # {A, C} = (5+1) + (5+1) = 12
    # {A, B} = (5+2) = 7
    # {B, C} = (2+1) + (2+1) = 6
    # {A, B, C} = 5+2+1 = 8

    # Minimum utility of 7
    df = hupm(transactions, utilities, min_utility=7.0)

    # Convert itemsets to sets for easy matching
    df["itemset_set"] = df["itemset"].apply(set)

    # Should find {A, C} (12), {A} (10), {A, B, C} (8), {A, B} (7)
    assert len(df) == 4

    # Check specific utilities
    ac_row = df[df["itemset_set"] == {1, 3}].iloc[0]
    assert ac_row["utility"] == 12.0

    a_row = df[df["itemset_set"] == {1}].iloc[0]
    assert a_row["utility"] == 10.0

    abc_row = df[df["itemset_set"] == {1, 2, 3}].iloc[0]
    assert abc_row["utility"] == 8.0


def test_mine_hupm_pandas():
    """Test the DataFrame wrapper for High-Utility Pattern Mining using Pandas."""
    import pandas as pd

    from rusket.hupm import mine_hupm

    # Same toy dataset as above
    data = pd.DataFrame(
        {"txn": [1, 1, 1, 2, 2, 3, 3], "item": [1, 2, 3, 1, 3, 2, 3], "util": [5.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0]}
    )

    df = mine_hupm(data, transaction_col="txn", item_col="item", utility_col="util", min_utility=7.0)

    df["itemset_set"] = df["itemset"].apply(set)
    assert len(df) == 4

    ac_row = df[df["itemset_set"] == {1, 3}].iloc[0]
    assert ac_row["utility"] == 12.0


def test_mine_hupm_polars():
    """Test the Polars DataFrame wrapper if polars is installed."""
    import pytest

    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    from rusket.hupm import mine_hupm

    data = pl.DataFrame(
        {"txn": [1, 1, 1, 2, 2, 3, 3], "item": [1, 2, 3, 1, 3, 2, 3], "util": [5.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0]}
    )

    df = mine_hupm(data, transaction_col="txn", item_col="item", utility_col="util", min_utility=7.0)

    df["itemset_set"] = df["itemset"].apply(set)
    assert len(df) == 4

    ac_row = df[df["itemset_set"] == {1, 3}].iloc[0]
    assert ac_row["utility"] == 12.0
