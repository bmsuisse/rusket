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
    import pandas as pd

    data = []
    for tx_id, (txn, util) in enumerate(zip(transactions, utilities, strict=False)):
        for item, u in zip(txn, util, strict=False):
            data.append({"txn": tx_id, "item": item, "util": u})
    df_raw = pd.DataFrame(data)

    from rusket.hupm import HUPM

    df = HUPM.from_transactions(
        df_raw, transaction_col="txn", item_col="item", utility_col="util", min_utility=7.0
    ).mine()

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

    # Same toy dataset as above
    data = pd.DataFrame(
        {"txn": [1, 1, 1, 2, 2, 3, 3], "item": [1, 2, 3, 1, 3, 2, 3], "util": [5.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0]}
    )

    from rusket.hupm import HUPM

    df = HUPM.from_transactions(
        data, transaction_col="txn", item_col="item", utility_col="util", min_utility=7.0
    ).mine()

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

    data = pl.DataFrame(
        {"txn": [1, 1, 1, 2, 2, 3, 3], "item": [1, 2, 3, 1, 3, 2, 3], "util": [5.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0]}
    )

    from rusket.hupm import HUPM

    df = HUPM.from_transactions(
        data, transaction_col="txn", item_col="item", utility_col="util", min_utility=7.0
    ).mine()

    df["itemset_set"] = df["itemset"].apply(set)
    assert len(df) == 4

    ac_row = df[df["itemset_set"] == {1, 3}].iloc[0]
    assert ac_row["utility"] == 12.0


import hypothesis.strategies as st  # noqa: E402
import pandas as pd  # noqa: E402
from hypothesis import given, settings  # noqa: E402


@given(
    transactions_with_utils=st.lists(
        st.dictionaries(
            keys=st.integers(min_value=1, max_value=50),
            values=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=20,
    ),
    min_utility=st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_hupm_invariants(transactions_with_utils, min_utility):
    from rusket.hupm import HUPM

    data = []
    for tx_id, item_util_map in enumerate(transactions_with_utils):
        for item, util in item_util_map.items():
            data.append({"txn": tx_id, "item": item, "util": util})

    df_raw = pd.DataFrame(data)

    df = HUPM.from_transactions(
        df_raw, transaction_col="txn", item_col="item", utility_col="util", min_utility=min_utility
    ).mine()

    if not df.empty:
        assert (df["utility"] >= (min_utility - 1e-5)).all()
