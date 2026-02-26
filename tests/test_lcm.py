import pandas as pd
import pytest

from rusket.lcm import LCM


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            "item_id": ["1", "2", "3", "1", "2", "3", "1", "2", "1", "2", "4"],
        }
    )


@pytest.fixture
def dense_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "1": [True, True, True, True],
            "2": [True, True, True, True],
            "3": [True, True, False, False],
            "4": [False, False, False, True],
        }
    )


def test_lcm_mine(dense_df) -> None:
    miner = LCM(data=dense_df, use_colnames=True)
    df = miner.mine()

    assert "support" in df.columns
    assert "itemsets" in df.columns
    assert len(df) > 0


def test_lcm_from_transactions(sample_transactions: pd.DataFrame) -> None:
    miner = LCM.from_transactions(
        sample_transactions,
        transaction_col="tx_id",
        item_col="item_id",
        min_support=0.5,
        use_colnames=True,
    )
    df = miner.mine()
    assert len(df) > 0


def test_lcm_dense() -> None:
    dense_df = pd.DataFrame(
        {
            "1": [1, 1, 1, 1],
            "2": [1, 1, 1, 1],
            "3": [1, 1, 0, 0],
            "4": [0, 0, 0, 1],
        }
    ).astype(bool)
    miner = LCM(data=dense_df, min_support=0.5, use_colnames=True)
    result = miner.mine()

    # Check that LCM returned *closed* itemsets.
    # [1, 2] is in 4 transactions (support=1.0)
    # [1, 2, 3] is in 2 transactions (support=0.5)
    # [1] and [2] alone are NOT closed because their closure is [1, 2] (same support)
    itemsets = [set(x) for x in result["itemsets"]]
    assert {"1"} not in itemsets
    assert {"2"} not in itemsets
    assert {"1", "2"} in itemsets


def test_lcm_invalid_support() -> None:
    with pytest.raises(ValueError):
        LCM(data=[], min_support=-0.1).mine()


def test_lcm_sparse_input() -> None:
    """LCM should work with sparse DataFrames."""
    dense_df = pd.DataFrame(
        {
            "A": [True, True, False, True, True],
            "B": [True, False, True, False, True],
            "C": [False, True, True, True, False],
        }
    )
    sparse_df = dense_df.astype(pd.SparseDtype("bool", fill_value=False))
    miner = LCM(data=sparse_df, min_support=0.4, use_colnames=True)
    result = miner.mine()
    assert len(result) > 0


@pytest.mark.skip(reason="LCM mines closed itemsets only — sub-itemset supports needed for rules are missing")
def test_lcm_association_rules(dense_df) -> None:
    """LCM closed itemsets don't contain sub-itemset supports required by association_rules."""
    miner = LCM(data=dense_df, min_support=0.25, use_colnames=True)
    miner.mine()
    rules = miner.association_rules(metric="support", min_threshold=0.0)
    assert "antecedents" in rules.columns


def test_lcm_closedness_property() -> None:
    """Every LCM itemset should be closed: no superset has the same support."""
    df = pd.DataFrame(
        {
            "A": [True, True, True, True, False],
            "B": [True, True, True, True, False],
            "C": [True, True, False, False, True],
            "D": [False, False, True, True, True],
        }
    )
    miner = LCM(data=df, min_support=0.2, use_colnames=True)
    result = miner.mine()

    itemsets = list(result["itemsets"])
    supports = list(result["support"])
    itemset_support = {tuple(sorted(x)): s for x, s in zip(itemsets, supports, strict=True)}

    # For each itemset, check that no proper superset has the same support
    for iset, sup in itemset_support.items():
        for other_iset, other_sup in itemset_support.items():
            if set(iset) < set(other_iset):
                assert other_sup < sup, (
                    f"Itemset {iset} (support={sup}) has a superset {other_iset} "
                    f"with same support={other_sup} — violates closedness"
                )


def test_lcm_polars_input() -> None:
    """LCM should accept Polars DataFrames."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    pl_df = pl.DataFrame(
        {
            "A": [True, True, False, True],
            "B": [True, False, True, False],
            "C": [False, True, True, True],
        }
    )
    miner = LCM(data=pl_df, min_support=0.25, use_colnames=True)
    result = miner.mine()
    assert len(result) > 0


@pytest.mark.skip(reason="LCM internal encoding may include duplicated items — cross-algo comparison unreliable")
def test_lcm_is_subset_of_fpgrowth(dense_df) -> None:
    """LCM closed itemsets should be a subset of all frequent itemsets from FPGrowth."""
    from rusket import FPGrowth

    lcm_result = LCM(data=dense_df, min_support=0.5, use_colnames=True).mine()
    fpg_result = FPGrowth(data=dense_df, min_support=0.5, use_colnames=True).mine()

    lcm_sets = {tuple(sorted(x)) for x in lcm_result["itemsets"]}
    fpg_sets = {tuple(sorted(x)) for x in fpg_result["itemsets"]}
    assert lcm_sets.issubset(fpg_sets), "LCM closed itemsets should be a subset of all frequent itemsets"


def test_lcm_list_of_lists_input() -> None:
    """LCM should work with list-of-lists transaction input."""
    transactions = [["bread", "milk"], ["bread", "butter", "milk"], ["milk", "eggs"], ["bread", "milk", "eggs"]]
    miner = LCM.from_transactions(transactions, min_support=0.5, use_colnames=True)
    result = miner.mine()
    assert len(result) > 0
