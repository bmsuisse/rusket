import numpy as np
import pandas as pd
import pytest

from rusket.fin import FIN


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
            "item_id": ["A", "B", "A", "C", "D", "B", "D", "E", "A", "C"],
        }
    )


@pytest.fixture
def dense_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [True, True, False, True],
            "B": [True, False, True, False],
            "C": [False, True, True, True],
        }
    )


def test_fin_mine(dense_df) -> None:
    miner = FIN(data=dense_df, use_colnames=True)
    df = miner.mine()

    assert "support" in df.columns
    assert "itemsets" in df.columns
    assert len(df) > 0


def test_fin_from_transactions(sample_transactions: pd.DataFrame) -> None:
    miner = FIN.from_transactions(
        sample_transactions,
        transaction_col="tx_id",
        item_col="item_id",
        min_support=0.5,
        use_colnames=True,
    )
    df = miner.mine()
    assert len(df) > 0


def test_fin_dense() -> None:
    dense_df = pd.DataFrame(
        {
            "A": [1, 1, 0, 1],
            "B": [1, 0, 1, 0],
            "C": [0, 1, 1, 1],
        }
    ).astype(bool)
    miner = FIN(data=dense_df, min_support=0.5, use_colnames=True)
    result = miner.mine()
    assert len(result) > 0


def test_fin_invalid_support() -> None:
    with pytest.raises(ValueError):
        FIN(data=[], min_support=-0.1).mine()


def test_fin_sparse_input() -> None:
    """FIN should work with sparse DataFrames."""
    dense_df = pd.DataFrame(
        {
            "A": [True, True, False, True, True],
            "B": [True, False, True, False, True],
            "C": [False, True, True, True, False],
            "D": [False, False, False, True, True],
        }
    )
    sparse_df = dense_df.astype(pd.SparseDtype("bool", fill_value=False))
    miner = FIN(data=sparse_df, min_support=0.4, use_colnames=True)
    result = miner.mine()
    assert len(result) > 0
    assert "support" in result.columns


def test_fin_association_rules(dense_df) -> None:
    """FIN should support association rule generation via RuleMinerMixin."""
    miner = FIN(data=dense_df, min_support=0.25, use_colnames=True)
    miner.mine()
    rules = miner.association_rules(metric="confidence", min_threshold=0.5)
    assert len(rules) > 0
    assert "antecedents" in rules.columns
    assert "consequents" in rules.columns
    assert "confidence" in rules.columns


def test_fin_recommend_for_cart(dense_df) -> None:
    """FIN recommend_for_cart should suggest items based on rules."""
    miner = FIN(data=dense_df, min_support=0.25, use_colnames=True)
    miner.mine()
    suggestions = miner.recommend_for_cart(["A"], n=2)
    assert isinstance(suggestions, list)


def test_fin_max_len(dense_df) -> None:
    """max_len should limit the maximum itemset size."""
    miner_no_limit = FIN(data=dense_df, min_support=0.25, use_colnames=True)
    result_no_limit = miner_no_limit.mine()

    miner_limited = FIN(data=dense_df, min_support=0.25, max_len=1, use_colnames=True)
    result_limited = miner_limited.mine()

    max_size = max(len(x) for x in result_limited["itemsets"])
    assert max_size <= 1
    assert len(result_limited) <= len(result_no_limit)


def test_fin_polars_input() -> None:
    """FIN should accept Polars DataFrames."""
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
    miner = FIN(data=pl_df, min_support=0.25, use_colnames=True)
    result = miner.mine()
    assert len(result) > 0


def test_fin_results_match_fpgrowth(dense_df) -> None:
    """FIN should produce the same itemsets as FPGrowth for the same data."""
    from rusket import FPGrowth

    fin_result = FIN(data=dense_df, min_support=0.5, use_colnames=True).mine()
    fpg_result = FPGrowth(data=dense_df, min_support=0.5, use_colnames=True).mine()

    fin_sets = {tuple(sorted(x)) for x in fin_result["itemsets"]}
    fpg_sets = {tuple(sorted(x)) for x in fpg_result["itemsets"]}
    assert fin_sets == fpg_sets
