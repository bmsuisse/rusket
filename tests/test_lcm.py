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


def test_lcm_mine() -> None:
    import pandas as pd
    dense_df = pd.DataFrame(
        {
            "1": [True, True, True, True],
            "2": [True, True, True, True],
            "3": [True, True, False, False],
            "4": [False, False, False, True],
        }
    )
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
    import pandas as pd
    dense_df = pd.DataFrame(
        {
            "1": [1, 1, 1, 1],
            "2": [1, 1, 1, 1],
            "3": [1, 1, 0, 0],
            "4": [0, 0, 0, 1],
        }
    )
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
