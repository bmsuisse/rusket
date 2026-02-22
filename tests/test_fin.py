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


def test_fin_mine() -> None:
    import pandas as pd
    dense_df = pd.DataFrame(
        {
            "A": [True, True, False, True],
            "B": [True, False, True, False],
            "C": [False, True, True, True],
        }
    )
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
    import pandas as pd
    dense_df = pd.DataFrame(
        {
            "A": [1, 1, 0, 1],
            "B": [1, 0, 1, 0],
            "C": [0, 1, 1, 1],
        }
    )
    miner = FIN(data=dense_df, min_support=0.5, use_colnames=True)
    result = miner.mine()
    assert len(result) > 0

def test_fin_invalid_support() -> None:
    with pytest.raises(ValueError):
        FIN(data=[], min_support=-0.1).mine()
