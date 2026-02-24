import pandas as pd
import pytest

import rusket


@pytest.fixture
def recbole_inter() -> pd.DataFrame:
    """Loads the standard RecBole test.inter mock dataset."""
    df = pd.read_csv("tests/test_data/test.inter", sep="\t")
    # Columns are user_id:token, item_id:token, rating:float, timestamp:float
    df.columns = [col.split(":")[0] for col in df.columns]
    return df


def test_recbole_itemknn(recbole_inter: pd.DataFrame) -> None:
    model = rusket.ItemKNN.from_transactions(
        recbole_inter, user_col="user_id", item_col="item_id", method="bm25", k=20
    ).fit()
    ids, scores = model.recommend_items(user_id=1, n=10)
    assert len(ids) > 0


def test_recbole_ease(recbole_inter: pd.DataFrame) -> None:
    model = rusket.EASE.from_transactions(recbole_inter, user_col="user_id", item_col="item_id", l2_reg=100.0).fit()
    ids, scores = model.recommend_items(user_id=1, n=10)
    assert len(ids) > 0


def test_recbole_als(recbole_inter: pd.DataFrame) -> None:
    model = rusket.ALS.from_transactions(
        recbole_inter, user_col="user_id", item_col="item_id", factors=32, iterations=5
    ).fit()
    ids, scores = model.recommend_items(user_id=1, n=10)
    assert len(ids) > 0


def test_recbole_bpr(recbole_inter: pd.DataFrame) -> None:
    model = rusket.BPR.from_transactions(
        recbole_inter, user_col="user_id", item_col="item_id", factors=32, iterations=5
    ).fit()
    ids, scores = model.recommend_items(user_id=1, n=10)
    assert len(ids) > 0


def test_recbole_fpmc(recbole_inter: pd.DataFrame) -> None:
    # FPMC needs sequential data grouped by user
    recbole_inter = recbole_inter.sort_values(by=["user_id", "timestamp"])

    users = recbole_inter["user_id"]
    items = recbole_inter["item_id"]

    user_codes, _ = pd.factorize(users)
    item_codes, _ = pd.factorize(items)

    sequences = [[] for _ in range(max(user_codes) + 1)]
    for u, i in zip(user_codes, item_codes, strict=False):
        if not pd.isna(u) and not pd.isna(i):
            sequences[int(u)].append(int(i))

    # filter out users with < 2 items
    sequences = [s for s in sequences if len(s) >= 2]

    model = rusket.FPMC(factors=16, iterations=5)
    model.fit(sequences)

    ids, scores = model.recommend_items(user_id=0, n=10)
    assert len(ids) > 0
