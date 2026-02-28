import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from rusket import PopularityRecommender


def test_popularity_fit_sparse() -> None:
    # U0 buys I0, I1
    # U1 buys I1, I2
    # U2 buys I0, I1, I2
    rows = [0, 0, 1, 1, 2, 2, 2]
    cols = [0, 1, 1, 2, 0, 1, 2]
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    model = PopularityRecommender()
    model.fit(X)

    assert model.fitted
    assert model._n_items == 3
    assert model._n_users == 3

    # I1 has 3 interactions (most popular), I0 has 2, I2 has 2
    pop = model.item_popularity
    assert pop[1] == 3.0
    assert pop[0] == 2.0
    assert pop[2] == 2.0


def test_popularity_recommend_exclude_seen() -> None:
    rows = [0, 0, 1, 1, 2, 2, 2]
    cols = [0, 1, 1, 2, 0, 1, 2]
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    model = PopularityRecommender().fit(X)

    # U0 has seen I0 and I1 → only I2 should be recommended
    items, scores = model.recommend_items(user_id=0, n=1, exclude_seen=True)
    assert len(items) == 1
    assert items[0] == 2

    # Include seen → all 3 items
    items_all, scores_all = model.recommend_items(user_id=0, n=3, exclude_seen=False)
    assert len(items_all) == 3
    # First should be I1 (most popular)
    assert items_all[0] == 1


def test_popularity_from_transactions() -> None:
    df = pd.DataFrame(
        {
            "user": [0, 0, 1, 1, 2, 2, 2],
            "item": ["A", "B", "B", "C", "A", "B", "C"],
        }
    )
    model = PopularityRecommender.from_transactions(df, user_col="user", item_col="item").fit()

    assert model.fitted
    assert model._n_users == 3
    assert model._n_items == 3

    items, scores = model.recommend_items(user_id=0, n=3, exclude_seen=False)
    assert len(items) == 3
    # Scores should be in descending order
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_popularity_unfitted_raises() -> None:
    model = PopularityRecommender()
    with pytest.raises(RuntimeError):
        model.recommend_items(0)


def test_popularity_already_fitted_raises() -> None:
    X = sp.csr_matrix(np.ones((2, 2)))
    model = PopularityRecommender().fit(X)
    with pytest.raises(RuntimeError):
        model.fit(X)


def test_popularity_user_out_of_bounds() -> None:
    X = sp.csr_matrix(np.ones((2, 3)))
    model = PopularityRecommender().fit(X)
    with pytest.raises(ValueError):
        model.recommend_items(5)
