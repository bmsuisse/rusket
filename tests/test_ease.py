import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from rusket import EASE


def test_ease_fit_sparse() -> None:
    # A tiny predictable setup:
    # U0 buys I0, I1
    # U1 buys I1, I2
    # U2 buys I0, I2
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 2]
    data = np.ones(len(rows), dtype=np.float32)

    X = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    model = EASE(regularization=1.0)
    model.fit(X)

    assert model.item_weights is not None
    assert model.item_weights.shape == (3, 3)
    # The diagonal of B must be zero
    np.testing.assert_array_equal(np.diag(model.item_weights), np.zeros(3))

    # Test recommendation for U0 (history: I0, I1)
    # We should recommend I2
    items, scores = model.recommend_items(user_id=0, n=1, exclude_seen=True)
    assert len(items) == 1
    assert items[0] == 2

    # Include seen
    items_all, scores_all = model.recommend_items(user_id=0, n=3, exclude_seen=False)
    assert len(items_all) == 3


def test_ease_from_transactions() -> None:
    # Create simple dataframe
    df = pd.DataFrame(
        {
            "user": [0, 0, 1, 1, 2, 2],
            "item": ["A", "B", "B", "C", "A", "C"],
            "rating": [1, 1, 1, 1, 1, 1],
        }
    )

    model = EASE.from_transactions(
        df,
        user_col="user",
        item_col="item",
        rating_col="rating",
        regularization=100.0,
    )

    assert model.fitted
    assert model._n_users == 3
    assert model._n_items == 3

    items, _ = model.recommend_items(user_id=0, n=2)
    # user 0 bought A (mapped to 0) and B (mapped to 1) -> recommend C (mapped to 2)
    assert len(items) > 0


def test_ease_fit_already_fitted() -> None:
    X = sp.csr_matrix(np.ones((2, 2)))
    model = EASE()
    model.fit(X)

    with pytest.raises(RuntimeError):
        model.fit(X)

def test_ease_recommend_unfitted() -> None:
    model = EASE()
    with pytest.raises(RuntimeError):
        model.recommend_items(0)

def test_ease_recommend_users_not_implemented() -> None:
    X = sp.csr_matrix(np.ones((2, 2)))
    model = EASE().fit(X)
    with pytest.raises(NotImplementedError):
        model.recommend_users(0)
