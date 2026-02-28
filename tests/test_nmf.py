import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from rusket import NMF


def test_nmf_fit_sparse() -> None:
    # 3 users, 3 items
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 2]
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    model = NMF(factors=4, iterations=50, seed=42)
    model.fit(X)

    assert model.fitted
    assert model.user_factors.shape == (3, 4)
    assert model.item_factors.shape == (3, 4)

    # All factors should be non-negative
    assert np.all(model.user_factors >= 0)
    assert np.all(model.item_factors >= 0)


def test_nmf_recommend_items() -> None:
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 2]
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    model = NMF(factors=4, iterations=80, seed=42).fit(X)

    # U0 has seen I0, I1 → should recommend I2
    items, scores = model.recommend_items(user_id=0, n=1, exclude_seen=True)
    assert len(items) == 1
    assert items[0] == 2

    # Include seen → 3 items
    items_all, scores_all = model.recommend_items(user_id=0, n=3, exclude_seen=False)
    assert len(items_all) == 3


def test_nmf_from_transactions() -> None:
    df = pd.DataFrame(
        {
            "user": [0, 0, 1, 1, 2, 2],
            "item": ["A", "B", "B", "C", "A", "C"],
        }
    )
    model = NMF.from_transactions(df, user_col="user", item_col="item", factors=4, iterations=50).fit()

    assert model.fitted
    assert model._n_users == 3
    assert model._n_items == 3

    items, scores = model.recommend_items(user_id=0, n=2)
    assert len(items) > 0


def test_nmf_unfitted_raises() -> None:
    model = NMF()
    with pytest.raises(RuntimeError):
        model.recommend_items(0)


def test_nmf_already_fitted_raises() -> None:
    X = sp.csr_matrix(np.ones((2, 2)))
    model = NMF(factors=2, iterations=5).fit(X)
    with pytest.raises(RuntimeError):
        model.fit(X)


def test_nmf_user_out_of_bounds() -> None:
    X = sp.csr_matrix(np.ones((2, 3)))
    model = NMF(factors=2, iterations=5).fit(X)
    with pytest.raises(ValueError):
        model.recommend_items(5)


def test_nmf_factors_nonnegative() -> None:
    """Verify that factors stay non-negative after fitting on a larger dataset."""
    rng = np.random.RandomState(123)
    X = sp.random(50, 30, density=0.1, format="csr", random_state=rng)
    X.data[:] = np.abs(X.data)  # ensure non-negative input

    model = NMF(factors=8, iterations=100, seed=123).fit(X)

    assert np.all(model.user_factors >= 0), "User factors contain negative values"
    assert np.all(model.item_factors >= 0), "Item factors contain negative values"
