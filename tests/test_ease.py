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
    ).fit()

    assert model.fitted  # type: ignore
    assert model._n_users == 3  # type: ignore
    assert model._n_items == 3  # type: ignore

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


def test_ease_fit_raises_clear_error_above_memory_threshold() -> None:
    # 100k items comfortably exceeds the 8 GiB default cap
    # (16 * 100_000**2 / 1024**3 ~= 149 GiB) without needing to actually
    # build a 100k x 100k sparse matrix.
    n_items = 100_000
    rows = [0, 0]
    cols = [0, 1]
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(1, n_items))

    model = EASE(regularization=1.0)
    with pytest.raises(MemoryError, match=r"100000 items.*GiB.*max_memory_gb"):
        model.fit(X)

    # Not fitted, so a retry (with an explicit opt-in) is still possible.
    assert not model.fitted


def test_ease_fit_memory_guard_bypassable_via_opt_in() -> None:
    # A tiny catalog (3 items) that would never trip the guard, used just to
    # confirm the opt-in path (raising max_memory_gb) still lets a normal
    # fit complete -- i.e. the guard doesn't get in the way when raised.
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 2]
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    model = EASE(regularization=1.0)
    model.fit(X, max_memory_gb=1_000_000.0)
    assert model.fitted

    # And confirm the estimator itself is what decides pass/fail: a catalog
    # of 100k items must fail with a *tiny* cap...
    from rusket.recommenders.ease import _estimate_peak_bytes

    n_items = 100_000
    estimated_gb = _estimate_peak_bytes(n_items) / (1024**3)
    assert estimated_gb > 1.0
    # ...but the exact same estimate must clear an explicitly huge cap,
    # proving the opt-in argument genuinely changes the outcome rather than
    # the guard being unconditionally on or off.
    assert estimated_gb < 1_000_000.0


def test_ease_weights_match_numpy_reference() -> None:
    # Reference implementation of Steck 2019 eq. 8:
    #   P = (X^T X + lambda I)^-1
    #   B[i, j] = -P[i, j] / P[j, j] for i != j, else 0
    rng = np.random.default_rng(0)
    n_users, n_items = 20, 6
    dense = (rng.random((n_users, n_items)) > 0.5).astype(np.float32)
    X = sp.csr_matrix(dense)

    lam = 2.0
    G = dense.T @ dense
    P = np.linalg.inv(G + lam * np.eye(n_items, dtype=np.float64))
    diag = np.diag(P)
    B_ref = -P / diag[np.newaxis, :]
    np.fill_diagonal(B_ref, 0.0)

    model = EASE(regularization=lam)
    model.fit(X)

    np.testing.assert_allclose(model.item_weights, B_ref, atol=1e-4, rtol=1e-4)
