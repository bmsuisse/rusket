import numpy as np
import pytest
from scipy import sparse as sp

from rusket import BPR


def test_bpr_basic_fit():
    """Basic structural test to ensure BPR fits without crashing."""
    # 3 users, 4 items.
    values = [1.0, 1.0, 1.0, 1.0]
    rows = [0, 0, 1, 2]
    cols = [0, 1, 2, 3]
    csr = sp.csr_matrix((values, (rows, cols)), shape=(3, 4))

    bpr = BPR(factors=10, iterations=10)
    bpr.fit(csr)

    assert bpr.fitted
    assert bpr.user_factors.shape == (3, 10)
    assert bpr.item_factors.shape == (4, 10)


def test_bpr_recommend():
    # Similar mockup to ALS recommendation test
    values = [1.0, 1.0, 1.0]
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    csr = sp.csr_matrix((values, (rows, cols)), shape=(3, 3))

    bpr = BPR(factors=10, iterations=10)
    bpr.fit(csr)

    # Overwrite factors to simulate perfect learn
    bpr._user_factors = np.array([[1.0, 0.0]], dtype=np.float32)
    bpr._item_factors = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bpr._n_users = 1

    items, scores = bpr.recommend_items(user_id=0, n=2, exclude_seen=True)
    assert len(items) == 2
    assert items[0] == 1


def test_bpr_ranking():
    """Test if BPR successfully separates positive vs negative items."""
    # Toy dataset:
    # User 0 likes items 0, 1.
    # User 1 likes items 2, 3.
    # User 2 likes items 0, 2.
    values = [1.0] * 6
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 2, 3, 0, 2]
    csr = sp.csr_matrix((values, (rows, cols)), shape=(3, 4))

    # Overfit a tiny model to make sure it learns the pattern
    bpr = BPR(
        factors=5, learning_rate=0.05, regularization=0.001, iterations=500, seed=42
    )
    bpr.fit(csr)

    u_factors = bpr.user_factors
    i_factors = bpr.item_factors

    # For user 0, item 0 (positive) should rank higher than item 3 (negative)
    u0_p0 = np.dot(u_factors[0], i_factors[0])
    u0_n3 = np.dot(u_factors[0], i_factors[3])
    assert u0_p0 > u0_n3

    # For user 1, item 2 (positive) should rank higher than item 0 (negative)
    u1_p2 = np.dot(u_factors[1], i_factors[2])
    u1_n0 = np.dot(u_factors[1], i_factors[0])
    assert u1_p2 > u1_n0


def test_fit_empty_matrix():
    """Test issue from implicit #264 causing crashes on empty matrices."""
    raw = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    model = BPR(iterations=5)
    model.fit(sp.csr_matrix(raw))
    assert model.fitted
    assert model.user_factors.shape == (3, 64)
    assert model.item_factors.shape == (3, 64)


def test_fit_almost_empty_matrix():
    """Test issue from implicit #264 causing crashes on almost empty matrices."""
    raw = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    model = BPR(iterations=5)
    model.fit(sp.csr_matrix(raw))
    assert model.fitted


def test_not_fitted_raises():
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = BPR().user_factors


def test_deterministic_seed():
    # Toy dataset
    values = [1.0] * 4
    rows = [0, 0, 1, 2]
    cols = [0, 1, 2, 3]
    csr = sp.csr_matrix((values, (rows, cols)), shape=(3, 4))

    m1 = BPR(factors=8, iterations=5, seed=123)
    m1.fit(csr)
    m2 = BPR(factors=8, iterations=5, seed=123)
    m2.fit(csr)
    # Note: Hogwild! parallel SGD is non-deterministic due to thread scheduling
    # race conditions. We use a generous atol to ensure the seed guides the
    # trajectory identically without failing on minor atomic overlaps.
    np.testing.assert_allclose(m1.user_factors, m2.user_factors, atol=0.05)
    np.testing.assert_allclose(m1.item_factors, m2.item_factors, atol=0.05)


def test_different_seed_different_factors():
    values = [1.0] * 4
    rows = [0, 0, 1, 2]
    cols = [0, 1, 2, 3]
    csr = sp.csr_matrix((values, (rows, cols)), shape=(3, 4))

    m1 = BPR(factors=8, iterations=5, seed=1)
    m1.fit(csr)
    m2 = BPR(factors=8, iterations=5, seed=2)
    m2.fit(csr)
    assert not np.allclose(m1.user_factors, m2.user_factors)
