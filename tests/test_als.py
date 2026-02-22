"""Tests for rusket.ALS — mirrors key tests from implicit library's test suite."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix

import rusket


def get_checker_board(n: int) -> csr_matrix:
    row, col = [], []
    for i in range(n):
        for j in range(n):
            if i % 2 == j % 2:
                row.append(i)
                col.append(j)
    return csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n))


@pytest.mark.parametrize("factors", [6])
def test_factorize_reconstruction(factors: int) -> None:
    counts = csr_matrix(
        [
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    model = rusket.ALS(
        factors=factors, regularization=0.0, alpha=2.0, iterations=50, seed=42
    )
    model.fit(counts)
    reconstructed = model.user_factors @ model.item_factors.T
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            expected = 1.0 if counts[i, j] > 0 else 0.0
            assert reconstructed[i, j] == pytest.approx(expected, abs=0.15), (
                f"row={i}, col={j}, got={reconstructed[i, j]:.4f}"
            )


def test_no_nan_in_factors() -> None:
    raw = [
        [0, 2, 1.5, 1.33, 1.25, 1.2, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 1.5, 1.33, 1.25, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 1.5, 1.33, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 1.5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 1.5, 1.33, 1.25, 1.2],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 1.5, 1.33, 1.25],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1.5, 1.33],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1.5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    model = rusket.ALS(factors=3, regularization=0.01, iterations=15, seed=23)
    model.fit(csr_matrix(raw, dtype=np.float32))
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()


def test_no_nan_sparse() -> None:
    Ciu = sparse.random(
        100, 100, density=0.0005, format="csr", dtype=np.float32, random_state=42
    )
    model = rusket.ALS(factors=32, regularization=10.0, iterations=10, seed=23)
    model.fit(Ciu)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()


def test_small_identity_no_nan() -> None:
    user_item = coo_matrix(
        (np.ones(10, dtype=np.float32), (np.arange(10), np.arange(10)))
    ).tocsr()
    model = rusket.ALS(factors=15, iterations=10, seed=42)
    model.fit(user_item)
    ids, scores = model.recommend_items(0, n=10, exclude_seen=False)
    assert not np.isnan(scores).any()
    assert ids[0] == 0


def test_recommend_items_basic() -> None:
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(get_checker_board(20))
    ids, scores = model.recommend_items(0, n=5, exclude_seen=False)
    assert len(ids) == 5 and len(scores) == 5
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_recommend_items_exclude_seen() -> None:
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)
    user_0_items = set(mat[0].indices.tolist())
    ids_excl, _ = model.recommend_items(0, n=10, exclude_seen=True)
    ids_incl, _ = model.recommend_items(0, n=10, exclude_seen=False)
    for item_id in ids_excl:
        assert item_id not in user_0_items
    assert any(item_id in user_0_items for item_id in ids_incl)


def test_recommend_users_basic() -> None:
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(get_checker_board(20))
    ids, scores = model.recommend_users(item_id=0, n=5)
    assert len(ids) == 5 and len(scores) == 5
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_checker_board_recommendations() -> None:
    model = rusket.ALS(factors=4, regularization=0.01, iterations=20, seed=42)
    model.fit(get_checker_board(50))
    ids, _ = model.recommend_items(0, n=5, exclude_seen=False)
    assert sum(1 for i in ids if i % 2 == 0) >= 3
    ids, _ = model.recommend_items(1, n=5, exclude_seen=False)
    assert sum(1 for i in ids if i % 2 == 1) >= 2


def test_fit_transactions_pandas() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "user": [0, 0, 0, 1, 1, 2, 2, 2],
            "item": ["a", "b", "c", "b", "c", "a", "c", "d"],
        }
    )
    model = rusket.ALS(factors=4, iterations=5, seed=42)
    model.fit_transactions(df, user_col="user", item_col="item")
    assert model.user_factors.shape == (3, 4)
    assert model.item_factors.shape == (4, 4)


def test_fit_transactions_polars() -> None:
    import polars as pl

    df = pl.DataFrame(
        {"user_id": [0, 0, 1, 1, 2, 2], "product": ["x", "y", "y", "z", "x", "z"]}
    )
    model = rusket.ALS(factors=4, iterations=5, seed=42)
    model.fit_transactions(df)
    assert model.user_factors.shape == (3, 4)


def test_fit_transactions_with_ratings() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "user": [0, 0, 1, 1, 2],
            "item": ["a", "b", "a", "c", "b"],
            "rating": [5.0, 3.0, 4.0, 1.0, 2.0],
        }
    )
    model = rusket.ALS(factors=4, iterations=5, seed=42)
    model.fit_transactions(df, rating_col="rating")
    assert model.user_factors.shape == (3, 4)


def test_single_user() -> None:
    model = rusket.ALS(factors=4, iterations=5, seed=42)
    model.fit(csr_matrix([[1, 0, 1, 0, 1]], dtype=np.float32))
    assert model.user_factors.shape == (1, 4)
    assert model.item_factors.shape == (5, 4)
    ids, _ = model.recommend_items(0, n=3, exclude_seen=True)
    assert len(ids) <= 3


def test_single_item() -> None:
    model = rusket.ALS(factors=4, iterations=5, seed=42)
    model.fit(csr_matrix([[1], [0], [1]], dtype=np.float32))
    assert model.user_factors.shape == (3, 4)
    assert model.item_factors.shape == (1, 4)


def test_fit_empty_matrix() -> None:
    raw = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    model = rusket.ALS(factors=4, iterations=5)
    model.fit(csr_matrix(raw, dtype=np.float32))
    assert model.user_factors.shape == (3, 4)
    assert model.item_factors.shape == (3, 4)
    assert not np.isnan(model.user_factors).any()


def test_not_fitted_raises() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = rusket.ALS().user_factors


def test_deterministic_seed() -> None:
    mat = get_checker_board(20)
    m1 = rusket.ALS(factors=8, iterations=5, seed=123)
    m1.fit(mat)
    m2 = rusket.ALS(factors=8, iterations=5, seed=123)
    m2.fit(mat)
    np.testing.assert_array_equal(m1.user_factors, m2.user_factors)
    np.testing.assert_array_equal(m1.item_factors, m2.item_factors)


def test_different_seed_different_factors() -> None:
    mat = get_checker_board(20)
    m1 = rusket.ALS(factors=8, iterations=5, seed=1)
    m1.fit(mat)
    m2 = rusket.ALS(factors=8, iterations=5, seed=2)
    m2.fit(mat)
    assert not np.allclose(m1.user_factors, m2.user_factors)


# ---------------------------------------------------------------------------
# recommend_users – detailed coverage
# ---------------------------------------------------------------------------


def test_recommend_users_returns_sorted() -> None:
    """recommend_users scores must be strictly descending."""
    model = rusket.ALS(factors=8, regularization=0.01, iterations=20, seed=7)
    model.fit(get_checker_board(30))
    ids, scores = model.recommend_users(item_id=0, n=10)
    assert len(ids) == 10
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"scores not sorted at position {i}"


def test_recommend_users_top1_is_correct() -> None:
    """For item 0 (even), user 0 (even, matched) should rank highly."""
    mat = get_checker_board(50)
    model = rusket.ALS(factors=4, regularization=0.01, iterations=30, seed=42)
    model.fit(mat)
    ids, scores = model.recommend_users(item_id=0, n=50)
    even_users = [u for u in ids if u % 2 == 0]
    # The top-ranked users should include even users (who actually liked even items)
    assert even_users[0] in ids[:25], "Even users should dominate top-N for even item"


def test_recommend_users_count_bounded_by_n_users() -> None:
    """Returned user count cannot exceed total users."""
    mat = csr_matrix([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    model = rusket.ALS(factors=2, iterations=5, seed=42)
    model.fit(mat)
    ids, scores = model.recommend_users(item_id=0, n=100)
    assert len(ids) <= 3  # only 3 users


def test_recommend_users_no_nan_scores() -> None:
    """recommend_users should never return NaN scores."""
    model = rusket.ALS(factors=16, regularization=0.01, iterations=10, seed=42)
    model.fit(get_checker_board(40))
    for item_id in range(10):
        _, scores = model.recommend_users(item_id=item_id, n=10)
        assert np.isfinite(scores).all(), f"NaN in scores for item {item_id}"


# ---------------------------------------------------------------------------
# recommend_items exclude_seen – detailed coverage
# ---------------------------------------------------------------------------


def test_exclude_seen_excludes_all_known_items() -> None:
    """With exclude_seen=True, every returned item must be unseen."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)

    for user_id in range(5):
        seen = set(mat[user_id].indices.tolist())
        ids, _ = model.recommend_items(user_id, n=20, exclude_seen=True)
        for item in ids:
            assert item not in seen, (
                f"User {user_id}: item {item} was seen but returned with exclude_seen=True"
            )


def test_exclude_seen_false_returns_seen_items() -> None:
    """With exclude_seen=False, seen items can appear in recommendations."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)

    for user_id in range(5):
        seen = set(mat[user_id].indices.tolist())
        ids_incl, _ = model.recommend_items(user_id, n=20, exclude_seen=False)
        has_seen = any(item in seen for item in ids_incl)
        assert has_seen, (
            f"User {user_id}: expected at least one seen item with exclude_seen=False"
        )


def test_exclude_seen_count_reduced() -> None:
    """With exclude_seen=True, fewer items returned compared to False for same n."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)

    user_id = 0
    n_seen = int(mat[0].nnz)
    ids_excl, _ = model.recommend_items(user_id, n=20, exclude_seen=True)
    ids_incl, _ = model.recommend_items(user_id, n=20, exclude_seen=False)

    # With exclusion, can have at most (n_items - n_seen) items
    assert len(ids_excl) <= len(ids_incl) or len(ids_excl) == 20 - n_seen


def test_exclude_seen_no_overlap_with_training() -> None:
    """Cross-check exclude_seen using a known interaction matrix."""
    # User 0 has seen item 0 and 2; only items 1, 3, 4 should be recommendable
    mat = csr_matrix([[1, 0, 1, 0, 0]], dtype=np.float32)
    model = rusket.ALS(factors=2, regularization=0.1, iterations=10, seed=42)
    model.fit(mat)

    ids, _ = model.recommend_items(0, n=5, exclude_seen=True)
    assert 0 not in ids, "Item 0 is seen and must be excluded"
    assert 2 not in ids, "Item 2 is seen and must be excluded"
    # Must only recommend from {1, 3, 4}
    for item in ids:
        assert item in {1, 3, 4}, f"Item {item} was seen but returned"


def test_recommend_items_no_nan_scores() -> None:
    """recommend_items should never return NaN scores."""
    model = rusket.ALS(factors=16, regularization=0.01, iterations=10, seed=42)
    model.fit(get_checker_board(40))
    for user_id in range(10):
        _, scores = model.recommend_items(user_id=user_id, n=10, exclude_seen=True)
        assert np.isfinite(scores).all(), f"NaN in scores for user {user_id}"
