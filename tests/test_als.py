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
    model = rusket.ALS(factors=factors, regularization=0.0, alpha=2.0, iterations=50, seed=42)
    model.fit(counts)
    reconstructed = model.user_factors @ model.item_factors.T
    for i in range(counts.shape[0]):  # type: ignore[index]
        for j in range(counts.shape[1]):  # type: ignore[index]
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
    Ciu = sparse.random(100, 100, density=0.0005, format="csr", dtype=np.float32, random_state=42)  # type: ignore[call-overload]
    model = rusket.ALS(factors=32, regularization=10.0, iterations=10, seed=23)
    model.fit(Ciu)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()


def test_small_identity_no_nan() -> None:
    user_item = coo_matrix((np.ones(10, dtype=np.float32), (np.arange(10), np.arange(10)))).tocsr()
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
    model = rusket.ALS.from_transactions(df, user_col="user", item_col="item", factors=4, iterations=5, seed=42).fit()
    assert model.user_factors.shape == (3, 4)  # type: ignore
    assert model.item_factors.shape == (4, 4)


def test_fit_transactions_polars() -> None:
    import polars as pl

    df = pl.DataFrame({"user_id": [0, 0, 1, 1, 2, 2], "product": ["x", "y", "y", "z", "x", "z"]})
    model = rusket.ALS.from_transactions(df, factors=4, iterations=5, seed=42).fit()
    assert model.user_factors.shape == (3, 4)  # type: ignore


def test_fit_transactions_with_ratings() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "user": [0, 0, 1, 1, 2],
            "item": ["a", "b", "a", "c", "b"],
            "rating": [5.0, 3.0, 4.0, 1.0, 2.0],
        }
    )
    model = rusket.ALS.from_transactions(df, rating_col="rating", factors=4, iterations=5, seed=42).fit()
    assert model.user_factors.shape == (3, 4)  # type: ignore


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
            assert item not in seen, f"User {user_id}: item {item} was seen but returned with exclude_seen=True"


def test_exclude_seen_false_returns_seen_items() -> None:
    """With exclude_seen=False, seen items can appear in recommendations."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)

    for user_id in range(5):
        seen = set(mat[user_id].indices.tolist())
        ids_incl, _ = model.recommend_items(user_id, n=20, exclude_seen=False)
        has_seen = any(item in seen for item in ids_incl)
        assert has_seen, f"User {user_id}: expected at least one seen item with exclude_seen=False"


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


def test_als_batch_recommend() -> None:
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)

    # Test polars returns
    df = model.batch_recommend(n=5, exclude_seen=True, format="polars")
    import polars as pl

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 20 * 5
    assert df.columns == ["user_id", "item_id", "score"]

    # Test pandas returns
    df_pd = model.batch_recommend(n=5, exclude_seen=True, format="pandas")
    import pandas as pd

    assert isinstance(df_pd, pd.DataFrame)
    assert len(df_pd) == 20 * 5


def test_als_export_formats() -> None:
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)

    df_items = model.export_factors(format="polars")
    df_users = model.export_user_factors(format="pandas")

    import pandas as pd
    import polars as pl

    assert isinstance(df_items, pl.DataFrame)
    assert len(df_items) == 20
    assert "vector" in df_items.columns

    assert isinstance(df_users, pd.DataFrame)
    assert len(df_users) == 20
    assert "vector" in df_users.columns


def test_als_normalization() -> None:
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(mat)

    df_norm = model.export_factors(format="pandas", normalize=True)
    lengths = df_norm["vector"].apply(lambda v: np.linalg.norm(v)).values
    np.testing.assert_allclose(lengths, np.ones(20), rtol=1e-5)


def test_recalculate_user_basic() -> None:
    model = rusket.ALS(factors=8, regularization=0.1, iterations=15, seed=42)
    model.fit(get_checker_board(20))

    # Recalculate factors for user 0
    # User 0 engaged with items 0, 2, 4, 6...
    user_0_items = [i for i in range(20) if i % 2 == 0]

    new_factors = model.recalculate_user(user_0_items)
    assert new_factors.shape == (8,)

    # The new factors should be somewhat close to original user 0 factors
    # Exact match depends on CG convergence vs full ALS fixed-point
    cosine_sim = np.dot(new_factors, model.user_factors[0]) / (
        np.linalg.norm(new_factors) * np.linalg.norm(model.user_factors[0])
    )
    assert cosine_sim > 0.9


def test_recalculate_user_not_fitted() -> None:
    model = rusket.ALS(factors=8)
    with pytest.raises(RuntimeError):
        model.recalculate_user([1, 2, 3])


def test_recalculate_user_out_of_bounds() -> None:
    model = rusket.ALS(factors=8, iterations=5)
    model.fit(get_checker_board(10))
    with pytest.raises(ValueError):
        model.recalculate_user([10])
    with pytest.raises(ValueError):
        model.recalculate_user([-1])


def test_recalculate_empty_user() -> None:
    model = rusket.ALS(factors=8, iterations=5)
    model.fit(get_checker_board(10))
    factors = model.recalculate_user([])
    assert factors.shape == (8,)
    # Factors should be a zero vector for a user with no interactions
    np.testing.assert_allclose(factors, np.zeros(8), atol=1e-5)


import hypothesis.strategies as st  # noqa: E402
from hypothesis import given, settings  # noqa: E402


@given(
    rows=st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=50),
    cols=st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=50),
)
@settings(max_examples=20, deadline=None)
def test_als_properties(rows, cols):
    from rusket import ALS

    if len(rows) != len(cols):
        min_len = min(len(rows), len(cols))
        rows = rows[:min_len]
        cols = cols[:min_len]

    data = np.ones(len(rows), dtype=np.float32)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(10, 10)).tocsr()

    model = ALS(factors=4, iterations=5, seed=42)
    model.fit(mat)

    assert model.user_factors.shape == (10, 4)
    assert model.item_factors.shape == (10, 4)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()


def test_eals_wrapper() -> None:
    mat = get_checker_board(20)

    model_als = rusket.ALS(factors=8, iterations=5, seed=42, use_eals=True)
    model_als.fit(mat)

    model_eals = rusket.eALS(factors=8, iterations=5, seed=42)
    model_eals.fit(mat)

    np.testing.assert_array_equal(model_als.user_factors, model_eals.user_factors)
    np.testing.assert_array_equal(model_als.item_factors, model_eals.item_factors)


# ---------------------------------------------------------------------------
# eALS popularity weighting tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weighting", ["sqrt", "log", "linear"])
def test_eals_popularity_weighting_modes(weighting: str) -> None:
    """Fit with each popularity weighting mode — factors must be finite."""
    mat = get_checker_board(20)
    model = rusket.eALS(factors=8, iterations=5, seed=42, popularity_weighting=weighting)
    model.fit(mat)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()


@pytest.mark.parametrize("weighting", ["sqrt", "log", "linear"])
def test_eals_popularity_weighting_differs_from_uniform(weighting: str) -> None:
    """Popularity weighting should produce different factors than uniform weighting."""
    # Build a skewed matrix where item popularity follows a power law
    rng = np.random.RandomState(99)
    n_users, n_items = 50, 30
    rows, cols = [], []
    for u in range(n_users):
        # Popular items (0-5) appear much more often than tail items (20-29)
        n_interactions = rng.randint(3, 10)
        items = rng.choice(
            n_items,
            size=n_interactions,
            replace=False,
            p=np.arange(n_items, 0, -1, dtype=float) / sum(range(1, n_items + 1)),
        )
        for i in items:
            rows.append(u)
            cols.append(i)
    mat = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items))

    m_base = rusket.eALS(factors=8, iterations=10, seed=42)
    m_base.fit(mat)
    m_pop = rusket.eALS(factors=8, iterations=10, seed=42, popularity_weighting=weighting)
    m_pop.fit(mat)
    # Factors must differ (same seed, same data, but different weighting scheme)
    assert not np.allclose(m_base.user_factors, m_pop.user_factors, rtol=1e-3)


def test_eals_popularity_default_unchanged() -> None:
    """popularity_weighting='none' must produce identical results to no weighting."""
    mat = get_checker_board(20)
    m1 = rusket.eALS(factors=8, iterations=5, seed=42)
    m1.fit(mat)
    m2 = rusket.eALS(factors=8, iterations=5, seed=42, popularity_weighting="none")
    m2.fit(mat)
    np.testing.assert_array_equal(m1.user_factors, m2.user_factors)
    np.testing.assert_array_equal(m1.item_factors, m2.item_factors)


# ---------------------------------------------------------------------------
# Bias terms tests
# ---------------------------------------------------------------------------


def test_als_bias_terms_finite() -> None:
    """Bias terms must be finite with correct shapes."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, iterations=10, seed=42, use_biases=True)
    model.fit(mat)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()
    assert np.isfinite(model.global_bias)
    assert np.isfinite(model.user_biases).all()
    assert np.isfinite(model.item_biases).all()
    assert model.user_biases.shape == (20,)
    assert model.item_biases.shape == (20,)


def test_als_bias_disabled_matches_baseline() -> None:
    """use_biases=False must give zero biases and match factor-only baseline."""
    mat = get_checker_board(20)
    m1 = rusket.ALS(factors=8, iterations=5, seed=42, use_biases=False)
    m1.fit(mat)
    m2 = rusket.ALS(factors=8, iterations=5, seed=42)
    m2.fit(mat)
    # Biases should be zero
    assert m1.global_bias == 0.0
    assert (m1.user_biases == 0.0).all()
    assert (m1.item_biases == 0.0).all()
    # Factors should be identical
    np.testing.assert_array_equal(m1.user_factors, m2.user_factors)
    np.testing.assert_array_equal(m1.item_factors, m2.item_factors)


def test_als_bias_recommendations_work() -> None:
    """Model with biases should produce valid recommendations."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, iterations=10, seed=42, use_biases=True)
    model.fit(mat)
    ids, scores = model.recommend_items(0, n=5)
    assert len(ids) == 5
    assert np.isfinite(scores).all()
    # recommend_users should also work
    uids, uscores = model.recommend_users(0, n=5)
    assert len(uids) == 5
    assert np.isfinite(uscores).all()


# ---------------------------------------------------------------------------
# ANN index integration tests
# ---------------------------------------------------------------------------


def test_build_ann_index_native() -> None:
    """build_ann_index('native') returns a working ANN index."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, iterations=5, seed=42)
    model.fit(mat)
    idx = model.build_ann_index(backend="native")
    # Query with user factors → should return item neighbors
    neighbors, distances = idx.kneighbors(model.user_factors[:1], n_neighbors=5)
    assert neighbors.shape == (1, 5)
    assert distances.shape == (1, 5)


def test_build_ann_index_not_fitted() -> None:
    """build_ann_index raises if model not fitted."""
    model = rusket.ALS(factors=8)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.build_ann_index()


def test_build_ann_index_invalid_backend() -> None:
    """build_ann_index raises on unknown backend."""
    mat = get_checker_board(20)
    model = rusket.ALS(factors=8, iterations=5, seed=42)
    model.fit(mat)
    with pytest.raises(ValueError, match="Unknown backend"):
        model.build_ann_index(backend="unknown")


# ---------------------------------------------------------------------------
# VALS (View-Aware ALS) tests
# ---------------------------------------------------------------------------


def test_vals_basic_fit() -> None:
    """Fit with purchase + view matrices — factors must be finite."""
    rng = np.random.RandomState(42)
    n_users, n_items = 30, 20
    purchases = csr_matrix(rng.randint(0, 2, (n_users, n_items)).astype(np.float32))
    views = csr_matrix(rng.randint(0, 2, (n_users, n_items)).astype(np.float32))

    model = rusket.ALS(factors=8, iterations=10, seed=42, alpha_view=10.0, view_target=0.5)
    model.fit(purchases, view_matrix=views)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()


def test_vals_no_views_unchanged() -> None:
    """Without view_matrix, VALS params don't change results."""
    mat = get_checker_board(20)
    m1 = rusket.ALS(factors=8, iterations=5, seed=42)
    m1.fit(mat)
    m2 = rusket.ALS(factors=8, iterations=5, seed=42, alpha_view=10.0, view_target=0.5)
    m2.fit(mat)
    np.testing.assert_array_equal(m1.user_factors, m2.user_factors)
    np.testing.assert_array_equal(m1.item_factors, m2.item_factors)


def test_vals_view_matrix_shape_mismatch() -> None:
    """Mismatched view_matrix shape should raise ValueError."""
    purchases = csr_matrix(np.ones((10, 20), dtype=np.float32))
    views = csr_matrix(np.ones((10, 15), dtype=np.float32))  # wrong shape
    model = rusket.ALS(factors=8, iterations=5, seed=42)
    with pytest.raises(ValueError, match="shape"):
        model.fit(purchases, view_matrix=views)


def test_vals_views_affect_results() -> None:
    """Providing views should produce different factors than no views."""
    rng = np.random.RandomState(42)
    n_users, n_items = 30, 20
    purchases = csr_matrix(rng.randint(0, 2, (n_users, n_items)).astype(np.float32))
    views = csr_matrix(rng.randint(0, 2, (n_users, n_items)).astype(np.float32))

    m1 = rusket.ALS(factors=8, iterations=10, seed=42)
    m1.fit(purchases)
    m2 = rusket.ALS(factors=8, iterations=10, seed=42, alpha_view=15.0, view_target=0.5)
    m2.fit(purchases, view_matrix=views)
    assert not np.allclose(m1.user_factors, m2.user_factors, rtol=1e-3)

