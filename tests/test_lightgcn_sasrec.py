"""Tests for LightGCN and SASRec recommendation models."""
import numpy as np
import pytest

import rusket


@pytest.fixture
def small_interactions() -> list[tuple[int, int]]:
    return [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 2), (2, 4),
        (3, 3), (3, 4), (3, 5),
        (4, 0), (4, 3), (4, 5),
    ]


@pytest.fixture
def interactions_df(small_interactions):
    import pandas as pd
    return pd.DataFrame(small_interactions, columns=["user_id", "item_id"])


# ─── LightGCN ────────────────────────────────────────────────────────────────

class TestLightGCN:
    def test_basic_fit_and_recommend(self, interactions_df):
        model = rusket.LightGCN.from_transactions(
            interactions_df,
            user_col="user_id",
            item_col="item_id",
            factors=16,
            k_layers=2,
            iterations=5,
            random_state=42,
        )
        ids, scores = model.recommend_items(user_id=0, n=3)
        assert len(ids) == 3
        assert len(scores) == 3
        # Scores should be sorted descending
        assert np.all(np.diff(scores) <= 0)

    def test_returns_valid_item_ids(self, interactions_df):
        model = rusket.LightGCN.from_transactions(
            interactions_df,
            user_col="user_id",
            item_col="item_id",
            factors=8,
            iterations=3,
            random_state=1,
        )
        ids, scores = model.recommend_items(user_id=0, n=10)
        assert len(ids) > 0
        # All returned IDs should be valid item IDs (0–5)
        assert all(0 <= i <= 5 for i in ids)

    def test_unknown_user_returns_empty(self, interactions_df):
        model = rusket.LightGCN.from_transactions(
            interactions_df,
            user_col="user_id",
            item_col="item_id",
            factors=8,
            iterations=2,
            random_state=0,
        )
        ids, scores = model.recommend_items(user_id=999, n=5)
        assert len(ids) == 0

    def test_reproducible_with_random_state(self, interactions_df):
        m1 = rusket.LightGCN.from_transactions(
            interactions_df, user_col="user_id", item_col="item_id",
            factors=16, iterations=5, random_state=42,
        )
        m2 = rusket.LightGCN.from_transactions(
            interactions_df, user_col="user_id", item_col="item_id",
            factors=16, iterations=5, random_state=42,
        )
        ids1, _ = m1.recommend_items(user_id=0, n=5)
        ids2, _ = m2.recommend_items(user_id=0, n=5)
        np.testing.assert_array_equal(ids1, ids2)

    def test_polars_input(self, interactions_df):
        pytest.importorskip("polars")
        import polars as pl
        df_pl = pl.from_pandas(interactions_df)
        model = rusket.LightGCN.from_transactions(
            df_pl, user_col="user_id", item_col="item_id",
            factors=8, iterations=2, random_state=0,
        )
        ids, scores = model.recommend_items(user_id=0, n=3)
        assert len(ids) > 0


# ─── SASRec ──────────────────────────────────────────────────────────────────

class TestSASRec:
    @pytest.fixture(autouse=True)
    def fitted_model(self):
        sequences = [
            [0, 1, 2, 3],
            [1, 2, 4, 5],
            [0, 3, 5, 6],
            [2, 3, 6, 7],
            [4, 5, 6, 7],
        ]
        model = rusket.SASRec(
            factors=16, n_layers=1, max_seq=10,
            iterations=3, random_state=42,
        )
        model.fit(sequences)
        self.model = model
        self.n_items = max(max(s) for s in sequences) + 1

    def test_fit_and_recommend(self):
        ids, scores = self.model.recommend_items([1, 2, 3], n=5)
        assert len(ids) > 0

    def test_scores_are_finite(self):
        ids, scores = self.model.recommend_items([1, 2], n=5)
        assert np.all(np.isfinite(scores))

    def test_from_transactions(self):
        import pandas as pd
        df = pd.DataFrame(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4)],
            columns=["user_id", "item_id"],
        )
        model = rusket.SASRec.from_transactions(
            df, user_col="user_id", item_col="item_id",
            factors=8, iterations=2, random_state=0,
        )
        ids, scores = model.recommend_items([1, 2, 3], n=3)
        assert len(ids) > 0

    def test_empty_sequence_returns_empty(self):
        ids, scores = self.model.recommend_items([], n=5)
        assert len(ids) == 0
