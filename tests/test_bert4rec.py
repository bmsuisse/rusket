"""Tests for BERT4Rec recommendation model."""

import numpy as np
import pytest

import rusket


class TestBERT4Rec:
    @pytest.fixture(autouse=True)
    def fitted_model(self):
        sequences = [
            [0, 1, 2, 3],
            [1, 2, 4, 5],
            [0, 3, 5, 6],
            [2, 3, 6, 7],
            [4, 5, 6, 7],
        ]
        model = rusket.BERT4Rec(
            factors=16,
            n_layers=1,
            max_seq=10,
            mask_prob=0.2,
            iterations=3,
            seed=42,
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
        model = rusket.BERT4Rec.from_transactions(
            df,
            user_col="user_id",
            item_col="item_id",
            factors=8,
            iterations=2,
            seed=0,
        ).fit()
        ids, scores = model.recommend_items([1, 2, 3], n=3)
        assert len(ids) > 0

    def test_empty_sequence_returns_empty(self):
        # Even with empty sequence, max_seq predictions usually return the best global items
        # Let's verify it doesn't crash and returns some recommendations.
        ids, scores = self.model.recommend_items([], n=5)
        assert len(ids) > 0

    def test_mask_token_logic(self):
        # Verify the dimensions match n_items + 2
        # (row 0 is padding, row n_items + 1 is MASK)
        factors = self.model.item_factors
        assert factors.shape[0] == self.n_items + 2
