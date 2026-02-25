"""Tests for model save/load serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rusket import ALS, BPR, EASE, FPMC, SVD, AutoMiner, Eclat, FPGrowth


@pytest.fixture
def als_model():
    """Train a small ALS model."""
    from scipy import sparse as sp

    rng = np.random.default_rng(42)
    n_users, n_items = 20, 30
    data = (rng.random((n_users, n_items)) < 0.15).astype(np.float32)
    csr = sp.csr_matrix(data)
    model = ALS(factors=8, iterations=5, seed=42)
    model.fit(csr)
    return model


@pytest.fixture
def miner_model():
    """Create a mined FPGrowth model."""
    df = pd.DataFrame(
        {
            "bread": [True, True, False, True, True],
            "milk": [True, False, True, True, True],
            "butter": [True, False, True, False, False],
            "eggs": [False, True, True, False, True],
        }
    )
    miner = FPGrowth(data=df, min_support=0.4, use_colnames=True)
    miner.mine()
    return miner


class TestSaveLoadRecommender:
    def test_als_round_trip(self, als_model):
        """ALS save → load should preserve recommendations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "als.pkl"
            als_model.save(path)

            loaded = ALS.load(path)
            assert loaded.fitted

            # Recommendations should match exactly
            orig_items, orig_scores = als_model.recommend_items(0, n=5, exclude_seen=False)
            load_items, load_scores = loaded.recommend_items(0, n=5, exclude_seen=False)

            np.testing.assert_array_equal(orig_items, load_items)
            np.testing.assert_array_almost_equal(orig_scores, load_scores)

    def test_als_factors_preserved(self, als_model):
        """Item/user factors should be identical after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "als.pkl"
            als_model.save(path)
            loaded = ALS.load(path)

            np.testing.assert_array_equal(als_model.item_factors, loaded.item_factors)
            np.testing.assert_array_equal(als_model.user_factors, loaded.user_factors)

    def test_save_creates_parent_dirs(self, als_model):
        """save() should create intermediate directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "model.pkl"
            als_model.save(path)
            assert path.exists()


class TestSaveLoadMiner:
    def test_fpgrowth_round_trip(self, miner_model):
        """FPGrowth miner save → load should preserve item_names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fpg.pkl"
            miner_model.save(path)
            loaded = FPGrowth.load(path)

            assert loaded.item_names == miner_model.item_names

    def test_eclat_round_trip(self):
        """Eclat save → load should work."""
        df = pd.DataFrame({"A": [True, True], "B": [True, False], "C": [False, True]})
        miner = Eclat(data=df, min_support=0.5, use_colnames=True)
        miner.mine()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eclat.pkl"
            miner.save(path)
            loaded = Eclat.load(path)
            assert loaded.item_names == miner.item_names


class TestSaveLoadEdgeCases:
    def test_load_wrong_class_warns(self, als_model):
        """Loading into the wrong class should warn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "als.pkl"
            als_model.save(path)

            with pytest.warns(UserWarning, match="saved as ALS but loaded as SVD"):
                SVD.load(path)

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            ALS.load("/nonexistent/path/model.pkl")
