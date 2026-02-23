"""pytest-benchmark: rusket vs LibRecommender on MovieLens 100k.

Statistically valid benchmarks with warmup, multiple rounds.
Only the actual model.fit() is timed — no data loading or startup overhead.

Usage:
    uv run pytest benchmarks/bench_pytest_librecommender.py -v --benchmark-columns=mean,stddev,rounds
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# ── Data fixtures (run once, shared across all benchmarks) ───────────────

DATA_DIR = Path(__file__).parent / "data" / "ml-100k"


@pytest.fixture(scope="module")
def ml100k_df() -> pd.DataFrame:
    """Download and return MovieLens 100k as a DataFrame."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "u.data"

    if not data_file.exists():
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = DATA_DIR.parent / "ml-100k.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR.parent)

    df = pd.read_csv(
        data_file, sep="\t", names=["user_id", "item_id", "rating", "timestamp"],
    )
    # Remap to 0-based contiguous IDs
    user_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item_map = {i: j for j, i in enumerate(sorted(df["item_id"].unique()))}
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)
    return df


@pytest.fixture(scope="module")
def ml100k_implicit_csr(ml100k_df: pd.DataFrame) -> sparse.csr_matrix:
    """Build implicit CSR matrix from ML-100k."""
    n_users = ml100k_df["user_id"].nunique()
    n_items = ml100k_df["item_id"].nunique()
    return sparse.csr_matrix(
        (np.ones(len(ml100k_df), dtype=np.float32),
         (ml100k_df["user_id"].values, ml100k_df["item_id"].values)),
        shape=(n_users, n_items),
    )


@pytest.fixture(scope="module")
def ml100k_explicit_csr(ml100k_df: pd.DataFrame) -> sparse.csr_matrix:
    """Build explicit CSR matrix from ML-100k."""
    n_users = ml100k_df["user_id"].nunique()
    n_items = ml100k_df["item_id"].nunique()
    return sparse.csr_matrix(
        (ml100k_df["rating"].values.astype(np.float32),
         (ml100k_df["user_id"].values, ml100k_df["item_id"].values)),
        shape=(n_users, n_items),
    )


# ── LibRecommender data fixture ─────────────────────────────────────────

HAS_LIBRECO = False
try:
    from libreco.data import DatasetPure, random_split
    HAS_LIBRECO = True
except ImportError:
    pass


@pytest.fixture(scope="module")
def libreco_data(ml100k_df: pd.DataFrame):
    """Build LibRecommender train data. Returns (train_df_raw, build_fn)."""
    if not HAS_LIBRECO:
        pytest.skip("LibRecommender not installed")
    lr_df = ml100k_df.rename(columns={
        "user_id": "user", "item_id": "item", "rating": "label",
    })
    lr_train_df, _ = random_split(lr_df, test_size=0.2, seed=42)
    return lr_train_df


def fresh_libreco_data(lr_train_df):
    """Build fresh train data (LibRecommender mutates state)."""
    train_data, data_info = DatasetPure.build_trainset(lr_train_df)
    return train_data, data_info


# ── Warmup fixture (warms up PyTorch JIT, etc.) ─────────────────────────

@pytest.fixture(scope="module", autouse=True)
def warmup_libreco(libreco_data):
    """Run a tiny model to warm up PyTorch/TF runtime."""
    if not HAS_LIBRECO:
        return
    try:
        from libreco.algorithms import BPR as _BPR
        train_data, data_info = fresh_libreco_data(libreco_data)
        m = _BPR(task="ranking", data_info=data_info, embed_size=4, n_epochs=1, use_tf=False)
        m.fit(train_data, neg_sampling=True, verbose=0)
        del m
    except Exception:
        pass


# ── rusket benchmarks ────────────────────────────────────────────────────

class TestRusketBenchmarks:
    def test_als_rusket(self, benchmark, ml100k_implicit_csr):
        import rusket
        mat = ml100k_implicit_csr

        def fit():
            m = rusket.ALS(factors=64, iterations=15, regularization=0.01, alpha=40.0, seed=42)
            m.fit(mat)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)

    def test_svd_rusket(self, benchmark, ml100k_explicit_csr):
        import rusket
        mat = ml100k_explicit_csr

        def fit():
            m = rusket.SVD(factors=64, iterations=20, learning_rate=0.005, regularization=0.02, seed=42)
            m.fit(mat)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)

    def test_bpr_rusket(self, benchmark, ml100k_implicit_csr):
        import rusket
        mat = ml100k_implicit_csr

        def fit():
            m = rusket.BPR(factors=64, iterations=10, learning_rate=0.01, regularization=0.01, seed=42)
            m.fit(mat)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)

    def test_itemknn_rusket(self, benchmark, ml100k_implicit_csr):
        import rusket
        mat = ml100k_implicit_csr

        def fit():
            m = rusket.ItemKNN(k=100)
            m.fit(mat)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)

    def test_ease_rusket(self, benchmark, ml100k_implicit_csr):
        import rusket
        mat = ml100k_implicit_csr

        def fit():
            m = rusket.EASE()
            m.fit(mat)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)


# ── LibRecommender benchmarks ───────────────────────────────────────────

@pytest.mark.skipif(not HAS_LIBRECO, reason="LibRecommender not installed")
class TestLibRecoBenchmarks:
    def test_als_libreco(self, benchmark, libreco_data):
        from libreco.algorithms import ALS as LibALS

        def fit():
            train_data, data_info = fresh_libreco_data(libreco_data)
            m = LibALS(task="ranking", data_info=data_info,
                       embed_size=64, n_epochs=15, reg=0.01, alpha=40)
            m.fit(train_data, neg_sampling=True, verbose=0)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)

    def test_bpr_libreco(self, benchmark, libreco_data):
        from libreco.algorithms import BPR as LibBPR

        def fit():
            train_data, data_info = fresh_libreco_data(libreco_data)
            m = LibBPR(task="ranking", data_info=data_info,
                       embed_size=64, n_epochs=10, lr=0.01, use_tf=False)
            m.fit(train_data, neg_sampling=True, verbose=0)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)

    def test_itemcf_libreco(self, benchmark, libreco_data):
        from libreco.algorithms import ItemCF as LibItemCF

        def fit():
            train_data, data_info = fresh_libreco_data(libreco_data)
            m = LibItemCF(task="ranking", data_info=data_info, k_sim=100)
            m.fit(train_data, neg_sampling=True, verbose=0)
            return m

        benchmark.pedantic(fit, rounds=5, warmup_rounds=1)
