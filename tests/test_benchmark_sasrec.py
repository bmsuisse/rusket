import numpy as np
import pandas as pd
import pytest

import rusket


@pytest.fixture
def big_interactions():
    np.random.seed(42)
    n_users = 2000
    n_items = 1000
    n_interactions = 100000

    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)

    return pd.DataFrame({"user_id": users, "item_id": items})


@pytest.fixture
def sasrec_unfitted(big_interactions):
    return rusket.SASRec.from_transactions(
        big_interactions,
        user_col="user_id",
        item_col="item_id",
        factors=64,
        n_layers=2,
        max_seq=50,
        iterations=10,
        seed=42,
    )


@pytest.fixture
def sasrec_fitted(sasrec_unfitted):
    # Fit once and return
    sasrec_unfitted.fitted = False
    return sasrec_unfitted.fit()


def test_benchmark_sasrec_fit(benchmark, sasrec_unfitted):
    def run_fit():
        sasrec_unfitted.fitted = False
        sasrec_unfitted.fit()

    # We may want to reduce rounds if it's too slow
    benchmark.pedantic(run_fit, iterations=1, rounds=3)


def test_benchmark_sasrec_predict_single(benchmark, sasrec_fitted):
    # Benchmark predicting for a single user
    def run_predict():
        sasrec_fitted.recommend_items(user_id=1, n=10)

    benchmark.pedantic(run_predict, iterations=1, rounds=3)


def test_benchmark_sasrec_predict_batch(benchmark, sasrec_fitted):
    # Benchmark predicting for 100 users
    def run_predict_batch():
        for i in range(100):
            sasrec_fitted.recommend_items(user_id=i, n=10)

    benchmark.pedantic(run_predict_batch, iterations=1, rounds=3)
