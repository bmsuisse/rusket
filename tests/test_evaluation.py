import numpy as np
import pandas as pd
import pytest

from rusket import ALS, evaluate
from rusket._rusket import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k  # type: ignore


def test_rust_metrics():
    actual = [1, 5, 9]
    predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # hit_rate_at_k
    assert hit_rate_at_k(actual, predicted, 1) == 1.0
    assert hit_rate_at_k(actual, predicted, 5) == 1.0
    assert hit_rate_at_k(actual, [11, 12, 13], 3) == 0.0

    # precision_at_k
    assert precision_at_k(actual, predicted, 1) == 1.0
    assert np.isclose(precision_at_k(actual, predicted, 5), 2.0 / 5.0, atol=1e-6)
    assert np.isclose(precision_at_k(actual, predicted, 10), 3.0 / 10.0, atol=1e-6)

    # recall_at_k
    assert np.isclose(recall_at_k(actual, predicted, 1), 1.0 / 3.0, atol=1e-6)
    assert np.isclose(recall_at_k(actual, predicted, 5), 2.0 / 3.0, atol=1e-6)
    assert np.isclose(recall_at_k(actual, predicted, 10), 1.0, atol=1e-6)

    # ndcg_at_k
    # predicted: 1 (hit), 2, 3, 4, 5 (hit)
    # i=0: 1/log2(2) = 1.0
    # i=4: 1/log2(6) = 0.3868
    # dcg = 1.3868
    # IDCG (ideal is 3 hits at pos 0, 1, 2) = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 = 2.1309
    # expected ndcg = 1.3868 / 2.1309 ~= 0.6508
    ndcg_5 = ndcg_at_k(actual, predicted, 5)
    assert 0.64 < ndcg_5 < 0.66


class MockRecommender:
    def recommend_items(self, user_id, n=10, exclude_seen=True):
        # Recommend items (user_id + i) for i in 1..n
        r_items = []
        r_scores = []
        for i in range(1, n + 1):
            r_items.append(user_id + i)
            r_scores.append(1.0)
        return np.array(r_items), np.array(r_scores)


def test_python_evaluate_wrapper():
    model = MockRecommender()

    # Ground truth:
    # User 1 likes 2, 3
    # User 2 likes 99 (not in recommendations)
    test_df = pd.DataFrame({"user": [1, 1, 2], "item": [2, 3, 99]})

    res = evaluate(model, test_df, k=2, metrics=["hr", "precipitation", "recall", "ndcg"])  # type: ignore

    # User 1 predictions: 2, 3 -> HR=1, Recall=1, NDCG=1
    # User 2 predictions: 3, 4 -> HR=0, Recall=0, NDCG=0

    assert np.isclose(res["hr"], 0.5, atol=1e-6)
    assert np.isclose(res["recall"], 0.5, atol=1e-6)
    assert np.isclose(res["ndcg"], 0.5, atol=1e-6)


def test_evaluate_with_label_mapping():
    """evaluate() should map original labels to internal indices
    when model has _user_labels / _item_labels from from_transactions()."""
    from rusket import train_test_split

    rng = np.random.default_rng(42)
    n = 1000
    # Use non-zero-based IDs to simulate real-world scenario
    df = pd.DataFrame(
        {
            "user_id": rng.integers(1000, 1050, n),
            "item_id": rng.integers(5000, 5030, n),
            "rating": rng.uniform(0.1, 5.0, n).astype(np.float32),
        }
    )

    train_df, test_df = train_test_split(df, user_col="user_id", item_col="item_id", test_size=0.2)

    model = ALS.from_transactions(
        train_df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        factors=16,
        iterations=5,
        seed=42,
    ).fit()

    # Build test data with the *original* labels (not 0-based)
    eval_df = test_df.rename(columns={"user_id": "user", "item_id": "item"})[["user", "item"]]

    metrics = evaluate(model, eval_df, k=10)
    for name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric {name}={val} out of range"
    # At least some metrics should be non-zero
    assert any(v > 0 for v in metrics.values()), f"All metrics are zero: {metrics}"


def test_evaluate_with_unknown_users():
    """evaluate() should gracefully skip unknown users and warn."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "user_id": rng.integers(100, 130, n),
            "item_id": rng.integers(200, 220, n),
        }
    )

    model = ALS.from_transactions(
        df,
        user_col="user_id",
        item_col="item_id",
        factors=8,
        iterations=2,
        seed=42,
    ).fit()

    # Test data includes users NOT in the model
    test_df = pd.DataFrame(
        {
            "user": [100, 100, 999, 999],  # 999 is unknown
            "item": [200, 201, 200, 201],
        }
    )

    with pytest.warns(UserWarning, match="unknown user labels"):
        metrics = evaluate(model, test_df, k=5)

    # Should still compute metrics for the known user
    for name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric {name}={val} out of range"


def test_evaluate_with_large_realistic_ids():
    """Regression: evaluate must work when user/item IDs are large customer_sk values.

    Before the fix, raw IDs like 50234 were passed directly to
    recommend_items() which expects 0-based indices, causing a ValueError
    silently caught and returning all-zero metrics.
    """
    from rusket import train_test_split

    rng = np.random.default_rng(42)
    n_users, n_items, n_interactions = 30, 20, 400
    user_ids = rng.integers(50000, 90000, size=n_users)
    item_ids = rng.integers(100000, 200000, size=n_items)
    rows = []
    for _ in range(n_interactions):
        u = rng.choice(user_ids)
        i = rng.choice(item_ids)
        rows.append({"user_id": int(u), "part_id": int(i), "label": float(rng.uniform(0.1, 5.0))})
    df = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "part_id"]).reset_index(drop=True)

    train_df, test_df = train_test_split(df, user_col="user_id", item_col="part_id", test_size=0.2)
    model = ALS.from_transactions(
        train_df,
        user_col="user_id",
        item_col="part_id",
        rating_col="label",
        factors=8,
        iterations=5,
        seed=42,
    ).fit()

    eval_df = test_df.rename(columns={"user_id": "user", "part_id": "item"})
    metrics = evaluate(model, eval_df, k=5)

    for name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric {name}={val} out of range"
    assert any(v > 0.0 for v in metrics.values()), f"All metrics zero — label-to-index mapping broken: {metrics}"
