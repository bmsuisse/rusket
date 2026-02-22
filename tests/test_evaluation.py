import numpy as np
import pandas as pd

from rusket import evaluate
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
    def recommend_users(self, user_ids, n=10, filter_already_liked_items=True):
        # Recommend items (user_id + i) for i in 1..n
        r_users = []
        r_items = []
        r_scores = []
        for u in user_ids:
            for i in range(1, n + 1):
                r_users.append(u)
                r_items.append(u + i)
                r_scores.append(1.0)
        return np.array(r_users), np.array(r_items), np.array(r_scores)


def test_python_evaluate_wrapper():
    model = MockRecommender()
    
    # Ground truth:
    # User 1 likes 2, 3
    # User 2 likes 99 (not in recommendations)
    test_df = pd.DataFrame({
        "user": [1, 1, 2],
        "item": [2, 3, 99]
    })

    res = evaluate(model, test_df, k=2, metrics=["hr", "precipitation", "recall", "ndcg"])  # type: ignore
    
    # User 1 predictions: 2, 3 -> HR=1, Recall=1, NDCG=1
    # User 2 predictions: 3, 4 -> HR=0, Recall=0, NDCG=0
    
    assert np.isclose(res["hr"], 0.5, atol=1e-6)
    assert np.isclose(res["recall"], 0.5, atol=1e-6)
    assert np.isclose(res["ndcg"], 0.5, atol=1e-6)
