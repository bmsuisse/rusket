import numpy as np
import pandas as pd

from rusket import ALS, Recommender, score_potential


def test_recommender_recommend_for_user():
    als = ALS()
    als._user_factors = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    als._item_factors = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    als._n_users = 1
    als.fitted = True

    # We need mock indptr and indices to test exclude_seen
    als._fit_indptr = np.array([0, 1], dtype=np.int64)
    als._fit_indices = np.array([0], dtype=np.int32)  # user 0 has seen item 0

    rec = Recommender(model=als)
    items, scores = rec.recommend_for_user(user_id=0, n=2)

    # User vector [1,0] dot item 1 [0.5, 0.5] = 0.5
    # User vector [1,0] dot item 2 [0.0, 1.0] = 0.0
    # User vector [1,0] dot item 0 [1.0, 0.0] = 1.0 (but excluded)
    assert len(items) == 2
    assert items[0] == 1
    assert items[1] == 2
    assert scores[0] == 0.5
    assert scores[1] == 0.0


def test_recommender_recommend_for_cart():
    rules_df = pd.DataFrame(
        {
            "antecedents": [(0,), (0, 1), (2,)],
            "consequents": [(1,), (2, 3), (0,)],
            "lift": [2.0, 5.0, 1.5],
            "confidence": [0.8, 0.9, 0.5],
        }
    )

    rec = Recommender(rules_df=rules_df)

    # Cart has item 0
    sugg = rec.recommend_for_cart([0], n=2)
    assert sugg == [1]  # Because only the first rule applies

    # Cart has items 0 and 1
    sugg = rec.recommend_for_cart([0, 1], n=5)
    # The second rule has highest lift (5.0), so we expect [2, 3] first.
    # Then the first rule also applies, giving [1], but 1 is in cart, so ignored.
    assert sugg == [2, 3]


def test_score_potential():
    als = ALS()
    als._user_factors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    als._item_factors = np.array(
        [
            [1.0, 0.0],  # Item 0
            [1.0, 1.0],  # Item 1
            [0.0, 1.0],  # Item 2
        ],
        dtype=np.float32,
    )
    als.fitted = True

    user_history = [[0], [2]]  # User 0 bought 0. User 1 bought 2.
    scores = score_potential(user_history, als)

    # Expected shapes
    assert scores.shape == (2, 3)

    # User 0
    assert scores[0, 0] == -np.inf  # Excluded
    assert scores[0, 1] == 1.0  # [1, 0] dot [1, 1]
    assert scores[0, 2] == 0.0  # [1, 0] dot [0, 1]

    # User 1
    assert scores[1, 0] == 0.0
    assert scores[1, 1] == 1.0
    assert scores[1, 2] == -np.inf


def test_score_potential_subset():
    als = ALS()
    als._user_factors = np.array([[1.0, 0.0]], dtype=np.float32)
    als._item_factors = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    als.fitted = True

    user_history = [[1]]
    scores = score_potential(user_history, als, target_categories=[0, 1])

    assert scores.shape == (1, 2)
    assert scores[0, 0] == 1.0
    assert scores[0, 1] == -np.inf  # item 1 is mapped to local idx 1 and excluded
