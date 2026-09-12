import numpy as np
import pandas as pd
from rusket.sasrec import SASRec


def test_sasrec_recommend_items_excludes_pad_id():
    """BUG 2 regression: Rust pads short result rows with (id=0, score=-inf)
    once exclude_seen has removed everything the user could be recommended.
    Item id 0 is not a real item (the item map is 1-indexed) and must never
    leak into recommend_items() output."""
    data = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [10, 20, 30]})
    model = SASRec.from_transactions(
        data, user_col="user_id", item_col="item_id", factors=8, iterations=2, seed=42
    ).fit()

    # User 0 has seen the entire (tiny) catalog, so with exclude_seen=True
    # and n larger than the catalog, Rust must pad the result rows.
    ids, scores = model.recommend_items(0, n=10, exclude_seen=True)

    assert 0 not in ids
    assert len(ids) == len(scores)


def test_sasrec_time_aware_from_transactions():
    data = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [10, 20, 30, 10, 20],
            # 1 day apart = 86400 seconds
            "timestamp": [
                1600000000,
                1600000000 + 86400,
                1600000000 + 86400 * 3,
                1600000000,
                1600000000 + 86400 * 5,
            ],
        }
    )

    # Fit a model without time-awareness to compare
    base_model = SASRec.from_transactions(
        data, user_col="user_id", item_col="item_id", factors=16, iterations=5, seed=42, time_aware=False
    ).fit()

    # Fit the new time-aware model
    time_model = SASRec.from_transactions(
        data,
        user_col="user_id",
        item_col="item_id",
        timestamp_col="timestamp",
        factors=16,
        iterations=5,
        seed=42,
        time_aware=True,
        max_time_steps=10,
    ).fit()

    assert time_model._time_emb is not None
    assert time_model._time_emb.shape == (10, 16)  # max_time_steps x factors

    # Evaluate predictions
    base_recs, base_scores = base_model.recommend_items(data["item_id"].tolist())
    time_recs, time_scores = time_model.recommend_items(data["item_id"].tolist(), timestamps=data["timestamp"].tolist())

    assert len(base_recs) > 0
    assert len(time_recs) > 0

    # We just ensure it runs cleanly; exact score comparison might be noisy depending on initialization.
    assert base_scores[0] != np.inf
    assert time_scores[0] != np.inf


def test_sasrec_time_aware_pandas_datetime():
    # Test handling of pandas datetime which rusket automatically converts to epoch
    data = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 20], "dt": pd.to_datetime(["2024-01-01", "2024-01-02"])})

    model = SASRec.from_transactions(
        data, user_col="user_id", item_col="item_id", timestamp_col="dt", factors=8, iterations=1, time_aware=True
    ).fit()

    assert model._time_emb is not None
    assert model._user_timestamps is not None
