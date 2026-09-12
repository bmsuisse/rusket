"""Tests for the MLflow integration."""

import pandas as pd
import pytest

pytest.importorskip("mlflow")
from rusket import ALS
from rusket.export.mlflow import autolog, save_model


def test_autolog():
    import mlflow

    from rusket import ALS

    autolog()

    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 10]})

    mlflow.set_experiment("test_experiment")

    with mlflow.start_run() as run:
        model = ALS.from_transactions(df, factors=4, iterations=1, seed=42)
        model.fit()

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run.info.run_id).data

    assert "factors" in run_data.params
    assert run_data.params["factors"] == "4"
    assert "iterations" in run_data.params
    assert run_data.params["iterations"] == "1"

    assert "training_duration_seconds" in run_data.metrics

    # disable to not affect other tests
    autolog(disable=True)


def test_save_load_model(tmp_path):
    import mlflow.pyfunc

    from rusket import ALS

    # item 30 is never bought by user 1, so it stays available as a
    # recommendation after exclude_seen (with only items 10/20, user 1 would
    # have bought everything and legitimately get zero recommendations).
    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 30]})

    model = ALS.from_transactions(df, factors=4, iterations=1, seed=42).fit()

    model_path = tmp_path / "mlflow_model"

    save_model(model, str(model_path))

    loaded_model = mlflow.pyfunc.load_model(str(model_path))

    # Test predict method
    input_df = pd.DataFrame(
        {
            "user_id": [1, 2, 3]  # 3 is unseen
        }
    )

    predictions = loaded_model.predict(input_df)

    assert len(predictions) == 3
    assert "scores" in predictions.columns
    assert "items" in predictions.columns
    assert "user" in predictions.columns

    # Regression: predict() used to call `scores.scores.tolist()` on an ndarray
    # (no `.scores` attribute), and the resulting AttributeError was swallowed
    # by a bare `except`, so every prediction silently came back empty.
    row = predictions.loc[predictions["user"] == 1].iloc[0]
    assert len(row["items"]) > 0
    assert len(row["scores"]) > 0

    # user 3 was never seen during fit — recommend_items rejects it as
    # out-of-range, and that (specific, expected) failure should still come
    # back as an empty recommendation instead of blowing up the whole batch.
    unseen_row = predictions.loc[predictions["user"] == 3].iloc[0]
    assert unseen_row["items"] == []
    assert unseen_row["scores"] == []


def test_predict_maps_external_labels_to_internal_indices():
    """Regression: predict() used to pass the external user label straight to
    ``recommend_items()``, which expects an internal 0-based row index. With
    non-trivial (non 0..n-1) string labels this either scores the wrong user
    or raises, and returned item indices were never mapped back to external
    item labels either.
    """
    from rusket.export.mlflow import _get_rusket_wrapper_cls

    # String user/item ids, deliberately not 0..n-1 and not sorted the same
    # as any obvious index assignment, so mixing up label vs. index changes
    # behavior visibly (either the wrong user's items, or a raised ValueError
    # for an out-of-range index, or item ids that are still internal indices).
    df = pd.DataFrame(
        {
            "user_id": ["cust_9", "cust_9", "cust_1", "cust_1", "cust_5"],
            "item_id": ["sku_z", "sku_y", "sku_x", "sku_z", "sku_x"],
        }
    )
    model = ALS.from_transactions(df, factors=4, iterations=1, seed=42).fit()

    wrapper_cls = _get_rusket_wrapper_cls()
    wrapper = wrapper_cls()
    wrapper.model = model

    predictions = wrapper.predict(None, pd.DataFrame({"user": ["cust_9"]}))
    row = predictions.loc[predictions["user"] == "cust_9"].iloc[0]

    assert len(row["items"]) > 0
    # Returned items must be external item labels, not raw internal indices.
    assert set(row["items"]).issubset(set(model._item_labels))

    # An unknown external user id must come back as "no recommendations"
    # rather than being silently coerced to some internal index.
    unseen = wrapper.predict(None, pd.DataFrame({"user": ["cust_unknown"]}))
    unseen_row = unseen.loc[unseen["user"] == "cust_unknown"].iloc[0]
    assert unseen_row["items"] == []
    assert unseen_row["scores"] == []


def test_predict_scores_call_does_not_raise_attribute_error():
    """Direct regression test for the `scores.scores.tolist()` bug (no save/load roundtrip)."""
    from rusket.export.mlflow import _get_rusket_wrapper_cls

    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 30]})
    model = ALS.from_transactions(df, factors=4, iterations=1, seed=42).fit()

    wrapper_cls = _get_rusket_wrapper_cls()
    wrapper = wrapper_cls()
    wrapper.model = model

    predictions = wrapper.predict(None, pd.DataFrame({"user": [1]}))

    assert len(predictions.loc[predictions["user"] == 1, "items"].iloc[0]) > 0
