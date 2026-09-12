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

    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 10]})

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


def test_predict_scores_call_does_not_raise_attribute_error():
    """Direct regression test for the `scores.scores.tolist()` bug (no save/load roundtrip)."""
    from rusket.export.mlflow import _get_rusket_wrapper_cls

    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 10]})
    model = ALS.from_transactions(df, factors=4, iterations=1, seed=42).fit()

    wrapper_cls = _get_rusket_wrapper_cls()
    wrapper = wrapper_cls()
    wrapper.model = model

    predictions = wrapper.predict(None, pd.DataFrame({"user": [1]}))

    assert len(predictions.loc[predictions["user"] == 1, "items"].iloc[0]) > 0
