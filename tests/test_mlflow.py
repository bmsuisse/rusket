"""Tests for the MLflow integration."""

import pandas as pd
import pytest

pytest.importorskip("mlflow")
from rusket.mlflow import autolog, save_model


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
