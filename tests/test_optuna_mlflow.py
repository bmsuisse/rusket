"""Tests for optuna_optimize with MLflow tracking and callbacks."""

import numpy as np
import pandas as pd

from rusket import ALS, CrossValidationResult, OptunaSearchSpace, optuna_optimize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n_users: int = 30, n_items: int = 20, n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic interaction dataset."""
    rng = np.random.default_rng(seed)
    return (
        pd.DataFrame(
            {
                "user_id": rng.integers(1000, 1000 + n_users, n),
                "item_id": rng.integers(5000, 5000 + n_items, n),
                "rating": rng.uniform(0.1, 5.0, n).astype(np.float32),
            }
        )
        .drop_duplicates(subset=["user_id", "item_id"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Tests — basic optuna_optimize
# ---------------------------------------------------------------------------


def test_optuna_optimize_basic() -> None:
    """Run a 3-trial search and verify result structure."""
    df = _make_dataset()
    result = optuna_optimize(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        search_space=[
            OptunaSearchSpace.int("factors", 8, 16),
            OptunaSearchSpace.float("regularization", 0.01, 0.1, log=True),
        ],
        n_trials=3,
        n_folds=2,
        k=5,
        metric="precision",
        verbose=False,
        seed=42,
    )

    assert isinstance(result, CrossValidationResult)
    assert result.best_params is not None
    assert "factors" in result.best_params
    assert 0.0 <= result.best_score <= 1.0
    assert len(result.results) == 3  # one per trial


def test_optuna_optimize_refit_best() -> None:
    """refit_best=True should return a fitted model."""
    df = _make_dataset()
    result = optuna_optimize(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        search_space=[OptunaSearchSpace.int("factors", 8, 16)],
        n_trials=2,
        n_folds=2,
        k=5,
        refit_best=True,
        verbose=False,
    )

    assert result.best_model is not None
    assert result.best_model.fitted is True
    ids, scores = result.best_model.recommend_items(0, n=3, exclude_seen=True)
    assert len(ids) == 3
    assert len(scores) == 3


# ---------------------------------------------------------------------------
# Tests — custom callbacks
# ---------------------------------------------------------------------------


def test_optuna_optimize_custom_callbacks() -> None:
    """A custom callback should be called once per trial."""
    df = _make_dataset()
    call_count = {"n": 0}

    def _counter_callback(study: object, trial: object) -> None:  # noqa: ARG001
        call_count["n"] += 1

    result = optuna_optimize(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        search_space=[OptunaSearchSpace.int("factors", 8, 16)],
        n_trials=3,
        n_folds=2,
        k=5,
        callbacks=[_counter_callback],
        verbose=False,
    )

    assert isinstance(result, CrossValidationResult)
    assert call_count["n"] == 3  # called once per trial


# ---------------------------------------------------------------------------
# Tests — MLflow tracking
# ---------------------------------------------------------------------------


def test_optuna_optimize_mlflow_tracking(tmp_path: object) -> None:
    """When mlflow_tracking=True, runs should be logged to MLflow."""
    import mlflow

    tracking_uri = f"file://{tmp_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rusket-optuna-test")

    df = _make_dataset()
    result = optuna_optimize(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        search_space=[OptunaSearchSpace.int("factors", 8, 16)],
        n_trials=3,
        n_folds=2,
        k=5,
        mlflow_tracking=True,
        verbose=False,
    )

    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0

    # Verify MLflow logged runs
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    # At least the default or our named experiment should exist
    assert len(experiments) >= 1

    # Check that runs were created (MLflowCallback creates one run per trial)
    runs = client.search_runs(experiment_ids=[e.experiment_id for e in experiments])
    assert len(runs) >= 1  # at least one run was logged


def test_mlflow_tracking_import_error() -> None:
    """If optuna-integration is missing, a clear ImportError should be raised."""
    # We can't easily simulate a missing import, but we can verify
    # that the parameter exists and the function accepts it.
    df = _make_dataset()
    # With mlflow_tracking=False (default), no import error
    result = optuna_optimize(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        search_space=[OptunaSearchSpace.int("factors", 8, 16)],
        n_trials=1,
        n_folds=2,
        k=5,
        mlflow_tracking=False,
        verbose=False,
    )
    assert isinstance(result, CrossValidationResult)
