"""Tests for the cross_validate grid-search utility."""

import numpy as np
import pandas as pd

from rusket import ALS, CrossValidationResult, cross_validate, eALS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n_users: int = 40, n_items: int = 25, n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic interaction dataset."""
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
# Tests
# ---------------------------------------------------------------------------


def test_basic_cross_validate():
    """Two param combos, 2 folds — verify structure and that best params are selected."""
    df = _make_dataset()

    result = cross_validate(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        param_grid={"factors": [8, 16]},
        n_folds=2,
        k=5,
        metric="precision",
        verbose=False,
    )

    assert isinstance(result, CrossValidationResult)
    assert result.best_params is not None
    assert "factors" in result.best_params
    assert result.best_params["factors"] in [8, 16]
    assert 0.0 <= result.best_score <= 1.0
    assert len(result.results) == 2  # 2 param combos
    for entry in result.results:
        assert "params" in entry
        assert "mean_precision" in entry
        assert "std_precision" in entry
        assert "fold_scores" in entry
        assert len(entry["fold_scores"]) == 2  # 2 folds
    # best_model should be None when refit_best=False (default)
    assert result.best_model is None


def test_cross_validate_returns_best_model():
    """refit_best=True should return a fitted model."""
    df = _make_dataset()

    result = cross_validate(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        param_grid={"factors": [8]},
        n_folds=2,
        k=5,
        refit_best=True,
        verbose=False,
    )

    assert result.best_model is not None
    assert result.best_model.fitted is True
    # The refitted model should be able to recommend
    ids, scores = result.best_model.recommend_items(0, n=3, exclude_seen=True)
    assert len(ids) == 3
    assert len(scores) == 3


def test_cross_validate_single_config():
    """One config → returns that config as best."""
    df = _make_dataset()

    result = cross_validate(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        param_grid={"factors": [8], "iterations": [3]},
        n_folds=2,
        k=5,
        verbose=False,
    )

    assert result.best_params == {"factors": 8, "iterations": 3}
    assert len(result.results) == 1


def test_cross_validate_custom_metric():
    """Run with metric='ndcg' to verify alternative target works."""
    df = _make_dataset()

    result = cross_validate(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        param_grid={"factors": [8, 16]},
        n_folds=2,
        k=5,
        metric="ndcg",
        verbose=False,
    )

    assert 0.0 <= result.best_score <= 1.0
    # All results should have ndcg means
    for entry in result.results:
        assert "mean_ndcg" in entry
        assert "std_ndcg" in entry


def test_cross_validate_no_param_grid():
    """No grid → single default run."""
    df = _make_dataset()

    result = cross_validate(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        n_folds=2,
        k=5,
        verbose=False,
    )

    assert len(result.results) == 1
    assert result.best_params == {}
    assert 0.0 <= result.best_score <= 1.0


def test_cross_validate_with_eals():
    """Verify cross_validate works with eALS subclass."""
    df = _make_dataset()

    result = cross_validate(
        eALS,
        df,
        user_col="user_id",
        item_col="item_id",
        param_grid={"factors": [8]},
        n_folds=2,
        k=5,
        verbose=False,
    )

    assert isinstance(result, CrossValidationResult)
    assert 0.0 <= result.best_score <= 1.0


def test_optuna_optimize_basic():
    """Optuna Bayesian HP search — 5 trials, verify result structure."""
    optuna = __import__("pytest").importorskip("optuna")  # noqa: F841
    from rusket import OptunaSearchSpace, optuna_optimize

    df = _make_dataset()

    result = optuna_optimize(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        search_space=[
            OptunaSearchSpace.int("factors", 8, 16),
            OptunaSearchSpace.float("regularization", 0.01, 0.1, log=True),
        ],
        n_trials=5,
        n_folds=2,
        k=5,
        metric="precision",
        verbose=False,
        seed=42,
    )

    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0
    assert "factors" in result.best_params
    assert "regularization" in result.best_params
    assert len(result.results) == 5  # 5 trials


def test_optuna_optimize_refit():
    """Optuna with refit_best=True returns a fitted model."""
    optuna = __import__("pytest").importorskip("optuna")  # noqa: F841
    from rusket import OptunaSearchSpace, optuna_optimize

    df = _make_dataset()

    result = optuna_optimize(
        ALS,
        df,
        user_col="user_id",
        item_col="item_id",
        search_space=[
            OptunaSearchSpace.int("factors", 8, 16),
        ],
        n_trials=3,
        n_folds=2,
        k=5,
        refit_best=True,
        verbose=False,
        seed=42,
    )

    assert result.best_model is not None
    assert result.best_model.fitted


def test_cross_validate_bpr_rust():
    """BPR should go through the Rust generic CV path."""
    from rusket import BPR

    df = _make_dataset()
    result = cross_validate(
        BPR,
        df,
        user_col="user_id",
        item_col="item_id",
        param_grid={"factors": [8, 16], "iterations": [5]},
        n_folds=2,
        k=5,
        verbose=False,
    )
    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0
    assert len(result.results) == 2


def test_cross_validate_svd_rust():
    """SVD should go through the Rust generic CV path."""
    from rusket import SVD

    df = _make_dataset()
    result = cross_validate(
        SVD,
        df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        param_grid={"factors": [8], "iterations": [5]},
        n_folds=2,
        k=5,
        verbose=False,
    )
    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0


def test_cross_validate_lightgcn_rust():
    """LightGCN should go through the Rust generic CV path."""
    from rusket import LightGCN

    df = _make_dataset()
    result = cross_validate(
        LightGCN,
        df,
        user_col="user_id",
        item_col="item_id",
        param_grid={"factors": [8], "iterations": [3], "k_layers": [2]},
        n_folds=2,
        k=5,
        verbose=False,
    )
    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0
