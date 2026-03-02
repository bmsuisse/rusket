"""pytest-benchmark: Rust vs Python cross-validation paths.

Run:
    uv run pytest tests/benchmarks/test_bench_cross_validate.py --benchmark-only -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rusket import ALS, CrossValidationResult, eALS
from rusket.model_selection import _cross_validate_python, _cross_validate_rust

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cv_dataset() -> pd.DataFrame:
    """Synthetic dataset: 100 users, 50 items, ~2k interactions."""
    rng = np.random.default_rng(42)
    n = 2000
    df = (
        pd.DataFrame(
            {
                "user_id": rng.integers(1000, 1100, n),
                "item_id": rng.integers(5000, 5050, n),
                "rating": rng.uniform(0.1, 5.0, n).astype(np.float32),
            }
        )
        .drop_duplicates(subset=["user_id", "item_id"])
        .reset_index(drop=True)
    )
    return df


PARAM_GRID = {"factors": [8, 16], "alpha": [10, 40], "regularization": [0.01, 0.1]}
# 2 × 2 × 2 = 8 configs, 2 folds each


# ── Benchmarks ──────────────────────────────────────────────────────


def test_bench_cv_rust(benchmark, cv_dataset: pd.DataFrame) -> None:
    """Cross-validation via the Rust backend (ALS)."""
    import itertools

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    param_combinations = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    def _run() -> CrossValidationResult:
        return _cross_validate_rust(
            model_class=ALS,
            df=cv_dataset,
            user_col="user_id",
            item_col="item_id",
            rating_col="rating",
            param_combinations=param_combinations,
            n_folds=2,
            k=5,
            metric="precision",
            metrics=["precision", "recall", "ndcg", "hr"],
            refit_best=False,
            verbose=False,
            seed=42,
        )

    result = benchmark(_run)
    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0


def test_bench_cv_python(benchmark, cv_dataset: pd.DataFrame) -> None:
    """Cross-validation via the Python fallback (ALS)."""
    import itertools

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    param_combinations = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    def _run() -> CrossValidationResult:
        return _cross_validate_python(
            model_class=ALS,
            df=cv_dataset,
            user_col="user_id",
            item_col="item_id",
            rating_col="rating",
            param_combinations=param_combinations,
            n_folds=2,
            k=5,
            metric="precision",
            metrics=["precision", "recall", "ndcg", "hr"],
            callbacks=None,
            refit_best=False,
            verbose=False,
            seed=42,
        )

    result = benchmark(_run)
    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0


def test_bench_cv_rust_eals(benchmark, cv_dataset: pd.DataFrame) -> None:
    """Cross-validation via the Rust backend (eALS)."""
    import itertools

    grid = {"factors": [8, 16], "alpha": [10, 40]}
    keys = list(grid.keys())
    values = list(grid.values())
    param_combinations = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    def _run() -> CrossValidationResult:
        return _cross_validate_rust(
            model_class=eALS,
            df=cv_dataset,
            user_col="user_id",
            item_col="item_id",
            rating_col="rating",
            param_combinations=param_combinations,
            n_folds=2,
            k=5,
            metric="ndcg",
            metrics=["precision", "recall", "ndcg", "hr"],
            callbacks=None,
            refit_best=False,
            verbose=False,
            seed=42,
        )

    result = benchmark(_run)
    assert isinstance(result, CrossValidationResult)
    assert result.best_score >= 0.0
