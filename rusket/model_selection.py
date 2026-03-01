"""Data preparation and model selection utilities."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd

from . import _rusket

# ---------------------------------------------------------------------------
# Data splitting helpers
# ---------------------------------------------------------------------------


def train_test_split(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into random train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The interaction dataframe.
    user_col : str
        Name of the user column.
    item_col : str
        Name of the item column.
    test_size : float, default=0.2
        Percentage of data to put in the test set.
    random_state : int, optional
        Set random seed (currently not used by Rust backend, but reserved for future).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    import numpy as np
    import pandas as _pd

    if not isinstance(df, _pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # Normally we do integer coercion when fitting, so here we assume
    # basic random splitting doesn't require actual encoded integers
    # but the Rust backend requires i32 for types (we use user_ids array just for length currently).
    # Since train_test_split algorithm randomly splits row indices, we can pass dummy int array

    dummy_ids = np.zeros(len(df), dtype=np.int32)
    train_idx, test_idx = _rusket.train_test_split(list(dummy_ids), test_size)  # type: ignore

    return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)


def leave_one_out_split(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave exactly one interaction per user for the test set.

    If a timestamp column is provided, the latest interaction is left out.
    If no timestamp is provided, a random interaction is chosen.

    Parameters
    ----------
    df : pd.DataFrame
        The interaction dataframe.
    user_col : str
        Name of the user column (must be numeric encoded to i32 ideally, or pandas int).
    item_col : str
        Name of the item column.
    timestamp_col : str, optional
        Name of the timestamp or ordering column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    import numpy as np
    import pandas as _pd

    if not isinstance(df, _pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # We need to ensure we can cast user IDs to int32 for the rust backend.
    try:
        user_ids = df[user_col].values.astype(np.int32)
        item_ids = df[item_col].values.astype(np.int32)
    except ValueError as e:
        raise ValueError(
            f"Columns {user_col} and {item_col} must be numeric/integer to use leave_one_out_split."
        ) from e

    timestamps = None
    if timestamp_col is not None:
        try:
            timestamps = list(df[timestamp_col].values.astype(np.float32))
        except ValueError as e:
            raise ValueError(f"Column {timestamp_col} must be numeric float to use leave_one_out_split.") from e

    train_idx, test_idx = _rusket.leave_one_out(list(user_ids), list(item_ids), timestamps)  # type: ignore

    return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationResult:
    """Result of a cross-validation hyperparameter search.

    Attributes
    ----------
    best_params : dict[str, Any]
        Hyperparameter configuration that maximised the target metric.
    best_score : float
        Mean score of the best configuration across folds.
    results : list[dict[str, Any]]
        Full results for every parameter configuration. Each dict contains
        ``"params"``, ``"mean_<metric>"``, ``"std_<metric>"`` for all
        computed metrics, and ``"fold_scores"`` with per-fold detail.
    best_model : Any | None
        If ``refit_best=True`` was passed to :func:`cross_validate`, this
        holds the model retrained on the **full** dataset with the best
        parameters.  Otherwise ``None``.
    """

    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=list)
    best_model: Any | None = None


def cross_validate(
    model_class: type,
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    n_folds: int = 3,
    k: int = 10,
    metric: str = "precision",
    metrics: list[str] | None = None,
    refit_best: bool = False,
    verbose: bool = True,
    seed: int = 42,
) -> CrossValidationResult:
    """K-fold grid-search cross-validation for implicit recommenders.

    Searches over every combination in *param_grid*, evaluating each one
    with *n_folds*-fold cross-validation.  The configuration that
    maximises the *metric* (averaged over folds) is returned as the best.

    Parameters
    ----------
    model_class : type
        The recommender class (e.g. ``ALS``, ``eALS``, ``BPR``).
    df : pd.DataFrame
        The full interaction dataframe.
    user_col : str
        Name of the user column.
    item_col : str
        Name of the item column.
    rating_col : str or None, default=None
        Name of the rating/confidence column (pass ``None`` for binary).
    param_grid : dict[str, list[Any]] or None
        Hyperparameter grid.  Keys are constructor kwargs, values are
        lists of values to try.  Example::

            {"factors": [32, 64], "alpha": [10, 40], "regularization": [0.01, 0.1]}

        If ``None`` or empty, a single run with the class defaults is
        performed (useful for getting baseline metrics).
    n_folds : int, default=3
        Number of cross-validation folds.
    k : int, default=10
        Cutoff for ranking metrics (precision@k, ndcg@k, …).
    metric : str, default="precision"
        Primary metric to maximise.  One of ``"precision"``, ``"recall"``,
        ``"ndcg"``, ``"hr"``.
    metrics : list[str] or None
        All metrics to compute per fold.  Defaults to
        ``["precision", "recall", "ndcg", "hr"]``.
    refit_best : bool, default=False
        If ``True``, retrain the best configuration on the entire dataset
        and store it in :attr:`CrossValidationResult.best_model`.
    verbose : bool, default=True
        Print progress to stdout.
    seed : int, default=42
        Random seed for shuffling the folds.

    Returns
    -------
    CrossValidationResult
        Object containing best params, best score, full per-config results,
        and optionally a refitted model.

    Examples
    --------
    >>> import rusket
    >>> result = rusket.cross_validate(
    ...     rusket.ALS,
    ...     df,
    ...     user_col="user_id",
    ...     item_col="item_id",
    ...     param_grid={"factors": [32, 64], "alpha": [10, 40]},
    ...     n_folds=3,
    ...     metric="precision",
    ... )
    >>> print(result.best_params)
    >>> print(result.best_score)
    """
    import numpy as np

    from .evaluation import MetricName, evaluate

    if metrics is None:
        metrics = ["precision", "recall", "ndcg", "hr"]

    if metric not in metrics:
        metrics = [metric, *metrics]

    # --- build parameter combinations ---
    if param_grid is None or len(param_grid) == 0:
        param_combinations: list[dict[str, Any]] = [{}]
    else:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        param_combinations = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    n_configs = len(param_combinations)

    # --- create folds via shuffled row indices ---
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    folds: list[np.ndarray] = np.array_split(indices, n_folds)  # type: ignore[assignment]

    all_results: list[dict[str, Any]] = []
    best_mean = -1.0
    best_params: dict[str, Any] = {}

    for ci, params in enumerate(param_combinations, 1):
        fold_scores: list[dict[str, float]] = []

        for fi in range(n_folds):
            test_idx = folds[fi]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # Build & fit model
            from_kw: dict[str, Any] = {
                "user_col": user_col,
                "item_col": item_col,
                "seed": seed,
                **params,
            }
            if rating_col is not None:
                from_kw["rating_col"] = rating_col

            model = model_class.from_transactions(train_df, **from_kw).fit()

            # Evaluate
            eval_df = test_df.rename(columns={user_col: "user", item_col: "item"})
            scores = evaluate(model, eval_df, k=k, metrics=cast(list["MetricName"], metrics))
            fold_scores.append(scores)

            if verbose:
                primary = scores.get(metric, 0.0)
                params_str = " ".join(f"{k_}={v}" for k_, v in params.items()) if params else "(defaults)"
                print(f"  [{ci}/{n_configs}] {params_str}  fold {fi + 1}/{n_folds}  {metric}@{k}={primary:.4f}")

            del model

        # Aggregate across folds
        result_entry: dict[str, Any] = {"params": params, "fold_scores": fold_scores}
        for m in metrics:
            vals = [fs.get(m, 0.0) for fs in fold_scores]
            result_entry[f"mean_{m}"] = float(np.mean(vals))
            result_entry[f"std_{m}"] = float(np.std(vals))

        all_results.append(result_entry)

        mean_primary = result_entry[f"mean_{metric}"]
        if mean_primary > best_mean:
            best_mean = mean_primary
            best_params = params

    if verbose:
        print(f"\n  Best: {metric}@{k}={best_mean:.4f}  params={best_params}")

    # Optionally refit on the full dataset
    best_model: Any = None
    if refit_best:
        from_kw = {
            "user_col": user_col,
            "item_col": item_col,
            "seed": seed,
            **best_params,
        }
        if rating_col is not None:
            from_kw["rating_col"] = rating_col
        best_model = model_class.from_transactions(df, **from_kw).fit()

    return CrossValidationResult(
        best_params=best_params,
        best_score=best_mean,
        results=all_results,
        best_model=best_model,
    )
