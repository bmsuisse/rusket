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

    For **ALS** and **eALS** models the entire train→recommend→evaluate
    loop runs natively in Rust with rayon parallelism across parameter
    configurations — typically an order-of-magnitude faster than the
    Python path used for other model classes.

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

    # --- Try Rust fast-path for ALS/eALS ---
    from .als import ALS

    if issubclass(model_class, ALS):
        result = _cross_validate_rust(
            model_class=model_class,
            df=df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            param_combinations=param_combinations,
            n_folds=n_folds,
            k=k,
            metric=metric,
            metrics=metrics,
            refit_best=refit_best,
            verbose=verbose,
            seed=seed,
        )
        return result

    # --- Python fallback for non-ALS models ---
    return _cross_validate_python(
        model_class=model_class,
        df=df,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        param_combinations=param_combinations,
        n_folds=n_folds,
        k=k,
        metric=metric,
        metrics=metrics,
        refit_best=refit_best,
        verbose=verbose,
        seed=seed,
    )


def _cross_validate_rust(
    *,
    model_class: type,
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str | None,
    param_combinations: list[dict[str, Any]],
    n_folds: int,
    k: int,
    metric: str,
    metrics: list[str],
    refit_best: bool,
    verbose: bool,
    seed: int,
) -> CrossValidationResult:
    """Rust-accelerated cross-validation for ALS/eALS models."""
    import numpy as np
    import pandas as _pd

    from .als import ALS

    # --- Factorise user/item labels to 0-based codes ---
    u_data = df[user_col]
    i_data = df[item_col]
    user_codes, _user_uniques = _pd.factorize(u_data, sort=False)
    item_codes, _item_uniques = _pd.factorize(i_data, sort=True)
    n_users = int(_user_uniques.__len__())
    n_items = int(_item_uniques.__len__())

    users = user_codes.astype(np.int32).tolist()
    items = item_codes.astype(np.int32).tolist()
    if rating_col is not None:
        values = np.asarray(df[rating_col], dtype=np.float32).tolist()
    else:
        values = [1.0] * len(users)

    # --- Build flat param arrays for Rust ---
    # Get defaults from ALS
    defaults = ALS()
    # Check if model_class forces use_eals (eALS subclass)
    is_eals_class = model_class is not ALS
    eals_default = True if is_eals_class else defaults.use_eals

    factors_list: list[int] = []
    regularization_list: list[float] = []
    alpha_list: list[float] = []
    iterations_list: list[int] = []
    use_eals_list: list[bool] = []
    eals_iters_list: list[int] = []
    cg_iters_list: list[int] = []
    use_cholesky_list: list[bool] = []
    seed_list: list[int] = []

    for params in param_combinations:
        factors_list.append(int(params.get("factors", defaults.factors)))
        regularization_list.append(float(params.get("regularization", defaults.regularization)))
        alpha_list.append(float(params.get("alpha", defaults.alpha)))
        iterations_list.append(int(params.get("iterations", defaults.iterations)))
        use_eals_list.append(bool(params.get("use_eals", eals_default)))
        eals_iters_list.append(int(params.get("eals_iters", defaults.eals_iters)))
        cg_iters_list.append(int(params.get("cg_iters", defaults.cg_iters)))
        use_cholesky_list.append(bool(params.get("use_cholesky", defaults.use_cholesky)))
        seed_list.append(int(params.get("seed", seed)))

    # --- Call Rust cross_validate_als ---
    (
        best_idx,
        best_mean,
        per_config_means,
        per_config_stds,
        per_config_fold_scores,
    ) = _rusket.cross_validate_als(
        users,
        items,
        values,
        n_users,
        n_items,
        factors_list,
        regularization_list,
        alpha_list,
        iterations_list,
        use_eals_list,
        eals_iters_list,
        cg_iters_list,
        use_cholesky_list,
        seed_list,
        n_folds,
        k,
        metric,
        seed,
        verbose,
    )

    # --- Reconstruct CrossValidationResult ---
    metric_names = ["precision", "recall", "ndcg", "hr"]
    all_results: list[dict[str, Any]] = []

    for ci, params in enumerate(param_combinations):
        means = per_config_means[ci]
        stds = per_config_stds[ci]
        fold_raw = per_config_fold_scores[ci]

        fold_scores: list[dict[str, float]] = []
        for fold_vals in fold_raw:
            fold_dict: dict[str, float] = {}
            for mi, mn in enumerate(metric_names):
                if mn in metrics:
                    fold_dict[mn] = fold_vals[mi]
            fold_scores.append(fold_dict)

        result_entry: dict[str, Any] = {"params": params, "fold_scores": fold_scores}
        for mi, mn in enumerate(metric_names):
            if mn in metrics:
                result_entry[f"mean_{mn}"] = float(means[mi])
                result_entry[f"std_{mn}"] = float(stds[mi])
        all_results.append(result_entry)

    best_params = param_combinations[best_idx]

    if verbose:
        print(f"\n  Best: {metric}@{k}={best_mean:.4f}  params={best_params}")

    # Optionally refit on the full dataset
    best_model: Any = None
    if refit_best:
        from_kw: dict[str, Any] = {
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
        best_score=float(best_mean),
        results=all_results,
        best_model=best_model,
    )


def _cross_validate_python(
    *,
    model_class: type,
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str | None,
    param_combinations: list[dict[str, Any]],
    n_folds: int,
    k: int,
    metric: str,
    metrics: list[str],
    refit_best: bool,
    verbose: bool,
    seed: int,
) -> CrossValidationResult:
    """Pure-Python cross-validation for non-ALS models (parallelized via threads).

    All rusket models release the GIL during ``.fit()``, so
    ``ThreadPoolExecutor`` gives true parallelism without needing
    model-specific Rust CV code.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np

    from .evaluation import MetricName, evaluate

    n_configs = len(param_combinations)

    # --- create folds via shuffled row indices ---
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    folds: list[np.ndarray] = np.array_split(indices, n_folds)  # type: ignore[assignment]

    # --- Pre-build fold DataFrames (shared across configs) ---
    fold_data: list[tuple[Any, Any]] = []
    for fi in range(n_folds):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        fold_data.append((train_df, test_df))

    def _eval_one_config(ci_params: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        ci, params = ci_params
        fold_scores: list[dict[str, float]] = []

        for fi, (train_df, test_df) in enumerate(fold_data):
            from_kw: dict[str, Any] = {
                "user_col": user_col,
                "item_col": item_col,
                "seed": seed,
                **params,
            }
            if rating_col is not None:
                from_kw["rating_col"] = rating_col

            model = model_class.from_transactions(train_df, **from_kw).fit()

            eval_df = test_df.rename(columns={user_col: "user", item_col: "item"})
            scores = evaluate(model, eval_df, k=k, metrics=cast(list["MetricName"], metrics))
            fold_scores.append(scores)

            if verbose:
                primary = scores.get(metric, 0.0)
                params_str = " ".join(f"{k_}={v}" for k_, v in params.items()) if params else "(defaults)"
                print(f"  [{ci + 1}/{n_configs}] {params_str}  fold {fi + 1}/{n_folds}  {metric}@{k}={primary:.4f}")

            del model

        result_entry: dict[str, Any] = {"params": params, "fold_scores": fold_scores}
        for m in metrics:
            vals = [fs.get(m, 0.0) for fs in fold_scores]
            result_entry[f"mean_{m}"] = float(np.mean(vals))
            result_entry[f"std_{m}"] = float(np.std(vals))
        return result_entry

    # --- Run configs in parallel (threads work because .fit() releases the GIL) ---
    max_workers = min(n_configs, os.cpu_count() or 4)
    all_results: list[dict[str, Any]] = [{}] * n_configs  # pre-allocate order

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one_config, (ci, params)): ci for ci, params in enumerate(param_combinations)}
        for future in as_completed(futures):
            ci = futures[future]
            all_results[ci] = future.result()

    # --- Find best ---
    best_mean = -1.0
    best_params: dict[str, Any] = {}
    for entry in all_results:
        mean_primary = entry[f"mean_{metric}"]
        if mean_primary > best_mean:
            best_mean = mean_primary
            best_params = entry["params"]

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


# ---------------------------------------------------------------------------
# Optuna Bayesian optimisation
# ---------------------------------------------------------------------------


@dataclass
class OptunaSearchSpace:
    """Defines the search space for a single hyperparameter.

    Use the class methods to create search space entries::

        OptunaSearchSpace.int("factors", 16, 256, log=True)
        OptunaSearchSpace.float("regularization", 1e-4, 1.0, log=True)
        OptunaSearchSpace.categorical("use_eals", [True, False])

    Attributes
    ----------
    name : str
        The hyperparameter name (must match the model constructor kwarg).
    kind : str
        One of ``"int"``, ``"float"``, or ``"categorical"``.
    low : Any
        Lower bound (for int/float).
    high : Any
        Upper bound (for int/float).
    choices : list[Any] | None
        The list of choices (for categorical).
    log : bool
        Whether to sample in log-space (for int/float).
    step : int | float | None
        Step size (for int/float).
    """

    name: str
    kind: str = "float"
    low: Any = None
    high: Any = None
    choices: list[Any] | None = None
    log: bool = False
    step: int | float | None = None

    @classmethod
    def int(cls, name: str, low: int, high: int, *, log: bool = False, step: int | None = None) -> OptunaSearchSpace:
        """Integer parameter."""
        return cls(name=name, kind="int", low=low, high=high, log=log, step=step)

    @classmethod
    def float(
        cls, name: str, low: float, high: float, *, log: bool = False, step: float | None = None
    ) -> OptunaSearchSpace:
        """Float parameter."""
        return cls(name=name, kind="float", low=low, high=high, log=log, step=step)

    @classmethod
    def categorical(cls, name: str, choices: list[Any]) -> OptunaSearchSpace:
        """Categorical parameter."""
        return cls(name=name, kind="categorical", choices=choices)


def optuna_optimize(
    model_class: type,
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str | None = None,
    search_space: list[OptunaSearchSpace] | None = None,
    n_trials: int = 50,
    n_folds: int = 3,
    k: int = 10,
    metric: str = "precision",
    refit_best: bool = False,
    verbose: bool = True,
    seed: int = 42,
    study: Any = None,
    mlflow_tracking: bool = False,
    callbacks: list[Any] | None = None,
    **study_kwargs: Any,
) -> CrossValidationResult:
    """Bayesian hyperparameter optimisation using `Optuna <https://optuna.org>`_.

    Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to
    intelligently explore the hyperparameter space.  For **ALS** / **eALS**
    models, each trial runs the Rust-native cross-validation backend,
    making each evaluation extremely fast.

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
    search_space : list[OptunaSearchSpace] or None
        The search space.  If ``None``, a sensible default for ALS is used::

            [
                OptunaSearchSpace.int("factors", 16, 256, log=True),
                OptunaSearchSpace.float("alpha", 1.0, 100.0, log=True),
                OptunaSearchSpace.float("regularization", 1e-4, 1.0, log=True),
                OptunaSearchSpace.int("iterations", 5, 50),
                OptunaSearchSpace.categorical("use_eals", [True, False]),
            ]

    n_trials : int, default=50
        Number of Optuna trials.
    n_folds : int, default=3
        Number of cross-validation folds.
    k : int, default=10
        Cutoff for ranking metrics.
    metric : str, default="precision"
        Primary metric to maximise.
    refit_best : bool, default=False
        If ``True``, retrain the best configuration on the full dataset.
    verbose : bool, default=True
        Print progress.
    seed : int, default=42
        Random seed.
    study : optuna.Study or None
        An existing Optuna study to resume.  If ``None``, a new one is created.
    mlflow_tracking : bool, default=False
        If ``True``, log every trial's parameters and metrics to MLflow
        using ``optuna_integration.MLflowCallback``.  Requires the
        ``mlflow`` and ``optuna-integration`` packages.
    callbacks : list[Any] or None
        Extra Optuna callbacks passed to ``study.optimize()``.
    **study_kwargs
        Extra kwargs passed to ``optuna.create_study()`` (e.g. ``sampler``).

    Returns
    -------
    CrossValidationResult
        Same result type as :func:`cross_validate`.

    Examples
    --------
    >>> import rusket
    >>> result = rusket.optuna_optimize(
    ...     rusket.ALS,
    ...     df,
    ...     user_col="user_id",
    ...     item_col="item_id",
    ...     n_trials=30,
    ...     metric="ndcg",
    ... )
    >>> print(result.best_params)
    >>> print(result.best_score)

    With MLflow tracking::

        result = rusket.optuna_optimize(
            rusket.ALS, df,
            user_col="user_id", item_col="item_id",
            n_trials=50, mlflow_tracking=True,
        )
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Optuna is required for optuna_optimize(). Install it with: pip install optuna") from e

    if search_space is None:
        search_space = [
            OptunaSearchSpace.int("factors", 16, 256, log=True),
            OptunaSearchSpace.float("alpha", 1.0, 100.0, log=True),
            OptunaSearchSpace.float("regularization", 1e-4, 1.0, log=True),
            OptunaSearchSpace.int("iterations", 5, 50),
            OptunaSearchSpace.categorical("use_eals", [True, False]),
        ]

    all_trial_results: list[dict[str, Any]] = []

    def _objective(trial: Any) -> float:
        params: dict[str, Any] = {}
        for sp in search_space:  # type: ignore[union-attr]
            if sp.kind == "int":
                kw: dict[str, Any] = {"log": sp.log}
                if sp.step is not None:
                    kw["step"] = sp.step
                params[sp.name] = trial.suggest_int(sp.name, sp.low, sp.high, **kw)
            elif sp.kind == "float":
                kw = {"log": sp.log}
                if sp.step is not None:
                    kw["step"] = sp.step
                params[sp.name] = trial.suggest_float(sp.name, sp.low, sp.high, **kw)
            elif sp.kind == "categorical":
                params[sp.name] = trial.suggest_categorical(sp.name, sp.choices)

        result = cross_validate(
            model_class=model_class,
            df=df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            param_grid={k_: [v] for k_, v in params.items()},
            n_folds=n_folds,
            k=k,
            metric=metric,
            refit_best=False,
            verbose=False,
            seed=seed,
        )
        all_trial_results.append(result.results[0])

        if verbose:
            params_str = " ".join(f"{k_}={v}" for k_, v in params.items())
            print(f"  Trial {trial.number + 1}: {metric}@{k}={result.best_score:.4f}  {params_str}")

        return result.best_score

    # --- Build callback list ---
    all_callbacks: list[Any] = list(callbacks) if callbacks else []

    if mlflow_tracking:
        try:
            from optuna_integration import MLflowCallback
        except ImportError as e:
            raise ImportError(
                "MLflow tracking requires 'mlflow' and 'optuna-integration'. "
                "Install them with: pip install mlflow optuna-integration"
            ) from e
        mlflow_cb = MLflowCallback(metric_name=metric)
        all_callbacks.insert(0, mlflow_cb)

    # Create or reuse study
    if study is None:
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", **study_kwargs)

    study.optimize(_objective, n_trials=n_trials, callbacks=all_callbacks or None)

    best_params = study.best_trial.params
    best_score = study.best_trial.value or 0.0

    if verbose:
        print(f"\n  Best: {metric}@{k}={best_score:.4f}  params={best_params}")

    # Optionally refit on the full dataset
    best_model: Any = None
    if refit_best:
        from_kw: dict[str, Any] = {
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
        best_score=best_score,
        results=all_trial_results,
        best_model=best_model,
    )
