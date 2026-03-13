"""Optuna Bayesian hyperparameter optimisation for rusket models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


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
    enable_pruning: bool = False,
    callbacks: list[Any] | None = None,
    **study_kwargs: Any,
) -> Any:
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
                OptunaSearchSpace.categorical("popularity_weighting", ["none", "sqrt", "log", "linear"]),
                OptunaSearchSpace.categorical("use_biases", [True, False]),
                OptunaSearchSpace.int("anderson_m", 0, 5),
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
    enable_pruning : bool, default=False
        If ``True``, raises `optuna.TrialPruned` if intermediate evaluation score
        is deemed unpromising, saving evaluation time.
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
    from .model_selection import (
        CrossValidationResult,
        OptunaPruningCallback,
        cross_validate,
    )

    try:
        from rusket._dependencies import import_optional_dependency

        optuna = import_optional_dependency("optuna")
    except ImportError as e:
        raise ImportError("Optuna is required for optuna_optimize(). Install it with: pip install optuna") from e

    if search_space is None:
        search_space = [
            OptunaSearchSpace.int("factors", 16, 256, log=True),
            OptunaSearchSpace.float("alpha", 1.0, 100.0, log=True),
            OptunaSearchSpace.float("regularization", 1e-4, 1.0, log=True),
            OptunaSearchSpace.int("iterations", 5, 50),
            OptunaSearchSpace.categorical("use_eals", [True, False]),
            OptunaSearchSpace.categorical("popularity_weighting", ["none", "sqrt", "log", "linear"]),
            OptunaSearchSpace.categorical("use_biases", [True, False]),
            OptunaSearchSpace.int("anderson_m", 0, 5),
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

        cv_callbacks = [OptunaPruningCallback(trial)] if enable_pruning else None

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
            callbacks=cv_callbacks,
        )
        all_trial_results.append(result.results[0])

        if verbose:
            params_str = " ".join(f"{k_}={v}" for k_, v in params.items())
            logger.info(f"Trial {trial.number}: {metric}@{k}={result.best_score:.4f}  {params_str}")

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
        mlflow_cb = MLflowCallback(
            metric_name=metric,
            create_experiment=False,
            mlflow_kwargs={"nested": True},
        )
        all_callbacks.insert(0, mlflow_cb)

    # We temporarily suppress optuna's own verbose logging which conflicts
    # with our standard logging and tqdm progress bars.
    original_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use our own tqdm progress bar instead of Optuna's built-in one,
    # which can get stuck / show incorrect percentages.
    _pbar = None
    if verbose:
        try:
            from tqdm.auto import tqdm

            _pbar = tqdm(total=n_trials, desc="🔍 Optuna hyperparameter search", unit="trial")
        except ImportError:
            pass

    def _progress_callback(study: Any, trial: Any) -> None:  # noqa: ARG001
        if _pbar is not None:
            best = study.best_trial
            _pbar.set_postfix_str(f"best trial: {best.number}, best value: {best.value:.6f}")
            _pbar.update(1)

    if _pbar is not None:
        all_callbacks.append(_progress_callback)

    try:
        # Create or reuse study
        if study is None:
            study = optuna.create_study(direction="maximize", **study_kwargs)

        study.optimize(
            _objective,
            n_trials=n_trials,
            callbacks=all_callbacks or None,
            show_progress_bar=False,
        )
    finally:
        if _pbar is not None:
            _pbar.close()
        optuna.logging.set_verbosity(original_verbosity)

    best_params = study.best_trial.params
    best_score = study.best_trial.value or 0.0

    if verbose:
        logger.info(f"Best: {metric}@{k}={best_score:.4f}  params={best_params}")

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
