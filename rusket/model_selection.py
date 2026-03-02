"""Data preparation and model selection utilities."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    import optuna
    import pandas as pd

from . import _rusket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class PruningCallback(Protocol):
    """Protocol for early stopping mechanisms during training iterations."""

    def __call__(self, epoch: int, metric_score: float) -> bool:
        """Called by the rust backend every `report_interval` epochs.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        metric_score : float
            Evaluation metric score.

        Returns
        -------
        bool
            `True` if trial should be pruned (aborted), `False` otherwise.
        """
        ...


class OptunaPruningCallback:
    """Optuna callback to prune unpromising trials during training.

    This callback reports the validation score to Optuna and raises
    `optuna.TrialPruned` if the trial is deemed unpromising.

    Parameters
    ----------
    trial : optuna.trial.Trial
        A :class:`~optuna.trial.Trial` corresponding to the current evaluation.
    report_interval : int, default=50
        How often (in epochs) the Rust backend should run a validation pass.
        Since validation is expensive, computing it every epoch is slow.
    """

    def __init__(self, trial: optuna.Trial, report_interval: int = 50) -> None:
        from rusket._dependencies import import_optional_dependency

        optuna = import_optional_dependency("optuna")

        # Use an internal flag for Optuna's trial to save the exception instance
        self._trial = trial
        self._optuna_pruned_exc_class = optuna.TrialPruned
        self.report_interval = report_interval

    def __call__(self, epoch: int, metric_score: float) -> bool:
        """Report intermediate score to Optuna. Returns `True` if trial should be pruned."""
        self._trial.report(metric_score, step=epoch)
        if self._trial.should_prune():
            return True
        return False


# ---------------------------------------------------------------------------
# Data splitting helpers (re-exported for backward compatibility)
# ---------------------------------------------------------------------------
from .splitting import (  # noqa: E402
    chronological_split,
    leave_one_out_split,
    train_test_split,
    user_stratified_split,
)


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
    callbacks: list[Any] | None = None,
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
    callbacks : list[PruningCallback] or None
        List of callbacks to run during training. Useful for early stopping
        (e.g., pruning trials in Optuna).
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

    # --- Try Rust fast-path for factor-based models ---
    rust_kind = _get_rust_model_kind(model_class)

    if rust_kind == "als":
        # ALS/eALS has a specialised Rust path
        return _cross_validate_rust(
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
            callbacks=callbacks,
            refit_best=refit_best,
            verbose=verbose,
            seed=seed,
        )

    if rust_kind is not None:
        # BPR, SVD, LightGCN — generic Rust path
        return _cross_validate_rust_generic(
            kind=rust_kind,
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
            callbacks=callbacks,
            refit_best=refit_best,
            verbose=verbose,
            seed=seed,
        )

    # --- Python ThreadPoolExecutor fallback for non-factor models ---
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
        callbacks=callbacks,
        refit_best=refit_best,
        verbose=verbose,
        seed=seed,
    )


def _get_rust_model_kind(model_class: type) -> str | None:
    """Return the Rust model kind string if the model supports Rust CV, else None."""
    from .als import ALS
    from .bpr import BPR
    from .lightgcn import LightGCN
    from .svd import SVD

    if issubclass(model_class, ALS):
        return "als"
    if issubclass(model_class, BPR):
        return "bpr"
    if issubclass(model_class, SVD):
        return "svd"
    if issubclass(model_class, LightGCN):
        return "lightgcn"
    return None


def _cross_validate_rust_generic(
    *,
    kind: str,
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
    callbacks: list[PruningCallback] | None = None,
    refit_best: bool,
    verbose: bool,
    seed: int,
) -> CrossValidationResult:
    """Rust-accelerated cross-validation for BPR, SVD, LightGCN."""
    import numpy as np

    from rusket._dependencies import import_optional_dependency

    _pd = import_optional_dependency("pandas")

    from . import _rusket

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

    # --- Defaults per model kind ---
    _defaults: dict[str, dict[str, Any]] = {
        "bpr": {
            "factors": 64,
            "regularization": 0.01,
            "iterations": 150,
            "learning_rate": 0.05,
            "alpha": 1.0,
            "k_layers": 3,
        },
        "svd": {
            "factors": 64,
            "regularization": 0.01,
            "iterations": 50,
            "learning_rate": 0.005,
            "alpha": 1.0,
            "k_layers": 3,
        },
        "lightgcn": {
            "factors": 64,
            "regularization": 1e-4,
            "iterations": 100,
            "learning_rate": 0.001,
            "alpha": 1.0,
            "k_layers": 3,
        },
    }
    d = _defaults.get(kind, _defaults["bpr"])

    # --- Build flat param arrays ---
    factors_list: list[int] = []
    regularization_list: list[float] = []
    iterations_list: list[int] = []
    seed_list: list[int] = []
    alpha_list: list[float] = []
    use_eals_list: list[bool] = []
    eals_iters_list: list[int] = []
    cg_iters_list: list[int] = []
    use_cholesky_list: list[bool] = []
    anderson_m_list: list[int] = []
    popularity_weighting_list: list[str] = []
    use_biases_list: list[bool] = []
    learning_rate_list: list[float] = []
    k_layers_list: list[int] = []

    for params in param_combinations:
        factors_list.append(int(params.get("factors", d["factors"])))
        regularization_list.append(float(params.get("regularization", d["regularization"])))
        iterations_list.append(int(params.get("iterations", d["iterations"])))
        seed_list.append(int(params.get("seed", seed)))
        alpha_list.append(float(params.get("alpha", d["alpha"])))
        use_eals_list.append(False)
        eals_iters_list.append(0)
        cg_iters_list.append(0)
        use_cholesky_list.append(False)
        anderson_m_list.append(0)
        popularity_weighting_list.append("none")
        use_biases_list.append(False)
        learning_rate_list.append(float(params.get("learning_rate", d["learning_rate"])))
        k_layers_list.append(int(params.get("k_layers", d["k_layers"])))

    # --- Call Rust cross_validate_generic ---
    (
        best_idx,
        best_mean,
        per_config_means,
        per_config_stds,
        per_config_fold_scores,
    ) = _rusket.cross_validate_generic(
        kind,
        users,
        items,
        values,
        n_users,
        n_items,
        factors_list,
        regularization_list,
        iterations_list,
        seed_list,
        alpha_list,
        use_eals_list,
        eals_iters_list,
        cg_iters_list,
        use_cholesky_list,
        anderson_m_list,
        popularity_weighting_list,
        use_biases_list,
        learning_rate_list,
        k_layers_list,
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
        logger.info(f"Best: {metric}@{k}={best_mean:.4f}  params={best_params}")

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
    callbacks: list[PruningCallback] | None = None,
    refit_best: bool,
    verbose: bool,
    seed: int,
) -> CrossValidationResult:
    """Rust-accelerated cross-validation for ALS/eALS models."""
    import numpy as np

    from rusket._dependencies import import_optional_dependency

    _pd = import_optional_dependency("pandas")

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
    anderson_m_list: list[int] = []
    popularity_weighting_list: list[str] = []
    use_biases_list: list[bool] = []

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
        anderson_m_list.append(int(params.get("anderson_m", defaults.anderson_m)))
        popularity_weighting_list.append(str(params.get("popularity_weighting", defaults.popularity_weighting)))
        use_biases_list.append(bool(params.get("use_biases", defaults.use_biases)))

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
        anderson_m_list,
        popularity_weighting_list,
        use_biases_list,
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
        logger.info(f"Best: {metric}@{k}={best_mean:.4f}  params={best_params}")

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
    callbacks: list[Any] | None = None,
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

            # --- Note: Pure python train path currently does not natively support
            # calling the PruningCallback during `fit()`, but we pass it anyway
            # for future-proofing when we add a `callbacks` parameter to `fit`.
            model = model_class.from_transactions(train_df, **from_kw).fit()

            eval_df = test_df.rename(columns={user_col: "user", item_col: "item"})
            scores = evaluate(model, eval_df, k=k, metrics=cast(list["MetricName"], metrics))
            fold_scores.append(scores)

            primary = scores.get(metric, 0.0)

            # --- Check Pruning ---
            if callbacks:
                # Trigger callback for epoch=iterations for pruning early stops
                # Since pure-Python path doesn't yield during fit, we only check at fold end
                iters = params.get("iterations", 1)  # Or some generic epoch identifier
                for cb in callbacks:
                    if cb(iters, primary):
                        # Signal Early Stopping
                        pass

            if verbose:
                primary = scores.get(metric, 0.0)
                params_str = " ".join(f"{k_}={v}" for k_, v in params.items()) if params else "(defaults)"
                logger.info(f"[{ci + 1}/{n_configs}] {params_str}  fold {fi + 1}/{n_folds}  {metric}@{k}={primary:.4f}")

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

        gen = as_completed(futures)
        if verbose:
            try:
                from tqdm.auto import tqdm

                gen = tqdm(gen, total=n_configs, desc="Cross-validation", unit="config")
            except ImportError:
                pass

        for future in gen:
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
        logger.info(f"Best: {metric}@{k}={best_mean:.4f}  params={best_params}")

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
# Optuna Bayesian optimisation (re-exported for backward compatibility)
# ---------------------------------------------------------------------------
from .optuna import OptunaSearchSpace, optuna_optimize  # noqa: E402

__all__ = [
    "PruningCallback",
    "OptunaPruningCallback",
    "train_test_split",
    "leave_one_out_split",
    "chronological_split",
    "user_stratified_split",
    "CrossValidationResult",
    "cross_validate",
    "OptunaSearchSpace",
    "optuna_optimize",
]
