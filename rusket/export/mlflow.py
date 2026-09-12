"""MLflow integration for rusket."""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Callable
from typing import Any

# Presence check ONLY — must not actually import mlflow at module scope, since
# that pulls in the full mlflow package (and its own heavy deps) just from
# ``import rusket``. The real import happens lazily inside the functions below.
HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None

logger = logging.getLogger(__name__)

_AUTOLOG_ENABLED = False
_ORIG_FIT_METHODS: dict[Any, Callable[..., Any]] = {}

_rusket_wrapper_cls: type | None = None


def _get_rusket_wrapper_cls() -> type:
    """Lazily build and memoize the ``_RusketWrapper`` PyFunc class.

    Deferred because its base class (``mlflow.pyfunc.PythonModel``) requires
    actually importing mlflow, which we avoid at module import time.
    """
    global _rusket_wrapper_cls
    if _rusket_wrapper_cls is not None:
        return _rusket_wrapper_cls

    from rusket._internal._dependencies import import_optional_dependency

    mlflow = import_optional_dependency("mlflow")
    import_optional_dependency("mlflow.pyfunc", "mlflow")

    class _RusketWrapper(mlflow.pyfunc.PythonModel):  # type: ignore
        """PyFunc wrapper for rusket models."""

        def load_context(self, context: Any) -> None:
            from ..model import load_model

            model_path = context.artifacts["model_path"]
            self.model = load_model(model_path)

        def predict(self, context: Any, model_input):
            """Predict recommendations for a dataframe of users.

            Input dataframe should have a 'user' column (or user inputs directly).
            """
            from rusket._internal._dependencies import import_optional_dependency

            pd = import_optional_dependency("pandas")

            if isinstance(model_input, pd.DataFrame):
                if "user" in model_input.columns:
                    users = model_input["user"].tolist()
                elif "user_id" in model_input.columns:
                    users = model_input["user_id"].tolist()
                else:
                    users = model_input.iloc[:, 0].tolist()
            else:
                users = list(model_input)

            # ``recommend_items`` expects an internal 0-based row index, but a
            # model built via ``from_transactions()`` lives in an external
            # label space (customer ids, SKUs, strings, ...). Resolve labels
            # to indices the same way ``evaluate()`` does
            # (rusket/evaluation/metrics.py), and map results back to
            # external item labels via ``_item_labels``.
            user_labels = getattr(self.model, "_user_labels", None)
            item_labels = getattr(self.model, "_item_labels", None)

            user_to_idx = None
            if user_labels is not None:
                user_to_idx = {}
                for idx, lbl in enumerate(user_labels):
                    user_to_idx[lbl] = idx
                    try:
                        user_to_idx[int(lbl)] = idx
                    except (ValueError, TypeError):
                        pass

            results = []
            for u in users:
                if user_to_idx is not None:
                    user_idx = user_to_idx.get(u)
                    if user_idx is None:
                        # Unknown/cold-start external user id — no recommendations
                        # rather than a serving failure.
                        results.append({"user": u, "items": [], "scores": []})
                        continue
                else:
                    user_idx = u

                try:
                    items, scores = self.model.recommend_items(user_idx, n=10, exclude_seen=True)  # type: ignore
                except ValueError:
                    # Unknown/out-of-range user id (e.g. cold-start) — recommend_items
                    # raises ValueError for this; treat it as "no recommendations"
                    # rather than a serving failure. Anything else (a genuine bug)
                    # is intentionally left to propagate.
                    results.append({"user": u, "items": [], "scores": []})
                    continue

                item_list: list[Any] = items.tolist()
                if item_labels is not None and len(item_labels) == self.model._n_items:  # type: ignore
                    item_list = [item_labels[i] for i in item_list]

                results.append({"user": u, "items": item_list, "scores": scores.tolist()})

            return pd.DataFrame(results)

    _rusket_wrapper_cls = _RusketWrapper
    return _RusketWrapper


def save_model(model: Any, path: str, **kwargs: Any) -> None:
    """Save a rusket model as an MLflow pyfunc model."""
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed. Install it with: pip install mlflow")

    import os
    import tempfile

    from rusket._internal._dependencies import import_optional_dependency

    mlflow = import_optional_dependency("mlflow")
    wrapper_cls = _get_rusket_wrapper_cls()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_path = os.path.join(tmpdir, "model.bin")
        model.save(local_model_path)

        artifacts = {"model_path": local_model_path}

        mlflow.pyfunc.save_model(path=path, python_model=wrapper_cls(), artifacts=artifacts, **kwargs)


def log_model(model: Any, artifact_path: str, **kwargs: Any) -> Any:
    """Log a rusket model as an MLflow pyfunc artifact."""
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed. Install it with: pip install mlflow")

    import os
    import tempfile

    from rusket._internal._dependencies import import_optional_dependency

    mlflow = import_optional_dependency("mlflow")
    wrapper_cls = _get_rusket_wrapper_cls()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_path = os.path.join(tmpdir, "model.bin")
        model.save(local_model_path)

        artifacts = {"model_path": local_model_path}

        return mlflow.pyfunc.log_model(
            artifact_path=artifact_path, python_model=wrapper_cls(), artifacts=artifacts, **kwargs
        )


def _get_hyperparameters(model: Any) -> dict[str, Any]:
    """Extract hyperparameters from a model instance."""
    params = {}
    items = [
        "factors",
        "regularization",
        "learning_rate",
        "iterations",
        "alpha",
        "use_eals",
        "k",
        "min_support",
        "max_len",
    ]
    for key in items:
        if hasattr(model, key):
            params[key] = getattr(model, key)
    return params


def _patch_fit(cls: type) -> None:
    """Monkey-patch the fit method of a class to add MLflow tracking."""
    if cls in _ORIG_FIT_METHODS:
        return  # already patched

    if not hasattr(cls, "fit"):
        return

    orig_fit = cls.fit
    _ORIG_FIT_METHODS[cls] = orig_fit

    def patched_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _AUTOLOG_ENABLED:
            return orig_fit(self, *args, **kwargs)

        from rusket._internal._dependencies import import_optional_dependency

        mlflow = import_optional_dependency("mlflow")

        params = _get_hyperparameters(self)

        # Determine if we should end the run automatically (if we started it)
        active_run = mlflow.active_run()
        end_run = False
        if not active_run:
            mlflow.start_run()
            end_run = True

        mlflow.log_params(params)

        start_time = time.time()
        try:
            result = orig_fit(self, *args, **kwargs)
        finally:
            duration = time.time() - start_time
            mlflow.log_metric("training_duration_seconds", duration)
            if end_run:
                mlflow.end_run()

        return result

    cls.fit = patched_fit


def _unpatch_fit(cls: type) -> None:
    """Restore the original fit method of a class."""
    if cls in _ORIG_FIT_METHODS:
        cls.fit = _ORIG_FIT_METHODS.pop(cls)


def autolog(disable: bool = False) -> None:
    """Enable or disable native MLflow autologging for rusket models.

    When enabled, calling ``.fit()`` on a rusket model will automatically log:
    - Model hyperparameters (e.g. factors, learning rate, iterations)
    - Training duration
    to the currently active MLflow run.
    """
    if not HAS_MLFLOW:
        if not disable:
            logger.warning("MLflow is not installed. Autologging cannot be enabled.")
        return

    global _AUTOLOG_ENABLED
    _AUTOLOG_ENABLED = not disable

    # Patch or unpatch models
    from ..miners.eclat import Eclat
    from ..miners.fpgrowth import FPGrowth
    from ..miners.prefixspan import PrefixSpan
    from ..recommenders.als import ALS, eALS
    from ..recommenders.bpr import BPR
    from ..recommenders.ease import EASE
    from ..recommenders.item_knn import ItemKNN
    from ..recommenders.lightgcn import LightGCN
    from ..recommenders.svd import SVD
    from ..sequential.sasrec import SASRec

    models = [ALS, eALS, BPR, EASE, ItemKNN, LightGCN, SVD, SASRec, FPGrowth, Eclat, PrefixSpan]

    if not disable:
        for m in models:
            _patch_fit(m)
    else:
        for m in models:
            _unpatch_fit(m)
