"""MLflow integration for rusket."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

try:
    from rusket._dependencies import import_optional_dependency

    mlflow = import_optional_dependency("mlflow")
    from rusket._dependencies import import_optional_dependency

    import_optional_dependency("mlflow.pyfunc", "mlflow")

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)

_AUTOLOG_ENABLED = False
_ORIG_FIT_METHODS: dict[Any, Callable[..., Any]] = {}


if HAS_MLFLOW:

    class _RusketWrapper(mlflow.pyfunc.PythonModel):  # type: ignore
        """PyFunc wrapper for rusket models."""

        def load_context(self, context: Any) -> None:
            from .model import load_model

            model_path = context.artifacts["model_path"]
            self.model = load_model(model_path)

        def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
            """Predict recommendations for a dataframe of users.

            Input dataframe should have a 'user' column (or user inputs directly).
            """
            from rusket._dependencies import import_optional_dependency

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

            results = []
            for u in users:
                try:
                    items, scores = self.model.recommend_items(u, n=10, exclude_seen=True)
                    results.append({"user": u, "items": items.tolist(), "scores": scores.scores.tolist()})
                except Exception:
                    results.append({"user": u, "items": [], "scores": []})

            return pd.DataFrame(results)
else:
    _RusketWrapper = None  # type: ignore


def save_model(model: Any, path: str, **kwargs: Any) -> None:
    """Save a rusket model as an MLflow pyfunc model."""
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed. Install it with: pip install mlflow")

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_path = os.path.join(tmpdir, "model.bin")
        model.save(local_model_path)

        artifacts = {"model_path": local_model_path}

        mlflow.pyfunc.save_model(path=path, python_model=_RusketWrapper(), artifacts=artifacts, **kwargs)


def log_model(model: Any, artifact_path: str, **kwargs: Any) -> Any:
    """Log a rusket model as an MLflow pyfunc artifact."""
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed. Install it with: pip install mlflow")

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_path = os.path.join(tmpdir, "model.bin")
        model.save(local_model_path)

        artifacts = {"model_path": local_model_path}

        return mlflow.pyfunc.log_model(
            artifact_path=artifact_path, python_model=_RusketWrapper(), artifacts=artifacts, **kwargs
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
    from .als import ALS, eALS
    from .bpr import BPR
    from .ease import EASE
    from .eclat import Eclat
    from .fpgrowth import FPGrowth
    from .item_knn import ItemKNN
    from .lightgcn import LightGCN
    from .prefixspan import PrefixSpan
    from .sasrec import SASRec
    from .svd import SVD

    models = [ALS, eALS, BPR, EASE, ItemKNN, LightGCN, SVD, SASRec, FPGrowth, Eclat, PrefixSpan]

    if not disable:
        for m in models:
            _patch_fit(m)
    else:
        for m in models:
            _unpatch_fit(m)
