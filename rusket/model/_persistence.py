"""Pickle-based persistence (save/load) for rusket models.

Note: this uses ``pickle`` (pre-existing behavior, moved unchanged from
``_base.py``) and therefore assumes model files come from a trusted source —
loading an untrusted ``.pkl`` file can execute arbitrary code.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

# Array attributes that ItemKNN/UserKNN/RuleBasedRecommender pass straight
# into Rust, which requires exact dtypes. fit() now casts these once (moved
# out of the hot recommend_items() path for performance), so a model pickled
# under an older release still carries raw scipy dtypes (e.g. int32 indptr,
# float64 data). Normalize them once here at load time rather than per-call.
_RUST_DTYPE_ATTRS: dict[str, str] = {
    "w_indptr": "int64",
    "_fit_indptr": "int64",
    "w_indices": "int32",
    "_fit_indices": "int32",
    "w_data": "float32",
    "_fit_data": "float32",
}


def _normalize_rust_dtypes(instance: Any) -> Any:
    """Coerce known array attributes to the dtypes the Rust extension expects."""
    import numpy as np

    for attr, dtype in _RUST_DTYPE_ATTRS.items():
        value = getattr(instance, attr, None)
        if value is None:
            continue
        if getattr(value, "dtype", None) != np.dtype(dtype):
            try:
                setattr(instance, attr, np.asarray(value).astype(dtype, copy=False))
            except (TypeError, ValueError):
                pass
    return instance


class PersistenceMixin:
    """Mixin providing pickle-based ``save``/``load`` for rusket models."""

    def save(self, path: str | Path) -> None:
        """Save the model to disk using pickle.

        Parameters
        ----------
        path : str or Path
            File path to write the model to (e.g. ``"model.pkl"``).
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "__rusket_version__": 1,
            "class": type(self).__name__,
            "module": type(self).__module__,
            "state": self.__dict__,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        Self
            The restored model.

        Raises
        ------
        TypeError
            If the file contains a different model class.
        """
        import pickle

        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301

        if isinstance(payload, dict) and "__rusket_version__" in payload:
            saved_cls_name = payload.get("class", "")
            state = payload["state"]
        else:
            # Legacy: plain pickled object
            if isinstance(payload, cls):
                return _normalize_rust_dtypes(payload)  # type: ignore[return-value]
            raise TypeError(f"Expected {cls.__name__}, got {type(payload).__name__}")

        # Construct an empty instance and restore state
        instance = cls.__new__(cls)  # type: ignore[arg-type]
        instance.__dict__.update(state)
        _normalize_rust_dtypes(instance)

        if saved_cls_name != cls.__name__:
            import warnings

            warnings.warn(
                f"Model was saved as {saved_cls_name} but loaded as {cls.__name__}. "
                "This may cause unexpected behaviour.",
                stacklevel=2,
            )

        return instance  # type: ignore[return-value]


def load_model(path: str | Path) -> Any:
    """Load a previously saved model from disk.

    This function automatically determines the correct model class
    and instantiates it.

    Parameters
    ----------
    path : str or Path
        File path to load from.

    Returns
    -------
    BaseModel
        The restored model.
    """
    import pickle

    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301

    if isinstance(payload, dict) and "__rusket_version__" in payload:
        saved_cls_name = payload.get("class", "")
        module_name = payload.get("module", "")
        state = payload["state"]

        # Import the class dynamically
        import importlib

        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, saved_cls_name)
        except (ImportError, AttributeError) as err:
            # Fallback to rusket namespace if old module moved
            import rusket

            cls = getattr(rusket, saved_cls_name, None)
            if cls is None:
                raise TypeError(f"Could not resolve class {saved_cls_name} from {module_name}") from err

        instance = cls.__new__(cls)
        instance.__dict__.update(state)
        return _normalize_rust_dtypes(instance)
    else:
        # Legacy: plain pickled object
        if hasattr(payload, "__dict__"):
            return _normalize_rust_dtypes(payload)
        raise TypeError(f"Expected a rusket model, got {type(payload).__name__}")
