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
                return payload  # type: ignore[return-value]
            raise TypeError(f"Expected {cls.__name__}, got {type(payload).__name__}")

        # Construct an empty instance and restore state
        instance = cls.__new__(cls)  # type: ignore[arg-type]
        instance.__dict__.update(state)

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
        return instance
    else:
        # Legacy: plain pickled object
        if hasattr(payload, "__dict__"):
            return payload
        raise TypeError(f"Expected a rusket model, got {type(payload).__name__}")
