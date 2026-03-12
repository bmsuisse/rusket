"""Abstract base class and model persistence for all rusket algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from typing_extensions import Self


class BaseModel(ABC):
    """Abstract base class for all rusket algorithms.

    Provides unified data ingestion methods (from_transactions, from_pandas, etc.)
    for any downstream Miner or Recommender.
    """

    @classmethod
    @abstractmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Initialize the model from a long-format DataFrame or sequences.

        Must be implemented by subclasses.
        """
        pass

    @classmethod
    def from_ratings(
        cls,
        data: Any,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Alias for from_transactions, specifically meant for Recommenders."""
        if "transaction_col" not in kwargs:
            kwargs["transaction_col"] = user_col
        if rating_col is not None:
            kwargs["rating_col"] = rating_col
        return cls.from_transactions(
            data,
            item_col=item_col,
            verbose=verbose,
            **kwargs,
        )

    def __dir__(self) -> list[str]:
        """Provides a clean public API surface for AI code assistants and REPLs.
        Filters out internal properties starting with underscores.
        """
        return [k for k in super().__dir__() if not k.startswith("_")]

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(df, transaction_col=transaction_col, item_col=item_col, verbose=verbose, **kwargs)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(df, transaction_col=transaction_col, item_col=item_col, verbose=verbose, **kwargs)

    @classmethod
    def from_spark(
        cls,
        df: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(df, transaction_col=transaction_col, item_col=item_col, **kwargs)

    @classmethod
    def from_arrow(
        cls,
        table: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(table, transaction_col, item_col)``.

        Parameters
        ----------
        table : pyarrow.Table
            An Arrow table with transaction and item columns.
        transaction_col : str, optional
            Name of the transaction ID column.
        item_col : str, optional
            Name of the item column.
        **kwargs
            Extra arguments forwarded to ``from_transactions``.
        """
        return cls.from_transactions(table, transaction_col=transaction_col, item_col=item_col, **kwargs)

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


def load_model(path: str | Path) -> BaseModel:
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
