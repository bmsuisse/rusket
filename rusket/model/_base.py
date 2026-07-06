"""Abstract base class for all rusket algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ._persistence import PersistenceMixin, load_model

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from typing_extensions import Self

__all__ = ["BaseModel", "load_model"]


class BaseModel(PersistenceMixin, ABC):
    """Abstract base class for all rusket algorithms.

    Provides unified data ingestion methods (from_transactions, from_pandas, etc.)
    for any downstream Miner or Recommender, plus pickle-based persistence
    (``save``/``load``, mixed in from :class:`PersistenceMixin`).
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
