from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    try:
        import pyspark.sql

        _SparkDataFrame = pyspark.sql.DataFrame
    except ImportError:
        _SparkDataFrame = Any

    DataFrameType = pd.DataFrame | pl.DataFrame | pa.Table | _SparkDataFrame
else:
    DataFrameType = Any


class SupportsItemFactors(Protocol):
    """Protocol for models that expose latent item factors.

    This is used by utility functions like `similar_items` and `export_item_factors`
    to guarantee that the provided model has an `item_factors` property.
    """

    @property
    def item_factors(self) -> np.ndarray: ...

    @property
    def _item_labels(self) -> list[Any] | None: ...


class SupportsUserFactors(Protocol):
    """Protocol for models that expose latent user factors."""

    @property
    def user_factors(self) -> np.ndarray: ...

    @property
    def _user_labels(self) -> list[Any] | None: ...
