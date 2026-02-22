from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    try:
        import pyspark.sql
        _SparkDataFrame = pyspark.sql.DataFrame
    except ImportError:
        _SparkDataFrame = Any

    DataFrameType = pd.DataFrame | pl.DataFrame | _SparkDataFrame
else:
    DataFrameType = Any

class SupportsItemFactors(Protocol):
    """Protocol for models that expose latent item factors.

    This is used by utility functions like `similar_items` and `export_item_factors`
    to guarantee that the provided model has an `item_factors` property.
    """

    @property
    def item_factors(self) -> np.ndarray:
        ...

    @property
    def _item_labels(self) -> list[Any] | None:
        ...
