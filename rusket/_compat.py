from __future__ import annotations

from typing import Any


def to_pandas(data: Any) -> Any:
    """Coerce a Spark or Polars DataFrame to pandas; return everything else unchanged."""
    mod = getattr(data, "__module__", "") or ""
    if mod.startswith("pyspark"):
        return data.toPandas()
    if mod.startswith("polars"):
        return data.to_pandas()
    return data
