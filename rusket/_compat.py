from __future__ import annotations

from typing import Any


def to_dataframe(data: Any) -> Any:
    """Coerce a Spark DataFrame to a zero-copy Polars DataFrame via PyArrow; return everything else unchanged."""
    mod = getattr(data, "__module__", "") or ""

    # We want Spark DataFrames to stay in Arrow memory, saving GC pauses
    if mod.startswith("pyspark"):
        import polars as pl
        import pyarrow as pa

        # PySpark 3.4+ supports toArrow() zero-copy serialization
        # Older PySpark may not have it, so we fallback gracefully but warn.
        if hasattr(data, "toArrow"):
            try:
                table = pa.Table.from_batches(data.toArrow())
                return pl.from_arrow(table)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to extract zero-copy Arrow batches from PySpark, falling back to Pandas. Exception: {e}",
                    stacklevel=2,
                )
                return data.toPandas()
        else:
            return data.toPandas()

    # Keep polars dataframes instead of coercing them to pandas so we use Rust arrow backend
    return data
