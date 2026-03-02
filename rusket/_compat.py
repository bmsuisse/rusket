from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from pyspark.sql import DataFrame as SparkDataFrame

    #: Union of all supported dense/tabular input types.
    #:
    #: Accepted by every ``df`` / ``data`` parameter in rusket:
    #:
    #: * ``pandas.DataFrame`` – including sparse-backed frames
    #: * ``polars.DataFrame``
    #: * ``numpy.ndarray`` – 2-D boolean / 0-1 matrix
    #: * ``pyspark.sql.DataFrame`` – converted to Polars via Arrow zero-copy
    #: * ``pyarrow.Table`` – converted to Polars via zero-copy
    DataFrame = Union[pd.DataFrame, pl.DataFrame, np.ndarray, SparkDataFrame, pa.Table]  # noqa: UP007


def to_dataframe(data: Any) -> Any:
    """Coerce Spark/PyArrow inputs to a zero-copy Polars DataFrame; return everything else unchanged."""
    mod = getattr(type(data), "__module__", "") or ""

    # PyArrow Table → Polars (zero-copy via Arrow memory)
    if type(data).__name__ == "Table" and mod.startswith("pyarrow"):
        from rusket._dependencies import import_optional_dependency

        pl = import_optional_dependency("polars")

        return pl.from_arrow(data)

    # We want Spark DataFrames to stay in Arrow memory, saving GC pauses
    if type(data).__name__ == "DataFrame" and mod.startswith("pyspark"):
        from rusket._dependencies import import_optional_dependency

        pl = import_optional_dependency("polars")
        from rusket._dependencies import import_optional_dependency

        pa = import_optional_dependency("pyarrow")

        # PySpark 3.4+ supports toArrow() zero-copy serialization
        # Older PySpark may not have it, so we fallback gracefully but warn.
        if hasattr(data, "toArrow"):
            try:
                arrow_data = data.toArrow()
                if isinstance(arrow_data, pa.Table):
                    table = arrow_data
                else:
                    table = pa.Table.from_batches(arrow_data)
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
