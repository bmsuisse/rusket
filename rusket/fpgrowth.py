from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl


def fpgrowth(
    df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    method: str = "fpgrowth",
    verbose: int = 0,
    column_names: list[str] | None = None,
) -> pd.DataFrame:
    if method not in ("fpgrowth", "eclat"):
        raise ValueError(f"`method` must be 'fpgrowth' or 'eclat'. Got: {method}")
    if min_support <= 0.0:
        raise ValueError(
            f"`min_support` must be a positive number within the interval `(0, 1]`. Got {min_support}."
        )

    from ._core import dispatch

    return dispatch(df, min_support, null_values, use_colnames, max_len, method, column_names)  # type: ignore[arg-type]
