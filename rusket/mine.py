from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

def mine(
    df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    method: str = "auto",
    verbose: int = 0,
    column_names: list[str] | None = None,
) -> pd.DataFrame:
    """Mine frequent itemsets using the optimal algorithm.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame | np.ndarray | scipy.sparse matrix
        The dataset to mine.
    min_support : float, default=0.5
        The minimum support threshold in the range (0, 1].
    null_values : bool, default=False
        Whether the input data contains null values (requires an extra pass).
    use_colnames : bool, default=False
        Return itemsets with column names rather than indices.
    max_len : int | None, default=None
        Maximum length of the itemsets generated.
    method : "auto" | "fpgrowth" | "eclat", default="auto"
        Algorithm to use. "auto" selects Eclat for sparse datasets and FP-Growth for dense ones.
    verbose : int, default=0
        Print timing and memory logs.
    column_names : list[str] | None, default=None
        Custom column names if `use_colnames=True` and no Pandas/Polars header exists.

    Returns
    -------
    pd.DataFrame
        DataFrame containing `support` and `itemsets` columns.
    """
    if method not in ("fpgrowth", "eclat", "auto"):
        raise ValueError(f"`method` must be 'fpgrowth', 'eclat', or 'auto'. Got: {method}")
    if min_support <= 0.0:
        raise ValueError(
            f"`min_support` must be a positive number within the interval `(0, 1]`. Got {min_support}."
        )

    from ._core import dispatch

    return dispatch(
        df,
        min_support,
        null_values,
        use_colnames,
        max_len,
        method,
        column_names,
        verbose,
    )  # type: ignore[arg-type]
