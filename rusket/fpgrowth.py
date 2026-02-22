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
    """Find frequent itemsets using the FP-growth algorithm.

    Parameters
    ----------
    df : pandas.DataFrame, polars.DataFrame, or numpy.ndarray
        The input dataset containing transactions. Can be a dense matrix of 0s and 1s,
        a scipy sparse matrix, or a boolean/integer DataFrame.
    min_support : float, default=0.5
        The minimum support threshold `[0.0, 1.0]`. Calculates as a percentage of total transactions.
    null_values : bool, default=False
        If True, ignore missing/null values in pandas DataFrames.
    use_colnames : bool, default=False
        If True, returns itemsets containing actual item names (column names)
        rather than their column indices.
    max_len : int | None, default=None
        Maximum length of the itemsets generated. If None, no limit is applied.
    method : str, default="fpgrowth"
        The mining method to use. Options are "fpgrowth", "eclat", or "auto".
        "auto" will choose between Eclat and FPGrowth based on matrix density.
    verbose : int, default=0
        If > 0, print progress and backend selection details to standard output.
    column_names : list[str] | None, default=None
        Custom column names to use if input is a numpy array or scipy sparse matrix
        and `use_colnames=True`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - `support`: the support score.
        - `itemsets`: list of items (indices or column names).
    """
    if method not in ("fpgrowth", "eclat", "auto"):
        raise ValueError(
            f"`method` must be 'fpgrowth', 'eclat', or 'auto'. Got: {method}"
        )
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
