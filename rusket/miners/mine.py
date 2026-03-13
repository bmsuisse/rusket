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
    use_colnames: bool = True,
    max_len: int | None = None,
    method: str = "fpgrowth",
    verbose: int = 0,
    column_names: list[str] | None = None,
) -> pd.DataFrame:
    """Mine frequent itemsets using the specified algorithm.

    This module-level function relies on the Object-Oriented APIs.
    """
    if method not in ("fpgrowth", "eclat", "fin", "lcm", "negfin"):
        raise ValueError(f"`method` must be 'fpgrowth', 'eclat', 'fin', 'lcm', or 'negfin'. Got: {method}")

    if method == "fpgrowth":
        from .fpgrowth import FPGrowth

        return FPGrowth(
            data=df,
            item_names=column_names,
            min_support=min_support,
            null_values=null_values,
            use_colnames=use_colnames,
            max_len=max_len,
            verbose=verbose,
        ).mine()
    elif method == "eclat":
        from .eclat import Eclat

        return Eclat(
            data=df,
            item_names=column_names,
            min_support=min_support,
            null_values=null_values,
            use_colnames=use_colnames,
            max_len=max_len,
            verbose=verbose,
        ).mine()
    elif method == "fin":
        from .fin import FIN

        return FIN(
            data=df,
            item_names=column_names,
            min_support=min_support,
            null_values=null_values,
            use_colnames=use_colnames,
            max_len=max_len,
            verbose=verbose,
        ).mine()
    elif method == "lcm":
        from .lcm import LCM

        return LCM(
            data=df,
            item_names=column_names,
            min_support=min_support,
            null_values=null_values,
            use_colnames=use_colnames,
            max_len=max_len,
            verbose=verbose,
        ).mine()
    elif method == "negfin":
        from .negfin import NegFIN

        return NegFIN(
            data=df,
            item_names=column_names,
            min_support=min_support,
            null_values=null_values,
            use_colnames=use_colnames,
            max_len=max_len,
            verbose=verbose,
        ).mine()

    raise ValueError(f"Unknown method {method}")
