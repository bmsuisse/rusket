"""Input validation utilities – ported from mlxtend.frequent_patterns.fpcommon."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def valid_input_check(df: pd.DataFrame, null_values: bool = False) -> None:
    """Validate a one-hot / boolean DataFrame before running FP-Growth.

    Parameters
    ----------
    df:
        Input DataFrame.  Allowed values: 0/1 or True/False (and NaN if
        ``null_values=True``).
    null_values:
        Whether NaN values are allowed in *df*.
    """
    if df is None:
        return

    if f"{type(df)}" == "<class 'pandas.core.frame.SparseDataFrame'>":
        raise TypeError(
            "SparseDataFrame support has been deprecated in pandas 1.0 "
            "and is no longer supported. "
            "Please see the pandas migration guide at "
            "https://pandas.pydata.org/pandas-docs/"
            "stable/user_guide/sparse.html#sparse-data-structures "
            "for supporting sparse data in DataFrames."
        )

    if df.size == 0:
        return

    if hasattr(df, "sparse"):
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError(
                "Due to current limitations in Pandas, "
                "if the sparse format has integer column names,"
                "names, please make sure they either start "
                "with `0` or cast them as string column names: "
                "`df.columns = [str(i) for i in df.columns`]."
            )

    # Fast path: all bool columns
    if null_values:
        all_bools = (
            df.apply(lambda col: col.apply(lambda x: pd.isna(x) or isinstance(x, bool)))
            .all()
            .all()
        )
    else:
        all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()

    if not all_bools:
        warnings.warn(
            "DataFrames with non-bool types result in worse computational "
            "performance and their support might be discontinued in the future. "
            "Please use a DataFrame with bool type",
            DeprecationWarning,
            stacklevel=3,
        )

        has_nans = pd.isna(df).any().any()
        if null_values and not has_nans:
            warnings.warn(
                "null_values=True is inefficient when there are no NaN values "
                "in the DataFrame. Set null_values=False for faster output.",
                stacklevel=3,
            )
        if not null_values and has_nans:
            raise ValueError(
                "NaN values are not permitted in the DataFrame when null_values=False."
            )

        if hasattr(df, "sparse"):
            if df.size == 0:
                values = df.values
            else:
                values = df.sparse.to_coo().tocoo().data
        else:
            values = df.values

        if null_values:
            idxs = np.where((values != 1) & (values != 0) & (~np.isnan(values)))
        else:
            idxs = np.where((values != 1) & (values != 0))

        if len(idxs[0]) > 0:
            val = values[tuple(loc[0] for loc in idxs)]
            if null_values:
                s = (
                    "The allowed values for a DataFrame "
                    "are True, False, 0, 1, NaN. Found value %s" % (val,)
                )
            else:
                s = (
                    "The allowed values for a DataFrame "
                    "are True, False, 0, 1. Found value %s" % (val,)
                )
            raise ValueError(s)
