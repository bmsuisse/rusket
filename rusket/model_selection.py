"""Data preparation and model selection utilities."""

from __future__ import annotations

import numpy as np

from . import _rusket


def train_test_split(
    df,
    user_col: str,
    item_col: str,
    test_size: float = 0.2,
    random_state: int | None = None,
):
    """Split interactions into random train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The interaction dataframe.
    user_col : str
        Name of the user column.
    item_col : str
        Name of the item column.
    test_size : float, default=0.2
        Percentage of data to put in the test set.
    random_state : int, optional
        Set random seed (currently not used by Rust backend, but reserved for future).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for model_selection utility.") from e

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # Normally we do integer coercion when fitting, so here we assume
    # basic random splitting doesn't require actual encoded integers
    # but the Rust backend requires i32 for types (we use user_ids array just for length currently).
    # Since train_test_split algorithm randomly splits row indices, we can pass dummy int array

    dummy_ids = np.zeros(len(df), dtype=np.int32)
    train_idx, test_idx = _rusket.train_test_split(list(dummy_ids), test_size)  # type: ignore

    return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)


def leave_one_out_split(
    df,
    user_col: str,
    item_col: str,
    timestamp_col: str | None = None,
):
    """Leave exactly one interaction per user for the test set.

    If a timestamp column is provided, the latest interaction is left out.
    If no timestamp is provided, a random interaction is chosen.

    Parameters
    ----------
    df : pd.DataFrame
        The interaction dataframe.
    user_col : str
        Name of the user column (must be numeric encoded to i32 ideally, or pandas int).
    item_col : str
        Name of the item column.
    timestamp_col : str, optional
        Name of the timestamp or ordering column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for model_selection utility.") from e

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # We need to ensure we can cast user IDs to int32 for the rust backend.
    try:
        user_ids = df[user_col].values.astype(np.int32)
        item_ids = df[item_col].values.astype(np.int32)
    except ValueError as e:
        raise ValueError(
            f"Columns {user_col} and {item_col} must be numeric/integer to use leave_one_out_split."
        ) from e

    timestamps = None
    if timestamp_col is not None:
        try:
            timestamps = list(df[timestamp_col].values.astype(np.float32))
        except ValueError as e:
            raise ValueError(f"Column {timestamp_col} must be numeric float to use leave_one_out_split.") from e

    train_idx, test_idx = _rusket.leave_one_out(list(user_ids), list(item_ids), timestamps)  # type: ignore

    return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)
