"""Data splitting utilities for train/test evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from . import _rusket


def train_test_split(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    import numpy as np

    from rusket._dependencies import import_optional_dependency

    _pd = import_optional_dependency("pandas")

    if not isinstance(df, _pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    dummy_ids = np.zeros(len(df), dtype=np.int32)
    train_idx, test_idx = _rusket.train_test_split(list(dummy_ids), test_size)  # type: ignore

    return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)


def leave_one_out_split(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    import numpy as np

    from rusket._dependencies import import_optional_dependency

    _pd = import_optional_dependency("pandas")

    if not isinstance(df, _pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

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


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions chronologically.

    Sorts the dataframe by timestamp and assigns the first ``1 - test_size``
    fraction to the training set and the remainder to the test set.

    Parameters
    ----------
    df : pd.DataFrame
        The interaction dataframe.
    timestamp_col : str
        Name of the timestamp column.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    df_sorted = df.sort_values(by=timestamp_col)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


def user_stratified_split(
    df: pd.DataFrame,
    user_col: str,
    test_size: float = 0.2,
    min_train_items: int = 1,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions stratifying by user.

    Assigns a fraction of each user's interactions to the test set, ensuring
    that at least ``min_train_items`` remain in the train set. Users with fewer
    than ``min_train_items`` interactions will have all their interactions
    in the training set.

    Parameters
    ----------
    df : pd.DataFrame
        The interaction dataframe.
    user_col : str
        Name of the user column.
    test_size : float, default=0.2
        The proportion of each user's interactions to include in the test split.
    min_train_items : int, default=1
        Minimum number of interactions a user must have in the training set.
    random_state : int or None, default=None
        Controls the shuffling applied before the split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    import numpy as np

    rng = np.random.default_rng(random_state)

    # Shuffle the dataframe
    indices = np.arange(len(df))
    rng.shuffle(indices)

    shuffled_df = df.iloc[indices].reset_index(drop=True)
    group_indices = shuffled_df.groupby(user_col).indices

    train_indices = []
    test_indices = []

    for _user, user_idxs in group_indices.items():
        n_inter = len(user_idxs)
        n_test = int(n_inter * test_size)

        # Ensure minimum train items
        if n_inter - n_test < min_train_items:
            n_test = n_inter - min_train_items

        n_test = max(0, n_test)

        if n_test > 0:
            test_indices.extend(user_idxs[:n_test])
            train_indices.extend(user_idxs[n_test:])
        else:
            train_indices.extend(user_idxs)

    train_df = shuffled_df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = shuffled_df.iloc[test_indices].copy().reset_index(drop=True)
    return train_df, test_df
