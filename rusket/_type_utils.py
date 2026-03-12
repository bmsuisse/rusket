"""Shared type-checking and coercion utilities for rusket.

Centralises patterns that were previously duplicated across model.py,
_core.py and transactions.py.
"""

from __future__ import annotations

from typing import Any


def is_dataframe_empty(df: Any) -> bool:
    """Check whether *df* is empty, supporting Pandas, Polars, PyArrow and plain sequences.

    Handles the quirks of each framework: ``None``, ``.empty``, ``.is_empty()``,
    ``.isEmpty()`` and ``len()``.

    Parameters
    ----------
    df : Any
        A DataFrame-like object (or ``None``).

    Returns
    -------
    bool
        ``True`` when the object is ``None``, empty, or has zero length.
    """
    if df is None:
        return True
    if hasattr(df, "empty"):
        return bool(df.empty)
    if hasattr(df, "is_empty"):
        return bool(df.is_empty())
    if hasattr(df, "isEmpty"):
        return bool(df.isEmpty())
    try:
        return len(df) == 0
    except TypeError:
        return False


def to_list_if_collection(x: Any) -> Any:
    """Convert *x* to a ``list`` if it is a ``tuple`` or ``set``, else return as-is.

    Used when building DataFrames for Arrow or Polars which do not
    natively support ``frozenset`` / ``tuple`` values.
    """
    if isinstance(x, (tuple, set)):
        return list(x)
    return x
