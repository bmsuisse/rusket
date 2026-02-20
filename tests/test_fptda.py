"""FP-TDA tests – same structure as test_fpgrowth.py.

Reuses all base mix-ins from test_fpbase.py to verify FP-TDA produces the
same results as FP-Growth, and adds explicit cross-checks.
"""

from __future__ import annotations

import unittest

import pandas as pd
import pytest

from test_fpbase import (
    FPTestEdgeCases,
    FPTestErrors,
    FPTestEx1All,
    FPTestEx2All,
    FPTestEx3All,
)

from rusket import fpgrowth, fptda


# ---------------------------------------------------------------------------
# Standard mix-in tests
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):
    def setUp(self) -> None:
        FPTestEdgeCases.setUp(self, fptda)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self) -> None:
        FPTestErrors.setUp(self, fptda)


class TestEx1(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:
        FPTestEx1All.setUp(self, fptda)


class TestEx2(unittest.TestCase, FPTestEx2All):
    def setUp(self) -> None:
        FPTestEx2All.setUp(self, fptda)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self) -> None:
        FPTestEx3All.setUp(self, fptda)


# ---------------------------------------------------------------------------
# Paper example (Table 1 from IJISRT25NOV1256)
# ---------------------------------------------------------------------------


def _paper_transactions() -> pd.DataFrame:
    """Nine transactions over items P1-P5 from Table 1 of the paper."""
    transactions = [
        ["P1", "P2", "P5"],    # T1
        ["P2", "P4"],           # T2
        ["P2", "P3"],           # T3
        ["P1", "P2", "P4"],    # T4
        ["P1", "P3"],           # T5
        ["P2", "P3"],           # T6
        ["P1", "P3"],           # T7
        ["P1", "P2", "P3", "P5"],  # T8
        ["P1", "P2", "P3"],    # T9
    ]
    items = ["P1", "P2", "P3", "P4", "P5"]
    data = [{item: (item in t) for item in items} for t in transactions]
    return pd.DataFrame(data)


def test_paper_example_minsup_half() -> None:
    """With minsup=50 % (≥ 5/9 rounds up to 5) verify known frequent items."""
    df = _paper_transactions()
    result = fptda(df, min_support=0.5, use_colnames=True)
    freq = {
        frozenset(row["itemsets"]): round(row["support"] * len(df))
        for _, row in result.iterrows()
    }
    # P1 appears in T1,T4,T5,T7,T8,T9 → 6 times
    assert freq[frozenset(["P1"])] == 6, freq
    # P2 appears in T1,T2,T3,T4,T6,T8,T9 → 7 times
    assert freq[frozenset(["P2"])] == 7, freq
    # P3 appears in T3,T5,T6,T7,T8,T9 → 6 times
    assert freq[frozenset(["P3"])] == 6, freq


# ---------------------------------------------------------------------------
# Correctness cross-check: FP-TDA must equal FP-Growth
# ---------------------------------------------------------------------------


def _to_bool_df(transactions: list[list[str | int]]) -> pd.DataFrame:
    items = sorted(set(item for t in transactions for item in t))
    data = [{item: (item in t) for item in items} for t in transactions]
    return pd.DataFrame(data)


def _support_map(
    df: pd.DataFrame, algo: object, min_support: float
) -> dict[frozenset, float]:
    import typing
    fn = typing.cast(object, algo)
    result = fn(df, min_support=min_support, use_colnames=True)  # type: ignore[operator]
    return {
        frozenset(row["itemsets"]): row["support"]
        for _, row in result.iterrows()
    }


_TRANSACTIONS_SETS = [
    _paper_transactions(),
    _to_bool_df([
        ["r", "z", "h", "k", "p"],
        ["z", "y", "x", "w", "v", "u", "t", "s"],
        ["s", "x", "o", "n", "r"],
        ["x", "z", "y", "m", "t", "s", "q", "e"],
        ["z"],
        ["x", "z", "y", "r", "q", "t", "p"],
    ]),
    _to_bool_df([
        [1, 2, 3],
        [1, 2, 3, 4],
        [5, 4, 3, 2, 1],
        [6, 5, 4, 3, 2, 1],
        [2, 4],
        [1, 3],
        [1, 7],
    ]),
]


@pytest.mark.parametrize("df", _TRANSACTIONS_SETS)
@pytest.mark.parametrize("min_support", [0.5, 0.3, 0.1])
def test_fptda_equals_fpgrowth(df: pd.DataFrame, min_support: float) -> None:
    """FP-TDA must return identical support values to FP-Growth."""
    expected = _support_map(df, fpgrowth, min_support)
    actual = _support_map(df, fptda, min_support)

    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    assert not missing, f"FP-TDA missing itemsets: {missing}"
    assert not extra, f"FP-TDA has extra itemsets: {extra}"

    for iset, sup in expected.items():
        assert abs(actual[iset] - sup) < 1e-9, (
            f"Support mismatch for {iset}: expected {sup}, got {actual[iset]}"
        )
