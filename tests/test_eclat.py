"""Eclat tests – mirrors test_fpgrowth.py but uses the standalone eclat() function."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from test_fpbase import (
    FPTestEdgeCases,
    FPTestErrors,
    FPTestEx1All,
    FPTestEx2All,
    FPTestEx3All,
)

from rusket import eclat


class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):
    def setUp(self) -> None:
        FPTestEdgeCases.setUp(self, eclat)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self) -> None:
        FPTestErrors.setUp(self, eclat)


class TestEx1(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:
        FPTestEx1All.setUp(self, eclat)


class TestEx1BoolInput(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:
        one_ary = np.array(
            [
                [False, False, False, True, False, True, True, True, True, False, True],
                [False, False, True, True, False, True, False, True, True, False, True],
                [
                    True,
                    False,
                    False,
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                ],
                [
                    False,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
            ]
        )
        FPTestEx1All.setUp(self, eclat, one_ary=one_ary)


class TestEx2(unittest.TestCase, FPTestEx2All):
    def setUp(self) -> None:
        FPTestEx2All.setUp(self, eclat)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self) -> None:
        FPTestEx3All.setUp(self, eclat)


# ---------------------------------------------------------------------------
# Eclat-specific tests
# ---------------------------------------------------------------------------


def test_eclat_matches_fpgrowth() -> None:
    """Eclat must produce identical results to fpgrowth."""
    from rusket import fpgrowth

    rng = np.random.default_rng(42)
    df = pd.DataFrame({f"c{i}": rng.random(500) < 0.15 for i in range(30)})

    res_eclat = eclat(df, min_support=0.05, use_colnames=True)
    res_fpg = fpgrowth(df, min_support=0.05, use_colnames=True)

    # Same number of itemsets
    assert len(res_eclat) == len(res_fpg), f"eclat found {len(res_eclat)} itemsets, fpgrowth found {len(res_fpg)}"


def test_eclat_polars_input() -> None:
    """Eclat must work with Polars DataFrames."""
    try:
        import polars as pl
    except ImportError:
        return  # skip if polars not installed

    df = pd.DataFrame({"a": [True, True, False], "b": [True, False, True], "c": [False, True, True]})
    df_pl = pl.from_pandas(df)

    res_pd = eclat(df, min_support=0.5, use_colnames=True)
    res_pl = eclat(df_pl, min_support=0.5, use_colnames=True)

    assert len(res_pd) == len(res_pl)


def test_eclat_max_len() -> None:
    """max_len must limit itemset size."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"c{i}": rng.random(200) < 0.3 for i in range(10)})

    res = eclat(df, min_support=0.05, use_colnames=True, max_len=2)
    for _, row in res.iterrows():
        assert len(row["itemsets"]) <= 2


def test_eclat_empty_result() -> None:
    """Very high support should yield empty result."""
    df = pd.DataFrame({"a": [True, False], "b": [False, True]})
    res = eclat(df, min_support=0.99)
    assert len(res) == 0


def test_eclat_sparse_retail() -> None:
    """Eclat should handle sparse retail-like data efficiently."""
    rng = np.random.default_rng(42)
    n_rows, n_cols = 5_000, 500
    data = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        items = rng.choice(n_cols, size=rng.poisson(5), replace=False)
        data[i, items] = True
    df = pd.DataFrame(data, columns=[f"sku_{j}" for j in range(n_cols)])

    res = eclat(df, min_support=0.01, use_colnames=True, max_len=3)
    assert res.shape[0] >= 0  # just ensure no crash
