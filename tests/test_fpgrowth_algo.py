"""FP-Growth tests – adapted from mlxtend/tests/test_fpgrowth.py."""

from __future__ import annotations

import functools
import time
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

from rusket import fpgrowth

fpgrowth_algo = functools.partial(fpgrowth, method="fpgrowth")
eclat_algo = functools.partial(fpgrowth, method="eclat")


class TestEdgeCases_FPGrowth(unittest.TestCase, FPTestEdgeCases):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEdgeCases.setUp(self, fpgrowth_algo)


class TestEdgeCases_Eclat(unittest.TestCase, FPTestEdgeCases):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEdgeCases.setUp(self, eclat_algo)


class TestErrors_FPGrowth(unittest.TestCase, FPTestErrors):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestErrors.setUp(self, fpgrowth_algo)


class TestErrors_Eclat(unittest.TestCase, FPTestErrors):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestErrors.setUp(self, eclat_algo)


class TestEx1_FPGrowth(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx1All.setUp(self, fpgrowth_algo)


class TestEx1_Eclat(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx1All.setUp(self, eclat_algo)


class Ex1BoolInputBase(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:  # type: ignore[override]
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
        FPTestEx1All.setUp(self, getattr(self, "method_func", fpgrowth_algo), one_ary=one_ary)


class TestEx2_FPGrowth(unittest.TestCase, FPTestEx2All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx2All.setUp(self, fpgrowth_algo)


class TestEx2_Eclat(unittest.TestCase, FPTestEx2All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx2All.setUp(self, eclat_algo)


class TestEx3_FPGrowth(unittest.TestCase, FPTestEx3All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx3All.setUp(self, fpgrowth_algo)


class TestEx3_Eclat(unittest.TestCase, FPTestEx3All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx3All.setUp(self, eclat_algo)


# ---------------------------------------------------------------------------
# Performance test – must complete in 5 seconds on 10k × 400 sparse data
# ---------------------------------------------------------------------------


def _create_dataframe(n_rows: int = 10_000, n_cols: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    support_values = np.zeros(n_cols)
    n_very_low = int(n_cols * 0.9)
    support_values[:n_very_low] = rng.uniform(0.0001, 0.009, n_very_low)
    n_medium = int(n_cols * 0.06)
    support_values[n_very_low : n_very_low + n_medium] = rng.uniform(0.01, 0.1, n_medium)
    n_high = n_cols - n_very_low - n_medium
    support_values[n_very_low + n_medium :] = rng.uniform(0.1, 0.65, n_high)
    return pd.DataFrame({f"feature_{i:04d}": rng.random(n_rows) < support_values[i] for i in range(n_cols)})


def test_fpgrowth_completes_within_5_seconds() -> None:
    df = _create_dataframe()
    start = time.perf_counter()
    frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)
    elapsed = time.perf_counter() - start
    assert elapsed < 5, (
        f"fpgrowth took {elapsed:.2f}s on df shape {df.shape} "
        f"and density {df.values.mean():.4f} with "
        f"{len(frequent_itemsets)} itemsets found"
    )


# ---------------------------------------------------------------------------
# Apache Spark MLlib Ported Tests
# ---------------------------------------------------------------------------


def _to_dataframe(transactions: list[list[str | int]]) -> pd.DataFrame:
    # Convert a list of transactions to a boolean one-hot encoded DataFrame
    items = sorted({item for t in transactions for item in t})
    data = [{item: (item in t) for item in items} for t in transactions]
    return pd.DataFrame(data)


def test_spark_mllib_fpgrowth_string() -> None:
    # Ported from Spark MLlib: FP-Growth using String type
    transactions = [
        "r z h k p".split(" "),
        "z y x w v u t s".split(" "),
        "s x o n r".split(" "),
        "x z y m t s q e".split(" "),
        ["z"],
        "x z y r q t p".split(" "),
    ]
    df = _to_dataframe(transactions)

    # min_support = 0.9 -> 0 itemsets
    res = fpgrowth(df, min_support=0.9, use_colnames=True)
    assert len(res) == 0

    # min_support = 0.5 -> 18 itemsets, verifiable frequencies
    res = fpgrowth(df, min_support=0.5, use_colnames=True)
    assert len(res) == 18

    freq_dict = {tuple(row["itemsets"]): row["support"] * len(df) for _, row in res.iterrows()}
    assert freq_dict[tuple(["z"])] == 5
    assert freq_dict[tuple(["x"])] == 4
    assert freq_dict[tuple(["t", "x", "y", "z"])] == 3

    # min_support = 0.3 -> 54 itemsets
    res2 = fpgrowth(df, min_support=0.3, use_colnames=True)
    assert len(res2) == 54

    # min_support = 0.1 -> 625 itemsets
    res1 = fpgrowth(df, min_support=0.1, use_colnames=True)
    assert len(res1) == 625


def test_spark_mllib_fpgrowth_int() -> None:
    # Ported from Spark MLlib: FP-Growth using Int type
    transactions = [
        [1, 2, 3],
        [1, 2, 3, 4],
        [5, 4, 3, 2, 1],
        [6, 5, 4, 3, 2, 1],
        [2, 4],
        [1, 3],
        [1, 7],
    ]
    df = _to_dataframe(transactions)  # type: ignore

    # min_support = 0.9 -> 0 itemsets
    res6 = fpgrowth(df, min_support=0.9, use_colnames=True)
    assert len(res6) == 0

    # min_support = 0.5 -> 9 itemsets
    res3 = fpgrowth(df, min_support=0.5, use_colnames=True)
    assert len(res3) == 9
    freq_dict = {tuple(row["itemsets"]): row["support"] * len(df) for _, row in res3.iterrows()}
    # Column names stay as their original type (int here), so keys are int
    expected = {
        tuple([1]): 6,
        tuple([2]): 5,
        tuple([3]): 5,
        tuple([4]): 4,
        tuple([1, 2]): 4,
        tuple([1, 3]): 5,
        tuple([2, 3]): 4,
        tuple([2, 4]): 4,
        tuple([1, 2, 3]): 4,
    }
    assert freq_dict == expected

    # min_support = 0.3 -> 15 itemsets
    res2 = fpgrowth(df, min_support=0.3, use_colnames=True)
    assert len(res2) == 15

    # min_support = 0.1 -> 65 itemsets
    res1 = fpgrowth(df, min_support=0.1, use_colnames=True)
    assert len(res1) == 65


def test_rust_fpgrowth_compat() -> None:
    # Ported from rust fp-growth crate: https://docs.rs/fp-growth/latest/src/fp_growth/lib.rs.html#1-128
    transactions = [
        ["a", "c", "e", "b", "f", "h", "a", "e", "f"],
        ["a", "c", "g"],
        ["e"],
        ["e", "c", "a", "g", "d"],
        ["a", "c", "e", "g"],
        ["e", "e"],
        ["a", "c", "e", "b", "f"],
        ["a", "c", "d"],
        ["g", "c", "e", "a"],
        ["a", "c", "e", "g"],
        ["i"],
    ]
    df = _to_dataframe(transactions)  # type: ignore

    # test cases: (minimum_support, frequent_patterns_num)
    test_cases = [
        (1, 88),
        (2, 43),
        (3, 15),
        (4, 15),
        (5, 11),
        (6, 7),
        (7, 4),
        (8, 4),
        (9, 0),
    ]

    for min_supp_count, expected_patterns in test_cases:
        res = fpgrowth(df, min_support=min_supp_count / len(df), use_colnames=True)
        assert len(res) == expected_patterns


class TestEx1BoolInput_FPGrowth(Ex1BoolInputBase):
    method_func = staticmethod(fpgrowth_algo)


class TestEx1BoolInput_Eclat(Ex1BoolInputBase):
    method_func = staticmethod(eclat_algo)
