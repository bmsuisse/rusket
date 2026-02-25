"""FP-Growth tests – adapted from mlxtend/tests/test_fpgrowth.py."""

from __future__ import annotations

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


class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEdgeCases.setUp(self, fpgrowth)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestErrors.setUp(self, fpgrowth)


class TestEx1(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx1All.setUp(self, fpgrowth)


class TestEx1BoolInput(unittest.TestCase, FPTestEx1All):
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
        FPTestEx1All.setUp(self, fpgrowth, one_ary=one_ary)


class TestEx2(unittest.TestCase, FPTestEx2All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx2All.setUp(self, fpgrowth)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self) -> None:  # type: ignore[override]
        FPTestEx3All.setUp(self, fpgrowth)


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

    freq_dict = {tuple(sorted(row["itemsets"])): row["support"] * len(df) for _, row in res.iterrows()}
    assert freq_dict[("z",)] == 5
    assert freq_dict[("x",)] == 4
    assert freq_dict[("t", "x", "y", "z")] == 3

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
    freq_dict = {tuple(sorted(row["itemsets"])): row["support"] * len(df) for _, row in res3.iterrows()}
    # Column names stay as their original type (int here), so keys are int
    expected = {
        (1,): 6,
        (2,): 5,
        (3,): 5,
        (4,): 4,
        (1, 2): 4,
        (1, 3): 5,
        (2, 3): 4,
        (2, 4): 4,
        (1, 2, 3): 4,
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


# ---------------------------------------------------------------------------
# Content-correctness tests: diverse datasets, exact itemset assertions
# ---------------------------------------------------------------------------


def _itemsets_as_tuples(res_df: pd.DataFrame) -> list[tuple]:
    """Convert result itemsets column to a list of sorted tuples for easy comparison."""
    return [tuple(sorted(row)) for row in res_df["itemsets"]]


def test_electronics_dataset_use_colnames_true() -> None:
    """Verify exact itemset content with a non-grocery dataset (use_colnames=True)."""
    # laptop + phone always co-occur, tablet + headphones always co-occur
    df = pd.DataFrame(
        {
            "laptop": [1, 1, 1, 1, 0, 0],
            "phone": [1, 1, 1, 1, 0, 0],
            "tablet": [0, 0, 1, 0, 1, 1],
            "headphones": [0, 0, 0, 0, 1, 1],
        }
    )

    res = fpgrowth(df, min_support=0.5, use_colnames=True)
    found = _itemsets_as_tuples(res)

    # laptop and phone both appear 4/6 = 0.667 and always together
    assert ("laptop",) in found
    assert ("phone",) in found
    assert ("laptop", "phone") in found  # sorted order

    # headphones only appears 2/6 = 0.333 -> below 0.5
    assert ("headphones",) not in found

    # No cross-contamination: no "bread" or "butter" should ever appear
    all_items = {item for fs in found for item in fs}
    assert "bread" not in all_items
    assert "butter" not in all_items
    assert "milk" not in all_items


def test_colors_dataset_use_colnames_false() -> None:
    """Verify integer index itemsets with use_colnames=False."""
    # Columns: red=0, green=1, blue=2, yellow=3
    df = pd.DataFrame(
        {
            "red": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "green": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            "blue": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            "yellow": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        }
    )

    res_named = fpgrowth(df, min_support=0.5, use_colnames=True)
    res_idx = fpgrowth(df, min_support=0.5, use_colnames=False)

    # Both should have same number of itemsets
    assert len(res_named) == len(res_idx)

    named_sets = _itemsets_as_tuples(res_named)
    idx_sets = _itemsets_as_tuples(res_idx)

    # green (col 1) appears 8/10 = 0.8 -> should be frequent
    assert ("green",) in named_sets
    assert (1,) in idx_sets

    # blue (col 2) appears 7/10 = 0.7 -> frequent
    assert ("blue",) in named_sets
    assert (2,) in idx_sets

    # red (col 0) appears 3/10 = 0.3 -> not frequent at 0.5
    assert ("red",) not in named_sets
    assert (0,) not in idx_sets

    # yellow (col 3) appears 2/10 = 0.2 -> not frequent
    assert ("yellow",) not in named_sets
    assert (3,) not in idx_sets


def test_sports_dataset_content_correctness() -> None:
    """Verify that items NOT co-occurring frequently don't appear in pair itemsets."""
    # soccer + basketball always co-occur (8/10)
    # tennis appears alone (6/10) but never with soccer
    df = pd.DataFrame(
        {
            "soccer": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            "basketball": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            "tennis": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            "swimming": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "cycling": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    res = fpgrowth(df, min_support=0.5, use_colnames=True)
    found = _itemsets_as_tuples(res)

    # soccer + basketball must co-occur (sorted order: basketball < soccer)
    assert ("basketball", "soccer") in found

    # swimming and cycling should NOT appear at all (< 0.5 support)
    all_items = {item for fs in found for item in fs}
    assert "swimming" not in all_items
    assert "cycling" not in all_items

    # soccer + tennis co-occur only 2/10 = 0.2 -> should NOT be a pair
    assert ("soccer", "tennis") not in found
    assert ("tennis", "soccer") not in found


def test_disjoint_groups_no_cross_contamination() -> None:
    """Two groups of items that never co-occur: no cross-group pairs should appear."""
    # Group A (rows 0-4): alpha + beta
    # Group B (rows 5-9): gamma + delta
    df = pd.DataFrame(
        {
            "alpha": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "beta": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "gamma": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "delta": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )

    res = fpgrowth(df, min_support=0.5, use_colnames=True)
    found = _itemsets_as_tuples(res)

    # Each singleton should appear (5/10 = 0.5)
    assert ("alpha",) in found
    assert ("beta",) in found
    assert ("gamma",) in found
    assert ("delta",) in found

    # Within-group pairs should appear (sorted order)
    assert ("alpha", "beta") in found
    assert ("delta", "gamma") in found

    # Cross-group pairs must NOT appear (0/10 co-occurrence)
    assert ("alpha", "gamma") not in found
    assert ("alpha", "delta") not in found
    assert ("beta", "gamma") not in found
    assert ("beta", "delta") not in found


def test_single_item_only_transactions() -> None:
    """When each transaction has exactly one item, no multi-item itemsets should appear."""
    df = pd.DataFrame(
        {
            "cat": [1, 0, 0, 0, 0],
            "dog": [0, 1, 0, 0, 0],
            "bird": [0, 0, 1, 0, 0],
            "fish": [0, 0, 0, 1, 0],
            "snake": [0, 0, 0, 0, 1],
        }
    )

    res = fpgrowth(df, min_support=0.1, use_colnames=True)
    found = _itemsets_as_tuples(res)

    # Each item appears exactly once -> support = 0.2
    assert len(found) == 5
    for item in ["cat", "dog", "bird", "fish", "snake"]:
        assert (item,) in found

    # No pairs should exist
    multi_item = [fs for fs in found if len(fs) > 1]
    assert len(multi_item) == 0


def test_all_methods_produce_same_results() -> None:
    """All mining methods should produce identical frequent itemsets."""
    df = pd.DataFrame(
        {
            "W": [1, 1, 0, 1, 0, 1, 1, 0],
            "X": [1, 1, 1, 1, 1, 0, 0, 0],
            "Y": [0, 1, 1, 0, 1, 1, 0, 0],
            "Z": [1, 0, 1, 1, 0, 0, 1, 1],
        }
    )

    methods = ["fpgrowth", "eclat", "fin", "lcm", "auto"]
    results: dict[str, set[tuple]] = {}

    for method in methods:
        res = fpgrowth(df, min_support=0.4, use_colnames=True, method=method)
        results[method] = set(_itemsets_as_tuples(res))

    # All methods should agree on the set of frequent itemsets
    reference = results["fpgrowth"]
    for method_name in methods[1:]:
        assert results[method_name] == reference, (
            f"Method '{method_name}' produced different itemsets than 'fpgrowth'.\n"
            f"  fpgrowth: {reference}\n"
            f"  {method_name}: {results[method_name]}"
        )

    # Also verify the reference is non-empty (sanity)
    assert len(reference) > 0


def test_varying_data_produces_different_results() -> None:
    """Smoke test: two clearly different datasets must produce different itemsets."""
    df_a = pd.DataFrame(
        {
            "P": [1, 1, 1, 1, 1],
            "Q": [1, 1, 1, 1, 1],
            "R": [0, 0, 0, 0, 0],
        }
    )

    df_b = pd.DataFrame(
        {
            "P": [0, 0, 0, 0, 0],
            "Q": [0, 0, 0, 0, 0],
            "R": [1, 1, 1, 1, 1],
        }
    )

    res_a = fpgrowth(df_a, min_support=0.5, use_colnames=True)
    res_b = fpgrowth(df_b, min_support=0.5, use_colnames=True)

    sets_a = set(_itemsets_as_tuples(res_a))
    sets_b = set(_itemsets_as_tuples(res_b))

    # Dataset A should have P, Q, {P,Q} — no R
    assert ("P",) in sets_a
    assert ("Q",) in sets_a
    assert ("P", "Q") in sets_a  # sorted order
    assert ("R",) not in sets_a

    # Dataset B should have R only — no P, Q
    assert ("R",) in sets_b
    assert ("P",) not in sets_b
    assert ("Q",) not in sets_b

    # The two results must be completely different
    assert sets_a != sets_b
    assert sets_a.isdisjoint(sets_b)
