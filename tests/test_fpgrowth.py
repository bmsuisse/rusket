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
    def setUp(self) -> None:
        FPTestEdgeCases.setUp(self, fpgrowth)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self) -> None:
        FPTestErrors.setUp(self, fpgrowth)


class TestEx1(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:
        FPTestEx1All.setUp(self, fpgrowth)


class TestEx1BoolInput(unittest.TestCase, FPTestEx1All):
    def setUp(self) -> None:
        one_ary = np.array(
            [
                [False, False, False, True,  False, True,  True,  True,  True,  False, True],
                [False, False, True,  True,  False, True,  False, True,  True,  False, True],
                [True,  False, False, True,  False, True,  True,  False, False, False, False],
                [False, True,  False, False, False, True,  True,  False, False, True,  True],
                [False, True,  False, True,  True,  True,  False, False, True,  False, False],
            ]
        )
        FPTestEx1All.setUp(self, fpgrowth, one_ary=one_ary)


class TestEx2(unittest.TestCase, FPTestEx2All):
    def setUp(self) -> None:
        FPTestEx2All.setUp(self, fpgrowth)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self) -> None:
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
    return pd.DataFrame(
        {f"feature_{i:04d}": rng.random(n_rows) < support_values[i] for i in range(n_cols)}
    )


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
