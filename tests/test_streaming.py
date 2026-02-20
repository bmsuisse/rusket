"""Tests verifying that FPMiner produces the same results as fpgrowth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rusket import FPMiner, fpgrowth, from_transactions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_sorted_frozensets(df: pd.DataFrame) -> set[tuple[float, frozenset]]:
    """Convert output DataFrame to a comparable set of (support, frozenset) tuples."""
    result = set()
    for _, row in df.iterrows():
        items = row["itemsets"]
        # Handle both ArrowDtype list and frozenset representations
        if hasattr(items, "__iter__") and not isinstance(items, (str, frozenset)):
            items = frozenset(str(x) for x in items)
        elif isinstance(items, frozenset):
            items = frozenset(str(x) for x in items)
        else:
            items = frozenset({str(items)})
        result.add((round(float(row["support"]), 6), items))
    return result


def _cmp(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Assert that two result DataFrames contain the same itemsets and supports."""
    sa = _to_sorted_frozensets(a)
    sb = _to_sorted_frozensets(b)
    assert sa == sb, (
        f"FPMiner and fpgrowth results differ.\n"
        f"Only in FPMiner:  {sa - sb}\n"
        f"Only in fpgrowth: {sb - sa}"
    )


def _build_reference(txn_ids: np.ndarray, item_ids: np.ndarray, n_items: int,
                     min_support: float, max_len: int | None) -> pd.DataFrame:
    """Build reference result using from_transactions + fpgrowth."""
    df = pd.DataFrame({"t": txn_ids, "i": item_ids})
    ohe = from_transactions(df)
    return fpgrowth(ohe, min_support=min_support, use_colnames=True, max_len=max_len)


# ---------------------------------------------------------------------------
# Small / deterministic tests
# ---------------------------------------------------------------------------


class TestFPMinerCorrectness:
    """FPMiner output must match fpgrowth on the same data."""

    def test_tiny_manual(self) -> None:
        """Hand-crafted example from the docs."""
        # txn 0: {0,1,2}, txn 1: {0,1,3}, txn 2: {0,2,3}, txn 3: {0,1}, txn 4: {1,2}
        txn_ids  = np.array([0,0,0, 1,1,1, 2,2,2, 3,3, 4,4], dtype=np.int64)
        item_ids = np.array([0,1,2, 0,1,3, 0,2,3, 0,1, 1,2], dtype=np.int32)

        n_items = 4
        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)

        for min_support in [0.4, 0.6, 0.8]:
            result  = miner.mine(min_support=min_support)
            ref     = _build_reference(txn_ids, item_ids, n_items, min_support, None)
            _cmp(result, ref)

    def test_tiny_with_max_len(self) -> None:
        """With max_len=1, only singletons should appear."""
        txn_ids  = np.array([0,0,0, 1,1,1, 2,2,2, 3,3, 4,4], dtype=np.int64)
        item_ids = np.array([0,1,2, 0,1,3, 0,2,3, 0,1, 1,2], dtype=np.int32)
        n_items = 4
        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        result = miner.mine(min_support=0.4, max_len=1)
        ref    = _build_reference(txn_ids, item_ids, n_items, 0.4, 1)
        _cmp(result, ref)

    def test_duplicates_in_same_transaction(self) -> None:
        """Duplicate (txn, item) pairs must be deduplicated."""
        txn_ids  = np.array([0,0,0, 0,0,  1,1, 1],   dtype=np.int64)
        item_ids = np.array([0,1,2, 0,1,  0,1, 0],   dtype=np.int32)  # 0 repeated
        n_items = 3
        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        result = miner.mine(min_support=0.5)
        ref    = _build_reference(txn_ids, item_ids, n_items, 0.5, None)
        _cmp(result, ref)

    def test_single_transaction(self) -> None:
        txn_ids  = np.array([0, 0, 0], dtype=np.int64)
        item_ids = np.array([0, 1, 2], dtype=np.int32)
        n_items = 3
        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        result = miner.mine(min_support=1.0)
        ref    = _build_reference(txn_ids, item_ids, n_items, 1.0, None)
        _cmp(result, ref)

    def test_all_below_threshold(self) -> None:
        """When nothing meets min_support, both should return empty DataFrames."""
        txn_ids  = np.array([0, 1, 2, 3], dtype=np.int64)
        item_ids = np.array([0, 1, 2, 3], dtype=np.int32)  # each item in only 1 txn
        n_items = 4
        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        result = miner.mine(min_support=0.5)
        assert len(result) == 0

    def test_multiple_chunks_same_as_single(self) -> None:
        """Splitting into 3 chunks must give same result as one chunk."""
        rng = np.random.default_rng(0)
        n_items = 20
        n_txns = 500
        txn_ids  = rng.integers(0, n_txns, size=5_000, dtype=np.int64)
        item_ids = rng.integers(0, n_items, size=5_000, dtype=np.int32)

        # Single chunk
        miner1 = FPMiner(n_items=n_items)
        miner1.add_chunk(txn_ids, item_ids)
        result1 = miner1.mine(min_support=0.1, max_len=3)

        # Three chunks
        miner3 = FPMiner(n_items=n_items)
        chunk = len(txn_ids) // 3
        for start in range(0, len(txn_ids), chunk):
            miner3.add_chunk(txn_ids[start:start+chunk], item_ids[start:start+chunk])
        result3 = miner3.mine(min_support=0.1, max_len=3)

        _cmp(result1, result3)

    def test_eclat_method(self) -> None:
        """FPMiner with method='eclat' must match fpgrowth results."""
        rng = np.random.default_rng(99)
        n_items = 15
        n_txns = 200
        txn_ids  = rng.integers(0, n_txns, size=2000, dtype=np.int64)
        item_ids = rng.integers(0, n_items, size=2000, dtype=np.int32)

        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        result_eclat   = miner.mine(min_support=0.1, max_len=3, method="eclat")
        result_fpgrowth = miner.mine(min_support=0.1, max_len=3, method="fpgrowth")
        _cmp(result_eclat, result_fpgrowth)

    def test_n_rows_property(self) -> None:
        txn_ids  = np.array([0, 0, 1], dtype=np.int64)
        item_ids = np.array([0, 1, 0], dtype=np.int32)
        miner = FPMiner(n_items=3)
        miner.add_chunk(txn_ids, item_ids)
        assert miner.n_rows == 3
        miner.add_chunk(txn_ids, item_ids)
        assert miner.n_rows == 6

    def test_reset(self) -> None:
        txn_ids  = np.array([0, 0, 1], dtype=np.int64)
        item_ids = np.array([0, 1, 0], dtype=np.int32)
        miner = FPMiner(n_items=3)
        miner.add_chunk(txn_ids, item_ids)
        miner.reset()
        assert miner.n_rows == 0
        result = miner.mine(min_support=0.5)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Medium-scale randomised test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed,n_txns,n_items,n_rows,min_support,max_len", [
    (1,  1_000, 50, 20_000, 0.05, 3),
    (2,  5_000, 30, 50_000, 0.05, 2),
    (3, 10_000, 20,100_000, 0.1,  3),
])
def test_fpminer_matches_fpgrowth_random(
    seed: int, n_txns: int, n_items: int, n_rows: int,
    min_support: float, max_len: int
) -> None:
    """FPMiner must produce same result as fpgrowth on random data."""
    rng = np.random.default_rng(seed)
    txn_ids  = rng.integers(0, n_txns,  size=n_rows, dtype=np.int64)
    item_ids = rng.integers(0, n_items, size=n_rows, dtype=np.int32)

    miner = FPMiner(n_items=n_items)
    miner.add_chunk(txn_ids, item_ids)
    result = miner.mine(min_support=min_support, max_len=max_len)
    ref    = _build_reference(txn_ids, item_ids, n_items, min_support, max_len)
    _cmp(result, ref)


# ---------------------------------------------------------------------------
# Result validation tests (correctness of output, not just equality)
# ---------------------------------------------------------------------------


def _make_realistic_data(
    rng: np.random.Generator,
    n_txns: int,
    n_items: int,
    avg_basket: float,
    item_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic basket data matching a given item frequency distribution."""
    basket_sizes = rng.poisson(avg_basket, size=n_txns).clip(1, n_items)
    total = int(basket_sizes.sum())
    txn_ids = np.repeat(np.arange(n_txns, dtype=np.int64), basket_sizes)
    item_ids = rng.choice(n_items, size=total, p=item_probs).astype(np.int32)
    return txn_ids, item_ids


class TestFPMinerResultValidation:
    """Validate properties of FPMiner output without needing an exact reference."""

    def _retail_probs(self, n_items: int) -> np.ndarray:
        """Simulate a skewed retail item distribution (power law)."""
        raw = np.arange(1, n_items + 1, dtype=np.float64) ** -1.2
        return raw / raw.sum()

    def test_support_values_in_bounds(self) -> None:
        """All reported supports must be in [min_support, 1.0]."""
        rng = np.random.default_rng(7)
        n_items = 30
        probs = self._retail_probs(n_items)
        txn_ids, item_ids = _make_realistic_data(rng, 5_000, n_items, 5.0, probs)

        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        min_support = 0.05
        result = miner.mine(min_support=min_support, max_len=3)

        assert len(result) > 0, "Expected some itemsets with retail distribution"
        assert (result["support"] >= min_support - 1e-9).all(), "Support below threshold found"
        assert (result["support"] <= 1.0 + 1e-9).all(), "Support above 1.0 found"

    def test_support_monotone_with_subset(self) -> None:
        """Support of a superset must be <= support of any of its subsets."""
        rng = np.random.default_rng(8)
        n_items = 20
        probs = self._retail_probs(n_items)
        txn_ids, item_ids = _make_realistic_data(rng, 3_000, n_items, 6.0, probs)

        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        result = miner.mine(min_support=0.05, max_len=3)

        if len(result) == 0:
            pytest.skip("No itemsets found")

        # Build support lookup
        support_map: dict[frozenset, float] = {}
        for _, row in result.iterrows():
            items = frozenset(int(x) for x in row["itemsets"])
            support_map[items] = float(row["support"])

        # Check: for every 2+ itemset, all subsets must have >= support
        for items, sup in support_map.items():
            if len(items) < 2:
                continue
            for item in items:
                subset = items - {item}
                if subset in support_map:
                    assert support_map[subset] >= sup - 1e-9, (
                        f"Monotonicity violated: {subset} support={support_map[subset]} "
                        f"< {items} support={sup}"
                    )

    def test_fpminer_matches_fpgrowth_realistic(self) -> None:
        """FPMiner must match fpgrowth on realistic retail-like bootstrapped data."""
        rng = np.random.default_rng(42)
        n_items = 50
        n_txns = 2_000
        avg_basket = 5.0
        probs = self._retail_probs(n_items)
        txn_ids, item_ids = _make_realistic_data(rng, n_txns, n_items, avg_basket, probs)

        # Run FPMiner in 3 chunks
        miner = FPMiner(n_items=n_items)
        chunk = len(txn_ids) // 3
        for start in range(0, len(txn_ids), chunk):
            miner.add_chunk(txn_ids[start:start + chunk], item_ids[start:start + chunk])
        result = miner.mine(min_support=0.05, max_len=3)

        # Reference
        ref = _build_reference(txn_ids, item_ids, n_items, 0.05, 3)
        _cmp(result, ref)

    def test_no_false_positives(self) -> None:
        """Every itemset in FPMiner output must actually meet min_support."""
        rng = np.random.default_rng(11)
        n_items = 15
        n_txns = 500
        txn_ids  = rng.integers(0, n_txns, size=5_000, dtype=np.int64)
        item_ids = rng.integers(0, n_items, size=5_000, dtype=np.int32)

        miner = FPMiner(n_items=n_items)
        miner.add_chunk(txn_ids, item_ids)
        min_support = 0.1
        result = miner.mine(min_support=min_support, max_len=2)

        if len(result) == 0:
            return  # trivially valid

        # Count actual support from raw data
        txn_item_sets: dict[int, set] = {}
        for t, i in zip(txn_ids.tolist(), item_ids.tolist()):
            txn_item_sets.setdefault(t, set()).add(i)

        total_txns = len(txn_item_sets)
        for _, row in result.iterrows():
            items = set(int(x) for x in row["itemsets"])
            actual_count = sum(1 for s in txn_item_sets.values() if items.issubset(s))
            actual_support = actual_count / total_txns
            assert actual_support >= min_support - 1e-6, (
                f"False positive: {items} has actual support {actual_support:.4f} < {min_support}"
            )
