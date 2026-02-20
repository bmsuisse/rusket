"""Benchmark: FPMiner streaming accumulator vs fpgrowth on a one-hot DataFrame.

Usage:
    uv run python benchmarks/bench_fpminer_vs_fpgrowth.py

Tests two end-to-end paths for the SAME data:
  A. from_transactions → fpgrowth  (standard pandas path)
  B. FPMiner.add_chunk → FPMiner.mine  (streaming Rust path)

Output is a markdown table.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rusket import FPMiner, fpgrowth, from_transactions


@dataclass
class Result:
    label: str
    rows: int
    n_txns: int
    n_items: int
    prep_s: float  # Convert → OHE or add_chunk time
    mine_s: float  # fpgrowth / mine() time
    itemsets: int


def bench_fpgrowth(
    rng: np.random.Generator,
    n_rows: int,
    n_txns: int,
    n_items: int,
    min_support: float,
    max_len: int,
) -> Result:
    txn_ids = rng.integers(0, n_txns, size=n_rows, dtype=np.int64)
    item_ids = rng.integers(0, n_items, size=n_rows, dtype=np.int32)

    df = pd.DataFrame({"t": txn_ids.astype(str), "i": item_ids.astype(str)})
    del txn_ids, item_ids

    t0 = time.perf_counter()
    ohe = from_transactions(df)
    prep_s = time.perf_counter() - t0
    del df
    gc.collect()

    t0 = time.perf_counter()
    freq = fpgrowth(ohe, min_support=min_support, max_len=max_len)
    mine_s = time.perf_counter() - t0
    del ohe
    gc.collect()

    return Result("fpgrowth", n_rows, n_txns, n_items, prep_s, mine_s, len(freq))


def bench_fpminer(
    rng: np.random.Generator,
    n_rows: int,
    n_txns: int,
    n_items: int,
    min_support: float,
    max_len: int,
    chunk_size: int = 10_000_000,
) -> Result:
    miner = FPMiner(n_items=n_items)

    t0 = time.perf_counter()
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        size = end - start
        txn_ids = rng.integers(0, n_txns, size=size, dtype=np.int64)
        item_ids = rng.integers(0, n_items, size=size, dtype=np.int32)
        miner.add_chunk(txn_ids, item_ids)
        del txn_ids, item_ids
    gc.collect()
    prep_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    freq = miner.mine(min_support=min_support, max_len=max_len)
    mine_s = time.perf_counter() - t0
    del miner
    gc.collect()

    return Result("FPMiner", n_rows, n_txns, n_items, prep_s, mine_s, len(freq))


def print_table(results: list[Result]) -> None:
    hdr = f"{'label':<10} {'rows':>12} {'n_txns':>10} {'n_items':>8} {'prep':>8} {'mine':>8} {'total':>8} {'itemsets':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        total = r.prep_s + r.mine_s
        print(
            f"{r.label:<10} {r.rows:>12,} {r.n_txns:>10,} {r.n_items:>8,} "
            f"{r.prep_s:>7.1f}s {r.mine_s:>7.1f}s {total:>7.1f}s {r.itemsets:>10,}"
        )


SCENARIOS = [
    # (n_rows, n_txns, n_items, min_support, max_len)
    (100_000, 10_000, 100, 0.05, 3),
    (1_000_000, 100_000, 200, 0.05, 3),
    (10_000_000, 1_000_000, 500, 0.02, 3),
    (50_000_000, 5_000_000, 1000, 0.01, 3),
]

FPMINER_ONLY_SCENARIOS = [
    # large scale — skip fpgrowth (OOM risk)
    (200_000_000, 10_000_000, 2000, 0.01, 2),
]


if __name__ == "__main__":
    print("=== FPMiner vs fpgrowth benchmark ===\n")
    results: list[Result] = []

    for n_rows, n_txns, n_items, min_support, max_len in SCENARIOS:
        rng = np.random.default_rng(42)
        print(f"[{n_rows:,} rows] fpgrowth ...", end=" ", flush=True)
        try:
            r = bench_fpgrowth(rng, n_rows, n_txns, n_items, min_support, max_len)
            results.append(r)
            print(f"done ({r.prep_s + r.mine_s:.1f}s)", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

        rng = np.random.default_rng(42)
        print(f"[{n_rows:,} rows] FPMiner  ...", end=" ", flush=True)
        r = bench_fpminer(rng, n_rows, n_txns, n_items, min_support, max_len)
        results.append(r)
        print(f"done ({r.prep_s + r.mine_s:.1f}s)", flush=True)

    print(f"\n{'─' * 40} FPMiner-only (large scale) {'─' * 40}")
    for n_rows, n_txns, n_items, min_support, max_len in FPMINER_ONLY_SCENARIOS:
        rng = np.random.default_rng(42)
        print(f"[{n_rows:,} rows] FPMiner ...", end=" ", flush=True)
        r = bench_fpminer(rng, n_rows, n_txns, n_items, min_support, max_len)
        results.append(r)
        print(f"done ({r.prep_s + r.mine_s:.1f}s)", flush=True)

    print("\n=== Results ===")
    print_table(results)
