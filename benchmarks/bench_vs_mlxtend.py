"""Benchmark: rusket FPMiner vs mlxtend.fpgrowth.

Usage:
    uv run python benchmarks/bench_vs_mlxtend.py

Tests end-to-end paths for the SAME data:
  A. mlxtend (pandas groupby/unstack -> mlxtend.fpgrowth)
  B. rusket FPMiner (streaming Rust path)

Output is a markdown table.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
from rusket import FPMiner


@dataclass
class Result:
    label: str
    rows: int
    n_txns: int
    n_items: int
    prep_s: float
    mine_s: float
    itemsets: int


def bench_mlxtend(
    rng: np.random.Generator,
    n_rows: int,
    n_txns: int,
    n_items: int,
    min_support: float,
    max_len: int,
) -> Result:
    txn_ids = rng.integers(0, n_txns, size=n_rows, dtype=np.int64)
    item_ids = rng.integers(0, n_items, size=n_rows, dtype=np.int32)

    df = pd.DataFrame({"t": txn_ids, "i": item_ids})
    del txn_ids, item_ids

    t0 = time.perf_counter()
    # Create one-hot encoded dataframe
    # mlxtend expects boolean DataFrame
    try:
        grouped = df.groupby(["t", "i"]).size().unstack(fill_value=0)
        ohe = grouped > 0
    except MemoryError:
        print(" [MemoryError during OHE prep] ", end="")
        return Result("mlxtend", n_rows, n_txns, n_items, -1, -1, 0)

    prep_s = time.perf_counter() - t0
    del df
    gc.collect()

    t0 = time.perf_counter()
    try:
        freq = mlxtend_fpgrowth(
            ohe, min_support=min_support, use_colnames=True, max_len=max_len
        )
        mine_s = time.perf_counter() - t0
        res_len = len(freq)
    except Exception as e:
        print(f" [Error: {type(e).__name__}] ", end="")
        return Result("mlxtend", n_rows, n_txns, n_items, prep_s, -1, 0)

    del ohe
    gc.collect()

    return Result("mlxtend", n_rows, n_txns, n_items, prep_s, mine_s, res_len)


def bench_fpminer(
    rng: np.random.Generator,
    n_rows: int,
    n_txns: int,
    n_items: int,
    min_support: float,
    max_len: int,
    chunk_size: int = 1_000_000,
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
    freq = miner.mine(min_support=min_support, max_len=max_len, method="eclat")
    mine_s = time.perf_counter() - t0
    del miner
    gc.collect()

    return Result("rusket", n_rows, n_txns, n_items, prep_s, mine_s, len(freq))


def print_table(results: list[Result]) -> None:
    hdr = f"{'label':<10} {'rows':>12} {'n_txns':>10} {'n_items':>8} {'prep':>8} {'mine':>8} {'total':>8} {'itemsets':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.prep_s < 0:
            print(
                f"{r.label:<10} {r.rows:>12,} {r.n_txns:>10,} {r.n_items:>8,} {'OOM':>8} {'-':>8} {'-':>8} {'-':>10}"
            )
        elif r.mine_s < 0:
            print(
                f"{r.label:<10} {r.rows:>12,} {r.n_txns:>10,} {r.n_items:>8,} {r.prep_s:>7.1f}s {'FAIL':>8} {'-':>8} {'-':>10}"
            )
        else:
            total = r.prep_s + r.mine_s
            print(
                f"{r.label:<10} {r.rows:>12,} {r.n_txns:>10,} {r.n_items:>8,} "
                f"{r.prep_s:>7.1f}s {r.mine_s:>7.1f}s {total:>7.1f}s {r.itemsets:>10,}"
            )


SCENARIOS = [
    # (n_rows, n_txns, n_items, min_support, max_len)
    (50_000, 10_000, 100, 0.05, 3),  # Small dataset
    (500_000, 50_000, 500, 0.02, 3),  # Medium dataset
    (2_000_000, 500_000, 2000, 0.01, 3),  # Larger dataset
]

if __name__ == "__main__":
    print("=== rusket vs mlxtend (Standard Python Data Science Library) ===\n")
    results: list[Result] = []

    for n_rows, n_txns, n_items, min_support, max_len in SCENARIOS:
        # Test mlxtend first
        rng = np.random.default_rng(42)
        print(f"[{n_rows:,} rows] mlxtend ...", end=" ", flush=True)
        r = bench_mlxtend(rng, n_rows, n_txns, n_items, min_support, max_len)
        results.append(r)
        if r.mine_s >= 0 and r.prep_s >= 0:
            print(f"done ({r.prep_s + r.mine_s:.1f}s)", flush=True)
        else:
            print("FAILED", flush=True)

        # Test rusket
        rng = np.random.default_rng(
            42
        )  # Reset RNG so the generated data is EXACTLY the same
        print(f"[{n_rows:,} rows] rusket  ...", end=" ", flush=True)
        r = bench_fpminer(rng, n_rows, n_txns, n_items, min_support, max_len)
        results.append(r)
        print(f"done ({r.prep_s + r.mine_s:.1f}s)", flush=True)
        print()

    print("\n=== Results ===")
    print_table(results)
