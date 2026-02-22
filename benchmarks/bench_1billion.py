import time

import numpy as np

import rusket


def run_1b_challenge():
    print("=" * 60)
    print(" 🚀 THE 1 BILLION ROW CHALLENGE 🚀")
    print("=" * 60)

    # 1 Billion items generated iteratively to save RAM
    n_rows_total = 1_000_000_000
    n_items = 5_000  # 5000 unique products
    chunk_size = 50_000_000  # 50M rows per chunk

    # We want actual itemsets!
    # To do this, we inject correlated "power items" (like Bread & Butter)
    # 90% random noise, 10% correlated transactions

    miner = rusket.FPMiner(n_items=n_items)

    print(f"Goal: {n_rows_total:,} rows streaming into FPMiner")
    t_start = time.perf_counter()
    rows_processed = 0
    rng = np.random.default_rng(42)

    while rows_processed < n_rows_total:
        t0 = time.perf_counter()

        # We need ~1M transactions per chunk (avg basket = 50 items)
        n_txns = chunk_size // 50

        # 1. Generate random base distribution
        txn_ids = rng.integers(0, n_txns, size=chunk_size, dtype=np.int64)
        # Shift txns to be globally unique
        txn_ids += rows_processed // 50

        item_ids = rng.integers(0, n_items, size=chunk_size, dtype=np.int32)

        # 2. Inject a highly correlated frequent itemset (Items 42, 43, 44, 45)
        # Force 100,000 transactions in this chunk to contain these 4 items
        correlated_txns = rng.choice(n_txns, size=100_000, replace=False) + (rows_processed // 50)

        # Flattened injection arrays
        inj_txns = np.repeat(correlated_txns, 4)
        inj_items = np.tile([42, 43, 44, 45], 100_000).astype(np.int32)

        # Combine
        final_txns = np.concatenate([txn_ids, inj_txns])
        final_items = np.concatenate([item_ids, inj_items])

        # 3. Add to FPMiner (sorts and stores)
        miner.add_chunk(final_txns, final_items)

        rows_processed += len(final_items)
        print(
            f"  Added {len(final_items):,} rows in {time.perf_counter() - t0:.1f}s — Total: {rows_processed:,}/{n_rows_total:,}"
        )

    t_build = time.perf_counter() - t_start
    print(f"\n✅ Successfully ingested {rows_processed:,} rows into FPMiner in {t_build:.1f}s")

    print("\nStarting K-Way Merge & Mining phase...")
    t0 = time.perf_counter()
    # 0.005 min_support = 0.5%. We injected 100k patterns per chunk = 2M total patterns / ~20M total transactions -> ~10% support
    freq = miner.mine(min_support=0.05, max_len=4)
    t_mine = time.perf_counter() - t0

    print(f"✅ Mining Complete in {t_mine:.1f}s")
    print(f"Found {len(freq)} frequent itemsets!")
    print("\nTop patterns:")
    print(freq.sort_values("support", ascending=False).head(10))
    print(f"\n🏆 Total Time: {t_build + t_mine:.1f}s")


if __name__ == "__main__":
    run_1b_challenge()
