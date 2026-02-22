import time

import numpy as np
import pandas as pd

from rusket import mine

try:
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
except ImportError:
    mlxtend_fpgrowth = None


def generate_basket_data(n_transactions=100_000, n_items=500, density=0.03):
    """Generates a boolean DataFrame simulating shopping baskets."""
    np.random.seed(42)
    mat = np.random.rand(n_transactions, n_items) < density
    df = pd.DataFrame(mat, columns=[f"Item_{i}" for i in range(n_items)])
    return df


def run_fpm_benchmark():
    n_txn = 50_000
    n_items = 500
    density = 0.04
    min_support = 0.05

    print("--- Generating synthetic retail dataset ---")
    print(f"Transactions: {n_txn:,}, Items: {n_items}, Density: {density * 100:.1f}%")
    df = generate_basket_data(n_txn, n_items, density=density)

    print(f"\nTarget Minimum Support: {min_support * 100:.1f}%")
    print("-" * 50)

    # 1. Rusket FP-Growth
    t0 = time.time()
    res_rusket_fp = mine(df, min_support=min_support, method="fpgrowth")
    t1 = time.time()
    rusket_fp_time = t1 - t0
    print(
        f"🏆 Rusket FP-Growth time: {rusket_fp_time:.4f}s (Found {len(res_rusket_fp)} itemsets)"
    )

    # 2. Rusket ECLAT
    t0 = time.time()
    _res_rusket_eclat = mine(df, min_support=min_support, method="eclat")
    t1 = time.time()
    rusket_eclat_time = t1 - t0
    print(f"🏆 Rusket ECLAT time:     {rusket_eclat_time:.4f}s")

    # 3. MLxtend FP-Growth
    if mlxtend_fpgrowth is not None:
        print(
            "\nEvaluating mlxtend (Standard Python Baseline)... this may take a while."
        )
        t0 = time.time()
        res_mlxtend = mlxtend_fpgrowth(df, min_support=min_support)
        t1 = time.time()
        mlxtend_time = t1 - t0
        print(
            f"🐢 MLxtend FP-Growth:     {mlxtend_time:.4f}s (Found {len(res_mlxtend)} itemsets)"
        )

        print("\n--- Benchmark Results ---")
        print(
            f"Speedup Rusket FP-Growth vs MLxtend: {mlxtend_time / rusket_fp_time:.2f}x faster"
        )
        print(
            f"Speedup Rusket ECLAT vs MLxtend:     {mlxtend_time / rusket_eclat_time:.2f}x faster"
        )
    else:
        print("\nMLxtend not installed. Skipping MLxtend benchmark.")


if __name__ == "__main__":
    run_fpm_benchmark()
