import time
import numpy as np

import rusket
import pandas as pd

try:
    from prefixspan import PrefixSpan as PyPrefixSpan
except ImportError:
    PyPrefixSpan = None


def generate_synthetic_sequences(n_users=2000, seq_len=10, vocab_size=50):
    np.random.seed(42)
    # create event log
    user_ids = np.repeat(np.arange(n_users), seq_len)
    item_ids = np.random.randint(0, vocab_size, size=n_users * seq_len)
    timestamps = np.tile(np.arange(seq_len), n_users)
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "timestamp": timestamps
    })
    return df

def run_benchmark():
    n_users = 2000
    seq_len = 15
    vocab_size = 100
    min_support = 20  # absolute support count
    
    # 1. Generate data
    df_long = generate_synthetic_sequences(n_users, seq_len, vocab_size)
    print("="*50)
    print("PREFIXSPAN BENCHMARK")
    print(f"Dataset: {n_users} users, {seq_len} events/user, vocab: {vocab_size}")
    print("="*50)
    
    # 2. Rusket
    print("\nRunning Rusket PrefixSpan...")
    t0 = time.perf_counter()
    rusket_model = rusket.PrefixSpan.from_transactions(
        df_long,
        transaction_col="user_id",
        item_col="item_id",
        time_col="timestamp",
        min_support=min_support,
        max_len=5
    )
    res_rusket = rusket_model.mine()
    rusket_time = time.perf_counter() - t0
    print(f"🏆 Rusket PrefixSpan time: {rusket_time:.4f}s (Found {len(res_rusket)} sequences)")

    # 3. pyprefixspan
    if PyPrefixSpan:
        print("\nRunning 'prefixspan' (PyPI library)...")
        # pyprefixspan expects a list of lists (sequences)
        grouped = df_long.sort_values(['user_id', 'timestamp']).groupby('user_id')['item_id'].apply(list).tolist()
        
        t0 = time.perf_counter()
        ps = PyPrefixSpan(grouped)
        res_py = ps.frequent(min_support, closed=False)
        py_time = time.perf_counter() - t0
        print(f"🐢 PyPrefixSpan time:      {py_time:.4f}s (Found {len(res_py)} sequences)")
        
        print("\n--- Benchmark Results ---")
        print(f"Speedup Rusket vs prefixspan pkg: {py_time / rusket_time:.2f}x faster")
    else:
        print("\n'prefixspan' package not installed. Skipping baseline.")

if __name__ == "__main__":
    run_benchmark()
