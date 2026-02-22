import time

import pandas as pd

import rusket


def run():
    print("Loading ml-100k...")
    df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    print("\n" + "=" * 50)
    print("LIGHTGCN BASELINE (ml-100k dataset)")
    print("=" * 50)

    # Let's train BPR first for a baseline, then mock LightGCN
    t0 = time.perf_counter()
    rusket.BPR.from_transactions(df, "user_id", "item_id", iterations=10)
    t1 = time.perf_counter()
    print(f"[BPR (10 Epo)]   Fit Time: {t1 - t0:.4f}s")


if __name__ == "__main__":
    run()
