import time

import pandas as pd

import rusket


def run():
    print("Loading ml-100k...")
    df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    print("\n" + "=" * 50)
    print("RUSKET ABSOLUTE SPEED BENCHMARK (ml-100k dataset)")
    print("100,000 interactions | 943 Users | 1682 Items")
    print("=" * 50)

    # --- ItemKNN ---
    t0 = time.perf_counter()
    rusket.ItemKNN.from_transactions(df, "user_id", "item_id", k=100)
    t1 = time.perf_counter()
    print(f"[ItemKNN]        Fit Time: {t1 - t0:.4f}s")

    # --- BPR ---
    t0 = time.perf_counter()
    rusket.BPR.from_transactions(df, "user_id", "item_id", iterations=10)
    t1 = time.perf_counter()
    print(f"[BPR (10 Epo)]   Fit Time: {t1 - t0:.4f}s")

    # --- EASE ---
    t0 = time.perf_counter()
    rusket.EASE.from_transactions(df, "user_id", "item_id")
    t1 = time.perf_counter()
    print(f"[EASE]           Fit Time: {t1 - t0:.4f}s")

    # --- ALS ---
    t0 = time.perf_counter()
    rusket.ALS.from_transactions(df, "user_id", "item_id", iterations=10)
    t1 = time.perf_counter()
    print(f"[ALS (10 Epo)]   Fit Time: {t1 - t0:.4f}s")


if __name__ == "__main__":
    run()
