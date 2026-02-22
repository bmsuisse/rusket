import numpy as np
import scipy.sparse as sp
import pandas as pd
import time
from typing import Any, Literal
import rusket


def benchmark():
    df = pd.read_parquet("tests/.dataset_cache/online_retail_II_sample.parquet")
    print(f"Loaded dataset: {df.shape[0]} interactions")

    n_users = df["Customer_ID"].nunique()
    n_items = df["Description"].nunique()
    print(f"Users: {n_users}, Items: {n_items}, Density: {len(df) / (n_users * n_items):.4%}")

    print("\n--- Fitting Models ---")

    # EASE
    t0 = time.time()
    ease_model = rusket.EASE.from_transactions(
        df,
        user_col="Customer_ID",
        item_col="Description",
        regularization=500.0,
    )
    ease_fit_time = time.time() - t0
    print(f"EASE fit time: {ease_fit_time:.3f}s")

    # ALS
    t0 = time.time()
    als_model = rusket.ALS.from_transactions(
        df,
        user_col="Customer_ID",
        item_col="Description",
        factors=64,
        iterations=15,
        regularization=0.01,
        alpha=40.0,
        seed=42,
    )
    als_fit_time = time.time() - t0
    print(f"ALS fit time:  {als_fit_time:.3f}s")

    # ItemKNN BM25
    t0 = time.time()
    itemknn_model = rusket.ItemKNN.from_transactions(
        df,
        user_col="Customer_ID",
        item_col="Description",
        method="bm25",
        k=20,
    )
    itemknn_fit_time = time.time() - t0
    print(f"ItemKNN fit:   {itemknn_fit_time:.3f}s")

    print("\n--- Inference: Recommending 10 items for ALL users ---")

    # EASE inference benchmark
    t0 = time.time()
    for u in range(n_users):
        ease_model.recommend_items(u, 10)
    ease_inf_time = time.time() - t0
    print(f"EASE total inference time: {ease_inf_time:.3f}s ({ease_inf_time / n_users * 1000:.2f}ms per user)")

    # ALS inference benchmark
    t0 = time.time()
    for u in range(n_users):
        als_model.recommend_items(u, 10)
    als_inf_time = time.time() - t0
    print(f"ALS inference:  {als_inf_time:.3f}s ({als_inf_time / n_users * 1000:.2f}ms per user)")

    # ItemKNN BM25 inference benchmark
    t0 = time.time()
    for u in range(n_users):
        itemknn_model.recommend_items(u, 10)
    itemknn_inf_time = time.time() - t0
    print(f"ItemKNN inference: {itemknn_inf_time:.3f}s ({itemknn_inf_time / n_users * 1000:.2f}ms per user)")

    print(f"\nResult: EASE inference is {als_inf_time / ease_inf_time:.1f}x faster than ALS inference")
    print(f"Result: ItemKNN fit is {ease_fit_time / itemknn_fit_time:.1f}x faster than EASE fit")


if __name__ == "__main__":
    benchmark()
