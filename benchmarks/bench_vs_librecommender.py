"""Benchmark rusket vs LibRecommender on MovieLens 100k.

Compares: ALS, SVD, BPR, LightGCN, ItemKNN/ItemCF, FM
Measures: fit time and NDCG@10

Usage:
    uv run python benchmarks/bench_vs_librecommender.py
"""

import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Data loading ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data" / "ml-100k"


def load_movielens_100k() -> pd.DataFrame:
    """Download and return the MovieLens 100k dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "u.data"

    if not data_file.exists():
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = DATA_DIR.parent / "ml-100k.zip"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR.parent)
        print("Done.")

    return pd.read_csv(
        data_file,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Random 80/20 split."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_size
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def build_sparse_matrix(df: pd.DataFrame, n_users: int, n_items: int, implicit: bool = True):
    """Build CSR matrix from DataFrame."""
    from scipy import sparse

    if implicit:
        vals = np.ones(len(df), dtype=np.float32)
    else:
        vals = df["rating"].values.astype(np.float32)

    return sparse.csr_matrix(
        (vals, (df["user_id"].values, df["item_id"].values)),
        shape=(n_users, n_items),
    )


def evaluate_ndcg(model, test_df, n=10):
    """Compute NDCG@k from a fitted rusket model."""
    user_true = {}
    for _, row in test_df.iterrows():
        u = int(row["user_id"])
        i = int(row["item_id"])
        if u not in user_true:
            user_true[u] = []
        user_true[u].append(i)

    ndcg_vals = []
    for u, true_items in user_true.items():
        try:
            recs, _ = model.recommend_items(u, n=n, exclude_seen=True)
        except Exception:
            ndcg_vals.append(0.0)
            continue

        recs_list = list(recs) if hasattr(recs, '__iter__') else [recs]
        true_set = set(true_items)

        dcg = 0.0
        for rank, item in enumerate(recs_list):
            if item in true_set:
                dcg += 1.0 / np.log2(rank + 2)

        ideal = sorted([1.0] * min(len(true_items), n), reverse=True)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal)))

        ndcg_vals.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcg_vals) if ndcg_vals else 0.0


# ── Benchmark functions ──────────────────────────────────────────────────

def bench_rusket(name: str, model_cls, train_df, test_df, n_users, n_items, **kwargs):
    """Benchmark a rusket model: fit time + NDCG@10."""
    implicit = kwargs.pop("implicit", True)
    mat = build_sparse_matrix(train_df, n_users, n_items, implicit=implicit)

    t0 = time.perf_counter()
    model = model_cls(**kwargs)
    model.fit(mat)
    fit_time = time.perf_counter() - t0

    ndcg = evaluate_ndcg(model, test_df)
    return fit_time, ndcg


def bench_libreco_model(model_cls_name, train_data, eval_data, data_info, pd_data, **model_kwargs):
    """Benchmark a LibRecommender model: fit time + NDCG@10."""
    from libreco.evaluation import evaluate as libreco_evaluate

    # Import the specific model class
    from libreco import algorithms
    model_cls = getattr(algorithms, model_cls_name)

    task = model_kwargs.pop("task", "ranking")
    neg_sampling = model_kwargs.pop("neg_sampling", True)

    model = model_cls(task=task, data_info=data_info, **model_kwargs)

    t0 = time.perf_counter()
    model.fit(train_data, neg_sampling=neg_sampling, verbose=0)
    fit_time = time.perf_counter() - t0

    # Evaluate
    metrics = libreco_evaluate(
        model, eval_data, neg_sampling=neg_sampling,
        metrics=["ndcg"], k=10,
    )
    ndcg = metrics.get("ndcg", {}).get(10, 0.0) if isinstance(metrics, dict) else 0.0

    return fit_time, ndcg


def run_benchmark():
    """Run the full benchmark suite."""
    print("\n" + "=" * 70)
    print("  BENCHMARK: rusket vs LibRecommender (MovieLens 100k)")
    print("=" * 70)

    # Load data
    df = load_movielens_100k()

    # Remap IDs to 0-based contiguous
    user_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item_map = {i: j for j, i in enumerate(sorted(df["item_id"].unique()))}
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    train_df, test_df = train_test_split(df)

    print(f"\n  Dataset: {len(df)} ratings, {n_users} users, {n_items} items")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Try importing LibRecommender
    has_libreco = False
    lr_train_data = None
    lr_eval_data = None
    lr_info = None
    try:
        from libreco.data import DatasetPure, random_split
        has_libreco = True
        print("  LibRecommender: ✅ available")

        # LibRecommender expects columns: user, item, label
        lr_df = df.rename(columns={"user_id": "user", "item_id": "item", "rating": "label"})
        lr_train_df, lr_eval_df = random_split(lr_df, test_size=0.2, seed=42)
        lr_train_data, lr_info = DatasetPure.build_trainset(lr_train_df)
        lr_eval_data = DatasetPure.build_evalset(lr_eval_df)
    except ImportError:
        print("  LibRecommender: ❌ not installed (install with: pip install LibRecommender)")
        print("  → Running rusket-only benchmarks\n")

    import rusket

    # Warmup LibRecommender runtime (TF/PyTorch JIT, graph compilation, etc.)
    if has_libreco and lr_train_data is not None:
        print("  Warming up LibRecommender runtime...")
        try:
            from libreco.algorithms import BPR as _WarmupBPR
            _warmup_model = _WarmupBPR(
                task="ranking", data_info=lr_info,
                embed_size=4, n_epochs=1, use_tf=False,
            )
            _warmup_model.fit(lr_train_data, neg_sampling=True, verbose=0)
            del _warmup_model
            # Rebuild data after warmup (LibRecommender mutates state)
            lr_train_data, lr_info = DatasetPure.build_trainset(lr_train_df)
            lr_eval_data = DatasetPure.build_evalset(lr_eval_df)
            print("  Warmup complete ✅\n")
        except Exception as e:
            print(f"  Warmup failed: {e}\n")

    results = []

    # ── ALS ──────────────────────────────────────────────────────────────
    print("\n  [ALS - Implicit]")
    t, ndcg = bench_rusket(
        "ALS", rusket.ALS, train_df, test_df, n_users, n_items,
        factors=64, iterations=15, regularization=0.01, alpha=40.0, seed=42,
    )
    print(f"    rusket ALS:          {t:.3f}s  NDCG@10={ndcg:.4f}")
    results.append(("ALS", "rusket", t, ndcg))

    if has_libreco and lr_train_data is not None:
        try:
            from libreco.algorithms import ALS as LibALS

            # Rebuild data (LibRecommender mutates state after each fit)
            lr_train_data, lr_info = DatasetPure.build_trainset(lr_train_df)
            lr_eval_data = DatasetPure.build_evalset(lr_eval_df)

            lr_model = LibALS(
                task="ranking", data_info=lr_info,
                embed_size=64, n_epochs=15, reg=0.01, alpha=40,
            )
            t0 = time.perf_counter()
            lr_model.fit(lr_train_data, neg_sampling=True, verbose=0)
            lr_t = time.perf_counter() - t0
            print(f"    LibRecommender ALS:  {lr_t:.3f}s")
            speedup = lr_t / t if t > 0 else 0
            print(f"    → rusket is {speedup:.1f}x faster")
            results.append(("ALS", "LibRecommender", lr_t, 0.0))
        except Exception as e:
            print(f"    LibRecommender ALS:  FAILED ({e})")

    # ── SVD ──────────────────────────────────────────────────────────────
    print("\n  [SVD - Explicit]")
    t, ndcg = bench_rusket(
        "SVD", rusket.SVD, train_df, test_df, n_users, n_items,
        factors=64, iterations=20, learning_rate=0.005, regularization=0.02, seed=42,
        implicit=False,
    )
    print(f"    rusket SVD:          {t:.3f}s  NDCG@10={ndcg:.4f}")
    results.append(("SVD", "rusket", t, ndcg))

    if has_libreco and lr_train_data is not None:
        try:
            from libreco.algorithms import SVD as LibSVD

            lr_train_data, lr_info = DatasetPure.build_trainset(lr_train_df)
            lr_eval_data = DatasetPure.build_evalset(lr_eval_df)

            lr_model = LibSVD(
                task="ranking", data_info=lr_info,
                embed_size=64, n_epochs=20, lr=0.005, reg=0.02,
                use_tf=False,
            )
            t0 = time.perf_counter()
            lr_model.fit(lr_train_data, neg_sampling=True, verbose=0)
            lr_t = time.perf_counter() - t0
            print(f"    LibRecommender SVD:  {lr_t:.3f}s")
            speedup = lr_t / t if t > 0 else 0
            print(f"    → rusket is {speedup:.1f}x faster")
            results.append(("SVD", "LibRecommender", lr_t, 0.0))
        except Exception as e:
            print(f"    LibRecommender SVD:  FAILED ({e})")

    # ── BPR ──────────────────────────────────────────────────────────────
    print("\n  [BPR - Implicit]")
    t, ndcg = bench_rusket(
        "BPR", rusket.BPR, train_df, test_df, n_users, n_items,
        factors=64, iterations=10, learning_rate=0.01, regularization=0.01, seed=42,
    )
    print(f"    rusket BPR:          {t:.3f}s  NDCG@10={ndcg:.4f}")
    results.append(("BPR", "rusket", t, ndcg))

    if has_libreco and lr_train_data is not None:
        try:
            from libreco.algorithms import BPR as LibBPR

            lr_train_data, lr_info = DatasetPure.build_trainset(lr_train_df)
            lr_eval_data = DatasetPure.build_evalset(lr_eval_df)

            lr_model = LibBPR(
                task="ranking", data_info=lr_info,
                embed_size=64, n_epochs=10, lr=0.01,
                use_tf=False,
            )
            t0 = time.perf_counter()
            lr_model.fit(lr_train_data, neg_sampling=True, verbose=0)
            lr_t = time.perf_counter() - t0
            print(f"    LibRecommender BPR:  {lr_t:.3f}s")
            speedup = lr_t / t if t > 0 else 0
            print(f"    → rusket is {speedup:.1f}x faster")
            results.append(("BPR", "LibRecommender", lr_t, 0.0))
        except Exception as e:
            print(f"    LibRecommender BPR:  FAILED ({e})")

    # ── ItemKNN ──────────────────────────────────────────────────────────
    print("\n  [ItemKNN]")
    t0 = time.perf_counter()
    model = rusket.ItemKNN.from_transactions(train_df, "user_id", "item_id", k=100).fit()
    t = time.perf_counter() - t0
    ndcg = evaluate_ndcg(model, test_df)
    print(f"    rusket ItemKNN:      {t:.3f}s  NDCG@10={ndcg:.4f}")
    results.append(("ItemKNN", "rusket", t, ndcg))

    if has_libreco and lr_train_data is not None:
        try:
            from libreco.algorithms import ItemCF as LibItemCF

            lr_train_data, lr_info = DatasetPure.build_trainset(lr_train_df)
            lr_eval_data = DatasetPure.build_evalset(lr_eval_df)

            lr_model = LibItemCF(task="ranking", data_info=lr_info, k_sim=100)
            t0 = time.perf_counter()
            lr_model.fit(lr_train_data, neg_sampling=True, verbose=0)
            lr_t = time.perf_counter() - t0
            print(f"    LibRecommender ItemCF: {lr_t:.3f}s")
            speedup = lr_t / t if t > 0 else 0
            print(f"    → rusket is {speedup:.1f}x faster")
            results.append(("ItemKNN", "LibRecommender", lr_t, 0.0))
        except Exception as e:
            print(f"    LibRecommender ItemCF: FAILED ({e})")

    # ── LightGCN ─────────────────────────────────────────────────────────
    print("\n  [LightGCN]")
    t, ndcg = bench_rusket(
        "LightGCN", rusket.LightGCN, train_df, test_df, n_users, n_items,
        factors=64, iterations=10, learning_rate=0.001, regularization=1e-5, seed=42,
    )
    print(f"    rusket LightGCN:     {t:.3f}s  NDCG@10={ndcg:.4f}")
    results.append(("LightGCN", "rusket", t, ndcg))

    if has_libreco and lr_train_data is not None:
        try:
            from libreco.algorithms import LightGCN as LibLightGCN

            lr_train_data, lr_info = DatasetPure.build_trainset(lr_train_df)
            lr_eval_data = DatasetPure.build_evalset(lr_eval_df)

            lr_model = LibLightGCN(
                task="ranking", data_info=lr_info,
                embed_size=64, n_epochs=10, lr=0.001,
                use_tf=False,
            )
            t0 = time.perf_counter()
            lr_model.fit(lr_train_data, neg_sampling=True, verbose=0)
            lr_t = time.perf_counter() - t0
            print(f"    LibRecommender LightGCN: {lr_t:.3f}s")
            speedup = lr_t / t if t > 0 else 0
            print(f"    → rusket is {speedup:.1f}x faster")
            results.append(("LightGCN", "LibRecommender", lr_t, 0.0))
        except Exception as e:
            print(f"    LibRecommender LightGCN: FAILED ({e})")

    # ── EASE ─────────────────────────────────────────────────────────────
    print("\n  [EASE]")
    t0 = time.perf_counter()
    model = rusket.EASE.from_transactions(train_df, "user_id", "item_id").fit()
    t = time.perf_counter() - t0
    ndcg = evaluate_ndcg(model, test_df)
    print(f"    rusket EASE:         {t:.3f}s  NDCG@10={ndcg:.4f}")
    results.append(("EASE", "rusket", t, ndcg))

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<12} {'Library':<18} {'Time (s)':<12} {'NDCG@10':<10}")
    print("  " + "-" * 54)
    for model_name, lib, t, ndcg in results:
        ndcg_str = f"{ndcg:.4f}" if ndcg > 0 else "N/A"
        print(f"  {model_name:<12} {lib:<18} {t:<12.3f} {ndcg_str:<10}")
    print()


if __name__ == "__main__":
    run_benchmark()
