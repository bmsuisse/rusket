"""ALS benchmark using real-world MovieLens data.

Auto-downloads MovieLens datasets on first run.

Usage:
    uv run python benchmarks/bench_als.py             # 1M + 10M + 25M (all three methods)
    uv run python benchmarks/bench_als.py --size 25m  # just 25M
    uv run python benchmarks/bench_als.py --size 200m # ~200M ratings from 1B dataset
    uv run python benchmarks/bench_als.py --size 1b   # full 1B (needs 8+ GB RAM)

Methods compared automatically:
    rusket (cg_iters=3)  — fast, slight quality tradeoff at large scale
    rusket (cg_iters=10) — default, best quality
    implicit             — popular Python ALS library (if installed)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tarfile
import time
import psutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from scipy import sparse

import rusket

# ---------------------------------------------------------------------------
# Published reference numbers (Spark MLlib, 4-node cluster, factors=100)
# ---------------------------------------------------------------------------
SPARK_REFERENCE = 180  # seconds for MovieLens 25M on a 4-node Spark cluster

MOVIELENS_URLS = {
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "1b": "https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar",
    "200m": "https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar",  # same source, sampled
}

FACTORS = 64
ITERS = 15
ALPHA = 40.0
REG = 0.01
TOP_N = 200  # users to time for recommend latency

# Target number of ratings for the 200M mode
TARGET_200M = 200_000_000


# ---------------------------------------------------------------------------
# Download / load MovieLens
# ---------------------------------------------------------------------------


def download_movielens(size: str, data_dir: Path) -> Path:
    """Download and extract MovieLens zip, return the ratings folder."""
    url = MOVIELENS_URLS[size]
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_to = data_dir / f"ml-{size}"

    # Walk existing extracted tree for ratings file
    if extract_to.exists():
        for root, _dirs, files in os.walk(extract_to):
            if "ratings.csv" in files or "ratings.dat" in files:
                return Path(root)
            if any(f.endswith(".npz") for f in files):
                return Path(root)

    is_tar = url.endswith(".tar")
    archive_path = data_dir / (f"ml-{size}.tar" if is_tar else f"ml-{size}.zip")
    if not archive_path.exists():
        print(f"  Downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, archive_path)
        print(f"  Saved {archive_path.stat().st_size / 1e6:.0f} MB", flush=True)

    print(f"  Extracting to {extract_to} ...", flush=True)
    extract_to.mkdir(parents=True, exist_ok=True)
    if is_tar:
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_to)
    else:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)

    for root, _dirs, files in os.walk(extract_to):
        if "ratings.csv" in files or "ratings.dat" in files:
            return Path(root)
        if any(f.endswith(".npz") for f in files):
            return Path(root)

    raise FileNotFoundError(f"No ratings file found after extracting {archive_path}")


def load_ratings_sampled_200m(folder: Path) -> sparse.csr_matrix:
    """Load ~200M ratings from the 1B dataset by reading the first N chunks.

    Builds a proper in-memory CSR without mmap so the solver runs at full speed.
    """

    npz_files = sorted(folder.glob("train*.npz"))
    all_u, all_i, all_r = [], [], []

    print(
        f"  Sampling ~{TARGET_200M // 1_000_000}M ratings from 1B dataset...",
        flush=True,
    )
    total = 0
    for npz in tqdm(npz_files, desc="  Reading NPZ files"):
        data = np.load(npz)
        arr = data["arr_0"]
        data.close()
        all_u.append(arr[:, 0].astype(np.int32))
        all_i.append(arr[:, 1].astype(np.int32))
        all_r.append(np.ones(len(arr), dtype=np.float32))
        total += len(arr)
        if total >= TARGET_200M:
            break

    uids = np.concatenate(all_u)
    iids = np.concatenate(all_i)
    vals = np.concatenate(all_r)
    del all_u, all_i, all_r
    gc.collect()

    # Remap to contiguous 0-based IDs
    u_map = {v: k for k, v in enumerate(np.unique(uids))}
    i_map = {v: k for k, v in enumerate(np.unique(iids))}
    u_idx = np.array([u_map[u] for u in uids], dtype=np.int32)
    i_idx = np.array([i_map[i] for i in iids], dtype=np.int32)
    n_users = len(u_map)
    n_items = len(i_map)

    mat = sparse.csr_matrix((vals, (u_idx, i_idx)), shape=(n_users, n_items))
    mat.sum_duplicates()
    print(
        f"  Loaded {n_users:,} users × {n_items:,} movies, {mat.nnz:,} ratings",
        flush=True,
    )
    return mat


def load_ratings_stream(folder: Path) -> sparse.csr_matrix:
    """Load ratings out-of-core into memory-mapped NumPy arrays.

    Safe for 1B dataset (~8 GB of CSR arrays stay on disk).
    Pass 1: Stream CSV, find shape and exact ratings per user.
    Pass 2: Allocate memmaps, stream again, and write data safely.
    """
    ratings_csv = folder / "ratings.csv"
    ratings_dat = folder / "ratings.dat"

    import pandas as pd  # type: ignore[import-untyped]

    is_csv = ratings_csv.exists()
    path = ratings_csv if is_csv else ratings_dat
    kwargs = (
        {"usecols": ["userId", "movieId", "rating"]}
        if is_csv
        else {
            "sep": "::",
            "names": ["userId", "movieId", "rating", "timestamp"],
            "usecols": [0, 1, 2],
            "engine": "python",
        }
    )

    # Pass 1: find max IDs and count ratings per user
    print("  Pass 1: Counting ratings to build indptr...", flush=True)
    nnz = 0
    max_user_id = -1
    max_item_id = -1

    # Pre-allocate dynamic array for user counts (safe up to 10M users)
    counts = np.zeros(1_000_000, dtype=np.int64)
    chunk_size = 5_000_000

    estimated_total = (
        1_000_000_000 if ("1b" in str(path) or "ml-20m" in str(path)) else None
    )

    # Generator to yield chunks from either CSV or NPZ files
    def get_chunks():
        if not is_csv and folder.name == "ml-20mx16x32":
            # 1B dataset: 16 .npz files containing arr_0 of shape (N, 2)
            npz_files = sorted(folder.glob("train*.npz"))
            for npz in npz_files:
                data = np.load(npz)
                arr = data["arr_0"]
                total_rows = len(arr)
                for start in range(0, total_rows, chunk_size):
                    end = min(start + chunk_size, total_rows)
                    # Yield user, item, rating (implicit 1.0 for this dataset)
                    yield (
                        arr[start:end, 0].astype(np.int32),
                        arr[start:end, 1].astype(np.int32),
                        np.ones(end - start, dtype=np.float32),
                    )
                data.close()
        else:
            for ch in pd.read_csv(path, chunksize=chunk_size, **kwargs):
                yield (
                    ch["userId"].to_numpy(dtype=np.int32),
                    ch["movieId"].to_numpy(dtype=np.int32),
                    ch["rating"].to_numpy(dtype=np.float32),
                )

    estimated_chunks = (
        iter(range(estimated_total // chunk_size + 1)) if estimated_total else None
    )

    for uids, iids, _ in tqdm(
        get_chunks(),
        desc="    Reading chunks",
        total=next(estimated_chunks) if estimated_chunks else None,
        unit="chunk",
    ):
        chunk_max_u = int(uids.max())
        if chunk_max_u > max_user_id:
            max_user_id = chunk_max_u
            if max_user_id >= len(counts):
                counts.resize(max_user_id + 5_000_000, refcheck=False)

        if int(iids.max()) > max_item_id:
            max_item_id = int(iids.max())

        chunk_counts = np.bincount(uids)
        counts[: len(chunk_counts)] += chunk_counts
        nnz += len(uids)

    n_users = max_user_id + 1
    n_items = max_item_id + 1
    counts = counts[:n_users]  # Shrink to exact bounds

    # Pass 2: allocate memmaps and scatter
    print(f"  Pass 2: Allocating 3 mmap files for {nnz:,} ratings...", flush=True)
    mmap_dir = folder.parent / f"{folder.name}_mmap"
    mmap_dir.mkdir(parents=True, exist_ok=True)

    indptr_file = mmap_dir / "indptr.dat"
    indices_file = mmap_dir / "indices.dat"
    data_file = mmap_dir / "data.dat"

    # Allocate & write indptr
    indptr = np.memmap(indptr_file, dtype=np.int64, mode="w+", shape=(n_users + 1,))
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indptr.flush()

    # Allocate indices and data
    indices = np.memmap(indices_file, dtype=np.int32, mode="w+", shape=(nnz,))
    data = np.memmap(data_file, dtype=np.float32, mode="w+", shape=(nnz,))

    # Track current writing position per user
    pos = indptr[:-1].copy()

    estimated_chunks = (
        iter(range(estimated_total // chunk_size + 1)) if estimated_total else None
    )
    for u, i, r in tqdm(
        get_chunks(),
        desc="    Writing mmap  ",
        total=next(estimated_chunks) if estimated_chunks else None,
        unit="chunk",
    ):
        # Sort by user so we can compute local target offsets vectorially.
        # Numpy argsort is fast for 5M rows
        sort_idx = np.argsort(u)
        u = u[sort_idx]
        i = i[sort_idx]
        r = r[sort_idx]

        # Vectorized generation of running offsets for duplicate users
        # e.g. users = [1, 1, 1, 2, 2] -> offsets = [0, 1, 2, 0, 1]
        idx = np.arange(len(u))
        changes = np.concatenate(([0], np.where(u[1:] != u[:-1])[0] + 1))
        starts = np.zeros(len(u), dtype=np.int32)
        starts[changes] = changes
        starts = np.maximum.accumulate(starts)
        local_offsets = idx - starts

        target_pos = pos[u] + local_offsets
        indices[target_pos] = i
        data[target_pos] = r

        chunk_counts = np.bincount(u)
        pos[: len(chunk_counts)] += chunk_counts

    indices.flush()
    data.flush()
    del counts
    del pos
    gc.collect()

    # Memory maps can be passed directly to scipy.sparse.csr_matrix
    # because they impl the buffer protocol perfectly.
    # To prevent scipy from downcasting int64->int32 or copying arrays:
    mat = sparse.csr_matrix((n_users, n_items))
    mat.indptr = indptr
    mat.indices = indices
    mat.data = data
    print(
        f"  Loaded {n_users:,} users × {n_items:,} movies, {nnz:,} ratings (out-of-core CSR)",
        flush=True,
    )
    return mat


def load_ratings(
    folder: Path, sample: float = 1.0, stream: bool = False
) -> sparse.csr_matrix:
    """Load ratings into a sparse CSR matrix, optionally sampling users."""
    import pandas as pd  # type: ignore[import-untyped]

    # Check if this is the 1B dataset folder structure
    if folder.name == "ml-20mx16x32":
        if stream:
            return load_ratings_stream(folder)

        # Fallback to in-memory load for subsets (if RAM allows)
        print("  Loading 1B .npz arrays into RAM...", flush=True)
        # We only implement stream mode for 1B to avoid OOM
        return load_ratings_stream(folder)

    if stream:
        return load_ratings_stream(folder)

    ratings_csv = folder / "ratings.csv"
    ratings_dat = folder / "ratings.dat"

    is_csv = ratings_csv.exists()

    if is_csv:
        df = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "rating"])
    else:
        df = pd.read_csv(
            ratings_dat,
            sep="::",
            names=["userId", "movieId", "rating", "timestamp"],
            usecols=[0, 1, 2],
            engine="python",
        )

    if sample < 1.0:
        users = df["userId"].unique()
        keep = np.random.default_rng(42).choice(
            users, size=int(len(users) * sample), replace=False
        )
        df = df[df["userId"].isin(keep)]
        print(
            f"  Sampled {sample * 100:.0f}% of users: {len(keep):,} users", flush=True
        )

    # Contiguous 0-based IDs
    df["user"] = df["userId"].astype("category").cat.codes.astype(np.int32)
    df["item"] = df["movieId"].astype("category").cat.codes.astype(np.int32)
    n_users = int(df["user"].max()) + 1
    n_items = int(df["item"].max()) + 1
    data = df["rating"].to_numpy(dtype=np.float32)
    rows = df["user"].to_numpy()
    cols = df["item"].to_numpy()
    del df
    gc.collect()

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    print(f"  {n_users:,} users × {n_items:,} movies, {mat.nnz:,} ratings", flush=True)
    return mat


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def _check_ram(fact_mb: float, csr_mb: float, is_mmap: bool) -> tuple[bool, str]:
    """Return (ok, reason). Skips if not enough RAM."""
    if fact_mb > 3_000:
        return False, "factor matrices > 3 GB"
    avail_mb = psutil.virtual_memory().available / 1e6
    required_mb = fact_mb if is_mmap else (fact_mb + csr_mb)
    if avail_mb - required_mb < 2_000:
        return (
            False,
            f"not enough RAM (need {required_mb:.0f} MB, have {avail_mb:.0f} MB)",
        )
    return True, ""


def run_method_rusket(
    mat: sparse.csr_matrix,
    label: str,
    cg_iters: int,
    verbose: bool = True,
    use_cholesky: bool = False,
) -> dict:
    """Benchmark rusket ALS with given cg_iters."""
    t0 = time.perf_counter()
    model = rusket.ALS(
        factors=FACTORS,
        regularization=REG,
        alpha=ALPHA,
        iterations=ITERS,
        seed=42,
        verbose=verbose,
        cg_iters=cg_iters,
        use_cholesky=use_cholesky,
    )
    model.fit(mat)
    fit_s = time.perf_counter() - t0

    n_users = mat.shape[0]
    n_rec = min(TOP_N, n_users)
    t0 = time.perf_counter()
    for uid in range(n_rec):
        model.recommend_items(uid, n=10, exclude_seen=True)
    rec_ms = (time.perf_counter() - t0) / n_rec * 1000

    print(f"    fit: {fit_s:.1f}s  |  rec/user: {rec_ms:.2f}ms", flush=True)
    del model
    gc.collect()
    return {"method": label, "fit_s": fit_s, "rec_ms": rec_ms}


def run_method_implicit(mat: sparse.csr_matrix) -> dict | None:
    """Benchmark the `implicit` library ALS (if installed)."""
    try:
        import implicit  # type: ignore[import-untyped]
    except ImportError:
        print(
            "    ⚠️  `implicit` not installed — skipping (pip install implicit)",
            flush=True,
        )
        return None

    # implicit expects item × user CSR
    mat_T = mat.T.tocsr().astype(np.float32)
    t0 = time.perf_counter()
    model = implicit.als.AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REG,
        alpha=ALPHA,
        iterations=ITERS,
        calculate_training_loss=False,
        use_gpu=False,
    )
    model.fit(mat_T, show_progress=True)
    fit_s = time.perf_counter() - t0

    # Latency: item recommendations per user
    n_users = mat.shape[0]
    n_rec = min(TOP_N, n_users)
    t0 = time.perf_counter()
    for uid in range(n_rec):
        model.recommend(
            uid, mat_T.T.tocsr()[uid], N=10, filter_already_liked_items=True
        )
    rec_ms = (time.perf_counter() - t0) / n_rec * 1000

    print(f"    fit: {fit_s:.1f}s  |  rec/user: {rec_ms:.2f}ms", flush=True)
    del model
    gc.collect()
    return {"method": "implicit", "fit_s": fit_s, "rec_ms": rec_ms}


def run_scenario(label: str, mat: sparse.csr_matrix) -> dict:
    """Run all methods against the same matrix and return combined results."""
    n_users, n_items = mat.shape
    nnz = mat.nnz
    csr_mb = (mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes) / 1e6
    fact_mb = (n_users + n_items) * FACTORS * 4 / 1e6
    is_mmap = isinstance(mat.data, np.memmap)

    print(f"\n  ─── {label} ───")
    print(
        f"  {n_users:,}u × {n_items:,}i  |  {nnz:,} nnz  |  "
        f"CSR={csr_mb:.0f} MB  factors={fact_mb:.0f} MB",
        flush=True,
    )

    ok, reason = _check_ram(fact_mb, csr_mb, is_mmap)
    if not ok:
        print(f"  ⚠️  Skipping: {reason}")
        return {
            "label": label,
            "n_users": n_users,
            "n_items": n_items,
            "nnz": nnz,
            "methods": [],
            "csr_mb": csr_mb,
            "fact_mb": fact_mb,
        }

    method_results = []

    print("\n  Method: rusket ALS  cg_iters=3  (fast)", flush=True)
    r = run_method_rusket(mat, "rusket cg=3", cg_iters=3, verbose=False)
    method_results.append(r)

    print("\n  Method: rusket ALS  Cholesky (exact)", flush=True)
    r = run_method_rusket(
        mat, "rusket cholesky", cg_iters=3, use_cholesky=True, verbose=False
    )
    method_results.append(r)

    print("\n  Method: rusket ALS  cg_iters=10 (default)", flush=True)
    r = run_method_rusket(mat, "rusket cg=10", cg_iters=10, verbose=False)
    method_results.append(r)

    print("\n  Method: implicit ALS", flush=True)
    r = run_method_implicit(mat)
    if r:
        method_results.append(r)

    # Summary
    print(f"\n  {'Method':<20} {'Fit (s)':>10} {'Rec (ms)':>10}")
    print(f"  {'-' * 42}")
    for m in method_results:
        print(f"  {m['method']:<20} {m['fit_s']:>10.1f} {m['rec_ms']:>10.2f}")

    return {
        "label": label,
        "n_users": n_users,
        "n_items": n_items,
        "nnz": nnz,
        "methods": method_results,
        "csr_mb": csr_mb,
        "fact_mb": fact_mb,
    }


# ---------------------------------------------------------------------------
# Plotly chart
# ---------------------------------------------------------------------------


def make_chart(results: list[dict], output_dir: Path) -> None:
    valid = [r for r in results if r.get("methods")]
    if not valid:
        return

    # Gather all method names in order
    method_names: list[str] = []
    for r in valid:
        for m in r["methods"]:
            if m["method"] not in method_names:
                method_names.append(m["method"])

    colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4"]
    dataset_labels = [r["label"] for r in valid]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("⚡ Fit Time (lower = faster)", "🔍 Rec Latency per User"),
        horizontal_spacing=0.14,
    )

    for idx, method in enumerate(method_names):
        fit_times = []
        rec_times = []
        for r in valid:
            match = next((m for m in r["methods"] if m["method"] == method), None)
            fit_times.append(match["fit_s"] if match else None)
            rec_times.append(match["rec_ms"] if match else None)

        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Bar(
                name=method,
                x=dataset_labels,
                y=fit_times,
                marker_color=color,
                text=[f"{t:.1f}s" if t else "N/A" for t in fit_times],
                textposition="outside",
                legendgroup=method,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name=method,
                x=dataset_labels,
                y=rec_times,
                marker_color=color,
                text=[f"{t:.2f}ms" if t else "N/A" for t in rec_times],
                textposition="outside",
                legendgroup=method,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(
        y=SPARK_REFERENCE,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Spark MLlib ~{SPARK_REFERENCE}s (4-node, ML-25M)",
        annotation_position="top right",
        row=1,
        col=1,
    )

    fig.update_layout(
        title=dict(text="rusket ALS — Multi-Method Benchmark", font=dict(size=20)),
        template="plotly_dark",
        height=480,
        width=1100,
        barmode="group",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        margin=dict(t=60, b=100),
    )
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    fig.update_yaxes(title_text="ms / user", row=1, col=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "als_benchmark.html"
    json_path = output_dir / "als_benchmark.json"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Chart → {html_path}")
    print(f"✅ JSON  → {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ALS MovieLens multi-method benchmark")
    parser.add_argument(
        "--size",
        choices=["1m", "10m", "25m", "200m", "1b", "all"],
        default="all",
        help="Dataset size (default: all = 1m + 10m + 25m)",
    )
    parser.add_argument("--data-dir", default="data/movielens", help="Data directory")
    args = parser.parse_args()

    sizes = ["1m", "10m", "25m"] if args.size == "all" else [args.size]
    data_dir = Path(args.data_dir)

    print("=" * 65)
    print("  rusket ALS — Multi-Method MovieLens Benchmark")
    print(f"  factors={FACTORS}, iterations={ITERS}, alpha={ALPHA}, reg={REG}")
    print("  Methods: rusket(cg=3), rusket(cg=10), implicit")
    print("=" * 65)

    results: list[dict] = []
    for size in sizes:
        size_label = size.upper()
        print(f"\n{'=' * 65}\n  MovieLens {size_label}")
        try:
            # 200M is a sampled subset of the 1B dataset
            dl_size = "1b" if size == "200m" else size
            folder = download_movielens(dl_size, data_dir)

            if size == "200m":
                mat = load_ratings_sampled_200m(folder)
            else:
                sample = 1.0
                stream = False
                mat = load_ratings(folder, sample=sample, stream=stream)

            result = run_scenario(f"MovieLens {size_label}", mat)
            results.append(result)
            del mat
            gc.collect()
        except Exception as e:
            print(f"  Error: {e}")

    output_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    make_chart(results, output_dir)
    print("\n✅ Done")


if __name__ == "__main__":
    main()
