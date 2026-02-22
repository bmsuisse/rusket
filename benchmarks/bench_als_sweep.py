"""25M ALS optimization sweep — runs all solver variants on MovieLens 25M.

Tests:
  CG iters =  1, 3, 5, 10      (standard CG at different depths)
  Cholesky                       (exact direct solve, no iterations)
  factors  = 32, 64, 128        (for CG-3 baseline, to see k scaling)

Usage:
    uv run python benchmarks/bench_als_sweep.py
    uv run python benchmarks/bench_als_sweep.py --iters 5  # fewer ALS iters for faster runs
"""

from __future__ import annotations

import argparse
import gc
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse

import rusket

DATA_DIR = Path("data/movielens")
ML25_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ALPHA = 40.0
REG = 0.01
TOP_N = 200


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


def get_ml25m() -> sparse.csr_matrix:
    dest = DATA_DIR / "ml-25m"
    ratings_csv = dest / "ratings.csv"

    if not ratings_csv.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        archive = DATA_DIR / "ml-25m.zip"
        if not archive.exists():
            print("  Downloading MovieLens 25M ...", flush=True)
            urllib.request.urlretrieve(ML25_URL, archive)
        print("  Extracting ...", flush=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_DIR)

    print("  Loading ratings.csv ...", flush=True)
    df = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "rating"])
    df["user"] = df["userId"].astype("category").cat.codes.astype(np.int32)
    df["item"] = df["movieId"].astype("category").cat.codes.astype(np.int32)
    n_users = int(df["user"].max()) + 1
    n_items = int(df["item"].max()) + 1
    mat = sparse.csr_matrix(
        (
            df["rating"].to_numpy(np.float32),
            (df["user"].to_numpy(), df["item"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
    mat.sum_duplicates()
    print(f"  {n_users:,} users × {n_items:,} items  {mat.nnz:,} ratings", flush=True)
    return mat


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def bench(
    mat: sparse.csr_matrix,
    label: str,
    factors: int,
    iterations: int,
    cg_iters: int,
    use_cholesky: bool,
) -> dict:
    model = rusket.ALS(
        factors=factors,
        regularization=REG,
        alpha=ALPHA,
        iterations=iterations,
        seed=42,
        verbose=False,
        cg_iters=cg_iters,
        use_cholesky=use_cholesky,
    )
    t0 = time.perf_counter()
    model.fit(mat)
    fit_s = time.perf_counter() - t0

    n_users = mat.shape[0]
    n_rec = min(TOP_N, n_users)
    t0 = time.perf_counter()
    for uid in range(n_rec):
        model.recommend_items(uid, n=10, exclude_seen=True)
    rec_ms = (time.perf_counter() - t0) / n_rec * 1000

    del model
    gc.collect()

    throughput = mat.nnz / fit_s / 1e6  # M ratings/sec
    print(
        f"  {label:<32}  fit={fit_s:>6.1f}s  {throughput:.2f}M rat/s  rec={rec_ms:.2f}ms",
        flush=True,
    )
    return {
        "label": label,
        "factors": factors,
        "fit_s": fit_s,
        "throughput": throughput,
        "rec_ms": rec_ms,
        "cg_iters": cg_iters,
        "use_cholesky": use_cholesky,
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def make_chart(results: list[dict], output_dir: Path) -> None:
    labels = [r["label"] for r in results]
    fit_times = [r["fit_s"] for r in results]
    throughput = [r["throughput"] for r in results]
    rec_times = [r["rec_ms"] for r in results]

    # Color by solver family
    def color(r: dict) -> str:
        if r["use_cholesky"]:
            return "#f59e0b"
        if r["cg_iters"] <= 1:
            return "#06b6d4"
        if r["cg_iters"] <= 3:
            return "#22c55e"
        if r["cg_iters"] <= 5:
            return "#6366f1"
        return "#a78bfa"

    colors = [color(r) for r in results]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "⏱ Fit Time (s)",
            "🚀 Throughput (M ratings/s)",
            "🔍 Rec latency (ms)",
        ),
        horizontal_spacing=0.1,
    )
    kw = {"marker_color": colors, "textposition": "outside"}

    fig.add_trace(
        go.Bar(x=labels, y=fit_times, text=[f"{v:.1f}" for v in fit_times], **kw),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=throughput, text=[f"{v:.2f}" for v in throughput], **kw),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=labels, y=rec_times, text=[f"{v:.2f}" for v in rec_times], **kw),
        row=1,
        col=3,
    )

    fig.update_layout(
        title={"text": "rusket ALS — 25M Optimization Sweep", "font": {"size": 20}},
        template="plotly_dark",
        height=500,
        width=1400,
        showlegend=False,
        margin={"t": 70, "b": 120},
    )
    for col in range(1, 4):
        fig.update_xaxes(tickangle=-40, row=1, col=col)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "als_sweep.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    print(f"\n✅ Chart → {html_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=15, help="Number of ALS iterations (default 15)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run CG-1/3/10 + Cholesky at factors=64",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  rusket ALS — 25M Optimization Sweep")
    print(f"  als_iters={args.iters}, alpha={ALPHA}, reg={REG}")
    print("=" * 65)

    mat = get_ml25m()
    results: list[dict] = []

    print("\n── CG iterations sweep (factors=64) ──────────────────────────", flush=True)
    for cg in [1, 3, 5, 10]:
        r = bench(mat, f"CG cg_iters={cg}  k=64", 64, args.iters, cg, False)
        results.append(r)

    print("\n── Cholesky (exact solve, factors=64) ───────────────────────", flush=True)
    r = bench(mat, "Cholesky  k=64", 64, args.iters, 0, True)
    results.append(r)

    if not args.quick:
        print("\n── Factor dimension sweep (CG cg_iters=3) ──────────────────", flush=True)
        for k in [32, 128]:
            r = bench(mat, f"CG cg_iters=3  k={k}", k, args.iters, 3, False)
            results.append(r)

    # Summary table
    print(f"\n  {'Variant':<32}  {'Fit (s)':>7}  {'M rat/s':>8}  {'Rec (ms)':>9}")
    print("  " + "-" * 62)
    best = min(results, key=lambda r: r["fit_s"])
    for r in results:
        marker = " ◀ fastest" if r is best else ""
        print(f"  {r['label']:<32}  {r['fit_s']:>7.1f}  {r['throughput']:>8.2f}  {r['rec_ms']:>9.2f}{marker}")

    output_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    make_chart(results, output_dir)
    print("\n✅ Done")


if __name__ == "__main__":
    main()
