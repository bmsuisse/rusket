"""
Anderson Acceleration ALS comparison on MovieLens 1M (small dataset).

Compares plain CG-ALS vs ALS+Anderson(m=5) for different iteration counts,
measuring fit time, convergence rate (RMSE proxy), and recommendation quality.
"""

import time
import zipfile
import io
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import rusket

# ─── Data loading ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_PATH = DATA_DIR / "ml-1m"


def load_ml1m() -> sp.csr_matrix:
    DATA_DIR.mkdir(exist_ok=True)
    ratings_csv = ML1M_PATH / "ratings.dat"
    if not ratings_csv.exists():
        print("  Downloading MovieLens 1M...")
        data = urllib.request.urlopen(ML1M_URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(DATA_DIR)
    print("  Loading ratings.dat ...")
    rows, cols, vals = [], [], []
    with open(ratings_csv) as f:
        for line in f:
            u, i, r, _ = line.strip().split("::")
            rows.append(int(u) - 1)
            cols.append(int(i) - 1)
            vals.append(float(r))
    mat = sp.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)),
        shape=(max(rows) + 1, max(cols) + 1)
    )
    print(f"  {mat.shape[0]:,} users × {mat.shape[1]:,} items  {mat.nnz:,} ratings")
    return mat


def bench(mat, label, *, factors=64, iterations=10, cg_iters=3, anderson_m=0):
    model = rusket.ALS(
        factors=factors,
        iterations=iterations,
        cg_iters=cg_iters,
        anderson_m=anderson_m,
        alpha=40.0,
        regularization=0.01,
        verbose=False,
        seed=42,
    )
    t0 = time.perf_counter()
    model.fit(mat)
    elapsed = time.perf_counter() - t0

    # Quick quality proxy: mean score for held-out 5-star ratings
    # (not a real train/test split — just sanity check that quality is maintained)
    uf = model.user_factors
    itf = model.item_factors
    # Score all known interactions
    sample_users = np.random.choice(mat.shape[0], size=100, replace=False)
    scores = []
    for u in sample_users:
        start = mat.indptr[u]
        end = mat.indptr[u + 1]
        if end == start:
            continue
        items = mat.indices[start:end]
        u_vec = uf[u]
        item_vecs = itf[items]
        pred = item_vecs @ u_vec
        scores.append(pred.mean())
    mean_score = float(np.mean(scores)) if scores else 0.0

    print(
        f"  {label:<40}  fit={elapsed:6.1f}s  "
        f"mean_pred={mean_score:6.3f}  "
        f"iters={iterations}"
    )
    return elapsed, mean_score


def main():
    print("=" * 72)
    print("  rusket ALS — Anderson Acceleration Comparison  (ML-1M)")
    print("=" * 72)
    mat = load_ml1m()
    np.random.seed(0)

    print()
    print("── Baseline: plain CG ALS ──────────────────────────────────────────")
    t_baseline, q_baseline = bench(mat, "Plain  CG-3  iters=15", iterations=15, cg_iters=3)
    bench(mat, "Plain  CG-3  iters=10",   iterations=10, cg_iters=3)
    bench(mat, "Plain  CG-3  iters=8",    iterations=8,  cg_iters=3)
    bench(mat, "Plain  CG-3  iters=5",    iterations=5,  cg_iters=3)

    print()
    print("── Anderson(m=5) acceleration ──────────────────────────────────────")
    bench(mat, "AA(m=5) CG-3  iters=15", iterations=15, cg_iters=3, anderson_m=5)
    bench(mat, "AA(m=5) CG-3  iters=10", iterations=10, cg_iters=3, anderson_m=5)
    bench(mat, "AA(m=5) CG-3  iters=8",  iterations=8,  cg_iters=3, anderson_m=5)
    bench(mat, "AA(m=5) CG-3  iters=5",  iterations=5,  cg_iters=3, anderson_m=5)

    print()
    print("── Anderson(m=3) acceleration ──────────────────────────────────────")
    bench(mat, "AA(m=3) CG-3  iters=15", iterations=15, cg_iters=3, anderson_m=3)
    bench(mat, "AA(m=3) CG-3  iters=10", iterations=10, cg_iters=3, anderson_m=3)
    bench(mat, "AA(m=3) CG-3  iters=5",  iterations=5,  cg_iters=3, anderson_m=3)

    print()
    print(f"  Baseline (iters=15, no AA): {t_baseline:.1f}s  mean_pred={q_baseline:.3f}")
    print("  Look for configs that match quality with fewer iterations → faster.")


if __name__ == "__main__":
    main()
