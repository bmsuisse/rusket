"""Benchmark rusket vs mlxtend on real-world datasets.

Downloads datasets automatically on first run.
Usage:
    uv run python benchmarks/bench_realworld.py
"""

from __future__ import annotations

import signal
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from rusket import eclat, fpgrowth

try:
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  mlxtend not installed — skipping mlxtend comparisons")


# ---------------------------------------------------------------------------
# Timeout helper (POSIX only)
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _alarm(signum: int, frame: object) -> None:
    raise _Timeout()


def timed_run(
    fn: object,
    *args: object,
    timeout_sec: int = 120,
    **kwargs: object,
) -> tuple[int | None, str | float]:
    """Run *fn* with a wall-clock timeout.  Returns ``(n_itemsets, elapsed)``."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_sec)
    try:
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        signal.alarm(0)
        return len(res), elapsed
    except _Timeout:
        signal.alarm(0)
        return None, f"TIMEOUT({timeout_sec}s)"
    except Exception as e:
        signal.alarm(0)
        return None, f"ERROR: {e}"
    finally:
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS: dict[str, dict] = {
    "andi_data": {
        "url": "https://raw.githubusercontent.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining/master/data/data.txt",
        "file": "andi_data.txt",
        "min_support": 0.02,
        "max_len": None,
        "timeout": 120,
    },
    "andi_data2": {
        "url": "https://raw.githubusercontent.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining/master/data/data2.txt",
        "file": "andi_data2.txt",
        "min_support": 0.005,
        "max_len": 3,
        "timeout": 120,
    },
}


def download_if_missing(name: str) -> Path:
    """Download a dataset if it's not already cached."""
    info = DATASETS[name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / info["file"]
    if not path.exists():
        print(f"  ⬇  Downloading {name} …", flush=True)
        urllib.request.urlretrieve(info["url"], path)
    return path


def load_basket(path: Path) -> pd.DataFrame:
    """Load a space-separated transaction file into a bool DataFrame."""
    transactions: list[list[str]] = []
    all_items: set[str] = set()
    with open(path) as f:
        for line in f:
            items = line.strip().split()
            if items:
                transactions.append(items)
                all_items.update(items)
    items_sorted = sorted(all_items)
    item_to_idx = {item: i for i, item in enumerate(items_sorted)}
    n_txn = len(transactions)
    n_items = len(items_sorted)
    data = np.zeros((n_txn, n_items), dtype=bool)
    for i, txn in enumerate(transactions):
        for item in txn:
            data[i, item_to_idx[item]] = True
    return pd.DataFrame(data, columns=items_sorted)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(name: str) -> None:
    """Download (if needed), load, and benchmark a single dataset."""
    info = DATASETS[name]
    path = download_if_missing(name)
    df = load_basket(path)

    density = df.values.mean()
    avg_basket = density * df.shape[1]

    print(
        f"\n{'=' * 72}\n"
        f"  {name}\n"
        f"  {df.shape[0]:,} transactions × {df.shape[1]:,} items\n"
        f"  density={density:.4f}  avg items/basket={avg_basket:.1f}\n"
        f"  min_support={info['min_support']}  max_len={info['max_len']}\n"
        f"{'=' * 72}"
    )

    timeout = info["timeout"]
    min_sup = info["min_support"]
    max_len = info["max_len"]

    # rusket fpgrowth
    n, t = timed_run(
        fpgrowth,
        df,
        min_support=min_sup,
        use_colnames=True,
        max_len=max_len,
        timeout_sec=timeout,
    )
    t_str = t if isinstance(t, str) else f"{t:.3f}s"
    print(
        f"  rusket fpgrowth  : {t_str:>14}  ({n:>12,} itemsets)"
        if n
        else f"  rusket fpgrowth  : {t_str}"
    )

    # rusket eclat
    n2, t2 = timed_run(
        eclat,
        df,
        min_support=min_sup,
        use_colnames=True,
        max_len=max_len,
        timeout_sec=timeout,
    )
    t2_str = t2 if isinstance(t2, str) else f"{t2:.3f}s"
    print(
        f"  rusket eclat     : {t2_str:>14}  ({n2:>12,} itemsets)"
        if n2
        else f"  rusket eclat     : {t2_str}"
    )

    # mlxtend
    if HAS_MLX:
        n3, t3 = timed_run(
            mlx_fpgrowth,
            df,
            min_support=min_sup,
            use_colnames=True,
            max_len=max_len if max_len else 999,
            timeout_sec=timeout,
        )
        t3_str = t3 if isinstance(t3, str) else f"{t3:.3f}s"
        if n3 is not None:
            print(f"  mlxtend fpgrowth : {t3_str:>14}  ({n3:>12,} itemsets)")
        else:
            print(f"  mlxtend fpgrowth : {t3_str}")

        # speedup
        if isinstance(t, (int, float)) and isinstance(t3, (int, float)):
            print(f"\n  → rusket is {t3 / t:.1f}× faster than mlxtend")
        elif isinstance(t3, str) and "TIMEOUT" in t3 and isinstance(t, (int, float)):
            print(f"\n  → mlxtend timed out, rusket finished in {t:.1f}s 🚀")


def main() -> None:
    print("=" * 72)
    print("  rusket vs mlxtend — Real-World Dataset Benchmark")
    print("=" * 72)

    for name in DATASETS:
        run_benchmark(name)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
