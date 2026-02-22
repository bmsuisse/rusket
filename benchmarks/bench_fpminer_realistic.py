"""FPMiner benchmark: iterate fast on 100M rows, validate best at 200M.

Compares:
  • Methods:      FP-Growth vs Eclat
  • Chunk sizes:  100k / 500k / 2M  (ingestion sensitivity)
  • Scale:        100M validation → 200M validation

Usage:
    uv run python benchmarks/bench_fpminer_realistic.py
"""

from __future__ import annotations

import gc
import time
import urllib.request
from pathlib import Path

import numpy as np

from rusket import FPMiner

DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS: dict[str, dict] = {
    "andi_data": {
        "url": "https://raw.githubusercontent.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining/master/data/data.txt",
        "file": "andi_data.txt",
        "min_support": 0.02,
        "max_len": 3,
    },
    "andi_data2": {
        "url": "https://raw.githubusercontent.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining/master/data/data2.txt",
        "file": "andi_data2.txt",
        "min_support": 0.02,
        "max_len": 3,
    },
}


def download_if_missing(name: str) -> Path:
    info = DATASETS[name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / info["file"]
    if not path.exists():
        print(f"  ⬇  Downloading {name} …")
        urllib.request.urlretrieve(info["url"], path)
    return path


def load_transactions(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    txn_ids_list, item_ids_list = [], []
    with open(path) as f:
        for txn_id, line in enumerate(f):
            items = [int(x) for x in line.split() if x.strip()]
            for item in items:
                txn_ids_list.append(txn_id)
                item_ids_list.append(item)
    txn_ids = np.array(txn_ids_list, dtype=np.int64)
    item_ids = np.array(item_ids_list, dtype=np.int32)
    n_items = int(item_ids.max()) + 1
    return txn_ids, item_ids, n_items


def compute_item_probs(
    txn_ids: np.ndarray, item_ids: np.ndarray, n_items: int, n_txns: int
) -> np.ndarray:
    counts = np.bincount(item_ids, minlength=n_items).astype(np.float64)
    return counts / counts.sum()


def generate_chunk(
    rng: np.random.Generator,
    item_probs: np.ndarray,
    avg_items: float,
    n_items: int,
    n_txns: int,
    txn_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = rng.poisson(avg_items, size=n_txns).clip(1, None)
    total = int(sizes.sum())
    item_ids = rng.choice(n_items, size=total, replace=True, p=item_probs).astype(
        np.int32
    )
    txn_ids = np.repeat(
        np.arange(txn_offset, txn_offset + n_txns, dtype=np.int64), sizes
    )
    return txn_ids, item_ids


def run_one(
    *,
    info: dict,
    item_probs: np.ndarray,
    avg_items: float,
    n_items: int,
    target_rows: int,
    chunk_txns: int,
    method: str,
    seed: int = 42,
) -> dict:
    target_txns = int(target_rows / avg_items)
    rng = np.random.default_rng(seed)
    miner = FPMiner(n_items=n_items)

    t0 = time.perf_counter()
    txn_offset = 0
    while txn_offset < target_txns:
        cs = min(chunk_txns, target_txns - txn_offset)
        txn_ids, item_ids = generate_chunk(
            rng, item_probs, avg_items, n_items, cs, txn_offset
        )
        miner.add_chunk(txn_ids, item_ids)
        txn_offset += cs
        del txn_ids, item_ids
    add_t = time.perf_counter() - t0
    actual_rows = miner.n_rows

    t1 = time.perf_counter()
    try:
        freq = miner.mine(
            min_support=info["min_support"], max_len=info["max_len"], method=method
        )  # type: ignore[call-arg]
        mine_t = time.perf_counter() - t1
        n_itemsets = len(freq)
    except Exception as e:
        mine_t = -1.0
        n_itemsets = -1
        print(f"    ⚠  {e}")

    del miner
    gc.collect()

    total = add_t + mine_t
    return {
        "rows": actual_rows,
        "add_t": add_t,
        "mine_t": mine_t,
        "total": total,
        "Mrows_s": actual_rows / total / 1e6 if total > 0 else 0,
        "n_itemsets": n_itemsets,
    }


ROW = "  {rows:>14,}  {method:>10}  {chunk:>6}  {add_t:>7.1f}s  {mine_t:>7.1f}s  {total:>7.1f}s  {Mrows_s:>9.2f}  {n_itemsets:>10,}"
HDR = (
    "  "
    + f"{'rows':>14}  {'method':>10}  {'chunk':>6}  {'add_t':>8}  {'mine_t':>8}  {'total':>8}  {'M rows/s':>9}  {'itemsets':>10}"
)

SEP = "─" * 88


SCALE_TARGETS = [100_000_000, 200_000_000, 500_000_000, 1_000_000_000]


def run_dataset(name: str) -> None:
    info = DATASETS[name]
    path = download_if_missing(name)

    txn_ids_real, item_ids_real, n_items = load_transactions(path)
    n_txns_real = int(txn_ids_real.max()) + 1
    avg_items = len(txn_ids_real) / n_txns_real
    item_probs = compute_item_probs(txn_ids_real, item_ids_real, n_items, n_txns_real)
    del txn_ids_real, item_ids_real

    print(f"\n{'═' * 88}")
    print(
        f"  Dataset: {name}  │  {n_txns_real:,} real txns × {n_items} items, avg {avg_items:.1f} items/txn"
    )
    print(f"  min_support={info['min_support']}, max_len={info['max_len']}")

    # ── Fast iteration on 100M ─────────────────────────────────────────────
    print("\n  ▶ Iteration phase — 100M rows")
    print("  " + SEP)
    print(HDR)
    print("  " + SEP)

    results_100m: list[dict] = []
    for method in ["fpgrowth", "eclat"]:
        for cname, csz in [("100k", 100_000), ("500k", 500_000), ("2M", 2_000_000)]:
            r = run_one(
                info=info,
                item_probs=item_probs,
                avg_items=avg_items,
                n_items=n_items,
                target_rows=100_000_000,
                chunk_txns=csz,
                method=method,
            )
            r["method"] = method
            r["chunk"] = cname
            results_100m.append(r)
            print(ROW.format(**r), flush=True)

    # ── Find best config (fastest total) ───────────────────────────────────
    best = min(
        results_100m, key=lambda r: r["total"] if r["n_itemsets"] > 0 else float("inf")
    )
    print(
        f"\n  🏆 Best @ 100M: method={best['method']}, chunk={best['chunk']} → {best['total']:.1f}s"
    )

    # ── Validate best + all methods at 200M ────────────────────────────────
    print("\n  ▶ Validation phase — 200M rows")
    print("  " + SEP)
    print(HDR)
    print("  " + SEP)

    for method in ["eclat", "fpgrowth"]:
        r = run_one(
            info=info,
            item_probs=item_probs,
            avg_items=avg_items,
            n_items=n_items,
            target_rows=200_000_000,
            chunk_txns=500_000,
            method=method,
        )
        r["method"] = method
        r["chunk"] = "500k"
        print(ROW.format(**r), flush=True)

    # ── Scale-up to 1B using Eclat (the winner) ───────────────────────────
    print("\n  ▶ Scale-up to 1B — Eclat (winner from iteration)")
    print("  " + SEP)
    print(HDR)
    print("  " + SEP)

    for target in SCALE_TARGETS:
        r = run_one(
            info=info,
            item_probs=item_probs,
            avg_items=avg_items,
            n_items=n_items,
            target_rows=target,
            chunk_txns=500_000,
            method="eclat",
        )
        r["method"] = "eclat"
        r["chunk"] = "500k"
        print(ROW.format(**r), flush=True)


def main() -> None:
    print("=" * 88)
    print("  FPMiner — Fast Iteration Benchmark")
    print("  Strategy: HashMap<txn_id, Vec<item>> — O(unique_txns × avg_items) memory")
    print("=" * 88)
    for name in DATASETS:
        run_dataset(name)
    print(f"\n{'═' * 88}")
    print("  ✅ Done!")
    print(f"{'═' * 88}")


if __name__ == "__main__":
    main()
