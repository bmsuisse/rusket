"""Quick focused benchmark: andi_data2, testing pre-filter speedup.

Compares FPGrowth vs Eclat at 100M, 200M, 500M, 1B on the sparse
catalogue dataset (2,603 items, avg 4.4 items/txn).
"""
from __future__ import annotations
import gc, time, urllib.request
from pathlib import Path
import numpy as np
from rusket import FPMiner

DATA_DIR = Path(__file__).resolve().parent / "data"
URL = "https://raw.githubusercontent.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining/master/data/data2.txt"
FILE = DATA_DIR / "andi_data2.txt"
MIN_SUPPORT = 0.02
MAX_LEN = 3

TARGETS = [100_000_000, 200_000_000, 500_000_000, 1_000_000_000]
CHUNK = 500_000


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not FILE.exists():
        print("Downloading andi_data2 …")
        urllib.request.urlretrieve(URL, FILE)


def load():
    txns, items = [], []
    with open(FILE) as f:
        for tid, line in enumerate(f):
            for item in line.split():
                txns.append(tid)
                items.append(int(item))
    txn_ids = np.array(txns, dtype=np.int64)
    item_ids = np.array(items, dtype=np.int32)
    n_items = int(item_ids.max()) + 1
    return txn_ids, item_ids, n_items


def gen_chunk(rng, probs, avg, n_items, n_txns, offset):
    sizes = rng.poisson(avg, size=n_txns).clip(1, None)
    total = int(sizes.sum())
    iids = rng.choice(n_items, size=total, replace=True, p=probs).astype(np.int32)
    tids = np.repeat(np.arange(offset, offset + n_txns, dtype=np.int64), sizes)
    return tids, iids


def run(n_items, probs, avg, target, method):
    n_txns = int(target / avg)
    rng = np.random.default_rng(42)
    miner = FPMiner(n_items=n_items)
    t0 = time.perf_counter()
    off = 0
    while off < n_txns:
        cs = min(CHUNK, n_txns - off)
        tids, iids = gen_chunk(rng, probs, avg, n_items, cs, off)
        miner.add_chunk(tids, iids)
        off += cs
        del tids, iids
    add_t = time.perf_counter() - t0
    actual = miner.n_rows
    t1 = time.perf_counter()
    try:
        freq = miner.mine(min_support=MIN_SUPPORT, max_len=MAX_LEN, method=method)  # type: ignore
        mine_t = time.perf_counter() - t1
        n = len(freq)
    except Exception as e:
        mine_t = -1.0; n = -1
        print(f"  ERROR: {e}")
    del miner; gc.collect()
    return actual, add_t, mine_t, n


def main():
    download()
    txn_ids, item_ids, n_items = load()
    n_txns_real = int(txn_ids.max()) + 1
    avg = len(txn_ids) / n_txns_real
    counts = np.bincount(item_ids, minlength=n_items).astype(np.float64)
    probs = counts / counts.sum()
    del txn_ids, item_ids

    print(f"\nandi_data2: {n_txns_real:,} real txns × {n_items} items, avg {avg:.1f} items/txn")
    print(f"min_support={MIN_SUPPORT}, max_len={MAX_LEN}")
    print()
    hdr = f"  {'rows':>14}  {'method':>10}  {'add_t':>8}  {'mine_t':>8}  {'total':>8}  {'M rows/s':>9}  {'itemsets':>10}"
    sep = "─" * 80
    print(hdr)
    print(sep)

    for target in TARGETS:
        for method in ["eclat", "fpgrowth"]:
            rows, add_t, mine_t, n = run(n_items, probs, avg, target, method)
            total = add_t + mine_t
            tput = rows / total / 1e6 if total > 0 else 0
            print(f"  {rows:>14,}  {method:>10}  {add_t:>7.1f}s  {mine_t:>7.1f}s  "
                  f"{total:>7.1f}s  {tput:>9.2f}  {n:>10,}", flush=True)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
