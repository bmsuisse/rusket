"""Benchmark: rusket FPGrowth vs arm-rs on the same dataset.

Generates synthetic transaction datasets with correlated items
(so multi-item frequent patterns and rules actually appear),
runs both implementations, and compares wall-clock times.

Usage:
    uv run python benchmarks/bench_fpgrowth_vs_armrs.py
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rusket import association_rules, fpgrowth


# ---------------------------------------------------------------------------
# Dataset generation — correlated baskets that produce real patterns
# ---------------------------------------------------------------------------

def generate_transactions(
    n_transactions: int = 50_000,
    n_items: int = 50,
    n_patterns: int = 15,
    pattern_size: int = 4,
    pattern_freq: float = 0.15,
    noise_items: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, Path]:
    """Generate transactions with embedded frequent patterns."""
    rng = np.random.default_rng(seed)

    # Create correlated patterns that will appear frequently
    patterns: list[list[int]] = []
    for _ in range(n_patterns):
        pattern = sorted(rng.choice(n_items, size=pattern_size, replace=False).tolist())
        patterns.append(pattern)

    transactions: list[list[int]] = []
    for _ in range(n_transactions):
        basket: set[int] = set()
        # Each pattern has a chance of appearing
        for pattern in patterns:
            if rng.random() < pattern_freq:
                # Add the full pattern or a subset
                subset_size = rng.integers(2, len(pattern) + 1)
                basket.update(rng.choice(pattern, size=subset_size, replace=False).tolist())
        # Add some noise items
        n_noise = rng.integers(0, noise_items + 1)
        basket.update(rng.choice(n_items, size=n_noise, replace=False).tolist())
        if basket:
            transactions.append(sorted(basket))
        else:
            transactions.append([int(rng.integers(0, n_items))])

    # Create one-hot DataFrame for rusket
    cols = {f"i{i}": np.zeros(n_transactions, dtype=bool) for i in range(n_items)}
    for row_idx, basket in enumerate(transactions):
        for item in basket:
            cols[f"i{item}"][row_idx] = True
    df = pd.DataFrame(cols)

    # Write CSV for arm-rs (space-separated items per line)
    csv_path = Path(tempfile.mktemp(suffix=".csv"))
    with open(csv_path, "w") as f:
        for basket in transactions:
            f.write(" ".join(f"i{i}" for i in basket) + "\n")

    return df, csv_path


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_rusket(df: pd.DataFrame, min_support: float) -> dict[str, float]:
    """Benchmark rusket's FPGrowth + association_rules."""
    # Warm up
    _ = fpgrowth(df, min_support=min_support, use_colnames=True)

    # FPGrowth mining
    t0 = time.perf_counter()
    freq = fpgrowth(df, min_support=min_support, use_colnames=True)
    t_mine = time.perf_counter() - t0

    # Association rules
    t0 = time.perf_counter()
    rules = association_rules(freq, metric="confidence", min_threshold=0.1)
    t_rules = time.perf_counter() - t0

    return {
        "mine_ms": t_mine * 1000,
        "rules_ms": t_rules * 1000,
        "total_ms": (t_mine + t_rules) * 1000,
        "n_itemsets": len(freq),
        "n_rules": len(rules),
    }


def bench_armrs(
    csv_path: Path, min_support: float, arm_rs_binary: Path,
) -> dict[str, float] | None:
    """Benchmark arm-rs binary on the same dataset."""
    if not arm_rs_binary.exists():
        print(f"  ⚠ arm-rs binary not found at {arm_rs_binary}, skipping")
        return None

    output_path = Path(tempfile.mktemp(suffix="_rules.csv"))

    cmd = [
        str(arm_rs_binary),
        "--input", str(csv_path),
        "--output", str(output_path),
        "--min-support", str(min_support),
        "--min-confidence", "0.1",
    ]

    # Warm up
    subprocess.run(cmd, capture_output=True, text=True)

    # Timed run
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t_total = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ⚠ arm-rs failed: {result.stderr}")
        return None

    # Parse timing from arm-rs stdout
    lines = result.stdout.strip().split("\n")
    n_itemsets = 0
    n_rules = 0
    mine_ms = 0.0
    rules_ms = 0.0

    for line in lines:
        if "FPGrowth generated" in line:
            # "FPGrowth generated 123 frequent itemsets in 456 ms."
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "frequent":
                    n_itemsets = int(parts[i - 1])
                if p == "ms.":
                    mine_ms = float(parts[i - 1])
        if "Generated" in line and "rules in" in line:
            # "Generated 78 rules in 12 ms, writing to disk."
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "rules" and i > 0:
                    n_rules = int(parts[i - 1])
                if p == "ms," and i > 0:
                    rules_ms = float(parts[i - 1])
        if "Total runtime:" in line:
            pass  # we use our own wall-clock

    # Clean up
    if output_path.exists():
        output_path.unlink()

    return {
        "mine_ms": mine_ms,
        "rules_ms": rules_ms,
        "total_ms": t_total * 1000,
        "n_itemsets": n_itemsets,
        "n_rules": n_rules,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    arm_rs_binary = Path(__file__).parent.parent / ".temp" / "arm-rs" / "target" / "release" / "arm"

    configs = [
        {
            "n_transactions": 20_000, "n_items": 50,
            "min_support": 0.05, "label": "Small (20k×50)",
            "n_patterns": 12, "pattern_size": 4, "pattern_freq": 0.15,
        },
        {
            "n_transactions": 100_000, "n_items": 80,
            "min_support": 0.03, "label": "Medium (100k×80)",
            "n_patterns": 20, "pattern_size": 5, "pattern_freq": 0.12,
        },
        {
            "n_transactions": 500_000, "n_items": 100,
            "min_support": 0.02, "label": "Large (500k×100)",
            "n_patterns": 25, "pattern_size": 5, "pattern_freq": 0.10,
        },
    ]

    results: list[dict[str, object]] = []

    for cfg in configs:
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  n_txn={cfg['n_transactions']}, n_items={cfg['n_items']}, "
              f"min_support={cfg['min_support']}")
        print(f"{'='*60}")

        df, csv_path = generate_transactions(
            n_transactions=cfg["n_transactions"],
            n_items=cfg["n_items"],
            n_patterns=cfg["n_patterns"],
            pattern_size=cfg["pattern_size"],
            pattern_freq=cfg["pattern_freq"],
        )

        # rusket
        print("\n  rusket:")
        r = bench_rusket(df, cfg["min_support"])
        print(f"    Mining:  {r['mine_ms']:>8.1f} ms  ({r['n_itemsets']} itemsets)")
        print(f"    Rules:   {r['rules_ms']:>8.1f} ms  ({r['n_rules']} rules)")
        print(f"    Total:   {r['total_ms']:>8.1f} ms")

        # arm-rs
        print("\n  arm-rs:")
        a = bench_armrs(csv_path, cfg["min_support"], arm_rs_binary)
        if a:
            print(f"    Mining:  {a['mine_ms']:>8.1f} ms  ({a['n_itemsets']} itemsets)")
            print(f"    Rules:   {a['rules_ms']:>8.1f} ms  ({a['n_rules']} rules)")
            print(f"    Total:   {a['total_ms']:>8.1f} ms")

            if a["total_ms"] > 0 and r["total_ms"] > 0:
                speedup = a["total_ms"] / r["total_ms"]
                print(f"\n  ⚡ rusket is {speedup:.1f}x faster (total)")
                results.append({
                    "label": label,
                    "rusket_ms": r["total_ms"], "armrs_ms": a["total_ms"],
                    "rusket_itemsets": r["n_itemsets"], "armrs_itemsets": a["n_itemsets"],
                    "rusket_rules": r["n_rules"], "armrs_rules": a["n_rules"],
                    "speedup": speedup,
                })
        else:
            results.append({
                "label": label, "rusket_ms": r["total_ms"],
                "armrs_ms": None, "speedup": None,
            })

        csv_path.unlink(missing_ok=True)

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<22} {'rusket':>10} {'arm-rs':>10} {'Speedup':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        armrs = f"{r['armrs_ms']:.0f} ms" if r.get("armrs_ms") else "N/A"
        speedup = f"{r['speedup']:.1f}x" if r.get("speedup") else "N/A"
        print(f"  {r['label']:<22} {r['rusket_ms']:>8.0f} ms {armrs:>10} {speedup:>10}")


if __name__ == "__main__":
    main()
