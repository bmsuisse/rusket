"""Automated scale benchmark with Plotly chart generation.

Runs from_transactions + fpgrowth at increasing scale and produces an
interactive Plotly HTML chart for the docs.

Usage:
    uv run python benchmarks/bench_scale.py
"""

from __future__ import annotations

import gc
import json
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from rusket import fpgrowth, from_transactions

try:
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# ---------------------------------------------------------------------------
# Timeout helper (POSIX only)
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _alarm(signum: int, frame: object) -> None:
    raise _Timeout()


def timed_run(
    fn: object, *args: object, timeout_sec: int = 600, **kwargs: object
) -> tuple[int | None, float | None]:
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
        return None, None
    finally:
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (label, n_rows, n_txns, n_items)
    ("1M", 1_000_000, 200_000, 10_000),
    ("5M", 5_000_000, 1_000_000, 20_000),
    ("10M", 10_000_000, 2_000_000, 50_000),
    ("50M", 50_000_000, 10_000_000, 100_000),
    ("100M", 100_000_000, 20_000_000, 200_000),
    ("200M", 200_000_000, 40_000_000, 200_000),
    ("500M", 500_000_000, 100_000_000, 500_000),
    ("1B", 1_000_000_000, 200_000_000, 1_000_000),
]


def run_all() -> list[dict]:
    """Run all benchmark scenarios and return results."""
    rng = np.random.default_rng(42)
    results: list[dict] = []

    for label, n_rows, n_txns, n_items in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"  {label}: {n_rows:,} rows → ~{n_txns:,} txns × {n_items:,} items")
        print(f"{'=' * 60}", flush=True)

        # Generate data
        t0 = time.perf_counter()
        txn_ids = rng.integers(0, n_txns, size=n_rows)
        item_ids = rng.integers(0, n_items, size=n_rows)
        df_long = pd.DataFrame({"txn_id": txn_ids, "item": item_ids})
        del txn_ids, item_ids
        gen_time = time.perf_counter() - t0
        print(f"  Gen: {gen_time:.1f}s", flush=True)

        # from_transactions
        t0 = time.perf_counter()
        ohe = from_transactions(df_long)
        conv_time = time.perf_counter() - t0
        del df_long
        gc.collect()
        mem_mb = ohe.memory_usage(deep=True).sum() / 1e6
        actual_txns = ohe.shape[0]
        actual_items = ohe.shape[1]
        print(
            f"  from_transactions: {conv_time:.1f}s → {actual_txns:,} × {actual_items:,}, {mem_mb:.0f} MB",
            flush=True,
        )

        import psutil

        avail_mb = psutil.virtual_memory().available / 1e6
        if avail_mb - mem_mb < 2_000:
            print(
                f"  ⚠️  Skipping: Not enough free RAM for fpgrowth (have {avail_mb:.0f} MB)",
                flush=True,
            )
            t_fpg, n, t_mlx = None, None, None
            results.append(
                {
                    "label": label,
                    "n_rows": n_rows,
                    "n_txns": actual_txns,
                    "n_items": actual_items,
                    "conv_time": conv_time,
                    "fpg_time": t_fpg,
                    "mlx_time": t_mlx,
                    "mem_mb": mem_mb,
                }
            )
            del ohe
            gc.collect()
            continue

        # rusket fpgrowth
        n, t_fpg = timed_run(
            fpgrowth,
            ohe,
            min_support=0.001,
            use_colnames=True,
            max_len=3,
            timeout_sec=600,
        )
        fpg_str = f"{t_fpg:.3f}s" if t_fpg is not None else "TIMEOUT"
        print(f"  fpgrowth: {fpg_str} ({n} itemsets)", flush=True)

        # mlxtend (with timeout) — only run at small/medium scales;
        # at >=5M rows mlxtend typically hangs or OOMs and CI kills the process
        # before our SIGALRM fires, so we skip it there.
        t_mlx = None
        if HAS_MLX and n_rows <= 1_000_000:
            n_mlx, t_mlx = timed_run(
                mlx_fpgrowth,
                ohe,
                min_support=0.001,
                use_colnames=True,
                max_len=3,
                timeout_sec=30,
            )
            mlx_str = f"{t_mlx:.3f}s" if t_mlx is not None else "TIMEOUT(30s)"
            print(f"  mlxtend:  {mlx_str}", flush=True)
        elif HAS_MLX:
            print(f"  mlxtend:  SKIPPED (n_rows > 1M)", flush=True)

        results.append(
            {
                "label": label,
                "n_rows": n_rows,
                "n_txns": actual_txns,
                "n_items": actual_items,
                "conv_time": conv_time,
                "fpg_time": t_fpg,
                "mlx_time": t_mlx,
                "mem_mb": mem_mb,
            }
        )

        del ohe
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Plotly chart
# ---------------------------------------------------------------------------


def make_chart(results: list[dict], output_dir: Path) -> None:
    """Generate interactive Plotly chart and save as HTML."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "⚡ Mining Time vs Scale",
            "📦 Memory Usage vs Scale",
        ),
        horizontal_spacing=0.12,
    )

    labels = [r["label"] for r in results]
    n_rows = [r["n_rows"] for r in results]

    # --- Panel 1: Time ---
    fig.add_trace(
        go.Scatter(
            x=n_rows,
            y=[r["conv_time"] for r in results],
            mode="lines+markers",
            name="from_transactions",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=10),
            hovertemplate="%{text}<br>%{y:.1f}s<extra></extra>",
            text=labels,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=n_rows,
            y=[r["fpg_time"] for r in results],
            mode="lines+markers",
            name="rusket fpgrowth",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=10, symbol="diamond"),
            hovertemplate="%{text}<br>%{y:.1f}s<extra></extra>",
            text=labels,
        ),
        row=1,
        col=1,
    )

    # mlxtend (may have Nones for timeouts)
    mlx_times = [r["mlx_time"] for r in results]
    if any(t is not None for t in mlx_times):
        # Only plot up to the last non-None
        valid_x = [x for x, t in zip(n_rows, mlx_times) if t is not None]
        valid_y = [t for t in mlx_times if t is not None]
        valid_labels = [lbl for lbl, t in zip(labels, mlx_times) if t is not None]
        fig.add_trace(
            go.Scatter(
                x=valid_x,
                y=valid_y,
                mode="lines+markers",
                name="mlxtend (TIMEOUT →)",
                line=dict(color="#ef4444", width=3, dash="dash"),
                marker=dict(size=10, symbol="x"),
                hovertemplate="%{text}<br>%{y:.1f}s<extra></extra>",
                text=valid_labels,
            ),
            row=1,
            col=1,
        )

    # --- Panel 2: Memory ---
    fig.add_trace(
        go.Scatter(
            x=n_rows,
            y=[r["mem_mb"] for r in results],
            mode="lines+markers",
            name="Sparse DataFrame",
            line=dict(color="#f59e0b", width=3),
            marker=dict(size=10, symbol="square"),
            hovertemplate="%{text}<br>%{y:.0f} MB<extra></extra>",
            text=labels,
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        type="log",
        title_text="Input rows (long-format)",
        row=1,
        col=1,
        gridcolor="rgba(128,128,128,0.15)",
    )
    fig.update_xaxes(
        type="log",
        title_text="Input rows (long-format)",
        row=1,
        col=2,
        gridcolor="rgba(128,128,128,0.15)",
    )
    fig.update_yaxes(
        type="log",
        title_text="Time (seconds)",
        row=1,
        col=1,
        range=[-1, 3.5], # 0.1s to 3000s
        gridcolor="rgba(128,128,128,0.15)",
    )
    fig.update_yaxes(
        title_text="Memory (MB)",
        row=1,
        col=2,
        rangemode="tozero",
        gridcolor="rgba(128,128,128,0.15)",
    )

    fig.update_layout(
        title=dict(
            text="rusket — Scale Benchmark (from_transactions → fpgrowth)",
            font=dict(size=20),
        ),
        template="plotly_dark",
        height=500,
        width=1100,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=60, b=100),
    )

    # Save HTML
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "scale_benchmark.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=False,
    )
    print(f"\n✅ Chart saved to {html_path}")

    # Also save JSON results for reference
    json_path = output_dir / "scale_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  rusket — Scale Benchmark")
    print("=" * 60)

    results = run_all()
    output_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    make_chart(results, output_dir)

    print("\n✅ Done! Embed in docs/benchmarks.md with:")
    print(
        '  <iframe src="../assets/scale_benchmark.html" '
        'width="100%" height="550px" style="border:none; border-radius:8px;"></iframe>'
    )


if __name__ == "__main__":
    main()
