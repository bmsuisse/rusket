"""Generate interactive Plotly benchmark report: rusket vs mlxtend.

Datasets are generated with Faker — realistic product/category names.
Size tiers: tiny / small / medium / large / HUGE (1M rows, ~GB scale).
"""

from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from faker import Faker
from plotly.subplots import make_subplots

from rusket import eclat, fpgrowth

try:
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("mlxtend not installed — skipping head-to-head comparison")

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# ---------------------------------------------------------------------------
# Realistic data generation with Faker
# ---------------------------------------------------------------------------


def _product_names(n: int, seed: int = 0) -> list[str]:
    """Generate n unique fake product/item names using Faker."""
    fake = Faker()
    Faker.seed(seed)
    seen: set[str] = set()
    names: list[str] = []
    generators = [
        # Combine adjective-style words with nouns for plausible product names
        lambda: f"{fake.word().capitalize()} {fake.word()}",
        lambda: fake.bs().split()[0].capitalize() + " " + fake.bs().split()[-1],
        lambda: fake.catch_phrase().split()[0] + " " + fake.catch_phrase().split()[-1],
    ]
    i = 0
    while len(names) < n:
        name = generators[i % len(generators)]()
        if name not in seen:
            seen.add(name)
            names.append(name)
        i += 1
    return names


def _make_transaction_df(
    n_rows: int,
    products: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate a realistic market-basket boolean DataFrame.
    Products follow a power-law popularity distribution (few blockbusters, many long-tail).
    """
    n_cols = len(products)
    # Power-law support: most products appear rarely, a few appear often
    rank = np.arange(1, n_cols + 1, dtype=np.float64)
    support = 0.6 / rank**0.5  # zipf-ish decay
    support = np.clip(support, 0.001, 0.6)

    # Generate boolean matrix vectorised — no Python loops
    matrix = rng.random((n_rows, n_cols)) < support
    return pd.DataFrame(matrix.astype(bool), columns=products)  # type: ignore


def _timed(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


# ---------------------------------------------------------------------------
# Size ladder
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

SIZES = [
    # label,   n_rows,  n_cols, min_sup
    ("tiny", 5, 11, 0.50),
    ("small", 1_000, 50, 0.10),
    ("medium", 10_000, 400, 0.01),
    ("large", 100_000, 1_000, 0.05),
    ("HUGE", 1_000_000, 2_000, 0.10),  # ~2GB raw boolean matrix
]

print("Generating product names…")
max_cols = max(c for _, _, c, _ in SIZES)
ALL_PRODUCTS = _product_names(max_cols)

results = []
for label, n_rows, n_cols, min_sup in SIZES:
    products = ALL_PRODUCTS[:n_cols]
    print(
        f"\n[{label}] {n_rows:>9,} rows × {n_cols:>5,} cols  min_sup={min_sup}",
        flush=True,
    )

    print("  Generating dataset…", end=" ", flush=True)
    t0 = time.perf_counter()
    df = _make_transaction_df(n_rows, products, RNG)
    print(f"{time.perf_counter() - t0:.2f}s  ({df.values.nbytes / 1e6:.0f} MB)")

    # rusket fpgrowth (pandas dense)
    print("  rusket fpgrowth (pandas)…", end=" ", flush=True)
    _, ours_t, ours_mem = _timed(fpgrowth, df, min_support=min_sup)
    n_fi = fpgrowth(df, min_support=min_sup).shape[0]
    print(f"{ours_t:.3f}s  peak={ours_mem / 1e6:.1f}MB  itemsets={n_fi:,}")

    # rusket eclat (pandas dense)
    print("  rusket eclat   (pandas)…", end=" ", flush=True)
    _, tda_t, tda_mem = _timed(eclat, df, min_support=min_sup)
    print(f"{tda_t:.3f}s  peak={tda_mem / 1e6:.1f}MB  ratio={tda_t / ours_t:.2f}×")

    row = {
        "label": label,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "raw_mb": df.values.nbytes / 1e6,
        "fpgrowth_time_s": ours_t,
        "fpgrowth_mem_mb": ours_mem / 1e6,
        "eclat_time_s": tda_t,
        "eclat_mem_mb": tda_mem / 1e6,
        "n_itemsets": n_fi,
    }

    # polars
    if HAS_POLARS:
        df_pl = pl.from_pandas(df)  # type: ignore[possibly-unbound]
        print("  rusket (polars)…", end=" ", flush=True)
        _, pol_t, pol_mem = _timed(fpgrowth, df_pl, min_support=min_sup)
        print(f"{pol_t:.3f}s  peak={pol_mem / 1e6:.1f}MB")
        row.update({"polars_time_s": pol_t, "polars_mem_mb": pol_mem / 1e6})

    # mlxtend — run on all sizes (no skipping; HUGE is slow but doesn't OOM)
    if HAS_MLXTEND:
        print("  mlxtend…", end=" ", flush=True)
        _, mlx_t, mlx_mem = _timed(mlx_fpgrowth, df, min_support=min_sup)  # type: ignore[possibly-unbound]
        mlx_result = mlx_fpgrowth(df, min_support=min_sup, use_colnames=True)  # type: ignore[possibly-unbound]
        speedup = mlx_t / ours_t
        print(f"{mlx_t:.3f}s  peak={mlx_mem / 1e6:.1f}MB  speedup={speedup:.1f}×")

        # ── Output correctness check ─────────────────────────────────────
        our_result = fpgrowth(df, min_support=min_sup, use_colnames=True)
        our_sets = {(round(irow.support, 6), tuple(irow.itemsets)) for irow in our_result.itertuples()}  # type: ignore[union-attr]
        mlx_sets = {(round(irow.support, 6), tuple(irow.itemsets)) for irow in mlx_result.itertuples()}  # type: ignore[union-attr]
        match = our_sets == mlx_sets
        sym_diff = our_sets.symmetric_difference(mlx_sets)
        if match:
            print(f"  ✅ Output matches mlxtend ({len(our_sets)} itemsets identical)")
        else:
            print(f"  ⚠️  Output differs! symmetric_diff={len(sym_diff)} itemsets")
            for x in list(sym_diff)[:3]:
                print(f"     {x}")

        row.update(
            {
                "mlxtend_time_s": mlx_t,
                "mlxtend_mem_mb": mlx_mem / 1e6,
                "speedup": speedup,
                "mem_ratio": mlx_mem / max(ours_mem, 1),
                "output_match": match,
            }
        )

    results.append(row)

df_res = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Plotly dark-theme report
# ---------------------------------------------------------------------------

RUSTKET_COLOR = "#6C63FF"
ECLAT_COLOR = "#FF9F43"
MLXTEND_COLOR = "#FF6584"
POLARS_COLOR = "#43C59E"
HUGE_GLOW = "#FFD166"
BG = "#0A0A14"
PANEL = "#12121F"
GRID = "#2A2A3A"
TEXT = "#E4E4FF"

x_labels = [
    f"<b>{r['label']}</b><br>{r['n_rows']:,} rows<br>{r['n_cols']:,} items<br>{r['raw_mb']:.0f} MB"
    for _, r in df_res.iterrows()
]

# Decide layout
has_compare = HAS_MLXTEND and "speedup" in df_res.columns and len(df_res[df_res["speedup"].notna()]) > 0
n_plot_rows = 2 if has_compare else 1
subplot_titles_all = [
    "⏱ Execution Time (s, log scale)",
    "🧠 Peak Memory (MB, log scale)",
    "⚡ Speedup vs mlxtend (×, higher = better)",
    "💾 Memory ratio vs mlxtend (×, higher = rusket saves more)",
]
subplot_titles = subplot_titles_all[: n_plot_rows * 2]

fig = make_subplots(
    rows=n_plot_rows,
    cols=2,
    subplot_titles=subplot_titles,
    vertical_spacing=0.2,
    horizontal_spacing=0.12,
)


# --- Bar helpers ---
def add_bars(col_idx, y_col, name, color, text_fmt, row=1, showlegend=True):
    mask = df_res[y_col].notna()
    sub = df_res[mask]
    fig.add_trace(
        go.Bar(
            name=name,
            x=[x_labels[i] for i in sub.index],
            y=sub[y_col],
            marker={
                "color": color,
                "line_width": 0,
                "pattern_shape": "" if name != "HUGE" else "/",
            },
            text=[text_fmt.format(v) for v in sub[y_col]],
            textposition="outside",
            textfont={"size": 11},
            showlegend=showlegend,
        ),
        row=row,
        col=col_idx,
    )


# Time (col 1)
add_bars(1, "fpgrowth_time_s", "🦀 fpgrowth (pandas)", RUSTKET_COLOR, "{:.3f}s")
add_bars(1, "eclat_time_s", "🟠 eclat   (pandas)", ECLAT_COLOR, "{:.3f}s")
if HAS_MLXTEND and "mlxtend_time_s" in df_res:
    add_bars(1, "mlxtend_time_s", "🐍 mlxtend", MLXTEND_COLOR, "{:.3f}s")
if HAS_POLARS and "polars_time_s" in df_res:
    add_bars(1, "polars_time_s", "🐻‍❄️ fpgrowth (polars)", POLARS_COLOR, "{:.3f}s")

# Memory (col 2)
add_bars(2, "fpgrowth_mem_mb", "🦀 fpgrowth mem", RUSTKET_COLOR, "{:.1f}MB", showlegend=False)
add_bars(2, "eclat_mem_mb", "🟠 eclat mem", ECLAT_COLOR, "{:.1f}MB", showlegend=False)
if HAS_MLXTEND and "mlxtend_mem_mb" in df_res:
    add_bars(
        2,
        "mlxtend_mem_mb",
        "🐍 mlxtend mem",
        MLXTEND_COLOR,
        "{:.1f}MB",
        showlegend=False,
    )
if HAS_POLARS and "polars_mem_mb" in df_res:
    add_bars(2, "polars_mem_mb", "🐻‍❄️ polars mem", POLARS_COLOR, "{:.1f}MB", showlegend=False)

# Speedup + memory ratio (row 2, only where mlxtend ran)
if has_compare:  # type: ignore[reportGeneralTypeIssues]
    sub = df_res[df_res["speedup"].notna()]
    colors = [f"hsl({min(140, int(s * 6))},75%,55%)" for s in sub["speedup"]]
    fig.add_trace(
        go.Bar(
            name="speedup (fpgrowth)",
            x=[x_labels[i] for i in sub.index],
            y=sub["speedup"],
            marker={"color": colors, "line_width": 0},
            text=[f"<b>{v:.1f}×</b>" for v in sub["speedup"]],
            textposition="outside",
            textfont={"size": 13},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            name="mem_ratio",
            x=[x_labels[i] for i in sub.index],
            y=sub["mem_ratio"],
            marker={"color": POLARS_COLOR, "line_width": 0},
            text=[f"{v:.1f}×" for v in sub["mem_ratio"]],
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_hline(
        y=1,
        row=2,  # type: ignore[arg-type]
        col=1,  # type: ignore[arg-type]
        line_color=MLXTEND_COLOR,
        line_dash="dot",
        annotation_text="baseline",
        annotation_position="bottom right",
        annotation_font_color=MLXTEND_COLOR,
    )
    fig.add_hline(
        y=1,
        row=2,  # type: ignore[arg-type]
        col=2,  # type: ignore[arg-type]
        line_color=MLXTEND_COLOR,
        line_dash="dot",
        annotation_text="baseline",
        annotation_position="bottom right",
        annotation_font_color=MLXTEND_COLOR,
    )

# Annotate HUGE bar specially
huge_rows = df_res[df_res["label"] == "HUGE"]
if not huge_rows.empty:
    h = huge_rows.iloc[0]
    fig.add_annotation(
        x=x_labels[int(huge_rows.index[0])],  # type: ignore
        y=h["fpgrowth_time_s"],
        text=f"🚀 HUGE: {h['n_rows']:,} rows<br>{h['raw_mb']:.0f} MB input",
        showarrow=True,
        arrowhead=2,
        arrowcolor=HUGE_GLOW,
        font={"color": HUGE_GLOW, "size": 12},
        bgcolor=PANEL,
        bordercolor=HUGE_GLOW,
        row=1,
        col=1,
    )

fig.update_layout(
    title={
        "text": "🦀 <b>rusket</b> — FP-Growth vs Eclat Benchmark (Faker synthetic market-basket data)",
        "font": {"size": 20, "color": TEXT, "family": "'Courier New', monospace"},
        "x": 0.5,
    },
    barmode="group",
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font={"color": TEXT, "family": "'Inter', 'Segoe UI', sans-serif", "size": 12},
    legend={"bgcolor": PANEL, "bordercolor": GRID, "borderwidth": 1, "font": {"size": 13}},
    height=800 if has_compare else 480,  # type: ignore[reportGeneralTypeIssues]
    margin={"t": 110, "b": 70, "l": 70, "r": 70},
)

for row in range(1, n_plot_rows + 1):
    for col in range(1, 3):
        fig.update_xaxes(gridcolor=GRID, linecolor=GRID, row=row, col=col)
        fig.update_yaxes(
            gridcolor=GRID,
            linecolor=GRID,
            row=row,
            col=col,
            type="log" if row == 1 else "linear",
        )

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"\n✅ Report → file://{out_path}")
