"""Generate a Plotly performance chart from FPMiner benchmark results.

Run the benchmark first:
    uv run python benchmarks/bench_fpminer_realistic.py > /tmp/bench_out.txt
Then generate the chart:
    uv run python benchmarks/plot_fpminer_scale.py

Or pass results directly via the RESULTS dict at the bottom of this file.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Results from the realistic retail benchmark (2603 items, avg basket=4.4) ---
# Format: (rows, add_t, mine_t, itemsets)  — update after each run
RESULTS = [
    (50_000_000, 4.8, 5.6, 1260),
    (100_000_000, 10.6, 13.9, 1254),
    (200_000_000, 22.7, 33.2, 1261),
    (300_000_000, 33.5, 54.6, 1260),
    # 500M, 800M, 1B will be filled in after benchmark completes
]

LABEL = "Real-world retail data (2,603 items, avg basket=4.4, min_support=0.1%)"


def make_chart(results: list[tuple]) -> go.Figure:
    rows_m = [r[0] / 1_000_000 for r in results]
    add_t = [r[1] for r in results]
    mine_t = [r[2] for r in results]
    total_t = [r[1] + r[2] for r in results]
    itemsets = [r[3] for r in results]

    colors = {
        "add": "#4361EE",
        "mine": "#F72585",
        "total": "#7209B7",
        "items": "#3A0CA3",
    }

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Wall-clock time vs rows processed",
            "Itemsets found vs rows processed",
        ),
        vertical_spacing=0.14,
        row_heights=[0.62, 0.38],
    )

    # -- Time chart --
    fig.add_trace(
        go.Bar(
            name="add_chunk()",
            x=rows_m,
            y=add_t,
            marker_color=colors["add"],
            opacity=0.85,
            text=[f"{v:.1f}s" for v in add_t],
            textposition="inside",
            width=0.35,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            name="mine()",
            x=rows_m,
            y=mine_t,
            marker_color=colors["mine"],
            opacity=0.85,
            text=[f"{v:.1f}s" for v in mine_t],
            textposition="inside",
            width=0.35,
            base=add_t,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            name="Total",
            x=rows_m,
            y=total_t,
            mode="lines+markers+text",
            line={"color": colors["total"], "width": 2.5, "dash": "dot"},
            marker={"size": 8},
            text=[f"{v:.1f}s" for v in total_t],
            textposition="top center",
        ),
        row=1,
        col=1,
    )

    # -- Itemsets chart --
    fig.add_trace(
        go.Scatter(
            name="Itemsets found",
            x=rows_m,
            y=itemsets,
            mode="lines+markers",
            line={"color": colors["items"], "width": 2.5},
            marker={"size": 8},
            fill="tozeroy",
            fillcolor="rgba(58,12,163,0.12)",
        ),
        row=2,
        col=1,
    )

    rows_label = [f"{int(r)}M" for r in rows_m]
    fig.update_xaxes(
        tickvals=rows_m, ticktext=rows_label, title_text="Rows processed", row=1, col=1
    )
    fig.update_xaxes(
        tickvals=rows_m, ticktext=rows_label, title_text="Rows processed", row=2, col=1
    )
    fig.update_yaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Frequent itemsets", row=2, col=1)

    fig.update_layout(
        title={
            "text": f"<b>rusket FPMiner — Billion-Row Streaming Benchmark</b><br>"
            f"<sup>{LABEL}</sup>",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        barmode="stack",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        font={"family": "Inter, system-ui, sans-serif", "size": 13},
        height=700,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
    )
    return fig


if __name__ == "__main__":
    fig = make_chart(RESULTS)
    out = "docs/assets/fpminer_benchmark.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Chart saved to {out}")

    # Also save as static PNG for README embedding
    try:
        img_out = "docs/assets/fpminer_benchmark.png"
        fig.write_image(img_out, width=1100, height=700, scale=2)
        print(f"PNG saved to {img_out}")
    except Exception as e:
        print(f"PNG skipped (install kaleido): {e}")
