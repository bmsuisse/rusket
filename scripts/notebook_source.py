import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

from rusket import association_rules, mine
from rusket.analytics import find_substitutes

# Use a crisp dark theme for all charts
pio.templates.default = "plotly_dark"

# ─── 1. Generating a Realistic Correlated Dataset ────────────────────────────


def generate_basket_data(n_transactions: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """
    Segment-based basket generator.

    Three customer segments create strong positive correlations (high lift).
    Two competing cola brands are negatively correlated (lift < 1, substitutes).

    Parameters
    ----------
    n_transactions : int
        Number of shopping baskets to generate.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = n_transactions

    cols = [
        # Tech accessories cluster
        "Mouse",
        "Keyboard",
        "USB_Hub",
        "Webcam",
        # Barista / coffee cluster
        "Espresso_Beans",
        "Milk_Frother",
        "Travel_Mug",
        # Home office cluster
        "Notebook",
        "Gel_Pen",
        "Highlighter",
        # Competing brands (substitutes — negative correlation)
        "Cola_A",
        "Cola_B",
    ]
    df = pd.DataFrame(False, index=range(n), columns=cols)

    # Tech buyers (40%) buy tech accessories together
    seg = rng.random(n) < 0.40
    for p in ["Mouse", "Keyboard", "USB_Hub", "Webcam"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.75

    # Coffee buyers (35%) often buy barista gear together
    seg = rng.random(n) < 0.35
    for p in ["Espresso_Beans", "Milk_Frother", "Travel_Mug"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.78

    # Home-office buyers (45%)
    seg = rng.random(n) < 0.45
    for p in ["Notebook", "Gel_Pen", "Highlighter"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.72

    # Substitutes: Cola_A is popular (~38%); Cola_B buyers mostly choose A OR B
    # Result: co-occurrence is ~6%, well below independence (0.38×0.21≈0.08) → lift ≈ 0.76
    a_mask = rng.random(n) < 0.38
    b_mask = a_mask & (rng.random(n) < 0.16)  # 16% of A buyers also buy B
    b_only = (~a_mask) & (rng.random(n) < 0.24)  # independent B buyers
    df["Cola_A"] = a_mask
    df["Cola_B"] = b_mask | b_only

    return df


df = generate_basket_data()
print(f"Dataset: {df.shape[0]:,} baskets × {df.shape[1]} products")
print(f"Avg basket size: {df.sum(axis=1).mean():.1f} items")
df.head(5)

# ─── 2. Benchmark: rusket vs mlxtend ─────────────────────────────────────────

min_support = 0.05

t0 = time.time()
rusket_res = mine(df, min_support=min_support, method="fpgrowth", use_colnames=True)
rusket_time = time.time() - t0
print(f"🚀 Rusket FP-Growth: {rusket_time:.4f}s  (Found {len(rusket_res):,} itemsets)")

t0 = time.time()
rusket_eclat_res = mine(df, min_support=min_support, method="eclat", use_colnames=True)
rusket_eclat_time = time.time() - t0
print(f"🚀 Rusket Eclat:     {rusket_eclat_time:.4f}s")

t0 = time.time()
mlxtend_res = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=True)
mlxtend_time = time.time() - t0
print(f"🐢 MLxtend FP-Growth:{mlxtend_time:.4f}s  (Found {len(mlxtend_res):,} itemsets)")
print("-" * 50)
print(f"🏆 Rusket is {mlxtend_time / rusket_time:.1f}× faster than MLxtend!")

# Benchmark chart
fig = px.bar(
    x=["MLxtend (Python)", "Rusket Eclat (Rust)", "Rusket FP-Growth (Rust)"],
    y=[mlxtend_time, rusket_eclat_time, rusket_time],
    title="⏱ Execution Time — Lower is Better",
    labels={"x": "Implementation", "y": "Time (seconds)"},
    color=["baseline", "optimized", "optimized"],
    color_discrete_map={"baseline": "#EF553B", "optimized": "#00CC96"},
    text_auto=".2f",
)
fig.update_traces(textfont_size=14)
fig.update_layout(showlegend=False, title_font_size=20)
fig.show()

# ─── 3. Association Rules ─────────────────────────────────────────────────────

t0 = time.time()
rules = association_rules(rusket_res, num_itemsets=len(df), min_threshold=0.01)
print(f"Generated {len(rules):,} rules in {time.time() - t0:.4f}s")

# Nicer display: keep key columns, round floats
top_rules = (
    rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    .sort_values("lift", ascending=False)
    .head(8)
    .assign(
        support=lambda d: d["support"].round(3),
        confidence=lambda d: d["confidence"].round(3),
        lift=lambda d: d["lift"].round(2),
    )
)
top_rules

# ─── 4. Cannibalization Detection — Substitute Products ──────────────────────

substitutes = find_substitutes(rules, max_lift=0.9)
print(f"Found {len(substitutes)} cannibalizing product pair(s).")

(
    substitutes[["antecedents", "consequents", "support", "confidence", "lift"]].assign(
        support=lambda d: d["support"].round(3),
        confidence=lambda d: d["confidence"].round(3),
        lift=lambda d: d["lift"].round(3),
    )
)

# Confidence vs Lift scatter — the product strategy quadrant
fig = px.scatter(
    rules,
    x="confidence",
    y="lift",
    size="support",
    color="lift",
    hover_data=["antecedents", "consequents"],
    color_continuous_scale="RdYlGn",
    title="📊 Product Strategy: Cross-Sells vs Substitutes",
    labels={"confidence": "Confidence →", "lift": "Lift →"},
)
fig.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="white",
    annotation_text="Lift = 1.0  (independent)",
    annotation_position="top left",
)
fig.add_annotation(
    x=0.85,
    y=max(rules["lift"]) * 0.95,
    text="✅ Cross-sell",
    showarrow=False,
    font=dict(color="#00CC96", size=14),
)
fig.add_annotation(
    x=0.15,
    y=0.5,
    text="⚠️ Substitutes",
    showarrow=False,
    font=dict(color="#EF553B", size=14),
)
fig.update_layout(
    title_font_size=20,
    coloraxis_showscale=True,
)
fig.show()
