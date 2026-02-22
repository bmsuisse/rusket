"""Verify the fixed notebook data generates non-empty rules and substitutes."""
import numpy as np
import pandas as pd

from rusket import association_rules, mine
from rusket.analytics import find_substitutes


def generate_basket_data(n_transactions: int = 30_000, seed: int = 42) -> pd.DataFrame:
    """
    Segment-based basket generator that produces realistic co-purchase correlations
    AND substitute anti-correlations so that:
    - association_rules returns >0 rows
    - find_substitutes returns >0 rows
    """
    rng = np.random.default_rng(seed)
    cols = [
        # Electronics accessories cluster
        "Wireless_Mouse", "Mechanical_Keyboard", "USB_Hub", "Monitor_Stand",
        "Webcam", "Laptop_Sleeve",
        # Coffee / barista cluster
        "Espresso_Beans", "Milk_Frother", "Coffee_Grinder", "Travel_Mug",
        "Descaler_Tablets",
        # Home office cluster
        "Notebook_A5", "Gel_Pen_Set", "Desk_Lamp", "Sticky_Notes",
        # Competing brands (substitutes)
        "Cola_Brand_A", "Cola_Brand_B",
    ]
    n = n_transactions

    df = pd.DataFrame(False, index=range(n), columns=cols)

    # ── Segment 1: Electronics buyers (45%) ──────────────────────────────────
    seg = rng.random(n) < 0.45
    for p in ["Wireless_Mouse", "Mechanical_Keyboard", "USB_Hub"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.72
    for p in ["Monitor_Stand", "Webcam", "Laptop_Sleeve"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.40

    # ── Segment 2: Coffee buyers (38%) ───────────────────────────────────────
    seg = rng.random(n) < 0.38
    for p in ["Espresso_Beans", "Milk_Frother"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.78
    for p in ["Coffee_Grinder", "Travel_Mug", "Descaler_Tablets"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.45

    # ── Segment 3: Home-office buyers (50%) ──────────────────────────────────
    seg = rng.random(n) < 0.50
    for p in ["Notebook_A5", "Gel_Pen_Set"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.68
    for p in ["Desk_Lamp", "Sticky_Notes"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.42

    # ── Substitutes: Cola A and Cola B (individually popular, rarely together) ─
    # Both popular individually (~35%) but negatively correlated
    a_buyers = rng.random(n) < 0.35
    b_buyers = rng.random(n) < 0.35
    # Remove 85% of co-occurrences → lift ≈ 0.15
    both = a_buyers & b_buyers
    keep_both = rng.random(n) < 0.15  # only 15% of co-occurrences survive
    a_buyers = a_buyers & (~both | keep_both)
    b_buyers = b_buyers & (~both | keep_both)
    df["Cola_Brand_A"] = a_buyers
    df["Cola_Brand_B"] = b_buyers

    return df


df = generate_basket_data()
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} products")
print(f"Density: {df.values.mean():.3f}")
print(f"Avg basket size: {df.sum(axis=1).mean():.1f} items")
print()

# Mine
freq = mine(df, min_support=0.05, use_colnames=True)
print(f"Frequent itemsets (min_support=0.05): {len(freq):,}")
pairs = freq[freq["itemsets"].apply(len) == 2]
print(f"  — pairs: {len(pairs)}")
print(freq.sort_values("support", ascending=False).head(8).to_string(index=False))
print()

# Rules
rules = association_rules(freq, num_itemsets=len(df), min_threshold=0.01)
print(f"Association rules: {len(rules):,}")
if len(rules):
    top = rules[["antecedents", "consequents", "confidence", "lift"]].sort_values("lift", ascending=False).head(5)
    print(top.to_string(index=False))
print()

# Substitutes
subs = find_substitutes(rules, max_lift=0.9)
print(f"Substitutes (lift < 0.9): {len(subs)}")
if len(subs):
    print(subs[["antecedents", "consequents", "lift", "confidence"]].head(5).to_string(index=False))
