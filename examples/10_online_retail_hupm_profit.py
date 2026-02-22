"""Example 10 — High-Utility Pattern Mining for profit/revenue on UCI Online Retail II.

Dataset
-------
Same Online Retail II (UCI) used in example 09. Run that example first to
pre-download the file, or let this script download it automatically.

Concepts demonstrated
---------------------
* Deriving per-row revenue as "utility" (`UnitPrice × Quantity`)
* `HUPM.from_transactions()` + `.mine()` — high-revenue product bundles
* `.association_rules()` on HUPM output
* `rusket.customer_saturation()` — segment customers by breadth of purchases

Business context
----------------
Frequent item counting (FP-Growth) finds commonly bought combinations, but a
bundle of cheap pens can look more "frequent" than a bundle of expensive crystal
glassware.  HUPM finds combinations that **generate the most revenue**, which is
often what the business actually wants to optimise for.

Run
---
    uv run python examples/10_online_retail_hupm_profit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
EXCEL_PATH = DATA_DIR / "online_retail_II.xlsx"


def _ensure_dataset() -> None:
    if not EXCEL_PATH.exists():
        print("Downloading dataset first … (importing example 09)")
        # Reuse download logic from example 09
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        ex09 = import_module("09_online_retail_basket_analysis")
        ex09.download_dataset()


def load_retail_with_revenue() -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_dataset()
    import pandas as pd

    print("Loading Excel …")
    sheets = []
    for sheet in [0, 1]:
        try:
            sheets.append(pd.read_excel(EXCEL_PATH, sheet_name=sheet, engine="openpyxl"))
        except Exception:
            pass
    df = pd.concat(sheets, ignore_index=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Filter: UK, positive qty, real stock codes, no returns, valid customer
    df = df[
        (df["Country"] == "United Kingdom")
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (~df["Invoice"].astype(str).str.startswith("C"))
        & df["Customer_ID"].notna()
        & df["StockCode"].astype(str).str.match(r"^\d{5}")
    ].copy()

    df["Description"] = df["Description"].str.strip()
    # Derive revenue (utility) per line
    df["Revenue"] = (df["Quantity"] * df["Price"]).round(2)
    print(
        f"  Rows: {len(df):,} | Invoices: {df['Invoice'].nunique():,} | "
        f"Total revenue: £{df['Revenue'].sum():,.0f}"
    )
    return df


def main() -> None:
    import rusket
    from rusket.hupm import HUPM

    df = load_retail_with_revenue()

    # -----------------------------------------------------------------------
    # Step 1: Mine high-revenue product bundles
    # -----------------------------------------------------------------------
    # We want bundles that together generate a high total revenue.
    # min_utility = £200 means the bundle's combined revenue across all
    # transactions must exceed £200.

    print("\n── Step 1: High-Utility Pattern Mining (min_utility=£200) ──")
    model = HUPM.from_transactions(
        df,
        transaction_col="Invoice",
        item_col="Description",
        utility_col="Revenue",
        min_utility=200.0,
        max_len=3,
    )
    hupm_results = model.mine()
    print(f"  Found {len(hupm_results):,} high-revenue bundles")

    if hupm_results.empty:
        print("  No bundles found — the dataset may be small. Try min_utility=50.")
        return

    top = hupm_results.sort_values("utility", ascending=False).head(10)
    print("\nTop 10 highest-revenue product bundles:")
    print(top.to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 2: Association rules on high-revenue itemsets
    # -----------------------------------------------------------------------
    print("\n── Step 2: Revenue-aware association rules ──")
    rules = model.association_rules(metric="confidence", min_threshold=0.3)
    if rules.empty:
        print("  No rules found. Try lowering min_utility or min_threshold.")
    else:
        print(f"  Found {len(rules):,} rules")
        top_rules = rules.sort_values("lift", ascending=False).head(10)
        print(top_rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False))

        # Cart recommendation based on revenue-aware rules
        if not rules.empty:
            first_antecedent = list(rules["antecedents"].iloc[0])
            recs = model.recommend_items(first_antecedent, n=3)
            print(f"\n  Cart {first_antecedent} → recommendations: {recs}")

    # -----------------------------------------------------------------------
    # Step 3: Customer saturation analysis
    # -----------------------------------------------------------------------
    print("\n── Step 3: Customer saturation by product category ──")
    # Use stock-code prefix as a proxy for category (first 3 chars)
    df["Category"] = df["StockCode"].astype(str).str[:3]

    saturation = rusket.customer_saturation(
        df,
        user_col="Customer_ID",
        category_col="Category",
    )
    print("Customer saturation deciles (top = highest category breadth):")
    print(saturation.head(10).to_string(index=False))

    # Revenue per decile
    df["Customer_ID"] = df["Customer_ID"].astype(str)
    saturation["Customer_ID"] = saturation["Customer_ID"].astype(str)
    rev_per_customer = df.groupby("Customer_ID")["Revenue"].sum().reset_index()
    merged = saturation.merge(rev_per_customer, on="Customer_ID", how="left")
    rev_by_decile = (
        merged.groupby("decile")["Revenue"]
        .agg(["mean", "sum"])
        .rename(columns={"mean": "avg_revenue_£", "sum": "total_revenue_£"})
        .round(0)
    )
    print("\nRevenue by saturation decile (decile 1 = broadest purchasers):")
    print(rev_by_decile.to_string())

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
