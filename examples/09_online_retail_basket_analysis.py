"""Example 09 — Market Basket Analysis on the UCI Online Retail II dataset.

Dataset
-------
Online Retail II (UCI Machine Learning Repository)
~1 million invoice lines from a UK online gift retailer (2009-2011).
Auto-downloaded as an Excel workbook (~6 MB) on first run.

Concepts demonstrated
---------------------
* `rusket.from_transactions()` — long-format → one-hot Boolean matrix
* `FPGrowth.from_transactions()` + `.mine()` — frequent itemset mining
* `.association_rules()` — rules from the OO API (cached on repeat calls)
* `.recommend_items()` — live cart recommendations
* `rusket.find_substitutes()` — detect cannibalising / substitute products
* `rusket.viz.to_networkx()` — export rule graph for community detection

Run
---
    uv run python examples/09_online_retail_basket_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Download the dataset (cached to examples/data/)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
EXCEL_PATH = DATA_DIR / "online_retail_II.xlsx"
UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def download_dataset() -> None:
    if EXCEL_PATH.exists():
        print(f"✔ Dataset already at {EXCEL_PATH.name}")
        return
    print("Downloading Online Retail II from UCI …", end=" ", flush=True)
    try:
        import urllib.request
        import zipfile

        zip_path = DATA_DIR / "online_retail_II.zip"
        urllib.request.urlretrieve(UCI_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            # Extract the Excel file — name may vary; pick the first .xlsx
            xlsx_names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
            if not xlsx_names:
                raise FileNotFoundError("No .xlsx found in downloaded zip")
            z.extract(xlsx_names[0], DATA_DIR)
            extracted = DATA_DIR / xlsx_names[0]
            extracted.rename(EXCEL_PATH)
        zip_path.unlink(missing_ok=True)
        print("done.")
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print(
            "Please download manually from:\n"
            "  https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            f"and save the Excel file to: {EXCEL_PATH}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Load and clean
# ---------------------------------------------------------------------------


def load_retail() -> "pd.DataFrame":  # type: ignore[name-defined]
    import pandas as pd

    print("Loading Excel … (this may take 15-30 s on first run)")
    # Sheet 0 = Year 2009-2010; sheet 1 = 2010-2011; we use both
    sheets = []
    for sheet in [0, 1]:
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, engine="openpyxl")
            sheets.append(df)
        except Exception:
            pass
    df = pd.concat(sheets, ignore_index=True)

    # Standard columns: Invoice, StockCode, Description, Quantity, InvoiceDate,
    #                   Price, Customer ID, Country
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    print(f"  Raw rows: {len(df):,}")

    # Keep UK sales with positive quantities; drop returns (Invoice starts with C)
    df = df[
        (df["Country"] == "United Kingdom")
        & (df["Quantity"] > 0)
        & (~df["Invoice"].astype(str).str.startswith("C"))
        & df["Customer_ID"].notna()
        & df["StockCode"].astype(str).str.match(r"^\d{5}")
    ].copy()

    df["Description"] = df["Description"].str.strip()
    print(f"  Clean rows: {len(df):,} | Invoices: {df['Invoice'].nunique():,} | Items: {df['StockCode'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# 3. Main workflow
# ---------------------------------------------------------------------------


def main() -> None:
    download_dataset()
    df = load_retail()

    import pandas as pd
    import rusket
    from rusket.fpgrowth import FPGrowth

    # -----------------------------------------------------------------------
    # 3a. Convert to one-hot encoded basket matrix
    # -----------------------------------------------------------------------
    print("\n── Step 1: One-hot encode transactions ──")
    basket = rusket.from_transactions(
        df,
        transaction_col="Invoice",
        item_col="Description",
    )
    print(f"  Matrix: {basket.shape[0]:,} baskets × {basket.shape[1]:,} items")

    # -----------------------------------------------------------------------
    # 3b. Mine frequent itemsets
    # -----------------------------------------------------------------------
    print("\n── Step 2: Mine frequent itemsets (FP-Growth, min_support=0.02) ──")
    model = FPGrowth.from_transactions(
        df,
        transaction_col="Invoice",
        item_col="Description",
        min_support=0.02,
        use_colnames=True,
        verbose=1,
    )
    freq = model.mine()
    print(f"  Found {len(freq):,} itemsets")
    print(freq.sort_values("support", ascending=False).head(10).to_string(index=False))

    # -----------------------------------------------------------------------
    # 3c. Generate association rules
    # -----------------------------------------------------------------------
    print("\n── Step 3: Association rules (min_confidence=0.5) ──")
    rules = model.association_rules(metric="confidence", min_threshold=0.5)
    print(f"  Found {len(rules):,} rules")
    top_rules = rules.sort_values("lift", ascending=False).head(10)
    print(top_rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # 3d. Cart recommendations
    # -----------------------------------------------------------------------
    print("\n── Step 4: Cart recommendations ──")
    example_cart = ["WHITE HANGING HEART T-LIGHT HOLDER", "RED WOOLLY HOTTIE WHITE HEART."]
    recs = model.recommend_items(example_cart, n=5)
    print(f"  Cart: {example_cart}")
    print(f"  Suggestions: {recs}")

    # -----------------------------------------------------------------------
    # 3e. Substitute / cannibalising products
    # -----------------------------------------------------------------------
    print("\n── Step 5: Cannibalising products (lift < 0.8) ──")
    substitutes = rusket.find_substitutes(rules, max_lift=0.8)
    if substitutes.empty:
        print("  No substitutes found at lift < 0.8 — try lowering max_lift.")
    else:
        print(substitutes[["antecedents", "consequents", "lift", "confidence"]].head(10).to_string(index=False))

    # -----------------------------------------------------------------------
    # 3f. Export rule graph for NetworkX
    # -----------------------------------------------------------------------
    print("\n── Step 6: Export to NetworkX graph ──")
    try:
        import networkx as nx

        G = rusket.viz.to_networkx(rules, edge_attr="lift")
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Optional: community detection
        try:
            communities = list(nx.algorithms.community.greedy_modularity_communities(G.to_undirected()))
            print(f"  Communities detected: {len(communities)}")
            for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:3]):
                print(f"    Cluster {i + 1} ({len(comm)} items): {list(comm)[:3]} …")
        except Exception as e:
            print(f"  (Community detection skipped: {e})")
    except ImportError:
        print("  networkx not installed — skipping graph export. `pip install networkx`")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
