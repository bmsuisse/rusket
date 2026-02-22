"""Example 11 — Grocery Recommender on the Instacart Market Basket 2017 dataset.

Dataset
-------
Instacart Market Basket Analysis (Kaggle, 2017)
3 million grocery orders from 200k+ Instacart users.

Download instructions (requires free Kaggle account + kaggle API token):
    pip install kaggle
    kaggle competitions download -c instacart-market-basket-analysis
    unzip instacart-market-basket-analysis.zip -d examples/data/instacart

If data is not available a small synthetic fallback is used automatically,
so the script always demonstrates the full API surface.

Concepts demonstrated
---------------------
* `ALS.from_transactions()` — collaborative filter on real grocery orders
* `ALS.recommend_items()` — top-N for a user
* `rusket.similar_items()` — grocery substitutes via ALS cosine similarity
* `rusket.export_item_factors()` — export vectors for FAISS / Pinecone / Qdrant
* `BPR.from_transactions()` — compare ranking model vs ALS
* `rusket.score_potential()` — cross-sell potential across product departments
* `PrefixSpan.from_transactions()` — sequential basket patterns across orders

Run
---
    # With real Instacart data:
    uv run python examples/11_instacart_recommender.py

    # With built-in synthetic data fallback:
    uv run python examples/11_instacart_recommender.py --synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "instacart"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _make_synthetic() -> tuple["pd.DataFrame", "pd.DataFrame"]:  # type: ignore[name-defined]
    """Return (orders_df, products_df) with synthetic grocery-like data."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_users, n_items, n_orders = 500, 120, 4_000

    categories = ["Produce", "Dairy", "Bakery", "Frozen", "Snacks", "Beverages", "Meat", "Seafood"]
    products = pd.DataFrame({
        "product_id": range(n_items),
        "product_name": [f"Product_{i:03d}" for i in range(n_items)],
        "department": [categories[i % len(categories)] for i in range(n_items)],
    })

    rows = []
    for order_id in range(n_orders):
        user_id = rng.integers(0, n_users)
        # Users have strong preference for 1-2 categories (simulate real shopping)
        fav_cat = user_id % len(categories)
        # 60% items from fav category, 40% random
        n_items_in_order = rng.integers(2, 12)
        fav_items = products[products["department"] == categories[fav_cat]]["product_id"].values
        other_items = products["product_id"].values

        chosen = []
        for _ in range(n_items_in_order):
            if rng.random() < 0.6 and len(fav_items) > 0:
                chosen.append(rng.choice(fav_items))
            else:
                chosen.append(rng.choice(other_items))

        for item in set(chosen):
            rows.append({"user_id": user_id, "order_id": order_id, "product_id": int(item)})

    orders = pd.DataFrame(rows)
    print(f"  [synthetic] {n_orders:,} orders · {n_users:,} users · {n_items:,} products")
    return orders, products


def _load_instacart() -> tuple["pd.DataFrame", "pd.DataFrame"] | None:
    """Try to load real Instacart data. Returns None if unavailable."""
    import pandas as pd

    prior_path = DATA_DIR / "order_products__prior.csv"
    orders_path = DATA_DIR / "orders.csv"
    products_path = DATA_DIR / "products.csv"
    depts_path = DATA_DIR / "departments.csv"

    if not prior_path.exists() or not orders_path.exists():
        return None

    print("Loading Instacart data …")
    prior = pd.read_csv(prior_path, usecols=["order_id", "product_id"])
    orders = pd.read_csv(orders_path, usecols=["order_id", "user_id"])
    products = pd.read_csv(products_path)
    depts = pd.read_csv(depts_path) if depts_path.exists() else None

    df = prior.merge(orders, on="order_id").merge(products, on="product_id")
    if depts is not None:
        df = df.merge(depts, on="department_id", how="left")
        df.rename(columns={"department": "department_name"}, inplace=True)
    else:
        df["department_name"] = "unknown"

    print(
        f"  {len(df):,} order-product records | "
        f"{df['user_id'].nunique():,} users | "
        f"{df['product_id'].nunique():,} products"
    )
    print("  (Tip: using the prior split — largest subset for ML)")
    return df[["user_id", "order_id", "product_id"]], products


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    use_synthetic = "--synthetic" in sys.argv

    import pandas as pd
    import numpy as np
    import rusket
    from rusket.als import ALS
    from rusket.bpr import BPR
    from rusket.prefixspan import PrefixSpan

    if use_synthetic:
        print("Using synthetic data …")
        orders, products = _make_synthetic()
    else:
        result = _load_instacart()
        if result is None:
            print(
                "Instacart data not found.\n"
                "To download:\n"
                "  pip install kaggle\n"
                "  kaggle competitions download -c instacart-market-basket-analysis\n"
                f"  unzip instacart-market-basket-analysis.zip -d {DATA_DIR}\n\n"
                "Running with synthetic fallback …"
            )
            orders, products = _make_synthetic()
        else:
            orders, products = result

    # Standardise column names
    orders = orders.rename(columns={"user_id": "user_id", "product_id": "product_id"})

    # -----------------------------------------------------------------------
    # Step 1: Train ALS collaborative filter
    # -----------------------------------------------------------------------
    print("\n── Step 1: Train ALS recommender ──")
    als = ALS.from_transactions(
        orders,
        user_col="user_id",
        item_col="product_id",
        factors=64,
        iterations=15,
        alpha=40.0,
        verbose=True,
    )

    # Recommend for user 0
    user_id = 0
    item_ids, scores = als.recommend_items(user_id=user_id, n=5)
    product_names = products.set_index("product_id")["product_name"]
    print(f"\nTop-5 recommendations for user {user_id}:")
    for iid, sc in zip(item_ids, scores):
        name = product_names.get(iid, f"product_{iid}")
        print(f"  {name!s:<40} score={sc:.3f}")

    # -----------------------------------------------------------------------
    # Step 2: Similar items (cosine similarity on ALS latent factors)
    # -----------------------------------------------------------------------
    print("\n── Step 2: Similar items via ALS cosine similarity ──")
    target_item = int(item_ids[0])
    sim_ids, sim_scores = rusket.similar_items(als, item_id=target_item, n=5)
    target_name = product_names.get(target_item, f"product_{target_item}")
    print(f"Products similar to '{target_name}':")
    for iid, sc in zip(sim_ids, sim_scores):
        name = product_names.get(iid, f"product_{iid}")
        print(f"  {name!s:<40} cosine={sc:.3f}")

    # -----------------------------------------------------------------------
    # Step 3: Export item factors for vector DB ingestion
    # -----------------------------------------------------------------------
    print("\n── Step 3: Export ALS item factors for vector DB ──")
    factors_df = rusket.export_item_factors(als, include_labels=True)
    print(f"  Exported {len(factors_df):,} item vectors, shape={np.stack(factors_df['vector'].values).shape}")
    print("  (Ready to ingest into FAISS / Pinecone / Qdrant)")
    print(factors_df.head(3).to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 4: BPR (ranking model) — compare to ALS
    # -----------------------------------------------------------------------
    print("\n── Step 4: BPR ranking model (compare to ALS) ──")
    bpr = BPR.from_transactions(
        orders,
        user_col="user_id",
        item_col="product_id",
        factors=64,
        iterations=50,
        verbose=True,
    )
    bpr_ids, bpr_scores = bpr.recommend_items(user_id=user_id, n=5)
    print(f"\nTop-5 BPR recommendations for user {user_id}:")
    for iid, sc in zip(bpr_ids, bpr_scores):
        name = product_names.get(iid, f"product_{iid}")
        print(f"  {name!s:<40} score={sc:.3f}")

    # -----------------------------------------------------------------------
    # Step 5: Cross-sell potential across departments
    # -----------------------------------------------------------------------
    print("\n── Step 5: Cross-sell potential by department ──")
    if "department" in products.columns or "department_name" in orders.columns:
        dept_col = "department_name" if "department_name" in orders.columns else "department"
        if dept_col in products.columns:
            dept_items = (
                products.groupby(dept_col)["product_id"]
                .apply(list)
                .to_dict()
            )
            # Pick one department as target category
            target_dept = list(dept_items.keys())[0]
            target_item_ids = [i for i in dept_items[target_dept] if i < als.item_factors.shape[0]]
            if target_item_ids:
                # Build user histories
                user_histories = (
                    orders.groupby("user_id")["product_id"]
                    .apply(list)
                    .tolist()
                )
                potential = rusket.score_potential(
                    user_histories[:50],  # first 50 users for demo
                    als,
                    target_categories=target_item_ids[:20],
                )
                top_user = int(np.argmax(potential.max(axis=1)))
                print(f"  Department: '{target_dept}' — {len(target_item_ids)} products")
                print(f"  User with highest cross-sell potential: user {top_user}")
                print(f"  Potential score matrix shape: {potential.shape}")
    else:
        print("  (department column not available in this data — skipping)")

    # -----------------------------------------------------------------------
    # Step 6: Sequential basket patterns (PrefixSpan on user order history)
    # -----------------------------------------------------------------------
    print("\n── Step 6: Sequential basket patterns with PrefixSpan ──")
    # Treat each user's ordered sequence of products as a session.
    # We create a synthetic event timestamp using cumulative order count per user
    # (for Instacart real data, use order_number; for synthetic we use order_id).
    top_users = orders["user_id"].value_counts().head(200).index
    sample = orders[orders["user_id"].isin(top_users)].copy()
    # Add a synthetic ordering column so PrefixSpan can sort events
    sample["event_order"] = sample.groupby("user_id").cumcount()

    model_ps = PrefixSpan.from_transactions(
        sample,
        user_col="user_id",
        time_col="event_order",
        item_col="product_id",
        min_support=5,
        max_len=3,
    )
    seq_df = model_ps.mine()
    print(f"  Found {len(seq_df):,} frequent sequences (min_support=5)")
    if not seq_df.empty:
        top_seqs = seq_df.sort_values("support", ascending=False).head(5)
        for _, row in top_seqs.iterrows():
            seq_names = [str(product_names.get(int(i), i)) for i in row["sequence"]]
            print(f"    support={row['support']:3d}  →  {' → '.join(seq_names)}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
