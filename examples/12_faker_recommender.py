"""
rusket — Faker-Driven Collaborative Filtering Recommender
==========================================================

Generates a synthetic e-commerce customer base with Faker
(realistic user names, product SKUs, and purchase timestamps),
then trains an ALS recommender, evaluates it, and shows
personalised recommendations with real-looking labels.
"""

import numpy as np
import pandas as pd
from faker import Faker

from rusket import ALS, FPGrowth, Recommender, evaluate, train_test_split

# ── 1. Generate fake customer purchase data ─────────────────────────────────

fake = Faker()
Faker.seed(42)
rng = np.random.default_rng(42)

N_CUSTOMERS = 500
N_PRODUCTS = 80
N_INTERACTIONS = 5_000


def make_customers(n: int) -> list[dict[str, str]]:
    """Generate realistic customer profiles."""
    customers = []
    for _ in range(n):
        customers.append(
            {
                "id": f"C-{fake.random_int(10000, 99999)}",
                "name": fake.name(),
                "email": fake.email(),
                "city": fake.city(),
            }
        )
    return customers


def make_products(n: int) -> list[dict[str, str]]:
    """Generate realistic product catalogue."""
    products = []
    seen: set[str] = set()
    while len(products) < n:
        name = f"{fake.word().capitalize()} {fake.color_name()}"
        if name not in seen:
            seen.add(name)
            products.append(
                {
                    "sku": f"SKU-{fake.bothify('??-####').upper()}",
                    "name": name,
                    "category": fake.word().capitalize(),
                }
            )
    return products


customers = make_customers(N_CUSTOMERS)
products = make_products(N_PRODUCTS)

customer_ids = [c["id"] for c in customers]
product_skus = [p["sku"] for p in products]

# Power-law purchase distribution (some products are much more popular)
product_weights = 1.0 / np.arange(1, N_PRODUCTS + 1, dtype=float) ** 0.7
product_weights /= product_weights.sum()

rows = []
pairs_seen: set[tuple[str, str]] = set()
while len(rows) < N_INTERACTIONS:
    c = rng.choice(customer_ids)
    p = rng.choice(product_skus, p=product_weights)
    if (c, p) not in pairs_seen:
        pairs_seen.add((c, p))
        rows.append(
            {
                "customer_id": c,
                "product_sku": p,
                "rating": round(float(rng.uniform(1.0, 5.0)), 1),
            }
        )

df = pd.DataFrame(rows)
print(f"Generated {len(df):,} unique customer–product interactions")
print(f"  Customers: {df['customer_id'].nunique()}")
print(f"  Products:  {df['product_sku'].nunique()}")
print(f"\nSample:\n{df.head()}\n")


# ── 2. Train/test split ─────────────────────────────────────────────────────

train_df, test_df = train_test_split(
    df, user_col="customer_id", item_col="product_sku", test_size=0.2
)
print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}\n")


# ── 3. Train ALS model ──────────────────────────────────────────────────────

model = ALS.from_transactions(
    train_df,
    user_col="customer_id",
    item_col="product_sku",
    rating_col="rating",
    factors=32,
    iterations=15,
    seed=42,
).fit()

print(f"ALS model trained: {model.user_factors.shape[0]} users × {model.item_factors.shape[0]} items")
print(f"  Factors: {model.user_factors.shape[1]}\n")


# ── 4. Evaluate ─────────────────────────────────────────────────────────────

eval_df = test_df.rename(columns={"customer_id": "user", "product_sku": "item"})
metrics = evaluate(model, eval_df, k=10)
print("Evaluation metrics @10:")
for name, val in metrics.items():
    print(f"  {name:>10s}: {val:.4f}")
print()


# ── 5. Personalised recommendations ─────────────────────────────────────────

# Build a product lookup for pretty printing
sku_to_name = {p["sku"]: p["name"] for p in products}

print("🎯 Personalised recommendations for 3 random customers:\n")
sample_users = rng.choice(range(model.user_factors.shape[0]), size=3, replace=False)
for uid in sample_users:
    label = model._user_labels[uid] if hasattr(model, "_user_labels") and model._user_labels is not None else uid
    cust_info = next((c for c in customers if c["id"] == label), None)
    display = f"{cust_info['name']} ({label})" if cust_info else str(label)

    item_ids, scores = model.recommend_items(int(uid), n=5, exclude_seen=True)

    print(f"  Customer: {display}")
    for rank, (iid, score) in enumerate(zip(item_ids, scores), 1):
        item_label = (
            model._item_labels[iid] if hasattr(model, "_item_labels") and model._item_labels is not None else iid
        )
        name = sku_to_name.get(str(item_label), str(item_label))
        print(f"    {rank}. {name} (score: {score:.3f})")
    print()


# ── 6. Market-basket cross-sell ──────────────────────────────────────────────

# Build basket matrix for association rule mining
basket_ohe = FPGrowth.from_transactions(
    train_df,
    transaction_col="customer_id",
    item_col="product_sku",
    min_support=0.03,
)
freq = basket_ohe.mine(use_colnames=True)
rules = basket_ohe.association_rules(metric="lift", min_threshold=1.2)  # type: ignore

print(f"Basket analysis: {len(freq):,} frequent itemsets, {len(rules):,} rules")

if len(rules) > 0:
    rec = Recommender(model=model, rules_df=rules)
    top_rules = (
        rules.query("confidence >= 0.3")
        .sort_values("lift", ascending=False)
        .head(5)[["antecedents", "consequents", "confidence", "lift"]]
    )
    print(f"\n🛒 Top 5 cross-sell rules:\n{top_rules.to_string()}\n")

print("Done! 🎉")
