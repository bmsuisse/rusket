import pandas as pd

from rusket import AutoMiner

orders = pd.DataFrame({
    "receipt_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1004, 1004, 1004, 1004],
    "product":    ["milk", "bread", "butter",
                   "milk", "eggs",
                   "bread", "butter",
                   "milk", "bread", "eggs", "coffee"],
})

model = AutoMiner.from_transactions(orders, transaction_col="receipt_id", item_col="product", min_support=0.4)
freq  = model.mine(use_colnames=True)
rules = model.association_rules(metric="lift", min_threshold=1.0)
print("Frequent Itemsets:")
print(freq.head())
print("\nRules:")
print(rules.head())
