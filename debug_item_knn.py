import pandas as pd
import rusket

df = pd.read_parquet("tests/.dataset_cache/online_retail_II_sample.parquet")
model = rusket.ItemKNN.from_transactions(df, user_col="Customer_ID", item_col="Description", method="bm25", k=20)
print(f"Model keys: {model.__dict__.keys()}")
