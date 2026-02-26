# Databricks ALS Cross-Sell Cookbook

This guide outlines how to use `rusket` within a Databricks/PySpark scale environment. It walks through the end-to-end process of generating recommendations, extracting latent factors for semantic search, and clustering users by cross-sell potential.

## 1. Setup and Sample Data

We'll start by loading sample transaction data into spark.

```python
import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
import rusket

spark = SparkSession.builder.getOrCreate()

# Create dummy sample data for articles and purchases
purchases = pd.DataFrame({
    "customer_id": np.random.randint(0, 1000, size=5000),
    "article_id": np.random.randint(0, 500, size=5000),
    "sales_amount": np.random.exponential(50, size=5000),
})

# Read in PySpark (simulate loading from a bronze layer)
spark_purchases = spark.createDataFrame(purchases)
```

## 2. Model Training

We fit the high-performance ALS model using implicit feedback (e.g., sales amounts).

```python
# Train the model, optionally passing pandas or polars dataframe directly 
# to avoid heavy JVM-to-Python serialization overhead when feasible
als = rusket.ALS(
    factors=64, 
    iterations=15, 
    alpha=40.0,
    seed=42
).from_transactions(
    spark_purchases.toPandas(), 
    transaction_col="customer_id",
    item_col="article_id",
    rating_col="sales_amount",
)
```

## 3. Extracting and Normalizing Latent Space (Embeddings)

Rather than keeping factors in memory or scoring one by one, we can export the trained underlying latent factors for semantic indexing or vector database lookups using LanceDB, FAISS, or Pinecone.

```python
# Export normalized vectors directly back to Spark DataFrames!
user_factors_df = als.export_user_factors(normalize=True, format="spark")
item_factors_df = als.export_factors(normalize=True, format="spark")

# Save to your Delta Lake silver tier
user_factors_df.write.format("delta").mode("overwrite").saveAsTable("silver_layer.user_embeddings")
item_factors_df.write.format("delta").mode("overwrite").saveAsTable("silver_layer.item_embeddings")
```

We can map our embeddings down to 3D and visualize them using Principal Component Analysis (PCA). `rusket` features a highly optimized Rust-backed `PCA` implementation.

You can use the fluent API to instantly project and visualize the latent space interactively:

```python
# Fluent API for instant 3D interactive visualization
fig = als.pca(n_components=3).plot(title="Latent Item Space")
fig.show()
```

If you need the principal components for downstream tasks or explicit ML pipelines, you can use the scikit-learn compatible object directly instead:

```python
# Object-oriented API for downstream machine learning tasks
pca_model = rusket.PCA(n_components=10)
reduced_embeddings = pca_model.fit_transform(als.item_embeddings)

print(pca_model.explained_variance_ratio_)
```

## 5. High-Speed Batch Scoring

We usually want to assign cross-sell scores to all users. Instead of a slow loop, `batch_recommend` accelerates this seamlessly across all cores using Rust.

```python
# Native Rust Rayon parallelism. Extremely fast.
recommendations = als.batch_recommend(n=20, exclude_seen=True, format="spark")

# Write out recommendations directly
recommendations.write.format("delta").mode("overwrite").saveAsTable("gold_layer.cross_sell_predictions")
```

## 6. Business Value: "Potential" Clustering

Using Databricks SQL or DataFrame APIs, we can categorize these recommendations into actionable tiers for email marketing queues (e.g. High / Medium / Low potential):

```python
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# Define quantiles over the `score` per item or overall
percent rank window
w = Window.partitionBy("item_id").orderBy(F.col("score").desc())

clustered = recommendations.withColumn(
    "percent_rank", F.percent_rank().over(w)
).withColumn(
    "potential", 
    F.when(F.col("percent_rank") <= 0.2, "High")
     .when(F.col("percent_rank") <= 0.6, "Medium")
     .otherwise("Low")
)

display(clustered.filter(F.col("potential") == "High"))
```
