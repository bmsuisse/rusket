import time
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS as PySparkALS
from pyspark.ml.fpm import FPGrowth as PySparkFPGrowth

from rusket import ALS, fpgrowth

def create_spark_session():
    return (
        SparkSession.builder.appName("rusket-vs-pyspark")
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def benchmark_als(spark, n_users=10_000, n_items=5_000, n_interactions=500_000):
    print(f"\n--- ALS Benchmark: {n_interactions:,} interactions ({n_users} U x {n_items} I) ---")
    
    np.random.seed(42)
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.rand(n_interactions) * 5.0
    
    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})
    
    # 1. rusket ALS
    t0 = time.time()
    rusket_model = ALS(factors=64, iterations=10, seed=42)
    rusket_model.fit_transactions(df, user_col="user_id", item_col="item_id", rating_col="rating")
    rusket_time = time.time() - t0
    print(f"🚀 rusket ALS Time:   {rusket_time:.3f}s")
    
    # 2. PySpark ALS
    print("Converting to PySpark df... (not timed)")
    spark_df = spark.createDataFrame(df)
    spark_df.cache()
    spark_df.count() # trigger cache
    
    t0 = time.time()
    pyspark_als = PySparkALS(
        userCol="user_id", 
        itemCol="item_id", 
        ratingCol="rating", 
        rank=64, 
        maxIter=10, 
        seed=42,
        coldStartStrategy="drop"
    )
    pyspark_model = pyspark_als.fit(spark_df)
    
    # Force evaluation
    # To be fair to PySpark, fit is mostly lazy, but we can force it by extracting item factors
    if hasattr(pyspark_model, "itemFactors"):
        pyspark_model.itemFactors.count()
        
    pyspark_time = time.time() - t0
    print(f"🐢 PySpark ALS Time:  {pyspark_time:.3f}s")
    
    speedup = pyspark_time / rusket_time
    print(f"🏆 Speedup vs PySpark: {speedup:.1f}x")
    

def generate_basket_data(n_transactions=50_000, n_items=1000, items_per_basket=10):
    np.random.seed(42)
    data = []
    
    # Generate Zipfian distribution for items to make it realistic
    a = 1.5
    item_probs = 1.0 / np.power(np.arange(1, n_items + 1), a)
    item_probs /= np.sum(item_probs)
    
    for i in range(n_transactions):
        # Vary basket size
        sz = int(np.random.normal(items_per_basket, items_per_basket // 2))
        sz = max(1, min(sz, n_items))
        
        # Select items
        basket = np.random.choice(n_items, size=sz, p=item_probs, replace=False)
        data.append({"txn_id": i, "items": [f"item_{it}" for it in basket]})
        
    return pd.DataFrame(data)

def generate_sparse_matrix_df(n_transactions=20_000, n_items=200, density=0.03):
    np.random.seed(42)
    mat = np.random.rand(n_transactions, n_items) < density
    df = pd.DataFrame(mat, columns=[f"Item_{i}" for i in range(n_items)])
    return df
    

def benchmark_fpgrowth(spark, n_transactions=20_000, n_items=500, min_support=0.01):
    print(f"\n--- FPGrowth Benchmark: {n_transactions:,} txns, {n_items} items, min_sup={min_support} ---")
    
    # Rusket takes one-hot encoded boolean DFs by default. 
    # PySpark takes arrays of strings.
    df_bool = generate_sparse_matrix_df(n_transactions, n_items, density=0.04)
    
    # Convert bool DF to list of items for PySpark
    print("Converting data formats...")
    # for PySpark we need an array of strings
    records = []
    for i, row in df_bool.iterrows():
        items = [col for col in df_bool.columns if row[col]]
        records.append({"items": items})
    df_array = pd.DataFrame(records)
    
    # 1. rusket FP-Growth
    t0 = time.time()
    res_rusket = fpgrowth(df_bool, min_support=min_support)
    rusket_time = time.time() - t0
    print(f"🚀 rusket FPGrowth:    {rusket_time:.3f}s (Found {len(res_rusket)} itemsets)")
    
    # 2. PySpark FP-Growth
    spark_df = spark.createDataFrame(df_array)
    spark_df.cache()
    spark_df.count()
    
    t0 = time.time()
    fp_model = PySparkFPGrowth(itemsCol="items", minSupport=min_support)
    model = fp_model.fit(spark_df)
    res_spark = model.freqItemsets.count()
    pyspark_time = time.time() - t0
    print(f"🐢 PySpark FPGrowth:   {pyspark_time:.3f}s (Found {res_spark} itemsets)")
    
    speedup = pyspark_time / rusket_time
    print(f"🏆 Speedup vs PySpark: {speedup:.1f}x")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    spark = create_spark_session()
    
    try:
        # Benchmark 1: FPGrowth
        benchmark_fpgrowth(spark, n_transactions=50_000, n_items=500, min_support=0.05)
        
        # Benchmark 2: ALS
        benchmark_als(spark, n_users=20_000, n_items=10_000, n_interactions=1_000_000)
    finally:
        spark.stop()
