"""
Example 13: Exporting and Serving models with LanceDB
=====================================================

This example demonstrates how to:
1. Train a model in rusket
2. Save it to disk, load it using a generic loader
3. Export items as embeddings
4. Ingest them into LanceDB for fast serving
"""

import rusket
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# 1. Create a dummy dataset (Users -> Items)
data = pd.DataFrame({
    "user_id": np.random.randint(0, 100, 1000),
    "item_id": np.random.randint(0, 50, 1000)
})

# 2. Train a Recommendation model
# Convert dataframe to internal representation using from_pandas
model = rusket.ALS.from_pandas(data, factors=8, iterations=5).fit()

# 3. Save the model 
with tempfile.TemporaryDirectory() as temp_dir:
    model_path = Path(temp_dir) / "als_model.pkl"
    model.save(model_path)
    print(f"✅ Saved model to {model_path}")

    # 4. generic load
    loaded_model = rusket.load_model(model_path)
    print(f"📦 Loaded model type: {type(loaded_model).__name__}")
    
    # 5. Export factors to the unified Pandas format
    items_df = rusket.export_item_factors(
        loaded_model, 
        normalize=True,     # L2 normalization for Cosine Similarity search
        format="pandas",
        include_labels=False
    )
    # The dataframe has `item_id` and `vector` columns.
    print(f"🧮 Extracted {len(items_df)} item embeddings")
    
    # 6. Serving via LanceDB
    print("\n🚀 Indexing into LanceDB...")
    import lancedb
    
    db = lancedb.connect(Path(temp_dir) / "lancedb")
    
    # Create LanceDB table
    table = db.create_table("items", data=items_df)
    
    # Example Query: finding items similar to User 10's embeddings
    print("🔍 Testing Vector Search:")
    user_vector = loaded_model.user_factors[10]
    
    # L2 normalize user vector for cosine similarity 
    user_vector = user_vector / max(np.linalg.norm(user_vector), 1e-9)
    
    results = table.search(user_vector).limit(5).to_pandas()
    
    print("\nTop 5 Items for User 10 via LanceDB vector search:")
    print(results[["item_id", "_distance"]])
