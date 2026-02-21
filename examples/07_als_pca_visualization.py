"""
Visualizing Latent Spaces (PCA) with ALS
========================================

This example demonstrates how to:
1. Load the Online Retail dataset using `pandas`.
2. Train a `rusket.ALS` collaborative filtering model.
3. Extract `item_factors` and perform L2 normalization and PCA using pure `numpy`.
4. Visualize the 3D space using `plotly`.

No external dependencies like `scikit-learn` or `PySpark` are required!
"""

import sys
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
from rusket import ALS

def download_online_retail(data_dir: Path) -> pd.DataFrame:
    """Downloads the Online Retail dataset."""
    url = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/refs/heads/master/data/retail-data/all/online-retail-dataset.csv"
    csv_path = data_dir / "online-retail-dataset.csv"
    
    if not csv_path.exists():
        print(f"Downloading Online Retail dataset to {data_dir}...")
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, csv_path)
        print("Download complete.")
        
    df = pd.read_csv(csv_path)
    # Filter out returns and missing customer IDs
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[df["Quantity"] > 0]
    return df

def compute_pca_3d(data: np.ndarray) -> np.ndarray:
    """Computes a 3D PCA projection using Singular Value Decomposition."""
    print("Computing SVD... ", end="")
    # Mean centering
    data_centered = data - np.mean(data, axis=0)
    
    # Singular Value Decomposition
    # Vt contains the principal components
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    
    # Extract the top 3 principal components
    components = Vt[:3]
    
    print("Done.")
    # Project data onto the components
    return np.dot(data_centered, components.T)

def main():
    # 1. Load the data
    data_dir = Path(__file__).parent.parent / "benchdata"
    purchases = download_online_retail(data_dir)
    print(f"Loaded {len(purchases):,} purchases across {purchases['StockCode'].nunique():,} items.")
    
    # 2. Fit the ALS model
    # We use anderson_m=5 to speed up convergence
    print("Fitting ALS model...")
    model = ALS(
        factors=64, 
        iterations=15, 
        alpha=40.0, 
        seed=42, 
        anderson_m=5,
        verbose=True
    )
    
    # `fit_transactions` will map original `StockCode` to internal matrix indices
    model.fit_transactions(purchases, user_col="CustomerID", item_col="StockCode")
    print(f"Successfully trained {model.item_factors.shape[0]} item embeddings.")
    
    # 3. L2 Normalization (Unit Sphere Projection)
    print("Applying L2 Normalization...")
    item_factors = model.item_factors
    item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    item_factors_norm = item_factors / np.clip(item_norms, a_min=1e-10, a_max=None)
    
    # 4. PCA Reduction
    print("Computing 3D PCA mapping...")
    item_pca = compute_pca_3d(item_factors_norm)
    
    # 5. Build DataFrame for Plotly
    df_viz = pd.DataFrame({
        "StockCode": model._item_labels, 
        "pca_1": item_pca[:, 0],
        "pca_2": item_pca[:, 1],
        "pca_3": item_pca[:, 2]
    })
    
    # Merge with the item descriptions
    df_items = purchases[["StockCode", "Description"]].drop_duplicates("StockCode")
    df_viz = df_viz.merge(df_items, on="StockCode", how="inner")
    
    print(f"Plotting {len(df_viz):,} items in 3D Latent Space...")
    
    # 6. Plotting
    fig = px.scatter_3d(
        df_viz,
        x="pca_1", 
        y="pca_2", 
        z="pca_3",
        hover_name="Description",
        title="ALS Latent Component Space (3D PCA Mapping) - Online Retail Dataset",
        opacity=0.7,
        height=800
    )
    
    fig.update_traces(marker=dict(size=3))
    
    # Save as HTML (or you can use fig.show() in a notebook)
    output_path = data_dir / "als_pca_visualization.html"
    fig.write_html(output_path)
    print(f"Successfully saved Plotly visualization to: {output_path}")

if __name__ == "__main__":
    import importlib.util
    if importlib.util.find_spec("plotly") is None:
        print("Please install plotly to run this example: `pip install plotly`")
        sys.exit(1)
        
    main()

