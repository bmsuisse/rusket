"""Export utilities for integrating with vector databases and feature stores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .als import ALS


def export_item_factors(als_model: ALS, include_labels: bool = True) -> pd.DataFrame:
    """Exports ALS latent item factors as a Pandas DataFrame for Vector DBs.

    This format is ideal for ingesting into FAISS, Pinecone, or Qdrant for
    Retrieval-Augmented Generation (RAG) and semantic search.

    Args:
        als_model: A fitted `rusket.ALS` model.
        include_labels: Whether to include the string item labels (if available).

    Returns:
        A Pandas DataFrame where each row is an item, containing its integer ID,
        (optionally) its label, and a `vector` column containing the dense latent numpy array.
    """
    if als_model.item_factors is None:
        raise ValueError("ALS model has not been fitted yet.")

    factors = als_model.item_factors
    n_items = factors.shape[0]

    df_data = {
        "item_id": np.arange(n_items, dtype=np.int32),
        "vector": list(factors),  # List of 1D numpy arrays
    }

    if include_labels and als_model._item_labels is not None:
        if len(als_model._item_labels) == n_items:
            df_data["item_label"] = als_model._item_labels

    # Ordering columns nicely
    cols = ["item_id"]
    if "item_label" in df_data:
        cols.append("item_label")
    cols.append("vector")

    return pd.DataFrame(df_data, columns=cols)
