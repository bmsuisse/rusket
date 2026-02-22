"""Export utilities for integrating with vector databases and feature stores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .typing import SupportsItemFactors


def export_item_factors(model: SupportsItemFactors, include_labels: bool = True) -> pd.DataFrame:
    """Exports latent item factors as a Pandas DataFrame for Vector DBs.

    This format is ideal for ingesting into FAISS, Pinecone, or Qdrant for
    Retrieval-Augmented Generation (RAG) and semantic search.

    Parameters
    ----------
    model : SupportsItemFactors
        A fitted model instance with an ``item_factors`` property.
    include_labels : bool, default=True
        Whether to include the string item labels (if available from
        the model's fitting method).

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row is an item with columns ``item_id``,
        optionally ``item_label``, and ``vector`` (a dense 1-D numpy array
        of the item's latent factors).

    Examples
    --------
    >>> model = rusket.ALS(factors=32).fit(interactions)
    >>> df = rusket.export_item_factors(model)
    >>> # Ingest into FAISS / Pinecone / Qdrant
    >>> vectors = np.stack(df["vector"].values)
    """
    factors = model.item_factors
    if factors is None:
        raise ValueError("Model has not been fitted yet.")

    n_items = factors.shape[0]

    df_data = {
        "item_id": np.arange(n_items, dtype=np.int32),
        "vector": list(factors),  # List of 1D numpy arrays
    }

    if include_labels and hasattr(model, "_item_labels") and model._item_labels is not None:
        if len(model._item_labels) == n_items:
            df_data["item_label"] = model._item_labels

    # Ordering columns nicely
    cols = ["item_id"]
    if "item_label" in df_data:
        cols.append("item_label")
    cols.append("vector")

    return pd.DataFrame(df_data, columns=cols)  # type: ignore[arg-type]
