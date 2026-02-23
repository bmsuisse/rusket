from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .typing import SupportsItemFactors, SupportsUserFactors


def export_item_factors(
    model: SupportsItemFactors, include_labels: bool = True, normalize: bool = False, format: str = "pandas"
) -> Any:
    """Exports latent item factors as a DataFrame for Vector DBs.

    This format is ideal for ingesting into FAISS, Pinecone, or Qdrant for
    Retrieval-Augmented Generation (RAG) and semantic search.

    Parameters
    ----------
    model : SupportsItemFactors
        A fitted model instance with an ``item_factors`` property.
    include_labels : bool, default=True
        Whether to include the string item labels (if available from
        the model's fitting method).
    normalize : bool, default=False
        Whether to L2-normalize the factors before export.
    format : str, default="pandas"
        The DataFrame format to return. One of "pandas", "polars", or "spark".

    Returns
    -------
    Any
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
    import numpy as np
    import pandas as pd

    factors = model.item_factors
    if factors is None:
        raise ValueError("Model has not been fitted yet.")

    n_items = factors.shape[0]

    if normalize:
        norms = np.linalg.norm(factors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        factors = factors / norms

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

    df = pd.DataFrame(df_data, columns=cols)

    if format == "pandas":
        return df
    elif format == "polars":
        import polars as pl

        return pl.from_pandas(df)
    elif format == "spark":
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found.")
        return spark.createDataFrame(df)
    else:
        raise ValueError(f"Unknown format: {format}")


def export_user_factors(
    model: SupportsUserFactors, include_labels: bool = True, normalize: bool = False, format: str = "pandas"
) -> Any:
    """Exports latent user factors as a DataFrame.

    Parameters
    ----------
    model : SupportsUserFactors
        A fitted model instance with a ``user_factors`` property.
    include_labels : bool, default=True
        Whether to include the string user labels (if available).
    normalize : bool, default=False
        Whether to L2-normalize the factors before export.
    format : str, default="pandas"
        The DataFrame format to return. One of "pandas", "polars", or "spark".

    Returns
    -------
    Any
        A DataFrame where each row is a user with columns ``user_id``,
        optionally ``user_label``, and ``vector``.
    """
    import numpy as np
    import pandas as pd

    factors = model.user_factors
    if factors is None:
        raise ValueError("Model has not been fitted yet.")

    n_users = factors.shape[0]

    if normalize:
        norms = np.linalg.norm(factors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        factors = factors / norms

    df_data = {
        "user_id": np.arange(n_users, dtype=np.int32),
        "vector": list(factors),
    }

    if include_labels and hasattr(model, "_user_labels") and model._user_labels is not None:
        if len(model._user_labels) == n_users:
            df_data["user_label"] = model._user_labels

    cols = ["user_id"]
    if "user_label" in df_data:
        cols.append("user_label")
    cols.append("vector")

    df = pd.DataFrame(df_data, columns=cols)

    if format == "pandas":
        return df
    elif format == "polars":
        import polars as pl

        return pl.from_pandas(df)
    elif format == "spark":
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found.")
        return spark.createDataFrame(df)
    else:
        raise ValueError(f"Unknown format: {format}")
