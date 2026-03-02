"""Export latent factor vectors to external vector databases.

Provides a single unified ``export_to_vectordb()`` function that supports
multiple backends (Qdrant, Meilisearch, etc.). Each backend requires its
respective Python client as an optional dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def export_vectors(
    factors: np.ndarray,
    backend: str = "qdrant",
    collection_name: str = "item_factors",
    *,
    ids: list[Any] | np.ndarray | None = None,
    payloads: list[dict[str, Any]] | None = None,
    url: str | None = None,
    api_key: str | None = None,
    batch_size: int = 1000,
    recreate: bool = True,
    **kwargs: Any,
) -> int:
    """Upload factor vectors to a vector database.

    Parameters
    ----------
    factors : ndarray of shape (n, d)
        The latent factor matrix (e.g. ``model.item_factors``).
    backend : str
        Vector DB backend. One of ``"qdrant"``, ``"meilisearch"``.
    collection_name : str
        Target collection/index name.
    ids : list or ndarray, optional
        Point/document IDs. Defaults to ``range(n)``.
    payloads : list[dict], optional
        Optional metadata per vector (e.g. item names, categories).
    url : str or None
        Server URL. Defaults to standard local ports.
    api_key : str or None
        API key for cloud deployments.
    batch_size : int
        Upload batch size.
    recreate : bool
        If True, delete and recreate the collection (Qdrant only).
    **kwargs
        Backend-specific options (e.g. ``distance="Cosine"`` for Qdrant,
        ``embedder_name="mf"`` for Meilisearch).

    Returns
    -------
    int
        Number of vectors uploaded.

    Examples
    --------
    >>> model = rusket.ALS(factors=64).fit(interactions)
    >>> rusket.export_to_vectordb(model.item_factors, backend="qdrant")
    >>> rusket.export_to_vectordb(model.item_factors, backend="meilisearch")
    """
    factors = np.ascontiguousarray(factors, dtype=np.float32)
    n, _d = factors.shape

    if ids is None:
        ids_list = list(range(n))
    elif isinstance(ids, np.ndarray):
        ids_list = ids.tolist()
    else:
        ids_list = list(ids)

    if backend == "qdrant":
        return _export_qdrant(
            factors,
            collection_name,
            ids_list,
            payloads,
            url or "http://localhost:6333",
            api_key,
            batch_size,
            recreate,
            **kwargs,
        )
    elif backend == "meilisearch":
        return _export_meilisearch(
            factors,
            collection_name,
            ids_list,
            payloads,
            url or "http://localhost:7700",
            api_key,
            batch_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported: 'qdrant', 'meilisearch'.")


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _export_qdrant(
    factors: np.ndarray,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    url: str,
    api_key: str | None,
    batch_size: int,
    recreate: bool,
    distance: str = "Dot",
    **client_kwargs: Any,
) -> int:
    try:
        from qdrant_client import QdrantClient  # type: ignore[import-untyped]
        from qdrant_client.models import (  # type: ignore[import-untyped]
            Distance,
            PointStruct,
            VectorParams,
        )
    except ImportError as e:
        raise ImportError("Qdrant client required. Install with: pip install qdrant-client") from e

    n, d = factors.shape
    dist_map = {"Dot": Distance.DOT, "Cosine": Distance.COSINE, "Euclid": Distance.EUCLID}
    if distance not in dist_map:
        raise ValueError(f"Unknown distance '{distance}'. Must be one of {list(dist_map)}")

    client = QdrantClient(url=url, api_key=api_key, **client_kwargs)
    if recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=d, distance=dist_map[distance]),
        )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        points = [
            PointStruct(
                id=ids[i],
                vector=factors[i].tolist(),
                payload=payloads[i] if payloads else {},
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection_name, points=points)

    return n


def _export_meilisearch(
    factors: np.ndarray,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    url: str,
    api_key: str | None,
    batch_size: int,
    embedder_name: str = "default",
    **_kwargs: Any,
) -> int:
    try:
        import meilisearch  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("Meilisearch client required. Install with: pip install meilisearch") from e

    n = factors.shape[0]
    client = meilisearch.Client(url, api_key)
    index = client.index(collection_name)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = []
        for i in range(start, end):
            doc = dict(payloads[i]) if payloads and i < len(payloads) else {}
            doc["id"] = ids[i]
            doc["_vectors"] = {embedder_name: factors[i].tolist()}
            batch.append(doc)
        index.add_documents(batch)

    return n
