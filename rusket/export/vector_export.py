"""Export latent factor vectors to external vector databases.

Provides a single ``export_vectors()`` function that auto-detects the
vector DB backend from the client/connection object you pass in.

Supported backends:

- **Qdrant** — ``pip install qdrant-client``
- **Meilisearch** — ``pip install meilisearch``
- **pgvector** — ``pip install psycopg2-binary`` (or ``psycopg``)
- **ChromaDB** — ``pip install chromadb``
- **Pinecone** — ``pip install pinecone-client``
- **Weaviate** — ``pip install weaviate-client``
- **Milvus** — ``pip install pymilvus``
- **Elasticsearch / OpenSearch** — ``pip install elasticsearch`` (or ``opensearch-py``)
- **MongoDB Atlas** — ``pip install pymongo``
- **LanceDB** — ``pip install lancedb``
- **Typesense** — ``pip install typesense``
"""

from __future__ import annotations

from typing import Any

import numpy as np


def export_vectors(
    factors: np.ndarray,
    client: Any,
    collection_name: str = "item_factors",
    *,
    ids: list[Any] | np.ndarray | None = None,
    payloads: list[dict[str, Any]] | None = None,
    batch_size: int = 1000,
    recreate: bool = True,
    **kwargs: Any,
) -> int:
    """Upload factor vectors to a vector database.

    The backend is **auto-detected** from the ``client`` object type.

    Parameters
    ----------
    factors : ndarray of shape (n, d)
        The latent factor matrix (e.g. ``model.item_factors``).
    client
        An initialized vector DB client or database connection:

        - ``qdrant_client.QdrantClient``
        - ``meilisearch.Client``
        - ``psycopg2.connection`` or ``psycopg.Connection`` (pgvector)
        - ``chromadb.Client`` / ``chromadb.PersistentClient``
        - ``pinecone.Index``
        - ``weaviate.Client`` / ``weaviate.WeaviateClient``
        - ``pymilvus.Collection``
        - ``elasticsearch.Elasticsearch`` / ``opensearchpy.OpenSearch``
        - ``pymongo.database.Database`` (MongoDB Atlas)
        - ``lancedb.DBConnection``
        - ``typesense.Client``
    collection_name : str
        Target collection/index/table name.
    ids : list or ndarray, optional
        Point/document IDs. Defaults to ``range(n)``.
    payloads : list[dict], optional
        Optional metadata per vector (e.g. item names, categories).
    batch_size : int
        Upload batch size.
    recreate : bool
        If True, drop and recreate the collection/table.
    **kwargs
        Backend-specific options (e.g. ``distance="Cosine"`` for Qdrant).

    Returns
    -------
    int
        Number of vectors uploaded.

    Examples
    --------
    pgvector (PostgreSQL)::

        import psycopg2
        conn = psycopg2.connect("dbname=mydb")
        rusket.export_vectors(model.item_factors, conn, "item_embeddings")

    Qdrant::

        from qdrant_client import QdrantClient
        rusket.export_vectors(model.item_factors, QdrantClient("localhost"))

    ChromaDB::

        import chromadb
        client = chromadb.PersistentClient("./chroma_db")
        rusket.export_vectors(model.item_factors, client, "items")
    """
    factors = np.ascontiguousarray(factors, dtype=np.float32)
    n, _d = factors.shape

    if ids is None:
        ids_list = list(range(n))
    elif isinstance(ids, np.ndarray):
        ids_list = ids.tolist()
    else:
        ids_list = list(ids)

    backend = _detect_backend(client)

    dispatch = {
        "qdrant": _export_qdrant,
        "meilisearch": _export_meilisearch,
        "pgvector": _export_pgvector,
        "chromadb": _export_chromadb,
        "pinecone": _export_pinecone,
        "weaviate": _export_weaviate,
        "milvus": _export_milvus,
        "elasticsearch": _export_elasticsearch,
        "mongodb": _export_mongodb,
        "lancedb": _export_lancedb,
        "typesense": _export_typesense,
    }

    fn = dispatch[backend]
    return fn(
        factors,
        client,
        collection_name,
        ids_list,
        payloads,
        batch_size,
        recreate,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Multi-vector export (DB-side fusion)
# ---------------------------------------------------------------------------

_MULTI_VECTOR_BACKENDS = frozenset({"qdrant", "meilisearch", "weaviate"})


def export_multi_vectors(
    named_vectors: dict[str, np.ndarray],
    client: Any,
    collection_name: str = "item_factors",
    *,
    ids: list[Any] | np.ndarray | None = None,
    payloads: list[dict[str, Any]] | None = None,
    batch_size: int = 1000,
    recreate: bool = True,
    **kwargs: Any,
) -> int:
    """Upload multiple named vector spaces to a vector database.

    Instead of fusing embeddings client-side, this exports each embedding
    space as a separate named vector, allowing the database to handle
    fusion / re-ranking at query time.

    Parameters
    ----------
    named_vectors : dict[str, ndarray]
        Mapping of vector space name → (n, d) embedding matrix.
        Example: ``{"cf": cf_factors, "semantic": text_vectors}``.
        All matrices must have the same number of rows.
    client
        An initialized vector DB client.  Only backends that natively
        support multiple vectors per point are accepted:

        - ``qdrant_client.QdrantClient`` (named vectors)
        - ``meilisearch.Client`` (multiple embedders)
        - ``weaviate.WeaviateClient`` (named vectors, v4)
    collection_name : str
        Target collection / index name.
    ids : list or ndarray, optional
        Point / document IDs.  Defaults to ``range(n)``.
    payloads : list[dict], optional
        Optional metadata per vector.
    batch_size : int
        Upload batch size.
    recreate : bool
        If True, drop and recreate the collection.
    **kwargs
        Backend-specific options (e.g. ``distance="Cosine"`` for Qdrant).

    Returns
    -------
    int
        Number of points uploaded.

    Raises
    ------
    ValueError
        If *named_vectors* is empty or matrices have mismatched row counts.
    NotImplementedError
        If the detected backend does not support multi-vector storage.

    Examples
    --------
    Qdrant named vectors::

        from qdrant_client import QdrantClient
        rusket.export_multi_vectors(
            {"cf": model.item_factors, "semantic": text_vectors},
            QdrantClient("localhost"),
            "hybrid_items",
        )

    Meilisearch multi-embedder::

        import meilisearch
        rusket.export_multi_vectors(
            {"cf": model.item_factors, "semantic": text_vectors},
            meilisearch.Client("http://localhost:7700"),
            "items",
        )
    """
    if not named_vectors:
        raise ValueError("named_vectors must be a non-empty dict.")

    # Validate & coerce
    coerced: dict[str, np.ndarray] = {}
    n: int | None = None
    for name, mat in named_vectors.items():
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"Vector matrix '{name}' must be 2-D, got ndim={mat.ndim}.")
        if n is None:
            n = mat.shape[0]
        elif mat.shape[0] != n:
            raise ValueError(f"Row count mismatch: first matrix has {n} rows, '{name}' has {mat.shape[0]} rows.")
        coerced[name] = mat

    assert n is not None  # guaranteed by non-empty check

    if ids is None:
        ids_list = list(range(n))
    elif isinstance(ids, np.ndarray):
        ids_list = ids.tolist()
    else:
        ids_list = list(ids)

    backend = _detect_backend(client)

    if backend not in _MULTI_VECTOR_BACKENDS:
        raise NotImplementedError(
            f"Backend '{backend}' does not support multi-vector storage. "
            f"Supported backends for multi-vector export: {sorted(_MULTI_VECTOR_BACKENDS)}."
        )

    dispatch = {
        "qdrant": _export_qdrant_multi,
        "meilisearch": _export_meilisearch_multi,
        "weaviate": _export_weaviate_multi,
    }

    return dispatch[backend](
        coerced,
        client,
        collection_name,
        ids_list,
        payloads,
        batch_size,
        recreate,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Multi-vector backends
# ---------------------------------------------------------------------------


def _export_qdrant_multi(
    named_vectors: dict[str, np.ndarray],
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    distance: str = "Dot",
    **_kw: Any,
) -> int:
    """Export named vectors to Qdrant (one vector space per embedding type)."""
    from qdrant_client.models import (  # type: ignore[import-untyped]
        Distance,
        PointStruct,
        VectorParams,
    )

    n = next(iter(named_vectors.values())).shape[0]
    dist_map = {"Dot": Distance.DOT, "Cosine": Distance.COSINE, "Euclid": Distance.EUCLID}
    if distance not in dist_map:
        raise ValueError(f"Unknown distance '{distance}'. Must be one of {list(dist_map)}")

    if recreate:
        vectors_config = {
            name: VectorParams(size=mat.shape[1], distance=dist_map[distance]) for name, mat in named_vectors.items()
        }
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        points = [
            PointStruct(
                id=ids[i],
                vector={name: mat[i].tolist() for name, mat in named_vectors.items()},
                payload=payloads[i] if payloads else {},
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection_name, points=points)
    return n


def _export_meilisearch_multi(
    named_vectors: dict[str, np.ndarray],
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    _recreate: bool,
    **_kw: Any,
) -> int:
    """Export multiple named embedders to Meilisearch."""
    n = next(iter(named_vectors.values())).shape[0]
    index = client.index(collection_name)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = []
        for i in range(start, end):
            doc = dict(payloads[i]) if payloads and i < len(payloads) else {}
            doc["id"] = ids[i]
            doc["_vectors"] = {name: mat[i].tolist() for name, mat in named_vectors.items()}
            batch.append(doc)
        index.add_documents(batch)
    return n


def _export_weaviate_multi(
    named_vectors: dict[str, np.ndarray],
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    _batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    """Export named vectors to Weaviate v4."""
    n = next(iter(named_vectors.values())).shape[0]

    if not hasattr(client, "collections"):
        raise NotImplementedError("Multi-vector export requires Weaviate v4 client (with client.collections API).")

    if recreate:
        try:
            client.collections.delete(collection_name)
        except Exception:
            pass
    col = client.collections.create(name=collection_name)
    with col.batch.dynamic() as batch:
        for i in range(n):
            props = payloads[i] if payloads and i < len(payloads) else {}
            batch.add_object(
                properties=props,
                vector={name: mat[i].tolist() for name, mat in named_vectors.items()},
                uuid=str(ids[i]),
            )
    return n


def _detect_backend(client: Any) -> str:
    """Auto-detect which vector DB the client belongs to."""
    client_mod = type(client).__module__
    client_name = type(client).__qualname__

    if "qdrant_client" in client_mod:
        return "qdrant"
    if "meilisearch" in client_mod:
        return "meilisearch"
    if "psycopg2" in client_mod or "psycopg" in client_mod:
        return "pgvector"
    if "chromadb" in client_mod:
        return "chromadb"
    if "pinecone" in client_mod:
        return "pinecone"
    if "weaviate" in client_mod:
        return "weaviate"
    if "pymilvus" in client_mod:
        return "milvus"
    if "elasticsearch" in client_mod or "opensearchpy" in client_mod:
        return "elasticsearch"
    if "pymongo" in client_mod:
        return "mongodb"
    if "lancedb" in client_mod:
        return "lancedb"
    if "typesense" in client_mod:
        return "typesense"

    # Fallback: check for common attribute patterns
    if hasattr(client, "cursor") and hasattr(client, "commit"):
        return "pgvector"  # DB-API 2.0 connection

    supported = (
        "QdrantClient, meilisearch.Client, psycopg2 connection, "
        "chromadb.Client, pinecone.Index, weaviate.Client, "
        "pymilvus.Collection, elasticsearch.Elasticsearch, "
        "pymongo.database.Database, lancedb.DBConnection, typesense.Client"
    )
    raise TypeError(f"Unsupported client type: {client_name} (module: {client_mod}). Supported: {supported}.")


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def _export_qdrant(
    factors: np.ndarray,
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    distance: str = "Dot",
    **_kw: Any,
) -> int:
    from qdrant_client.models import (  # type: ignore[import-untyped]
        Distance,
        PointStruct,
        VectorParams,
    )

    n, d = factors.shape
    dist_map = {"Dot": Distance.DOT, "Cosine": Distance.COSINE, "Euclid": Distance.EUCLID}
    if distance not in dist_map:
        raise ValueError(f"Unknown distance '{distance}'. Must be one of {list(dist_map)}")

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


# ---------------------------------------------------------------------------
# Meilisearch
# ---------------------------------------------------------------------------


def _export_meilisearch(
    factors: np.ndarray,
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    _recreate: bool,
    embedder_name: str = "default",
    **_kw: Any,
) -> int:
    n = factors.shape[0]
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


# ---------------------------------------------------------------------------
# pgvector (PostgreSQL)
# ---------------------------------------------------------------------------


def _export_pgvector(
    factors: np.ndarray,
    conn: Any,
    table_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    """Export to pgvector via a psycopg2/psycopg DB-API connection."""
    import json

    n, d = factors.shape
    cur = conn.cursor()

    # Ensure pgvector extension exists
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    if recreate:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"CREATE TABLE {table_name} (  id TEXT PRIMARY KEY,  embedding vector({d}),  payload JSONB DEFAULT '{{}}')"
        )
        conn.commit()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        values = []
        for i in range(start, end):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in factors[i]) + "]"
            payload_json = json.dumps(payloads[i]) if payloads and i < len(payloads) else "{}"
            values.append((str(ids[i]), vec_str, payload_json))

        args_str = ",".join(
            cur.mogrify("(%s, %s::vector, %s::jsonb)", v).decode()
            if hasattr(cur.mogrify("", ()), "decode")
            else cur.mogrify("(%s, %s::vector, %s::jsonb)", v)
            for v in values
        )
        cur.execute(
            f"INSERT INTO {table_name} (id, embedding, payload) VALUES {args_str} "
            f"ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload"
        )
    conn.commit()
    cur.close()
    return n


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------


def _export_chromadb(
    factors: np.ndarray,
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "ip"},  # inner product
    )

    n = factors.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_ids = [str(ids[i]) for i in range(start, end)]
        chunk_vecs = factors[start:end].tolist()
        chunk_meta = [payloads[i] for i in range(start, end)] if payloads else None
        collection.upsert(
            ids=chunk_ids,
            embeddings=chunk_vecs,
            metadatas=chunk_meta,
        )
    return n


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------


def _export_pinecone(
    factors: np.ndarray,
    index: Any,
    _collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    _recreate: bool,
    **_kw: Any,
) -> int:
    """Export to a Pinecone Index object."""
    n = factors.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        vectors = []
        for i in range(start, end):
            v: dict[str, Any] = {"id": str(ids[i]), "values": factors[i].tolist()}
            if payloads and i < len(payloads):
                v["metadata"] = payloads[i]
            vectors.append(v)
        index.upsert(vectors=vectors)
    return n


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------


def _export_weaviate(
    factors: np.ndarray,
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    n = factors.shape[0]

    # Weaviate v4 API
    if hasattr(client, "collections"):
        if recreate:
            try:
                client.collections.delete(collection_name)
            except Exception:
                pass
        col = client.collections.create(name=collection_name)
        with col.batch.dynamic() as batch:
            for i in range(n):
                props = payloads[i] if payloads and i < len(payloads) else {}
                batch.add_object(properties=props, vector=factors[i].tolist(), uuid=str(ids[i]))
    else:
        # Weaviate v3 fallback
        with client.batch as batch:
            for i in range(n):
                props = payloads[i] if payloads and i < len(payloads) else {}
                batch.add_data_object(
                    data_object=props,
                    class_name=collection_name,
                    vector=factors[i].tolist(),
                    uuid=str(ids[i]),
                )
    return n


# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------


def _export_milvus(
    factors: np.ndarray,
    collection: Any,
    _collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    _recreate: bool,
    **_kw: Any,
) -> int:
    """Export to a pymilvus Collection object (must already be created)."""
    n = factors.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        data = [
            [ids[i] for i in range(start, end)],
            factors[start:end].tolist(),
        ]
        if payloads:
            import json

            data.append([json.dumps(payloads[i]) for i in range(start, end)])
        collection.insert(data)
    collection.flush()
    return n


# ---------------------------------------------------------------------------
# Elasticsearch / OpenSearch
# ---------------------------------------------------------------------------


def _export_elasticsearch(
    factors: np.ndarray,
    client: Any,
    index_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    """Export to Elasticsearch or OpenSearch using kNN dense_vector fields."""
    n, d = factors.shape

    if recreate:
        try:
            client.indices.delete(index=index_name)
        except Exception:
            pass
        client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": d,
                            "index": True,
                            "similarity": "dot_product",
                        },
                        "payload": {"type": "object", "enabled": False},
                    }
                }
            },
        )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        body: list[Any] = []
        for i in range(start, end):
            body.append({"index": {"_index": index_name, "_id": str(ids[i])}})
            doc: dict[str, Any] = {"embedding": factors[i].tolist()}
            if payloads and i < len(payloads):
                doc["payload"] = payloads[i]
            body.append(doc)
        client.bulk(body=body)
    return n


# ---------------------------------------------------------------------------
# MongoDB Atlas Vector Search
# ---------------------------------------------------------------------------


def _export_mongodb(
    factors: np.ndarray,
    db: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    """Export to MongoDB Atlas (expects a pymongo.database.Database)."""
    if recreate:
        db.drop_collection(collection_name)

    col = db[collection_name]
    n = factors.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        docs = []
        for i in range(start, end):
            doc: dict[str, Any] = {
                "_id": ids[i],
                "embedding": factors[i].tolist(),
            }
            if payloads and i < len(payloads):
                doc["payload"] = payloads[i]
            docs.append(doc)
        col.insert_many(docs, ordered=False)
    return n


# ---------------------------------------------------------------------------
# LanceDB
# ---------------------------------------------------------------------------


def _export_lancedb(
    factors: np.ndarray,
    db: Any,
    table_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    _batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    """Export to LanceDB (expects a lancedb.DBConnection)."""
    import pyarrow as pa  # type: ignore[import-untyped]

    n, d = factors.shape
    data = {
        "id": [str(x) for x in ids],
        "vector": [factors[i].tolist() for i in range(n)],
    }
    if payloads:
        import json

        data["payload"] = [json.dumps(payloads[i]) if i < len(payloads) else "{}" for i in range(n)]

    table = pa.table(data)

    if recreate:
        try:
            db.drop_table(table_name)
        except Exception:
            pass
    db.create_table(table_name, table)
    return n


# ---------------------------------------------------------------------------
# Typesense
# ---------------------------------------------------------------------------


def _export_typesense(
    factors: np.ndarray,
    client: Any,
    collection_name: str,
    ids: list[Any],
    payloads: list[dict[str, Any]] | None,
    batch_size: int,
    recreate: bool,
    **_kw: Any,
) -> int:
    """Export to Typesense (expects a typesense.Client)."""
    n, d = factors.shape

    if recreate:
        try:
            client.collections[collection_name].delete()
        except Exception:
            pass
        client.collections.create(
            {
                "name": collection_name,
                "fields": [
                    {"name": "id", "type": "string"},
                    {"name": "vec", "type": "float[]", "num_dim": d},
                    {"name": "payload", "type": "object", "optional": True},
                ],
            }
        )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        docs = []
        for i in range(start, end):
            doc: dict[str, Any] = {
                "id": str(ids[i]),
                "vec": factors[i].tolist(),
            }
            if payloads and i < len(payloads):
                doc["payload"] = payloads[i]
            docs.append(doc)
        client.collections[collection_name].documents.import_(docs, {"action": "upsert"})
    return n
