"""Abstract VectorStore interface and concrete implementations.

Provides a uniform OOP interface for exporting embeddings to vector
databases, with support for both single-vector and multi-vector (named
vector) storage.

Supported backends
------------------
- :class:`QdrantVectorStore`
- :class:`MeilisearchVectorStore`
- :class:`WeaviateVectorStore`
- :class:`PgVectorStore`
- :class:`ChromaVectorStore`

Example
-------
>>> store = QdrantVectorStore(QdrantClient("localhost"))
>>> store.upload(model.item_factors, collection_name="items")
>>> # Multi-vector (DB-side fusion):
>>> store.upload_multi(
...     {"cf": model.item_factors, "semantic": text_vectors},
...     collection_name="hybrid_items",
... )
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import numpy.typing as npt


class VectorStore(abc.ABC):
    """Abstract base class for vector database backends.

    Subclasses must implement :meth:`upload`.  Backends that support
    multiple named vectors per point should also override
    :meth:`upload_multi`.

    Parameters
    ----------
    client : Any
        An initialised vector DB client or connection object.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        """The underlying vector DB client."""
        return self._client

    # -- required ---------------------------------------------------------

    @abc.abstractmethod
    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **kwargs: Any,
    ) -> int:
        """Upload a single vector matrix to the database.

        Parameters
        ----------
        vectors : ndarray of shape (n, d)
            The embedding matrix (e.g. ``model.item_factors``).
        collection_name : str
            Target collection / index / table name.
        ids : list or ndarray, optional
            Point IDs.  Defaults to ``range(n)``.
        payloads : list[dict], optional
            Metadata per point.
        batch_size : int
            Upload batch size.
        recreate : bool
            If True, drop and recreate the collection.
        **kwargs
            Backend-specific options.

        Returns
        -------
        int
            Number of vectors uploaded.
        """

    # -- optional (multi-vector) ------------------------------------------

    @property
    def supports_multi_vector(self) -> bool:
        """Whether this backend supports multiple named vectors per point."""
        return False

    def upload_multi(
        self,
        named_vectors: dict[str, npt.NDArray[np.float32]],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **kwargs: Any,
    ) -> int:
        """Upload multiple named vector spaces to the database.

        Parameters
        ----------
        named_vectors : dict[str, ndarray]
            Mapping of vector space name → (n, d) matrix.
        collection_name : str
            Target collection / index name.
        ids : list or ndarray, optional
            Point IDs.  Defaults to ``range(n)``.
        payloads : list[dict], optional
            Metadata per point.
        batch_size : int
            Upload batch size.
        recreate : bool
            If True, drop and recreate the collection.
        **kwargs
            Backend-specific options.

        Returns
        -------
        int
            Number of points uploaded.

        Raises
        ------
        NotImplementedError
            If the backend does not support multi-vector storage.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support multi-vector storage.")

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _prepare_ids(
        n: int,
        ids: list[Any] | np.ndarray | None,
    ) -> list[Any]:
        if ids is None:
            return list(range(n))
        if isinstance(ids, np.ndarray):
            return ids.tolist()
        return list(ids)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(client={type(self._client).__name__})"


# =====================================================================
# Concrete implementations
# =====================================================================


class QdrantVectorStore(VectorStore):
    """Qdrant vector store with named-vector support.

    Parameters
    ----------
    client : qdrant_client.QdrantClient
        An initialised Qdrant client.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        distance: str = "Dot",
        **_kw: Any,
    ) -> int:
        from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore[import-untyped]

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = factors.shape
        ids_list = self._prepare_ids(n, ids)

        dist_map = {"Dot": Distance.DOT, "Cosine": Distance.COSINE, "Euclid": Distance.EUCLID}
        if distance not in dist_map:
            raise ValueError(f"Unknown distance '{distance}'. Must be one of {list(dist_map)}")

        if recreate:
            self._client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=d, distance=dist_map[distance]),
            )

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = [
                PointStruct(
                    id=ids_list[i],
                    vector=factors[i].tolist(),
                    payload=payloads[i] if payloads else {},
                )
                for i in range(start, end)
            ]
            self._client.upsert(collection_name=collection_name, points=points)
        return n

    @property
    def supports_multi_vector(self) -> bool:
        return True

    def upload_multi(
        self,
        named_vectors: dict[str, npt.NDArray[np.float32]],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        distance: str = "Dot",
        **_kw: Any,
    ) -> int:
        from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore[import-untyped]

        coerced = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in named_vectors.items()}
        n = next(iter(coerced.values())).shape[0]
        ids_list = self._prepare_ids(n, ids)

        dist_map = {"Dot": Distance.DOT, "Cosine": Distance.COSINE, "Euclid": Distance.EUCLID}
        if distance not in dist_map:
            raise ValueError(f"Unknown distance '{distance}'. Must be one of {list(dist_map)}")

        if recreate:
            vectors_config = {
                name: VectorParams(size=mat.shape[1], distance=dist_map[distance]) for name, mat in coerced.items()
            }
            self._client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = [
                PointStruct(
                    id=ids_list[i],
                    vector={name: mat[i].tolist() for name, mat in coerced.items()},
                    payload=payloads[i] if payloads else {},
                )
                for i in range(start, end)
            ]
            self._client.upsert(collection_name=collection_name, points=points)
        return n


class MeilisearchVectorStore(VectorStore):
    """Meilisearch vector store with multi-embedder support.

    Parameters
    ----------
    client : meilisearch.Client
        An initialised Meilisearch client.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        embedder_name: str = "default",
        **_kw: Any,
    ) -> int:
        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = factors.shape[0]
        ids_list = self._prepare_ids(n, ids)
        index = self._client.index(collection_name)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = []
            for i in range(start, end):
                doc = dict(payloads[i]) if payloads and i < len(payloads) else {}
                doc["id"] = ids_list[i]
                doc["_vectors"] = {embedder_name: factors[i].tolist()}
                batch.append(doc)
            index.add_documents(batch)
        return n

    @property
    def supports_multi_vector(self) -> bool:
        return True

    def upload_multi(
        self,
        named_vectors: dict[str, npt.NDArray[np.float32]],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        coerced = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in named_vectors.items()}
        n = next(iter(coerced.values())).shape[0]
        ids_list = self._prepare_ids(n, ids)
        index = self._client.index(collection_name)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = []
            for i in range(start, end):
                doc = dict(payloads[i]) if payloads and i < len(payloads) else {}
                doc["id"] = ids_list[i]
                doc["_vectors"] = {name: mat[i].tolist() for name, mat in coerced.items()}
                batch.append(doc)
            index.add_documents(batch)
        return n


class WeaviateVectorStore(VectorStore):
    """Weaviate vector store (v4 API) with named-vector support.

    Parameters
    ----------
    client : weaviate.WeaviateClient
        An initialised Weaviate v4 client.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = factors.shape[0]
        ids_list = self._prepare_ids(n, ids)

        if hasattr(self._client, "collections"):
            if recreate:
                try:
                    self._client.collections.delete(collection_name)
                except Exception:
                    pass
            col = self._client.collections.create(name=collection_name)
            with col.batch.dynamic() as batch:
                for i in range(n):
                    props = payloads[i] if payloads and i < len(payloads) else {}
                    batch.add_object(
                        properties=props,
                        vector=factors[i].tolist(),
                        uuid=str(ids_list[i]),
                    )
        else:
            with self._client.batch as batch:
                for i in range(n):
                    props = payloads[i] if payloads and i < len(payloads) else {}
                    batch.add_data_object(
                        data_object=props,
                        class_name=collection_name,
                        vector=factors[i].tolist(),
                        uuid=str(ids_list[i]),
                    )
        return n

    @property
    def supports_multi_vector(self) -> bool:
        return hasattr(self._client, "collections")

    def upload_multi(
        self,
        named_vectors: dict[str, npt.NDArray[np.float32]],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        if not hasattr(self._client, "collections"):
            raise NotImplementedError("Multi-vector export requires Weaviate v4 (client.collections API).")

        coerced = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in named_vectors.items()}
        n = next(iter(coerced.values())).shape[0]
        ids_list = self._prepare_ids(n, ids)

        if recreate:
            try:
                self._client.collections.delete(collection_name)
            except Exception:
                pass
        col = self._client.collections.create(name=collection_name)
        with col.batch.dynamic() as batch:
            for i in range(n):
                props = payloads[i] if payloads and i < len(payloads) else {}
                batch.add_object(
                    properties=props,
                    vector={name: mat[i].tolist() for name, mat in coerced.items()},
                    uuid=str(ids_list[i]),
                )
        return n


class PgVectorStore(VectorStore):
    """pgvector (PostgreSQL) vector store.

    Parameters
    ----------
    client : psycopg2 connection or psycopg Connection
        A database connection with the ``vector`` extension enabled.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_pgvector

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_pgvector(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store.

    Parameters
    ----------
    client : chromadb.Client or chromadb.PersistentClient
        An initialised ChromaDB client.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_chromadb

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_chromadb(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )


class PineconeVectorStore(VectorStore):
    """Pinecone vector store.

    Parameters
    ----------
    client : pinecone.Index
        An initialised Pinecone index object.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_pinecone

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_pinecone(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )


class MilvusVectorStore(VectorStore):
    """Milvus vector store.

    Parameters
    ----------
    client : pymilvus.Collection
        An initialised Milvus collection object.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_milvus

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_milvus(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )


class ElasticsearchVectorStore(VectorStore):
    """Elasticsearch / OpenSearch vector store.

    Parameters
    ----------
    client : elasticsearch.Elasticsearch or opensearchpy.OpenSearch
        An initialised Elasticsearch or OpenSearch client.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_elasticsearch

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_elasticsearch(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )


class MongoDBVectorStore(VectorStore):
    """MongoDB Atlas vector store.

    Parameters
    ----------
    client : pymongo.database.Database
        A pymongo database object (with Atlas Vector Search enabled).
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_mongodb

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_mongodb(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )


class LanceDBVectorStore(VectorStore):
    """LanceDB vector store.

    Parameters
    ----------
    client : lancedb.DBConnection
        An initialised LanceDB connection.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_lancedb

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_lancedb(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )


class TypesenseVectorStore(VectorStore):
    """Typesense vector store.

    Parameters
    ----------
    client : typesense.Client
        An initialised Typesense client.
    """

    def upload(
        self,
        vectors: npt.NDArray[np.float32],
        collection_name: str = "item_factors",
        *,
        ids: list[Any] | np.ndarray | None = None,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 1000,
        recreate: bool = True,
        **_kw: Any,
    ) -> int:
        from .vector_export import _export_typesense

        factors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_list = self._prepare_ids(factors.shape[0], ids)
        return _export_typesense(
            factors,
            self._client,
            collection_name,
            ids_list,
            payloads,
            batch_size,
            recreate,
        )
