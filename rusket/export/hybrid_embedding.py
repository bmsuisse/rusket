"""Hybrid embedding fusion: combine CF and semantic embeddings for retrieval.

Provides :class:`HybridEmbeddingIndex` for building a single ANN-searchable
vector space from collaborative-filtering item factors (ALS, BPR, …) and
external semantic embeddings (e.g. sentence-transformers, OpenAI, TF-IDF).

The index can either fuse embeddings client-side into a single dense vector
or export each embedding space as a **separate named vector** to databases
that support multi-vector storage (Qdrant, Meilisearch, Weaviate).

Three fusion strategies are available:

* ``"concat"`` — L2-normalise each space, then concatenate ``[cf; sem]``.
* ``"weighted_concat"`` (default) — normalise, scale by ``alpha`` / ``1-alpha``,
  then concatenate.  ``alpha=1`` gives pure CF, ``alpha=0`` gives pure semantic.
* ``"projection"`` — concatenate and project down to ``projection_dim`` via PCA,
  producing a compact, de-correlated representation.

Example
-------
>>> import rusket
>>> model = rusket.ALS(factors=32).fit(interactions)
>>> hybrid = rusket.HybridEmbeddingIndex(
...     cf_embeddings=model.item_factors,
...     semantic_embeddings=text_vectors,  # e.g. from sentence-transformers
...     strategy="weighted_concat",
...     alpha=0.6,
... )
>>> ids, scores = hybrid.query(item_id=42, n=10)
>>> # Or build an ANN index for sub-ms retrieval:
>>> ann = hybrid.build_ann_index(backend="native")
>>> ids, dists = ann.kneighbors(hybrid.fused_embeddings[[42]], n_neighbors=10)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

_STRATEGIES = ("concat", "weighted_concat", "projection")


# ── helpers ──────────────────────────────────────────────────────────────


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (safe for zero-norm rows)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, a_min=1e-10, a_max=None)


def _validate_inputs(
    cf: np.ndarray,
    sem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce input embeddings."""
    cf = np.asarray(cf, dtype=np.float32)
    sem = np.asarray(sem, dtype=np.float32)

    if cf.ndim != 2 or sem.ndim != 2:
        raise ValueError(f"Both embedding matrices must be 2-D. Got cf.ndim={cf.ndim}, sem.ndim={sem.ndim}.")
    if cf.shape[0] != sem.shape[0]:
        raise ValueError(f"Number of items must match: cf has {cf.shape[0]} rows, sem has {sem.shape[0]} rows.")
    return cf, sem


# ── public standalone function ───────────────────────────────────────────


def fuse_embeddings(
    cf_embeddings: npt.NDArray[np.float32],
    semantic_embeddings: npt.NDArray[np.float32],
    strategy: str = "weighted_concat",
    alpha: float = 0.5,
    projection_dim: int | None = None,
) -> npt.NDArray[np.float32]:
    """Fuse collaborative-filtering and semantic embedding matrices.

    Parameters
    ----------
    cf_embeddings : ndarray of shape (n_items, d_cf)
        Collaborative-filtering item factors (e.g. ``model.item_factors``).
    semantic_embeddings : ndarray of shape (n_items, d_sem)
        Text / content embeddings (e.g. from sentence-transformers).
    strategy : str, default="weighted_concat"
        One of ``"concat"``, ``"weighted_concat"``, ``"projection"``.
    alpha : float, default=0.5
        Weight for CF embeddings in ``"weighted_concat"`` strategy.
        ``1.0`` = pure CF, ``0.0`` = pure semantic.
    projection_dim : int | None, default=None
        Target dimensionality for the ``"projection"`` strategy.
        Defaults to ``min(d_cf + d_sem, 64)``.

    Returns
    -------
    ndarray of shape (n_items, d_fused)
        The fused embedding matrix.

    Raises
    ------
    ValueError
        If strategy is unknown or inputs have mismatched row counts.
    """
    cf, sem = _validate_inputs(cf_embeddings, semantic_embeddings)

    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Must be one of {_STRATEGIES}.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    cf_norm = _l2_normalise(cf)
    sem_norm = _l2_normalise(sem)

    if strategy == "concat":
        fused = np.hstack([cf_norm, sem_norm])

    elif strategy == "weighted_concat":
        fused = np.hstack([alpha * cf_norm, (1.0 - alpha) * sem_norm])

    elif strategy == "projection":
        concat = np.hstack([cf_norm, sem_norm])
        dim = projection_dim or min(concat.shape[1], 64)
        dim = min(dim, concat.shape[1], concat.shape[0])  # clamp

        # Centre
        mean = concat.mean(axis=0)
        centred = concat - mean

        # SVD-based PCA (no sklearn dependency)
        _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
        fused = centred @ vt[:dim].T
    else:
        raise ValueError(f"Unknown strategy {strategy!r}.")  # pragma: no cover

    return fused.astype(np.float32)


# ── index class ──────────────────────────────────────────────────────────


class HybridEmbeddingIndex:
    """Fuse CF and semantic embeddings, then query via cosine similarity or ANN.

    Parameters
    ----------
    cf_embeddings : ndarray of shape (n_items, d_cf)
        Collaborative-filtering item factors (e.g. ``model.item_factors``).
    semantic_embeddings : ndarray of shape (n_items, d_sem)
        Text / content embeddings.
    strategy : str, default="weighted_concat"
        Fusion strategy: ``"concat"``, ``"weighted_concat"``, ``"projection"``.
    alpha : float, default=0.5
        Weight for CF embeddings when using ``"weighted_concat"``.
    projection_dim : int | None, default=None
        Target dimensionality for ``"projection"`` strategy.

    Example
    -------
    >>> hybrid = HybridEmbeddingIndex(
    ...     cf_embeddings=als_model.item_factors,
    ...     semantic_embeddings=sentence_vectors,
    ...     alpha=0.6,
    ... )
    >>> item_ids, scores = hybrid.query(item_id=0, n=5)
    """

    def __init__(
        self,
        cf_embeddings: npt.NDArray[np.float32],
        semantic_embeddings: npt.NDArray[np.float32],
        strategy: str = "weighted_concat",
        alpha: float = 0.5,
        projection_dim: int | None = None,
    ) -> None:
        self._cf = np.asarray(cf_embeddings, dtype=np.float32)
        self._sem = np.asarray(semantic_embeddings, dtype=np.float32)
        self.strategy = strategy
        self.alpha = alpha
        self.projection_dim = projection_dim

        self._fused = fuse_embeddings(
            self._cf,
            self._sem,
            strategy=strategy,
            alpha=alpha,
            projection_dim=projection_dim,
        )
        self._fused_normed = _l2_normalise(self._fused)
        self._n_items = self._fused.shape[0]

    def __repr__(self) -> str:
        d_cf = self._cf.shape[1]
        d_sem = self._sem.shape[1]
        d_fused = self._fused.shape[1]
        return (
            f"HybridEmbeddingIndex(n_items={self._n_items}, "
            f"d_cf={d_cf}, d_sem={d_sem}, d_fused={d_fused}, "
            f"strategy={self.strategy!r}, alpha={self.alpha})"
        )

    # ── properties ───────────────────────────────────────────────────

    @property
    def fused_embeddings(self) -> npt.NDArray[np.float32]:
        """The fused embedding matrix, shape ``(n_items, d_fused)``."""
        return self._fused

    @property
    def n_items(self) -> int:
        """Number of items in the index."""
        return self._n_items

    @property
    def named_embeddings(self) -> dict[str, npt.NDArray[np.float32]]:
        """L2-normalised CF and semantic matrices as a named dict.

        Useful for multi-vector export where each embedding space is
        stored separately in the vector database.

        Returns
        -------
        dict[str, ndarray]
            ``{"cf": cf_normed, "semantic": sem_normed}``.
        """
        return {
            "cf": _l2_normalise(self._cf),
            "semantic": _l2_normalise(self._sem),
        }

    # ── query ────────────────────────────────────────────────────────

    def query(
        self,
        item_id: int,
        n: int = 10,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
        """Find the *n* most similar items to ``item_id`` via cosine similarity.

        Parameters
        ----------
        item_id : int
            Internal item index.
        n : int, default=10
            Number of similar items to return.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(item_ids, cosine_scores)`` sorted by descending similarity.
            The query item itself is excluded.
        """
        if item_id < 0 or item_id >= self._n_items:
            raise ValueError(f"item_id {item_id} out of bounds for {self._n_items} items.")

        scores = self._fused_normed @ self._fused_normed[item_id]
        scores[item_id] = -np.inf  # exclude self

        n = min(n, self._n_items - 1)
        if n <= 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

        top = np.argpartition(scores, -n)[-n:]
        top = top[np.argsort(-scores[top])]
        return top.astype(np.intp), scores[top].astype(np.float32)

    def query_vector(
        self,
        vector: npt.NDArray[np.float32],
        n: int = 10,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
        """Find the *n* closest items to an arbitrary vector.

        Parameters
        ----------
        vector : ndarray of shape (d_fused,)
            A query vector in the fused embedding space.
        n : int, default=10
            Number of results to return.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(item_ids, cosine_scores)`` sorted by descending similarity.
        """
        vec = np.asarray(vector, dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-10:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        vec = vec / vec_norm

        scores = self._fused_normed @ vec
        n = min(n, self._n_items)
        top = np.argpartition(scores, -n)[-n:]
        top = top[np.argsort(-scores[top])]
        return top.astype(np.intp), scores[top].astype(np.float32)

    # ── ANN index ────────────────────────────────────────────────────

    def build_ann_index(
        self,
        backend: str = "native",
        index_type: str = "hnsw",
        **kwargs: Any,
    ) -> Any:
        """Build an ANN index over the fused embeddings.

        Parameters
        ----------
        backend : str, default="native"
            ``"native"`` uses the built-in Rust random-projection forest
            (:class:`~rusket.ApproximateNearestNeighbors`).
            ``"faiss"`` uses a FAISS index (requires ``faiss-cpu`` or ``faiss-gpu``).
        index_type : str, default="hnsw"
            FAISS index type (only used when ``backend="faiss"``).
        **kwargs
            Extra arguments forwarded to the index constructor.

        Returns
        -------
        index
            A fitted ANN index with a ``query()`` / ``kneighbors()`` method.
        """
        if backend == "native":
            from .ann import ApproximateNearestNeighbors

            ann = ApproximateNearestNeighbors(**kwargs)
            ann.fit(self._fused)
            return ann
        elif backend == "faiss":
            from .faiss_ann import FAISSIndex

            idx = FAISSIndex(index_type=index_type, **kwargs)
            idx.build(self._fused)
            return idx
        else:
            raise ValueError(f"Unknown backend {backend!r}. Use 'native' or 'faiss'.")

    # ── export ───────────────────────────────────────────────────────

    def export_vectors(
        self,
        client: Any,
        *,
        mode: Literal["fused", "multi"] = "fused",
        **kwargs: Any,
    ) -> int:
        """Export embeddings to a vector database.

        Parameters
        ----------
        client : Any
            A vector DB client (Qdrant, pgvector, ChromaDB, Pinecone, …).
        mode : {"fused", "multi"}, default="fused"
            - ``"fused"`` — export the single fused embedding matrix
              (default, backward-compatible).
            - ``"multi"`` — export CF and semantic embeddings as
              **separate named vectors** so the database handles
              fusion at query time.  Requires a backend that supports
              multi-vector storage (Qdrant, Meilisearch, Weaviate).
        **kwargs
            Extra arguments forwarded to the export function.

        Returns
        -------
        int
            Number of vectors exported.

        Examples
        --------
        Fused (single vector, any backend)::

            hybrid.export_vectors(qdrant_client, collection_name="items")

        Multi-vector (DB-side fusion)::

            hybrid.export_vectors(qdrant_client, mode="multi",
                                  collection_name="items")
        """
        if mode == "multi":
            from .vector_export import export_multi_vectors

            return export_multi_vectors(self.named_embeddings, client=client, **kwargs)

        from .vector_export import export_vectors

        return export_vectors(self._fused, client=client, **kwargs)
