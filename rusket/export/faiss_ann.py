"""FAISS-based Approximate Nearest Neighbor index for latent factor retrieval.

This is an **optional** module — requires ``pip install faiss-cpu`` or ``faiss-gpu``.
For a zero-dependency alternative, use :class:`rusket.ApproximateNearestNeighbors`
which uses a native Rust random-projection forest.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _import_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-untyped]

        return faiss
    except ImportError as e:
        raise ImportError(
            "FAISS is required for this feature. Install it via:\n"
            "  pip install faiss-cpu    # CPU-only\n"
            "  pip install faiss-gpu    # GPU-accelerated"
        ) from e


class FAISSIndex:
    """Wrapper around a FAISS index for sub-millisecond top-K retrieval.

    Parameters
    ----------
    index_type : str
        Index type to build. One of:

        - ``"flat"`` — exact brute-force (baseline, no approximation)
        - ``"hnsw"`` — Hierarchical Navigable Small Worlds (recommended)
        - ``"ivfflat"`` — Inverted File with flat quantizer
        - ``"ivfpq"`` — Inverted File with Product Quantization (smallest memory)

    hnsw_m : int
        Number of neighbors per HNSW layer (default 32). Higher = better recall
        but slower build.
    hnsw_ef_construction : int
        Search depth during construction (default 200).
    hnsw_ef_search : int
        Search depth during query (default 128). Higher = better recall.
    nlist : int
        Number of Voronoi cells for IVF indexes (default 100).
    nprobe : int
        Number of cells to search for IVF indexes (default 10).
    pq_m : int
        Number of sub-quantizers for IVF-PQ (default 8).
    """

    def __init__(
        self,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        nlist: int = 100,
        nprobe: int = 10,
        pq_m: int = 8,
    ) -> None:
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self._index: Any = None
        self._dim: int = 0

    def build(self, factors: np.ndarray) -> FAISSIndex:
        """Build the FAISS index from a factor matrix.

        Parameters
        ----------
        factors : ndarray of shape (n, d)
            The latent factor matrix (e.g. ``model.item_factors``).

        Returns
        -------
        self
        """
        faiss = _import_faiss()
        factors = np.ascontiguousarray(factors, dtype=np.float32)
        n, d = factors.shape
        self._dim = d

        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(d)
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = self.hnsw_ef_construction
            self._index.hnsw.efSearch = self.hnsw_ef_search
        elif self.index_type == "ivfflat":
            nlist = min(self.nlist, n // 2) if n > 1 else 1
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.nprobe = self.nprobe
            self._index.train(factors)
        elif self.index_type == "ivfpq":
            nlist = min(self.nlist, n // 2) if n > 1 else 1
            pq_m = min(self.pq_m, d)
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
            self._index.nprobe = self.nprobe
            self._index.train(factors)
        else:
            raise ValueError(
                f"Unknown index_type: '{self.index_type}'. Must be one of: 'flat', 'hnsw', 'ivfflat', 'ivfpq'."
            )

        # Normalize for inner product search (FAISS METRIC_INNER_PRODUCT)
        self._index.add(factors)
        return self

    def query(self, vectors: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Find the top-K nearest items for each query vector.

        Parameters
        ----------
        vectors : ndarray of shape (n_queries, d) or (d,)
            Query vector(s). For ALS, this is typically ``model.user_factors[uid]``.
        k : int
            Number of nearest neighbors to return.

        Returns
        -------
        ids : ndarray of shape (n_queries, k)
            Indices of the nearest items.
        scores : ndarray of shape (n_queries, k)
            Inner-product scores (higher = more similar).
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        scores, ids = self._index.search(vectors, k)
        return ids.astype(np.int32), scores.astype(np.float32)

    def save(self, path: str) -> None:
        """Save the index to disk."""
        faiss = _import_faiss()
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        faiss.write_index(self._index, path)

    def load(self, path: str) -> FAISSIndex:
        """Load an index from disk."""
        faiss = _import_faiss()
        self._index = faiss.read_index(path)
        self._dim = self._index.d
        return self


def build_faiss_index(
    factors: np.ndarray,
    index_type: str = "hnsw",
    **kwargs: Any,
) -> FAISSIndex:
    """Build a FAISS index from a latent factor matrix.

    Convenience function — equivalent to ``FAISSIndex(index_type, **kwargs).build(factors)``.

    Parameters
    ----------
    factors : ndarray of shape (n, d)
        Latent factor matrix.
    index_type : str
        One of ``"flat"``, ``"hnsw"``, ``"ivfflat"``, ``"ivfpq"``.
    **kwargs
        Passed to :class:`FAISSIndex`.

    Returns
    -------
    FAISSIndex
    """
    return FAISSIndex(index_type=index_type, **kwargs).build(factors)
