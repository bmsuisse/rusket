"""CUDA-accelerated operations via CuPy or PyTorch.

This module provides CUDA implementations for all rusket models:
- Gramian computation (Y^T Y)
- Factor solve via Cholesky
- Batch recommendation (matmul + top-K)
- Dense matrix multiplication
- Sparse-dense matrix multiplication (for EASE, KNN)
- Attention forward pass (for SASRec, BERT4Rec)
- Layer normalization

Falls back gracefully when no CUDA library is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ── Backend detection ────────────────────────────────────────────────────────


def _get_cuda_backend() -> tuple[str, Any]:
    """Detect available CUDA backend.

    Returns
    -------
    (name, module) : tuple[str, module]
        ``("cupy", cupy)`` or ``("torch", torch)`` or raises ImportError.
    """
    try:
        import cupy  # type: ignore[import-untyped]

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return ("cupy", cupy)
    except (ImportError, Exception):
        pass

    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            return ("torch", torch)
    except (ImportError, Exception):
        pass

    raise ImportError(
        "No CUDA backend available. Install one of:\n"
        "  pip install cupy-cuda12x    # CuPy (recommended)\n"
        "  pip install torch           # PyTorch\n"
        "Or set use_cuda=False to use the CPU backend."
    )


def check_cuda_available() -> bool:
    """Check if a CUDA backend is available without raising."""
    try:
        _get_cuda_backend()
        return True
    except ImportError:
        return False


# ── Gramian ──────────────────────────────────────────────────────────────────


def gpu_gramian(factors: np.ndarray, backend: str, lib: Any) -> np.ndarray:
    """Compute Y^T Y on CUDA.

    Parameters
    ----------
    factors : ndarray of shape (n, k)
        Factor matrix (row-major).
    backend : str
        ``"cupy"`` or ``"torch"``.
    lib : module
        The CUDA library module.

    Returns
    -------
    ndarray of shape (k, k)
        The Gramian matrix, back on CPU.
    """
    if backend == "cupy":
        d = lib.asarray(factors)
        g = d.T @ d
        return lib.asnumpy(g)
    elif backend == "torch":
        d = lib.tensor(factors, device="cuda", dtype=lib.float32)
        g = d.T @ d
        return g.cpu().numpy()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Cholesky solve ──────────────────────────────────────────────────────────


def gpu_solve_cholesky(
    gramian: np.ndarray,
    interactions_yt: np.ndarray,
    regularization: float,
    backend: str,
    lib: Any,
) -> np.ndarray:
    """Solve (Y^TY + λI)X^T = Y^T R^T on CUDA via Cholesky.

    Parameters
    ----------
    gramian : ndarray (k, k)
        Pre-computed Y^T Y.
    interactions_yt : ndarray (k, n_rhs)
        Right-hand side: Y^T * (weighted interactions).
    regularization : float
        Regularization lambda.
    backend : str
        ``"cupy"`` or ``"torch"``.
    lib : module
        The CUDA library module.

    Returns
    -------
    ndarray (k, n_rhs)
        Solved factors.
    """
    k = gramian.shape[0]
    if backend == "cupy":
        A = lib.asarray(gramian) + regularization * lib.eye(k, dtype=lib.float32)
        B = lib.asarray(interactions_yt)
        X = lib.linalg.solve(A, B)
        return lib.asnumpy(X)
    elif backend == "torch":
        A = lib.tensor(gramian, device="cuda", dtype=lib.float32) + regularization * lib.eye(
            k, device="cuda", dtype=lib.float32
        )
        B = lib.tensor(interactions_yt, device="cuda", dtype=lib.float32)
        X = lib.linalg.solve(A, B)
        return X.cpu().numpy()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Batch recommend (matmul + top-K) ─────────────────────────────────────────


def gpu_batch_recommend(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    k: int,
    backend: str,
    lib: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-K items for all users via CUDA matmul + topk.

    Parameters
    ----------
    user_factors : ndarray (n_users, d)
    item_factors : ndarray (n_items, d)
    k : int
        Number of items per user.
    backend : str
    lib : module

    Returns
    -------
    (ids, scores) : tuple of ndarrays
        ids: (n_users, k), scores: (n_users, k)
    """
    if backend == "cupy":
        U = lib.asarray(user_factors)
        V = lib.asarray(item_factors)
        scores = U @ V.T
        top_idx = lib.argsort(-scores, axis=1)[:, :k]
        top_scores = lib.take_along_axis(scores, top_idx, axis=1)
        return lib.asnumpy(top_idx).astype(np.int32), lib.asnumpy(top_scores).astype(np.float32)
    elif backend == "torch":
        U = lib.tensor(user_factors, device="cuda", dtype=lib.float32)
        V = lib.tensor(item_factors, device="cuda", dtype=lib.float32)
        scores = U @ V.T
        top_scores, top_idx = lib.topk(scores, k=min(k, scores.shape[1]), dim=1)
        return top_idx.cpu().numpy().astype(np.int32), top_scores.cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Dense matrix multiply ────────────────────────────────────────────────────


def gpu_matmul(
    a: np.ndarray,
    b: np.ndarray,
    backend: str,
    lib: Any,
) -> np.ndarray:
    """Dense matrix multiply A @ B on CUDA.

    Parameters
    ----------
    a : ndarray (m, k)
    b : ndarray (k, n)
    backend : str
    lib : module

    Returns
    -------
    ndarray (m, n)
    """
    if backend == "cupy":
        A = lib.asarray(a)
        B = lib.asarray(b)
        C = A @ B
        return lib.asnumpy(C)
    elif backend == "torch":
        A = lib.tensor(a, device="cuda", dtype=lib.float32)
        B = lib.tensor(b, device="cuda", dtype=lib.float32)
        C = A @ B
        return C.cpu().numpy()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Top-K selection ──────────────────────────────────────────────────────────


def gpu_topk(
    scores: np.ndarray,
    k: int,
    backend: str,
    lib: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K selection per row on CUDA.

    Parameters
    ----------
    scores : ndarray (n, m)
        Score matrix.
    k : int
        Number of top entries per row.
    backend : str
    lib : module

    Returns
    -------
    (ids, top_scores) : tuple of ndarrays
        ids: (n, k), top_scores: (n, k)
    """
    if backend == "cupy":
        S = lib.asarray(scores)
        top_idx = lib.argsort(-S, axis=1)[:, :k]
        top_scores = lib.take_along_axis(S, top_idx, axis=1)
        return lib.asnumpy(top_idx).astype(np.int32), lib.asnumpy(top_scores).astype(np.float32)
    elif backend == "torch":
        S = lib.tensor(scores, device="cuda", dtype=lib.float32)
        top_scores, top_idx = lib.topk(S, k=min(k, S.shape[1]), dim=1)
        return top_idx.cpu().numpy().astype(np.int32), top_scores.cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── User scoring (single user x all items) ───────────────────────────────────


def gpu_score_user(
    user_vec: np.ndarray,
    item_factors: np.ndarray,
    backend: str,
    lib: Any,
) -> np.ndarray:
    """Compute scores for a single user against all items on CUDA.

    Parameters
    ----------
    user_vec : ndarray (d,)
        Single user factor vector.
    item_factors : ndarray (n_items, d)
        Item factor matrix.
    backend : str
    lib : module

    Returns
    -------
    ndarray (n_items,)
        Scores for each item.
    """
    if backend == "cupy":
        u = lib.asarray(user_vec)
        V = lib.asarray(item_factors)
        scores = V @ u
        return lib.asnumpy(scores)
    elif backend == "torch":
        u = lib.tensor(user_vec, device="cuda", dtype=lib.float32)
        V = lib.tensor(item_factors, device="cuda", dtype=lib.float32)
        scores = V @ u
        return scores.cpu().numpy()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Matrix inversion via Cholesky ─────────────────────────────────────────────


def gpu_cholesky_inverse(
    gram: np.ndarray,
    regularization: float,
    backend: str,
    lib: Any,
) -> np.ndarray:
    """Invert (G + λI) via Cholesky decomposition on CUDA.

    Parameters
    ----------
    gram : ndarray (n, n)
        Gram matrix.
    regularization : float
        Regularization lambda.
    backend : str
    lib : module

    Returns
    -------
    ndarray (n, n)
        Inverted matrix.
    """
    n = gram.shape[0]
    if backend == "cupy":
        G = lib.asarray(gram.astype(np.float64)) + regularization * lib.eye(n, dtype=lib.float64)
        identity = lib.eye(n, dtype=lib.float64)
        P = lib.linalg.solve(G, identity)
        return lib.asnumpy(P).astype(np.float32)
    elif backend == "torch":
        G = lib.tensor(gram.astype(np.float64), device="cuda", dtype=lib.float64)
        G = G + regularization * lib.eye(n, device="cuda", dtype=lib.float64)
        L = lib.linalg.cholesky(G)
        P = lib.cholesky_inverse(L)
        return P.cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Sparse-dense matrix multiply ─────────────────────────────────────────────


def gpu_sparse_dense_matmul(
    sparse_data: np.ndarray,
    sparse_indices: np.ndarray,
    sparse_indptr: np.ndarray,
    shape: tuple[int, int],
    dense: np.ndarray,
    backend: str,
    lib: Any,
) -> np.ndarray:
    """Sparse CSR × dense matrix multiply on CUDA.

    Parameters
    ----------
    sparse_data, sparse_indices, sparse_indptr : ndarrays
        CSR components of the sparse matrix.
    shape : tuple[int, int]
        Shape of the sparse matrix (rows, cols).
    dense : ndarray
        Dense matrix to multiply with.
    backend : str
    lib : module

    Returns
    -------
    ndarray
        Result of sparse @ dense.
    """
    if backend == "cupy":
        import cupyx.scipy.sparse as cusp  # type: ignore[import-untyped]

        sp_gpu = cusp.csr_matrix(
            (
                lib.asarray(sparse_data.astype(np.float32)),
                lib.asarray(sparse_indices.astype(np.int32)),
                lib.asarray(sparse_indptr.astype(np.int32)),
            ),
            shape=shape,
        )
        D = lib.asarray(dense.astype(np.float32))
        result = sp_gpu @ D
        return lib.asnumpy(result)
    elif backend == "torch":
        import scipy.sparse as sp

        csr = sp.csr_matrix(
            (sparse_data.astype(np.float32), sparse_indices.astype(np.int32), sparse_indptr),
            shape=shape,
        )
        coo = csr.tocoo()
        indices = lib.tensor(
            np.vstack([coo.row, coo.col]).astype(np.int64),
            device="cuda",
        )
        values = lib.tensor(coo.data.astype(np.float32), device="cuda")
        sp_gpu = lib.sparse_coo_tensor(indices, values, size=shape).to_sparse_csr()
        D = lib.tensor(dense.astype(np.float32), device="cuda")
        result = sp_gpu @ D
        return result.cpu().numpy()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Convenience: get backend or None ──────────────────────────────────────────


def get_cuda_backend_safe() -> tuple[str, Any] | None:
    """Try to get a CUDA backend, returning None on failure."""
    try:
        return _get_cuda_backend()
    except ImportError:
        return None


# ── Backward compatibility aliases ──────────────────────────────────────────

_get_gpu_backend = _get_cuda_backend
check_gpu_available = check_cuda_available
get_gpu_backend_safe = get_cuda_backend_safe
