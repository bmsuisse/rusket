"""GPU-accelerated ALS operations via CuPy or PyTorch.

This module provides GPU implementations of the heavy ALS operations:
- Gramian computation (Y^T Y)
- Factor solve via Cholesky or CG

Falls back gracefully when no GPU library is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_gpu_backend() -> tuple[str, Any]:
    """Detect available GPU backend.

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
        "No GPU backend available. Install one of:\n"
        "  pip install cupy-cuda12x    # CuPy (recommended)\n"
        "  pip install torch           # PyTorch\n"
        "Or set use_gpu=False to use the CPU backend."
    )


def gpu_gramian(factors: np.ndarray, backend: str, lib: Any) -> np.ndarray:
    """Compute Y^T Y on GPU.

    Parameters
    ----------
    factors : ndarray of shape (n, k)
        Factor matrix (row-major).
    backend : str
        ``"cupy"`` or ``"torch"``.
    lib : module
        The GPU library module.

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


def gpu_solve_cholesky(
    gramian: np.ndarray,
    interactions_yt: np.ndarray,
    regularization: float,
    backend: str,
    lib: Any,
) -> np.ndarray:
    """Solve (Y^TY + λI)X^T = Y^T R^T on GPU via Cholesky.

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
        The GPU library module.

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


def gpu_batch_recommend(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    k: int,
    backend: str,
    lib: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-K items for all users via GPU matmul + topk.

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


def check_gpu_available() -> bool:
    """Check if a GPU backend is available without raising."""
    try:
        _get_gpu_backend()
        return True
    except ImportError:
        return False
