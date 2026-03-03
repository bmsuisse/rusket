"""Backward compatibility shim — use ``rusket.cuda`` instead."""

from __future__ import annotations

# Re-export everything from the new cuda module
from .cuda import (  # noqa: F401
    _get_cuda_backend as _get_gpu_backend,
)
from .cuda import (
    check_cuda_available as check_gpu_available,
)
from .cuda import (
    get_cuda_backend_safe as get_gpu_backend_safe,
)
from .cuda import (
    gpu_batch_recommend,
    gpu_cholesky_inverse,
    gpu_gramian,
    gpu_matmul,
    gpu_score_user,
    gpu_solve_cholesky,
    gpu_sparse_dense_matmul,
    gpu_topk,
)

__all__ = [
    "_get_gpu_backend",
    "check_gpu_available",
    "get_gpu_backend_safe",
    "gpu_batch_recommend",
    "gpu_cholesky_inverse",
    "gpu_gramian",
    "gpu_matmul",
    "gpu_score_user",
    "gpu_solve_cholesky",
    "gpu_sparse_dense_matmul",
    "gpu_topk",
]
