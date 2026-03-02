"""Tests for GPU acceleration and vector DB export modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import rusket
from rusket.gpu import (
    check_gpu_available,
    gpu_batch_recommend,
    gpu_gramian,
    gpu_solve_cholesky,
)


# ---------------------------------------------------------------------------
# GPU operations (tested with mock CuPy)
# ---------------------------------------------------------------------------


def _make_mock_cupy() -> MagicMock:
    """Create a mock CuPy module with basic ndarray-like behaviour."""
    mock_cp = MagicMock()
    mock_cp.asarray = lambda x: np.asarray(x, dtype=np.float32)
    mock_cp.asnumpy = lambda x: np.asarray(x, dtype=np.float32)
    mock_cp.float32 = np.float32
    mock_cp.eye = np.eye
    mock_cp.linalg.solve = np.linalg.solve
    mock_cp.argsort = np.argsort
    mock_cp.take_along_axis = np.take_along_axis
    mock_cp.cuda.runtime.getDeviceCount.return_value = 1
    return mock_cp


class TestGPUGramian:
    def test_cupy_gramian_matches_numpy(self) -> None:
        rng = np.random.RandomState(42)
        Y = rng.randn(50, 8).astype(np.float32)
        expected = Y.T @ Y
        result = gpu_gramian(Y, backend="cupy", lib=_make_mock_cupy())
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            gpu_gramian(np.ones((5, 3), dtype=np.float32), backend="xla", lib=MagicMock())


class TestGPUSolveCholesky:
    def test_cupy_solve_identity(self) -> None:
        k = 4
        gramian = np.eye(k, dtype=np.float32)
        rhs = np.ones((k, 1), dtype=np.float32)
        reg = 0.1
        result = gpu_solve_cholesky(gramian, rhs, reg, "cupy", _make_mock_cupy())
        expected = np.linalg.solve(np.eye(k) + reg * np.eye(k), rhs)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestGPUBatchRecommend:
    def test_cupy_topk(self) -> None:
        users = np.eye(5, 3, dtype=np.float32)
        items = np.eye(10, 3, dtype=np.float32)
        ids, scores = gpu_batch_recommend(users, items, k=3, backend="cupy", lib=_make_mock_cupy())
        assert ids.shape == (5, 3)
        assert scores.shape == (5, 3)
        assert ids.dtype == np.int32


class TestGPUAvailability:
    def test_no_gpu_returns_false(self) -> None:
        with patch.dict("sys.modules", {"cupy": None, "torch": None}):
            assert check_gpu_available() is False


# ---------------------------------------------------------------------------
# ALS use_gpu parameter
# ---------------------------------------------------------------------------


class TestALSGPUParam:
    def test_use_gpu_default_false(self) -> None:
        assert rusket.ALS(factors=8).use_gpu is False

    def test_use_gpu_accepted(self) -> None:
        assert rusket.ALS(factors=8, use_gpu=True).use_gpu is True

    def test_use_gpu_fits_normally_when_false(self) -> None:
        mat = csr_matrix(np.random.randint(0, 2, (10, 15)).astype(np.float32))
        model = rusket.ALS(factors=4, iterations=3, seed=42, use_gpu=False)
        model.fit(mat)
        assert model.user_factors is not None


# ---------------------------------------------------------------------------
# Vector DB export (mocked clients)
# ---------------------------------------------------------------------------


class TestExportVectors:
    def test_unknown_backend_raises(self) -> None:
        from rusket.vector_export import export_vectors

        with pytest.raises(ValueError, match="Unknown backend"):
            export_vectors(np.zeros((5, 3), dtype=np.float32), backend="pinecone")

    def test_qdrant_import_error(self) -> None:
        with patch.dict("sys.modules", {"qdrant_client": None, "qdrant_client.models": None}):
            import importlib

            import rusket.vector_export as vmod

            importlib.reload(vmod)
            with pytest.raises(ImportError, match="qdrant-client"):
                vmod.export_vectors(np.zeros((5, 3), dtype=np.float32), backend="qdrant")

    def test_meilisearch_import_error(self) -> None:
        with patch.dict("sys.modules", {"meilisearch": None}):
            import importlib

            import rusket.vector_export as vmod

            importlib.reload(vmod)
            with pytest.raises(ImportError, match="meilisearch"):
                vmod.export_vectors(np.zeros((5, 3), dtype=np.float32), backend="meilisearch")
