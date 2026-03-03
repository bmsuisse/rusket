"""Tests for CUDA acceleration and vector DB export modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import rusket
from rusket._config import disable_cuda
from rusket.cuda import (
    check_cuda_available,
    gpu_batch_recommend,
    gpu_gramian,
    gpu_score_user,
    gpu_solve_cholesky,
)

# ---------------------------------------------------------------------------
# CUDA operations (tested with mock CuPy)
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


class TestCUDAGramian:
    def test_cupy_gramian_matches_numpy(self) -> None:
        rng = np.random.RandomState(42)
        Y = rng.randn(50, 8).astype(np.float32)
        expected = Y.T @ Y
        result = gpu_gramian(Y, backend="cupy", lib=_make_mock_cupy())
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            gpu_gramian(np.ones((5, 3), dtype=np.float32), backend="xla", lib=MagicMock())


class TestCUDASolveCholesky:
    def test_cupy_solve_identity(self) -> None:
        k = 4
        gramian = np.eye(k, dtype=np.float32)
        rhs = np.ones((k, 1), dtype=np.float32)
        reg = 0.1
        result = gpu_solve_cholesky(gramian, rhs, reg, "cupy", _make_mock_cupy())
        expected = np.linalg.solve(np.eye(k) + reg * np.eye(k), rhs)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestCUDABatchRecommend:
    def test_cupy_topk(self) -> None:
        users = np.eye(5, 3, dtype=np.float32)
        items = np.eye(10, 3, dtype=np.float32)
        ids, scores = gpu_batch_recommend(users, items, k=3, backend="cupy", lib=_make_mock_cupy())
        assert ids.shape == (5, 3)
        assert scores.shape == (5, 3)
        assert ids.dtype == np.int32


class TestCUDAScoreUser:
    def test_cupy_score_matches_numpy(self) -> None:
        rng = np.random.RandomState(42)
        user_vec = rng.randn(8).astype(np.float32)
        item_factors = rng.randn(20, 8).astype(np.float32)
        expected = item_factors @ user_vec
        result = gpu_score_user(user_vec, item_factors, backend="cupy", lib=_make_mock_cupy())
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestCUDAAvailability:
    def test_no_cuda_returns_false(self) -> None:
        with patch.dict("sys.modules", {"cupy": None, "torch": None}):
            assert check_cuda_available() is False


# ---------------------------------------------------------------------------
# ALS use_cuda parameter
# ---------------------------------------------------------------------------


class TestALSCUDAParam:
    def test_use_cuda_default(self) -> None:
        disable_cuda()
        assert rusket.ALS(factors=8).use_cuda is False

    def test_use_cuda_accepted(self) -> None:
        disable_cuda()
        assert rusket.ALS(factors=8, use_cuda=True).use_cuda is True

    def test_use_cuda_fits_normally_when_false(self) -> None:
        disable_cuda()
        mat = csr_matrix(np.random.randint(0, 2, (10, 15)).astype(np.float32))
        model = rusket.ALS(factors=4, iterations=3, seed=42, use_cuda=False)
        model.fit(mat)
        assert model.user_factors is not None


# ---------------------------------------------------------------------------
# CUDA codepath tests (mock backend)
# ---------------------------------------------------------------------------


class TestModelCUDACodepaths:
    """Test CUDA codepaths for models using mocked CuPy backend."""

    @staticmethod
    def _patch_cuda_backend():
        """Patch the CUDA backend detection to return mock CuPy."""
        return patch("rusket.cuda._get_cuda_backend", return_value=("cupy", _make_mock_cupy()))

    def test_als_cuda_recommend(self) -> None:
        disable_cuda()
        mat = csr_matrix(np.random.randint(0, 2, (10, 15)).astype(np.float32))
        model = rusket.ALS(factors=4, iterations=3, seed=42, use_cuda=True)
        model.fit(mat)
        with self._patch_cuda_backend():
            ids, scores = model.recommend_items(0, n=5)
            assert len(ids) == 5
            assert len(scores) == 5

    def test_bpr_cuda_recommend(self) -> None:
        disable_cuda()
        mat = csr_matrix(np.random.randint(0, 2, (10, 15)).astype(np.float32))
        model = rusket.BPR(factors=4, iterations=3, seed=42, use_cuda=True)
        model.fit(mat)
        with self._patch_cuda_backend():
            ids, scores = model.recommend_items(0, n=5)
            assert len(ids) == 5
            assert len(scores) == 5

    def test_svd_cuda_recommend(self) -> None:
        disable_cuda()
        rng = np.random.RandomState(42)
        mat = csr_matrix(rng.rand(10, 15).astype(np.float32))
        model = rusket.SVD(factors=4, iterations=3, seed=42, use_cuda=True)
        model.fit(mat)
        with self._patch_cuda_backend():
            ids, scores = model.recommend_items(0, n=5)
            assert len(ids) == 5
            assert len(scores) == 5

    def test_nmf_cuda_recommend(self) -> None:
        disable_cuda()
        mat = csr_matrix(np.random.rand(10, 15).astype(np.float32))
        model = rusket.NMF(factors=4, iterations=3, seed=42, use_cuda=True)
        model.fit(mat)
        with self._patch_cuda_backend():
            ids, scores = model.recommend_items(0, n=5)
            assert len(ids) == 5
            assert len(scores) == 5


# ---------------------------------------------------------------------------
# Vector export — auto-detect from client type
# ---------------------------------------------------------------------------


class TestExportVectors:
    def test_unsupported_client_raises_typeerror(self) -> None:
        from rusket.vector_export import export_vectors

        with pytest.raises(TypeError, match="Unsupported client type"):
            export_vectors(np.zeros((5, 3), dtype=np.float32), client="not_a_client")

    def test_qdrant_auto_detected(self) -> None:
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "qdrant_client.qdrant_client"
        mock_client.__class__.__qualname__ = "QdrantClient"

        mock_models = MagicMock()
        mock_models.Distance.DOT = "Dot"
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": MagicMock(),
                "qdrant_client.models": mock_models,
            },
        ):
            from rusket.vector_export import export_vectors

            n = export_vectors(np.random.randn(5, 3).astype(np.float32), client=mock_client)
            assert n == 5
            mock_client.upsert.assert_called_once()

    def test_meilisearch_auto_detected(self) -> None:
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "meilisearch.client"
        mock_client.__class__.__qualname__ = "Client"

        from rusket.vector_export import export_vectors

        n = export_vectors(np.random.randn(5, 3).astype(np.float32), client=mock_client)
        assert n == 5
        mock_client.index.assert_called_once()

    def test_pgvector_auto_detected(self) -> None:
        """A mock psycopg2 connection should be auto-detected as pgvector."""
        mock_conn = MagicMock()
        mock_conn.__class__.__module__ = "psycopg2.extensions"
        mock_conn.__class__.__qualname__ = "connection"
        mock_cursor = MagicMock()
        mock_cursor.mogrify = lambda fmt, vals: (
            fmt % tuple(f"'{v}'" if isinstance(v, str) else str(v) for v in vals)
        ).encode()
        mock_conn.cursor.return_value = mock_cursor

        from rusket.vector_export import export_vectors

        n = export_vectors(np.random.randn(5, 3).astype(np.float32), client=mock_conn)
        assert n == 5
        mock_conn.commit.assert_called()
        mock_cursor.execute.assert_called()

    def test_chromadb_auto_detected(self) -> None:
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "chromadb.api.client"
        mock_client.__class__.__qualname__ = "Client"

        from rusket.vector_export import export_vectors

        n = export_vectors(np.random.randn(5, 3).astype(np.float32), client=mock_client)
        assert n == 5
        mock_client.get_or_create_collection.assert_called_once()

    def test_dbapi_connection_detected_as_pgvector(self) -> None:
        """Any DB-API 2.0 connection (cursor+commit) should fall back to pgvector."""
        mock_conn = MagicMock()
        mock_conn.__class__.__module__ = "some_custom_driver"
        mock_conn.__class__.__qualname__ = "Connection"
        # DB-API 2.0 protocol
        mock_conn.cursor = MagicMock()
        mock_conn.commit = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = lambda fmt, vals: (
            fmt % tuple(f"'{v}'" if isinstance(v, str) else str(v) for v in vals)
        ).encode()
        mock_conn.cursor.return_value = mock_cursor

        from rusket.vector_export import export_vectors

        n = export_vectors(np.random.randn(3, 2).astype(np.float32), client=mock_conn)
        assert n == 3
