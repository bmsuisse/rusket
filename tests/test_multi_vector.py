"""Tests for multi-vector export and HybridEmbeddingIndex multi-mode."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rusket.hybrid_embedding import HybridEmbeddingIndex
from rusket.vector_export import export_multi_vectors

# ── helpers ──────────────────────────────────────────────────────────────


def _cf(n: int = 5, d: int = 4) -> np.ndarray:
    return np.random.default_rng(42).standard_normal((n, d)).astype(np.float32)


def _sem(n: int = 5, d: int = 6) -> np.ndarray:
    return np.random.default_rng(99).standard_normal((n, d)).astype(np.float32)


def _mock_qdrant_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "qdrant_client.qdrant_client"
    client.__class__.__qualname__ = "QdrantClient"
    return client


def _mock_meilisearch_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "meilisearch.client"
    client.__class__.__qualname__ = "Client"
    return client


def _mock_weaviate_v4_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "weaviate.client"
    client.__class__.__qualname__ = "WeaviateClient"
    client.collections = MagicMock()
    return client


def _qdrant_models_patch() -> dict[str, MagicMock]:
    mock_models = MagicMock()
    mock_models.Distance.DOT = "Dot"
    mock_models.Distance.COSINE = "Cosine"
    mock_models.Distance.EUCLID = "Euclid"
    return {
        "qdrant_client": MagicMock(),
        "qdrant_client.models": mock_models,
    }


# =====================================================================
# export_multi_vectors (functional API)
# =====================================================================


class TestExportMultiVectors:
    """Tests for the standalone export_multi_vectors function."""

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            export_multi_vectors({}, client=MagicMock())

    def test_row_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Row count mismatch"):
            export_multi_vectors(
                {"a": _cf(5, 4), "b": _cf(3, 4)},
                client=_mock_qdrant_client(),
            )

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            export_multi_vectors(
                {"a": np.ones(4, dtype=np.float32)},
                client=_mock_qdrant_client(),
            )

    def test_unsupported_backend_raises(self) -> None:
        mock = MagicMock()
        mock.__class__.__module__ = "chromadb.api.client"
        mock.__class__.__qualname__ = "Client"
        with pytest.raises(NotImplementedError, match="does not support multi-vector"):
            export_multi_vectors({"a": _cf()}, client=mock)

    def test_qdrant_multi_named_vectors(self) -> None:
        client = _mock_qdrant_client()
        with patch.dict("sys.modules", _qdrant_models_patch()):
            n = export_multi_vectors(
                {"cf": _cf(5, 4), "semantic": _sem(5, 6)},
                client=client,
                collection_name="test_col",
            )
        assert n == 5
        # recreate_collection should be called with dict vectors_config
        client.recreate_collection.assert_called_once()
        call_kwargs = client.recreate_collection.call_args
        vectors_config = call_kwargs.kwargs.get(
            "vectors_config",
            call_kwargs[1].get("vectors_config") if len(call_kwargs) > 1 else None,
        )
        assert isinstance(vectors_config, dict)
        assert "cf" in vectors_config
        assert "semantic" in vectors_config

    def test_meilisearch_multi_embedders(self) -> None:
        client = _mock_meilisearch_client()
        n = export_multi_vectors(
            {"cf": _cf(3, 4), "semantic": _sem(3, 6)},
            client=client,
            collection_name="items",
        )
        assert n == 3
        client.index.assert_called_once_with("items")
        # Verify documents contain multi-embedder _vectors
        mock_index = client.index.return_value
        docs = mock_index.add_documents.call_args[0][0]
        assert "_vectors" in docs[0]
        assert "cf" in docs[0]["_vectors"]
        assert "semantic" in docs[0]["_vectors"]

    def test_weaviate_multi_named_vectors(self) -> None:
        client = _mock_weaviate_v4_client()
        n = export_multi_vectors(
            {"cf": _cf(3, 4), "semantic": _sem(3, 6)},
            client=client,
            collection_name="items",
        )
        assert n == 3


# =====================================================================
# HybridEmbeddingIndex.export_vectors(mode="multi")
# =====================================================================


class TestHybridEmbeddingMultiExport:
    def test_named_embeddings_property(self) -> None:
        cf = _cf(5, 4)
        sem = _sem(5, 6)
        idx = HybridEmbeddingIndex(cf, sem)
        named = idx.named_embeddings
        assert set(named.keys()) == {"cf", "semantic"}
        assert named["cf"].shape == (5, 4)
        assert named["semantic"].shape == (5, 6)
        # Should be L2-normalised (row norms ≈ 1)
        norms = np.linalg.norm(named["cf"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_export_fused_mode_default(self) -> None:
        """Default mode='fused' calls export_vectors."""
        cf = _cf(5, 4)
        sem = _sem(5, 6)
        idx = HybridEmbeddingIndex(cf, sem)

        with patch("rusket.vector_export.export_vectors") as mock_exp:
            mock_exp.return_value = 5
            n = idx.export_vectors(MagicMock(), collection_name="test")
        assert n == 5
        mock_exp.assert_called_once()

    def test_export_multi_mode(self) -> None:
        """mode='multi' calls export_multi_vectors with named dict."""
        cf = _cf(5, 4)
        sem = _sem(5, 6)
        idx = HybridEmbeddingIndex(cf, sem)

        with patch("rusket.vector_export.export_multi_vectors") as mock_exp:
            mock_exp.return_value = 5
            n = idx.export_vectors(MagicMock(), mode="multi", collection_name="test")
        assert n == 5
        mock_exp.assert_called_once()
        call_args = mock_exp.call_args
        named = call_args[0][0]
        assert isinstance(named, dict)
        assert "cf" in named
        assert "semantic" in named

    def test_export_multi_qdrant_integration(self) -> None:
        """Full integration: HybridEmbeddingIndex → Qdrant multi-vector."""
        cf = _cf(5, 4)
        sem = _sem(5, 6)
        idx = HybridEmbeddingIndex(cf, sem)
        client = _mock_qdrant_client()

        with patch.dict("sys.modules", _qdrant_models_patch()):
            n = idx.export_vectors(client, mode="multi", collection_name="hybrid")
        assert n == 5
        client.recreate_collection.assert_called_once()
        # Verify named vectors in config
        call_kwargs = client.recreate_collection.call_args
        vectors_config = call_kwargs.kwargs.get(
            "vectors_config",
            call_kwargs[1].get("vectors_config") if len(call_kwargs) > 1 else None,
        )
        assert isinstance(vectors_config, dict)
        assert "cf" in vectors_config
        assert "semantic" in vectors_config
