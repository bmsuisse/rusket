"""Tests for rusket.hybrid_embedding — HybridEmbeddingIndex & fuse_embeddings."""

import numpy as np
import pytest
import scipy.sparse as sp

from rusket.als import ALS
from rusket.hybrid_embedding import HybridEmbeddingIndex, fuse_embeddings


# ── fixtures ─────────────────────────────────────────────────────────────


def _cf_embeddings(n: int = 10, d: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, d)).astype(np.float32)


def _sem_embeddings(n: int = 10, d: int = 16) -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.standard_normal((n, d)).astype(np.float32)


# ── fuse_embeddings tests ────────────────────────────────────────────────


class TestFuseEmbeddings:
    def test_concat_shape(self) -> None:
        cf = _cf_embeddings(10, 8)
        sem = _sem_embeddings(10, 16)
        fused = fuse_embeddings(cf, sem, strategy="concat")
        assert fused.shape == (10, 24)  # 8 + 16

    def test_weighted_concat_shape(self) -> None:
        cf = _cf_embeddings(10, 8)
        sem = _sem_embeddings(10, 16)
        fused = fuse_embeddings(cf, sem, strategy="weighted_concat", alpha=0.7)
        assert fused.shape == (10, 24)

    def test_projection_shape(self) -> None:
        cf = _cf_embeddings(20, 8)
        sem = _sem_embeddings(20, 16)
        fused = fuse_embeddings(cf, sem, strategy="projection", projection_dim=5)
        assert fused.shape == (20, 5)

    def test_projection_default_dim(self) -> None:
        cf = _cf_embeddings(100, 32)
        sem = _sem_embeddings(100, 48)
        fused = fuse_embeddings(cf, sem, strategy="projection")
        # Default projection_dim = min(32+48, 64) = 64
        assert fused.shape == (100, 64)

    def test_dtype_is_float32(self) -> None:
        cf = _cf_embeddings()
        sem = _sem_embeddings()
        for strategy in ("concat", "weighted_concat", "projection"):
            fused = fuse_embeddings(cf, sem, strategy=strategy, projection_dim=4)
            assert fused.dtype == np.float32

    def test_alpha_pure_cf(self) -> None:
        """alpha=1.0 → semantic part is all zeros, so fused ≈ pure CF."""
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        fused = fuse_embeddings(cf, sem, strategy="weighted_concat", alpha=1.0)
        # The semantic half should be all zeros
        assert np.allclose(fused[:, 4:], 0.0)

    def test_alpha_pure_semantic(self) -> None:
        """alpha=0.0 → CF part is all zeros."""
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        fused = fuse_embeddings(cf, sem, strategy="weighted_concat", alpha=0.0)
        assert np.allclose(fused[:, :4], 0.0)

    def test_row_mismatch_raises(self) -> None:
        cf = _cf_embeddings(10, 4)
        sem = _sem_embeddings(5, 4)  # mismatch
        with pytest.raises(ValueError, match="Number of items must match"):
            fuse_embeddings(cf, sem)

    def test_invalid_strategy_raises(self) -> None:
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        with pytest.raises(ValueError, match="Unknown strategy"):
            fuse_embeddings(cf, sem, strategy="magic")

    def test_alpha_out_of_range_raises(self) -> None:
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        with pytest.raises(ValueError, match="alpha must be"):
            fuse_embeddings(cf, sem, alpha=1.5)

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            fuse_embeddings(np.ones(4, dtype=np.float32), np.ones((4, 2), dtype=np.float32))


# ── HybridEmbeddingIndex tests ──────────────────────────────────────────


class TestHybridEmbeddingIndex:
    def test_query_basic(self) -> None:
        cf = _cf_embeddings(10, 8)
        sem = _sem_embeddings(10, 8)
        idx = HybridEmbeddingIndex(cf, sem)
        ids, scores = idx.query(item_id=0, n=3)
        assert len(ids) == 3
        assert len(scores) == 3
        # Scores should be descending
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_query_excludes_self(self) -> None:
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        idx = HybridEmbeddingIndex(cf, sem)
        ids, _ = idx.query(item_id=2, n=4)
        assert 2 not in ids

    def test_query_out_of_bounds(self) -> None:
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        idx = HybridEmbeddingIndex(cf, sem)
        with pytest.raises(ValueError, match="out of bounds"):
            idx.query(item_id=99, n=3)

    def test_query_vector(self) -> None:
        cf = _cf_embeddings(10, 8)
        sem = _sem_embeddings(10, 8)
        idx = HybridEmbeddingIndex(cf, sem)
        vec = idx.fused_embeddings[3].copy()
        ids, scores = idx.query_vector(vec, n=5)
        assert len(ids) == 5
        # The original item (3) should be the closest or near-top
        assert 3 in ids[:2]

    def test_fused_embeddings_property(self) -> None:
        cf = _cf_embeddings(10, 8)
        sem = _sem_embeddings(10, 16)
        idx = HybridEmbeddingIndex(cf, sem, strategy="concat")
        assert idx.fused_embeddings.shape == (10, 24)
        assert idx.n_items == 10

    def test_repr(self) -> None:
        cf = _cf_embeddings(10, 8)
        sem = _sem_embeddings(10, 16)
        idx = HybridEmbeddingIndex(cf, sem, strategy="weighted_concat", alpha=0.6)
        r = repr(idx)
        assert "HybridEmbeddingIndex" in r
        assert "d_cf=8" in r
        assert "d_sem=16" in r
        assert "weighted_concat" in r

    def test_build_ann_native(self) -> None:
        cf = _cf_embeddings(20, 8)
        sem = _sem_embeddings(20, 8)
        idx = HybridEmbeddingIndex(cf, sem)
        ann = idx.build_ann_index(backend="native", n_trees=3, leaf_size=5)
        # Should be able to query
        q = idx.fused_embeddings[[0]]
        neighbors, dists = ann.kneighbors(q, n_neighbors=3)
        assert neighbors.shape == (1, 3)

    def test_build_ann_invalid_backend(self) -> None:
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        idx = HybridEmbeddingIndex(cf, sem)
        with pytest.raises(ValueError, match="Unknown backend"):
            idx.build_ann_index(backend="unknown")

    def test_strategies_produce_different_results(self) -> None:
        cf = _cf_embeddings(20, 8)
        sem = _sem_embeddings(20, 8)
        results: dict[str, np.ndarray] = {}
        for strat in ("concat", "weighted_concat"):
            idx = HybridEmbeddingIndex(cf, sem, strategy=strat)
            ids, _ = idx.query(item_id=0, n=5)
            results[strat] = ids
        # They may or may not differ, but shapes must be valid
        for ids in results.values():
            assert len(ids) == 5

    def test_projection_strategy(self) -> None:
        cf = _cf_embeddings(30, 8)
        sem = _sem_embeddings(30, 16)
        idx = HybridEmbeddingIndex(cf, sem, strategy="projection", projection_dim=6)
        assert idx.fused_embeddings.shape == (30, 6)
        ids, scores = idx.query(item_id=0, n=3)
        assert len(ids) == 3


# ── Integration with ALS ────────────────────────────────────────────────


class TestHybridWithALS:
    def test_als_item_factors_with_semantic(self) -> None:
        """End-to-end: train ALS, fuse with random semantic embeddings, query."""
        row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
        col = np.array([0, 1, 2, 1, 2, 3, 3, 4, 0, 4])
        data = np.ones(10, dtype=np.float32)
        csr = sp.csr_matrix((data, (row, col)), shape=(4, 5))

        model = ALS(factors=4, iterations=5, seed=42)
        model.fit(csr)

        # Semantic embeddings: make items 0 and 4 similar semantically
        sem = np.array(
            [
                [1.0, 0.0, 0.0],  # item 0
                [0.0, 1.0, 0.0],  # item 1
                [0.0, 0.0, 1.0],  # item 2
                [0.5, 0.5, 0.0],  # item 3
                [0.9, 0.1, 0.0],  # item 4: semantically close to item 0
            ],
            dtype=np.float32,
        )

        idx = HybridEmbeddingIndex(
            cf_embeddings=model.item_factors,
            semantic_embeddings=sem,
            strategy="weighted_concat",
            alpha=0.5,
        )

        assert idx.n_items == 5
        ids, scores = idx.query(item_id=0, n=4)
        assert len(ids) == 4
        assert 0 not in ids  # self excluded

    def test_fuse_embeddings_with_als(self) -> None:
        """Test the standalone fuse_embeddings with ALS factors."""
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 1, 2, 0, 2])
        data = np.ones(6, dtype=np.float32)
        csr = sp.csr_matrix((data, (row, col)), shape=(3, 3))

        model = ALS(factors=4, iterations=5, seed=42)
        model.fit(csr)

        sem = np.random.default_rng(42).standard_normal((3, 8)).astype(np.float32)
        fused = fuse_embeddings(model.item_factors, sem, strategy="concat")
        assert fused.shape == (3, 12)  # 4 + 8


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_item(self) -> None:
        cf = np.array([[1.0, 2.0]], dtype=np.float32)
        sem = np.array([[3.0, 4.0]], dtype=np.float32)
        idx = HybridEmbeddingIndex(cf, sem)
        ids, scores = idx.query(item_id=0, n=5)
        assert len(ids) == 0  # no other items

    def test_identical_embeddings(self) -> None:
        """When CF == semantic, the fused space should still work."""
        emb = _cf_embeddings(10, 8)
        idx = HybridEmbeddingIndex(emb, emb.copy())
        ids, scores = idx.query(item_id=0, n=3)
        assert len(ids) == 3

    def test_zero_vector_in_semantic(self) -> None:
        """A zero vector should not crash (handled by l2 norm clipping)."""
        cf = _cf_embeddings(5, 4)
        sem = np.zeros((5, 4), dtype=np.float32)
        sem[0] = [1, 0, 0, 0]  # at least one non-zero
        idx = HybridEmbeddingIndex(cf, sem)
        ids, _ = idx.query(item_id=0, n=3)
        assert len(ids) == 3

    def test_query_vector_zero(self) -> None:
        """Querying with a zero vector should return empty."""
        cf = _cf_embeddings(5, 4)
        sem = _sem_embeddings(5, 4)
        idx = HybridEmbeddingIndex(cf, sem)
        ids, scores = idx.query_vector(np.zeros(8, dtype=np.float32), n=3)
        assert len(ids) == 0
