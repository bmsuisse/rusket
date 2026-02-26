"""Tests for rusket.PCA — validates Rust-backed PCA against scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest

import rusket


# ── Fixtures ───────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
X_SMALL = RNG.standard_normal((100, 20)).astype(np.float32)
X_TALL = RNG.standard_normal((500, 5)).astype(np.float32)
X_WIDE = RNG.standard_normal((10, 50)).astype(np.float32)


# ── Shape tests ────────────────────────────────────────────────────────────


def test_fit_shapes() -> None:
    pca = rusket.PCA(n_components=5)
    pca.fit(X_SMALL)
    assert pca.components_.shape == (5, 20)
    assert pca.explained_variance_.shape == (5,)
    assert pca.explained_variance_ratio_.shape == (5,)
    assert pca.singular_values_.shape == (5,)
    assert pca.mean_.shape == (20,)
    assert pca.n_components_ == 5


def test_transform_shape() -> None:
    pca = rusket.PCA(n_components=3)
    pca.fit(X_SMALL)
    Xt = pca.transform(X_SMALL)
    assert Xt.shape == (100, 3)


def test_fit_transform_shape() -> None:
    pca = rusket.PCA(n_components=3)
    Xt = pca.fit_transform(X_SMALL)
    assert Xt.shape == (100, 3)


# ── Variance tests ─────────────────────────────────────────────────────────


def test_explained_variance_ratio_sums_to_one_full() -> None:
    """When keeping all components, ratios should sum to ~1.0."""
    n_components = min(X_SMALL.shape)
    pca = rusket.PCA(n_components=n_components)
    pca.fit(X_SMALL)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-4)


def test_explained_variance_descending() -> None:
    pca = rusket.PCA(n_components=10)
    pca.fit(X_SMALL)
    ev = pca.explained_variance_
    for i in range(len(ev) - 1):
        assert ev[i] >= ev[i + 1], f"explained_variance not descending at {i}"


def test_singular_values_descending() -> None:
    pca = rusket.PCA(n_components=10)
    pca.fit(X_SMALL)
    sv = pca.singular_values_
    for i in range(len(sv) - 1):
        assert sv[i] >= sv[i + 1], f"singular_values not descending at {i}"


# ── Reconstruction test ───────────────────────────────────────────────────


def test_reconstruction_close() -> None:
    """X_transformed @ components + mean ≈ X when all components kept."""
    n = min(X_SMALL.shape)
    pca = rusket.PCA(n_components=n)
    Xt = pca.fit_transform(X_SMALL)
    X_recon = Xt @ pca.components_ + pca.mean_
    np.testing.assert_allclose(X_recon, X_SMALL, atol=1e-3)


# ── Orthogonality test ────────────────────────────────────────────────────


def test_components_orthogonal() -> None:
    pca = rusket.PCA(n_components=5)
    pca.fit(X_SMALL)
    gram = pca.components_ @ pca.components_.T
    np.testing.assert_allclose(gram, np.eye(5), atol=1e-4)


# ── Convenience functions ─────────────────────────────────────────────────


def test_pca2_shape() -> None:
    result = rusket.pca2(X_SMALL)
    assert result.shape == (100, 2)


def test_pca3_shape() -> None:
    result = rusket.pca3(X_SMALL)
    assert result.shape == (100, 3)


def test_pca_function() -> None:
    result = rusket.pca(X_SMALL, n_components=7)
    assert result.shape == (100, 7)


# ── Edge cases ─────────────────────────────────────────────────────────────


def test_single_component() -> None:
    result = rusket.pca(X_SMALL, n_components=1)
    assert result.shape == (100, 1)


def test_n_components_gt_features_clipped() -> None:
    """n_components > n_features should be clipped gracefully."""
    pca = rusket.PCA(n_components=100)
    pca.fit(X_WIDE)
    # Clipped to min(n_samples, n_features) = min(10, 50) = 10
    assert pca.n_components_ == 10
    assert pca.components_.shape == (10, 50)


def test_tall_matrix() -> None:
    pca = rusket.PCA(n_components=3)
    Xt = pca.fit_transform(X_TALL)
    assert Xt.shape == (500, 3)


def test_wide_matrix() -> None:
    pca = rusket.PCA(n_components=3)
    Xt = pca.fit_transform(X_WIDE)
    assert Xt.shape == (10, 3)


# ── Determinism ────────────────────────────────────────────────────────────


def test_deterministic() -> None:
    pca1 = rusket.PCA(n_components=3).fit(X_SMALL)
    pca2_model = rusket.PCA(n_components=3).fit(X_SMALL)
    np.testing.assert_array_equal(pca1.components_, pca2_model.components_)
    np.testing.assert_array_equal(pca1.explained_variance_, pca2_model.explained_variance_)


# ── Not fitted ─────────────────────────────────────────────────────────────


def test_not_fitted_raises() -> None:
    pca = rusket.PCA(n_components=3)
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = pca.components_


# ── sklearn comparison ─────────────────────────────────────────────────────


def test_matches_sklearn() -> None:
    """Compare explained variance ratios against scikit-learn PCA."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA as SkPCA

    n_comp = 5
    rpca = rusket.PCA(n_components=n_comp)
    rpca.fit(X_SMALL)

    skpca = SkPCA(n_components=n_comp)
    skpca.fit(X_SMALL)

    # Explained variance ratios should match closely
    np.testing.assert_allclose(
        rpca.explained_variance_ratio_,
        skpca.explained_variance_ratio_.astype(np.float32),
        atol=1e-4,
    )

    # Components should match in absolute value (sign may flip)
    np.testing.assert_allclose(
        np.abs(rpca.components_),
        np.abs(skpca.components_.astype(np.float32)),
        atol=1e-3,
    )

    # Singular values should match
    np.testing.assert_allclose(
        rpca.singular_values_,
        skpca.singular_values_.astype(np.float32),
        atol=0.05,
    )


# ── Plotting tests ─────────────────────────────────────────────────────────


def test_plot_pca_2d() -> None:
    """plot_pca should return a plotly Figure for 2D data."""
    plotly = pytest.importorskip("plotly")
    import plotly.graph_objects as go

    coords = rusket.pca2(X_SMALL)
    fig = rusket.viz.plot_pca(coords, title="Test 2D")
    assert isinstance(fig, go.Figure)


def test_plot_pca_3d() -> None:
    """plot_pca should return a plotly Figure for 3D data."""
    plotly = pytest.importorskip("plotly")
    import plotly.graph_objects as go

    coords = rusket.pca3(X_SMALL)
    fig = rusket.viz.plot_pca(coords, labels=[f"item_{i}" for i in range(100)])
    assert isinstance(fig, go.Figure)


def test_plot_pca_invalid_dims() -> None:
    """plot_pca should raise ValueError for data with wrong number of columns."""
    pytest.importorskip("plotly")
    data_4d = rusket.pca(X_SMALL, n_components=4)
    with pytest.raises(ValueError, match="2 or 3 columns"):
        rusket.viz.plot_pca(data_4d)


# ── 2D input (degenerate) ──────────────────────────────────────────────────


def test_pca_2d_input() -> None:
    """PCA on a 2D dataset should work fine."""
    X = RNG.standard_normal((50, 2)).astype(np.float32)
    pca = rusket.PCA(n_components=2)
    Xt = pca.fit_transform(X)
    assert Xt.shape == (50, 2)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-4)
