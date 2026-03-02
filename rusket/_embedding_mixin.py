"""Shared embedding visualization mixin for recommender models."""

from __future__ import annotations

from typing import Any


class EmbeddingMixin:
    """Mixin providing PCA and PaCMAP dimensionality reduction for item embeddings.

    Any class mixing this in must provide:
    - ``item_factors`` property returning a (n_items, factors) ndarray
    - ``_item_labels`` attribute (optional list of item labels)
    """

    def pca(self, n_components: int = 3, normalize: bool = True) -> Any:
        """Reduces the item embeddings to `n_components` dimensions using PCA.

        This enables a fluent visualization API:
        ```python
        model.fit().pca().plot()
        ```

        Parameters
        ----------
        n_components : int, default=3
            Number of principal components to keep.
        normalize : bool, default=True
            Whether to L2-normalize the item factors before PCA computation.
            Normalizing factors often creates a better visualization for cosine distance.

        Returns
        -------
        ProjectedSpace
            A wrapper object containing the projected coordinates, with a ``.plot()`` method.
        """
        import numpy as np

        from .pca import ProjectedSpace, pca

        factors = self.item_factors  # type: ignore[attr-defined]
        if normalize:
            norms = np.linalg.norm(factors, axis=1, keepdims=True)
            factors = factors / np.clip(norms, a_min=1e-10, a_max=None)

        coords: np.ndarray = pca(factors, n_components=n_components)
        return ProjectedSpace(coords, self._item_labels)  # type: ignore[attr-defined]

    def pacmap(self, n_components: int = 2, normalize: bool = True, **kwargs: Any) -> Any:
        """Reduces the item embeddings to `n_components` dimensions using PaCMAP.

        PaCMAP provides superior preservation of both local and global structure
        compared to PCA, making it ideal for visualizing latent item clusters.

        This enables a fluent visualization API:
        ```python
        model.fit().pacmap(n_components=2).plot()
        ```

        Parameters
        ----------
        n_components : int, default=2
            Number of dimensions to embed into.
        normalize : bool, default=True
            Whether to L2-normalize the item factors before PaCMAP computation.
            Normalizing factors often creates a better visualization for cosine distance.
        **kwargs : Any
            Additional arguments passed to ``rusket.pacmap()`` (e.g., ``n_neighbors``, ``lr``).

        Returns
        -------
        ProjectedSpace
            A wrapper object containing the projected coordinates, with a ``.plot()`` method.
        """
        import numpy as np

        from .pacmap import pacmap
        from .pca import ProjectedSpace

        factors = self.item_factors  # type: ignore[attr-defined]
        if normalize:
            norms = np.linalg.norm(factors, axis=1, keepdims=True)
            factors = factors / np.clip(norms, a_min=1e-10, a_max=None)

        coords: np.ndarray = pacmap(factors, n_components=n_components, **kwargs)
        return ProjectedSpace(coords, self._item_labels)  # type: ignore[attr-defined]

    def pacmap2(self, normalize: bool = True, **kwargs: Any) -> Any:
        """Shorthand for ``pacmap(n_components=2)``."""
        return self.pacmap(n_components=2, normalize=normalize, **kwargs)

    def pacmap3(self, normalize: bool = True, **kwargs: Any) -> Any:
        """Shorthand for ``pacmap(n_components=3)``."""
        return self.pacmap(n_components=3, normalize=normalize, **kwargs)
