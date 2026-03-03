"""Global rusket configuration — GPU toggle and runtime settings."""

from __future__ import annotations

_GPU_ENABLED: bool = False


def enable_gpu() -> None:
    """Enable GPU acceleration globally for all rusket models.

    After calling this, every model created without an explicit ``use_gpu``
    argument will default to GPU.  Individual models can still override
    with ``use_gpu=False``.

    Example
    -------
    >>> import rusket
    >>> rusket.enable_gpu()
    >>> model = rusket.ALS(factors=64)  # uses GPU automatically
    """
    global _GPU_ENABLED
    _GPU_ENABLED = True


def disable_gpu() -> None:
    """Disable GPU acceleration globally (the default)."""
    global _GPU_ENABLED
    _GPU_ENABLED = False


def is_gpu_enabled() -> bool:
    """Return ``True`` if GPU acceleration is globally enabled."""
    return _GPU_ENABLED


def _resolve_gpu(use_gpu: bool | None) -> bool:
    """Resolve per-model ``use_gpu`` against the global setting.

    Parameters
    ----------
    use_gpu : bool | None
        - ``True``/``False``: explicit per-model choice (always wins).
        - ``None``: fall back to the global ``enable_gpu()`` setting.
    """
    if use_gpu is not None:
        return use_gpu
    return _GPU_ENABLED
