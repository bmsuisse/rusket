"""Global rusket configuration — CUDA toggle and runtime settings."""

from __future__ import annotations

_CUDA_ENABLED: bool = False


def enable_cuda() -> None:
    """Enable CUDA acceleration globally for all rusket models.

    After calling this, every model created without an explicit ``use_cuda``
    argument will default to CUDA.  Individual models can still override
    with ``use_cuda=False``.

    Example
    -------
    >>> import rusket
    >>> rusket.enable_cuda()
    >>> model = rusket.ALS(factors=64)  # uses CUDA automatically
    """
    global _CUDA_ENABLED
    _CUDA_ENABLED = True


def disable_cuda() -> None:
    """Disable CUDA acceleration globally (the default)."""
    global _CUDA_ENABLED
    _CUDA_ENABLED = False


def is_cuda_enabled() -> bool:
    """Return ``True`` if CUDA acceleration is globally enabled."""
    return _CUDA_ENABLED


def _resolve_cuda(use_cuda: bool | None) -> bool:
    """Resolve per-model ``use_cuda`` against the global setting.

    Parameters
    ----------
    use_cuda : bool | None
        - ``True``/``False``: explicit per-model choice (always wins).
        - ``None``: fall back to the global ``enable_cuda()`` setting.
    """
    if use_cuda is not None:
        return use_cuda
    if _CUDA_AUTODETECT_PENDING:
        _run_pending_cuda_autodetect()
    return _CUDA_ENABLED


def _auto_detect_cuda() -> None:
    """Probe for CuPy or PyTorch CUDA and enable globally if found.

    Called once at ``import rusket`` time.  Safe to call multiple times.

    Only checks *presence* of cupy/torch via ``importlib.util.find_spec`` —
    it must NOT actually import either package, since importing torch alone
    can take 1-3s and would defeat the point of a fast ``import rusket``.
    The real (heavier) import + CUDA-availability probe is deferred to
    ``_resolve_cuda()``, which only runs it once, on first use.
    """
    import importlib.util

    if importlib.util.find_spec("cupy") is not None or importlib.util.find_spec("torch") is not None:
        # Defer the actual (possibly slow) import + device probe until a model
        # actually asks to resolve CUDA (see _resolve_cuda below).
        global _CUDA_AUTODETECT_PENDING
        _CUDA_AUTODETECT_PENDING = True


_CUDA_AUTODETECT_PENDING: bool = False


def _run_pending_cuda_autodetect() -> None:
    """Run the deferred cupy/torch import + device probe, once."""
    global _CUDA_AUTODETECT_PENDING
    if not _CUDA_AUTODETECT_PENDING:
        return
    _CUDA_AUTODETECT_PENDING = False

    try:
        import cupy  # type: ignore[import-untyped]

        if cupy.cuda.runtime.getDeviceCount() > 0:
            enable_cuda()
            return
    except Exception:
        pass

    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            enable_cuda()
            return
    except Exception:
        pass


# ── Backward compatibility aliases ──────────────────────────────────────────

enable_gpu = enable_cuda
disable_gpu = disable_cuda
is_gpu_enabled = is_cuda_enabled
_resolve_gpu = _resolve_cuda
