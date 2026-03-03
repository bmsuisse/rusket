"""Tests for global CUDA enable/disable configuration."""

import rusket
from rusket._config import (
    _auto_detect_cuda,
    _resolve_cuda,
    disable_cuda,
    enable_cuda,
    is_cuda_enabled,
)


def test_default_without_cuda_hardware():
    """CUDA may be auto-enabled on import if hardware is present; test the toggle API."""
    disable_cuda()  # reset
    assert is_cuda_enabled() is False


def test_enable_disable_toggle():
    """enable_cuda / disable_cuda toggles the global flag."""
    disable_cuda()
    assert is_cuda_enabled() is False

    enable_cuda()
    assert is_cuda_enabled() is True

    disable_cuda()
    assert is_cuda_enabled() is False


def test_resolve_cuda_explicit_true():
    """Explicit use_cuda=True always wins, even if global is off."""
    disable_cuda()
    assert _resolve_cuda(True) is True


def test_resolve_cuda_explicit_false():
    """Explicit use_cuda=False always wins, even if global is on."""
    enable_cuda()
    assert _resolve_cuda(False) is False
    disable_cuda()


def test_resolve_cuda_none_follows_global():
    """use_cuda=None falls back to the global setting."""
    disable_cuda()
    assert _resolve_cuda(None) is False

    enable_cuda()
    assert _resolve_cuda(None) is True

    disable_cuda()


def test_model_inherits_global_cuda():
    """Models created after enable_cuda() should have use_cuda=True."""
    disable_cuda()
    model_cpu = rusket.ALS(factors=8)
    assert model_cpu.use_cuda is False

    enable_cuda()
    model_gpu = rusket.ALS(factors=8)
    assert model_gpu.use_cuda is True

    disable_cuda()


def test_model_override_when_global_enabled():
    """Per-model use_cuda=False overrides the global enable_cuda()."""
    enable_cuda()
    model = rusket.ALS(factors=8, use_cuda=False)
    assert model.use_cuda is False
    disable_cuda()


def test_public_api_exports():
    """enable_cuda / disable_cuda / is_cuda_enabled are importable from rusket."""
    assert hasattr(rusket, "enable_cuda")
    assert hasattr(rusket, "disable_cuda")
    assert hasattr(rusket, "is_cuda_enabled")


# ── Backward compatibility ──────────────────────────────────────────────────


def test_backward_compat_enable_gpu():
    """Old enable_gpu / disable_gpu / is_gpu_enabled still work."""
    assert hasattr(rusket, "enable_gpu")
    assert hasattr(rusket, "disable_gpu")
    assert hasattr(rusket, "is_gpu_enabled")

    rusket.disable_gpu()
    assert rusket.is_gpu_enabled() is False

    rusket.enable_gpu()
    assert rusket.is_gpu_enabled() is True

    rusket.disable_gpu()


def test_backward_compat_use_gpu_kwarg():
    """Old use_gpu kwarg still accepted by models."""
    disable_cuda()
    model = rusket.ALS(factors=8, use_gpu=True)
    assert model.use_cuda is True
    disable_cuda()


def test_auto_detect_callable():
    """_auto_detect_cuda runs without error."""
    _auto_detect_cuda()
    disable_cuda()  # reset
