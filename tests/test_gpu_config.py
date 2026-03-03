"""Tests for global GPU enable/disable configuration."""

import rusket
from rusket._config import _resolve_gpu, disable_gpu, enable_gpu, is_gpu_enabled


def test_default_is_disabled():
    """GPU should be off by default."""
    disable_gpu()  # reset
    assert is_gpu_enabled() is False


def test_enable_disable_toggle():
    """enable_gpu / disable_gpu toggles the global flag."""
    disable_gpu()
    assert is_gpu_enabled() is False

    enable_gpu()
    assert is_gpu_enabled() is True

    disable_gpu()
    assert is_gpu_enabled() is False


def test_resolve_gpu_explicit_true():
    """Explicit use_gpu=True always wins, even if global is off."""
    disable_gpu()
    assert _resolve_gpu(True) is True


def test_resolve_gpu_explicit_false():
    """Explicit use_gpu=False always wins, even if global is on."""
    enable_gpu()
    assert _resolve_gpu(False) is False
    disable_gpu()


def test_resolve_gpu_none_follows_global():
    """use_gpu=None falls back to the global setting."""
    disable_gpu()
    assert _resolve_gpu(None) is False

    enable_gpu()
    assert _resolve_gpu(None) is True

    disable_gpu()


def test_model_inherits_global_gpu():
    """Models created after enable_gpu() should have use_gpu=True."""
    disable_gpu()
    model_cpu = rusket.ALS(factors=8)
    assert model_cpu.use_gpu is False

    enable_gpu()
    model_gpu = rusket.ALS(factors=8)
    assert model_gpu.use_gpu is True

    disable_gpu()


def test_model_override_when_global_enabled():
    """Per-model use_gpu=False overrides the global enable_gpu()."""
    enable_gpu()
    model = rusket.ALS(factors=8, use_gpu=False)
    assert model.use_gpu is False
    disable_gpu()


def test_public_api_exports():
    """enable_gpu / disable_gpu / is_gpu_enabled are importable from rusket."""
    assert hasattr(rusket, "enable_gpu")
    assert hasattr(rusket, "disable_gpu")
    assert hasattr(rusket, "is_gpu_enabled")
