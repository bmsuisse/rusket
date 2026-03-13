"""Backward-compatibility shim — canonical location: ``rusket.export.hybrid_embedding``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.export.hybrid_embedding")
_sys.modules[__name__] = _real
