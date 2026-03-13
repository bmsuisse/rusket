"""Backward-compatibility shim — canonical location: ``rusket._internal.analytics``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket._internal.analytics")
_sys.modules[__name__] = _real
