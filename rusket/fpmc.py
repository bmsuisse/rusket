"""Backward-compatibility shim — canonical location: ``rusket.sequential.fpmc``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.sequential.fpmc")
_sys.modules[__name__] = _real
