"""Backward-compatibility shim — canonical location: ``rusket.recommenders.ease``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.ease")
_sys.modules[__name__] = _real
