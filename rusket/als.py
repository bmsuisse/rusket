"""Backward-compatibility shim — canonical location: ``rusket.recommenders.als``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.als")
_sys.modules[__name__] = _real
