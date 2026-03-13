"""Backward-compatibility shim — canonical location: ``rusket.recommenders.bpr``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.bpr")
_sys.modules[__name__] = _real
