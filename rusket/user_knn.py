"""Backward-compatibility shim — canonical location: ``rusket.recommenders.user_knn``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.user_knn")
_sys.modules[__name__] = _real
