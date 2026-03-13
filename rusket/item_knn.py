"""Backward-compatibility shim — canonical location: ``rusket.recommenders.item_knn``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.item_knn")
_sys.modules[__name__] = _real
