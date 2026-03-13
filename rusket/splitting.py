"""Backward-compatibility shim — canonical location: ``rusket.evaluation.splitting``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.evaluation.splitting")
_sys.modules[__name__] = _real
