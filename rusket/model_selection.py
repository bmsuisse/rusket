"""Backward-compatibility shim — canonical location: ``rusket.evaluation.model_selection``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.evaluation.model_selection")
_sys.modules[__name__] = _real
