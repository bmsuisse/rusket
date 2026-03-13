"""Backward-compatibility shim — canonical location: ``rusket.miners.association_rules``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.miners.association_rules")
_sys.modules[__name__] = _real
