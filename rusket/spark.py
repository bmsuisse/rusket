"""Backward-compatibility shim — canonical location: ``rusket.integrations.spark``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.integrations.spark")
_sys.modules[__name__] = _real
