"""Backward-compatibility shim — canonical location: ``rusket._internal._type_utils``."""
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket._internal._type_utils")
_sys.modules[__name__] = _real
