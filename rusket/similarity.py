"""Backward-compatibility shim — canonical location: ``rusket._internal.similarity``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket._internal.similarity")
_sys.modules[__name__] = _real
