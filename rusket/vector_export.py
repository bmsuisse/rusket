"""Backward-compatibility shim — canonical location: ``rusket.export.vector_export``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.export.vector_export")
_sys.modules[__name__] = _real
