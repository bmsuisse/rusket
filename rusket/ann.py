"""Backward-compatibility shim — canonical location: ``rusket.export.ann``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.export.ann")
_sys.modules[__name__] = _real
