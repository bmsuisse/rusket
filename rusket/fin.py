"""Backward-compatibility shim — canonical location: ``rusket.miners.fin``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.miners.fin")
_sys.modules[__name__] = _real
