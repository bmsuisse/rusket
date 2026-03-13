"""Backward-compatibility shim — canonical location: ``rusket.miners.streaming``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.miners.streaming")
_sys.modules[__name__] = _real
