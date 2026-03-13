"""Backward-compatibility shim — canonical location: ``rusket.miners.lcm``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.miners.lcm")
_sys.modules[__name__] = _real
