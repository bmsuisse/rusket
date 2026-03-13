"""Backward-compatibility shim — canonical location: ``rusket._internal._embedding_mixin``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket._internal._embedding_mixin")
_sys.modules[__name__] = _real
