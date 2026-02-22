"""pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys

# Ensure tests/ dir is on path so test_fpbase imports work
sys.path.insert(0, os.path.dirname(__file__))
