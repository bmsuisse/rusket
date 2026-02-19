"""pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
import os

# Ensure tests/ dir is on path so test_fpbase imports work
sys.path.insert(0, os.path.dirname(__file__))
