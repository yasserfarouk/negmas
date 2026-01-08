# ruff: noqa: F403
"""Module for utilities functionality."""

from __future__ import annotations

from negmas import warnings


from .preferences import *

warnings.deprecated("Module `utilities` is deprecated. Use `preferences` instead")
