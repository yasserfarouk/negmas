"""Backward-compatibility shim.

``ParetoSampler`` has been moved to ``negmas.preferences.protocols`` so that it
lives alongside ``InverseUFun``.  This module re-exports it from there to avoid
breaking any code that imports from the old location.
"""

from __future__ import annotations

from negmas.preferences.protocols import ParetoSampler

__all__ = ["ParetoSampler"]
