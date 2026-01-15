# ruff: noqa: F403
"""Re-exports limited outcomes negotiators from the GB module for SAO compatibility."""

import negmas.gb.negotiators.limited as _n
from negmas.gb.negotiators.limited import *

__all__ = [_ for _ in _n.__all__ if not _.startswith("GB")]
