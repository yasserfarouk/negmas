# ruff: noqa: F403
"""Re-exports tit-for-tat negotiators from the GB module for SAO compatibility."""

import negmas.gb.negotiators.titfortat as _n
from negmas.gb.negotiators.titfortat import *

__all__ = [_ for _ in _n.__all__ if not _.startswith("GB")]
