import negmas.gb.negotiators.scs as _n
from negmas.gb.negotiators.scs import *

__all__ = [_ for _ in _n.__all__ if not _.startswith("GB")]
