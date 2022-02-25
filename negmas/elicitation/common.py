from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..common import Value

np.seterr(all="raise")  # setting numpy to raise exceptions in case of errors


__all__ = ["_loc", "_locs", "_scale", "_upper", "_uppers", "argmax", "argmin", "argmin"]


def _loc(u: Value):
    """Returns the lower bound of a Value"""
    return u if isinstance(u, float) else u.loc


def _locs(us: Iterable[Value]):
    """Returns the lower bound of an iterable of Value(s)"""
    return [u if isinstance(u, float) else u.loc for u in us]


def _scale(u: Value):
    """Returns the difference between the upper and lower bounds of a Value"""
    return 0.0 if isinstance(u, float) else u.scale


def _upper(u: Value):
    """Returns the upper bound of a Value"""
    return u if isinstance(u, float) else (u.loc + u.scale)


def _uppers(us: Iterable[Value]):
    """Returns the upper bounds of an Iterble of Values"""
    return [u if isinstance(u, float) else (u.loc + u.scale) for u in us]


def argmax(iterable: Iterable[Any]):
    """Returns the index of the maximum"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def argsort(iterable: Iterable[Any]):
    """Returns a list of indices that would sort the iterable"""
    return [_[0] for _ in sorted(enumerate(iterable), key=lambda x: x[1])]


def argmin(iterable):
    """Returns the index of the minimum"""
    return min(enumerate(iterable), key=lambda x: x[1])[0]
