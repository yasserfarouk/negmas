#!/usr/bin/env python
"""
Extra helpers
"""
from __future__ import annotations

import itertools
import numbers
from typing import Any, Callable, Iterable

from ..protocols import HasMinMax

__all__ = [
    "monotonic_minmax",
    "monotonic_multi_minmax",
    "nonmonotonic_multi_minmax",
    "nonmonotonic_minmax",
]


def nonmonotonic_minmax(
    input: Iterable, f: Callable[[Any], numbers.Real]
) -> tuple[float, float]:
    """Finds the limits of a function `f` for the input assuming that it is non-monotonic and input is iterable"""
    mn, mx = float("inf"), float("-inf")
    fmn, fmx = float("-inf"), float("inf")
    for x in input:
        fx = float(f(x))
        if fx < mn:
            mn, fmn = x, fx
        if fx > mx:
            mx, fmx = x, fx
    return fmn, fmx


def nonmonotonic_multi_minmax(
    input: Iterable[Iterable], f: Callable[[Any], numbers.Real]
) -> tuple[float, float]:
    """Finds the limits of a function `f` for the each input assuming that it is non-monotonic and input is iterable"""
    return nonmonotonic_minmax(itertools.product(*input), f)


def monotonic_minmax(
    input: HasMinMax, f: Callable[[Any], numbers.Real]
) -> tuple[float, float]:
    """Finds the limits of a function `f` for the input assuming that it is monotonic and input has `min_value` and `max_value` members"""
    a, b = input.min_value, input.max_value
    fa, fb = f(a), f(b)
    if fb < fa:
        return float(fb), float(fa)
    return float(fa), float(fb)


def monotonic_multi_minmax(
    input: Iterable[HasMinMax], f: Callable[[Any], numbers.Real]
) -> tuple[float, float]:
    """Finds the limits of a function `f` for the input assuming that it is monotonic and each input has `min_value` and `max_value` members"""
    vals = [(i.min_value, i.max_value) for i in input]
    return nonmonotonic_minmax(itertools.product(*vals), f)
