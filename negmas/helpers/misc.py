#!/usr/bin/env python
"""
Extra helpers
"""
from __future__ import annotations

import itertools
import random
import socket
from math import exp, log
from typing import Any, Callable, Iterable, Sequence

from ..protocols import HasMinMax

__all__ = [
    "monotonic_minmax",
    "monotonic_multi_minmax",
    "nonmonotonic_multi_minmax",
    "nonmonotonic_minmax",
    "make_callable",
    "get_free_tcp_port",
    "intin",
    "floatin",
]


def floatin(
    x: float | tuple[float, float] | Sequence[float], log_uniform: bool
) -> float:
    """Samples a value. A random choice is made if x is another sequence, a value between the two limits is used if the input is a tuple otherwise x is returned"""
    if isinstance(x, tuple) and len(x) == 2:
        x = tuple(float(_) for _ in x)
        if x[0] == x[-1]:
            return x[0]
        if log_uniform:
            l = [log(_) for _ in x]
            return min(x[1], max(x[0], exp(random.random() * (l[1] - l[0]) + l[0])))

        return random.random() * (x[1] - x[0]) + x[0]
    if isinstance(x, Sequence):
        return float(random.choice(x))
    return float(x)


def intin(x: int | tuple[int, int] | Sequence[int], log_uniform: bool = False) -> int:
    """Samples a value. A random choice is made if x is another sequence, a value between the two limits is used if the input is a tuple otherwise x is returned"""
    if isinstance(x, tuple) and len(x) == 2:
        x = tuple(int(_) for _ in x)
        if x[0] == x[-1]:
            return int(x[0])
        if log_uniform:
            l = [log(_) for _ in x]
            return min(
                x[1], max(x[0], int(0.5 + exp(random.random() * (l[1] - l[0]) + l[0])))
            )

        return random.randint(*x)
    if isinstance(x, Sequence):
        return int(random.choice(x))
    return int(x)


def nonmonotonic_minmax(
    input: Iterable, f: Callable[[Any], float]
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
    input: Iterable[Iterable], f: Callable[[Any], float]
) -> tuple[float, float]:
    """Finds the limits of a function `f` for each input assuming that it is non-monotonic and input is iterable"""
    return nonmonotonic_minmax(itertools.product(*input), f)


def monotonic_minmax(
    input: HasMinMax, f: Callable[[Any], float]
) -> tuple[float, float]:
    """Finds the limits of a function `f` for the input assuming that it is monotonic and input has `min_value` and `max_value` members"""
    a, b = input.min_value, input.max_value
    fa, fb = f(a), f(b)
    if fb < fa:
        return float(fb), float(fa)
    return float(fa), float(fb)


def monotonic_multi_minmax(
    input: Iterable[HasMinMax], f: Callable[[Any], float]
) -> tuple[float, float]:
    """Finds the limits of a function `f` for the input assuming that it is monotonic and each input has `min_value` and `max_value` members"""
    vals = [(i.min_value, i.max_value) for i in input]
    return nonmonotonic_minmax(itertools.product(*vals), f)


def remove_qoutes(s: str) -> str:
    return s.replace('"', "~`")


def recover_qoutes(s: str) -> str:
    return s.replace("~`", '"')


def make_callable(x: dict | Sequence | Callable | None) -> Callable:
    """
    Converts its input to a callable (i.e. can be called using () operator).


    Examples:

        >>> make_callable(lambda  x: 2 * x) (4)
        8

        >>> make_callable(dict(a=1, b=3))("a")
        1

        >>> make_callable((3, 4, 5))(1)
        4
    """
    if x is None:
        return lambda a: a
    if isinstance(x, Callable):
        return x
    return lambda a: x[a]


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port
