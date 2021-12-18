"""Defines value functions for single issues"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, reduce
from math import cos, e, exp, log, pow, sin
from operator import add
from typing import Any, Callable, Iterable

from negmas.helpers.misc import (
    monotonic_minmax,
    monotonic_multi_minmax,
    nonmonotonic_minmax,
    nonmonotonic_multi_minmax,
)
from negmas.outcomes.base_issue import Issue

from .protocols import MultiIssueFun, SingleIssueFun

MAX_CARINALITY = 10_000

__all__ = [
    "ConstFun",
    "IdentityFun",
    "AffineFun",
    "LinearFun",
    "TriangularFun",
    "LambdaFun",
    "PolynomialFun",
    "QuadraticFun",
    "ExponentialFun",
    "LogFun",
    "SinFun",
    "CosFun",
    "TableFun",
    "TableMultiFun",
    "AffineMultiFun",
    "LinearMultiFun",
    "LambdaMultiFun",
]


@dataclass
class TableFun(SingleIssueFun):
    d: dict

    def __call__(self, x):
        return self.d[x]

    @lru_cache
    def minmax(self, input: Issue) -> tuple[float, float]:
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> TableFun:
        d = dict()
        for k in self.d.keys():
            d[k] = self.d[k] + offset
        return TableFun(d)

    def scale_by(self, scale: float) -> TableFun:
        d = dict()
        for k in self.d.keys():
            d[k] = self.d[k] * scale
        return TableFun(d)


@dataclass
class AffineFun(SingleIssueFun):
    slope: float
    bias: float = 0

    def __call__(self, x: float):
        return x * self.slope + self.bias

    @lru_cache
    def minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        return AffineFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineFun:
        return AffineFun(slope=self.slope * scale, bias=self.bias * scale)


@dataclass
class IdentityFun(SingleIssueFun):
    def __call__(self, x: float):
        return x

    def minmax(self, input: Issue) -> tuple[float, float]:
        return (input.min_value, input.max_value)

    def shift_by(self, offset: float) -> ConstFun:
        return ConstFun(bias=offset)

    def scale_by(self, scale: float) -> LinearFun:
        return LinearFun(slope=scale)


@dataclass
class ConstFun(SingleIssueFun):
    bias: float

    def __call__(self, x: float):
        return self.bias

    def minmax(self, input: Issue) -> tuple[float, float]:
        return (self.bias, self.bias)

    def shift_by(self, offset: float) -> ConstFun:
        return ConstFun(bias=offset + self.bias)

    def scale_by(self, scale: float) -> AffineFun:
        return AffineFun(slope=scale, bias=self.bias)


@dataclass
class LinearFun(SingleIssueFun):
    slope: float

    @property
    def bias(sef):
        return 0.0

    def __call__(self, x: float):
        return x * self.slope

    @lru_cache
    def minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        return AffineFun(bias=offset, slope=self.slope)

    def scale_by(self, scale: float) -> LinearFun:
        return LinearFun(slope=scale * self.slope)


@dataclass
class LambdaFun(SingleIssueFun):
    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

    def __call__(self, x: Any) -> float:
        return self.f(x) + self.bias

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        if self.min_value is not None and self.max_value is not None:
            return self.min_value, self.max_value
        mn, mx = nonmonotonic_minmax(input, self.f)
        if self.min_value is not None:
            mn = min(mn, self.min_value)
        if self.max_value is not None:
            mx = min(mx, self.max_value)
        return mn, mx

    def shift_by(self, offset: float, change_bias: bool = False) -> LambdaFun:
        mn, mx = self.min_value, self.max_value
        return LambdaFun(
            self.f if change_bias else lambda x: offset + self.f(x),
            bias=(self.bias + offset) if change_bias else self.bias,
            min_value=mn if mn is None else mn + offset,
            max_value=mx if mx is None else mx + offset,
        )

    def scale_by(self, scale: float) -> LambdaFun:
        mn, mx = self.min_value, self.max_value
        if scale < 0:
            mn, mx = mx, mn
        return LambdaFun(
            lambda x: scale * self.f(x),
            bias=self.bias * scale,
            min_value=mn if mn is None else mn * scale,
            max_value=mx if mx is None else mx * scale,
        )


@dataclass
class QuadraticFun(SingleIssueFun):
    a2: float
    a1: float
    bias: float = 0

    def __call__(self, x: float):
        return self.a2 * x * x + self.a1 * x + self.bias

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        mn, mx = input.min_value, input.max_value
        middle = -self.a1 / (2 * self.a2)
        fmn, fmx = self(mn), self(mx)
        if middle < mn or middle > mx:
            if fmn < fmx:
                return fmn, fmx
            return fmx, fmn
        if fmn > fmx:
            fmn = fmx
        fmiddle = self(middle)  # type: ignore
        if fmn < fmiddle:
            return fmn, fmiddle
        return fmiddle, fmn

    def shift_by(self, offset: float) -> QuadraticFun:
        return QuadraticFun(bias=self.bias + offset, a1=self.a1, a2=self.a2)

    def scale_by(self, scale: float) -> QuadraticFun:
        return QuadraticFun(
            bias=self.bias * scale, a1=self.a1 * scale, a2=self.a2 * scale
        )


@dataclass
class PolynomialFun(SingleIssueFun):
    a: tuple[float]
    bias: float = 0

    def __call__(self, x: float):
        return reduce(add, [b * pow(x, p + 1) for p, b in enumerate(self.a)], self.bias)

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> PolynomialFun:
        return PolynomialFun(bias=self.bias + offset, a=self.a)

    def scale_by(self, scale: float) -> PolynomialFun:
        return PolynomialFun(bias=self.bias * scale, a=[_ * scale for _ in self.a])


@dataclass
class TriangularFun(SingleIssueFun):
    start: float
    middle: float
    end: float
    bias: float = 0

    def __call__(self, x: float):
        bias1, slope1 = self.start, (self.middle - self.start)
        bias2, slope2 = self.middle, (self.middle - self.end)
        return self.bias + (
            bias1 + slope1 * float(x) if x < self.middle else bias2 + slope2 * float(x)
        )

    def shift_by(self, offset: float) -> TriangularFun:
        return TriangularFun(
            bias=self.bias + offset, start=self.start, middle=self.middle, end=self.end
        )

    def scale_by(self, scale: float) -> TriangularFun:
        return TriangularFun(
            bias=self.bias * scale,
            start=self.start * scale,
            middle=self.middle * scale,
            end=self.end * scale,
        )

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)


@dataclass
class ExponentialFun(SingleIssueFun):
    tau: float
    bias: float = 0
    base: float = e

    def __call__(self, x: float):
        return pow(self.base, self.tau * x) + self.bias

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> ExponentialFun:
        return ExponentialFun(bias=self.bias + offset, tau=self.tau)

    def scale_by(self, scale: float) -> ExponentialFun:
        return ExponentialFun(
            bias=self.bias * scale, tau=self.tau + math.log(scale, base=self.base)
        )


@dataclass
class LogFun(SingleIssueFun):
    tau: float
    bias: float = 0
    base: float = e

    def __call__(self, x: float):
        return log(self.tau * x, base=self.base) + self.bias

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> LogFun:
        return LogFun(bias=self.bias + offset, tau=self.tau)

    def scale_by(self, scale: float) -> LambdaFun:
        return LambdaFun(
            lambda x: log(self.tau * x, base=self.base) * scale, bias=self.bias * scale
        )


@dataclass
class TableMultiFun(MultiIssueFun):
    d: dict

    def __call__(self, x):
        return self.d[x]

    @lru_cache
    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return nonmonotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> TableMultiFun:
        d = dict()
        for k in self.d.keys():
            d[k] = self.d[k] + offset
        return TableMultiFun(d)

    def scale_by(self, scale: float) -> TableMultiFun:
        d = dict()
        for k in self.d.keys():
            d[k] = self.d[k] * scale
        return TableMultiFun(d)


@dataclass
class AffineMultiFun(MultiIssueFun):
    slope: tuple[float, ...]
    bias: float = 0

    def __call__(self, x: tuple):
        return reduce(add, [a * b for a, b in zip(self.slope, x)], self.bias)

    @lru_cache
    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineMultiFun:
        return AffineMultiFun(
            slope=tuple(scale * _ for _ in self.slope), bias=self.bias * scale
        )


@dataclass
class LinearMultiFun(MultiIssueFun):
    slope: tuple[float, ...]

    @property
    def bias(self):
        return 0

    def __call__(self, x: tuple):
        return reduce(add, [a * b for a, b in zip(self.slope, x)], 0)

    @lru_cache
    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> LinearMultiFun:
        return LinearMultiFun(slope=tuple(scale * _ for _ in self.slope))


@dataclass
class LambdaMultiFun(MultiIssueFun):
    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

    def __call__(self, x: Any) -> float:
        return self.f(x) + self.bias

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        if self.min_value is not None and self.max_value is not None:
            return self.min_value, self.max_value
        mn, mx = nonmonotonic_multi_minmax(input, self.f)
        if self.min_value is not None:
            mn = min(mn, self.min_value)
        if self.max_value is not None:
            mx = min(mx, self.max_value)
        return mn, mx
