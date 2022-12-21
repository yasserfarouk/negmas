"""Defines value functions for single issues"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import lru_cache, reduce
from math import cos, e, log, pow, sin
from operator import add
from typing import Any, Callable, Iterable

from attr import asdict, define

from negmas.helpers.misc import (
    monotonic_minmax,
    monotonic_multi_minmax,
    nonmonotonic_minmax,
    nonmonotonic_multi_minmax,
)
from negmas.helpers.types import is_lambda_function
from negmas.outcomes.base_issue import Issue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .protocols import MultiIssueFun

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


class BaseFun(ABC):
    @property
    def dim(self) -> int:
        return 1

    @abstractmethod
    def minmax(self, input: Issue) -> tuple[float, float]:
        ...

    @classmethod
    def from_dict(cls, d: dict):
        if isinstance(d, cls):
            return d
        _ = d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(**deserialize(d))  # type: ignore Concrete classes will have constructor params

    def to_dict(self) -> dict[str, Any]:
        return serialize(asdict(self))  # type: ignore

    def min(self, input: Issue) -> float:
        mn, _ = self.minmax(input)
        return mn

    def max(self, input: Issue) -> float:
        _, mx = self.minmax(input)
        return mx


@define
class TableFun(BaseFun):
    d: dict

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

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        issue_name = issue.name
        dtype = "discrete"
        vtype = (
            "integer"
            if issue.is_integer()
            else "real"
            if issue.is_float()
            else "discrete"
        )
        output = f'<issue index="{indx+1}" etype="{dtype}" type="{dtype}" vtype="{vtype}" name="{issue_name}">\n'
        vals = issue.all  # type: ignore
        for i, issue_value in enumerate(vals):
            uu = self(issue_value) + bias
            output += (
                f'    <item index="{i+1}" value="{issue_value}" evaluation="{uu}" />\n'
            )
        output += "</issue>\n"
        return output

    def __call__(self, x):
        return self.d[x]


@define
class AffineFun(BaseFun):
    slope: float
    bias: float = 0

    @lru_cache
    def minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        return AffineFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineFun:
        return AffineFun(slope=self.slope * scale, bias=self.bias * scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="linear" parameter0="{bias+self.bias}" parameter1="{self.slope}"></evaluator>\n'
        # elif isinstance(issue, ContiguousIssue) and issue.cardinality > 50_000:
        #     output = f'<issue index="{indx + 1}" etype="real" type="integer" vtype="integer" name="{issue_name}">\n'
        #     output += f'    <evaluator ftype="linear" parameter0="{bias+self.bias}" parameter1="{self.slope}"></evaluator>\n'
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        return x * self.slope + self.bias


@define
class ConstFun(BaseFun):
    bias: float

    @lru_cache
    def minmax(self, input: Issue) -> tuple[float, float]:
        return (self.bias, self.bias)

    def shift_by(self, offset: float) -> ConstFun:
        return ConstFun(bias=offset + self.bias)

    def scale_by(self, scale: float) -> AffineFun:
        return AffineFun(slope=scale, bias=self.bias)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        return AffineFun(0.0, self.bias).xml(indx, issue, bias)

    def __call__(self, x: float):
        return self.bias


@define
class LinearFun(BaseFun):
    slope: float

    @property
    def bias(sef):
        return 0.0

    @lru_cache
    def minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        return AffineFun(bias=offset, slope=self.slope)

    def scale_by(self, scale: float) -> LinearFun:
        return LinearFun(slope=scale * self.slope)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        return AffineFun(self.slope, 0.0).xml(indx, issue, bias)

    def __call__(self, x: float):
        return x * self.slope


@define
class IdentityFun(BaseFun):
    def minmax(self, input: Issue) -> tuple[float, float]:
        return (input.min_value, input.max_value)

    def shift_by(self, offset: float) -> ConstFun:
        return ConstFun(bias=offset)

    def scale_by(self, scale: float) -> LinearFun:
        return LinearFun(slope=scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        return LinearFun(1.0).xml(indx, issue, bias)

    def __call__(self, x: float):
        return x


@define
class LambdaFun(BaseFun):
    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self):
        # we need to be sure that f is a lambda function so that it can
        # correctly be serialized
        if not is_lambda_function(self.f):
            f = self.f
            self.f = lambda x: f(x)

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

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        if issue.is_discrete():
            values = list(issue.all)  # type: ignore (I know it is discrete)
            return TableFun(dict(zip(values, [self(_) for _ in values]))).xml(
                indx, issue, bias
            )
        raise ValueError(
            f"LambdaFun with a continuous issue cannot be converted to XML"
        )

    def __call__(self, x: Any) -> float:
        return self.f(x) + self.bias


@define
class QuadraticFun(BaseFun):
    a2: float
    a1: float
    bias: float = 0

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

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="quadratic" parameter0="{bias+self.bias}" parameter1="{self.a1} parameter2={self.a2}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="quadratic" parameter0="{bias+self.bias}" parameter1="{self.a1} parameter2={self.a2}"></evaluator>\n'
        else:
            vals = list(issue.all)  # type: ignore
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        return self.a2 * x * x + self.a1 * x + self.bias


@define
class PolynomialFun(BaseFun):
    a: tuple[float]
    bias: float = 0

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> PolynomialFun:
        return PolynomialFun(bias=self.bias + offset, a=self.a)

    def scale_by(self, scale: float) -> PolynomialFun:
        return PolynomialFun(bias=self.bias * scale, a=tuple(_ * scale for _ in self.a))

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="poynomial" parameter0="{bias+self.bias}"'
            for i, x in enumerate(self.a):
                output += f'parameter{i}="{x}"'

            output += "></evaluator>\n"
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="poynomial" parameter0="{bias+self.bias}"'
            for i, x in enumerate(self.a):
                output += f'parameter{i}="{x}"'

            output += "></evaluator>\n"
        else:
            vals = list(issue.all)  # type: ignore
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        return reduce(add, [b * pow(x, p + 1) for p, b in enumerate(self.a)], self.bias)


@define
class TriangularFun(BaseFun):
    start: float
    middle: float
    end: float
    bias: float = 0

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

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            assert abs(bias + self.bias) < 1e-6
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="triangular" parameter0="{self.start}" parameter1="{self.end} parameter2={self.middle}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            assert abs(bias + self.bias) < 1e-6
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="triangular" parameter0="{self.start}" parameter1="{self.end} parameter2={self.middle}"></evaluator>\n'
        else:
            vals = list(issue.all)  # type: ignore
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        bias1, slope1 = self.start, (self.middle - self.start)
        bias2, slope2 = self.middle, (self.middle - self.end)
        return self.bias + (
            bias1 + slope1 * float(x) if x < self.middle else bias2 + slope2 * float(x)
        )


@define
class ExponentialFun(BaseFun):
    tau: float
    bias: float = 0
    base: float = e

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> ExponentialFun:
        return ExponentialFun(bias=self.bias + offset, tau=self.tau)

    def scale_by(self, scale: float) -> ExponentialFun:
        return ExponentialFun(
            bias=self.bias * scale, tau=self.tau + math.log(scale, base=self.base)
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="exponential" parameter0="{bias+self.bias}" parameter1="{self.tau} parameter2={self.base}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="exponential" parameter0="{bias+self.bias}" parameter1="{self.tau} parameter2={self.base}"></evaluator>\n'
        else:
            vals = list(issue.all)  # type: ignore
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        return pow(self.base, self.tau * x) + self.bias


@define
class CosFun(BaseFun):
    multiplier: float = 1.0
    bias: float = 0.0
    phase: float = 0.0
    amplitude: float = 1.0

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> CosFun:
        return CosFun(
            bias=self.bias + offset,
            multiplier=self.multiplier,
            phase=self.phase,
            amplitude=self.amplitude,
        )

    def scale_by(self, scale: float) -> CosFun:
        return CosFun(
            amplitude=self.amplitude * scale,
            bias=self.bias * scale,
            multiplier=self.multiplier,
            phase=self.phase,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="cos" parameter0="{bias+self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="cos" parameter0="{bias+self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        else:
            vals = list(issue.all)  # type: ignore
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        return self.amplitude * (cos(self.multiplier * x + self.phase)) + self.bias


@define
class SinFun(BaseFun):
    multiplier: float = 1.0
    bias: float = 0.0
    phase: float = 0.0
    amplitude: float = 1.0

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> SinFun:
        return SinFun(
            bias=self.bias + offset,
            multiplier=self.multiplier,
            phase=self.phase,
            amplitude=self.amplitude,
        )

    def scale_by(self, scale: float) -> SinFun:
        return SinFun(
            amplitude=self.amplitude * scale,
            bias=self.bias * scale,
            multiplier=self.multiplier,
            phase=self.phase,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="sin" parameter0="{bias+self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="sin" parameter0="{bias+self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        else:
            vals = list(issue.all)  # type: ignore
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        return self.amplitude * (sin(self.multiplier * x + self.phase)) + self.bias


@define
class LogFun(BaseFun):
    tau: float
    bias: float = 0
    base: float = e
    scale: float = 1.0

    @lru_cache
    def minmax(self, input) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> LogFun:
        return LogFun(
            bias=self.bias + offset, tau=self.tau, scale=self.scale, base=self.base
        )

    def scale_by(self, scale: float) -> LogFun:
        return LogFun(
            bias=self.bias * scale,
            tau=self.tau,
            scale=self.scale * scale,
            base=self.base,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="log" parameter0="{bias+self.bias}" parameter1="{self.tau} parameter2={self.base} paramter3={self.scale}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="log" parameter0="{bias+self.bias}" parameter1="{self.tau} parameter2={self.base} parameter3={self.scale}"></evaluator>\n'
        else:
            vals = list(issue.all)  # type: ignore
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        return self.scale * log(self.tau * x, base=self.base) + self.bias


@define
class TableMultiFun(MultiIssueFun):
    d: dict[tuple, Any]

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

    def dim(self) -> int:
        if not len(self.d):
            raise ValueError("Unkonwn dictionary in TableMultiFun")
        return len(list(self.d.keys())[0])

    def xml(self, indx: int, issues: list[Issue], bias=0) -> str:
        raise NotImplementedError()

    def __call__(self, x):
        return self.d[x]


@define
class AffineMultiFun(MultiIssueFun):
    slope: tuple[float, ...]
    bias: float = 0

    @lru_cache
    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineMultiFun:
        return AffineMultiFun(
            slope=tuple(scale * _ for _ in self.slope), bias=self.bias * scale
        )

    def dim(self) -> int:
        raise NotImplementedError()

    def xml(self, indx: int, issues: list[Issue], bias=0) -> str:
        raise NotImplementedError()

    def __call__(self, x: tuple):
        return reduce(add, [a * b for a, b in zip(self.slope, x)], self.bias)


@define
class LinearMultiFun(MultiIssueFun):
    slope: tuple[float, ...]

    @property
    def bias(self):
        return 0

    @lru_cache
    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> LinearMultiFun:
        return LinearMultiFun(slope=tuple(scale * _ for _ in self.slope))

    def dim(self) -> int:
        raise NotImplementedError()

    def xml(self, indx: int, issues: list[Issue], bias=0) -> str:
        raise NotImplementedError()

    def __call__(self, x: tuple):
        return reduce(add, [a * b for a, b in zip(self.slope, x)], 0)


@define
class LambdaMultiFun(MultiIssueFun):
    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

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

    def shift_by(self, offset: float) -> AffineMultiFun:
        raise NotImplementedError()

    def scale_by(self, scale: float) -> LinearMultiFun:
        raise NotImplementedError()

    def dim(self) -> int:
        raise NotImplementedError()

    def xml(self, indx: int, issues: list[Issue], bias=0) -> str:
        raise NotImplementedError()

    def __call__(self, x: Any) -> float:
        return self.f(x) + self.bias


def make_fun_from_xml(item) -> tuple[BaseFun, str]:

    if item.attrib["ftype"] == "linear":
        offset = item.attrib.get(
            "offset",
            item.attrib.get("parameter0", item.attrib.get("offset", 0.0)),
        )
        slope = item.attrib.get(
            "slope",
            item.attrib.get("parameter1", item.attrib.get("slope", 1.0)),
        )
        offset, slope = float(offset), float(slope)
        if offset != 0:
            return AffineFun(bias=offset, slope=slope), "affine"
        else:
            return LinearFun(slope=slope), "linear"
    elif item.attrib["ftype"] == "quadratic":
        offset = float(item.attrib.get("parameter0", item.attrib.get("offset", 0.0)))
        a1 = float(item.attrib.get("parameter1", item.attrib.get("a1", 1.0)))
        a2 = float(item.attrib.get("parameter2", item.attrib.get("a2", 1.0)))
        return QuadraticFun(a1=a1, a2=a2, bias=offset), "quadratic"
    elif item.attrib["ftype"] == "triangular":
        strt = float(item.attrib.get("parameter0", item.attrib.get("start", 0.0)))
        end = float(item.attrib.get("parameter1", item.attrib.get("end", 1.0)))
        middle = float(item.attrib.get("parameter2", item.attrib.get("middle", 1.0)))
        return TriangularFun(start=strt, end=end, middle=middle), "triangular"
    else:
        # todo: implement all other functions defined in value_fun.py
        raise ValueError(f'Unknown ftype {item.attrib["ftype"]}')
