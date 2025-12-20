"""Defines value functions for single issues"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import lru_cache, reduce
from math import cos, e, log, pow, sin
from operator import add
from typing import Any, Callable, Iterable

from attrs import asdict, define

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

MAX_CARINALITY = 10_000_000_000

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
    "make_fun_from_xml",
]


class BaseFun(ABC):
    @property
    def dim(self) -> int:
        """Dim.

        Returns:
            int: The result.
        """
        return 1

    @abstractmethod
    def minmax(self, input: Issue) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        ...

    @abstractmethod
    def __call__(self, x) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        ...

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """From dict.

        Args:
            d: D.
            python_class_identifier: Python class identifier.
        """
        if isinstance(d, cls):
            return d
        _ = d.pop(python_class_identifier, None)
        return cls(**deserialize(d, python_class_identifier=python_class_identifier))  # type: ignore

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """To dict.

        Args:
            python_class_identifier: Python class identifier.

        Returns:
            dict[str, Any]: The result.
        """
        return serialize(asdict(self), python_class_identifier=python_class_identifier)

    def min(self, input: Issue) -> float:
        """Min.

        Args:
            input: Input.

        Returns:
            float: The result.
        """
        mn, _ = self.minmax(input)
        return mn

    def max(self, input: Issue) -> float:
        """Max.

        Args:
            input: Input.

        Returns:
            float: The result.
        """
        _, mx = self.minmax(input)
        return mx

    def to_table(self, issue: Issue) -> "TableFun":
        """Converts the function to  a table"""
        if not issue.is_finite():
            issue = issue.to_discrete()
        vals = list(issue.all)
        return TableFun(mapping=dict(zip(vals, [self(_) for _ in vals])))


@define(frozen=True)
class TableFun(BaseFun):
    """TableFun implementation."""

    mapping: dict

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    def _minmax(self, input: Issue) -> tuple[float, float]:
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> TableFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            TableFun: The result.
        """
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] + offset
        return TableFun(d)

    def scale_by(self, scale: float) -> TableFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            TableFun: The result.
        """
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] * scale
        return TableFun(d)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        dtype = "discrete"
        vtype = (
            "integer"
            if issue.is_integer()
            else "real"
            if issue.is_float()
            else "discrete"
        )
        output = f'<issue index="{indx + 1}" etype="{dtype}" type="{dtype}" vtype="{vtype}" name="{issue_name}">\n'
        vals = issue.all
        for i, issue_value in enumerate(vals):
            uu = self(issue_value) + bias
            output += f'    <item index="{i + 1}" value="{issue_value}" evaluation="{uu}" />\n'
        output += "</issue>\n"
        return output

    def __call__(self, x) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        return self.mapping[x]


@define(frozen=True)
class AffineFun(BaseFun):
    """AffineFun implementation."""

    slope: float
    bias: float = 0

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            AffineFun: The result.
        """
        return AffineFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            AffineFun: The result.
        """
        return AffineFun(slope=self.slope * scale, bias=self.bias * scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="linear" parameter0="{bias + self.bias}" parameter1="{self.slope}"></evaluator>\n'
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

    def __call__(self, x: float) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        return x * self.slope + self.bias


@define(frozen=True)
class ConstFun(BaseFun):
    """ConstFun implementation."""

    bias: float

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        _ = input
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Issue) -> tuple[float, float]:
        _ = input
        return (self.bias, self.bias)

    def shift_by(self, offset: float) -> ConstFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            ConstFun: The result.
        """
        return ConstFun(bias=offset + self.bias)

    def scale_by(self, scale: float) -> AffineFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            AffineFun: The result.
        """
        return AffineFun(slope=scale, bias=self.bias)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        return AffineFun(0.0, self.bias).xml(indx, issue, bias)

    def __call__(self, x: float) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        _ = x
        return self.bias


@define(frozen=True)
class LinearFun(BaseFun):
    """LinearFun implementation."""

    slope: float

    @property
    def bias(sef):
        """Bias.

        Args:
            sef: Sef.
        """
        return 0.0

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            AffineFun: The result.
        """
        return AffineFun(bias=offset, slope=self.slope)

    def scale_by(self, scale: float) -> LinearFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            LinearFun: The result.
        """
        return LinearFun(slope=scale * self.slope)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        return AffineFun(self.slope, 0.0).xml(indx, issue, bias)

    def __call__(self, x: float) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        return x * self.slope


@define(frozen=True)
class IdentityFun(BaseFun):
    """IdentityFun implementation."""

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return (input.min_value, input.max_value)

    def shift_by(self, offset: float) -> ConstFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            ConstFun: The result.
        """
        return ConstFun(bias=offset)

    def scale_by(self, scale: float) -> LinearFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            LinearFun: The result.
        """
        return LinearFun(slope=scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        return LinearFun(1.0).xml(indx, issue, bias)

    def __call__(self, x: float) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        return x


@define(frozen=True)
class LambdaFun(BaseFun):
    """LambdaFun implementation."""

    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self):
        # we need to be sure that f is a lambda function so that it can
        # correctly be serialized
        """post init  ."""
        if not is_lambda_function(self.f):
            f = self.f
            object.__setattr__(self, "f", lambda x: f(x))

    def minmax(self, input) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        if self.min_value is not None and self.max_value is not None:
            return self.min_value, self.max_value
        mn, mx = nonmonotonic_minmax(input, self.f)
        if self.min_value is not None:
            mn = min(mn, self.min_value)
        if self.max_value is not None:
            mx = min(mx, self.max_value)
        return mn, mx

    def shift_by(self, offset: float, change_bias: bool = False) -> LambdaFun:
        """Shift by.

        Args:
            offset: Offset.
            change_bias: Change bias.

        Returns:
            LambdaFun: The result.
        """
        mn, mx = self.min_value, self.max_value
        return LambdaFun(
            self.f if change_bias else lambda x: offset + self.f(x),
            bias=(self.bias + offset) if change_bias else self.bias,
            min_value=mn if mn is None else mn + offset,
            max_value=mx if mx is None else mx + offset,
        )

    def scale_by(self, scale: float) -> LambdaFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            LambdaFun: The result.
        """
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
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        if issue.is_discrete():
            values = list(issue.all)
            return TableFun(dict(zip(values, [self(_) for _ in values]))).xml(
                indx, issue, bias
            )
        raise ValueError("LambdaFun with a continuous issue cannot be converted to XML")

    def __call__(self, x: Any) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        return self.f(x) + self.bias


@define(frozen=True)
class QuadraticFun(BaseFun):
    """QuadraticFun implementation."""

    a2: float
    a1: float
    bias: float = 0

    def minmax(self, input) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        mn, mx = input.min_value, input.max_value
        middle = -self.a1 / (2 * self.a2)
        fmn, fmx = self(mn), self(mx)
        if middle < mn or middle > mx:
            if fmn < fmx:
                return fmn, fmx
            return fmx, fmn
        if fmn > fmx:
            fmn = fmx
        fmiddle = self(middle)
        if fmn < fmiddle:
            return fmn, fmiddle
        return fmiddle, fmn

    def shift_by(self, offset: float) -> QuadraticFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            QuadraticFun: The result.
        """
        return QuadraticFun(bias=self.bias + offset, a1=self.a1, a2=self.a2)

    def scale_by(self, scale: float) -> QuadraticFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            QuadraticFun: The result.
        """
        return QuadraticFun(
            bias=self.bias * scale, a1=self.a1 * scale, a2=self.a2 * scale
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="quadratic" parameter0="{bias + self.bias}" parameter1="{self.a1} parameter2={self.a2}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="quadratic" parameter0="{bias + self.bias}" parameter1="{self.a1} parameter2={self.a2}"></evaluator>\n'
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        """Make instance callable.

        Args:
            x: X.
        """
        return self.a2 * x * x + self.a1 * x + self.bias


@define(frozen=True)
class PolynomialFun(BaseFun):
    """PolynomialFun implementation."""

    coefficients: tuple[float, ...]
    bias: float = 0

    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> PolynomialFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            PolynomialFun: The result.
        """
        return PolynomialFun(bias=self.bias + offset, coefficients=self.coefficients)

    def scale_by(self, scale: float) -> PolynomialFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            PolynomialFun: The result.
        """
        return PolynomialFun(
            bias=self.bias * scale,
            coefficients=tuple(_ * scale for _ in self.coefficients),
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += (
                f'    <evaluator ftype="poynomial" parameter0="{bias + self.bias}"'
            )
            for i, x in enumerate(self.coefficients):
                output += f'parameter{i}="{x}"'

            output += "></evaluator>\n"
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += (
                f'    <evaluator ftype="poynomial" parameter0="{bias + self.bias}"'
            )
            for i, x in enumerate(self.coefficients):
                output += f'parameter{i}="{x}"'

            output += "></evaluator>\n"
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        """Make instance callable.

        Args:
            x: X.
        """
        return reduce(
            add, [b * pow(x, p + 1) for p, b in enumerate(self.coefficients)], self.bias
        )


@define(frozen=True)
class TriangularFun(BaseFun):
    """Triangular value function.

    Returns bias at x <= start, rises linearly to bias + 1 at middle,
    then falls linearly back to bias at x >= end.

    Args:
        start: x value where function starts rising from bias
        middle: x value where function reaches maximum (bias + 1)
        end: x value where function returns to bias
        bias: offset added to all values (default 0)
    """

    start: float
    middle: float
    end: float
    bias: float = 0
    scale: float = 1

    def shift_by(self, offset: float) -> "TriangularFun":
        return TriangularFun(
            bias=self.bias + offset,
            start=self.start,
            middle=self.middle,
            end=self.end,
            scale=self.scale,
        )

    def scale_by(self, scale: float) -> "TriangularFun":
        # Scale the output amplitude, not the x-coordinates
        return TriangularFun(
            bias=self.bias * scale,
            start=self.start,
            middle=self.middle,
            end=self.end,
            scale=self.scale * scale,
        )

    def minmax(self, input) -> tuple[float, float]:
        # Triangular function ranges from bias (at edges) to bias + 1 (at middle)
        if hasattr(input, "min_value") and hasattr(input, "max_value"):
            input_min = input.min_value
            input_max = input.max_value
            # Check if middle is within input range
            if input_min <= self.middle <= input_max:
                max_val = self.bias + self.scale
            else:
                max_val = max(self(input_min), self(input_max))
            min_val = min(self(input_min), self(input_max))
            return (min_val, max_val)
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        return nonmonotonic_minmax(input, self)

    def xml(self, indx: int, issue: Issue, bias) -> str:
        issue_name = issue.name
        if issue.is_continuous():
            assert abs(bias + self.bias) < 1e-6
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="triangular" parameter0="{self.start}" parameter1="{self.end}" parameter2="{self.middle}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            assert abs(bias + self.bias) < 1e-6
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="triangular" parameter0="{self.start}" parameter1="{self.end}" parameter2="{self.middle}"></evaluator>\n'
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        x = float(x)
        if x <= self.start:
            return self.bias
        if x >= self.end:
            return self.bias
        if x <= self.middle:
            # Linear interpolation from 0 at start to 1 at middle
            if self.middle == self.start:
                return self.bias + self.scale
            return self.bias + self.scale * (x - self.start) / (
                self.middle - self.start
            )
        # Linear interpolation from 1 at middle to 0 at end
        if self.end == self.middle:
            return self.bias + self.scale
        return self.bias + self.scale * (self.end - x) / (self.end - self.middle)


@define(frozen=True)
class ExponentialFun(BaseFun):
    """ExponentialFun implementation."""

    tau: float
    bias: float = 0
    base: float = e

    def minmax(self, input) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> ExponentialFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            ExponentialFun: The result.
        """
        return ExponentialFun(bias=self.bias + offset, tau=self.tau)

    def scale_by(self, scale: float) -> ExponentialFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            ExponentialFun: The result.
        """
        return ExponentialFun(
            bias=self.bias * scale, tau=self.tau + math.log(scale, base=self.base)
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="exponential" parameter0="{bias + self.bias}" parameter1="{self.tau} parameter2={self.base}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="exponential" parameter0="{bias + self.bias}" parameter1="{self.tau} parameter2={self.base}"></evaluator>\n'
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        """Make instance callable.

        Args:
            x: X.
        """
        return pow(self.base, self.tau * x) + self.bias


@define(frozen=True)
class CosFun(BaseFun):
    """CosFun implementation."""

    multiplier: float = 1.0
    bias: float = 0.0
    phase: float = 0.0
    amplitude: float = 1.0

    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> CosFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            CosFun: The result.
        """
        return CosFun(
            bias=self.bias + offset,
            multiplier=self.multiplier,
            phase=self.phase,
            amplitude=self.amplitude,
        )

    def scale_by(self, scale: float) -> CosFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            CosFun: The result.
        """
        return CosFun(
            amplitude=self.amplitude * scale,
            bias=self.bias * scale,
            multiplier=self.multiplier,
            phase=self.phase,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="cos" parameter0="{bias + self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="cos" parameter0="{bias + self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        """Make instance callable.

        Args:
            x: X.
        """
        return self.amplitude * (cos(self.multiplier * x + self.phase)) + self.bias


@define(frozen=True)
class SinFun(BaseFun):
    """SinFun implementation."""

    multiplier: float = 1.0
    bias: float = 0.0
    phase: float = 0.0
    amplitude: float = 1.0

    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> SinFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            SinFun: The result.
        """
        return SinFun(
            bias=self.bias + offset,
            multiplier=self.multiplier,
            phase=self.phase,
            amplitude=self.amplitude,
        )

    def scale_by(self, scale: float) -> SinFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            SinFun: The result.
        """
        return SinFun(
            amplitude=self.amplitude * scale,
            bias=self.bias * scale,
            multiplier=self.multiplier,
            phase=self.phase,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="sin" parameter0="{bias + self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="sin" parameter0="{bias + self.bias}" parameter1="{self.amplitude} parameter2={self.multiplier} parameter3={self.phase}"></evaluator>\n'
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        """Make instance callable.

        Args:
            x: X.
        """
        return self.amplitude * (sin(self.multiplier * x + self.phase)) + self.bias


@define(frozen=True)
class LogFun(BaseFun):
    """LogFun implementation."""

    tau: float
    bias: float = 0
    base: float = e
    scale: float = 1.0

    def minmax(self, input) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> LogFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            LogFun: The result.
        """
        return LogFun(
            bias=self.bias + offset, tau=self.tau, scale=self.scale, base=self.base
        )

    def scale_by(self, scale: float) -> LogFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            LogFun: The result.
        """
        return LogFun(
            bias=self.bias * scale,
            tau=self.tau,
            scale=self.scale * scale,
            base=self.base,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        issue_name = issue.name
        if issue.is_continuous():
            output = f'<issue index="{indx + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
            output += f'    <evaluator ftype="log" parameter0="{bias + self.bias}" parameter1="{self.tau} parameter2={self.base} paramter3={self.scale}"></evaluator>\n'
        elif isinstance(issue, ContiguousIssue):
            output = f'<issue index="{indx + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            output += f'    <evaluator ftype="log" parameter0="{bias + self.bias}" parameter1="{self.tau} parameter2={self.base} parameter3={self.scale}"></evaluator>\n'
        else:
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        output += "</issue>\n"
        return output

    def __call__(self, x: float):
        """Make instance callable.

        Args:
            x: X.
        """
        return self.scale * log(self.tau * x, base=self.base) + self.bias


@define(frozen=True)
class TableMultiFun(MultiIssueFun):
    """TableMultiFun implementation."""

    mapping: dict[tuple, Any]

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return nonmonotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> TableMultiFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            TableMultiFun: The result.
        """
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] + offset
        return TableMultiFun(d)

    def scale_by(self, scale: float) -> TableMultiFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            TableMultiFun: The result.
        """
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] * scale
        return TableMultiFun(d)

    @property
    def dim(self) -> int:
        """Dim.

        Returns:
            int: The result.
        """
        if not len(self.mapping):
            raise ValueError("Unkonwn dictionary in TableMultiFun")
        return len(list(self.mapping.keys())[0])

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issues: Issues.
            bias: Bias.

        Returns:
            str: The result.
        """
        raise NotImplementedError()

    def __call__(self, x):
        """Make instance callable.

        Args:
            x: X.
        """
        return self.mapping[x]


@define(frozen=True)
class AffineMultiFun(MultiIssueFun):
    """AffineMultiFun implementation."""

    slope: tuple[float, ...]
    bias: float = 0

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            AffineMultiFun: The result.
        """
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineMultiFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            AffineMultiFun: The result.
        """
        return AffineMultiFun(
            slope=tuple(scale * _ for _ in self.slope), bias=self.bias * scale
        )

    @property
    def dim(self) -> int:
        """Dim.

        Returns:
            int: The result.
        """
        raise NotImplementedError()

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issues: Issues.
            bias: Bias.

        Returns:
            str: The result.
        """
        raise NotImplementedError()

    def __call__(self, x: tuple):
        """Make instance callable.

        Args:
            x: X.
        """
        return reduce(add, [a * b for a, b in zip(self.slope, x)], self.bias)


@define(frozen=True)
class LinearMultiFun(MultiIssueFun):
    """LinearMultiFun implementation."""

    slope: tuple[float, ...]

    @property
    def bias(self):
        """Bias."""
        return 0

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            AffineMultiFun: The result.
        """
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> LinearMultiFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            LinearMultiFun: The result.
        """
        return LinearMultiFun(slope=tuple(scale * _ for _ in self.slope))

    @property
    def dim(self) -> int:
        """Dim.

        Returns:
            int: The result.
        """
        raise NotImplementedError()

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issues: Issues.
            bias: Bias.

        Returns:
            str: The result.
        """
        raise NotImplementedError()

    def __call__(self, x: tuple):
        """Make instance callable.

        Args:
            x: X.
        """
        return reduce(add, [a * b for a, b in zip(self.slope, x)], 0)


@define(frozen=True)
class LambdaMultiFun(MultiIssueFun):
    """LambdaMultiFun implementation."""

    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        if self.min_value is not None and self.max_value is not None:
            return self.min_value, self.max_value
        mn, mx = nonmonotonic_multi_minmax(input, self.f)
        if self.min_value is not None:
            mn = min(mn, self.min_value)
        if self.max_value is not None:
            mx = min(mx, self.max_value)
        return mn, mx

    def shift_by(self, offset: float) -> AffineMultiFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            AffineMultiFun: The result.
        """
        raise NotImplementedError()

    def scale_by(self, scale: float) -> LinearMultiFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            LinearMultiFun: The result.
        """
        raise NotImplementedError()

    @property
    def dim(self) -> int:
        """Dim.

        Returns:
            int: The result.
        """
        raise NotImplementedError()

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issues: Issues.
            bias: Bias.

        Returns:
            str: The result.
        """
        raise NotImplementedError()

    def __call__(self, x: Any) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        return self.f(x) + self.bias


def make_fun_from_xml(item) -> tuple[BaseFun, str]:
    """Make fun from xml.

    Args:
        item: Item.

    Returns:
        tuple[BaseFun, str]: The result.
    """
    if item.attrib["ftype"] == "linear":
        offset = item.attrib.get(
            "offset", item.attrib.get("parameter0", item.attrib.get("offset", 0.0))
        )
        slope = item.attrib.get(
            "slope", item.attrib.get("parameter1", item.attrib.get("slope", 1.0))
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
        raise ValueError(f"Unknown ftype {item.attrib['ftype']}")
