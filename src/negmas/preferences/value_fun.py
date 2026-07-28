"""Value functions for mapping issue values to real numbers.

This module provides value functions used in utility function construction.
Value functions map values from one or more issues to real numbers, serving
as building blocks for utility functions like `LinearAdditiveUtilityFunction`
and `GLAUtilityFunction`.

Classes are organized into two hierarchies:

Single-Issue Functions (inherit from `BaseFun`):
    These map a single issue value to a real number.

    - `ConstFun`: Constant function (always returns same value)
    - `IdentityFun`: Identity function (returns input unchanged)
    - `LinearFun`: Linear function f(x) = slope * x
    - `AffineFun`: Affine function f(x) = slope * x + bias
    - `QuadraticFun`: Quadratic function f(x) = a2*x^2 + a1*x + bias
    - `PolynomialFun`: General polynomial function
    - `TriangularFun`: Triangular/tent function (rises then falls)
    - `TrapezoidalFun`: Trapezoidal function (triangular with a flat top)
    - `GaussianFun`: Gaussian/bell-curve bump function
    - `AggregatingFun`: Weighted sum of other `BaseFun` instances, with
      optional normalization to [0, 1]
    - `BiasedFun`: Wraps any `BaseFun`, adding a constant bias (with optional
      normalization to [0, 1])
    - `MultiModalTrapezoidalFun`: Weighted mixture of several `TrapezoidalFun`
    - `MultiModalGaussianFun`: Weighted mixture of several `GaussianFun`
    - `ExponentialFun`: Exponential function f(x) = base^(tau*x) + bias
    - `LogFun`: Logarithmic function f(x) = scale * log_base(tau*x) + bias
    - `SinFun`: Sinusoidal function
    - `CosFun`: Cosine function
    - `TableFun`: Lookup table (dictionary mapping)
    - `LambdaFun`: Arbitrary function via lambda/callable

Multi-Issue Functions (inherit from `BaseMultiFun`):
    These map a tuple of values from multiple issues to a real number.

    - `LinearMultiFun`: Linear combination f(x) = sum(slope[i] * x[i])
    - `AffineMultiFun`: Affine combination f(x) = sum(slope[i] * x[i]) + bias
    - `BilinearMultiFun`: Two-issue bilinear f(x,y) = a*x + b*y + c*x*y + d
    - `QuadraticMultiFun`: Full quadratic with linear, squared, and interaction terms
    - `PolynomialMultiFun`: General multivariate polynomial
    - `ProductMultiFun`: Scaled product f(x) = scale * prod(x[i]^powers[i]) + bias
    - `TableMultiFun`: Lookup table for value tuples
    - `LambdaMultiFun`: Arbitrary function via lambda/callable

Example:
    >>> from negmas.outcomes import make_issue
    >>> from negmas.preferences.value_fun import AffineFun, TableFun

    >>> # Create an affine value function
    >>> price_value = AffineFun(slope=-0.1, bias=100)
    >>> price_value(500)  # Higher price = lower value
    50.0

    >>> # Create a table-based value function for categorical issue
    >>> color_value = TableFun(mapping={"red": 1.0, "blue": 0.8, "green": 0.6})
    >>> color_value("red")
    1.0

    >>> # Multi-issue function with interaction
    >>> from negmas.preferences.value_fun import BilinearMultiFun
    >>> interaction = BilinearMultiFun(a=0.5, b=0.3, c=0.1, bias=0.0)
    >>> interaction((10, 20))  # 0.5*10 + 0.3*20 + 0.1*10*20 = 5 + 6 + 20 = 31
    31.0
"""

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
from negmas.helpers.types import get_full_type_name, is_lambda_function
from negmas.outcomes.base_issue import Issue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

MAX_CARINALITY = 10_000_000_000
# Number of samples used to approximate minmax() for value functions that combine
# several components (`AggregatingFun`, `MultiModalTrapezoidalFun`,
# `MultiModalGaussianFun`) over a continuous issue, where no closed-form extremum
# generally exists. The default per-issue iteration (`Issue.__iter__`) only yields
# `DEFAULT_LEVELS` (10) points, too coarse to reliably find narrow peaks/troughs.
_AGGREGATE_MINMAX_SAMPLES = 1000


def _dense_nonmonotonic_minmax(
    input: Issue, f: Callable[[Any], float]
) -> tuple[float, float]:
    """Like `nonmonotonic_minmax`, but samples continuous issues on a dense
    grid (`_AGGREGATE_MINMAX_SAMPLES` points) instead of the coarse default
    (`DEFAULT_LEVELS` points) used by plain iteration. Approximate: for
    functions without a closed-form extremum (e.g. a mixture of bumps), the
    true min/max may lie between samples.
    """
    if hasattr(input, "is_continuous") and input.is_continuous():
        return nonmonotonic_minmax(
            input.value_generator(n=_AGGREGATE_MINMAX_SAMPLES, endpoints=True), f
        )
    return nonmonotonic_minmax(input, f)


__all__ = [
    "BaseFun",
    "BaseMultiFun",
    "ConstFun",
    "IdentityFun",
    "AffineFun",
    "LinearFun",
    "TriangularFun",
    "TrapezoidalFun",
    "GaussianFun",
    "AggregatingFun",
    "BiasedFun",
    "MultiModalTrapezoidalFun",
    "MultiModalGaussianFun",
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
    "QuadraticMultiFun",
    "BilinearMultiFun",
    "PolynomialMultiFun",
    "ProductMultiFun",
    "make_fun_from_xml",
]


class BaseFun(ABC):
    """Abstract base class for single-issue value functions.

    A value function maps values from a single issue to real numbers.
    These are used as building blocks in utility functions like
    `LinearAdditiveUtilityFunction`.

    All subclasses must implement:
        - `__call__(x)`: Evaluate the function at value x
        - `minmax(issue)`: Find min/max values over an issue's domain

    Subclasses should also implement:
        - `shift_by(offset)`: Return a shifted version of the function
        - `scale_by(scale)`: Return a scaled version of the function
        - `xml(indx, issue, bias)`: Export to GENIUS XML format

    Example:
        >>> from negmas.outcomes import make_issue
        >>> from negmas.preferences.value_fun import AffineFun

        >>> # Create an affine function f(x) = 2x + 1
        >>> f = AffineFun(slope=2.0, bias=1.0)
        >>> f(5)
        11.0

        >>> # Find min/max over an issue
        >>> issue = make_issue(10, "quantity")  # 0-9
        >>> f.minmax(issue)
        (1.0, 19.0)
    """

    @property
    def dim(self) -> int:
        """Return the dimensionality of this function (always 1 for single-issue).

        Returns:
            int: Always returns 1.
        """
        return 1

    @abstractmethod
    def minmax(self, input: Issue) -> tuple[float, float]:
        """Find the minimum and maximum values over an issue's domain.

        Args:
            input: The issue whose domain to evaluate over.

        Returns:
            tuple[float, float]: A tuple of (minimum, maximum) values.
        """
        ...

    @abstractmethod
    def __call__(self, x) -> float:
        """Evaluate the function at a given value.

        Args:
            x: The input value (type depends on the issue).

        Returns:
            float: The function value at x.
        """
        ...

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Deserialize a value function from a dictionary.

        Args:
            d: Dictionary representation of the function.
            python_class_identifier: Key used for class type identification.

        Returns:
            BaseFun: The deserialized value function.
        """
        if isinstance(d, cls):
            return d
        _ = d.pop(python_class_identifier, None)
        return cls(**deserialize(d, python_class_identifier=python_class_identifier))  # type: ignore

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serialize the value function to a dictionary.

        Args:
            python_class_identifier: Key to use for class type identification.

        Returns:
            dict[str, Any]: Dictionary representation of the function, tagged
            with its concrete type so that it can be reconstructed correctly
            even when nested inside a heterogeneous collection of `BaseFun`
            (e.g. `AggregatingFun.funs`) via the generic `deserialize`.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(
            serialize(asdict(self), python_class_identifier=python_class_identifier)
        )
        return d

    def min(self, input: Issue) -> float:
        """Find the minimum value over an issue's domain.

        Args:
            input: The issue whose domain to evaluate over.

        Returns:
            float: The minimum value.
        """
        mn, _ = self.minmax(input)
        return mn

    def max(self, input: Issue) -> float:
        """Find the maximum value over an issue's domain.

        Args:
            input: The issue whose domain to evaluate over.

        Returns:
            float: The maximum value.
        """
        _, mx = self.minmax(input)
        return mx

    def to_table(self, issue: Issue) -> "TableFun":
        """Convert this function to a table lookup function.

        Useful for exporting functions to formats that only support
        discrete value mappings.

        Args:
            issue: The issue to enumerate values from.

        Returns:
            TableFun: A table function with the same values.

        Note:
            For continuous issues, the issue will be discretized first.
        """
        if not issue.is_finite():
            issue = issue.to_discrete()
        vals = list(issue.all)
        return TableFun(mapping=dict(zip(vals, [self(_) for _ in vals])))


class BaseMultiFun(ABC):
    """Abstract base class for multi-issue value functions.

    A multi-issue value function maps a tuple of values from multiple issues
    to a single real number. These are used in utility functions like
    `GLAUtilityFunction` to model interactions between issues.

    All subclasses must implement:
        - `__call__(x)`: Evaluate the function at value tuple x
        - `minmax(issues)`: Find min/max values over the issues' domains
        - `dim`: Property returning the number of issues

    Subclasses should also implement:
        - `shift_by(offset)`: Return a shifted version of the function
        - `scale_by(scale)`: Return a scaled version of the function
        - `xml(indx, issues, bias)`: Export to GENIUS XML format (if possible)

    Example:
        >>> from negmas.outcomes import make_issue
        >>> from negmas.preferences.value_fun import AffineMultiFun

        >>> # Create a multi-issue function f(x,y) = 0.5*x + 0.3*y + 0.1
        >>> f = AffineMultiFun(slope=(0.5, 0.3), bias=0.1)
        >>> f((10, 20))
        11.1

        >>> # Find min/max over issues
        >>> issues = [make_issue(10, "x"), make_issue(5, "y")]
        >>> f.minmax(tuple(issues))
        (0.1, 5.8)
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the number of issues this function operates on.

        Returns:
            int: The dimensionality (number of issues).
        """
        ...

    @abstractmethod
    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the minimum and maximum values over the issues' domains.

        Args:
            input: The issues whose combined domain to evaluate over.

        Returns:
            tuple[float, float]: A tuple of (minimum, maximum) values.
        """
        ...

    @abstractmethod
    def __call__(self, x: tuple) -> float:
        """Evaluate the function at a tuple of values.

        Args:
            x: Tuple of input values, one per issue.

        Returns:
            float: The function value at x.
        """
        ...

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Deserialize a value function from a dictionary.

        Args:
            d: Dictionary representation of the function.
            python_class_identifier: Key used for class type identification.

        Returns:
            BaseMultiFun: The deserialized value function.
        """
        if isinstance(d, cls):
            return d
        _ = d.pop(python_class_identifier, None)
        return cls(**deserialize(d, python_class_identifier=python_class_identifier))  # type: ignore

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serialize the value function to a dictionary.

        Args:
            python_class_identifier: Key to use for class type identification.

        Returns:
            dict[str, Any]: Dictionary representation of the function, tagged
            with its concrete type so that it can be reconstructed correctly
            even when nested inside a heterogeneous collection via the
            generic `deserialize`.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(
            serialize(asdict(self), python_class_identifier=python_class_identifier)
        )
        return d

    @abstractmethod
    def shift_by(self, offset: float) -> "BaseMultiFun":
        """Create a new function shifted by a constant offset.

        Args:
            offset: The amount to add to all output values.

        Returns:
            BaseMultiFun: A new function where f'(x) = f(x) + offset.
        """
        ...

    @abstractmethod
    def scale_by(self, scale: float) -> "BaseMultiFun":
        """Create a new function scaled by a constant factor.

        Args:
            scale: The factor to multiply all output values by.

        Returns:
            BaseMultiFun: A new function where f'(x) = scale * f(x).
        """
        ...

    def xml(
        self, indx: int, issues: list[Issue] | tuple[Issue, ...], bias: float = 0
    ) -> str:
        """Export this function to GENIUS XML format.

        Args:
            indx: The index of this function in the utility function.
            issues: The issues this function operates on.
            bias: Additional bias to add to values.

        Returns:
            str: XML string representation.

        Raises:
            NotImplementedError: Multi-issue functions generally cannot be
                exported to GENIUS XML format.
        """
        raise NotImplementedError(
            f"XML export is not implemented for {self.__class__.__name__}"
        )


@define(frozen=True)
class TableFun(BaseFun):
    """Table-based value function using dictionary lookup.

    Maps discrete issue values to real numbers using a dictionary.
    This is the most flexible single-issue function, suitable for
    categorical issues or when values don't follow a mathematical pattern.

    Args:
        mapping: Dictionary mapping issue values to utility values.

    Example:
        >>> f = TableFun(mapping={"red": 1.0, "blue": 0.8, "green": 0.6})
        >>> f("red")
        1.0
        >>> f("green")
        0.6

        >>> # Can also map numeric values
        >>> f = TableFun(mapping={0: 0.0, 1: 0.5, 2: 1.0})
        >>> f(1)
        0.5

    Note:
        Raises KeyError if called with a value not in the mapping.
    """

    mapping: dict

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Find the minimum and maximum values over an issue's domain."""
        return self._minmax(input)

    def _minmax(self, input: Issue) -> tuple[float, float]:
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> TableFun:
        """Create a new TableFun with all values shifted by a constant.

        Args:
            offset: The amount to add to each mapped value.

        Returns:
            A new TableFun where each value is increased by offset.
        """
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] + offset
        return TableFun(d)

    def scale_by(self, scale: float) -> TableFun:
        """Create a new TableFun with all values scaled by a constant.

        Args:
            scale: The factor to multiply each mapped value by.

        Returns:
            A new TableFun where each value is multiplied by scale.
        """
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] * scale
        return TableFun(d)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this function to GENIUS XML format.

        Args:
            indx: The 0-based index of this issue in the utility function.
            issue: The issue this function is defined over.
            bias: Additional bias to add to all evaluation values.

        Returns:
            XML string representing this value function.
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
        """Look up x in the mapping dictionary and return its value."""
        return self.mapping[x]


@define(frozen=True)
class AffineFun(BaseFun):
    r"""Affine value function: f(x) = slope * x + bias.

    Maps numeric issue values to utilities using an affine (linear + constant)
    transformation. This is one of the most commonly used value functions.

    Mathematical definition:

    .. math::

        f(x) = a \cdot x + b

    where :math:`a` is the slope and :math:`b` is the bias.

    Args:
        slope: The coefficient :math:`a` multiplying the input value.
        bias: The constant offset :math:`b` (default: 0).

    Example:
        >>> f = AffineFun(slope=2.0, bias=1.0)
        >>> f(5)  # 2*5 + 1 = 11
        11.0
        >>> f(0)  # 2*0 + 1 = 1
        1.0

    Note:
        - For ``slope > 0``: higher input values yield higher utilities
        - For ``slope < 0``: higher input values yield lower utilities
        - ``AffineFun(slope=m, bias=0)`` is equivalent to ``LinearFun(slope=m)``
    """

    slope: float
    bias: float = 0

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Compute the minimum and maximum values this function can produce for the given issue.

        Args:
            input: The issue whose value range should be evaluated.

        Returns:
            Tuple of (minimum_value, maximum_value) for this function over the issue's range.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        """Return a new AffineFun with the bias shifted by the given offset."""
        return AffineFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineFun:
        """Return a new AffineFun with slope and bias scaled by the given factor."""
        return AffineFun(slope=self.slope * scale, bias=self.bias * scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this value function to GENIUS XML format."""
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
        """Evaluate the affine function at x, returning slope * x + bias."""
        return x * self.slope + self.bias


@define(frozen=True)
class ConstFun(BaseFun):
    r"""Constant value function: f(x) = bias.

    Returns the same value regardless of input. Useful for issues that
    don't affect utility, or as a building block for more complex functions.

    Mathematical definition:

    .. math::

        f(x) = c

    where :math:`c` is a constant (the bias parameter).

    Args:
        bias: The constant value :math:`c` to return.

    Example:
        >>> f = ConstFun(bias=5.0)
        >>> f(0)
        5.0
        >>> f(100)
        5.0
        >>> f("anything")
        5.0
    """

    bias: float

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Return (bias, bias) since the function is constant."""
        # A constant maps every value to ``bias``; there is nothing to cache and
        # ``@lru_cache`` on a method of this (unhashable) attrs class raised
        # ``unhashable type: 'ConstFun'`` whenever a ConstFun was used as a value
        # function (e.g. for a zero-contribution issue produced by from_xml_str).
        _ = input
        return (self.bias, self.bias)

    def shift_by(self, offset: float) -> ConstFun:
        """Return a new ConstFun with the bias shifted by the given offset."""
        return ConstFun(bias=offset + self.bias)

    def scale_by(self, scale: float) -> ConstFun:
        """Return a new ConstFun with the bias scaled by the given factor.

        Since f(x) = c for all x, scaling gives f'(x) = c * scale for all x.
        """
        return ConstFun(bias=self.bias * scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this constant function to GENIUS XML format."""
        return AffineFun(0.0, self.bias).xml(indx, issue, bias)

    def __call__(self, x: float) -> float:
        """Return the constant bias value, ignoring x."""
        _ = x
        return self.bias


@define(frozen=True)
class LinearFun(BaseFun):
    r"""Linear value function: f(x) = slope * x.

    Maps numeric issue values to utilities using a linear transformation
    through the origin (no constant offset).

    Mathematical definition:

    .. math::

        f(x) = a \cdot x

    where :math:`a` is the slope.

    Args:
        slope: The coefficient :math:`a` multiplying the input value.

    Example:
        >>> f = LinearFun(slope=0.5)
        >>> f(10)  # 0.5 * 10 = 5
        5.0
        >>> f(0)  # 0.5 * 0 = 0
        0.0

    Note:
        This is equivalent to ``AffineFun(slope=a, bias=0)``.
        Use ``AffineFun`` if you need a non-zero intercept.
    """

    slope: float

    @property
    def bias(sef):
        """Return the constant offset (always 0 for LinearFun)."""
        return 0.0

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Find the minimum and maximum values over an issue's domain."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Issue) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> AffineFun:
        """Return an AffineFun with this slope and the given offset as bias."""
        return AffineFun(bias=offset, slope=self.slope)

    def scale_by(self, scale: float) -> LinearFun:
        """Return a new LinearFun with the slope scaled by the given factor."""
        return LinearFun(slope=scale * self.slope)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this linear function to GENIUS XML format."""
        return AffineFun(self.slope, 0.0).xml(indx, issue, bias)

    def __call__(self, x: float) -> float:
        """Evaluate the linear function at x, returning slope * x."""
        return x * self.slope


@define(frozen=True)
class IdentityFun(BaseFun):
    r"""Identity value function: f(x) = x.

    Returns the input value unchanged. Useful when the issue value
    directly represents the utility.

    Mathematical definition:

    .. math::

        f(x) = x

    Example:
        >>> f = IdentityFun()
        >>> f(5)
        5
        >>> f(3.14)
        3.14
    """

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Return the issue's min and max values directly."""
        return (input.min_value, input.max_value)

    def shift_by(self, offset: float) -> AffineFun:
        """Return f(x) = x + offset as an AffineFun (slope 1).

        Note:
            Returning ``ConstFun(bias=offset)`` here would collapse the identity
            map to a constant, dropping the issue's contribution during
            normalization (the overall maximum would fall below 1). The shift of
            ``f(x) = x`` by ``offset`` is ``f(x) = x + offset``, i.e. slope 1.
        """
        return AffineFun(slope=1.0, bias=offset)

    def scale_by(self, scale: float) -> LinearFun:
        """Return a LinearFun with the given scale as slope."""
        return LinearFun(slope=scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this identity function to GENIUS XML format."""
        return LinearFun(1.0).xml(indx, issue, bias)

    def __call__(self, x: float) -> float:
        """Return x unchanged."""
        return x


@define(frozen=True)
class LambdaFun(BaseFun):
    r"""Lambda-based value function using an arbitrary callable.

    Wraps any callable (function, lambda, etc.) as a value function.
    This provides maximum flexibility when predefined functions don't
    suffice.

    Mathematical definition:

    .. math::

        f(x) = g(x) + b

    where :math:`g` is the user-provided callable and :math:`b` is an
    optional bias.

    Args:
        f: A callable that takes an issue value and returns a float.
        bias: Constant offset added to the callable's output (default: 0).
        min_value: Known minimum value (optional, for optimization).
        max_value: Known maximum value (optional, for optimization).

    Example:
        >>> f = LambdaFun(f=lambda x: x**2)
        >>> f(3)  # 3^2 = 9
        9

        >>> f = LambdaFun(f=lambda x: x**2, bias=1.0)
        >>> f(3)  # 3^2 + 1 = 10
        10.0

    Warning:
        Lambda functions can be serialized, but complex callables
        (e.g., closures with external state) may not serialize correctly.
    """

    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self):
        # we need to be sure that f is a lambda function so that it can
        # correctly be serialized
        """Ensure the callable is wrapped as a lambda for serialization."""
        if not is_lambda_function(self.f):
            f = self.f
            object.__setattr__(self, "f", lambda x: f(x))

    def minmax(self, input) -> tuple[float, float]:
        """Find the min/max values, using cached bounds if available."""
        if self.min_value is not None and self.max_value is not None:
            return self.min_value, self.max_value
        mn, mx = nonmonotonic_minmax(input, self.f)
        if self.min_value is not None:
            mn = min(mn, self.min_value)
        if self.max_value is not None:
            mx = min(mx, self.max_value)
        return mn, mx

    def shift_by(self, offset: float, change_bias: bool = False) -> LambdaFun:
        """Return a new LambdaFun shifted by the given offset."""
        mn, mx = self.min_value, self.max_value
        return LambdaFun(
            self.f if change_bias else lambda x: offset + self.f(x),
            bias=(self.bias + offset) if change_bias else self.bias,
            min_value=mn if mn is None else mn + offset,
            max_value=mx if mx is None else mx + offset,
        )

    def scale_by(self, scale: float) -> LambdaFun:
        """Return a new LambdaFun scaled by the given factor."""
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
        """Export this function to GENIUS XML format (discrete issues only)."""
        if issue.is_discrete():
            values = list(issue.all)
            return TableFun(dict(zip(values, [self(_) for _ in values]))).xml(
                indx, issue, bias
            )
        raise ValueError("LambdaFun with a continuous issue cannot be converted to XML")

    def __call__(self, x: Any) -> float:
        """Evaluate the wrapped function at x and add the bias."""
        return self.f(x) + self.bias


@define(frozen=True)
class QuadraticFun(BaseFun):
    r"""Quadratic value function: f(x) = a2*x^2 + a1*x + bias.

    Maps numeric issue values to utilities using a quadratic polynomial.

    Mathematical definition:

    .. math::

        f(x) = a_2 x^2 + a_1 x + b

    where :math:`a_2` is the quadratic coefficient, :math:`a_1` is the linear
    coefficient, and :math:`b` is the bias (constant term).

    Args:
        a2: Quadratic coefficient :math:`a_2`.
        a1: Linear coefficient :math:`a_1`.
        bias: Constant term :math:`b` (default: 0).

    Example:
        >>> f = QuadraticFun(a2=1.0, a1=-2.0, bias=1.0)  # (x-1)^2
        >>> f(1)  # Minimum at x=1
        0.0
        >>> f(0)  # 0 - 0 + 1 = 1
        1.0
        >>> f(3)  # 9 - 6 + 1 = 4
        4.0

    Note:
        - If :math:`a_2 > 0`: parabola opens upward (has a minimum)
        - If :math:`a_2 < 0`: parabola opens downward (has a maximum)
        - Extremum occurs at :math:`x = -a_1 / (2 a_2)`
    """

    a2: float
    a1: float
    bias: float = 0

    def minmax(self, input) -> tuple[float, float]:
        """Find the min/max values, accounting for the parabola's extremum."""
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
        """Return a new QuadraticFun with the bias shifted by the given offset."""
        return QuadraticFun(bias=self.bias + offset, a1=self.a1, a2=self.a2)

    def scale_by(self, scale: float) -> QuadraticFun:
        """Return a new QuadraticFun with all coefficients scaled."""
        return QuadraticFun(
            bias=self.bias * scale, a1=self.a1 * scale, a2=self.a2 * scale
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Export this quadratic function to GENIUS XML format."""
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
        """Evaluate a2*x^2 + a1*x + bias at the given x."""
        return self.a2 * x * x + self.a1 * x + self.bias


@define(frozen=True)
class PolynomialFun(BaseFun):
    r"""General polynomial value function.

    Maps numeric issue values to utilities using a polynomial of arbitrary degree.

    Mathematical definition:

    .. math::

        f(x) = b + \sum_{k=1}^{n} c_k x^k

    where :math:`b` is the bias (constant term) and :math:`c_k` are the
    coefficients for powers 1 through n.

    Args:
        coefficients: Tuple of coefficients :math:`(c_1, c_2, ..., c_n)` for
            powers 1, 2, ..., n respectively.
        bias: Constant term :math:`b` (default: 0).

    Example:
        >>> # f(x) = 1 + 2x + 3x^2
        >>> f = PolynomialFun(coefficients=(2.0, 3.0), bias=1.0)
        >>> f(0)  # 1 + 0 + 0 = 1
        1.0
        >>> f(1)  # 1 + 2 + 3 = 6
        6.0
        >>> f(2)  # 1 + 4 + 12 = 17
        17.0

    Note:
        The coefficients tuple is indexed from power 1, not power 0.
        Use the ``bias`` parameter for the constant (power 0) term.
    """

    coefficients: tuple[float, ...]
    bias: float = 0

    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        """Find the min/max values by sampling (exact method not yet implemented)."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> PolynomialFun:
        """Return a new PolynomialFun with the bias shifted by the given offset."""
        return PolynomialFun(bias=self.bias + offset, coefficients=self.coefficients)

    def scale_by(self, scale: float) -> PolynomialFun:
        """Return a new PolynomialFun with all coefficients scaled."""
        return PolynomialFun(
            bias=self.bias * scale,
            coefficients=tuple(_ * scale for _ in self.coefficients),
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Export this polynomial function to GENIUS XML format."""
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
        """Evaluate the polynomial at x by summing coefficient*x^power terms."""
        return reduce(
            add, [b * pow(x, p + 1) for p, b in enumerate(self.coefficients)], self.bias
        )


@define(frozen=True)
class TriangularFun(BaseFun):
    r"""Triangular (tent) value function.

    A piecewise-linear function that rises from bias at ``start`` to
    ``bias + scale`` at ``middle``, then falls back to bias at ``end``.

    Mathematical definition:

    .. math::

        f(x) = \begin{cases}
            b & \text{if } x \le s \\
            b + k \cdot \frac{x - s}{m - s} & \text{if } s < x \le m \\
            b + k \cdot \frac{e - x}{e - m} & \text{if } m < x < e \\
            b & \text{if } x \ge e
        \end{cases}

    where :math:`s` is start, :math:`m` is middle, :math:`e` is end,
    :math:`b` is bias, and :math:`k` is scale.

    Args:
        start: x value where function starts rising from bias.
        middle: x value where function reaches maximum (bias + scale).
        end: x value where function returns to bias.
        bias: Offset added to all values (default: 0).
        scale: Height of the peak above bias (default: 1).

    Example:
        >>> f = TriangularFun(start=0.0, middle=5.0, end=10.0)
        >>> f(0)   # At start: bias = 0
        0
        >>> f(5)   # At middle: bias + scale = 1
        1.0
        >>> f(10)  # At end: bias = 0
        0
        >>> f(2.5) # Halfway up: 0.5
        0.5
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
    r"""Exponential value function: f(x) = base^(tau*x) + bias.

    Maps numeric issue values to utilities using an exponential function.

    Mathematical definition:

    .. math::

        f(x) = a^{\tau x} + b

    where :math:`a` is the base, :math:`\tau` is the rate parameter,
    and :math:`b` is the bias.

    Args:
        tau: Rate parameter :math:`\tau` controlling growth/decay speed.
        bias: Constant offset :math:`b` (default: 0).
        base: Exponential base :math:`a` (default: e ≈ 2.718).

    Example:
        >>> import math
        >>> f = ExponentialFun(tau=1.0)  # f(x) = e^x
        >>> abs(f(0) - 1.0) < 1e-10  # e^0 = 1
        True
        >>> abs(f(1) - math.e) < 1e-10  # e^1 = e
        True

    Note:
        - For :math:`\tau > 0`: function grows with increasing x
        - For :math:`\tau < 0`: function decays with increasing x
        - Common bases: e (natural), 2 (binary), 10 (decimal)
    """

    tau: float
    bias: float = 0
    base: float = e

    def minmax(self, input) -> tuple[float, float]:
        """Find the min/max values over the monotonic exponential function."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> ExponentialFun:
        """Return a new ExponentialFun with the bias shifted by the given offset."""
        return ExponentialFun(bias=self.bias + offset, tau=self.tau)

    def scale_by(self, scale: float) -> ExponentialFun:
        """Return a new ExponentialFun scaled by adjusting tau and bias."""
        return ExponentialFun(
            bias=self.bias * scale, tau=self.tau + math.log(scale, base=self.base)
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Export this exponential function to GENIUS XML format."""
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
        """Evaluate base^(tau*x) + bias at the given x."""
        return pow(self.base, self.tau * x) + self.bias


@define(frozen=True)
class CosFun(BaseFun):
    r"""Cosine value function.

    Maps numeric issue values to utilities using a cosine function.

    Mathematical definition:

    .. math::

        f(x) = A \cos(\omega x + \phi) + b

    where :math:`A` is the amplitude, :math:`\omega` is the angular frequency
    (multiplier), :math:`\phi` is the phase shift, and :math:`b` is the bias.

    Args:
        multiplier: Angular frequency :math:`\omega` (default: 1.0).
        bias: Vertical offset :math:`b` (default: 0.0).
        phase: Phase shift :math:`\phi` in radians (default: 0.0).
        amplitude: Amplitude :math:`A` (default: 1.0).

    Example:
        >>> import math
        >>> f = CosFun()  # f(x) = cos(x)
        >>> abs(f(0) - 1.0) < 1e-10  # cos(0) = 1
        True
        >>> abs(f(math.pi) - (-1.0)) < 1e-10  # cos(pi) = -1
        True

    Note:
        The function oscillates between ``bias - amplitude`` and
        ``bias + amplitude`` with period :math:`2\pi / \omega`.
    """

    multiplier: float = 1.0
    bias: float = 0.0
    phase: float = 0.0
    amplitude: float = 1.0

    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        """Find the min/max values by sampling (exact method not yet implemented)."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> CosFun:
        """Return a new CosFun with the bias shifted by the given offset."""
        return CosFun(
            bias=self.bias + offset,
            multiplier=self.multiplier,
            phase=self.phase,
            amplitude=self.amplitude,
        )

    def scale_by(self, scale: float) -> CosFun:
        """Return a new CosFun with amplitude and bias scaled."""
        return CosFun(
            amplitude=self.amplitude * scale,
            bias=self.bias * scale,
            multiplier=self.multiplier,
            phase=self.phase,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Export this cosine function to GENIUS XML format."""
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
        """Evaluate amplitude * cos(multiplier*x + phase) + bias at the given x."""
        return self.amplitude * (cos(self.multiplier * x + self.phase)) + self.bias


@define(frozen=True)
class SinFun(BaseFun):
    r"""Sine value function.

    Maps numeric issue values to utilities using a sine function.

    Mathematical definition:

    .. math::

        f(x) = A \sin(\omega x + \phi) + b

    where :math:`A` is the amplitude, :math:`\omega` is the angular frequency
    (multiplier), :math:`\phi` is the phase shift, and :math:`b` is the bias.

    Args:
        multiplier: Angular frequency :math:`\omega` (default: 1.0).
        bias: Vertical offset :math:`b` (default: 0.0).
        phase: Phase shift :math:`\phi` in radians (default: 0.0).
        amplitude: Amplitude :math:`A` (default: 1.0).

    Example:
        >>> import math
        >>> f = SinFun()  # f(x) = sin(x)
        >>> abs(f(0) - 0.0) < 1e-10  # sin(0) = 0
        True
        >>> abs(f(math.pi / 2) - 1.0) < 1e-10  # sin(pi/2) = 1
        True

    Note:
        The function oscillates between ``bias - amplitude`` and
        ``bias + amplitude`` with period :math:`2\pi / \omega`.
    """

    multiplier: float = 1.0
    bias: float = 0.0
    phase: float = 0.0
    amplitude: float = 1.0

    def minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        """Find the min/max values by sampling (exact method not yet implemented)."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        # todo: implement this exactly without sampling
        return nonmonotonic_minmax(input, self)

    def shift_by(self, offset: float) -> SinFun:
        """Return a new SinFun with the bias shifted by the given offset."""
        return SinFun(
            bias=self.bias + offset,
            multiplier=self.multiplier,
            phase=self.phase,
            amplitude=self.amplitude,
        )

    def scale_by(self, scale: float) -> SinFun:
        """Return a new SinFun with amplitude and bias scaled."""
        return SinFun(
            amplitude=self.amplitude * scale,
            bias=self.bias * scale,
            multiplier=self.multiplier,
            phase=self.phase,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Export this sine function to GENIUS XML format."""
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
        """Evaluate amplitude * sin(multiplier*x + phase) + bias at the given x."""
        return self.amplitude * (sin(self.multiplier * x + self.phase)) + self.bias


@define(frozen=True)
class LogFun(BaseFun):
    r"""Logarithmic value function: f(x) = scale * log_base(tau*x) + bias.

    Maps numeric issue values to utilities using a logarithmic function.

    Mathematical definition:

    .. math::

        f(x) = s \cdot \log_a(\tau x) + b

    where :math:`s` is the scale, :math:`a` is the base, :math:`\tau` is the
    rate parameter, and :math:`b` is the bias.

    Args:
        tau: Rate parameter :math:`\tau` that scales the input.
        bias: Constant offset :math:`b` (default: 0).
        base: Logarithm base :math:`a` (default: e ≈ 2.718).
        scale: Output scale :math:`s` (default: 1.0).

    Example:
        >>> import math
        >>> f = LogFun(tau=1.0)  # f(x) = ln(x)
        >>> abs(f(1) - 0.0) < 1e-10  # ln(1) = 0
        True
        >>> abs(f(math.e) - 1.0) < 1e-10  # ln(e) = 1
        True

    Warning:
        The input ``tau * x`` must be positive, as logarithm is undefined
        for non-positive values.

    Note:
        - For :math:`\tau > 0` and :math:`s > 0`: function grows with increasing x
        - For :math:`\tau > 0` and :math:`s < 0`: function decreases with increasing x
        - Common bases: e (natural log), 2 (binary log), 10 (common log)
    """

    tau: float
    bias: float = 0
    base: float = e
    scale: float = 1.0

    def minmax(self, input) -> tuple[float, float]:
        """Find the min/max values over the monotonic logarithmic function."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        return monotonic_minmax(input, self)

    def shift_by(self, offset: float) -> LogFun:
        """Return a new LogFun with the bias shifted by the given offset."""
        return LogFun(
            bias=self.bias + offset, tau=self.tau, scale=self.scale, base=self.base
        )

    def scale_by(self, scale: float) -> LogFun:
        """Return a new LogFun with scale and bias multiplied by the factor."""
        return LogFun(
            bias=self.bias * scale,
            tau=self.tau,
            scale=self.scale * scale,
            base=self.base,
        )

    def xml(self, indx: int, issue: Issue, bias) -> str:
        """Export this logarithmic function to GENIUS XML format."""
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
        """Evaluate scale * log_base(tau*x) + bias at the given x."""
        return self.scale * log(self.tau * x, self.base) + self.bias


@define(frozen=True)
class TrapezoidalFun(BaseFun):
    r"""Trapezoidal value function: a triangular function with a flat top.

    A piecewise-linear function that rises from bias at ``start`` to
    ``bias + scale`` at ``rise_end``, stays at ``bias + scale`` until
    ``fall_start``, then falls back to bias at ``end``.

    Mathematical definition:

    .. math::

        f(x) = \begin{cases}
            b & \text{if } x \le s \\
            b + k \cdot \frac{x - s}{r - s} & \text{if } s < x < r \\
            b + k & \text{if } r \le x \le f \\
            b + k \cdot \frac{e - x}{e - f} & \text{if } f < x < e \\
            b & \text{if } x \ge e
        \end{cases}

    where :math:`s` is start, :math:`r` is rise_end, :math:`f` is fall_start,
    :math:`e` is end, :math:`b` is bias, and :math:`k` is scale.

    Args:
        start: x value where function starts rising from bias.
        rise_end: x value where function first reaches the plateau (bias + scale).
        fall_start: x value where function starts falling from the plateau.
        end: x value where function returns to bias.
        bias: Offset added to all values (default: 0).
        scale: Height of the plateau above bias (default: 1).

    Example:
        >>> f = TrapezoidalFun(start=0.0, rise_end=2.0, fall_start=8.0, end=10.0)
        >>> f(0)    # At start: bias = 0
        0
        >>> f(1)    # Halfway up the rise: 0.5
        0.5
        >>> f(5)    # On the plateau: bias + scale = 1
        1.0
        >>> f(9)    # Halfway down the fall: 0.5
        0.5
        >>> f(10)   # At end: bias = 0
        0

    Note:
        Setting ``start == rise_end`` and ``fall_start == end`` reduces this
        to a rectangular (step) function. Setting ``rise_end == fall_start``
        reduces this to `TriangularFun`.
    """

    start: float
    rise_end: float
    fall_start: float
    end: float
    bias: float = 0
    scale: float = 1

    def __attrs_post_init__(self):
        if not (self.start <= self.rise_end <= self.fall_start <= self.end):
            raise ValueError(
                "TrapezoidalFun requires start <= rise_end <= fall_start <= end, "
                f"got {self.start=}, {self.rise_end=}, {self.fall_start=}, {self.end=}"
            )

    def shift_by(self, offset: float) -> "TrapezoidalFun":
        return TrapezoidalFun(
            start=self.start,
            rise_end=self.rise_end,
            fall_start=self.fall_start,
            end=self.end,
            bias=self.bias + offset,
            scale=self.scale,
        )

    def scale_by(self, scale: float) -> "TrapezoidalFun":
        # Scale the output amplitude, not the x-coordinates
        return TrapezoidalFun(
            start=self.start,
            rise_end=self.rise_end,
            fall_start=self.fall_start,
            end=self.end,
            bias=self.bias * scale,
            scale=self.scale * scale,
        )

    def minmax(self, input) -> tuple[float, float]:
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        if hasattr(input, "min_value") and hasattr(input, "max_value"):
            lo, hi = input.min_value, input.max_value
            vals = [self(lo), self(hi)]
            top_lo, top_hi = self.rise_end, self.fall_start
            if max(lo, top_lo) <= min(hi, top_hi):
                vals.append(self.bias + self.scale)
            return min(vals), max(vals)
        return nonmonotonic_minmax(input, self)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this function to GENIUS XML format.

        GENIUS has no native trapezoidal evaluator, so this is only
        supported for discrete issues via a lookup table.
        """
        if issue.is_discrete():
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        raise NotImplementedError(
            "TrapezoidalFun has no native GENIUS evaluator type; only discrete "
            "issues can be exported to XML."
        )

    def __call__(self, x: float):
        x = float(x)
        if x < self.start or x > self.end:
            return self.bias
        if x <= self.rise_end:
            if self.rise_end == self.start:
                return self.bias + self.scale
            return self.bias + self.scale * (x - self.start) / (
                self.rise_end - self.start
            )
        if x <= self.fall_start:
            return self.bias + self.scale
        # Unreachable when fall_start == end: `x <= self.fall_start` above
        # would already have matched every in-domain x (since fall_start ==
        # end and x <= end is guaranteed by the first check), so no 0/0
        # division from `end - fall_start` can occur here.
        return self.bias + self.scale * (self.end - x) / (self.end - self.fall_start)


@define(frozen=True)
class GaussianFun(BaseFun):
    r"""Gaussian (bell-curve) value function.

    Maps numeric issue values to utilities using a Gaussian bump centered at
    ``center``. The center is a free parameter (it need not lie inside the
    issue's range); the function is simply evaluated (and its min/max
    computed) over whatever domain the issue restricts it to -- i.e. it is
    "restricted to the edges of the issue" in the sense that `minmax` only
    considers the bounded interval ``[issue.min_value, issue.max_value]``
    rather than the whole real line.

    Mathematical definition:

    .. math::

        f(x) = b + k \cdot \exp\left(-\frac{(x - c)^2}{2 \sigma^2}\right)

    where :math:`c` is the center, :math:`\sigma` is the standard deviation,
    :math:`b` is the bias, and :math:`k` is the scale (peak height above bias).

    Args:
        center: The x value :math:`c` where the peak (or trough, if
            ``scale < 0``) occurs.
        sigma: Standard deviation :math:`\sigma` controlling the width of the
            bump. Must be strictly positive.
        bias: Offset added to all values (default: 0).
        scale: Height of the peak above bias (default: 1). Use a negative
            value for a dip instead of a bump.

    Example:
        >>> f = GaussianFun(center=5.0, sigma=1.0)
        >>> f(5)  # At the center: bias + scale = 1
        1.0
        >>> 0 < f(6) < 1  # One sigma away: less than the peak
        True
        >>> f(6) == f(4)  # Symmetric around the center
        True
    """

    center: float
    sigma: float
    bias: float = 0
    scale: float = 1

    def __attrs_post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {self.sigma}")

    def shift_by(self, offset: float) -> "GaussianFun":
        return GaussianFun(
            center=self.center,
            sigma=self.sigma,
            bias=self.bias + offset,
            scale=self.scale,
        )

    def scale_by(self, scale: float) -> "GaussianFun":
        return GaussianFun(
            center=self.center,
            sigma=self.sigma,
            bias=self.bias * scale,
            scale=self.scale * scale,
        )

    def minmax(self, input) -> tuple[float, float]:
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input) -> tuple[float, float]:
        if hasattr(input, "min_value") and hasattr(input, "max_value"):
            lo, hi = input.min_value, input.max_value
            vals = [self(lo), self(hi)]
            if lo <= self.center <= hi:
                vals.append(self.bias + self.scale)
            return min(vals), max(vals)
        return nonmonotonic_minmax(input, self)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this function to GENIUS XML format.

        GENIUS has no native Gaussian evaluator, so this is only supported
        for discrete issues via a lookup table.
        """
        if issue.is_discrete():
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        raise NotImplementedError(
            "GaussianFun has no native GENIUS evaluator type; only discrete "
            "issues can be exported to XML."
        )

    def __call__(self, x: float):
        x = float(x)
        return self.bias + self.scale * math.exp(
            -((x - self.center) ** 2) / (2 * self.sigma * self.sigma)
        )


@define(frozen=True)
class AggregatingFun(BaseFun):
    r"""Linearly aggregating value function: a weighted sum of base value funs.

    Combines multiple single-issue value functions (all defined over the
    same issue) into one, computing a weighted sum plus a bias. Optionally
    normalizes its output to the range ``[0, 1]`` over a given issue.

    Mathematical definition:

    .. math::

        f(x) = b + \sum_{i=1}^{n} w_i \cdot g_i(x)

    where :math:`g_i` are the component value functions, :math:`w_i` are
    their weights, and :math:`b` is the bias.

    Args:
        funs: Tuple of component `BaseFun` instances to combine.
        weights: Tuple of weights, one per component (default: 1.0 each).
        bias: Constant offset added to the weighted sum (default: 0).
        normalize: If True, rescale the combination so that its range over
            ``issue`` is exactly ``[0, 1]``. Requires ``issue`` to be given.
        issue: The issue used to compute the normalization range. Only
            needed (and only used) when ``normalize=True``; once the
            normalization is baked into ``weights``/``bias`` at construction
            time, this is reset to ``None``.

    Example:
        >>> from negmas.outcomes import make_issue
        >>> f = AggregatingFun(
        ...     funs=(AffineFun(slope=1.0), ConstFun(bias=1.0)), weights=(0.5, 2.0)
        ... )
        >>> f(10)  # 0.5*10 + 2.0*1 = 7
        7.0

        >>> # Normalize a mixture of two Gaussians to [0, 1] over an issue
        >>> issue = make_issue((0.0, 10.0), "x")
        >>> g = AggregatingFun(
        ...     funs=(
        ...         GaussianFun(center=2.0, sigma=1.0),
        ...         GaussianFun(center=8.0, sigma=1.0),
        ...     ),
        ...     normalize=True,
        ...     issue=issue,
        ... )
        >>> mn, mx = g.minmax(issue)
        >>> abs(mn - 0.0) < 1e-6 and abs(mx - 1.0) < 1e-6
        True
    """

    funs: tuple[BaseFun, ...]
    weights: tuple[float, ...] | None = None
    bias: float = 0
    normalize: bool = False
    issue: Issue | None = None

    def __attrs_post_init__(self):
        if not self.funs:
            raise ValueError("AggregatingFun requires at least one component fun")
        if self.weights is None:
            object.__setattr__(self, "weights", tuple(1.0 for _ in self.funs))
        if len(self.weights) != len(self.funs):
            raise ValueError(
                f"Got {len(self.funs)} funs but {len(self.weights)} weights"
            )
        if self.normalize:
            if self.issue is None:
                raise ValueError(
                    "normalize=True requires `issue` to determine the value range"
                )
            mn, mx = _dense_nonmonotonic_minmax(self.issue, self._raw_call)
            span = mx - mn
            s = 1.0 / span if abs(span) > 1e-12 else 1.0
            offset = -mn * s
            object.__setattr__(self, "weights", tuple(w * s for w in self.weights))
            object.__setattr__(self, "bias", self.bias * s + offset)
            object.__setattr__(self, "normalize", False)
            object.__setattr__(self, "issue", None)

    def _raw_call(self, x) -> float:
        return self.bias + sum(w * f(x) for w, f in zip(self.weights, self.funs))

    def shift_by(self, offset: float) -> "AggregatingFun":
        return AggregatingFun(
            funs=self.funs, weights=self.weights, bias=self.bias + offset
        )

    def scale_by(self, scale: float) -> "AggregatingFun":
        return AggregatingFun(
            funs=self.funs,
            weights=tuple(scale * w for w in self.weights),
            bias=self.bias * scale,
        )

    def minmax(self, input) -> tuple[float, float]:
        # A weighted sum of arbitrary component funs need not be monotonic or
        # unimodal, so we cannot in general compute this in closed form; fall
        # back to sampling the issue's domain, using a denser grid than the
        # default for continuous issues (see `_dense_nonmonotonic_minmax`) so
        # that narrow peaks (e.g. a mixture of `GaussianFun`s) are not missed.
        # This is still approximate: the true extremum may lie between
        # samples. We deliberately avoid `@lru_cache` here: `funs` may hold
        # non-hashable components (e.g. `TableFun`, whose `mapping` is a
        # dict), which would make `self` unhashable (see `ConstFun.minmax`
        # for prior art).
        return _dense_nonmonotonic_minmax(input, self)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this function to GENIUS XML format.

        GENIUS has no native evaluator for an arbitrary weighted sum, so
        this is only supported for discrete issues via a lookup table.
        """
        if issue.is_discrete():
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        raise NotImplementedError(
            "AggregatingFun has no native GENIUS evaluator type; only discrete "
            "issues can be exported to XML."
        )

    def __call__(self, x) -> float:
        return self._raw_call(x)

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serialize, preserving the concrete type of each component fun."""
        d = {python_class_identifier: get_full_type_name(type(self))}
        return dict(
            **d,
            funs=serialize(self.funs, python_class_identifier=python_class_identifier),
            weights=list(self.weights) if self.weights is not None else None,
            bias=self.bias,
            normalize=self.normalize,
            issue=serialize(self.issue, python_class_identifier=python_class_identifier)
            if self.issue is not None
            else None,
        )

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        d = deserialize(
            d,
            deep=True,
            remove_type_field=True,
            python_class_identifier=python_class_identifier,
        )  # type: ignore
        return cls(**d)  # type: ignore


@define(frozen=True)
class BiasedFun(BaseFun):
    r"""Wraps any value function, adding a constant bias (and optional 0-1 normalization).

    Useful for value function types that don't already expose a ``bias``
    parameter of their own (e.g. `IdentityFun`): ``f(x) = g(x) + b``.

    Mathematical definition:

    .. math::

        f(x) = g(x) + b

    where :math:`g` is the wrapped value function and :math:`b` is the bias.

    Args:
        fun: The `BaseFun` to wrap.
        bias: Constant offset added to the wrapped function's output
            (default: 0).
        normalize: If True, rescale ``fun(x) + bias`` so that its range over
            ``issue`` is exactly ``[0, 1]``. Requires ``issue`` to be given.
        issue: The issue used to compute the normalization range. Only
            needed (and only used) when ``normalize=True``; once the
            normalization is baked in at construction time, this is reset to
            ``None``.

    Example:
        >>> f = BiasedFun(fun=IdentityFun(), bias=3.0)
        >>> f(5)  # 5 + 3
        8.0

        >>> from negmas.outcomes import make_issue
        >>> issue = make_issue((0.0, 10.0), "x")
        >>> g = BiasedFun(fun=IdentityFun(), normalize=True, issue=issue)
        >>> g.minmax(issue)
        (0.0, 1.0)
    """

    fun: BaseFun
    bias: float = 0
    normalize: bool = False
    issue: Issue | None = None

    def __attrs_post_init__(self):
        if self.normalize:
            if self.issue is None:
                raise ValueError(
                    "normalize=True requires `issue` to determine the value range"
                )
            mn, mx = self.fun.minmax(self.issue)
            mn, mx = mn + self.bias, mx + self.bias
            span = mx - mn
            s = 1.0 / span if abs(span) > 1e-12 else 1.0
            offset = -mn * s
            object.__setattr__(self, "fun", self.fun.scale_by(s))
            object.__setattr__(self, "bias", self.bias * s + offset)
            object.__setattr__(self, "normalize", False)
            object.__setattr__(self, "issue", None)

    def shift_by(self, offset: float) -> "BiasedFun":
        return BiasedFun(fun=self.fun, bias=self.bias + offset)

    def scale_by(self, scale: float) -> "BiasedFun":
        return BiasedFun(fun=self.fun.scale_by(scale), bias=self.bias * scale)

    def minmax(self, input) -> tuple[float, float]:
        mn, mx = self.fun.minmax(input)
        return mn + self.bias, mx + self.bias

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Export this function to GENIUS XML format by delegating to the
        wrapped function, shifted by this wrapper's bias."""
        return self.fun.shift_by(self.bias).xml(indx, issue, bias)

    def __call__(self, x) -> float:
        return self.fun(x) + self.bias

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serialize, preserving the concrete type of the wrapped fun."""
        d = {python_class_identifier: get_full_type_name(type(self))}
        return dict(
            **d,
            fun=serialize(self.fun, python_class_identifier=python_class_identifier),
            bias=self.bias,
            normalize=self.normalize,
            issue=serialize(self.issue, python_class_identifier=python_class_identifier)
            if self.issue is not None
            else None,
        )

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        d = deserialize(
            d,
            deep=True,
            remove_type_field=True,
            python_class_identifier=python_class_identifier,
        )  # type: ignore
        return cls(**d)  # type: ignore


@define(frozen=True)
class MultiModalTrapezoidalFun(BaseFun):
    r"""A mixture (weighted sum) of several `TrapezoidalFun`, for multi-modal values.

    Equivalent to building an `AggregatingFun` out of several `TrapezoidalFun`
    instances, but takes each `TrapezoidalFun` parameter as a tuple (one entry
    per component) plus a ``weights`` tuple, which is more convenient when
    all components are trapezoidal.

    Note:
        `minmax` is approximate for continuous issues (a mixture of several
        trapezoids need not have a closed-form extremum): it delegates to
        `AggregatingFun.minmax`, which samples the issue's domain on a dense
        grid.

    Mathematical definition:

    .. math::

        f(x) = b + \sum_{i=1}^{n} w_i \cdot T_i(x)

    where each :math:`T_i` is a `TrapezoidalFun` built from the i-th entries
    of ``starts``, ``rise_ends``, ``fall_starts``, ``ends``, ``biases`` and
    ``scales``.

    Args:
        starts: Tuple of ``start`` values, one per component.
        rise_ends: Tuple of ``rise_end`` values, one per component.
        fall_starts: Tuple of ``fall_start`` values, one per component.
        ends: Tuple of ``end`` values, one per component.
        biases: Tuple of per-component biases (default: 0 each).
        scales: Tuple of per-component scales (default: 1 each).
        weights: Tuple of weights multiplying each component before summing
            (default: 1 each).
        bias: Overall constant offset added after summing (default: 0).

    Example:
        >>> f = MultiModalTrapezoidalFun(
        ...     starts=(0.0, 5.0),
        ...     rise_ends=(1.0, 6.0),
        ...     fall_starts=(2.0, 7.0),
        ...     ends=(3.0, 8.0),
        ... )
        >>> f(1.5)  # peak of the first trapezoid
        1.0
        >>> f(6.5)  # peak of the second trapezoid
        1.0
        >>> f(4.0)  # between the two, both are at bias (0)
        0.0
    """

    starts: tuple[float, ...]
    rise_ends: tuple[float, ...]
    fall_starts: tuple[float, ...]
    ends: tuple[float, ...]
    biases: tuple[float, ...] | None = None
    scales: tuple[float, ...] | None = None
    weights: tuple[float, ...] | None = None
    bias: float = 0

    def __attrs_post_init__(self):
        n = len(self.starts)
        for name, seq in (
            ("rise_ends", self.rise_ends),
            ("fall_starts", self.fall_starts),
            ("ends", self.ends),
        ):
            if len(seq) != n:
                raise ValueError(f"{name} must have length {n} (same as starts)")
        if self.biases is None:
            object.__setattr__(self, "biases", tuple(0.0 for _ in range(n)))
        if self.scales is None:
            object.__setattr__(self, "scales", tuple(1.0 for _ in range(n)))
        if self.weights is None:
            object.__setattr__(self, "weights", tuple(1.0 for _ in range(n)))
        for name, seq in (
            ("biases", self.biases),
            ("scales", self.scales),
            ("weights", self.weights),
        ):
            if len(seq) != n:
                raise ValueError(f"{name} must have length {n} (same as starts)")

    def _components(self) -> tuple[TrapezoidalFun, ...]:
        return tuple(
            TrapezoidalFun(start=s, rise_end=r, fall_start=f, end=e, bias=b, scale=sc)
            for s, r, f, e, b, sc in zip(
                self.starts,
                self.rise_ends,
                self.fall_starts,
                self.ends,
                self.biases,
                self.scales,
            )
        )

    def _agg(self) -> AggregatingFun:
        return AggregatingFun(
            funs=self._components(), weights=self.weights, bias=self.bias
        )

    def shift_by(self, offset: float) -> "MultiModalTrapezoidalFun":
        return MultiModalTrapezoidalFun(
            starts=self.starts,
            rise_ends=self.rise_ends,
            fall_starts=self.fall_starts,
            ends=self.ends,
            biases=self.biases,
            scales=self.scales,
            weights=self.weights,
            bias=self.bias + offset,
        )

    def scale_by(self, scale: float) -> "MultiModalTrapezoidalFun":
        return MultiModalTrapezoidalFun(
            starts=self.starts,
            rise_ends=self.rise_ends,
            fall_starts=self.fall_starts,
            ends=self.ends,
            biases=self.biases,
            scales=self.scales,
            weights=tuple(scale * w for w in self.weights),
            bias=self.bias * scale,
        )

    def minmax(self, input) -> tuple[float, float]:
        # Delegates to `AggregatingFun.minmax`, which is approximate for
        # continuous issues (dense grid sampling; see `_dense_nonmonotonic_minmax`)
        # since a mixture of bumps need not have a closed-form extremum.
        return self._agg().minmax(input)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        if issue.is_discrete():
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        raise NotImplementedError(
            "MultiModalTrapezoidalFun has no native GENIUS evaluator type; only "
            "discrete issues can be exported to XML."
        )

    def __call__(self, x) -> float:
        return self._agg()(x)


@define(frozen=True)
class MultiModalGaussianFun(BaseFun):
    r"""A mixture (weighted sum) of several `GaussianFun`, for multi-modal values.

    Equivalent to building an `AggregatingFun` out of several `GaussianFun`
    instances, but takes each `GaussianFun` parameter as a tuple (one entry
    per component) plus a ``weights`` tuple, which is more convenient when
    all components are Gaussian. This is the standard way to build a
    multi-modal (multi-peak) value function out of Gaussian bumps.

    Note:
        `minmax` is approximate for continuous issues (a mixture of several
        bumps need not have a closed-form extremum): it delegates to
        `AggregatingFun.minmax`, which samples the issue's domain on a dense
        grid.

    Mathematical definition:

    .. math::

        f(x) = b + \sum_{i=1}^{n} w_i \cdot \left(b_i + k_i \exp\left(
            -\frac{(x - c_i)^2}{2 \sigma_i^2}\right)\right)

    where each component is a `GaussianFun` built from the i-th entries of
    ``centers``, ``sigmas``, ``biases`` and ``scales``.

    Args:
        centers: Tuple of ``center`` values, one per component (peak).
        sigmas: Tuple of ``sigma`` values, one per component.
        biases: Tuple of per-component biases (default: 0 each).
        scales: Tuple of per-component scales (default: 1 each).
        weights: Tuple of weights multiplying each component before summing
            (default: 1 each).
        bias: Overall constant offset added after summing (default: 0).

    Example:
        >>> f = MultiModalGaussianFun(centers=(2.0, 8.0), sigmas=(0.5, 0.5))
        >>> abs(f(2.0) - 1.0) < 1e-9  # peak of the first bump
        True
        >>> abs(f(8.0) - 1.0) < 1e-9  # peak of the second bump
        True
        >>> f(5.0) < 0.1  # far from both peaks
        True
    """

    centers: tuple[float, ...]
    sigmas: tuple[float, ...]
    biases: tuple[float, ...] | None = None
    scales: tuple[float, ...] | None = None
    weights: tuple[float, ...] | None = None
    bias: float = 0

    def __attrs_post_init__(self):
        n = len(self.centers)
        if len(self.sigmas) != n:
            raise ValueError(f"sigmas must have length {n} (same as centers)")
        if self.biases is None:
            object.__setattr__(self, "biases", tuple(0.0 for _ in range(n)))
        if self.scales is None:
            object.__setattr__(self, "scales", tuple(1.0 for _ in range(n)))
        if self.weights is None:
            object.__setattr__(self, "weights", tuple(1.0 for _ in range(n)))
        for name, seq in (
            ("biases", self.biases),
            ("scales", self.scales),
            ("weights", self.weights),
        ):
            if len(seq) != n:
                raise ValueError(f"{name} must have length {n} (same as centers)")

    def _components(self) -> tuple[GaussianFun, ...]:
        return tuple(
            GaussianFun(center=c, sigma=s, bias=b, scale=sc)
            for c, s, b, sc in zip(self.centers, self.sigmas, self.biases, self.scales)
        )

    def _agg(self) -> AggregatingFun:
        return AggregatingFun(
            funs=self._components(), weights=self.weights, bias=self.bias
        )

    def shift_by(self, offset: float) -> "MultiModalGaussianFun":
        return MultiModalGaussianFun(
            centers=self.centers,
            sigmas=self.sigmas,
            biases=self.biases,
            scales=self.scales,
            weights=self.weights,
            bias=self.bias + offset,
        )

    def scale_by(self, scale: float) -> "MultiModalGaussianFun":
        return MultiModalGaussianFun(
            centers=self.centers,
            sigmas=self.sigmas,
            biases=self.biases,
            scales=self.scales,
            weights=tuple(scale * w for w in self.weights),
            bias=self.bias * scale,
        )

    def minmax(self, input) -> tuple[float, float]:
        # Delegates to `AggregatingFun.minmax`, which is approximate for
        # continuous issues (dense grid sampling; see `_dense_nonmonotonic_minmax`)
        # since a mixture of bumps need not have a closed-form extremum.
        return self._agg().minmax(input)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        if issue.is_discrete():
            vals = list(issue.all)
            return TableFun(dict(zip(vals, [self(_) for _ in vals]))).xml(
                indx, issue, bias
            )
        raise NotImplementedError(
            "MultiModalGaussianFun has no native GENIUS evaluator type; only "
            "discrete issues can be exported to XML."
        )

    def __call__(self, x) -> float:
        return self._agg()(x)


@define(frozen=True)
class TableMultiFun(BaseMultiFun):
    r"""Table-based multi-issue value function using dictionary lookup.

    Maps tuples of issue values to real numbers using a dictionary.
    This is the most flexible multi-issue function, suitable for modeling
    arbitrary interactions between issues.

    Mathematical definition:

    .. math::

        f(\mathbf{x}) = T[\mathbf{x}]

    where :math:`T` is the mapping dictionary and :math:`\mathbf{x}` is a
    tuple of issue values.

    Args:
        mapping: Dictionary mapping value tuples to utility values.

    Example:
        >>> f = TableMultiFun(
        ...     mapping={
        ...         ("red", "large"): 1.0,
        ...         ("red", "small"): 0.8,
        ...         ("blue", "large"): 0.6,
        ...         ("blue", "small"): 0.4,
        ...     }
        ... )
        >>> f(("red", "large"))
        1.0
        >>> f(("blue", "small"))
        0.4

    Note:
        Raises KeyError if called with a value tuple not in the mapping.
    """

    mapping: dict[tuple, Any]

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the min/max values by sampling all value combinations."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return nonmonotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> TableMultiFun:
        """Return a new TableMultiFun with all mapped values shifted by offset."""
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] + offset
        return TableMultiFun(d)

    def scale_by(self, scale: float) -> TableMultiFun:
        """Return a new TableMultiFun with all mapped values scaled."""
        d = dict()
        for k in self.mapping.keys():
            d[k] = self.mapping[k] * scale
        return TableMultiFun(d)

    @property
    def dim(self) -> int:
        """Return the number of issues from the first mapping key's tuple length."""
        if not len(self.mapping):
            raise ValueError("Unkonwn dictionary in TableMultiFun")
        return len(list(self.mapping.keys())[0])

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Export to GENIUS XML format (not implemented for multi-issue tables)."""
        raise NotImplementedError()

    def __call__(self, x):
        """Look up the value tuple x in the mapping dictionary."""
        return self.mapping[x]


@define(frozen=True)
class AffineMultiFun(BaseMultiFun):
    r"""Affine multi-issue value function: f(x) = sum(slope[i] * x[i]) + bias.

    Maps tuples of numeric issue values to utilities using an affine combination
    (weighted sum plus constant). This models additive interactions between issues.

    Mathematical definition:

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} a_i x_i + b

    where :math:`a_i` are the slopes (weights) for each issue, :math:`x_i` are the
    issue values, and :math:`b` is the bias.

    Args:
        slope: Tuple of coefficients :math:`(a_1, a_2, ..., a_n)` for each issue.
        bias: Constant offset :math:`b` (default: 0).

    Example:
        >>> f = AffineMultiFun(slope=(0.5, 0.3), bias=0.1)
        >>> f((10, 20))  # 0.5*10 + 0.3*20 + 0.1 = 11.1
        11.1

    Note:
        This is equivalent to ``LinearMultiFun`` when ``bias=0``.
    """

    slope: tuple[float, ...]
    bias: float = 0

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the min/max values over the monotonic affine function."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        """Return a new AffineMultiFun with the bias shifted by offset."""
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> AffineMultiFun:
        """Return a new AffineMultiFun with all slopes and bias scaled."""
        return AffineMultiFun(
            slope=tuple(scale * _ for _ in self.slope), bias=self.bias * scale
        )

    @property
    def dim(self) -> int:
        """Return the number of issues (not implemented)."""
        raise NotImplementedError()

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Export to GENIUS XML format (not implemented)."""
        raise NotImplementedError()

    def __call__(self, x: tuple):
        """Evaluate sum(slope[i] * x[i]) + bias for the given value tuple."""
        return reduce(add, [a * b for a, b in zip(self.slope, x)], self.bias)


@define(frozen=True)
class LinearMultiFun(BaseMultiFun):
    r"""Linear multi-issue value function: f(x) = sum(slope[i] * x[i]).

    Maps tuples of numeric issue values to utilities using a weighted sum
    (no constant offset).

    Mathematical definition:

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} a_i x_i

    where :math:`a_i` are the slopes (weights) for each issue and :math:`x_i`
    are the issue values.

    Args:
        slope: Tuple of coefficients :math:`(a_1, a_2, ..., a_n)` for each issue.

    Example:
        >>> f = LinearMultiFun(slope=(0.5, 0.3))
        >>> f((10, 20))  # 0.5*10 + 0.3*20 = 11
        11.0

    Note:
        This is equivalent to ``AffineMultiFun(slope=slope, bias=0)``.
        Use ``AffineMultiFun`` if you need a non-zero constant term.
    """

    slope: tuple[float, ...]

    @property
    def bias(self):
        """Return the constant offset (always 0 for LinearMultiFun)."""
        return 0

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the min/max values over the monotonic linear function."""
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return monotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> AffineMultiFun:
        """Return an AffineMultiFun with this slope and the given offset as bias."""
        return AffineMultiFun(slope=self.slope, bias=self.bias + offset)

    def scale_by(self, scale: float) -> LinearMultiFun:
        """Return a new LinearMultiFun with all slopes scaled."""
        return LinearMultiFun(slope=tuple(scale * _ for _ in self.slope))

    @property
    def dim(self) -> int:
        """Return the number of issues (not implemented)."""
        raise NotImplementedError()

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Export to GENIUS XML format (not implemented)."""
        raise NotImplementedError()

    def __call__(self, x: tuple):
        """Evaluate sum(slope[i] * x[i]) for the given value tuple."""
        return reduce(add, [a * b for a, b in zip(self.slope, x)], 0)


@define(frozen=True)
class LambdaMultiFun(BaseMultiFun):
    r"""Lambda-based multi-issue value function using an arbitrary callable.

    Wraps any callable (function, lambda, etc.) as a multi-issue value function.
    This provides maximum flexibility for modeling complex interactions between
    issues.

    Mathematical definition:

    .. math::

        f(\mathbf{x}) = g(\mathbf{x}) + b

    where :math:`g` is the user-provided callable and :math:`b` is an optional bias.

    Args:
        f: A callable that takes a tuple of issue values and returns a float.
        bias: Constant offset added to the callable's output (default: 0).
        min_value: Known minimum value (optional, for optimization).
        max_value: Known maximum value (optional, for optimization).

    Example:
        >>> # Model interaction: value is product of issue values
        >>> f = LambdaMultiFun(f=lambda x: x[0] * x[1])
        >>> f((3, 4))  # 3 * 4 = 12
        12

        >>> # With bias
        >>> f = LambdaMultiFun(f=lambda x: x[0] * x[1], bias=1.0)
        >>> f((3, 4))  # 3 * 4 + 1 = 13
        13.0

    Warning:
        Lambda functions can be serialized, but complex callables
        (e.g., closures with external state) may not serialize correctly.
    """

    f: Callable[[Any], float]
    bias: float = 0
    min_value: float | None = None
    max_value: float | None = None

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the min/max values, using cached bounds if available."""
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
        """Shift operation (not implemented for lambda functions)."""
        raise NotImplementedError()

    def scale_by(self, scale: float) -> LinearMultiFun:
        """Scale operation (not implemented for lambda functions)."""
        raise NotImplementedError()

    @property
    def dim(self) -> int:
        """Return the number of issues (not implemented)."""
        raise NotImplementedError()

    def xml(self, indx, issues, bias: float = 0) -> str:
        """Export to GENIUS XML format (not implemented)."""
        raise NotImplementedError()

    def __call__(self, x: Any) -> float:
        """Evaluate the wrapped function at value tuple x and add the bias."""
        return self.f(x) + self.bias


@define(frozen=True)
class QuadraticMultiFun(BaseMultiFun):
    r"""Quadratic multi-issue value function with cross-terms.

    Maps tuples of numeric issue values to utilities using a full quadratic form
    including linear terms, squared terms, and all pairwise interaction terms.

    Mathematical definition:

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} a_i x_i + \sum_{i=1}^{n} b_i x_i^2
                        + \sum_{i=1}^{n} \sum_{j=i+1}^{n} c_{ij} x_i x_j + d

    where:
        - :math:`a_i` are the linear coefficients
        - :math:`b_i` are the quadratic (squared) coefficients
        - :math:`c_{ij}` are the interaction coefficients for pairs (i, j)
        - :math:`d` is the bias (constant term)

    Args:
        linear: Tuple of linear coefficients :math:`(a_1, ..., a_n)`.
        quadratic: Tuple of quadratic coefficients :math:`(b_1, ..., b_n)` for squared terms.
        interactions: Tuple of interaction coefficients for pairs (0,1), (0,2), ..., (n-2,n-1).
            Length should be n*(n-1)/2 for n issues.
        bias: Constant offset :math:`d` (default: 0).

    Example:
        >>> # f(x,y) = 2x + 3y + x^2 + 2y^2 + 4xy + 1
        >>> f = QuadraticMultiFun(
        ...     linear=(2.0, 3.0), quadratic=(1.0, 2.0), interactions=(4.0,), bias=1.0
        ... )
        >>> f((1, 1))  # 2*1 + 3*1 + 1*1 + 2*1 + 4*1*1 + 1 = 13
        13.0
        >>> f((2, 3))  # 2*2 + 3*3 + 1*4 + 2*9 + 4*2*3 + 1 = 4+9+4+18+24+1 = 60
        60.0

    Note:
        For two issues, this is equivalent to the full bivariate quadratic:
        :math:`f(x, y) = a_1 x + a_2 y + b_1 x^2 + b_2 y^2 + c_{12} xy + d`
    """

    linear: tuple[float, ...]
    quadratic: tuple[float, ...]
    interactions: tuple[float, ...]
    bias: float = 0

    @property
    def dim(self) -> int:
        """Return the number of issues this function operates on.

        Returns:
            int: The dimensionality (number of issues).
        """
        return len(self.linear)

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the minimum and maximum values over the issues' domains.

        Args:
            input: The issues whose combined domain to evaluate over.

        Returns:
            tuple[float, float]: A tuple of (minimum, maximum) values.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return nonmonotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> "QuadraticMultiFun":
        """Create a new function shifted by a constant offset.

        Args:
            offset: The amount to add to all output values.

        Returns:
            QuadraticMultiFun: A new function where f'(x) = f(x) + offset.
        """
        return QuadraticMultiFun(
            linear=self.linear,
            quadratic=self.quadratic,
            interactions=self.interactions,
            bias=self.bias + offset,
        )

    def scale_by(self, scale: float) -> "QuadraticMultiFun":
        """Create a new function scaled by a constant factor.

        Args:
            scale: The factor to multiply all output values by.

        Returns:
            QuadraticMultiFun: A new function where f'(x) = scale * f(x).
        """
        return QuadraticMultiFun(
            linear=tuple(scale * c for c in self.linear),
            quadratic=tuple(scale * c for c in self.quadratic),
            interactions=tuple(scale * c for c in self.interactions),
            bias=self.bias * scale,
        )

    def __call__(self, x: tuple) -> float:
        """Evaluate the function at a tuple of values.

        Args:
            x: Tuple of input values, one per issue.

        Returns:
            float: The function value at x.
        """
        n = len(x)
        result = self.bias

        # Linear and quadratic terms
        for i in range(n):
            result += self.linear[i] * x[i]
            result += self.quadratic[i] * x[i] * x[i]

        # Interaction terms (upper triangular)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                result += self.interactions[idx] * x[i] * x[j]
                idx += 1

        return result


@define(frozen=True)
class BilinearMultiFun(BaseMultiFun):
    r"""Bilinear two-issue value function: f(x, y) = a*x + b*y + c*x*y + d.

    A specialized function for exactly two issues with a bilinear interaction term.
    This is simpler than `QuadraticMultiFun` when you only need linear terms and
    their product (no squared terms).

    Mathematical definition:

    .. math::

        f(x, y) = a x + b y + c x y + d

    where :math:`a` and :math:`b` are linear coefficients, :math:`c` is the
    interaction coefficient, and :math:`d` is the bias.

    Args:
        a: Coefficient for the first issue :math:`x`.
        b: Coefficient for the second issue :math:`y`.
        c: Interaction coefficient for :math:`xy`.
        bias: Constant offset :math:`d` (default: 0).

    Example:
        >>> f = BilinearMultiFun(a=2.0, b=3.0, c=1.0, bias=0.0)
        >>> f((1, 1))  # 2*1 + 3*1 + 1*1*1 = 6
        6.0
        >>> f((2, 3))  # 2*2 + 3*3 + 1*2*3 = 4 + 9 + 6 = 19
        19.0

    Note:
        This is a special case of `QuadraticMultiFun` with quadratic=(0,0).
        Use `QuadraticMultiFun` if you need squared terms.
    """

    a: float
    b: float
    c: float
    bias: float = 0

    @property
    def dim(self) -> int:
        """Return the number of issues (always 2).

        Returns:
            int: Always returns 2.
        """
        return 2

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the minimum and maximum values over the issues' domains.

        Args:
            input: The issues whose combined domain to evaluate over.

        Returns:
            tuple[float, float]: A tuple of (minimum, maximum) values.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return nonmonotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> "BilinearMultiFun":
        """Create a new function shifted by a constant offset.

        Args:
            offset: The amount to add to all output values.

        Returns:
            BilinearMultiFun: A new function where f'(x) = f(x) + offset.
        """
        return BilinearMultiFun(a=self.a, b=self.b, c=self.c, bias=self.bias + offset)

    def scale_by(self, scale: float) -> "BilinearMultiFun":
        """Create a new function scaled by a constant factor.

        Args:
            scale: The factor to multiply all output values by.

        Returns:
            BilinearMultiFun: A new function where f'(x) = scale * f(x).
        """
        return BilinearMultiFun(
            a=self.a * scale, b=self.b * scale, c=self.c * scale, bias=self.bias * scale
        )

    def __call__(self, x: tuple) -> float:
        """Evaluate the function at a pair of values.

        Args:
            x: Tuple of two input values (x, y).

        Returns:
            float: The function value at (x, y).
        """
        return self.a * x[0] + self.b * x[1] + self.c * x[0] * x[1] + self.bias


@define(frozen=True)
class PolynomialMultiFun(BaseMultiFun):
    r"""General polynomial multi-issue value function.

    Maps tuples of numeric issue values to utilities using a multivariate polynomial.
    Each term is specified by its coefficient and the powers to apply to each issue.

    Mathematical definition:

    .. math::

        f(\mathbf{x}) = \sum_{k} c_k \prod_{i=1}^{n} x_i^{p_{ki}} + d

    where :math:`c_k` are the term coefficients, :math:`p_{ki}` are the powers
    for each issue in term k, and :math:`d` is the bias.

    Args:
        terms: Tuple of (coefficient, powers) pairs, where powers is a tuple of
            exponents for each issue. For example, ((2.0, (1, 0)), (3.0, (0, 1)))
            represents 2*x + 3*y.
        bias: Constant offset :math:`d` (default: 0).

    Example:
        >>> # f(x, y) = 2*x + 3*y + x^2*y + 1
        >>> f = PolynomialMultiFun(
        ...     terms=((2.0, (1, 0)), (3.0, (0, 1)), (1.0, (2, 1))), bias=1.0
        ... )
        >>> f((1, 1))  # 2*1 + 3*1 + 1*1*1 + 1 = 7
        7.0
        >>> f((2, 3))  # 2*2 + 3*3 + 4*3 + 1 = 4 + 9 + 12 + 1 = 26
        26.0

        >>> # Three-issue polynomial: f(x, y, z) = x*y + y*z + x*z
        >>> f = PolynomialMultiFun(
        ...     terms=((1.0, (1, 1, 0)), (1.0, (0, 1, 1)), (1.0, (1, 0, 1))), bias=0.0
        ... )
        >>> f((2, 3, 4))  # 2*3 + 3*4 + 2*4 = 6 + 12 + 8 = 26
        26.0

    Note:
        - Use powers of 0 to exclude an issue from a term
        - This is the most general polynomial form but may be slower than
          specialized classes like `QuadraticMultiFun` or `BilinearMultiFun`
    """

    terms: tuple[tuple[float, tuple[int, ...]], ...]
    bias: float = 0

    @property
    def dim(self) -> int:
        """Return the number of issues this function operates on.

        Returns:
            int: The dimensionality (number of issues).

        Raises:
            ValueError: If there are no terms to infer dimensionality from.
        """
        if not self.terms:
            raise ValueError("Cannot determine dimensionality with no terms")
        return len(self.terms[0][1])

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the minimum and maximum values over the issues' domains.

        Args:
            input: The issues whose combined domain to evaluate over.

        Returns:
            tuple[float, float]: A tuple of (minimum, maximum) values.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return nonmonotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> "PolynomialMultiFun":
        """Create a new function shifted by a constant offset.

        Args:
            offset: The amount to add to all output values.

        Returns:
            PolynomialMultiFun: A new function where f'(x) = f(x) + offset.
        """
        return PolynomialMultiFun(terms=self.terms, bias=self.bias + offset)

    def scale_by(self, scale: float) -> "PolynomialMultiFun":
        """Create a new function scaled by a constant factor.

        Args:
            scale: The factor to multiply all output values by.

        Returns:
            PolynomialMultiFun: A new function where f'(x) = scale * f(x).
        """
        return PolynomialMultiFun(
            terms=tuple((scale * coef, powers) for coef, powers in self.terms),
            bias=self.bias * scale,
        )

    def __call__(self, x: tuple) -> float:
        """Evaluate the function at a tuple of values.

        Args:
            x: Tuple of input values, one per issue.

        Returns:
            float: The function value at x.
        """
        result = self.bias
        for coef, powers in self.terms:
            term = coef
            for i, p in enumerate(powers):
                if p != 0:
                    term *= pow(x[i], p)
            result += term
        return result


@define(frozen=True)
class ProductMultiFun(BaseMultiFun):
    r"""Product multi-issue value function: f(x) = scale * prod(x[i]^powers[i]) + bias.

    Maps tuples of numeric issue values to utilities using a scaled product
    with optional powers for each issue.

    Mathematical definition:

    .. math::

        f(\mathbf{x}) = s \cdot \prod_{i=1}^{n} x_i^{p_i} + b

    where :math:`s` is the scale, :math:`p_i` are the powers for each issue,
    and :math:`b` is the bias.

    Args:
        powers: Tuple of exponents :math:`(p_1, ..., p_n)` for each issue.
            Default power of 1.0 for all issues if not specified.
        scale: Multiplicative scale :math:`s` (default: 1.0).
        bias: Constant offset :math:`b` (default: 0).

    Example:
        >>> # Simple product: f(x, y) = x * y
        >>> f = ProductMultiFun(powers=(1.0, 1.0))
        >>> f((3, 4))  # 3 * 4 = 12
        12.0

        >>> # Cobb-Douglas style: f(x, y) = x^0.5 * y^0.5
        >>> f = ProductMultiFun(powers=(0.5, 0.5))
        >>> abs(f((4, 9)) - 6.0) < 1e-10  # 4^0.5 * 9^0.5 = 2 * 3 = 6
        True

        >>> # With scale and bias: f(x, y) = 2 * x * y + 1
        >>> f = ProductMultiFun(powers=(1.0, 1.0), scale=2.0, bias=1.0)
        >>> f((3, 4))  # 2 * 3 * 4 + 1 = 25
        25.0

    Note:
        This is useful for Cobb-Douglas utility functions commonly used in
        economics: :math:`U(x, y) = x^a y^{1-a}`.
    """

    powers: tuple[float, ...]
    scale: float = 1.0
    bias: float = 0

    @property
    def dim(self) -> int:
        """Return the number of issues this function operates on.

        Returns:
            int: The dimensionality (number of issues).
        """
        return len(self.powers)

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the minimum and maximum values over the issues' domains.

        Args:
            input: The issues whose combined domain to evaluate over.

        Returns:
            tuple[float, float]: A tuple of (minimum, maximum) values.
        """
        return self._minmax(input)

    @lru_cache
    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        return nonmonotonic_multi_minmax(input, self)

    def shift_by(self, offset: float) -> "ProductMultiFun":
        """Create a new function shifted by a constant offset.

        Args:
            offset: The amount to add to all output values.

        Returns:
            ProductMultiFun: A new function where f'(x) = f(x) + offset.
        """
        return ProductMultiFun(
            powers=self.powers, scale=self.scale, bias=self.bias + offset
        )

    def scale_by(self, scale: float) -> "ProductMultiFun":
        """Create a new function scaled by a constant factor.

        Args:
            scale: The factor to multiply all output values by.

        Returns:
            ProductMultiFun: A new function where f'(x) = scale * f(x).
        """
        return ProductMultiFun(
            powers=self.powers, scale=self.scale * scale, bias=self.bias * scale
        )

    def __call__(self, x: tuple) -> float:
        """Evaluate the function at a tuple of values.

        Args:
            x: Tuple of input values, one per issue.

        Returns:
            float: The function value at x.
        """
        result = self.scale
        for i, p in enumerate(self.powers):
            result *= pow(x[i], p)
        return result + self.bias


def make_fun_from_xml(item) -> tuple[BaseFun, str]:
    """Parse a GENIUS XML evaluator element and return the corresponding value function."""
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
