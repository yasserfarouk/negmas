#!/usr/bin/env python
"""
A set of utilities to handle probability distributions
"""
from __future__ import annotations

import copy
import numbers
import random

import numpy as np
import scipy.stats as stats

from negmas.common import Distribution

__all__ = [
    "Distribution",  # THe interface of a distribution class
    "ScipyDistribution",  # A probability distribution using scipy stats
    "NormalDistribution",  # A probability distribution using scipy stats
    "UniformDistribution",  # A probability distribution using scipy stats
    "Real",
    "make_distribution",
]

EPSILON = 1e-8


class Real(Distribution):
    """A real number implementing the `Distribution` interface"""

    def __init__(
        self,
        loc: int | float | numbers.Real = 0.0,
        type: str = "",
        scale: float = 0.0,
        **kwargs,
    ):
        loc = float(loc)
        if not (0 <= scale <= EPSILON):
            raise ValueError(
                f"Cannot construct a Real with this scale ({scale}) It must be zero"
            )
        self._loc: float = loc
        self._scale = 0.0
        self._type = type if type else "uniform"

    @property
    def loc(self) -> float:
        """Returns the location of the distributon (usually mean)"""
        return self._loc

    def __copy__(self):
        """Copies the distribution"""
        return Real(type="", loc=self.loc)

    def __deepcopy__(self, memo):
        """Performs deep copying"""
        return Real(loc=self.loc)
        # todo: check that this works. I think it does not
        # super().__init__(type=type if type else "uniform")

    def type(self) -> str:
        return self._type

    def mean(self) -> float:
        """Finds the mean"""
        return self._loc

    def prob(self, val: float) -> float:
        """Returns the probability for the given value"""
        return 1.0 if self == val < EPSILON else 0.0

    def cum_prob(self, mn: float, mx: float) -> float:
        """Returns the probability for the given range"""
        return int(mn <= self.loc <= mx)

    def sample(self, size: int = 1) -> np.ndarray:
        """Samples `size` elements from the distribution"""
        return self._loc * np.ones(size)

    @property
    def scale(self) -> float:
        """Returns the scale of the distribution (may be std. dev.)"""
        return 0.0

    @property
    def min(self) -> float:
        """Returns the minimum"""
        return self._loc

    @property
    def max(self) -> float:
        """Returns the maximum"""
        return self._loc

    def is_gaussian(self):
        return True

    def is_uniform(self):
        return True

    def is_crisp(self) -> bool:
        """Returns true if this is a distribution with all probability at one point (delta(v))"""
        return self.scale < EPSILON / 1000

    def __add__(self, other):
        """Returns the distribution for the sum of samples of `self` and `other`"""
        if isinstance(other, float):
            return super().__add__(other)
        return other.__class__(loc=other.loc + self._loc, scale=other.scale)

    def __sub__(self, other):
        """Returns the distribution for the difference between samples of `self` and `other`"""
        if isinstance(other, float):
            return super().__sub__(other)
        return other.__class__(loc=self.loc - other.loc, scale=other.scale)

    def __mul__(self, other):
        """Returns the distribution for the sum of samples of `self` and `other`"""
        if isinstance(other, float):
            return self._loc * other
        return other * self._loc

    def __lt__(self, other):
        """Check that a sample from `self` is ALWAYS less than a sample from other `other`"""
        if isinstance(other, float):
            return super().__lt__(other)
        return self.max < other.min

    def __le__(self, other):
        """Check that a sample from `self` is ALWAYS less or equal a sample from other `other`"""
        if isinstance(other, float):
            return super().__le__(other)
        return self.max <= other.min

    def __eq__(self, other):
        """Checks for equality of the two distributions"""
        if isinstance(other, Distribution):
            return other.scale < EPSILON and abs(other.loc - self.loc) < EPSILON
        if isinstance(other, float):
            return abs(self.loc - other) < EPSILON
        raise ValueError(f"Cannot compare CrispValues with {type(other)}")

    def __ne__(self, other):
        """Checks for ineqlaity of the distributions"""
        return not self == other

    def __gt__(self, other):
        """Check that a sample from `self` is ALWAYS greater than a sample from other `other`"""
        if isinstance(other, float):
            return super().__gt__(other)
        return self.min > other.max

    def __ge__(self, other):
        """Check that a sample from `self` is ALWAYS greater or equal a sample from other `other`"""
        if isinstance(other, float):
            return super().__ge__(other)
        return self.min >= other.max

    def __str__(self):
        """Converts to a readable string"""
        return str(float(self._loc))

    __repr__ = __str__


def make_distribution(x: int | float | numbers.Real | Distribution) -> Distribution:
    """
    Ensures the output is `Distribution`

    Remarks:
        The input can either be `Distribution` or a number

    """
    if not isinstance(x, Distribution):
        return Real(x)
    return x


class ScipyDistribution(Distribution):
    """
    Any distribution from scipy.stats with overloading of addition and multiplication.

    Args:
            type (str): Data type of the distribution as a string. It must be one defined in `scipy.stats`
            loc (float): The location of the distribution (corresponds to mean in Gaussian)
            scale (float): The _scale of the distribution (corresponds to standard deviation in Gaussian)
            **kwargs:

    Examples:

        >>> d2 = ScipyDistribution('uniform')
        >>> print(d2.mean())
        0.5

        >>> try:
        ...     d = ScipyDistribution('something crazy')
        ... except ValueError as e:
        ...     print(str(e))
        Unknown distribution something crazy

    """

    def __init__(self, type: str, **kwargs) -> None:
        dist = getattr(stats, type.lower(), None)
        if dist is None:
            raise ValueError(f"Unknown distribution {type}")
        if "loc" not in kwargs.keys():
            kwargs["loc"] = 0.0
        if "scale" not in kwargs.keys():
            kwargs["scale"] = 1.0

        self._dist = dist(**kwargs)
        self._type = type

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def type(self) -> str:
        return self._type

    def _make_dist(self, type: str, loc: float, scale: float):
        dist = getattr(stats, type.lower(), None)
        if dist is None:
            raise ValueError(f"Unknown distribution {type}")
        return dist(loc=loc, scale=scale)

    @property
    def loc(self):
        return self._dist.kwds.get("loc", 0.5)

    @property
    def scale(self):
        return self._dist.kwds.get("scale", 0.0)

    def mean(self) -> float:
        if self._type != "uniform":
            raise NotImplementedError(
                "Only uniform distributions are supported for now"
            )
        if self.scale < 1e-6:
            return float(self.loc)
        mymean = self._dist.mean()
        return float(mymean)

    def prob(self, val: float) -> float:
        """Returns the probability for the given value"""
        return self._dist.prob(val)

    def cum_prob(self, mn: float, mx: float) -> float:
        """Returns the probability for the given range"""
        return self._dist.cdf(mx) - self._dist.cdf(mn)

    def sample(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)

    @property
    def min(self):
        return self.loc - self.scale

    @property
    def max(self):
        return self.loc + self.scale

    def is_gaussian(self):
        return self._type == "normal"

    def is_uniform(self):
        return self._type == "uniform"

    def is_crisp(self) -> bool:
        """Returns true if this is a distribution with all probability at one point (delta(v))"""
        return self.scale < EPSILON / 1000

    def __call__(self, val: float) -> float:
        """Returns the probability for the given value"""
        return self.prob(val)

    def __add__(self, other):
        if isinstance(other, float) and self._type in ("uniform", "normal"):
            self._dist = self._make_dist(self._type, self.loc + other, self.scale)
        raise NotImplementedError()

    def __sub__(self, other):
        if isinstance(other, float) and self._type in ("uniform", "normal"):
            self._dist = self._make_dist(self._type, self.loc - other, self.scale)
        raise NotImplementedError()

    def __mul__(self, weight: float):
        if isinstance(weight, float) and self._type in ("uniform", "normal"):
            self._dist = self._make_dist(
                self._type, self.loc * weight, self.scale * weight
            )
        raise NotImplementedError()

    def __lt__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS less than a sample from other `other`"""
        if isinstance(other, numbers.Real):
            return self.max < float(other)
        return self.max < other.min

    def __le__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS less or equal a sample from other `other`"""
        return self < other or self == other

    def __eq__(self, other):
        if isinstance(other, ScipyDistribution):
            return (
                self._type == other._type
                and abs(self.loc - other.loc) < EPSILON
                and abs(self.scale - other.scale) < EPSILON
            )

        if isinstance(other, float):
            return abs(self.loc - other) < EPSILON and self.loc < EPSILON
        raise ValueError(f"Cannot compare Distribution with {type(other)}")

    def __eq__(self, other) -> bool:
        """Checks for equality of the two distributions"""
        if isinstance(other, numbers.Real):
            other = Real(other)
        return (
            self.loc == other.loc
            and self.scale == other.scale
            and (
                isinstance(other, Real)
                or isinstance(self, Real)
                or self.type == other.type
            )
        )

    def __ne__(self, other) -> bool:
        """Checks for ineqlaity of the distributions"""
        return not (self == other)

    def __gt__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS greater than a sample from other `other`"""
        if isinstance(other, numbers.Real):
            return self.min > other
        return self.min > other.max

    def __ge__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS greater or equal a sample from other `other`"""
        return self > other or self == other

    def __float__(self) -> float:
        """Converts to a float (usually by calling mean())"""
        return self.mean()

    def __str__(self):
        if self._type == "uniform":
            return f"U({self.loc}, {self.loc+self.scale})"
        if self._type == "normal":
            return f"G({self.loc}, {self.loc+self.scale})"
        return f"{self._type}(loc:{self.loc}, scale:{self.scale})"

    __repr__ = __str__


class UniformDistribution(ScipyDistribution):
    """A `ScipyDistribution` reprsenting a unifrom distribution"""

    def __init__(
        self, loc: float = 0.0, scale: float = 1.0, *, type: str = "uniform", **kwargs
    ) -> None:
        super().__init__(loc=loc, scale=scale, type="uniform", **kwargs)


class NormalDistribution(ScipyDistribution):
    """A `ScipyDistribution` reprsenting a unifrom distribution"""

    def __init__(
        self, loc: float = 0.0, scale: float = 1.0, *, type: str = "norm", **kwargs
    ) -> None:
        super().__init__(loc=loc, scale=scale, type="norm", **kwargs)


def uniform_around(
    value: float = 0.5,
    range: tuple[float, float] = (0.0, 1.0),
    uncertainty: float = 0.5,
    cls: type[Distribution] = ScipyDistribution,
) -> Distribution:
    """
    Generates a uniform distribution around the input value in the given range with given uncertainty

    Args:
        value: The value to generate the distribution around
        range: The range of possible values
        uncertainty: The uncertainty level required. 0.0 means no uncertainty and 1.0 means full uncertainty

    Returns:
        Distribution A uniform distribution around `value` with uncertainty (scale) `uncertainty`
    """
    if uncertainty >= 1.0:
        return cls(type="uniform", loc=range[0], scale=range[1])
    if uncertainty <= 0.0:
        return Real(loc=value)
    scale = uncertainty * (range[1] - range[0])
    loc = max(range[0], (random.random() - 1.0) * scale + value)
    if loc + scale > range[1]:
        loc -= loc + scale - range[1]
    return cls(type="uniform", loc=loc, scale=scale)
