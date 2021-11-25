#!/usr/bin/env python
"""
A set of utilities to handle probability distributions
"""
from __future__ import annotations

import copy
import random
from typing import Protocol

import numpy as np
import scipy.stats as stats

__all__ = [
    "DistributionProtocol",  # THe interface of a distribution class
    "Distribution",  # A probability distribution
    "Value",
]

Value = float
"""Crisp values are real numbers"""


class DistributionProtocol(Protocol):
    @classmethod
    def around(
        cls,
        value: float = 0.5,
        range: tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
    ) -> "Distribution":
        """
        Generates a uniform distribution around the input value in the given range with given uncertainty

        Args:
            value: The value to generate the distribution around
            range: The range of possible values
            uncertainty: The uncertainty level required. 0.0 means no uncertainty and 1.0 means full uncertainty

        Returns:
            Distribution A uniform distribution around `value` with uncertainty (scale) `uncertainty`
        """

    def mean(self) -> float:
        """Finds the mean"""

    def __float__(self):
        """Converts to a float (usually by calling mean())"""

    def __and__(self, other):
        """And operation"""

    def __or__(self, other):
        """Or operation"""

    def prob(self, val: float) -> float:
        """Returns the probability for the given value"""

    def sample(self, size: int = 1) -> np.ndarray:
        """Samples `size` elements from the distribution"""

    @property
    def loc(self):
        """Returns the location of the distributon (usually mean)"""

    @property
    def scale(self):
        """Returns the scale of the distribution (may be std. dev.)"""

    def min(self):
        """Returns the minimum"""

    def max(self):
        """Returns the maximum"""

    def __str__(self):
        """Converts to a readable string"""

    def __copy__(self):
        """Copies the distribution"""

    def __deepcopy__(self, memo):
        """Performs deep copying"""

    def __eq__(self, other):
        """Checks for equality of the two distributions"""

    def __ne__(self, other):
        """Checks for ineqlaity of the distributions"""

    def __lt__(self, other):
        """Check that a sample from `self` is ALWAYS less than a sample from other `other`"""

    def __le__(self, other):
        """Check that a sample from `self` is ALWAYS less or equal a sample from other `other`"""

    def __gt__(self, other):
        """Check that a sample from `self` is ALWAYS greater than a sample from other `other`"""

    def __ge__(self, other):
        """Check that a sample from `self` is ALWAYS greater or equal a sample from other `other`"""

    def __sub__(self, other):
        """Returns the distribution for the difference between samples of `self` and `other`"""

    def __add__(self, other):
        """Returns the distribution for the sum of samples of `self` and `other`"""

    def __radd__(self, other):
        """Returns the distribution for the sum of samples of `self` and `other`"""

    def __mul__(self, other):
        """Returns the distribution for the multiplication of samples of `self` and `other`"""

    def __rmul__(self, other):
        """Returns the distribution for the multiplication of samples of `self` and `other`"""

    def __divmod__(self, other):
        """Returns the distribution for the divition of samples of `self` and `other`"""


class Distribution:
    """
    Any distribution from scipy.stats with overloading of addition and multiplication.

    Args:
            type (str): Data type of the distribution as a string. It must be one defined in `scipy.stats`
            loc (float): The location of the distribution (corresponds to mean in Gaussian)
            scale (float): The _scale of the distribution (corresponds to standard deviation in Gaussian)
            **kwargs:

    Examples:

        >>> d2 = Distribution('uniform')
        >>> print(d2.mean())
        0.5

        >>> try:
        ...     d = Distribution('something crazy')
        ... except ValueError as e:
        ...     print(str(e))
        Unknown distribution something crazy

    """

    def __init__(self, type: str, **kwargs) -> None:
        super().__init__()
        dist = getattr(stats, type.lower(), None)
        if dist is None:
            raise ValueError(f"Unknown distribution {type}")
        if "loc" not in kwargs.keys():
            kwargs["loc"] = 0.0
        if "scale" not in kwargs.keys():
            kwargs["scale"] = 1.0

        self._dist = dist(**kwargs)
        self._type = type

    @classmethod
    def uniform_around(
        cls,
        value: float = 0.5,
        range: tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
    ) -> "Distribution":
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
            return cls(type="uniform", loc=value, scale=0.0)
        scale = uncertainty * (range[1] - range[0])
        loc = max(range[0], (random.random() - 1.0) * scale + value)
        if loc + scale > range[1]:
            loc -= loc + scale - range[1]
        return cls(type="uniform", loc=loc, scale=scale)

    def mean(self) -> float:
        if self._type != "uniform":
            raise NotImplementedError(
                "Only uniform distributions are supported for now"
            )
        if self.scale < 1e-6:
            return self.loc
        mymean = self._dist.mean()
        return float(mymean)

    def __float__(self):
        return float(self.mean())

    def __and__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return float(other)
        if self._type == "uniform":
            beg = max(self.loc, other.loc)
            end = min(self.scale + self.loc, other.loc + other.scale)
            return Distribution(self._type, loc=beg, scale=end - beg)
        raise NotImplementedError()

    def __or__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return float(other)
        # if self.type == "uniform":
        #     raise NotImplementedError(
        #         "Current implementation assumes an overlap otherwise a mixture must be returned"
        #     )
        # beg = min(self.loc, other.loc)
        # end = max(self.scale + self.loc, other.loc + other.scale)
        # return Distribution(self.type, loc=beg, scale=end - beg)
        raise NotImplementedError()

    def prob(self, val: float) -> float:
        """Returns the probability for the given value"""
        return self._dist.prob(val)

    def sample(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)

    @property
    def loc(self):
        return self._dist.kwds.get("loc", 0.0)

    @property
    def scale(self):
        return self._dist.kwds.get("scale", 0.0)

    def min(self):
        return self.loc - self.scale

    def max(self):
        return self.loc + self.scale

    def __str__(self):
        if self._type == "uniform":
            return f"U({self.loc}, {self.loc+self.scale})"
        return f"{self._type}(loc:{self.loc}, scale:{self.scale})"

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

    __repr__ = __str__

    def __eq__(self, other):
        return float(self) == other

    def __ne__(self, other):
        return float(self) == other

    def __lt__(self, other):
        return float(self) == other

    def __le__(self, other):
        return float(self) == other

    def __gt__(self, other):
        return float(self) == other

    def __ge__(self, other):
        return float(self) == other

    def __sub__(self, other):
        return float(self) - other

    def __add__(self, other):
        return float(self) + other

    def __radd__(self, other):
        return float(self) + other

    def __mul__(self, other):
        return float(self) * float(other)

    def __rmul__(self, other):
        return float(other) * float(self)

    def __divmod__(self, other):
        return float(self).__divmod__(other)
