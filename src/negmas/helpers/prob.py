#!/usr/bin/env python
"""
A set of utilities to handle probability distributions
"""

from __future__ import annotations

import copy
import numbers
import random
from typing import TYPE_CHECKING

import numpy as np

from negmas.common import Distribution

if TYPE_CHECKING:
    pass

__all__ = [
    "Distribution",  # THe interface of a distribution class
    "ScipyDistribution",  # A probability distribution using scipy stats
    "NormalDistribution",  # A probability distribution using scipy stats
    "UniformDistribution",  # A probability distribution using scipy stats
    "Real",
    "make_distribution",
    "UNIFORM",
    "NORMAL",
    "canonical_distribution_type",
]

EPSILON = 1e-8

#: Canonical name of the continuous uniform distribution family. This is the
#: single source of truth for the string identifying a uniform distribution
#: throughout NegMAS (and downstream packages such as ``negmas-elicit``).
UNIFORM = "uniform"
#: Canonical name of the normal (Gaussian) distribution family. Single source
#: of truth for the string identifying a normal distribution.
NORMAL = "normal"

#: Case-insensitive aliases accepted for each canonical family name.
_UNIFORM_ALIASES = frozenset({"uniform", "unif", "u"})
_NORMAL_ALIASES = frozenset({"normal", "norm", "gaussian", "gauss", "n"})
#: Mapping from a canonical family name to the corresponding ``scipy.stats`` name
#: (``scipy`` calls the normal family ``"norm"`` and the uniform one ``"uniform"``).
_SCIPY_NAME = {UNIFORM: "uniform", NORMAL: "norm"}


def canonical_distribution_type(type: str) -> str:
    """Normalize a distribution family name to its canonical NegMAS constant.

    Accepts common aliases case-insensitively (e.g. ``"norm"``, ``"gaussian"``
    -> `NORMAL`; ``"unif"`` -> `UNIFORM`). Any unrecognized name is returned
    lower-cased and unchanged so that arbitrary ``scipy.stats`` families keep
    working.

    Args:
        type: The (possibly aliased) distribution family name.

    Returns:
        The canonical family name (`UNIFORM`, `NORMAL`, or the lower-cased input).
    """
    t = (type or "").strip().lower()
    if t in _UNIFORM_ALIASES:
        return UNIFORM
    if t in _NORMAL_ALIASES:
        return NORMAL
    return t


class Real(Distribution):
    """A real number implementing the `Distribution` interface"""

    def __init__(
        self,
        loc: int | float | numbers.Real = 0.0,
        type: str = "",
        scale: float = 0.0,
        **kwargs,
    ):
        """Initialize a Real distribution representing a single deterministic value.

        Args:
            loc: The location (value) of this deterministic distribution.
            type: Distribution type name (defaults to "uniform" for compatibility).
            scale: Must be zero or near-zero since Real represents a single point.
            **kwargs: Additional keyword arguments (ignored).
        """
        loc = float(loc)
        if not (0 <= scale <= EPSILON):
            raise ValueError(
                f"Cannot construct a Real with this scale ({scale}) It must be zero"
            )
        self._loc: float = loc
        self._scale = 0.0
        self._type = type if type else "uniform"

    def __float__(self) -> float:
        """Convert this distribution to a float by returning its location value."""
        return self.loc

    def __call__(self, val: float) -> float:
        """Return the probability density at the given value (1.0 if equal to loc, else 0.0)."""
        return 1.0 if abs(val - self.loc) < 1e-10 else 0.0

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
        """Return the distribution type name."""
        return self._type

    def mean(self) -> float:
        """Finds the mean"""
        return self._loc

    def prob(self, val: float) -> float:
        r"""Return the (degenerate) density of the point mass ``delta(v)`` at ``val``.

        Math: ``1`` when ``val == v`` (within `EPSILON`) and ``0`` otherwise, i.e.
        the indicator ``[|val - v| < EPSILON]``.
        """
        return 1.0 if abs(val - self._loc) < EPSILON else 0.0

    def cum_prob(self, mn: float, mx: float) -> float:
        r"""Return the mass of ``delta(v)`` inside ``[mn, mx]``.

        Math: the indicator ``[mn <= v <= mx]`` (``1`` if the point ``v`` lies in
        the closed interval, else ``0``).
        """
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
        """Check if gaussian."""
        return True

    def is_uniform(self):
        """Check if uniform."""
        return True

    def is_crisp(self) -> bool:
        """Returns true if this is a distribution with all probability at one point (delta(v))"""
        return self.scale < EPSILON / 1000

    def __add__(self, other):
        r"""Return ``self + other`` where ``self`` is the point mass ``delta(v)``.

        Math:
            - ``delta(v) + c`` (scalar ``c``) = ``delta(v + c)`` = ``Real(v + c)``.
            - ``delta(v) + D`` (distribution ``D``) = ``D`` shifted right by ``v``
              (adding the constant ``v`` to the random variable ``D``); the family
              and scale of ``D`` are preserved.
        """
        if isinstance(other, numbers.Real):
            return Real(self._loc + float(other))
        if isinstance(other, Distribution):
            return other + self._loc
        return NotImplemented

    def __radd__(self, other):
        """Reflected addition ``other + self`` (addition is commutative)."""
        return self.__add__(other)

    def __sub__(self, other):
        r"""Return ``self - other`` where ``self`` is the point mass ``delta(v)``.

        Math:
            - ``delta(v) - c`` (scalar ``c``) = ``delta(v - c)`` = ``Real(v - c)``.
            - ``delta(v) - D`` = ``v - D`` = ``(-1) * D`` shifted by ``v``
              (mean becomes ``v - mean(D)``; the family and scale of ``D`` are
              preserved).
        """
        if isinstance(other, numbers.Real):
            return Real(self._loc - float(other))
        if isinstance(other, Distribution):
            return (other * -1.0) + self._loc
        return NotImplemented

    def __rsub__(self, other):
        r"""Reflected subtraction ``other - self`` for a scalar: ``c - v = Real(c - v)``."""
        if isinstance(other, numbers.Real):
            return Real(float(other) - self._loc)
        return NotImplemented

    def __mul__(self, other):  # type: ignore
        r"""Return ``self * other`` where ``self`` is the point mass ``delta(v)``.

        Math:
            - ``delta(v) * k`` (scalar ``k``) = ``delta(v * k)`` = ``Real(v * k)``.
            - ``delta(v) * D`` (distribution ``D``) = ``D`` scaled by ``v``
              (equivalent to ``v * D``).
        """
        if isinstance(other, numbers.Real):
            return Real(self._loc * float(other))
        if isinstance(other, Distribution):
            return other * self._loc
        return NotImplemented

    def __rmul__(self, other):
        """Reflected multiplication ``other * self`` (multiplication by a scalar is commutative)."""
        return self.__mul__(other)

    def __and__(self, other):
        r"""Intersection of the point mass ``delta(v)`` with ``other``.

        The intersection of a point mass with any distribution that covers ``v``
        is the point mass itself, so this returns ``Real(v)`` (the crisp value
        dominates the fused belief).
        """
        return Real(self._loc)

    __rand__ = __and__

    def __lt__(self, other):
        """Check that a sample from `self` is ALWAYS less than a sample from other `other`"""
        if isinstance(other, float):
            return float(self) < other
        return self.max < other.min

    def __le__(self, other):
        """Check that a sample from `self` is ALWAYS less or equal a sample from other `other`"""
        if isinstance(other, float):
            return float(self) <= other
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
            return float(self) > other
        return self.min > other.max

    def __ge__(self, other):
        """Check that a sample from `self` is ALWAYS greater or equal a sample from other `other`"""
        if isinstance(other, float):
            return float(self) >= other
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

        >>> d2 = ScipyDistribution("uniform")
        >>> print(d2.mean())
        0.5

        >>> try:
        ...     d = ScipyDistribution("something crazy")
        ... except ValueError as e:
        ...     print(str(e))
        Unknown distribution something crazy

    """

    def __init__(self, type: str, **kwargs) -> None:
        """Initializes the instance.

        The ``type`` is normalized to a canonical family name (`UNIFORM` or
        `NORMAL`) via `canonical_distribution_type`, so ``"norm"``, ``"normal"``
        and ``"gaussian"`` all produce the same distribution. The canonical name
        is what `type` returns; the underlying ``scipy.stats`` object is built
        from the scipy-specific name (``"norm"`` for the normal family).

        For the uniform family the parameters follow the ``scipy`` convention:
        ``loc`` is the lower bound and ``scale`` is the width, so the support is
        ``[loc, loc + scale]``. For the normal family ``loc`` is the mean and
        ``scale`` is the standard deviation.
        """
        import scipy.stats as stats

        canonical = canonical_distribution_type(type)
        scipy_name = _SCIPY_NAME.get(canonical, canonical)
        dist = getattr(stats, scipy_name, None)
        if dist is None:
            raise ValueError(f"Unknown distribution {type}")
        if "loc" not in kwargs.keys():
            kwargs["loc"] = 0.0
        if "scale" not in kwargs.keys():
            kwargs["scale"] = 1.0

        self._dist = dist(**kwargs)
        self._type = canonical

    def __copy__(self):
        """Create a shallow copy of this distribution."""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Create a deep copy of this distribution."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def type(self) -> str:
        """Return the distribution type name (e.g., 'uniform', 'normal')."""
        return self._type

    def _make_dist(self, type: str, loc: float, scale: float):
        """Build a raw ``scipy.stats`` frozen distribution for the given family.

        ``type`` is canonicalized then mapped to its ``scipy`` name before the
        frozen distribution is created with the given ``loc`` and ``scale``.
        """
        import scipy.stats as stats

        scipy_name = _SCIPY_NAME.get(
            canonical_distribution_type(type), canonical_distribution_type(type)
        )
        dist = getattr(stats, scipy_name, None)
        if dist is None:
            raise ValueError(f"Unknown distribution {type}")
        return dist(loc=loc, scale=scale)

    @property
    def loc(self):
        """Return the location parameter of the distribution (typically the mean)."""
        return self._dist.kwds.get("loc", 0.5)

    @property
    def scale(self):
        """Return the scale parameter of the distribution (typically the standard deviation)."""
        return self._dist.kwds.get("scale", 0.0)

    def mean(self) -> float:
        r"""Return the mean (expected value) of the distribution.

        Math:
            - Uniform ``U(loc=a, scale=w)`` over ``[a, a+w]``: ``mean = a + w/2``.
            - Normal ``N(loc=m, scale=s)``: ``mean = m``.
            - A (near) zero-scale distribution is a point mass, so ``mean = loc``.

        Both cases coincide with ``scipy.stats.<family>.mean()``.
        """
        if self.scale < EPSILON:
            return float(self.loc)
        return float(self._dist.mean())

    def prob(self, val: float) -> float:
        r"""Return the probability density ``f(val)`` at ``val``.

        This is the ``scipy.stats`` pdf of the underlying frozen distribution:

            - Uniform ``U(a, w)``: ``1/w`` for ``a <= val <= a+w`` else ``0``.
            - Normal ``N(m, s)``: ``exp(-((val-m)**2)/(2 s**2)) / (s sqrt(2 pi))``.
        """
        return self._dist.pdf(val)

    def cum_prob(self, mn: float, mx: float) -> float:
        r"""Return the probability mass in ``[mn, mx]``: ``F(mx) - F(mn)`` where
        ``F`` is the cumulative distribution function of the underlying scipy
        distribution."""
        return self._dist.cdf(mx) - self._dist.cdf(mn)

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw random samples from the distribution."""
        return self._dist.rvs(size=size)

    @property
    def min(self):
        r"""Return the (effective) minimum value of the distribution.

        Math:
            - Uniform ``U(loc=a, scale=w)``: ``min = a`` (the true lower bound of
              the support ``[a, a+w]``).
            - Normal ``N(loc=m, scale=s)``: ``min = m - s`` (a one-standard-
              deviation heuristic, since a Gaussian has unbounded support). This
              is used only by the "always less/greater than" comparison operators.
        """
        if self._type == UNIFORM:
            return self.loc
        return self.loc - self.scale

    @property
    def max(self):
        r"""Return the (effective) maximum value of the distribution.

        Math:
            - Uniform ``U(loc=a, scale=w)``: ``max = a + w`` (upper bound of the
              support ``[a, a+w]``).
            - Normal ``N(loc=m, scale=s)``: ``max = m + s`` (one-standard-
              deviation heuristic; see `min`).
        """
        return self.loc + self.scale

    def is_gaussian(self):
        """Return ``True`` iff this distribution belongs to the normal family."""
        return self._type == NORMAL

    def is_uniform(self):
        """Return ``True`` iff this distribution belongs to the uniform family."""
        return self._type == UNIFORM

    def is_crisp(self) -> bool:
        """Returns true if this is a distribution with all probability at one point (delta(v))"""
        return self.scale < EPSILON / 1000

    def __call__(self, val: float) -> float:
        """Returns the probability for the given value"""
        return self.prob(val)

    def __add__(self, other):
        r"""Return a NEW distribution for the sum ``self + other`` (never mutates).

        Let ``self`` be ``U(a1, w1)`` / ``N(m1, s1)``.

        Adding a scalar ``c`` (or a crisp/point-mass distribution at ``c``) shifts
        the location and keeps the scale (adding a constant to a random variable):

            - Uniform: ``U(a1 + c, w1)``          (mean -> ``a1 + w1/2 + c``)
            - Normal:  ``N(m1 + c, s1)``

        Adding another distribution of the **same family** models the sum of two
        independent random variables (``loc`` adds for both families):

            - Uniform + Uniform: ``U(a1 + a2, w1 + w2)`` -- the exact support of
              the sum is ``[a1+a2, (a1+w1)+(a2+w2)]``; kept uniform over that
              support (mean-preserving: ``mean = mean1 + mean2``).
            - Normal + Normal: ``N(m1 + m2, sqrt(s1**2 + s2**2))`` -- means add and
              variances add (exact for independent Gaussians).

        Mixing different families raises `TypeError`.
        """
        return self._combine(other, sign=+1.0)

    def __radd__(self, other):
        """Reflected addition (``scalar + distribution``); addition is commutative."""
        return self.__add__(other)

    def __sub__(self, other):
        r"""Return a NEW distribution for the difference ``self - other``.

        Let ``self`` be ``U(a1, w1)`` / ``N(m1, s1)``.

        Subtracting a scalar ``c`` (or a crisp point mass at ``c``) shifts the
        location and keeps the scale:

            - Uniform: ``U(a1 - c, w1)``
            - Normal:  ``N(m1 - c, s1)``

        Subtracting another distribution of the **same family** models the
        difference of two independent random variables (mean subtracts, variance
        still adds):

            - Uniform - Uniform: support ``[a1-(a2+w2), (a1+w1)-a2]`` ->
              ``U(a1 - a2 - w2, w1 + w2)`` (``mean = mean1 - mean2``).
            - Normal - Normal: ``N(m1 - m2, sqrt(s1**2 + s2**2))``.

        Mixing different families raises `TypeError`.
        """
        return self._combine(other, sign=-1.0)

    def __rsub__(self, other):
        r"""Reflected subtraction ``other - self`` for a scalar ``other = c``.

        Math (negate then shift, ``c - X``):

            - Uniform: support ``[c-(a+w), c-a]`` -> ``U(c - a - w, w)``.
            - Normal:  ``N(c - m, s)``.
        """
        if not isinstance(other, numbers.Real):
            return NotImplemented
        c = float(other)
        if self._type == UNIFORM:
            return ScipyDistribution(UNIFORM, loc=c - self.max, scale=self.scale)
        return ScipyDistribution(self._type, loc=c - self.loc, scale=self.scale)

    def _combine(self, other, sign: float):
        """Shared implementation of `__add__` (``sign=+1``) and `__sub__` (``sign=-1``)."""
        # A scalar (or a crisp point-mass) is treated as a constant shift.
        shift = None
        if isinstance(other, numbers.Real):
            shift = float(other)
        elif isinstance(other, Distribution) and other.is_crisp():
            shift = float(other.loc)
        if shift is not None:
            return ScipyDistribution(
                self._type, loc=self.loc + sign * shift, scale=self.scale
            )
        if not isinstance(other, Distribution):
            return NotImplemented
        if self.is_crisp():
            # A crisp self behaves as the scalar ``self.loc``.
            return (self.loc + other) if sign > 0 else (self.loc - other)
        if self._type != other._type:
            raise TypeError(
                f"Cannot combine distributions of different families "
                f"({self._type!r} and {other._type!r}). Only same-family "
                "addition/subtraction is supported."
            )
        if self._type == UNIFORM:
            if sign > 0:  # X + Y support: [a1+a2, (a1+w1)+(a2+w2)]
                loc = self.loc + other.loc
            else:  # X - Y support: [a1-(a2+w2), (a1+w1)-a2]
                loc = self.loc - other.max
            return ScipyDistribution(UNIFORM, loc=loc, scale=self.scale + other.scale)
        # Normal: means add/subtract, variances add.
        loc = self.loc + sign * other.loc
        scale = (self.scale**2 + other.scale**2) ** 0.5
        return ScipyDistribution(NORMAL, loc=loc, scale=scale)

    def __mul__(self, weight):
        r"""Return a NEW distribution scaled by a scalar ``weight = k`` (``k * X``).

        Scaling a random variable by ``k`` scales its mean by ``k`` and its
        standard deviation / range by ``|k|``:

            - Uniform ``U(a, w)``: support becomes ``{k*a, k*(a+w)}`` ->
              ``U(min(k*a, k*(a+w)), |k|*w)`` (handles negative ``k`` correctly).
            - Normal ``N(m, s)``: ``N(k*m, |k|*s)``.

        Multiplying by another distribution is not supported (returns
        ``NotImplemented``).
        """
        if not isinstance(weight, numbers.Real):
            return NotImplemented
        k = float(weight)
        if self._type == UNIFORM:
            e1, e2 = k * self.loc, k * (self.loc + self.scale)
            return ScipyDistribution(
                UNIFORM, loc=min(e1, e2), scale=abs(k) * self.scale
            )
        return ScipyDistribution(
            self._type, loc=k * self.loc, scale=abs(k) * self.scale
        )

    def __rmul__(self, weight):
        """Reflected multiplication (``scalar * distribution``); multiplication by a scalar is commutative."""
        return self.__mul__(weight)

    def __and__(self, other):
        r"""Return the intersection (evidence fusion) of ``self`` and ``other``.

        Both must belong to the **same family** (else `TypeError`).

        Math:
            - Uniform & Uniform: intersection of the two supports.
              With ``[a1, a1+w1]`` and ``[a2, a2+w2]`` let ``lo = max(a1, a2)`` and
              ``hi = min(a1+w1, a2+w2)``. If ``hi > lo`` the result is
              ``U(lo, hi - lo)``; otherwise the supports are disjoint and the
              result is the zero-width point mass ``U((lo+hi)/2, 0)`` (guarded so
              the scale is never negative). This is used by the VOI elicitors to
              combine a hard range answer with the current belief interval.
            - Normal & Normal: the normalized product of two Gaussian densities is
              Gaussian with precision-weighted parameters::

                  1/s**2 = 1/s1**2 + 1/s2**2
                  m      = s**2 * (m1/s1**2 + m2/s2**2)

              (Implemented for completeness; not exercised by the current
              elicitation code, which keeps beliefs uniform.)
        """
        if not isinstance(other, Distribution):
            return NotImplemented
        if other.is_crisp():
            return Real(float(other.loc))
        if self.is_crisp():
            return Real(float(self.loc))
        if self._type != other._type:
            raise TypeError(
                f"Cannot intersect distributions of different families "
                f"({self._type!r} and {other._type!r})."
            )
        if self._type == UNIFORM:
            lo = max(self.loc, other.loc)
            hi = min(self.loc + self.scale, other.loc + other.scale)
            if hi > lo:
                return ScipyDistribution(UNIFORM, loc=lo, scale=hi - lo)
            return ScipyDistribution(UNIFORM, loc=0.5 * (lo + hi), scale=0.0)
        v1, v2 = self.scale**2, other.scale**2
        variance = 1.0 / (1.0 / v1 + 1.0 / v2)
        loc = variance * (self.loc / v1 + other.loc / v2)
        return ScipyDistribution(NORMAL, loc=loc, scale=variance**0.5)

    def __rand__(self, other):
        """Reflected intersection; intersection is symmetric."""
        return self.__and__(other)

    def __lt__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS less than a sample from other `other`"""
        if isinstance(other, numbers.Real):
            return self.max < float(other)
        return self.max < other.min

    def __le__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS less or equal a sample from other `other`"""
        return self < other or self == other

    # def __eq__(self, other):
    #     if isinstance(other, ScipyDistribution):
    #         return (
    #             self._type == other._type
    #             and abs(self.loc - other.loc) < EPSILON
    #             and abs(self.scale - other.scale) < EPSILON
    #         )
    #
    #     if isinstance(other, float):
    #         return abs(self.loc - other) < EPSILON and self.loc < EPSILON
    #     raise ValueError(f"Cannot compare Distribution with {type(other)}")

    def __eq__(self, other) -> bool:
        """Return ``True`` iff the two distributions have the same ``loc``, ``scale``
        and family.

        A scalar is first promoted to a crisp `Real`. Because a `Real` is a
        family-agnostic point mass, comparing against one only checks
        ``loc``/``scale``. Note ``type`` is a method, so the family is compared
        through the underlying ``_type`` attribute (comparing the bound methods
        directly would always be unequal).
        """
        if isinstance(other, numbers.Real):
            other = Real(other)
        return (
            self.loc == other.loc
            and self.scale == other.scale
            and (
                isinstance(other, Real)
                or isinstance(self, Real)
                or self._type == other._type
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
        """Return a human-readable string representation of the distribution.

        Uniform is shown as ``U(lower, upper)`` = ``U(loc, loc+scale)`` and normal
        as ``G(mean, std)`` = ``G(loc, scale)``.
        """
        if self._type == UNIFORM:
            return f"U({self.loc}, {self.loc + self.scale})"
        if self._type == NORMAL:
            return f"G({self.loc}, {self.scale})"
        return f"{self._type}(loc:{self.loc}, scale:{self.scale})"

    __repr__ = __str__


class UniformDistribution(ScipyDistribution):
    """A `ScipyDistribution` reprsenting a unifrom distribution"""

    def __init__(
        self, loc: float = 0.0, scale: float = 1.0, *, type: str = "uniform", **kwargs
    ) -> None:
        """Initialize a uniform distribution.

        Args:
            loc: The lower bound of the distribution range.
            scale: The width of the distribution (upper bound = loc + scale).
            **kwargs: Additional keyword arguments passed to scipy.stats.uniform.
        """
        super().__init__(loc=loc, scale=scale, type="uniform", **kwargs)


class NormalDistribution(ScipyDistribution):
    """A `ScipyDistribution` representing a normal (Gaussian) distribution"""

    def __init__(
        self, loc: float = 0.0, scale: float = 1.0, *, type: str = "norm", **kwargs
    ) -> None:
        """Initialize a normal (Gaussian) distribution.

        Args:
            loc: The mean of the distribution.
            scale: The standard deviation of the distribution.
            **kwargs: Additional keyword arguments passed to scipy.stats.norm.
        """
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
        return cls(type="uniform", loc=range[0], scale=range[1])  # type: ignore
    if uncertainty <= 0.0:
        return Real(loc=value)
    scale = uncertainty * (range[1] - range[0])
    loc = max(range[0], (random.random() - 1.0) * scale + value)
    if loc + scale > range[1]:
        loc -= loc + scale - range[1]
    return cls(type="uniform", loc=loc, scale=scale)  # type: ignore
