"""Infinite issue implementations for unbounded integer and continuous ranges.

`IntIssue` and `RealIssue` are factories over these classes.

Examples:

    >>> from negmas.outcomes import IntIssue, RealIssue
    >>> q = IntIssue("quantity", min=1)  # any positive integer
    >>> q.is_valid(10**12), q.is_valid(0), q.cardinality
    (True, False, inf)
    >>> all(1 <= q.rand() for _ in range(100))
    True
    >>> p = RealIssue("price", typical=(10.0, 20.0))  # any real number
    >>> p.is_valid(-1e9), p.is_continuous()
    (True, True)
    >>> IntIssue("n", min=0, max=5)  # both bounds: a finite range
    ContiguousIssue((0, 5), n)
"""

from __future__ import annotations

import math
import numbers
import random
import sys
from typing import Generator

import numpy as np

from negmas.outcomes.base_issue import Issue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.outcomes.continuous_issue import ContinuousIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize

__all__ = [
    "CountableInfiniteIssue",
    "ContinuousInfiniteIssue",
    "InfiniteIssue",
    "IntIssue",
    "RealIssue",
]

INFINITE_INT = sys.maxsize
# Width of the default typical range next to a finite bound (or around 0).
DEFAULT_TYPICAL_SPAN = 100
# Probability that `rand()` samples outside the typical range (within bounds).
TAIL_PROBABILITY = 0.1


def _default_typical(lo: float, hi: float) -> tuple[float, float]:
    """A typical range: next to the finite bound, or around 0 when unbounded."""
    if math.isfinite(lo) and math.isfinite(hi):
        return lo, hi
    if math.isfinite(lo):
        return lo, lo + DEFAULT_TYPICAL_SPAN
    if math.isfinite(hi):
        return hi - DEFAULT_TYPICAL_SPAN, hi
    return -DEFAULT_TYPICAL_SPAN / 2, DEFAULT_TYPICAL_SPAN / 2


def _typical(lo: float, hi: float, typical) -> tuple[float, float]:
    """The typical range clipped to the bounds (raises if it is empty)."""
    t = _default_typical(lo, hi) if typical is None else tuple(typical)
    if len(t) != 2 or t[1] < t[0]:
        raise ValueError(f"typical must be a (min, max) range but got {typical}")
    a, b = max(t[0], lo), min(t[1], hi)
    if b < a:
        raise ValueError(f"typical range {typical} is outside the bounds ({lo}, {hi})")
    return a, b


def _rand_real(lo: float, hi: float, t: tuple[float, float]) -> float:
    """Mostly uniform in the typical range; sometimes an exponential tail."""
    a, b = t
    if random.random() >= TAIL_PROBABILITY:
        return random.uniform(a, b)
    scale = max(b - a, 1.0)
    sides = [s for s, room in ((1, hi - b), (-1, a - lo)) if room > 0]
    if not sides:
        return random.uniform(a, b)
    side = random.choice(sides)
    x = (
        b + random.expovariate(1 / scale)
        if side > 0
        else a - random.expovariate(1 / scale)
    )
    return min(max(x, lo), hi)


def _none_if_inf(x):
    return None if not math.isfinite(x) else x


def _from_bounds(b):
    """(min, max) from serialized bounds, None meaning infinite."""
    if not b:
        return None
    return (
        float("-inf") if b[0] is None else b[0],
        float("inf") if b[1] is None else b[1],
    )


class InfiniteIssue:
    """
    Indicates that the issue is infinite (i.e. one or more of its limits is infinity)
    """


class CountableInfiniteIssue(ContiguousIssue, InfiniteIssue):
    """
    An issue that can have all integer values (optionally bounded on one side).

    Args:
        values: The (min, max) limits; use ``float("inf")``/``-inf`` for no limit.
        name: The issue name.
        typical: The range `rand()` and the samplers draw from mostly. Defaults
            to 100 values next to the finite bound (or around 0).

    Remarks:
        - Infinities are stored as +- INFINITE_INT.
        - The issue is not discrete (it cannot be enumerated): it is sampled.
    """

    def __init__(
        self,
        values: tuple[int | float, int | float],
        name: str | None = None,
        typical: tuple[int, int] | None = None,
    ) -> None:
        lo, hi = values
        # finite bounds round inward; infinite ones stay infinite
        lo = lo if not math.isfinite(lo) else int(math.ceil(lo))
        hi = hi if not math.isfinite(hi) else int(math.floor(hi))
        if hi < lo:
            raise ValueError(f"No integers between {values[0]} and {values[1]}")
        # stand-ins for the infinite limits (kept beyond any finite bound)
        v = (
            lo
            if math.isfinite(lo)
            else -INFINITE_INT
            if not math.isfinite(hi)
            else min(-INFINITE_INT, hi - INFINITE_INT),
            hi
            if math.isfinite(hi)
            else INFINITE_INT
            if not math.isfinite(lo)
            else max(INFINITE_INT, lo + INFINITE_INT),
        )
        a, b = _typical(lo, hi, typical)
        # set before the base init, which samples values
        self._lo, self._hi = lo, hi
        self.typical = (math.ceil(a), math.floor(b))
        if self.typical[1] < self.typical[0]:
            raise ValueError(f"typical range {typical} has no integers")
        super().__init__(v, name=name)
        self._value_type = int
        self._n_values = float("inf")
        self.min_value, self.max_value = lo, hi

    def is_continuous(self) -> bool:
        return False

    def is_discrete(self) -> bool:
        """False: infinitely many values (sampled, never enumerated)."""
        return False

    def is_finite(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return True

    @property
    def cardinality(self) -> float:
        return float("inf")

    def is_valid(self, v) -> bool:
        if isinstance(v, bool):
            return False
        if not isinstance(v, numbers.Integral):
            if not (isinstance(v, numbers.Real) and math.isfinite(v) and v == int(v)):
                return False
        return self.min_value <= v <= self.max_value

    def rand(self) -> int:
        """A random integer, mostly from the typical range."""
        x = _rand_real(self._lo, self._hi, self.typical)
        return int(min(max(round(x), self._lo), self._hi))

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list[int]:
        if with_replacement:
            return [self.rand() for _ in range(n)]
        seen: dict[int, None] = {}
        for _ in range(50 * n):
            if len(seen) >= n:
                break
            seen.setdefault(self.rand(), None)
        # fill deterministically outward from the typical range
        up, down = self.typical[1] + 1, self.typical[0] - 1
        while len(seen) < n and (up <= self.max_value or down >= self.min_value):
            if up <= self.max_value:
                seen.setdefault(up, None)
                up += 1
            if len(seen) < n and down >= self.min_value:
                seen.setdefault(down, None)
                down -= 1
        return list(seen)[:n]

    def ordered_value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[int, None, None]:
        """`n` (default 10) evenly spaced integers over the typical range."""
        a, b = self.typical
        n = 10 if n is None or not math.isfinite(n) else int(n)
        if b - a + 1 <= n:
            yield from range(a, b + 1)
            return
        yield from sorted(set(np.linspace(a, b, num=n).round().astype(int).tolist()))

    def value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[int, None, None]:
        """Like `ordered_value_generator` (the typical range, never everything)."""
        yield from self.ordered_value_generator(n, grid, compact, endpoints)

    def to_discrete(
        self, n: int | float | None = 10, grid=True, compact=False, endpoints=True
    ):
        """A finite issue of `n` integers spread over the typical range."""
        from negmas.outcomes.cardinal_issue import DiscreteCardinalIssue

        return DiscreteCardinalIssue(
            list(self.ordered_value_generator(n)), name=self.name
        )

    @property
    def all(self):
        raise ValueError(f"The infinite issue {self.name} cannot be enumerated")

    def rand_invalid(self):
        """An integer outside the bounds, or None if the issue is unbounded."""
        if math.isfinite(self.max_value):
            return int(self.max_value) + random.randint(1, DEFAULT_TYPICAL_SPAN)
        if math.isfinite(self.min_value):
            return int(self.min_value) - random.randint(1, DEFAULT_TYPICAL_SPAN)
        return None

    def value_at(self, index: int):
        raise ValueError("Cannot index an infinite issue")

    def contains(self, issue: Issue) -> bool:
        """Checks whether every value of `issue` is in this issue."""
        return (
            issubclass(issue.value_type, numbers.Integral)
            and issue.min_value is not None
            and issue.max_value is not None
            and issue.min_value >= self.min_value
            and issue.max_value <= self.max_value
        )

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        d = super().to_dict(python_class_identifier=python_class_identifier)
        d["typical"] = list(self.typical)
        d["bounds"] = [_none_if_inf(self.min_value), _none_if_inf(self.max_value)]
        return d

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        values = _from_bounds(d.get("bounds")) or deserialize(
            d["values"], python_class_identifier=python_class_identifier
        )
        t = d.get("typical")
        return cls(tuple(values), name=d.get("name"), typical=tuple(t) if t else None)

    def _clone(self):
        return type(self)(
            (self.min_value, self.max_value), name=self.name, typical=self.typical
        )

    def __copy__(self):
        return self._clone()

    def __deepcopy__(self, memodict={}):
        return self._clone()

    def __repr__(self):
        return f"{self.__class__.__name__}({(self.min_value, self.max_value)}, {self.name})"

    def __str__(self):
        return f"{self.name}: ({self.min_value}, {self.max_value})"


class ContinuousInfiniteIssue(ContinuousIssue, InfiniteIssue):
    """
    An issue that can represent all real numbers (optionally bounded on one side).

    Args:
        values: The (min, max) limits; use ``float("inf")``/``-inf`` for no limit.
        name: The issue name.
        n_levels: As in `ContinuousIssue`.
        typical: The range `rand()` and the samplers draw from mostly. Defaults
            to a width of 100 next to the finite bound (or around 0).
    """

    def __init__(self, values, name=None, n_levels=10, typical=None) -> None:
        values = tuple(float(_) for _ in values)
        self._lo, self._hi = values
        self.typical = _typical(values[0], values[1], typical)
        super().__init__(values, name=name, n_levels=n_levels)
        self._value_type = float

    def is_valid(self, v) -> bool:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            return False
        if not math.isfinite(v):
            return False
        return self.min_value <= v <= self.max_value

    def rand(self) -> float:
        """A random real, mostly from the typical range."""
        return _rand_real(self._lo, self._hi, self.typical)

    def ordered_value_generator(
        self, n: int | float | None = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[float, None, None]:
        """`n` values over the typical range (a grid, or random if not `grid`)."""
        if n is None or not math.isfinite(n):
            raise ValueError(f"Cannot generate {n} values from issue: {self}")
        a, b = self.typical
        if grid:
            yield from np.linspace(a, b, num=int(n), endpoint=True).tolist()
            return
        yield from sorted(random.uniform(a, b) for _ in range(int(n)))

    def value_generator(
        self, n: int | float | None = 100, grid=True, compact=False, endpoints=True
    ) -> Generator[float, None, None]:
        yield from self.ordered_value_generator(n, grid, compact, endpoints)

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list[float]:
        if with_replacement:
            return [self.rand() for _ in range(n)]
        return list(self.ordered_value_generator(n))

    def rand_invalid(self):
        """A real outside the bounds, or None if the issue is unbounded."""
        span = max(self.typical[1] - self.typical[0], 1.0)
        if math.isfinite(self.max_value):
            return self.max_value + random.uniform(0.01, 1.0) * span
        if math.isfinite(self.min_value):
            return self.min_value - random.uniform(0.01, 1.0) * span
        return None

    def value_at(self, index: int):
        raise ValueError("Cannot index an infinite issue")

    def contains(self, issue: Issue) -> bool:
        """Checks whether every value of `issue` is in this issue."""
        return (
            issubclass(issue.value_type, numbers.Real)
            and issue.min_value is not None
            and issue.max_value is not None
            and issue.min_value >= self.min_value
            and issue.max_value <= self.max_value
        )

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        d = super().to_dict(python_class_identifier=python_class_identifier)
        d["typical"] = list(self.typical)
        d["bounds"] = [_none_if_inf(self.min_value), _none_if_inf(self.max_value)]
        return d

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        values = _from_bounds(d.get("bounds")) or deserialize(
            d["values"], python_class_identifier=python_class_identifier
        )
        t = d.get("typical")
        return cls(
            tuple(values),
            name=d.get("name"),
            n_levels=d.get("n_levels", 10),
            typical=tuple(t) if t else None,
        )

    def _clone(self):
        return type(self)(
            self._values, name=self.name, n_levels=self._n_levels, typical=self.typical
        )

    def __copy__(self):
        return self._clone()

    def __deepcopy__(self, memodict={}):
        return self._clone()


def _bound(x):
    """None for a missing or infinite bound."""
    return None if x is None or not math.isfinite(x) else x


def IntIssue(  # noqa: N802 (a factory named like the issue it makes)
    name: str | None = None,
    min: int | None = None,
    max: int | None = None,
    typical: tuple[int, int] | None = None,
) -> Issue:
    """An integer issue: any integer, or one bounded on one or both sides.

    Returns a `CountableInfiniteIssue` (a `ContiguousIssue` if both bounds are
    given). ``min=0`` or ``min=1`` gives a non-negative or positive issue.
    """
    min, max = _bound(min), _bound(max)
    if min is not None and max is not None:
        return ContiguousIssue((math.ceil(min), math.floor(max)), name=name)
    lo = float("-inf") if min is None else min
    hi = float("inf") if max is None else max
    return CountableInfiniteIssue((lo, hi), name=name, typical=typical)


def RealIssue(  # noqa: N802 (a factory named like the issue it makes)
    name: str | None = None,
    min: float | None = None,
    max: float | None = None,
    typical: tuple[float, float] | None = None,
) -> Issue:
    """A real issue: any real number, or one bounded on one or both sides.

    Returns a `ContinuousInfiniteIssue` (a `ContinuousIssue` if both bounds
    are given).
    """
    min, max = _bound(min), _bound(max)
    if min is not None and max is not None:
        return ContinuousIssue((float(min), float(max)), name=name)
    lo = float("-inf") if min is None else float(min)
    hi = float("inf") if max is None else float(max)
    return ContinuousInfiniteIssue((lo, hi), name=name, typical=typical)
