from __future__ import annotations

import math
import random
import warnings

# from bisect import bisect_left, bisect_right
from typing import Any, Iterable  # , Sequence

import numpy as np
from numpy import floating, integer
from numpy.typing import NDArray

from negmas.outcomes import Outcome
from negmas.warnings import NegmasUnexpectedValueWarning, warn_if_slow

from .base_ufun import BaseUtilityFunction
from .protocols import InverseUFun

__all__ = [
    "PresortingInverseUtilityFunction",
    "SamplingInverseUtilityFunction",
]

EPS = 1e-6


def index_above_or_equal(a: NDArray, x: Any, lo: int = 0, hi: int | None = None) -> int:
    "Locate the smallest value greater than or equal to x"
    if hi is None:
        hi = len(a)
    i = np.searchsorted(a[lo : hi + 1], x, side="left")
    return i


def index_below_or_equal(a: NDArray, x: Any, lo: int = 0, hi: int | None = None) -> int:
    "Locate the greatest value less than or equal to x"
    if hi is None:
        hi = len(a)
    i = np.searchsorted(a[lo : hi + 1], x, side="right")
    return i - 1


class SamplingInverseUtilityFunction(InverseUFun):
    """
    A utility function inverter that uses sampling.

    Nothing is done during initialization so the fixed cost of this inverter is minimal.
    Nevertheless, each time the system is asked to find an outcome within some range, it uses
    random sampling which is very inefficient and suffers from the curse of dimensionality.
    """

    def __init__(self, ufun: BaseUtilityFunction, max_samples_per_call: int = 10_000):
        self._ufun = ufun
        self.max_samples_per_call = max_samples_per_call
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    @property
    def ufun(self):
        return self._ufun

    def init(self):
        pass

    def all(
        self,
        rng: float | tuple[float, float],
    ) -> list[Outcome]:
        """
        Finds all outcomes with in the given utility value range

        Args:
            rng: The range. If a value, outcome utilities must match it exactly

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        raise ValueError(
            f"Cannot find all outcomes in a range ({rng}) using a SamplingInverseUtilityFunction. Try a PresortedInverseUtilityFunction"
        )

    def some(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        n: int | None = None,
    ) -> list[Outcome]:
        """
        Finds some outcomes with the given utility value (if discrete, all)

        Args:
            rng: The range. If a value, outcome utilities must match it exactly
            n: The maximum number of outcomes to return

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        if not isinstance(rng, Iterable):
            rng = (rng - EPS, rng + EPS)
        if not n:
            n = self.max_samples_per_call
        else:
            n *= 3
        if not self._ufun.outcome_space:
            return []
        outcomes = list(self._ufun.outcome_space.sample(n, False, False))
        mn, mx = rng
        u = self.ufun.eval_normalized if normalized else self.ufun.eval
        return [_ for _ in outcomes if mn <= u(_) <= mx]

    def worst_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        some = self.some(rng, normalized)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        worst_util, worst = float("inf"), None
        for o in some:
            util = self._ufun(o)
            if util < worst_util:
                worst_util, worst = util, o
        return worst

    def best_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        some = self.some(rng, normalized)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        best_util, best = float("-inf"), None
        for o in some:
            util = self._ufun(o)
            if util < best_util:
                best_util, best = util, o
        return best

    def one_in(
        self,
        rng: float | tuple[float, float],
        normalized: float,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        if not self._ufun.outcome_space:
            return None
        if not isinstance(rng, Iterable):
            rng = (rng - EPS, rng + EPS)
        u = self.ufun.eval_normalized if normalized else self.ufun.eval
        for _ in range(self.max_samples_per_call):
            o = list(self._ufun.outcome_space.sample(1))[0]
            if rng[0] <= u(o) <= rng[1]:
                return o
        if fallback_to_higher and rng[1] < 1 - EPS:
            return self.one_in(
                (rng[0], 1),
                normalized,
                fallback_to_higher=False,
                fallback_to_best=fallback_to_best,
            )
        if fallback_to_best:
            return self._ufun.best()
        return None

    def minmax(self) -> tuple[float, float]:
        """
        Finds the minimum and maximum utility values that can be returned.

        Remarks:
            These may be different from the results of `ufun.minmax()` as they can be approximate.
        """
        return self.min(), self.max()

    def extreme_outcomes(self) -> tuple[Outcome | None, Outcome | None]:
        """
        Finds the worst and best outcomes that can be returned.

        Remarks:
            These may be different from the results of `ufun.extreme_outcomes()` as they can be approximate.
        """
        return self.worst(), self.best()

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Calling an inverse ufun directly is equivalent to calling `one_in()`
        """
        return self.one_in(rng, normalized)


class PresortingInverseUtilityFunction(InverseUFun):
    """
    A utility function inverter that uses pre-sorting.

    The outcome-space is sampled if it is continuous and enumerated if it is discrete
    during the call to `init()` and an ordered list of outcomes with their utility
    values is then cached.


    Args:
        ufun: The utility function to be inverted
        levels: discretization levels per issue
        max_cache_size: maximum allowed number of outcomes in the resulting inverse
        rational_only: If true, rational outcomes will be sorted but irrational outcomes will not be sorted (should be faster if the reserved value is high)
        n_waypoints: Used to speedup sampling outcomes at given utilities. The larger, the slower init() will be but the faster worst_in() and best_in()
        eps: Absolute difference between utility values to consider them equal (zero or negative to disable).
        rel_eps: Relative difference between utility values to consider them equal (zero or negative to disable).

    Remarks:
        - The actual limit used to judge ufun equality is max(eps, rel_eps * range) where range is the difference between max and min utilities for rational outcomes.
          Set both eps, rel_eps to zero or a negative number to disable cycling through outcomes with equal utilities (or set cycle=False when calling the appropriate function).
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        levels: int = 10,
        max_cache_size: int = 10_000_000_000,
        rational_only: bool = False,
        n_waypints: int = 10,
        eps: float = 1e-12,
        rel_eps: float = 1e-6,
    ):
        self._ufun = ufun
        self.max_cache_size = max_cache_size
        self.levels = levels
        self._initialized = False
        self.outcomes: list[Outcome] = []
        self._last_rational: int = -1
        self.utils: NDArray[floating[Any]] = []  # type: ignore
        self.rational_only = rational_only
        self._waypoints: NDArray[integer[Any]] = []  # type: ignore
        self._waypoint_values: NDArray[floating[Any]] = []  # type: ignore
        self.__nwaypoints = n_waypints
        self._smallest_indx, self._smallest_val = 0, float("inf")
        self._largest_indx, self._largest_val = -1, float("-inf")
        self._last_returned_from_next: int = -1
        self._eps = eps
        self._rel_eps = rel_eps
        self._near_range: dict[int, tuple[int, int]] = dict()

    @property
    def initialized(self):
        return self._initialized

    @property
    def ufun(self):
        return self._ufun

    def reset(self) -> None:
        self._initialized = False
        self.outcomes, self.utils = [], []  # type: ignore
        self._waypoints, self._waypoint_values = [], []  # type: ignore

    def init(self):
        outcome_space = self._ufun.outcome_space
        if outcome_space is None:
            raise ValueError("Cannot find the outcome space.")
        self._worst, self._best = self._ufun.extreme_outcomes()
        self._min, self._max = float(self._ufun(self._worst)), float(
            self._ufun(self._best)
        )
        self._range = self._max - self._min
        self._offset = self._min / self._range if self._range > EPS else self._min
        for l in range(self.levels, 0, -1):
            n = outcome_space.cardinality_if_discretized(l)
            if n <= self.max_cache_size:
                break
        else:
            raise ValueError(
                f"Cannot discretize keeping cache size at {self.max_cache_size}. Outcome space cardinality is {outcome_space.cardinality}\nOutcome space: {outcome_space}"
            )
        os = outcome_space.to_discrete(levels=l, max_cardinality=self.max_cache_size)
        outcomes = list(os.enumerate_or_sample(max_cardinality=self.max_cache_size))
        utils = [float(self._ufun.eval(_)) for _ in outcomes]
        # x = len(utils)
        warn_if_slow(
            len(utils),
            "Inverting a large utility function",
            lambda x: (x * math.log2(x)) if x else x,
        )
        r = self._ufun.reserved_value
        r = float(r) if r is not None else float("-inf")
        if self.rational_only:
            rational, irrational = [], []
            ur, uir = [], []
            for u, o in zip(utils, outcomes):
                if u >= r:
                    rational.append(o)
                    ur.append(u)
                else:
                    irrational.append(o)
                    uir.append(u)
        else:
            rational, irrational = outcomes, []
            ur, uir = utils, []
        ur, uir = (
            np.asarray(ur, dtype=float).flatten(),
            np.asarray(uir, dtype=float).flatten(),
        )
        indices = np.argsort(ur)
        ur_sorted = ur[indices]
        if len(ur_sorted) > 0:
            eps = max(self._eps, self._rel_eps * (ur_sorted[-1] - ur_sorted[0]))
        else:
            eps = -1
        if eps > 0:
            try:
                n = len(ur_sorted)
                if n >= 2:
                    scaled = np.asarray(ur_sorted / eps, dtype=int)
                    diffs = np.diff(scaled) != 0
                    indexes = np.nonzero(diffs)[0] + 1
                    groups = np.split(scaled, indexes)
                    lengths = np.asarray([_.size for _ in groups], dtype=int)
                    starts = np.hstack((np.asarray([0], dtype=int), indexes))
                    ends = starts + lengths - 1
                    extended = np.nonzero(lengths > 1)[0]
                    starts, ends = starts[extended], ends[extended]
                    for mn, mx in zip(starts, ends):
                        # assert scaled[mn] == scaled[mx], f"{scaled[mn]=}, {scaled[mx]=}"
                        for indx in range(mn, mx + 1):
                            # assert scaled[indx] == scaled[mx], f"{scaled[indx]=}, {scaled[mx]=}"
                            self._near_range[indx] = (mn, mx)
            except:
                pass
            # for indx, current in enumerate(scaled):
            #     mn_indx = indx
            #     for i in range(indx - 1, -1, -1):
            #         u = scaled[i]
            #         if current != scaled[i]:
            #             mn_indx = i + 1
            #             break
            #     mx_indx = indx
            #     for i in range(indx + 1, n):
            #         u = scaled[i]
            #         if current != scaled[i]:
            #             mx_indx = i - 1
            #             break
            #     if mn_indx < mx_indx:
            #         self._near_range[indx] = (mn_indx, mx_indx)

        # ordered_outcomes = sorted(zip(ur, rational, strict=True))
        # if irrational:
        #     ordered_outcomes += list(zip(uir, irrational, strict=True))
        self._initialized = True
        self._last_rational = len(rational) - 1
        # save waypoints within the sorted outcomes list with known utilities.
        # Used to limit the lo, hi limits when doing bisection later
        self.__nwaypoints = min(self.__nwaypoints, self._last_rational + 1)
        waypoints: NDArray[integer[Any]] = (
            np.asarray(
                np.linspace(0, self._last_rational, self.__nwaypoints, endpoint=True),
                dtype=int,
            )
            if self.__nwaypoints > 0
            else np.empty(0, dtype=int)
        )
        waypoint_values: NDArray[floating[Any]] = (
            ur_sorted[waypoints] if self.__nwaypoints > 0 else np.empty(0, dtype=float)
        )
        assert not any(
            _ is None for _ in waypoints
        ), f"{waypoints=}\n{waypoint_values}\n{self._last_rational}"
        assert not any(
            a < b for a, b in zip(waypoints[1:], waypoints[:-1])
        ), f"{waypoints=}\n{waypoint_values}\n{self._last_rational}"
        assert not any(
            a < b for a, b in zip(waypoint_values[1:], waypoint_values[:-1])
        ), f"{waypoints=}\n{waypoint_values}\n{self._last_rational}"
        self._waypoints, self._waypoint_values = waypoints, waypoint_values
        self.outcomes = [rational[_] for _ in indices] + irrational
        self.utils = np.hstack((ur_sorted, uir))
        self._smallest_indx, self._smallest_val = 0, self.utils[0]
        self._largest_indx, self._largest_val = len(self.utils) - 1, self.utils[-1]

    def _un_normalize_range(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        for_best: bool,
    ) -> tuple[float, float]:
        if not isinstance(rng, Iterable):
            rng = (rng - EPS, rng + EPS)
        if not normalized:
            return rng
        mn, mx = self._min, self._max
        d = mx - mn
        if d < EPS:
            return tuple(0.0 if not for_best else 1.0 for _ in rng)
        return tuple(_ * d + mn for _ in rng)

    def _get_limiting_waypoints(self, mn: float, mx: float) -> tuple[int, int]:
        """Returns indices of the smallest utility >= mx and largest utility <= mn in self._utils"""
        if len(self._waypoints) <= 0:
            return (-1, -1)
        lo, hi = self._waypoints[0], self._waypoints[-1]
        lo_val, hi_val = self._waypoint_values[0], self._waypoint_values[-1]
        n = len(self.utils)
        for i, u in zip(self._waypoints, self._waypoint_values):
            if u == lo_val:
                lo = i
            elif u > lo_val:
                lo = max(0, lo - 1)
            if u == hi_val:
                hi = i
            elif u < hi_val:
                hi = min(n - 1, hi - 1)
        # adjust limits to known largest and smallest. This will be specially useful for
        # strategies that call worst_in or best_in repeatedly with descending/ascending values
        if self._smallest_val <= mn and self._smallest_indx > lo:
            lo = self._smallest_indx
        if self._largest_val >= mx and self._largest_indx < hi:
            hi = self._largest_indx
        return lo, hi

    def next_worse(self) -> Outcome | None:
        """Returns the rational outcome with utility just below the last one returned from this function"""
        if self._last_returned_from_next < 0:
            self._last_returned_from_next = self._last_rational
            return self.best()
        if self._last_returned_from_next > 0:
            self._last_returned_from_next -= 1
            return self.outcomes[self._last_returned_from_next]
        return None

    def next_better(self) -> Outcome | None:
        """Returns the rational outcome with utility just above the last one returned from this function"""
        if self._last_returned_from_next < 0:
            self._last_returned_from_next = 0
            return self.worst()
        if self._last_returned_from_next < self._last_rational:
            self._last_returned_from_next += 1
            return self.outcomes[self._last_returned_from_next]
        return None

    def some(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        n: int | None = None,
    ) -> list[Outcome]:
        """
        Finds some outcomes with the given utility value (if discrete, all)

        Args:
            rng: The range. If a value, outcome utilities must match it exactly
            normalized: if given, consider the range as a normalized range betwwen 0 and 1 representing lowest and highest utilities.
            n: The maximum number of outcomes to return

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        if not self._ufun.is_stationary():
            self.init()

        if len(self._waypoints) <= 0:
            return []
        rng = self._un_normalize_range(rng, normalized, True)
        mn, mx = rng
        lo, hi = self._get_limiting_waypoints(mn, mx)
        results = []
        for util, w in zip(self.utils[lo : hi + 1], self.outcomes[lo : hi + 1]):
            if util > mx:
                break
            if util < mn:
                continue
            results.append(w)
        if n and len(results) >= n:
            return random.sample(results, n)
        return results

    def all(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
    ) -> list[Outcome]:
        """
        Finds all outcomes with in the given utility value range

        Args:
            rng: The range. If a value, outcome utilities must match it exactly

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        os_ = self._ufun.outcome_space
        if not os_:
            raise ValueError(f"Unknown outcome space. Cannot invert the ufun")

        if os_.is_discrete():
            return self.some(rng, normalized)
        raise ValueError(
            f"Cannot find all outcomes in a range for a continuous outcome space (there is in general an infinite number of them)"
        )

    def _indx_of_worst_in(self, rng: tuple[float, float] | float, normalized: bool):
        if not self._ufun.is_stationary():
            self.init()
        rng = self._un_normalize_range(rng, normalized, False)
        mn, mx = rng
        lo, hi = self._get_limiting_waypoints(mn, mx)
        return index_above_or_equal(self.utils, mn, lo, hi), mn, mx

    def _indx_of_best_in(self, rng: tuple[float, float] | float, normalized: bool):
        if not self._ufun.is_stationary():
            self.init()
        rng = self._un_normalize_range(rng, normalized, True)
        mn, mx = rng
        lo, hi = self._get_limiting_waypoints(mn, mx)
        return index_below_or_equal(self.utils, mx, lo, hi), mn, mx

    def worst_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        cycle: bool = True,
    ) -> Outcome | None:
        """
        Finds an outcome with the lowest possible utility within the given range (if any)

        Args:
            rng: Range of utilities to find the outcome within
            normalized: If true, the range will be interpreted as on a scaled ufun with range (0-1)
            cycle: If given, the system will cycle between outcomes with the same utility value (up to `eps`)

        Remarks:
            - Returns None if no such outcome exists, or the worst outcome in the given range is irrational
            - This is an O(log(n)) operation (where n is the number of outcomes)
        """
        indx, mn, mx = self._indx_of_worst_in(rng, normalized)
        if indx > self._last_rational:
            # fail if this worst is irrational
            return None
        if self.utils[indx] > mx:
            # fail if the found outcome is actually worse than the maximum allowed. Should never happen
            return None
        if mn < self._smallest_val:
            self._smallest_indx, self._smallest_val = indx, mn
        if cycle and indx:
            self._cycle_around(indx)
        return self.outcomes[indx]

    def best_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        cycle: bool = True,
    ) -> Outcome | None:
        """
        Finds an outcome with the highest possible utility within the given range (if any)

        Args:
            rng: Range of utilities to find the outcome within
            normalized: If true, the range will be interpreted as on a scaled ufun with range (0-1)
            cycle: If given, the system will cycle between outcomes with the same utility value (up to `eps`)

        Remarks:
            - Returns None if no such outcome exists
            - This is an O(log(n)) operation (where n is the number of outcomes)
        """
        indx, mn, mx = self._indx_of_best_in(rng, normalized)
        if indx < 0:
            return None
        if self.utils[indx] < mn:
            return None
        if mx > self._largest_val:
            self._largest_indx, self._largest_val = indx, mx
        if cycle and indx:
            self._cycle_around(indx)
        return self.outcomes[indx]

    def _in(self, x, rng):
        return rng[0] - EPS <= x <= rng[1] + EPS

    def one_in(
        self,
        rng: tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        """
        Finds an outcome within the given range of utilities.

        Remarks:
            - This is an O(log(n)) operation (where n is the number of outcomes)
            - We only search rational outcomes
        """
        mx, rmn, rmx = self._indx_of_best_in(rng, normalized)
        if mx > self._last_rational or not self._in(self.utils[mx], (rmn, rmx)):
            return None
        mn, rmn, rmx = self._indx_of_worst_in(rng, normalized)
        if mn < 0 or not self._in(self.utils[mn], (rmn, rmx)):
            return None
        if mn < 0:
            mn = 0
        if mx > self._last_rational:
            mx = self._last_rational

        def recover_from_failure():
            if fallback_to_higher and rng[1] < 1 - EPS:
                return self.one_in(
                    (rng[0], 1 if normalized else float(self._ufun.max())),
                    normalized,
                    fallback_to_higher=False,
                    fallback_to_best=fallback_to_best,
                )
            if fallback_to_best:
                return self._best
            return None

        if mx < mn:
            # TODO: Something is wrong here. When using stochastic aspiration mn > mx a lot and this leads to strange behavior.
            warnings.warn(
                f"Utility Inverter: {mx=}, {mn=} but we expect mn < mx. Could not find any outcomes in the range given",
                NegmasUnexpectedValueWarning,
            )
            return recover_from_failure()
            # return self.outcomes[min(mx, mn)]
            # mx, mn = mn, mx

        if mn == mx:
            return self.outcomes[mn]
        indx = random.randint(mn, mx)
        return self.outcomes[indx]

    def _cycle_around(self, indx: int) -> None:
        mn_indx, mx_indx = self._near_range.get(indx, (indx, indx))
        if mn_indx < mx_indx:
            np.roll(self.utils[mn_indx : mx_indx + 1], 1)

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        """
        Finds outcomes within the given fractions of utility values (the fractions must be between zero and one).

        Remarks:
            - This is an O(n) operation (where n is the number of outcomes)
        """
        if not self._ufun.is_stationary():
            self.init()
        n = self._last_rational + 1
        rng = (rng[0] * n, rng[1] * n)
        rng = (
            max(self._last_rational - math.floor(rng[1]), 0),
            min(n, self._last_rational - math.floor(rng[0])),
        )
        return list(reversed(self.outcomes[int(rng[0]) : int(rng[1]) + 1]))

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        """
        Finds outcomes within the given indices with the best at index 0 and the worst at largest index.

        Remarks:
            - Works only for discrete outcome spaces
        """
        if not self._ufun.is_stationary():
            self.init()
        n = self._last_rational + 1
        rng = (self._last_rational - rng[1], self._last_rational - rng[0])
        rng = (max(rng[0], 0), min(rng[1], n))
        return list(reversed(self.outcomes[rng[0] : rng[1] + 1]))

    def min(self) -> float:
        """
        Finds the minimum utility value (of the rational outcomes if constructed with `sort_rational_only`)
        """
        if not self._ufun.is_stationary():
            self.init()
        return self.utils[0] if self._last_rational >= 0 else float("inf")

    def max(self) -> float:
        """
        Finds the maximum rational utility value (of the rational outcomes if constructed with `sort_rational_only`)
        """
        if not self._ufun.is_stationary():
            self.init()
        return (
            self.utils[self._last_rational]
            if self._last_rational >= 0
            else float("-inf")
        )

    def worst(self) -> Outcome | None:
        """
        Finds the worst  outcome (of the rational outcomes if constructed with `sort_rational_only`)
        """
        if not self._ufun.is_stationary():
            self.init()
        return self.outcomes[0] if self._last_rational >= 0 else None

    def best(self) -> Outcome | None:
        """
        Finds the best  outcome (of the rational outcomes if constructed with `sort_rational_only`)
        """
        if not self._ufun.is_stationary():
            self.init()
        return self.outcomes[self._last_rational] if self._last_rational >= 0 else None

    def minmax(self) -> tuple[float, float]:
        """
        Finds the minimum and maximum utility values that can be returned.

        Remarks:

            - These may be different from the results of `ufun.minmax()` as they can be approximate.
            - Will only consider rational outcomes if constructed with `sort_rational_only`
        """
        return self.min(), self.max()

    def extreme_outcomes(self) -> tuple[Outcome | None, Outcome | None]:
        """
        Finds the worst and best outcomes that can be returned.

        Remarks:
            These may be different from the results of `ufun.extreme_outcomes()` as they can be approximate.
        """
        return self.worst(), self.best()

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Calling an inverse ufun directly is equivalent to calling `one_in()`
        """
        return self.one_in(rng, normalized)

    def outcome_at(self, indx: int) -> Outcome | None:
        n = len(self.outcomes)
        if indx >= n:
            return None
        if indx <= self._last_rational:
            return self.outcomes[self._last_rational - indx]
        return self.outcomes[indx]

    def utility_at(self, indx: int) -> float:
        n = len(self.outcomes)
        if indx >= n:
            return float("-inf")
        if indx <= self._last_rational:
            return self.utils[self._last_rational - indx]
        return self.utils[indx]


class PresortingInverseUtilityFunctionBruteForce(InverseUFun):
    """
    A utility function inverter that uses pre-sorting.

    The outcome-space is sampled if it is continuous and enumerated if it is discrete
    during the call to `init()` and an ordered list of outcomes with their utility
    values is then cached.


    Args:
        ufun: The utility function to be inverted
        levels: discretization levels per issue
        max_cache_size: maximum allowed number of outcomes in the resulting inverse
        sort_rational_only: If true, rational outcomes will be sorted but irrational outcomes will not be sorted (should be faster if the reserved value is high)
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        levels: int = 10,
        max_cache_size: int = 10_000_000_000,
        rational_only: bool = False,
    ):
        self._ufun = ufun
        self.max_cache_size = max_cache_size
        self.levels = levels
        self._initialized = False
        self._ordered_outcomes: list[tuple[float, Outcome]] = []
        self.rational_only = rational_only

    @property
    def initialized(self):
        return self._initialized

    @property
    def ufun(self):
        return self._ufun

    def reset(self) -> None:
        self._initialized = False
        self._orddered_outcomes = []

    def init(self):
        outcome_space = self._ufun.outcome_space
        if outcome_space is None:
            raise ValueError("Cannot find the outcome space.")
        self._worst, self._best = self._ufun.extreme_outcomes()
        self._min, self._max = float(self._ufun(self._worst)), float(
            self._ufun(self._best)
        )
        self._range = self._max - self._min
        self._offset = self._min / self._range if self._range > EPS else self._min
        for l in range(self.levels, 0, -1):
            n = outcome_space.cardinality_if_discretized(l)
            if n <= self.max_cache_size:
                break
        else:
            raise ValueError(
                f"Cannot discretize keeping cache size at {self.max_cache_size}. "
                f"Outcome space cardinality is {outcome_space.cardinality}\nOutcome space: {outcome_space}"
            )
        os = outcome_space.to_discrete(levels=l, max_cardinality=self.max_cache_size)
        outcomes = os.enumerate_or_sample(max_cardinality=self.max_cache_size)
        utils = [float(self._ufun.eval(_)) for _ in outcomes]
        warn_if_slow(
            len(utils),
            "Inverting a large utility function",
            lambda x: (x * math.log2(x)) if x else x,
        )
        r = self._ufun.reserved_value
        r = float(r) if r is not None else float("-inf")
        if self.rational_only:
            rational, irrational = [], []
            ur, uir = [], []
            for u, o in zip(utils, outcomes):
                if u >= r:
                    rational.append(o)
                    ur.append(u)
                else:
                    irrational.append(o)
                    uir.append(u)
        else:
            rational, irrational = list(outcomes), []
            ur, uir = utils, []
        self._ordered_outcomes = sorted(
            zip(ur, rational, strict=True), key=lambda x: -x[0]
        )
        self._last_rational = len(rational) - 1
        if irrational:
            self._ordered_outcomes += list(zip(uir, irrational, strict=True))
        self._initialized = True

    def _normalize_range(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        for_best: bool,
    ) -> tuple[float, float]:
        if not isinstance(rng, Iterable):
            rng = (rng - EPS, rng + EPS)
        if not normalized:
            return rng
        mn, mx = self._min, self._max
        d = mx - mn
        if d < EPS:
            return tuple(1.0 if for_best else 0.0 for _ in rng)
        return tuple(_ * d + mn for _ in rng)

    def some(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        n: int | None = None,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> list[Outcome]:
        """
        Finds some outcomes with the given utility value (if discrete, all)

        Args:
            rng: The range. If a value, outcome utilities must match it exactly
            normalized: if given, consider the range as a normalized range betwwen 0 and 1 representing lowest and highest utilities.
            n: The maximum number of outcomes to return

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        rmx = rng[1] if isinstance(rng, Iterable) else rng
        rmn = rng[0] if isinstance(rng, Iterable) else rng

        def recover_from_failure():
            if fallback_to_higher and rmx < 1 - EPS:
                return self.some(
                    (rmn, 1 if normalized else float(self._ufun.max())),
                    normalized,
                    fallback_to_higher=False,
                    fallback_to_best=fallback_to_best,
                )
            if fallback_to_best:
                return [self._ufun.best()]
            return []

        if not self._ufun.is_stationary():
            self.init()
        rng = self._normalize_range(rng, normalized, True)
        mn, mx = rng
        # todo use bisection
        results = []
        for util, w in self._ordered_outcomes:
            if util > mx:
                continue
            if util < mn:
                break
            results.append(w)
        if n and len(results) >= n:
            return random.sample(results, n)
        if not results:
            return recover_from_failure()
        return results

    def all(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
    ) -> list[Outcome]:
        """
        Finds all outcomes with in the given utility value range

        Args:
            rng: The range. If a value, outcome utilities must match it exactly

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        os_ = self._ufun.outcome_space
        if not os_:
            raise ValueError(f"Unknown outcome space. Cannot invert the ufun")

        if os_.is_discrete():
            return self.some(
                rng, normalized, fallback_to_higher=False, fallback_to_best=False
            )
        raise ValueError(
            f"Cannot find all outcomes in a range for a continuous outcome space (there is in general an infinite number of them)"
        )

    def worst_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        if not self._ufun.is_stationary():
            self.init()
        rng = self._normalize_range(rng, normalized, False)
        mn, mx = rng
        if not self._ordered_outcomes:
            return None
        wbefore = None
        for util, w in self._ordered_outcomes[: self._last_rational + 1]:
            if util > mx:
                continue
            if util < mn:
                return wbefore
            wbefore = w
        return wbefore

    def best_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        if not self._ufun.is_stationary():
            self.init()
        rng = self._normalize_range(rng, normalized, True)
        mn, mx = rng
        for util, w in self._ordered_outcomes[: self._last_rational + 1]:
            if util > mx:
                continue
            if util < mn:
                return None
            return w
        return None

    def one_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        lst = self.some(
            rng,
            normalized,
            fallback_to_higher=fallback_to_higher,
            fallback_to_best=fallback_to_best,
        )

        if not lst:
            return None
        if len(lst) == 1:
            return lst[0]
        return lst[random.randint(0, len(lst) - 1)]

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        """
        Finds outcomes within the given fractions of utility values (the fractions must be between zero and one)
        """
        if not self._ufun.is_stationary():
            self.init()
        n = self._last_rational
        rng = (max(rng[0] * n, 0), min(rng[1] * n, n))
        return [_[1] for _ in self._ordered_outcomes[int(rng[0]) : int(rng[1]) + 1]]

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        """
        Finds outocmes within the given indices with the best at index 0 and the worst at largest index.

        Remarks:
            - Works only for discrete outcome spaces
        """
        if not self._ufun.is_stationary():
            self.init()
        n = self._last_rational + 1
        rng = (max(rng[0], 0), min(rng[1], n))
        return [_[1] for _ in self._ordered_outcomes[rng[0] : rng[1] + 1]]

    def min(self) -> float:
        """
        Finds the minimum utility value
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return (
            self._ordered_outcomes[self._last_rational][0]
            if self._last_rational >= 0
            else float("inf")
        )

    def max(self) -> float:
        """
        Finds the maximum utility value
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return (
            self._ordered_outcomes[0][0] if self._last_rational >= 0 else float("-inf")
        )

    def worst(self) -> Outcome | None:
        """
        Finds the worst  outcome
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return (
            self._ordered_outcomes[self._last_rational][1]
            if self._last_rational >= 0
            else None
        )

    def best(self) -> Outcome | None:
        """
        Finds the best  outcome
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return self._ordered_outcomes[0][1] if self._last_rational >= 0 else None

    def minmax(self) -> tuple[float, float]:
        """
        Finds the minimum and maximum utility values that can be returned.

        Remarks:
            These may be different from the results of `ufun.minmax()` as they can be approximate.
        """
        return self.min(), self.max()

    def extreme_outcomes(self) -> tuple[Outcome | None, Outcome | None]:
        """
        Finds the worst and best outcomes that can be returned.

        Remarks:
            These may be different from the results of `ufun.extreme_outcomes()` as they can be approximate.
        """
        return self.worst(), self.best()

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Calling an inverse ufun directly is equivalent to calling `one_in()`
        """
        return self.one_in(rng, normalized)

    def outcome_at(self, indx: int) -> Outcome | None:
        if indx >= len(self._ordered_outcomes):
            return None
        return self._ordered_outcomes[indx][1]

    def utility_at(self, indx: int) -> float:
        if indx >= len(self._ordered_outcomes):
            return float("-inf")
        return self._ordered_outcomes[indx][0]
