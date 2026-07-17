"""Utility function inversion through pre-sorting all (or a sample of) outcomes.

This is the main/default presorting-based inverter. See `presorting_legacy.py` for a
functionally-equivalent (but slower, legacy) sibling that adds coarse waypoint-based
search-window narrowing.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

import numpy as np
from numpy import floating
from numpy.typing import NDArray

from negmas.outcomes import Outcome
from negmas.warnings import warn_if_slow

from ..base_ufun import BaseUtilityFunction
from ..protocols import InverseUFun
from ._common import (
    EPS,
    _nearest_around,
    _un_normalize_range,
    index_above_or_equal,
    index_below_or_equal,
)

__all__ = ["PresortingInverseUtilityFunction"]


class PresortingInverseUtilityFunction(InverseUFun):
    """
    A utility function inverter that uses pre-sorting.

    The outcome-space is sampled if it is continuous and enumerated if it is discrete
    during the call to `init()` and an ordered list of outcomes with their utility
    values is then cached. Queries (`worst_in`/`best_in`/`some`/`one_in`) then
    binary-search this cached, sorted array using `numpy.searchsorted`, giving
    `O(log n)` lookups after an `O(n log n)` one-time construction cost.

    This is the **recommended/default** presorting-based inverter.
    `PresortingLegacyInverseUtilityFunction` is a functionally-equivalent (but slower,
    legacy) sibling that adds coarse waypoint-based search-window narrowing before the
    final bisection; use it only if you specifically need that legacy behavior for
    backwards compatibility.

    Args:
        ufun: The utility function to be inverted
        levels: discretization levels per issue
        max_cache_size: maximum allowed number of outcomes in the resulting inverse
        rational_only: If true, rational outcomes will be sorted but irrational outcomes will not be sorted (should be faster if the reserved value is high)
        eps: Absolute difference between utility values to consider them equal (zero or negative to disable).
        rel_eps: Relative difference between utility values to consider them equal (zero or negative to disable).
        clamp_tolerance: Maximum distance (in utility units) that `worst_in`/`best_in`/`one_in`
            are allowed to *clamp* outside the requested range when the range contains no
            outcome. When the requested range is empty, these methods return the nearest
            achievable outcome on the appropriate side (`worst_in` clamps up, `best_in`
            clamps down) so negotiators never stall. If that nearest outcome lies further
            than `clamp_tolerance` outside the range, `None` is returned instead (and
            `one_in` falls back per its `fallback_to_*` flags). Defaults to
            ``float("inf")`` which disables the cap entirely (unbounded clamping — the
            historical behavior).

    Remarks:
        - The actual limit used to judge ufun equality is max(eps, rel_eps * range) where range is the difference between max and min utilities for rational outcomes.
          Set both eps, rel_eps to zero or a negative number to disable cycling through outcomes with equal utilities (or set cycle=False when calling the appropriate function).
        - `worst_in`/`best_in`/`some`/`one_in` binary-search directly over the full
          sorted range of rational outcomes (`[0, last_rational]`).
        - Clamping is always to the single nearest achievable outcome (it never skips a
          closer one), but its distance is only bounded by the gap in the utility
          distribution; use `clamp_tolerance` to cap it.
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        levels: int = 10,
        max_cache_size: int = 10_000_000_000,
        rational_only: bool = False,
        eps: float = 1e-12,
        rel_eps: float = 1e-6,
        clamp_tolerance: float = float("inf"),
    ):
        self._ufun = ufun
        self.max_cache_size = max_cache_size
        self.levels = levels
        self._initialized = False
        self.outcomes: list[Outcome] = []
        self._last_rational: int = -1
        self.utils: NDArray[floating[Any]] = []  # type: ignore
        self.rational_only = rational_only
        self._last_returned_from_next: int = -1
        self._eps = eps
        self._rel_eps = rel_eps
        self._clamp_tolerance = clamp_tolerance
        self._near_range: dict[int, tuple[int, int]] = dict()

    @property
    def initialized(self):
        """Whether the inverter has been initialized with sorted outcomes."""
        return self._initialized

    @property
    def ufun(self):
        """The utility function being inverted."""
        return self._ufun

    def reset(self) -> None:
        """Clears cached outcomes and resets the inverter to uninitialized state."""
        self._initialized = False
        self.outcomes, self.utils = [], []  # type: ignore
        self._last_returned_from_next = -1
        self._near_range = dict()

    def init(self):
        """Initializes the inverter by enumerating and sorting outcomes by utility."""
        outcome_space = self._ufun.outcome_space
        if outcome_space is None:
            raise ValueError("Cannot find the outcome space.")
        self._worst, self._best = self._ufun.extreme_outcomes()
        self._min, self._max = (
            float(self._ufun(self._worst)),
            float(self._ufun(self._best)),
        )
        self._range = self._max - self._min
        self._offset = self._min / self._range if self._range > EPS else self._min
        for L in range(self.levels, 0, -1):
            n = outcome_space.cardinality_if_discretized(L)
            if n <= self.max_cache_size:
                break
        else:
            raise ValueError(
                f"Cannot discretize keeping cache size at {self.max_cache_size}. Outcome space cardinality is {outcome_space.cardinality}\nOutcome space: {outcome_space}"
            )
        os = outcome_space.to_discrete(levels=L, max_cardinality=self.max_cache_size)
        outcomes = list(os.enumerate_or_sample(max_cardinality=self.max_cache_size))
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
            rational, irrational = outcomes, []
            ur, uir = utils, []
        ur, uir = (
            np.asarray(ur, dtype=float).flatten(),
            np.asarray(uir, dtype=float).flatten(),
        )
        indices = np.argsort(ur)
        ur_sorted = ur[indices]
        if len(ur_sorted) > 0:
            with np.errstate(under="ignore"):
                relative_part = self._rel_eps * (ur_sorted[-1] - ur_sorted[0])

            eps = max(self._eps, relative_part)
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
                        for indx in range(mn, mx + 1):
                            self._near_range[indx] = (mn, mx)
            except Exception:
                pass

        self._initialized = True
        self._last_rational = len(rational) - 1
        self.outcomes = [rational[_] for _ in indices] + irrational
        self.utils = np.hstack((ur_sorted, uir))

    def _un_normalize_range(
        self, rng: float | tuple[float, float], normalized: bool, for_best: bool
    ) -> tuple[float, float] | None:
        return _un_normalize_range(rng, normalized, self._min, self._max, for_best)  # type: ignore

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
        self, rng: float | tuple[float, float], normalized: bool, n: int | None = None
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

        if self._last_rational < 0:
            return []
        rng = self._un_normalize_range(rng, normalized, False)
        if rng is None:
            return []
        mn, mx = rng
        lo, hi = 0, self._last_rational
        if lo > hi:
            return []
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

    def all(self, rng: float | tuple[float, float], normalized: bool) -> list[Outcome]:
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
            raise ValueError("Unknown outcome space. Cannot invert the ufun")

        if os_.is_discrete():
            return self.some(rng, normalized)
        raise ValueError(
            "Cannot find all outcomes in a range for a continuous outcome space (there is in general an infinite number of them)"
        )

    def _indx_of_worst_in(self, rng: tuple[float, float] | float, normalized: bool):
        """Returns the index of the outcome with the lowest utility that is still
        `>= mn` (i.e. the worst rational outcome within `[mn, mx]`, if any). Callers
        must check `self.utils[indx]` is actually `<= mx` (and `indx <= last_rational`)
        before using the result."""
        if not self._ufun.is_stationary():
            self.init()
        rng = self._un_normalize_range(rng, normalized, False)
        if rng is None:
            return 0, float("inf"), float("-inf")
        mn, mx = rng
        if self._last_rational < 0:
            return 0, mn, mx
        return index_above_or_equal(self.utils, mn, 0, self._last_rational), mn, mx

    def _indx_of_best_in(self, rng: tuple[float, float] | float, normalized: bool):
        """Returns the index of the outcome with the highest utility that is still
        `<= mx` (i.e. the best rational outcome within `[mn, mx]`, if any). Callers
        must check `self.utils[indx]` is actually `>= mn` before using the result."""
        if not self._ufun.is_stationary():
            self.init()
        rng = self._un_normalize_range(rng, normalized, True)
        if rng is None:
            return -1, float("inf"), float("-inf")
        mn, mx = rng
        if self._last_rational < 0:
            return -1, mn, mx
        return index_below_or_equal(self.utils, mx, 0, self._last_rational), mn, mx

    def worst_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        cycle: bool = True,
        eps: float = EPS,
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

        # print(f"{self.__class__.__name__}.worst_in({rng}, normalized={normalized})")
        indx_, mn, mx = self._indx_of_worst_in(rng, normalized)
        if self._last_rational < 0 or indx_ > self._last_rational:
            # fail if this worst is irrational or there are no rational outcomes
            return None
        # Clamp to the nearest achievable outcome around indx_. When the requested
        # range lies entirely outside the currently available (possibly discounted)
        # utilities, this returns the nearest boundary outcome instead of None so that
        # negotiators relying on worst_in never stall.
        indx = _nearest_around(
            mn, self.utils, indx_, mn - eps, mx + 2 * eps, eps=2 * eps
        )
        if indx is None:
            # print("indx ix None")
            return None
        if not (
            mn - self._clamp_tolerance <= self.utils[indx] <= mx + self._clamp_tolerance
        ):
            # The nearest achievable outcome is further than the allowed clamp
            # tolerance outside the requested range: refuse to clamp.
            return None
        if cycle and indx:
            self._cycle_around(indx)
        # print(f"{indx=}, {self.outcomes[indx]}")
        return self.outcomes[indx]

    def best_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        cycle: bool = True,
        eps: float = EPS,
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
        indx_, mn, mx = self._indx_of_best_in(rng, normalized)
        if self._last_rational < 0:
            return None
        if indx_ < 0:
            indx_ = 0
        # Clamp to the nearest achievable outcome around indx_ (see worst_in).
        indx = _nearest_around(
            mx, self.utils, indx_, mn - eps, mx + 2 * eps, eps=2 * eps
        )
        if indx is None:
            return None
        if not (
            mn - self._clamp_tolerance <= self.utils[indx] <= mx + self._clamp_tolerance
        ):
            # The nearest achievable outcome is further than the allowed clamp
            # tolerance outside the requested range: refuse to clamp.
            return None
        if cycle and indx:
            self._cycle_around(indx)
        return self.outcomes[indx]

    def _in(self, x, rng):
        return rng[0] - EPS <= x <= rng[1] + EPS

    def one_in(
        self,
        rng: float | tuple[float, float],
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
        # print(
        # f"{self.__class__.__name__}.one_in({rng}, normalized={normalized}, fallback_to_higher={fallback_to_higher}, fallback_to_best={fallback_to_best})"
        # )

        def recover_from_failure():
            """Attempts fallback strategies when no outcome is found in range."""
            rmx = rng[1] if isinstance(rng, Iterable) else rng
            if fallback_to_higher and (not normalized or rmx < 1 - EPS):
                return self.one_in(
                    (rng[0] if isinstance(rng, Iterable) else rng, 1)
                    if normalized
                    else (
                        rng[0] if isinstance(rng, Iterable) else rng,
                        float(self._max),
                    ),
                    normalized,
                    fallback_to_higher=False,
                    fallback_to_best=fallback_to_best,
                )
            if fallback_to_best:
                return self.best()
            return None

        if self._last_rational < 0:
            return recover_from_failure()

        mn_indx, rmn, rmx = self._indx_of_worst_in(rng, normalized)
        mx_indx, _, _ = self._indx_of_best_in(rng, normalized)

        # Clamp both bounds into the valid rational range. When the requested range
        # lies entirely outside the currently available (possibly discounted)
        # utilities, both bounds collapse onto the nearest achievable boundary outcome
        # so we return the closest achievable outcome instead of failing/falling back
        # to the best (which previously made negotiators always offer their top
        # outcome under discounting).
        mn_indx = max(0, min(mn_indx, self._last_rational))
        mx_indx = max(0, min(mx_indx, self._last_rational))
        if mn_indx > mx_indx:
            mn_indx, mx_indx = mx_indx, mn_indx

        indx = random.randint(mn_indx, mx_indx)
        if not (
            rmn - self._clamp_tolerance
            <= self.utils[indx]
            <= rmx + self._clamp_tolerance
        ):
            # The nearest achievable outcome is further than the allowed clamp
            # tolerance outside the requested range: fall back instead of clamping.
            return recover_from_failure()
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
        """
        Returns the outcome at the given rank index (0 = best).

        Args:
            indx: The rank index, where 0 is the best outcome.

        Returns:
            The outcome at that rank, or None if index is out of bounds.
        """
        n = len(self.outcomes)
        if indx >= n:
            return None
        if indx <= self._last_rational:
            return self.outcomes[self._last_rational - indx]
        return self.outcomes[indx]

    def utility_at(self, indx: int) -> float:
        """
        Returns the utility value at the given rank index (0 = best).

        Args:
            indx: The rank index, where 0 is the best outcome.

        Returns:
            The utility value at that rank, or -inf if index is out of bounds.
        """
        n = len(self.outcomes)
        if indx >= n:
            return float("-inf")
        if indx <= self._last_rational:
            return self.utils[self._last_rational - indx]
        return self.utils[indx]
