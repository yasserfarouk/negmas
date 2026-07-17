"""A slow, but always-correct brute-force utility function inverter.

This inverter is the project's **ground truth**: every query is answered by a
straightforward *linear scan* over the (sorted) list of outcomes, with no
bisection, waypoints, clamping or other performance optimizations that could
(subtly) introduce bugs. It is the reference implementation used in tests to
validate the accuracy of the faster inverters (e.g.
`PresortingInverseUtilityFunction`) on small outcome spaces, and should never
need optimization itself.

Caching policy
--------------
The only thing cached is the array of outcomes and their utilities, and this is
done **only for stationary utility functions**. For non-stationary ufuns, the
outcome/utility arrays are rebuilt on every call (via ``init()``) so that the
inverter always reflects the current utilities. Because tests use stationary
ufuns, this makes the ground truth reasonably fast while keeping the logic
trivially correct.
"""

from __future__ import annotations

import math
import random

from negmas.outcomes import Outcome
from negmas.warnings import warn_if_slow

from ..base_ufun import BaseUtilityFunction
from ..protocols import InverseUFun
from ._common import EPS

__all__ = ["BruteForceInverseUtilityFunction"]


class BruteForceInverseUtilityFunction(InverseUFun):
    """
    A dead-simple, always-correct utility function inverter used as ground truth.

    The outcome-space is sampled if it is continuous and enumerated if it is discrete
    during the call to `init()` and an ordered list of outcomes with their utility
    values is then cached (for stationary ufuns only). Every query is answered by a
    plain linear scan over that list.

    This is a **strict** inverter (see module docs). ``worst_in``/``best_in``
    return ``None`` when no outcome's utility falls inside the requested range —
    they never clamp, expand the range, or fall back to an out-of-range outcome.
    ``one_in`` is the exception: it always had ``fallback_to_higher`` and
    ``fallback_to_best`` parameters (both default ``True``), so it never returns
    ``None`` for a non-empty outcome space.

    Args:
        ufun: The utility function to be inverted
        levels: discretization levels per issue
        max_cache_size: maximum allowed number of outcomes in the resulting inverse
        rational_only: If true, rational outcomes will be sorted but irrational outcomes will not be sorted (should be faster if the reserved value is high)

    Remarks:
        - This class is intentionally implemented using simple linear scans over the fully
          sorted outcome list rather than bisection so that it can serve as a trusted,
          easy-to-verify ground truth for testing other (faster, more complex) inverters
          such as `PresortingInverseUtilityFunction`.
        - Unlike the clamping inverters, `best_in`/`worst_in` here are
          **strict**: they return ``None`` when no outcome exists in the requested range
          (no clamping to the nearest boundary outcome). Callers that need a
          non-None result (e.g. `AspirationNegotiator`) must handle ``None`` by
          falling back to the best outcome themselves.
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
        self._last_rational: int = -1
        self._last_returned_from_next: int = -1

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
        self._ordered_outcomes = []
        self._last_rational = -1
        self._last_returned_from_next = -1

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
                f"Cannot discretize keeping cache size at {self.max_cache_size}. "
                f"Outcome space cardinality is {outcome_space.cardinality}\nOutcome space: {outcome_space}"
            )
        os = outcome_space.to_discrete(levels=L, max_cardinality=self.max_cache_size)
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
        self._last_returned_from_next = -1

    def _ensure(self) -> None:
        """Rebuilds the outcome/utility cache when needed.

        For non-stationary ufuns the cache is always rebuilt (utilities may have
        changed). For stationary ufuns it is built once and reused.
        """
        if not self._ufun.is_stationary() or not self._initialized:
            self.init()

    def _un_normalize_range(
        self, rng: float | tuple[float, float], normalized: bool, for_best: bool
    ) -> tuple[float, float]:
        if not isinstance(rng, tuple):
            rng = (float(rng) - EPS, float(rng) + EPS)
        lo, hi = float(rng[0]), float(rng[1])
        if not normalized:
            return (lo, hi)
        mn, mx = self._min, self._max
        d = mx - mn
        if d < EPS:
            v = 0.0 if not for_best else 1.0
            return (v, v)
        return (lo * d + mn, hi * d + mn)

    def _normalize_range(
        self, rng: float | tuple[float, float], normalized: bool, for_best: bool
    ) -> tuple[float, float]:
        if not isinstance(rng, tuple):
            rng = (float(rng) - EPS, float(rng) + EPS)
        lo, hi = float(rng[0]), float(rng[1])
        if not normalized:
            return (lo, hi)
        mn, mx = self._min, self._max
        d = mx - mn
        if d < EPS:
            v = 1.0 if for_best else 0.0
            return (v, v)
        return (lo * d + mn, hi * d + mn)

    def next_worse(self) -> Outcome | None:
        """Returns the rational outcome with utility just below the last one returned from this function"""
        self._ensure()
        if self._last_rational < 0:
            return None
        if self._last_returned_from_next < 0:
            self._last_returned_from_next = 0
            return self._ordered_outcomes[0][1]
        if self._last_returned_from_next < self._last_rational:
            self._last_returned_from_next += 1
            return self._ordered_outcomes[self._last_returned_from_next][1]
        return None

    def next_better(self) -> Outcome | None:
        """Returns the rational outcome with utility just above the last one returned from this function"""
        self._ensure()
        if self._last_rational < 0:
            return None
        if self._last_returned_from_next < 0:
            self._last_returned_from_next = self._last_rational
            return self._ordered_outcomes[self._last_rational][1]
        if self._last_returned_from_next > 0:
            self._last_returned_from_next -= 1
            return self._ordered_outcomes[self._last_returned_from_next][1]
        return None

    def some(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        n: int | None = None,
    ) -> list[Outcome]:
        """
        Finds some outcomes within the given utility range (if discrete, all of them).

        Args:
            rng: The range. If a value, outcome utilities must match it exactly
            normalized: if given, consider the range as a normalized range betwwen 0 and 1 representing lowest and highest utilities.
            n: The maximum number of outcomes to return

        Remarks:
            - Returns only outcomes whose utility lies inside the range (never falls
              back to out-of-range outcomes). May return an empty list.
            - If the outcome-space is discrete, this method will return all outcomes in the given range
        """
        self._ensure()
        if self._last_rational < 0:
            return []
        rng = self._un_normalize_range(rng, normalized, False)
        mn, mx = rng
        results = []
        # `_ordered_outcomes` is sorted by utility descending (index 0 == best).
        for util, w in self._ordered_outcomes[: self._last_rational + 1]:
            if util > mx:
                continue
            if util < mn:
                break
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

    def worst_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        eps: float = EPS,
    ) -> Outcome | None:
        """
        Finds an outcome with the lowest utility within the given range.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].
            eps: Tolerance used when comparing utilities against the range bounds.

        Returns:
            The outcome with lowest utility in the range, or None if not found.
        """
        self._ensure()
        rng = self._un_normalize_range(rng, normalized, False)
        mn, mx = rng
        if self._last_rational < 0:
            return None
        wbefore = None
        # `_ordered_outcomes` is sorted by utility descending (index 0 == best).
        for util, w in self._ordered_outcomes[: self._last_rational + 1]:
            if util > mx + eps:
                continue
            if util < mn - eps:
                return wbefore
            wbefore = w
        return wbefore

    def best_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        eps: float = EPS,
    ) -> Outcome | None:
        """
        Finds an outcome with the highest utility within the given range.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].
            eps: Tolerance used when comparing utilities against the range bounds.

        Returns:
            The outcome with highest utility in the range, or None if not found.
        """
        self._ensure()
        rng = self._un_normalize_range(rng, normalized, True)
        mn, mx = rng
        if self._last_rational < 0:
            return None
        # `_ordered_outcomes` is sorted by utility descending (index 0 == best).
        for util, w in self._ordered_outcomes[: self._last_rational + 1]:
            if util > mx + eps:
                continue
            if util < mn - eps:
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
        """
        Finds any outcome within the given utility range.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].
            fallback_to_higher: If True, expand range upward when no match found.
            fallback_to_best: If True, return best outcome as last resort.

        Returns:
            An outcome within the range, or None if not found and fallbacks disabled.
        """
        lst = self.some(rng, normalized)
        if lst:
            if len(lst) == 1:
                return lst[0]
            return lst[random.randint(0, len(lst) - 1)]

        # No outcome inside the range: apply the same fallbacks as presorting.one_in.
        rmn = rng[0] if isinstance(rng, tuple) else rng
        rmx = rng[1] if isinstance(rng, tuple) else rng
        if fallback_to_higher and (not normalized or rmx < 1 - EPS):
            new_rng = (rmn, 1) if normalized else (rmn, float(self._max))
            return self.one_in(
                new_rng,
                normalized,
                fallback_to_higher=False,
                fallback_to_best=fallback_to_best,
            )
        if fallback_to_best:
            return self._ufun.best()
        return None

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        """
        Finds outcomes within the given fractions of utility values (the fractions must be between zero and one)
        """
        self._ensure()
        n = self._last_rational + 1
        rng = (max(rng[0] * n, 0), min(rng[1] * n, n))
        return [_[1] for _ in self._ordered_outcomes[int(rng[0]) : int(rng[1]) + 1]]

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        """
        Finds outocmes within the given indices with the best at index 0 and the worst at largest index.

        Remarks:
            - Works only for discrete outcome spaces
        """
        self._ensure()
        n = self._last_rational + 1
        rng = (max(rng[0], 0), min(rng[1], n))
        return [_[1] for _ in self._ordered_outcomes[rng[0] : rng[1] + 1]]

    def min(self) -> float:
        """
        Finds the minimum utility value
        """
        self._ensure()
        if not self._ordered_outcomes:
            raise ValueError("No outcomes to find the best")
        return (
            self._ordered_outcomes[self._last_rational][0]
            if self._last_rational >= 0
            else float("inf")
        )

    def max(self) -> float:
        """
        Finds the maximum utility value
        """
        self._ensure()
        if not self._ordered_outcomes:
            raise ValueError("No outcomes to find the best")
        return (
            self._ordered_outcomes[0][0] if self._last_rational >= 0 else float("-inf")
        )

    def worst(self) -> Outcome | None:
        """
        Finds the worst  outcome
        """
        self._ensure()
        if not self._ordered_outcomes:
            raise ValueError("No outcomes to find the best")
        return (
            self._ordered_outcomes[self._last_rational][1]
            if self._last_rational >= 0
            else None
        )

    def best(self) -> Outcome | None:
        """
        Finds the best  outcome
        """
        self._ensure()
        if not self._ordered_outcomes:
            raise ValueError("No outcomes to find the best")
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
        """
        Returns the outcome at the given rank index (0 = best).

        Args:
            indx: The rank index, where 0 is the best outcome.

        Returns:
            The outcome at that rank, or None if index is out of bounds.
        """
        self._ensure()
        if indx >= len(self._ordered_outcomes):
            return None
        return self._ordered_outcomes[indx][1]

    def utility_at(self, indx: int) -> float:
        """
        Returns the utility value at the given rank index (0 = best).

        Args:
            indx: The rank index, where 0 is the best outcome.

        Returns:
            The utility value at that rank, or -inf if index is out of bounds.
        """
        self._ensure()
        if indx >= len(self._ordered_outcomes):
            return float("-inf")
        return self._ordered_outcomes[indx][0]
