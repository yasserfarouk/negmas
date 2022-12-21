from __future__ import annotations

import random
from typing import Iterable

from negmas.outcomes import Outcome

from .base_ufun import BaseUtilityFunction
from .protocols import InverseUFun

__all__ = [
    "PresortingInverseUtilityFunction",
    "SamplingInverseUtilityFunction",
]


class SamplingInverseUtilityFunction(InverseUFun):
    """
    A utility function inverter that uses sampling.

    Nothing is done during initialization so the fixed cost of this inverter is minimal.
    Nevertheless, each time the system is asked to find an outcome within some range, it uses
    random sampling which is very inefficient and suffers from the curse of dimensinality.
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
            f"Cannot find all outcomes in a range using a SamplingInverseUtilityFunction. Try a PresortedInverseUtilityFunction"
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
            rng = (rng - 1e-5, rng + 1e-5)
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
        self, rng: float | tuple[float, float], normalized: float
    ) -> Outcome | None:
        if not self._ufun.outcome_space:
            return None
        if not isinstance(rng, Iterable):
            rng = (rng - 1e-5, rng + 1e-5)
        u = self.ufun.eval_normalized if normalized else self.ufun.eval
        for _ in range(self.max_samples_per_call):
            o = list(self._ufun.outcome_space.sample(1))[0]
            if rng[0] <= u(o) <= rng[1]:
                return o
        return None

    def minmax(self) -> tuple[float, float]:
        """
        Finds the minimum and maximum utility values that can be returned.

        Remarks:
            These may be different from the results of `ufun.minmax()` as they can be approximate.
        """
        return self.min(), self.max()

    def extreme_outcomes(self) -> tuple[Outcome, Outcome]:
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
    """

    def __init__(
        self, ufun: BaseUtilityFunction, levels: int = 10, max_cache_size: int = 100_000
    ):
        self._ufun = ufun
        self.max_cache_size = max_cache_size
        self.levels = levels
        self._initialized = False
        self._ordered_outcomes: list[tuple[float, Outcome]] = []

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
        self._offset = self._min / self._range if self._range > 1e-5 else self._min
        for l in range(self.levels, 0, -1):
            n = outcome_space.cardinality_if_discretized(l)
            if n <= self.max_cache_size:
                break
        else:
            raise ValueError(
                f"Cannot discretize keeping cach size at {self.max_cache_size}. Outocme space cardinality is {outcome_space.cardinality}\nOutcome space: {outcome_space}"
            )
        os = outcome_space.to_discrete(levels=l, max_cardinality=self.max_cache_size)
        if os.cardinality <= self.max_cache_size:
            outcomes = list(os.sample(self.max_cache_size, False, False))
        else:
            outcomes = list(os.enumerate())[: self.max_cache_size]
        utils = [float(self._ufun.eval(_)) for _ in outcomes]
        self._ordered_outcomes = sorted(zip(utils, outcomes), key=lambda x: -x[0])
        self._initialized = True

    def _normalize_range(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
    ) -> tuple[float, float]:
        if not isinstance(rng, Iterable):
            rng = (rng - 1e-5, rng + 1e-5)
        if not normalized:
            return rng
        mn, mx = self._min, self._max
        d = mx - mn
        if d < 1e-8:
            return tuple(1.0 for _ in rng)
        return tuple(_ * d + mn for _ in rng)

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
        rng = self._normalize_range(rng, normalized)
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
            raise ValueError(f"Unkonwn outcome space. Cannot invert the ufun")

        if os_.is_discrete():
            return self.some(rng, normalized)
        raise ValueError(
            f"Cannot find all outcomes in a range for a continous outcome space (there is in general an infinite number of them)"
        )

    def worst_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        if not self._ufun.is_stationary():
            self.init()
        rng = self._normalize_range(rng, normalized)
        mn, mx = rng
        if not self._ordered_outcomes:
            return None
        wbefore = self._ordered_outcomes[0][1]
        for i, (util, w) in enumerate(self._ordered_outcomes):
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
        rng = self._normalize_range(rng, normalized)
        mn, mx = rng
        for util, w in self._ordered_outcomes:
            if util > mx:
                continue
            if util < mn:
                return None
            return w
        return None

    def one_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        lst = self.some(rng, normalized)
        if not lst:
            return None
        return lst[random.randint(0, len(lst) - 1)]

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        """
        Finds outocmes within the given fractions of utility values (the fractions must be between zero and one)
        """
        if not self._ufun.is_stationary():
            self.init()
        n = len(self._ordered_outcomes)
        rng = (max(rng[0] * n, 0), min(rng[1] * n, len(self._ordered_outcomes)))
        return [_[1] for _ in self._ordered_outcomes[int(rng[0]) : int(rng[1])]]

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        """
        Finds outocmes within the given indices with the best at index 0 and the worst at largest index.

        Remarks:
            - Works only for discrete outcome spaces
        """
        if not self._ufun.is_stationary():
            self.init()
        rng = (max(rng[0], 0), min(rng[1], len(self._ordered_outcomes)))
        return [_[1] for _ in self._ordered_outcomes[rng[0] : rng[1]]]

    def min(self) -> float:
        """
        Finds the minimum utility value
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return self._ordered_outcomes[-1][0]

    def max(self) -> float:
        """
        Finds the maximum utility value
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return self._ordered_outcomes[0][0]

    def worst(self) -> Outcome:
        """
        Finds the worst  outcome
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return self._ordered_outcomes[-1][1]

    def best(self) -> Outcome:
        """
        Finds the best  outcome
        """
        if not self._ufun.is_stationary():
            self.init()
        if not self._ordered_outcomes:
            raise ValueError(f"No outcomes to find the best")
        return self._ordered_outcomes[0][1]

    def minmax(self) -> tuple[float, float]:
        """
        Finds the minimum and maximum utility values that can be returned.

        Remarks:
            These may be different from the results of `ufun.minmax()` as they can be approximate.
        """
        return self.min(), self.max()

    def extreme_outcomes(self) -> tuple[Outcome, Outcome]:
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
