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
        if not n:
            n = self.max_samples_per_call
        if not self._ufun.outcome_space:
            return []
        return list(self._ufun.outcome_space.sample(n, False, False))

    def worst_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        some = self.some(rng)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        worst_util, worst = float("inf"), None
        for o in some:
            util = self._ufun(o)
            if util < worst_util:
                worst_util, worst = util, o
        return worst

    def best_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        some = self.some(rng)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        best_util, best = float("-inf"), None
        for o in some:
            util = self._ufun(o)
            if util < best_util:
                best_util, best = util, o
        return best

    def one_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        if not self._ufun.outcome_space:
            return None
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        for _ in range(self.max_samples_per_call):
            o = list(self._ufun.outcome_space.sample(1))[0]
            if rng[0] + 1e-7 <= self._ufun(o) <= rng[1] - 1e-7:
                return o
        return None


class PresortingInverseUtilityFunction(InverseUFun):
    def __init__(
        self, ufun: BaseUtilityFunction, levels: int = 10, max_cache_size: int = 100_000
    ):
        self._ufun = ufun
        self.max_cache_size = max_cache_size
        self.levels = levels
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    @property
    def ufun(self):
        return self._ufun

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
        utils = [float(self._ufun(_)) for _ in outcomes]
        self._ordered_outcomes = sorted(zip(utils, outcomes), key=lambda x: -x[0])
        self._initialized = True

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
        os_ = self._ufun.outcome_space
        if not os_:
            raise ValueError(f"Unkonwn outcome space. Cannot invert the ufun")

        if os_.is_discrete():
            return self.some(rng)
        raise ValueError(
            f"Cannot find all outcomes in a range for a continous outcome space (there is in general an infinite number of them)"
        )

    def some(
        self,
        rng: float | tuple[float, float],
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
        if not self._ufun.is_stationary():
            self.init()
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
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
                return results
        return results

    def worst_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        if not self._ufun.is_stationary():
            self.init()
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        mn, mx = rng
        for i, (util, _) in enumerate(self._ordered_outcomes):
            if util >= mn:
                continue
            ubefore, wbefore = self._ordered_outcomes[i - 1 if i > 0 else 0]
            if ubefore > mx:
                return None
            return wbefore
        ubefore, wbefore = self._ordered_outcomes[-1]
        if ubefore > mx:
            return None
        return wbefore

    def best_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        if not self._ufun.is_stationary():
            self.init()
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        mn, mx = rng
        for util, w in self._ordered_outcomes:
            if util <= mx:
                if util < mn:
                    return None
                return w
        return None

    def one_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        lst = self.some(rng)
        if not lst:
            return None
        return lst[random.randint(0, len(lst) - 1)]
