"""Utility function inversion through random sampling of the outcome space."""

from __future__ import annotations

import math
from typing import Iterable

from negmas.outcomes import Outcome

from ..base_ufun import BaseUtilityFunction
from ..protocols import InverseUFun
from ._common import EPS

__all__ = ["SamplingInverseUtilityFunction"]


class SamplingInverseUtilityFunction(InverseUFun):
    """
    A utility function inverter that uses sampling.

    Nothing is done during initialization so the fixed cost of this inverter is minimal.
    Nevertheless, each time the system is asked to find an outcome within some range, it uses
    random sampling which is very inefficient and suffers from the curse of dimensionality.

    Remarks:
        - `min`, `max`, `worst`, `best` and `minmax` are **not** computed through sampling.
          They simply delegate to the wrapped ufun's own `minmax`/`extreme_outcomes` (which
          may use closed-form solutions for some ufun types, e.g. linear ufuns, or fall back
          to sampling internally). This avoids duplicating (weaker) sampling logic here.
        - `within_fractions` and `within_indices` are approximate: because this inverter never
          maintains a full ranking of outcomes, a fresh sample is ranked every time these are
          called. Use `PresortingInverseUtilityFunction` if you need exact results.
    """

    def __init__(self, ufun: BaseUtilityFunction, max_samples_per_call: int = 10_000):
        """Initializes the instance."""
        self._ufun = ufun
        self.max_samples_per_call = max_samples_per_call
        self._initialized = True

    @property
    def initialized(self):
        """Whether the inverter has been initialized."""
        return self._initialized

    @property
    def ufun(self):
        """The utility function being inverted."""
        return self._ufun

    def init(self):
        """Initialize the inverter (no-op for sampling-based inverter)."""
        pass

    def all(self, rng: float | tuple[float, float]) -> list[Outcome]:
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
        self, rng: float | tuple[float, float], normalized: bool, n: int | None = None
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
        """
        Finds an outcome with the lowest utility within the given range using sampling.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].

        Returns:
            The outcome with lowest utility in the range, or None if not found.
        """
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
        """
        Finds an outcome with the highest utility within the given range using sampling.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].

        Returns:
            The outcome with highest utility in the range, or None if not found.
        """
        some = self.some(rng, normalized)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        best_util, best = float("-inf"), None
        for o in some:
            util = self._ufun(o)
            if util > best_util:
                best_util, best = util, o
        return best

    def one_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        """
        Finds any outcome within the given utility range using random sampling.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].
            fallback_to_higher: If True, expand range upward when no match found.
            fallback_to_best: If True, return best outcome as last resort.

        Returns:
            An outcome within the range, or None if not found and fallbacks disabled.
        """
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
            - Delegates to `ufun.minmax()` (which may itself use closed-form solutions for
              some ufun types or fall back to sampling) instead of re-implementing sampling
              here, since this is cheaper and at least as accurate.
        """
        mn, mx = self._ufun.minmax()
        return float(mn), float(mx)

    def min(self) -> float:
        """Finds the minimum utility value that can be returned."""
        return self.minmax()[0]

    def max(self) -> float:
        """Finds the maximum utility value that can be returned."""
        return self.minmax()[1]

    def extreme_outcomes(self) -> tuple[Outcome | None, Outcome | None]:
        """
        Finds the worst and best outcomes that can be returned.

        Remarks:
            - Delegates to `ufun.extreme_outcomes()` for the same reason as `minmax()`.
        """
        return self._ufun.extreme_outcomes()

    def worst(self) -> Outcome | None:
        """Finds the worst outcome."""
        return self.extreme_outcomes()[0]

    def best(self) -> Outcome | None:
        """Finds the best outcome."""
        return self.extreme_outcomes()[1]

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        """
        Approximates the outcomes within the given fractions of utility values using sampling.

        Remarks:
            - This is inherently approximate: `max_samples_per_call` outcomes are drawn,
              sorted by utility, and the outcomes whose rank (as a fraction of the sample)
              falls within `rng` are returned. Use `PresortingInverseUtilityFunction` for
              exact results.
        """
        if not self._ufun.outcome_space:
            return []
        outcomes = list(
            self._ufun.outcome_space.sample(self.max_samples_per_call, False, False)
        )
        if not outcomes:
            return []
        scored = sorted(
            ((float(self._ufun(o)), o) for o in outcomes), key=lambda x: -x[0]
        )
        n = len(scored)
        lo, hi = min(rng), max(rng)
        start = max(0, int(lo * n))
        end = min(n, math.ceil(hi * n))
        return [o for _, o in scored[start:end]]

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        """
        Approximates the outcomes within the given rank indices (0 = best) using sampling.

        Remarks:
            - This is inherently approximate as `SamplingInverseUtilityFunction` never
              maintains a full ranking of outcomes. Use `PresortingInverseUtilityFunction`
              for exact index-based access.
        """
        if not self._ufun.outcome_space:
            return []
        lo, hi = max(0, min(rng)), max(rng)
        n_samples = max(self.max_samples_per_call, hi + 1)
        outcomes = list(self._ufun.outcome_space.sample(n_samples, False, False))
        if not outcomes:
            return []
        scored = sorted(
            ((float(self._ufun(o)), o) for o in outcomes), key=lambda x: -x[0]
        )
        return [o for _, o in scored[lo : min(len(scored), hi + 1)]]

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Calling an inverse ufun directly is equivalent to calling `one_in()`
        """
        return self.one_in(rng, normalized)
