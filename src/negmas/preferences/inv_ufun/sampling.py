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

    This is a **clamping** inverter (see module docs). When ``worst_in``/``best_in``
    fail to find an in-range outcome (common when the range is narrow and contains
    only a single outcome that the random sample missed), they expand the range
    upward and, if still nothing is found, fall back to the nearest boundary
    outcome (best if the range is in the upper half of the utility range, worst if
    in the lower half). ``one_in`` always had these fallbacks; ``worst_in``/
    ``best_in`` gained them to avoid returning ``None`` and breaking the SAO
    mechanism.

    Args:
        ufun: The utility function to invert
        max_samples_per_call: The maximum number of samples to draw per call
        rel_max_samples_per_call: The maximum number of samples to draw per call relative to the size of the outcome space (default: 1.5)
        eps: The absolute tolerance for utility matching (default: 0.001)
        rel_eps: The relative tolerance for utility matching (default: 0.05)
        multiplier: The number of samples to draw per call relative to the requested number of outcomes (default: 3)
        strict_limit: The fraction of the sampling loop during which we search with zero tolerance  (default: 0.1)


    Remarks:
        - `min`, `max`, `worst`, `best` and `minmax` are **not** computed through sampling.
          They simply delegate to the wrapped ufun's own `minmax`/`extreme_outcomes` (which
          may use closed-form solutions for some ufun types, e.g. linear ufuns, or fall back
          to sampling internally). This avoids duplicating (weaker) sampling logic here.
        - `within_fractions` and `within_indices` are approximate: because this inverter never
          maintains a full ranking of outcomes, a fresh sample is ranked every time these are
          called. Use `PresortingInverseUtilityFunction` if you need exact results.

    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        max_samples_per_call: int = 10_000,
        rel_max_samples_per_call: float = 1.5,
        eps: float = 0.001,
        rel_eps: float = 0.05,
        multiplier: int = 3,
        strict_limit: float = 0.1,
    ):
        """Initializes the instance."""
        self._ufun = ufun
        self.max_samples_per_call = max_samples_per_call
        self.rel_max_samples_per_call = rel_max_samples_per_call
        self._initialized = True
        self.eps = eps
        self.rel_eps = rel_eps
        self.multiplier = multiplier
        self.strict_limit = strict_limit
        self._max_samples: int = self.max_samples_per_call

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
        if self.ufun and self.ufun.outcome_space is not None:
            rl = self.ufun.outcome_space.cardinality * self.rel_max_samples_per_call
            try:
                rl = int(rl)
                self._max_samples = min(rl, self._max_samples)
            except OverflowError:
                pass

    def _tolerance(self, x: float | tuple[float, float]) -> float:
        """Returns the tolerance for utility matching."""
        if isinstance(x, Iterable):
            x = min(x)
        return min(self.eps, self.rel_eps * x)

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
        tol_down = self._tolerance(rng[0])
        tol_up = self._tolerance(rng[1])

        if not n:
            n = n_samples = self._max_samples
        else:
            n_samples = n * self.multiplier
        if self._ufun.outcome_space is None:
            return []
        outcomes = list(self._ufun.outcome_space.sample(n, False, False))
        mn, mx = rng
        u = self.ufun.eval_normalized if normalized else self.ufun.eval

        samples = []
        extra_samples = []
        for _ in outcomes:
            util = u(_)
            if mn <= util <= mx:
                samples.append(_)
            elif mn - tol_down <= util <= mx + tol_up:
                extra_samples.append(_)
        n_samples = len(samples)
        if n_samples >= n:
            return samples[:n]
        if len(extra_samples) >= n - n_samples:
            extra_samples = extra_samples[: n - n_samples]
        return samples + extra_samples

    def worst_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_worst: bool = True,
    ) -> Outcome | None:
        """
        Finds an outcome with the lowest utility within the given range using sampling.

        Fallback behavior (clamping inverter — see module docs):

        * If no in-range outcome is found, and ``fallback_to_higher`` is ``True``,
          the range is expanded upward to ``[rng[0], 1]`` (normalized) or
          ``[rng[0], self.max]`` (raw) and the search is retried without
          ``fallback_to_higher``.
        * If still nothing is found and ``fallback_to_worst`` is ``True``, the
          **nearest boundary** outcome is returned: the best outcome if the
          range lies above the maximum utility, the worst outcome if it lies
          below the minimum. This keeps the fallback close to the requested
          range (important for negotiators that ask for high-utility outcomes).
        * Otherwise ``None`` is returned.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].
            fallback_to_higher: If True, expand the range upward when no match found.
            fallback_to_worst: If True, return the nearest boundary outcome as a
                last resort.

        Returns:
            The outcome with lowest utility in the range, or a fallback, or None.
        """
        some = self.some(rng, normalized)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        worst_util, worst = float("inf"), None
        for o in some:
            util = self._ufun(o)
            if util < worst_util:
                worst_util, worst = util, o
        if worst is not None:
            return worst
        # No in-range outcome found: apply fallbacks.
        rmx = rng[1] if isinstance(rng, Iterable) else rng
        if fallback_to_higher and (not normalized or rmx < 1 - EPS):
            new_rng = (rng[0], 1) if normalized else (rng[0], float(self.max()))
            return self.worst_in(
                new_rng,
                normalized,
                fallback_to_higher=False,
                fallback_to_worst=fallback_to_worst,
            )
        if fallback_to_worst:
            # Return the nearest boundary outcome. If the requested range is
            # in the upper half of the utility range, the best outcome is
            # closer; otherwise the worst is closer.
            u_min, u_max = self.minmax()
            mid = (u_min + u_max) / 2.0
            r_lo = rng[0] if isinstance(rng, Iterable) else rng
            if r_lo >= mid:
                return self._ufun.extreme_outcomes()[1]
            return self._ufun.extreme_outcomes()[0]
        return None

    def best_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        """
        Finds an outcome with the highest utility within the given range using sampling.

        Fallback behavior (clamping inverter — see module docs):

        * If no in-range outcome is found, and ``fallback_to_higher`` is ``True``,
          the range is expanded upward to ``[rng[0], 1]`` (normalized) or
          ``[rng[0], self.max]`` (raw) and the search is retried without
          ``fallback_to_higher``.
        * If still nothing is found and ``fallback_to_best`` is ``True``, the
          best outcome overall is returned.
        * Otherwise ``None`` is returned.

        Args:
            rng: The utility range (single value or (min, max) tuple).
            normalized: Whether the range is normalized to [0, 1].
            fallback_to_higher: If True, expand the range upward when no match found.
            fallback_to_best: If True, return the best outcome as a last resort.

        Returns:
            The outcome with highest utility in the range, or a fallback, or None.
        """
        some = self.some(rng, normalized)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        best_util, best = float("-inf"), None
        for o in some:
            util = self._ufun(o)
            if util > best_util:
                best_util, best = util, o
        if best is not None:
            return best
        # No in-range outcome found: apply fallbacks.
        rmx = rng[1] if isinstance(rng, Iterable) else rng
        if fallback_to_higher and (not normalized or rmx < 1 - EPS):
            new_rng = (rng[0], 1) if normalized else (rng[0], float(self.max()))
            return self.best_in(
                new_rng,
                normalized,
                fallback_to_higher=False,
                fallback_to_best=fallback_to_best,
            )
        if fallback_to_best:
            return self._ufun.extreme_outcomes()[1]
        return None

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
        if self._ufun.outcome_space is None:
            return None
        if not isinstance(rng, Iterable):
            rng = (rng - EPS, rng + EPS)
        u = self.ufun.eval_normalized if normalized else self.ufun.eval
        tol_down = self._tolerance(rng[0])
        tol_up = self._tolerance(rng[1])
        for _ in range(self._max_samples):
            o = list(self._ufun.outcome_space.sample(1))[0]
            if rng[0] <= u(o) <= rng[1]:
                return o
            if _ > self._max_samples * self.strict_limit:
                if rng[0] - tol_down <= u(o) <= rng[1] + tol_up:
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
        if self._ufun.outcome_space is None:
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
        if self._ufun.outcome_space is None:
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
