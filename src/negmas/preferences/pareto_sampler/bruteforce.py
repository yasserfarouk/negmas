"""Exact brute-force Pareto sampler using negmas Pareto frontier utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
import numpy as np

from negmas.preferences.ops import pareto_frontier_numpy

if TYPE_CHECKING:
    from negmas.preferences.base_ufun import BaseUtilityFunction

__all__ = ["BruteForceParetoSampler"]


class BruteForceParetoSampler:
    """Exact Pareto sampler via full enumeration and exact frontier extraction.

    The utility functions are supplied to `init` (not the constructor): the
    constructor only stores configuration, and the expensive frontier build
    happens in `init`. A single instance can therefore be re-``init``-ed with
    new operands instead of being recreated.

    Args:
        max_cardinality: Max outcomes to enumerate.

    *AI supported (config-only constructor; frontier built in ``init``; uses the
    vectorized ``pareto_frontier_numpy``).*
    """

    def __init__(self, max_cardinality: int = 1_000_000) -> None:
        self._ufun: BaseUtilityFunction | None = None
        self._opponent_ufun: BaseUtilityFunction | None = None
        self._max_cardinality = max_cardinality
        self._initialized = False
        self._pareto_front: list[Outcome] = []

    @property
    def ufun(self) -> BaseUtilityFunction | None:
        return self._ufun

    @property
    def initialized(self) -> bool:
        return self._initialized

    def init(
        self,
        ufun: BaseUtilityFunction | None = None,
        opponent_ufun: BaseUtilityFunction | None = None,
    ) -> None:
        if ufun is not None:
            self._ufun = ufun
        if opponent_ufun is not None:
            self._opponent_ufun = opponent_ufun
        if self._opponent_ufun is None:
            self._pareto_front = []
            self._initialized = False
            return
        if self._ufun is None:
            raise ValueError(
                "BruteForceParetoSampler.init requires a ufun (pass ufun=...)."
            )
        os = self._ufun.outcome_space
        if os is None:
            raise ValueError("BruteForceParetoSampler requires an outcome space.")
        outcomes = list(os.enumerate_or_sample(max_cardinality=self._max_cardinality))
        points = np.asarray(
            [
                [float(self._ufun(outcome)), float(self._opponent_ufun(outcome))]
                for outcome in outcomes
            ],
            dtype=np.float64,
        )
        # ``pareto_frontier_numpy`` is the vectorized exact frontier extractor —
        # ~200x faster than the O(n^2) ``pareto_frontier_bf`` on ~hundreds of
        # points (both give the identical frontier). This matters because callers
        # such as the Nice Tit for Tat offering policy re-run ``init`` every round
        # as the opponent model learns.
        indices = pareto_frontier_numpy(points, sort_by_welfare=False)
        self._pareto_front = [outcomes[int(i)] for i in indices]
        self._initialized = True

    def _to_raw_util(self, norm: float) -> float:
        ufun = self._ufun
        assert ufun is not None  # guaranteed once initialized
        mn, mx = ufun.minmax()
        return float(mn) + norm * float(mx - mn)

    def pareto_outcomes(
        self,
        n: int | None = None,
        *,
        min_util: float = 0.0,
        normalized: bool = False,
        opponent_ufun: BaseUtilityFunction | None = None,
    ) -> list[Outcome]:
        if opponent_ufun is not None:
            self.init(opponent_ufun=opponent_ufun)
        elif not self._initialized:
            self.init()
        if not self._initialized:
            return []
        ufun = self._ufun
        assert ufun is not None  # guaranteed once initialized
        raw_min = self._to_raw_util(min_util) if normalized else min_util
        result = [o for o in self._pareto_front if float(ufun(o)) >= raw_min]
        if n is not None:
            result = result[:n]
        return result

    def best_for_opponent(
        self,
        *,
        min_util: float,
        normalized: bool = False,
        opponent_ufun: BaseUtilityFunction | None = None,
    ) -> Outcome | None:
        if opponent_ufun is not None:
            self.init(opponent_ufun=opponent_ufun)
        elif not self._initialized:
            self.init()
        if not self._initialized or self._opponent_ufun is None:
            return None
        opp = self._opponent_ufun
        ufun = self._ufun
        assert ufun is not None  # guaranteed once initialized
        raw_min = self._to_raw_util(min_util) if normalized else min_util
        feasible = [o for o in self._pareto_front if float(ufun(o)) >= raw_min]
        if not feasible:
            return None
        return max(feasible, key=lambda o: float(opp(o)))
