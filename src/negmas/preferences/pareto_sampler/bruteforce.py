"""Exact brute-force Pareto sampler using negmas Pareto frontier utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
import numpy as np

from negmas.preferences.ops import pareto_frontier_bf

if TYPE_CHECKING:
    from negmas.preferences.base_ufun import BaseUtilityFunction

__all__ = ["BruteForceParetoSampler"]


class BruteForceParetoSampler:
    """Exact Pareto sampler via full enumeration and exact frontier extraction.

    Args:
        ufun: Own utility function.
        opponent_ufun: Opponent utility function estimate.
        max_cardinality: Max outcomes to enumerate.
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        opponent_ufun: BaseUtilityFunction | None = None,
        max_cardinality: int = 1_000_000,
    ) -> None:
        self._ufun = ufun
        self._opponent_ufun = opponent_ufun
        self._max_cardinality = max_cardinality
        self._initialized = False
        self._pareto_front: list[Outcome] = []

    @property
    def ufun(self) -> BaseUtilityFunction:
        return self._ufun

    @property
    def initialized(self) -> bool:
        return self._initialized

    def init(self, opponent_ufun: BaseUtilityFunction | None = None) -> None:
        if opponent_ufun is not None:
            self._opponent_ufun = opponent_ufun
        if self._opponent_ufun is None:
            self._pareto_front = []
            self._initialized = False
            return
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
        indices = pareto_frontier_bf(points, sort_by_welfare=False)
        self._pareto_front = [outcomes[int(i)] for i in indices]
        self._initialized = True

    def _to_raw_util(self, norm: float) -> float:
        mn, mx = self._ufun.minmax()
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
            self.init(opponent_ufun)
        elif not self._initialized:
            self.init()
        if not self._initialized:
            return []
        raw_min = self._to_raw_util(min_util) if normalized else min_util
        result = [o for o in self._pareto_front if float(self._ufun(o)) >= raw_min]
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
            self.init(opponent_ufun)
        elif not self._initialized:
            self.init()
        if not self._initialized or self._opponent_ufun is None:
            return None
        opp = self._opponent_ufun
        raw_min = self._to_raw_util(min_util) if normalized else min_util
        feasible = [o for o in self._pareto_front if float(self._ufun(o)) >= raw_min]
        if not feasible:
            return None
        return max(feasible, key=lambda o: float(opp(o)))
