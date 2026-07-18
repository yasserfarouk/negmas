"""MOBANOS (Multi-Objective Branch-And-bound Non-dominated Outcome Sampler)
ParetoSampler implementation.

MOBANOS is a variant of IPS that constructs the Pareto frontier using exact
(non-rounded) utilities, avoiding the discretisation error of IPS at the cost
of a potentially larger intermediate Pareto set.

Reference
---------
De Jonge, D., Rovatsos, M., & Sierra, C. (2019).
*Multi-objective automated negotiation with incomplete information.*
In ECAI 2019.

Description from:
Koça, T., de Jonge, D., & Baarslag, T. (2024).
*Search algorithms for automated negotiation in large domains.*
Algorithms 17(5), 200.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from negmas.outcomes import Outcome


if TYPE_CHECKING:
    from negmas.preferences.base_ufun import BaseUtilityFunction
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

__all__ = ["MOBANOSParetoSampler"]


class MOBANOSParetoSampler:
    """Pareto sampler using the MOBANOS exact iterative construction algorithm.

    MOBANOS is similar to IPS (Iterative Pareto Search) but uses **exact**
    utilities instead of rounded ones.  This avoids any discretisation error
    but can make the intermediate Pareto set exponentially large for many-valued
    issues.  A ``max_pareto_size`` parameter limits memory usage by keeping only
    the most promising candidates when the set grows too large.

    **Requirements**: both ufuns must be ``LinearAdditiveUtilityFunction``
    instances.  ``init()`` raises ``TypeError`` otherwise.

    The utility functions are supplied to `init` (``init(ufun, opponent_ufun)``),
    not the constructor: the constructor takes configuration only.

    Args:
        max_pareto_size: Maximum number of candidates retained at any point
            during the iterative construction.  Default: 100000.
    """

    def __init__(self, max_pareto_size: int = 100000) -> None:
        self._ufun: BaseUtilityFunction | None = None
        self._opponent_ufun: BaseUtilityFunction | None = None
        self._max_pareto_size = max_pareto_size
        self._initialized = False
        self._pareto_front: list[Outcome] = []

    # ------------------------------------------------------------------
    # ParetoSampler protocol properties
    # ------------------------------------------------------------------

    @property
    def ufun(self) -> BaseUtilityFunction | None:
        return self._ufun

    @property
    def initialized(self) -> bool:
        return self._initialized

    def set_opponent_ufun(self, opponent_ufun: BaseUtilityFunction | None) -> None:
        """Update the opponent utility function and mark as not initialized."""
        self._opponent_ufun = opponent_ufun
        self._initialized = False

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init(
        self,
        ufun: BaseUtilityFunction | None = None,
        opponent_ufun: BaseUtilityFunction | None = None,
    ) -> None:
        """Build the exact Pareto frontier using MOBANOS.

        Args:
            ufun: The agent's own utility function (supplied here, not in the
                constructor). Omit to keep the current one.
            opponent_ufun: Estimate of the opponent's utility function. Omit to
                keep the current estimate.

        Raises:
            TypeError: if own or opponent ufun is not a
                ``LinearAdditiveUtilityFunction``.
        """
        from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

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
                "MOBANOSParetoSampler.init requires a ufun (pass ufun=...)."
            )

        own_base = self._unwrap(self._ufun)
        opp_base = self._unwrap(self._opponent_ufun)

        for label, base in [("own", own_base), ("opponent", opp_base)]:
            if not isinstance(base, LinearAdditiveUtilityFunction):
                raise TypeError(
                    f"MOBANOSParetoSampler requires {label} ufun to be a "
                    f"LinearAdditiveUtilityFunction but got {type(base).__name__}."
                )

        issues = list(own_base.issues)  # type: ignore[union-attr]
        if not issues:
            raise ValueError(
                "MOBANOSParetoSampler requires a ufun with an outcome space."
            )

        self._pareto_front = self._run_mobanos(
            own_base,  # type: ignore[arg-type]
            opp_base,  # type: ignore[arg-type]
            issues,  # type: ignore[arg-type]
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # ParetoSampler queries
    # ------------------------------------------------------------------

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
        result = [o for o in self._pareto_front if float(ufun(o)) >= raw_min - 1e-9]
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
        feasible = [o for o in self._pareto_front if float(ufun(o)) >= raw_min - 1e-9]
        if not feasible:
            return None
        return max(feasible, key=lambda o: float(opp(o)))

    # ------------------------------------------------------------------
    # MOBANOS algorithm internals
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(ufun: BaseUtilityFunction) -> BaseUtilityFunction:
        from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

        base = ufun
        while hasattr(base, "ufun") and not isinstance(
            base, LinearAdditiveUtilityFunction
        ):
            base = base.ufun  # type: ignore[attr-defined]
        return base

    def _to_raw_util(self, norm: float) -> float:
        ufun = self._ufun
        assert ufun is not None  # guaranteed once initialized
        mn, mx = ufun.minmax()
        return float(mn) + norm * float(mx - mn)

    @staticmethod
    def _remove_dominated(
        candidates: list[tuple[Any, float, float]],
    ) -> list[tuple[Any, float, float]]:
        """Remove dominated solutions.

        Each element is ``(partial_outcome, own_util, opp_util)``.
        A candidate is dominated if there exists another that is ≥ on both
        utilities and > on at least one.
        """
        if not candidates:
            return []
        n = len(candidates)
        dominated: set[int] = set()
        for a in range(n):
            if a in dominated:
                continue
            _, ua, va = candidates[a]
            for b in range(n):
                if b == a or b in dominated:
                    continue
                _, ub, vb = candidates[b]
                if ua >= ub and va >= vb and (ua > ub or va > vb):
                    dominated.add(b)
        return [c for i, c in enumerate(candidates) if i not in dominated]

    def _run_mobanos(
        self,
        own: LinearAdditiveUtilityFunction,
        opp: LinearAdditiveUtilityFunction,
        issues: list[Any],
    ) -> list[Outcome]:
        """Core MOBANOS algorithm — exact IPS without rounding."""

        def per_issue_pareto(i: int) -> list[tuple[tuple[Any, ...], float, float]]:
            vals = list(issues[i].all)
            own_vfun = own.values[i]
            opp_vfun = opp.values[i]
            own_w = own.weights[i]
            opp_w = opp.weights[i]
            candidates: list[tuple[tuple[Any, ...], float, float]] = [
                ((v,), own_w * float(own_vfun(v)), opp_w * float(opp_vfun(v)))
                for v in vals
            ]
            return self._remove_dominated(candidates)

        ps: list[tuple[tuple[Any, ...], float, float]] = per_issue_pareto(0)

        for i in range(1, len(issues)):
            ri = per_issue_pareto(i)
            if not ri:
                continue
            product: list[tuple[tuple[Any, ...], float, float]] = [
                (partial + singleton, u_own + u_i, u_opp + v_i)
                for partial, u_own, u_opp in ps
                for singleton, u_i, v_i in ri
            ]
            ps = self._remove_dominated(product)

            # Safety valve: if Pareto set is too large, keep the most promising ones
            if len(ps) > self._max_pareto_size:
                ps.sort(key=lambda x: x[1] + x[2], reverse=True)
                ps = ps[: self._max_pareto_size]

        return [partial for partial, _, _ in ps]
