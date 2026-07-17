"""NB3 (Negotiating by Branch-and-Bound) ParetoSampler implementation.

NB3 uses branch-and-bound with per-agent utility upper bounds to efficiently
prune the search space and find the Pareto-optimal outcomes for a pair of
additive utility functions.

Reference
---------
De Jonge, D., & Sierra, C. (2017).
*NB3: A Multilateral Negotiation Algorithm for Large, Non-Linear Agreement
Spaces with Limited Time.*
In AAMAS 2017.

Description from:
Koça, T., de Jonge, D., & Baarslag, T. (2024).
*Search algorithms for automated negotiation in large domains.*
Algorithms 17(5), 200.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any

from negmas.outcomes import Outcome


if TYPE_CHECKING:
    from negmas.preferences.base_ufun import BaseUtilityFunction
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

__all__ = ["NB3ParetoSampler"]


class NB3ParetoSampler:
    """Pareto sampler using the NB3 Branch-and-Bound algorithm.

    NB3 exploits the additive structure of both utility functions to prune the
    search space via per-agent upper-bound estimates, finding the Pareto frontier
    in an anytime fashion (stops after ``max_nodes`` node expansions).

    **Requirements**: both the agent's own ufun and the opponent ufun must be
    ``LinearAdditiveUtilityFunction`` instances.  ``init()`` raises ``TypeError``
    if this is not satisfied.

    Args:
        ufun: The agent's own utility function.
        opponent_ufun: Estimate of the opponent's utility function.
        max_nodes: Maximum number of nodes to expand.  Default: 10000.
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        opponent_ufun: BaseUtilityFunction | None = None,
        max_nodes: int = 10000,
    ) -> None:
        self._ufun = ufun
        self._opponent_ufun = opponent_ufun
        self._max_nodes = max_nodes
        self._initialized = False
        # list of (own_util, opp_util, outcome)
        self._pareto_front: list[tuple[float, float, Outcome]] = []

    # ------------------------------------------------------------------
    # ParetoSampler protocol properties
    # ------------------------------------------------------------------

    @property
    def ufun(self) -> BaseUtilityFunction:
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

    def init(self, opponent_ufun: BaseUtilityFunction | None = None) -> None:
        """Run the NB3 branch-and-bound algorithm to build the Pareto frontier.

        Raises:
            TypeError: if own or opponent ufun is not a
                ``LinearAdditiveUtilityFunction``.
        """
        from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

        if opponent_ufun is not None:
            self._opponent_ufun = opponent_ufun

        if self._opponent_ufun is None:
            self._pareto_front = []
            self._initialized = False
            return

        own_base = self._unwrap(self._ufun)
        opp_base = self._unwrap(self._opponent_ufun)

        for label, base in [("own", own_base), ("opponent", opp_base)]:
            if not isinstance(base, LinearAdditiveUtilityFunction):
                raise TypeError(
                    f"NB3ParetoSampler requires {label} ufun to be a "
                    f"LinearAdditiveUtilityFunction but got {type(base).__name__}."
                )

        issues = list(own_base.issues)  # type: ignore[union-attr]
        if not issues:
            raise ValueError("NB3ParetoSampler requires a ufun with an outcome space.")

        self._pareto_front = self._run_nb3(
            own_base,
            opp_base,
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
            self.init(opponent_ufun)
        elif not self._initialized:
            self.init()
        if not self._initialized:
            return []

        raw_min = self._to_raw_util(min_util) if normalized else min_util
        result = [o for u_own, _, o in self._pareto_front if u_own >= raw_min - 1e-9]
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

        raw_min = self._to_raw_util(min_util) if normalized else min_util
        feasible = [
            (u_opp, o)
            for u_own, u_opp, o in self._pareto_front
            if u_own >= raw_min - 1e-9
        ]
        if not feasible:
            return None
        return max(feasible, key=lambda x: x[0])[1]

    # ------------------------------------------------------------------
    # NB3 algorithm internals
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
        mn, mx = self._ufun.minmax()
        return float(mn) + norm * float(mx - mn)

    def _run_nb3(
        self,
        own: LinearAdditiveUtilityFunction,
        opp: LinearAdditiveUtilityFunction,
        issues: list[Any],
    ) -> list[tuple[float, float, Outcome]]:
        """Core NB3 branch-and-bound algorithm."""
        n_issues = len(issues)

        # Precompute per-issue max and min contributions for own and opp
        # max_contrib[i] = max weighted contribution of issue i
        # min_contrib[i] = min weighted contribution of issue i
        own_max = []
        own_min = []
        opp_max = []
        opp_min = []

        val_lists: list[list[Any]] = []
        own_contribs: list[list[float]] = []
        opp_contribs: list[list[float]] = []

        for i, (issue, own_vfun, own_w, opp_vfun, opp_w) in enumerate(
            zip(issues, own.values, own.weights, opp.values, opp.weights)
        ):
            vals = list(issue.all)
            val_lists.append(vals)
            oc = [own_w * float(own_vfun(v)) for v in vals]
            pc = [opp_w * float(opp_vfun(v)) for v in vals]
            own_contribs.append(oc)
            opp_contribs.append(pc)
            own_max.append(max(oc))
            own_min.append(min(oc))
            opp_max.append(max(pc))
            opp_min.append(min(pc))

        # Suffix sums for upper bounds: suffix_own_max[i] = sum_{k>=i} own_max[k]
        suffix_own_max = [0.0] * (n_issues + 1)
        suffix_opp_max = [0.0] * (n_issues + 1)
        for i in range(n_issues - 1, -1, -1):
            suffix_own_max[i] = suffix_own_max[i + 1] + own_max[i]
            suffix_opp_max[i] = suffix_opp_max[i + 1] + opp_max[i]

        # Current non-dominated front: list of (own_util, opp_util, outcome)
        pareto_front: list[tuple[float, float, Outcome]] = []

        def is_dominated(u_own: float, u_opp: float) -> bool:
            """True if (u_own, u_opp) is dominated by some solution in the front."""
            for s_own, s_opp, _ in pareto_front:
                if s_own >= u_own and s_opp >= u_opp:
                    return True
            return False

        def update_front(u_own: float, u_opp: float, outcome: Outcome) -> None:
            """Add (u_own, u_opp, outcome) to front if non-dominated; prune dominated."""
            nonlocal pareto_front
            # Check if this new solution is dominated
            if is_dominated(u_own, u_opp):
                return
            # Remove solutions dominated by the new one
            pareto_front = [
                (so, sp, o)
                for so, sp, o in pareto_front
                if not (u_own >= so and u_opp >= sp and (u_own > so or u_opp > sp))
            ]
            pareto_front.append((u_own, u_opp, outcome))

        # Node: (partial_outcome_as_tuple, depth, own_partial, opp_partial)
        # Heap: max-heap on h = ub_own + ub_opp, so negate for Python's min-heap
        def h(depth: int, own_partial: float, opp_partial: float) -> float:
            return (
                own_partial
                + suffix_own_max[depth]
                + opp_partial
                + suffix_opp_max[depth]
            )

        root = ((), 0, 0.0, 0.0)
        root_h = h(0, 0.0, 0.0)
        heap: list[tuple[float, Any]] = [(-root_h, root)]
        n_expanded = 0

        while heap and n_expanded < self._max_nodes:
            _, node = heapq.heappop(heap)
            partial, depth, own_partial, opp_partial = node

            ub_own = own_partial + suffix_own_max[depth]
            ub_opp = opp_partial + suffix_opp_max[depth]
            # Prune: dominated if any solution (s_own, s_opp) satisfies
            # s_own >= ub_own AND s_opp >= ub_opp
            if is_dominated(ub_own, ub_opp):
                continue

            n_expanded += 1

            if depth == n_issues:
                # Terminal node: update Pareto front
                update_front(own_partial, opp_partial, partial)
                continue

            # Expand children for each value of issue[depth]
            for j, v in enumerate(val_lists[depth]):
                new_own = own_partial + own_contribs[depth][j]
                new_opp = opp_partial + opp_contribs[depth][j]
                new_partial = partial + (v,)
                child_ub_own = new_own + suffix_own_max[depth + 1]
                child_ub_opp = new_opp + suffix_opp_max[depth + 1]
                # Prune child if dominated
                if not is_dominated(child_ub_own, child_ub_opp):
                    child_h = child_ub_own + child_ub_opp
                    child_node = (new_partial, depth + 1, new_own, new_opp)
                    heapq.heappush(heap, (-child_h, child_node))

        return pareto_front
