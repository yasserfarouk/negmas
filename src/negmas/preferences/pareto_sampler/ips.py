"""IPS (Iterative Pareto Search) ParetoSampler implementation.

IPS exploits the additive structure of both the agent's own utility function and
the opponent's utility function to iteratively construct an approximate Pareto
frontier via a dynamic-programming-style combination of per-issue Pareto sets.

Reference
---------
Koça, T., de Jonge, D., & Baarslag, T. (2024).
*Search algorithms for automated negotiation in large domains.*
Algorithms 17(5), 200.  (Section 4.3 and Algorithm 3.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


from negmas.outcomes import Outcome


if TYPE_CHECKING:
    from negmas.preferences.base_ufun import BaseUtilityFunction
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

__all__ = ["IPSParetoSampler"]

_MIN_RANGE = 1e-12


class IPSParetoSampler:
    """Pareto sampler using the IPS (Iterative Pareto Search) algorithm.

    IPS builds an approximate Pareto frontier for a pair of additive utility
    functions by iteratively combining per-issue non-dominated solution sets,
    with utilities rounded to a *p*-decimal-place grid (controlled by
    ``precision``) to induce overlapping sub-problems.

    **Requirements**: both the agent's own ufun and the opponent ufun must be
    ``LinearAdditiveUtilityFunction`` (or ``LinearUtilityFunction``) instances
    with the same set of issues and the same issue ordering.  ``init()`` (or
    any query method) will raise ``TypeError`` if this is not satisfied.

    The utility functions are supplied to `init` (``init(ufun, opponent_ufun)``),
    not the constructor: the constructor takes configuration only, and the
    expensive frontier build happens in `init`.

    Args:
        precision: Grid precision *p*.  Utilities are rounded to *p* decimal
            places when computing dominance.  Default: 3.

    Remarks:
        - The algorithm has worst-case space complexity *O(|I|·|V|·10^p)* and
          time complexity *O(|I|·|V|·10^p)*.
        - ``init(ufun, opponent_ufun)`` must be called before any query method
          is used.  If *opponent_ufun* changes, call ``init`` again to rebuild
          the frontier (the same instance is reused).
        - IPS requires both ufuns to be additive (``LinearAdditiveUtilityFunction``
          or its subclass ``LinearUtilityFunction``).  For non-additive ufuns,
          use a sampling-based approach instead.

    *AI supported (config-only constructor; operands supplied to ``init``).*
    """

    def __init__(self, precision: int = 3) -> None:
        self._ufun: BaseUtilityFunction | None = None
        self._opponent_ufun: BaseUtilityFunction | None = None
        self._precision = precision
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
        """Update the opponent utility function.  Call ``init()`` afterwards to
        rebuild the Pareto frontier with the new estimate."""
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
        """Build the approximate Pareto frontier using IPS.

        Args:
            ufun: The agent's own utility function. If provided, replaces the
                one from a previous ``init`` (the expensive build happens here,
                not in the constructor). Omit to keep the current ufun.
            opponent_ufun: Estimate of the opponent's utility function. Omit to
                keep the current estimate (e.g. to rebuild after the own ufun
                changed).

        Raises:
            TypeError: if own or opponent ufun is not a
                ``LinearAdditiveUtilityFunction``/``LinearUtilityFunction``.
        """
        from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

        if ufun is not None:
            self._ufun = ufun
        if opponent_ufun is not None:
            self._opponent_ufun = opponent_ufun

        if self._opponent_ufun is None:
            # Not an error during construction; just mark as not initialised
            # so callers can detect this.
            self._pareto_front = []
            self._initialized = False
            return

        if self._ufun is None:
            raise ValueError("IPSParetoSampler.init requires a ufun (pass ufun=...).")

        own_base = self._unwrap(self._ufun)
        opp_base = self._unwrap(self._opponent_ufun)

        for name, base in [("own", own_base), ("opponent", opp_base)]:
            if not isinstance(base, LinearAdditiveUtilityFunction):
                raise TypeError(
                    f"IPSParetoSampler requires {name} ufun to be a "
                    f"LinearAdditiveUtilityFunction but got "
                    f"{type(base).__name__}."
                )

        issues = list(own_base.issues)  # type: ignore[union-attr]
        if not issues:
            raise ValueError("IPSParetoSampler requires a ufun with an outcome space.")

        self._pareto_front = self._run_ips(own_base, opp_base, issues)  # type: ignore[arg-type]
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
        """Return approximately Pareto-optimal outcomes, optionally filtered.

        Args:
            n: Maximum number of outcomes (``None`` = all).
            min_util: Minimum required own utility (raw or normalised).
            normalized: if ``True``, *min_util* is in normalised [0,1] space.
            opponent_ufun: If provided, reinitialises IPS with this opponent ufun
                before answering the query.
        """
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
        """Return the Pareto-optimal outcome maximising opponent utility subject
        to own utility ≥ *min_util*.

        Args:
            min_util: Minimum required own utility.
            normalized: if ``True``, *min_util* is in normalised [0,1] space.
            opponent_ufun: If provided, reinitialises IPS with this opponent ufun.
        """
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
    # IPS algorithm internals
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(ufun: BaseUtilityFunction) -> BaseUtilityFunction:
        """Unwrap discounted/scaled wrappers to get the underlying base ufun."""
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

    def _round(self, x: float) -> float:
        """Round to *precision* decimal places (discretisation function d_p)."""
        return round(x, self._precision)

    def _remove_approx_dominated(
        self, candidates: list[tuple[Any, float, float]]
    ) -> list[tuple[Any, float, float]]:
        """Remove approximately dominated solutions from *candidates*.

        Each element of *candidates* is ``(partial_outcome, own_util, opp_util)``
        where utilities have already been rounded.  A candidate is dominated if
        there exists another candidate that is at least as good on BOTH utilities
        and strictly better on at least one.

        Returns the non-dominated subset.
        """
        if not candidates:
            return []
        dominated: set[int] = set()
        n = len(candidates)
        for a in range(n):
            if a in dominated:
                continue
            _, ua, va = candidates[a]
            for b in range(n):
                if b == a or b in dominated:
                    continue
                _, ub, vb = candidates[b]
                # a dominates b if ua >= ub AND va >= vb with at least one strict
                if ua >= ub and va >= vb and (ua > ub or va > vb):
                    dominated.add(b)
        return [c for i, c in enumerate(candidates) if i not in dominated]

    def _run_ips(
        self,
        own: LinearAdditiveUtilityFunction,
        opp: LinearAdditiveUtilityFunction,
        issues: list[Any],
    ) -> list[Outcome]:
        """Core IPS algorithm (Algorithm 3 in Koça et al. 2024).

        Returns a list of outcomes on the approximate Pareto frontier.
        """
        p = self._precision

        def d(x: float) -> float:
            return round(max(0.0, x), p)

        # ------------------------------------------------------------------
        # Build per-issue Pareto sets R_i (for i >= 1)
        # and the initial PS from issue 0
        # ------------------------------------------------------------------
        # Each element in a per-issue set is:
        #   (partial_outcome_tuple, own_partial_util, opp_partial_util)
        # where partial utilities are ROUNDED via d().
        # ------------------------------------------------------------------

        def per_issue_candidates(i: int) -> list[tuple[tuple[Any, ...], float, float]]:
            vals = list(issues[i].all)
            own_vfun = own.values[i]
            opp_vfun = opp.values[i]
            own_w = own.weights[i]
            opp_w = opp.weights[i]
            return [
                ((v,), d(own_w * float(own_vfun(v))), d(opp_w * float(opp_vfun(v))))
                for v in vals
            ]

        # issue 0: initial PS
        ps: list[tuple[tuple[Any, ...], float, float]] = self._remove_approx_dominated(
            per_issue_candidates(0)
        )

        # issues 1 .. n-1: iterative Cartesian product + dominance pruning
        for i in range(1, len(issues)):
            ri = self._remove_approx_dominated(per_issue_candidates(i))
            if not ri:
                continue

            # Cartesian product of ps x ri
            product: list[tuple[tuple[Any, ...], float, float]] = []
            for partial, u_own, u_opp in ps:
                for singleton, u_own_i, u_opp_i in ri:
                    combined = partial + singleton
                    product.append((combined, d(u_own + u_own_i), d(u_opp + u_opp_i)))

            ps = self._remove_approx_dominated(product)

        # Convert to full outcomes (add missing bias / offset contributions)
        # The per-issue partial utilities exclude the bias; this is fine for
        # dominance computation.  Return the actual outcome tuples.
        return [partial for partial, _, _ in ps]
