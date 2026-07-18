"""Adaptive Pareto sampler that picks a concrete sampler based on domain size.

The concrete choice can only be made once the operands are known — the own and
opponent utility functions (and therefore the outcome space) are supplied to
`init`, not the constructor — so the selection happens there.

*AI supported (adaptive dispatch over the exact/large ParetoSampler backends).*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome

from .bruteforce import BruteForceParetoSampler
from .mobanos import MOBANOSParetoSampler

if TYPE_CHECKING:
    from negmas.preferences.base_ufun import BaseUtilityFunction
    from negmas.preferences.protocols import ParetoSampler

__all__ = ["AdaptiveParetoSampler"]


class AdaptiveParetoSampler:
    """Selects the best ``ParetoSampler`` backend based on the outcome-space size.

    * For **small** outcome spaces (``cardinality <= max_bruteforce_outcomes``)
      it uses `BruteForceParetoSampler` — exact, works with *any* utility
      function, and fast enough when the space fits comfortably in memory.
    * For **larger** spaces it uses the more scalable ``large_sampler`` (the
      *most accurate* non-brute-force backend, `MOBANOSParetoSampler` by default
      — exact, no rounding). If that backend cannot handle the ufuns (e.g. it
      requires additive utilities and they are not additive) it falls back to
      `BruteForceParetoSampler` with the ``max_cardinality`` sampling cap.

    Because a sampler's operands are supplied to :meth:`init` (not the
    constructor), the decision is made there — the constructor only stores the
    thresholds/config. This is the default sampler used by
    ``BaseUtilityFunction.make_pareto_sampler`` (aliased ``DefaultParetoSampler``).

    Args:
        max_bruteforce_outcomes: Largest outcome-space cardinality for which the
            exact `BruteForceParetoSampler` is used. Above it, ``large_sampler``
            is tried first. Default ``10_000``.
        large_sampler: The ``ParetoSampler`` type used for large spaces. Defaults
            to `MOBANOSParetoSampler`. Pass `IPSParetoSampler` for very large
            additive domains where an exact frontier is too costly.
        max_cardinality: Enumeration/sampling cap for the brute-force backend.
    """

    def __init__(
        self,
        max_bruteforce_outcomes: int = 10_000,
        large_sampler: type[ParetoSampler] | None = None,
        max_cardinality: int = 1_000_000,
    ) -> None:
        self._ufun: BaseUtilityFunction | None = None
        self._opponent_ufun: BaseUtilityFunction | None = None
        self._max_bruteforce_outcomes = max_bruteforce_outcomes
        self._large_sampler_type = large_sampler
        self._max_cardinality = max_cardinality
        self._impl: ParetoSampler | None = None
        self._initialized = False

    @property
    def ufun(self) -> BaseUtilityFunction | None:
        return self._ufun

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def impl(self) -> ParetoSampler | None:
        """The concrete sampler chosen at :meth:`init` (``None`` before init)."""
        return self._impl

    def _cardinality(self) -> float:
        try:
            os_ = self._ufun.outcome_space if self._ufun is not None else None
        except Exception:  # pragma: no cover - defensive
            return float("inf")
        if os_ is None:
            return float("inf")
        try:
            return float(os_.cardinality)
        except Exception:  # pragma: no cover - defensive
            return float("inf")

    def init(
        self,
        ufun: BaseUtilityFunction | None = None,
        opponent_ufun: BaseUtilityFunction | None = None,
    ) -> None:
        if ufun is not None:
            self._ufun = ufun
        if opponent_ufun is not None:
            self._opponent_ufun = opponent_ufun
        if self._ufun is None or self._opponent_ufun is None:
            self._impl = None
            self._initialized = False
            return

        card = self._cardinality()
        impl: ParetoSampler | None = None
        if card <= self._max_bruteforce_outcomes:
            impl = BruteForceParetoSampler(max_cardinality=self._max_cardinality)
        else:
            large_type = self._large_sampler_type or MOBANOSParetoSampler
            candidate = large_type()  # config-only constructor
            try:
                candidate.init(self._ufun, self._opponent_ufun)
            except Exception:
                # e.g. an additive-only backend on a non-additive ufun: fall back
                # to the (sampling-capped) exact brute-force sampler.
                candidate = None  # type: ignore[assignment]
            if candidate is not None and candidate.initialized:
                impl = candidate
            else:
                impl = BruteForceParetoSampler(max_cardinality=self._max_cardinality)

        if not impl.initialized:
            impl.init(self._ufun, self._opponent_ufun)
        self._impl = impl
        self._initialized = impl.initialized

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
        if self._impl is None:
            return []
        return self._impl.pareto_outcomes(n, min_util=min_util, normalized=normalized)

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
        if self._impl is None:
            return None
        return self._impl.best_for_opponent(min_util=min_util, normalized=normalized)
