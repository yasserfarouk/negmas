"""Adaptive utility function inverter that automatically selects the best available
inverter for the given ufun.

``AdaptiveInverseUtilityFunction`` inspects the wrapped ufun's type and outcome-space
size at ``init()`` time and delegates all queries to the most appropriate concrete
inverter:

* Small/medium discrete outcome spaces (cardinality ≤ ``max_presorting_outcomes``) →
  `PresortingInverseUtilityFunction` (exact enumeration + sorting, always preferred
  when the space is small enough to enumerate).
* Large or continuous ``LinearAdditiveUtilityFunction`` /``LinearUtilityFunction``
  outcome spaces (cardinality > ``max_presorting_outcomes``) →
  `BIDSInverseUtilityFunction` (approximate, but scalable to very large spaces
  via dynamic programming).
* Any other ufun type → `PresortingInverseUtilityFunction` regardless of size
  (BIDS only works with additive ufuns).

This class is the recommended entry point when you do not want to couple your code
to a specific inverter implementation.
"""

from __future__ import annotations

from typing import Any

from negmas.outcomes import Outcome

from ..protocols import InverseUFun
from .bids import BIDSInverseUtilityFunction
from .presorting import PresortingInverseUtilityFunction

__all__ = ["AdaptiveInverseUtilityFunction"]

# Default cardinality threshold above which BIDS is preferred over Presorting
# for linear-additive ufuns.  1,000,000 outcomes can be sorted in ~1-2 seconds;
# above that, the O(n log n) init cost of Presorting starts to dominate.
_DEFAULT_MAX_PRESORTING = 1_000_000


class AdaptiveInverseUtilityFunction(InverseUFun):
    """Automatically selects the best ``InverseUFun`` implementation.

    At ``init()`` time the outcome-space cardinality is checked and the best
    concrete inverter is chosen:

    * **Small/medium spaces** (cardinality ≤ ``max_presorting_outcomes``) →
      `PresortingInverseUtilityFunction`: exact enumeration + sort.  Preferred
      whenever feasible because BIDS is inherently approximate.
    * **Large additive spaces** (``LinearAdditiveUtilityFunction`` or
      ``LinearUtilityFunction`` with cardinality > ``max_presorting_outcomes``) →
      `BIDSInverseUtilityFunction`: approximate dynamic programming, scalable to
      hundreds of issues / 10^250 outcomes.
    * **Large non-additive spaces** → `PresortingInverseUtilityFunction` with
      sampling (``max_cache_size`` limits the cached outcomes).

    All ``InverseUFun`` method calls are forwarded to the selected delegate.

    Args:
        ufun: The utility function to invert.
        max_presorting_outcomes: Outcome-space cardinality threshold below which
            `PresortingInverseUtilityFunction` is always used (even for additive
            ufuns), because it is exact.  Default: 1,000,000.
        bids_precision: ``precision`` argument forwarded to
            `BIDSInverseUtilityFunction` when BIDS is selected (default: 3).
        bids_n_samples: ``n_samples`` argument forwarded to
            `BIDSInverseUtilityFunction` (default: 50).
        **presorting_kwargs: Keyword arguments forwarded to
            `PresortingInverseUtilityFunction` when presorting is selected
            (e.g. ``rational_only``, ``levels``, ``max_cache_size``).
    """

    def __init__(
        self,
        ufun: Any,
        max_presorting_outcomes: int = _DEFAULT_MAX_PRESORTING,
        bids_precision: int = 3,
        bids_n_samples: int = 50,
        **presorting_kwargs: Any,
    ) -> None:
        self._ufun = ufun
        self._max_presorting = max_presorting_outcomes
        self._bids_precision = bids_precision
        self._bids_n_samples = bids_n_samples
        self._presorting_kwargs = presorting_kwargs
        self._delegate: InverseUFun | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def ufun(self) -> Any:
        return self._ufun

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def delegate(self) -> InverseUFun | None:
        """The concrete inverter currently in use (set after ``init()``)."""
        return self._delegate

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Select and initialise the best concrete inverter for *ufun*."""
        from ..crisp.linear import LinearAdditiveUtilityFunction

        # Unwrap wrappers to inspect the base ufun type
        base = self._ufun
        while hasattr(base, "ufun") and not isinstance(
            base, LinearAdditiveUtilityFunction
        ):
            base = base.ufun  # type: ignore[attr-defined]

        is_additive = isinstance(base, LinearAdditiveUtilityFunction) and bool(
            getattr(base, "issues", None)
        )

        # Check outcome-space cardinality to decide whether Presorting is feasible
        use_bids = False
        if is_additive:
            os = self._ufun.outcome_space
            cardinality = os.cardinality if os is not None else float("inf")
            use_bids = cardinality > self._max_presorting

        if use_bids:
            self._delegate = BIDSInverseUtilityFunction(
                self._ufun,
                precision=self._bids_precision,
                n_samples=self._bids_n_samples,
            )
        else:
            self._delegate = PresortingInverseUtilityFunction(
                self._ufun, **self._presorting_kwargs
            )

        self._delegate.init()
        self._initialized = True

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def _d(self) -> InverseUFun:
        if self._delegate is None:
            raise RuntimeError(
                "AdaptiveInverseUtilityFunction.init() has not been called."
            )
        return self._delegate

    # ------------------------------------------------------------------
    # InverseUFun protocol
    # ------------------------------------------------------------------

    def some(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        n: int | None = None,
    ) -> list[Outcome]:
        return self._d().some(rng, normalized, n)

    def one_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        return self._d().one_in(rng, normalized, fallback_to_higher, fallback_to_best)

    def best_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        return self._d().best_in(rng, normalized)

    def worst_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        return self._d().worst_in(rng, normalized)

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        return self._d().within_fractions(rng)

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        return self._d().within_indices(rng)

    def min(self) -> float:
        return self._d().min()

    def max(self) -> float:
        return self._d().max()

    def worst(self) -> Outcome:
        return self._d().worst()

    def best(self) -> Outcome:
        return self._d().best()

    def minmax(self) -> tuple[float, float]:
        return self._d().minmax()

    def extreme_outcomes(self) -> tuple[Outcome, Outcome]:
        return self._d().extreme_outcomes()

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """Calling an inverse ufun directly is equivalent to calling ``one_in()``."""
        return self._d().one_in(rng, normalized)
