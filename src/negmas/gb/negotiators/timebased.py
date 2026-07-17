"""Negotiators base classes."""

from __future__ import annotations

from typing import Callable, Literal, Sequence, TypeVar

from negmas.gb.components.selectors import (
    AdditivePartnerOffersOrientedSelector,
    BestOfferOrientedSelector,
    FirstOfferOrientedSelector,
    KeepFirst,
    KeepLast,
    LastOfferOrientedSelector,
    MultiplicativePartnerOffersOrientedSelector,
    NoFiltering,
    OfferFilterProtocol,
    OfferOrientedSelector,
    OfferSelector,
)
from negmas.negotiators.helpers import Aspiration, PolyAspiration, TimeCurve
from negmas.preferences import DefaultInverseUtilityFunction, InverseUFun

from ...outcomes import DistanceFun, Outcome, generalized_minkowski_distance
from .utilbased import UtilBasedNegotiator

__all__ = [
    "TimeBasedNegotiator",
    "TimeBasedConcedingNegotiator",
    "BoulwareTBNegotiator",
    "LinearTBNegotiator",
    "ConcederTBNegotiator",
    "AspirationNegotiator",
    "FirstOfferOrientedTBNegotiator",
    "LastOfferOrientedTBNegotiator",
    "BestOfferOrientedTBNegotiator",
    "AdditiveParetoFollowingTBNegotiator",
    "MultiplicativeParetoFollowingTBNegotiator",
    "MultiplicativeLastOfferFollowingTBNegotiator",
    "AdditiveLastOfferFollowingTBNegotiator",
    "MultiplicativeFirstFollowingTBNegotiator",
    "AdditiveFirstFollowingTBNegotiator",
]

TC = TypeVar("TC", bound=TimeCurve)


def make_curve(
    curve: TC | Literal["boulware"] | Literal["conceder"] | Literal["linear"] | float,
    starting_utility: float = 1.0,
) -> TC | PolyAspiration:
    """
    Generates a `TimeCurve` or `Aspiration` with optional `starting_utility`self.

    Default behavior is to return a `PolyAspiration` object.
    """
    if isinstance(curve, TimeCurve):
        return curve
    return PolyAspiration(starting_utility, curve)


def make_offer_selector(
    inverse_ufun: InverseUFun,
    selector: Callable[[Sequence[Outcome]], Outcome | None]
    | Literal["best"]
    | Literal["worst"]
    | None = None,
) -> Callable[[tuple[float, float], bool], Outcome | None]:
    """
    Generates a callable that can be used to select a specific outcome in a range of utility values.

    Args:
        inverse_ufun: The inverse utility function used
        selector: Any callable that selects an outcome, or "best"/"worst" for the one with highest/lowest utility. `None` for random.
    """
    if selector is None:
        return inverse_ufun.one_in
    if isinstance(selector, Callable):
        return lambda x: selector(inverse_ufun.some(x, normalized=False))  # type: ignore
    if selector == "best":
        return inverse_ufun.best_in
    if selector == "worst":
        return inverse_ufun.worst_in
    raise ValueError(f"Unknown selector type: {selector}")


class TimeBasedNegotiator(UtilBasedNegotiator):
    """
    A time-based negotiation strategy that concedes independently of the offers
    received during the negotiation.

    The negotiator maintains two concession curves: an *offering* curve that
    controls the utility range of its own proposals, and an *accepting* curve
    that controls the utility range of offers it will accept. Both are
    `TimeCurve` objects mapping relative time ``t ∈ [0, 1]`` to a utility
    range. At each step, the negotiator asks the inverter for an outcome within
    the offering curve's range (for ``propose``) or checks whether the
    opponent's offer falls within the accepting curve's range (for
    ``respond``).

    Args:
        offering_curve (TimeCurve | str | float): A `TimeCurve` (or a string
            ``"boulware"`` / ``"linear"`` / ``"conceder"`` or a float exponent)
            used to sample outcomes when offering. Defaults to ``"boulware"``.
        accepting_curve (TimeCurve | str | float | None): A `TimeCurve` (or
            string/float as above) used to decide the utility range to accept.
            If ``None``, the offering curve is reused (same range for offering
            and accepting).
        offer_selector (OfferSelector | None): See `UtilBasedNegotiator`.
        **kwargs: Forwarded to `UtilBasedNegotiator` (e.g. ``stochastic``,
            ``ufun_inverter``, ``eps``, ``rank_only``, ``max_cardinality``).

    Remarks:
        - This negotiator is *time-only*: it never reacts to the opponent's
          offers (except accepting/rejecting them). For opponent-aware
          behavior, use `NaiveTitForTatNegotiator` or the
          ``OfferOrientedTBNegotiator`` family.
    """

    def __init__(
        self,
        *args,
        offering_curve: TimeCurve
        | Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | float = "boulware",
        accepting_curve: TimeCurve
        | Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | float
        | None = None,
        offer_selector: OfferSelector | None = None,
        **kwargs,
    ):
        """Initializes the negotiator with offering and accepting curves."""
        super().__init__(*args, offer_selector=offer_selector, **kwargs)
        offering_curve = make_curve(offering_curve, 1.0)
        if accepting_curve is None:
            accepting_curve = offering_curve
        else:
            accepting_curve = make_curve(accepting_curve, 1.0)
        self._offering_curve = offering_curve
        self._accepting_curve = accepting_curve

    def utility_range_to_propose(self, state) -> tuple[float, float]:
        """Returns the acceptable utility range for making proposals at the current negotiation state."""
        return self._offering_curve.utility_range(state.relative_time)

    def utility_range_to_accept(self, state) -> tuple[float, float]:
        """Returns the acceptable utility range for accepting offers at the current negotiation state."""
        return self._accepting_curve.utility_range(state.relative_time)


class TimeBasedConcedingNegotiator(TimeBasedNegotiator):
    """
    A time-based conceding negotiator using an `Aspiration` curve.

    This is the main entry point for aspiration-based time-only negotiators.
    It accepts an `Aspiration` curve (or a string/float shorthand) for both
    offering and accepting, plus a ``starting_utility`` that controls the
    first offer's utility level.

    Args:
        offering_curve (Aspiration | str | float): An `Aspiration` curve (or
            ``"boulware"`` / ``"linear"`` / ``"conceder"`` or a float exponent)
            controlling how fast the negotiator concedes when offering.
            Defaults to ``"boulware"`` (slow concession).
        accepting_curve (Aspiration | str | float | None): An `Aspiration`
            curve (or string/float) controlling the acceptance threshold. If
            ``None`` or falsy, the offering curve is reused.
        starting_utility (float): The relative utility (in ``[0, 1]``) at which
            the first offer is made. Only used when ``offering_curve`` is a
            string/float (not a pre-built `Aspiration` object). Defaults to
            ``1.0`` (start at the best outcome).
        **kwargs: Forwarded to `TimeBasedNegotiator` (e.g. ``stochastic``,
            ``ufun_inverter``, ``eps``, ``offer_selector``).

    Remarks:
        - ``BoulwareTBNegotiator``, ``LinearTBNegotiator``, and
          ``ConcederTBNegotiator`` are convenience subclasses that fix
          ``offering_curve`` to ``"boulware"``, ``"linear"``, and
          ``"conceder"`` respectively (with ``stochastic=False``).
        - `AspirationNegotiator` is a simplified interface to this class with
          ``presort`` and ``tolerance`` parameters.
    """

    def __init__(
        self,
        *args,
        offering_curve: Aspiration
        | Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | float = "boulware",
        accepting_curve: Aspiration
        | Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | float
        | None = None,
        starting_utility: float = 1.0,
        **kwargs,
    ):
        """Initializes the negotiator with configurable concession curves."""
        offering_curve = make_curve(offering_curve, starting_utility)
        if not accepting_curve:
            accepting_curve = offering_curve
        super().__init__(
            *args,
            offering_curve=offering_curve,
            accepting_curve=accepting_curve,
            **kwargs,
        )


class BoulwareTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A time-based negotiator that concedes sub-linearly (boulware).

    Uses a `PolyAspiration` curve with exponent 4 (``"boulware"``) and
    ``stochastic=False`` (proposes the worst outcome within the aspiration band).
    """

    def __init__(self, *args, **kwargs):
        """Initializes the instance."""
        kwargs["offering_curve"] = PolyAspiration(1.0, "boulware")
        kwargs["stochastic"] = False
        super().__init__(*args, **kwargs)


class LinearTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A time-based negotiator that concedes linearly.

    Uses a `PolyAspiration` curve with exponent 1 (``"linear"``) and
    ``stochastic=False``.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the instance."""
        kwargs["offering_curve"] = PolyAspiration(1.0, "linear")
        kwargs["stochastic"] = False
        super().__init__(*args, **kwargs)


class ConcederTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A time-based negotiator that concedes super-linearly (conceder).

    Uses a `PolyAspiration` curve with exponent 0.25 (``"conceder"``) and
    ``stochastic=False``.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the instance."""
        kwargs["offering_curve"] = PolyAspiration(1.0, "conceder")
        kwargs["stochastic"] = False
        super().__init__(*args, **kwargs)


class AspirationNegotiator(TimeBasedConcedingNegotiator):
    """
    A time-based conceding negotiator with a simplified interface.

    This is the most commonly used negotiator in the library. It concedes over
    time according to a polynomial aspiration curve (boulware / linear /
    conceder, or a custom exponent) and uses an `InverseUFun` to find outcomes
    within the current aspiration band.

    Args:
        max_aspiration (float): The aspiration level (relative utility in
            ``[0, 1]``) to use for the first offer (and first acceptance
            decision). Defaults to ``1.0`` (start at the best outcome).
        aspiration_type (str | float): The polynomial aspiration curve type.
            Pass a string (``"boulware"`` for slow concession, ``"linear"``
            for constant-rate concession, ``"conceder"`` for fast concession)
            or a real-valued exponent. Defaults to ``"boulware"``.
        stochastic (bool): If ``False`` (default), the negotiator proposes the
            outcome with the *lowest* utility still within its aspiration band
            (i.e. just above the aspiration level) via ``worst_in``. If
            ``True``, it proposes a random in-range outcome via ``one_in``.
        presort (bool): If ``True`` (default), a `DefaultInverseUtilityFunction`
            (i.e. `AdaptiveInverseUtilityFunction`) is used, which presorts
            outcomes for exact ``O(log n)`` lookups on small/medium spaces and
            falls back to BIDS for large additive spaces. If ``False``, no
            inverter is used and the negotiator cannot propose (it will always
            fall back to the best outcome).
        tolerance (float): A tolerance used for sampling outcomes near the
            aspiration level (passed as ``eps`` to the inverter). Defaults to
            ``0.001``.
        ufun_inverter (type[InverseUFun] | None): An optional `InverseUFun`
            **type** to use for inverting the utility function. If given, it
            overrides the ``presort`` default. See `negmas.preferences.inv_ufun`
            for the full list of available inverters and their trade-offs.
        **kwargs: Forwarded to `TimeBasedConcedingNegotiator` (e.g.
            ``name``, ``ufun``, ``parent``, ``owner``).

    Remarks:
        - This class provides a simpler interface to
          `TimeBasedConcedingNegotiator` with less control over the accepting
          curve and offer selector. For more control, use
          `TimeBasedConcedingNegotiator` or `TimeBasedNegotiator` directly.
        - ``propose`` never returns ``None`` mid-negotiation: if the inverter
          finds no outcome in the aspiration range (e.g. for strict inverters
          like `BruteForceInverseUtilityFunction`, or when the aspiration band
          is empty), it falls back to the best outcome rather than breaking the
          SAO mechanism.
    """

    def __init__(
        self,
        *args,
        max_aspiration=1.0,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | float = "boulware",
        stochastic=False,
        presort: bool = True,
        tolerance: float = 0.001,
        ufun_inverter: type[InverseUFun] | None = None,
        **kwargs,
    ):
        """Initializes the aspiration-based negotiator with concession parameters.

        Args:
            ufun_inverter: An optional `InverseUFun` **type** to use for inverting the
                utility function. If given, it overrides the ``presort`` default (and
                forces presorting-style offering). If `None`, a
                `DefaultInverseUtilityFunction` is used when ``presort`` is `True` and no
                inverter is used otherwise.
        """
        if ufun_inverter is None:
            ufun_inverter = None if not presort else DefaultInverseUtilityFunction
        super().__init__(
            *args,
            offering_curve=aspiration_type,
            accepting_curve=None,
            stochastic=stochastic,
            starting_utility=max_aspiration,
            ufun_inverter=ufun_inverter,
            eps=tolerance,
            **kwargs,
        )

    @property
    def tolerance(self):
        """Returns the tolerance used when sampling outcomes near the aspiration level."""
        return self._inverter.tolerance

    def utility_at(self, t):
        """Returns the aspiration utility level at relative time t (0.0 to 1.0)."""
        if not self._offering_curve:
            raise ValueError("No inverse ufun is known yet")
        return self._offering_curve.utility_at(t)  # type: ignore (I know it is an Aspiration not a TimeCurve)

    @property
    def ufun_max(self):
        """Returns the maximum utility value from the inverter."""
        return self._inverter.ufun_max

    @property
    def ufun_min(self):
        """Returns the minimum utility value from the inverter."""
        return self._inverter.ufun_min


class OfferOrientedNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that orients its offers toward some pivot outcome (See `OfferOrientedSelector` )
    """

    def __init__(self, *args, offer_selector: OfferOrientedSelector, **kwargs):
        """Initializes the instance."""
        super().__init__(*args, offer_selector=offer_selector, **kwargs)


class FirstOfferOrientedTBNegotiator(OfferOrientedNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on their utility value and how near they are to the partner's first offer
    """

    def __init__(
        self,
        *args,
        distance_fun: DistanceFun = generalized_minkowski_distance,
        **kwargs,
    ):
        """Initializes the negotiator with first-offer orientation."""
        kwargs["offer_selector"] = FirstOfferOrientedSelector(distance_fun)
        super().__init__(*args, **kwargs)


class BestOfferOrientedTBNegotiator(FirstOfferOrientedTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on their utility value and how near they are to the partner's past offer with the highest utility for me
    """

    def __init__(
        self,
        *args,
        distance_fun: DistanceFun = generalized_minkowski_distance,
        **kwargs,
    ):
        """Initializes the negotiator with best-offer orientation."""
        kwargs["offer_selector"] = BestOfferOrientedSelector(distance_fun)
        super().__init__(*args, **kwargs)


class LastOfferOrientedTBNegotiator(FirstOfferOrientedTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on their utility value and how near they are to the partner's last offer
    """

    def __init__(
        self,
        *args,
        distance_fun: DistanceFun = generalized_minkowski_distance,
        **kwargs,
    ):
        """Initializes the negotiator with last-offer orientation."""
        kwargs["offer_selector"] = LastOfferOrientedSelector(distance_fun)
        super().__init__(*args, **kwargs)


class MultiplicativeParetoFollowingTBNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the
    current utility level based on a weighted sum of their normalized utilities and
    distances to previous offers
    """

    def __init__(
        self,
        *args,
        dist_power: float = 2,
        issue_weights: list[float] | None = None,
        offer_filter: OfferFilterProtocol = NoFiltering,
        **kwargs,
    ):
        """Initializes the negotiator with multiplicative Pareto-following selection."""
        super().__init__(
            *args,
            offer_selector=MultiplicativePartnerOffersOrientedSelector(
                distance_power=dist_power,
                weights=issue_weights,
                offer_filter=offer_filter,
            ),
            **kwargs,
        )


class AdditiveParetoFollowingTBNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by
    the  current utility level based on a weighted sum of their normalized
    utilities and distances to previous offers
    """

    def __init__(
        self,
        *args,
        dist_power: float = 2,
        issue_weights: list[float] | None = None,
        offer_filter: OfferFilterProtocol = NoFiltering,
        **kwargs,
    ):
        """Initializes the negotiator with additive Pareto-following selection."""
        super().__init__(
            *args,
            offer_selector=AdditivePartnerOffersOrientedSelector(
                distance_power=dist_power,
                weights=issue_weights,
                offer_filter=offer_filter,
            ),
            **kwargs,
        )


class MultiplicativeLastOfferFollowingTBNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the
    current utility level based on a weighted sum of their normalized utilities and
    distances to previous offers
    """

    def __init__(
        self,
        *args,
        dist_power: float = 2,
        issue_weights: list[float] | None = None,
        **kwargs,
    ):
        """Initializes the negotiator with multiplicative last-offer following."""
        super().__init__(
            *args,
            offer_selector=MultiplicativePartnerOffersOrientedSelector(
                distance_power=dist_power, weights=issue_weights, offer_filter=KeepLast
            ),
            **kwargs,
        )


class AdditiveLastOfferFollowingTBNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by
    the  current utility level based on a weighted sum of their normalized
    utilities and distances to previous offers
    """

    def __init__(
        self,
        *args,
        dist_power: float = 2,
        issue_weights: list[float] | None = None,
        **kwargs,
    ):
        """Initializes the negotiator with additive last-offer following."""
        super().__init__(
            *args,
            offer_selector=AdditivePartnerOffersOrientedSelector(
                distance_power=dist_power, weights=issue_weights, offer_filter=KeepLast
            ),
            **kwargs,
        )


class MultiplicativeFirstFollowingTBNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the
    current utility level based on a weighted sum of their normalized utilities and
    distances to previous offers
    """

    def __init__(
        self,
        *args,
        dist_power: float = 2,
        issue_weights: list[float] | None = None,
        **kwargs,
    ):
        """Initializes the negotiator with multiplicative first-offer following."""
        super().__init__(
            *args,
            offer_selector=MultiplicativePartnerOffersOrientedSelector(
                distance_power=dist_power, weights=issue_weights, offer_filter=KeepFirst
            ),
            **kwargs,
        )


class AdditiveFirstFollowingTBNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by
    the  current utility level based on a weighted sum of their normalized
    utilities and distances to previous offers
    """

    def __init__(
        self,
        *args,
        dist_power: float = 2,
        issue_weights: list[float] | None = None,
        **kwargs,
    ):
        """Initializes the negotiator with additive first-offer following."""
        super().__init__(
            *args,
            offer_selector=AdditivePartnerOffersOrientedSelector(
                distance_power=dist_power, weights=issue_weights, offer_filter=KeepFirst
            ),
            **kwargs,
        )
