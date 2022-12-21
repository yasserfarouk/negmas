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
from negmas.preferences import InverseUFun, PresortingInverseUtilityFunction

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
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        offering_curve (TimeCurve): A `TimeCurve` that is to be used to sample outcomes when offering
        accepting_curve (TimeCurve): A `TimeCurve` that is to be used to decide utility range to accept
        inverter (UtilityInverter): A component used to keep track of the ufun inverse
        stochastic (bool): If `False` the worst outcome in the current utility range will be used
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
        super().__init__(*args, **kwargs)
        offering_curve = make_curve(offering_curve, 1.0)
        if accepting_curve is None:
            accepting_curve = offering_curve
        else:
            accepting_curve = make_curve(accepting_curve, 1.0)
        self._offering_curve = offering_curve
        self._accepting_curve = accepting_curve

    def utility_range_to_propose(self, state) -> tuple[float, float]:
        return self._offering_curve.utility_range(state.relative_time)

    def utility_range_to_accept(self, state) -> tuple[float, float]:
        return self._accepting_curve.utility_range(state.relative_time)


class TimeBasedConcedingNegotiator(TimeBasedNegotiator):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        offering_curve (TimeCurve): A `TimeCurve` that is to be used to sample outcomes when offering
        accepting_curve (TimeCurve): A `TimeCurve` that is to be used to decide utility range to accept
        starting_utility (float): The relative utility (range 1.0 to 0.0) at which to give the first offer. Only used if `offering_curve` was not given
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
    A Boulware time-based negotiator that conceeds sub-linearly
    """

    def __init__(self, *args, **kwargs):
        kwargs["offering_curve"] = PolyAspiration(1.0, "boulware")
        kwargs["stochastic"] = False
        super().__init__(*args, **kwargs)


class LinearTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A Boulware time-based negotiator that conceeds linearly
    """

    def __init__(self, *args, **kwargs):
        kwargs["offering_curve"] = PolyAspiration(1.0, "linear")
        kwargs["stochastic"] = False
        super().__init__(*args, **kwargs)


class ConcederTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A Boulware time-based negotiator that conceeds super-linearly
    """

    def __init__(self, *args, **kwargs):
        kwargs["offering_curve"] = PolyAspiration(1.0, "conceder")
        kwargs["stochastic"] = False
        super().__init__(*args, **kwargs)


class AspirationNegotiator(TimeBasedConcedingNegotiator):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        name: The agent name
        preferences:  The utility function to attache with the agent
        max_aspiration: The aspiration level to use for the first offer (or first acceptance decision).
        aspiration_type: The polynomial aspiration curve type. Here you can pass the exponent as a real value or
                         pass a string giving one of the predefined types: linear, conceder, boulware.
        stochastic: If True, the agent will propose outcomes with utility >= the current aspiration level not
                         outcomes just above it.
        can_propose: If True, the agent is allowed to propose
        ranking: If True, the aspiration level will not be based on the utility value but the ranking of the outcome
                 within the presorted list. It is only effective when presort is set to True
        presort: If True, the negotiator will catch a list of outcomes, presort them and only use them for offers
                 and responses. This is much faster then other option for general continuous utility functions
                 but with the obvious problem of only exploring a discrete subset of the issue space (Decided by
                 the `discrete_outcomes` property of the `NegotiatorMechanismInterface` . If the number of outcomes is
                 very large (i.e. > 10000) and discrete, presort will be forced to be True. You can check if
                 presorting is active in realtime by checking the "presorted" attribute.
        tolerance: A tolerance used for sampling of outcomes when `presort` is set to False
        owner: The `Agent` that owns the negotiator.
        parent: The parent which should be an `GBController`

    Remarks:

        - This class provides a different interface to the `TimeBasedConcedingNegotiator` with less control. It is recommonded to use
          `TimeBasedConcedingNegotiator` or other `TimeBasedNegotiator` negotiators isntead

    """

    def __init__(
        self,
        *args,
        max_aspiration=1.0,
        aspiration_type="boulware",
        stochastic=False,
        presort: bool = False,
        tolerance: float = 0.001,
        **kwargs,
    ):
        ufun_inverter = None if not presort else PresortingInverseUtilityFunction
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
        return self._inverter.tolerance

    def utility_at(self, t):
        if not self._offering_curve:
            raise ValueError(f"No inverse ufun is known yet")
        return self._offering_curve.utility_at(t)  # type: ignore (I know it is an Aspiration not a TimeCurve)

    @property
    def ufun_max(self):
        return self._inverter.ufun_max

    @property
    def ufun_min(self):
        return self._inverter.ufun_min


class OfferOrientedNegotiator(TimeBasedNegotiator):
    """
    A time-based negotiator that orients its offers toward some pivot outcome (See `OfferOrientedSelector` )
    """

    def __init__(self, *args, offer_selector: OfferOrientedSelector, **kwargs):
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
        super().__init__(
            *args,
            offer_selector=AdditivePartnerOffersOrientedSelector(
                distance_power=dist_power, weights=issue_weights, offer_filter=KeepFirst
            ),
            **kwargs,
        )
