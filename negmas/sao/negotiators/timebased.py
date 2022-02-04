from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Sequence, TypeVar

from negmas import warnings
from negmas.helpers.numeric import isint, isreal
from negmas.negotiators.components import Aspiration, PolyAspiration, TimeCurve
from negmas.outcomes.cardinal_issue import CardinalIssue
from negmas.preferences import (
    BaseUtilityFunction,
    InverseUFun,
    PresortingInverseUtilityFunction,
    RankOnlyUtilityFunction,
    SamplingInverseUtilityFunction,
)
from negmas.sao import SAOState

from ...common import PreferencesChange
from ...outcomes import CartesianOutcomeSpace, Outcome, OutcomeSpace
from ..common import ResponseType
from .base import SAONegotiator

__all__ = [
    "AspirationNegotiator",
    "TimeBasedNegotiator",
    "TimeBasedConcedingNegotiator",
    "BoulwareTBNegotiator",
    "LinearTBNegotiator",
    "ConcederTBNegotiator",
    "FirstOfferOrientedTBNegotiator",
    "ParetoFollowingTBNegotiator",
    "AdditiveParetoFollowingTBNegotiator",
    "MultiplicativeParetoFollowingTBNegotiator",
]


def generalized_euclidean_distance(
    outcome_space: OutcomeSpace | None,
    a: Outcome,
    b: Outcome,
    weights: Sequence[float] | None = None,
    dist_power: float = 2,
) -> float:
    """
    Calculates the difference between two outcomes given an outcome-space (optionally with issue weights). This is defined as the distance.

    Remarks:

        - Becomes the Euclidean distance if all issues are numeric and no weights are given
        - You can control the power:

            - Setting it to 1 is the city-block distance
            - Setting it to 0 is the maximum issue difference
    """
    if not weights:
        weights = [1] * len(a)
    if dist_power <= 0 or dist_power == float("inf"):
        if not isinstance(outcome_space, CartesianOutcomeSpace):
            return max(
                (w * abs(x - y))
                if (isint(x) or isreal(x)) and (isint(y) or isreal(y))
                else (w * int(x == y))
                for w, x, y in zip(weights, a, b)
            )
        d = float("-inf")
        for issue, w, x, y in zip(outcome_space.issues, weights, a, b):
            if isinstance(issue, CardinalIssue):
                c = w * abs(x - y)
            else:
                c = w * int(x == y)
            if c > d:
                d = c
        return d
    if not isinstance(outcome_space, CartesianOutcomeSpace):
        return math.pow(
            sum(
                (w * math.pow(abs(x - y), dist_power))
                if (isint(x) or isreal(x)) and (isint(y) or isreal(y))
                else (w * int(x == y))
                for w, x, y in zip(weights, a, b)
            ),
            1.0 / dist_power,
        )
    d = 0.0
    for issue, w, x, y in zip(outcome_space.issues, weights, a, b):
        if isinstance(issue, CardinalIssue):
            d += w * math.pow(abs(x - y), dist_power)
            continue
        d += w * int(x == y)
    return math.pow(d, 1.0 / dist_power)


def min_dist(
    outcome: Outcome,
    outcomes: Sequence[Outcome],
    outcome_space: OutcomeSpace | None,
    weights: Sequence[float] | None,
    **kwargs,
) -> float:
    """
    Minimum distance between an outcome and a set of outcomes in an outcome-spaceself.

    See Also:

        `diff`
    """
    if not outcomes:
        return 1.0
    return min(
        generalized_euclidean_distance(outcome_space, outcome, _, weights, **kwargs)
        for _ in outcomes
    )


def additive_outcome_selector(
    outcomes: Sequence[Outcome],
    partner_offers: Sequence[Outcome],
    ufun: BaseUtilityFunction,
    weights: Sequence[float] | None = None,
    u_weight: float = 0.5,
    dist_power: float = 2,
) -> Outcome:
    """
    Selects the outcome that maximizes the weightd sum of utility value and distance to the partner offers (with `u_weight` weighing the utility value)

    See Also:

        `min_dist` , `diff`
    """
    utils = [float(ufun(_)) for _ in outcomes]
    dists = [
        min_dist(
            w,
            partner_offers,
            ufun.outcome_space,
            weights=weights,
            dist_power=dist_power,
        )
        for w in outcomes
    ]
    max_dist = max(dists)
    if abs(max_dist) < 1e-8:
        scores = sorted(zip(utils, outcomes), reverse=True)
        return scores[0][1]
    dists = [(max_dist - _) / max_dist for _ in dists]
    scores = sorted(
        (
            (
                u * u_weight + (1 - u_weight) * d,
                w,
            )
            for (u, d, w) in zip(utils, dists, outcomes)
        ),
        reverse=True,
    )
    return scores[0][1]


def multiplicative_outcome_selector(
    outcomes: Sequence[Outcome],
    partner_offers: Sequence[Outcome],
    ufun: BaseUtilityFunction,
    weights: Sequence[float] | None = None,
    dist_power: float = 2,
) -> Outcome:
    """
    Selects the outcome that maximizes the product of utility value and distance to the partner offers.

    See Also:

        `min_dist` , `diff`
    """
    utils = [float(ufun(_)) for _ in outcomes]
    dists = [
        min_dist(w, partner_offers, ufun.outcome_space, weights, dist_power=dist_power)
        for w in outcomes
    ]
    max_dist = max(dists)
    if abs(max_dist) < 1e-8:
        scores = sorted(zip(utils, outcomes), reverse=True)
        return scores[0][1]
    dists = [(max_dist - _) / max_dist for _ in dists]
    scores = sorted(
        ((u * d, w) for (u, d, w) in zip(utils, dists, outcomes)),
        reverse=True,
    )
    return scores[0][1]


def make_inverter(
    ufun: BaseUtilityFunction,
    ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
    rank_only: bool = False,
    max_cardinality: int = 10_000,
) -> InverseUFun:
    """
    Creates an `InverseUFun` object from the given ufun with appropriate type if the type is not given

    Args:
        ufun (BaseUtilityFunction): The ufun to invert
        rank_only (bool): If True, only the relative ranks of outcomes will be used in inversion not the values themselves.
        ufun_inverter (Callable[[BaseUtilityFunction], InverseUFun] | None): An optional factory to generate a `InverseUFun` from a `BaseUtilityFunction` .
        max_cardinality (int): The maximum cardinality at which we switch to using a `SamplingInverseUtilityFunction`
    """
    if rank_only:
        ufun = RankOnlyUtilityFunction(ufun, randomize_equal=False, name=ufun.name)
    if ufun_inverter:
        return ufun_inverter(ufun)
    return (
        SamplingInverseUtilityFunction(ufun)
        if ufun.outcome_space is None
        or ufun.outcome_space.cardinality >= max_cardinality
        else PresortingInverseUtilityFunction(ufun)
    )


def make_offer_selector(
    inverse_ufun: InverseUFun,
    selector: Callable[[Sequence[Outcome]], Outcome]
    | Literal["best"]
    | Literal["worst"]
    | None = None,
) -> Callable[[tuple[float, float]], Outcome | None]:
    """
    Generates a callable that can be used to select a specific outcome in a range of utility values.

    Args:

        inverse_ufun: The inverse utility function used
        selector: Any callable that selects an outcome, or "best"/"worst" for the one with highest/lowest utility. `None` for random.
    """
    if selector is None:
        return inverse_ufun.one_in
    if isinstance(selector, Callable):
        return lambda x: selector(inverse_ufun.some(x))
    if selector == "best":
        return inverse_ufun.best_in
    if selector == "worst":
        return inverse_ufun.worst_in
    raise ValueError(f"Unknown selector type: {selector}")


TC = TypeVar("TC", bound=TimeCurve, contravariant=False)


def make_curve(
    curve: TC | Literal["boulware"] | Literal["conceder"] | Literal["linear"] | float,
    starting_utility: float = 1.0,
) -> TC:
    """
    Generates a `TimeCurve` or `Aspiration` with optional `starting_utility`
    """
    if isinstance(curve, TimeCurve):
        return curve
    return PolyAspiration(starting_utility, curve)


class TimeBasedNegotiator(SAONegotiator):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        name: The agent name
        preferences:  The utility function to attache with the agent
        owner: The `Agent` that owns the negotiator.
        parent: The parent which should be an `SAOController`
        offering_curve (TimeCurve): A `TimeCurve` that is to be used to sample outcomes when offering
        accepting_curve (TimeCurve): A `TimeCurve` that is to be used to decide utility range to accept
        rank_only (bool): rank_only
        max_cardinality (int): The maximum outcome space cardinality at which to use a `SamplingInverseUtilityFunction`. Only used if `ufun_inverter` is `None` .
        can_propose (bool): If `False` the agent cannot propose (can only accept/reject)
        eps: A fraction of maximum utility to use as slack when checking if an outcome's utility lies within the current range

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
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        offer_selector: Callable[[Sequence[Outcome]], Outcome]
        | Literal["best"]
        | Literal["worst"]
        | None = None,
        max_cardinality: int = 10_000,
        eps: float = 0.0001,
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
        self._inv: InverseUFun | None = None
        self._min = self._max = self._best = None
        self._rank_only = rank_only
        self._max_cartinality = max_cardinality
        self._inverter_factory = ufun_inverter
        self._selector_type = offer_selector
        self._selector: Callable[[Sequence[Outcome]], Outcome | None] | None = None
        self._inv_method = None
        self._eps = eps

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        super().on_preferences_changed(changes)
        if not self.ufun:
            self._inv, self._selector = None, None
            self._min = self._max = self._best = None
            return
        self._inv = make_inverter(
            self.ufun, self._inverter_factory, self._rank_only, self._max_cartinality
        )
        if self._selector_type is None:
            self._inv_method = self._inv.one_in
            self._selector = None
        elif isinstance(self._selector_type, Callable):
            self._selector, self._inv_method = self._selector_type, self._inv.some
        elif self._selector_type == "worst":
            self._inv_method = self._inv.worst_in
            self._selector = None
        elif self._selector_type == "best_in":
            self._inv_method = self._inv.best_in
            self._selector = None

        _worst, self._best = self.ufun.extreme_outcomes()
        self._min, self._max = float(self.ufun(_worst)), float(self.ufun(self._best))
        if self._min < self.reserved_value:
            self._min = self.reserved_value

    def respond(self, state, offer):
        if self.ufun is None:
            raise ValueError("Unkonwn ufun.")
        if self._max is None or self._min is None:
            warnings.warn(
                f"It seems that on_prefrences_changed() was not called until propose for a ufun of type {self.ufun.__class__.__name__} for negotiator {self.id} of type {self.__class__.__name__}"
            )
            self.on_preferences_changed([PreferencesChange.General])
        if self._max is None or self._min is None:
            raise ValueError("Cannot find extreme outcomes.")
        urange = self._accepting_curve.utility_range(state.relative_time)
        w, b = ((self._max - self._min) * _ + self._min for _ in urange)
        urange = (w - self._eps, b + self._eps)
        if urange[0] <= self.ufun(offer) <= urange[1]:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state):
        if self.ufun is None:
            raise ValueError("Unkonwn ufun.")
        if self._inv is None or self._max is None or self._min is None:
            warnings.warn(
                f"It seems that on_prefrences_changed() was not called until propose for a ufun of type {self.ufun.__class__.__name__} for negotiator {self.id} of type {self.__class__.__name__}"
            )
            self.on_preferences_changed([PreferencesChange.General])
        if self._inv is None or self._max is None or self._min is None:
            raise ValueError(
                "Failed to find an invertor, a selector, or exreme outputs"
            )
        if not self._inv.initialized:
            self._inv.init()
        urange = self._offering_curve.utility_range(state.relative_time)
        w, b = ((self._max - self._min) * _ + self._min for _ in urange)
        urange = (w - self._eps, b + self._eps)
        outcomes = self._inv_method(urange)  # type: ignore
        outcome = self._selector(outcomes) if self._selector else outcomes  # type: ignore
        if not outcome:
            return self._best
        return outcome


class TimeBasedConcedingNegotiator(TimeBasedNegotiator):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        name: The agent name
        preferences:  The utility function to attache with the agent
        owner: The `Agent` that owns the negotiator.
        parent: The parent which should be an `SAOController`
        offering_curve (TimeCurve): A `TimeCurve` that is to be used to sample outcomes when offering
        accepting_curve (TimeCurve): A `TimeCurve` that is to be used to decide utility range to accept
        rank_only (bool): rank_only
        max_cardinality (int): The maximum outcome space cardinality at which to use a `SamplingInverseUtilityFunction`. Only used if `ufun_inverter` is `None` .
        stochastic (bool): If True, the negotiator will use the
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
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        max_cardinality: int = 10_000,
        stochastic: bool = False,
        starting_utility: float = 1.0,
        offer_selector: Callable[[Sequence[Outcome]], Outcome]
        | Literal["best"]
        | Literal["worst"]
        | None = None,
        **kwargs,
    ):
        offering_curve = make_curve(offering_curve, starting_utility)
        if not accepting_curve:
            accepting_curve = offering_curve
        if offer_selector is None and not stochastic:
            offer_selector = "worst"
        super().__init__(
            *args,
            offering_curve=offering_curve,
            accepting_curve=accepting_curve,
            rank_only=rank_only,
            ufun_inverter=ufun_inverter,
            max_cardinality=max_cardinality,
            offer_selector=offer_selector,
            **kwargs,
        )


class BoulwareTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A Boulware time-based negotiator that conceeds sub-linearly
    """

    def __init__(
        self,
        *args,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        max_cardinality: int = 10_000,
        stochastic: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            offering_curve=PolyAspiration(1.0, "boulware"),
            accepting_curve=None,
            rank_only=rank_only,
            ufun_inverter=ufun_inverter,
            max_cardinality=max_cardinality,
            offer_selector=None if stochastic else "worst",
            **kwargs,
        )


class LinearTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A Boulware time-based negotiator that conceeds linearly
    """

    def __init__(
        self,
        *args,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        max_cardinality: int = 10_000,
        stochastic: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            offering_curve=PolyAspiration(1.0, "linear"),
            accepting_curve=None,
            rank_only=rank_only,
            ufun_inverter=ufun_inverter,
            max_cardinality=max_cardinality,
            offer_selector=None if stochastic else "worst",
            **kwargs,
        )


class ConcederTBNegotiator(TimeBasedConcedingNegotiator):
    """
    A Boulware time-based negotiator that conceeds super-linearly
    """

    def __init__(
        self,
        *args,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        max_cardinality: int = 10_000,
        stochastic: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            offering_curve=PolyAspiration(1.0, "conceder"),
            accepting_curve=None,
            rank_only=rank_only,
            ufun_inverter=ufun_inverter,
            max_cardinality=max_cardinality,
            offer_selector=None if stochastic else "worst",
            **kwargs,
        )


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
        parent: The parent which should be an `SAOController`

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
        ranking_only=False,
        presort: bool = True,
        tolerance: float = 0.001,
        **kwargs,
    ):
        super().__init__(
            *args,
            offering_curve=aspiration_type,
            accepting_curve=aspiration_type,
            rank_only=ranking_only,
            ufun_inverter=None if not presort else PresortingInverseUtilityFunction,
            stochastic=stochastic,
            starting_utility=max_aspiration,
            eps=tolerance,
            **kwargs,
        )

    @property
    def tolerance(self):
        return self._eps

    def utility_at(self, t):
        if not self._offering_curve:
            raise ValueError(f"No inverse ufun is known yet")
        return self._offering_curve.utility_at(t)  # type: ignore (I know it is an Aspiration not a TimeCurve)

    @property
    def ufun_max(self):
        return self._max

    @property
    def ufun_min(self):
        return self._min


class DynamicSelectorTBNegotiator(TimeBasedNegotiator, ABC):
    """
    Time-based negotiators that use a dynamic way to select which offer to give the opponent from the set selected by the current utility level
    """

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        super().on_preferences_changed(changes)
        self._inv_method = self._inv.some  # type: ignore
        self._selector = self.outcome_selector

    @abstractmethod
    def outcome_selector(self, outcomes: Sequence[Outcome]) -> Outcome | None:
        ...


class OfferOrientedTBNegotiator(DynamicSelectorTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on their utility value and how near they are to the partner's first offer (abstract).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs["stochastic"] = True
        self._followed_offer = None

    @abstractmethod
    def pivot_selector(
        self, old: Outcome | None, new: Outcome | None, state: SAOState
    ) -> Outcome | None:
        ...

    def outcome_selector(self, outcomes: Sequence[Outcome]) -> Outcome | None:
        if not self.ufun:
            raise ValueError(f"Unkonwn ufun")
        if not self._followed_offer:
            return random.choice(outcomes)
        nearest, ndist = None, float("inf")
        for o in outcomes:
            d = generalized_euclidean_distance(
                self.ufun.outcome_space, o, self._followed_offer, None
            )
            if d < ndist:
                nearest, ndist = o, d
        return nearest

    def respond(self, state, offer):
        self._followed_offer = self.pivot_selector(self._followed_offer, offer, state)
        if not self._followed_offer:
            self._followed_offer = offer
        return super().respond(state, offer)


class FirstOfferOrientedTBNegotiator(OfferOrientedTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on their utility value and how near they are to the partner's first offer
    """

    def pivot_selector(
        self, old: Outcome | None, new: Outcome | None, state: SAOState
    ) -> Outcome | None:
        if old is None:
            return new
        return old


class LastOfferOrientedTBNegotiator(FirstOfferOrientedTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on their utility value and how near they are to the partner's last offer
    """

    def pivot_selector(
        self, old: Outcome | None, new: Outcome | None, state: SAOState
    ) -> Outcome | None:
        return new


class ParetoFollowingTBNegotiator(DynamicSelectorTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on their utility value and how near they are to the partner's previous offers.


    Remarks:
        - You can pass extra parameters to the `offer_selector` using `offer_selector_params` .
        - Builtin selectors (i.e. `additive_outcome_selector`, `multiplicative_outcome_selector` ) can be passed the following parameters:

           - weights: To choose a weight for each issue when evaluating outcome distances. Default is 1 for all issues
           - dist_power: To choose a power for calculating distances. Default is 2 (for Euclidean distance)
    """

    def __init__(
        self,
        *args,
        offer_selector: Callable[
            [Sequence[Outcome], Sequence[Outcome], BaseUtilityFunction], Outcome
        ] = multiplicative_outcome_selector,
        offer_selector_params: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, offer_selector=None, **kwargs)
        self._partner_offers: set[Outcome] = set()
        self.offer_selector = offer_selector
        self._selector_params = (
            offer_selector_params if offer_selector_params else dict()
        )

    def respond(self, state, offer):
        self._partner_offers.add(offer)
        return super().respond(state, offer)

    def outcome_selector(self, outcomes: Sequence[Outcome]) -> Outcome | None:
        if self.ufun is None:
            raise ValueError(f"Unknown ufun")
        return self.offer_selector(
            outcomes, list(self._partner_offers), self.ufun, **self._selector_params
        )


class MultiplicativeParetoFollowingTBNegotiator(ParetoFollowingTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on a weighted sum of their normalized utilities and distances to previous offers


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
            offer_selector=multiplicative_outcome_selector,
            offer_selector_params=dict(weights=issue_weights, dist_power=dist_power),
            **kwargs,
        )


class AdditiveParetoFollowingTBNegotiator(ParetoFollowingTBNegotiator):
    """
    A time-based negotiator that selectes outcomes from the list allowed by the  current utility level based on a weighted sum of their normalized utilities and distances to previous offers


    """

    def __init__(
        self,
        *args,
        dist_power: float = 2,
        utility_weight: float = 0.25,
        issue_weights: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            offer_selector=additive_outcome_selector,
            offer_selector_params=dict(
                weights=issue_weights, u_weight=utility_weight, dist_power=dist_power
            ),
            **kwargs,
        )
