from __future__ import annotations

from abc import abstractmethod
from random import choice
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

from yaml import warnings

from negmas.gb.components import GBComponent
from negmas.outcomes.outcome_ops import generalized_minkowski_distance, min_dist
from negmas.preferences import (
    BaseUtilityFunction,
    InverseUFun,
    PresortingInverseUtilityFunction,
    RankOnlyUtilityFunction,
)

if TYPE_CHECKING:
    from negmas.gb import GBState
    from negmas.outcomes import Outcome
    from negmas.outcomes.outcome_space import DistanceFun


__all__ = [
    "OfferSelectorProtocol",
    "OfferSelector",
    "RandomOfferSelector",
    "BestOfferSelector",
    "MedianOfferSelector",
    "WorstOfferSelector",
    "OfferOrientedSelector",
    "FirstOfferOrientedSelector",
    "LastOfferOrientedSelector",
    "BestOfferOrientedSelector",
    "OutcomeSetOrientedSelector",
    "PartnerOffersOrientedSelector",
    "MultiplicativePartnerOffersOrientedSelector",
    "AdditivePartnerOffersOrientedSelector",
]


def additive_score(
    outcomes: Sequence[Outcome],
    partner_offers: Sequence[Outcome],
    ufun: BaseUtilityFunction,
    distance_fun: DistanceFun = generalized_minkowski_distance,
    u_weight: float = 0.5,
    **kwargs,
) -> Sequence[tuple[float, Outcome]]:
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
            distance_fun=distance_fun,
            **kwargs,
        )
        for w in outcomes
    ]
    max_dist = max(dists)
    if abs(max_dist) < 1e-8:
        scores = sorted(zip(utils, outcomes), reverse=True)
        return scores[0][1]
    dists = [(max_dist - _) / max_dist for _ in dists]
    scores = [
        (u * u_weight + (1 - u_weight) * d, w)
        for (u, d, w) in zip(utils, dists, outcomes)
    ]
    return scores


def multiplicative_score(
    outcomes: Sequence[Outcome],
    partner_offers: Sequence[Outcome],
    ufun: BaseUtilityFunction,
    distance_fun: DistanceFun = generalized_minkowski_distance,
    **kwargs,
) -> Sequence[tuple[float, Outcome]]:
    """
    Selects the outcome that maximizes the product of utility value and distance to the partner offers.

    See Also:

        `min_dist` , `diff`
    """
    utils = [float(ufun(_)) for _ in outcomes]
    dists = [
        min_dist(
            w,
            partner_offers,
            ufun.outcome_space,
            distance_fun=distance_fun,
            **kwargs,
        )
        for w in outcomes
    ]
    max_dist = max(dists)
    if abs(max_dist) < 1e-8:
        scores = sorted(zip(utils, outcomes), reverse=True)
        return scores[0][1]
    dists = [(max_dist - _) / max_dist for _ in dists]
    scores = [(u * d, w) for (u, d, w) in zip(utils, dists, outcomes)]
    return scores


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
    return PresortingInverseUtilityFunction(ufun, max_cache_size=max_cardinality)


class OfferFilterProtocol(Protocol):
    """
    Can select *the best* offers in some  sense from a list of offers based on an inverter
    """

    def __call__(
        self, outcomes: Sequence[Outcome], state: GBState
    ) -> Sequence[Outcome]:
        ...


def NoFiltering(outcomes: Sequence[Outcome], state: GBState) -> Sequence[Outcome]:
    return outcomes


def KeepFirst(outcomes: Sequence[Outcome], state: GBState) -> Sequence[Outcome]:
    return [] if not outcomes else [outcomes[0]]


def KeepLast(outcomes: Sequence[Outcome], state: GBState) -> Sequence[Outcome]:
    return [] if not outcomes else [outcomes[-1]]


class OfferSelectorProtocol(Protocol):
    """
    Can select *the best* offer in some  sense from a list of offers based on an inverter
    """

    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        ...


class OfferSelector(OfferSelectorProtocol, GBComponent):
    """
    Can select *the best* offer in some  sense from a list of offers based on an inverter
    """

    @abstractmethod
    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        ...


class RandomOfferSelector(OfferSelector):
    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        if not outcomes:
            return None
        return choice(outcomes)


class BestOfferSelector(OfferSelector):
    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        if not self._negotiator:
            warnings.warn(
                "Asked to select an outcome with unkonwn negotiator",
                warnings.NegmasUnexpectedValueWarning,
            )
            return None
        if not self._negotiator.ufun:
            warnings.warn(
                "Asked to select an outcome with unkonwn utility function",
                warnings.NegmasUnexpectedValueWarning,
            )
            return None
        if not outcomes:
            return None
        return sorted(((self._negotiator.ufun(_), _) for _ in outcomes), reverse=True)[
            0
        ][1]


class MedianOfferSelector(OfferSelector):
    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        if not self._negotiator:
            warnings.warn(
                "Asked to select an outcome with unkonwn negotiator",
                warnings.NegmasUnexpectedValueWarning,
            )
            return None
        if not self._negotiator.ufun:
            warnings.warn(
                "Asked to select an outcome with unkonwn utility function",
                warnings.NegmasUnexpectedValueWarning,
            )
            return None
        if not outcomes:
            return None
        ordered = sorted(
            ((self._negotiator.ufun(_), _) for _ in outcomes), reverse=False
        )
        return ordered[len(ordered) // 2][1]


class WorstOfferSelector(OfferSelector):
    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        if not self._negotiator:
            warnings.warn(
                "Asked to select an outcome with unkonwn negotiator",
                warnings.NegmasUnexpectedValueWarning,
            )
            return None
        if not self._negotiator.ufun:
            warnings.warn(
                "Asked to select an outcome with unkonwn utility function",
                warnings.NegmasUnexpectedValueWarning,
            )
            return None
        if not outcomes:
            return None
        return sorted(((self._negotiator.ufun(_), _) for _ in outcomes), reverse=False)[
            0
        ][1]


class OfferOrientedSelector(OfferSelector):
    """
    Selects the nearest outcome to the pivot outcome which is updated before responding
    """

    def __init__(
        self, distance_fun: DistanceFun = generalized_minkowski_distance, **kwargs
    ):
        self._pivot: Outcome | None = None
        self._distance_fun = distance_fun
        self._distance_fun_params = kwargs

    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        if not self._negotiator or not self._negotiator.ufun:
            raise ValueError(f"Unknown ufun or negotiator")
        if not self._pivot:
            return choice(outcomes)
        nearest, ndist = None, float("inf")
        for o in outcomes:
            d = self._distance_fun(
                o,
                self._pivot,
                self._negotiator.ufun.outcome_space,
                **self._distance_fun_params,
            )
            if d < ndist:
                nearest, ndist = o, d
        return nearest


class FirstOfferOrientedSelector(OfferOrientedSelector):
    """
    Selects the offer nearest the partner's first offer
    """

    def before_responding(
        self, state: GBState, offer: Outcome | None, source: str
    ) -> Outcome | None:
        if self._pivot or offer is None:
            return
        self._pivot = offer


class LastOfferOrientedSelector(OfferOrientedSelector):
    """
    Selects the offer nearest the partner's last offer
    """

    def before_responding(self, state: GBState, offer: Outcome | None, source: str):
        if offer is None:
            return
        self._pivot = offer


class BestOfferOrientedSelector(OfferOrientedSelector):
    """
    Selects the offer nearest the partner's best offer for me so far
    """

    _pivot_util: float = float("-inf")

    def before_responding(self, state: GBState, offer: Outcome | None, source: str):
        if offer is None:
            return
        if not self._negotiator or not self._negotiator.ufun:
            raise ValueError(f"Unknown ufun or negotiator")
        u = self._negotiator.ufun(offer)
        if u is None:
            return
        u = float(u)
        if u > self._pivot_util:
            self._pivot_util = u
            self._pivot = offer
        self._pivot_util = u


class OutcomeSetOrientedSelector(OfferSelector):
    """
    Selects the nearest outcome to a set of pivot outcomes which is updated before responding
    """

    def __init__(
        self,
        distance_fun: DistanceFun = generalized_minkowski_distance,
        offer_filter: OfferFilterProtocol = NoFiltering,
        **kwargs,
    ):
        self._pivots: list[Outcome] = []
        self._distance_fun = distance_fun
        self._distnace_fun_params = kwargs
        self._offer_filter = offer_filter

    @abstractmethod
    def calculate_scores(
        self, outcomes: Sequence[Outcome], pivots: list[Outcome], state: GBState
    ) -> Sequence[tuple[float, Outcome]]:
        ...

    def __call__(self, outcomes: Sequence[Outcome], state: GBState) -> Outcome | None:
        if not self._negotiator or not self._negotiator.ufun:
            raise ValueError(f"Unknown ufun or negotiator")
        outcomes = self._offer_filter(outcomes, state)
        if not self._pivots:
            return choice(outcomes)
        scores = self.calculate_scores(outcomes, self._pivots, state)
        scores = sorted(
            scores,
            reverse=True,
        )
        return scores[0][1]


class PartnerOffersOrientedSelector(OutcomeSetOrientedSelector):
    """
    Orients offes toward the set of past opponent offers
    """

    def before_responding(self, state: GBState, offer: Outcome | None, source: str):
        if offer is None:
            return
        self._pivots.append(offer)


class MultiplicativePartnerOffersOrientedSelector(PartnerOffersOrientedSelector):
    """
    Orients offes toward the set of past opponent offers.

    The score of an offer is the product of its utility to self and its distance
    to opponent's past offers after normalization
    """

    def calculate_scores(
        self, outcomes: Sequence[Outcome], pivots: list[Outcome], state: GBState
    ) -> Sequence[tuple[float, Outcome]]:
        if not self._negotiator or not self._negotiator.ufun:
            raise ValueError(f"Unknown ufun or negotiator")
        return multiplicative_score(
            outcomes,
            pivots,
            self._negotiator.ufun,
            self._distance_fun,
            **self._distnace_fun_params,
        )


class AdditivePartnerOffersOrientedSelector(PartnerOffersOrientedSelector):
    """
    Orients offes toward the set of past opponent offers.

    The score of an offer is the product of its utility to self and its distance
    to opponent's past offers after normalization
    """

    def __init__(self, *args, u_weight: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_weight = u_weight

    def calculate_scores(
        self, outcomes: Sequence[Outcome], pivots: list[Outcome], state: GBState
    ) -> Sequence[tuple[float, Outcome]]:
        if not self._negotiator or not self._negotiator.ufun:
            raise ValueError(f"Unknown ufun or negotiator")
        return additive_score(
            outcomes,
            pivots,
            self._negotiator.ufun,
            u_weight=self.u_weight,
            distance_fun=self._distance_fun,
            **self._distnace_fun_params,
        )
