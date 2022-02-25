from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable

from attr import define, field

from negmas import warnings
from negmas.common import PreferencesChange, PreferencesChangeType
from negmas.sao.common import ResponseType

from .base import AcceptanceStrategy, FilterResult
from .concession import ConcessionRecommender
from .models.ufun import UFunModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
    from negmas.outcomes import Outcome
    from negmas.sao import SAOState
    from negmas.sao.negotiators.base import SAONegotiator

    from .offering import OfferingStrategy

__all__ = [
    "LimitedOutcomesAcceptanceStrategy",
    "NegotiatorAcceptanceStrategy",
    "ConcensusAcceptanceStrategy",
    "AllAcceptanceStrategies",
    "AnyAcceptanceStrategy",
    "AcceptImmediately",
    "RejectAlways",
    "EndImmediately",
    "AcceptAbove",
    "RandomAcceptanceStrategy",
    "AcceptTop",
    "AcceptBest",
    "TFTAcceptanceStrategy",
    "ACNext",
    "ACLast",
    "ACLastKReceived",
    "ACLastFractionReceived",
    "ACTime",
    "ACConst",
]


@define
class ACConst(AcceptanceStrategy):
    """
    Accepts $\\omega$ if $\alpha u(my-next-offer) + \beta > f(u(\text{utils of offers received in the last k steps))$
    """

    c: float = 0.9
    alpha: float = 1.0
    beta: float = 0.0

    def after_join(self, nmi) -> None:
        k = nmi.n_steps if self.k <= 0 else self.k
        self._best = [float("inf")] * k

    def before_responding(self, state: SAOState, offer: Outcome | None):
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._best.append(float(self.negotiator.ufun(offer)))
        self._best = self._best[1:]

    def __call__(self, state, offer):
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        u = float(self.negotiator.ufun(offer))
        if self.alpha * u + self.beta > self.c:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class ACLastKReceived(AcceptanceStrategy):
    """
    Accepts $\\omega$ if $\alpha u(my-next-offer) + \beta > f(u(\text{utils of offers received in the last k steps))$
    """

    k: int = 0
    alpha: float = 1.0
    beta: float = 0.0
    op: Callable[[list[float]], float] = max
    _best: list[float] = field(init=False, default=[])

    def after_join(self, nmi) -> None:
        k = nmi.n_steps if self.k <= 0 else self.k
        self._best = [float("inf")] * k

    def before_responding(self, state: SAOState, offer: Outcome | None):
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._best.append(float(self.negotiator.ufun(offer)))
        self._best = self._best[1:]

    def __call__(self, state, offer):
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        best = self.op(self._best)

        u = float(self.negotiator.ufun(offer))
        if self.alpha * u + self.beta > best:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class ACLastFractionReceived(AcceptanceStrategy):
    """
    Accepts $\\omega$ if $\alpha u(my-next-offer) + \beta > f(u(\text{utils of offers received in the given fraction of time}))$
    """

    fraction: float = 1.0
    alpha: float = 1.0
    beta: float = 0.0
    op: Callable[[list[float]], float] = max
    _best: list[tuple[float, float]] = field(init=False, default=[])

    def before_responding(self, state: SAOState, offer: Outcome | None):
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._best.append((float(self.negotiator.ufun(offer)), state.relative_time))

    def __call__(self, state, offer):
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        cutoff = state.relative_time - self.fraction
        lst = []
        for i in range(len(self._best) - 1, -1, -1):
            if self._best[i][1] < cutoff:
                self._best = self._best[i:]
                break
            lst.append(self._best[i][0])
        best = self.op(lst)

        u = float(self.negotiator.ufun(offer))
        if self.alpha * u + self.beta > best:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class ACLast(AcceptanceStrategy):
    """
    Implements the AClast acceptance strategy based on our last offer.

    Accepts $\\omega$ if $\alpha u(my-next-offer) + \beta > u(\\omega)$
    """

    last_offer_util: float = field(init=False, default=float("inf"))
    alpha: float = 1.0
    beta: float = 0.0

    def after_proposing(self, state: SAOState, offer: Outcome | None):
        if not self.negotiator or not self.negotiator.ufun:
            return
        self.last_offer_util = float(self.negotiator.ufun(offer))

    def __call__(self, state, offer):
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        last = self.last_offer_util
        u = float(self.negotiator.ufun(offer))
        if self.alpha * u + self.beta > last:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class ACTime(AcceptanceStrategy):
    """
    Implements the ACnext acceptance strategy based on our next offer.

    Accepts $\\omega$ if $\alpha u(my-next-offer) + \beta > u(\\omega)$
    """

    tau: float

    def __call__(self, state, offer):
        if state.relative_time > self.tau:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class ACNext(AcceptanceStrategy):
    """
    Implements the ACnext acceptance strategy based on our next offer.

    Accepts $\\omega$ if $\alpha u(my-next-offer) + \beta > u(\\omega)$
    """

    offering_strategy: OfferingStrategy
    alpha: float = 1.0
    beta: float = 0.0

    def __call__(self, state, offer):
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        next = float(self.negotiator.ufun(self.offering_strategy(state)))
        u = float(self.negotiator.ufun(offer))
        if self.alpha * u + self.beta > next:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class TFTAcceptanceStrategy(AcceptanceStrategy):
    """
    An acceptance strategy that concedes as much as the partner (or more)
    """

    partner_ufun: UFunModel
    recommender: ConcessionRecommender

    def __call__(self, state, offer):
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        partner_u = float(self.partner_ufun.eval_normalized(offer)) if offer else 1.0
        partner_concession = 1.0 - partner_u
        my_concession = self.recommender(partner_concession, state)
        if not math.isfinite(my_concession):
            warnings.warn(
                f"Got {my_concession} for concession which is unacceptable. Will use no concession"
            )
            my_concession = 0.0
        if not (-1e-6 <= my_concession <= 1.0000001):
            warnings.warn(f"{my_concession} is negative or above 1")
            my_concession = 0.0
        u = 1.0 - float(my_concession)
        if self.negotiator.ufun.eval_normalized(offer) >= u:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class RandomAcceptanceStrategy(AcceptanceStrategy):
    p_acceptance: float = 0.15
    p_rejection: float = 0.25
    p_ending: float = 0.1

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        r = random.random()
        if r <= self.p_acceptance + 1e-8:
            return ResponseType.ACCEPT_OFFER
        if r <= self.p_acceptance + self.p_rejection + 1e-8:
            return ResponseType.REJECT_OFFER
        if r <= self.p_acceptance + self.p_rejection + self.p_ending + 1e-8:
            return ResponseType.END_NEGOTIATION
        return ResponseType.NO_RESPONSE


@define
class AcceptBest(AcceptanceStrategy):
    """
    Accepts Only the best outcome.

    Remarks:
        - If the best possible utility cannot be found, nothing will be accepted
    """

    _best_util: float = float("inf")

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if self.negotiator.ufun(offer) >= self.negotiator.ufun.max() - 1e-10:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class AcceptTop(AcceptanceStrategy):
    """
    Accepts outcomes that are in the given top fraction or top `k`. If neither is given it reverts to accepting the best outcome only.

    Remarks:
        - The outcome-space is always discretized and the constraints `fraction` and `k` are applied to the discretized space
    """

    fraction: float = 0.0
    k: int = 1

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if any(
            _.type
            not in (
                PreferencesChangeType.Scaled,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            self.negotiator.ufun.invert().init()

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        top_k = self.negotiator.ufun.invert().within_indices((0, self.k))
        if offer in top_k:
            return ResponseType.ACCEPT_OFFER
        top_f = self.negotiator.ufun.invert().within_fractions((0.0, self.fraction))
        if offer in top_f:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class AcceptAbove(AcceptanceStrategy):
    """
    Accepts outcomes with utilities in the given top `limit` fraction above reserve/minimum (based on `above_resrve` ).
    """

    limit: float
    above_reserve: bool = True

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER
        if (
            self.negotiator.ufun.eval_normalized(offer, self.above_reserve)
            >= self.limit
        ):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class EndImmediately(AcceptanceStrategy):
    """
    Rejects immediately anything
    """

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        return ResponseType.END_NEGOTIATION


@define
class RejectAlways(AcceptanceStrategy):
    """
    Rejects everything
    """

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        return ResponseType.REJECT_OFFER


@define
class AcceptImmediately(AcceptanceStrategy):
    """
    Accepts immediately anything
    """

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        return ResponseType.ACCEPT_OFFER


@define
class LimitedOutcomesAcceptanceStrategy(AcceptanceStrategy):
    """
    Accepts from a list of predefined outcomes


    Remarks:
        - if `prob` is a number, it is taken as the probability of aceptance for any outcome.
        - if `prob` is `None`, the probability of acceptance of any outcome will be set to the relative time
    """

    prob: dict[Outcome, float] | float | None
    p_ending: float = 0.0

    @classmethod
    def from_outcome_list(
        cls,
        outcomes: list[Outcome],
        prob: list[float] | float = 1.0,
        p_ending: float = 0.0,
    ):
        if not isinstance(prob, Iterable):
            prob = [prob] * len(outcomes)
        return LimitedOutcomesAcceptanceStrategy(
            prob=dict(zip(outcomes, prob)), p_ending=p_ending
        )

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        if random.random() < self.p_ending - 1e-12:
            return ResponseType.END_NEGOTIATION
        if self.prob is None:
            prob = state.relative_time
        elif isinstance(self.prob, float):
            prob = self.prob
        else:
            prob = self.prob.get(offer, 0.0)
        if random.random() <= prob:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class NegotiatorAcceptanceStrategy(AcceptanceStrategy):
    """
    Uses a negotiator as an offering strategy
    """

    acceptor: SAONegotiator

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        return self.acceptor.respond(state, offer)


@define
class ConcensusAcceptanceStrategy(AcceptanceStrategy, ABC):
    """
    Accepts based on concensus of multiple strategies
    """

    strategies: list[AcceptanceStrategy]

    def filter(self, indx: int, response: ResponseType) -> FilterResult:
        """
        Called with the decision of each strategy in order.


        Remarks:
            - Two decisions need to be made:

              1. Should we continue trying other strategies
              2. Should we save this result.
        """
        return FilterResult(True, True)

    @abstractmethod
    def decide(self, indices: list[int], responses: list[ResponseType]) -> ResponseType:
        """
        Called to make a final decsision given the decisions of the stratgeis with indices `indices` (see `filter` for filtering rules)
        """

    def __call__(self, state: SAOState, offer: Outcome) -> ResponseType:
        selected, selected_indices = [], []
        for i, s in enumerate(self.strategies):
            response = s.respond(state, offer)
            r = self.filter(i, response)
            if not r.next:
                break
            if r.save:
                selected.append(response)
                selected_indices.append(i)

        return self.decide(selected_indices, selected)


@define
class AllAcceptanceStrategies(ConcensusAcceptanceStrategy):
    """Accept only if all children accept, end only if all of them end, otherwise reject"""

    def filter(self, indx: int, response: ResponseType) -> FilterResult:
        if response == ResponseType.REJECT_OFFER:
            return FilterResult(False, True)
        if response == ResponseType.END_NEGOTIATION:
            return FilterResult(False, True)
        return FilterResult(True, False)

    def decide(self, indices: list[int], responses: list[ResponseType]) -> ResponseType:
        if not responses:
            return ResponseType.ACCEPT_OFFER
        return responses[0]


@define
class AnyAcceptanceStrategy(ConcensusAcceptanceStrategy):
    """Accept any children accept, end or reject only if all of them end or reject"""

    def filter(self, indx: int, response: ResponseType) -> FilterResult:
        if response == ResponseType.ACCEPT_OFFER:
            return FilterResult(False, True)
        if response == ResponseType.END_NEGOTIATION:
            return FilterResult(False, True)
        return FilterResult(True, False)

    def decide(self, indices: list[int], responses: list[ResponseType]) -> ResponseType:
        if not responses:
            return ResponseType.REJECT_OFFER
        return responses[0]
