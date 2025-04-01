from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import bisect

from attrs import define, field

from negmas import warnings
from negmas.common import PreferencesChangeType, Value
from negmas.negotiators.helpers import PolyAspiration
from negmas.outcomes.common import ExtendedOutcome
from negmas.outcomes.protocols import DiscreteOutcomeSpace
from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction

from .base import FilterResult, OfferingPolicy
from .concession import ConcessionRecommender
from .models.ufun import UFunModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
    from negmas.gb import GBState
    from negmas.gb.negotiators.base import GBNegotiator
    from negmas.outcomes import Outcome


__all__ = [
    "CABOfferingPolicy",
    "WAROfferingPolicy",
    "LimitedOutcomesOfferingPolicy",
    "NegotiatorOfferingPolicy",
    "ConcensusOfferingPolicy",
    "RandomConcensusOfferingPolicy",
    "UnanimousConcensusOfferingPolicy",
    "UtilBasedConcensusOfferingPolicy",
    "MyBestConcensusOfferingPolicy",
    "MyWorstConcensusOfferingPolicy",
    "NoneOfferingPolicy",
    "RandomOfferingPolicy",
    "OfferTop",
    "OfferBest",
    "TFTOfferingPolicy",
    "MiCROOfferingPolicy",
    "TimeBasedOfferingPolicy",
    "HybridOfferingPolicy",
]


def index_of_nearest_value(
    values: list[float], x: float, above_only: bool = False
) -> int:
    """
    Finds the index of the nearest value to x

    Args:
        values: A list of sorted values (sorted ascendingly)
        x: The target float value.
        above_only: If True, only consider items with values above x.

    Returns:
        The index of the item with the nearest value. -1 means an empty list
    """

    if not values:
        return -1

    n = len(values)
    if above_only:
        # Find the first item with a value greater than or equal to x
        first_above_index = bisect.bisect_left(values, x)
        if first_above_index >= n - 1:
            return n - 1
        return first_above_index

    else:
        # Find the nearest value using bisect
        index = bisect.bisect_left(values, x)
        if index == 0:
            return 0
        elif index == n:
            return n - 1
        if abs(values[index - 1] - x) <= abs(values[index] - x):
            return index - 1
        return index


@define
class HybridOfferingPolicy(OfferingPolicy):
    initial_utility: float = float("nan")
    concession_ratio: float = float("nan")
    final_utility: float = float("nan")
    empathy_score: float = float("nan")
    frac_time_based: dict[int, tuple[float, ...]] = field(
        factory=lambda: {
            1: (1.0,),
            2: (0.25, 0.75),
            3: (0.11, 0.22, 0.66),
            4: (0.05, 0.15, 0.3, 0.5),
        }
    )
    above_only: bool = False
    _sent_offers: list[Outcome] = field(init=False, factory=list)
    _sent_utils: list[float] = field(init=False, factory=list)
    _received_offers: list[Outcome] = field(init=False, factory=list)
    _received_utils: list[float] = field(init=False, factory=list)
    _outcomes: list[Outcome] = field(init=False, factory=list)
    _values: list[float] = field(init=False, factory=list)

    def _adjust_params(self):
        self.initial_utility = 1.0
        self.concession_ratio = 0.75
        self.final_utility = 0.55
        self.empathy_score = 0.5

        ufun = self.negotiator.ufun
        assert ufun, "Unknown ufun. Cannot continue"
        domain_size = int(ufun.outcome_space.cardinality)  # type: ignore

        if domain_size < 450:
            self.final_utility = 0.80
        elif domain_size < 1500:
            self.final_utility = 0.775
        elif domain_size < 4500:
            self.final_utility = 0.75
        elif domain_size < 18000:
            self.final_utility = 0.725
        elif domain_size < 33000:
            self.final_utility = 0.70
        else:
            self.final_utility = 0.675

        self._sent_utils = [ufun(_) for _ in self._sent_offers]
        self._received_utils = [ufun(_) for _ in self._received_utils]

        self.final_utility = max(self.final_utility, float(ufun.reserved_value))
        os = ufun.outcome_space
        assert os
        outcomes = (
            os.enumerate_or_sample(levels=10, max_cardinality=1_000_000)
            if not isinstance(os, DiscreteOutcomeSpace)
            else os.enumerate()
        )
        r = float(ufun.reserved_value)
        outcome_util = sorted([(u, _) for _ in outcomes if (u := ufun(_)) >= r])
        self._outcomes = [_[1] for _ in outcome_util]
        self._values = [_[0] for _ in outcome_util]

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        self._adjust_params()
        return super().on_preferences_changed(changes)

    def time_based(self, t: float) -> float:
        """
            Target utility calculation of Time-Based strategy
        :param t: Negotiation time
        :return: Target utility
        """
        return (
            (1 - t) * (1 - t) * self.initial_utility
            + 2 * (1 - t) * t * self.concession_ratio
            + t * t * self.final_utility
        )

    def behaviour_based(self, t: float) -> float:
        """
            Target utility calculation of Behavior-Based strategy
        :param t: Negotiation time
        :return: Target utility
        """

        # Utility differences of consecutive offers of opponent
        diff = [
            self._received_utils[i + 1] - self._received_utils[i]
            for i in range(len(self._received_utils) - 1)
        ]

        # Fixed size window
        if len(diff) > len(self.frac_time_based):
            diff = diff[-len(self.frac_time_based) :]

        # delta = diff * window
        delta = sum([u * w for u, w in zip(diff, self.frac_time_based[len(diff)])])

        # Calculate target utility by updating the last offered bid
        target_utility = (
            self._sent_utils[-1] - (self.empathy_score + self.empathy_score * t) * delta
        )

        return target_utility

    def __call__(self, state: GBState, dest: str | None = None):
        if not self._values:
            self._adjust_params()
        t = state.relative_time
        # Target utility of Time-Based strategy
        target_utility = self.time_based(t)

        # If first 2 round, apply only Time-Based strategy
        if len(self._received_offers) > 2:
            # Target utility of Behavior-Based strategy
            behavior_utility = self.behaviour_based(t)

            # Combining Time-Based and Behavior-Based strategy
            target_utility = (1.0 - t * t) * behavior_utility + t * t * target_utility

        ufun = self.negotiator.ufun
        assert ufun, "Unknown ufun. Cannot continue"
        r = float(ufun.reserved_value)
        # Target utility cannot be lower than the reservation value.
        if target_utility < r:
            target_utility = r

        # # AC_Next strategy to decide accepting or not
        # if self.can_accept() and target_utility <= self.last_received_bids[-1].utility:
        #     return self.accept_action

        # Find the closest bid to target utility
        indx = index_of_nearest_value(self._values, target_utility)
        outcome = self._outcomes[indx]

        self._sent_offers.append(outcome)
        self._sent_utils.append(float(ufun(outcome)))

        return outcome

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        ufun = self.negotiator.ufun
        assert ufun, "Unknown ufun. Cannot continue"
        self._received_offers.append(offer)
        self._received_utils.append(float(ufun(offer)))
        return super().on_partner_proposal(state, partner_id, offer)


@define
class TimeBasedOfferingPolicy(OfferingPolicy):
    curve: PolyAspiration = field(factory=lambda: PolyAspiration(1.0, "boulware"))
    stochastic: bool = False

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if self.sorter is not None:
            warnings.warn(
                "Sorter is already initialized. May be on_preferences_changed is called twice!!"
            )
        self.sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self.sorter.init()

    def __call__(self, state: GBState, dest: str | None = None):
        assert self.negotiator.ufun is not None
        asp = self.curve.utility_at(state.relative_time)
        mn, mx = self.sorter.minmax()
        assert mn >= self.negotiator.ufun.reserved_value
        asp = asp * (mx - mn) + mn
        if self.stochastic:
            outcome = self.sorter.one_in((asp, mx), normalized=True)
        else:
            outcome = self.sorter.worst_in((asp - 1e-5, mx), normalized=True)
        if outcome:
            return outcome
        return self.sorter.best()


@define
class MiCROOfferingPolicy(OfferingPolicy):
    next_indx: int = 0
    sorter: PresortingInverseUtilityFunction | None = field(repr=False, default=None)
    _received: set[Outcome] = field(factory=set)
    _sent: set[Outcome] = field(factory=set)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if any(
            _.type
            not in (
                PreferencesChangeType.Scale,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
            self.next_indx = 0
            self._received = set()
            self._sent = set()

    def sample_sent(self) -> Outcome | None:
        if not len(self._sent):
            return None
        return random.choice(list(self._sent))

    def ensure_sorter(self):
        if not self.sorter:
            assert self.negotiator.ufun
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
        return self.sorter

    def next_offer(self) -> Outcome | None:
        return self.ensure_sorter().outcome_at(self.next_indx)

    def best_offer_so_far(self) -> Outcome | None:
        if self.next_indx > 0:
            return self.ensure_sorter().outcome_at(self.next_indx - 1)
        return None

    def ready_to_concede(self) -> bool:
        return len(self._sent) <= len(self._received)

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        outcome = self.next_offer()
        assert self.sorter
        assert self.negotiator.ufun
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
            or not self.ready_to_concede()
        ):
            return self.sample_sent()
        self.next_indx += 1
        self._sent.add(outcome)
        return outcome

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        self._received.add(offer)
        return super().on_partner_proposal(state, partner_id, offer)


@define
class CABOfferingPolicy(OfferingPolicy):
    next_indx: int = 0
    sorter: PresortingInverseUtilityFunction | None = field(repr=False, default=None)
    _last_offer: Outcome | None = field(init=False, default=None)
    _repeating: bool = field(init=False, default=False)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if any(
            _.type
            not in (
                PreferencesChangeType.Scale,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            if self.sorter is not None:
                warnings.warn(
                    "Sorter is already initialized. May be on_preferences_changed is called twice!!"
                )
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
            self.next_indx = 0
            self._repeating = False

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        if (
            self._repeating
            or not self.negotiator
            or not self.negotiator.ufun
            or not self.negotiator.nmi
        ):
            return self._last_offer
        if self.next_indx >= self.negotiator.nmi.n_outcomes:
            return self._last_offer
        if not self.sorter:
            warnings.warn(
                "Sorter is not initialized. May be on_preferences_changed is never called before propose!!"
            )
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
        outcome = self.sorter.outcome_at(self.next_indx)
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
        ):
            # self.negotiator.nmi.mechanism.plot()
            # breakpoint()
            self._repeating = True
            return self._last_offer
        self.next_indx += 1
        self._last_offer = outcome
        return outcome


@define
class WAROfferingPolicy(OfferingPolicy):
    next_indx: int = 0
    sorter: PresortingInverseUtilityFunction | None = field(repr=False, default=None)
    _last_offer: Outcome | None = field(init=False, default=None)
    _repeating: bool = field(init=False, default=False)
    _irrational: bool = field(init=False, default=True)
    _irrational_index: int = field(init=False, default=-1)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._irrational = True
        self._irrational_index = int(self.negotiator.nmi.n_outcomes) - 1
        if any(
            _.type
            not in (
                PreferencesChangeType.Scale,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            if self.sorter is not None:
                warnings.warn(
                    "Sorter is already initialized. May be on_preferences_changed is called twice!!"
                )
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
            self.next_indx = 0
            self._repeating = False

    def on_negotiation_start(self, state) -> None:
        self._repeating = False
        self._irrational = True
        self._irrational_index = self.negotiator.nmi.n_outcomes - 1  # type: ignore
        return super().on_negotiation_start(state)

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        if not self.negotiator or not self.negotiator.ufun or not self.negotiator.nmi:
            return self._last_offer
        if self._repeating:
            return self._last_offer
        if not self._irrational and self.next_indx >= self.negotiator.nmi.n_outcomes:
            return self._last_offer
        if not self.sorter:
            warnings.warn(
                "Sorter is not initialized. May be on_preferences_changed is never called before propose!!"
            )
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
        nxt = self._irrational_index if self._irrational else self.next_indx
        outcome = self.sorter.outcome_at(nxt)
        if self._irrational:
            if (
                outcome is None
                or self.sorter.utility_at(self._irrational_index)
                >= self.negotiator.ufun.reserved_value
            ):
                self._irrational = False
                assert self._last_offer is None
                outcome = self.sorter.outcome_at(self.next_indx)
            else:
                self._irrational_index -= 1
                return outcome
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
        ):
            self._repeating = True
            return self._last_offer
        self.next_indx += 1
        self._last_offer = outcome
        return outcome


@define
class TFTOfferingPolicy(OfferingPolicy):
    """
    An acceptance strategy that concedes as much as the partner (or more)
    """

    partner_ufun: UFunModel
    recommender: ConcessionRecommender
    stochastic: bool = False
    _partner_offer: Outcome | None = field(init=False, default=None)

    def before_responding(
        self, state: GBState, offer: Outcome | None, source: str | None = None
    ):
        self._partner_offer = offer

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        super().on_preferences_changed(changes)
        self.partner_ufun.on_preferences_changed(changes)

    def __call__(self, state: GBState, dest: str | None = None):
        if not self.negotiator or not self.negotiator.ufun:
            return None
        partner_u = (
            float(self.partner_ufun.eval_normalized(self._partner_offer, True))
            if self._partner_offer
            else 1.0
        )
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
        target_utility = 1.0 - float(my_concession)
        if self.stochastic:
            return self.negotiator.ufun.invert().one_in(
                (target_utility, 1.0), normalized=True
            )
        return self.negotiator.ufun.invert().worst_in(
            (target_utility, 1.0), normalized=True
        )


@define
class OfferBest(OfferingPolicy):
    """
    Offers Only the best outcome.

    Remarks:
        - You can pass the  best outcome if you know it as `best` otherwise it will find it.
    """

    _best: Outcome | None = None

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        _, self._best = self.negotiator.ufun.extreme_outcomes()

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        return self._best


@define
class OfferTop(OfferingPolicy):
    """
    Offers outcomes that are in the given top fraction or top `k`. If neither is given it reverts to only offering the best outcome

    Remarks:
        - The outcome-space is always discretized and the constraints `fraction` and `k` are applied to the discretized space
    """

    fraction: float = 0.0
    k: int = 1
    _top: list[Outcome] | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if any(
            _.type
            not in (
                PreferencesChangeType.Scale,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            inverter = self.negotiator.ufun.invert()
            inverter.init()
            top_k = inverter.within_indices((0, self.k))
            top_f = inverter.within_fractions((0.0, self.fraction))
            self._top = list(set(top_k + top_f))

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        if not self.negotiator or not self.negotiator.ufun:
            return None
        if self._top is None:
            self.on_preferences_changed([])
        if not self._top:
            return None
        return random.choice(self._top)


@define
class NoneOfferingPolicy(OfferingPolicy):
    """
    Always offers `None` which means it never gets an agreement.
    """

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        return None


@define
class RandomOfferingPolicy(OfferingPolicy):
    """
    Always offers `None` which means it never gets an agreement.
    """

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        if not self.negotiator or not self.negotiator.nmi:
            return None
        return self.negotiator.nmi.random_outcome()


@define
class LimitedOutcomesOfferingPolicy(OfferingPolicy):
    """
    Offers from a given list of outcomes
    """

    outcomes: list[Outcome] | None
    prob: list[float] | None = None
    p_ending: float = 0.0

    def _run(
        self, state: GBState, dest: str | None = None, second_trial: bool = False
    ) -> Outcome | None:
        if not self.negotiator or not self.negotiator.nmi:
            return None
        if random.random() < self.p_ending - 1e-7:
            return None
        if not self.prob or not self.outcomes:
            return random.choice(
                self.outcomes
                if self.outcomes
                else list(self.negotiator.nmi.discrete_outcomes())
            )
        r, s = random.random(), 0.0
        for w, p in zip(self.outcomes, self.prob):
            s += p
            if r <= s:
                return w
        if second_trial:
            return None
        if s > 0.999:
            return self.outcomes[-1]
        self.prob = [_ / s for _ in self.prob]
        return self._run(state, dest, True)

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        return self._run(state, dest)


@define
class NegotiatorOfferingPolicy(OfferingPolicy):
    """
    Uses a negotiator as an offering strategy
    """

    proposer: GBNegotiator = field(kw_only=True)

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        r = self.proposer.propose(state)
        if isinstance(r, ExtendedOutcome):
            return r.outcome
        return r


@define
class ConcensusOfferingPolicy(OfferingPolicy, ABC):
    """
    Offers based on concensus of multiple strategies
    """

    strategies: list[OfferingPolicy]

    def filter(self, indx: int, offer: Outcome | None) -> FilterResult:
        """
        Called with the decision of each strategy in order.


        Remarks:
            - Two decisions need to be made:

              1. Should we continue trying other strategies
              2. Should we save this result.
        """
        return FilterResult(True, True)

    @abstractmethod
    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        """
        Called to make a final decsision given the decisions of the stratgeis with indices `indices` (see `filter` for filtering rules)
        """

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        selected, selected_indices = [], []
        for i, s in enumerate(self.strategies):
            response = s.propose(state)
            r = self.filter(i, response)
            if not r.next:
                break
            if r.save:
                selected.append(response)
                selected_indices.append(i)

        return self.decide(selected_indices, selected)


@define
class UnanimousConcensusOfferingPolicy(ConcensusOfferingPolicy):
    """
    Offers only if all offering strategies gave exactly the same outcome
    """

    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        outcomes = set(responses)
        if len(outcomes) != 1:
            return None
        return list(outcomes)[0]


@define
class RandomConcensusOfferingPolicy(ConcensusOfferingPolicy):
    """
    Offers a random response from the list of strategies (different strategy every time).
    """

    prob: list[float] | None = None

    def __attrs_post_init__(self):
        if not self.prob:
            return
        s = sum(self.prob)
        self.prob = [_ / s for _ in self.prob]

    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        if not self.prob:
            return random.choice(responses)

        r, s = random.random(), 0.0
        for i, p in enumerate(self.prob):
            s += p
            if r <= s:
                return responses[i]
        if s > 0.999:
            return responses[-1]
        raise ValueError(f"sum of probabilities is less than 1: {s}")


@define
class UtilBasedConcensusOfferingPolicy(ConcensusOfferingPolicy, ABC):
    """
    Offers from the list of stratgies (different strategy every time) based on outcome utilities
    """

    @abstractmethod
    def decide_util(self, utils: list[Value]) -> int:
        """
        Returns the index to chose based on utils
        """

    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        if not self.negotiator.ufun:
            raise ValueError("Cannot decide because I have no ufun")
        return responses[
            self.decide_util([self.negotiator.ufun(_) for _ in set(responses)])
        ]


@define
class MyBestConcensusOfferingPolicy(UtilBasedConcensusOfferingPolicy):
    """
    Offers my best outcome from the list of stratgies (different strategy every time).
    """

    def decide_util(self, utils: list[Value]) -> int:
        return max(range(len(utils)), key=lambda x: utils[x])


@define
class MyWorstConcensusOfferingPolicy(UtilBasedConcensusOfferingPolicy):
    """
    Offers my worst outcome from the list of stratgies (different strategy every time) based on outcome utilities
    """

    def decide_util(self, utils: list[Value]) -> int:
        return min(range(len(utils)), key=lambda x: utils[x])
