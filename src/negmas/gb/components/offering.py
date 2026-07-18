"""Offering strategies and policies for negotiations."""

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
from negmas.preferences.inv_ufun import DefaultInverseUtilityFunction
from negmas.preferences.pareto_sampler import DefaultParetoSampler
from negmas.preferences.protocols import InverseUFun, ParetoSampler

from .base import FilterResult, OfferingPolicy
from .concession import ConcessionRecommender
from .models.ufun import UFunModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
    from negmas.gb import GBState
    from negmas.gb.negotiators.base import GBNegotiator
    from negmas.outcomes import Outcome
    from negmas.preferences.base_ufun import BaseUtilityFunction


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
    "NiceTitForTatOfferingPolicy",
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
    """HybridOffering policy implementation."""

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
        domain_size = ufun.outcome_space.cardinality  # type: ignore
        if domain_size > 100_000:
            domain_size = 100_000

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

        self._sent_utils = [float(ufun(_)) for _ in self._sent_offers]
        self._received_utils = [float(ufun(_)) for _ in self._received_offers]

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
        self._values = [float(_[0]) for _ in outcome_util]

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Recalculates parameters and outcome utilities when preferences change."""
        self._adjust_params()
        return super().on_preferences_changed(changes)

    def time_based(self, t: float) -> float:
        """Computes target utility using a quadratic Bezier curve over time.

        Args:
            t: Normalized negotiation time in [0, 1].

        Returns:
            The target utility value at time t.
        """
        return (
            (1 - t) * (1 - t) * self.initial_utility
            + 2 * (1 - t) * t * self.concession_ratio
            + t * t * self.final_utility
        )

    def behaviour_based(self, t: float) -> float:
        """Computes target utility based on opponent's concession patterns.

        Args:
            t: Normalized negotiation time in [0, 1].

        Returns:
            The target utility value adjusted by opponent behavior.
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
        """Generates an offer by combining time-based and behavior-based strategies.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.
        """
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
        n_outcomes = len(self._outcomes)
        if n_outcomes < 1:
            return None
        if indx < 0:
            indx = indx % n_outcomes
        else:
            indx = max(0, min(n_outcomes - 1, indx))

        try:
            outcome = self._outcomes[indx]
        except Exception as e:
            raise ValueError(f"{indx=}, {len(self._outcomes)=}\n{e=}")

        self._sent_offers.append(outcome)
        self._sent_utils.append(float(ufun(outcome)))

        return outcome

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Records partner's offer and its utility for behavior-based strategy."""
        ufun = self.negotiator.ufun
        assert ufun, "Unknown ufun. Cannot continue"
        self._received_offers.append(offer)
        self._received_utils.append(float(ufun(offer)))
        return super().on_partner_proposal(state, partner_id, offer)


@define
class TimeBasedOfferingPolicy(OfferingPolicy):
    """TimeBasedOffering policy implementation."""

    curve: PolyAspiration = field(factory=lambda: PolyAspiration(1.0, "boulware"))
    stochastic: bool = False
    sorter: InverseUFun | None = field(repr=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Initializes the outcome sorter when preferences are set or changed.

        Handles different change types appropriately:
        - Initialization/General: Initialize the sorter (warn if already initialized)
        - Scale/ReservedValue/ReservedOutcome: Ignored (don't affect outcome ordering)
        """
        if not self.negotiator or not self.negotiator.ufun:
            return

        # Skip changes that don't affect outcome ordering
        if all(
            _.type
            in (
                PreferencesChangeType.Scale,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            return

        # For Initialization or General changes, initialize the sorter
        if self.sorter is not None:
            warnings.warn(
                "Sorter is already initialized. May be on_preferences_changed is called twice!!"
            )
        self.sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self.sorter.init()

    def __call__(self, state: GBState, dest: str | None = None):
        """Generates an offer based on aspiration level at current time.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.
        """
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
    """MiCROOffering policy implementation."""

    next_indx: int = 0
    sorter: InverseUFun | None = field(repr=False, default=None)
    _received: set[Outcome] = field(factory=set)
    _sent: set[Outcome] = field(factory=set)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Reinitializes the sorter and resets offer tracking on significant preference changes."""
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
            self.sorter = DefaultInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
            self.next_indx = 0
            self._received = set()
            self._sent = set()

    def sample_sent(self) -> Outcome | ExtendedOutcome | None:
        """Returns a random outcome from previously sent offers, or None if empty."""
        if not len(self._sent):
            return None
        return random.choice(list(self._sent))

    def ensure_sorter(self):
        """Initializes the outcome sorter if not already initialized and returns it."""
        if not self.sorter:
            assert self.negotiator.ufun
            self.sorter = DefaultInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
        return self.sorter

    def next_offer(self) -> Outcome | ExtendedOutcome | None:
        """Returns the next outcome to offer based on current concession level."""
        return self.ensure_sorter().outcome_at(self.next_indx)

    def best_offer_so_far(self) -> Outcome | ExtendedOutcome | None:
        """Returns the highest-utility outcome offered so far, or None if none sent."""
        if self.next_indx > 0:
            return self.ensure_sorter().outcome_at(self.next_indx - 1)
        return None

    def ready_to_concede(self) -> bool:
        """Checks if we should concede based on offer exchange balance."""
        return len(self._sent) <= len(self._received)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generates an offer using MiCRO's tit-for-tat concession strategy.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.

        Returns:
            The next outcome to propose, or a previously sent offer if not ready to concede.
        """
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
        """Records the partner's offer to track concession balance."""
        self._received.add(offer)
        return super().on_partner_proposal(state, partner_id, offer)


@define
class FastMiCROOfferingPolicy(MiCROOfferingPolicy):
    """FastMiCROOffering policy implementation."""

    _skipped: set[Outcome] = field(factory=set)

    def ready_to_concede(self) -> bool:
        """Checks concession readiness, allowing faster concession near deadline."""
        return (
            len(self._sent) <= len(self._received)
            or self.negotiator.nmi.state.relative_time > 0.95
        )

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generates an offer with adaptive concession rate based on remaining time.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.

        Returns:
            The next outcome to propose, potentially skipping some for efficiency.
        """
        outcome = self.next_offer()
        assert self.sorter
        assert self.negotiator.ufun
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
            or not self.ready_to_concede()
        ):
            if self._skipped and random.random() < (
                len(self._skipped) / (len(self._skipped) + len(self._sent))
            ):
                o = random.choice(list(self._skipped))
                self._skipped.remove(o)
                self._sent.add(o)
                return o
            return self.sample_sent()
        t = state.relative_time
        n_sent = len(self._sent)
        if t < 0.1 or n_sent < 5:
            self.next_indx += 1
        else:
            n_remaining = len(self.sorter.outcomes) - 1 - self.next_indx
            t_per_offer = (t) / n_sent
            n_expected = int((1 - t) / t_per_offer + 0.5)
            if n_remaining <= n_expected:
                self.next_indx += 1
            else:
                n_skip = max(1, (n_expected - n_remaining))
                for i in range(n_skip - 1):
                    o = self.sorter.outcome_at(self.next_indx + 1)
                    if o is None:
                        break
                    self._skipped.add(o)
                    self.next_indx += 1
                else:
                    self.next_indx += 1

        self._sent.add(outcome)
        return outcome


@define
class CABOfferingPolicy(OfferingPolicy):
    """CABOffering policy implementation."""

    next_indx: int = 0
    sorter: InverseUFun | None = field(repr=False, default=None)
    _last_offer: Outcome | None = field(init=False, default=None)
    _repeating: bool = field(init=False, default=False)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Initializes the outcome sorter on significant preference changes."""
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
            self.sorter = DefaultInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
            self.next_indx = 0
            self._repeating = False

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generates an offer by conceding one outcome at a time from best to worst.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.

        Returns:
            The next outcome in descending utility order, or repeats last if exhausted.
        """
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
            self.sorter = DefaultInverseUtilityFunction(
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
    """WAROffering policy implementation."""

    next_indx: int = 0
    sorter: InverseUFun | None = field(repr=False, default=None)
    _last_offer: Outcome | None = field(init=False, default=None)
    _repeating: bool = field(init=False, default=False)
    _irrational: bool = field(init=False, default=True)
    _irrational_index: int = field(init=False, default=-1)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Initializes outcome sorter and irrational offer tracking on preference changes."""
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
            self.sorter = DefaultInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
            self.next_indx = 0
            self._repeating = False

    def on_negotiation_start(self, state) -> None:
        """Resets state to start with irrational (worst) offers."""
        self._repeating = False
        self._irrational = True
        self._irrational_index = self.negotiator.nmi.n_outcomes - 1  # type: ignore
        return super().on_negotiation_start(state)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generates offers starting from worst, then conceding from best to worst.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.

        Returns:
            An irrational offer initially, then rational offers in descending utility.
        """
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
            self.sorter = DefaultInverseUtilityFunction(
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
        """Stores the partner's latest offer for concession calculation."""
        self._partner_offer = offer

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Propagates preference changes to the partner utility model."""
        super().on_preferences_changed(changes)
        self.partner_ufun.on_preferences_changed(changes)

    def __call__(self, state: GBState, dest: str | None = None):
        """Generates an offer matching or exceeding the partner's concession level.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.
        """
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
class NiceTitForTatOfferingPolicy(OfferingPolicy):
    """
    The bidding (offering) strategy of the Nice Tit for Tat agent
    (Baarslag, Hindriks & Jonker, 2013, "A Tit for Tat Negotiation Strategy
    for Real-Time Bilateral Negotiations").

    The strategy reciprocates in the agent's *own* utility (the opponent's
    utility is unknown and any model of it is unreliable, so we measure the
    opponent's concession as the change in *our* utility of the opponent's
    successive offers) and aims for a bargaining solution point of the
    scenario rather than naively mirroring the opponent's raw concession
    (naive mirroring settles for ~0.5 utility and misses win-win deals).
    By default the target is the Nash bargaining point (as in the paper);
    other solution concepts can be selected with ``target`` (see below):

    This follows the algorithm of the reference Genius implementation
    (Baarslag's ``NiceTitForTat``), working entirely in the agent's *normalized*
    ``[0, 1]`` utility:

    1. **Cooperate first.** Initially the agent offers its best outcome
       (it never defects first).
    2. **Estimate ``my_nash``.** Using the opponent model
       (``negotiator.opponent_model``) and the calculators in
       `negmas.preferences.ops`, estimate the chosen bargaining solution and take
       the agent's own utility ``p_me`` at it. Scale it by a multiplier that
       depends on how far the opponent started from the agent
       (``1.4 - 0.6 * initial_gap``) and floor it: for the Nash target at
       ``nash_min`` (``0.5`` by default, as in the reference); for other targets
       only at the reserved value.
    3. **Reciprocate the opponent's observed concession.** Measure the
       opponent's concession *in the agent's own utility* as
       ``max_offered_to_me - first_offered_to_me`` (how far the best bid it has
       offered us has moved from its starting bid — directly observed, not
       model-estimated) and express it as a fraction of the way to ``my_nash``.
       The agent concedes the same fraction of its own gap from ``1`` down to
       ``my_nash``: ``target = 1 - factor * (1 - my_nash)``. Then a **concession
       bonus** (a small constant baseline from the discount factor plus a ramp
       near the deadline) pulls the target the rest of the way toward ``my_nash``
       — this is what lets a mirror match concede and agree instead of
       deadlocking at maximum utility.
    4. **Make the offer attractive to the opponent.** Among the outcomes in the
       agent's acceptable utility band ``[target, 1]``, pick the one the
       opponent model rates highest — the trade-off query
       ``argmax_{u_me >= target} u_opp``, answered by a `ParetoSampler`
       (``pareto_sampler_type``) built on the normalized agent ufun with the
       opponent model as the opponent ufun (via ``ufun.make_pareto_sampler``,
       cached and re-initialized as the same model instance learns). If the
       sampler cannot answer (no result, or its additivity requirements are not
       met), the policy falls back to inverting the normalized ufun over the
       band. Finally, if the best bid the opponent has already offered us is
       worth at least as much to us as our planned bid, we offer *that* bid
       instead (the reference's ``makeAppropriate``), so consensus forms as soon
       as the opponent's standing offer beats our plan.

    The opponent model is accessed through ``self.negotiator.opponent_model``
    (a `UFunModel`), which may be provided to the negotiator or left to a
    default (see `NiceTitForTatNegotiator`).

    *AI Generated (implementation of the Baarslag 2013 Nice Tit for Tat bidding
    strategy for negmas).*

    Args:
        sample_size: Maximum number of outcomes to sample from the utility band
            when selecting the opponent-attractive offer.
        max_cardinality: Maximum number of outcomes to enumerate/sample when
            estimating the bargaining point.
        nash_refresh: Re-estimate the bargaining point every this many rounds
            (1 = every round, reflecting an opponent model that keeps
            learning).
        stochastic: If ``True`` and no opponent model is available, pick a
            random in-band outcome instead of the worst-in-band one.
        target: The bargaining solution to aim for. One of ``"nash"`` (default,
            Nash bargaining), ``"kalai"`` (Kalai egalitarian),
            ``"kalai_smorodinsky"`` / ``"ks"`` (Kalai-Smorodinsky),
            ``"max_welfare"`` (utilitarian / max sum of utilities), or
            ``"max_relative_welfare"`` (max sum of relative gains). Selected via
            the corresponding calculator in `negmas.preferences.ops`.
        nash_min: Lower clamp on the estimated ``my_nash`` target utility, applied
            **only** for the ``"nash"`` target (the reference agent never asks for
            less than ``0.5``). For other solution concepts the target is floored
            at the reserved value instead, so a legitimately lower ``p_me`` is not
            inflated.
        pareto_sampler_type: The `ParetoSampler` implementation used for step iv
            (the opponent-attractive trade-off query). Defaults to
            `DefaultParetoSampler` (``AdaptiveParetoSampler``), which uses the
            exact `BruteForceParetoSampler` on small outcome spaces and a
            scalable backend on large ones. Pass a specific sampler type (e.g.
            `BruteForceParetoSampler`, or `IPSParetoSampler` for very large
            additive domains) to override. The sampler is queried each round via
            ``best_for_opponent`` with the opponent model as the opponent ufun;
            if it cannot answer the query the policy falls back to the inverter
            path.

    Remarks:
        - Requires the negotiator to expose ``opponent_model`` (a `UFunModel`
          or ``None``). When it is ``None`` the policy degrades to naive
          tit-for-tat.
    """

    sample_size: int = 100
    max_cardinality: int = 10000
    nash_refresh: int = 1
    stochastic: bool = False
    target: str = "nash"
    levels: int = 20
    nash_min: float = 0.5
    pareto_sampler_type: type[ParetoSampler] = DefaultParetoSampler
    _partner_offer: Outcome | None = field(init=False, default=None)
    _prev_partner_offer: Outcome | None = field(init=False, default=None)
    _target_util: float | None = field(init=False, default=None)
    _point_cache: tuple | None = field(init=False, default=None)
    _point_cache_step: int = field(init=False, default=-1)
    _norm_ufun: BaseUtilityFunction | None = field(init=False, default=None)
    _norm_inverter: InverseUFun | None = field(init=False, default=None)
    _sampler_failed: bool = field(init=False, default=False)
    # opponent-offer history, measured in the agent's own normalized utility
    # (mirrors the Java reference: reciprocation is driven by the utility the
    # opponent's *observed* offers give us, not by the opponent model).
    _opp_first_util: float | None = field(init=False, default=None)
    _opp_max_util: float | None = field(init=False, default=None)
    _opp_best_bid: Outcome | None = field(init=False, default=None)

    def before_responding(
        self, state: GBState, offer: Outcome | None, source: str | None = None
    ):
        """Remember the opponent's offers and update the offer-history stats.

        Tracks, in the agent's own normalized utility, the utility of the
        opponent's *first* bid (the reference point for its concession) and the
        running maximum utility it has offered us (a ratchet), plus the bid that
        achieved it (used to avoid overshooting — see ``makeAppropriate`` in the
        reference).
        """
        self._prev_partner_offer = self._partner_offer
        self._partner_offer = offer
        if offer is None or not self.negotiator or not self.negotiator.ufun:
            return
        na = self._normalized_ufun(self.negotiator.ufun)
        base = na if na is not None else self.negotiator.ufun
        try:
            u = float(base.eval_normalized(offer))
        except Exception:  # pragma: no cover - defensive
            return
        if not math.isfinite(u):
            return
        if self._opp_first_util is None:
            self._opp_first_util = u
        if self._opp_max_util is None or u > self._opp_max_util:
            self._opp_max_util = u
            self._opp_best_bid = offer

    def _normalized_ufun(self, ufun) -> BaseUtilityFunction | None:
        """A cached, normalized (``[0, 1]``) copy of the agent's ufun.

        The opponent model already returns values in ``[0, 1]``; normalizing the
        agent's ufun puts both on the same footing for `pareto_frontier` and the
        bargaining-solution calculators, and (via its inverter) for the utility
        band in step iv. ``normalize_reserved_values`` keeps the reserved value
        finite so the calculators can use it.
        """
        if self._norm_ufun is None:
            try:
                self._norm_ufun = ufun.normalize(normalize_reserved_values=True)
            except Exception:  # pragma: no cover - defensive
                return None
        return self._norm_ufun

    def _sampler(self, na, opponent_model):
        """The cached `ParetoSampler` (own = ``na``, opponent = model), or ``None``.

        Built once (and cached on ``na``) via ``na.make_pareto_sampler``. The
        *same* instance is shared by `_target_point` (which reads its frontier to
        find the bargaining point) and step iv (which answers the trade-off
        query), so the Pareto frontier is built once per ``nash_refresh`` cycle
        instead of twice per round.

        ``max_cardinality`` is forwarded only to samplers whose constructor
        accepts it (e.g. `BruteForceParetoSampler`), so the frontier honours the
        same enumeration cap `pareto_frontier` uses. Any construction error
        (e.g. an additive-only sampler on a non-additive ufun) disables the
        sampler path for the rest of the negotiation.
        """
        if self._sampler_failed or opponent_model is None or na is None:
            return None
        extra: dict = {}
        try:
            import inspect

            if (
                "max_cardinality"
                in inspect.signature(self.pareto_sampler_type).parameters
            ):
                extra["max_cardinality"] = self.max_cardinality
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
        try:
            return na.make_pareto_sampler(
                opponent_ufun=opponent_model,
                pareto_sampler=self.pareto_sampler_type,
                **extra,
            )
        except Exception:
            self._sampler_failed = True
            return None

    def _target_point(self, ufun, opponent_model) -> tuple | None:
        """Estimate ``(P_me, P_opp, best_opp)`` for the chosen bargaining solution.

        Sources the Pareto frontier from the shared `ParetoSampler` (see
        `_sampler`) — re-initialized here with the current (learning) opponent
        model — and runs the bargaining-solution calculators (``nash_points``,
        ``kalai_points``, ``ks_points``, ``max_welfare_points``,
        ``max_relative_welfare_points``) from `negmas.preferences.ops` over it.
        The calculators stay in `ops` (single source of truth); the sampler only
        supplies the frontier. Step iv reuses the *same* initialized sampler, so
        the frontier is computed once per round rather than twice.

        The sampler's frontier is reserve-filtered (rational for *both* ufuns)
        before the calculators see it. This matches
        `pareto_frontier`'s rational frontier exactly — a rational Pareto point
        can only be dominated by another rational point — so the target point is
        identical to the pre-sampler behaviour on finite spaces. The *real*
        ufuns (the normalized agent ufun and the opponent model) are passed to
        the calculator so each ufun's ``reserved_value`` is used.

        If the chosen sampler cannot handle these ufuns (e.g. `IPSParetoSampler`
        on a non-additive learned model), this falls back to
        `pareto_frontier` (exact enumeration) so the target point is still found.

        Returns ``None`` if no estimate can be made (no outcome space, no
        opponent model, normalization failure, or the model is
        degenerate/uninformative).
        """
        if opponent_model is None:
            return None
        os_ = ufun.outcome_space
        if os_ is None:
            return None
        issues = getattr(os_, "issues", None)
        if issues is None:
            return None
        na = self._normalized_ufun(ufun)
        if na is None:
            return None

        def _reserve(u) -> float:
            try:
                r = float(u.reserved_value)
            except Exception:  # pragma: no cover - defensive
                return float("-inf")
            return r if math.isfinite(r) else float("-inf")

        frontier: list[tuple[float, float]] | None = None
        sampler = self._sampler(na, opponent_model)
        if sampler is not None:
            try:
                sampler.init(
                    opponent_ufun=opponent_model
                )  # refresh as the model learns
                outcomes = sampler.pareto_outcomes()
            except Exception:  # pragma: no cover - defensive
                self._sampler_failed = True
                outcomes = []
            if outcomes:
                r_me, r_opp = _reserve(na), _reserve(opponent_model)
                pts: list[tuple[float, float]] = []
                for o in outcomes:
                    um = float(na(o))
                    uo = float(opponent_model(o))
                    if (
                        math.isfinite(um)
                        and math.isfinite(uo)
                        and um >= r_me
                        and uo >= r_opp
                    ):
                        pts.append((um, uo))
                frontier = pts

        if not frontier:
            # Fallback: exact enumeration always works, so the target point is
            # still found even when the chosen sampler cannot handle these ufuns.
            try:
                from negmas.preferences.ops import pareto_frontier

                fr, _idx = pareto_frontier(
                    [na, opponent_model],
                    issues=issues,
                    n_discretization=self.levels,
                    max_cardinality=self.max_cardinality,
                )
                frontier = [(float(a), float(b)) for a, b in fr]
            except Exception:  # pragma: no cover - defensive
                return None
        if not frontier:
            return None
        best_opp = max(uo for _, uo in frontier)
        ums = [p[0] for p in frontier]
        uos = [p[1] for p in frontier]
        ranges = [(min(ums), max(ums)), (min(uos), max(uos))]
        calculators = {
            "nash": "nash_points",
            "kalai": "kalai_points",
            "kalai_smorodinsky": "ks_points",
            "ks": "ks_points",
            "max_welfare": "max_welfare_points",
            "max_relative_welfare": "max_relative_welfare_points",
        }
        name = calculators.get(self.target, "nash_points")
        try:
            from negmas.preferences import ops as _ops

            fn = getattr(_ops, name)
            # Pass the REAL ufuns (no proxies): the calculator reads
            # ``reserved_value`` from each, so the opponent model's estimate of
            # the opponent's disagreement utility is used (and the agent's
            # normalized reserve), instead of assuming 0 for the opponent.
            results = fn(
                ufuns=[na, opponent_model],
                frontier=frontier,
                ranges=ranges,
                outcome_space=os_,
            )
        except Exception:  # pragma: no cover - defensive
            return None
        if not results:
            return None
        util_tuple, _idx = results[0]
        p_me, p_opp = float(util_tuple[0]), float(util_tuple[1])
        return p_me, p_opp, best_opp

    def _best_via_sampler(
        self, na: BaseUtilityFunction, opponent_model, target: float
    ) -> Outcome | None:
        """``argmax_{u_me >= target} u_opp`` via the shared `ParetoSampler`.

        Reuses the *same* sampler instance `_target_point` already initialized
        this round (see `_sampler`) and answers the trade-off query with
        ``best_for_opponent(min_util=target, normalized=True)``. Crucially it
        does **not** pass ``opponent_ufun`` here — doing so would re-run
        ``init`` and rebuild the frontier a second time; instead it reuses the
        frontier built for the target-point search, so the frontier is computed
        once per round. ``normalized=True`` matches the ``[0, 1]`` scale
        ``target`` lives on (``na.minmax() == (0, 1)``).

        Returns the opponent-attractive in-band outcome, or ``None`` if the
        sampler is unavailable or cannot answer (no feasible outcome), in which
        case the caller falls back to the inverter path.
        """
        sampler = self._sampler(na, opponent_model)
        if sampler is None:
            return None
        try:
            return sampler.best_for_opponent(min_util=target, normalized=True)
        except Exception:
            self._sampler_failed = True
            return None

    @staticmethod
    def _nash_multiplier(gap: float) -> float:
        """Reference multiplier applied to the Nash utility estimate.

        A large ``gap`` (opponent started far from us) shrinks the multiplier
        (ask for a bit less); a small gap grows it (ask for a bit more). Matches
        the Baarslag reference ``1.4 - 0.6 * gap`` (floored at 0).
        """
        return max(0.0, 1.4 - 0.6 * gap)

    def _bonus(self, relative_time: float) -> float:
        """Concession *bonus* toward the Nash utility (Baarslag reference).

        Returns a value in ``[0, 1]`` multiplying the remaining gap to Nash:

        * a constant **discount baseline** ``0.5 - 0.4 * discount_factor`` (``0.1``
          with no discounting) — a small immediate concession that produces the
          initial cooperative gesture, and
        * a **time ramp** near the deadline (from ``t = 0.91``, or ``0.85`` on
          big domains, rising ``0 -> 1`` by ``t ~ 0.96``).

        The bonus is what lets a mirror match concede toward Nash and agree
        rather than deadlocking at maximum utility.
        """
        ufun = self.negotiator.ufun if self.negotiator else None
        discount = 1.0
        raw = getattr(ufun, "discount", None)
        try:
            if raw is not None:
                discount = float(raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            discount = 1.0
        if not (0.0 < discount <= 1.0):
            discount = 1.0
        discount_bonus = 0.5 - 0.4 * discount
        big = False
        try:
            os_ = ufun.outcome_space if ufun is not None else None
            big = os_ is not None and float(os_.cardinality) > 3000
        except Exception:  # pragma: no cover - defensive
            big = False
        min_time = 0.85 if big else 0.91
        time_bonus = 0.0
        if relative_time > min_time:
            time_bonus = min(1.0, 20.0 * (relative_time - min_time))
        return min(1.0, max(0.0, max(discount_bonus, time_bonus)))

    def __call__(self, state: GBState, dest: str | None = None):
        """Generate the next Nice Tit for Tat offer.

        Args:
            state: The current negotiation state.
            dest: Optional destination partner identifier.

        Returns:
            An outcome in the agent's acceptable utility band, chosen to be as
            attractive as possible to the opponent.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return None
        ufun = self.negotiator.ufun
        opponent_model = getattr(self.negotiator, "opponent_model", None)
        na = self._normalized_ufun(ufun)
        if na is None:
            return None

        # Refresh the bargaining point (``p_me`` = the agent's normalized utility
        # at the chosen solution) on the ``nash_refresh`` cadence.
        if (
            state.step - self._point_cache_step >= self.nash_refresh
            or self._point_cache is None
        ):
            self._point_cache = self._target_point(ufun, opponent_model)
            self._point_cache_step = state.step
        point = self._point_cache

        # ``my_nash`` = the agent's target utility at the solution, scaled by the
        # reference nash-multiplier (depends on how far the opponent started from
        # us) and floored. For the Nash target we reproduce the reference's
        # ``[nash_min, 1]`` clamp; for other solution concepts we only floor at
        # the (normalized) reserved value, so a legitimately lower ``p_me`` (e.g.
        # ``max_welfare``) is not inflated to 0.5.
        reserved = 0.0
        try:
            r = float(na.reserved_value)
            reserved = r if math.isfinite(r) else 0.0
        except Exception:  # pragma: no cover - defensive
            reserved = 0.0
        first = self._opp_first_util if self._opp_first_util is not None else 1.0
        max_off = self._opp_max_util if self._opp_max_util is not None else first
        initial_gap = 1.0 - first
        base_nash = point[0] if point is not None else 0.7
        my_nash = base_nash * self._nash_multiplier(initial_gap)
        low = self.nash_min if self.target == "nash" else reserved
        my_nash = min(1.0, max(low, reserved, my_nash))

        # Reciprocate the opponent's *observed* concession (in our own utility):
        # how far the best bid it has offered us has moved from its starting bid,
        # as a fraction of the way to ``my_nash``. We concede the same fraction
        # of our own gap from the maximum (1.0) down to ``my_nash``.
        denom = my_nash - first
        if denom > 1e-9:
            factor = min(1.0, max(0.0, (max_off - first) / denom))
        else:
            factor = 0.0
        my_concession = factor * (1.0 - my_nash)
        target = 1.0 - my_concession

        # Concession bonus toward Nash (small constant baseline + late-time ramp):
        # this breaks the mutual-maximum deadlock and drives agreement.
        gap_to_nash = max(0.0, target - my_nash)
        target -= self._bonus(state.relative_time) * gap_to_nash

        # keep within [reserved, 1].
        target = min(max(target, reserved), 1.0)
        self._target_util = target

        # step iv: opponent-attractive offer within the utility band [target, 1]:
        # the trade-off query ``argmax_{u_me >= target} u_opp``.

        # Preferred path: a `ParetoSampler` built on the normalized agent ufun
        # with the opponent model as the opponent ufun answers the trade-off
        # query directly. ``make_pareto_sampler`` caches the sampler on ``na``
        # and initializes it once; the opponent model is the same instance, so
        # passing it as ``opponent_ufun`` refreshes the frontier as the model
        # learns while reusing the cached sampler object. ``normalized=True``
        # makes ``min_util=target`` match the normalized [0, 1] scale ``target``
        # was computed on (``na.minmax() == (0, 1)``).
        bid: Outcome | None = None
        if opponent_model is not None and not self._sampler_failed:
            bid = self._best_via_sampler(na, opponent_model, target)

        # fallback: invert the normalized ufun over the band (same [0, 1] scale
        # as the frontier) and pick the opponent-best in band. Reached when there
        # is no opponent model, the sampler returned nothing, or the sampler
        # cannot handle these ufuns (e.g. an additive-only sampler such as
        # `IPSParetoSampler` on a non-additive learned model).
        if bid is None:
            if self._norm_inverter is None:
                try:
                    self._norm_inverter = na.invert()
                except Exception:  # pragma: no cover - defensive
                    self._norm_inverter = None
            inv = (
                self._norm_inverter
                if self._norm_inverter is not None
                else ufun.invert()
            )
            band = (target, 1.0)
            candidates = inv.some(band, normalized=True, n=self.sample_size)
            if candidates:
                if opponent_model is not None:
                    bid = max(
                        candidates,
                        key=lambda o: float(opponent_model.eval_normalized(o)),
                    )
                elif self.stochastic:
                    bid = inv.one_in(band, normalized=True)
                else:
                    # no model: standard TFT concession (worst-in-band for us)
                    bid = min(candidates, key=lambda o: float(ufun.eval_normalized(o)))
            else:
                # clamping inverter worst_in never returns None on a non-empty
                # space; if it does, refuse to propose.
                bid = inv.worst_in(band, normalized=True)

        # ``makeAppropriate`` (reference): never offer a bid worth less to us than
        # the best bid the opponent has already offered — offer that bid instead.
        # This prevents overshooting and lets consensus form as soon as the
        # opponent's standing offer beats our planned one.
        if bid is not None and self._opp_best_bid is not None:
            try:
                u_bid = float(na.eval_normalized(bid))
                u_best = float(na.eval_normalized(self._opp_best_bid))
                if math.isfinite(u_best) and math.isfinite(u_bid) and u_best >= u_bid:
                    return self._opp_best_bid
            except Exception:  # pragma: no cover - defensive
                pass
        return bid


@define
class OfferBest(OfferingPolicy):
    """
    Offers Only the best outcome.

    Remarks:
        - You can pass the  best outcome if you know it as `best` otherwise it will find it.
    """

    _best: Outcome | None = None

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Finds and caches the best outcome when preferences change."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        _, self._best = self.negotiator.ufun.extreme_outcomes()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Returns the best outcome according to the negotiator's preferences."""
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
        """Computes the set of top outcomes based on fraction and k constraints."""
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

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Returns a random outcome from the top outcomes set."""
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

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Always returns None, preventing any agreement."""
        return None


@define
class RandomOfferingPolicy(OfferingPolicy):
    """
    Offers random outcomes from the negotiation outcome space.
    """

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Returns a uniformly random outcome from the outcome space."""
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
    ) -> Outcome | ExtendedOutcome | None:
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

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Samples an outcome from the predefined list with optional probability weights."""
        return self._run(state, dest)


@define
class NegotiatorOfferingPolicy(OfferingPolicy):
    """
    Uses a negotiator as an offering strategy
    """

    proposer: GBNegotiator = field(kw_only=True)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Delegates offer generation to the wrapped proposer negotiator."""
        r = self.proposer.propose(state)
        if isinstance(r, ExtendedOutcome):
            return (
                r.best_for(self.negotiator.ufun)
                if self.negotiator and self.negotiator.ufun
                else r.outcome
            )
        return r


@define
class ConcensusOfferingPolicy(OfferingPolicy, ABC):
    """
    Offers based on concensus of multiple strategies
    """

    strategies: list[OfferingPolicy]

    def filter(
        self, indx: int, offer: Outcome | ExtendedOutcome | None
    ) -> FilterResult:
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
        self, indices: list[int], responses: list[Outcome | ExtendedOutcome | None]
    ) -> Outcome | ExtendedOutcome | None:
        """
        Called to make a final decsision given the decisions of the stratgeis with indices `indices` (see `filter` for filtering rules)
        """

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Collects filtered offers from all strategies and decides on the final offer."""
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
        self, indices: list[int], responses: list[Outcome | ExtendedOutcome | None]
    ) -> Outcome | ExtendedOutcome | None:
        """Returns the outcome only if all strategies agree, otherwise None.

        Args:
            indices: Indices of strategies that passed the filter.
            responses: Outcomes proposed by the filtered strategies.

        Returns:
            The unanimous outcome, or None if strategies disagree.
        """
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
        """Normalizes probability weights to sum to 1."""
        if not self.prob:
            return
        s = sum(self.prob)
        self.prob = [_ / s for _ in self.prob]

    def decide(
        self, indices: list[int], responses: list[Outcome | ExtendedOutcome | None]
    ) -> Outcome | ExtendedOutcome | None:
        """Randomly selects an outcome from responses using probability weights.

        Args:
            indices: Indices of strategies that passed the filter.
            responses: Outcomes proposed by the filtered strategies.

        Returns:
            A randomly selected outcome based on probability distribution.
        """
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
        self, indices: list[int], responses: list[Outcome | ExtendedOutcome | None]
    ) -> Outcome | ExtendedOutcome | None:
        """Selects an outcome based on utility values using the decide_util method.

        Args:
            indices: Indices of strategies that passed the filter.
            responses: Outcomes proposed by the filtered strategies.

        Returns:
            The outcome selected by the utility-based decision rule.
        """
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
        """Returns the index of the outcome with the highest utility.

        Args:
            utils: List of utility values for each candidate outcome.

        Returns:
            Index of the maximum utility outcome.
        """
        return max(range(len(utils)), key=lambda x: utils[x])


@define
class MyWorstConcensusOfferingPolicy(UtilBasedConcensusOfferingPolicy):
    """
    Offers my worst outcome from the list of stratgies (different strategy every time) based on outcome utilities
    """

    def decide_util(self, utils: list[Value]) -> int:
        """Returns the index of the outcome with the lowest utility.

        Args:
            utils: List of utility values for each candidate outcome.

        Returns:
            Index of the minimum utility outcome.
        """
        return min(range(len(utils)), key=lambda x: utils[x])
