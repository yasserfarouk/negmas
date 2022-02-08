from __future__ import annotations

import random

import numpy as np

from ...common import MechanismState, PreferencesChange
from ...negotiators import Controller
from ...outcomes import Outcome
from ..common import ResponseType
from .base import SAONegotiator

__all__ = [
    "ToughNegotiator",
    "TopFractionNegotiator",
    "BestOutcomeOnlyNegotiator",
]


class ToughNegotiator(SAONegotiator):
    """
    Accepts and proposes only the top offer (i.e. the one with highest utility).

    Args:
         name: Negotiator name
         parent: Parent controller if any
         can_propose: If `False` the negotiator will never propose but can only accept
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         owner: The `Agent` that owns the negotiator.

    Remarks:
        - If there are multiple outcome with the same maximum utility, only one of them will be used.

    """

    def __init__(
        self,
        name=None,
        parent: Controller = None,
        can_propose=True,
        **kwargs,
    ):
        super().__init__(name=name, parent=parent, **kwargs)
        self.best_outcome = None
        self._offerable_outcomes = None
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        super().on_preferences_changed(changes)
        if self.ufun is None:
            return
        _, self.best_outcome = self.ufun.extreme_outcomes()

    def respond(self, state: MechanismState, offer: Outcome) -> "ResponseType":
        if offer == self.best_outcome:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Outcome | None:
        if not self._capabilities["propose"]:
            return None


class BestOutcomeOnlyNegotiator(SAONegotiator):
    """
    Offers and accepts only its absolute best outcome(s)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_outcome = None
        self.best_util = float("-inf")
        self.__end_negotiation = False

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        if self.ufun is None:
            return
        _, self.best_outcome = self.ufun.extreme_outcomes()
        self.best_util = float(self.ufun(self.best_outcome))
        if (
            self.ufun.reserved_value is not None
            and self.best_util < self.reserved_value
        ):
            self.__end_negotiation = True

    def propose(self, state: MechanismState) -> Outcome | None:
        return self.best_outcome

    def respond(self, state: MechanismState, offer: Outcome) -> ResponseType:
        if self.__end_negotiation:
            return ResponseType.END_NEGOTIATION
        if offer == self.best_outcome or (
            self.ufun is not None and float(self.ufun(offer)) >= self.best_util
        ):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class TopFractionNegotiator(SAONegotiator):
    """
    Offers and accepts only one of the top outcomes for the negotiator.

    Args:
        name: Negotiator name
        parent: Parent controller if any
        can_propose: If `False` the negotiator will never propose but can only accept
        preferences: The preferences of the negotiator
        ufun: The ufun of the negotiator (overrides preferences)
        min_utility: The minimum utility to offer or accept
        top_fraction: The fraction of the outcomes (ordered decreasingly by utility) to offer or accept
        best_first: Guarantee offering will non-increasing in terms of utility value
        probabilistic_offering: Offer randomly from the outcomes selected based on `top_fraction` and `min_utility`
        owner: The `Agent` that owns the negotiator.
    """

    def __init__(
        self,
        name=None,
        parent: Controller = None,
        min_utility=0.95,
        top_fraction=0.05,
        best_first=True,
        probabilistic_offering=True,
        can_propose=True,
        preferences=None,
        ufun=None,
        **kwargs,
    ):
        self._offerable_outcomes = None
        self.best_outcome = []
        self.ordered_outcomes = []
        self.acceptable_outcomes = []
        self.wheel = np.array([])
        self.offered = set()
        self.top_fraction = top_fraction
        self.min_utility = min_utility
        self.best_first = best_first
        self.probabilistic_offering = probabilistic_offering
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        super().__init__(
            name=name, parent=parent, ufun=ufun, preferences=preferences, **kwargs
        )

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        super().on_preferences_changed(changes)
        if not self.ufun:
            self.acceptable_outcomes, self.wheel = [], []
            return
        if self._offerable_outcomes is not None:
            outcomes = self._offerable_outcomes
        elif self.nmi is not None:
            outcomes = list(self.nmi.discrete_outcomes())
        elif self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=1000)
            )
        else:
            outcomes = []
        eu_outcome = list(zip([self.ufun(_) for _ in outcomes], outcomes))
        self.ordered_outcomes = sorted(eu_outcome, key=lambda x: x[0], reverse=True)
        if self.min_utility is None:
            selected, selected_utils = [], []
        else:
            util_limit = self.min_utility * self.ordered_outcomes[0][0]
            selected, selected_utils = [], []
            for u, o in self.ordered_outcomes:
                if u >= util_limit:
                    selected.append(o)
                    selected_utils.append(u)
                else:
                    break
        if self.top_fraction is not None:
            frac_limit = max(1, round(self.top_fraction * len(self.ordered_outcomes)))
        else:
            frac_limit = len(outcomes)

        if frac_limit >= len(selected) > 0:
            sum = np.asarray(selected_utils).sum()
            if sum > 0.0:
                selected_utils /= sum
                selected_utils = np.cumsum(selected_utils)
            else:
                selected_utils = np.linspace(0.0, 1.0, len(selected_utils))
            self.acceptable_outcomes, self.wheel = selected, selected_utils
            return
        if frac_limit > 0:
            n_sel = len(selected)
            fsel = [_[1] for _ in self.ordered_outcomes[n_sel:frac_limit]]
            futil = [_[0] for _ in self.ordered_outcomes[n_sel:frac_limit]]
            selected_utils = selected_utils + futil
            sum = np.asarray(selected_utils).sum()
            if sum > 0.0:
                selected_utils /= sum
                selected_utils = np.cumsum(selected_utils)
            else:
                selected_utils = np.linspace(0.0, 1.0, len(selected_utils))
            self.acceptable_outcomes, self.wheel = selected + fsel, selected_utils
            return
        self.acceptable_outcomes, self.wheel = [], []
        return

    def respond(self, state: MechanismState, offer: Outcome) -> "ResponseType":
        if offer in self.acceptable_outcomes:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Outcome | None:
        if not self._capabilities["propose"]:
            return None
        if self.best_first:
            for o in self.acceptable_outcomes:
                if o not in self.offered:
                    self.offered.add(o)
                    return o
        if len(self.acceptable_outcomes) > 0:
            if self.probabilistic_offering:
                r = random.random()
                for o, w in zip(self.acceptable_outcomes, self.wheel):
                    if w > r:
                        return o
                return random.sample(self.acceptable_outcomes, 1)[0]
            return random.sample(self.acceptable_outcomes, 1)[0]
        return None
