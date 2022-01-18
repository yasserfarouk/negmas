from __future__ import annotations

import random
import warnings

import numpy as np

from ...common import MechanismState
from ...negotiators import Controller
from ...outcomes import Outcome
from ..common import ResponseType
from .base import SAONegotiator

__all__ = [
    "ToughNegotiator",
    "OnlyBestNegotiator",
]


class ToughNegotiator(SAONegotiator):
    """
    Accepts and proposes only the top offer (i.e. the one with highest utility).

    Args:
         name: Negotiator name
         parent: Parent controller if any
         dynamic_preferences: If `True`, assumes a dynamic ufun that can change during the negotiation
         can_propose: If `False` the negotiator will never propose but can only accept
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                            reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
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

    def on_preferences_changed(self):
        super().on_preferences_changed()
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
        return self.best_outcome


class OnlyBestNegotiator(SAONegotiator):
    """
    Offers and accepts only one of the top outcomes for the negotiator.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         dynamic_preferences: If `True`, assumes a dynamic ufun that can change during the negotiation
         can_propose: If `False` the negotiator will never propose but can only accept
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         min_utility: The minimum utility to offer or accept
         top_fraction: The fraction of the outcomes (ordered decreasingly by utility) to offer or accept
         best_first: Guarantee offering will non-increasing in terms of utility value
         probabilistic_offering: Offer randomly from the outcomes selected based on `top_fraction` and `min_utility`
         rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                            reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
         owner: The `Agent` that owns the negotiator.
    """

    def __init__(
        self,
        name=None,
        parent: Controller = None,
        dynamic_preferences=True,
        min_utility=0.95,
        top_fraction=0.05,
        best_first=False,
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
        super().__init__(
            name=name, parent=parent, ufun=ufun, preferences=preferences, **kwargs
        )
        if not dynamic_preferences:
            warnings.warn(
                "dynamic_preferences is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
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

    def on_preferences_changed(self):
        super().on_preferences_changed()
        outcomes = list(
            self._nmi.discrete_outcomes()
            if self._offerable_outcomes is None
            else self._offerable_outcomes
        )
        eu_outcome = list(zip([self.ufun.eval(_) for _ in outcomes], outcomes))
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
