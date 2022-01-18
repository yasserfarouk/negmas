from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ...common import MechanismState
from ...negotiators import Controller
from ...outcomes import Outcome
from ..common import ResponseType
from .base import SAONegotiator

if TYPE_CHECKING:
    from negmas.preferences import Preferences, UtilityFunction


__all__ = [
    "NaiveTitForTatNegotiator",
    "SimpleTitForTatNegotiator",
]


class NaiveTitForTatNegotiator(SAONegotiator):
    """
    Implements a naive tit-for-tat strategy that does not depend on the availability of an opponent model.

    Args:
        name: Negotiator name
        preferences: negotiator preferences
        ufun: negotiator ufun (overrides preferences)
        parent: A controller
        kindness: How 'kind' is the agent. A value of zero is standard tit-for-tat. Positive values makes the negotiator
                  concede faster and negative values slower.
        randomize_offer: If `True`, the offers will be randomized above the level determined by the current concession
                        which in turn reflects the opponent's concession.
        always_concede: If `True` the agent will never use a negative concession rate
        initial_concession: How much should the agent concede in the beginning in terms of utility. Should be a number
                            or the special string value 'min' for minimum concession

    Remarks:
        - This negotiator does not keep an opponent model. It thinks only in terms of changes in its own utility.
          If the opponent's last offer was better for the negotiator compared with the one before it, it considers
          that the opponent has conceded by the difference. This means that it implicitly assumes a zero-sum
          situation.
    """

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        preferences: "Preferences" | None = None,
        ufun: "UtilityFunction" | None = None,
        kindness=0.0,
        randomize_offer=False,
        always_concede=True,
        initial_concession: float | str = "min",
        **kwargs,
    ):
        self.received_utilities = []
        self.proposed_utility = None
        self.ordered_outcomes: list[tuple[float, Outcome]] = []
        self.sent_offer_index = None
        self.n_sent = 0
        super().__init__(
            name=name, ufun=ufun, preferences=preferences, parent=parent, **kwargs
        )
        self.kindness = kindness
        self.initial_concession = (
            initial_concession if isinstance(initial_concession, float) else -1.0
        )
        self.randomize_offer = randomize_offer
        self.always_concede = always_concede

    def on_preferences_changed(self):
        super().on_preferences_changed()
        outcomes = self._nmi.discrete_outcomes()
        self.ordered_outcomes = sorted(
            ((self.ufun(outcome), outcome) for outcome in outcomes),
            key=lambda x: x[0],
            reverse=True,
        )

    def respond(self, state: MechanismState, offer: Outcome) -> "ResponseType":
        if self.ufun is None:
            return ResponseType.REJECT_OFFER
        offered_utility = self.ufun(offer)
        if len(self.received_utilities) < 2:
            self.received_utilities.append(offered_utility)
        else:
            self.received_utilities[0] = self.received_utilities[1]
            self.received_utilities[-1] = offered_utility
        indx = self._propose(state=state)
        my_utility, _ = self.ordered_outcomes[indx]
        if offered_utility >= my_utility:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _outcome_just_below(self, ulevel: float) -> int:
        for i, (u, _) in enumerate(self.ordered_outcomes):
            if u is None:
                continue
            if u < ulevel:
                if self.randomize_offer:
                    return random.randint(0, i)
                return i
        if self.randomize_offer:
            return random.randint(0, len(self.ordered_outcomes) - 1)
        return -1

    def _propose(self, state: MechanismState) -> int:
        if self.proposed_utility is None:
            return 0
        if len(self.received_utilities) < 2:
            if isinstance(self.initial_concession, str) and self.initial_concession < 0:
                return self._outcome_just_below(ulevel=self.ordered_outcomes[0][0])
            else:
                asp = self.ordered_outcomes[0][0] * (1.0 - self.initial_concession)
            return self._outcome_just_below(ulevel=asp)

        if self.always_concede:
            opponent_concession = max(
                0.0, self.received_utilities[1] - self.received_utilities[0]
            )
        else:
            opponent_concession = (
                self.received_utilities[1] - self.received_utilities[0]
            )
        indx = self._outcome_just_below(
            ulevel=self.proposed_utility
            - opponent_concession
            - self.kindness * max(0.0, opponent_concession)
        )
        return indx

    def propose(self, state: MechanismState) -> Outcome | None:
        indx = self._propose(state)
        self.proposed_utility = self.ordered_outcomes[indx][0]
        return self.ordered_outcomes[indx][1]


SimpleTitForTatNegotiator = NaiveTitForTatNegotiator
"""A simple tit-for-tat negotiator"""
