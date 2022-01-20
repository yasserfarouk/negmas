from __future__ import annotations

import math
import random
import warnings

from ...common import MechanismState
from ...negotiators.mixins import AspirationMixin
from ...outcomes import Outcome
from ..common import ResponseType
from .base import SAONegotiator

__all__ = [
    "AspirationNegotiator",
]


class AspirationNegotiator(SAONegotiator, AspirationMixin):
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
        assume_normalized: If True, the ufun will just be assumed to have the range [0, 1] inclusive
        ranking: If True, the aspiration level will not be based on the utility value but the ranking of the outcome
                 within the presorted list. It is only effective when presort is set to True
        presort: If True, the negotiator will catch a list of outcomes, presort them and only use them for offers
                 and responses. This is much faster then other option for general continuous utility functions
                 but with the obvious problem of only exploring a discrete subset of the issue space (Decided by
                 the `discrete_outcomes` property of the `NegotiatorMechanismInterface` . If the number of outcomes is
                 very large (i.e. > 10000) and discrete, presort will be forced to be True. You can check if
                 presorting is active in realtime by checking the "presorted" attribute.
        tolerance: A tolerance used for sampling of outcomes when `presort` is set to False
        rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                        reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
        owner: The `Agent` that owns the negotiator.
        parent: The parent which should be an `SAOController`

    """

    def __init__(
        self,
        max_aspiration=1.0,
        aspiration_type="boulware",
        stochastic=False,
        can_propose=True,
        assume_normalized=False,
        ranking_only=False,
        ufun_max=None,
        ufun_min=None,
        presort: bool = True,
        tolerance: float = 0.01,
        **kwargs,
    ):
        self.ufun_max = ufun_max
        self.ufun_min = ufun_min
        self.ranking = ranking_only
        self.tolerance = tolerance
        if assume_normalized:
            self.ufun_max, self.ufun_min = 1.0, 0.0
        super().__init__(
            assume_normalized=assume_normalized,
            **kwargs,
        )
        self.aspiration_init(
            max_aspiration=max_aspiration, aspiration_type=aspiration_type
        )
        self.randomize_offer = stochastic
        self.best_outcome, self.worst_outcome = None, None
        self.presort = presort
        self.presorted = False
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.__last_offer_util, self.__last_offer = float("inf"), None
        self.n_outcomes_to_force_presort = 10000
        self.n_trials = 1

    def on_preferences_changed(self):
        super().on_preferences_changed()
        if self.ufun is None or self._nmi is None:
            self.ufun_max = self.ufun_min = None
            return
        presort = self.presort
        if (
            not presort
            and self.nmi.outcome_space.is_discrete()
            and self.nmi.outcome_space.cardinality >= self.n_outcomes_to_force_presort
        ):
            presort = True
        if presort:
            outcomes = self._nmi.discrete_outcomes()
            uvals = [self.ufun.eval(_) for _ in outcomes]
            uvals_outcomes = [
                (u, o) for u, o in zip(uvals, outcomes) if u >= self.ufun.reserved_value
            ]
            self.ordered_outcomes = sorted(
                uvals_outcomes,
                key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
                reverse=True,
            )
            if self.assume_normalized:
                self.ufun_min, self.ufun_max = 0.0, 1.0
            elif len(self.ordered_outcomes) < 1:
                self.ufun_max = self.ufun_min = self.ufun.reserved_value
            else:
                if self.ufun_max is None:
                    self.ufun_max = self.ordered_outcomes[0][0]

                if self.ufun_min is None:
                    # we set the minimum utility to the minimum finite value above both reserved_value
                    for j in range(len(self.ordered_outcomes) - 1, -1, -1):
                        self.ufun_min = self.ordered_outcomes[j][0]
                        if self.ufun_min is not None and self.ufun_min > float("-inf"):
                            break
                    if (
                        self.ufun_min is not None
                        and self.ufun_min < self.reserved_value
                    ):
                        self.ufun_min = self.reserved_value
        else:
            if (
                self.ufun_min is None
                or self.ufun_max is None
                or self.best_outcome is None
                or self.worst_outcome is None
            ):
                self.worst_outcome, self.best_outcome = self.ufun.extreme_outcomes()
                mn, mx = self.ufun(self.worst_outcome), self.ufun(self.best_outcome)
                if self.ufun_min is None:
                    self.ufun_min = mn
                if self.ufun_max is None:
                    self.ufun_max = mx

        if self.ufun_min < self.reserved_value:
            self.ufun_min = self.reserved_value
        if self.ufun_max < self.ufun_min:  # type: ignore
            self.ufun_max = self.ufun_min

        self.presorted = presort
        self.n_trials = 10

    def respond(self, state: MechanismState, offer: Outcome) -> "ResponseType":
        if self.ufun_max is None or self.ufun_min is None:
            self.on_preferences_changed()
        if self._preferences is None or self.ufun_max is None or self.ufun_min is None:
            return ResponseType.REJECT_OFFER
        u = self.ufun(offer)
        if u is None or u < self.reserved_value:
            return ResponseType.REJECT_OFFER
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if u >= asp and u > self.reserved_value:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Outcome | None:
        if self.ufun_max is None or self.ufun_min is None:
            self.on_preferences_changed()
        if self._preferences is None or self.ufun_max is None or self.ufun_min is None:
            return None
        if self.ufun_max < self.reserved_value:
            return None
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if asp < self.reserved_value:
            return None
        if self.presorted:
            if len(self.ordered_outcomes) < 1:
                return None
            for i, (u, _) in enumerate(self.ordered_outcomes):
                if u is None:
                    continue
                if u < asp:
                    if u < self.reserved_value:
                        return None
                    if i == 0:
                        return self.ordered_outcomes[i][1]
                    if self.randomize_offer:
                        return random.sample(self.ordered_outcomes[:i], 1)[0][1]
                    return self.ordered_outcomes[i - 1][1]
            if self.randomize_offer:
                return random.sample(self.ordered_outcomes, 1)[0][1]
            return self.ordered_outcomes[-1][1]
        else:
            if asp >= 0.99999999999 and self.best_outcome is not None:
                return self.best_outcome
            if self.randomize_offer:
                return self.ufun.sample_outcome_with_utility(
                    (asp, float("inf")), outcome_space=self._nmi.outcome_space
                )
            tol = self.tolerance
            for _ in range(self.n_trials):
                rng = self.ufun_max - self.ufun_min
                mx = min(asp + tol * rng, self.__last_offer_util)
                outcome = self.ufun.sample_outcome_with_utility(
                    (asp, mx), outcome_space=self._nmi.outcome_space
                )
                if outcome is not None:
                    break
                tol = math.sqrt(tol)
            else:
                outcome = (
                    self.best_outcome
                    if self.__last_offer is None
                    else self.__last_offer
                )
            self.__last_offer_util = self.ufun(outcome)
            self.__last_offer = outcome
            return outcome
