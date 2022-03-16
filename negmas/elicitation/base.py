from __future__ import annotations

import copy
import time
from abc import abstractmethod
from typing import Callable

import numpy as np

from ..common import MechanismState, NegotiatorMechanismInterface, Value
from ..helpers.prob import ScipyDistribution
from ..models.acceptance import AdaptiveDiscreteAcceptanceModel, DiscreteAcceptanceModel
from ..outcomes import Outcome
from ..preferences import IPUtilityFunction, MappingUtilityFunction, Preferences
from ..sao import AspirationNegotiator, ResponseType, SAONegotiator
from .common import _locs, _uppers
from .expectors import Expector, MeanExpector
from .strategy import EStrategy
from .user import User

__all__ = ["BaseElicitor"]


class BaseElicitor(SAONegotiator):
    def accuracy_limit(self, cost: float) -> float:
        """The accuracy limit given the cost and `epsilon`."""
        return 0.5 * max(self.epsilon, cost)

    def __init__(
        self,
        user: User,
        *,
        strategy: EStrategy | None = None,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        opponent_model_factory: None
        | (
            Callable[[NegotiatorMechanismInterface], DiscreteAcceptanceModel]
        ) = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(nmi=x),
        expector_factory: Expector | Callable[[], Expector] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        true_utility_on_zero_cost=False,
    ) -> None:
        """
        ABC for all elicitation algorithms.

        Args:
            user: A `User` object that
            strategy: An elicitation strategy that determines the order of deep
                      elicitation queries.
            base_negotiator: A negotiator that is used to propose and rspond to
                      proposals based on the current state of the utility function.
            opponent_model_factory: A callable that can be used to create an opponent
                                    model.
            expector_factory: A callable that can be used to construct an `Expector`
                              object responsible of reducing a probabilistic utility
                              value into a real number to be used by the negotiator.
            single_elicitation_per_round: Forces a single elicitation opportunity per
                                          negotiation round. If the elicitor uses deep
                                          elicitation this will correspond to multiple
                                          calls to the `strategy`.
            continue_eliciting_past_reserved_val: If `True`, elicitation continues even
                                                  if the expector returns a value under
                                                  the reserved value of the negotiator.
            epsilon: A small number to stop elicitation when the uncertainty in the ufun
                     is under.
            true_utility_on_zero_cost: If `True`, zero cost will force the final elicited
                                       value of any outcome to exactly match the utility
                                       function. If `False`, the final utility value after
                                       elicitation may be within `epsilon` from the true
                                       value.
        """
        super().__init__()
        self.add_capabilities(
            {
                "propose": True,
                "respond": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.strategy = strategy
        self.opponent_model_factory = opponent_model_factory
        self.expector_factory = expector_factory
        self.single_elicitation = single_elicitation_per_round
        self.continue_eliciting_past_reserved_val = continue_eliciting_past_reserved_val
        self.epsilon = epsilon
        self.true_utility_on_zero_cost = true_utility_on_zero_cost
        self.elicitation_history = []
        self.opponent_model = None
        self._elicitation_time = None
        self.asking_time = 0.0
        self.offerable_outcomes = (
            []
        )  # will contain outcomes with known or at least elicited utilities
        self.indices = None
        self.initial_utility_priors = None
        self.user = user
        self.acc_limit = self.accuracy_limit(self.user.cost_of_asking())
        self.base_negotiator = base_negotiator
        self.expect = None
        if strategy is not None:
            strategy.resolution = max(self.acc_limit, strategy.resolution)

    def init_elicitation(
        self,
        preferences: None
        | (IPUtilityFunction | ScipyDistribution | list[ScipyDistribution]),
        **kwargs,
    ) -> None:
        """
        Called once to initialize the elicitation process

        Args:
            preferences: The probabilistic utility function
            **kwargs:

        Remarks:
            - If no `ufun` is given one will be created with 0-1 uniform distributions and
              zero reserved value.
            - If a single `Distribution` is given as `ufun`, it is repeated for all
              outcomes (and the reserved value is set to zero).
            - If a list of `Distribution` s is given, it must have the same length as
              the list of outcomes of this negotiation and is used to set the `ufun`.
            - The opponent model

        """
        nmi = self._nmi
        if nmi is None:
            raise ValueError(f"Unkown NMI")
        self.elicitation_history = []
        outcomes = list(nmi.discrete_outcomes())
        self.indices = dict(zip(outcomes, range(len(outcomes))))
        self.offerable_outcomes = []
        self._elicitation_time = 0.0
        if self.opponent_model_factory is None:
            self.opponent_model = None
        else:
            self.opponent_model = self.opponent_model_factory(nmi)
            self.base_negotiator.opponent_model = self.opponent_model
        if preferences is None:
            dists = [
                ScipyDistribution(type="uniform", loc=0.0, scale=1.0) for _ in outcomes
            ]
            preferences = IPUtilityFunction(
                outcomes=outcomes, distributions=dists, reserved_value=0.0
            )
        elif isinstance(preferences, ScipyDistribution):
            preferences = [copy.copy(preferences) for _ in outcomes]
            preferences = IPUtilityFunction(
                outcomes=outcomes, distributions=preferences, reserved_value=0.0
            )
        elif (
            isinstance(preferences, list)
            and len(preferences) > 0
            and isinstance(preferences[0], ScipyDistribution)
        ):
            preferences = IPUtilityFunction(
                outcomes=outcomes, distributions=preferences, reserved_value=0.0
            )
        self.set_preferences(preferences)
        self.initial_utility_priors = copy.copy(preferences)

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        role: str = "negotiator",
        **kwargs,
    ) -> bool:
        """
        Called to join a negotiation.

        Remarks:
            - uses the base_negotiator to join the negotiation.
            - creates a `MappingUtilityFunction` that maps every outcome to the
              result of the expector applied to the corresponding utility value.
            - The reserved value of the created ufun is set to -inf
        """
        if preferences is None:
            preferences = IPUtilityFunction(outcomes=nmi.outcomes, reserved_value=0.0)
        if not super().join(nmi=nmi, state=state, preferences=preferences, role=role):
            return False
        self.expect = self.expector_factory(self._nmi)
        self.init_elicitation(preferences=preferences, **kwargs)
        self.base_negotiator.join(
            nmi,
            state,
            preferences=MappingUtilityFunction(
                mapping=lambda x: self.expect(self.preferences(x), state=state),
                reserved_value=float("-inf"),
            ),
        )
        return True

    def on_negotiation_start(self, state: MechanismState):
        """Called when the negotiation starts. Just passes the call to
        base_negotiator."""
        self.base_negotiator.on_negotiation_start(state=state)

    def utility_distributions(self) -> list[ScipyDistribution]:
        """
        Returns a `Distribution` for every outcome
        """
        if self.preferences is None:
            return [None] * len(self._nmi.outcomes)
        if self.preferences.base_type == "ip":
            return list(self.preferences.distributions.values())
        else:
            return [self.preferences(o) for o in self._nmi.outcomes]

    def user_preferences(self, outcome: Outcome | None) -> float:
        """
        Finds the total utility obtained by the user for this outcome after
        discounting elicitation cost.

        Args:
            outcome: The outcome to find the user utility for. If None, it
                     returns the reserved value.

        Remarks:
            The total elicitation cost is *not* discounted from the reserved
            value when the input is None
        """
        return (
            self.user.ufun(outcome) - self.user.total_cost
            if outcome is not None
            else self.user.ufun(outcome)
        )

    @property
    def elicitation_cost(self) -> float:
        """
        The total elicitation cost.
        """
        return self.user.total_cost

    @property
    def elicitation_time(self) -> float:
        """The total elicitation time in seconds."""
        return self._elicitation_time

    def maximum_attainable_utility(self) -> float:
        """
        Maximum utility that could even in principle be attained which
        simply means the utility value of the outcome with maximum utility.
        """
        return max(_uppers(self.utility_distributions()))

    def minimum_guaranteed_utility(self):
        """
        Minimum utility that could even in principle be attained which
        simply means the utility value of the outcome with minimum utility.
        """
        return min(_locs(self.utility_distributions()))

    def on_opponent_model_updated(
        self, outcomes: list[Outcome], old: list[float], new: list[float]
    ) -> None:
        """
        Called whenever an opponents model is updated.

        Args:
            outcomes: A list of outcomes for which the acceptance probability are changed
            old: The old acceptance probability
            new: The new acceptance probability
        """

    def on_partner_proposal(
        self, state: MechanismState, partner_id: str, offer: Outcome
    ):
        """
        Called when one of the partners propose (only if enable_callbacks is set
        in the `SAOMechanism`).

        Args:
            state: mechanism state
            partner_id: the partner who proposed
            offer: The offer from the partner

        Remarks:
            - Used to update the opponent model by calling `update_offered` then
              `on_opponent_model_updated`.
        """
        self.base_negotiator.on_partner_proposal(
            partner_id=partner_id, offer=offer, state=state
        )
        old_prob = self.opponent_model.probability_of_acceptance(offer)
        self.opponent_model.update_offered(offer)
        new_prob = self.opponent_model.probability_of_acceptance(offer)
        self.on_opponent_model_updated([offer], old=[old_prob], new=[new_prob])

    def on_partner_response(
        self,
        state: MechanismState,
        partner_id: str,
        outcome: Outcome,
        response: ResponseType,
    ):
        """
        Called when one of the partners respond  (only if enable_callbacks is set
        in the `SAOMechanism`).

        Args:
            state: mechanism state
            partner_id: the partner who offered
            outcome: The outcome responded to
            response: The partner response including both the response and outcome

        Remarks:
            - Used to update the opponent model by calling `update_rejected` or
              `update_accepted1 then `on_opponent_model_updated`.
        """
        self.base_negotiator.on_partner_response(
            state=state, partner_id=partner_id, outcome=outcome, response=response
        )
        if response == ResponseType.REJECT_OFFER:
            old_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.opponent_model.update_rejected(outcome)
            new_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.on_opponent_model_updated([outcome], old=old_probs, new=new_probs)
        elif response == ResponseType.ACCEPT_OFFER:
            old_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.opponent_model.update_accepted(outcome)
            new_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.on_opponent_model_updated([outcome], old=old_probs, new=new_probs)

    def before_eliciting(self) -> None:
        """Called by apply just before continuously calling elicit_single"""

    @abstractmethod
    def elicit_single(self, state: MechanismState) -> None:
        """Does a single elicitation act

        Args:
            state: mechanism state
        """
        raise NotImplementedError()

    def elicit(self, state: MechanismState) -> None:
        """
        Called to do utility elicitation whenever needed.

        Args:
            state: mechanism state

        Remarks:
            - Keeps track of elicitation time and asking time.
            - If the maximum attainable utility minus elicitation cost is less than the
              reserved value, no elicitation will take place because we will end this
              negotiation anyway. Note that the maximum attainable utility can **never**
              go up.
            - Calls `before_eliciting` once to initialize the process then calls
              `elicit_single` which does the actual elicitation. This is done only once
              if `single_elicitation` is set, otherwise it is repeated until one of the
              following conditiosn is met:

                - `elicit_single` returns False
                - The maximum attainable utility (minus elicitation cost) is less than
                  the reserved value.
        """
        if (
            self.maximum_attainable_utility() - self.elicitation_cost
            <= self.reserved_value
        ):
            return
        start = time.perf_counter()
        self.before_eliciting()
        if self.single_elicitation:
            self.elicit_single(state=state)
        else:
            while self.elicit_single(state=state):
                if (
                    self.maximum_attainable_utility() - self.elicitation_cost
                    <= self.reserved_value
                    or state.relative_time >= 1
                ):
                    break
        elapsed = time.perf_counter() - start
        self._elicitation_time += elapsed
        self.asking_time += elapsed

    def utility_on_rejection(self, outcome: Outcome, state: MechanismState) -> Value:
        """Estimated utility if this outcome rejected at this state.

        Args:
            outcome: The outcome tested
            state: The mechanism state

        Remarks:
            - MUST be implemented by any Elicitor.
        """
        raise NotImplementedError(
            f"Must override utility_on_rejection in {self.__class__.__name__}"
        )

    def offering_utility(self, outcome, state) -> Value:
        """
        returns expected utlity of offering `outcome` in `state`

        Args:
            outcome: The outcome
            state: The state

        Returns:
            A utility value

        Remarks:
            - returns $u(o) p(o) + ru(o) (1-p(o))$ where $p$ is the opponent model,
              $u$ is the utility function, and $r$ is the utility in case of rejections.
            - `state` is needed when calculating $r(o)$ by calling `utility_on_rejection`.
            - Note that if $u$ or $r$ return a `Distribution`, this method will return
              a `Distribution` not a real number.
        """
        if self.opponent_model is None:
            return self.preferences(outcome)
        u = self.preferences(outcome)
        p = self.opponent_model.probability_of_acceptance(outcome)
        return p * u + (1 - p) * self.utility_on_rejection(outcome, state=state)

    def best_offer(self, state: MechanismState) -> tuple[Outcome | None, float]:
        """The outcome with maximum expected utility given the expector and its utility

        Args:
            state: The mechanism state

        Returns:
            A tuple containing the best outcome (or None) and its expected utility using the
            expector (or reserved value)

        Remarks:
            - if there are no offerable outcomes, elicitation is done and if still there are no
              offerable outcomes, the reserved value is returned (with None as outcome)
            - Only offerable outcomes are checked.
            - The best outcome is defined as the one with maximum `expect` applied to
              `offering_utility`.
        """
        if len(self.offerable_outcomes) == 0:
            self.elicit(state=state)
        if len(self.offerable_outcomes) == 0:
            return None, self.reserved_value
        best, best_utility, bsf = None, self.reserved_value, self.reserved_value
        for outcome in self.offerable_outcomes:
            if outcome is None:
                continue
            utilitiy = self.offering_utility(outcome, state=state)
            expected_utility = self.expect(utilitiy, state=state)
            if expected_utility >= bsf:
                best, best_utility, bsf = outcome, utilitiy, expected_utility
        return best, self.expect(best_utility, state=state)

    def respond_(self, state: MechanismState, offer: Outcome) -> ResponseType:
        """
        Called by the mechanism directly (through `counter` ) to respond to offers.

        Args:
            state: mechanism state
            offer: the offer to respond to

        Remarks:
            - Does the following steps:

                1. Finds the the best offer using `best_offer` and uses the base negotiator
                   to respond if that offer was `None`
                2. Looks at `offerable_outcomes` and applies the elicitation strategy (one
                   step) to the outcome if it was not offerable (or if there are no offerable
                   outcomes defined).
                3. Finds the utility of the offer using `utility_function` not taking into accout
                   elicitation cost and uses the base negotiator if that fails (i.e. `utility_function`
                   returns `None`).
                4. Finds the expected utility of the offer using the `expect` () method which calls the
                   expector passed during construction.
                5. If the maximum attainable utility now (judging from the current estimation of
                   the utility value of each outcome taking elicitation cost into account) is less
                   than the reserved value, end the negotiation
                6. If the utility of my best offer (returned from `best_offer`) is less than the offered
                   utility, accept the offer
                7. Otherwise, call bhe base negotiator to respond.

        """
        my_offer, meu = self.best_offer(state=state)
        if my_offer is None:
            return self.base_negotiator.respond_(state=state, offer=offer)
        if (
            self.strategy
            and self.offerable_outcomes is not None
            and offer not in self.offerable_outcomes
        ):
            self.strategy.apply(user=self.user, outcome=offer)
        offered_utility = self.preferences(offer)
        if offered_utility is None:
            return self.base_negotiator.respond_(state=state, offer=offer)
        offered_utility = self.expect(offered_utility, state=state)
        if (
            self.maximum_attainable_utility() - self.user.total_cost
            < self.reserved_value
        ):
            return ResponseType.END_NEGOTIATION
        if meu < offered_utility:
            return ResponseType.ACCEPT_OFFER
        else:
            return self.base_negotiator.respond_(state=state, offer=offer)

    @abstractmethod
    def can_elicit(self) -> bool:
        """Returns whether we can do more elicitation"""
        raise NotImplementedError()

    def propose(self, state: MechanismState) -> Outcome:
        """
        Called to propose an outcome

        Args:
            state: mechanism state

        Remarks:

            - if the negotiator `can_elicit`, it will `elicit`.
            - always then calls the base negotiator to propose.
        """
        if self.can_elicit():
            self.elicit(state=state)
        return self.base_negotiator.propose(state=state)

    def offering_utilities(self, state) -> np.ndarray:
        """
        Calculates the offering utility for all outcomes

        Args:
            state: Calculates the state at which the offering utilities are to be calculated

        Returns:
            An ndarray with the offering utility of every outcome (in order)

        Remarks:
            - This is just a faster version of calling `offering_utility` in a loop.

        """
        us = np.asarray(self.utility_distributions())
        ps = np.asarray(self.opponent_model.acceptance_probabilities())
        return ps * us + (1 - ps) * np.asarray(self.preferences(state=state))

    def utility_on_acceptance(self, outcome: Outcome) -> Value:
        """
        The utility of acceptance which is simply the utility function applied to `outcome`.
        """
        return self.preferences(outcome)

    def utilities_on_rejection(self, state: MechanismState) -> list[Value]:
        """Finds the utility of rejection for all outputs.

        Remarks:
            - By default it calls `utility_on_rejection` repeatedly for all outcomes.
              Override this method if a faster versin can be implemented
        """
        return [
            self.utility_on_rejection(outcome=outcome, state=state)
            for outcome in self._nmi.outcomes
        ]

    def __getattr__(self, item):
        return getattr(self.base_negotiator, item)
        # TODO extend this to take the partner_id as a parameter to handle multiparty negotiation

    def __str__(self):
        return f"{self.name}"
