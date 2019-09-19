"""Implements single text negotiation mechanisms"""
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from negmas import MechanismRoundResult, Outcome, MechanismState
from negmas.mechanisms import Mechanism

__all__ = [
    "VetoSTMechanism",
]


@dataclass
class STState(MechanismState):
    """Defines extra values to keep in the mechanism state. This is accessible to all negotiators"""
    current_offer: Optional["Outcome"] = None


class VetoSTMechanism(Mechanism):
    """Base class for all single text mechanisms

    Args:
        *args: positional arguments to be passed to the base Mechanism
        **kwargs: keyword arguments to be passed to the base Mechanism
        initial_outcome: initial outcome. If None, it will be selected by `next_outcome` which by default will choose it
                         randomly.
        initial_responses: Initial set of responses.

    Remarks:

        - initial_responses is only of value when the number of negotiators that will join the negotiation is less then
          or equal to its length. By default it is not used for anything. Nevertheless, it is here because
          `next_outcome` may decide to use it with the `initial_outcome`


    """

    def __init__(self, *args, epsilon: float = 1e-6, initial_outcome=None, initial_responses: Tuple[bool] = tuple()
                 , **kwargs):
        kwargs["state_factory"] = STState
        super().__init__(*args, **kwargs)

        self.add_requirements({"compare-binary": True}) # assert that all agents must have compare-binary capability
        self.current_offer = initial_outcome
        """The current offer"""
        self.initial_outcome = initial_outcome
        """The initial offer"""
        self.last_responses = list(initial_responses)
        """The responses of all negotiators for the last offer"""
        self.initial_responses = self.last_responses
        """The initial set of responses. See the remarks of this class to understand its role."""
        self.epsilon = epsilon

    def extra_state(self):
        return STState(
            current_offer=self.current_offer,
        )

    def next_outcome(self, outcome: Optional[Outcome]) -> Optional[Outcome]:
        """Generate the next outcome given some outcome.

        Args:
             outcome: The current outcome

        Returns:
            a new outcome or None to end the mechanism run

        """
        return self.random_outcomes(1)[0]

    def round(self) -> MechanismRoundResult:
        """Single round of the protocol"""
        new_offer = self.next_outcome(self.current_offer)
        responses = []

        for neg in self.negotiators:
            strt = time.perf_counter()
            responses.append(neg.is_better(new_offer, self.current_offer, epsilon=self.epsilon) is not False)
            if time.perf_counter() - strt > self.ami.step_time_limit:
                return MechanismRoundResult(
                    broken=False, timedout=True, agreement=None
                )

        self.last_responses = responses

        if all(responses):
            self.current_offer = new_offer

        return MechanismRoundResult(broken=False, timedout=False, agreement=None)

    def on_negotiation_end(self) -> None:
        """Used to pass the final offer for agreement between all negotiators"""
        if self.current_offer is not None and all(neg.is_acceptable(self.current_offer) for neg in self.negotiators):
            self._agreement = self.current_offer

        super().on_negotiation_end()


