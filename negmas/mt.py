"""Implements single text negotiation mechanisms"""
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .mechanisms import Mechanism, MechanismRoundResult, MechanismState

__all__ = [
    "VetoMTMechanism",
]


@dataclass
class MTState(MechanismState):
    """Defines extra values to keep in the mechanism state. This is accessible to all negotiators"""

    current_offers: List[Optional["Outcome"]] = field(default_factory=list)


class VetoMTMechanism(Mechanism):
    """Base class for all muti-text mechanisms

    Args:
        *args: positional arguments to be passed to the base Mechanism
        **kwargs: keyword arguments to be passed to the base Mechanism
        n_texts: Number of current offers to keep
        initial_outcomes: initial set of outcomes outcome. If None, it will be selected by `next_outcome` which by default will choose it
                         randomly.
        initial_responses: Initial set of responses. One list per outcome in initial_outcomes.

    Remarks:

        - initial_responses is only of value when the number of negotiators that will join the negotiation is less then
          or equal to its length. By default it is not used for anything. Nevertheless, it is here because
          `next_outcome` may decide to use it with the `initial_outcome`


    """

    def __init__(
        self,
        *args,
        epsilon: float = 1e-6,
        n_texts: int = 10,
        initial_outcomes=None,
        initial_responses: Tuple[Tuple[bool]] = None,
        **kwargs
    ):
        kwargs["state_factory"] = MTState
        super().__init__(*args, **kwargs)

        self.add_requirements(
            {"compare-binary": True}
        )  # assert that all agents must have compare-binary capability
        self.current_offers = (
            initial_outcomes if initial_outcomes is not None else [None] * n_texts
        )
        """The current offer"""
        self.initial_outcomes = deepcopy(self.current_offers.copy)
        """The initial offer"""
        self.last_responses = (
            [list(_) for _ in initial_responses]
            if initial_responses is not None
            else [None] * n_texts
        )
        """The responses of all negotiators for the last offer"""
        self.initial_responses = deepcopy(self.last_responses)
        """The initial set of responses. See the remarks of this class to understand its role."""
        self.epsilon = epsilon

    def extra_state(self):
        return dict(current_offers=deepcopy(self.current_offers),)

    def next_outcome(self, outcome: Optional["Outcome"]) -> Optional["Outcome"]:
        """Generate the next outcome given some outcome.

        Args:
             outcome: The current outcome

        Returns:
            a new outcome or None to end the mechanism run

        """
        return self.random_outcomes(1)[0]

    def round(self) -> MechanismRoundResult:
        """Single round of the protocol"""
        for i, current_offer in enumerate(self.current_offers):
            new_offer = self.next_outcome(current_offer)
            responses = []

            for neg in self.negotiators:
                strt = time.perf_counter()
                responses.append(
                    neg.is_better(new_offer, current_offer, epsilon=self.epsilon)
                    is not False
                )
                if time.perf_counter() - strt > self.ami.step_time_limit:
                    return MechanismRoundResult(
                        broken=False, timedout=True, agreement=None
                    )

            self.last_responses = responses

            if all(responses):
                self.current_offers[i] = new_offer

        return MechanismRoundResult(broken=False, timedout=False, agreement=None)

    def on_negotiation_end(self) -> None:
        """Used to pass the final offer for agreement between all negotiators"""
        self._agreement = self.current_offers

        super().on_negotiation_end()
