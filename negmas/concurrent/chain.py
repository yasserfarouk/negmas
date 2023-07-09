"""Implements a concurrent set of negotiations creating a chain of bilateral negotiations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass

from ..common import NegotiatorMechanismInterface
from ..mechanisms import Mechanism, MechanismState, MechanismStepResult
from ..negotiators import Negotiator
from ..outcomes import Outcome
from ..preferences import Preferences
from ..sao import ResponseType

__all__ = [
    "ChainNegotiationsMechanism",
    "ChainNegotiator",
    "MultiChainNegotiationsMechanism",
    "MultiChainNegotiator",
]


@dataclass
class Offer:
    """Defines an offer"""

    outcome: Outcome
    left: bool
    temp: bool
    partner: str | None = None


Agreement = namedtuple("Agreement", ["outcome", "negotiators", "level"])
"""Defines an agreement for a multi-channel mechanism"""


class ChainAMI(NegotiatorMechanismInterface):
    def __init__(
        self,
        *args,
        parent: ChainNegotiationsMechanism,
        negotiator: ChainNegotiator,
        level: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__parent = parent
        self.__negotiator = negotiator
        self.__level = level

    def confirm(self, parent: bool) -> bool:
        return self.__parent.on_confirm(self.__level, parent)


class ChainNegotiator(Negotiator, ABC):
    """Base class for all nested negotiations negotiator"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nmi: ChainAMI
        self.__level = -1

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        role: str = "negotiator",
    ) -> bool:
        to_join = super().join(nmi, state, preferences=preferences, role=role)
        if to_join:
            self.__level = int(role)
        return to_join

    def confirm(self, left: bool) -> bool:
        """
        Called to confirm a offer to either the left or the right

        Args:
            left: If true confirm the offer sent to the left (otherwise the right)

        Returns:

        """
        return self._nmi.confirm(left)

    @abstractmethod
    def on_acceptance(self, state: MechanismState, offer: Offer) -> Offer:
        """
        Called when one of the negotiator's proposals gets accepted

        Args:
            state: mechanism state
            offer: The offer accepted

        Returns:
            A new offer (possibly to another negotiator)
        """

    @abstractmethod
    def propose(self, state: MechanismState) -> Offer:
        """
        Called to allow the agent to propose to either its left or its right in the chain

        Args:
            state: Mechanism state

        Returns:
            The offer
        """

    @abstractmethod
    def respond(
        self, state: MechanismState, outcome: Outcome, from_left: bool, temp: bool
    ) -> ResponseType:
        """
        Called to respond to an offer

        Args:
            state: Mechanism state
            outcome: The offer to respond to
            from_left: Whether the offer is coming from the left
            temp: Whether the offer is a temporary offer

        Returns:
            A response type which can only be ACCEPT_OFFER, REJECT_OFFER, or END_NEGOTIATION
        """


class MultiChainNegotiator(Negotiator, ABC):
    """Base class for all nested negotiations negotiator"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__level = -1

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        role: str = "negotiator",
    ) -> bool:
        to_join = super().join(nmi, state, preferences=preferences, role=role)
        if to_join:
            self.__level = int(role)
        return to_join

    def confirm(self, left: bool) -> bool:
        """
        Called to confirm a offer to either the left or the right

        Args:
            left: If true confirm the offer sent to the left (otherwise the right)

        Returns:

        """
        return self.nmi.confirm(left)  # type: ignore

    @abstractmethod
    def on_acceptance(self, state: MechanismState, offer: Offer) -> Offer:
        """
        Called when one of the negotiator's proposals gets accepted

        Args:
            state: mechanism state
            offer: The offer accepted

        Returns:
            A new offer (possibly to another negotiator)
        """

    @abstractmethod
    def propose(self, state: MechanismState) -> Offer:
        """
        Called to allow the agent to propose to either its left or its right in the chain

        Args:
            state: Mechanism state

        Returns:
            The offer
        """

    @abstractmethod
    def respond(
        self,
        state: MechanismState,
        outcome: Outcome,
        from_left: bool,
        temp: bool,
        source: str,
    ) -> ResponseType:
        """
        Called to respond to an offer

        Args:
            state: Mechanism state
            outcome: The offer to respond to
            from_left: Whether the offer is coming from the left
            temp: Whether the offer is a temporary offer
            source: The ID of the source agent

        Returns:
            A response type which can only be ACCEPT_OFFER, REJECT_OFFER, or END_NEGOTIATION
        """


class ChainNegotiationsMechanism(Mechanism):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__chain: list[ChainNegotiator | None] = []
        self.__next_agent = 0
        self.__last_proposal: Offer | None = None
        self.__last_proposer_index: int = -1
        self.__agreements: dict[int, Outcome | None] = defaultdict(lambda: None)
        self.__temp_agreements: dict[int, Outcome | None] = defaultdict(lambda: None)

    def _get_ami(
        self, negotiator: Negotiator, role: str
    ) -> NegotiatorMechanismInterface:
        """
        Returns a chain AMI instead of the standard AMI.

        Args:
            negotiator: The negotiator to create the AMI for
            role: Its role in the negotiation (an integer >= -1 as a string)

        Returns:

        """
        return ChainAMI(
            id=self.id,
            n_outcomes=self.nmi.n_outcomes,
            issues=self.nmi.outcome_space,
            outcomes=self.nmi.outcomes,
            time_limit=self.nmi.time_limit,
            step_time_limit=self.nmi.step_time_limit,
            n_steps=self.nmi.n_steps,
            dynamic_entry=self.nmi.dynamic_entry,
            max_n_agents=self.nmi.max_n_agents,
            annotation=self.nmi.annotation,
            parent=self,
            negotiator=negotiator,  # type: ignore
            level=int(role) + 1,
        )

    def add(
        self,
        negotiator: Negotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        **kwargs,
    ) -> bool | None:
        if role is None:
            raise ValueError(
                "You cannot join this protocol without specifying the role. "
                "Possible roles are integers >= -1 "
            )
        added = super().add(negotiator, preferences=preferences, role=role, **kwargs)
        if not added:
            return added
        level = int(role) + 1
        if len(self.__chain) > level:
            if self.__chain[level] is not None:
                raise ValueError(f"A negotiator already exist in the role {role}")
            self.__chain[level] = negotiator
            return
        self.__chain += [None] * (level - len(self.__chain) + 1)
        self.__chain[level] = negotiator

    def _update_next(self) -> None:
        if self.__last_proposal.left:
            self.__next_agent = (self.__last_proposer_index - 1) % len(self.__chain)
        else:
            self.__next_agent = (self.__last_proposer_index + 1) % len(self.__chain)

    def __call__(self, state, action=None) -> MechanismStepResult:
        # check that the chain is complete
        if not all(self.__chain):
            if self.dynamic_entry:
                state.has_error = True
                state.error_details = "The chain is not complete"
                return MechanismStepResult(state=state)
            raise ValueError(
                "The chain is not complete and dynamic entry is not allowed"
            )

        # find the next negotiator to ask
        negotiator = self.__chain[self.__next_agent]

        # if this is the first proposal, get it from the left most agent and ask the next one to respond
        if self.__last_proposal is None:
            self.__last_proposal = negotiator.propose(self.state)
            self._update_next()
            assert self.__next_agent == 1
            return MechanismStepResult(state)

        # if all agreements are finalized end the mechanism session with success
        agreements = [
            self.__agreements[l]
            for l in range(len(self.__chain))
            if self.__agreements[l] is not None
        ]

        if len(agreements) == len(self.__chain) - 1:
            state.agreement = agreements
            return MechanismStepResult(state)
        response = negotiator.respond(
            self.state,
            self.__last_proposal.outcome,
            not self.__last_proposal.left,
            self.__last_proposal.temp,
        )

        # handle unacceptable responses
        if response not in (
            ResponseType.ACCEPT_OFFER,
            ResponseType.REJECT_OFFER,
            ResponseType.END_NEGOTIATION,
        ):
            state.has_error = True
            state.error_details = "An unacceptable response was returned"
            return MechanismStepResult(state)

        # If the response is to end the negotiation, end it but only if there are not partial negotiations
        if response == ResponseType.END_NEGOTIATION:
            if len(self.__agreements) > 0:
                state.has_error = True
                state.error_details = (
                    "Cannot end a negotiation chain with some agreements"
                )
                return MechanismStepResult(state)
            state.broken = True
            return MechanismStepResult(state)

        # if the response is an acceptance then either register an agreement or a temporary agreement depending on
        # proposal
        if response == ResponseType.ACCEPT_OFFER:
            agreement_index = (
                self.__next_agent
                if self.__last_proposal.left
                else self.__next_agent - 1
            )
            if not self.__last_proposal.temp:
                assert agreement_index >= 0
                self.__agreements[agreement_index] = self.__last_proposal.outcome
            else:
                assert self.__temp_agreements[agreement_index] is None
                self.__temp_agreements[agreement_index] = self.__last_proposal.outcome
            self.__last_proposal = self.__chain[
                self.__last_proposer_index
            ].on_acceptance(self.state, self.__last_proposal)
            self.__last_proposer_index = self.__next_agent
            self._update_next()
            return MechanismStepResult(state)

        # now it must be a rejection, ask the one who rejected to propose (in either direction)
        self.__last_proposal = self.__chain[self.__next_agent].propose(self.state)
        self.__last_proposer_index = self.__next_agent
        self._update_next()
        return MechanismStepResult()

    def on_confirm(self, level: int, left: bool) -> bool:
        """
        Called by negotiators to confirm their temporary accepted agreements

        Args:
            level: The caller level
            left: Whether to confirm its left or right temporary accepted agreement
        """
        if left:
            level = (level - 1) % len(self.__chain)
        else:
            level = (level + 1) % len(self.__chain)
        if self.__temp_agreements[level] is None:
            raise ValueError(f"No temporary agreement exists at level {level}")
        if self.__agreements[level] is not None:
            raise ValueError(f"An agreement already exists at level {level}")
        self.__agreements[level] = self.__temp_agreements[level]
        return True


class MultiChainNegotiationsMechanism(Mechanism):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__chain: list[list[MultiChainNegotiator]] = []
        self.__next_agent_level = 0
        self.__next_agent_number = 0
        self.__last_proposal: Offer = None
        self.__last_proposer_level: int = -1
        self.__last_proposer_number: int = -1
        self.__agreements: dict[int, Agreement] = defaultdict(lambda: None)
        self.__temp_agreements: dict[int, Agreement] = defaultdict(lambda: None)
        self.__level: dict[str, int] = {}
        self.__number: dict[str, int] = {}

    def _get_ami(
        self, negotiator: Negotiator, role: str
    ) -> NegotiatorMechanismInterface:
        """
        Returns a chain AMI instead of the standard AMI.

        Args:
            negotiator: The negotiator to create the AMI for
            role: Its role in the negotiation (an integer >= -1 as a string)

        Returns:

        """
        return ChainAMI(
            id=self.id,
            n_outcomes=self.nmi.n_outcomes,
            issues=self.nmi.outcome_space,
            outcomes=self.nmi.outcomes,
            time_limit=self.nmi.time_limit,
            step_time_limit=self.nmi.step_time_limit,
            n_steps=self.nmi.n_steps,
            dynamic_entry=self.nmi.dynamic_entry,
            max_n_agents=self.nmi.max_n_agents,
            annotation=self.nmi.annotation,
            parent=self,
            negotiator=negotiator,
            level=int(role) + 1,
        )

    def add(
        self,
        negotiator: Negotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        **kwargs,
    ) -> bool | None:
        if role is None:
            raise ValueError(
                "You cannot join this protocol without specifying the role. "
                "Possible roles are integers >= -1 "
            )
        added = super().add(negotiator, preferences=preferences, role=role, **kwargs)
        if not added:
            return added
        level = int(role) + 1
        if len(self.__chain) > level:
            if self.__chain[level] is not None:
                raise ValueError(f"A negotiator already exist in the role {role}")
            self.__chain[level] = negotiator
            return
        self.__chain += [list() for _ in range(level - len(self.__chain) + 1)]
        self.__chain[level].append(negotiator)
        self.__level[negotiator.id] = level
        self.__number[negotiator.id] = len(self.__chain[level]) - 1

    def _update_next(self) -> None:
        if self.__last_proposal.left:
            self.__next_agent_level = (self.__last_proposer_level - 1) % len(
                self.__chain
            )
        else:
            self.__next_agent_level = (self.__last_proposer_level + 1) % len(
                self.__chain
            )
        self.__next_agent_number = self.__number[self.__last_proposal.partner]

    def __call__(self, state, action=None) -> MechanismStepResult:
        # check that the chain is complete
        if not all(len(_) > 0 for _ in self.__chain):
            if self.dynamic_entry:
                state.has_error = True
                state.error_details = "The chain is not complete"
                return MechanismStepResult(state)
            raise ValueError(
                "The chain is not complete and dynamic entry is not allowed"
            )

        # find the next negotiator to ask
        negotiator = self.__chain[self.__next_agent_level][self.__next_agent_number]

        # if this is the first proposal, get it from the left most agent and ask the next one to respond
        if self.__last_proposal is None:
            self.__last_proposal = negotiator.propose(self.state)
            self._update_next()
            assert self.__next_agent_level == 1
            return MechanismStepResult(state)

        # if all agreements are finalized end the mechanism session with success
        agreements = [
            self.__agreements[l]
            for l in range(len(self.__chain))
            if self.__agreements[l] is not None
        ]

        if len(agreements) == len(self.__chain) - 1:
            state.agreement = agreements
            return MechanismStepResult(state)
        response = negotiator.respond(
            self.state,
            self.__last_proposal.outcome,
            not self.__last_proposal.left,
            self.__last_proposal.temp,
            source=self.__chain[self.__next_agent_level][self.__next_agent_number].id,
        )

        # handle unacceptable responses
        if response not in (
            ResponseType.ACCEPT_OFFER,
            ResponseType.REJECT_OFFER,
            ResponseType.END_NEGOTIATION,
        ):
            state.has_error = True
            error_details = "An unacceptable response was returned"
            return MechanismStepResult(state)

        # If the response is to end the negotiation, end it but only if there are not partial negotiations
        if response == ResponseType.END_NEGOTIATION:
            if len(self.__agreements) > 0:
                state.has_error = True
                stae.error_details = (
                    "Cannot end a negotiation chain with some agreements",
                )
                return MechanismStepResult(stae)
            state.broken = True
            return MechanismStepResult(state)

        # if the response is an acceptance then either register an agreement or a temporary agreement depending on
        # proposal
        if response == ResponseType.ACCEPT_OFFER:
            agreement_index = (
                self.__next_agent_level
                if self.__last_proposal.left
                else self.__next_agent_level - 1
            )
            if not self.__last_proposal.temp:
                assert agreement_index >= 0
                self.__agreements[agreement_index] = self.__last_proposal.outcome
            else:
                assert self.__temp_agreements[agreement_index] is None
                self.__temp_agreements[agreement_index] = self.__last_proposal.outcome
            self.__last_proposal = self.__chain[self.__last_proposer_level][
                self.__last_proposer_number
            ].on_acceptance(self.state, self.__last_proposal)
            self.__last_proposer_level = self.__next_agent_level
            self.__last_proposer_number = self.__next_agent_number
            self._update_next()
            return MechanismStepResult(state)

        # now it must be a rejection, ask the one who rejected to propose (in either direction)
        self.__last_proposal = self.__chain[self.__next_agent_level][
            self.__next_agent_number
        ].propose(self.state)
        self.__last_proposer_level = self.__next_agent_level
        self.__last_proposer_number = self.__next_agent_number
        self._update_next()
        return MechanismStepResult(state)

    def on_confirm(self, level: int, left: bool) -> None:
        """
        Called by negotiators to confirm their temporary accepted agreements

        Args:
            level: The caller level
            left: Whether to confirm its left or right temporary accepted agreement
        """
        if left:
            level = (level - 1) % len(self.__chain)
        else:
            level = (level + 1) % len(self.__chain)
        if self.__temp_agreements[level] is None:
            raise ValueError(f"No temporary agreement exists at level {level}")
        if self.__agreements[level] is not None:
            raise ValueError(f"An agreement already exists at level {level}")
        self.__agreements[level] = self.__temp_agreements[level]
