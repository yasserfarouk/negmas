"""Implements a concurrent set of negotiations creating a chain of bilateral negotiations."""
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..common import AgentMechanismInterface
from ..mechanisms import Mechanism, MechanismRoundResult, MechanismState
from ..negotiators import Negotiator
from ..outcomes import Outcome, ResponseType
from ..utilities import UtilityFunction

__all__ = [
    "ChainNegotiationsMechanism",
    "ChainNegotiator",
    "MultiChainNegotiationsMechanism",
    "MultiChainNegotiator",
]


@dataclass
class Offer:
    """Defines an offer"""

    outcome: "Outcome"
    left: bool
    temp: bool
    partner: str = None


Agreement = namedtuple("Agreement", ["outcome", "negotiators", "level"])
"""Defines an agreement for a multi-channel mechanism"""


class ChainAMI(AgentMechanismInterface):
    def __init__(
        self,
        *args,
        parent: "ChainNegotiationsMechanism",
        negotiator: "ChainNegotiator",
        level: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__parent = parent
        self.__negotiator = negotiator
        self.__level = level

    def confirm(self, parent: bool) -> None:
        return self.__parent.on_confirm(self.__level, parent)


class ChainNegotiator(Negotiator, ABC):
    """Base class for all nested negotiations negotiator"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__level = -1

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        to_join = super().join(ami, state, ufun=ufun, role=role)
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
        self._ami.confirm(left)

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
        self, state: MechanismState, outcome: "Outcome", from_left: bool, temp: bool
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
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        to_join = super().join(ami, state, ufun=ufun, role=role)
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
        self._ami.confirm(left)

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
        outcome: "Outcome",
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
        self.__chain: List[Optional[ChainNegotiator]] = []
        self.__next_agent = 0
        self.__last_proposal: Offer = None
        self.__last_proposer_index: int = -1
        self.__agreements: Dict[int, Outcome] = defaultdict(lambda: None)
        self.__temp_agreements: Dict[int, Outcome] = defaultdict(lambda: None)

    def _get_ami(self, negotiator: Negotiator, role: str) -> AgentMechanismInterface:
        """
        Returns a chain AMI instead of the standard AMI.

        Args:
            negotiator: The negotiator to create the AMI for
            role: Its role in the negotiation (an integer >= -1 as a string)

        Returns:

        """
        return ChainAMI(
            id=self.id,
            n_outcomes=self.ami.n_outcomes,
            issues=self.ami.issues,
            outcomes=self.ami.outcomes,
            time_limit=self.ami.time_limit,
            step_time_limit=self.ami.step_time_limit,
            n_steps=self.ami.n_steps,
            dynamic_entry=self.ami.dynamic_entry,
            max_n_agents=self.ami.max_n_agents,
            annotation=self.ami.annotation,
            parent=self,
            negotiator=negotiator,
            level=int(role) + 1,
        )

    def add(
        self,
        negotiator: "Negotiator",
        *,
        ufun: Optional[UtilityFunction] = None,
        role: Optional[str] = None,
        **kwargs,
    ) -> Optional[bool]:
        if role is None:
            raise ValueError(
                "You cannot join this protocol without specifying the role. "
                "Possible roles are integers >= -1 "
            )
        added = super().add(negotiator, ufun=ufun, role=role, **kwargs)
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

    def round(self) -> MechanismRoundResult:

        # check that the chain is complete
        if not all(self.__chain):
            if self.dynamic_entry:
                return MechanismRoundResult(
                    error=True, error_details="The chain is not complete"
                )
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
            return MechanismRoundResult()

        # if all agreements are finalized end the mechanism session with success
        agreements = [
            self.__agreements[l]
            for l in range(len(self.__chain))
            if self.__agreements[l] is not None
        ]

        if len(agreements) == len(self.__chain) - 1:
            return MechanismRoundResult(agreement=agreements)
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
            return MechanismRoundResult(
                error=True, error_details="An unacceptable response was returned"
            )

        # If the response is to end the negotiation, end it but only if there are not partial negotiations
        if response == ResponseType.END_NEGOTIATION:
            if len(self.__agreements) > 0:
                return MechanismRoundResult(
                    error=True,
                    error_details="Cannot end a negotiation chain with some "
                    "agreements",
                )
            return MechanismRoundResult(broken=True)

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
            return MechanismRoundResult()

        # now it must be a rejection, ask the one who rejected to propose (in either direction)
        self.__last_proposal = self.__chain[self.__next_agent].propose(self.state)
        self.__last_proposer_index = self.__next_agent
        self._update_next()
        return MechanismRoundResult()

    def _update_next(self) -> None:
        if self.__last_proposal.left:
            self.__next_agent = (self.__last_proposer_index - 1) % len(self.__chain)
        else:
            self.__next_agent = (self.__last_proposer_index + 1) % len(self.__chain)

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


class MultiChainNegotiationsMechanism(Mechanism):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__chain: List[List[MultiChainNegotiator]] = []
        self.__next_agent_level = 0
        self.__next_agent_number = 0
        self.__last_proposal: Offer = None
        self.__last_proposer_level: int = -1
        self.__last_proposer_number: int = -1
        self.__agreements: Dict[int, Agreement] = defaultdict(lambda: None)
        self.__temp_agreements: Dict[int, Agreement] = defaultdict(lambda: None)
        self.__level: Dict[str, int] = {}
        self.__number: Dict[str, int] = {}

    def _get_ami(self, negotiator: Negotiator, role: str) -> AgentMechanismInterface:
        """
        Returns a chain AMI instead of the standard AMI.

        Args:
            negotiator: The negotiator to create the AMI for
            role: Its role in the negotiation (an integer >= -1 as a string)

        Returns:

        """
        return ChainAMI(
            id=self.id,
            n_outcomes=self.ami.n_outcomes,
            issues=self.ami.issues,
            outcomes=self.ami.outcomes,
            time_limit=self.ami.time_limit,
            step_time_limit=self.ami.step_time_limit,
            n_steps=self.ami.n_steps,
            dynamic_entry=self.ami.dynamic_entry,
            max_n_agents=self.ami.max_n_agents,
            annotation=self.ami.annotation,
            parent=self,
            negotiator=negotiator,
            level=int(role) + 1,
        )

    def add(
        self,
        negotiator: "Negotiator",
        *,
        ufun: Optional[UtilityFunction] = None,
        role: Optional[str] = None,
        **kwargs,
    ) -> Optional[bool]:
        if role is None:
            raise ValueError(
                "You cannot join this protocol without specifying the role. "
                "Possible roles are integers >= -1 "
            )
        added = super().add(negotiator, ufun=ufun, role=role, **kwargs)
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

    def round(self) -> MechanismRoundResult:

        # check that the chain is complete
        if not all(len(_) > 0 for _ in self.__chain):
            if self.dynamic_entry:
                return MechanismRoundResult(
                    error=True, error_details="The chain is not complete"
                )
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
            return MechanismRoundResult()

        # if all agreements are finalized end the mechanism session with success
        agreements = [
            self.__agreements[l]
            for l in range(len(self.__chain))
            if self.__agreements[l] is not None
        ]

        if len(agreements) == len(self.__chain) - 1:
            return MechanismRoundResult(agreement=agreements)
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
            return MechanismRoundResult(
                error=True, error_details="An unacceptable response was returned"
            )

        # If the response is to end the negotiation, end it but only if there are not partial negotiations
        if response == ResponseType.END_NEGOTIATION:
            if len(self.__agreements) > 0:
                return MechanismRoundResult(
                    error=True,
                    error_details="Cannot end a negotiation chain with some "
                    "agreements",
                )
            return MechanismRoundResult(broken=True)

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
            return MechanismRoundResult()

        # now it must be a rejection, ask the one who rejected to propose (in either direction)
        self.__last_proposal = self.__chain[self.__next_agent_level][
            self.__next_agent_number
        ].propose(self.state)
        self.__last_proposer_level = self.__next_agent_level
        self.__last_proposer_number = self.__next_agent_number
        self._update_next()
        return MechanismRoundResult()

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
