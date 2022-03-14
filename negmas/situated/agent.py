from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from negmas.common import MechanismState, NegotiatorMechanismInterface
from negmas.events import Event, EventSink, EventSource, Notifier
from negmas.helpers.inout import ConfigReader
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue
from negmas.preferences import Preferences, UtilityFunction
from negmas.types import Rational

from .awi import AgentWorldInterface
from .breaches import Breach
from .common import (
    NegotiationInfo,
    NegotiationRequestInfo,
    RenegotiationRequest,
    RunningNegotiationInfo,
)
from .contract import Contract
from .entity import Entity

__all__ = ["Agent"]


class Agent(Entity, EventSink, ConfigReader, Notifier, Rational, ABC):
    """Base class for all agents that can run within a `World` and engage in situated negotiations"""

    # def __getstate__(self):
    #     return self.name, self.awi
    #
    # def __setstate__(self, state):
    #     name, awi = state
    #     super().__init__(name=name)
    #     self._awi = awi

    def __init__(
        self,
        name: str = None,
        type_postfix: str = "",
        preferences: Preferences = None,
        ufun: UtilityFunction = None,
    ):
        super().__init__(type_postfix=type_postfix)
        Rational.__init__(self, name=name, preferences=preferences, ufun=ufun)
        self._running_negotiations: dict[str, RunningNegotiationInfo] = {}
        self._requested_negotiations: dict[str, NegotiationRequestInfo] = {}
        self._accepted_requests: dict[str, NegotiationRequestInfo] = {}
        self.contracts: list[Contract] = []
        self._unsigned_contracts: set[Contract] = set()
        self._awi: AgentWorldInterface = None  # type: ignore

    # def to_dict(self) -> Dict[str, Any]:
    #     """Converts the agent into  dict for storage purposes.
    #
    #     The agent need not be recoverable from this representation.
    #
    #     """
    #     try:
    #         d = to_dict(vars(dict), deep=False, keep_private=False, add_type_field=False)
    #         # _ = json.dumps(d)
    #         return d
    #     except:
    #         return {"id": self.id, "name": self.name}

    @property
    def initialized(self) -> bool:
        """Was the agent initialized (i.e. was init_() called)"""
        return self._initialized

    @property
    def unsigned_contracts(self) -> list[Contract]:
        """
        All contracts that are not yet signed.
        """
        return list(self._unsigned_contracts)

    @property
    def requested_negotiations(self) -> list[NegotiationRequestInfo]:
        """The negotiations currently requested by the agent.

        Returns:

            A list of negotiation request information objects (`NegotiationRequestInfo`)
        """
        return list(self._requested_negotiations.values())

    @property
    def accepted_negotiation_requests(self) -> list[NegotiationRequestInfo]:
        """
        A list of negotiation requests sent to this agent that are already accepted by it.

        Remarks:
            - These negotiations did not start yet as they are still not accepted  by all partners.
              Once that happens, they will be moved to `running_negotiations`
        """
        return list(self._accepted_requests.values())

    @property
    def negotiation_requests(self) -> list[NegotiationRequestInfo]:
        """A list of the negotiation requests sent by this agent that are not yet accepted or rejected.

        Remarks:
            - These negotiations did not start yet as they are still not accepted  by all partners.
              Once that happens, they will be moved to `running_negotiations`
        """
        return list(self._requested_negotiations.values())

    @property
    def running_negotiations(self) -> list[RunningNegotiationInfo]:
        """The negotiations currently requested by the agent.

        Returns:

            A list of negotiation information objects (`RunningNegotiationInfo`)
        """
        return list(self._running_negotiations.values())

    @property
    def awi(self) -> AgentWorldInterface:
        """Gets the Agent-world interface."""
        return self._awi

    @awi.setter
    def awi(self, awi: AgentWorldInterface):
        """Sets the Agent-world interface. Should only be called by the world."""
        self._awi = awi

    def create_negotiation_request(
        self,
        issues: list[Issue],
        partners: list[str],
        annotation: dict[str, Any] | None,
        negotiator: Negotiator | None,
        extra: dict[str, Any] | None,
    ) -> str:
        """
        Creates a new `NegotiationRequestInfo` record and returns its ID

        Args:
            issues: negotiation issues
            partners: partners
            annotation: annotation
            negotiator: the negotiator to use
            extra: any extra information

        Returns:
            A unique identifier for this negotiation info structure

        """
        req_id = str(uuid.uuid4())
        self._requested_negotiations[req_id] = NegotiationRequestInfo(
            issues=issues,
            partners=partners,
            annotation=annotation,
            negotiator=negotiator,
            extra=extra,
            requested=True,
            uuid=req_id,
        )
        return req_id

    def _request_negotiation(
        self,
        issues: list[Issue],
        partners: list[str],
        roles: list[str] = None,
        annotation: dict[str, Any] | None = None,
        mechanism_name: str = None,
        mechanism_params: dict[str, Any] = None,
        negotiator: Negotiator = None,
        extra: dict[str, Any] | None = None,
        group: str | None = None,
    ) -> bool:
        """
        Requests to start a negotiation with some other agents

        Args:
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation.
                      Note that the caller itself may not be in this list which
                      makes it possible for an agent to request a negotaition
                      that it does not participate in. If that is not to be
                      allowed in some world, override this method and explicitly
                      check for these kinds of negotiations and return False.
                      If partners is passed as a single string or as a list
                      containing a single string, then he caller will be added
                      at the beginning of the list. This will only be done if
                      `roles` was passed as None.
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object
            negotiator: My negotiator to use in this negotiation. Can be none
            extra: Any extra information I would like to keep to myself for this negotiation
            group: The negotiation group

        Returns:

            List["Agent"] the list of partners who rejected the negotiation if any. If None then the negotiation was
            accepted. If empty then the negotiation was not started from the world manager


        Remarks:

            - The function will create a request ID that will be used in callbacks `on_neg_request_accepted` and
              `on_neg_request_rejected`.
            - This function is a private function as the name implies and should not be called directly in any world.
            - World designers extending this class for their worlds, should define a way to start negotiations that
              calls this function. The simplest way is to just define a `request_negotiation` function that calls this
              private version directly with the same parameters.


        """
        if roles is None:
            if isinstance(partners, str) or isinstance(partners, Agent):
                partners = [partners]  # type: ignore
            if len(partners) == 1 and partners[0] != self.id:
                partners = [self.id, partners[0]]
        req_id = self.create_negotiation_request(
            issues=issues,
            partners=partners,
            annotation=annotation,
            negotiator=negotiator,
            extra=extra,
        )
        return self.awi.request_negotiation_about(
            issues=issues,
            partners=partners,
            req_id=req_id,
            roles=roles,
            annotation=annotation,
            group=group,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
        )

    def on_event(self, event: Event, sender: EventSource):
        if not isinstance(sender, Mechanism) and not isinstance(sender, Mechanism):
            raise ValueError(
                f"Sender of the negotiation end event is of type {sender.__class__.__name__} "
                f"not Mechanism!!"
            )

    @abstractmethod
    def on_neg_request_rejected(self, req_id: str, by: list[str] | None):
        """Called when a requested negotiation is rejected

        Args:
            req_id: The request ID passed to _request_negotiation
            by: A list of agents that refused to participate or None if the failure was for another reason


        """
        # if event.type == "negotiation_end":
        #     # will be sent by the World once a negotiation in which this agent is involved is completed            l
        #     mechanism_id = sender.id
        #     self._running_negotiations.pop(mechanism_id, None)

    # ------------------------------------------------------------------
    # EVENT CALLBACKS (Called by the `World` when certain events happen)
    # ------------------------------------------------------------------

    def on_neg_request_rejected_(self, req_id: str, by: list[str] | None):
        """Called when a requested negotiation is rejected

        Args:
            req_id: The request ID passed to _request_negotiation
            by: A list of agents that refused to participate or None if the failure was for another reason


        """
        self.on_neg_request_rejected(req_id, by)
        self._requested_negotiations.pop(req_id, None)

    @abstractmethod
    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        """Called when a requested negotiation is accepted"""

    def on_neg_request_accepted_(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        """Called when a requested negotiation is accepted"""
        my_request = req_id is not None
        _request_dict = self._requested_negotiations
        if req_id is None:
            # I am not the requesting agent
            req_id = mechanism.id
            _request_dict = self._accepted_requests
        neg = _request_dict.get(req_id, None)
        if neg is None:
            return
        if my_request:
            self.on_neg_request_accepted(req_id, mechanism)
        self._running_negotiations[mechanism.id] = RunningNegotiationInfo(
            extra=_request_dict[req_id].extra,
            negotiator=neg.negotiator,
            annotation=_request_dict[req_id].annotation,
            uuid=req_id,
            my_request=my_request,
        )
        _request_dict.pop(req_id, None)

    @abstractmethod
    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""

    def on_negotiation_failure_(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""
        self.on_negotiation_failure(partners, annotation, mechanism, state)
        self._running_negotiations.pop(mechanism.id, None)

    @abstractmethod
    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""

    def on_negotiation_success_(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""
        self.on_negotiation_success(contract, mechanism)
        self._unsigned_contracts.add(contract)
        self._running_negotiations.pop(mechanism.id, None)

    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""

    def on_contract_signed_(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""
        self.on_contract_signed(contract)
        if contract in self._unsigned_contracts:
            self._unsigned_contracts.remove(contract)
        self.contracts.append(contract)

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""

    def on_contract_cancelled_(self, contract: Contract, rejectors: list[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""
        self.on_contract_cancelled(contract, rejectors)
        if contract in self._unsigned_contracts:
            self._unsigned_contracts.remove(contract)

    @abstractmethod
    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: list[str],
        issues: list[Issue],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        role: str | None,
        req_id: str | None,
    ) -> Negotiator | None:
        """
        Called by the mechanism to ask for joining a negotiation. The agent can refuse by returning a None

        Args:
            initiator: The ID of the agent that initiated the negotiation request
            partners: The partner list (will include this agent)
            issues: The list of issues
            annotation: Any annotation specific to this negotiation.
            mechanism: The mechanism that started the negotiation
            role: The role of this agent in the negotiation
            req_id: The req_id passed to the AWI when starting the negotiation (only to the initiator).

        Returns:
            None to refuse the negotiation or a `Negotiator` object appropriate to the given mechanism to accept it.

        Remarks:

            - It is expected that world designers will introduce a better way to respond and override this function to
              call it

        """

    def respond_to_negotiation_request_(
        self,
        initiator: str,
        partners: list[str],
        issues: list[Issue],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        role: str | None,
        req_id: str | None,
    ) -> Negotiator | None:
        """Called when a negotiation request is received"""
        if req_id is not None:
            # I am the one who requested this negotiation
            info = self._requested_negotiations.get(req_id, None)
            if info and info.negotiator is not None:
                return info.negotiator
        negotiator = self._respond_to_negotiation_request(
            initiator=initiator,
            partners=partners,
            issues=issues,
            annotation=annotation,
            mechanism=mechanism,
            role=role,
            req_id=req_id,
        )
        if negotiator is not None:
            self._accepted_requests[mechanism.id] = NegotiationRequestInfo(
                partners,
                issues,
                annotation,
                uuid,
                negotiator,
                extra={"my_request": False},
                requested=False,
            )
        return negotiator

    def on_simulation_step_ended(self):
        """Will be called at the end of the simulation step after everything else"""

    def on_simulation_step_started(self):
        """Will be called at the beginning of the simulation step before everything else (except init)"""

    @abstractmethod
    def step(self):
        """Called by the simulator at every simulation step"""

    @abstractmethod
    def init(self):
        """Called to initialize the agent **after** the world is initialized. the AWI is accessible at this point."""

    def on_contracts_finalized(
        self,
        signed: list[Contract],
        cancelled: list[Contract],
        rejectors: list[list[str]],
    ) -> None:
        """
        Called for all contracts in a single step to inform the agent about which were finally signed
        and which were rejected by any agents (including itself)

        Args:
            signed: A list of signed contracts. These are binding
            cancelled: A list of cancelled contracts. These are not binding
            rejectors: A list of lists where each of the internal lists gives the rejectors of one of the
                       cancelled contracts. Notice that it is possible that this list is empty which
                       means that the contract other than being rejected by any agents (if that was possible in
                       the specific world).

        Remarks:

            The default implementation is to call `on_contract_signed` for singed contracts and `on_contract_cancelled`
            for cancelled contracts

        """
        for contract in signed:
            self.on_contract_signed_(contract)
        for contract, r in zip(cancelled, rejectors):
            self.on_contract_cancelled_(contract, r)

    @abstractmethod
    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        """
        Received by partners in ascending order of their total breach levels in order to set the
        renegotiation agenda when contract execution fails

        Args:

            contract: The contract being breached
            breaches: All breaches on `contract`

        Returns:

            Renegotiation agenda (issues to negotiate about to avoid reporting the breaches).

        """

    @abstractmethod
    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        """
        Called to respond to a renegotiation request

        Args:

            agenda:
            contract:
            breaches:

        Returns:

        """

    def sign_contract(self, contract: Contract) -> str | None:
        """Called after the signing delay from contract conclusion to sign the contract. Contracts become binding
        only after they are signed."""
        return self.id

    def sign_all_contracts(
        self, contracts: list[Contract]
    ) -> None | str | dict[str, str | None] | list[str | None]:
        """Called to sign all contracts concluded in a single step by this agent

        Args:
            contracts: A list of contracts to sign/ refuse to sign

        Return:
            You can return any of the following:

            - `None` to indicate refusing to sign all contracts.
            - `str` (specifically, the agent ID) to indicate signing ALL contracts.
            - `List[Optional[str]]` A list with a value for each input contract where `None` means refusal to sign that
              contract and a string (agent ID) indicates acceptance to sign it. Note that in this case, the number of
              values in the returned list must match that of the contacts (and they should obviously correspond to the
              contracts).
            - `Dict[str, Optional[str]]` A mapping from contract ID to either a `None` for rejection to sign or a string
              (for acceptance to sign). Contracts with IDs not in the keys will assumed not to be signed.

        Remarks:

            - default implementation calls `sign_contract` for each contract returning the results

        """
        return [self.sign_contract(contract) for contract in contracts]

    @abstractmethod
    def on_contract_executed(self, contract: Contract) -> None:
        """
        Called after successful contract execution for which the agent is one of the partners.
        """

    @abstractmethod
    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        """
        Called after complete processing of a contract that involved a breach.

        Args:
            contract: The contract
            breaches: All breaches committed (even if they were resolved)
            resolution: The resolution contract if re-negotiation was successful. None if not.
        """

    def __str__(self):
        return f"{self.name}"

    __repr__ = __str__
