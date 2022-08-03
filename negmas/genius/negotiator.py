"""
Genius Negotiator
An agent used to connect to GENIUS agents (ver 8.0.4) and allow them to join negotiation mechanisms

"""
from __future__ import annotations

import math
import os
import random
import tempfile
from pathlib import Path

from negmas import warnings
from negmas.outcomes.base_issue import Issue

from ..common import MechanismState, NegotiatorMechanismInterface
from ..config import CONFIG_KEY_GENIUS_BRIDGE_JAR, negmas_config
from ..inout import get_domain_issues
from ..negotiators import Controller
from ..outcomes import CartesianOutcomeSpace, Outcome, issues_to_xml_str
from ..preferences import UtilityFunction, make_discounted_ufun
from ..sao.common import ResponseType, SAOResponse, SAOState
from ..sao.negotiators import SAONegotiator
from .bridge import GeniusBridge
from .common import (
    DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT,
    DEFAULT_JAVA_PORT,
    get_free_tcp_port,
)
from .ginfo import AGENT_BASED_NEGOTIATORS, PARTY_BASED_NEGOTIATORS, TESTED_NEGOTIATORS

__all__ = [
    "GeniusNegotiator",
]

INTERNAL_SEP, ENTRY_SEP, FIELD_SEP = "<<s=s>>", "<<y,y>>", "<<sy>>"
FAILED, OK, TIMEOUT = "__FAILED__", "__OK__", "__TIMEOUT__"


class GeniusNegotiator(SAONegotiator):
    """
    Encapsulates a Genius Negotiator

    Args:
        preferences: The ufun of the negotiator [optional]
        name: Negotiator name [optional]
        parent: Parent `Controller`
        owner: The agent that owns the negotiator (if any)
        java_class_name: The java class name of the Geinus underlying agent
        domain_file_name: Optional domain file name (containing the negotiation issues or agenda)
        utility_file_name: Optional ufun file name (xml) from which a ufun will be loaded for the agent
        can_propose: The negotiator can propose
        auto_load_java: Load the genius bridge if needed
        port: The port to load the genius bridge to (or use if it is already loaded)
        genius_bridge_path: The path to the genius bridge
        strict: If True, raise exceptions if any exception is thrown by the agent or the bridge
                (or if the agent could not choose an action).
                If false, ignore these exceptions and assume a None return.
                If None use strict for n_steps limited negotiations and not strict for time_limit
                limited ones.
    """

    def __init__(
        self,
        preferences: UtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent = None,  # type: ignore
        java_class_name: str | None = None,
        domain_file_name: str | Path | None = None,
        utility_file_name: str | Path | None = None,
        can_propose=True,
        auto_load_java: bool = True,
        port: int = DEFAULT_JAVA_PORT,
        genius_bridge_path: str | None = None,
        strict: bool | None = None,
    ):
        super().__init__(
            name=name,
            preferences=None,
            parent=parent,
            owner=owner,
        )
        self.__frozen_relative_time = None
        self.__destroyed = False
        self.__started = False
        self._strict = strict
        self.capabilities["propose"] = can_propose
        self.add_capabilities({"genius": True})
        self.genius_bridge_path = (
            genius_bridge_path
            if genius_bridge_path is not None
            else str(
                negmas_config(
                    CONFIG_KEY_GENIUS_BRIDGE_JAR,
                    Path.home() / "negmas" / "files" / "geniusbridge.jar",
                )
            )
        )
        self.java = None
        self.java_class_name = (
            java_class_name if java_class_name is not None else GeniusNegotiator.random_negotiator_name()
        )
        self._port = port
        self.domain_file_name = str(domain_file_name) if domain_file_name else None
        self.utility_file_name = str(utility_file_name) if utility_file_name else None
        self._my_last_offer = None
        self._preferences, self.discount = None, None
        self.issue_names = self.issues = self.issue_index = None
        self.auto_load_java = auto_load_java
        if domain_file_name is not None:
            # we keep original issues details so that we can create appropriate answers to Java
            self.issues = get_domain_issues(
                domain_file_name=domain_file_name,
            )
            if not self.issues:
                raise ValueError(f"Cannot read domain file {domain_file_name}")
            self.issue_names = [_.name for _ in self.issues]
            self.issue_index = dict(zip(self.issue_names, range(len(self.issue_names))))
        self.discount = None
        if utility_file_name is not None:
            self._preferences, self.discount = UtilityFunction.from_genius(
                utility_file_name,
                issues=self.issues,
            )
        # if ufun is not None:
        #     self._preferences = ufun
        self.base_utility = self._preferences
        self.__preferences_received = preferences
        self._temp_domain_file = self._temp_preferences_file = False
        self.connected = False

    @property
    def is_connected(self):
        return self.connected and self.java is not None

    @property
    def port(self):
        # if a port was not specified then we just set any random empty port to be used
        if not self.is_connected and self._port <= 0:
            self._port = get_free_tcp_port()
        return self._port

    @port.setter
    def port(self, port):
        self._port = port

    @classmethod
    def robust_negotiators(cls) -> list[str]:
        """
        Returns a list of genius agents that were tested and seem to be robustly working with negmas
        """
        return TESTED_NEGOTIATORS

    @classmethod
    def negotiators(cls, agent_based=True, party_based=True) -> list[str]:
        """
        Returns a list of all available agents in genius 8.4.0

        Args:
            agent_based: Old agents based on the Java class Negotiator
            party_based: Newer agents based on the Java class AbstractNegotiationParty

        Returns:

        """
        r = []
        if agent_based:
            r += AGENT_BASED_NEGOTIATORS
        if party_based:
            r += PARTY_BASED_NEGOTIATORS
        return r

    @classmethod
    def random_negotiator_name(
        cls,
        agent_based=True,
        party_based=True,
    ):
        agent_names = cls.negotiators(agent_based=agent_based, party_based=party_based)
        return random.choice(agent_names)

    @classmethod
    def random_negotiator(
        cls,
        agent_based=True,
        party_based=True,
        port: int = DEFAULT_JAVA_PORT,
        domain_file_name: str | None = None,
        utility_file_name: str | None = None,
        auto_load_java: bool = False,
        can_propose=True,
        name: str | None = None,
    ) -> GeniusNegotiator:
        """
        Returns an agent with a random class name

        Args:
            name: negotiator name
            can_propose: Can this negotiator propose?
            auto_load_java: load the JVM if needed
            utility_file_name: Name of the utility xml file
            domain_file_name: Name of the domain XML file
            port: port number to use if the JVM is to be started
            agent_based: Old agents based on the Java class Negotiator
            party_based: Newer agents based on the Java class AbstractNegotiationParty


        Returns:
            GeniusNegotiator an agent with a random java class
        """
        agent_name = cls.random_negotiator_name(agent_based=agent_based, party_based=party_based)
        return GeniusNegotiator(
            java_class_name=agent_name,
            port=port,
            domain_file_name=domain_file_name,
            utility_file_name=utility_file_name,
            auto_load_java=auto_load_java,
            can_propose=can_propose,
            name=name,
        )

    def _create(self):
        """Creates the corresponding java agent"""
        if self.java is None:
            raise ValueError(f"Cannot create Genius Agent (no java instance)")
        aid = self.java.create_agent(self.java_class_name)  # type: ignore
        if aid == FAILED:
            raise ValueError(f"Cannot initialized {self.java_class_name}")
        return aid

    def _connect(self, path: str, port: int, auto_load_java: bool = False) -> bool:
        """
        Connects the negotiator to an appropriate genius-bridge running the actual agent
        """
        try:
            gateway = GeniusBridge.gateway(port)
        except:
            gateway = None
        if gateway is None:
            if auto_load_java:
                GeniusBridge.start(port=port, path=path)
            try:
                gateway = GeniusBridge.gateway(port)
            except:
                gateway = None
            if gateway == None:
                self.java = None
                return False
        self.java = gateway.entry_point  # type: ignore
        return True

    @property
    def java_name(self):
        if not self.java:
            return None
        return self.java.get_name(self.java_uuid)  # type: ignore

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: UtilityFunction | None = None,
        ufun: UtilityFunction | None = None,
        role: str = "negotiator",
    ) -> bool:
        if ufun:
            preferences = ufun
        if not preferences:
            preferences = self.__preferences_received
        result = super().join(nmi=nmi, state=state, preferences=preferences, ufun=None, role=role)
        if not result or self.ufun is None:
            return False
        # only connect to the JVM running genius-bridge if you are going to join a negotiation.
        if not self.is_connected:
            mechanism_port: int = nmi.params.get("genius_port", 0)
            if mechanism_port > 0:
                self.port = mechanism_port
            self.connected = self._connect(
                path=self.genius_bridge_path,
                port=self.port,
                auto_load_java=self.auto_load_java,
            )
            if not self.is_connected:
                return False

        self.java_uuid = self._create()

        if not isinstance(nmi.cartesian_outcome_space, CartesianOutcomeSpace):
            raise ValueError(
                f"Genius negotiators cannot be used with an out come space of type {nmi.cartesian_outcome_space.__class__.__name__}. Must use `CartesianOutcomeSpace`"
            )
        self.issues = nmi.cartesian_outcome_space.issues
        self.issue_names = [_.name for _ in self.issues]
        self.issue_index = dict(zip(self.issue_names, range(len(self.issue_names))))
        if nmi.cartesian_outcome_space is not None and self.domain_file_name is None:
            domain_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.domain_file_name = domain_file.name
            domain_file.write(issues_to_xml_str(nmi.cartesian_outcome_space.issues))
            domain_file.close()
            self._temp_domain_file = True
        if preferences is not None and self.utility_file_name is None:
            utility_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.utility_file_name = utility_file.name
            utility_file.write(
                UtilityFunction.to_xml_str(
                    preferences,
                    issues=nmi.cartesian_outcome_space.issues,
                    discount_factor=self.discount,
                )
            )
            utility_file.close()
            self._temp_preferences_file = True
        return result

    def _outcome2str(self, outcome):
        output = ""
        if not self.issue_names:
            nmi = self.nmi
            if not nmi:
                raise ValueError("Cannot send outcome to java without an NMI")
            self.issue_names = [_.name for _ in nmi.cartesian_outcome_space.issues]
        outcome_dict = dict(zip(self.issue_names, outcome))
        for i, v in outcome_dict.items():
            # todo check that the order here will be correct!!
            output += f"{i}{INTERNAL_SEP}{v}{ENTRY_SEP}"
        output = output[: -len(ENTRY_SEP)]
        return output

    def destroy_java_counterpart(self, state=None) -> None:
        if self.__started and not self.__destroyed:
            if self.java is not None:
                self.__frozen_relative_time = self.java.get_relative_time(self.java_uuid)  # type: ignore
                result = self.java.on_negotiation_end(  # type: ignore
                    self.java_uuid,
                    None
                    if state is None
                    else self._outcome2str(state.agreement)
                    if state.agreement is not None
                    else None,
                )
                if result in (OK, TIMEOUT):
                    result = self.java.destroy_agent(self.java_uuid)  # type: ignore
                    if result in (FAILED, TIMEOUT) and self._strict:
                        raise ValueError(
                            f"{self._me()} ended the negotiation but failed to destroy the agent. A possible memory leak"
                        )
                elif self._strict:
                    raise ValueError(f"{self._me()} failed to end the negotiation!!")
            self.__destroyed = True
        if self._temp_preferences_file:
            try:
                os.unlink(self.utility_file_name)  # type: ignore
            except (FileNotFoundError, PermissionError):
                pass
            self._temp_preferences_file = False

        if self._temp_domain_file:
            try:
                os.unlink(self.domain_file_name)  # type: ignore
            except (FileNotFoundError, PermissionError):
                pass
            self._temp_domain_file = False

    def on_negotiation_end(self, state: MechanismState) -> None:
        """called when a negotiation is ended"""
        super().on_negotiation_end(state)
        self.destroy_java_counterpart(state)

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Called when the info starts. Connects to the JVM."""
        super().on_negotiation_start(state=state)
        if self._strict is None and self.nmi is not None:
            self._strict = self.nmi.n_steps is not None and self.nmi.n_steps != float("inf")
        info = self.nmi
        if info is None:
            raise ValueError("Cannot start a negotiation without a NMI")
        if self._preferences is not None and self.utility_file_name is None:
            domain_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.domain_file_name = domain_file.name
            domain_file.write(issues_to_xml_str(self._preferences.outcome_space.issues))  # type: ignore
            domain_file.close()
            self._temp_domain_file = True
            utility_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.utility_file_name = utility_file.name
            if not isinstance(self.ufun, UtilityFunction):
                raise ValueError(
                    f"Genius Negotiator must have `UtilityFunctions` but you passed {self.ufun.__class__.__name__}"
                )
            utility_file.write(
                UtilityFunction.to_xml_str(
                    self.ufun,
                    issues=info.cartesian_outcome_space.issues,
                    discount_factor=self.discount,
                )
            )
            utility_file.close()
            self._temp_preferences_file = True
        if self.discount is not None and self.discount != 1.0 and self._preferences:
            if not isinstance(self.ufun, UtilityFunction):
                raise ValueError(
                    f"Genius Negotiator must have `UtilityFunctions` but you passed {self.ufun.__class__.__name__}"
                )
            self.set_preferences(
                make_discounted_ufun(
                    self.ufun,
                    discount_per_round=self.discount,
                    power_per_round=1.0,
                )
            )
        n_steps = -1 if info.n_steps is None else int(info.n_steps)  # number of steps
        n_seconds = -1 if info.time_limit is None or math.isinf(info.time_limit) else int(info.time_limit)  # time limit
        timeout = (
            info.negotiator_time_limit
            if info.negotiator_time_limit and not math.isinf(info.negotiator_time_limit)
            else info.step_time_limit
            if info.step_time_limit and not math.isinf(info.step_time_limit)
            else info.time_limit
            if info.time_limit and not math.isinf(info.time_limit)
            else DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT
        )
        if timeout is None or math.isinf(timeout) or timeout < 0:
            timeout = DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT

        if info.mechanism:
            timeout = min(timeout, info.mechanism._hidden_time_limit)

        if n_steps * n_seconds > 0:
            if n_steps < 0:
                if self._strict:
                    raise ValueError(
                        f"{self._me()}: Neither n_steps ({n_steps}) nor n_seconds ({n_seconds}) are given. Not allowed in strict mode"
                    )
                warnings.warn(
                    f"{self._me()}: Neither n_steps ({n_steps}) nor n_seconds ({n_seconds}) are given. This may lead to an infinite negotiation",
                    warnings.NegmasInfiniteNegotiationWarning,
                )
            else:
                # n_steps take precedence
                if self._strict:
                    raise ValueError(
                        f"{self._me()}: Both n_steps ({n_steps}) and n_seconds ({n_seconds}) are given. Not allowed in strict execution"
                    )
                warnings.warn(
                    f"{self._me()}: Both n_steps ({n_steps}) and n_seconds ({n_seconds}) are given. time_limit will be ignored",
                    warnings.NegmasStepAndTimeLimitWarning,
                )
                n_seconds, timeout = -1, -1
        timeout = int(timeout)
        try:
            result = self.java.on_negotiation_start(  # type: ignore
                self.java_uuid,  # java_uuid
                info.n_negotiators,  # number of agents
                n_steps,
                n_seconds,
                n_seconds > 0,
                self.domain_file_name,  # domain file
                self.utility_file_name,  # Negotiator file
                timeout,
                self._strict,
            )
            self.__started = result == OK
        except Exception as e:
            raise ValueError(
                f"{self._me()}: Cannot start negotiation: {str(e)}\nDomain file: {self.domain_file_name}\nUFun file: {self.domain_file_name}\n{'strict' if self._strict else 'non-strict'} {n_steps=} {timeout=}"
            )

    def cancel(self, reason=None) -> None:
        try:
            self.java.cancel(self.java_uuid)  # type: ignore
        except:
            pass

    @property
    def relative_time(self) -> float | None:
        if self.nmi is None or not self.nmi.state.started:
            return 0
        if self.nmi is not None and self.nmi.state.ended:
            return self.__frozen_relative_time
        t = self.java.get_relative_time(self.java_uuid)  # type: ignore
        if t < 0:
            if self._strict:
                raise ValueError(f"{self._me()} cannot read relative time (returned {t})")
            return None
        return t

    def _me(self):
        return f"Agent {self.name} (jid: {self.java_uuid}) of type {self.java_class_name}"

    def parse(self, action: str) -> tuple[ResponseType, Outcome | None]:
        """
        Parses an action into a ResponseType and an Outcome (if one is included)
        Args:
            action:

        Returns:

        """
        outcome = None
        if len(action) < 1:
            if self._strict:
                raise ValueError(f"{self._me()} received no actions while parsing")
            return ResponseType.REJECT_OFFER, None
        _, typ_, bid_str = action.split(FIELD_SEP)
        nmi = self.nmi
        if not nmi:
            raise ValueError("Cannot parse without an NMI")
        if typ_ == FAILED:
            raise ValueError(f"{self._me()} sent an action that cannot be parsed ({action})")
        elif typ_ == TIMEOUT:
            if nmi.state.relative_time < 1.0:
                raise ValueError(
                    f"{self._me()} indicated that it timedout at relative time ({nmi.state.relative_time})"
                )

        issues = nmi.cartesian_outcome_space.issues

        def map_value(issue: Issue, val: str):
            if not issue.value_type:
                return val
            if issubclass(issue.value_type, tuple):
                return eval(val)
            return issue.value_type(val)  # type: ignore (It will always be a constructable type)

        if typ_ in ("Offer",) and (bid_str is not None and len(bid_str) > 0):
            try:
                values = {_[0]: _[1] for _ in [_.split(INTERNAL_SEP) for _ in bid_str.split(ENTRY_SEP)]}
                outcome = []
                for issue in issues:
                    outcome.append(map_value(issue, values[issue.name]))
                outcome = tuple(outcome)
            except Exception as e:
                if self._strict:
                    raise ValueError(
                        f"{self._me()} failed to parse {bid_str} of action {action} with exception {str(e)}"
                    )
                warnings.warn(
                    f"{self._me()} failed in parsing bid string: {bid_str} of action {action} with exception {str(e)}",
                    warnings.NegmasBrdigeParsingWarning,
                )

        if typ_ in (TIMEOUT, "Offer"):
            response = ResponseType.REJECT_OFFER
        elif typ_ == "Accept":
            response = ResponseType.ACCEPT_OFFER
        elif typ_ == "EndNegotiation":
            response = ResponseType.END_NEGOTIATION
        elif typ_ in ("NullOffer", "Failure", "NoAction"):
            if self._strict:
                raise ValueError(f"{self._me()} received {typ_} in action {action}")
            response = ResponseType.REJECT_OFFER
            outcome = None
        else:
            raise ValueError(f"{self._me()}: Unknown response: {typ_} in action {action}")
        return response, tuple(outcome) if outcome else None

    def counter(self, state: SAOState, offer: Outcome | None):
        if offer is None and self._my_last_offer is not None and self._strict:
            raise ValueError(f"{self._me()} got counter with a None offer.")
        if offer is not None:
            received = self.java.receive_message(  # type: ignore
                self.java_uuid,
                state.current_proposer,
                "Offer",
                self._outcome2str(offer),
                state.step,
            )
            if self._strict and received == FAILED:
                raise ValueError(f"{self._me()} failed to receive message in step {state.step}")
        response, outcome = self.parse(self.java.choose_action(self.java_uuid, state.step))  # type: ignore
        if response is None or (
            self._strict
            and (
                response
                not in (
                    ResponseType.REJECT_OFFER,
                    ResponseType.ACCEPT_OFFER,
                    ResponseType.END_NEGOTIATION,
                )
                or (response == ResponseType.REJECT_OFFER and outcome is None)
            )
        ):
            raise ValueError(f"{self._me()} returned a None counter offer in step {state.step}")

        self._my_last_offer = outcome
        return SAOResponse(response, outcome)

    def propose(self, state):
        raise ValueError(f"{self._me()}: propose should never be called directly on GeniusNegotiator")

    def __str__(self):
        name = super().__str__().split("/")
        return "/".join(name[:-1]) + f"/{self.java_class_name}/" + name[-1]
