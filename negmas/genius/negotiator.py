"""
Genius Negotiator
An agent used to connect to GENIUS agents (ver 8.0.4) and allow them to join negotiation mechanisms

"""
import math
import os
import pathlib
import random
import tempfile
import warnings
from typing import List, Optional, Tuple, Union

from ..common import AgentMechanismInterface, MechanismState
from ..config import CONFIG_KEY_GENIUS_BRIDGE_JAR, NEGMAS_CONFIG
from ..inout import get_domain_issues
from ..negotiators import Controller
from ..outcomes import Issue, ResponseType
from ..sao.common import SAOResponse
from ..sao.negotiators import SAONegotiator
from ..utilities import UtilityFunction, make_discounted_ufun, normalize
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
    """Encapsulates a Genius Negotiator

    Args:
        assume_normalized: Assume that the utility function is already normalized (or do not need to be normalized)
        ufun: The ufun of the negotiator [optional]
        name: Negotiator name [optional]
        rational_proposal: If true, the negotiator will not offer anything less than their reserved-value
        parent: Parent `Controller`
        owner: The agent that owns the negotiator (if any)
        java_class_name: The java class name of the Geinus underlying agent
        domain_file_name: Optional domain file name (containing the negotiation issues or agenda)
        utility_file_name: Optional ufun file name (xml) from which a ufun will be loaded for the agent
        keep_issue_names: When reading domainand utility files keep issue names
        keep_value_names: When reading domainand utility files keep value names
        can_propose: The negotiator can propose
        normalize_utility: Normalize the ufun [0-1] if it is not already normalized and not assumed-normalized.
        normalize_max_only: Normalize the max to 1.0 but do not normalize the min.
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
        assume_normalized=True,
        ufun: Optional[UtilityFunction] = None,
        name: str = None,
        rational_proposal=False,
        parent: Controller = None,
        owner: "Agent" = None,  # type: ignore
        java_class_name: str = None,
        domain_file_name: Union[str, pathlib.Path] = None,
        utility_file_name: Union[str, pathlib.Path] = None,
        keep_issue_names: bool = True,
        keep_value_names: bool = True,
        can_propose=True,
        normalize_utility: bool = False,
        normalize_max_only: bool = False,
        auto_load_java: bool = True,
        port: int = DEFAULT_JAVA_PORT,
        genius_bridge_path: str = None,
        strict: bool = None,
    ):
        super().__init__(
            name=name,
            assume_normalized=assume_normalized,
            ufun=None,
            rational_proposal=rational_proposal,
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
            else NEGMAS_CONFIG.get(
                CONFIG_KEY_GENIUS_BRIDGE_JAR, "~/negmas/files/geniusbridge.jar"
            )
        )
        self.java = None
        self.java_class_name = (
            java_class_name
            if java_class_name is not None
            else GeniusNegotiator.random_negotiator_name()
        )
        self._port = port
        self._normalize_utility = normalize_utility
        self._normalize_max_only = normalize_max_only
        self.domain_file_name = str(domain_file_name) if domain_file_name else None
        self.utility_file_name = str(utility_file_name) if utility_file_name else None
        self._my_last_offer = None
        self.keep_issue_names = keep_issue_names
        self._utility_function, self.discount = None, None
        self.issue_names = self.issues = self.issue_index = None
        self.auto_load_java = auto_load_java
        if domain_file_name is not None:
            # we keep original issues details so that we can create appropriate answers to Java
            self.issues = get_domain_issues(
                domain_file_name=domain_file_name,  # type: ignore
                keep_issue_names=True,
                keep_value_names=True,
            )
            self.issue_names = [_.name for _ in self.issues]  # type: ignore
            self.issue_index = dict(zip(self.issue_names, range(len(self.issue_names))))
        self.discount = None
        if utility_file_name is not None:
            self._utility_function, self.discount = UtilityFunction.from_genius(  # type: ignore
                utility_file_name,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
            )
        # if ufun is not None:
        #     self._utility_function = ufun
        self.base_utility = self._utility_function
        self.__ufun_received = ufun
        self._temp_domain_file = self._temp_ufun_file = False
        self.connected = False

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
    def robust_negotiators(cls) -> List[str]:
        """
        Returns a list of genius agents that were tested and seem to be robustly working with negmas
        """
        return TESTED_NEGOTIATORS

    @classmethod
    def negotiators(cls, agent_based=True, party_based=True) -> List[str]:
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
        domain_file_name: str = None,
        utility_file_name: str = None,
        keep_issue_names: bool = True,
        keep_value_names: bool = True,
        auto_load_java: bool = False,
        can_propose=True,
        name: str = None,
    ) -> "GeniusNegotiator":
        """
        Returns an agent with a random class name

        Args:
            name: negotiator name
            can_propose: Can this negotiator propose?
            auto_load_java: load the JVM if needed
            keep_value_names: Keep value names if values are strings
            keep_issue_names: Use dictionaries instead of tuples for representing outcomes
            utility_file_name: Name of the utility xml file
            domain_file_name: Name of the domain XML file
            port: port number to use if the JVM is to be started
            agent_based: Old agents based on the Java class Negotiator
            party_based: Newer agents based on the Java class AbstractNegotiationParty


        Returns:
            GeniusNegotiator an agent with a random java class
        """
        agent_name = cls.random_negotiator_name(
            agent_based=agent_based, party_based=party_based
        )
        return GeniusNegotiator(
            java_class_name=agent_name,
            port=port,
            domain_file_name=domain_file_name,
            utility_file_name=utility_file_name,
            keep_issue_names=keep_issue_names,
            keep_value_names=keep_value_names,
            auto_load_java=auto_load_java,
            can_propose=can_propose,
            name=name,
        )

    @property
    def is_connected(self):
        return self.connected and self.java is not None

    def _create(self):
        """Creates the corresponding java agent"""
        aid = self.java.create_agent(self.java_class_name)
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
        return self.java.get_name(self.java_uuid)

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        if ufun is None:
            ufun = self.__ufun_received
        result = super().join(ami=ami, state=state, ufun=ufun, role=role)
        if not result:
            return False
        # only connect to the JVM running genius-bridge if you are going to join a negotiation.
        if not self.is_connected:
            mechanism_port = ami.params.get("genius_port", 0)
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

        if self._normalize_utility:
            self._utility_function = normalize(
                self.ufun,  # type: ignore
                outcomes=ami.discrete_outcomes,  # type: ignore
                max_only=self._normalize_max_only,
            )
        self.issue_names = [_.name for _ in ami.issues]
        self.issues = ami.issues
        self.issue_index = dict(zip(self.issue_names, range(len(self.issue_names))))
        self.keep_issue_names = self.keep_value_names = ami.outcome_type == dict
        if ami.issues is not None and self.domain_file_name is None:
            domain_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.domain_file_name = domain_file.name
            domain_file.write(Issue.to_xml_str(ami.issues))
            domain_file.close()
            self._temp_domain_file = True
        if ufun is not None and self.utility_file_name is None:
            utility_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.utility_file_name = utility_file.name
            utility_file.write(
                UtilityFunction.to_xml_str(
                    ufun, issues=ami.issues, discount_factor=self.discount
                )
            )
            utility_file.close()
            self._temp_ufun_file = True
        return result

    def destroy_java_counterpart(self, state=None) -> None:
        if self.__started and not self.__destroyed:
            if self.java is not None:
                self.__frozen_relative_time = self.java.get_relative_time(
                    self.java_uuid
                )
                result = self.java.on_negotiation_end(
                    self.java_uuid,
                    None
                    if state is None
                    else self._outcome2str(state.agreement)
                    if state.agreement is not None
                    else None,
                )
                if result in (OK, TIMEOUT):
                    results = self.java.destroy_agent(self.java_uuid)
                    if result in (FAILED, TIMEOUT) and self._strict:
                        raise ValueError(
                            f"{self._me()} ended the negotiation but failed to destroy the agent. A possible memory leak"
                        )
                elif self._strict:
                    raise ValueError(f"{self._me()} failed to end the negotiation!!")
            self.__destroyed = True
        if self._temp_ufun_file:
            try:
                os.unlink(self.utility_file_name)  # type: ignore
            except (FileNotFoundError, PermissionError):
                pass
            self._temp_ufun_file = False

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
        if self._strict is None:
            self._strict = self.ami.n_steps is not None and self.ami.n_steps != float(
                "inf"
            )
        if self._utility_function is not None and self.utility_file_name is None:
            utility_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.utility_file_name = utility_file.name
            utility_file.write(
                UtilityFunction.to_xml_str(
                    self._utility_function,
                    issues=self.ami.issues,
                    discount_factor=self.discount,
                )
            )
            utility_file.close()
            self._temp_ufun_file = True
        info = self._ami
        if self.discount is not None and self.discount != 1.0:
            self._utility_function = make_discounted_ufun(
                self._utility_function,
                ami=info,
                discount_per_round=self.discount,
                power_per_round=1.0,
            )
        n_steps = -1 if info.n_steps is None else int(info.n_steps)  # number of steps
        n_seconds = (
            -1
            if info.time_limit is None or math.isinf(info.time_limit)
            else int(info.time_limit)
        )  # time limit
        timeout = (
            info.negotiator_time_limit
            if info.negotiator_time_limit and not math.isinf(info.negotiator_time_limit)
            else info.step_time_limit
            if info.step_time_limit and not math.isinf(info.step_time_limit)
            else info.time_limit
            if info.time_limit and not math.isinf(info.time_limit)
            else DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT
        )
        if timeout is None or math.isinf(timeout) or timeout <= 0:
            timeout = DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT

        if n_steps * n_seconds > 0:
            # n_seconds take precedence
            if self._strict:
                raise ValueError(
                    f"{self._me()}: Both n_steps ({n_steps}) and n_seconds ({n_seconds}) are given. Not allowed in strict execution"
                )
            warnings.warn(
                f"{self._me()}: Both n_steps ({n_steps}) and n_seconds ({n_seconds}) are given. n_steps will be ignored"
            )
            n_steps = -1
        try:
            result = self.java.on_negotiation_start(
                self.java_uuid,  # java_uuid
                info.n_negotiators,  # number of agents
                n_steps,
                n_seconds,
                n_seconds > 0,
                self.domain_file_name,  # domain file
                self.utility_file_name,  # Negotiator file
                int(timeout),
                self._strict,
            )
            self.__started = result == OK
        except Exception as e:
            raise ValueError(f"{self._me()}: Cannot start negotiation: {str(e)}")

    def cancel(self, reason=None) -> None:
        try:
            self.java.cancel(self.java_uuid)
        except:
            pass

    @property
    def relative_time(self) -> Optional[float]:
        if self.ami is None or not self.ami.state.started:
            return 0
        if self.ami is not None and self.ami.state.ended:
            return self.__frozen_relative_time
        t = self.java.get_relative_time(self.java_uuid)
        if t < 0:
            if self._strict:
                raise ValueError(
                    f"{self._me()} cannot read relative time (returned {t})"
                )
            return None
        return t

    def _me(self):
        return (
            f"Agent {self.name} (jid: {self.java_uuid}) of type {self.java_class_name}"
        )

    def counter(self, state: MechanismState, offer: Optional["Outcome"]):
        if offer is None and self._my_last_offer is not None and self._strict:
            raise ValueError(f"{self._me()} got counter with a None offer.")
        if offer is not None:
            received = self.java.receive_message(
                self.java_uuid,
                state.current_proposer,
                "Offer",
                self._outcome2str(offer),
                state.step,
            )
            if self._strict and received == FAILED:
                raise ValueError(
                    f"{self._me()} failed to receive message in step {state.step}"
                )
        response, outcome = self.parse(
            self.java.choose_action(self.java_uuid, state.step)
        )
        if self._strict and (
            response
            not in (
                ResponseType.REJECT_OFFER,
                ResponseType.ACCEPT_OFFER,
                ResponseType.END_NEGOTIATION,
            )
            or (response == ResponseType.REJECT_OFFER and outcome is None)
        ):
            raise ValueError(
                f"{self._me()} returned a None counter offer in step {state.step}"
            )

        self._my_last_offer = outcome
        return SAOResponse(response, outcome)

    def propose(self, state):
        raise ValueError(
            f"{self._me()}: propose should never be called directly on GeniusNegotiator"
        )

    def parse(self, action: str) -> Tuple[Optional[ResponseType], Optional["Outcome"]]:
        """
        Parses an action into a ResponseType and an Outcome (if one is included)
        Args:
            action:

        Returns:

        """
        response, outcome = None, None
        if len(action) < 1:
            if self._strict:
                raise ValueError(f"{self._me()} received no actions while parsing")
            return ResponseType.REJECT_OFFER, None
        _, typ_, bid_str = action.split(FIELD_SEP)
        if typ_ == FAILED:
            raise ValueError(
                f"{self._me()} sent an action that cannot be parsed ({action})"
            )
        elif typ_ == TIMEOUT:
            if self.ami.state.relative_time < 1.0:
                raise ValueError(
                    f"{self._me()} indicated that it timedout at relative time ({self.ami.state.relative_time})"
                )

        issues = self._ami.issues

        if typ_ in ("Offer",) and (bid_str is not None and len(bid_str) > 0):
            try:
                if self._ami.outcome_type == tuple:
                    values = {
                        _[0]: _[1]
                        for _ in [
                            _.split(INTERNAL_SEP) for _ in bid_str.split(ENTRY_SEP)
                        ]
                    }
                    outcome = []
                    for issue in issues:
                        outcome.append(issue.value_type(values[issue.name]))
                    outcome = tuple(outcome)
                else:
                    outcome = {
                        _[0]: _[1]
                        for _ in [
                            _.split(INTERNAL_SEP) for _ in bid_str.split(ENTRY_SEP)
                        ]
                    }
                    issue_map = {i.name: i for i in issues}
                    for k, v in outcome.items():
                        outcome[k] = issue_map[k].value_type(v)
                    if self._ami.outcome_type != dict:
                        outcome = self._ami.outcome_type(
                            **{
                                _[0]: _[1]
                                for _ in [
                                    _.split(INTERNAL_SEP)
                                    for _ in bid_str.split(ENTRY_SEP)
                                ]
                            }
                        )
            except Exception as e:
                if self._strict:
                    raise ValueError(
                        f"{self._me()} failed to parse {bid_str} of action {action} with exception {str(e)}"
                    )
                warnings.warn(
                    f"{self._me()} failed in parsing bid string: {bid_str} of action {action} with exception {str(e)}"
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
            raise ValueError(
                f"{self._me()}: Unknown response: {typ_} in action {action}"
            )
        return response, outcome

    def _outcome2str(self, outcome):
        output = ""
        if not isinstance(outcome, dict):
            outcome_dict = dict(zip(self.issue_names, outcome))
        else:
            outcome_dict = outcome
        for i, v in outcome_dict.items():
            # todo check that the order here will be correct!!
            output += f"{i}{INTERNAL_SEP}{v}{ENTRY_SEP}"
        output = output[: -len(ENTRY_SEP)]
        return output

    def __str__(self):
        name = super().__str__().split("/")
        return "/".join(name[:-1]) + f"/{self.java_class_name}/" + name[-1]
