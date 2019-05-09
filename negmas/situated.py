"""This module defines the base classes for worlds within which multiple agents engage in situated negotiations

The `Agent` class encapsulates the managing entity that creates negotiators to engage in negotiations within a world
`Simulation` in order to maximize its own total utility.

Remarks:
--------

    - When immediate_negotiations is true, negotiations start in the same step they are registered in (they may also end
      in that step) and `negotiation_speed_multiple` steps of it are conducted. That entails that requesting a
      negotiation may result in new contracts in the same time-step only of `immediate_negotiations` is true.

Simulation steps:
-----------------

    #. prepare custom stats (call `_pre_step_stats`)
    #. sign contracts that are to be signed at this step calling `on_contract_signed` as needed
    #. step all existing negotiations `negotiation_speed_multiple` times handling any failed negotiations and creating
       contracts for any resulting agreements
    #. run all `Entity` objects registered (i.e. all agents) in the predefined `simulation_order`.
    #. execute contracts that are executable at this time-step handling any breaches
    #. allow custom simulation steps to run (call `_simulation_step`)
    #. remove any negotiations that are completed!
    #. update basic stats
    #. update custom stats (call `_post_step_stats`)

"""
import copy
import logging
import os
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from enum import Enum
from pathlib import Path
from typing import Dict, Sized
from typing import (
    Optional,
    List,
    Any,
    Tuple,
    Callable,
    Union,
    Iterable,
    Set,
    Iterator,
    Collection,
)

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, field

from negmas.common import AgentMechanismInterface, MechanismState
from negmas.common import NamedObject
from negmas.events import Event, EventSource, EventSink, Notifier
from negmas.helpers import (
    ConfigReader,
    instantiate,
    get_class,
    unique_name,
    snake_case,
    dump,
    create_loggers,
    add_records,
)
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.outcomes import OutcomeType, Issue

__all__ = [
    "Action",  # An action that an `Agent` can execute in the `World`.
    "Contract",  # A agreement definition which encapsulates an agreement with partners and extra information
    "Breach",  # A breach in executing a contract
    "BreachProcessing",
    "Agent",  # Negotiator capable of engaging in multiple negotiations
    "BulletinBoard",
    "World",
    "Entity",
    "AgentWorldInterface",  # the interface though which an agent can interact with the world
    "NegotiationInfo",
    "RenegotiationRequest",
    "save_stats",
]

PROTOCOL_CLASS_NAME_FIELD = "__mechanism_class_name"

try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({"YAMLLoadWarning": False})
except:
    pass


@dataclass
class Action:
    """An action that an `Agent` can execute in a `World` through the `Simulator`."""

    type: str
    """Action name."""
    params: dict
    """Any extra parameters to be passed for the action."""

    def __str__(self):
        return f"{self.type}: {self.params}"


Signature = namedtuple("Signature", ["id", "signature"])
"""A signature with the name of signature and her signature"""


@dataclass
class Contract(OutcomeType):
    """A agreement definition which encapsulates an agreement with partners and extra information"""

    partners: List[str] = field(default_factory=list)
    """The partners"""
    agreement: OutcomeType = None
    """The actual agreement of the negotiation in the form of an `Outcome` in the `Issue` space defined by `issues`"""
    annotation: Dict[str, Any] = field(default_factory=dict)
    """Misc. information to be kept with the agreement."""
    issues: List[Issue] = field(default_factory=list)
    """Issues of the negotiations from which this agreement was concluded. It may be empty"""
    signed_at: Optional[int] = None
    """The time-step at which the contract was signed"""
    concluded_at: Optional[int] = None
    """The time-step at which the contract was concluded (but it is still not binding until signed)"""
    nullified_at: Optional[int] = None
    """The time-step at which the contract was nullified after being signed. That can happen if a partner declares 
    bankruptcy"""
    to_be_signed_at: Optional[int] = None
    """The time-step at which the contract should be signed"""
    signatures: List[Signature] = field(default_factory=list)
    """A list of signatures giving agent name, signature"""
    mechanism_state: Optional[MechanismState] = None
    """The mechanism state at the contract conclusion"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""

    def __str__(self):
        return f'{", ".join(self.partners)} agreed on {str(self.agreement)}'

    def __hash__(self):
        """The hash depends only on the name"""
        return self.id.__hash__()

    class Java:
        implements = ["jnegmas.situated.Contract"]


@dataclass
class Breach:
    contract: Contract
    """The agreement being breached"""
    perpetrator: str
    """ID of the agent committing the breach"""
    type: str
    """The type of the breach. Can be one of: `refusal`, `product`, `money`, `penalty`."""
    victims: List[str] = field(default_factory=list)
    """Specific victims of the breach. If not given all partners in the agreement (except perpetrator) are considered 
    victims"""
    level: float = 1.0
    """Breach level defaulting to full breach (a number between 0 and 1)"""
    step: int = -1
    """The simulation step at which the breach occurred"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""

    def __hash__(self):
        """The hash depends only on the name"""
        return self.id.__hash__()

    def __str__(self):
        return f"Breach ({self.level} {self.type}) by {self.perpetrator} on {self.contract.id} at {self.step}"

    def as_dict(self):
        return {
            "contract": str(self.contract),
            "contract_id": self.contract.id,
            "type": self.type,
            "level": self.level,
            "id": self.id,
            "perpetrator": self.perpetrator,
            "perpetrator_type": self.perpetrator.__class__.__name__,
            "victims": [_ for _ in self.victims],
            "step": self.step,
            "resolved": None,
        }

    class Java:
        implements = ["jnegmas.situated.Breach"]


class BreachProcessing(Enum):
    """The way breaches are to be handled"""

    NONE = 0
    """The breach should always be reported in the breach list and no re-negotiation is allowed."""
    VICTIM_THEN_PERPETRATOR = 1
    """The victim is asked to set the re-negotiation agenda then the perpetrator."""
    META_NEGOTIATION = 2
    """A meta negotiation is instantiated between victim and perpetrator to set re-negotiation issues."""


@dataclass
class RenegotiationRequest:
    """A request for renegotiation.

    The issues can be any or all of the following:

    immediate_delivery: int
    immediate_unit_price: float
    later_quantity: int
    later_unit_price: int
    later_penalty: float
    later_time: int

    """

    publisher: "Agent"
    issues: List[Issue]
    annotation: Dict[str, Any] = field(default_factory=dict)

    class Java:
        implements = ["jnegmas.situated.RenegotiationRequest"]


class Entity(NamedObject):
    """Defines an entity that is a part of the world but does not participate in the simulation"""

    def __init__(self, name: str = None):
        super().__init__(name=name)

    @property
    def type_name(self):
        """Returns the name of the type of this entity"""
        return snake_case(self.__class__.__name__)

    @property
    def short_type_name(self):
        """Returns a short name of the type of this entity"""
        long_name = self.type_name
        name = (
            long_name.split(".")[-1]
            .lower()
            .replace("factory_manager", "")
            .replace("manager", "")
        )
        name = (
            name.replace("factory", "")
            .replace("agent", "")
            .replace("miner", "")
            .replace("consumer", "")
        )
        if long_name.startswith("jnegmas"):
            name = f"j-{name}"
        name = name.strip("_")
        return name


class BulletinBoard(Entity, EventSource, ConfigReader):
    """The bulletin-board which carries all public information. It consists of sections each with a dictionary of records.

    """

    def __getstate__(self):
        return self.name, self._data

    def __setstate__(self, state):
        name, self._data = state
        super().__init__(name=name)

    def __init__(self, name: str = None):
        """
        Constructor

        Args:
            name: BulletinBoard name
        """
        super().__init__(name=name)
        self._data: Dict[str, Dict[str, Any]] = {}

    def add_section(self, name: str) -> None:
        """
        Adds a section to the bulletin Board

        Args:
            name: Section name

        Returns:

        """
        self._data[name] = {}

    def query(
        self, section: Optional[Union[str, List[str]]], query: Any, query_keys=False
    ) -> Optional[Dict[str, Any]]:
        """
        Returns all records in the given section/sections of the bulletin-board that satisfy the query

        Args:
            section: Either a section name, a list of sections or None specifying ALL public sections (see remarks)
            query: The query which is USUALLY a dict with conditions on it when querying values and a RegExp when
            querying keys
            query_keys: Whether the query is to be applied to the keys or values.

        Returns:

            - A dictionary with key:value pairs giving all records that satisfied the given requirements.

        Remarks:

            - A public section is a section with a name that does not start with an underscore
            - If a set of sections is given, and two records in different sections had the same key, only one of them
              will be returned
            - Key queries use regular expressions and match from the beginning using the standard re.match function

        """
        if section is None:
            return self.query(
                section=[_ for _ in self._data.keys() if not _.startswith("_")],
                query=query,
                query_keys=query_keys,
            )

        if isinstance(section, Iterable) and not isinstance(section, str):
            results = [
                self.query(section=_, query=query, query_keys=query_keys)
                for _ in section
            ]
            if len(results) == 0:
                return dict()
            final: Dict[str, Any] = {}
            for _ in results:
                final.update(_)
            return final

        sec = self._data.get(section, None)
        if sec is None:
            return {}
        if query is None:
            return copy.deepcopy(sec)
        if query_keys:
            return {k: v for k, v in sec.items() if re.match(str(query), k) is not None}
        return {k: v for k, v in sec.items() if BulletinBoard.satisfies(v, query)}

    @classmethod
    def satisfies(cls, value: Any, query: Any) -> bool:
        method = getattr(value, "satisfies", None)
        if method is not None and isinstance(method, Callable):
            return method(query)
        if isinstance(value, dict) and isinstance(query, dict):
            for k, v in query.items():
                if value.get(k, None) != v:
                    return False
        else:
            raise ValueError(
                f"Cannot check satisfaction of {type(query)} against value {type(value)}"
            )
        return True

    def read(self, section: str, key: str) -> Any:
        """
        Reads the value associated with given key

        Args:
            section: section name
            key: key

        Returns:

            Content of that key in the bulletin-board

        """
        sec = self._data.get(section, None)
        if sec is None:
            return None
        return sec.get(key, None)

    def record(self, section: str, value: Any, key: Optional[str] = None) -> None:
        """
        Records data in the given section of the bulletin-board

        Args:
            section: section name (can contain subsections separated by '/')
            key: The key
            value: The value

        """
        if key is None:
            try:
                skey = str(hash(value))
            except:
                skey = str(uuid.uuid4())
        else:
            skey = key
        self._data[section][skey] = value
        self.announce(
            Event("new_record", data={"section": section, "key": skey, "value": value})
        )

    def remove(
        self,
        section: Optional[Union[List[str], str]],
        *,
        query: Optional[Any] = None,
        key: str = None,
        query_keys: bool = False,
        value: Any = None,
    ) -> bool:
        """
        Removes a value or a set of values from the bulletin Board

        Args:
            section: The section
            query: the query to use to select what to remove
            key: the key to remove (no need to give a full query)
            query_keys: Whether to apply the query (if given) to keys or values
            value: Value to be removed

        Returns:
            bool: Success of failure
        """
        if section is None:
            return self.remove(
                section=[_ for _ in self._data.keys() if not _.startswith("_")],
                query=query,
                key=key,
                query_keys=query_keys,
            )

        if isinstance(section, Iterable) and not isinstance(section, str):
            return all(
                self.remove(section=_, query=query, key=key, query_keys=query_keys)
                for _ in section
            )

        sec = self._data.get(section, None)
        if sec is None:
            return False
        if value is not None:
            for k, v in sec.items():
                if v == value:
                    key = k
                    break
        if key is not None:
            try:
                self.announce(
                    Event(
                        "will_remove_record",
                        data={"section": sec, "key": key, "value": sec[key]},
                    )
                )
                sec.pop(key, None)
                return True
            except KeyError:
                return False

        if query is None:
            return False

        if query_keys:
            keys = [k for k, v in sec.items() if re.match(str(query), k) is not None]
        else:
            keys = [k for k, v in sec.items() if v.satisfies(query)]
        if len(keys) == 0:
            return False
        for k in keys:
            self.announce(
                Event(
                    "will_remove_record",
                    data={"section": sec, "key": k, "value": sec[k]},
                )
            )
            sec.pop(k, None)
        return True

    @property
    def data(self):
        """This property is intended for use only by the world manager. No other agent is allowed to use it"""
        return self._data


def safe_min(a, b):
    """Returns min(a, b) assuming None is less than anything."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


@dataclass
class NegotiationInfo:
    """Saves information about a negotiation"""

    mechanism: Optional[Mechanism]
    partners: List["Agent"]
    annotation: Dict[str, Any]
    issues: List["Issue"]
    requested_at: int
    rejectors: Optional[List["Agent"]] = None


class MechanismFactory:
    """A mechanism creation class. It can invite agents to join a mechanism and then run it."""

    def __init__(
        self,
        world: "World",
        mechanism_name: str,
        mechanism_params: Dict[str, Any],
        issues: List["Issue"],
        req_id: str,
        caller: "Agent",
        partners: List["Agent"],
        roles: Optional[List[str]] = None,
        annotation: Dict[str, Any] = None,
        neg_n_steps: int = None,
        neg_time_limit: int = None,
        neg_step_time_limit=None,
        log_ufuns_file=None,
    ):
        self.mechanism_name, self.mechanism_params = mechanism_name, mechanism_params
        self.caller = caller
        self.partners = partners
        self.roles = roles
        self.annotation = annotation
        self.neg_n_steps = neg_n_steps
        self.neg_time_limit = neg_time_limit
        self.neg_step_time_limit = neg_step_time_limit
        self.world = world
        self.req_id = req_id
        self.issues = issues
        self.mechanism = None
        self.log_ufuns_file = log_ufuns_file

    def _create_negotiation_session(
        self,
        mechanism: Mechanism,
        responses: Iterator[Tuple[Negotiator, str]],
        partners: List["Agent"],
    ) -> Mechanism:
        if self.neg_n_steps is not None:
            mechanism.ami.n_steps = self.neg_n_steps
        if self.neg_time_limit is not None:
            mechanism.ami.time_limit = self.neg_time_limit
        if self.neg_step_time_limit is not None:
            mechanism.ami.step_time_limit = self.neg_step_time_limit
        for partner in partners:
            mechanism.register_listener(event_type="negotiation_end", listener=partner)

        ufun = []
        if self.log_ufuns_file is not None:
            for outcome in mechanism.discrete_outcomes(astype=dict):
                record = {"mechanism_id": mechanism.id, "outcome": outcome}
                ufun.append(record)
        for i, (partner_, (_negotiator, _role)) in enumerate(zip(partners, responses)):
            if self.log_ufuns_file is not None:
                for record in ufun:
                    record[f"agent{i}"] = partner_.name
                    # record[f"agent_type{i}"] = partner_.type_name
                    # record[f"negotiator{i}"] = _negotiator.name
                    record[f"reserved{i}"] = _negotiator.reserved_value
                    record[f"u{i}"] = _negotiator.utility_function(record["outcome"])
            mechanism.add(negotiator=_negotiator, role=_role)

        if self.log_ufuns_file is not None:
            for record in ufun:
                outcome = record.pop("outcome", {})
                record.update(outcome)
            add_records(self.log_ufuns_file, ufun)

        return mechanism

    def _start_negotiation(
        self,
        mechanism_name,
        mechanism_params,
        roles,
        caller,
        partners,
        annotation,
        issues,
        req_id,
    ) -> Optional[NegotiationInfo]:
        """Tries to prepare the negotiation to start by asking everyone to join"""
        mechanisms = self.world.mechanisms
        if issues is None:
            caller.on_neg_request_rejected_(req_id=req_id, by=None)
            return None
        if (
            mechanisms is not None
            and mechanism_name is not None
            and mechanism_name not in mechanisms.keys()
        ):
            caller.on_neg_request_rejected_(req_id=req_id, by=None)
            return None
        if mechanisms is not None and mechanism_name is not None:
            mechanism_name = mechanisms[mechanism_name].pop(
                PROTOCOL_CLASS_NAME_FIELD, mechanism_name
            )
        if mechanism_params is None:
            mechanism_params = {}
        if mechanisms and mechanisms.get(mechanism_name, None) is not None:
            mechanism_params.update(mechanisms[mechanism_name])
        # mechanism_params = {k: v for k, v in mechanism_params.items() if k != PROTOCOL_CLASS_NAME_FIELD}
        mechanism_params["n_steps"] = self.neg_n_steps
        mechanism_params["time_limit"] = self.neg_time_limit
        mechanism_params["step_time_limit"] = self.neg_step_time_limit
        mechanism_params["issues"] = issues
        mechanism_params["annotation"] = annotation
        mechanism_params["name"] = "-".join(_.id for _ in partners)
        if mechanism_name is None:
            if mechanisms is not None and len(mechanisms) == 1:
                mechanism_name = list(mechanisms.keys())[0]
            else:
                mechanism_name = "negmas.sao.SAOMechanism"
            if mechanisms and mechanisms.get(mechanism_name, None) is not None:
                mechanism_params.update(mechanisms[mechanism_name])
        try:
            mechanism = instantiate(class_name=mechanism_name, **mechanism_params)
        except:
            mechanism = None
            self.world.logerror(
                f"Failed to create {mechanism_name} with params {mechanism_params}"
            )
        self.mechanism = mechanism
        if mechanism is None:
            return None

        if roles is None:
            roles = [None] * len(partners)

        partner_names = [p.id for p in partners]
        responses = [
            partner.respond_to_negotiation_request_(
                initiator=caller.id,
                partners=partner_names,
                issues=issues,
                annotation=annotation,
                role=role,
                mechanism=mechanism.ami,
                req_id=req_id if partner == caller else None,
            )
            for role, partner in zip(roles, partners)
        ]
        if not all(responses):
            rejectors = [p for p, response in zip(partners, responses) if not response]
            caller.on_neg_request_rejected_(req_id=req_id, by=[_.id for _ in rejectors])
            self.world.loginfo(f"{caller.id} request was rejected by {rejectors}")
            return NegotiationInfo(
                mechanism=None,
                partners=partners,
                annotation=annotation,
                issues=issues,
                rejectors=rejectors,
                requested_at=self.world.current_step,
            )
        mechanism = self._create_negotiation_session(
            mechanism=mechanism, responses=zip(responses, roles), partners=partners
        )
        neg_info = NegotiationInfo(
            mechanism=mechanism,
            partners=partners,
            annotation=annotation,
            issues=issues,
            requested_at=self.world.current_step,
        )
        caller.on_neg_request_accepted_(req_id=req_id, mechanism=mechanism.ami)
        self.world.loginfo(f"{caller.id} request was accepted")
        return neg_info

    def init(self) -> Optional[NegotiationInfo]:
        return self._start_negotiation(
            mechanism_name=self.mechanism_name,
            mechanism_params=self.mechanism_params,
            roles=self.roles,
            caller=self.caller,
            partners=self.partners,
            annotation=self.annotation,
            issues=self.issues,
            req_id=self.req_id,
        )


class AgentWorldInterface:
    """Agent World Interface class"""

    def __getstate__(self):
        return self._world, self.agent.id

    def __setstate__(self, state):
        self._world, agent_id = state
        self.agent = self._world.agents[agent_id]

    def __init__(self, world: "World", agent: "Agent"):
        self._world, self.agent = world, agent

    def execute(
        self, action: Action, callback: Callable[[Action, bool], Any] = None
    ) -> bool:
        """Executes an action in the world simulation"""
        return self._world.execute(action=action, agent=self.agent, callback=callback)

    @property
    def state(self) -> Any:
        """Returns the private state of the agent in that world"""
        return self._world.get_private_state(self.agent)

    @property
    def relative_time(self) -> float:
        """Relative time of the simulation going from 0 to 1"""
        return self._world.relative_time

    @property
    def current_step(self) -> int:
        """Current simulation step"""
        return self._world.current_step

    @property
    def n_steps(self) -> int:
        """Number of steps in a simulation"""
        return self._world.n_steps

    @property
    def default_signing_delay(self) -> int:
        return self._world.default_signing_delay

    def request_negotiation_about(
        self,
        issues: List[Issue],
        partners: List[str],
        req_id: str,
        roles: List[str] = None,
        annotation: Optional[Dict[str, Any]] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
    ) -> bool:
        """
        Requests to start a negotiation with some other agents

        Args:
            req_id:
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object

        Returns:

            List["Agent"] the list of partners who rejected the negotiation if any. If None then the negotiation was
            accepted. If empty then the negotiation was not started from the world manager


        Remarks:

            - The function will create a request ID that will be used in callbacks `on_neg_request_accepted` and
            `on_neg_request_rejected`


        """
        partner_agents = [self._world.agents[_] for _ in partners]
        return self._world.request_negotiation_about(
            req_id=req_id,
            caller=self.agent,
            partners=partner_agents,
            roles=roles,
            issues=issues,
            annotation=annotation,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
        )

    def loginfo(self, msg: str) -> None:
        """
        Logs an INFO message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.loginfo(msg)

    def logwarning(self, msg: str) -> None:
        """
        Logs a WARNING message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logwarning(msg)

    def logdebug(self, msg: str) -> None:
        """
        Logs a WARNING message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logdebug(msg)

    def logerror(self, msg: str) -> None:
        """
        Logs a WARNING message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logerror(msg)

    def bb_query(
        self, section: Optional[Union[str, List[str]]], query: Any, query_keys=False
    ) -> Optional[Dict[str, Any]]:
        """
        Returns all records in the given section/sections of the bulletin-board that satisfy the query

        Args:
            section: Either a section name, a list of sections or None specifying ALL public sections (see remarks)
            query: The query which is USUALLY a dict with conditions on it when querying values and a RegExp when
            querying keys
            query_keys: Whether the query is to be applied to the keys or values.

        Returns:

            - A dictionary with key:value pairs giving all records that satisfied the given requirements.

        Remarks:

            - A public section is a section with a name that does not start with an underscore
            - If a set of sections is given, and two records in different sections had the same key, only one of them
              will be returned
            - Key queries use regular expressions and match from the beginning using the standard re.match function

        """
        return self._world.bulletin_board.query(
            section=section, query=query, query_keys=query_keys
        )

    def bb_read(self, section: str, key: str) -> Optional[Any]:
        """
        Reads the value associated with given key from the bulletin board

        Args:
            section: section name
            key: key

        Returns:

            Content of that key in the bulletin-board

        """
        return self._world.bulletin_board.read(section=section, key=key)

    def bb_record(self, section: str, value: Any, key: Optional[str] = None) -> None:
        """
        Records data in the given section of the bulletin board

        Args:
            section: section name (can contain subsections separated by '/')
            key: The key
            value: The value

        """
        return self._world.bulletin_board.record(section=section, value=value, key=key)

    def bb_remove(
        self,
        section: Optional[Union[List[str], str]],
        *,
        query: Optional[Any] = None,
        key: str = None,
        query_keys: bool = False,
        value: Any = None,
    ) -> bool:
        """
        Removes a value or a set of values from the bulletin Board

        Args:
            section: The section
            query: the query to use to select what to remove
            key: the key to remove (no need to give a full query)
            query_keys: Whether to apply the query (if given) to keys or values
            value: Value to be removed

        Returns:
            bool: Success of failure
        """
        return self._world.bulletin_board.remove(
            section=section, query=query, key=key, query_keys=query_keys, value=value
        )

    class Java:
        implements = ["jnegmas.situated.AgentWorldInterface"]


class World(EventSink, EventSource, ConfigReader, ABC):
    """Base world class encapsulating a world that runs a simulation with several agents interacting within some
    dynamically changing environment.

    A world maintains its own session.

    """

    def __getstate__(self):
        state = self.__dict__.copy()
        if "logger" in state.keys():
            state.pop("logger", None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.logger = create_loggers(
            file_name=self.log_file_name,
            module_name=None,
            screen_level=self.log_screen_level if self.log_to_screen else None,
            file_level=self.log_file_level,
            app_wide_log_file=True,
        )

    def __init__(
        self,
        bulletin_board: BulletinBoard = None,
        n_steps=10000,
        time_limit=60 * 60,
        negotiation_speed=None,
        neg_n_steps=100,
        neg_time_limit=3 * 60,
        neg_step_time_limit=60,
        default_signing_delay=0,
        breach_processing=BreachProcessing.VICTIM_THEN_PERPETRATOR,
        log_file_name="",
        log_to_screen: bool = False,
        log_file_level=logging.DEBUG,
        log_screen_level=logging.ERROR,
        log_ufuns_file=None,
        mechanisms: Dict[str, Dict[str, Any]] = None,
        awi_type: str = "negmas.situated.AgentWorldInterface",
        start_negotiations_immediately: bool = False,
        save_signed_contracts: bool = True,
        save_cancelled_contracts: bool = True,
        save_negotiations: bool = True,
        save_resolved_breaches: bool = True,
        save_unresolved_breaches: bool = True,
        name=None,
    ):
        """

        Args:
            bulletin_board:
            n_steps: Total simulation time in steps
            time_limit: Real-time limit on the simulation
            negotiation_speed: The number of negotiation steps per simulation step. None means infinite
            neg_n_steps: Maximum number of steps allowed for a negotiation.
            neg_step_time_limit: Time limit for single step of the negotiation protocol.
            neg_time_limit: Real-time limit on each single negotiation
            name: Name of the simulator
        """
        super().__init__()
        self.log_file_name = log_file_name
        self.log_file_level = log_file_level
        self.log_screen_level = log_screen_level
        self.log_to_screen = log_to_screen
        self.logger = create_loggers(
            file_name=log_file_name,
            module_name=None,
            screen_level=log_screen_level if log_to_screen else None,
            file_level=log_file_level,
            app_wide_log_file=True,
        )
        self.bulletin_board: BulletinBoard = bulletin_board
        self.set_bulletin_board(bulletin_board=bulletin_board)
        self._negotiations: Dict[str, NegotiationInfo] = {}
        self.unsigned_contracts: Dict[int, Set[Contract]] = defaultdict(set)
        self.breach_processing = breach_processing
        self.n_steps = n_steps
        self.save_signed_contracts = save_signed_contracts
        self.save_cancelled_contracts = save_cancelled_contracts
        self.save_negotiations = save_negotiations
        self.save_resolved_breaches = save_resolved_breaches
        self.save_unresolved_breaches = save_unresolved_breaches
        self.current_step = 0
        self.negotiation_speed = negotiation_speed
        self.default_signing_delay = default_signing_delay
        self.time_limit = time_limit
        self.neg_n_steps = neg_n_steps
        self.neg_time_limit = neg_time_limit
        self.neg_step_time_limit = neg_step_time_limit
        self._entities: Dict[int, Set[Entity]] = defaultdict(set)
        self._negotiations: Dict[str, NegotiationInfo] = {}
        self._start_time = -1
        self._log_ufuns_file = log_ufuns_file
        if isinstance(mechanisms, Collection) and not isinstance(mechanisms, dict):
            mechanisms = dict(zip(mechanisms, [dict()] * len(mechanisms)))
        self.mechanisms: Optional[Dict[str, Dict[str, Any]]] = mechanisms
        self.awi_type = get_class(awi_type, scope=globals())
        self.name = (
            name
            if name is not None
            else unique_name(base=self.__class__.__name__, add_time=True, rand_digits=5)
        )
        self._stats: Dict[str, List[Any]] = defaultdict(list)
        self.__n_negotiations = 0
        self.__n_contracts_signed = 0
        self.__n_contracts_concluded = 0
        self.__n_contracts_cancelled = 0
        self._saved_contracts: Dict[str, Dict[str, Any]] = {}
        self._saved_negotiations: Dict[str, Dict[str, Any]] = {}
        self._saved_breaches: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Agent] = {}
        self.immediate_negotiations = start_negotiations_immediately
        self.loginfo(f"{self.name}: World Created")

    def loginfo(self, s: str) -> None:
        """logs info-level information

        Args:
            s (str): The string to log

        """
        self.logger.info(f"{self._log_header()}: " + s.strip())

    def logdebug(self, s) -> None:
        """logs debug-level information

        Args:
            s (str): The string to log

        """
        self.logger.debug(f"{self._log_header()}: " + s.strip())

    def logwarning(self, s) -> None:
        """logs warning-level information

        Args:
            s (str): The string to log

        """
        self.logger.warning(f"{self._log_header()}: " + s.strip())

    def logerror(self, s) -> None:
        """logs error-level information

        Args:
            s (str): The string to log

        """
        self.logger.error(f"{self._log_header()}: " + s.strip())

    def set_bulletin_board(self, bulletin_board):
        self.bulletin_board = (
            bulletin_board if bulletin_board is not None else BulletinBoard()
        )
        self.bulletin_board.add_section("breaches")
        self.bulletin_board.add_section("stats")
        self.bulletin_board.add_section("settings")

    @property
    def time(self) -> Optional[float]:
        """Elapsed time since mechanism started in seconds. None if the mechanism did not start running"""
        if self._start_time is None:
            return None
        return time.monotonic() - self._start_time

    @property
    def remaining_time(self) -> Optional[float]:
        """Returns remaining time in seconds. None if no time limit is given."""
        if self.time_limit is None:
            return None
        limit = self.time_limit - (time.monotonic() - self._start_time)
        if limit < 0.0:
            return 0.0

        return limit

    @property
    def relative_time(self) -> float:
        """Returns a number between ``0`` and ``1`` indicating elapsed relative time or steps."""
        if self.time_limit is None and self.n_steps is None:
            return 0.0
        relative_step = (
            (self.current_step + 1) / self.n_steps
            if self.n_steps is not None
            else np.nan
        )
        relative_time = (
            self.time / self.time_limit if self.time_limit is not None else np.nan
        )
        return max([relative_step, relative_time])

    @property
    def remaining_steps(self) -> Optional[int]:
        """Returns the remaining number of steps until the end of the mechanism run. None if unlimited"""
        if self.n_steps is None:
            return None

        return self.n_steps - self.current_step

    def _register_breach(self, breach: Breach) -> None:
        # we do not report breachs with no victims
        if breach.victims is None or len(breach.victims) < 1:
            return
        self.bulletin_board.record(
            section="breaches", key=breach.id, value=self._breach_record(breach)
        )

    @property
    def saved_negotiations(self) -> List[Dict[str, Any]]:
        return list(self._saved_negotiations.values())

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    def step(self) -> bool:
        """A single simulation step"""
        if self.current_step >= self.n_steps:
            self.logerror(
                f"Asked  to step after the simulation ({self.n_steps}). Will just ignore this"
            )
            return False
        self.loginfo(
            f"{len(self._negotiations)} Negotiations/{len(self._entities)} _entities"
        )

        def _run_negotiations(n_steps: Optional[int] = None):
            """ Runs all bending negotiations """
            mechanisms = list(
                (k, _.mechanism) for k, _ in self._negotiations.items() if _ is not None
            )
            current_step = 0
            while len(mechanisms) > 0:
                random.shuffle(mechanisms)
                for puuid, mechanism in mechanisms:
                    result = mechanism.step()
                    agreement, is_broken = result.agreement, result.broken
                    if agreement is not None or is_broken:  # or not mechanism.running:
                        negotiation = self._negotiations.get(puuid, None)
                        if agreement is None:
                            self._register_failed_negotiation(
                                mechanism.ami, negotiation
                            )
                        else:
                            self._register_contract(mechanism.ami, negotiation)
                        if negotiation:
                            self._negotiations.pop(mechanism.uuid, None)
                mechanisms = list(
                    (k, _.mechanism)
                    for k, _ in self._negotiations.items()
                    if _ is not None
                )
                current_step += 1
                if n_steps is not None and current_step >= n_steps:
                    break

        # initialize stats
        # ----------------
        n_new_contract_executions = 0
        n_new_breaches = 0
        n_cancelled = 0
        activity_level = 0

        self._pre_step_stats()
        self._stats["n_registered_negotiations_before"].append(len(self._negotiations))

        # sign contacts that are to be signed in this step
        # ------------------------------------------------
        # this is done first to allow these contracts to be executed immediately
        unsigned = self.unsigned_contracts.get(self.current_step, None)
        signed = []
        cancelled = []
        if unsigned:
            for contract in unsigned:
                rejectors = self._sign_contract(contract=contract)
                if rejectors is not None and len(rejectors) == 0:
                    signed.append(contract)
                else:
                    cancelled.append(contract)
            for contract in signed:
                self.on_contract_signed(contract=contract)
            for contract in cancelled:
                self.on_contract_cancelled(contract=contract)

        # run all negotiations before the simulation step if that is the meeting strategy
        # --------------------------------------------------------------------------------
        if self.negotiation_speed is None:
            _run_negotiations()

        # Step all entities in the world once:
        # ------------------------------------
        # note that entities are simulated in the partial-order specified by their priority value
        tasks: List[Entity] = []
        for priority in sorted(self._entities.keys()):
            tasks += [_ for _ in self._entities[priority]]

        for task in tasks:
            task.step_()

        # execute contracts that are executable at this step
        # --------------------------------------------------
        current_contracts = [
            _ for _ in self._get_executable_contracts() if _.nullified_at is None
        ]
        if len(current_contracts) > 0:
            # remove expired contracts
            executed = set()
            current_contracts = self._contract_execution_order(current_contracts)

            for contract in current_contracts:
                contract_breaches = self._execute_contract(contract)
                if len(contract_breaches) < 1:
                    self._saved_contracts[contract.id]["executed"] = True
                    self._saved_contracts[contract.id]["breaches"] = ""
                    executed.add(contract)
                    n_new_contract_executions += 1
                    activity_level += self._contract_size(contract)
                    for partner in contract.partners:
                        self.agents[partner].on_contract_executed(contract)
                else:
                    self._saved_contracts[contract.id]["executed"] = False
                    self._saved_contracts[contract.id]["breaches"] = "; ".join(
                        str(_) for _ in current_contracts
                    )
                    for b in contract_breaches:
                        self._saved_breaches[b.id] = b.as_dict()
                    resolution = self._process_breach(contract, list(contract_breaches))
                    n_new_breaches += 1 - int(resolution is not None)
                    self._complete_contract_execution(
                        contract, list(contract_breaches), resolution
                    )
                    for partner in contract.partners:
                        self.agents[partner].on_contract_breached(
                            contract, list(contract_breaches), resolution
                        )
            self._delete_executed_contracts()  # note that all contracts even breached ones are to be deleted

        # World Simulation Step:
        # ----------------------
        # The world manager should execute a single step of simulation in this function. It may lead to new negotiations
        self._simulation_step()

        # do one step of all negotiations if that is specified as the meeting strategy
        if self.negotiation_speed is not None:
            _run_negotiations(n_steps=self.negotiation_speed)

        # remove all negotiations that are completed
        # ------------------------------------------
        completed = list(
            k
            for k, _ in self._negotiations.items()
            if _ is not None and _.mechanism.completed
        )
        for key in completed:
            self._negotiations.pop(key, None)

        # update stats
        # ------------
        n_total_contracts = n_new_contract_executions + n_new_breaches
        self._stats["n_contracts_executed"].append(n_new_contract_executions)
        self._stats["n_contracts_cancelled"].append(self.__n_contracts_cancelled)
        self._stats["n_breaches"].append(n_new_breaches)
        self._stats["breach_level"].append(
            n_new_breaches / n_total_contracts if n_total_contracts > 0 else np.nan
        )
        self._stats["n_contracts_signed"].append(self.__n_contracts_signed)
        self._stats["n_contracts_concluded"].append(self.__n_contracts_concluded)
        self._stats["n_negotiations"].append(self.__n_negotiations)
        self._stats["n_registered_negotiations_after"].append(len(self._negotiations))
        self._stats["activity_level"].append(activity_level)
        self._post_step_stats()
        self.__n_negotiations = 0
        self.__n_contracts_signed = 0
        self.__n_contracts_concluded = 0
        self.__n_contracts_cancelled = 0
        self.current_step += 1

        # always indicate that the simulation is to continue
        return True

    @property
    def saved_breaches(self) -> List[Dict[str, Any]]:
        return list(self._saved_breaches.values())

    @property
    def resolved_breaches(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_breaches.values() if _["resolved"])

    @property
    def unresolved_breaches(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_breaches.values() if not _["resolved"])

    def run(self):
        """Runs the simulation until it ends"""
        self._start_time = time.monotonic()
        for _ in range(self.n_steps):
            if (
                self.time_limit is not None
                and (time.monotonic() - self._start_time) >= self.time_limit
            ):
                break
            if not self.step():
                break

    def register(self, x: "Entity", simulation_priority: int = 0):
        """
        Registers an entity in the world so it can be looked up by name. Should not be called directly

        Args:
            x: The entity to be registered
            simulation_priority: The simulation periority. Entities with lower periorities will be stepped first during

        Returns:

        """
        # super().register(x) # If we inherit from session, we can do that but it is not needed as we do not do string
        # based resolution now
        if hasattr(x, "_world"):
            x._world = self
        if hasattr(x, "step_"):
            self._entities[simulation_priority].add(x)

    def join(self, x: "Agent", simulation_priority: int = 0):
        """Add an agent to the world.

        Args:
            x: The agent to be registered
            simulation_priority: The simulation periority. Entities with lower periorities will be stepped first during

        Returns:

        """
        self.loginfo(f"{x.name} joined")
        self.register(x, simulation_priority=simulation_priority)
        self.agents[x.id] = x
        x.awi = self.awi_type(self, x)

    def _register_negotiation(
        self,
        mechanism_name,
        mechanism_params,
        roles,
        caller,
        partners,
        annotation,
        issues,
        req_id,
        run_to_completion=False,
    ) -> Optional[NegotiationInfo]:
        """Registers a negotiation and returns the list of rejectors if any or None"""
        factory = MechanismFactory(
            world=self,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
            issues=issues,
            req_id=req_id,
            caller=caller,
            partners=partners,
            roles=roles,
            annotation=annotation,
            neg_n_steps=self.neg_n_steps,
            neg_time_limit=self.neg_time_limit,
            neg_step_time_limit=self.neg_step_time_limit,
            log_ufuns_file=self._log_ufuns_file,
        )
        neg = factory.init()
        if neg is None:
            return None
        if neg.mechanism is None:
            return neg
        self.__n_negotiations += 1
        if run_to_completion:
            pass
        else:
            self._negotiations[neg.mechanism.uuid] = neg
            if self.immediate_negotiations:
                mechanism = neg.mechanism
                puuid = mechanism.uuid
                result = mechanism.step()
                agreement, is_broken = result.agreement, result.broken
                if agreement is not None or is_broken:  # or not mechanism.running:
                    negotiation = self._negotiations.get(puuid, None)
                    if agreement is None:
                        self._register_failed_negotiation(mechanism.ami, negotiation)
                    else:
                        self._register_contract(mechanism.ami, negotiation)
                    if negotiation:
                        self._negotiations.pop(mechanism.uuid, None)
        # self.loginfo(
        #    f'{caller.id} request was accepted')
        return neg

    def request_negotiation_about(
        self,
        req_id: str,
        caller: "Agent",
        issues: List[Issue],
        partners: List["Agent"],
        roles: List[str] = None,
        annotation: Optional[Dict[str, Any]] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
    ) -> bool:
        """
        Requests to start a negotiation with some other agents

        Args:
            req_id: An ID For the request that is unique to the caller
            caller: The agent requesting the negotiation
            partners: The list of partners that the agent wants to negotiate with. Roles will be determined by these agents.
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object

        Returns:

            None. The caller will be informed by a callback function `on_neg_request_accepted` or
            `on_neg_request_rejected` about the status of the negotiation.

        """
        self.loginfo(
            f"{caller.name} requested "
            f"{mechanism_name}[{mechanism_params}] with {[_.name for _ in partners]} (ID {req_id})"
        )
        neg = self._register_negotiation(
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
            roles=roles,
            caller=caller,
            partners=partners,
            annotation=annotation,
            issues=issues,
            req_id=req_id,
            run_to_completion=False,
        )
        success = neg is not None and neg.mechanism is not None

        return success

    def run_negotiation(
        self,
        caller: "Agent",
        issues: Collection[Issue],
        partners: Collection["Agent"],
        roles: Collection[str] = None,
        annotation: Optional[Dict[str, Any]] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
    ) -> Optional[Tuple[Contract, AgentMechanismInterface]]:
        """
        Requests to start a negotiation with some other agents

        Args:
            caller: The agent requesting the negotiation
            partners: The list of partners that the agent wants to negotiate with. Roles will be determined by these agents.
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object

        Returns:

            Contract: The agreed upon contract if negotiation was successful otherwise, None.

        """
        self.loginfo(
            f"{caller.name} requested immediate negotiation "
            f"{mechanism_name}[{mechanism_params}] with {[_.name for _ in partners]}"
        )
        neg = self._register_negotiation(
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
            roles=roles,
            caller=caller,
            partners=partners,
            annotation=annotation,
            issues=issues,
            req_id=None,
            run_to_completion=True,
        )
        if neg and neg.mechanism:
            mechanism = neg.mechanism
            mechanism.run()
            if mechanism.agreement is None:
                contract = None
                self._register_failed_negotiation(
                    mechanism=mechanism.ami, negotiation=neg
                )
            else:
                contract = self._register_contract(
                    mechanism=mechanism.ami, negotiation=neg, force_signature_now=True
                )
            return contract, mechanism.ami
        return None

    def _log_header(self):
        if self.time is None:
            return f"{self.name} (not started)"
        return f"{self.current_step}/{self.n_steps} [{self.relative_time:0.2%}]"

    def _register_contract(
        self, mechanism, negotiation, force_signature_now=False
    ) -> Optional[Contract]:
        partners = negotiation.partners
        if self.save_negotiations:
            _stats = {
                "final_status": "failed",
                "partners": [_.id for _ in partners],
                "partner_types": [_.__class__.__name__ for _ in partners],
                "ended_at": self.current_step,
                "requested_at": negotiation.requested_at,
                "mechanism_type": mechanism.__class__.__name__,
            }
            _stats.update(mechanism.state.__dict__)
            self._saved_negotiations[mechanism.id] = _stats
        if mechanism.state.agreement is None or negotiation is None:
            return None
        signed_at = None
        if force_signature_now:
            signing_delay = 0
        else:
            signing_delay = mechanism.state.agreement.get(
                "signing_delay", self.default_signing_delay
            )
        contract = Contract(
            partners=list(_.id for _ in partners),
            annotation=negotiation.annotation,
            issues=negotiation.issues,
            agreement=mechanism.state.agreement,
            concluded_at=self.current_step,
            to_be_signed_at=self.current_step + signing_delay,
            signed_at=signed_at,
            mechanism_state=mechanism.state,
        )
        if not force_signature_now:
            self.on_contract_concluded(
                contract, to_be_signed_at=self.current_step + signing_delay
            )
        for partner in partners:
            partner.on_negotiation_success_(contract=contract, mechanism=mechanism)
        if signing_delay == 0:
            rejectors = self._sign_contract(contract)
            signed = rejectors is not None and len(rejectors) == 0
            if signed and not force_signature_now:
                self.on_contract_signed(contract=contract)
            sign_status = (
                "signed"
                if signed
                else f"cancelled by {rejectors if rejectors is not None else 'error!!'}"
            )
        else:
            sign_status = f"to be signed at {contract.to_be_signed_at}"
            self.on_contract_cancelled(contract=contract)
        if negotiation.annotation is not None:
            annot_ = dict(
                zip(
                    negotiation.annotation.keys(),
                    (str(_) for _ in negotiation.annotation.values()),
                )
            )
        else:
            annot_ = ""
        self.logdebug(
            f'Contract<{"immediate-signature-forced" if force_signature_now else ""}> [{sign_status}]: {[_.name for _ in partners]}'
            f" > {str(mechanism.state.agreement)} on annotation {annot_}"
        )
        return contract

    def _register_failed_negotiation(self, mechanism, negotiation) -> None:
        partners = negotiation.partners
        mechanism_state = mechanism.state
        annotation = negotiation.annotation
        if self.save_negotiations:
            _stats = {
                "final_status": "failed",
                "partners": [_.id for _ in partners],
                "partner_types": [_.__class__.__name__ for _ in partners],
                "ended_at": self.current_step,
                "requested_at": negotiation.requested_at,
                "mechanism_type": mechanism.__class__.__name__,
            }
            _stats.update(mechanism.state.__dict__)
            self._saved_negotiations[mechanism.id] = _stats
        for partner in partners:
            partner.on_negotiation_failure_(
                partners=[_.id for _ in partners],
                annotation=annotation,
                mechanism=mechanism,
                state=mechanism_state,
            )

        self.logdebug(
            f"Negotiation failure between {[_.name for _ in partners]}"
            f" on annotation {negotiation.annotation} "
        )

    def _sign_contract(self, contract: Contract) -> Optional[List[str]]:
        """Called to sign a contract and returns whether or not it was signed"""
        # if self._contract_finalization_time(contract) >= self.n_steps or \
        #     self._contract_execution_time(contract) < self.current_step:
        #     return None
        partners = [self.agents[_] for _ in contract.partners]
        signatures = list(
            zip(
                partners,
                (partner.sign_contract(contract=contract) for partner in partners),
            )
        )
        rejectors = [partner for partner, signature in signatures if signature is None]
        if len(rejectors) == 0:
            contract.signatures = [
                Signature(id=a.id, signature=s) for a, s in signatures
            ]
            contract.signed_at = self.current_step
            for partner in partners:
                partner.on_contract_signed_(contract=contract)
        else:
            # if self.save_cancelled_contracts:
            record = self._contract_record(contract)
            record["signed"] = False
            record["executed"] = None
            record["breaches"] = ""
            self._saved_contracts[contract.id] = record
            # else:
            #     self._saved_contracts.pop(contract.id, None)
            for partner in partners:
                partner.on_contract_cancelled_(
                    contract=contract, rejectors=[_.id for _ in rejectors]
                )
        return [_.id for _ in rejectors]

    def on_contract_signed(self, contract: Contract) -> None:
        """Called to add a contract to the existing set of contract after it is signed

        Args:

            contract: The contract to add

        Remarks:

            - By default this function just adds the contract to the set of contracts maintaned by the world.
            - You should ALWAYS call this function when overriding it.

        """
        self.__n_contracts_signed += 1
        self.unsigned_contracts[self.current_step].remove(contract)
        record = self._contract_record(contract)
        if self.save_signed_contracts:
            record["signed"] = True
            record["executed"] = None
            record["breaches"] = ""
            self._saved_contracts[contract.id] = record
        else:
            self._saved_contracts.pop(contract.id, None)

    def on_contract_cancelled(self, contract):
        """Called whenever a concluded contract is not signed (cancelled)

                Args:

                    contract: The contract to add

                Remarks:

                    - By default this function just adds the contract to the set of contracts maintaned by the world.
                    - You should ALWAYS call this function when overriding it.

        """
        self.__n_contracts_cancelled += 1
        unsigned = self.unsigned_contracts.get(self.current_step, None)
        if unsigned is None:
            return
        try:
            unsigned.remove(contract)
        except KeyError:
            pass

    @property
    def saved_contracts(self) -> List[Dict[str, Any]]:
        return list(self._saved_contracts.values())

    @property
    def signed_contracts(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_contracts.values() if _["signed"])

    @property
    def cancelled_contracts(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_contracts.values() if not _["signed"])

    def on_contract_concluded(self, contract: Contract, to_be_signed_at: int) -> None:
        """Called to add a contract to the existing set of contract after it is signed

        Args:

            contract: The contract to add
            to_be_signed_at: The timestep at which the contract is to be signed

        Remarks:

            - By default this function just adds the contract to the set of contracts maintaned by the world.
            - You should ALWAYS call this function when overriding it.

        """
        self.__n_contracts_concluded += 1
        self.unsigned_contracts[to_be_signed_at].add(contract)
        # self.saved_contracts.append(self._contract_record(contract))

    def save_config(self, file_name: str):
        """
        Saves the config of the world as a yaml file

        Args:
            file_name: Name of file to save the config to

        Returns:

        """
        with open(file_name, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    @abstractmethod
    def _delete_executed_contracts(self) -> None:
        """Called after processing executable contracts at every simulation step to delete processed contracts"""

    @abstractmethod
    def _get_executable_contracts(self) -> Collection[Contract]:
        """Called at every time-step to get the contracts that are `executable` at this point of the simulation"""

    @abstractmethod
    def _post_step_stats(self):
        """Called at the end of the simulation step to update all stats"""
        pass

    @abstractmethod
    def _pre_step_stats(self):
        """Called at the beginning of the simulation step to prepare stats or update them"""
        pass

    @abstractmethod
    def _contract_execution_order(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        """Orders the contracts in a specific time-step that are about to be executed"""

    @abstractmethod
    def _contract_record(self, contract: Contract) -> Dict[str, Any]:
        """Converts a contract to a record suitable for permanent storage"""

    @abstractmethod
    def _breach_record(self, breach: Breach) -> Dict[str, Any]:
        """Converts a breach to a record suitable for storage during the simulation"""

    @abstractmethod
    def _execute_contract(self, contract: Contract) -> Set[Breach]:
        """
        Tries to execute the contract

        Args:
            contract:

        Returns:
            Set[Breach]: The set of breaches committed if any. If there are no breaches return an empty set

        Remarks:

            - You must call super() implementation of this method before doing anything

        """
        self.loginfo(f"Executing {str(contract)}")
        return set()

    @abstractmethod
    def _complete_contract_execution(
        self, contract: Contract, breaches: List[Breach], resolution: Contract
    ) -> None:
        """
        Called after breach resolution is completed for contracts for which some potential breaches occurred.

        Args:
            contract: The contract considered.
            breaches: The list of potential breaches that was generated by `_execute_contract`.
            resolution: The agreed upon resolution

        Returns:

        """

    def _process_breach(
        self, contract: Contract, breaches: List[Breach], force_immediate_signing=True
    ) -> Optional[Contract]:
        new_contract = None

        # calculate total breach level
        total_breach_levels = defaultdict(int)
        for breach in breaches:
            total_breach_levels[breach.perpetrator] += breach.level

        # give agents the chance to set renegotiation agenda in ascending order of their total breach levels
        for agent_name, _ in sorted(
            zip(total_breach_levels.keys(), total_breach_levels.values()),
            key=lambda x: x[1],
        ):
            agent = self.agents[agent_name]
            agenda = agent.set_renegotiation_agenda(
                contract=contract, breaches=breaches
            )
            if agenda is None:
                continue
            negotiators = []
            for partner in contract.partners:
                negotiator = self.agents[partner].respond_to_renegotiation_request(
                    contract=contract, breaches=breaches, agenda=agenda
                )
                if negotiator is None:
                    break
                negotiators.append(negotiator)
            else:
                # everyone accepted this renegotiation
                results = self.run_negotiation(
                    caller=agent,
                    issues=agenda.issues,
                    partners=[self.agents[_] for _ in contract.partners],
                )
                if results is not None:
                    new_contract, mechanism = results
                    self._register_contract(
                        mechanism=mechanism,
                        negotiation=None,
                        force_signature_now=force_immediate_signing,
                    )
                    break

        if new_contract is not None:
            for breach in breaches:
                if self.save_resolved_breaches:
                    self._saved_breaches[breach.id]["resolved"] = True
                else:
                    self._saved_breaches.pop(breach.id, None)
            return new_contract
        for breach in breaches:
            if self.save_unresolved_breaches:
                self._saved_breaches[breach.id]["resolved"] = False
            else:
                self._saved_breaches.pop(breach.id, None)
            self._register_breach(breach)
        return None

    @abstractmethod
    def execute(
        self, action: Action, agent: "Agent", callback: Callable = None
    ) -> bool:
        """Executes the given action by the given agent"""

    @abstractmethod
    def get_private_state(self, agent: "Agent") -> dict:
        """Reads the private state of the given agent"""

    @abstractmethod
    def _simulation_step(self):
        """A single step of the simulation if any"""

    @abstractmethod
    def _contract_finalization_time(self, contract: Contract) -> int:
        """
        Returns the time at which the given contract will complete execution
        Args:
            contract:

        Returns:

        """

    @abstractmethod
    def _contract_execution_time(self, contract: Contract) -> int:
        """
        Returns the time at which the given contract will start execution
        Args:
            contract:

        Returns:

        """

    @abstractmethod
    def _contract_size(self, contract: Contract) -> float:
        """
        Returns an estimation of the **activity level** associated with this contract. Higher is better
        Args:
            contract:

        Returns:

        """


RunningNegotiationInfo = namedtuple(
    "RunningNegotiationInfo", ["negotiator", "annotation", "uuid", "extra"]
)
"""Keeps track of running negotiations for an agent"""

NegotiationRequestInfo = namedtuple(
    "NegotiationRequestInfo",
    ["partners", "issues", "annotation", "uuid", "negotiator", "extra"],
)
"""Keeps track to negotiation requests that an agent sent"""


class Agent(Entity, EventSink, ConfigReader, Notifier, ABC):
    """Base class for all agents that can run within a `World` and engage in situated negotiations"""

    def __getstate__(self):
        return self.name, self.awi

    def __setstate__(self, state):
        name, awi = state
        super().__init__(name=name)
        self._awi = awi

    def __init__(self, name: str = None):
        super().__init__(name=name)
        self._running_negotiations: Dict[str, RunningNegotiationInfo] = {}
        self._requested_negotiations: Dict[str, NegotiationRequestInfo] = {}
        self.contracts: List[Contract] = []
        self._unsigned_contracts: Set[Contract] = set()
        self._awi: AgentWorldInterface = None

    @property
    def unsigned_contracts(self) -> List[Contract]:
        """
        All contracts that are not yet signed.
        """
        return list(self._unsigned_contracts)

    @property
    def requested_negotiations(self) -> List[NegotiationRequestInfo]:
        """The negotiations currently requested by the agent.

        Returns:

            A list of negotiation request information objects (`NegotiationRequestInfo`)
        """
        return list(self._requested_negotiations.values())

    @property
    def running_negotiations(self) -> List[RunningNegotiationInfo]:
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

    def _add_negotiation_request_info(
        self,
        issues: List[Issue],
        partners: List[str],
        annotation: Optional[Dict[str, Any]],
        negotiator: Optional[Negotiator],
        extra: Optional[Dict[str, Any]],
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
            uuid=req_id,
        )
        return req_id

    def _request_negotiation(
        self,
        issues: List[Issue],
        partners: List[str],
        roles: List[str] = None,
        annotation: Optional[Dict[str, Any]] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
        negotiator: Negotiator = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Requests to start a negotiation with some other agents

        Args:
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object
            negotiator: My negotiator to use in this negotiation. Can be none
            extra: Any extra information I would like to keep to myself for this negotiation
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
        req_id = self._add_negotiation_request_info(
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
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
        )

    def init_(self):
        """Called to initialize the agent **after** the world is initialized. the AWI is accessible at this point."""
        self.init()

    def step_(self):
        """Called at every time-step. This function is called directly by the world."""
        self.step()

    def on_event(self, event: Event, sender: EventSource):
        if not isinstance(sender, Mechanism) and not isinstance(sender, Mechanism):
            raise ValueError(
                f"Sender of the negotiation end event is of type {sender.__class__.__name__} "
                f"not Mechanism!!"
            )
        if event.type == "negotiation_end":
            # will be sent by the World once a negotiation in which this agent is involved is completed            l
            mechanism_id = sender.id
            self._running_negotiations.pop(mechanism_id, None)

    # ------------------------------------------------------------------
    # EVENT CALLBACKS (Called by the `World` when certain events happen)
    # ------------------------------------------------------------------

    def on_neg_request_rejected_(self, req_id: str, by: Optional[List[str]]):
        """Called when a requested negotiation is rejected

        Args:
            req_id: The request ID passed to _request_negotiation
            by: A list of agents that refused to participate or None if the failure was for another reason


        """
        self.on_neg_request_rejected(req_id, by)
        self._requested_negotiations.pop(req_id, None)

    def on_neg_request_accepted_(self, req_id: str, mechanism: AgentMechanismInterface):
        """Called when a requested negotiation is accepted"""
        self.on_neg_request_accepted(req_id, mechanism)
        neg = self._requested_negotiations[req_id].negotiator
        annotation = self._requested_negotiations[req_id].annotation
        self._running_negotiations[mechanism.id] = RunningNegotiationInfo(
            extra=self._requested_negotiations[req_id].extra,
            negotiator=neg,
            annotation=annotation,
            uuid=req_id,
        )
        self._requested_negotiations.pop(req_id, None)

    def on_negotiation_failure_(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""
        self.on_negotiation_failure(partners, annotation, mechanism, state)
        self._running_negotiations.pop(mechanism.id, None)

    def on_negotiation_success_(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""
        self.on_negotiation_success(contract, mechanism)
        self._unsigned_contracts.add(contract)
        self._running_negotiations.pop(mechanism.id, None)

    def on_contract_signed_(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""
        self.on_contract_signed(contract)
        if contract in self._unsigned_contracts:
            self._unsigned_contracts.remove(contract)
        self.contracts.append(contract)

    def on_contract_cancelled_(self, contract: Contract, rejectors: List[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""
        self.on_contract_cancelled(contract, rejectors)
        if contract in self._unsigned_contracts:
            self._unsigned_contracts.remove(contract)

    def respond_to_negotiation_request_(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
        """Called when a negotiation request is received"""
        if req_id is not None:
            info = self._requested_negotiations.get(req_id, None)
            if info and info.negotiator is not None:
                return info.negotiator
        return self._respond_to_negotiation_request(
            initiator=initiator,
            partners=partners,
            issues=issues,
            annotation=annotation,
            mechanism=mechanism,
            role=role,
            req_id=req_id,
        )

    def __str__(self):
        return f"{self.name}"

    __repr__ = __str__

    @abstractmethod
    def step(self):
        """Called by the simulator at every simulation step"""

    @abstractmethod
    def init(self):
        """Called to initialize the agent **after** the world is initialized. the AWI is accessible at this point."""

    @abstractmethod
    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
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

    @abstractmethod
    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        """Called when a requested negotiation is rejected

        Args:
            req_id: The request ID passed to _request_negotiation
            by: A list of agents that refused to participate or None if the failure was for another reason


        """

    @abstractmethod
    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        """Called when a requested negotiation is accepted"""

    @abstractmethod
    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""

    @abstractmethod
    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""

    @abstractmethod
    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""

    @abstractmethod
    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""

    @abstractmethod
    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
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
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        """
        Called to respond to a renegotiation request

        Args:

            agenda:
            contract:
            breaches:

        Returns:

        """

    @abstractmethod
    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Called after the signing delay from contract conclusion to sign the contract. Contracts become binding
        only after they are signed."""
        return self.id

    @abstractmethod
    def on_contract_executed(self, contract: Contract) -> None:
        """
        Called after successful contract execution for which the agent is one of the partners.
        """

    @abstractmethod
    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """
        Called after complete processing of a contract that involved a breach.

        Args:
            contract: The contract
            breaches: All breaches committed (even if they were resolved)
            resolution: The resolution contract if re-negotiation was successful. None if not.
        """


def save_stats(world: World, log_dir: str, params: Dict[str, Any] = None):
    """
    Saves the statistics of a world run.

    Args:

        world:
        log_dir:
        params:

    Returns:

    """
    log_dir = Path(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    dump(params, log_dir / "params")
    dump(world.stats, log_dir / "stats")

    try:
        data = pd.DataFrame.from_dict(world.stats)
        data.to_csv(str(log_dir / "stats.csv"), index_label="index")
    except:
        pass

    if world.save_negotiations:
        if len(world.saved_negotiations) > 0:
            data = pd.DataFrame(world.saved_negotiations)
            data.to_csv(str(log_dir / "negotiations.csv"), index_label="index")
        else:
            with open(log_dir / "negotiations.csv", "w") as f:
                f.write("")

    if world.save_resolved_breaches or world.save_unresolved_breaches:
        if len(world.saved_breaches) > 0:
            data = pd.DataFrame(world.saved_breaches)
            data.to_csv(str(log_dir / "breaches.csv"), index_label="index")
        else:
            with open(log_dir / "breaches.csv", "w") as f:
                f.write("")

    if world.save_signed_contracts:
        if len(world.signed_contracts) > 0:
            data = pd.DataFrame(world.signed_contracts)
            data = data.sort_values(["delivery_time"])
            data = data.loc[
                :,
                [
                    "seller_type",
                    "buyer_type",
                    "seller_name",
                    "buyer_name",
                    "delivery_time",
                    "unit_price",
                    "quantity",
                    "product_name",
                    "n_neg_steps",
                    "signed_at",
                    "concluded_at",
                    "cfp",
                ],
            ]
            data.to_csv(str(log_dir / "signed_contracts.csv"), index_label="index")
        else:
            with open(log_dir / "signed_contracts.csv", "w") as f:
                f.write("")

    if world.save_cancelled_contracts:
        if len(world.cancelled_contracts) > 0:
            data = pd.DataFrame(world.cancelled_contracts)
            data = data.sort_values(["delivery_time"])
            data = data.loc[
                :,
                [
                    "seller_type",
                    "buyer_type",
                    "seller_name",
                    "buyer_name",
                    "delivery_time",
                    "unit_price",
                    "quantity",
                    "product_name",
                    "n_neg_steps",
                    "signed_at",
                    "concluded_at",
                    "cfp",
                ],
            ]
            data.to_csv(str(log_dir / "cancelled_contracts.csv"), index_label="index")
        else:
            with open(log_dir / "cancelled_contracts.csv", "w") as f:
                f.write("")

    if world.save_signed_contracts or world.save_cancelled_contracts:
        if len(world.saved_contracts) > 0:
            data = pd.DataFrame(world.saved_contracts)
            data = data.sort_values(["delivery_time"])
            data.to_csv(str(log_dir / "contracts_full_info.csv"), index_label="index")
            data = data.loc[
                :,
                [
                    "seller_type",
                    "buyer_type",
                    "seller_name",
                    "buyer_name",
                    "delivery_time",
                    "unit_price",
                    "quantity",
                    "product_name",
                    "n_neg_steps",
                    "signed_at",
                    "concluded_at",
                    "cfp",
                ],
            ]
            if world.save_signed_contracts and world.save_cancelled_contracts:
                data.to_csv(str(log_dir / "all_contracts.csv"), index_label="index")
        else:
            with open(log_dir / "contracts_full_info.csv", "w") as f:
                f.write("")
            if world.save_signed_contracts and world.save_cancelled_contracts:
                with open(log_dir / "all_contracts.csv", "w") as f:
                    f.write("")
