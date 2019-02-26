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
    #. run all `ActiveEntity` objects registered (i.e. all agents) in the predefined `simulation_order`.
    #. execute contracts that are executable at this time-step handling any breaches
    #. allow custom simulation steps to run (call `_simulation_step`)
    #. remove any negotiations that are completed!
    #. update basic stats
    #. update custom stats (call `_post_step_stats`)

"""
import concurrent.futures as futures
import copy
import itertools
import json
import os
import pathlib
import random
import re
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import Optional, List, Any, Tuple, Callable, Union, Iterable, Set, Iterator, Collection, Type, Sequence

import distributed
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing_extensions import Protocol

from negmas.common import MechanismInfo, MechanismState
from negmas.common import NamedObject
from negmas.events import Event, EventSource, EventSink, Notifier
from negmas.helpers import ConfigReader, LoggerMixin, instantiate, get_class, unique_name, import_by_name, \
    get_full_type_name
from negmas.mechanisms import MechanismProxy, Mechanism
from negmas.negotiators import NegotiatorProxy
from negmas.outcomes import OutcomeType, Issue

__all__ = [
    'Action',  # An action that an `Agent` can execute in the `World`.
    'Contract',  # A agreement definition which encapsulates an agreement with partners and extra information
    'Breach',  # A breach in executing a contract
    'BreachProcessing',
    'Agent',  # Negotiator capable of engaging in multiple negotiations
    'BulletinBoard',
    'World',
    'ActiveEntity',  # an entity that can be stepped by the simulator
    'Entity',
    'AgentWorldInterface',  # the interface though which an agent can interact with the world
    'NegotiationInfo',
    'RenegotiationRequest',
    'save_stats',
    'tournament',
    'WorldGenerator',
    'WorldRunResults',
    'TournamentResults',
    'run_world',
    'process_world_run',
]

PROTOCOL_CLASS_NAME_FIELD = '__mechanism_class_name'


@dataclass
class Action:
    """An action that an `Agent` can execute in a `World` through the `Simulator`."""
    type: str
    """Action name."""
    params: dict
    """Any extra parameters to be passed for the action."""

    def __str__(self):
        return f'{self.type}: {self.params}'


Signature = namedtuple('Signature', ['id', 'signature'])
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
    to_be_signed_at: Optional[int] = None
    """The time-step at which the contract should be signed"""
    signatures: List[Signature] = field(default_factory=list)
    """A list of signatures giving agent name, signature"""
    mechanism_state: MechanismState = None
    """The mechanism state at the contract conclusion"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""

    def __str__(self):
        return f'{", ".join(self.partners)} agreed on {str(self.agreement)}'

    def __hash__(self):
        """The hash depends only on the name"""
        return self.id.__hash__()


@dataclass
class Breach:
    contract: Contract
    """The agreement being breached"""
    perpetrator: 'Agent'
    """The agent committing the breach"""
    type: str
    """The type of the breach. Can be one of: `refusal`, `product`, `money`, `penalty`."""
    victims: List['Agent'] = field(default_factory=list)
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
        return f'Breach ({self.level} {self.type}) by {self.perpetrator.name} on {self.contract.id} at {self.step}'

    def as_dict(self):
        return {
            'contract': str(self.contract),
            'contract_id': self.contract.id,
            'type': self.type,
            'level': self.level,
            'id': self.id,
            'perpetrator': self.perpetrator.id,
            'perpetrator_type': self.perpetrator.__class__.__name__,
            'victims': [_.id for _ in self.victims],
            'victim_types': [_.__class__.__name__ for _ in self.victims],
            'step': self.step,
            'resolved': None,
        }


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
    publisher: 'Agent'
    issues: List[Issue]
    annotation: Dict[str, Any] = field(default_factory=dict)


class Entity(NamedObject):
    """Defines an entity that is a part of the world but does not participate in the simulation"""

    def __init__(self, name: str = None):
        super().__init__(name=name)
        self._world: Optional['World'] = None

    def init(self):
        """Will be called by the world once the world itself is initialized to initialize itself."""


class BulletinBoard(Entity, EventSource, ConfigReader):
    """The white-board which carries all public information. It consists of sections each with a dictionary of records.

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

    def query(self, section: Optional[Union[str, List[str]]], query: Any, query_keys=False) -> Optional[Dict[str, Any]]:
        """
        Returns all records in the given section/sections of the white-board that satisfy the query

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
            return self.query(section=[_ for _ in self._data.keys() if not _.startswith('_')]
                              , query=query, query_keys=query_keys)

        if isinstance(section, Iterable) and not isinstance(section, str):
            results = [self.query(section=_, query=query, query_keys=query_keys) for _ in section]
            if len(results) == 0:
                return dict()
            final: Dict[str, Any] = {}
            for _ in results:
                final.update(_)
            return final

        sec = self._data.get(section, None)
        if not sec:
            return None
        if query is None:
            return sec
        if query_keys:
            return {k: v for k, v in sec.items() if re.match(str(query), k) is not None}
        return {k: v for k, v in sec.items() if BulletinBoard.satisfies(v, query)}

    @classmethod
    def satisfies(cls, value: Any, query: Any) -> bool:
        method = getattr(value, 'satisfies', None)
        if method is not None and isinstance(method, Callable):
            return method(query)
        if isinstance(value, dict) and isinstance(query, dict):
            for k, v in query.items():
                if value.get(k, None) != v:
                    return False
        else:
            raise ValueError(f'Cannot check satisfaction of {type(query)} against value {type(value)}')
        return True

    def read(self, section: str, key: str) -> Optional[Any]:
        """
        Reads the value associated with given key

        Args:
            section: section name
            key: key

        Returns:

            Content of that key in the white-board

        """
        sec = self._data.get(section, None)
        if sec is None:
            return None
        return sec.get(key, None)

    def record(self, section: str, value: Any, key: Optional[str] = None) -> None:
        """
        Records data in the given section of the white-board

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
        self.announce(Event('new_record', data={'section': section, 'key': skey, 'value': value}))

    def remove(self, section: Optional[Union[List[str], str]], *
               , query: Optional[Any] = None, key: str = None, query_keys: bool = False
               , value: Any = None) -> bool:
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
            return self.remove(section=[_ for _ in self._data.keys() if not _.startswith('_')]
                               , query=query, key=key, query_keys=query_keys)

        if isinstance(section, Iterable) and not isinstance(section, str):
            return all(self.remove(section=_, query=query, key=key, query_keys=query_keys) for _ in section)

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
                self.announce(Event('will_remove_record', data={'section': sec, 'key': key, 'value': sec[key]}))
                del sec[key]
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
            self.announce(Event('will_remove_record', data={'section': sec, 'key': k, 'value': sec[k]}))
            del sec[k]
        return True


BulletinBoardProxy = BulletinBoard
"""A proxy to the bulletin board"""


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
    mechanism: Optional[MechanismProxy]
    partners: List['Agent']
    annotation: Dict[str, Any]
    issues: List['Issue']
    requested_at: int
    rejectors: Optional[List['Agent']] = None


class MechanismFactory:
    """A mechanism creation class. It can invite agents to join a mechanism and then run it."""

    def __init__(self, world: 'World', mechanism_name: str, mechanism_params: Dict[str, Any]
                 , issues: List['Issue'], req_id: str
                 , caller: 'Agent', partners: List['Agent'], roles: Optional[List[str]] = None
                 , annotation: Dict[str, Any] = None, neg_n_steps: int = None, neg_time_limit: int = None
                 ):
        self.mechanism_name, self.mechanism_params = mechanism_name, mechanism_params
        self.caller = caller
        self.partners = partners
        self.roles = roles
        self.annotation = annotation
        self.neg_n_steps = neg_n_steps
        self.neg_time_limit = neg_time_limit
        self.world = world
        self.req_id = req_id
        self.issues = issues
        self.mechanism = None

    def _create_negotiation_session(self, mechanism: MechanismProxy
                                    , responses: Iterator[Tuple[NegotiatorProxy, str]]
                                    , partners: List["Agent"]) -> MechanismProxy:
        if self.neg_n_steps is not None:
            mechanism.info.n_steps = self.neg_n_steps
        if self.neg_time_limit is not None:
            mechanism.info.time_limit = self.neg_time_limit
        for partner in partners:
            mechanism.register_listener(event_type='negotiation_end', listener=partner)
        for _negotiator, _role in responses:
            mechanism.add(negotiator=_negotiator, role=_role)
        return mechanism

    def _start_negotiation(self, mechanism_name, mechanism_params, roles, caller, partners,
                           annotation
                           , issues, req_id) -> Optional[NegotiationInfo]:
        """Tries to prepare the negotiation to start by asking everyone to join"""
        mechanisms = self.world.mechanisms
        if issues is None:
            caller.on_neg_request_rejected(req_id=req_id, by=None)
            return None
        if mechanisms is not None and mechanism_name not in mechanisms.keys():
            caller.on_neg_request_rejected(req_id=req_id, by=None)
            return None
        if mechanisms is not None:
            mechanism_name = mechanisms[mechanism_name].get(PROTOCOL_CLASS_NAME_FIELD, mechanism_name)
        if mechanism_params is None:
            mechanism_params = {}
        if mechanisms and mechanisms[mechanism_name]:
            mechanism_params.update(mechanisms[mechanism_name])
        mechanism_params = {k: v for k, v in mechanism_params.items() if k != PROTOCOL_CLASS_NAME_FIELD}
        mechanism_params['n_steps'] = self.neg_n_steps
        mechanism_params['time_limit'] = self.neg_time_limit
        mechanism_params['issues'] = issues
        mechanism_params['annotation'] = annotation
        mechanism_params['name'] = '-'.join(_.id for _ in partners)
        if mechanism_name is None:
            mechanism_name = 'negmas.sao.SAOMechanism'
        try:
            mechanism = instantiate(class_name=mechanism_name, **mechanism_params)
        except:
            mechanism = None
            self.world.logerror(f'Failed to create {mechanism_name} with params {mechanism_params}')
        self.mechanism = mechanism
        if mechanism is None:
            return None

        if roles is None:
            roles = [None] * len(partners)

        partner_names = [p.id for p in partners]
        responses = [partner.respond_to_negotiation_request(initiator=caller.id, partners=partner_names, issues=issues
                                                            , annotation=annotation, role=role, mechanism=mechanism
                                                            , req_id=req_id if partner == caller else None)
                     for role, partner in zip(roles, partners)]
        if not all(responses):
            rejectors = [p for p, response in zip(partners, responses) if not response]
            caller.on_neg_request_rejected(req_id=req_id, by=[_.id for _ in rejectors])
            self.world.loginfo(f'{caller.id} request was rejected by {rejectors}')
            return NegotiationInfo(mechanism=None, partners=partners, annotation=annotation, issues=issues
                                   , rejectors=rejectors, requested_at=self.world.current_step)
        mechanism = self._create_negotiation_session(mechanism=mechanism
                                                     , responses=zip(responses, roles), partners=partners)
        neg_info = NegotiationInfo(mechanism=mechanism, partners=partners, annotation=annotation, issues=issues
                                   , requested_at=self.world.current_step)
        caller.on_neg_request_accepted(req_id=req_id, mechanism=mechanism)
        self.world.loginfo(f'{caller.id} request was accepted')
        return neg_info

    def init(self) -> Optional[NegotiationInfo]:
        return self._start_negotiation(mechanism_name=self.mechanism_name, mechanism_params=self.mechanism_params
                                       , roles=self.roles, caller=self.caller, partners=self.partners
                                       , annotation=self.annotation, issues=self.issues
                                       , req_id=self.req_id)


class AgentWorldInterface:
    """Agent World Interface class"""

    def __getstate__(self):
        return self._world, self.agent.id

    def __setstate__(self, state):
        self._world, agent_id = state
        self.agent = self._world.agents[agent_id]

    def __init__(self, world: 'World', agent: 'Agent'):
        self._world, self.agent = world, agent

    def execute(self, action: Action, callback: Callable[[Action, bool], Any] = None) -> bool:
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
    def bulletin_board(self) -> BulletinBoardProxy:
        """The white-board"""
        return self._world.bulletin_board

    @property
    def default_signing_delay(self) -> int:
        return self._world.default_signing_delay

    def request_negotiation(self
                            , issues: List[Issue]
                            , partners: List[str]
                            , req_id: str
                            , roles: List[str] = None
                            , annotation: Optional[Dict[str, Any]] = None
                            , mechanism_name: str = None
                            , mechanism_params: Dict[str, Any] = None
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
        return self._world.request_negotiation(req_id=req_id, caller=self.agent
                                               , partners=partner_agents
                                               , roles=roles, issues=issues, annotation=annotation
                                               , mechanism_name=mechanism_name, mechanism_params=mechanism_params)

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


class World(EventSink, EventSource, ConfigReader, LoggerMixin, ABC):
    """Base world class encapsulating a world that runs a simulation with several agents interacting within some
    dynamically changing environment.

    A world maintains its own session.

    """

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state.keys():
            del state['logger']
        state['log_file_name'] = self.log_file_name
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        LoggerMixin.__init__(self, file_name=state['log_file_name'], screen_log=['screen_log'])
        if 'log_file_name' in state.keys():
            del self.__dict__['log_file_name']

    def __init__(self, bulletin_board: BulletinBoard = None
                 , n_steps=10000
                 , time_limit=60 * 60
                 , negotiation_speed=None
                 , neg_n_steps=100
                 , neg_time_limit=3 * 60
                 , default_signing_delay=0
                 , breach_processing=BreachProcessing.VICTIM_THEN_PERPETRATOR
                 , log_file_name=''
                 , mechanisms: Dict[str, Dict[str, Any]] = None
                 , screen_log: bool = False
                 , awi_type: str = 'negmas.apps.scml.AgentWorldInterface'
                 , start_negotiations_immediately: bool = False
                 , save_signed_contracts: bool = True
                 , save_cancelled_contracts: bool = True
                 , save_negotiations: bool = True
                 , save_resolved_breaches: bool = True
                 , save_unresolved_breaches: bool = True
                 , name=None
                 ):
        """

        Args:
            bulletin_board:
            n_steps: Total simulation time in steps
            time_limit: Real-time limit on the simulation
            negotiation_speed: The number of negotiation steps per simulation step. None means infinite
            neg_n_steps: Maximum number of steps allowed for a negotiation.
            neg_time_limit: Real-time limit on each single negotiation
            name: Name of the simulator
        """
        LoggerMixin.__init__(self, file_name=log_file_name, screen_log=screen_log)
        super().__init__()
        self.screen_log = screen_log
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
        self._entities: Dict[int, Set[ActiveEntity]] = defaultdict(set)
        self._negotiations: Dict[str, NegotiationInfo] = {}
        self._start_time = -1
        self.mechanisms: Optional[Dict[str, Dict[str, Any]]] = mechanisms
        self.awi_type = get_class(awi_type, scope=globals())
        self.name = name if name is not None else unique_name(base=self.__class__.__name__, add_time=True
                                                              , rand_digits=5)
        self._stats: Dict[str, List[Any]] = defaultdict(list)
        self.__n_negotiations = 0
        self.__n_contracts_signed = 0
        self.__n_contracts_concluded = 0
        self._saved_contracts: Dict[str, Dict[str, Any]] = {}
        self._saved_negotiations: Dict[str, Dict[str, Any]] = {}
        self._saved_breaches: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Agent] = {}
        self.immediate_negotiations = start_negotiations_immediately
        self.loginfo(f'{self.name}: World Created')

    def loginfo(self, s: str) -> None:
        """logs info-level information

        Args:
            s (str): The string to log

        """
        self.logger.info(f'{self._log_header()}: ' + s)

    def logdebug(self, s) -> None:
        """logs debug-level information

        Args:
            s (str): The string to log

        """
        self.logger.debug(f'{self._log_header()}: ' + s)

    def logwarning(self, s) -> None:
        """logs warning-level information

        Args:
            s (str): The string to log

        """
        self.logger.warning(f'{self._log_header()}: ' + s)

    def logerror(self, s) -> None:
        """logs error-level information

        Args:
            s (str): The string to log

        """
        self.logger.error(f'{self._log_header()}: ' + s)

    def set_bulletin_board(self, bulletin_board):
        self.bulletin_board = bulletin_board if bulletin_board is not None else BulletinBoard()
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
        relative_step = (self.current_step + 1) / self.n_steps if self.n_steps is not None else np.nan
        relative_time = self.time / self.time_limit if self.time_limit is not None else np.nan
        return max([relative_step, relative_time])

    @property
    def remaining_steps(self) -> Optional[int]:
        """Returns the remaining number of steps until the end of the mechanism run. None if unlimited"""
        if self.n_steps is None:
            return None

        return self.n_steps - self.current_step

    def _register_breach(self, breach: Breach) -> None:
        self.bulletin_board.record(section='breaches', key=breach.id, value=self._breach_record(breach))

    @property
    def saved_negotiations(self) -> List[Dict[str, Any]]:
        return list(self._saved_negotiations.values())

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    def step(self) -> bool:
        """A single simulation step"""
        if self.current_step >= self.n_steps:
            self.logerror(f'Asked  to step after the simulation ({self.n_steps}). Will just ignore this')
            return False
        self.loginfo(f'{len(self._negotiations)} Negotiations/{len(self._entities)} _entities')

        def _run_negotiations(n_steps: Optional[int] = None):
            """ Runs all bending negotiations """
            mechanisms = list((k, _.mechanism) for k, _ in self._negotiations.items() if _ is not None)
            current_step = 0
            while len(mechanisms) > 0:
                random.shuffle(mechanisms)
                for puuid, mechanism in mechanisms:
                    result = mechanism.step()
                    agreement, is_broken = result.agreement, result.broken
                    if agreement is not None or is_broken:  # or not mechanism.running:
                        negotiation = self._negotiations.get(puuid, None)
                        if agreement is None:
                            self._register_failed_negotiation(mechanism, negotiation)
                        else:
                            self._register_contract(mechanism, negotiation)
                        if negotiation:
                            del self._negotiations[mechanism.uuid]
                mechanisms = list((k, _.mechanism) for k, _ in self._negotiations.items() if _ is not None)
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
        self._stats['n_registered_negotiations_before'].append(len(self._negotiations))

        # sign contacts that are to be signed in this step
        # ------------------------------------------------
        # this is done first to allow these contracts to be executed immediately
        unsigned = self.unsigned_contracts.get(self.current_step, None)
        signed = []
        if unsigned:
            for contract in unsigned:
                if self._sign_contract(contract=contract):
                    signed.append(contract)
                else:
                    n_cancelled += 1
            for contract in signed:
                self.on_contract_signed(contract=contract)

        # run all negotiations before the simulation step if that is the meeting strategy
        # --------------------------------------------------------------------------------
        if self.negotiation_speed is None:
            _run_negotiations()

        # Step all entities in the world once:
        # ------------------------------------
        # note that entities are simulated in the partial-order specified by their priority value
        tasks: List[ActiveEntity] = []
        for priority in sorted(self._entities.keys()):
            tasks += [_ for _ in self._entities[priority]]

        for task in tasks:
            task.step()

        # execute contracts that are executable at this step
        # --------------------------------------------------
        current_contracts = self._get_executable_contracts()
        if len(current_contracts) > 0:
            # remove expired contracts
            executed = set()
            current_contracts = self._contract_execution_order(current_contracts)
            breached_contracts = []
            something_executed = True
            while something_executed:
                something_executed = False
                for contract in current_contracts:
                    contract_breaches = self._execute_contract(contract)
                    if len(contract_breaches) < 1:
                        self._saved_contracts[contract.id]['executed'] = True
                        self._saved_contracts[contract.id]['breaches'] = ''
                        executed.add(contract)
                        n_new_contract_executions += 1
                        activity_level += self._contract_size(contract)
                        # something_executed = True # @todo I am disabling this for now as this approach may result in multiple loans
                    else:
                        self._saved_contracts[contract.id]['executed'] = False
                        self._saved_contracts[contract.id]['breaches'] = '; '.join(str(_) for _ in current_contracts)
                        breached_contracts.append((contract, contract_breaches))
                        for b in contract_breaches:
                            self._saved_breaches[b.id] = b.as_dict()
                current_contracts = [_[0] for _ in breached_contracts]
            for contract, contract_breaches in breached_contracts:
                self._process_breach(contract, list(contract_breaches))

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
        completed = list(k for k, _ in self._negotiations.items() if _ is not None and _.mechanism.completed)
        for key in completed:
            del self._negotiations[key]

        # update stats
        # ------------
        n_total_contracts = n_new_contract_executions + n_new_breaches
        self._stats['n_contracts_executed'].append(n_new_contract_executions)
        self._stats['n_contracts_cancelled'].append(n_cancelled)
        self._stats['n_breaches'].append(n_new_breaches)
        self._stats['breach_level'].append(n_new_breaches / n_total_contracts
                                           if n_total_contracts > 0 else np.nan)
        self._stats['n_contracts_signed'].append(self.__n_contracts_signed)
        self._stats['n_contracts_concluded'].append(self.__n_contracts_concluded)
        self._stats['n_negotiations'].append(self.__n_negotiations)
        self._stats['n_registered_negotiations_after'].append(len(self._negotiations))
        self._stats['activity_level'].append(activity_level)
        self._post_step_stats()
        self.__n_negotiations = 0
        self.__n_contracts_signed = 0
        self.__n_contracts_concluded = 0
        self.current_step += 1

        # always indicate that the simulation is to continue
        return True

    @property
    def saved_breaches(self) -> List[Dict[str, Any]]:
        return list(self._saved_breaches.values())

    @property
    def resolved_breaches(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_breaches.values() if _['resolved'])

    @property
    def unresolved_breaches(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_breaches.values() if not _['resolved'])

    def run(self):
        """Runs the simulation until it ends"""
        self._start_time = time.monotonic()
        for _ in range(self.n_steps):
            if self.time_limit is not None and (time.monotonic() - self._start_time) >= self.time_limit:
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
        # based resoluton now
        x._world = self
        if isinstance(x, ActiveEntity):
            self._entities[simulation_priority].add(x)

    def join(self, x: 'Agent', simulation_priority: int = 0):
        """Add an agent to the world.

        Args:
            x: The agent to be registered
            simulation_priority: The simulation periority. Entities with lower periorities will be stepped first during

        Returns:

        """
        self.loginfo(f'{x.name} joined')
        self.register(x, simulation_priority=simulation_priority)
        self.agents[x.id] = x
        x.awi = self.awi_type(self, x)

    def _register_negotiation(self, mechanism_name, mechanism_params, roles, caller, partners
                              , annotation
                              , issues, req_id, run_to_completion=False) -> Optional[NegotiationInfo]:
        """Registers a negotiation and returns the list of rejectors if any or None"""
        factory = MechanismFactory(world=self, mechanism_name=mechanism_name, mechanism_params=mechanism_params
                                   , issues=issues, req_id=req_id, caller=caller, partners=partners
                                   , roles=roles, annotation=annotation
                                   , neg_n_steps=self.neg_n_steps, neg_time_limit=self.neg_time_limit)
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
                        self._register_failed_negotiation(mechanism, negotiation)
                    else:
                        self._register_contract(mechanism, negotiation)
                    if negotiation:
                        del self._negotiations[mechanism.uuid]
        # self.loginfo(
        #    f'{caller.id} request was accepted')
        return neg

    def request_negotiation(self, req_id: str
                            , caller: "Agent"
                            , issues: List[Issue]
                            , partners: List["Agent"]
                            , roles: List[str] = None
                            , annotation: Optional[Dict[str, Any]] = None
                            , mechanism_name: str = None
                            , mechanism_params: Dict[str, Any] = None) -> bool:
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
        self.loginfo(f'{caller.name} requested '
                     f'{mechanism_name}[{mechanism_params}] with {[_.name for _ in partners]} (ID {req_id})')
        neg = self._register_negotiation(mechanism_name=mechanism_name, mechanism_params=mechanism_params
                                         , roles=roles, caller=caller
                                         , partners=partners, annotation=annotation, issues=issues
                                         , req_id=req_id, run_to_completion=False)
        success = neg is not None and neg.mechanism is not None

        return success

    def run_negotiation(self, caller: "Agent"
                        , issues: Collection[Issue]
                        , partners: Collection["Agent"]
                        , roles: Collection[str] = None
                        , annotation: Optional[Dict[str, Any]] = None
                        , mechanism_name: str = None
                        , mechanism_params: Dict[str, Any] = None) -> Optional[Contract]:
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
        self.loginfo(f'{caller.name} requested immediate negotiation '
                     f'{mechanism_name}[{mechanism_params}] with {[_.name for _ in partners]}')
        contract = None
        neg = self._register_negotiation(mechanism_name=mechanism_name, mechanism_params=mechanism_params, roles=roles
                                         , caller=caller, partners=partners, annotation=annotation, issues=issues
                                         , req_id=None, run_to_completion=True)
        if neg and neg.mechanism:
            mechanism = neg.mechanism
            mechanism.run()
            if mechanism.agreement is None:
                contract = None
                self._register_failed_negotiation(mechanism=mechanism, negotiation=neg)
            else:
                contract = self._register_contract(mechanism=mechanism, negotiation=neg, force_signature_now=True)
        return contract

    def _log_header(self):
        if self.time is None:
            return f'{self.name} (not started)'
        return f'{self.current_step}/{self.n_steps} [{self.relative_time:0.2%}]'

    def _register_contract(self, mechanism, negotiation, force_signature_now=False) -> Optional[Contract]:
        partners = negotiation.partners
        if self.save_negotiations:
            _stats = {'final_status': 'failed', 'partners': [_.id for _ in partners]
                , 'partner_types': [_.__class__.__name__ for _ in partners]
                , 'ended_at': self.current_step, 'requested_at': negotiation.requested_at
                , 'mechanism_type': mechanism.__class__.__name__}
            _stats.update(mechanism.state.__dict__)
            self._saved_negotiations[mechanism.id] = _stats
        if mechanism.agreement is None or negotiation is None:
            return None
        signed_at = None
        if force_signature_now:
            signing_delay = 0
        else:
            signing_delay = mechanism.agreement.get('signing_delay', self.default_signing_delay)
        contract = Contract(
            partners=list(_.id for _ in partners),
            annotation=negotiation.annotation,
            issues=negotiation.issues,
            agreement=mechanism.agreement,
            concluded_at=self.current_step,
            to_be_signed_at=self.current_step + signing_delay,
            signed_at=signed_at,
            mechanism_state=mechanism.state
        )
        self.on_contract_concluded(contract, to_be_signed_at=self.current_step + signing_delay)
        for partner in partners:
            partner.on_negotiation_success(contract=contract, mechanism=mechanism.info)
        if signing_delay == 0:
            signed = self._sign_contract(contract)
            if signed:
                self.on_contract_signed(contract=contract)

            sign_status = "signed" if signed else "cancelled"
        else:
            sign_status = f"to be signed at {contract.to_be_signed_at}"
        if negotiation.annotation is not None:
            annot_ = dict(zip(negotiation.annotation.keys(), (str(_) for _ in negotiation.annotation.values())))
        else:
            annot_ = ''
        self.logdebug(f'Contract [{sign_status}]: {[_.name for _ in partners]}'
                      f' > {str(mechanism.agreement)} on annotation {annot_}')
        return contract

    def _register_failed_negotiation(self, mechanism, negotiation) -> None:
        partners = negotiation.partners
        mechanism_state = mechanism.state
        annotation = negotiation.annotation
        if self.save_negotiations:
            _stats = {'final_status': 'failed', 'partners': [_.id for _ in partners]
                , 'partner_types': [_.__class__.__name__ for _ in partners]
                , 'ended_at': self.current_step, 'requested_at': negotiation.requested_at
                , 'mechanism_type': mechanism.__class__.__name__}
            _stats.update(mechanism.state.__dict__)
            self._saved_negotiations[mechanism.id] = _stats
        for partner in partners:
            partner.on_negotiation_failure(partners=[_.id for _ in partners], annotation=annotation
                                           , mechanism=mechanism.info, state=mechanism_state)

        self.logdebug(f'Negotiation failure between {[_.name for _ in partners]}'
                      f' on annotation {negotiation.annotation} ')

    def _sign_contract(self, contract: Contract) -> bool:
        """Called to sign a contract and returns whether or not it was signed"""
        if self._contract_finalization_time(contract) >= self.n_steps or \
            self._contract_execution_time(contract) < self.current_step:
            return False
        partners = [self.agents[_] for _ in contract.partners]
        signatures = list(zip(partners, (partner.sign_contract(contract=contract) for partner in partners)))
        rejectors = [partner for partner, signature in signatures if signature is None]
        if len(rejectors) == 0:
            contract.signatures = [Signature(id=a.id, signature=s) for a, s in signatures]
            contract.signed_at = self.current_step
            for partner in partners:
                partner.on_contract_signed(contract=contract)
        else:
            if self.save_cancelled_contracts:
                record = self._contract_record(contract)
                record['signed'] = False
                record['executed'] = None
                record['breaches'] = ''
                self._saved_contracts[contract.id] = record
            else:
                del self._saved_contracts[contract.id]
            for partner in partners:
                partner.on_contract_cancelled(contract=contract, rejectors=[_.id for _ in rejectors])
        return len(rejectors) == 0

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
            record['signed'] = True
            record['executed'] = None
            record['breaches'] = ''
            self._saved_contracts[contract.id] = record
        else:
            del self._saved_contracts[contract.id]

    @property
    def saved_contracts(self) -> List[Dict[str, Any]]:
        return list(self._saved_contracts.values())

    @property
    def signed_contracts(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_contracts.values() if _['signed'])

    @property
    def cancelled_contracts(self) -> List[Dict[str, Any]]:
        return list(_ for _ in self._saved_contracts.values() if not _['signed'])

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
    def _contract_execution_order(self, contracts: Collection[Contract]) -> Collection[Contract]:
        """Orders the contracts in a specific time-step that are about to be executed"""

    @abstractmethod
    def _contract_record(self, contract: Contract) -> Dict[str, Any]:
        """Converts a contract to a record suitable for permenant storage"""

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
            Set[Breach]: The set of breaches sommitted if any. If there are no breaches return an empty set

        Remarks:

            - You must call super() implementation of this method before doing anything

        """
        self.loginfo(f'Executing {str(contract)}')
        return set()

    def _process_breach(self, contract: Contract, breaches: List[Breach]) -> bool:
        resolved = False
        # @todo add breach processing
        if resolved:
            for breach in breaches:
                if self.save_resolved_breaches:
                    self._saved_breaches[breach.id]['resolved'] = True
                else:
                    del self._saved_breaches[breach.id]
            return True
        for breach in breaches:
            if self.save_unresolved_breaches:
                self._saved_breaches[breach.id]['resolved'] = False
            else:
                del self._saved_breaches[breach.id]
            self._register_breach(breach)
        return False

    @abstractmethod
    def execute(self, action: Action, agent: 'Agent', callback: Callable = None) -> bool:
        """Executes the given action by the given agent"""

    @abstractmethod
    def get_private_state(self, agent: 'Agent') -> dict:
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


class ActiveEntity(Entity):
    """Defines an entity that is a part of the world and participates in the simulation"""

    @abstractmethod
    def step(self):
        """Called by the simulator at every simulation step"""


RunningNegotiationInfo = namedtuple('RunningNegotiationInfo', ['negotiator', 'annotation', 'uuid', 'extra'])
"""Keeps track of running negotiations for an agent"""

NegotiationRequestInfo = namedtuple('NegotiationRequestInfo', ['partners', 'issues', 'annotation', 'uuid'
    , 'negotiator', 'extra'])
"""Keeps track to negotiation requests that an agent sent"""


class Agent(ActiveEntity, EventSink, ConfigReader, Notifier, ABC):
    """Base class for all agents that can run within a `World` and engage in situated negotiations"""

    def __getstate__(self):
        return self.name, self.awi

    def __setstate__(self, state):
        name, awi = state
        super().__init__(name=name)
        self.awi = awi

    def __init__(self, name: str = None):
        super().__init__(name=name)
        self.running_negotiations: Dict[str, RunningNegotiationInfo] = {}
        self._neg_requests: Dict[str, NegotiationRequestInfo] = {}
        self.contracts: Set[Contract] = set()
        self._unsigned_contracts: Set[Contract] = set()
        self.awi: AgentWorldInterface = None

    def request_negotiation(self
                            , issues: List[Issue]
                            , partners: List[str]
                            , roles: List[str] = None
                            , annotation: Optional[Dict[str, Any]] = None
                            , mechanism_name: str = None
                            , mechanism_params: Dict[str, Any] = None
                            , negotiator: NegotiatorProxy = None
                            , extra: Optional[Dict[str, Any]] = None
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
            `on_neg_request_rejected`


        """
        req_id = str(uuid.uuid4())
        self._neg_requests[req_id] = NegotiationRequestInfo(issues=issues, partners=partners, annotation=annotation
                                                            , negotiator=negotiator, extra=extra, uuid=req_id)
        return self.awi.request_negotiation(issues=issues, partners=partners, req_id=req_id, roles=roles
                                            , annotation=annotation, mechanism_name=mechanism_name
                                            , mechanism_params=mechanism_params)

    def init(self):
        """Called to initialize the agent **after** the world is initialized. the AWI is accessible at this point."""
        pass

    def on_event(self, event: Event, sender: EventSource):
        if not isinstance(sender, MechanismProxy) and not isinstance(sender, Mechanism):
            raise ValueError(f'Sender of the negotiation end event is of type {sender.__class__.__name__} '
                             f'not MechanismProxy!!')
        if event.type == 'negotiation_end':
            # will be sent by the World once a negotiation in which this agent is involved is completed            l
            mechanism_id = sender.id
            negotiation = self.running_negotiations.get(mechanism_id, None)
            # if negotiation is None:
            #    print('Cannot find the negotiation')
            if negotiation:
                del self.running_negotiations[mechanism_id]

    # ------------------------------------------------------------------
    # EVENT CALLBACKS (Called by the `World` when certain events happen)
    # ------------------------------------------------------------------

    def respond_to_negotiation_request(self, initiator: str, partners: List[str], issues: List[Issue]
                                       , annotation: Dict[str, Any], mechanism: MechanismProxy, role: Optional[str]
                                       , req_id: str) -> Optional[NegotiatorProxy]:
        """Called by the mechanism to ask for joining a negotiation. The agent can refuse by returning a None"""

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any], mechanism: MechanismInfo
                               , state: MechanismState) -> None:
        """Called whenever a negotiation ends without agreement"""
        if mechanism.id in self.running_negotiations.keys():
            del self.running_negotiations[mechanism.id]

    def on_negotiation_success(self, contract: Contract, mechanism: MechanismInfo) -> None:
        """Called whenever a negotiation ends with agreement"""
        self._unsigned_contracts.add(contract)
        if mechanism.id in self.running_negotiations.keys():
            del self.running_negotiations[mechanism.id]

    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""
        if contract in self._unsigned_contracts:
            self._unsigned_contracts.remove(contract)
        self.contracts.add(contract)

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""
        if contract in self._unsigned_contracts:
            self._unsigned_contracts.remove(contract)

    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Called after the signing delay from contract conclusion to sign the contract. Contracts become binding
        only after they are signed."""
        return self.id

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        """Called when a requested negotiation is rejected

        Args:
            req_id: The request ID passed to request_negotiation
            by: A list of agents that refused to participate or None if the failure was for another reason


        """
        del self._neg_requests[req_id]

    def on_neg_request_accepted(self, req_id: str, mechanism: MechanismProxy):
        """Called when a requested negotiation is accepted"""
        neg, annotation = self._neg_requests[req_id].negotiator, self._neg_requests[req_id].annotation
        self.running_negotiations[mechanism.uuid] = RunningNegotiationInfo(extra=self._neg_requests[req_id].extra
                                                                           , negotiator=neg, annotation=annotation
                                                                           , uuid=req_id)
        del self._neg_requests[req_id]

    def __str__(self):
        return f'{self.name}'

    __repr__ = __str__

    @abstractmethod
    def set_renegotiation_agenda(self, contract: Contract
                                 , breaches: List[Dict[str, Any]]) -> Optional[RenegotiationRequest]:
        """
        Received by partners in ascending order of their total breach levels in order to set the
        renegotiation agenda when contract execution fails

        Args:
            contract:
            breaches:

        Returns:

        """

    @abstractmethod
    def respond_to_renegotiation_request(self, contract: Contract, breaches: List[Dict[str, Any]]
                                         , agenda: RenegotiationRequest) -> Optional[NegotiatorProxy]:
        """
        Called to respond to a renegotiation request

        Args:
            agenda:
            contract:
            breaches:

        Returns:

        """

    @abstractmethod
    def on_renegotiation_request(self, contract: Contract, cfp: "CFP", partner: str) -> bool:
        """Called to respond to a re-negotiation request"""


def save_stats(world: World, log_dir: str, params: Dict[str, Any] = None):
    log_dir = Path(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if params is not None:
        with open(log_dir / 'params.csv', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

    with open(log_dir / 'stats.csv', 'w') as f:
        json.dump(world.stats, f, indent=4, sort_keys=True)
    try:
        data = pd.DataFrame.from_dict(world.stats)
        data.to_csv(str(log_dir / 'stats.csv'), index_label='index')
    except:
        pass
    if len(world.saved_negotiations) > 0:
        data = pd.DataFrame(world.saved_negotiations)
        data.to_csv(str(log_dir / 'negotiations.csv'), index_label='index')
    else:
        with open(log_dir / 'negotiations.csv', 'w') as f:
            f.write('')

    if len(world.saved_breaches) > 0:
        data = pd.DataFrame(world.saved_breaches)
        data.to_csv(str(log_dir / 'breaches.csv'), index_label='index')
    else:
        with open(log_dir / 'breaches.csv', 'w') as f:
            f.write('')

    if len(world.signed_contracts) > 0:
        data = pd.DataFrame(world.signed_contracts)
        data = data.sort_values(['delivery_time'])
        data = data.loc[:, ['seller_type', 'buyer_type', 'seller_name', 'buyer_name', 'delivery_time', 'unit_price'
                               , 'quantity', 'product_name', 'n_neg_steps', 'signed_at', 'concluded_at', 'cfp']]
        data.to_csv(str(log_dir / 'signed_contracts.csv'), index_label='index')
    else:
        with open(log_dir / 'signed_contracts.csv', 'w') as f:
            f.write('')

    if len(world.cancelled_contracts) > 0:
        data = pd.DataFrame(world.cancelled_contracts)
        data = data.sort_values(['delivery_time'])
        data = data.loc[:, ['seller_type', 'buyer_type', 'seller_name', 'buyer_name', 'delivery_time', 'unit_price'
                               , 'quantity', 'product_name', 'n_neg_steps', 'signed_at', 'concluded_at', 'cfp']]
        data.to_csv(str(log_dir / 'cancelled_contracts.csv'), index_label='index')
    else:
        with open(log_dir / 'cancelled_contracts.csv', 'w') as f:
            f.write('')

    if len(world.saved_contracts) > 0:
        data = pd.DataFrame(world.saved_contracts)
        data = data.sort_values(['delivery_time'])
        data.to_csv(str(log_dir / 'contracts_full_info.csv'), index_label='index')
        data = data.loc[:, ['seller_type', 'buyer_type', 'seller_name', 'buyer_name', 'delivery_time', 'unit_price'
                               , 'quantity', 'product_name', 'n_neg_steps', 'signed_at', 'concluded_at', 'cfp']]
        data.to_csv(str(log_dir / 'all_contracts.csv'), index_label='index')
    else:
        with open(log_dir / 'contracts_full_info.csv', 'w') as f:
            f.write('')
        with open(log_dir / 'all_contracts.csv', 'w') as f:
            f.write('')


class WorldGenerator(Protocol):
    """A callback-protocol specifying the signature of a world generator function that can be passed to `tournament`

    Args:
            name: world name. If None, a random name should be generated
            competitors: A list of `Agent` types that can be used to create the agents of the competitor types
            log_File_name: A log file name to keep logs
            randomize: If true, competitors should be assigned randomly within the world. The meaning of "random
            assignment" can vary from a world to another. In general it should be the case that if randomize is False,
            all worlds generated given kwargs, and a selection competitors will be the same.
            agent_names_reveal_type: Whether the type of an agent should be apparent in its name
            kwargs: key-value pairs of arguments.

    See Also:
        `tournament`

    """

    def __call__(self, name: Optional[str] = None, competitors: Sequence[Type[Agent]] = ()
                 , log_file_name: Optional[str] = None, randomize: bool = True, agent_names_reveal_type: bool = False
                 , **kwargs) -> World: ...


@dataclass
class WorldRunResults:
    """Results of a world run"""
    world_name: str
    """World name"""
    log_file_name: str
    """Log file name"""
    names: List[str] = field(default_factory=list, init=False)
    """Agent names"""
    scores: List[float] = field(default_factory=list, init=False)
    """Agent scores"""
    types: List[str] = field(default_factory=list, init=False)
    """Agent type names"""


@dataclass
class TournamentResults:
    scores: pd.DataFrame
    total_scores: pd.DataFrame
    winners: List[str]
    """Winner type name(s) which may be a list"""
    winners_scores: np.array
    """Winner score (accumulated)"""
    ttest: pd.DataFrame


def run_world(world_info: dict):
    """Runs a world and returns stats. This function is designed to be used with distributed systems like dask.

    Args:
        world_info: World info dict. See remarks for its parameters
        
    Remarks:
    
        The `world_info` dict should have the following members:

            - name: world name [Defaults to random]
            - competitors: list of strings giving competitor types [Defaults to an empty list]
            - log_file_name: file name to store the world log [Defaults to random]
            - randomize: whether to randomize assignment [Defaults to True]
            - agent_names_reveal_type: whether agent names reveal type [Defaults to False]
            - __dir_name: directory to store the world stats [Defaults to random]
            - __world_generator: full name of the world generator function (including its module) [Required]
            - __score_calculator: full name of the score calculator function [Required]
            - __tournament_name: name of the tournament [Defaults to random]
            - others: values of all other keys are passed to the world generator as kwargs
    """
    world_generator = world_info.get('__world_generator', None)
    score_calculator = world_info.get('__score_calculator', None)
    tournament_name = world_info.get('__tournament_name', unique_name(base=""))
    assert world_generator and score_calculator, f'Cannot run without specifying both a world generator and a score ' \
        f'calculator'

    world_generator = import_by_name(world_generator)
    score_calculator = import_by_name(score_calculator)
    world_info['competitors'] = [get_class(_) for _ in world_info.get('competitors', [])]
    default_name = unique_name(base="")
    world_info['name'] = world_info.get('name', default_name)
    world_name = world_info['name']
    default_dir = (Path(f'~') / 'negmas' / 'tournaments' / tournament_name / world_name).absolute()
    world_info['log_file_name'] = world_info.get('log_file_name', str(default_dir / 'log.txt'))
    world_info['agent_names_reveal_type'] = world_info.get('agent_names_reveal_type', False)
    world_info['randomize'] = world_info.get('randomzie', True)
    world_info['__dir_name'] = world_info.get('__dir_name', str(default_dir))

    # delete the parameters not used by _run_world
    for k in ('__world_generator', '__tournament_name', '__score_calculator'):
        if k in world_info.keys():
            del world_info[k]
    return _run_world(world_info=world_info, world_generator=world_generator, score_calculator=score_calculator)


def _run_world(world_info: dict, world_generator: WorldGenerator,
               score_calculator: Callable[[World], WorldRunResults]
               , world_progress_callback: Callable[[Optional[World]], None] = None
               ):
    """Runs a world and returns stats

    Args:
        world_info: World info dict. See remarks for its parameters
        world_generator: World generator function.
        score_calculator: Score calculator function
        world_progress_callback: world progress callback

    Remarks:

        The `world_info` dict should have the following members:

            - name: world name
            - competitors: list of types giving competitor types
            - log_file_name: file name to store the world log
            - randomize: whether to randomize assignment
            - agent_names_reveal_type: whether agent names reveal type
            - __dir_name: directory to store the world stats
            - others: values of all other keys are passed to the world generator as kwargs
    """
    world_info = world_info.copy()
    dir_name = world_info['__dir_name']
    del world_info['__dir_name']
    world = world_generator(**world_info)
    if world_progress_callback is None:
        world.run()
    else:
        _start_time = time.monotonic()
        for _ in range(world.n_steps):
            if world.time_limit is not None and (time.monotonic() - _start_time) >= world.time_limit:
                break
            if not world.step():
                break
            world_progress_callback(world)
    save_stats(world=world, log_dir=dir_name)
    scores = score_calculator(world)
    return scores, dir_name


def process_world_run(results: WorldRunResults, tournament_name: str, dir_name: str) -> pd.DataFrame:
    """
    Generates a dataframe with the results of this world run

    Args:
        results: Results of the world run
        tournament_name: tournament name
        dir_name: directory name to store the stats.

    Returns:

        A pandas DataFrame with agent_name, agent_type, score, log_file, world, and stats_folder columns

    """
    log_file, world_name_ = results.log_file_name, results.world_name
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'\nPART of TOURNAMENT {tournament_name}. This world run completed successfully\n')
    scores = []
    for name_, type_, score in zip(results.names, results.types, results.scores):
        scores.append({'agent_name': name_, 'agent_type': type_, 'score': score, 'log_file': log_file
                          , 'world': world_name_, 'stats_folder': dir_name})
    return pd.DataFrame(data=scores)


def _run_dask(scheduler_ip, scheduler_port, verbose, world_infos, world_generator, tournament_progress_callback
              , n_worlds, name, score_calculator) -> List[pd.DataFrame]:
    """Runs the tournament on dask"""
    scores = []
    if scheduler_ip is None and scheduler_port is None:
        address = None
    else:
        if scheduler_ip is None:
            scheduler_ip = '127.0.0.1'
        if scheduler_port is None:
            scheduler_port = '8786'
        address = f'{scheduler_ip}:{scheduler_port}'
    if verbose:
        print(f'Will use DASK on {address}')
    client = distributed.Client(address=address, set_as_default=True)
    future_results = []
    for world_info in world_infos:
        future_results.append(client.submit(_run_world, world_info, world_generator, score_calculator))
    print(f'Submitted all processes ({len(world_infos)})')
    for i, (future, result) in enumerate(
        distributed.as_completed(future_results, with_results=True, raise_errors=False)):
        try:
            score_, dir_name = result
            if tournament_progress_callback is not None:
                tournament_progress_callback(score_, i, n_worlds)
            scores.append(process_world_run(score_, tournament_name=name, dir_name=str(dir_name)))
        except Exception as e:
            if tournament_progress_callback is not None:
                tournament_progress_callback(None, i, n_worlds)
            print(traceback.format_exc())
            print(e)
    client.shutdown()
    return scores


def tournament(competitors: Sequence[Union[str, Type[Agent]]]
               , world_generator: WorldGenerator
               , score_calculator: Callable[[World], WorldRunResults]
               , randomize=False
               , agent_names_reveal_type=False
               , max_n_runs: int = 100
               , n_runs_per_config: int = 5
               , tournament_path: str = './logs/tournaments'
               , total_timeout: Optional[int] = None
               , parallelism='local'
               , scheduler_ip: Optional[str] = None
               , scheduler_port: Optional[str] = None
               , tournament_progress_callback: Callable[[Optional[WorldRunResults], int, int], None] = lambda x: None
               , world_progress_callback: Callable[[Optional[World]], None] = None
               , name: str = None
               , verbose: bool = False
               , configs_only: bool = False
               , **kwargs
               ) -> Union[TournamentResults, pathlib.Path]:
    """
    Runs a tournament

    Args:

        name: Tournament name
        world_generator: A functions to generate worlds for the tournament
        score_calculator: A function for calculating the score of a world *After it finishes running*
        competitors: A list of class names for the competitors
        randomize: If true, then instead of trying all possible permutations of assignment random shuffles will be used.
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its 
        beginning).
        max_n_runs: No more than n_runs_max worlds will be run. If `randomize` then it cannot be None and that is exactly
        the number of worlds to run. If not `randomize` then at most this number of worlds will be run if it is not None
        n_runs_per_config: Number of runs per configuration.
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
        keep detailed logs
        parallelism: Type of parallelism. Can be 'none' for serial, 'local' for parallel and 'dist' for distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip:   IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
        evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
        processing
        verbose: Verbosity
        configs_only: If true, a config file for each 
        kwargs: Arguments to pass to the `world_generator` function

    Returns:
        `TournamentResults` The results of the tournament

    """
    dask_options = ('dist', 'distributed', 'dask', 'd')
    multiprocessing_options = ('local', 'parallel', 'par', 'p')
    serial_options = ('none', 'serial', 's')
    assert total_timeout is None or parallelism not in dask_options, f'Cannot use {parallelism} with a total-timeout'
    assert world_progress_callback is None or parallelism not in dask_options, f'Cannot use {parallelism} with a world callback'
    if name is None:
        name = unique_name('', add_time=True, rand_digits=0)
    competitors = list(competitors)
    if tournament_path.startswith('~'):
        tournament_path = Path.home() / ('/'.join(tournament_path.split('/')[1:]))
    tournament_path = (pathlib.Path(tournament_path) / name).absolute()
    tournament_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'Results of Tournament {name} will be saved to {str(tournament_path)}')
    params = {
        'competitors': [get_class(_).__name__ for _ in competitors],
        'randomize': randomize,
        'n_runs': max_n_runs,
        'tournament_path': str(tournament_path),
        'total_timeout': total_timeout,
        'parallelism': parallelism,
        'scheduler_ip': scheduler_ip,
        'scheduler_port': scheduler_port,
        'name': name,
        'n_worlds_to_run': None
    }
    params.update(kwargs)
    with (tournament_path / 'params.json').open('w') as f:
        json.dump(params, f, sort_keys=True, indent=4)
    world_infos = []
    if randomize:
        for i in range(max_n_runs):
            random.shuffle(competitors)
            world_name = unique_name(f'{i:05}', add_time=True, rand_digits=4)
            dir_name = tournament_path / world_name
            world_info = {'name': world_name, 'competitors': competitors, 'log_file_name': str(dir_name / 'log.txt')
                , 'randomize': True, 'agent_names_reveal_type': agent_names_reveal_type
                , '__dir_name': str(dir_name)}
            world_info.update(kwargs)
            world_infos += [world_info.copy() for _ in range(n_runs_per_config)]
    else:
        c_list = list(itertools.permutations(competitors))
        extra_runs = 0
        if max_n_runs is not None:
            if len(c_list) > max_n_runs:
                print(f'Need {len(c_list)} permutations but allowed to only use {max_n_runs} of them'
                      f' ({max_n_runs / len(c_list):0.2%})')
                c_list = random.shuffle(c_list)[:max_n_runs]
            elif len(c_list) < max_n_runs:
                extra_runs = max_n_runs - len(c_list)

        for i, c in enumerate(c_list):
            world_name = unique_name(f'{i:05}', add_time=True, rand_digits=4)
            dir_name = tournament_path / world_name
            world_info = {'name': world_name, 'competitors': list(c), 'log_file_name': str(dir_name / 'log.txt')
                , 'randomize': False, 'agent_names_reveal_type': agent_names_reveal_type
                , '__dir_name': str(dir_name)}
            world_info.update(kwargs)
            world_infos += [world_info.copy() for _ in range(n_runs_per_config)]

        # if extra_runs > 0:
        #     for j in range(extra_runs):
        #         random.shuffle(competitors)
        #         world_name = unique_name(f'{j:05}', add_time=True, rand_digits=4)
        #         dir_name = tournament_path / world_name
        #         world_info = {'name': world_name, 'competitors': competitors,
        #                       'log_file_name': str(dir_name / 'log.txt')
        #             , 'randomize': True, 'agent_names_reveal_type': agent_names_reveal_type
        #             , '__dir_name': str(dir_name)}
        #         world_info.update(kwargs)
        #         world_infos += [world_info.copy() for _ in range(n_runs_per_config)]

    if configs_only:
        saved_configs = [{k: copy.copy(v) if k != 'competitors' else
        [get_full_type_name(c) if not isinstance(c, str) else c for c in v]
                          for k, v in _.items()} for _ in world_infos]
        score_calculator_name = get_full_type_name(score_calculator) if not isinstance(score_calculator,
                                                                                       str) else score_calculator
        world_generator_name = get_full_type_name(world_generator) if not isinstance(world_generator,
                                                                                     str) else world_generator
        for d in saved_configs:
            d['__score_calculator'] = score_calculator_name
            d['__world_generator'] = world_generator_name
            d['__tournament_name'] = name
        config_path = tournament_path / 'configs'
        config_path.mkdir(exist_ok=True, parents=True)
        for i, conf in enumerate(saved_configs):
            f_name = config_path / f'{i:06}.json'
            with open(f_name, 'w') as f:
                json.dump(conf, f, sort_keys=True, indent=4)
        return config_path

    scores = []
    scores_file = str(tournament_path / 'scores.csv')
    n_worlds = len(world_infos)
    params['n_worlds_to_run'] = n_worlds
    with (tournament_path / 'params.json').open('w') as f:
        json.dump(params, f, sort_keys=True, indent=4)
    if verbose:
        print(f'Will run {n_worlds} worlds')
    if parallelism in serial_options:
        strt = time.perf_counter()
        for i, world_info in enumerate(world_infos):
            if total_timeout is not None and time.perf_counter() - strt > total_timeout:
                break
            try:
                score_, _ = _run_world(world_info=world_info, world_generator=world_generator
                                       , world_progress_callback=world_progress_callback
                                       , score_calculator=score_calculator)
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_worlds)
                scores.append(process_world_run(score_, tournament_name=name, dir_name=str(world_info['__dir_name'])))
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_worlds)
                print(traceback.format_exc())
                print(e)
    elif parallelism in multiprocessing_options:
        executor = futures.ProcessPoolExecutor(max_workers=None)
        future_results = []
        for world_info in world_infos:
            future_results.append(executor.submit(_run_world, world_info, world_generator, score_calculator
                                                  , world_progress_callback))
        if verbose:
            print(f'Submitted all processes ({len(world_infos)})')
        for i, future in enumerate(futures.as_completed(future_results, timeout=total_timeout)):
            try:
                score_, dir_name = future.result()
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_worlds)
                scores.append(process_world_run(score_, tournament_name=name, dir_name=str(dir_name)))
            except futures.TimeoutError:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_worlds)
                print('Tournament timed-out')
                break
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_worlds)
                print(traceback.format_exc())
                print(e)
    elif parallelism in dask_options:
        scores = _run_dask(scheduler_ip, scheduler_port, verbose, world_infos, world_generator
                           , tournament_progress_callback, n_worlds, name, score_calculator)
    if verbose:
        print(f'Finding winners')

    scores: pd.DataFrame = pd.concat(scores, ignore_index=True)
    scores = pd.DataFrame(data=scores)
    scores.to_csv(scores_file, index_label='index')
    scores = scores.loc[~scores['agent_type'].isnull(), :]
    scores = scores.loc[scores['agent_type'].str.len() > 0, :]
    total_scores = scores.groupby(['agent_type'])['score'].sum().sort_values(ascending=False).reset_index()
    winner_table = total_scores.loc[total_scores['score'] == total_scores['score'].max(), :]
    winners = winner_table['agent_type'].values.tolist()
    winner_scores = winner_table['score'].values
    types = list(scores['agent_type'].unique())

    ttest_results = []
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types[i + 1:]):
            from scipy.stats import ttest_ind
            t, p = ttest_ind(scores[scores['agent_type'] == t1].score, scores[scores['agent_type'] == t2].score)
            ttest_results.append({'a': t1, 'b': t2, 't': t, 'p': p})

    if verbose:
        print(f'Tournament completed successfully\nWinners: {list(zip(winners, winner_scores))}')

    return TournamentResults(scores=scores, total_scores=total_scores, winners=winners, winners_scores=winner_scores
                             , ttest=pd.DataFrame(data=ttest_results))
