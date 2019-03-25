"""Common data-structures and classes used by all other modules.

This module does not import anything from the library except during type checking
"""
import typing
import uuid
from copy import deepcopy
from typing import List, Optional, Any, TYPE_CHECKING

from dataclasses import dataclass, field, fields

from negmas.helpers import snake_case, unique_name

if TYPE_CHECKING:
    from negmas.mechanisms import Mechanism
    from negmas.outcomes import Issue, Outcome

__all__ = [
    'NamedObject',
    'MechanismInfo',
    'MechanismState',
    'register_all_mechanisms',
    'NegotiatorInfo',
]

_running_negotiations: typing.Dict[str, 'Mechanism'] = {}


@dataclass
class NegotiatorInfo:
    name: str
    """Name of this negotiator"""
    id: str
    """ID unique to this negotiator"""
    type: str
    """Type of the negotiator as a string"""


@dataclass
class MechanismState:
    """Encapsulates the mechanism state at any point"""
    running: bool = False
    """Whether the negotiation has started and did not yet finish"""
    started: bool = False
    """Whether the negotiation has started"""
    step: int = 0
    """The current round of the negotiation"""
    time: float = 0.0
    """The current real time of the negotiation."""
    relative_time: float = 0.0
    """A number in the period [0, 1] giving the relative time of the negotiation.
    Relative time is calculated as ``max(step/n_steps, time/time_limit)``.
    """
    broken: bool = False
    """True if the negotiation has started and ended with an END_NEGOTIATION"""
    timedout: bool = False
    """True if the negotiation was timedout"""
    agreement: Optional['Outcome'] = None
    """Agreement at the end of the negotiation (it is always None until an agreement is reached)"""
    n_negotiators: int = 0
    """Number of agents currently in the negotiation. Notice that this may change over time if the mechanism supports 
    dynamic entry"""
    has_error: bool = False
    """Does the mechanism have any errors"""
    error_details: str = ''
    """Details of the error if any"""
    info: 'MechanismInfo' = None
    """Mechanism information"""

    @property
    def ended(self):
        return self.started and (self.broken or self.timedout or (self.agreement is not None))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __copy__(self):
        return MechanismInfo(**self.__dict__)

    def __deepcopy__(self, memodict={}):
        d = {k: deepcopy(v) for k, v in self.__dict__.items()}
        return MechanismInfo(**d)

    def __getitem__(self, item):
        """Makes the outcome type behave like a dict"""
        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def asdict(self):
        """Converts the outcome to a dict containing all fields"""
        return {_.name: self.__dict__[_.name] for _ in fields(self)}

    class Java:
        implements = ['jnegmas.common.MechanismState']


@dataclass
class MechanismInfo:
    """All information of a negotiation visible to negotiators."""
    id: str
    """Mechanism session ID. That is unique for all mechanisms"""
    n_outcomes: Optional[int]
    """Number of outcomes which may be None indicating infinity"""
    issues: List['Issue']
    """Negotiation issues as a list of `Issue` objects"""
    outcomes: Optional[List['Outcome']]
    """A lit of *all possible* outcomes for a negotiation. None if the number of outcomes is uncountable"""
    time_limit: float
    """The time limit in seconds for this negotiation session. None indicates infinity"""
    step_time_limit: float
    """The time limit in seconds for each step of this negotiation session. None indicates infinity"""
    n_steps: int
    """The allowed number of steps for this negotiation. None indicates infinity"""
    dynamic_entry: bool
    """Whether it is allowed for agents to enter/leave the negotiation after it starts"""
    max_n_agents: int
    """Maximum allowed number of agents in the session. None indicates no limit"""
    annotation: typing.Dict[str, Any] = field(default_factory=dict)
    """An arbitrary annotation as a `Dict[str, Any]` that is always available for all agents"""

    def random_outcomes(self, n: int = 1, astype: typing.Type['Outcome'] = dict) -> List['Outcome']:
        """
        A set of random outcomes from the issues of this negotiation

        Args:
            n: number of outcomes requested
            astype: A type to cast the resulting outcomes to.

        Returns:

            List[Outcome]: List of `n` or less outcomes

        """
        return _running_negotiations[self.id].random_outcomes(n=n, astype=astype)

    def discrete_outcomes(self, n_max: int = None, astype: typing.Type['Outcome'] = dict) -> List['Outcome']:
        """
        A discrete set of outcomes that spans the outcome space

        Args:
            n_max: The maximum number of outcomes to return. If None, all outcomes will be returned for discrete issues
            astype: A type to cast the resulting outcomes to.

        Returns:

            List[Outcome]: List of `n` or less outcomes

        """
        return _running_negotiations[self.id].discrete_outcomes(n_max=n_max, astype=astype)

    def outcome_index(self, outcome: 'Outcome') -> Optional[int]:
        """
        The index of an outcome

        Args:
            outcome: The outcome asked about

        Returns:

            int: The index of this outcome in the list of outcomes. Only valid if n_outcomes is finite and not None.
        """
        return _running_negotiations[self.id].outcome_index(outcome)

    @property
    def participants(self) -> List[NegotiatorInfo]:
        return _running_negotiations[self.id].participants

    @property
    def state(self) -> MechanismState:
        """
        Access the current state of the mechanism.

        Remarks:

            - Whenever a method receives a `MechanismInfo` object, it can always access the *current* state of the
              protocol by accessing this property.

        """
        return _running_negotiations[self.id].state

    @property
    def requirements(self) -> dict:
        """
        The protocol requirements

        Returns:
            - A dict of str/Any pairs giving the requirements
        """
        return _running_negotiations[self.id].requirements

    @property
    def n_negotiators(self) -> int:
        """Syntactic sugar for state.n_agents"""
        return self.state.n_negotiators

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        d1 = self.__dict__
        d2 = other.__dict__
        for k in d1.keys():
            if d2[k] != d1[k]:
                return False
        return True

    def __copy__(self):
        return MechanismInfo(**self.__dict__)

    def __deepcopy__(self, memodict={}):
        d = {k: deepcopy(v) for k, v in self.__dict__.items()}
        return MechanismInfo(**d)

    def __getitem__(self, item):
        """Makes the outcome type behave like a dict"""
        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def asdict(self):
        """Converts the outcome to a dict containing all fields"""
        return {_.name: self.__dict__[_.name] for _ in fields(self)}

    class Java:
        implements = ['jnegmas.common.MechanismInfo']


def register_all_mechanisms(mechanisms: typing.Dict[str, 'Mechanism']) -> None:
    """registers the running mechanisms. Used internally. **DO NOT CALL THIS.**"""
    global _running_negotiations
    _running_negotiations = mechanisms


class NamedObject(object):
    """The base class of all named entities.

    All named entities need to call this class's __init__() somewhere during initialization.

    Args:
        name (str): The given name of the entity. Notice that the class will add this to a  base that depends
                    on the child's class name.

    """

    def __init__(self, name: str = None) -> None:
        if name is not None:
            name = str(name)
        self.__uuid = (f'{name}-' if name is not None else "") + str(uuid.uuid4())
        if name is None or len(name) == 0:
            name = unique_name('', add_time=False, rand_digits=16)
        self.__name = name
        super().__init__()

    @classmethod
    def create(cls, *args, **kwargs):
        """Creates an object and returns a proxy to it."""
        return cls(*args, **kwargs)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def uuid(self):
        return self.__uuid

    @uuid.setter
    def uuid(self, uuid):
        self.__uuid = uuid

    @property
    def id(self):
        return self.__uuid

    @id.setter
    def id(self, id):
        self.__uuid = id
