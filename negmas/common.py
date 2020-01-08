"""Common data-structures and classes used by all other modules.

This module does not import anything from the library except during type checking
"""
import datetime
import uuid
import dill
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Any, TYPE_CHECKING, Union, Dict, Tuple, Type
from typing_extensions import Protocol, runtime

from .helpers import unique_name, load, dump, get_full_type_name
from .java import to_java, to_dict

if TYPE_CHECKING:
    from .mechanisms import Mechanism
    from .outcomes import Issue, Outcome


__all__ = [
    "NamedObject",
    "AgentMechanismInterface",
    "MechanismState",
    "NegotiatorInfo",
    "_ShadowAgentMechanismInterface",
]


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
    waiting: bool = False
    """Whether the negotiation is waiting for some negotiator to respond"""
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
    agreement: Optional["Outcome"] = None
    """Agreement at the end of the negotiation (it is always None until an agreement is reached)."""
    results: Optional[Union["Outcome", List["Outcome"], List["Issue"]]] = None
    """In its simplest form, an agreement is a single outcome (or None for failure). Nevertheless, it can be a list of
    outcomes or even a list of negotiation issues for future negotiations.
    """
    n_negotiators: int = 0
    """Number of agents currently in the negotiation. Notice that this may change over time if the mechanism supports
    dynamic entry"""
    has_error: bool = False
    """Does the mechanism have any errors"""
    error_details: str = ""
    """Details of the error if any"""

    @property
    def ended(self):
        return self.started and (
            self.broken or self.timedout or (self.agreement is not None)
        )

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __copy__(self):
        return MechanismState(**self.__dict__)

    def __deepcopy__(self, memodict={}):
        d = {k: deepcopy(v) for k, v in self.__dict__.items()}
        return MechanismState(**d)

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
        implements = ["jnegmas.common.MechanismState"]


@dataclass
class AgentMechanismInterface:
    """All information of a negotiation visible to negotiators."""

    id: str
    """Mechanism session ID. That is unique for all mechanisms"""
    n_outcomes: Optional[int]
    """Number of outcomes which may be None indicating infinity"""
    issues: List["Issue"]
    """Negotiation issues as a list of `Issue` objects"""
    outcomes: Optional[List["Outcome"]]
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
    annotation: Dict[str, Any] = field(default_factory=dict)
    """An arbitrary annotation as a `Dict[str, Any]` that is always available for all agents"""

    _mechanism = None

    def random_outcomes(
        self, n: int = 1, astype: Type["Outcome"] = dict
    ) -> List["Outcome"]:
        """
        A set of random outcomes from the issues of this negotiation

        Args:
            n: number of outcomes requested
            astype: A type to cast the resulting outcomes to.

        Returns:

            List[Outcome]: List of `n` or less outcomes

        """
        return self._mechanism.random_outcomes(n=n, astype=astype)

    def discrete_outcomes(
        self, n_max: int = None, astype: Type["Outcome"] = dict
    ) -> List["Outcome"]:
        """
        A discrete set of outcomes that spans the outcome space

        Args:
            n_max: The maximum number of outcomes to return. If None, all outcomes will be returned for discrete issues
            astype: A type to cast the resulting outcomes to.

        Returns:

            List[Outcome]: List of `n` or less outcomes

        """
        return self._mechanism.discrete_outcomes(n_max=n_max, astype=astype)

    def outcome_index(self, outcome: "Outcome") -> Optional[int]:
        """
        The index of an outcome

        Args:
            outcome: The outcome asked about

        Returns:

            int: The index of this outcome in the list of outcomes. Only valid if n_outcomes is finite and not None.
        """
        return self._mechanism.outcome_index(outcome)

    @property
    def participants(self) -> List[NegotiatorInfo]:
        return self._mechanism.participants

    @property
    def state(self) -> MechanismState:
        """
        Access the current state of the mechanism.

        Remarks:

            - Whenever a method receives a `AgentMechanismInterface` object, it can always access the *current* state of the
              protocol by accessing this property.

        """
        return self._mechanism.state

    @property
    def requirements(self) -> dict:
        """
        The protocol requirements

        Returns:
            - A dict of str/Any pairs giving the requirements
        """
        return self._mechanism.requirements

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
        return AgentMechanismInterface(**self.__dict__)

    def __deepcopy__(self, memodict={}):
        d = {k: deepcopy(v) for k, v in self.__dict__.items()}
        return AgentMechanismInterface(**d)

    def __getitem__(self, item):
        """Makes the outcome type behave like a dict"""
        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def asdict(self):
        """Converts the object to a dict containing all fields"""
        return {_.name: self.__dict__[_.name] for _ in fields(self)}


class _ShadowAgentMechanismInterface:
    """Used to represent an AMI to Java.
    """

    def randomOutcomes(self, n: int):
        return to_java(self.shadow.random_outcomes(n))

    def discreteOutcomes(self, nMax: int):
        return to_java(self.shadow.discrete_outcomes(n_max=nMax))

    def outcomeIndex(self, outcome) -> int:
        return to_java(self.shadow.outcome_index(outcome))

    def getParticipants(self) -> List[NegotiatorInfo]:
        return to_java(to_java(self.shadow.participants))

    def getOutcomes(self):
        return to_java(to_java(self.shadow.outcomes))

    def getState(self) -> MechanismState:
        return to_java(self.shadow.state)

    def getRequirements(self) -> Dict[str, Any]:
        return to_java(self.shadow.requirements)

    def getNNegotiators(self) -> int:
        return self.shadow.n_negotiators

    def __init__(self, ami: AgentMechanismInterface):
        self.shadow = ami

    def to_java(self):
        return to_dict(self.shadow)

    class Java:
        implements = ["jnegmas.common.AgentMechanismInterface"]


@runtime
class Runnable(Protocol):
    """A protocol defining runnable objects"""

    @property
    def current_step(self) -> int:
        pass

    def step(self) -> Any:
        pass

    def run(self) -> Any:
        pass


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
        self.__uuid = (f"{name}-" if name is not None else "") + str(uuid.uuid4())
        if name is None or len(name) == 0:
            name = unique_name("", add_time=False, rand_digits=16)
        self.__name = name
        super().__init__()

    @classmethod
    def create(cls, *args, **kwargs):
        """Creates an object and returns a proxy to it."""
        return cls(*args, **kwargs)

    @property
    def name(self):
        """A convenient name of the entity (intended primarily for printing/logging/debugging)."""
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def uuid(self):
        """The unique ID of this entity"""
        return self.__uuid

    @uuid.setter
    def uuid(self, uuid):
        self.__uuid = uuid

    @property
    def id(self):
        """The unique ID of this entity"""
        return self.__uuid

    @id.setter
    def id(self, id):
        self.__uuid = id

    def checkpoint(
        self,
        path: Union[Path, str],
        file_name: str = None,
        info: Dict[str, Any] = None,
        exist_ok: bool = False,
        single_checkpoint: bool = True,
        step_attribs: Tuple[str] = (
            "current_step",
            "_current_step",
            "_Entity__current_step",
            "_step",
        ),
    ) -> Path:
        """
        Saves a checkpoint of the current object at  the given path.

        Args:

            path: Full path to a directory to store the checkpoint
            file_name: Name of the file to dump into. If not given, a unique name is created
            info: Information to save with the checkpoint (must be json serializable)
            exist_ok: If true, override existing dump
            single_checkpoint: If true, keep a single checkpoint for the last step
            step_attribs: Attributes to represent the time-step of the object. Any of the given attributes will be
                          used in the file name generated if single_checkpoint is False. If single_checkpoint is True, the
                          filename will not contain time-step information

        Returns:
            full path to the file used to save the checkpoint

        """
        if file_name is None:
            base_name = (
                f"{self.__class__.__name__.split('.')[-1].lower()}.{unique_name('', add_time=False, rand_digits=8, sep='-')}"
                f".{self.id.replace('/', '_')}"
            )
        else:
            base_name = file_name
        path = Path(path)
        if path.exists() and path.is_file():
            raise ValueError(f"{str(path)} is a file. It must be a directory")
        path.mkdir(parents=True, exist_ok=True)
        current_step = None
        for attrib in step_attribs:
            try:
                a = getattr(self, attrib)
                if isinstance(a, int):
                    current_step = a
                    break
            except AttributeError:
                pass
        if not single_checkpoint and current_step is not None:
            base_name = f"{current_step:05}.{base_name}"
        file_name = path / base_name

        if info is None:
            info = {}
        info.update(
            {
                "type": get_full_type_name(self.__class__),
                "id": self.id,
                "name": self.name,
                "time": datetime.datetime.now().isoformat(),
                "step": current_step,
                "filename": str(file_name),
            }
        )

        if (not exist_ok) and file_name.exists():
            raise ValueError(
                f"{str(file_name)} already exists. Pass exist_ok=True if you want to override it"
            )

        with open(file_name, "wb") as f:
            dill.dump(self, f)

        info_file_name = path / (base_name + ".json")
        dump(info, info_file_name)
        return file_name

    @classmethod
    def from_checkpoint(
        cls, file_name: Union[str, Path], return_info=False
    ) -> Union["NamedObject", Tuple["NamedObject", Dict[str, Any]]]:
        """
        Creates an object from a saved checkpoint

        Args:
            file_name:
            return_info: If True, tbe information saved when the file was dumped are returned

        Returns:
            Either the object or the object and dump-info as a dict (if return_info was true)

        Remarks:

            - If info is returned, it is guaranteed to have the following members:
                - time: Dump time
                - type: Type of the dumped object
                - id: ID
                - name: name
        """
        file_name = Path(file_name).absolute()
        with open(file_name, "rb") as f:
            obj = dill.load(f)
        if return_info:
            return obj, cls.checkpoint_info(file_name)
        return obj

    @classmethod
    def checkpoint_info(cls, file_name: Union[str, Path]) -> Dict[str, Any]:
        """
        Returns the information associated with a dump of the object saved in the given file

        Args:
            file_name: Name of the object

        Returns:

        """
        file_name = Path(file_name).absolute()
        return load(file_name.parent / (file_name.name + ".json"))
