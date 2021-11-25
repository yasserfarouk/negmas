"""
Common data-structures and classes used by all other modules.

This module does not import anything from the library except during type checking
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from .java import to_dict, to_java

# from .actors.thread_actors import ThreadProxy

if TYPE_CHECKING:
    from .mechanisms import Mechanism
    from .outcomes import Issue, Outcome


__all__ = [
    "NegotiatorInfo",
    "NegotiatorMechanismInterface",
    "MechanismState",
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
    agreement: "Outcome" | None = None
    """Agreement at the end of the negotiation (it is always None until an agreement is reached)."""
    results: "Outcome" | list["Outcome"] | list["Issue"] | None = None
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
        d = {k: deepcopy(v, memo=memodict) for k, v in self.__dict__.items()}
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
class NegotiatorMechanismInterface:
    """All information of a negotiation visible to negotiators."""

    id: str
    """Mechanism session ID. That is unique for all mechanisms"""
    n_outcomes: int | None
    """Number of outcomes which may be None indicating infinity"""
    issues: list["Issue"]
    """Negotiation issues as a list of `Issue` objects"""
    outcomes: list["Outcome"] | None
    """A lit of *all possible* outcomes for a negotiation. None if the number of outcomes is uncountable"""
    time_limit: float
    """The time limit in seconds for this negotiation session. None indicates infinity"""
    step_time_limit: float
    """The time limit in seconds for each step of this negotiation session. None indicates infinity"""
    negotiator_time_limit: float
    """The time limit in seconds to wait for negotiator responses of this negotiation session. None indicates infinity"""
    n_steps: int
    """The allowed number of steps for this negotiation. None indicates infinity"""
    dynamic_entry: bool
    """Whether it is allowed for agents to enter/leave the negotiation after it starts"""
    max_n_agents: int
    """Maximum allowed number of agents in the session. None indicates no limit"""
    imap: dict[str | int, str | int]
    """A map that translates issue names to indices and issue indices to names"""
    annotation: dict[str, Any] = field(default_factory=dict)
    """An arbitrary annotation as a `dict[str, Any]` that is always available for all agents"""
    _mechanism: "Mechanism" | None = None

    @property
    def issue_names(self):
        """Returns issue names"""
        return [_.name for _ in self.issues]

    @property
    def params(self):
        """Returns the parameters used to initialize the mechanism."""
        return self._mechanism.params

    def random_outcomes(self, n: int = 1) -> list["Outcome"]:
        """
        A set of random outcomes from the issues of this negotiation

        Args:
            n: number of outcomes requested

        Returns:

            list[Outcome]: list of `n` or less outcomes

        """
        return self._mechanism.random_outcomes(n=n)

    def discrete_outcomes(self, n_max: int = None) -> list["Outcome"]:
        """
        A discrete set of outcomes that spans the outcome space

        Args:
            n_max: The maximum number of outcomes to return. If None, all outcomes will be returned for discrete issues

        Returns:

            list[Outcome]: list of `n` or less outcomes

        """
        return self._mechanism.discrete_outcomes(n_max=n_max)

    def outcome_index(self, outcome: "Outcome") -> int | None:
        """
        The index of an outcome

        Args:
            outcome: The outcome asked about

        Returns:

            int: The index of this outcome in the list of outcomes. Only valid if n_outcomes is finite and not None.
        """
        return self._mechanism.outcome_index(outcome)

    @property
    def participants(self) -> list[NegotiatorInfo]:
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
        """Syntactic sugar for state.n_negotiators"""
        return self.state.n_negotiators

    @property
    def negotiator_ids(self) -> list[str]:
        """Gets the IDs of all negotiators"""
        return self._mechanism.negotiator_ids

    @property
    def negotiator_names(self) -> list[str]:
        """Gets the namess of all negotiators"""
        return self._mechanism.negotiator_names

    @property
    def agent_ids(self) -> list[str]:
        """Gets the IDs of all agents owning all negotiators"""
        return self._mechanism.agent_ids

    @property
    def agent_names(self) -> list[str]:
        """Gets the names of all agents owning all negotiators"""
        return self._mechanism.agent_names

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
        return NegotiatorMechanismInterface(**vars(self))

    def __deepcopy__(self, memodict={}):
        d = {k: deepcopy(v, memo=memodict) for k, v in vars(self).items()}
        if "_mechanism" in d.keys():
            del d["_mechanism"]
        return NegotiatorMechanismInterface(**d)

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
    """Used to represent an AMI to Java."""

    def randomOutcomes(self, n: int):
        return to_java(self.shadow.random_outcomes(n))

    def discreteOutcomes(self, nMax: int):
        return to_java(self.shadow.discrete_outcomes(n_max=nMax))

    def outcomeIndex(self, outcome):
        return to_java(self.shadow.outcome_index(outcome))

    def getParticipants(self):
        return to_java(self.shadow.participants)

    def getOutcomes(self):
        return to_java(self.shadow.outcomes)

    def getState(self):
        return to_java(self.shadow.state)

    def getRequirements(self):
        return to_java(self.shadow.requirements)

    def getNNegotiators(self):
        return self.shadow.n_negotiators

    def __init__(self, ami: NegotiatorMechanismInterface):
        self.shadow = ami

    def to_java(self):
        return to_dict(self.shadow)

    class Java:
        implements = ["jnegmas.common.AgentMechanismInterface"]
