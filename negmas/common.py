"""
Common data-structures and classes used by all other modules.

This module does not import anything from the library except during type checking
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum, auto, unique
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Union

from negmas.helpers.prob import Distribution

if TYPE_CHECKING:
    from .mechanisms import Mechanism
    from .outcomes import (
        CartesianOutcomeSpace,
        DiscreteOutcomeSpace,
        Issue,
        Outcome,
        OutcomeSpace,
    )


__all__ = [
    "NegotiatorInfo",
    "NegotiatorMechanismInterface",
    "MechanismState",
    "Value",
    "PreferencesChange",
    "PreferencesChangeType",
    "AgentMechanismInterface",
]

Value = Union[Distribution, float]
"""
A value in NegMAS can either be crisp ( float ) or probabilistic ( `Distribution` )
"""


@unique
class PreferencesChangeType(Enum):
    """
    The type of change in preferences.

    Remarks:

        - Returned from `changes` property of `Preferences` to help the owner of the preferences in deciding what to do with the change.
        - Received by the `on_preferences_changed` method of `Rational` entities to inform them about a change in preferences.
        - Note that the `Rational` entity needs to call `changes` explicitly and call its own `on_preferences_changed` to handle changes that happen without assignment to `preferences` of the `Rational` entity.
        - If the `preferences` of the `Rational` agent are changed through assignmen, its `on_preferences_changed` will be called with the appropriate `PreferencesChange` list.
    """

    General = auto()
    Scaled = auto()
    Shifted = auto()
    ReservedValue = auto()
    ReservedOutcome = auto()
    UncertaintyReduced = auto()
    UncertaintyIncreased = auto()


@dataclass
class PreferencesChange:
    type: PreferencesChangeType = PreferencesChangeType.General
    data: Any = None


@dataclass
class NegotiatorInfo:
    """
    Keeps information about a negotiator. Mostly for use with controllers.
    """

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
    results: "Outcome" | "OutcomeSpace" | None = None
    """In its simplest form, an agreement is a single outcome (or None for failure). Nevertheless, it can be a list of
    outcomes or even a complete outcome space.
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


@dataclass
class NegotiatorMechanismInterface:
    """All information of a negotiation visible to negotiators."""

    id: str
    """Mechanism session ID. That is unique for all mechanisms"""
    n_outcomes: int | float
    """Number of outcomes which may be `float('inf')` indicating infinity"""
    outcome_space: OutcomeSpace
    """Negotiation agenda as as an `OutcomeSpace` object. The most common type is `CartesianOutcomeSpace` which represents the cartesian product of a list of issues"""
    time_limit: float
    """The time limit in seconds for this negotiation session. None indicates infinity"""
    step_time_limit: float
    """The time limit in seconds for each step of ;this negotiation session. None indicates infinity"""
    negotiator_time_limit: float
    """The time limit in seconds to wait for negotiator responses of this negotiation session. None indicates infinity"""
    n_steps: int | None
    """The allowed number of steps for this negotiation. None indicates infinity"""
    dynamic_entry: bool
    """Whether it is allowed for agents to enter/leave the negotiation after it starts"""
    max_n_agents: int | None
    """Maximum allowed number of agents in the session. None indicates no limit"""
    _mechanism: "Mechanism" = field(init=False)
    annotation: dict[str, Any] = field(default_factory=dict)
    """An arbitrary annotation as a `dict[str, Any]` that is always available for all agents"""

    @property
    def cartesian_outcome_space(self) -> CartesianOutcomeSpace:
        """
        Returns the `outcome_space` as a `CartesianOutcomeSpace` or raises a `ValueError` if that was not possible.

        Remarks:

            - Useful for negotiators that only work with `CartesianOutcomeSpace` s (i.e. `GeniusNegotiator` )
        """
        from negmas.outcomes import CartesianOutcomeSpace

        if not isinstance(self.outcome_space, CartesianOutcomeSpace):
            raise ValueError(
                f"{self.outcome_space} is of type {self.outcome_space.__class__.__name__} and cannot be cast as a `CartesianOutcomeSpace`"
            )
        return self.outcome_space

    def discrete_outcome_space(
        self, levels: int = 5, max_cardinality: int = 100_000
    ) -> DiscreteOutcomeSpace:
        """
        Returns a stable discrete version of the given outcome-space
        """
        return self._mechanism.discrete_outcome_space(levels, max_cardinality)

    @property
    def params(self):
        """Returns the parameters used to initialize the mechanism."""
        return self._mechanism.params

    def random_outcomes(self, n: int = 1) -> list["Outcome"]:
        """
        A set of random outcomes from the outcome-space of this negotiation

        Args:
            n: number of outcomes requested

        Returns:

            list[Outcome]: list of `n` or less outcomes

        """
        return self._mechanism.random_outcomes(n=n)

    def discrete_outcomes(
        self, max_cardinality: int | float = float("inf")
    ) -> Iterable["Outcome"]:
        """
        A discrete set of outcomes that spans the outcome space

        Args:
            max_cardinality: The maximum number of outcomes to return. If None, all outcomes will be returned for discrete outcome-spaces

        Returns:

            list[Outcome]: list of `n` or less outcomes

        """
        return self._mechanism.discrete_outcomes(max_cardinality=max_cardinality)

    @property
    def issues(self) -> tuple[Issue, ...]:
        os = self._mechanism.outcome_space
        if hasattr(os, "issues"):
            return os.issues  # type: ignore I am just checking that the attribute issues exists
        raise ValueError(
            f"{os} of type {os.__class__.__name__} has no issues attribute"
        )

    @property
    def outcomes(self) -> Iterable[Outcome] | None:
        """All outcomes for discrete outcome spaces or None for continuous outcome spaces. See `discrete_outcomes`"""
        from negmas.outcomes.protocols import DiscreteOutcomeSpace

        return (
            self._mechanism.outcome_space.enumerate()
            if isinstance(self._mechanism.outcome_space, DiscreteOutcomeSpace)
            else None
        )

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


AgentMechanismInterface = NegotiatorMechanismInterface
"""A **depricated** alias for `NegotiatorMechanismInterface`"""
