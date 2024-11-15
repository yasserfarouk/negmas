"""
Common data-structures and classes used by all other modules.

This module does not import anything from the library except during type checking
"""

from __future__ import annotations
from collections import namedtuple
from enum import Enum, auto, unique
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Union,
    runtime_checkable,
)

from attrs import asdict, define, field
from .outcomes import (
    CartesianOutcomeSpace,
    DiscreteOutcomeSpace,
    Issue,
    Outcome,
    OutcomeSpace,
)

if TYPE_CHECKING:
    from .mechanisms import Mechanism


__all__ = [
    "NegotiatorInfo",
    "NegotiatorMechanismInterface",
    "MechanismState",
    "Value",
    "PreferencesChange",
    "PreferencesChangeType",
    "AgentMechanismInterface",
    "TraceElement",
    "DEFAULT_JAVA_PORT",
    "MechanismAction",
]

DEFAULT_JAVA_PORT = 25337
"""Default port to use for connecting to GENIUS"""


@runtime_checkable
class Distribution(Protocol):
    """
    A protocol representing a probability distribution
    """

    def type(self) -> str:
        """Returns the distribution type (e.g. uniform, normal, ...)"""
        ...

    def mean(self) -> float:
        """Finds the mean"""
        ...

    def prob(self, val: float) -> float:
        """Returns the probability for the given value"""
        ...

    def cum_prob(self, mn: float, mx: float) -> float:
        """Returns the probability for the given range"""
        ...

    def sample(self, size: int = 1) -> Iterable[float]:
        """Samples `size` elements from the distribution"""
        ...

    @property
    def loc(self) -> float:
        """Returns the location of the distributon (usually mean)"""
        ...

    @property
    def scale(self) -> float:
        """Returns the scale of the distribution (may be std. dev.)"""
        ...

    @property
    def min(self) -> float:
        """Returns the minimum"""
        ...

    @property
    def max(self) -> float:
        """Returns the maximum"""
        ...

    def is_uniform(self) -> bool:
        """Returns true if this is a uniform distribution"""
        ...

    def is_gaussian(self) -> bool:
        """Returns true if this is a gaussian distribution"""
        ...

    def is_crisp(self) -> bool:
        """Returns true if this is a distribution with all probability at one point (delta(v))"""
        ...

    def __call__(self, val: float) -> float:
        """Returns the probability for the given value"""
        ...

    def __add__(self, other) -> Distribution:
        """Returns the distribution for the sum of samples of `self` and `other`"""
        ...

    def __sub__(self, other) -> Distribution:
        """Returns the distribution for the difference between samples of `self` and `other`"""
        ...

    def __mul__(self, weight: float) -> Distribution:
        """Returns the distribution for the multiplicaiton of samples of `self` with `weight`"""
        ...

    def __lt__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS less than a sample from other `other`"""
        ...

    def __le__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS less or equal a sample from other `other`"""
        ...

    def __eq__(self, other) -> bool:
        """Checks for equality of the two distributions"""
        ...

    def __ne__(self, other) -> bool:
        """Checks for ineqlaity of the distributions"""
        ...

    def __gt__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS greater than a sample from other `other`"""
        ...

    def __ge__(self, other) -> bool:
        """Check that a sample from `self` is ALWAYS greater or equal a sample from other `other`"""
        ...

    def __float__(self) -> float:
        """Converts to a float (usually by calling mean())"""
        ...


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
        - If the `preferences` of the `Rational` agent are changed through assignment, its `on_preferences_changed` will be called with the appropriate `PreferencesChange` list.
    """

    General = auto()
    Scale = auto()
    Shift = auto()
    ReservedValue = auto()
    ReservedOutcome = auto()
    UncertaintyReduced = auto()
    UncertaintyIncreased = auto()
    OSRestricted = auto()
    OSExpanded = auto()


@define(frozen=True)
class PreferencesChange:
    type: PreferencesChangeType = PreferencesChangeType.General
    data: Any = None


@define(frozen=True)
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


@define
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
    agreement: Outcome | None = None
    """Agreement at the end of the negotiation (it is always None until an agreement is reached)."""
    results: Outcome | OutcomeSpace | tuple[Outcome] | None = None
    """In its simplest form, an agreement is a single outcome (or None for failure).
    Nevertheless, it can be a tuple of outcomes or even a complete outcome space.
    """
    n_negotiators: int = 0
    """Number of agents currently in the negotiation. Notice that this may change
    over time if the mechanism supports dynamic entry"""
    has_error: bool = False
    """Does the mechanism have any errors"""
    error_details: str = ""
    """Details of the error if any"""
    erred_negotiator: str = ""
    """ID of the negotiator that raised the last error"""
    erred_agent: str = ""
    """ID of the agent owning the negotiator that raised the last error"""

    def __hash__(self):
        return hash(self.asdict())

    #     def __copy__(self):
    #         return MechanismState(**self.__dict__)
    #
    #     def __deepcopy__(self, memodict={}):
    #         d = {k: deepcopy(v, memo=memodict) for k, v in self.__dict__.items()}
    #         return MechanismState(**d)

    @property
    def ended(self):
        return self.started and (
            self.broken or self.timedout or (self.agreement is not None)
        )

    @property
    def completed(self):
        return self.started and (
            self.broken or self.timedout or (self.agreement is not None)
        )

    @property
    def done(self):
        return self.started and (
            self.broken or self.timedout or (self.agreement is not None)
        )

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def asdict(self):
        """Converts the outcome to a dict containing all fields"""
        return asdict(self)

    def __getitem__(self, item):
        """Makes the outcome type behave like a dict"""
        return getattr(self, item)

    # def __hash__(self):
    #     return hash(str(self))


#     def __eq__(self, other):
#         return self.__hash__() == other.__hash__()
#
#     def __repr__(self):
#         return self.__dict__.__repr__()
#
#     def __str__(self):
#         return str(self.__dict__)


@define(frozen=True)
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
    pend: float
    """The probability that the negotiation times out at every step. Must be less than one. If <= 0, it is ignored"""
    pend_per_second: float
    """The probability that the negotiation times out every second. Must be less than one. If <= 0, it is ignored"""
    step_time_limit: float
    """The time limit in seconds for each step of ;this negotiation session. None indicates infinity"""
    negotiator_time_limit: float
    """The time limit in seconds to wait for negotiator responses of this negotiation session. None indicates infinity"""
    n_steps: int | None
    """The allowed number of steps for this negotiation. None indicates infinity"""
    dynamic_entry: bool
    """Whether it is allowed for negotiators to enter/leave the negotiation after it starts"""
    max_n_negotiators: int | None
    """Maximum allowed number of negotiators in the session. None indicates no limit"""
    _mechanism: Mechanism = field(alias="_mechanism")
    """A reference to the mechanism. MUST NEVER BE USED BY NEGOTIATORS. **must be treated as a private member**"""
    annotation: dict[str, Any] = field(default=dict)
    """An arbitrary annotation as a `dict[str, Any]` that is always available for all negotiators"""

    #     def __copy__(self):
    #         return NegotiatorMechanismInterface(**vars(self))
    #
    #     def __deepcopy__(self, memodict={}):
    #         d = {k: deepcopy(v, memo=memodict) for k, v in vars(self).items()}
    #         if "_mechanism" in d.keys():
    #             del d["_mechanism"]
    #         return NegotiatorMechanismInterface(**d)
    #

    @property
    def estimated_n_steps(self) -> int:
        """Return an estimate of the number of steps for this negotiation."""
        estimates: list[int] = []
        if self.n_steps is not None:
            estimates.append(self.n_steps)
        if self.pend is not None:
            estimates.append(int(1 / self.pend + 0.5))
        if self.pend_per_second is not None:
            time_limit = 1 / self.pend
            estimates.append(int(time_limit * self.state.step / self.state.time + 0.5))
        if self.time_limit is not None:
            estimates.append(int(self.state.step / self.state.relative_time + 0.5))
        return min(estimates)

    @property
    def estimated_time_limit(self) -> float:
        """Return an estimate of the number of seconds for this negotiation."""
        estimates: list[float] = []
        if self.time_limit is not None:
            estimates.append(self.time_limit)
        if self.n_steps is not None:
            estimates.append(self.n_steps / self.state.relative_time)
        if self.pend_per_second is not None:
            estimates.append(int(1 / self.pend_per_second + 0.5))
        if self.pend is not None:
            n_steps = 1 / self.pend
            estimates.append(int(n_steps * self.state.time / self.state.step + 0.5))
        return min(estimates)

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
        self, levels: int = 5, max_cardinality: int = 10_000_000_000
    ) -> DiscreteOutcomeSpace:
        """
        Returns a stable discrete version of the given outcome-space
        """
        return self._mechanism.discrete_outcome_space(levels, max_cardinality)

    @property
    def params(self):
        """Returns the parameters used to initialize the mechanism."""
        return self._mechanism.params

    def random_outcome(self) -> Outcome:
        """A single random outcome."""
        return self._mechanism.random_outcome()

    def random_outcomes(self, n: int = 1) -> list[Outcome]:
        """
        A set of random outcomes from the outcome-space of this negotiation

        Args:
            n: number of outcomes requested

        Returns:

            list[Outcome]: list of `n` or less outcomes

        """
        return self._mechanism.random_outcomes(n=n)

    @property
    def atomic_steps(self) -> bool:
        return self._mechanism.atomic_steps

    def discrete_outcomes(
        self, max_cardinality: int | float = float("inf")
    ) -> Iterable[Outcome]:
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
    def history(self) -> list:
        return self._mechanism.history

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
    def genius_negotiator_ids(self) -> list[str]:
        """Gets the Java IDs of all negotiators (if the negotiator is not a GeniusNegotiator, its normal ID is returned)"""
        return self._mechanism.genius_negotiator_ids

    def genius_id(self, id: str | None) -> str | None:
        """Gets the Genius ID corresponding to the given negotiator if known otherwise its normal ID"""
        return self._mechanism.genius_id(id)

    @property
    def mechanism_id(self) -> str:
        """Gets the ID of the mechanism"""
        return self._mechanism.id

    @property
    def negotiator_ids(self) -> list[str]:
        """Gets the IDs of all negotiators"""
        return self._mechanism.negotiator_ids

    def negotiator_index(self, source: str) -> int:
        """Returns the negotiator index for the given negotiator. Raises an exception if not found"""
        indx = self._mechanism.negotiator_index(source)
        if indx is None:
            raise ValueError(f"No known index for negotiator {source}")
        return indx

    # @property
    # def negotiator_names(self) -> list[str]:
    #     """Gets the namess of all negotiators"""
    #     return self.mechanism.negotiator_names

    @property
    def agent_ids(self) -> list[str]:
        """Gets the IDs of all agents owning all negotiators"""
        return self._mechanism.agent_ids

    @property
    def agent_names(self) -> list[str]:
        """Gets the names of all agents owning all negotiators"""
        return self._mechanism.agent_names

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def asdict(self):
        """Converts the object to a dict containing all fields"""
        return asdict(self)

    def __getitem__(self, item):
        """Makes the NMI behave like a dict"""
        return getattr(self, item)

    def log_info(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at info level"""
        self._mechanism.log(nid, level="info", data=data)

    def log_debug(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at debug level"""
        self._mechanism.log(nid, level="debug", data=data)

    def log_warning(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at warning level"""
        self._mechanism.log(nid, level="warning", data=data)

    def log_error(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at error level"""
        self._mechanism.log(nid, level="error", data=data)

    def log_critical(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at critical level"""
        self._mechanism.log(nid, level="critical", data=data)


TraceElement = namedtuple(
    "TraceElement",
    ["time", "relative_time", "step", "negotiator", "offer", "responses", "state"],
)
"""An element of the trace returned by `full_trace` representing the history of the negotiation"""

AgentMechanismInterface = NegotiatorMechanismInterface
"""A **depricated** alias for `NegotiatorMechanismInterface`"""


class MechanismAction:
    """Defines a negotiation action"""


ReactiveStrategy = (
    Mapping[MechanismState, MechanismAction]
    | Callable[[MechanismState], MechanismAction]
)
"""Defines a negotiation strategy as a mapping from a mechanism state to an action"""
