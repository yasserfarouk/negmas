from __future__ import annotations

"""Provides interfaces for defining negotiation mechanisms."""

import copy
import math
import pprint
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from attrs import define
from rich.progress import Progress

from negmas import warnings
from negmas.checkpoints import CheckpointMixin
from negmas.common import (
    DEFAULT_JAVA_PORT,
    Action,
    MechanismState,
    NegotiatorInfo,
    NegotiatorMechanismInterface,
    ReactiveStrategy,
)
from negmas.events import Event, EventSource
from negmas.helpers import snake_case
from negmas.helpers.misc import get_free_tcp_port
from negmas.helpers.strings import humanize_time
from negmas.negotiators import Negotiator
from negmas.outcomes import Outcome
from negmas.outcomes.common import check_one_and_only, ensure_os
from negmas.outcomes.protocols import OutcomeSpace
from negmas.preferences import (
    kalai_points,
    nash_points,
    pareto_frontier,
    pareto_frontier_bf,
)
from negmas.preferences.ops import max_relative_welfare_points, max_welfare_points
from negmas.types import NamedObject

if TYPE_CHECKING:
    from negmas.outcomes.base_issue import Issue
    from negmas.outcomes.protocols import DiscreteOutcomeSpace
    from negmas.preferences import Preferences
    from negmas.preferences.base_ufun import BaseUtilityFunction

__all__ = ["Mechanism", "MechanismStepResult"]


@define(frozen=True)
class MechanismStepResult:
    """
    Represents the results of a negotiation step.

    This is what `round()` should return.
    """

    state: MechanismState
    """The returned state."""
    completed: bool = True
    """Whether the current round is completed or not."""
    broken: bool = False
    """True only if END_NEGOTIATION was selected by one agent."""
    timedout: bool = False
    """True if a timeout occurred."""
    agreement: Outcome | None = None
    """The agreement if any. Allows for a single outcome or a collection of outcomes."""
    error: bool = False
    """True if an error occurred in the mechanism."""
    error_details: str = ""
    """Error message."""
    waiting: bool = False
    """Whether to consider that the round is still running and call the round
    method again without increasing the step number."""
    exceptions: dict[str, list[str]] | None = None
    """A mapping from negotiator ID to a list of exceptions raised by that
    negotiator in this round."""
    times: dict[str, float] | None = None
    """A mapping from negotiator ID to the time it consumed during this round."""


# noinspection PyAttributeOutsideInit
class Mechanism(NamedObject, EventSource, CheckpointMixin, ABC):
    """Base class for all negotiation Mechanisms.

    Override the `round` function of this class to implement a round of your mechanism

    Args:
        outcome_space: The negotiation agenda
        outcomes: list of outcomes (optional as you can pass `issues`). If an int then it is the number of outcomes
        n_steps: Number of rounds allowed (None means infinity)
        time_limit: Number of real seconds allowed (None means infinity)
        pend: Probability of ending the negotiation at any step
        pend_per_second: Probability of ending the negotiation every second
        hidden_time_limit: Number of real seconds allowed but not visilbe to the negotiators
        max_n_agents:  Maximum allowed number of agents
        dynamic_entry: Allow agents to enter/leave negotiations between rounds
        cache_outcomes: If true, a list of all possible outcomes will be cached
        max_cardinality: The maximum allowed number of outcomes in the cached set
        annotation: Arbitrary annotation
        checkpoint_every: The number of steps to checkpoint after. Set to <= 0 to disable
        checkpoint_folder: The folder to save checkpoints into. Set to None to disable
        checkpoint_filename: The base filename to use for checkpoints (multiple checkpoints will be prefixed with
                             step number).
        single_checkpoint: If true, only the most recent checkpoint will be saved.
        extra_checkpoint_info: Any extra information to save with the checkpoint in the corresponding json file as
                               a dictionary with string keys
        exist_ok: IF true, checkpoints override existing checkpoints with the same filename.
        name: Name of the mechanism session. Should be unique. If not given, it will be generated.
        genius_port: the port used to connect to Genius for all negotiators in this mechanism (0 means any).
        id: An optional system-wide unique identifier. You should not change
            the default value except in special circumstances like during
            serialization and should always guarantee system-wide uniquness
            if you set this value explicitly
    """

    def __init__(
        self,
        initial_state: MechanismState | None = None,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | int | None = None,
        n_steps: int | float | None = None,
        time_limit: float | None = None,
        pend: float = 0,
        pend_per_second: float = 0,
        step_time_limit: float | None = None,
        negotiator_time_limit: float | None = None,
        hidden_time_limit: float = float("inf"),
        max_n_agents: int | None = None,
        dynamic_entry=False,
        annotation: dict[str, Any] | None = None,
        nmi_factory=NegotiatorMechanismInterface,
        extra_callbacks=False,
        checkpoint_every: int = 1,
        checkpoint_folder: PathLike | None = None,
        checkpoint_filename: str | None = None,
        extra_checkpoint_info: dict[str, Any] | None = None,
        single_checkpoint: bool = True,
        exist_ok: bool = True,
        name=None,
        genius_port: int = DEFAULT_JAVA_PORT,
        id: str | None = None,
        type_name: str | None = None,
        verbosity: int = 0,
    ):
        check_one_and_only(outcome_space, issues, outcomes)
        outcome_space = ensure_os(outcome_space, issues, outcomes)
        self.__verbosity = verbosity
        super().__init__(name, id=id, type_name=type_name)

        self._negotiator_times = defaultdict(float)
        CheckpointMixin.checkpoint_init(
            self,
            step_attrib="_step",
            every=checkpoint_every,
            folder=checkpoint_folder,
            filename=checkpoint_filename,
            info=extra_checkpoint_info,
            exist_ok=exist_ok,
            single=single_checkpoint,
        )
        self.__last_second_tried = 0
        self._hidden_time_limit = hidden_time_limit
        time_limit = time_limit if time_limit is not None else float("inf")
        step_time_limit = (
            step_time_limit if step_time_limit is not None else float("inf")
        )
        negotiator_time_limit = (
            negotiator_time_limit if negotiator_time_limit is not None else float("inf")
        )

        # parameters fixed for all runs

        self.id = str(uuid.uuid4())
        if n_steps == float("inf"):
            n_steps = None
        if isinstance(n_steps, float):
            n_steps = int(n_steps)
        if pend is None:
            pend = 0
        if pend_per_second is None:
            pend_per_second = 0
        self.nmi = nmi_factory(
            id=self.id,
            n_outcomes=outcome_space.cardinality,
            outcome_space=outcome_space,
            time_limit=time_limit,
            pend=pend,
            pend_per_second=pend_per_second,
            n_steps=n_steps,
            step_time_limit=step_time_limit,
            negotiator_time_limit=negotiator_time_limit,
            dynamic_entry=dynamic_entry,
            max_n_agents=max_n_agents,
            annotation=annotation if annotation is not None else dict(),
            mechanism=self,
        )

        self._current_state = initial_state if initial_state else MechanismState()

        self._history: list[MechanismState] = []
        self._stats: dict[str, Any] = dict()
        self._stats["round_times"] = list()
        self._stats["times"] = defaultdict(float)
        self._stats["exceptions"] = defaultdict(list)
        # if self.nmi.issues is not None:
        #     self.nmi.issues = tuple(self.nmi.issues)
        # if self.nmi.outcomes is not None:
        #     self.nmi.outcomes = tuple(self.nmi.outcomes)

        self._requirements = {}
        self._negotiators = []
        self._negotiator_map: dict[str, Negotiator] = dict()
        self._negotiator_index: dict[str, int] = dict()
        self._roles = []
        self._start_time = None
        self.__discrete_os = None
        self.__discrete_outcomes = None
        self._extra_callbacks = extra_callbacks

        self.agents_of_role = defaultdict(list)
        self.role_of_agent = {}
        # mechanisms do not differentiate between RANDOM_JAVA_PORT and ANY_JAVA_PORT.
        # if either is given as the genius_port, it will fix a port and all negotiators
        # that are not explicitly assigned to a port (by passing port>0 to them) will just
        # use that port.
        self.genius_port = genius_port if genius_port > 0 else get_free_tcp_port()

        self.params: dict[str, Any] = dict(
            dynamic_entry=dynamic_entry,
            genius_port=genius_port,
            annotation=annotation,
        )

    @property
    def negotiator_times(self) -> dict[str, float]:
        """The total time consumed by every negotiator.

        Each mechanism class is responsible of updating this for any activities of the negotiator it controls."""
        return self._negotiator_times

    @property
    def negotiators(self):
        return self._negotiators

    @property
    def participants(self) -> list[NegotiatorInfo]:
        """Returns a list of all participant names."""
        return [
            NegotiatorInfo(name=_.name, id=_.id, type=snake_case(_.__class__.__name__))
            for _ in self.negotiators
        ]

    def is_valid(self, outcome: Outcome):
        """Checks whether the outcome is valid given the issues."""
        return outcome in self.nmi.outcome_space

    @property
    def outcome_space(self) -> OutcomeSpace:
        return self.nmi.outcome_space

    def discrete_outcome_space(
        self, levels: int = 5, max_cardinality: int = 10_000_000_000
    ) -> DiscreteOutcomeSpace:
        """Returns a stable discrete version of the given outcome-space."""
        if self.__discrete_os:
            return self.__discrete_os
        self.__discrete_os = self.outcome_space.to_discrete(
            levels=levels,
            max_cardinality=max_cardinality
            if max_cardinality is not None
            else float("inf"),
        )
        return self.__discrete_os

    @property
    def outcomes(self):
        return self.nmi.outcomes

    def discrete_outcomes(
        self, levels: int = 5, max_cardinality: int | float = float("inf")
    ) -> list[Outcome]:
        """A discrete set of outcomes that spans the outcome space.

        Args:
            max_cardinality: The maximum number of outcomes to return. If None, all outcomes will be returned for discrete issues
            and *10_000* if any of the issues was continuous

        Returns:

            list[Outcome]: list of `n` or less outcomes
        """
        if self.outcomes is not None:
            return list(self.outcomes)
        if self.__discrete_outcomes:
            return self.__discrete_outcomes
        self.__discrete_os = self.outcome_space.to_discrete(
            levels=levels,
            max_cardinality=max_cardinality
            if max_cardinality is not None
            else float("inf"),
        )
        self.__discrete_outcomes = list(self.__discrete_os.enumerate_or_sample())
        return self.__discrete_outcomes

    def random_outcomes(
        self, n: int = 1, with_replacement: bool = False
    ) -> list[Outcome]:
        """Returns random offers.

        Args:
              n: Number of outcomes to generate
              with_replacement: If true, outcomes may be repeated

        Returns:
              A list of outcomes of at most n outcomes.

        Remarks:

                - If the number of outcomes `n` cannot be satisfied, a smaller number will be returned
                - Sampling is done without replacement (i.e. returned outcomes are unique).
        """
        return list(
            self.outcome_space.sample(
                n, with_replacement=with_replacement, fail_if_not_enough=False
            )
        )

    def random_outcome(self) -> Outcome:
        """Returns a single random offer"""
        return self.outcome_space.random_outcome()

    @property
    def time(self) -> float:
        """Elapsed time since mechanism started in seconds.

        0.0 if the mechanism did not start running
        """
        if self._start_time is None:
            return 0.0

        return time.perf_counter() - self._start_time

    @property
    def remaining_time(self) -> float | None:
        """
        Returns remaining time in seconds.

        None if no time limit is given.
        """
        if self.nmi.time_limit == float("+inf"):
            return None
        if not self._start_time:
            return self.nmi.time_limit

        limit = self.nmi.time_limit - (time.perf_counter() - self._start_time)
        if limit < 0.0:
            return 0.0

        return limit

    @property
    def expected_remaining_time(self) -> float | None:
        """
        Returns remaining time in seconds (expectation).

        None if no time limit or pend_per_second is given.
        """
        rem = self.remaining_time
        pend = self.nmi.pend_per_second
        if pend <= 0:
            return rem
        return min(rem, (1 / pend)) if rem is not None else (1 / pend)

    @property
    def relative_time(self) -> float:
        """
        Returns a number between ``0`` and ``1`` indicating elapsed relative time or steps.

        Remarks:
            - If pend or pend_per_second are defined in the `NegotiatorMechanismInterface`,
              and time_limit/n_steps are not given, this becomes an expectation that is limited above by one.
        """
        n_steps = self.nmi.n_steps
        time_limit = self.nmi.time_limit
        if time_limit == float("+inf") and n_steps is None:
            if self.nmi.pend <= 0 and self.nmi.pend_per_second <= 0:
                return 0.0
            if self.nmi.pend > 0:
                n_steps = int(math.ceil(1 / self.nmi.pend))
            if self.nmi.pend_per_second > 0:
                time_limit = 1 / self.nmi.pend_per_second

        relative_step = (
            (self._current_state.step + 1) / (n_steps + 1)
            if n_steps is not None
            else -1.0
        )
        relative_time = self.time / time_limit if time_limit is not None else -1.0
        return min(1.0, max([relative_step, relative_time]))

    @property
    def expected_relative_time(self) -> float:
        """
        Returns a positive number indicating elapsed relative time or steps.

        Remarks:
            - This is relative to the expected time/step at which the negotiation ends given all timing
              conditions (time_limit, n_step, pend, pend_per_second).
        """
        n_steps = self.nmi.n_steps
        time_limit = self.nmi.time_limit
        if time_limit == float("+inf") and n_steps is None:
            if self.nmi.pend <= 0 and self.nmi.pend_per_second <= 0:
                return 0.0
        if n_steps is None:
            # set the expected number of steps to the reciprocal of the probability of ending at every step
            n_steps = (
                int(math.ceil(1 / self.nmi.pend)) if self.nmi.pend > 0 else n_steps
            )
        else:
            n_steps = min(int(math.ceil(1 / self.nmi.pend)), n_steps)
        if time_limit == float("inf"):
            # set the expected number of seconds to the reciprocal of the probability of ending every second
            time_limit = (
                int(math.ceil(1 / self.nmi.pend_per_second))
                if self.nmi.pend_per_second > 0
                else time_limit
            )
        else:
            time_limit = (
                min(time_limit, int(math.ceil(1 / self.nmi.pend_per_second)))
                if self.nmi.pend_per_second > 0
                else time_limit
            )

        relative_step = (
            (self._current_state.step + 1) / (n_steps + 1)
            if n_steps is not None
            else -1.0
        )
        relative_time = self.time / time_limit if time_limit is not None else -1.0
        return max([relative_step, relative_time])

    @property
    def remaining_steps(self) -> int | None:
        """
        Returns the remaining number of steps until the end of the mechanism run.

        None if unlimited
        """
        if self.nmi.n_steps is None:
            return None

        return self.nmi.n_steps - self._current_state.step

    @property
    def expected_remaining_steps(self) -> int | None:
        """
        Returns the expected remaining number of steps until the end of the mechanism run.

        None if unlimited
        """
        rem = self.remaining_steps
        pend = self.nmi.pend
        if pend <= 0:
            return rem

        return (
            min(rem, int(math.ceil(1 / pend)))
            if rem is not None
            else int(math.ceil(1 / pend))
        )

    @property
    def requirements(self):
        """A dictionary specifying the requirements that must be in the
        capabilities of any agent to join the mechanism."""
        return self._requirements

    @requirements.setter
    def requirements(
        self,
        requirements: dict[
            str,
            (
                tuple[int | float | str, int | float | str]
                | list
                | set
                | int
                | float
                | str
            ),
        ],
    ):
        self._requirements = {
            k: set(v) if isinstance(v, list) else v for k, v in requirements.items()
        }

    def is_satisfying(self, capabilities: dict) -> bool:
        """Checks if the  given capabilities are satisfying mechanism
        requirements.

        Args:
            capabilities: capabilities to check

        Returns:
            bool are the requirements satisfied by the capabilities.

        Remarks:

            - Requirements are also a dict with the following meanings:

                - tuple: Min and max acceptable values
                - list/set: Any value in the iterable is acceptable
                - Single value: The capability must match this value

            - Capabilities can also have the same three possibilities.
        """
        requirements = self.requirements
        for r, v in requirements.items():
            if v is None:
                if r not in capabilities.keys():
                    return False

                else:
                    continue

            if r not in capabilities.keys():
                return False

            if capabilities[r] is None:
                continue

            c = capabilities[r]
            if isinstance(c, tuple):
                # c is range
                if isinstance(v, tuple):
                    # both ranges
                    match = v[0] <= c[0] <= v[1] or v[0] <= c[1] <= v[1]
                else:
                    # c is range and cutoff_utility is not a range
                    match = (
                        any(c[0] <= _ <= c[1] for _ in v)
                        if isinstance(v, set)
                        else c[0] <= v <= c[1]
                    )
            elif isinstance(c, list) or isinstance(c, set):
                # c is list
                if isinstance(v, tuple):
                    # c is a list and cutoff_utility is a range
                    match = any(v[0] <= _ <= v[1] for _ in c)
                else:
                    # c is a list and cutoff_utility is not a range
                    match = any(_ in v for _ in c) if isinstance(v, set) else v in c
            else:
                # c is a single value
                if isinstance(v, tuple):
                    # c is a singlton and cutoff_utility is a range
                    match = v[0] <= c <= v[1]
                else:
                    # c is a singlton and cutoff_utility is not a range
                    match = c in v if isinstance(v, set) else c == v
            if not match:
                return False

        return True

    def can_participate(self, agent: Negotiator) -> bool:
        """Checks if the agent can participate in this type of negotiation in
        general.

        Args:
            agent:

        Returns:
            bool: True if it  can

        Remarks:
            The only reason this may return `False` is if the mechanism requires some requirements
            that are not within the capabilities of the agent.

            When evaluating compatibility, the agent is considered incapable of participation if any
            of the following conditions hold:
            * A mechanism requirement is not in the capabilities of the agent
            * A mechanism requirement is in the capabilities of the agent by the values required for it
              is not in the values announced by the agent.

            An agent that lists a `None` value for a capability is announcing that it can work with all its
            values. On the other hand, a mechanism that lists a requirement as None announces that it accepts
            any value for this requirement as long as it exist in the agent
        """
        return self.is_satisfying(agent.capabilities)

    def can_accept_more_agents(self) -> bool:
        """Whether the mechanism can **currently** accept more negotiators."""
        return (
            True
            if self.nmi.max_n_agents is None or self._negotiators is None
            else len(self._negotiators) < self.nmi.max_n_agents
        )

    def can_enter(self, agent: Negotiator) -> bool:
        """Whether the agent can enter the negotiation now."""
        return self.can_accept_more_agents() and self.can_participate(agent)

    # def extra_state(self) -> dict[str, Any] | None:
    #     """Returns any extra state information to be kept in the `state` and `history` properties"""
    #     return dict()

    @property
    def state(self) -> MechanismState:
        """Returns the current state.

        Override `extra_state` if you want to keep extra state
        """
        return self._current_state

    def _get_nmi(self, negotiator: Negotiator) -> NegotiatorMechanismInterface:  # type: ignore
        _ = negotiator
        return self.nmi

    def add(
        self,
        negotiator: Negotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        ufun: BaseUtilityFunction | None = None,
    ) -> bool | None:
        """Add an agent to the negotiation.

        Args:

            negotiator: The agent to be added.
            preferences: The utility function to use. If None, then the agent must already have a stored
                  utility function otherwise it will fail to enter the negotiation.
            ufun: [depricated] same as preferences but must be a `UFun` object.
            role: The role the agent plays in the negotiation mechanism. It is expected that mechanisms inheriting from
                  this class will check this parameter to ensure that the role is a valid role and is still possible for
                  negotiators to join on that role. Roles may include things like moderator, representative etc based
                  on the mechanism


        Returns:

            * True if the agent was added.
            * False if the agent was already in the negotiation.
            * None if the agent cannot be added. This can happen in the following cases:

              1. The capabilities of the negotiator do not match the requirements of the negotiation
              2. The outcome-space of the negotiator's preferences do not contain the outcome-space of the negotiation
              3. The negotiator refuses to join (by returning False from its `join` method) see `Negotiator.join` for possible reasons of that
        """

        from negmas.preferences import (
            BaseUtilityFunction,
            MappingUtilityFunction,
            Preferences,
        )

        if ufun is not None:
            if not isinstance(ufun, BaseUtilityFunction):
                ufun = MappingUtilityFunction(ufun, outcome_space=self.outcome_space)
            preferences = ufun
        if (
            preferences is not None
            and not isinstance(preferences, Preferences)
            and isinstance(preferences, Callable)
        ):
            preferences = MappingUtilityFunction(
                preferences, outcome_space=self.outcome_space
            )
        if (
            preferences
            and preferences.outcome_space
            and self.outcome_space not in preferences.outcome_space
        ):
            return None
        if not self.can_enter(negotiator):
            return None

        if negotiator in self._negotiators:
            return False

        if role is None:
            role = "negotiator"

        if negotiator.join(
            nmi=self._get_nmi(negotiator),
            state=self.state,
            preferences=preferences,
            role=role,
        ):
            self._negotiators.append(negotiator)
            self._current_state.n_negotiators += 1
            self._negotiator_map[negotiator.id] = negotiator
            self._negotiator_index[negotiator.id] = len(self._negotiators) - 1
            self._roles.append(role)
            self.role_of_agent[negotiator.uuid] = role
            self.agents_of_role[role].append(negotiator)
            return True
        return None

    def get_negotiator(self, source: str) -> Negotiator | None:
        """Returns the negotiator with the given ID if present in the
        negotiation."""
        return self._negotiator_map.get(source, None)

    def can_leave(self, agent: Negotiator) -> bool:
        """Can the agent leave now?"""
        return (
            True
            if self.nmi.dynamic_entry
            else not self.nmi.state.running and agent in self._negotiators
        )

    def remove(self, negotiator: Negotiator) -> bool | None:
        """Remove the agent from the negotiation.

        Args:
            agent:

        Returns:
            * True if the agent was removed.
            * False if the agent was not in the negotiation already.
            * None if the agent cannot be removed.
        """
        if not self.can_leave(negotiator):
            return False
        n = self._negotiator_map.get(negotiator.id, None)
        if n is None:
            return False
        self._negotiators.remove(negotiator)
        self._negotiator_map.pop(negotiator.id)
        self._negotiator_index.pop(negotiator.id)
        if self._extra_callbacks:
            strt = time.perf_counter()
            negotiator.on_leave(self.nmi.state)
            self._negotiator_times[negotiator.id] += time.perf_counter() - strt
        return True

    def add_requirements(self, requirements: dict) -> None:
        """Adds requirements."""
        requirements = {
            k: set(v) if isinstance(v, list) else v for k, v in requirements.items()
        }
        if hasattr(self, "_requirements"):
            self._requirements.update(requirements)
        else:
            self._requirements = requirements

    def remove_requirements(self, requirements: Iterable) -> None:
        """Adds requirements."""
        for r in requirements:
            if r in self._requirements.keys():
                self._requirements.pop(r, None)

    def negotiator_index(self, source: str) -> int | None:
        """Gets the negotiator index.

        Args:
            source (str): source

        Returns:
            int | None:
        """

        return self._negotiator_index.get(source, None)

    @property
    def negotiator_ids(self) -> list[str]:
        return [_.id for _ in self._negotiators]

    @property
    def genius_negotiator_ids(self) -> list[str]:
        return [
            _.java_uuid if hasattr(_, "java_uuid") else _.id for _ in self._negotiators
        ]

    def genius_id(self, id: str | None) -> str | None:
        """Gets the Genius ID corresponding to the given negotiator if known otherwise its normal ID"""
        if id is None:
            return None
        negotiator = self._negotiator_map.get(id, None)
        if not negotiator:
            return id
        return negotiator.java_uuid if hasattr(negotiator, "java_uuid") else negotiator.id  # type: ignore

    @property
    def agent_ids(self) -> list[str]:
        return [_.owner.id for _ in self._negotiators if _.owner]

    @property
    def agent_names(self) -> list[str]:
        return [_.owner.name for _ in self._negotiators if _.owner]

    @property
    def negotiator_names(self) -> list[str]:
        return [_.name for _ in self._negotiators]

    @property
    def agreement(self):
        return self._current_state.agreement

    @property
    def n_outcomes(self):
        return self.nmi.n_outcomes

    @property
    def issues(self) -> list[Issue]:
        """Returns the issues of the outcomespace.

        Will raise an exception if the outcome space has no defined
        issues
        """
        return self.nmi.outcome_space.issues  # type: ignore

    @property
    def completed(self):
        """Ended without timing out (either with agreement or broken by a negotiator)"""
        return self.agreement is not None or self._current_state.broken

    @property
    def ended(self):
        """Ended in any way"""
        return self._current_state.ended

    @property
    def n_steps(self):
        return self.nmi.n_steps

    @property
    def time_limit(self):
        return self.nmi.time_limit

    @property
    def running(self):
        return self._current_state.running

    @property
    def dynamic_entry(self):
        return self.nmi.dynamic_entry

    @property
    def max_n_agents(self):
        return self.nmi.max_n_agents

    @property
    def state4history(self) -> Any:
        """Returns the state as it should be stored in the history."""
        return copy.deepcopy(self._current_state)

    def _add_to_history(self, state4history):
        if len(self._history) == 0:
            self._history.append(state4history)
            return
        last = self._history[-1]
        if last["step"] == state4history:
            self._history[-1] = state4history
            return
        self._history.append(state4history)

    def on_mechanism_error(self) -> None:
        """Called when there is a mechanism error.

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        state = self.state
        if self._extra_callbacks:
            for a in self.negotiators:
                strt = time.perf_counter()
                a.on_mechanism_error(state)
                self._negotiator_times[a.id] += time.perf_counter() - strt

    def on_negotiation_end(self) -> None:
        """Called at the end of each negotiation.

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        state = self.state
        for a in self.negotiators:
            strt = time.perf_counter()
            a._on_negotiation_end(state)
            self._negotiator_times[a.id] += time.perf_counter() - strt
        self.announce(
            Event(
                type="negotiation_end",
                data={
                    "agreement": self.agreement,
                    "state": state,
                    "annotation": self.nmi.annotation,
                },
            )
        )
        self.checkpoint_final_step()

    @property
    def verbosity(self) -> int:
        """
        Verbosity level.

        - Children of this class should only print if verbosity > 1
        """
        return self.__verbosity

    def on_negotiation_start(self) -> bool:
        """Called before starting the negotiation.

        If it returns False then negotiation will end immediately
        """
        return True

    @abstractmethod
    def __call__(
        self, state: MechanismState, action: dict[str, Action | None] | None = None
    ) -> MechanismStepResult:
        """
        Implements a single step of the mechanism. Override this!

        Args:
            state: The mechanism state. When overriding, set the type of this
                   to the specific `MechanismState` descendent for your mechanism.
            action: An optional action (value) of the next negotiator (key). If given, the call
                   should just execute the action without calling the next negotiator.

        Returns:
            `MechanismStepResult` showing the result of the negotiation step
        """
        ...

    def step(self, action: dict[str, Action | None] | None = None) -> MechanismState:
        """Runs a single step of the mechanism.

        Returns:
            MechanismState: The state of the negotiation *after* the round is conducted
            action: An optional action (value) for the next negotiator (key). If given, the call
                   should just execute the action without calling the next negotiator.

        Remarks:

            - Every call yields the results of one round (see `round()`)
            - If the mechanism was yet to start, it will start it and runs one round
            - There is another function (`run()`) that runs the whole mechanism in blocking mode
        """

        if self._start_time is None or self._start_time < 0:
            self._start_time = time.perf_counter()
        if self.__verbosity >= 1:
            if self.current_step == 0:
                print(
                    f"{self.name}: Step {self.current_step} starting after {datetime.now()}",
                    flush=True,
                )
            else:
                _elapsed = time.perf_counter() - self._start_time
                remaining = self.expected_remaining_steps
                etatime = self.expected_remaining_time
                etatime = etatime if etatime is not None else float("inf")
                if remaining is not None:
                    _eta = (
                        humanize_time(
                            min(
                                (_elapsed * remaining) / self.current_step,
                                etatime,
                            )
                        )
                        + f" {remaining} steps"
                    )
                else:
                    _eta = "--"
                print(
                    f"{self.name}: Step {self.current_step} starting after {humanize_time(_elapsed, show_ms=True)} [ETA {_eta}]",
                    flush=True,
                    end="\r" if self.verbosity == 1 else "\n",
                )
        self.checkpoint_on_step_started()
        state = self.state
        state4history = self.state4history
        rs, rt = random.random(), 2

        # end with a timeout if condition is met
        current_time = self.time
        if self.__last_second_tried < int(current_time):
            rt, self.__last_second_tried = random.random(), int(current_time)

        if (
            (current_time > self.time_limit)
            or (self.nmi.n_steps and self._current_state.step >= self.nmi.n_steps)
            or current_time > self._hidden_time_limit
            or rs < self.nmi.pend - 1e-8
            or rt < self.nmi.pend_per_second - 1e-8
        ):
            (
                self._current_state.running,
                self._current_state.broken,
                self._current_state.timedout,
            ) = (False, False, True)
            self.on_negotiation_end()
            return self.state

        # if there is a single negotiator and no other negotiators can be added,
        # end without starting
        if len(self._negotiators) < 2:
            if self.nmi.dynamic_entry:
                return self.state
            else:
                (
                    self._current_state.running,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (False, False, False)
                self.on_negotiation_end()
                return self.state

        # if the mechanism states that it is broken, timedout or ended with
        # agreement, report that
        if (
            self._current_state.broken
            or self._current_state.timedout
            or self._current_state.agreement is not None
        ):
            self._current_state.running = False
            self.on_negotiation_end()
            return self.state

        if not self._current_state.running:
            # if we did not start, just start
            self._current_state.running = True
            self._current_state.step = 0
            self._start_time = time.perf_counter()
            self._current_state.started = True
            state = self.state
            # if the mechanism indicates that it cannot start, keep trying
            if self.on_negotiation_start() is False:
                (
                    self._current_state.agreement,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (None, False, False)
                return self.state
            for a in self.negotiators:
                strt = time.perf_counter()
                a._on_negotiation_start(state=state)
                self._negotiator_times[a.id] += time.perf_counter() - strt
            self.announce(Event(type="negotiation_start", data=None))
        else:
            # if no steps are remaining, end with a timeout
            remaining_steps, remaining_time = self.remaining_steps, self.remaining_time
            if (remaining_steps is not None and remaining_steps <= 0) or (
                remaining_time is not None and remaining_time <= 0.0
            ):
                self._current_state.running = False
                (
                    self._current_state.agreement,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (None, False, True)
                self.on_negotiation_end()
                return self.state

        # send round start only if the mechanism is not waiting for anyone
        # TODO check this.
        if not self._current_state.waiting and self._extra_callbacks:
            for agent in self._negotiators:
                strt = time.perf_counter()
                agent.on_round_start(state)
                self._negotiator_times[agent.id] += time.perf_counter() - strt

        # run a round of the mechanism and get the new state
        step_start = (
            time.perf_counter() if not self._current_state.waiting else self._last_start
        )
        self._last_start = step_start
        self._current_state.waiting = False
        try:
            result = self(self._current_state, action=action)
        except TypeError:
            result = self(self._current_state)
        self._current_state = result.state
        step_time = time.perf_counter() - step_start
        self._stats["round_times"].append(step_time)

        # if negotaitor times are reported, save them
        if result.times:
            for k, v in result.times.items():
                if v is not None:
                    self._stats["times"][k] += v
        # if negotaitor exceptions are reported, save them
        if result.exceptions:
            for k, v in result.exceptions.items():
                if v:
                    self._stats["exceptions"][k] += v

        # update current state variables from the result of the round just run
        (
            self._current_state.has_error,
            self._current_state.error_details,
            self._current_state.waiting,
        ) = (
            result.state.has_error,
            result.state.error_details,
            result.state.waiting,
        )
        if self._current_state.has_error:
            self.on_mechanism_error()
        if (
            self.nmi.step_time_limit is not None
            and step_time > self.nmi.step_time_limit
        ):
            (
                self._current_state.broken,
                self._current_state.timedout,
                self._current_state.agreement,
            ) = (False, True, None)
        else:
            (
                self._current_state.broken,
                self._current_state.timedout,
                self._current_state.agreement,
            ) = (
                result.state.broken,
                result.state.timedout,
                result.state.agreement,
            )
        if (
            (self._current_state.agreement is not None)
            or self._current_state.broken
            or self._current_state.timedout
        ):
            self._current_state.running = False

        # now switch to the new state
        state = self.state
        if not self._current_state.waiting and result.completed:
            state4history = self.state4history
            if self._extra_callbacks:
                for agent in self._negotiators:
                    strt = time.perf_counter()
                    agent.on_round_end(state)
                    self._negotiator_times[agent.id] += time.perf_counter() - strt
            self._add_to_history(state4history)
            # we only indicate a new step if no one is waiting
            self._current_state.step += 1
            self._current_state.time = self.time
            self._current_state.relative_time = self.relative_time

        if not self._current_state.running:
            self.on_negotiation_end()
        return self.state

    def __next__(self) -> MechanismState:
        result = self.step()
        if not self._current_state.running:
            raise StopIteration

        return result

    def abort(self) -> MechanismState:
        """Aborts the negotiation."""
        (
            self._current_state.has_error,
            self._current_state.error_details,
            self._current_state.waiting,
        ) = (
            True,
            "Uncaught Exception",
            False,
        )
        self.on_mechanism_error()
        (
            self._current_state.broken,
            self._current_state.timedout,
            self._current_state.agreement,
        ) = (True, False, None)
        state = self.state
        state4history = self.state4history
        self._current_state.running = False
        if self._extra_callbacks:
            for agent in self._negotiators:
                strt = time.perf_counter()
                agent.on_round_end(state)
                self._negotiator_times[agent.id] += time.perf_counter() - strt
        self._add_to_history(state4history)
        self._current_state.step += 1
        self.on_negotiation_end()
        return state

    @classmethod
    def runall(
        cls,
        mechanisms: list[Mechanism] | tuple[Mechanism, ...],
        keep_order=True,
        method="serial",
    ) -> list[MechanismState | None]:
        """Runs all mechanisms.

        Args:
            mechanisms: list of mechanisms
            keep_order: if True, the mechanisms will be run in order every step otherwise the order will be randomized
                        at every step. This is only allowed if the method is serial
            method: the method to use for running all the sessions.  Acceptable options are: serial, threads, processes

        Returns:
            - list of states of all mechanisms after completion
            - None for any such states indicates disagreements
        """
        completed = [_ is None for _ in mechanisms]
        states: list[MechanismState | None] = [None] * len(mechanisms)
        indices = list(range(len(list(mechanisms))))
        if method == "serial":
            while not all(completed):
                if not keep_order:
                    random.shuffle(indices)
                for i in indices:
                    done, mechanism = completed[i], mechanisms[i]
                    if done:
                        continue
                    result = mechanism.step()
                    if result.running:
                        continue
                    completed[i] = True
                    states[i] = mechanism.state
                    if all(completed):
                        break
        elif method == "threads":
            raise NotImplementedError()
        elif method == "processes":
            raise NotImplementedError()
        else:
            raise ValueError(
                f"method {method} is unknown. Acceptable options are serial, threads, processes"
            )
        return states

    @classmethod
    def stepall(
        cls, mechanisms: list[Mechanism] | tuple[Mechanism, ...], keep_order=True
    ) -> list[MechanismState]:
        """Step all mechanisms.

        Args:
            mechanisms: list of mechanisms
            keep_order: if True, the mechanisms will be run in order every step otherwise the order will be randomized
                        at every step

        Returns:
            - list of states of all mechanisms after completion
        """
        indices = list(range(len(list(mechanisms))))
        if not keep_order:
            random.shuffle(indices)

        completed = [_ is None for _ in mechanisms]
        for i in indices:
            done, mechanism = completed[i], mechanisms[i]
            if done:
                continue
            result = mechanism.step()
            if result.running:
                continue
            completed[i] = True
        return [_.state for _ in mechanisms]

    def run_with_progress(self, timeout=None) -> MechanismState:
        if timeout is None:
            with Progress() as progress:
                task = progress.add_task("Negotiating ...", total=100)
                for _ in self:
                    progress.update(task, completed=int(self.relative_time * 100))
        else:
            start_time = time.perf_counter()
            with Progress() as progress:
                task = progress.add_task("Negotiating ...", total=100)
                for _ in self:
                    progress.update(task, completed=int(self.relative_time * 100))
                    if time.perf_counter() - start_time > timeout:
                        (
                            self._current_state.running,
                            self._current_state.timedout,
                            self._current_state.broken,
                        ) = (False, True, False)
                        self.on_negotiation_end()
                        break
        return self.state

    def run(self, timeout=None) -> MechanismState:
        if timeout is None:
            for _ in self:
                pass
        else:
            start_time = time.perf_counter()
            for _ in self:
                if time.perf_counter() - start_time > timeout:
                    (
                        self._current_state.running,
                        self._current_state.timedout,
                        self._current_state.broken,
                    ) = (False, True, False)
                    self.on_negotiation_end()
                    break
        return self.state

    @property
    def history(self):
        return self._history

    @property
    def stats(self):
        return self._stats

    @property
    def current_step(self):
        return self._current_state.step

    def _get_preferences(self):
        preferences = []
        for a in self.negotiators:
            preferences.append(a.preferences)
        return preferences

    def _pareto_frontier(
        self,
        method: Callable,
        max_cardinality: int | float = float("inf"),
        sort_by_welfare=True,
    ) -> tuple[list[tuple[float, ...]], list[Outcome]]:
        ufuns = tuple(self._get_preferences())
        if any(_ is None for _ in ufuns):
            raise ValueError(
                "Some negotiators have no ufuns. Cannot calcualate the pareto frontier"
            )
        outcomes = self.discrete_outcomes(max_cardinality=max_cardinality)
        points = [tuple(ufun(outcome) for ufun in ufuns) for outcome in outcomes]
        reservs = tuple(
            _.reserved_value if _ is not None else float("-inf") for _ in ufuns
        )
        rational_indices = [
            i for i, _ in enumerate(points) if all(a >= b for a, b in zip(_, reservs))
        ]
        points = [points[_] for _ in rational_indices]
        indices = list(
            method(
                points,
                sort_by_welfare=sort_by_welfare,
            )
        )
        frontier = [points[_] for _ in indices]
        if frontier is None:
            raise ValueError("Could not find the pareto-frontier")
        return frontier, [outcomes[rational_indices[_]] for _ in indices]

    def pareto_frontier_bf(
        self, max_cardinality: float = float("inf"), sort_by_welfare=True
    ) -> tuple[list[tuple[float, ...]], list[Outcome]]:
        return self._pareto_frontier(
            pareto_frontier_bf, max_cardinality, sort_by_welfare
        )

    def pareto_frontier(
        self, max_cardinality: float = float("inf"), sort_by_welfare=True
    ) -> tuple[tuple[tuple[float, ...]], list[Outcome]]:
        ufuns = tuple(self._get_preferences())
        if any(_ is None for _ in ufuns):
            raise ValueError(
                "Some negotiators have no ufuns. Cannot calculate the pareto frontier"
            )
        outcomes = self.discrete_outcomes(max_cardinality=max_cardinality)
        results = pareto_frontier(
            ufuns,
            outcomes=outcomes,
            n_discretization=None,
            max_cardinality=float("inf"),
            sort_by_welfare=sort_by_welfare,
        )
        return results[0], [outcomes[_] for _ in results[1]]

    def max_welfare_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...]] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome]]:
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = max_welfare_points(
            ufuns, frontier, outcome_space=self.outcome_space
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def max_relative_welfare_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...]] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome]]:
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = max_relative_welfare_points(
            ufuns, frontier, outcome_space=self.outcome_space
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def modified_kalai_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...]] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome]]:
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = kalai_points(
            ufuns,
            frontier,
            outcome_space=self.outcome_space,
            subtract_reserved_value=False,
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def kalai_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...]] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome]]:
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = kalai_points(
            ufuns,
            frontier,
            outcome_space=self.outcome_space,
            subtract_reserved_value=True,
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def nash_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...]] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome]]:
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        nash_pts = nash_points(ufuns, frontier, outcome_space=self.outcome_space)
        return tuple(
            (nash_utils, frontier_outcomes[indx]) for nash_utils, indx in nash_pts
        )

    def plot(self, **kwargs) -> Any:
        """A method for plotting a negotiation session."""
        _ = kwargs

    def _get_ami(self, negotiator: ReactiveStrategy) -> NegotiatorMechanismInterface:  # type: ignore
        _ = negotiator
        warnings.deprecated(f"_get_ami is depricated. Use `get_nmi` instead of it")
        return self.nmi

    def __iter__(self):
        return self

    def __str__(self):
        d = self.__dict__.copy()
        return pprint.pformat(d)

    __repr__ = __str__
