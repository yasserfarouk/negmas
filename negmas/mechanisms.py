"""Provides interfaces for defining negotiation mechanisms.
"""
import itertools
import math
import pprint
import time
import uuid
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Tuple, List, Optional, Any, Iterable, Union, Dict, Set, Type

import pandas as pd
from dataclasses import dataclass

from negmas.utilities import UtilityFunction, MappingUtilityFunction, pareto_frontier
from negmas.outcomes import outcome_is_valid, Issue, Outcome, enumerate_outcomes
from negmas.common import (
    AgentMechanismInterface,
    MechanismState,
    register_all_mechanisms,
    NamedObject,
)
from negmas.common import NegotiatorInfo
from negmas.events import *
from negmas.generics import ikeys
from negmas.helpers import snake_case
from negmas.negotiators import Negotiator

__all__ = ["Mechanism", "Protocol", "MechanismRoundResult"]


@dataclass
class MechanismRoundResult:
    broken: bool = False
    """True only if END_NEGOTIATION was selected by one agent"""
    timedout: bool = False
    """True if a timeout occurred. Usually not used"""
    agreement: Optional["Outcome"] = None
    """The agreement if any"""
    error: bool = False
    """True if an error occurred in the mechanism"""
    error_details: str = ""
    """Error message"""


# noinspection PyAttributeOutsideInit
class Mechanism(NamedObject, EventSource, ABC):
    """
    Base class for all negotiation Mechanisms.

    Override the `round` function of this class to implement a round of your mechanism
    """

    all: Dict[str, "Mechanism"] = {}
    register_all_mechanisms(all)

    def __init__(
        self,
        issues: List["Issue"] = None,
        outcomes: Union[int, List["Outcome"]] = None,
        n_steps: int = None,
        time_limit: float = None,
        step_time_limit: float = None,
        max_n_agents: int = None,
        dynamic_entry=False,
        cache_outcomes=True,
        max_n_outcomes: int = 1000000,
        keep_issue_names=True,
        annotation: Optional[Dict[str, Any]] = None,
        state_factory=MechanismState,
        enable_callbacks=False,
        name=None,
    ):
        """

        Args:
            issues: List of issues to use (optional as you can pass `outcomes`)
            outcomes: List of outcomes (optional as you can pass `issues`). If an int then it is the number of outcomes
            n_steps: Number of rounds allowed (None means infinity)
            time_limit: Number of real seconds allowed (None means infinity)
            max_n_agents:  Maximum allowed number of agents
            dynamic_entry: Allow agents to enter/leave negotiations between rounds
            cache_outcomes: If true, a list of all possible outcomes will be cached
            max_n_outcomes: The maximum allowed number of outcomes in the cached set
            keep_issue_names: If True, dicts with issue names will be used for outcomes otherwise tuples
            annotation: Arbitrary annotation
            state_factory: A callable that receives an arbitrary set of key-value pairs and return a MechanismState
            descendant object
            name: Name of the mechanism session. Should be unique. If not given, it will be generated.
        """
        super().__init__(name=name)
        # parameters fixed for all runs
        if issues is None:
            if outcomes is None:
                raise ValueError(
                    "Not issues or outcomes are given to this mechanism. Cannot be constructed"
                )
            else:
                if isinstance(outcomes, int):
                    outcomes = [(_,) for _ in range(outcomes)]
                else:
                    outcomes = list(outcomes)
                n_issues = len(outcomes[0])
                issues = []
                issue_names = ikeys(outcomes[0])
                for issue in range(n_issues):
                    vals = list(set([_[issue] for _ in outcomes]))
                    issues.append(vals)
                __issues = [
                    Issue(_, name=name_) for _, name_ in zip(issues, issue_names)
                ]
        else:
            __issues = list(issues)
            if outcomes is None and cache_outcomes:
                try:
                    if len(__issues) == 0:
                        outcomes = []
                    else:
                        n_outcomes = 1
                        for issue in __issues:
                            if issue.is_continuous():
                                break
                            n_outcomes *= issue.cardinality()
                            if n_outcomes > max_n_outcomes:
                                break
                        else:
                            outcomes = enumerate_outcomes(
                                __issues, keep_issue_names=keep_issue_names
                            )

                except ValueError:
                    pass
            elif outcomes is not None:
                issue_names = [_.name for _ in issues]
                assert (keep_issue_names and isinstance(outcomes[0], dict)) or (
                    not keep_issue_names and isinstance(outcomes[0], tuple)
                ), (
                    f"Either you request to keep issue"
                    f" names but use tuple outcomes or "
                    f"vice versa (names={keep_issue_names}"
                    f", type={type(outcomes[0])})"
                )
                if keep_issue_names:
                    for i, outcome in enumerate(outcomes):
                        assert list(outcome.keys()) == issue_names
                else:
                    for i, outcome in enumerate(outcomes):
                        assert len(outcome) == len(issue_names)

        __outcomes = outcomes
        # we have now __issues is a List[Issue] and __outcomes is Optional[List[Outcome]]

        # create a couple of ways to access outcomes by indices effeciently
        self.outcome_index = lambda x: None
        self.outcome_indices = []
        if __outcomes is not None and cache_outcomes:
            self.outcome_indices = range(len(__outcomes))
            try:
                _outcome_index = dict(zip(__outcomes, self.outcome_indices))
                self.outcome_index = lambda x: _outcome_index[x]
            except:
                self.outcome_index = lambda x: __outcomes.index(x)

        self.id = str(uuid.uuid4())
        self.ami = AgentMechanismInterface(
            id=self.id,
            n_outcomes=None if outcomes is None else len(outcomes),
            issues=__issues,
            outcomes=__outcomes,
            time_limit=time_limit,
            n_steps=n_steps,
            step_time_limit=step_time_limit,
            dynamic_entry=dynamic_entry,
            max_n_agents=max_n_agents,
            annotation=annotation,
        )

        self._history = []
        # if self.ami.issues is not None:
        #     self.ami.issues = tuple(self.ami.issues)
        # if self.ami.outcomes is not None:
        #     self.ami.outcomes = tuple(self.ami.outcomes)
        self._state_factory = state_factory
        Mechanism.all[self.id] = self

        self._requirements = {}
        self._negotiators = []
        self._roles = []
        self._start_time = None
        self._started = False
        self._step = 0
        self._n_accepting_agents = 0
        self._broken = False
        self._agreement = None
        self._timedout = False
        self._running = False
        self._error = False
        self._error_details = ""
        self.__discrete_outcomes: List[Outcome] = None
        self._enable_callbacks = enable_callbacks

        self.agents_of_role = defaultdict(list)
        self.role_of_agent = {}

    @classmethod
    def get_info(cls, id: str) -> AgentMechanismInterface:
        """Returns the mechanism information which contains its static config plus methods to access current state"""
        return cls.all[id].ami

    @property
    def participants(self) -> List[NegotiatorInfo]:
        """Returns a list of all participant names"""
        return [
            NegotiatorInfo(name=_.name, id=_.id, type=snake_case(_.__class__.__name__))
            for _ in self.negotiators
        ]

    def is_valid(self, outcome: "Outcome"):
        """Checks whether the outcome is valid given the issues"""
        if self.ami.issues is None or len(self.ami.issues) == 0:
            raise ValueError("I do not have any issues to check")

        return outcome_is_valid(outcome, self.ami.issues)

    def discrete_outcomes(
        self, n_max: int = None, astype: Type["Outcome"] = dict
    ) -> List["Outcome"]:
        """
        A discrete set of outcomes that spans the outcome space

        Args:
            n_max: The maximum number of outcomes to return. If None, all outcomes will be returned for discrete issues
            and *100* if any of the issues was continuous
            astype: A type to cast the resulting outcomes to.

        Returns:

            List[Outcome]: List of `n` or less outcomes

        """
        if self.outcomes is not None:
            return self.outcomes
        if self.__discrete_outcomes is None:
            if all(issue.is_discrete() for issue in self.issues):
                self.__discrete_outcomes = Issue.sample(
                    issues=self.issues,
                    n_outcomes=n_max,
                    astype=astype,
                    with_replacement=False,
                    fail_if_not_enough=False,
                )
            else:
                self.__discrete_outcomes = Issue.sample(
                    issues=self.issues,
                    n_outcomes=n_max if n_max is not None else 100,
                    astype=astype,
                    with_replacement=False,
                    fail_if_not_enough=False,
                )
        return self.__discrete_outcomes

    def random_outcomes(
        self, n: int = 1, astype: Type[Outcome] = dict
    ) -> List["Outcome"]:
        """Returns random offers"""
        if self.ami.issues is None or len(self.ami.issues) == 0:
            raise ValueError("I do not have any issues to generate offers from")
        return Issue.sample(
            issues=self.issues,
            n_outcomes=n,
            astype=astype,
            with_replacement=False,
            fail_if_not_enough=False,
        )

    @property
    def time(self) -> Optional[float]:
        """Elapsed time since mechanism started in seconds. None if the mechanism did not start running"""
        if self._start_time is None:
            return 0.0

        return time.monotonic() - self._start_time

    @property
    def remaining_time(self) -> Optional[float]:
        """Returns remaining time in seconds. None if no time limit is given."""
        if self.ami.time_limit is None:
            return None

        limit = self.ami.time_limit - (time.monotonic() - self._start_time)
        if limit < 0.0:
            return 0.0

        return limit

    @property
    def relative_time(self) -> Optional[float]:
        """Returns a number between ``0`` and ``1`` indicating elapsed relative time or steps."""
        if self.ami.time_limit is None and self.ami.n_steps is None:
            return None

        relative_step = (
            (self._step + 1) / self.ami.n_steps
            if self.ami.n_steps is not None
            else -1.0
        )
        relative_time = (
            self.time / self.ami.time_limit if self.ami.time_limit is not None else -1.0
        )
        return max([relative_step, relative_time])

    @property
    def remaining_steps(self) -> Optional[int]:
        """Returns the remaining number of steps until the end of the mechanism run. None if unlimited"""
        if self.ami.n_steps is None:
            return None

        return self.ami.n_steps - self._step

    def add(
        self,
        negotiator: "Negotiator",
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: Optional[str] = None,
        **kwargs,
    ) -> Optional[bool]:
        """Add an agent to the negotiation.

        Args:

            negotiator: The agent to be added.
            ufun: The utility function to use. If None, then the agent must already have a stored
                  utility function otherwise it will fail to enter the negotiation.
            role: The role the agent plays in the negotiation mechanism. It is expected that mechanisms inheriting from
                  this class will check this parameter to ensure that the role is a valid role and is still possible for
                  negotiators to join on that role. Roles may include things like moderator, representative etc based
                  on the mechanism


        Returns:

            * True if the agent was added.
            * False if the agent was already in the negotiation.
            * None if the agent cannot be added.

        """
        if not self.can_enter(negotiator):
            return None

        if negotiator in self._negotiators:
            return False

        if isinstance(ufun, Iterable) and not isinstance(ufun, UtilityFunction):
            if isinstance(ufun, dict):
                ufun = MappingUtilityFunction(mapping=ufun, ami=self.ami)
            else:
                ufun = MappingUtilityFunction(
                    mapping=dict(zip(self.outcomes, ufun)), ami=self.ami
                )

        if role is None:
            role = "agent"

        if negotiator.join(ami=self.ami, state=self.state, ufun=ufun, role=role):
            self._negotiators.append(negotiator)
            self._roles.append(role)
            self.role_of_agent[negotiator.uuid] = role
            self.agents_of_role[role].append(negotiator)
            return True
        return None

    def remove(self, agent: "Negotiator", **kwargs) -> Optional[bool]:
        """Remove the agent from the negotiation.

        Args:
            agent:

        Returns:
            * True if the agent was removed.
            * False if the agent was not in the negotiation already.
            * None if the agent cannot be removed.
        """
        try:
            indx = self._negotiators.index(agent)
        except ValueError:
            return False

        if not self.can_leave(agent):
            return None
        self._negotiators = self._negotiators[0:indx] + self._negotiators[indx + 1 :]
        if self._enable_callbacks:
            agent.on_leave(self.ami, **kwargs)
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

    @property
    def negotiators(self):
        return self._negotiators

    @property
    def requirements(self):
        """A dictionary specifying the requirements that must be in the capabilities of any agent
        to join the mechanism.

        """
        return self._requirements

    @requirements.setter
    def requirements(
        self,
        requirements: Dict[
            str,
            Union[
                Tuple[Union[int, float, str], Union[int, float, str]],
                List,
                Set,
                Union[int, float, str],
            ],
        ],
    ):
        self._requirements = {
            k: set(v) if isinstance(v, list) else v for k, v in requirements.items()
        }

    @property
    def agreement(self):
        return self._agreement

    def is_satisfying(self, capabilities: dict) -> bool:
        """Checks if the  given capabilities are satisfying mechanism requirements.

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

    def can_participate(self, agent: "Negotiator") -> bool:
        """Checks if the agent can participate in this type of negotiation in general.

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

    @property
    def n_outcomes(self):
        return self.ami.n_outcomes

    @property
    def issues(self):
        return self.ami.issues

    @property
    def completed(self):
        return self.agreement is not None or self._broken

    @property
    def outcomes(self):
        return self.ami.outcomes

    @property
    def n_steps(self):
        return self.ami.n_steps

    @property
    def time_limit(self):
        return self.ami.time_limit

    @property
    def running(self):
        return self._running

    @property
    def dynamic_entry(self):
        return self.ami.dynamic_entry

    @property
    def max_n_agents(self):
        return self.ami.max_n_agents

    @max_n_agents.setter
    def max_n_agents(self, n: int):
        self.ami.max_n_agents = n

    def can_accept_more_agents(self) -> bool:
        """Whether the mechanism can **currently** accept more negotiators."""
        return (
            True
            if self.ami.max_n_agents is None or self._negotiators is None
            else len(self._negotiators) < self.ami.max_n_agents
        )

    def can_leave(self, agent: "Negotiator") -> bool:
        """Can the agent leave now?"""
        return (
            True
            if self.ami.dynamic_entry
            else not self.ami.state.running and agent in self._negotiators
        )

    def can_enter(self, agent: "Negotiator") -> bool:
        """Whether the agent can enter the negotiation now."""
        return self.can_accept_more_agents() and self.can_participate(agent)

    def __iter__(self):
        return self

    def __next__(self) -> MechanismState:
        result = self.step()
        if not self._running:
            raise StopIteration

        return result

    def step(self) -> MechanismState:
        """Runs a single step of the mechanism.

        Returns:
            MechanismState: The state of the negotiation *after* the round is conducted

        Remarks:

            - Every call yields the results of one round (see `round()`)
            - If the mechanism was yet to start, it will start it and runs one round
            - There is another function (`run()`) that runs the whole mechanism in blocking mode

        """
        if self.time_limit is not None and self.time > self.time_limit:
            self._agreement, self._broken, self._timedout = None, False, True
            self._history.append(self.state)
            return self.state
        if len(self._negotiators) < 2:
            if self.ami.dynamic_entry:
                self._history.append(self.state)
                return self.state
            else:
                self.ami.state.running = False
                self._agreement, self._broken, self._timedout = None, False, False
                self._history.append(self.state)
                self.on_negotiation_end()
                return self.state

        if self._broken or self._timedout or self._agreement is not None:
            self._history.append(self.state)
            return self.state

        if not self._running:
            self._running = True
            self._step = 0
            self._start_time = time.monotonic()
            self._started = True
            if self.on_negotiation_start() is False:
                self._agreement, self._broken, self._timedout = None, False, False
                self._history.append(self.state)
                return self.state
            if self._enable_callbacks:
                for a in self.negotiators:
                    a.on_negotiation_start(state=self.state)
            self.announce(Event(type="negotiation_start", data=None))
        else:
            remaining_steps, remaining_time = self.remaining_steps, self.remaining_time
            if (remaining_steps is not None and remaining_steps <= 0) or (
                remaining_time is not None and remaining_time <= 0.0
            ):
                self._running = False
                self._agreement, self._broken, self._timedout = None, False, True
                self._history.append(self.state)
                self.on_negotiation_end()
                return self.state

        if self._enable_callbacks:
            for agent in self._negotiators:
                agent.on_round_start(state=self.state)
        step_start = time.perf_counter()
        result = self.round()
        step_time = time.perf_counter() - step_start
        self._error, self._error_details = result.error, result.error_details
        if self._error:
            self.on_mechanism_error()
        if (
            self.ami.step_time_limit is not None
            and step_time > self.ami.step_time_limit
        ):
            self._broken, self._timedout, self._agreement = False, True, None
        else:
            self._broken, self._timedout, self._agreement = (
                result.broken,
                result.timedout,
                result.agreement,
            )
        if (self._agreement is not None) or self._broken or self._timedout:
            self._running = False
        self._step += 1
        if self._enable_callbacks:
            for agent in self._negotiators:
                agent.on_round_end(state=self.state)
        if not self._running:
            self.on_negotiation_end()
        self._history.append(self.state)
        return self.state

    def run(self, timeout=None) -> MechanismState:
        if timeout is None:
            for _ in self:
                pass
        else:
            start_time = time.perf_counter()
            for _ in self:
                if time.perf_counter() - start_time > timeout:
                    self._running, self._timedout = False, True
                    self.on_negotiation_end()
                    break
        return self.state

    def on_mechanism_error(self) -> None:
        """
        Called when there is a mechanism error

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        if self._enable_callbacks:
            for a in self.negotiators:
                a.on_mechanism_error(state=self.state)

    def on_negotiation_end(self) -> None:
        """
        Called at the end of each negotiation

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        if self._enable_callbacks:
            for a in self.negotiators:
                a.on_negotiation_end(state=self.state)
        self.announce(
            Event(
                type="negotiation_end",
                data={
                    "agreement": self.agreement,
                    "state": self.state,
                    "annotation": self.ami.annotation,
                },
            )
        )

    def on_negotiation_start(self) -> bool:
        """Called before starting the negotiation. If it returns False then negotiation will end immediately"""
        return True

    @property
    def history(self):
        return self._history

    @property
    def state(self):
        """Returns the current state. Override `extra_state` if you want to keep extra state"""
        current_state = self.extra_state()
        if current_state is None:
            current_state = {}
        else:
            current_state = dict(current_state)
        current_state.update(
            {
                "running": self._running,
                "step": self._step,
                "time": self.time,
                "relative_time": self.relative_time,
                "broken": self._broken,
                "timedout": self._timedout,
                "started": self._started,
                "agreement": self._agreement,
                "n_negotiators": len(self.negotiators),
                "has_error": self._error,
                "error_details": self._error_details,
            }
        )
        return self._state_factory(**current_state)

    def pareto_frontier(
        self, n_max=None, sort_by_welfare=True
    ) -> Tuple[List[Tuple[float]], List["Outcome"]]:
        ufuns = self._get_ufuns()
        if any(_ is None for _ in ufuns):
            return [], []
        frontier, indices = pareto_frontier(
            ufuns=ufuns,
            n_discretization=None,
            sort_by_welfare=sort_by_welfare,
            outcomes=self.discrete_outcomes(n_max=n_max),
        )
        return frontier, [self.discrete_outcomes(n_max=n_max)[_] for _ in indices]

    def __str__(self):
        d = self.__dict__.copy()
        return pprint.pformat(d)

    __repr__ = __str__

    def _get_ufuns(self):
        ufuns = []
        for a in self.negotiators:
            ufuns.append(a.utility_function)
        return ufuns

    def plot(self, plot_utils=True, plot_outcomes=True):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if len(self.negotiators) > 2:
            print("Cannot visualize negotiations with more than 2 negotiators")
        else:
            # has_front = int(len(self.outcomes[0]) <2)
            has_front = 1
            n_agents = len(self.negotiators)
            history = pd.DataFrame(data=[_.__dict__ for _ in self.history])
            # history['time'] = [_.time for _ in self.history]
            # history['relative_time'] = [_.relative_time for _ in self.history]
            # history['step'] = [_.step for _ in self.history]
            history = history.loc[~history.current_offer.isnull(), :]
            ufuns = self._get_ufuns()
            outcomes = self.outcomes

            utils = [tuple(f(o) for f in ufuns) for o in outcomes]
            agent_names = [a.name for a in self.negotiators]
            history["offer_index"] = [outcomes.index(_) for _ in history.current_offer]
            frontier, frontier_outcome = self.pareto_frontier(sort_by_welfare=True)
            frontier_outcome_indices = [outcomes.index(_) for _ in frontier_outcome]
            if plot_utils:
                fig_util = plt.figure()
            if plot_outcomes:
                fig_outcome = plt.figure()
            gs_util = gridspec.GridSpec(n_agents, has_front + 1) if plot_utils else None
            gs_outcome = (
                gridspec.GridSpec(n_agents, has_front + 1) if plot_outcomes else None
            )
            axs_util, axs_outcome = [], []

            for a in range(n_agents):
                if a == 0:
                    if plot_utils:
                        axs_util.append(fig_util.add_subplot(gs_util[a, has_front]))
                    if plot_outcomes:
                        axs_outcome.append(
                            fig_outcome.add_subplot(gs_outcome[a, has_front])
                        )
                else:
                    if plot_utils:
                        axs_util.append(
                            fig_util.add_subplot(
                                gs_util[a, has_front], sharex=axs_util[0]
                            )
                        )
                    if plot_outcomes:
                        axs_outcome.append(
                            fig_outcome.add_subplot(
                                gs_outcome[a, has_front], sharex=axs_outcome[0]
                            )
                        )
                if plot_utils:
                    axs_util[-1].set_ylabel(agent_names[a])
                if plot_outcomes:
                    axs_outcome[-1].set_ylabel(agent_names[a])
            for a, (au, ao) in enumerate(
                zip(
                    itertools.chain(axs_util, itertools.repeat(None)),
                    itertools.chain(axs_outcome, itertools.repeat(None)),
                )
            ):
                if au is None and ao is None:
                    break
                h = history.loc[
                    history.current_proposer == self.negotiators[a].id,
                    ["relative_time", "offer_index", "current_offer"],
                ]
                h["utility"] = h["current_offer"].apply(ufuns[a])
                if plot_outcomes:
                    ao.plot(h.relative_time, h["offer_index"])
                if plot_utils:
                    au.plot(h.relative_time, h.utility)
                    au.set_ylim(0.0, 1.0)

            if has_front:
                if plot_utils:
                    axu = fig_util.add_subplot(gs_util[:, 0])
                    axu.scatter(
                        [_[0] for _ in utils],
                        [_[1] for _ in utils],
                        label="outcomes",
                        color="gray",
                        marker="s",
                        s=20,
                    )
                if plot_outcomes:
                    axo = fig_outcome.add_subplot(gs_outcome[:, 0])
                clrs = ("blue", "green")
                if plot_utils:
                    for a in range(n_agents):
                        h = history.loc[
                            history.current_proposer == self.negotiators[a].id,
                            ["relative_time", "offer_index", "current_offer"],
                        ]
                        h["u0"] = h["current_offer"].apply(ufuns[0])
                        h["u1"] = h["current_offer"].apply(ufuns[1])
                        axu.scatter(
                            h.u0, h.u1, color=clrs[a], label=f"{agent_names[a]}"
                        )
                if plot_outcomes:
                    steps = sorted(history.step.unique().tolist())
                    aoffers = [[], []]
                    for step in steps[::2]:
                        offrs = []
                        for a in range(n_agents):
                            a_offer = history.loc[
                                (history.current_proposer == agent_names[a])
                                & ((history.step == step) | (history.step == step + 1)),
                                "offer_index",
                            ]
                            if len(a_offer) > 0:
                                offrs.append(a_offer.values[-1])
                        if len(offrs) == 2:
                            aoffers[0].append(offrs[0])
                            aoffers[1].append(offrs[1])
                    axo.scatter(aoffers[0], aoffers[1], color=clrs[0], label=f"offers")

                if self.state.agreement is not None:
                    if plot_utils:
                        axu.scatter(
                            [ufuns[0](self.state.agreement)],
                            [ufuns[1](self.state.agreement)],
                            color="black",
                            marker="*",
                            s=120,
                            label="SCMLAgreement",
                        )
                    if plot_outcomes:
                        axo.scatter(
                            [outcomes.index(self.state.agreement)],
                            [outcomes.index(self.state.agreement)],
                            color="black",
                            marker="*",
                            s=120,
                            label="SCMLAgreement",
                        )

                if plot_utils:
                    f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
                    axu.scatter(f1, f2, label="frontier", color="red", marker="x")
                    axu.legend()
                    axu.set_xlabel(agent_names[0] + " utility")
                    axu.set_ylabel(agent_names[1] + " utility")
                    if self.agreement is not None:
                        pareto_distance = 1e9
                        cu = (ufuns[0](self.agreement), ufuns[1](self.agreement))
                        for pu in frontier:
                            dist = math.sqrt(
                                (pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2
                            )
                            if dist < pareto_distance:
                                pareto_distance = dist
                        axu.text(
                            0,
                            0.95,
                            f"Pareto-distance={pareto_distance:5.2}",
                            verticalalignment="top",
                            transform=axu.transAxes,
                        )

                if plot_outcomes:
                    axo.scatter(
                        frontier_outcome_indices,
                        frontier_outcome_indices,
                        color="red",
                        marker="x",
                        label="frontier",
                    )
                    axo.legend()
                    axo.set_xlabel(agent_names[0])
                    axo.set_ylabel(agent_names[1])

            if plot_utils:
                fig_util.show()
            if plot_outcomes:
                fig_outcome.show()

    @abstractmethod
    def round(self) -> MechanismRoundResult:
        """ Implements a single step of the mechanism. Override this!

        Returns:
            MechanismRoundResult giving whether the negotiation was broken or timedout and the agreement if any.

        """
        raise NotImplementedError(
            "You must inherit from Mechanism and override its round() function"
        )

    def extra_state(self) -> Optional[Dict[str, Any]]:
        """Returns any extra state information to be kept in the `state` and `history` properties"""
        return None


Protocol = Mechanism
"""An alias for `Mechanism`"""
