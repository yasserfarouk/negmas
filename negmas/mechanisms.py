"""
Provides interfaces for defining negotiation mechanisms.
"""
import pprint
import random
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from negmas.checkpoints import CheckpointMixin
from negmas.common import (
    AgentMechanismInterface,
    MechanismState,
    NamedObject,
    NegotiatorInfo,
)
from negmas.events import Event, EventSource
from negmas.generics import ikeys
from negmas.helpers import snake_case
from negmas.negotiators import Negotiator
from negmas.outcomes import (
    Issue,
    Outcome,
    enumerate_outcomes,
    outcome_as_tuple,
    outcome_is_valid,
)
from negmas.utilities import MappingUtilityFunction, UtilityFunction, pareto_frontier
from negmas.genius import (
    DEFAULT_JAVA_PORT,
    get_free_tcp_port,
)

__all__ = ["Mechanism", "Protocol", "MechanismRoundResult"]


@dataclass
class MechanismRoundResult:
    broken: bool = False
    """True only if END_NEGOTIATION was selected by one agent"""
    timedout: bool = False
    """True if a timeout occurred. Usually not used"""
    agreement: Optional[Union[Collection["Outcome"], "Outcome"]] = None
    """The agreement if any. Allows for a single outcome or a collection of outcomes"""
    error: bool = False
    """True if an error occurred in the mechanism"""
    error_details: str = ""
    """Error message"""
    waiting: bool = False
    """whether to consider that the round is still running and call the round method again without increasing
    the step number"""
    exceptions: Optional[Dict[str, List[str]]] = None
    """A mapping from negotiator ID to a list of exceptions raised by that negotiator in this round"""
    times: Optional[Dict[str, float]] = None
    """A mapping from negotiator ID to the time it consumed during this round"""


# noinspection PyAttributeOutsideInit
class Mechanism(NamedObject, EventSource, CheckpointMixin, ABC):
    """
    Base class for all negotiation Mechanisms.

    Override the `round` function of this class to implement a round of your mechanism

    Args:
        issues: List of issues to use (optional as you can pass `outcomes`)
        outcomes: List of outcomes (optional as you can pass `issues`). If an int then it is the number of outcomes
        n_steps: Number of rounds allowed (None means infinity)
        time_limit: Number of real seconds allowed (None means infinity)
        max_n_agents:  Maximum allowed number of agents
        dynamic_entry: Allow agents to enter/leave negotiations between rounds
        cache_outcomes: If true, a list of all possible outcomes will be cached
        max_n_outcomes: The maximum allowed number of outcomes in the cached set
        keep_issue_names: DEPRICATED. Use `outcome_type` instead. If True, dicts with issue names will be used for outcomes otherwise tuples
        annotation: Arbitrary annotation
        state_factory: A callable that receives an arbitrary set of key-value pairs and return a MechanismState
                      descendant object
        checkpoint_every: The number of steps to checkpoint after. Set to <= 0 to disable
        checkpoint_folder: The folder to save checkpoints into. Set to None to disable
        checkpoint_filename: The base filename to use for checkpoints (multiple checkpoints will be prefixed with
                             step number).
        single_checkpoint: If true, only the most recent checkpoint will be saved.
        extra_checkpoint_info: Any extra information to save with the checkpoint in the corresponding json file as
                               a dictionary with string keys
        exist_ok: IF true, checkpoints override existing checkpoints with the same filename.
        name: Name of the mechanism session. Should be unique. If not given, it will be generated.
        outcome_type: The type used for representing outcomes. Can be tuple, dict or any `OutcomeType`
        genius_port: the port used to connect to Genius for all negotiators in this mechanism (0 means any).
        id: An optional system-wide unique identifier. You should not change
            the default value except in special circumstances like during
            serialization and should always guarantee system-wide uniquness
            if you set this value explicitly
    """

    def __init__(
        self,
        issues: List["Issue"] = None,
        outcomes: Union[int, List["Outcome"]] = None,
        n_steps: int = None,
        time_limit: float = None,
        step_time_limit: float = None,
        negotiator_time_limit: float = None,
        max_n_agents: int = None,
        dynamic_entry=False,
        cache_outcomes=True,
        max_n_outcomes: int = 1000000,
        keep_issue_names=None,
        annotation: Optional[Dict[str, Any]] = None,
        state_factory=MechanismState,
        enable_callbacks=False,
        checkpoint_every: int = 1,
        checkpoint_folder: Optional[Union[str, Path]] = None,
        checkpoint_filename: str = None,
        extra_checkpoint_info: Dict[str, Any] = None,
        single_checkpoint: bool = True,
        exist_ok: bool = True,
        name=None,
        outcome_type=tuple,
        genius_port: int = DEFAULT_JAVA_PORT,
        id: str = None,
    ):
        super().__init__(name, id=id)
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
        time_limit = time_limit if time_limit is not None else float("inf")
        step_time_limit = (
            step_time_limit if step_time_limit is not None else float("inf")
        )
        negotiator_time_limit = (
            negotiator_time_limit if negotiator_time_limit is not None else float("inf")
        )

        if keep_issue_names is not None:
            warnings.warn(
                "keep_issue_names is depricated. Use outcome_type instead.\n"
                "keep_issue_names=True <--> outcome_type=dict\n"
                "keep_issue_names=False <--> outcome_type=tuple\n",
                DeprecationWarning,
            )
            outcome_type = dict if keep_issue_names else tuple
        keep_issue_names = not issubclass(outcome_type, tuple)
        # parameters fixed for all runs
        if issues is None:
            if outcomes is None:
                __issues = []
                outcomes = []
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
                            if issue.is_uncountable():
                                break
                            n_outcomes *= issue.cardinality
                            if n_outcomes > max_n_outcomes:
                                break
                        else:
                            outcomes = enumerate_outcomes(__issues, astype=outcome_type)

                except ValueError:
                    pass
            elif outcomes is not None:
                issue_names = [_.name for _ in issues]
                assert (not keep_issue_names and isinstance(outcomes[0], tuple)) or (
                    keep_issue_names and not isinstance(outcomes[0], tuple)
                ), (
                    f"Either you request to keep issue"
                    f" names but use tuple outcomes or "
                    f"vice versa (names={keep_issue_names}"
                    f", type={type(outcomes[0])})"
                )
                if keep_issue_names:
                    for outcome in outcomes:
                        assert list(outcome.keys()) == issue_names
                else:
                    for i, outcome in enumerate(outcomes):
                        assert len(outcome) == len(issue_names)

        self.__outcomes = outcomes
        # we have now __issues is a List[Issue] and __outcomes is Optional[List[Outcome]]

        # create a couple of ways to access outcomes by indices effeciently
        self.outcome_indices = []
        self.__outcome_index = None
        if self.__outcomes is not None and cache_outcomes:
            self.outcome_indices = range(len(self.__outcomes))
            self.__outcome_index = dict(
                zip(
                    (outcome_as_tuple(o) for o in self.__outcomes),
                    range(len(self.__outcomes)),
                )
            )

        self.id = str(uuid.uuid4())
        _imap = dict(zip((_.name for _ in __issues), range(len(__issues))))
        _imap.update(dict(zip(range(len(__issues)), (_.name for _ in __issues))))
        self.ami = AgentMechanismInterface(
            id=self.id,
            n_outcomes=None if outcomes is None else len(outcomes),
            issues=__issues,
            outcomes=self.__outcomes,
            time_limit=time_limit,
            n_steps=n_steps,
            step_time_limit=step_time_limit,
            negotiator_time_limit=negotiator_time_limit,
            dynamic_entry=dynamic_entry,
            max_n_agents=max_n_agents,
            annotation=annotation,
            outcome_type=dict if keep_issue_names else tuple,
            imap=_imap,
        )
        self.ami._mechanism = self

        self._history: List[MechanismState] = []
        self._stats: Dict[str, Any] = dict()
        self._stats["round_times"] = list()
        self._stats["times"] = defaultdict(float)
        self._stats["exceptions"] = defaultdict(list)
        # if self.ami.issues is not None:
        #     self.ami.issues = tuple(self.ami.issues)
        # if self.ami.outcomes is not None:
        #     self.ami.outcomes = tuple(self.ami.outcomes)
        self._state_factory = state_factory

        self._requirements = {}
        self._negotiators = []
        self._negotiator_map: Dict[str, "SAONegotiator"] = dict()
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
        self._waiting = False
        self.__discrete_outcomes: List[Outcome] = None
        self._enable_callbacks = enable_callbacks

        self.agents_of_role = defaultdict(list)
        self.role_of_agent = {}
        # mechanisms do not differentiate between RANDOM_JAVA_PORT and ANY_JAVA_PORT.
        # if either is given as the genius_port, it will fix a port and all negotiators
        # that are not explicitly assigned to a port (by passing port>0 to them) will just
        # use that port.
        self.genius_port = genius_port if genius_port > 0 else get_free_tcp_port()

        self.params = dict(
            dynamic_entry=dynamic_entry,
            genius_port=genius_port,
            cache_outcomes=cache_outcomes,
            annotation=annotation,
        )

    def outcome_index(self, outcome) -> Optional[int]:
        """Returns the index of the outcome if that was possible"""
        if self.__outcomes is None:
            return None
        if self.__outcome_index is not None:
            return self.__outcome_index[outcome_as_tuple(outcome)]
        return self.__outcomes.index(outcome)

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
        self, n_max: int = None, astype: Type["Outcome"] = None
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
        if astype is None:
            astype = self.ami.outcome_type
        if self.outcomes is not None:
            return self.outcomes
        if self.__discrete_outcomes is None:
            if all(issue.is_countable() for issue in self.issues):
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
        self, n: int = 1, astype: Type[Outcome] = None
    ) -> List["Outcome"]:
        """Returns random offers.

        Args:
              n: Number of outcomes to generate
              astype: The type to use for the generated outcomes

        Returns:
              A list of outcomes (in the type specified using `astype`) of at most n outcomes.

        Remarks:

                - If the number of outcomes `n` cannot be satisfied, a smaller number will be returned
                - Sampling is done without replacement (i.e. returned outcomes are unique).

        """
        if astype is None:
            astype = self.ami.outcome_type
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
    def time(self) -> float:
        """Elapsed time since mechanism started in seconds. 0.0 if the mechanism did not start running"""
        if self._start_time is None:
            return 0.0

        return time.perf_counter() - self._start_time

    @property
    def remaining_time(self) -> Optional[float]:
        """Returns remaining time in seconds. None if no time limit is given."""
        if self.ami.time_limit == float("+inf"):
            return None

        limit = self.ami.time_limit - (time.perf_counter() - self._start_time)
        if limit < 0.0:
            return 0.0

        return limit

    @property
    def relative_time(self) -> Optional[float]:
        """Returns a number between ``0`` and ``1`` indicating elapsed relative time or steps."""
        if self.ami.time_limit == float("+inf") and self.ami.n_steps is None:
            return None

        relative_step = (
            self._step / self.ami.n_steps if self.ami.n_steps is not None else -1.0
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

        if negotiator.join(
            ami=self._get_ami(negotiator, role), state=self.state, ufun=ufun, role=role
        ):
            self._negotiators.append(negotiator)
            self._negotiator_map[negotiator.id] = negotiator
            self._roles.append(role)
            self.role_of_agent[negotiator.uuid] = role
            self.agents_of_role[role].append(negotiator)
            if negotiator.ufun is not None:
                if hasattr(negotiator.ufun, "ami"):
                    negotiator.ufun.ami = self.ami
            return True
        return None

    def get_negotiator(self, nid: str) -> Optional["SAONegotiator"]:
        """Returns the negotiator with the given ID if present in the negotiation"""
        return self._negotiator_map.get(nid, None)

    def remove(self, negotiator: "Negotiator", **kwargs) -> Optional[bool]:
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
        if self._enable_callbacks:
            negotiator.on_leave(self.ami, **kwargs)
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
    def negotiator_ids(self) -> List[str]:
        return [_.id for _ in self._negotiators]

    @property
    def agent_ids(self) -> List[str]:
        return [_.owner.id for _ in self._negotiators if _.owner]

    @property
    def agent_names(self) -> List[str]:
        return [_.owner.name for _ in self._negotiators if _.owner]

    @property
    def negotiator_names(self) -> List[str]:
        return [_.name for _ in self._negotiators]

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

    def abort(self) -> MechanismState:
        """
        Aborts the negotiation
        """
        self._error, self._error_details, self._waiting = (
            True,
            "Uncaught Exception",
            False,
        )
        self.on_mechanism_error()
        self._broken, self._timedout, self._agreement = (True, False, None)
        state = self.state
        state4history = self.state4history
        self._running = False
        if self._enable_callbacks:
            for agent in self._negotiators:
                agent.on_round_end(state)
        self._history.append(state4history)
        self._step += 1
        self.on_negotiation_end()
        return state

    @property
    def state4history(self) -> Any:
        """Returns the state as it should be stored in the history"""
        return self.state

    def step(self) -> MechanismState:
        """Runs a single step of the mechanism.

        Returns:
            MechanismState: The state of the negotiation *after* the round is conducted

        Remarks:

            - Every call yields the results of one round (see `round()`)
            - If the mechanism was yet to start, it will start it and runs one round
            - There is another function (`run()`) that runs the whole mechanism in blocking mode

        """
        if self._start_time is None or self._start_time < 0:
            self._start_time = time.perf_counter()
        self.checkpoint_on_step_started()
        state = self.state
        state4history = self.state4history
        if (self.time > self.time_limit) or (
            self.ami.n_steps and self._step >= self.ami.n_steps
        ):
            self._running, self._broken, self._timedout = False, False, True
            # self._history.append(self.state4history)
            self.on_negotiation_end()
            return self.state
        if len(self._negotiators) < 2:
            if self.ami.dynamic_entry:
                return state
            else:
                self._running, self._broken, self._timedout = False, False, False
                # self._history.append(self.state4history)
                self.on_negotiation_end()
                return self.state

        if self._broken or self._timedout or self._agreement is not None:
            self._running = False
            # self._history.append(self.state4history)
            self.on_negotiation_end()
            return self.state

        if not self._running:
            self._running = True
            self._step = 0
            self._start_time = time.perf_counter()
            self._started = True
            state = self.state
            if self.on_negotiation_start() is False:
                self._agreement, self._broken, self._timedout = None, False, False
                # self._history.append(self.state4history)
                return self.state
            for a in self.negotiators:
                a.on_negotiation_start(state=state)
            self.announce(Event(type="negotiation_start", data=None))
        else:
            remaining_steps, remaining_time = self.remaining_steps, self.remaining_time
            if (remaining_steps is not None and remaining_steps <= 0) or (
                remaining_time is not None and remaining_time <= 0.0
            ):
                self._running = False
                self._agreement, self._broken, self._timedout = None, False, True
                # self._history.append(self.state4history)
                self.on_negotiation_end()
                return self.state

        if not self._waiting:
            if self._enable_callbacks:
                for agent in self._negotiators:
                    agent.on_round_start(state)
        step_start = time.perf_counter() if not self._waiting else self._last_start
        self._last_start = step_start
        self._waiting = False
        result = self.round()
        step_time = time.perf_counter() - step_start
        self._stats["round_times"].append(step_time)
        if result.times:
            for k, v in result.times.items():
                if v is not None:
                    self._stats["times"][k] += v
        if result.exceptions:
            for k, v in result.exceptions.items():
                if v:
                    self._stats["exceptions"][k] += v
        state = self.state
        self._error, self._error_details, self._waiting = (
            result.error,
            result.error_details,
            result.waiting,
        )
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
        state = self.state
        if not self._waiting:
            state4history = self.state4history
            if self._enable_callbacks:
                for agent in self._negotiators:
                    agent.on_round_end(state)
            self._history.append(state4history)
            self._step += 1
        if not self._running:
            self.on_negotiation_end()
        return state

    @classmethod
    def runall(
        cls, mechanisms: List["Mechanism"], keep_order=True, method="serial"
    ) -> List[MechanismState]:
        """
        Runs all mechanisms

        Args:
            mechanisms: List of mechanisms
            keep_order: if True, the mechanisms will be run in order every step otherwise the order will be randomized
                        at every step. This is only allowed if the method is serial
            method: the method to use for running all the sessions.  Acceptable options are: serial, threads, processes

        Returns:
            - List of states of all mechanisms after completion

        """
        completed = [_ is None for _ in mechanisms]
        states = [None] * len(mechanisms)
        if method == "serial":
            while not all(completed):
                lst = zip(completed, mechanisms)
                if not keep_order:
                    lst = list(lst)
                    random.shuffle(lst)
                for i, (done, mechanism) in enumerate(lst):
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
        cls, mechanisms: List["Mechanism"], keep_order=True
    ) -> List[MechanismState]:
        """
        Step all mechanisms

        Args:
            mechanisms: List of mechanisms
            keep_order: if True, the mechanisms will be run in order every step otherwise the order will be randomized
                        at every step

        Returns:
            - List of states of all mechanisms after completion

        """
        if not keep_order:
            raise NotImplementedError(
                "running mechanisms in random order is not yet supported"
            )

        completed = [_ is None for _ in mechanisms]
        states = [None] * len(mechanisms)
        for i, (done, mechanism) in enumerate(zip(completed, mechanisms)):
            if done:
                continue
            result = mechanism.step()
            if result.running:
                continue
            completed[i] = True
            states[i] = mechanism.state
        return states

    def run(self, timeout=None) -> MechanismState:
        if timeout is None:
            for _ in self:
                pass
        else:
            start_time = time.perf_counter()
            for _ in self:
                if time.perf_counter() - start_time > timeout:
                    self._running, self._timedout, self._broken = False, True, False
                    self.on_negotiation_end()
                    break
        return self.state

    def on_mechanism_error(self) -> None:
        """
        Called when there is a mechanism error

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        state = self.state
        if self._enable_callbacks:
            for a in self.negotiators:
                a.on_mechanism_error(state)

    def on_negotiation_end(self) -> None:
        """
        Called at the end of each negotiation.

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        state = self.state
        for a in self.negotiators:
            a.on_negotiation_end(state)
        self.announce(
            Event(
                type="negotiation_end",
                data={
                    "agreement": self.agreement,
                    "state": state,
                    "annotation": self.ami.annotation,
                },
            )
        )
        self.checkpoint_final_step()

    def on_negotiation_start(self) -> bool:
        """Called before starting the negotiation. If it returns False then negotiation will end immediately"""
        return True

    @property
    def history(self):
        return self._history

    @property
    def stats(self):
        return self._stats

    @property
    def current_step(self):
        return self._step

    @property
    def state(self):
        """Returns the current state. Override `extra_state` if you want to keep extra state"""
        d = dict(
            running=self._running,
            step=self._step,
            time=self.time,
            relative_time=self.relative_time,
            broken=self._broken,
            timedout=self._timedout,
            started=self._started,
            agreement=self._agreement,
            n_negotiators=len(self.negotiators),
            has_error=self._error,
            error_details=self._error_details,
            waiting=self._waiting,
        )
        d2 = self.extra_state()
        if d2:
            d.update(d2)
        return self._state_factory(
            **d,
        )

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

    def plot(self, **kwargs):
        """A method for plotting a negotiation session"""

    @abstractmethod
    def round(self) -> MechanismRoundResult:
        """Implements a single step of the mechanism. Override this!

        Returns:
            MechanismRoundResult giving whether the negotiation was broken or timedout and the agreement if any.

        """
        raise NotImplementedError(
            "You must inherit from Mechanism and override its round() function"
        )

    def extra_state(self) -> Optional[Dict[str, Any]]:
        """Returns any extra state information to be kept in the `state` and `history` properties"""
        return dict()

    def _get_ami(self, negotiator: Negotiator, role: str) -> AgentMechanismInterface:
        return self.ami


# @dataclass
# class MechanismSequenceState(MechanismState):
#     current_mechanism_ami: AgentMechanismInterfacea = None
#     current_mechanism_state: MechanismState = None
#
#
# def safemin(a, b):
#     """Returns the minimum assuming None is larger than anything"""
#     if a is None:
#         return b
#     if b is None:
#         return a
#     return min(a, b)
#
# class MechanismSequence(Mechanism):
#     """Represents a sequence of mechanisms with the agreements of one of them as the starting point of the next"""
#
#     def __init__(self, mechanisms: List[Mechanism],
#                  one_round_per_mechanism = True,
#                  issues: List["Issue"] = None,
#                  outcomes: Union[int, List["Outcome"]] = None,
#                  n_steps: int = None,
#                  time_limit: float = None,
#                  step_time_limit: float = None,
#                  max_n_agents: int = None,
#                  dynamic_entry=False,
#                  cache_outcomes=True,
#                  max_n_outcomes: int = 1000000,
#                  keep_issue_names=True,
#                  annotation: Optional[Dict[str, Any]] = None,
#                  state_factory=MechanismState,
#                  enable_callbacks=False,
#                  name=None,
#                  ):
#         super().__init__(issues=issues, outcomes=outcomes, n_steps=n_steps, time_limit=time_limit
#                          , step_time_limit=step_time_limit, max_n_agents=max_n_agents, dynamic_entry=dynamic_entry
#                          , cache_outcomes=cache_outcomes, max_n_outcomes=max_n_outcomes, keep_issue_names=keep_issue_names
#                          , annotation=annotation, state_factory=state_factory, enable_callbacks=enable_callbacks
#                          , name=name)
#         self.mechanisms = list(mechanisms)
#         for mechanism in mechanisms:
#             if mechanism.state.started:
#                 raise ValueError(f'Mechanism {mechanism.id} of type {mechanism.__class__.__name__} is already started. '
#                                  f'Cannot create a mechanism sequence with a mechanism that is already started.')
#             mechanism.ami.time_limit = safemin(mechanism.ami.time_limit, self.time_limit)
#             mechanism.ami.step_time_limit = safemin(mechanism.ami.step_time_limit, self.step_time_limit)
#             if not one_round_per_mechanism:
#                 mechanism.ami.n_steps = safemin(mechanism.ami.n_steps, n_steps)
#             mechanism.ami.dynamic_entry = self.dynamic_entry if not self.dynamic_entry else mechanism.ami.dynamic_entry
#         self._current_mechanism = self.mechanisms[0]
#         self._current_mechanism_index = 0
#         self._current_mechanism.outcomes = self.outcomes
#         self.one_round_per_mechanism = one_round_per_mechanism
#
#     def round(self) -> MechanismRoundResult:
#         if self.one_round_per_mechanism:
#
#


Protocol = Mechanism
"""An alias for `Mechanism`"""
