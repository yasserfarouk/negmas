"""
The implementation file for all entities needed for ANAC-SCML 2019.

Participants need to provide a class inherited from `FactoryManager` that implements all of its abstract functions.

Participants can optionally override any other methods of this class or implement new `NegotiatorUtility` class.

Simulation steps:
-----------------

    #. prepare custom stats (call `_pre_step_stats`)
    #. sign contracts that are to be signed at this step calling `on_contract_signed` as needed
    #. step all existing negotiations `negotiation_speed_multiple` times handling any failed negotaitions and creating
       contracts for any resulting agreements
    #. run all `ActiveEntity` objects registered (i.e. all agents). `Consumer` s run first then `FactoryManager` s then
       `Miner` s
    #. execute contracts that are executable at this time-step handling any breaches
    #. Custom Simulation Steps:
        #. step all factories (`Factory` objects) running any prescheduled commands
        #. Apply interests and pay loans
        #. remove expired `CFP` s
        #. Deliver any products that are in transportation
    #. remove any negotiations that are completed!
    #. update basic stats
    #. update custom stats (call `_post_step_stats`)

Remarks about re-negotiation on breaches:
-----------------------------------------

    - The victim is asked first to specify the negotiation agenda (issues) then the perpetrator
    - renegotiations for breaches run immediately to completion independent from settings of
      `negotiation_speed_multiplier` and `immediate_negotiations`. That include conclusion and signing of any resulting
      agreements.

Remarks about timing:
-------------------------

    - The order of events within a single time-step are as follows:

        #. Scheduled negotiation conclusions which may lead new negotiations if they are done on `on_contract_concluded`
           or `on_negotiation_failure`.
        #. If `immediate_negotiations`, some of the newly added negotiations may be concluded/failed.
        #. Contract signing for any concluded negotiations.
        #. Contract executions including conclusion of any re-negotiations and breach handling. Notice that if
           re-negotiation leads to new contracts, these will be concluded and signed immediately at this step. Please
           note the following about contract execution:

           - products are moved from the seller's storage to a temporary *truck* as long as they are available at the
             time of contract execution. Because contract execution happens *before* actual production, outputs from
             production processes *CANNOT* be sold at the same time-step.

        #. Production is executed on all factories. For a `Process` to start/continue on a `Line`, all its inputs
           required at this time-step **MUST** be available in storage of the corresponding factory *by this pint*.
           This implies that it is impossible for any processes to start at time-step *0* except if initial storage was
           nonzero. `FactoryManager` s are informed about processes that cannot start due to storage or fund shortage
           (or cannot continue due to storage shortage) through an `on_production_failure` call.
        #. Outputs of the `Process` are generated at *the end* of the corresponding time-step. It is immediately moved
           to storage. Because outputs are generated at the *end* of the step and inputs are consumed at the beginning,
           a factory cannot use outputs of a process as inputs to another process that starts at the same time-step.
        #. Products are moved from the temporary *truck* to the buyer's storage after the `transportation_delay` have
           passed at the *end* of the time-step. Transportation completes at the *end* of the time-step no matter
           what is the value for `transportation_delay`. This means that if a `FactoryManager` believes
           that it can produce some product at time *t*, it should never contract to sell it before *t+d + 1* where
           *d* is the `transporation_delay` (the *1* comes from the fact that contract execution happens *before*
           production). Even for a zero transportation delay, you cannot produce something and sell it in the same
           time-step. Moreover, the buyer should never use the product to be delivered at time *t* as an input to a
           production process that needs it before step *t+1*.


Remarks about ANAC 2019 SCML League:
------------------------------------

    Given the information above, and settings for the ANAC 2019 SCML you can confirm for yourself that the following
    rules are all correct:

        #. No agents except miners should contract on delivery at time *0*.
        #. `FactoryManager` s should never sign contracts to sell the output of their production with delivery at *t*
           except if this production starts at step *t* and the contract is signed no later than than *t-1*.
        #. If not all inputs are available in storage, `FactoryManager` s should never sign contracts to sell the output
           of production with delivery at *t* later than *t-2* (and that is optimistic).


"""
import functools
import itertools
import math
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from random import randint, random, sample, choices, gauss
from typing import Dict, Iterable, Any, Callable, Set, Collection, Type
from typing import List, Optional, Tuple, Union

import numpy as np
from dataclasses import dataclass, field
from numpy.random import dirichlet

from negmas.common import NamedObject, MechanismState, MechanismInfo
from negmas.events import Event, EventSource, Notification
from negmas.helpers import ConfigReader, get_class
from negmas.mechanisms import MechanismProxy
from negmas.negotiators import NegotiatorProxy
from negmas.outcomes import Issue, OutcomeType, Outcome
from negmas.sao import AspirationNegotiator
from negmas.situated import Agent, Contract, World, Action, AgentWorldInterface, Breach, BreachProcessing
from negmas.utilities import UtilityFunctionProxy, UtilityValue, LinearUtilityAggregationFunction, normalize, \
    ComplexWeightedUtilityFunction, MappingUtilityFunction

__all__ = [
    'SCMLWorld',
    'SCMLAgent',
    'SCMLAgreement',
    'CFP',
    'FactoryManager',
    'Product',
    'Process',
    'ManufacturingProfile',
    'FactoryState',
    'Factory',
    'LineState',
    'Line',
    'SCMLAction',
    'Consumer',
    'Miner',
    'GreedyFactoryManager',
    'NegotiatorUtility',
    'ScheduleInfo',
    'Scheduler',
    'GreedyScheduler',
]


g_last_product_id = 0
g_last_process_id = 0
g_agents: Dict[str, 'SCMLAgent'] = {}
g_products: Dict[int, 'Product'] = {}
g_processes: Dict[int, 'Process'] = {}


@dataclass
class Product:
    __slots__ = ['id', 'production_level', 'name', 'expires_in', 'catalog_price']
    """A product that can be transacted in."""
    id: int
    """Product index. Must be set during construction and **MUST** be unique for products in the same world"""
    production_level: int
    """The level of this product in the production graph."""
    name: str
    """Object name"""
    expires_in: Optional[int]
    """Number of steps within which the product must be consumed. None means never"""
    catalog_price: Optional[float]
    """Catalog price of the product."""

    def __str__(self):
        """String representation is simply the name"""
        return self.name + (f'(cp:{self.catalog_price})' if self.catalog_price is not None else "")

    def __post_init__(self):
        global g_last_product_id
        if self.id is None:
            self.id = g_last_product_id
            g_last_product_id += 1
        if self.name is None:
            self.name = str(self.id)

    def __hash__(self):
        return self.id.__hash__()


@dataclass(frozen=True)
class InputOutput:
    """An input/output to a production process"""
    __slots__ = ['product', 'quantity', 'step']
    product: int
    """Index of the product used as input or output"""
    quantity: int
    """Quantity needed/produced"""
    step: float
    """Relative time within the production at which the input is needed (output is produced)"""


@dataclass
class Process:
    __slots__ = ['id', 'production_level', 'name', 'inputs', 'outputs', 'historical_cost']
    id: int
    """A manufacturing process."""
    production_level: int
    """The level of this process in the production graph"""
    name: str
    """Object name"""
    inputs: Set[InputOutput]
    """list of input product name + quantity required and time of consumption relative to the time required for 
    production (value from 0 to 1)"""
    outputs: Set[InputOutput]
    """list of output product names, quantity required and when it becomes available relative to the time required for 
    production (value from 0 to 1)"""
    historical_cost: Optional[float]
    """Average cost for running this process in some world. Filled by the world"""

    def __str__(self):
        """String representation is simply the name"""
        return self.name + (f'(cp:{self.historical_cost})' if self.historical_cost is not None else "")

    def __post_init__(self):
        global g_last_process_id
        if self.id is None:
            self.id = g_last_process_id
            g_last_process_id += 1
        if self.name is None:
            self.name = str(self.id)

    def __hash__(self):
        """The hash depends only on the name"""
        return str(self).__hash__()


@dataclass
class ManufacturingProfile:
    __slots__ = ['n_steps', 'cost', 'initial_pause_cost', 'running_pause_cost', 'resumption_cost', 'cancellation_cost']
    n_steps: int
    """Number of steps needed to complete the manufacturing"""
    cost: float
    """Cost of manufacturing"""
    initial_pause_cost: float
    """Cost of pausing incurred only at the step a pause is started"""
    running_pause_cost: float
    """Running cost of pausing"""
    resumption_cost: float
    """Cost of resuming a process"""
    cancellation_cost: float
    """Cost of cancelling the process before the last step"""


# -----------------------------------------------
# Classes needed to support the scheduler (begin)
# -----------------------------------------------

@dataclass
class ProductionNeed:
    __slots__ = ['product', 'needed_for', 'quantity_to_buy', 'quantity_in_storage', 'step']
    product: int
    """The product needed"""
    needed_for: Contract
    """The contract for which the product is needed"""
    quantity_to_buy: int
    """The quantity need to be bought"""
    quantity_in_storage: int
    """The quantity already found in storage"""
    step: int
    """The time step at which the product is needed"""

    def __str__(self):
        """String representation is simply the name"""
        return f'Need {self.quantity_to_buy} ({self.quantity_in_storage} exist) of {self.product} at ' + \
               f' {self.step} for {self.needed_for}'


@dataclass
class RunningCommandInfo:
    __slots__ = ['process', 'beg', 'end', 'command', 'scheduled_at', 'ready', 'cancelled']
    process: int
    """The process for run commands"""
    beg: int
    """The time the command is to be executed"""
    end: int
    """The number of steps starting at `beg` for this command to end (it ends at end - 1)"""
    command: str
    """The command type. For the current implementation it will always be run"""

    @property
    def n_steps(self) -> int:
        return self.end - self.beg


@dataclass
class Job:
    __slots__ = ['process', 'time', 'line_name', 'command', 'contract', 'updates']
    process: int
    """The process for run commands"""
    time: int
    """The time the command is to be executed"""
    line_name: str
    """The line the job is scheduled for."""
    command: str
    """The command type. For the current implementation it will always be run"""
    contract: Contract
    """The sell contract associated with the command"""
    updates: Dict[int, 'FactoryStatusUpdate']
    """The status updates implied by this job"""

    def __str__(self):
        return f'{self.command} {self.process if self.command == "run" else ""} at {self.time} on {self.line_name}'


@dataclass
class FactoryStatusUpdate:
    __slots__ = ['balance', 'storage']
    balance: float
    """The update to the balance"""
    storage: Dict[int, int]
    """The updates to be applied to the storage after this step"""

    def __post_init__(self):
        if not isinstance(self.storage, defaultdict):
            self.storage = defaultdict(int, self.storage)

    def combine(self, other: "FactoryStatusUpdate") -> None:
        """
        Combines this status update with another one in place

        Args:
            other: The other status update

        Returns:
            None
        """
        if other is None:
            return
        self.balance += other.balance
        to_remove = []
        for k in itertools.chain(self.storage.keys(), other.storage.keys()):
            self.storage[k] += other.storage[k]
            if self.storage[k] == 0:
                to_remove.append(k)
        for k in to_remove:
            del self.storage[k]

    @classmethod
    def combine_sets(cls, dst: Dict[int, "FactoryStatusUpdate"], src: Dict[int, "FactoryStatusUpdate"]):
        """
        Combines a set of updates over time with another in place (overriding `first`)
        Args:
            dst: First set of updates to be combined into
            src: second set of updates to be combined from

        Returns:

        """
        to_remove = []
        for i, update in src.items():
            dst[i].combine(update)
            if dst[i].balance == 0 and len(dst[i].storage) == 0:
                to_remove.append(i)
        for i in to_remove:
            del dst[i]
        return None

    def empty(self):
        return self.balance == 0 and len(self.storage) == 0

    def __str__(self):
        return f'balance: {self.balance}, ' + \
               f'{str({k: v for k, v in self.storage.items()}) if self.storage is not None else ""}'


@dataclass
class ManufacturingInfo:
    """Gives full information about a production command"""
    line_name: str
    """The line name specifying a `Line`"""
    profile: ManufacturingProfile
    """The `ManufacturingProfile` """
    process: int
    """The `Process` index"""
    quantity: int
    """The quantity generated/consumed by running this manufacturing info"""
    step: int
    """The step from the beginning at which the outcome is received/consumed"""


# -----------------------------------------------
# Classes needed to support the scheduler (end)
# -----------------------------------------------

INVALID_STEP = -1000
EMPTY_STEP = -1


@dataclass
class LineState:
    __slots__ = ['current_process', 'running_for', 'paused_at', 'commands', 'jobs', 'updates', 'schedule']
    current_process: Optional[int]
    """Current process running on the line"""
    running_for: int
    """For how many steps was this process running"""
    paused_at: Optional[int]
    """If not None, it gives the time-step at which the process was paused"""
    commands: List[Optional[RunningCommandInfo]]
    """The production commands waiting to be run. It is always a run command"""
    jobs: Dict[int, List[Job]]
    """The jobs scheduled at every time-step"""
    updates: Dict[int, FactoryStatusUpdate]
    """The updates scheduled at every time-step"""
    schedule: np.array
    """The schedule of the line as a N*1 array giving the index of the process running at every timestep to the end
    of time or -1 if no process is running. The special value INVALID means that this time step cannot run anything
    """

    def __post_init__(self):
        if not isinstance(self.jobs, defaultdict):
            self.jobs = defaultdict(list, self.jobs)
        if not isinstance(self.updates, defaultdict):
            self.updates = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}), self.updates)


@dataclass
class MissingInput:
    __slots__ = ['product', 'quantity']
    product: int
    quantity: int


@dataclass
class ProductionFailure:
    __slots__ = ['line', 'command', 'missing_inputs', 'missing_money']
    line: str
    """ID of the line that failed"""
    command: RunningCommandInfo
    """Information about the command that failed"""
    missing_inputs: List[MissingInput]
    """The missing inputs if any with their quantities"""
    missing_money: float
    """The amount of money needed for production that is not available"""


@dataclass
class Line:
    processes: List[Process]
    """All processes. They may be runnable and may not be runnable on the line"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""
    profiles: Dict[int, ManufacturingProfile] = field(default_factory=dict)
    """Production profiles mapping a `Process` index to its `ManufacturingProfile`"""
    _state: LineState = field(default_factory=lambda: LineState(current_process=None
                                                                , running_for=0, paused_at=None, jobs={}, updates={}
                                                                , schedule=None, commands=[]))
    """The state of the line"""
    i2p: Dict[int, Process] = field(default_factory=dict)
    """Mapping process indices to processes"""

    @property
    def n_steps(self) -> int:
        """The number of steps for which the line is keeping a schedule"""
        return len(self._state.schedule) if self._state.schedule is not None else 0

    @n_steps.setter
    def n_steps(self, n_steps):
        if self.n_steps == n_steps:
            return
        if n_steps < self.n_steps:
            if self._state.schedule is not None:
                self._state.schedule = self._state.schedule[:n_steps]
            if self._state.commands is not None:
                self._state.commands = [c for c in self._state.commands[:n_steps]]
            self._state.jobs = defaultdict(list, {k: v for k, v in self._state.jobs.items() if k < n_steps})
            self._state.updates = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={})
                                              , {k: v for k, v in self._state.updates.items()
                                                 if k < n_steps})
        else:
            if self._state.schedule is not None:
                s = self._state.schedule
                self._state.schedule = np.zeros(n_steps)
                self._state.schedule[:len(s)] = s
                self._state.schedule[len(s):] = s[-1]
            if self._state.commands is not None:
                self._state.commands = [c for c in self._state.commands]
            self._state.jobs = defaultdict(list, {k: v for k, v in self._state.jobs.items()})
            self._state.updates = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={})
                                              , {k: v for k, v in self._state.updates.items()})

    def copy(self):
        """Copies the line trying to preserve memory."""
        line = Line(id=self.id, profiles=self.profiles, processes=self.processes)
        line._state.current_process = self._state.current_process
        line._state.running_for = self._state.running_for
        line._state.paused_at = self._state.paused_at
        line._state.jobs = self._state.jobs.copy()
        line._state.commands = self._state.commands[:]
        line._state.schedule = np.copy(self._state.schedule)
        return line

    @property
    def schedule(self) -> np.array:
        """Returns a 1d array with the process index running at every timestep from time 0 till end of time"""
        return self._state.schedule

    def __post_init__(self):
        if self._state is None:
            self._state = LineState(current_process=None, running_for=0, paused_at=None, commands=[]
                                    , jobs={}, updates={}, schedule=None)
        if self.profiles is not None:
            processes = self.processes
            if self.i2p is None or len(self.i2p) == 0:
                self.i2p = dict(zip((_.id for _ in processes), processes))

    def _cancel_running_command(self, running_command: RunningCommandInfo) -> Optional[Dict[int, FactoryStatusUpdate]]:
        """
        Cancels a running command as if it did not ever happen

        Args:

            running_command: The running command to cancel

        Returns:

            Dict[int, FactoryStatusUpdate]: The status updated for all times that need to be updated to cancel the
                command

        Remarks:
            - The output of a process that runs from step t to step t + n - 1 will only be in storage at step t + n
            -

        """
        if running_command.command != 'run':
            raise NotImplementedError('We only support run jobs now')
        process_index, beg, end = running_command.process, running_command.beg, running_command.end
        profile = self.profiles[process_index]
        if profile is None:
            return {}
        n, cost = profile.n_steps, profile.cost
        self._state.schedule[beg: end] = EMPTY_STEP
        self._state.commands[beg: end] = [None] * (end - beg)
        results: Dict[int, FactoryStatusUpdate] = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}))
        process = self.i2p[process_index]
        for need in process.inputs:
            results[beg + int(math.floor(need.step * n))].storage[need.product] += need.quantity
        for output in process.outputs:
            results[beg + int(math.ceil(output.step * n))].storage[output.product] -= output.quantity
        results[beg].balance += cost  # notice that we do not need to pay cancellation cost here
        FactoryStatusUpdate.combine_sets(self._state.updates, results)
        return results

    def _simulate_run(self, t: int, process: Process, override=True) -> Optional[Dict[int, FactoryStatusUpdate]]:
        """running is executed at the beginning of the step t

        Args:

            t: time-step to start the process
            process: the process to start
            override: If true, override any running processes paying cancellation cost for these processes

        Returns:

            Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
            the command if it is not None. If None is returned then scheduling failed.

        Remarks:

            - The output of a process that runs from step t to step t + n - 1 will only be in storage at step t + n



        """
        pid = process.id
        profile = self.profiles[pid]
        n, cost = profile.n_steps, profile.cost

        def do_run() -> Dict[int, FactoryStatusUpdate]:
            self._state.schedule[t: t + n] = pid
            c = RunningCommandInfo(command='run', process=pid, beg=t, end=t + n)
            self._state.commands[t: t + n] = [c] * n
            results: Dict[int, FactoryStatusUpdate] = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}))
            for need in process.inputs:
                results[t + int(math.floor(need.step * n))].storage[need.product] -= need.quantity
            for output in process.outputs:
                results[t + int(math.ceil(output.step * n))].storage[output.product] += output.quantity
            results[t].balance -= cost
            return results

        # run immediately if possible
        if np.all(self._state.schedule[t: t + n] == EMPTY_STEP):
            updates = do_run()
            FactoryStatusUpdate.combine_sets(self._state.updates, updates)
            return updates

        # if I am not allowed to override, then this command has no effect and I return an empty status update
        if not override:
            return FactoryStatusUpdate(balance=0, storage={})

        # requires some stopping and cancellation
        updates = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}))
        for current in range(t, t + n):
            current_command = self._state.commands[current]
            if current_command is None:
                continue
            if current_command == self._state.commands[current - 1]:
                # that is a running process, stop it
                # @todo if the process has not produced any outcomes, then cancel it
                update_set = self._simulate_stop(current)
            else:
                # that is a new process that is to be started. Do not start it
                update_set = self._cancel_running_command(current_command)
            # if I cannot cancel or stop the running command, then fail
            if update_set is None:
                return None
            for i, change in update_set.items():
                updates[i].combine(change)
        new_updates = do_run()
        for i, change in new_updates.items():
            updates[i].combine(change)
        FactoryStatusUpdate.combine_sets(self._state.updates, updates)
        return updates

    def _simulate_pause(self, t: int) -> Optional[Dict[int, FactoryStatusUpdate]]:
        """pausing is executed at the end of the step

        Args:

            t: time-step to start the process

        Returns:

            Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
            the command if it is not None. If None is returned then scheduling failed.

        Remarks:

            - Not implemented yet
            - pausing when nothing is running is not an error and will return an empty status update

        """
        raise NotImplementedError('Pause is not implemented')

    def _simulate_resume(self, t: int) -> Optional[Dict[int, FactoryStatusUpdate]]:
        """resumption is executed at the end of the step (starting next step count down)


        Args:

            t: time-step to start the process

        Returns:

            Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
            the command if it is not None. If None is returned then scheduling failed.

        Remarks:

            - Not implemented yet
            - resuming when nothing is paused is not an error and will return an empty status update

        """
        raise NotImplementedError('Resume is not implemented')

    def _simulate_stop(self, t: int) -> Optional[Dict[int, FactoryStatusUpdate]]:
        """stopping is executed at the beginning of the current step

        Args:

            t: time-step to start the process

        Returns:

            Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
            the command if it is not None. If None is returned then scheduling failed.

        Remarks:

            - stopping when nothing is running is not an error and will just return an empty schedule
        """
        current_command = self._state.commands[t]
        if current_command is None:
            return {}
        running_process_index = self._state.schedule[t]
        if current_command.beg >= t:
            return self._cancel_running_command(current_command)
        beg, end = current_command.beg, current_command.end
        current_command.end = t
        self._state.schedule[t: end] = running_process_index
        process_index = current_command.process
        profile = self.profiles[process_index]
        # current_command.costs[t] = profile.cancellation_cost
        n = profile.n_steps
        updates: Dict[int, FactoryStatusUpdate] = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}))
        process = self.i2p[process_index]
        for need in process.inputs:
            need_time = beg + int(math.floor(need.step * n))
            if need_time > t:
                updates[need_time].storage[need.product] += need.quantity
        for output in process.outputs:
            output_time = beg + int(math.floor(output.step * n))
            if output_time >= t:
                updates[output_time].storage[output.product] -= output.quantity
        updates[t].balance -= profile.cancellation_cost
        FactoryStatusUpdate.combine_sets(self._state.updates, updates)
        return updates

    def _schedule(self, command: str, t: int, process: Process = None
                  , override=True) -> Optional[Dict[int, FactoryStatusUpdate]]:
        """
        Schedules the given command at the given time for the given process.

        Args:

            command: Can be run, stop, pause, resume
            t:
            process:
            override: IF true running commands will be overridden to make this command possible

        Returns:

            Dict[int, FactoryStatusUpdate]: The status updated for all times that need to be updated to cancel the
                command
        """
        result = None
        if command == 'run':
            if process is None:
                raise ValueError('Cannot run an unspecified process')
            result = self._simulate_run(t=t, process=process, override=True)
        elif command == 'pause':
            result = self._simulate_pause(t=t)
        elif command == 'resume':
            result = self._simulate_resume(t=t)
        elif command == 'stop':
            result = self._simulate_stop(t=t)
        return result

    def init_schedule(self, n_steps: int, jobs: Dict[int, List[Job]] = None, override=True) -> bool:
        """
        Initializes the internal schedule of the line

        Args:
            n_steps: The number of steps to keep in the schedule
            jobs: Initial set of jobs to run after the initialization
            override: If true then later jobs override earlier ones if they are conflicting

        Returns:
            bool: Success indicator
        """
        self._state.schedule = np.ones(shape=n_steps, dtype=np.int) * EMPTY_STEP
        self._state.commands = [None] * n_steps
        if jobs is None:
            return True
        failed = True
        for t, js in jobs.items():
            for job in js:
                command, process_index = job.command, job.process
                result = self._schedule(command=command, t=t, process=self.processes[process_index], override=override)
                if result is None:
                    failed = True
                    continue
                job.time = t
                job.updates = result
                self._state.jobs[t].append(job)
        return not failed

    def schedule_job(self, job: Job, override=True) -> Optional[Dict[int, FactoryStatusUpdate]]:
        """
        Schedules the given job

        Args:
            job: A job to schedule
            override: If true, override any preexisting jobs to make this one run

        Returns:
            Dict[int, FactoryStatusUpdate]: The status updated for all times that need to be updated to cancel the
                command

        Remarks:
            The job is updated as follows:

            - This line is set as the line member in job
            - The updates that result from this schedule to balance and storage are added to the updates in the job
        """
        time, command, process_index = job.time, job.command, job.process
        self._state.jobs[time].append(job)
        job.line_name = self.id
        result = self._schedule(command=command, t=time, process=self.processes[process_index], override=override)
        if result is None:
            return result
        if job.updates:
            FactoryStatusUpdate.combine_sets(job.updates, result)
        else:
            job.updates = result
        return result

    def step(self, t: int, storage: Dict[int, int], wallet: float) -> Union[FactoryStatusUpdate, ProductionFailure]:
        """
        Steps the line to the time-step `t` assuming that it is already stepped to time-step t-1 given the storage

        Args:
            t:
            storage:

        Returns:
            None: Failed to step because the storage is not enough
            FactoryStatueUpdate(storage=None): Nothing is running at this step
            FactoryStatueUpdate(storage!=None): can step and returns the updates to the storage and balance
        """
        state = self._state
        updates = state.updates.get(t, None)
        if updates is None:
            updates = FactoryStatusUpdate(balance=0, storage={})
        self._state.paused_at, self._state.running_for, self._state.current_process = None, 0, None

        command = state.commands[t]
        if command is None:
            state.schedule[:t + 1] = INVALID_STEP
            return updates
        process_index = command.process
        missing_inputs = []
        missing_money = 0
        failed = False
        if updates.balance < 0 and wallet < -updates.balance:
            failed = True
            missing_money = -updates.balance - wallet
        for product_id, quantity in updates.storage.items():
            if storage.get(product_id, 0) < -quantity:
                failed = True
                missing_inputs.append(MissingInput(product=product_id, quantity=quantity))
        if failed:
            failure = ProductionFailure(line=self.id, command=command, missing_money=missing_money
                                        , missing_inputs=missing_inputs)
            self._cancel_running_command(command)
            state.schedule[:t + 1] = INVALID_STEP
            return failure
        state.current_process = process_index
        state.paused_at = 0  # @todo pause is disabled
        state.running_for = t - command.beg  # @todo pause is disabled
        state.schedule[:t + 1] = INVALID_STEP
        return updates

    @property
    def state(self):
        """Read only access to the state"""
        return self._state

    def __str__(self):
        """String representation is simply the name"""
        return self.id

    def __hash__(self):
        """The hash depends only on the name"""
        return str(self).__hash__()


# @dataclass
# class ManufacturingInfo:
#     """Gives full information about a manufacturing process on a line"""
#     line: Line
#     """The line name specifying a `Line`"""
#     profile: ManufacturingProfile
#     """The `ManufacturingProfile` """
#     process: Process
#     """The `Process`"""


@dataclass
class Loan:
    amount: float = 0.0
    """Loan amount"""
    total: float = 0.0
    """The total to be paid including the amount + interests"""
    interest: float = 0.0
    """The interest rate per step"""
    installment: float = 0.0
    """The amount to be paid in one installment"""
    n_installments: int = 1
    """The number of installments"""


@dataclass
class FactoryState:
    storage: Dict[int, int] = field(default_factory=lambda: defaultdict(int))  # mapping from product to quantity
    wallet: float = 0.0
    """Money available for purchases"""
    loans: float = 0.0
    """The total money owned as loans"""

    _balance_prediction: np.array = None
    """Balances over simulation time as 1D array N. steps"""
    _storage_prediction: np.array = None
    """Storage state over simulation time as 2D array N. Products * N. steps"""
    _reserved_storage: np.array = None
    """Amounts of different products that need to exist at the storage over time"""
    _total_storage: np.array = None
    """Total storage over time as 1D array N. steps"""

    def __post_init__(self):
        if self._storage_prediction is not None:
            self._total_storage = self._storage_prediction.sum(axis=0)

    @property
    def balance(self):
        """The total balance of the factory"""
        return self.wallet - self.loans

    def copy(self):
        return FactoryState(storage={k: v for k, v in self.storage.items()}
                            , wallet=self.wallet, loans=self.loans
                            , _balance_prediction=self._balance_prediction.copy()
                            , _storage_prediction=self._storage_prediction.copy()
                            , _reserved_storage=self._reserved_storage.copy())


@dataclass
class RenegotiationRequest:
    publisher: 'SCMLAgent'
    partner: 'SCMLAgent'
    issues: List[Issue]
    annotation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SCMLAgreement(OutcomeType):
    time: int
    """delivery time"""
    unit_price: float
    """unit price"""
    quantity: int
    """quantity"""
    penalty: Optional[float] = None
    """penalty"""
    signing_delay: int = 0
    """Delay between agreement conclusion and signing it to be binding"""


class SCMLAgent(Agent, ABC):
    """The base for all SCM Agents"""

    def __init__(self, factory: Optional['Factory'] = None, negotiator_type='negmas.sao.AspirationNegotiator'
                 , **kwargs):
        super().__init__(**kwargs)
        self.factory = factory
        if self.factory is not None:
            self.factory.manager = self
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.immediate_negotiations = True
        self.negotiation_speed_multiple: int = 1
        self.transportation_delay: int = 0
        self.products: List[Product] = []
        self.processes: List[Process] = []
        self.interesting_products: List[int] = []

    def attach(self, factory: 'Factory'):
        self.factory = factory
        self.factory.manager = self

    def step(self):
        pass

    def init(self):
        super().init()
        # noinspection PyUnresolvedReferences
        self.products = self.awi.products  # type: ignore
        self.interesting_products = [_.id for _ in self.products]
        # noinspection PyUnresolvedReferences
        self.processes = self.awi.processes  # type: ignore
        self.negotiation_speed_multiple = self.awi.bulletin_board.read('settings', 'negotiation_speed_multiple')
        self.immediate_negotiations = self.awi.bulletin_board.read('settings', 'immediate_negotiations')
        self.transportation_delay = self.awi.bulletin_board.read(section='settings', key='transportation_delay')

    def can_expect_agreement(self, cfp: 'CFP'):
        """
        Checks if it is possible in principle to get an agreement on this CFP by the time it becomes executable
        Args:
            cfp:

        Returns:

        """
        return cfp.max_time >= self.awi.current_step + 1 - int(self.immediate_negotiations) #@todo check that this is correct now

    def before_joining_negotiation(self, initiator: str, partners: List[str], issues: List[Issue]
                                   , annotation: Dict[str, Any], mechanism: MechanismProxy, role: Optional[str]
                                   , req_id: str):
        """When a negotiation request is received"""
        if req_id is not None:
            info = self._neg_requests.get(req_id, None)
            if info and info.negotiator is not None:
                return info.negotiator
        return self.on_negotiation_request(partner=initiator, cfp=annotation['cfp'])

    def _create_annotation(self, cfp: 'CFP'):
        """Creates full annotation based on a cfp that the agent is receiving"""
        partners = [self.id, cfp.publisher]
        annotation = {'cfp': cfp, 'partners': partners}
        if cfp.is_buy:
            annotation['seller'] = self.id
            annotation['buyer'] = cfp.publisher
        else:
            annotation['buyer'] = self.id
            annotation['seller'] = cfp.publisher
        return annotation

    def on_new_cfp(self, cfp: 'CFP') -> None:
        """Called whenever a new CFP is published"""

    def on_remove_cfp(self, cfp: 'CFP') -> None:
        """Called whenever an existing CFP is about to be removed"""

    @abstractmethod
    def confirm_loan(self, loan: Loan) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""

    @abstractmethod
    def confirm_contract_execution(self, contract: Contract) -> bool:
        """Called before executing any agreement"""
        return True

    @abstractmethod
    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[NegotiatorProxy]:
        """Called when a prospective partner requests a negotiation to start"""

    @abstractmethod
    def on_breach_by_self(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        """Called when the agent is about to commit a breach"""

    @abstractmethod
    def on_breach_by_another(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        """Called when a partner is about to cause a breach"""

    @abstractmethod
    def on_breach_meta_negotiation(self, contract: Contract, partner: str, issues: List[Issue]) \
        -> Optional[NegotiatorProxy]:
        """Called when a partner or self is about to cause a breach if the breach_processing setting is set to
        start a meta negotiation. The agent should either return None or a negotiator"""

    @abstractmethod
    def on_renegotiation_request(self, contract: Contract, cfp: "CFP", partner: str) -> bool:
        """Called to respond to a re-negotiation request"""

    @property
    def factory_state(self):
        """Read only access to factory state"""
        return self.factory.state

    def on_production_failure(self, failures: List[ProductionFailure]) -> None:
        """Will be called when production of some command fails"""


def zero_runs(a: np.array) -> np.array:
    """
    Finds all runs of zero in an array

    Args:
        a: Input array (assumed to be 1D)

    Returns:
        np.array: A 2D array giving beginning and end (exclusive) of zero stretches in the input array.
    """
    if len(a) == 0:
        return []
    if np.all(np.equal(a, 0).view(np.int8)):
        return np.array([[0, len(a)]])
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    if len(ranges) == 0 and a[0] == 0:
        return np.array([[0, len(a)]])
    return ranges


@dataclass
class ScheduleInfo:
    factory: 'Factory'
    """The factory on which the schedule is executed"""
    valid: bool = True
    """Is this a valid schedule?"""
    end: Optional[int] = None
    """The step after the last step in this simulation"""
    # these two attributes are not directly related to the schedule but with the scheduling operation that generated it
    needs: List[ProductionNeed] = field(default_factory=list)
    """The products needed but not still in storage needed to complete this schedule."""
    jobs: List[Job] = field(default_factory=list)
    """The jobs that need to be scheduled"""
    updates: Dict[int, FactoryStatusUpdate] = field(default_factory=
                                                    lambda: defaultdict(lambda: FactoryStatusUpdate(
                                                        balance=0, storage={})))
    """A list of updates to storage and balance for successful buy contracts"""
    failed_contracts: List[Contract] = field(default_factory=list)
    """A list of contracts that failed to be scheduled."""

    def __str__(self):
        fail = str("fail " + "|".join(str(_) for _ in self.failed_contracts)) if len(self.failed_contracts) > 0 else ""
        jobs = str("jobs " + "|".join(str(_) for _ in self.jobs)) if len(self.jobs) > 0 else ""
        updates, needs = '', ''
        if len(self.updates) > 0:
            updates = "updates "
            for t, update in self.updates.items():
                updates += f'\n\t @{t}: {str(update)}'
        if len(self.needs) > 0:
            needs = "needs :" + "\n\t".join(str(_) for _ in self.needs)

        result = f'{"valid" if self.valid else "invalid"} (ends before {self.end}):'
        for x in (fail, jobs, updates, needs):
            if len(x) > 0:
                result += '\n' + x
        return result

    def combine(self, other: 'ScheduleInfo') -> None:
        self.valid = self.valid and other.valid
        if other.end is not None and self.end is not None:
            self.end = max(self.end, other.end)
        if other.needs is not None:
            self.needs.extend(other.needs)
        FactoryStatusUpdate.combine_sets(self.updates, other.updates)
        if other.jobs is not None:
            self.jobs.extend(other.jobs)
        if other.failed_contracts is not None:
            self.failed_contracts.extend(other.failed_contracts)



@dataclass
class Factory:
    """The factory object representing a single factory"""
    products: List[Product]
    """All products"""

    processes: List[Process]
    """All processes"""

    name: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""

    _state: FactoryState = field(default_factory=FactoryState)
    """The dyanmic state of the factory under control of the simulator"""

    _jobs: List[Job] = field(default_factory=list)
    """The jobs to be executed for this factory"""

    lines: Dict[str, Line] = field(default_factory=dict)
    """Lines with their config if any"""

    max_storage: Optional[int] = None
    """Maximum storage for this factory. None means infinity"""

    n_scheduling_steps: Optional[int] = None
    """The number of steps I am expected to run"""

    manager: Optional['SCMLAgent'] = None
    """The factory manager which is an object of the `FactoryManager` class that controls production."""

    # manufacturing_profiles: Dict[Process, List[ManufacturingInfo]] = field(default_factory=lambda: defaultdict(list))
    # """Mapping every process to all lines that can produce it with full information"""

    producing: Dict[int, List[ManufacturingInfo]] = field(default_factory=lambda: defaultdict(list))
    """Mapping from a product to all manufacturing processes that can generate it"""

    consuming: Dict[int, List[ManufacturingInfo]] = field(default_factory=lambda: defaultdict(list))
    """Mapping from a product to all manufacturing processes that can consume it"""

    process2ind: Dict[Process, int] = field(default_factory=dict)
    """Maps processes to indices from 0"""

    product2ind: Dict[Product, int] = field(default_factory=dict)
    """Maps Products to indices from 0"""

    def __post_init__(self):
        self.update_fields()

    def __str__(self):
        """String representation is simply the name"""
        return self.name

    def __hash__(self):
        """The hash depends only on the name"""
        return str(self).__hash__()

    @property
    def n_steps(self):
        return self.n_scheduling_steps

    @n_steps.setter
    def n_steps(self, n_steps):
        if self.n_scheduling_steps == n_steps:
            return
        if n_steps < self.n_scheduling_steps:
            self._state._balance_prediction = self.predicted_balance[:n_steps]
            self._state._total_storage = self.predicted_total_storage[:n_steps]
            self._state._storage_prediction = self.predicted_storage[:, :n_steps]
            self._state._reserved_storage = self.predicted_reserved_storage[:, :n_steps]
        else:
            storage, balance = self.predicted_storage, self.predicted_balance
            total, reserved = self.predicted_total_storage, self.predicted_reserved_storage
            self._state._balance_prediction = np.ones(n_steps) * balance[-1]
            self._state._total_storage = np.ones(n_steps) * total[-1]
            self._state._storage_prediction = np.ones(shape=(storage.shape[0], n_steps)
                                                      , dtype=np.int) * storage[:, -1].reshape(storage.shape[0], 1)
            self._state._reserved_storage = np.zeros(shape=(storage.shape[0], n_steps), dtype=np.int)
            self._state._balance_prediction[:self.n_scheduling_steps] = balance
            self._state._total_storage[:self.n_scheduling_steps] = total
            self._state._storage_prediction[:, :self.n_scheduling_steps] = storage
            self._state._reserved_storage[:, :self.n_scheduling_steps] = reserved
        self.n_scheduling_steps = n_steps
        for line in self.lines.values():
            line.n_steps = n_steps

    @property
    def state(self):
        """Read only access to the state"""
        return self._state

    @property
    def n_products(self):
        return len(self.products)

    @property
    def n_processes(self):
        return len(self.processes)

    @property
    def schedule(self) -> Dict[Line, np.array]:
        return {l: l.schedule for l in self.lines.values()}

    @property
    def predicted_storage(self) -> np.array:
        return self._state._storage_prediction

    @property
    def predicted_balance(self) -> np.array:
        return self._state._balance_prediction

    @property
    def predicted_total_storage(self) -> np.array:
        return self._state._total_storage

    @property
    def predicted_reserved_storage(self) -> np.array:
        return self._state._reserved_storage

    def get_line(self, name: str):
        """Finds a line by its name"""
        return self.lines.get(name, None)

    def step(self, t: int) -> List[ProductionFailure]:
        failures = []
        for line in self.lines.values():
            # step the current production process
            results = line.step(t=t, storage=self._state.storage, wallet=self._state.wallet)
            if isinstance(results, ProductionFailure):
                failures.append(results)
            else:
                if results.balance != 0:
                    self._state.wallet -= results.balance
                if results.storage is not None:
                    for k, v in results.storage.items():
                        self.state.storage[k] += v
        return failures

    def update_fields(self):
        """Updates all dynamic fields (that are not initialized in __init__)."""
        if not isinstance(self.lines, dict) and isinstance(self.lines, Iterable):
            # noinspection PyUnresolvedReferences
            self.lines = {_.id: _ for _ in self.lines}
        if self._state is None:
            self._state = FactoryState()
        for line_name, config in self.lines.items():
            for process_index, profile in config.profiles.items():
                process = self.processes[process_index]
                for outpt in process.outputs:
                    step = int(math.ceil(outpt.step * profile.n_steps))
                    self.producing[outpt.product].append(ManufacturingInfo(line_name=line_name, profile=profile
                                                                           , process=process_index
                                                                           , quantity=outpt.quantity
                                                                           , step=step))

                for inpt in process.inputs:
                    self.consuming[inpt.product].append(ManufacturingInfo(line_name=line_name, profile=profile
                                                                          , process=process_index
                                                                          , quantity=inpt.quantity
                                                                          ,
                                                                          step=int(
                                                                              math.floor(inpt.step * profile.n_steps))))

        # sort production and consumption mappings so that the cheapest methods come first
        # for k, v in self.producing.items():
        #     self.producing[k] = sorted(v)
        # for k, v in self.consuming.items():
        #     self.consuming[k] = sorted(v)

        self.product2ind = dict(zip(self.products, (_.id for _ in self.products)))
        self.process2ind = dict(zip(self.processes, (_.id for _ in self.processes)))
        for line in self.lines.values():
            line.p2i = self.process2ind

    def copy(self):
        f = Factory(_state=self._state.copy(), lines={k: v.copy() for k, v in self.lines.items()}
                    , max_storage=self.max_storage
                    , n_scheduling_steps=self.n_scheduling_steps, manager=self.manager
                    , process2ind=self.process2ind, product2ind=self.product2ind
                    , products=self.products, processes=self.processes)
        return f

    def _apply_updates(self, updates: FactoryStatusUpdate, t: int, end: int, set_reserved: bool) -> bool:
        """
        Applies the given set of updates to predicted storage/balance

        Args:
            updates:
            t:
            end:

        Remarks:

            If end == None, only time t is updated

        """
        if updates is None:
            return False
        if updates.empty():
            return True
        for k, v in updates.storage.items():
            if set_reserved and v < 0:
                self.predicted_reserved_storage[k, t] -= v
            self.predicted_storage[k, t: end] += v
            self.predicted_total_storage[t:end] += v
        self.predicted_balance[t:end] += updates.balance
        return True

    def init_schedule(self, n_steps: int, initial_storage: Dict[int, int] = None, override: bool = True
                      , initial_balance: float = None, initial_jobs: List[Job] = None) -> bool:
        if initial_balance is None:
            initial_balance = self._state.wallet
        if initial_storage is None:
            initial_storage = self._state.storage
        if initial_jobs is None:
            initial_jobs = self._jobs
        self.n_scheduling_steps = n_steps
        n_products = self.n_products
        self._state._balance_prediction = np.ones(n_steps) * initial_balance
        self._state._storage_prediction = np.zeros(shape=(n_products, n_steps), dtype=np.int)
        self._state._reserved_storage = np.zeros(shape=(n_products, n_steps), dtype=np.int)
        self._state._total_storage = np.zeros(n_steps)
        if len(initial_storage) > 0:
            for i in range(n_products):
                self.predicted_storage[i, :] = initial_storage.get(i, 0)
            self.predicted_total_storage[:] = self.predicted_storage[:, 0].sum()
        for line in self.lines.values():
            line.init_schedule(n_steps=n_steps, jobs={})
        if len(initial_jobs) < 1:
            return True
        initial_jobs = sorted(initial_jobs, key=lambda x: x.time)
        return all(self.schedule_job(job=job, end=n_steps, override=override) for job in initial_jobs)

    def schedule_job(self, job: Job, end=None, override=True) -> bool:
        """
        Schedules a job returning whether it is possible to apply all required updates or not

        Args:
            job:
            end:
            overrride: if true, override any existing jobs

        Returns:

        """
        line = self.lines[job.line_name]
        if line is None:
            return False
        updates = line.schedule_job(job=job, override=override)
        if updates is None:
            return False
        if end is None:
            end = self.n_scheduling_steps
        set_reserved = job.command == 'run'
        return all(self._apply_updates(updates=updates[t], t=t, end=end, set_reserved=set_reserved)
                   for t in sorted(updates.keys()))

    def find_needs(self, job: Job) -> List[ProductionNeed]:
        """
        Finds the production needs for job

        Args:
            job:

        Returns:

        """
        needs = []
        if job is None:
            return needs
        for t, update in job.updates.items():
            # ignore updates with no change in storage
            if update.storage is None or len(update.storage) < 1 or max(abs(_) for _ in update.storage.values()) == 0:
                continue

            for pind, quantity in update.storage.items():
                # ignore updates that add to storage.
                if quantity > 0:
                    continue
                quantity = -quantity

                # do not count reserved storage (for other jobs) into available quantity
                available = max(0,
                                self.predicted_storage[pind, t] - self.predicted_reserved_storage[pind, t] + quantity)
                if available >= quantity:
                    instore, tobuy = quantity, 0
                else:
                    instore, tobuy = available, quantity - available
                if tobuy > 0 or instore > 0:
                    needs.append(ProductionNeed(product=pind, needed_for=job.contract
                                                , quantity_in_storage=instore, quantity_to_buy=tobuy, step=job.time))
        return needs


@dataclass
class CFP(OutcomeType):
    """A Call for proposal upon which a negotiation can start"""
    is_buy: bool
    """If true, the author wants to buy otherwise to sell. Non-negotiable."""
    publisher: str
    """the publisher name. Non-negotiable."""
    product: int
    """product ID. Non-negotiable."""
    time: Union[int, Tuple[int, int], List[int]]
    """delivery time. May be negotiable."""
    unit_price: Union[float, Tuple[float, float], List[float]]
    """unit price. May be negotiable."""
    quantity: Union[int, Tuple[int, int], List[int]]
    """quantity. May be negotiable."""
    penalty: Optional[Union[float, Tuple[float, float], List[float]]] = None
    """penalty per missing item in case the seller cannot provide the required quantity. May be negotiable."""
    signing_delay: Optional[Union[int, Tuple[int, int], List[int]]] = None
    """The grace period after which the agents are asked to confirm signing the contract"""
    money_resolution: Optional[float] = None
    """If not None then it is the minimum unit of money (e.g. 1 for dollar, 0.01 for cent, etc)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Unique CFP ID"""

    def __str__(self):
        s = f'{g_agents[self.publisher].name}: {"buy" if self.is_buy else "sell"} '
        s += f'{g_products[self.product]} '
        s += f'(t: {self.time}, u: {self.unit_price}, q: {self.quantity}'
        if self.penalty is not None:
            s += f', penalty: {self.penalty}'
        if self.signing_delay is not None:
            s += f', sign after: {self.signing_delay}'
        s += ')'
        return s

    def satisfies(self, query: Dict[str, Any]) -> bool:
        """
        Tests whether the CFP satisfies the conditions set by the query

        Args:

            query: A dictionary given the conditions. See `Remarks` for details

        Remarks:

           - The query dictionary can be used to specify any conditions that are required in the CFP. Only CFPs that
             satisfy ALL the conditions specified in the query are considered satisfying the query. The following keys
             can be set with corresponding meanings:

             is_buy
               True or False. If both are OK, just do not add this key

             publisher
               A string or `SCMLAgent` specifying a specific publisher

             publishers
               A list of publishers (see publisher key)

             product
              A string specifying a product name

             products
               A list of products (see product key)

             time
               A number, list or 2-items-tuple (range) specifying possible times to consider satisfactory

             unit_price
               A number, list or 2-items-tuple (range) specifying possible prices to consider satisfactory

             quantity
                A number, list or 2-items-tuple (range) specifying possible quantities to consider OK

             penalty
               A number, list or 2-items-tuple (range) specifying possible penalties to consider satisfactory

        """

        def _overlap(a: Union[int, float, Tuple[float, float], List[float], Tuple[int, int], List[int]]
                     , b: Union[float, Tuple[float, float], List[float], int, Tuple[int, int], List[int]]):

            def _test_single(a, b):
                if not isinstance(b, Iterable):
                    return a == b
                if isinstance(b, tuple):
                    return b[0] <= a <= b[1]
                return a in b

            if not isinstance(b, Iterable):
                a, b = b, a

            if not isinstance(a, Iterable):
                return _test_single(a, b)

            if isinstance(a, tuple):
                if isinstance(b, tuple):
                    return b[0] <= a[0] <= b[1] or b[0] <= a[1] <= b[1]
                return any(_test_single(_, a) for _ in b)  # type: ignore

            return any(_test_single(_, b) for _ in a)

        for k, v in query.items():
            if k == 'is_buy' and self.is_buy != v:
                return False
            if k == 'publisher' and self.publisher != v:
                return False
            if k == 'publishers' and self.publisher not in v:
                return False
            if k == 'products' and self.product not in v:
                return False
            if k == 'product_ids' and self.product not in v:
                return False
            if k == 'product_indices' and self.product not in v:
                return False
            if k == 'product' and self.product != v:
                return False
            if k == 'product_id' and self.product != v:
                return False
            if k == 'product_index' and self.product != v:
                return False
            if k == 'time' and not _overlap(v, self.time):
                return False
            if k == 'unit_price' and not _overlap(v, self.unit_price):
                return False
            if k == 'penalty':
                if self.penalty is None and v is None:
                    return True
                if self.penalty is None or v is None:
                    return False
                if not _overlap(v, self.penalty):
                    return False
            if k == 'quantity' and not _overlap(v, self.quantity):
                return False
        return True

    @property
    def issues(self):
        """Returns the set of issues associated with this CFP. Notice that some of the issues may have a single value"""

        def _values(x, ensure_list=False, ensure_int=False):
            if isinstance(x, tuple) and ensure_list:
                if x[0] == x[1]:
                    if ensure_list and self.money_resolution is not None:
                        if ensure_int:
                            return [int(math.floor(x[0] / self.money_resolution) * self.money_resolution)]
                        return [math.floor(x[0] / self.money_resolution) * self.money_resolution]
                    else:
                        if ensure_int:
                            return [int(x[0])]
                        return [x[0]]
                if isinstance(x[0], float) or isinstance(x[1], float):
                    xs = (int(math.floor(x[0] / self.money_resolution))
                          , int(math.floor(x[1] / self.money_resolution)))
                    xs = list(_ * self.money_resolution for _ in range(xs[0], xs[1] + 1))
                elif isinstance(x[0], int):
                    xs = list(range(x[0], x[1] + 1))
                else:
                    xs = list(range(int(x[0]), int(x[1]) + 1))
                if len(xs) == 0:
                    if ensure_list and self.money_resolution is not None:
                        if ensure_int:
                            return [int(math.floor(x[0] / self.money_resolution) * self.money_resolution)]
                        return [math.floor(x[0] / self.money_resolution) * self.money_resolution]
                    if ensure_int:
                        return [int(x[0])]
                    return [x[0]]
                if ensure_int:
                    return list(set(int(_) for _ in xs))
                return list(set(xs))
            if isinstance(x, Iterable):
                if ensure_int:
                    return list(set(int(_) for _ in x))
                return list(set(x))
            if ensure_int:
                return [int(x)]
            return [x]

        issues = [Issue(name='time', values=_values(self.time, ensure_list=True, ensure_int=True))
            , Issue(name='quantity', values=_values(self.quantity, ensure_list=True, ensure_int=True))
            , Issue(name='unit_price', values=_values(self.unit_price, ensure_list=self.money_resolution is not None))]
        if self.penalty is not None:
            issues.append(Issue(name='penalty'
                                , values=_values(self.penalty, ensure_list=self.money_resolution is not None)))
        if self.signing_delay is not None:
            issues.append(Issue(name='signing_delay', values=_values(self.quantity, ensure_list=True, ensure_int=True)))
        return issues

    @property
    def outcomes(self):
        return Issue.enumerate(issues=self.issues, max_n_outcomes=1000)

    @property
    def min_time(self):
        if isinstance(self.time, tuple):
            return self.time[0]
        elif isinstance(self.time, list):
            return min(self.time)
        return self.time

    @property
    def max_time(self):
        if isinstance(self.time, tuple):
            return self.time[1]
        elif isinstance(self.time, list):
            return max(self.time)
        return self.time

    @property
    def min_quantity(self):
        if isinstance(self.quantity, tuple):
            return self.quantity[0]
        elif isinstance(self.quantity, list):
            return min(self.quantity)
        return self.quantity

    @property
    def max_quantity(self):
        if isinstance(self.quantity, tuple):
            return self.quantity[1]
        elif isinstance(self.quantity, list):
            return max(self.quantity)
        return self.quantity

    @property
    def min_unit_price(self):
        if isinstance(self.unit_price, tuple):
            return self.unit_price[0]
        elif isinstance(self.unit_price, list):
            return min(self.unit_price)
        return self.unit_price

    @property
    def max_unit_price(self):
        if isinstance(self.unit_price, tuple):
            return self.unit_price[1]
        elif isinstance(self.unit_price, list):
            return max(self.unit_price)
        return self.unit_price


@dataclass
class SCMLAction:
    line: str
    process: Process
    action: str
    """The action which may be start, stop, pause, resume"""
    time: int = 0


def pos_gauss(mu, sigma):
    """Returns a sample from a rectified gaussian"""
    x = gauss(mu, sigma)
    return abs(x)


@dataclass
class ConsumptionProfile:
    schedule: Union[int, List[int]] = 1
    underconsumption: float = 0.1
    overconsumption: float = 0.0
    dynamicity: float = 0.0
    cv: float = 0.1

    alpha_q: float = 0.5
    alpha_u: float = 1.0

    beta_q: float = 10.0
    beta_u: float = 10.0

    tau_q: float = 2
    tau_u: float = 0.25

    @classmethod
    def random(cls):
        return ConsumptionProfile(schedule=randint(1, 20), overconsumption=2 * random(), underconsumption=2 * random()
                                  , cv=random(), beta_q=99 * random() + 1, beta_u=99 * random() + 1, tau_q=3 * random()
                                  , tau_u=random())

    def schedule_at(self, time: int) -> int:
        if isinstance(self.schedule, int):
            return self.schedule
        else:
            return self.schedule[time % len(self.schedule)]

    def schedule_within(self, time: Union[int, List[int], Tuple[int, int]]) -> int:
        if isinstance(time, int):
            return self.schedule_at(time)
        if isinstance(time, tuple):
            times = list(range(time[0], time[1] + 1))
        else:
            times = time
        if isinstance(self.schedule, int):
            return self.schedule * len(times)
        return sum(self.schedule_at(t) for t in times)

    def set_schedule_at(self, time: int, value: int, n_steps: int) -> None:
        if isinstance(self.schedule, int):
            self.schedule = [self.schedule] * n_steps
        elif len(self.schedule) < n_steps:
            self.schedule = list(itertools.chain(*([self.schedule] * int(math.ceil(n_steps / len(self.schedule))))))
        self.schedule[time % len(self.schedule)] = value


class Consumer(SCMLAgent, ConfigReader):
    """Consumer class"""

    MAX_UNIT_PRICE = 1e5
    RELATIVE_MAX_PRICE = 2

    def __init__(self, profiles: Dict[int, ConsumptionProfile] = None
                 , negotiator_type='negmas.sao.AspirationNegotiator'
                 , consumption_horizon: Optional[int] = 20
                 , immediate_cfp_update: bool = True
                 , **kwargs):
        super().__init__(negotiator_type=negotiator_type, **kwargs)
        self.profiles: Dict[int, ConsumptionProfile] = dict()
        self.secured_quantities: Dict[int, int] = dict()
        if profiles is not None:
            self.set_profiles(profiles=profiles)
        self.consumption_horizon = consumption_horizon
        self.immediate_cfp_update = immediate_cfp_update

    def on_new_cfp(self, cfp: 'CFP') -> None:
        pass  # consumers never respond to CFPs

    def init(self):
        super().init()
        if self.factory is None:
            self.factory = Factory(products=self.products, processes=self.processes)
        if self.consumption_horizon is None:
            self.consumption_horizon = self.awi.n_steps
        self.interesting_products = list(self.profiles.keys())

    def set_profiles(self, profiles: Dict[int, ConsumptionProfile]):
        self.profiles = profiles if profiles is not None else dict()
        self.secured_quantities = dict(zip(profiles.keys(), itertools.repeat(0)))

    def register_product_cfps(self, p: int, t: int, profile: ConsumptionProfile):
        current_schedule = profile.schedule_at(t)
        product = self.products[p]
        awi: SCMLAWI = self.awi
        if current_schedule <= 0:
            awi.bulletin_board.remove(section='cfps', query={'publisher': self.id, 'time': t, 'product_index': p})
            return
        max_price = Consumer.RELATIVE_MAX_PRICE * product.catalog_price if product.catalog_price is not None \
            else Consumer.MAX_UNIT_PRICE
        cfps = awi.bulletin_board.query(section='cfps', query={'publisher': self.id, 'time': t, 'product': p})
        if cfps is not None and len(cfps) > 0:
            for _, cfp in cfps.items():
                if cfp.max_quantity != current_schedule:
                    cfp = CFP(is_buy=True, publisher=self.id, product=p
                              , time=t, unit_price=(0, max_price), quantity=(1, current_schedule))
                    awi.bulletin_board.remove(section='cfps', query={'publisher': self.id, 'time': t
                        , 'product': p})
                    awi.register_cfp(cfp)
                    break
        else:
            cfp = CFP(is_buy=True, publisher=self.id, product=p
                      , time=t, unit_price=(0, max_price), quantity=(1, current_schedule))
            awi.register_cfp(cfp)

    def step(self):
        super().step()
        if self.consumption_horizon is None:
            horizon = self.awi.n_steps
        else:
            horizon = min(self.awi.current_step + self.consumption_horizon + 1, self.awi.n_steps)
        for p, profile in self.profiles.items():
            for t in range(self.awi.current_step, horizon):  # + self.transportation_delay
                self.register_product_cfps(p=p, t=t, profile=profile)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    @staticmethod
    def _qufun(outcome: Dict[str, Any], tau: float, profile: ConsumptionProfile):
        """The ufun value for quantity"""
        q, t = outcome['quantity'], outcome['time']
        y = profile.schedule_within(t)
        o = profile.overconsumption
        u = profile.underconsumption
        if q == 0 and y != 0:
            return 0.0
        if y == 0:
            result = -o
        elif q > y:
            result = - o * (((q - y) / y) ** tau)
        elif q < y:
            result = - u * (((y - q) / y) ** tau)
        else:
            result = 1.0
        return math.exp(result)

    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[NegotiatorProxy]:
        profile = self.profiles[cfp.product]
        if profile.cv == 0:
            alpha_u, alpha_q = profile.alpha_u, profile.alpha_q
        else:
            alpha_u, alpha_q = tuple(dirichlet((profile.alpha_u, profile.alpha_q), size=1)[0])
        beta_u = pos_gauss(profile.beta_u, profile.cv)
        tau_u = pos_gauss(profile.tau_u, profile.cv)
        tau_q = pos_gauss(profile.tau_q, profile.cv)
        ufun = normalize(ComplexWeightedUtilityFunction(ufuns=[
            MappingUtilityFunction(mapping=lambda x: 1 - x['unit_price'] ** tau_u / beta_u)
            , MappingUtilityFunction(mapping=functools.partial(Consumer._qufun, tau=tau_q, profile=profile))]
            , weights=[alpha_u, alpha_q], name=self.name + '_' + partner)
            , outcomes=cfp.outcomes, infeasible_cutoff=-1500)
        if self.negotiator_type == AspirationNegotiator:
            negotiator = self.negotiator_type(assume_normalized=True, name=self.name + '*' + partner
                                              , aspiration_type='conceder')
        else:
            negotiator = self.negotiator_type(name=self.name + '*' + partner)
        negotiator.name = self.name + '_' + partner
        negotiator.utility_function = ufun
        return negotiator

    def on_breach_by_self(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        raise ValueError('Consumers should never cause a breach')

    def on_breach_by_another(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        return None  # reject all re-negotiations

    def on_breach_meta_negotiation(self, contract: Contract, partner: str, issues: List[Issue]) \
        -> Optional[NegotiatorProxy]:
        return None

    def on_renegotiation_request(self, contract: Contract, cfp: "CFP", partner: str) -> bool:
        return False

    def confirm_loan(self, loan: Loan) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""
        return False

    def sign_contract(self, contract: Contract) -> Optional[str]:
        if contract is None:
            return
        cfp: CFP = contract.annotation['cfp']
        agreement = contract.agreement  # type: ignore
        schedule = self.profiles[cfp.product].schedule_at(agreement['time'])
        if schedule - agreement['quantity'] < 0:
            return None
        return super().sign_contract(contract=contract)

    def on_contract_signed(self, contract: Contract):
        super().on_contract_signed(contract)
        if contract is None:
            return
        cfp: CFP = contract.annotation['cfp']
        agreement = contract.agreement  # type: ignore
        self.secured_quantities[cfp.product] += agreement['quantity']
        old_quantity = self.profiles[cfp.product].schedule_at(agreement['time'])
        new_quantity = old_quantity - agreement['quantity']
        t = agreement['time']
        self.profiles[cfp.product].set_schedule_at(time=t
                                                   , value=new_quantity
                                                   , n_steps=self.awi.n_steps)
        if self.immediate_cfp_update and new_quantity != old_quantity:
            self.register_product_cfps(p=cfp.product, t=t, profile=self.profiles[cfp.product])
        for negotiation in self.running_negotiations.values():
            self.notify(negotiation.negotiator, Notification(type='ufun_modified', data=None))


@dataclass
class MiningProfile:
    cv: float = 0.05

    alpha_t: float = 1.0
    alpha_q: float = 1.0
    alpha_u: float = 1.0

    beta_t: float = 1.0
    beta_q: float = 100.0
    beta_u: float = 100.0

    tau_t: float = -0.25
    tau_q: float = 0.25
    tau_u: float = 1.0

    @classmethod
    def random(cls):
        alpha_t, alpha_q, alpha_u = dirichlet((1, 1, 1), size=1)[0]
        tau_t, tau_q, tau_u = 2 * random() - 1, 2 * random() - 1, 2 * random() - 1
        return MiningProfile(cv=random(), alpha_t=alpha_t, alpha_q=alpha_q, alpha_u=alpha_u
                             , tau_t=tau_t, tau_q=tau_q, tau_u=tau_u
                             , beta_t=1.5 * random(), beta_q=99 * random() + 1, beta_u=99 * random() + 1)


class Miner(SCMLAgent, ConfigReader):
    """Raw Material Generator"""

    def __init__(self, profiles: Dict[int, MiningProfile] = None, negotiator_type='negmas.sao.AspirationNegotiator'
                 , n_retrials=0, reactive=True, **kwargs):
        super().__init__(negotiator_type=negotiator_type, **kwargs)
        self.profiles: Dict[int, MiningProfile] = {}
        self.n_neg_trials: Dict[str, int] = defaultdict(int)
        self.n_retrials = n_retrials
        self.reactive = reactive
        if profiles is not None:
            self.set_profiles(profiles=profiles)

    def init(self):
        super().init()
        if self.factory is None:
            self.factory = Factory(products=self.products, processes=self.processes)
        self.interesting_products = list(self.profiles.keys())

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any], mechanism: MechanismInfo
                               , state: MechanismState) -> None:
        # noinspection PyUnusedLocal
        cfp = annotation['cfp']
        super().on_negotiation_failure(partners=partners, annotation=annotation, mechanism=mechanism, state=state)
        thiscfp = self.awi.bulletin_board.query(section='cfps', query=cfp.id, query_keys=True)
        if cfp.publisher != self.id and thiscfp is not None and len(thiscfp) > 0 \
             and self.n_neg_trials[cfp.id] < self.n_retrials:
            self.awi.logdebug(f'Renegotiating {self.n_neg_trials[cfp.id]} on {cfp}')
            self.on_new_cfp(cfp=annotation['cfp'])

    def set_profiles(self, profiles: Dict[int, MiningProfile]):
        self.profiles = profiles if profiles is not None else dict()

    def _process_cfp(self, cfp: 'CFP'):
        if not self.can_expect_agreement(cfp=cfp):
            return
        profile = self.profiles.get(cfp.product, None)
        if profile is None:
            return
        if profile.cv == 0:
            alpha_u, alpha_q, alpha_t = profile.alpha_u, profile.alpha_q, profile.alpha_t
        else:
            alpha_u, alpha_q, alpha_t = tuple(
                dirichlet((profile.alpha_u, profile.alpha_q, profile.alpha_t), size=1)[0])
        beta_u = pos_gauss(profile.beta_u, profile.cv)
        beta_t = pos_gauss(profile.beta_t, profile.cv)
        beta_q = pos_gauss(profile.beta_q, profile.cv)

        tau_u = pos_gauss(profile.tau_u, profile.cv)
        tau_t = pos_gauss(profile.tau_t, profile.cv)
        tau_q = pos_gauss(profile.tau_q, profile.cv)

        ufun = normalize(LinearUtilityAggregationFunction(issue_utilities={
            'time': lambda x: x ** tau_t / beta_t,
            'quantity': lambda x: x ** tau_q / beta_q,
            'unit_price': lambda x: x ** tau_u / beta_u if x > 1e-7 else -2000.0,
        }, weights={'time': alpha_t, 'quantity': alpha_q, 'unit_price': alpha_u})
            , outcomes=cfp.outcomes, infeasible_cutoff=-1500)
        if self.negotiator_type == AspirationNegotiator:
            negotiator = self.negotiator_type(assume_normalized=True, name=self.name + '*' + cfp.publisher
                                              , dynamic_ufun=False, aspiration_type='conceder')
        else:
            negotiator = self.negotiator_type(assume_normalized=True, name=self.name + '*' + cfp.publisher)
        negotiator.utility_function = normalize(ufun, outcomes=cfp.outcomes, infeasible_cutoff=None)
        self.n_neg_trials[cfp.id] += 1
        self.request_negotiation(partners=[cfp.publisher, self.id], issues=cfp.issues, negotiator=negotiator
                                 , annotation=self._create_annotation(cfp=cfp), extra=None
                                 , mechanism_name='negmas.sao.SAOMechanism', roles=None)

    def on_new_cfp(self, cfp: 'CFP'):
        if self.reactive:
            if not cfp.satisfies(query={'products': list(self.profiles.keys())}):
                return
            self._process_cfp(cfp)

    def step(self):
        super().step()
        if not self.reactive:
            cfps = self.awi.bulletin_board.query(section='cfps', query_keys=False
                                                 , query={'products': list(self.profiles.keys())})
            if cfps is None:
                return
            cfps = cfps.values()
            for cfp in cfps:
                self._process_cfp(cfp)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[NegotiatorProxy]:
        raise ValueError('Miners should never receive negotiation requests as they publish no CFPs')

    def on_breach_by_self(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        raise ValueError('Miners should never cause a breach')

    def on_breach_by_another(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        return None  # reject all re-negotiations

    def on_breach_meta_negotiation(self, contract: Contract, partner: str, issues: List[Issue]) \
        -> Optional[NegotiatorProxy]:
        return None

    def on_renegotiation_request(self, contract: Contract, cfp: "CFP", partner: str) -> bool:
        return False

    def confirm_loan(self, loan: Loan) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""
        return False


def safe_max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


class Scheduler(ABC):
    """Base class for all schedulers"""

    def __init__(self, factory: Factory, n_steps: int, products: List[Product], processes: List[Process]
                 , manager_id: str):
        self.managed_factory = factory
        self.scheduling_factory: Factory = None
        self.n_steps = n_steps
        self.products = products
        self.processes = processes
        self.manager_id = manager_id

    def init(self):
        """Called by the FactoryManager after it is initialized"""

    def schedule(self, contracts: Collection[Contract] = None
                 , initial_storage: Dict[int, int] = None
                 , initial_balance: float = None, initial_jobs: List[Job] = None
                 , n_steps: int = None, assume_no_further_negotiations=False
                 , ensure_storage_for: int = 0
                 , reset=False) -> ScheduleInfo:
        """
        Schedules a set of contracts and returns either the search_for_schedule or None if infeasible

        Args:
            contracts:
            initial_storage:
            initial_balance:
            initial_jobs:
            n_steps (int):
            assume_no_further_negotiations:
            ensure_storage_for:
            reset: If true, then reset the schedule before finding a new one

        Returns:

        """
        if reset or self.scheduling_factory is None:
            self.reset_schedule(to_real=True)
        factory = self.scheduling_factory

        # initialize the state of the scheduler with initial knowledge
        success = True
        if initial_storage is not None or initial_balance is not None or initial_jobs is not None:
            if n_steps is None:
                n_steps = self.n_steps
            success = factory.init_schedule(n_steps=n_steps, initial_jobs=initial_jobs, initial_balance=initial_balance
                                            , initial_storage=initial_storage)

        # if we have not new contracts, then we are done
        if contracts is None or len(contracts) == 0:
            return ScheduleInfo(valid=success, failed_contracts=[], end=n_steps, factory=factory)

        return self.find_schedule(contracts=contracts, n_steps=n_steps
                                  , assume_no_further_negotiations=assume_no_further_negotiations
                                  , ensure_storage_for=ensure_storage_for)

    @abstractmethod
    def find_schedule(self, contracts: Collection[Contract]
                      , n_steps: int = None
                      , assume_no_further_negotiations=False
                      , ensure_storage_for: int = 0) -> ScheduleInfo:
        """
        Schedules a set of contracts and returns either the search_for_schedule or None if infeasible

        Args:
            contracts:
            n_steps (int):
            assume_no_further_negotiations:
            ensure_storage_for:

        Returns:

        """

    def reset_schedule(self, to_real=True) -> None:
        """Resets the schedule stored

        Arg:
            to_real: If true, the scheduler is reset to the state of it's manager's factory otherwise to a clean factory

        """
        if to_real:
            if self.managed_factory is None:
                raise ValueError("Cannot init to the manager's factory because it does not have one")
            self.scheduling_factory = self.managed_factory.copy()
        else:
            self.scheduling_factory = Factory(products=self.products, processes=self.processes)


class GreedyScheduler(Scheduler):
    """Default scheduler used by the DefaultFactoryManager"""

    def __init__(self, factory: Factory, n_steps: int, products: List[Product], processes: List[Process]
                 , manager_id: str, awi: 'SCMLAWI', max_insurance_premium: Optional[float] = None
                 , add_catalog_prices=True, strategy: str = 'latest', profile_sorter: str = 'total-cost>time'):
        """

        Args:
            add_catalog_prices: Whether to add total catalog price costs to costs of production
            strategy: How to schedule production. Possible values are earliest, latest, shortest, longest

        Remarks:

            The following `production_strategy` values are supported:

            - earliest: Try to produce things as early as possible. Useful for infinite storage
            - latest: Try to produce things as late as possible. Useful for finite storage
            - shortest: Schedule in the time/line that has the shortest empty slot that is enough for production
            - longest: Schedule in the time/line that has the longest empty slot that is enough for production

            The `profile_sorter` string consists of one or more of the following sections separated by ``>`` characters
            to indicate sorting order. Costs are sorted ascendingly and times descendingly. Costs and times refer to
            unit cost/time (total divided by quantity generated):

            - time, t: profile production time per unit
            - input-cost, ic, icost: Input cost per unit only using catalog prices
            - production-cost, pc, pcost: Production cost as specified in the profile per unit
            - total-cost, tc, tcost: Total cost per unit including input cost


        """
        super().__init__(factory=factory, n_steps=n_steps, products=products, processes=processes
                         , manager_id=manager_id)
        self.add_catalog_prices = add_catalog_prices
        self.strategy = strategy
        self.awi = awi
        self.max_insurance_premium = max_insurance_premium
        self.fields: List[Callable[[ManufacturingInfo], float]] = [self.total_unit_cost, self.unit_time
            , self.production_unit_cost, self.input_unit_cost]
        mapper = {'tc': 0, 't': 1, 'pc': 2, 'ic': 3}
        self.field_order: List[int] = []
        sort_fields = profile_sorter.split('>')
        self.producing: Dict[int, List[ManufacturingInfo]] = {}
        for field_name in sort_fields:
            if field_name in ('time', 't'):
                self.field_order.append(mapper['t'])
            elif field_name in ('total-cost', 'tc', 'tcost'):
                self.field_order.append(mapper['tc'])
            elif field_name in ('production-cost', 'pc', 'pcost'):
                self.field_order.append(mapper['pc'])
            elif field_name in ('input-cost', 'ic', 'icost'):
                self.field_order.append(mapper['ic'])

    def init(self):
        self.producing = {k: sorted(v, key=self._profile_sorter) for k, v in self.managed_factory.producing.items()}

    def _profile_sorter(self, info: ManufacturingInfo) -> Any:
        vals = [field(info) for field in self.fields]
        return tuple([vals[indx] for indx in self.field_order] + [info.line_name, info.process])

    def unit_time(self, info: ManufacturingInfo) -> float:
        return info.profile.n_steps / info.quantity

    def total_cost(self, info: ManufacturingInfo) -> float:
        products = self.managed_factory.products
        process = self.managed_factory.processes[info.process]
        production_cost = info.profile.cost

        def safe(x):
            return 0.0 if x is None else x

        inputs_cost = sum(safe(products[inp.product].catalog_price) * inp.quantity for inp in process.inputs)
        return production_cost + inputs_cost

    def total_unit_cost(self, info: ManufacturingInfo) -> float:
        return self.total_cost(info=info) / info.quantity

    def production_cost(self, info: ManufacturingInfo) -> float:
        return info.profile.cost

    def production_unit_cost(self, info: ManufacturingInfo) -> float:
        return self.production_cost(info=info) / info.quantity

    def input_cost(self, info: ManufacturingInfo):
        products = self.managed_factory.products
        process = self.managed_factory.processes[info.process]

        def safe(x):
            return 0.0 if x is None else x

        return sum(safe(products[inp.product].catalog_price) * inp.quantity for inp in process.inputs)

    def input_unit_cost(self, info: ManufacturingInfo) -> float:
        return self.input_cost(info=info) / info.quantity

    # noinspection PyUnusedLocal
    def schedule_contract(self, contract: Contract, end: int = None, ensure_storage_for: int = 0) -> ScheduleInfo:
        """
        Schedules this contract if possible and returns information about the resulting schedule

        Args:
            contract:
            end:
            ensure_storage_for: The number of steps all needs must be in storage before they are consumed in production

        Returns:

        """
        if self.scheduling_factory is None:
            self.reset_schedule(to_real=True)
        factory = self.scheduling_factory
        if end is None:
            end = factory.n_scheduling_steps
        if contract.agreement is None:
            return ScheduleInfo(factory=factory, end=end)
        agreement: SCMLAgreement
        if isinstance(contract.agreement, dict):
            agreement = SCMLAgreement(**contract.agreement)
        else:
            agreement = contract.agreement  # type: ignore
        t = agreement['time']
        if t >= factory.n_steps:
            factory.n_steps = t + 1  # @todo check if I can just make the schedule start from now
        q, u = int(agreement['quantity']), agreement['unit_price']
        p = u * q
        pid: int = contract.annotation['cfp'].product
        balances, storage = factory.predicted_balance, factory.predicted_storage
        total = factory.predicted_total_storage
        if contract.annotation['buyer'] == self.manager_id:
            # I am a buyer
            if self.max_insurance_premium is None:
                insurance = 0.0
            else:
                insurance = self.awi.evaluate_insurance(contract=contract, t=self.awi.current_step)
            if balances[t] >= p + insurance and \
                (factory.max_storage is None or np.max(total[t:]) <= factory.max_storage - q):
                storage[pid, t:] += q
                total[t:] += q
                balances[t:] -= p
                return ScheduleInfo(factory=factory, end=end
                                    , updates={t: FactoryStatusUpdate(balance=-p, storage={pid: q})})
            return ScheduleInfo(factory=factory, valid=False, failed_contracts=[contract], end=end)
        elif contract.annotation['seller'] == self.manager_id:
            # I am a seller

            # if enough is available in storage and not reserved, just sell it
            q_needed = q - storage[pid, t] - factory.predicted_reserved_storage[pid, t]
            if q_needed <= 0:
                storage[pid, t:] -= q
                total[t:] -= q
                balances[t:] += p
                return ScheduleInfo(factory=factory, end=end
                                    , updates={t: FactoryStatusUpdate(balance=p, storage={pid: -q})})
            jobs: List[Job] = []
            needs: List[ProductionNeed] = []

            saved_factory = factory.copy()
            some_production = True
            while q_needed > 0 and some_production:
                some_production = False
                # I need now to schedule the production needed and calculate all required input products
                for info in factory.producing[pid]:
                    # find if it is possible to use the current process for producing the product
                    line_name, process_index, profile = info.line_name, info.process, info.profile
                    line = factory.lines[line_name]
                    q_produced, t_production = info.quantity, info.step
                    current_schedule = line.state.schedule.view()[:t - ensure_storage_for]
                    if len(current_schedule) < t_production:
                        continue
                    locs = zero_runs((current_schedule != EMPTY_STEP).astype(int))
                    lengths = locs[:, 1] - locs[:, 0]
                    indices = np.array(range(len(lengths)))
                    indices = indices[lengths >= t_production]
                    if len(indices) < 1:
                        continue
                    lengths, locs = lengths[indices], locs[indices]
                    if self.strategy == 'earliest':
                        # find the first location that fits the required production time
                        loc = locs[0, :]
                    elif self.strategy == 'latest':
                        # there is a max storage, produce as need the deadline as possible
                        loc = locs[-1, :] - 1
                        loc[0] = loc[1] - t_production
                    elif self.strategy == 'shortest':
                        sorted_lengths = sorted(zip(range(len(lengths)), lengths), key=lambda x: x[1])
                        loc = locs[sorted_lengths[0][0], :]
                    elif self.strategy == 'longest':
                        sorted_lengths = sorted(zip(range(len(lengths)), lengths), key=lambda x: x[1], reverse=True)
                        loc = locs[sorted_lengths[0][0], :]
                    else:
                        raise ValueError(f'Unknown production strategy {self.strategy}')
                    ptime = loc[0]
                    job = Job(line_name=line_name, command='run', time=ptime, process=process_index
                              , contract=contract, updates={})
                    if not factory.schedule_job(job, end=end):
                        continue  # should never hit this
                    jobs.append(job)
                    needs += factory.find_needs(job)
                    # @todo consider stopping production after the product is available (+ ensure_storage_for) if needed
                    q_needed -= q_produced
                    # storage[pid, ptime:] -= q_produced
                    # total[ptime:] -= q_produced
                    # balances[ptime:] += info.profile.cost
                    some_production = True
                    break
            if q_needed <= 0:
                # add the effect of selling
                storage[pid, t:] -= q
                total[t:] -= q
                balances[t:] += p

                # create schedule
                schedule = ScheduleInfo(factory=factory, jobs=jobs, end=end, needs=needs, failed_contracts=[])

                # add the effect of buying raw materials
                for need in needs:
                    product_index = need.product
                    product = self.products[product_index]
                    catalog_price = product.catalog_price
                    if catalog_price == 0 or need.quantity_to_buy <= 0:
                        continue
                    price = need.quantity_to_buy * catalog_price
                    factory.predicted_balance[need.step:] -= price
                    schedule.updates[need.step].balance -= price
                return schedule
            self.scheduling_factory = saved_factory
            return ScheduleInfo(factory=factory, valid=False, failed_contracts=[contract], end=end)
        raise ValueError(f'{self.manager_id} Not a seller of a buyer in Contract: {contract} with '
                         f'annotation: {contract.annotation}')

    def schedule_contracts(self, contracts: Collection[Contract], end: int = None
                           , ensure_storage_for: int = 0) -> ScheduleInfo:
        """
        Schedules a set of contracts and returns the `ScheduleInfo`.

        Args:
            contracts: Contracts to schedule
            end: The end of the simulation for the schedule (exclusive)
            ensure_storage_for: Ensure that the outcome will be at the storage for at least this time

        Returns:

            ScheduleInfo giving the schedule after these contracts is included. `valid` member can be used to check
            whether this is a valid contract

        """
        factory = self.scheduling_factory
        if factory is None:
            raise ValueError('Cannot schedule without a factory')
        if end is None:
            end = factory.n_scheduling_steps
        result = ScheduleInfo(factory=factory, valid=True, end=end)
        contracts = sorted(contracts, key=lambda x: x.agreement['time'])
        for contract in contracts:
            result.combine(self.schedule_contract(contract, end=end, ensure_storage_for=ensure_storage_for))
        return result

    def find_schedule(self, contracts: Collection[Contract]
                      , n_steps: int = None
                      , assume_no_further_negotiations=False
                      , ensure_storage_for: int = 0):
        """
        Schedules a set of contracts and returns either the search_for_schedule or None if infeasible

        Args:
            contracts:
            n_steps (int):
            assume_no_further_negotiations:
            ensure_storage_for:

        Returns:

        """
        factory = self.scheduling_factory

        # set the number of steps to simulate (No more than what is necessary to speedup calculation).
        n_steps_max = max([contract.agreement['time'] + 1 for contract in contracts])
        n_steps = self.n_steps if n_steps is None else n_steps
        if n_steps is not None:
            if factory.n_steps is not None:
                n_steps = max(n_steps, self.n_steps)
            end = min(n_steps_max, n_steps)
        else:
            end = n_steps_max
        if factory.n_steps is None:
            factory.n_scheduling_steps = end
        if factory.n_steps > end:
            factory.n_steps = end

        # Now, schedule the contracts
        schedule = self.schedule_contracts(contracts=contracts, end=end, ensure_storage_for=ensure_storage_for)

        # Mark the schedule as invalid if it has any production needs and we assume_no_further_negotiations
        if assume_no_further_negotiations and schedule.needs is not None and len(schedule.needs) > 0:
            schedule.valid = False
            return schedule

        # add the expected cost of inputs to the updates using catalog prices
        # if schedule.valid and self.add_catalog_prices and schedule.needs is not None:
        #     for need in schedule.needs:
        #         product_index = need.product
        #         product = self.products[product_index]
        #         catalog_price = product.catalog_price
        #         if catalog_price == 0 or need.quantity_to_buy <= 0:
        #             continue
        #         price = need.quantity_to_buy * catalog_price
        #         schedule.factory.predicted_balance[need.step:] -= price
        #         schedule.updates[need.step].balance -= price
        return schedule


class FactoryManager(SCMLAgent, ConfigReader, ABC):
    """Base factory manager class that will be inherited by participant negmas in ANAC 2019"""

    def __init__(self, factory: Factory = None, negotiator_type='negmas.sao.AspirationNegotiator', **kwargs):
        super().__init__(factory=factory, negotiator_type=negotiator_type, **kwargs)
        self.transportation_delay = 0
        self.scheduler: Optional[Scheduler] = None

    def init(self):
        super().init()
        if self.scheduler:
            self.scheduler.init()
        self.interesting_products = list(self.factory.producing.keys())
        self.interesting_products += list(self.factory.consuming.keys())

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    def on_breach_by_self(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        return None

    def on_breach_by_another(self, contract: Contract, partner: str) -> Optional[RenegotiationRequest]:
        return None

    def on_renegotiation_request(self, contract: Contract, cfp: "CFP", partner: str) -> bool:
        return False

    def confirm_loan(self, loan: Loan) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""
        return True

    def on_breach_meta_negotiation(self, contract: Contract
                                   , partner: str, issues: List[Issue]) -> Optional[NegotiatorProxy]:
        return None

    def on_event(self, event: Event, sender: EventSource):
        super().on_event(event=event, sender=sender)

    def total_utility(self, contracts: Collection[Contract] = None) -> float:
        """Calculates the total utility for the agent of a collection of contracts"""
        if self.scheduler is None:
            raise ValueError('Cannot calculate total utility without a scheduler')
        schedule = self.scheduler.schedule(contracts=contracts, assume_no_further_negotiations=False
                                           , ensure_storage_for=self.transportation_delay
                                           , reset=contracts is None or len(contracts) == 0)
        if not schedule.valid:
            return -2000.0
        return schedule.factory.predicted_balance[-1]


# @dataclass
# class ProductionInfo:
#     process: Process
#     cost: float
#     n_steps: int
#     lines: List[Line] = field(default_factory=lambda: defaultdict(list))


class GreedyFactoryManager(FactoryManager):
    """The default factory manager that will be implemented by the committee of ANAC-SCML 2019"""

    def __init__(self, factory=None, name=None, optimism: float = 0.0, p_negotiation=0.25
                 , negotiator_type='negmas.sao.AspirationNegotiator', single_sell_order: bool = False
                 , n_retrials=0, use_consumer=False, reactive=True, sign_only_guaranteed_contracts=True
                 , max_insurance_premium=None):
        super().__init__(factory=factory, name=name, negotiator_type=negotiator_type)
        self.optimism = optimism
        self.p_negotiation = p_negotiation
        self.ufun_factory: Union[Type[NegotiatorUtility], Callable[[Any, Any], NegotiatorUtility]]
        if optimism < 1e-6:
            self.ufun_factory = PessimisticNegotiatorUtility
        elif optimism > 1 - 1e-6:
            self.ufun_factory = OptimisticNegotiatorUtility
        else:
            self.ufun_factory: NegotiatorUtility = lambda agent, annotation: \
                AveragingNegotiatorUtility(agent=agent, annotation=annotation, optimism=self.optimism)
        self.single_sell_order = single_sell_order
        self.max_insurance_premium = max_insurance_premium
        self.n_retrials = n_retrials
        self.n_neg_trials: Dict[str, int] = defaultdict(int)
        self.consumer = None
        self.use_consumer = use_consumer
        self.reactive = reactive
        self.sign_only_guaranteed_contracts = sign_only_guaranteed_contracts
        self.contract_schedules: Dict[str, ScheduleInfo] = {}

    def init(self):
        super().init()
        if self.use_consumer:
            self.consumer: Consumer = Consumer(profiles=dict(zip(self.factory.consuming.keys()
                                                                 , (ConsumptionProfile(schedule=[_] * self.awi.n_steps)
                                                                    for _ in itertools.repeat(0))))
                                               , consumption_horizon=self.awi.n_steps, immediate_cfp_update=True
                                               , name=self.name)
            self.consumer.id = self.id
            self.consumer.awi = self.awi
            self.consumer.init()
        self.scheduler = GreedyScheduler(factory=self.factory, n_steps=self.awi.n_steps, products=self.products
                                         , processes=self.processes, manager_id=self.id
                                         , awi=self.awi, max_insurance_premium=self.max_insurance_premium)

    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[NegotiatorProxy]:
        if self.use_consumer:
            return self.consumer.on_negotiation_request(cfp=cfp, partner=partner)
        else:
            if self.negotiator_type == AspirationNegotiator:
                neg = self.negotiator_type(assume_normalized=True, aspiration_type='conceder'
                                           , name=self.name + '*' + partner)
            else:
                neg = self.negotiator_type(name=self.name + '*' + partner)
            neg.utility_function = normalize(self.ufun_factory(self, self._create_annotation(cfp=cfp)),
                                             outcomes=cfp.outcomes, infeasible_cutoff=-1500)
            return neg

    def on_negotiation_success(self, contract: Contract, mechanism: MechanismInfo):
        super().on_negotiation_success(contract=contract, mechanism=mechanism)
        if self.use_consumer:
            self.consumer.on_negotiation_success(contract, mechanism)

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any], mechanism: MechanismInfo
                               , state: MechanismState) -> None:
        super().on_negotiation_failure(partners=partners, annotation=annotation, mechanism=mechanism, state=state)
        if self.use_consumer:
            self.consumer.on_negotiation_failure(partners, annotation, mechanism, state)
        cfp = annotation['cfp']
        thiscfp = self.awi.bulletin_board.query(section='cfps', query=cfp.id, query_keys=True)
        if cfp.publisher != self.id and thiscfp is not None and len(thiscfp) > 0 \
            and self.n_neg_trials[cfp.id] < self.n_retrials:
            self.awi.logdebug(f'Renegotiating {self.n_neg_trials[cfp.id]} on {cfp}')
            self.on_new_cfp(cfp=annotation['cfp'])

    def _execute_schedule(self, schedule: ScheduleInfo, contract: Contract) -> None:
        if self.factory is None:
            raise ValueError('No factory is defined')
        awi: SCMLAWI = self.awi
        if contract.annotation['buyer'] == self.id:
            if self.max_insurance_premium is not None and contract is not None:
                premium = awi.evaluate_insurance(contract=contract)
                if premium is not None:
                    relative_premium = premium / (contract.agreement['unit_price'] * contract.agreement['quantity'])
                    if relative_premium <= self.max_insurance_premium:
                        awi.buy_insurance(contract=contract)
                        self.factory.predicted_balance[self.awi.current_step:] -= premium
            for t, update in schedule.updates.items():
                self.factory.predicted_balance[t:] -= update.balance
                for product_id, q in update.storage.items():
                    self.factory.predicted_reserved_storage[product_id, t] += q
        else:
            for job in schedule.jobs:
                awi.execute(action=Action(type=job.command, params={'line_name': job.line_name, 'time': job.time
                    , 'process': job.process, 'contract': contract, 'override': True}))
            for t, update in schedule.updates.items():
                self.factory.predicted_balance[t:] -= update.balance
                for product_id, q in update.storage.items():
                    self.factory.predicted_storage[product_id, t:] += q
            for need in schedule.needs:
                if need.quantity_to_buy <= 0:
                    continue
                product_id = need.product
                if self.use_consumer:
                    self.consumer.profiles[product_id].schedule[need.step] += need.quantity_to_buy
                    self.consumer.register_product_cfps(p=product_id, t=need.step
                                                        , profile=self.consumer.profiles[product_id])
                else:
                    product = self.factory.products[product_id]
                    if product.catalog_price is None:
                        price_range = (0.0, 100.0)
                    else:
                        price_range = (0.5, 1.5 * product.catalog_price)
                    # @todo check this. This error is raised sometimes
                    if need.step < awi.current_step:
                        continue
                        # raise ValueError(f'need {need} at {need.step} while running at step {awi.current_step}')
                    time = need.step if self.factory.max_storage is not None else (awi.current_step, need.step)
                    cfp = CFP(is_buy=True, publisher=self.id, product=product_id
                              , time=time, unit_price=price_range
                              , quantity=(1, int(1.1 * need.quantity_to_buy)))
                    awi.register_cfp(cfp)

    def sign_contract(self, contract: Contract):
        signature = super().sign_contract(contract)
        if signature is None:
            return None
        schedule = self.scheduler.schedule(assume_no_further_negotiations=False, contracts=[contract]
                                           , ensure_storage_for=self.transportation_delay, reset=True)
        if self.sign_only_guaranteed_contracts and (not schedule.valid or len(schedule.needs) > 1):
            return None
        self.contract_schedules[contract.id] = schedule
        return signature

    def on_contract_signed(self, contract: Contract):
        super().on_contract_signed(contract)
        if contract.annotation['buyer'] == self.id and self.use_consumer:
            self.consumer.on_contract_signed(contract)
        schedule = self.contract_schedules[contract.id]
        if schedule is not None and schedule.valid:
            self._execute_schedule(schedule=schedule, contract=contract)
        if contract.annotation['buyer'] != self.id or not self.use_consumer:
            for negotiation in self.running_negotiations.values():
                self.notify(negotiation.negotiator, Notification(type='ufun_modified', data=None))

    def _process_buy_cfp(self, cfp: 'CFP') -> None:
        if self.factory is None or not self.can_expect_agreement(cfp=cfp) or not self.can_produce(cfp=cfp):
            return
        if self.negotiator_type == AspirationNegotiator:
            neg = self.negotiator_type(assume_normalized=True, name=self.name + '>' + cfp.publisher)
        else:
            neg = self.negotiator_type(name=self.name + '>' + cfp.publisher)
        neg.utility_function = normalize(self.ufun_factory(self, self._create_annotation(cfp=cfp)),
                                         outcomes=cfp.outcomes, infeasible_cutoff=-1500)
        self.request_negotiation(negotiator=neg, extra=None, partners=[cfp.publisher, self.id]
                                 , issues=cfp.issues, annotation=self._create_annotation(cfp=cfp)
                                 , mechanism_name='negmas.sao.SAOMechanism')

    def _process_sell_cfp(self, cfp: 'CFP'):
        if self.use_consumer:
            self.consumer.on_new_cfp(cfp=cfp)

    def on_new_cfp(self, cfp: 'CFP') -> None:
        if not self.reactive:
            return
        if cfp.satisfies(query={'is_buy': True, 'products': list(self.factory.producing.keys())}):
            self._process_buy_cfp(cfp)
        if cfp.satisfies(query={'is_buy': False, 'products': list(self.factory.consuming.keys())}):
            self._process_sell_cfp(cfp)

    def step(self):
        if self.use_consumer:
            self.consumer.step()
        if self.reactive:
            return
        # 0. remove all my CFPs
        # self.awi.bulletin_board.remove(section='cfps', query={'publisher': self})

        # respond to interesting CFPs
        # todo: should check time and sort products by interest etc
        cfps = self.awi.bulletin_board.query(section='cfps'
                                             , query={'products': self.factory.producing.keys(), 'is_buy': True})
        if cfps is None:
            return
        for cfp in cfps.values():
            self._process_buy_cfp(cfp)
            if self.single_sell_order:
                break

    def can_produce(self, cfp: CFP, assume_no_further_negotiations=False) -> bool:
        """Whether or not we can produce the required item in time"""
        if cfp.product not in self.factory.producing.keys():
            return False
        agreement = SCMLAgreement(time=cfp.max_time, unit_price=cfp.max_unit_price, quantity=cfp.min_quantity)
        min_concluded_at = self.awi.current_step + 1 - int(self.immediate_negotiations)
        min_sign_at = min_concluded_at + self.awi.default_signing_delay
        if cfp.max_time < min_sign_at + 1:  # 1 is minimum time to produce the product
            return False
        schedule = self.scheduler.schedule(contracts=[Contract(partners=[self.id, cfp.publisher]
                                                               , agreement=agreement
                                                               , annotation=self._create_annotation(cfp=cfp)
                                                               , issues=cfp.issues, signed_at=min_sign_at
                                                               , concluded_at=min_concluded_at)]
                                           , ensure_storage_for=self.transportation_delay
                                           , reset=True
                                           , assume_no_further_negotiations=assume_no_further_negotiations)
        return schedule.valid and self.needs_securable(schedule=schedule, step=self.awi.current_step)

    def needs_securable(self, schedule: ScheduleInfo, step: int):
        """
        Finds if it is possible in principle to arrange these needs at the given time.

        Args:
            schedule:
            step:

        Returns:

        """
        needs = schedule.needs
        if len(needs) < 1:
            return True
        for need in needs:
            if need.quantity_to_buy > 0 and need.step < step + 1 - int(self.immediate_negotiations): #@todo check this
                return False
        return True


class DoNothingFactoryManager(FactoryManager):
    """The default factory manager that will be implemented by the committee of ANAC-SCML 2019"""

    def on_new_cfp(self, cfp: 'CFP') -> None:
        pass

    def __init__(self, factory=None, name=None):
        super().__init__(factory=factory, name=name)

    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[NegotiatorProxy]:
        return None

    def step(self):
        pass


TotalUtilityFun = Callable[[Collection[Contract]], float]


class NegotiatorUtility(UtilityFunctionProxy):
    """The utility function of a negotiator."""

    def __init__(self, agent: FactoryManager, annotation: Dict[str, Any], name: Optional[str] = None):
        if name is None:
            name = agent.name + '*' + '*'.join(_ for _ in annotation['partners'] if _ != agent.id)
        super().__init__(name=name)
        self.agent = agent
        self.annotation = annotation

    def _contracts(self, agreements: Iterable[SCMLAgreement]) -> Collection[Contract]:
        """Converts agreements/outcomes into contracts"""
        if self.info is None:
            raise ValueError('No annotation is stored (No mechanism info)')
        annotation = self.info.annotation
        return [Contract(partners=annotation['partners'], agreement=a, annotation=annotation, issues=self.info.issues)
                for a in agreements]

    def _contract(self, agreement: SCMLAgreement) -> Contract:
        """Converts an agreement/outcome into a contract"""
        annotation = self.annotation
        return Contract(partners=annotation['partners'], agreement=agreement
                        , annotation=annotation, issues=annotation['cfp'].issues)

    def _free_sale(self, agreement: SCMLAgreement) -> bool:
        return self.annotation['seller'] == self.agent and agreement['unit_price'] < 1e-6

    def __call__(self, outcome: Outcome) -> Optional[UtilityValue]:
        if isinstance(outcome, dict):
            return self.call(agreement=SCMLAgreement(**outcome))
        if isinstance(outcome, SCMLAgreement):
            return self.call(agreement=outcome)
        raise ValueError(f'Outcome: {outcome} cannot be converted to an SCMLAgreement')

    @abstractmethod
    def call(self, agreement: SCMLAgreement) -> Optional[UtilityValue]:
        """Called to evaluate a agreement"""

    def xml(self, issues: List[Issue]) -> str:
        return 'NegotiatorUtility has not xml representation'


class PessimisticNegotiatorUtility(NegotiatorUtility):
    """The utility function of a negotiator that assumes other negotiations currently open will fail."""

    def call(self, agreement: SCMLAgreement) -> Optional[UtilityValue]:
        """An offer will be a tuple of one value which in turn will be a list of contracts"""
        if self._free_sale(agreement):
            return -2000.0
        # contracts = self.agent.contracts
        # hypothetical = list(contracts)
        # hypothetical.append(self._contract(agreement))
        hypothetical = [self._contract(agreement)]
        base_util = self.agent.total_utility(contracts=None)
        return self.agent.total_utility(hypothetical) - base_util


class OptimisticNegotiatorUtility(NegotiatorUtility):
    """The utility function of a negotiator that assumes other negotiations currently open will succeed."""

    def call(self, agreement: SCMLAgreement) -> Optional[UtilityValue]:
        if self._free_sale(agreement):
            return -2000.0
        # contracts = self.agent.contracts
        # hypothetical = list(contracts)
        # hypothetical.append(self._contract(agreement))
        hypothetical = [self._contract(agreement)]
        for negotiation in self.agent.running_negotiations.values():  # type: ignore
            negotiator = negotiation.negotiator
            current_offer = negotiator.my_last_proposal
            if current_offer is not None:
                hypothetical.append(self._contract(current_offer))
        base_util = self.agent.total_utility(contracts=None)
        return self.agent.total_utility(list(hypothetical)) - base_util


class AveragingNegotiatorUtility(NegotiatorUtility):
    """A utility function that combines optimistic and pessimistic evaluators linearly using adjustable weight"""

    def __init__(self, agent: FactoryManager, annotation: Dict[str, Any], name: Optional[str] = None
                 , optimism: float = 0.5):
        NamedObject.__init__(self=self, name=name)
        self.optimism = optimism
        self.optimistic = OptimisticNegotiatorUtility(agent=agent, annotation=annotation)
        self.pessimistic = PessimisticNegotiatorUtility(agent=agent, annotation=annotation)

    def call(self, agreement: SCMLAgreement) -> Optional[UtilityValue]:
        if self._free_sale(agreement):
            return -2000.0
        opt, pess = self.optimistic(agreement), self.pessimistic(agreement)
        if opt is None or pess is None:
            return None
        return self.optimism * opt + (1 - self.optimism) * pess


# todo: implement snapshots and save them.
@dataclass
class SnapshotRecord:
    manager_type: str
    manager_id: str
    balance: float
    step: int
    n_contacts: int
    n_negotiations: int


@dataclass
class InsurancePolicy:
    premium: float
    contract: Contract
    at_time: float
    against: SCMLAgent


class SCMLAWI(AgentWorldInterface):

    def evaluate_insurance(self, contract: Contract, t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breachs committed by others

        Args:

            contract: hypothetical contract
            t: time at which the policy is to be bought. If None, it means current step
        """
        self._world: SCMLWorld
        self.agent: SCMLAgent
        return self._world.evaluate_insurance(contract=contract, agent=self.agent, t=t)

    def register_cfp(self, cfp: CFP) -> None:
        """Registers a CFP"""
        self._world: SCMLWorld
        self.agent: SCMLAgent
        self._world.n_new_cfps += 1
        cfp.money_resolution = self._world.money_resolution
        self._world.bulletin_board.record(section='cfps', key=cfp.id, value=cfp)

    def remove_cfp(self, cfp: CFP) -> bool:
        """Removes a CFP"""
        self._world: SCMLWorld
        self.agent: SCMLAgent
        if self.agent.id != cfp.publisher:
            return False
        return self._world.bulletin_board.remove(section='cfps', key=str(hash(cfp)))

    def buy_insurance(self, contract: Contract) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        self._world: SCMLWorld
        self.agent: SCMLAgent
        return self._world.buy_insurance(contract=contract, agent=self.agent)

    @property
    def products(self):
        """Products in the world"""
        return self._world.products

    @property
    def processes(self):
        """Processes in the world"""
        return self._world.processes


class SCMLWorld(World):
    """The `World` class running a simulation of supply chain management."""

    def __init__(self
                 , products: Collection[Product]
                 , processes: Collection[Process]
                 , factories: List[Factory]
                 , consumers: List[Consumer]
                 , miners: List[Miner]
                 , factory_managers: Optional[List[FactoryManager]] = None
                 , initial_wallet_balances=1e6
                 , n_steps=10000
                 , time_limit=60 * 60
                 , negotiation_speed=None
                 , neg_n_steps=100
                 , neg_time_limit=3 * 60
                 , minimum_balance=0
                 , interest_rate=None
                 , interest_max=None
                 , max_allowed_breach_level=None
                 , catalog_profit=0.15
                 , avg_process_cost_is_public=True
                 , catalog_prices_are_public=True
                 , breach_processing=BreachProcessing.VICTIM_THEN_PERPETRATOR
                 , breach_penalty_society=1.0
                 , breach_penalty_society_min=2.0
                 , breach_penalty_victim=0.0
                 , breach_move_max_product=True
                 , premium=0.1
                 , money_resolution=0.5
                 , premium_time_increment=0.1
                 , premium_breach_increment=0.1
                 , default_signing_delay=0
                 , transportation_delay: int = 1
                 , loan_installments=1
                 , start_negotiations_immediately=False
                 , log_file_name=None
                 , name: str = None):
        """

        Args:
            products:
            processes:
            factories:
            consumers:
            miners:
            factory_managers:
            initial_wallet_balances:
            n_steps:
            time_limit:
            negotiation_speed:
            neg_n_steps:
            neg_time_limit:
            minimum_balance:
            interest_rate:
            interest_max:
            max_allowed_breach_level:
            catalog_profit:
            avg_process_cost_is_public:
            catalog_prices_are_public:
            breach_processing:
            breach_penalty_society:
            breach_penalty_society_min:
            breach_penalty_victim:
            breach_move_max_product:
            premium:
            money_resolution:
            premium_time_increment:
            premium_breach_increment:
            default_signing_delay:
            transportation_delay:
            loan_installments:
            log_file_name:
            name:
        """
        super().__init__(bulletin_board=None
                         , n_steps=n_steps, time_limit=time_limit
                         , negotiation_speed=negotiation_speed
                         , neg_n_steps=neg_n_steps, neg_time_limit=neg_time_limit, breach_processing=breach_processing
                         , log_file_name=log_file_name, awi_type='negmas.apps.scml.SCMLAWI'
                         , default_signing_delay=default_signing_delay
                         , start_negotiations_immediately=start_negotiations_immediately
                         , name=name)
        global g_products, g_processes
        self.contracts: Dict[int, Set[Contract]] = defaultdict(set)
        self.interest_max = interest_max
        self.premium_time_increment = premium_time_increment
        self.premium_breach_increment = premium_breach_increment
        self.premium = premium
        self.bulletin_board.register_listener(event_type='new_record', listener=self)
        self.bulletin_board.register_listener(event_type='will_remove_record', listener=self)
        self.bulletin_board.add_section("cfps")
        self.bulletin_board.add_section("products")
        self.bulletin_board.add_section("processes")
        self.bulletin_board.add_section("raw_materials")
        self.bulletin_board.add_section("final_products")
        self.minimum_balance = minimum_balance
        self.transportation_delay = transportation_delay
        self.breach_penalty_society = breach_penalty_society
        self.breach_move_max_product = breach_move_max_product
        self.breach_penalty_society_min = breach_penalty_society_min
        self.interest_rate = interest_rate
        self.penalties = 0.0
        self.max_allowed_breach_level = max_allowed_breach_level
        self.catalog_profit = catalog_profit
        self.loan_installments = loan_installments
        self.breach_penalty_victim = breach_penalty_victim
        self.bulletin_board.record(section='settings', key='breach_penalty_society', value=breach_penalty_society)
        self.bulletin_board.record(section='settings', key='breach_penalty_victim', value=breach_penalty_victim)
        self.bulletin_board.record(section='settings', key='immediate_negotiations'
                                   , value=start_negotiations_immediately)
        self.bulletin_board.record(section='settings', key='negotiation_speed_multiple'
                                   , value=negotiation_speed)
        self.bulletin_board.record(section='settings', key='negotiation_n_steps', value=neg_n_steps)
        self.bulletin_board.record(section='settings', key='transportation_delay', value=transportation_delay)
        self.avg_process_cost_is_public = avg_process_cost_is_public
        self.catalog_prices_are_public = catalog_prices_are_public
        self.initial_wallet_balances = initial_wallet_balances
        self.factories: List[Factory] = []
        self.products: List[Product] = []
        self.processes: List[Process] = []
        self.factory_managers: List[FactoryManager] = []
        self.miners: List[Miner] = []
        self.consumers: List[Consumer] = []
        self.set_products(products)
        self.set_processes(processes)
        self.set_factories(factories)
        self.set_miners(miners)
        self.set_consumers(consumers)
        self.set_factory_managers(factory_managers)
        self.money_resolution = money_resolution
        # self._remove_processes_not_used_by_factories()
        # self._remove_products_not_used_by_processes()
        if catalog_prices_are_public or avg_process_cost_is_public:
            self._update_dynamic_product_process_info()

        for factory, manager in zip(self.factories, self.factory_managers):
            factory.state.wallet = initial_wallet_balances
            factory.manager = manager
            manager.factory = factory

        for agent in itertools.chain(self.miners, self.consumers, self.factory_managers):  # type: ignore
            agent.init()

        self.loans: Dict[SCMLAgent, List[Loan]] = defaultdict(list)
        self.insured_contracts: Dict[Tuple[Contract, SCMLAgent], InsurancePolicy] = dict()
        self.n_new_cfps = 0
        self._transport: Dict[int, List[Tuple[SCMLAgent, int, int]]] = defaultdict(list)
        # self.standing_jobs: Dict[int, List[Tuple[Factory, Job]]] = defaultdict(list)
        
        for product in self.products:
            g_products[product.id] = product
        for process in self.processes:
            g_processes[process.id] = process

    def join(self, x: 'Agent', simulation_priority: int = 0):
        """Add an agent to the world.

        Args:
            x: The agent to be registered
            simulation_priority: The simulation priority. Entities with lower priorities will be stepped first during

        Returns:

        """
        global g_agents
        super().join(x=x, simulation_priority=simulation_priority)
        g_agents[x.id] = x

    def set_factories(self, factories):
        self.factories = factories
        for factory in factories:
            factory.init_schedule(n_steps=self.n_steps, initial_balance=self.initial_wallet_balances)

    @classmethod
    def random_small(cls, n_production_levels: int = 1, n_factories: int = 10, factory_kwargs: Dict[str, Any] = None
                     , miner_kwargs: Dict[str, Any] = None, consumer_kwargs: Dict[str, Any] = None
                     , **kwargs):
        return cls.random(n_raw_materials=3
                          , raw_material_price=(1, 3), n_final_products=3
                          , n_production_levels=n_production_levels, n_products_per_level=4
                          , n_processes_per_level=2, n_inputs_per_process=(1, 2)
                          , bias_toward_last_level_products=1.0, quantity_per_input=1
                          , quantity_per_output=1, process_relative_cost=0.15
                          , n_outputs_per_process=1, n_lines=2, lines_are_similar=True
                          , n_processes_per_line=None, cost_for_line=(1.0, 5.0)
                          , n_production_steps=(1, 5), n_factories=n_factories
                          , n_consumers=1, n_products_per_consumer=None, n_miners=1
                          , n_products_per_miner=None, factory_kwargs=factory_kwargs
                          , miner_kwargs=miner_kwargs, consumer_kwargs=consumer_kwargs, **kwargs)

    @classmethod
    def single_path_world(cls, n_intermediate_levels=0, n_miners=1, n_factories_per_level=1, n_consumers=1, n_steps=10
                          , n_lines_per_factory=1
                          , log_file_name: str = None, negotiator_type: str = 'negmas.sao.AspirationNegotiator'
                          , max_storage=None
                          , factory_kwargs: Dict[str, Any] = None, miner_kwargs: Dict[str, Any] = None, consumption=1
                          , consumer_kwargs: Dict[str, Any] = None
                          , negotiation_speed: Optional[int] = None
                          , **kwargs):
        """
        Creates a very small world in which only one raw material and one final product. The production graph is a
        series with `n_intermediate_levels` intermediate levels between the single raw material and single final product

        Args:

            consumption:
            n_intermediate_levels: The number of intermediate products
            n_miners: number of miners of the single raw material
            n_factories_per_level: number of factories at every production level
            n_consumers: number of consumers of the final product
            n_steps: number of simulation steps
            n_lines_per_factory: number of lines in each factory
            log_file_name: File name to store the logs
            negotiator_type: The negotiation factory used to create all negotiators
            max_storage: maximum storage capacity for all factory negmas If None then it is unlimited
            factory_kwargs: keyword arguments to be used for constructing factory negmas
            consumer_kwargs: keyword arguments to be used for constructing consumers
            miner_kwargs: keyword arguments to be used for constructing miners
            negotiation_speed: The number of negotiation steps per simulation step. None means infinite
            kwargs: Any other parameters are just passed to the world constructor

        Returns:
            SCMLWorld ready to run

        Remarks:

            - Every production level n has one process only that takes n steps to complete


        """
        if factory_kwargs is None:
            factory_kwargs = {}
        if consumer_kwargs is None:
            consumer_kwargs = {}
        if miner_kwargs is None:
            miner_kwargs = {}
        if negotiator_type is not None:
            for args in (factory_kwargs, consumer_kwargs, miner_kwargs):
                if 'negotiator_type' not in args.keys():
                    args['negotiator_type'] = negotiator_type

        products = [Product(id=0, name='p0', catalog_price=1.0, production_level=0, expires_in=0)]
        processes = []
        miners = [Miner(profiles={products[-1].id: MiningProfile(cv=0)}, name=f'm_{i}', **miner_kwargs)
                  for i in range(n_miners)]
        factories, negmas = [], []
        n_steps_profile = 1

        def _s(x):
            return x if x is not None else 0

        for level in range(n_intermediate_levels + 1):
            new_product = Product(name=f'p{level + 1}', catalog_price=_s(products[-1].catalog_price) + level + 1
                                  , production_level=level + 1, id=level + 1, expires_in=0)
            p = Process(name=f'p{level + 1}', inputs={InputOutput(product=level, quantity=1, step=0.0)}
                        , production_level=level + 1
                        , outputs={InputOutput(product=level + 1, quantity=1, step=1.0)}, historical_cost=level + 1
                        , id=level)
            processes.append(p)
            products.append(new_product)

        for level in range(n_intermediate_levels + 1):
            for j in range(n_factories_per_level):
                lines = []
                for k in range(n_lines_per_factory):
                    lines.append(Line(id=f'l{level + 1}_{j}_{k}'
                              , profiles={level: ManufacturingProfile(n_steps=n_steps_profile, cost=j + 1
                                                                      , initial_pause_cost=0
                                                                      , running_pause_cost=0
                                                                      , resumption_cost=0
                                                                      , cancellation_cost=0)}
                              , processes=processes))
                factory = Factory(name=f'f{level + 1}_{j}', max_storage=max_storage
                                  , lines=dict(zip((_.id for _ in lines), lines))
                                  , products=products, processes=processes)
                manager = GreedyFactoryManager(factory=factory, name=f'f{level + 1}_{j}', p_negotiation=1.0
                                               , **factory_kwargs)
                factory.manager = manager
                factories.append(factory)
                negmas.append(manager)

        consumers = [Consumer(profiles={products[-1].id: ConsumptionProfile(cv=0, schedule=consumption)}
                              , name=f'c_{i}', **consumer_kwargs)
                     for i in range(n_consumers)]

        return SCMLWorld(products=products, processes=processes, factories=factories  # type: ignore
                         , consumers=consumers, miners=miners
                         , factory_managers=negmas, initial_wallet_balances=1000, n_steps=n_steps
                         , minimum_balance=-100, interest_rate=0.1, interest_max=0.2, log_file_name=log_file_name
                         , negotiation_speed=negotiation_speed
                         , **kwargs)

    @classmethod
    def random(cls
               , n_raw_materials: Union[int, Tuple[int, int]] = (5, 10)
               , raw_material_price: Union[float, Tuple[float, float]] = (1.0, 30.0)
               , n_final_products: Union[int, Tuple[int, int]] = (3, 5)
               , n_production_levels: Union[int, Tuple[int, int]] = (3, 5)
               , n_products_per_level: Union[int, Tuple[int, int]] = (3, 5)
               , n_processes_per_level: Union[int, Tuple[int, int]] = (6, 10)
               , n_inputs_per_process: Union[int, Tuple[int, int]] = (2, 5)
               , bias_toward_last_level_products: float = 0.0
               , quantity_per_input: Union[int, Tuple[int, int]] = (1, 10)
               , input_step: Union[float, Tuple[float, float]] = 0.0
               , quantity_per_output: Union[int, Tuple[int, int]] = (1, 1)
               , output_step: Union[float, Tuple[float, float]] = 1.0
               , process_relative_cost: Union[float, Tuple[float, float]] = (0.05, 0.4)
               , n_outputs_per_process: Union[int, Tuple[int, int]] = (1, 1)
               , n_lines: Union[int, Tuple[int, int]] = (3, 5)
               , lines_are_similar: bool = False
               , n_processes_per_line: Union[int, Tuple[int, int]] = None
               , cost_for_line: Union[float, Tuple[float, float]] = (5.0, 50.0)
               , n_production_steps: Union[int, Tuple[int, int]] = (2, 10)
               , max_storage: Union[int, Tuple[int, int]] = 2000
               , n_factories: Union[int, Tuple[int, int]] = 20
               , n_consumers: Union[int, Tuple[int, int]] = 5
               , n_products_per_consumer: Union[int, Tuple[int, int]] = None
               , n_miners: Union[int, Tuple[int, int]] = 5
               , n_products_per_miner: Optional[Union[int, Tuple[int, int]]] = None
               , factory_manager_types: Union[Type[FactoryManager], List[Type[FactoryManager]]] = GreedyFactoryManager
               , consumer_types: Union[Type[Consumer], List[Type[Consumer]]] = Consumer
               , miner_types: Union[Type[Miner], List[Type[Miner]]] = Miner
               , negotiator_type='negmas.sao.AspirationNegotiator'
               , initial_wallet_balance=1000
               , factory_kwargs: Dict[str, Any] = None
               , miner_kwargs: Dict[str, Any] = None, consumer_kwargs: Dict[str, Any] = None
               , **kwargs
               ):
        """
        Creates a random SCML scenario with adjustable parameters.

        Args:
            n_raw_materials: Number of raw materials. Can be a value or a range.
            raw_material_price: Catalog prices for raw materials. Can be a value or a range.
            n_final_products: Number of final products. Can be a value or a range.
            n_production_levels: How deep is the production graph (number of intermediate products). Can be a value or
            a range.
            n_products_per_level: How many intermediate products per intermediate level. Can be a value or a range.
            n_processes_per_level: Number of processes in intermediate levels. Can be a value or a range.
            n_inputs_per_process: Number of inputs per process. Can be a value or a range.
            bias_toward_last_level_products: How biased are production processes toward using products from the last
            level below them (0 means not bias, 1 means only sample from this last level). A value between 0 and 1.
            quantity_per_input: How many items are needed for each input to a process. Can be a value or a range.
            input_step: When are inputs consumed during the production process. Can be a value or a range. Default 0
            quantity_per_output: How many items are produced per output. Can be a value or a range.
            output_step: When are outputs created during the production process. Can be a value or a range. Default 1
            process_relative_cost: Intrinsic relative cost of processes [Outputs will be produced
            at a cost of sum(input costs) * (1 + process_relative_cost)]. Can be a value or a range.
            n_outputs_per_process: Number of outputs per process. Can be a value or a range.
            n_lines: Number of lines per factory. Can be a value or a range.
            lines_are_similar: If true then all lins of the same factory will have the same production processes.
            n_processes_per_line: Number of processes that can be run on each line per factory. Can be a value or a
            range.
            cost_for_line: Cost for running a process on a line. Can be a value or a range.
            n_production_steps: Number of production steps per line. Can be a value or a range.
            max_storage: Maximum storage per factory. Can be a value or a range.
            n_factories: Number of factories. Can be a value or a range.
            n_consumers: Number of consumers. Can be a value or a range.
            n_products_per_consumer: Number of products per miner. If None then all final products will be assigned to
            every customer. Can be a value or a range.
            n_miners: Number of miners. Can be a value or a range.
            n_products_per_miner: Number of products per miner. If None then all raw materials will be assigned to every
            miner. Can be a value or a range.
            factory_manager_types: A callable for creating factory negmas for the factories
            consumer_types: A callable for creating `Consumer` objects
            miner_types: A callable for creating `Miner` objects
            negotiator_type: A string that can be `eval`uated to a negotiator.
            initial_wallet_balance: The initial balance of all wallets
            factory_kwargs: keyword arguments to be used for constructing factory negmas
            consumer_kwargs: keyword arguments to be used for constructing consumers
            miner_kwargs: keyword arguments to be used for constructing miners
            **kwargs:

        Returns:

            `SCMLWorld` The random world generated

        Remarks:

            - Most parameters accept either a single value or a 2-valued tuple. In the later case, it will sample a
              value within the range specified by the tuple (low, high) inclusive. For example the number of lines
              (n_lines) follows this pattern

        """
        if factory_kwargs is None:
            factory_kwargs = {}
        if consumer_kwargs is None:
            consumer_kwargs = {}
        if miner_kwargs is None:
            miner_kwargs = {}

        if negotiator_type is not None:
            for args in (factory_kwargs, consumer_kwargs, miner_kwargs):
                if 'negotiator_type' not in args.keys():
                    args['negotiator_type'] = negotiator_type

        def realin(rng: Union[Tuple[float, float], float]) -> float:
            if isinstance(rng, float):
                return rng
            if abs(rng[1] - rng[0]) < 1e-8:
                return rng[0]
            return rng[0] + random() * (rng[1] - rng[0])

        def intin(rng: Union[Tuple[int, int], int]) -> int:
            if isinstance(rng, int):
                return rng
            if rng[0] == rng[1]:
                return rng[0]
            return randint(rng[0], rng[1])

        def _sample_product(products: list, old_products: list, last_level_products: list, k: int):
            if bias_toward_last_level_products < 1e-7:
                return sample(products, k=min(k, len(products)))
            elif bias_toward_last_level_products > 1 - 1e-7:
                return sample(last_level_products, min(k, len(last_level_products)))
            n_old, n_last = len(old_products), len(last_level_products)
            p_old = n_old / (n_old + n_last / bias_toward_last_level_products)
            if random() < p_old:
                return sample(old_products, min(k, len(old_products)))
            else:
                return sample(last_level_products, min(k, len(last_level_products)))

        products = [Product(name=f'r_{ind}'
                            , catalog_price=realin(raw_material_price)
                            , production_level=0, expires_in=0, id=ind)
                    for ind in range(intin(n_raw_materials))]
        raw_materials = products.copy()
        last_level_products = products  # last level of products
        old_products: List[Product] = []  # products not including last level
        processes: List[Process] = []

        def _adjust_level_of_production(new_products, new_processes):
            product_prices: Dict[Product, List[float]] = defaultdict(
                list)  # will keep the costs for generating products
            for process in new_processes:
                process.inputs = set(InputOutput(product=_.index, quantity=intin(quantity_per_input)
                                                 , step=realin(input_step))
                                     for _ in _sample_product(products=products, old_products=old_products
                                                              , last_level_products=last_level_products
                                                              , k=intin(n_inputs_per_process)))
                process.outputs = set(InputOutput(product=_.index, quantity=intin(quantity_per_output)
                                                  , step=realin(output_step))
                                      for _ in sample(new_products, intin(n_outputs_per_process)))
                process.historical_cost = sum(products[_.product].catalog_price * _.quantity
                                              for _ in process.inputs)
                process.historical_cost *= 1 + realin(process_relative_cost)
                for output in process.outputs:
                    product_prices[products[output.product]].append(process.historical_cost)

            new_products = [_ for _ in product_prices.keys()]
            for product in new_products:
                product.catalog_price = sum(product_prices[product]) / len(product_prices[product])
            return new_products, new_processes

        n_levels = intin(n_production_levels)
        if n_levels > 0:
            for level in range(n_levels):
                new_products = [Product(name=f'intermediate_{level}_{ind}', production_level=level + 1
                                        , id=len(products) + ind, catalog_price=0, expires_in=0)
                                for ind in range(intin(n_products_per_level))]
                new_processes = [Process(name=f'process_{level}_{ind}', production_level=level + 1
                                         , id=len(processes) + ind, historical_cost=0, inputs=set(), outputs=set())
                                 for ind in range(intin(n_processes_per_level))]
                new_products, new_processes = _adjust_level_of_production(new_products, new_processes)
                products += new_products
                old_products += last_level_products
                last_level_products = new_products
                processes += new_processes

            final_products = [Product(name=f'f_{ind}', production_level=n_levels + 1
                                      , id=len(products) + ind, catalog_price=0, expires_in=0)
                              for ind in range(intin(n_final_products))]
            new_processes = [Process(name=f'process_final_{ind}', production_level=n_levels + 1
                                     , id=len(processes) + ind, historical_cost=0, inputs=set(), outputs=set())
                             for ind in range(intin(n_processes_per_level))]

            final_products, new_processes = _adjust_level_of_production(final_products, new_processes)
            products += final_products
            processes += new_processes
        else:
            final_products = raw_materials

        if n_processes_per_line is None:
            n_processes_per_line = len(processes)

        n_factories = intin(n_factories)
        factories = []
        for i in range(n_factories):
            if lines_are_similar:
                line_processes = sample(processes, intin(n_processes_per_line))
                profiles = [ManufacturingProfile(n_steps=intin(n_production_steps), cost=realin(cost_for_line)
                                                 , initial_pause_cost=0
                                                 , running_pause_cost=0
                                                 , resumption_cost=0
                                                 , cancellation_cost=0)
                            for _ in line_processes]
                lines = [Line(profiles=dict(zip((_.index for _ in line_processes), profiles)), processes=processes)
                         for _ in range(intin(n_lines))] * intin(n_lines)
            else:
                lines = []
                for _ in range(intin(n_lines)):
                    line_processes = sample(processes, intin(n_processes_per_line))
                    profiles = [ManufacturingProfile(n_steps=intin(n_production_steps), cost=realin(cost_for_line)
                                                     , initial_pause_cost=0
                                                     , running_pause_cost=0
                                                     , resumption_cost=0
                                                     , cancellation_cost=0)
                                for _ in line_processes
                                ]
                    lines.append(Line(profiles=dict(zip((_.index for _ in line_processes), profiles))
                                      , processes=processes))
            factories.append(Factory(lines=dict(zip((_.id for _ in lines), lines)), max_storage=intin(max_storage)
                                     , processes=processes, products=products))

        def _ensure_list(x):
            if isinstance(x, Iterable):
                return list(x)
            else:
                return [x]

        miner_types_list, consumer_types_list, factory_manager_types_list = [_ensure_list(_)
                                                                             for _ in (miner_types, consumer_types
                                                                                       , factory_manager_types)]

        factory_managers = [current(**factory_kwargs)
                            for current in choices(factory_manager_types_list, k=n_factories)]
        miners = [current(**miner_kwargs)
                  for current in choices(miner_types_list, k=intin(n_miners))]
        if n_products_per_miner is None:
            n_products_per_miner = len(raw_materials)
        if n_products_per_consumer is None:
            n_products_per_consumer = len(final_products)
        n_products_per_miner = min(n_products_per_miner, len(raw_materials))
        n_products_per_consumer = min(n_products_per_consumer, len(final_products))
        for miner in miners:
            _n = intin(n_products_per_miner)
            mining_profiles = dict(zip((_.index for _ in sample(raw_materials, _n))
                                       , [MiningProfile.random() for _ in range(_n)]))
            miner.set_profiles(mining_profiles)
        consumers = [current(**consumer_kwargs)
                     for current in choices(consumer_types_list, k=intin(n_consumers))]
        for consumer in consumers:
            _n = intin(n_products_per_consumer)
            consumer_profiles = dict(zip((_.index for _ in sample(final_products, _n))
                                         , [ConsumptionProfile.random() for _ in range(_n)]))
            consumer.set_profiles(consumer_profiles)

        return SCMLWorld(products=products, processes=processes, factories=factories, consumers=consumers,
                         miners=miners, factory_managers=factory_managers
                         , initial_wallet_balances=initial_wallet_balance, **kwargs)

    # def _remove_processes_not_used_by_factories(self):
    #     """Removes all process that no factory can run"""
    #     found: Set[Process] = set()
    #     for factory in self.factories:
    #         for line in factory.lines.values():
    #             found = found.union(set(self.processes[_] for _ in line.profiles.keys()))
    #     self.processes = list(set(self.processes).intersection(found))
    #
    # def _remove_products_not_used_by_processes(self):
    #     """removes products that can neither be produced nor consumed by any process"""
    #     found: Set[Product] = set()
    #     for process in self.processes:
    #         found = found.union(set(self.products[i.product] for i in process.inputs))
    #         found = found.union(set(self.products[i.product] for i in process.outputs))
    #     self.products = list(set(self.products).intersection(found))

    def _update_dynamic_product_process_info(self):
        """Updates the catalog prices of all products based on the prices of their inputs"""

        # for process in self.processes:
        #     for current in process.inputs:
        #         self.products[current.product].consuming_processes.add((process, current.quantity))
        #     for current in process.outputs:
        #         self.products[current.product].generating_processes.add((process, current.quantity))

        # noinspection PyUnusedLocal
        def _or_for_none(a, b):
            return (a is None and b) or a or b

        product_costs: Dict[Product, List[float]] = defaultdict(list)
        process_costs: Dict[Process, List[float]] = defaultdict(list)

        for factory in self.factories:
            for line in factory.lines.values():
                for process_index, profile in line.profiles.items():
                    process = self.processes[process_index]
                    if self.avg_process_cost_is_public:
                        process_costs[process].append(profile.cost)
                    if self.catalog_prices_are_public:
                        input_price = sum(self.products[i.product].catalog_price * i.quantity
                                          for i in process.inputs)
                        for output in process.outputs:
                            if output.quantity == 0:
                                continue
                            product_costs[self.products[output.product]].append(
                                (input_price + profile.cost) / output.quantity)

        if self.catalog_prices_are_public:
            product_costs_avg = {p: sum(v) / len(v) if len(v) > 0 else math.inf for p, v in product_costs.items()}
            for product in self.products:
                if product.catalog_price is None:
                    product.catalog_price = product_costs_avg[product] * (1 + self.catalog_profit)

        if self.avg_process_cost_is_public:
            process_costs_avg = {k: sum(v) / len(v) if len(v) > 0 else math.inf for k, v in process_costs.items()}
            for process in self.processes:
                if process.historical_cost is None:
                    process.historical_cost = process_costs_avg[process]

    def set_consumers(self, consumers: List[Consumer]):
        self.consumers = consumers
        [self.join(f, simulation_priority=1) for f in consumers]

    def set_miners(self, miners: List[Miner]):
        self.miners = miners
        [self.join(f, simulation_priority=3) for f in miners]

    def set_factory_managers(self, factory_managers: Optional[List[FactoryManager]]):
        if factory_managers is None:
            factory_managers = []
        self.factory_managers = factory_managers
        [self.join(f, simulation_priority=2) for f in factory_managers]

    def set_processes(self, processes: Collection[Process]):
        if processes is None:
            self.processes = []
        else:
            self.processes = list(processes)
        for v in self.processes:
            self.bulletin_board.record('processes', key=str(v), value=v)

    def set_products(self, products: Collection[Product]):
        if products is None:
            self.products = []
        else:
            self.products = list(products)
        for v in self.products:
            self.bulletin_board.record('products', key=str(v), value=v)

    def _contract_execution_order(self, contracts: Collection[Contract]):
        def order(x: Contract):
            o = self.products[x.annotation['cfp'].product].production_level
            if o:
                return o
            return 0

        return sorted(contracts, key=order)

    def execute(self, action: Action, agent: 'Agent', callback: Callable[[Action, bool], Any] = None) -> bool:
        if not isinstance(agent, FactoryManager):
            if callback is not None:
                callback(action, False)
            return False
        line_name, process_index = action.params.get('line_name', None), action.params.get('process', None)
        contract = action.params.get('contract', None)
        time = action.params.get('time', None)
        override = action.params.get('override', True)
        if line_name is None or process_index is None or time is None or time < 0 or time > self.n_steps - 1:
            if callback is not None:
                callback(action, False)
            return False
        line = agent.factory.lines.get(line_name, None)
        if line is None:
            if callback is not None:
                callback(action, False)
            return False
        job = Job(command=action.type, process=process_index, line_name=line_name, time=time, contract=contract
                  , updates={})
        # self.standing_jobs[time].append((agent.factory, job))
        agent.factory.schedule_job(job=job, override=override)
        if callback is not None:
            callback(action, True)
        return True

    def state(self, agent: 'Agent') -> dict:
        if isinstance(agent, FactoryManager):
            return agent.factory_state.__dict__
        return {}

    def _update_factory_balance(self, factory: Factory, value: float) -> None:
        if abs(value) < 1e-10:
            return
        factory.state.wallet += value
        if factory.state.wallet < self.minimum_balance:
            if self.interest_rate is not None:
                factory.state.wallet *= (1 + self.interest_rate)

    def _simulation_step(self):
        """A step of SCML simulation"""

        # run standing jobs
        # -----------------
        # for factory, job in self.standing_jobs.get(self.current_step, []):
        #    factory.schedule_job(job=job, end=self.n_steps)

        # apply interests and pay loans
        # -----------------------------
        for agent, loans in self.loans.items():
            for l in loans:
                loan: Loan = l
                if loan.n_installments <= 0:
                    continue
                wallet = agent.factory_state.wallet
                payment = 0
                if wallet > 0:
                    payment = min(loan.installment, wallet)
                    loan.amount -= payment
                    agent.factory_state.wallet -= payment
                if payment < loan.installment:
                    unpaid = loan.installment - payment
                    penalty = unpaid * ((1 + loan.interest) ** loan.n_installments)
                    loan.total += penalty
                    loan.installment += penalty / loan.n_installments
                    agent.factory_state.loans += penalty

        # run factories
        # -------------
        for factory in self.factories:
            failures = factory.step(t=self.current_step)
            if len(failures) > 0:
                self.logdebug(f'{factory.manager.name}\'s failed @ {[(_.line, _.command) for _ in failures]}')
                factory.manager.on_production_failure(failures)

        # finish transportation
        # ---------------------
        transports = self._transport.get(self.current_step, [])
        for transport in transports:
            manager, product_id, q = transport
            manager.factory_state.storage[product_id] += q

        # remove expired CFPs
        # -------------------
        cfps = self.bulletin_board.query(section='cfps', query=None)
        if cfps is not None:
            # new_cfps = dict()
            toremove = []
            for key, cfp in cfps.items():
                # we remove CFP with a max_time less than *or equal* to current step as all processing for current step
                # should already be complete by now
                if cfp.max_time <= self.current_step:
                    toremove.append(key)
                # new_cfps[key] = cfp
            for key in toremove:
                self.bulletin_board.remove(section='cfps', query=key, query_keys=True)
            # noinspection PyProtectedMember
            # self.bulletin_board._data['cfps'] = new_cfps

    def _pre_step_stats(self):
        # noinspection PyProtectedMember
        cfps = self.bulletin_board._data['cfps']
        self._stats['n_cfps_on_board_before'].append(len(cfps) if cfps else 0)
        pass

    def _post_step_stats(self):
        """Saves relevant stats"""
        self._stats['n_cfps'].append(self.n_new_cfps)
        self.n_new_cfps = 0
        # noinspection PyProtectedMember
        cfps = self.bulletin_board._data['cfps']
        self._stats['n_cfps_on_board_after'].append(len(cfps) if cfps else 0)
        market_size = 0
        for a in itertools.chain(self.miners, self.consumers, self.factory_managers):
            self._stats[f'balance_{a.name}'].append(a.factory_state.balance)
            self._stats[f'storage_{a.name}'].append(sum(a.factory_state.storage.values()))
            market_size += a.factory_state.balance
        self._stats['market_size'].append(market_size)

    # noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
    def _execute_contract(self, contract: Contract) -> Set[Breach]:
        super()._execute_contract(contract=contract)
        partners: Set[SCMLAgent]
        cfp: CFP
        partners, agreement = set(self.agents[_] for _ in contract.partners), contract.agreement
        cfp = contract.annotation['cfp']  # type: ignore
        breaches = set()
        quantity, unit_price = agreement['quantity'], agreement['unit_price']
        penalty_victim = agreement.get('penalty', None)
        if penalty_victim is not None and self.breach_penalty_victim is not None:
            # there is a defined penalty, find its value
            penalty_victim = (penalty_victim if penalty_victim is not None else 0.0) + \
                             (self.breach_penalty_victim if self.breach_penalty_victim is not None else 0)
        penalty_society = self.breach_penalty_society

        # ask each partner to confirm the execution
        for partner in partners:
            if not partner.confirm_contract_execution(contract=contract):
                self.logdebug(
                    f'{partner.name} refused execution og Contract {contract.id}')
                breaches.add(Breach(contract=contract, perpetrator=partner  # type: ignore
                                    , victims=partners - {partner}
                                    , level=1.0, type='refusal'))
        if len(breaches) > 0:
            return breaches

        # all partners agreed to execute the agreement -> execute it.
        pind = cfp.product
        buyer_id, seller_id = contract.annotation['buyer'], contract.annotation['seller']
        seller: SCMLAgent
        buyer: SCMLAgent
        buyer, seller = self.agents[buyer_id], self.agents[seller_id]
        product_breach = penalty_breach = money_breach = None
        money = unit_price * quantity

        # check the seller
        available_quantity = seller.factory_state.storage.get(pind, 0) if not isinstance(seller, Miner) else quantity
        missing_quantity = max(0, quantity - available_quantity)
        penalty_value, payable, paid_for_quantity = 0.0, 0.0, 0
        if missing_quantity > 0:
            product_breach = missing_quantity / quantity
            for penalty, is_victim in ((penalty_victim, True), (penalty_society, False)):
                if penalty is not None:
                    # if a penalty is defined in the contract try to take it from the seller
                    penalty_value = penalty * product_breach
                    if not is_victim:
                        penalty_value = max(penalty_value,
                                            self.breach_penalty_society_min
                                            if self.breach_penalty_society_min is not None else 0)
                    seller_balance = seller.factory_state.wallet
                    # if the seller needs more than her wallet to pay the penalty, try a loan
                    if seller_balance < penalty_value:
                        self.buy_loan(agent=seller, amount=penalty_value - seller_balance
                                      , n_installments=self.loan_installments)

                    # if the seller can pay the penalty then pay it and now there is no breach, otherwise pay as much as
                    # possible
                    missing_quantity_unpaid_for = 0
                    if seller.factory_state.wallet >= penalty_value:
                        payable = penalty_value
                        if is_victim:
                            missing_quantity_unpaid_for = 0
                    else:
                        if unit_price == 0:
                            paid_for_quantity = missing_quantity
                            payable = unit_price * paid_for_quantity
                        else:
                            paid_for_quantity = int(seller.factory_state.wallet / unit_price)
                            payable = unit_price * paid_for_quantity
                        if is_victim:
                            missing_quantity_unpaid_for = missing_quantity - paid_for_quantity
                    seller.factory_state.wallet -= payable
                    if is_victim:
                        buyer.factory_state.wallet += payable
                        # if we agreed on a penalty and the buyer paid it, then clear the product breach
                        if missing_quantity_unpaid_for <= 0:
                            product_breach = None
                        else:
                            missing_quantity = missing_quantity_unpaid_for
                            product_breach = missing_quantity / quantity
                    else:
                        # if this is the society penalty, it does not affect the product_breach
                        self.penalties += payable

        # check the seller
        available_money = buyer.factory_state.wallet if not isinstance(buyer, Consumer) else money
        missing_money = max(0.0, money - available_money)
        if missing_money > 0.0:
            # if the buyer cannot pay, then offer him a loan
            self.buy_loan(agent=buyer, amount=money - available_money, n_installments=self.loan_installments)
            available_money = buyer.factory_state.wallet
            missing_money = max(0.0, money - available_money)
        if missing_money > 0.0:
            money_breach = missing_money / money

        # apply insurances if they exist
        if penalty_breach is not None:
            breaches.add(Breach(contract=contract, perpetrator=seller, victims={buyer}
                                , level=penalty_breach, type='penalty', step=self.current_step))
        if product_breach is not None:
            if (contract, seller) in self.insured_contracts.keys():
                seller.factory_state.storage[pind] = seller.factory_state.storage.get(pind, 0) + missing_quantity
                del self.insured_contracts[(contract, seller)]
            breaches.add(Breach(contract=contract, perpetrator=seller, victims={buyer}
                                , level=product_breach, type='product', step=self.current_step))
        if money_breach is not None:
            if (contract, buyer) in self.insured_contracts.keys():
                buyer.factory_state.wallet += missing_money
                del self.insured_contracts[(contract, buyer)]
            breaches.add(Breach(contract=contract, perpetrator=buyer, victims={seller}
                                , level=money_breach, type='money', step=self.current_step))

        if len(breaches) > 0:
            self.logdebug(f'Contract {contract.id} has {len(breaches)} breaches:')
            for breach in breaches:
                self.logdebug(f'{breach}')
        self._move_product(buyer=buyer, seller=seller, quantity=quantity, money=money, product_id=pind)
        return breaches

    def _move_product(self, buyer: SCMLAgent, seller: SCMLAgent, product_id: int, quantity: int, money: float):
        """Moves as much product and money between the buyer and seller"""
        if isinstance(seller, Miner):
            available_quantity = quantity
        else:
            available_quantity = min(quantity, seller.factory_state.storage.get(product_id, 0))
        if isinstance(buyer, Consumer):
            available_money = money
        else:
            available_money = min(money, buyer.factory_state.wallet)
        self.logdebug(f'Moving {quantity} (available {available_quantity}) units of '
                      f'{self.products[product_id].name} from {seller.name} '
                      f'to {buyer.name} for {money} (available {available_money}) dollars')
        if available_quantity > 0:
            seller.factory_state.storage[product_id] -= available_quantity
            if self.transportation_delay < 1:
                buyer.factory_state.storage[product_id] += available_quantity
            else:
                self._transport[self.current_step + self.transportation_delay].append(
                    (buyer, product_id, available_quantity))
        if available_money > 0:
            buyer.factory_state.wallet -= available_money
            seller.factory_state.wallet += available_money

    def _process_breach(self, breach: Breach) -> bool:
        if super()._process_breach(breach=breach):
            return True
        breach_resolved = False

        if self.breach_processing == BreachProcessing.META_NEGOTIATION:
            raise NotImplementedError('Meta negotiations are still not implemented')
        elif self.breach_processing == BreachProcessing.VICTIM_THEN_PERPETRATOR:
            # noinspection PyUnusedLocal
            perpetrator: SCMLAgent
            # noinspection PyUnusedLocal
            victim: SCMLAgent
            # noinspection PyUnusedLocal
            victims: Set[SCMLAgent]
            contract, perpetrator = breach.contract, breach.perpetrator  # type: ignore
            partners = set(self.agents[_] for _ in contract.partners)
            victims = partners - {perpetrator}  # type: ignore

            for victim in victims:
                request = victim.on_breach_by_another(contract=contract, partner=perpetrator.id)
                if request is not None and request.annotation is not None:
                    responses = []
                    # noinspection PyUnusedLocal
                    partner: SCMLAgent
                    for partner in partners - {victim}:  # type: ignore
                        responses.append(partner.on_renegotiation_request(contract=contract
                                                                          , partner=victim.id
                                                                          , cfp=request.annotation['cfp']))
                    if not all(responses):
                        continue
                    if not breach_resolved:
                        partner_names = [_.id for _ in breach.victims.union({breach.perpetrator})]
                        breach_resolved = self.run_negotiation(caller=victim
                                                               , issues=request.issues
                                                               , partners=partner_names
                                                               , roles=None
                                                               , annotation=request.annotation
                                                               , mechanism_name=None
                                                               , mechanism_params=None) is not None

            request = perpetrator.on_breach_by_self(contract=contract, partner=victim.id)
            if request is not None:
                responses = []
                for partner in partners - {perpetrator}:  # type: ignore
                    responses.append(partner.on_renegotiation_request(contract=contract
                                                                      , partner=perpetrator.id
                                                                      , cfp=request.annotation['cfp']))
                if all(responses):
                    if not breach_resolved:
                        breach_resolved = self.run_negotiation(caller=perpetrator, issues=request.issues
                                                               , partners=breach.victims.union({breach.perpetrator})
                                                               , roles=None
                                                               , annotation=request.annotation
                                                               , mechanism_name=None
                                                               , mechanism_params=None) is not None

        if breach_resolved:
            return True
        self._register_breach(breach)
        return False

    def run_negotiation(self, caller: "Agent"
                        , issues: Collection[Issue]
                        , partners: Collection["Agent"]
                        , roles: Collection[str] = None
                        , annotation: Optional[Dict[str, Any]] = None
                        , mechanism_name: str = None
                        , mechanism_params: Dict[str, Any] = None):
        if annotation is None:
            annotation = {}
        cfp: CFP = annotation['cfp']
        annotation['partners'] = [_.id for _ in partners]
        if cfp.is_buy:
            annotation['buyer'] = [p for p in partners if p != caller][0].id
            annotation['seller'] = caller.id
        else:
            annotation['seller'] = [p for p in partners if p != caller][0].id
            annotation['buyer'] = caller.id
        return super().run_negotiation(caller=caller, issues=issues, annotation=annotation
                                       , partners=partners, roles=roles
                                       , mechanism_name=mechanism_name, mechanism_params=mechanism_params)

    def request_negotiation(self, req_id: str
                            , caller: "Agent"
                            , issues: List[Issue]
                            , partners: List["Agent"]
                            , roles: List[str] = None
                            , annotation: Optional[Dict[str, Any]] = None
                            , mechanism_name: str = None
                            , mechanism_params: Dict[str, Any] = None):
        if annotation is None:
            annotation = {}
        cfp: CFP = annotation['cfp']
        annotation['partners'] = [_.id for _ in partners]
        if cfp.is_buy:
            annotation['buyer'] = [p for p in partners if p != caller][0].id
            annotation['seller'] = caller.id
        else:
            annotation['seller'] = [p for p in partners if p != caller][0].id
            annotation['buyer'] = caller.id
        return super().request_negotiation(req_id=req_id, caller=caller, issues=issues, annotation=annotation
                                           , partners=partners, roles=roles, mechanism_name=mechanism_name
                                           , mechanism_params=mechanism_params)

    def evaluate_insurance(self, contract: Contract, agent: SCMLAgent, t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breachs committed by others

        Args:

            contract: hypothetical contract
            agent: The `SCMLAgent` I am ensuring against
            t: time at which the policy is to be bought. If None, it means current step
        """
        if self.premium is None or contract.signed_at is None:
            return None
        if t is None:
            t = self.current_step
        dt = t - contract.signed_at
        if dt > 0:
            return None
        other = [self.agents[_] for _ in contract.partners if _ != agent.id]
        if len(other) != 1:
            return None
        other = other[0]
        breaches = self.bulletin_board.query(section='breaches', query={'perpetrator': other})
        b = 0
        if breaches is not None:
            for _, breach in breaches.items():
                b += breach.level
        return (self.premium + b * self.premium_breach_increment) * (1 + self.premium_time_increment * dt)

    def buy_insurance(self, contract: Contract, agent: SCMLAgent) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        premium = self.evaluate_insurance(contract=contract, t=self.current_step, agent=agent)
        if premium is None or agent.factory_state.wallet < premium:
            return False
        other = [self.agents[_] for _ in contract.partners if _ != agent.id]
        if len(other) != 1:
            return False
        other = other[0]
        agent.factory_state.wallet -= premium
        if not isinstance(other, SCMLAgent):
            raise ValueError('The partner must be an SCML agent')
        self.insured_contracts[(contract, other)] = InsurancePolicy(contract=contract, at_time=self.current_step
                                                                    , against=other, premium=premium)
        return True

    def evaluate_loan(self, agent: SCMLAgent, amount: float, n_installments: int) -> Optional[Loan]:
        """Evaluates the interest that will be imposed on the agent to buy_loan that amount"""
        balance = agent.factory_state.wallet
        if self.minimum_balance is not None and agent.factory_state.balance - amount < - self.minimum_balance:
            return None

        if self.interest_rate is None:
            return None
        interest = self.interest_rate
        if balance < 0 and self.interest_max is not None:
            interest *= (1 + (self.interest_max - 1) / (-balance * interest))
        total = amount * (1 + interest) ** n_installments
        installment = total / n_installments
        return Loan(amount=amount, total=total, interest=interest
                    , n_installments=n_installments, installment=installment)

    def buy_loan(self, agent: SCMLAgent, amount: float, n_installments: int):
        """Gives a loan of amount to agent at the interest calculated using `evaluate_loan`"""

        loan = self.evaluate_loan(amount=amount, agent=agent, n_installments=n_installments)
        if loan is not None:
            if agent.confirm_loan(loan=loan):
                self.loans[agent].append(loan)
                agent.factory_state.wallet += loan.amount
                agent.factory_state.loans += loan.total

    @property
    def winners(self):
        """The winners of this world (factory maangers with maximum wallet balance"""
        if len(self.factory_managers) < 1:
            return []
        balances = sorted(((_.factory_state.balance, _) for _ in self.factory_managers), key=lambda x: x[0]
                          , reverse=True)
        max_balance = balances[0][0]
        return [_[1] for _ in balances if _[0] >= max_balance]

    def on_event(self, event: Event, sender: 'EventSource') -> None:
        """
        Called whenever an event is raised for which the `World` is registered asa listener

        Args:
            event: The event
            sender: The sender

        Returns:
            None
        """
        if event.type == 'new_record' and event.data['section'] == 'cfps':
            cfp = event.data['value']
            for m in itertools.chain(self.miners, self.factory_managers, self.consumers):  # type: ignore
                if m.id != cfp.publisher and cfp.product in m.interesting_products:
                    m.on_new_cfp(cfp)
        elif event.type == 'will_remove_record' and event.data['section'] == 'cfps':
            for m in itertools.chain(self.miners, self.factory_managers, self.consumers):  # type: ignore
                cfp = event.data['value']
                if m.id != cfp.publisher and cfp.product in m.interesting_products:
                    m.on_remove_cfp(cfp)

    def _contract_record(self, contract: Contract) -> Dict[str, Any]:
        c = {
            'seller_name': self.agents[contract.annotation['seller']].name,
            'buyer_name': self.agents[contract.annotation['buyer']].name,
            'product_name': g_products[contract.annotation['cfp'].product],
            'delivery_time': contract.agreement['time'],
            'quantity': contract.agreement['quantity'],
            'unit_price': contract.agreement['unit_price'],
            'signed_at': contract.signed_at if contract.signed_at is not None else -1,
            'concluded_at': contract.concluded_at,
            'penalty': contract.agreement.get('penalty', np.nan),
            'signing_delay': contract.agreement.get('signing_delay', 0),
            'signatures': '|'.join(str(_) for _ in contract.signatures),
            'issues': contract.issues,
            'seller': contract.annotation['seller'],
            'buyer': contract.annotation['buyer'],
        }
        c.update(contract.annotation)
        c['n_neg_steps'] = contract.mechanism_state.step
        return c

    def _breach_record(self, breach: Breach) -> Dict[str, Any]:
        return {
            'perpetrator': breach.perpetrator.id,
            'perpetrator_name': breach.perpetrator.name,
            'level': breach.level,
            'type': breach.type,
            'time': breach.step
        }

    def on_contract_signed(self, contract: Contract):
        super().on_contract_signed(contract=contract)
        self.contracts[contract.agreement['time']].add(contract)

    def _get_executable_contracts(self) -> Collection[Contract]:
        """Called at every time-step to get the contracts that are `executable` at this point of the simulation"""
        return self.contracts.get(self.current_step, [])

    def _delete_executed_contracts(self) -> None:
        del self.contracts[self.current_step]

    def _contract_finalization_time(self, contract: Contract) -> int:
        """
        Returns the time at which the given contract will complete execution
        Args:
            contract:

        Returns:

        """
        return contract.agreement['time'] + self.transportation_delay

    def _contract_executation_time(self, contract: Contract) -> int:
        """
        Returns the time at which the given contract will start execution
        Args:
            contract:

        Returns:

        """
        return contract.agreement['time']

    def _contract_size(self, contract: Contract) -> float:
        return contract.agreement['unit_price'] * contract.agreement['quantity']
