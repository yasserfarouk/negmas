from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, List, Optional

import numpy as np
from dataclasses import dataclass, field

from .common import ManufacturingProfile, Job, Factory

INVALID_STEP = -1000
NO_PRODUCTION = -1

__all__ = [
    'FactorySimulator',
    'SlowFactorySimulator'
]


class FactorySimulator(ABC):
    """Simulates a factory allowing for prediction of storage/balance in the future"""

    def __init__(self, initial_wallet: float, initial_storage: Dict[int, int], n_steps: int, n_products: int
                 , profiles: List[ManufacturingProfile], max_storage: Optional[int] = None):
        self._n_steps = n_steps
        self._max_storage = max_storage
        self._initial_wallet = initial_wallet
        self._initial_storage = np.zeros(n_products)
        for k, v in initial_storage.items():
            self._initial_storage[k] = v
        self._profiles = profiles
        self._n_products = n_products
        self._reserved_storage = np.zeros(shape=(n_products, n_steps))

    # -----------------
    # FIXED PROPERTIES
    # -----------------

    @property
    def max_storage(self) -> Optional[int]:
        return self._max_storage

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def initial_wallet(self) -> float:
        return self._initial_wallet

    @property
    def initial_storage(self) -> np.array:
        return self._initial_storage

    @property
    @abstractmethod
    def n_lines(self):
        """Number of lines"""

    @property
    @abstractmethod
    def final_balance(self) -> float:
        """Final balance given everything scheduled so-far"""

    # -------------------------------
    # DYNAMIC PROPERTIES (READ STATE)
    # -------------------------------

    @abstractmethod
    def wallet_to(self, t: int) -> np.array:
        """
        Returns the cash in wallet at time t
        Args:
            t: 

        Returns:

        """

    def wallet(self, t: int) -> float:
        return self.wallet_to(t)[-1]

    @abstractmethod
    def storage_to(self, t: int) -> np.array:
        """
        Returns the storage at time t
        
        Args:
            t: 

        Returns:

        """

    def storage(self, t: int) -> np.array:
        return self.storage_to(t)[:, -1]

    @abstractmethod
    def line_schedules_to(self, t: int) -> np.array:
        """
        Returns the schedule of each line

        Args:
            t:

        Returns:
            - A `NO_PRODUCTION` value means no production, otherwise the index of the process being run
        """

    def line_schedules(self, t: int) -> np.array:
        return self.line_schedules_to(t)[:, -1]

    def total_storage_to(self, t: int) -> np.array:
        return self.storage_to(t).sum(axis=0)

    def total_storage(self, t: int) -> int:
        return self.total_storage_to(t)[-1]

    def reserved_storage_to(self, t: int) -> np.array:
        return self._reserved_storage[:, :t+1]

    def reserved_storage(self, t: int) -> np.array:
        return self._reserved_storage[:, t]

    def available_storage(self, t: int) -> np.array:
        return self.storage(t) - self.reserved_storage(t)

    def available_storage_to(self, t: int) -> np.array:
        return self.storage_to(t) - self.reserved_storage_to(t)

    @abstractmethod
    def loans_to(self, t: int) -> np.array:
        """
        Returns loans up to time t
        
        Args:
            t: 

        Returns:

        """

    def loans(self, t: int) -> float:
        """
        Returns loans at time t
        Args:
            t:

        Returns:

        """
        return self.loans_to(t)[-1]

    def balance(self, t: int) -> float:
        """
        Returns the balance fo the factory at time t
        Args:
            t: 

        Returns:

        """
        return self.wallet(t) - self.loans(t)

    # -------------------------
    # OPERATIONS (UPDATE STATE)
    # -------------------------

    @abstractmethod
    def add_loan(self, total: float, t: int) -> bool:
        """
        Adds a loan at the given time
        Args:
            total: 
            t: 

        Returns:

        """

    def receive(self, payment: float, t: int) -> bool:
        """
        Simulates receiving payment at time t
        Args:
            payment: 
            t: 

        Returns:

        """
        return self.pay(-payment, t)

    @abstractmethod
    def pay(self, payment: float, t: int) -> bool:
        """
        Simulate payment at time t
        
        Args:
            payment: 
            t: 

        Returns:

        """

    @abstractmethod
    def transport_to(self, inventory: Dict[int, int], t: int) -> bool:
        """
        Simulates transporting products to/from storage at time t
        Args:
            inventory: 
            t: 

        Returns:

        """

    @abstractmethod
    def buy(self, product: int, quantity: int, price: int, t: int) -> bool:
        """
        Buy a given quantity of a product for a given price at some time t
        Args:
            product:
            quantity:
            price:
            t: time

        Returns:

        """

    @abstractmethod
    def sell(self, product: int, quantity: int, price: int, t: int) -> bool:
        """
        sell a given quantity of a product for a given price at some time t
        Args:
            product:
            quantity:
            price:
            t:

        Returns:

        """

    @abstractmethod
    def schedule(self, job: Job, override=True) -> bool:
        """
        Simulates scheduling the given job at its `time` and `line` optionally overriding whatever was already scheduled
        Args:
            job:
            override:

        Returns:
            Success/failure
        """

    def reserve(self, product: int, quantity: int, t: int) -> bool:
        """
        Simulates reserving the given quantity of the given product at time t
        Args:
            product:
            quantity:
            t: time

        Returns:

        """
        self._reserved_storage[product, t] += quantity
        return True

    # ------------------
    # HISTORY MANAGEMENT
    # ------------------

    @abstractmethod
    def fix_before(self, t: int) -> bool:
        """
        Fix the history before this point

        Args:
            t:

        Returns:

        """

    @abstractmethod
    def bookmark(self) -> int:
        """Sets a bookmark to the current location

        Returns:
            bookmark ID
        """

    @abstractmethod
    def rollback(self, bookmark_id: int) -> bool:
        """Rolls back to the given bookmark ID

        Args:
            bookmark_id The bookmark ID returned from bookmark

        Remarks:

            - You can only rollback in the reverse order of bookmarks. If the bookmark ID given here is not the one
              at the top of the bookmarks stack, the rollback will fail (return False)

        """

    @abstractmethod
    def delete_bookmark(self, bookmark_id: int) -> bool:
        """Commits everything since the bookmark so it cannot be rolled back

        Args:
            bookmark_id The bookmark ID returned from bookmark

        Remarks:

            - You can only rollback in the reverse order of bookmarks. If the bookmark ID given here is not the one
              at the top of the bookmarks stack, the deletion will fail (return False)

        """


@dataclass
class Bookmark:
    id: int
    jobs: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list), init=False)
    buy_contracts: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list), init=False)
    sell_contracts: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list), init=False)
    payment_updates: Dict[int, float] = field(default_factory=lambda: defaultdict(float), init=False)
    loans_updates: Dict[int, float] = field(default_factory=lambda: defaultdict(float), init=False)
    storage_updates: Dict[int, Dict[int, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int))
                                                       , init=False)


class SlowFactorySimulator(FactorySimulator):
    """A slow factory simulator that runs an internal factory to find-out what will happen in the future

    Remarks:

        - It is *much* faster to always access the properties/methods of this class in ascending time. If that is not
          the case, each time reversal will cause a complete reset.
        - It is recommended to call `fix_before` () to fix the past once a production step is completed. That will speed
          up operations
    """

    def delete_bookmark(self, bookmark_id: int) -> bool:
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            return False
        self._bookmarks, self._bookmarked_at = self._bookmarks[:-1], self._bookmarked_at[:-1]
        self._active_bookmark = self._bookmarks[-1] if len(self._bookmarks) > 0 else None
        self._active_bookmarked_at = self._bookmarked_at[-1] if len(self._bookmarked_at) > 0 else -1
        return True

    def bookmark(self) -> int:
        bookmark = Bookmark(id=len(self._bookmarks))
        self._bookmarks.append(bookmark)
        self._bookmarked_at.append(self._factory.next_step)
        self._active_bookmark = bookmark
        self._active_bookmarked_at = self._bookmarked_at[-1]
        return bookmark.id

    def rollback(self, bookmark_id: int) -> bool:
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            return False
        for t, payment in self._active_bookmark.payment_updates.items():
            self._payment_updates[t] += payment
        for t, payment in self._active_bookmark.loans_updates.items():
            self._loans_updates[t] += payment
        for t, storage in self._active_bookmark.storage_updates.items():
            s = self._storage_updates[t]
            for k, v in storage:
                s[k] -= v
        for t, rolled_indices in self._active_bookmark.jobs.items():
            self._jobs[t] = [_ for i, _ in enumerate(self._jobs[t]) if i not in rolled_indices]
        for t, rolled_indices in self._active_bookmark.buy_contracts.items():
            self._buy_contracts[t] = [_ for i, _ in enumerate(self._buy_contracts[t]) if i not in rolled_indices]
        for t, rolled_indices in self._active_bookmark.sell_contracts.items():
            self._sell_contracts[t] = [_ for i, _ in enumerate(self._sell_contracts[t]) if i not in rolled_indices]

        if self._factory.next_step != self._bookmarked_at:
            self.goto(self._active_bookmarked_at)

        self._bookmarks, self._bookmarked_at = self._bookmarks[:-1], self._bookmarked_at[:-1]
        self._active_bookmark = self._bookmarks[-1] if len(self._bookmarks) > 0 else None
        self._active_bookmarked_at = self._bookmarked_at[-1] if len(self._bookmarked_at) > 0 else -1
        return True

    @property
    def final_balance(self) -> float:
        self.goto(self.n_steps - 1)
        return self.balance(self.n_steps - 1)

    @property
    def n_lines(self):
        return self._factory.n_lines

    def fix_before(self, t: int) -> bool:
        self.goto(t)
        self._fixed_before = t
        invalid = [i for i, bt in enumerate(self._bookmarked_at) if bt < t]
        self._bookmarks = [_ for i, _ in enumerate(self._bookmarks) if i not in invalid]
        self._bookmarked_at = [_ for i, _ in enumerate(self._bookmarked_at) if i not in invalid]
        return True

    def __init__(self, initial_wallet: float, initial_storage: Dict[int, int], n_steps: int, n_products: int
                 , profiles: List[ManufacturingProfile], max_storage: Optional[int]):
        super().__init__(initial_wallet=initial_wallet, initial_storage=initial_storage, n_steps=n_steps
                         , n_products=n_products, profiles=profiles, max_storage=max_storage)
        self._factory = Factory(initial_storage=initial_storage, initial_wallet=initial_wallet
                                , profiles=profiles, max_storage=max_storage)
        self._jobs: Dict[int, List[(Job, bool)]] = defaultdict(list)
        self._buy_contracts: Dict[int, List[(int, int, float)]] = defaultdict(list)
        self._sell_contracts: Dict[int, List[(int, int, float)]] = defaultdict(list)
        self._payment_updates: Dict[int, float] = defaultdict(float)
        self._loans_updates: Dict[int, float] = defaultdict(float)
        self._storage_updates: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._wallet = np.zeros(n_steps)
        self._loans = np.zeros(n_steps)
        self._storage = np.zeros(shape=(n_products, n_steps))
        self._line_schedules = np.zeros(shape=(self._factory.n_lines, self._n_steps))
        self._fixed_before = 0
        self._bookmarks: List[Bookmark] = []
        self._active_bookmark: Optional[Bookmark] = None
        self._active_bookmarked_at: int = -1
        self._bookmarked_at: List[int] = []

    def _as_array(self, storage: Dict[int, int]) -> np.array:
        a = np.zeros(self._n_products)
        for k, v in storage.items():
            a[k] = v
        return a

    def _update_state(self) -> None:
        t = self._factory.next_step - 1
        if t < 0:
            return
        self._wallet[t] = self._factory.wallet
        self._loans[t] = self._factory.loans
        self._storage[:, t] = self._as_array(self._factory.storage)
        self._line_schedules[:, t] = np.array(list(NO_PRODUCTION if command.is_none else command.profile.process.id
                                                   for command in self._factory.commands))

    def reset_to(self, t: int) -> None:
        self._factory = Factory(initial_storage={i: v for i, v in enumerate(self._initial_storage) if v != 0}
                                , initial_wallet=self._initial_wallet
                                , profiles=self._profiles)
        for step in range(t + 1):
            self._factory.receive(payment=self._payment_updates.get(step, 0.0))
            self._factory.add_loan(total=self._loans_updates.get(step, 0.0))
            jobs = self._jobs.get(t, [])
            for job, override in jobs:
                self._factory.schedule(job=job, override=override)
            contracts = self._buy_contracts.get(t, [])
            for product, quantity, price in contracts:
                self._factory.buy(product=product, quantity=quantity, price=price)
            contracts = self._sell_contracts.get(t, [])
            for product, quantity, price in contracts:
                self._factory.sell(product=product, quantity=quantity, price=price)
            self._factory.transport_to(inventory=self._storage_updates.get(step, {}))
            self._update_state()

    def goto(self, t: int) -> None:
        """
        Steps the factory to the end of step t
        Args:
            t:

        Returns:

        """
        if t > self.n_steps - 1:
            t = self.n_steps - 1
        if self._factory.next_step > t + 1:
            if t < self._fixed_before:
                return
            self.reset_to(t)
        while self._factory.next_step <= t:
            step = self._factory.next_step
            loan = self._loans_updates.get(step, None)
            if loan is not None:
                self._factory.add_loan(loan)
            payment = self._payment_updates.get(step, None)
            if payment is not None:
                self._factory.pay(payment)
            jobs = self._jobs.get(step, [])
            for job, override in jobs:
                self._factory.schedule(job=job, override=override)
            self._factory.step()
            inventory = self._storage_updates.get(step, None)
            if inventory is not None:
                self._factory.transport_to(inventory)
            self._update_state()

    def wallet_to(self, t: int) -> np.array:
        if t < self._fixed_before:
            return self._wallet[:t + 1]
        self.goto(t)
        return self._wallet[:t + 1]

    def line_schedules_to(self, t: int) -> np.array:
        if t < self._fixed_before:
            return self._storage[:, :t + 1]
        self.goto(t)
        return self._line_schedules[:, :t + 1]

    def storage_to(self, t: int) -> np.array:
        if t < self._fixed_before:
            return self._storage[:, :t + 1]
        self.goto(t)
        return self._storage[:, :t + 1]

    def loans_to(self, t: int) -> float:
        if t < self._fixed_before:
            return self._loans[:t + 1]
        self.goto(t)
        return self._loans[:t + 1]

    def add_loan(self, total: float, t: int) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._loans_updates[t] += total
        if self._active_bookmark:
            self._active_bookmark.loans_updates[t] += total
        return True

    def pay(self, payment: float, t: int) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._payment_updates[t] += payment
        if self._active_bookmark:
            self._active_bookmark.payment_updates[t] += payment
        return True

    def transport_to(self, inventory: Dict[int, int], t: int) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        s = self._storage_updates[t]
        for k, v in inventory:
            s[k] += v
        if self._active_bookmark:
            s = self._active_bookmark.storage_updates[t]
            for k, v in inventory:
                s[k] += v
        return True

    def schedule(self, job: Job, override=True) -> bool:
        t = job.time
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._jobs[t].append((job, override))
        if self._active_bookmark:
            self._active_bookmark.jobs[t].append(len(self._jobs[t]))
        return True

    def buy(self, product: int, quantity: int, price: int, t: int) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._buy_contracts[t].append((product, quantity, price))
        if self._active_bookmark:
            self._active_bookmark.buy_contracts[t].append(len(self._buy_contracts[t]))
        return True

    def sell(self, product: int, quantity: int, price: int, t: int) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._sell_contracts[t].append((product, quantity, price))
        if self._active_bookmark:
            self._active_bookmark.sell_contracts[t].append(len(self._sell_contracts[t]))
        return True

#
# @dataclass
# class FastFactorySimulator(FactorySimulator):
#     n_steps: int
#     """The number of steps for which the factory is going to be running"""
#     n_lines: int = field(init=False)
#     """The number of lines in the factory, will be set using the `profiles` input"""
#     costs: np.array
#     """An n_lines*n_processes array giving the cost of running this process on this line"""
#     cancellation_costs: np.array = field(init=False)
#     """An n_lines*n_processes array giving the cost of cancelling this process on this line"""
#     pause_initial_costs: np.array = field(init=False)
#     """An n_lines*n_processes array giving the initial cost of pausing this process on this line"""
#     pause_running_costs: np.array = field(init=False)
#     """An n_lines*n_processes array giving the running cost of pausing this process on this line"""
#     resume_costs: np.array = field(init=False)
#     """An n_lines*n_processes array giving the cost of resuming this process on this line"""
#     times: np.array = field(init=False)
#     """An n_lines*n_processes array giving the running time of this process on this line"""
#     schedule: np.array = field(init=False)
#     """The schedule of lines as a n_lines*n_steps array giving the index of the process running at every timestep
#      or NO_PRODUCTION if no process is running. The special value INVALID means that this time step cannot run anything
#     """
#
#     jobs: Dict[int, List[Job]] = field(init=False, default_factory=defaultdict(lambda: list))
#     """The jobs scheduled at every time-step"""
#     current_step: int = field(init=False, default=0)
#     """Current simulation step"""
#
#     profiles: InitVar[List[ManufacturingProfile]]
#     """A list of profiles used to initialize the factory"""
#     initial_storage: InitVar[Dict[int, int]]
#     """Mapping from product index to the amount available in the inventory"""
#     initial_wallet: InitVar[float]
#     """Money available for purchases"""
#
#     max_storage: Optional[int] = None
#     """Maximum storage allowed in this factory"""
#     id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
#     """Object name"""
#     manager: Optional['SCMLAgent'] = field(init=False, default=None)
#     """The factory manager which is an object of the `FactoryManager` class that controls production."""
#
#     def wallet(self, t: int) -> float:
#         pass
#
#     def storage(self, t: int) -> np.array:
#         pass
#
#     def loans(self, t: int) -> float:
#         pass
#
#     def add_loan(self, total: float, t: int) -> bool:
#         pass
#
#     def receive(self, payment: float, t: int) -> bool:
#         pass
#
#     def pay(self, payment: float, t: int) -> bool:
#         pass
#
#     def transport(self, inventory: Dict[int, int], t: int) -> bool:
#         pass
#
#     def __post_init__(self, profiles, initial_storage=None, initial_wallet=0.0):
#         given_lines = sorted(list(set(p.line for p in profiles)))
#         mapping = dict(zip(given_lines, range(len(given_lines))))
#         for profile in profiles:
#             profile.line = mapping[profile.line]
#         self.n_lines = len(given_lines)
#         self.commands = np.array([[[None] * self.n_steps] * self.n_lines], dtype=object)
#         self.schedule = np.zeros(shape=(self.n_lines, self.n_steps), dtype=int)
#         self.jobs = defaultdict(list)
#
#     @property
#     def balance(self):
#         """The total balance of the factory"""
#         return self.wallet - self.loans
#
#     def copy(self):
#         return Factory(storage={k: v for k, v in self.storage.items()}
#                        , wallet=self.wallet, loans=self.loans, max_storage=self.max_storage
#                        , n_steps=self.n_steps, profiles=self.profiles)
#
#     def step(self) -> List[ProductionReport]:
#         reports = []
#         for line in range(self.n_lines):
#             # step the current production process
#             report = self._step_line(line=line)
#             reports.append(report)
#             updates = report.updates
#             if updates.balance != 0:
#                 self.wallet += updates.balance
#             if updates.storage is not None:
#                 for k, v in updates.storage.items():
#                     self.storage[k] += v
#         self.current_step += 1
#         return reports
#
#     def schedule_job(self, job: Job, override=True) -> Optional[Dict[int, FactoryStatusUpdate]]:
#         """
#         Schedules the given job
#
#         Args:
#             job: A job to schedule
#             override: If true, override any preexisting jobs to make this one run
#
#         Returns:
#             None if it is not possible to schedule this command, otherwise a mapping from time-steps to
#             `FactoryStatusUpdate` to apply at this time-step.
#
#         Remarks:
#             The job is updated as follows:
#
#             - This line is set as the line member in job
#             - The updates that result from this schedule to balance and storage are added to the updates in the job
#         """
#         t = job.time
#         self.jobs[t].append(job)
#         result = self._schedule(command=job.command, t=t, profile=self.profiles[job.profile]
#                                 , line=job.line, override=override)
#         if result is None:
#             return result
#         if job.updates:
#             FactoryStatusUpdate.combine_sets(job.updates, result)
#         else:
#             job.updates = result
#         return result
#
#     def _cancel_running_command(self, running_command: RunningCommandInfo) -> Optional[Dict[int, FactoryStatusUpdate]]:
#         """
#         Cancels a running command as if it did not ever happen
#
#         Args:
#             running_command: The running command to cancel
#
#         Returns:
#
#             Dict[int, FactoryStatusUpdate]: The status updated for all times that need to be updated to cancel the
#                 command
#
#         Remarks:
#             - The output of a process that runs from step t to step t + n - 1 will only be in storage at step t + n
#             -
#
#         """
#         if running_command is None:
#             return {}
#         if running_command.command != 'run':
#             raise NotImplementedError('We only support run jobs now')
#         profile = running_command.profile
#         if profile is None:
#             return {}
#         line = running_command.profile.line
#         process = running_command.profile.process
#         process_index = process.id
#         beg, end = running_command.beg, running_command.end
#         n, cost = profile.n_steps, profile.cost
#         self.schedule[line, beg: end] = NO_PRODUCTION
#         self.commands[line, beg: end] = [None] * (end - beg)
#         results: Dict[int, FactoryStatusUpdate] = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}))
#         for need in process.inputs:
#             results[beg + int(math.floor(need.step * n))].storage[need.product] += need.quantity
#         for output in process.outputs:
#             results[beg + int(math.ceil(output.step * n))].storage[output.product] -= output.quantity
#         results[beg].balance += cost  # notice that we do not need to pay cancellation cost here
#         FactoryStatusUpdate.combine_sets(self.updates, results)
#         return results
#
#     def _simulate_run(self, t: int, profile: ManufacturingProfile
#                       , override=True) -> Optional[Dict[int, FactoryStatusUpdate]]:
#         """running is executed at the beginning of the step t
#
#         Args:
#             t: time-step to start the process
#             profile: the profile to start giving both the line and process
#             override: If true, override any running processes paying cancellation cost for these processes
#
#         Returns:
#
#             Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
#             the command if it is not None. If None is returned then scheduling failed.
#
#         Remarks:
#
#             - The output of a process that runs from step t to step t + n - 1 will only be in storage at step t + n
#
#         """
#         process = profile.process
#         line = profile.line
#         pid = process.id
#         n, cost = profile.n_steps, profile.cost
#
#         def do_run() -> Dict[int, FactoryStatusUpdate]:
#             self.schedule[line, t: t + n] = pid
#             c = RunningCommandInfo(command='run', process=pid, beg=t, end=t + n)
#             self.commands[line, t: t + n] = [c] * n
#             results: Dict[int, FactoryStatusUpdate] = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}))
#             for need in process.inputs:
#                 results[t + int(math.floor(need.step * n))].storage[need.product] -= need.quantity
#             for output in process.outputs:
#                 results[t + int(math.ceil(output.step * n))].storage[output.product] += output.quantity
#             results[t].balance -= cost
#             return results
#
#         # run immediately if possible
#         if np.all(self.schedule[line, t: t + n] == NO_PRODUCTION):
#             updates = do_run()
#             FactoryStatusUpdate.combine_sets(self.updates, updates)
#             return updates
#
#         # if I am not allowed to override, then this command has no effect and I return an empty status update
#         if not override:
#             return {}
#
#         # requires some stopping and cancellation
#         updates = defaultdict(lambda: FactoryStatusUpdate(balance=0, storage={}))
#         for current in range(t, t + n):
#             current_command = self.commands[line, current]
#             if current_command is None:
#                 continue
#             if current_command == self.commands[line, current - 1]:
#                 # that is a running process, stop it
#                 # @todo if the process has not produced any outcomes, then cancel it
#                 update_set = self._simulate_stop(t=current, line=line)
#             else:
#                 # that is a new process that is to be started. Do not start it
#                 update_set = self._cancel_running_command(current_command)
#             # if I cannot cancel or stop the running command, then fail
#             if update_set is None:
#                 return None
#             for i, change in update_set.items():
#                 updates[i].combine(change)
#         new_updates = do_run()
#         for i, change in new_updates.items():
#             updates[i].combine(change)
#         FactoryStatusUpdate.combine_sets(self.updates, updates)
#         return updates
#
#     def _simulate_pause(self, t: int, line: int) -> Optional[Dict[int, FactoryStatusUpdate]]:
#         """pausing is executed at the end of the step
#
#         Args:
#
#             t: time-step to start the process
#             line: the line on which the process is running
#
#         Returns:
#
#             Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
#             the command if it is not None. If None is returned then scheduling failed.
#
#         Remarks:
#
#             - Not implemented yet
#             - pausing when nothing is running is not an error and will return an empty status update
#
#         """
#         raise NotImplementedError('Pause is not implemented')
#
#     def _simulate_resume(self, t: int, line: int) -> Optional[Dict[int, FactoryStatusUpdate]]:
#         """resumption is executed at the end of the step (starting next step count down)
#
#
#         Args:
#
#             t: time-step to start the process
#             line: the line on which the process is running
#
#         Returns:
#
#             Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
#             the command if it is not None. If None is returned then scheduling failed.
#
#         Remarks:
#
#             - Not implemented yet
#             - resuming when nothing is paused is not an error and will return an empty status update
#
#         """
#         raise NotImplementedError('Resume is not implemented')
#
#     def _simulate_stop(self, t: int, line: int) -> Optional[Dict[int, FactoryStatusUpdate]]:
#         """stopping is executed at the beginning of the current step
#
#         Args:
#
#             t: time-step to start the process
#             line: the line on which the process is running
#
#         Returns:
#
#             Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
#             the command if it is not None. If None is returned then scheduling failed.
#
#         Remarks:
#
#             - stopping when nothing is running is not an error and will just return an empty schedule
#         """
#         current_command: RunningCommandInfo = self.commands[line, t]
#         if current_command is None:
#             return {}
#         running_process_index = self.schedule[line, t]
#         if current_command.beg >= t:
#             return self._cancel_running_command(current_command)
#         beg, end = current_command.beg, current_command.end
#         current_command.end = t
#         self.schedule[line, t: end] = running_process_index
#         profile = current_command.profile
#         process = profile.process
#         process_index = process.id
#         # current_command.costs[t] = profile.cancellation_cost
#         n = profile.n_steps
#         updates: Dict[int, FactoryStatusUpdate] = defaultdict(lambda: FactoryStatusUpdate.is_empty())
#         for need in process.inputs:
#             need_time = beg + int(math.floor(need.step * n))
#             if need_time > t:
#                 updates[need_time].storage[need.product] += need.quantity
#         for output in process.outputs:
#             output_time = beg + int(math.floor(output.step * n))
#             if output_time >= t:
#                 updates[output_time].storage[output.product] -= output.quantity
#         updates[t].balance -= profile.cancellation_cost
#         FactoryStatusUpdate.combine_sets(self.updates, updates)
#         return updates
#
#     def _schedule(self, command: str, t: int, profile: ManufacturingProfile = None, line: Optional[int] = None
#                   , override=True) -> Optional[Dict[int, FactoryStatusUpdate]]:
#         """
#         Schedules the given command at the given time for the given process.
#
#         Args:
#
#             command: Can be run, stop, pause, resume
#             t: The time to schedule
#             profile: The profile to schedule
#             line: The line to schedule at.
#             override: IF true running commands will be overridden to make this command possible
#
#         Returns:
#
#             None if it is not possible to schedule this command, otherwise a mapping from time-steps to
#             `FactoryStatusUpdate` to apply at this time-step.
#
#         Remarks:
#             - cannot give profile and line in the same time. For run commands give profile, otherwise give line
#
#         """
#         if line is not None and profile is not None and profile.line != line:
#             raise ValueError('Cannot specify both the line and profile at the same time with different line in '
#                              ' the profile')
#         if command == 'run':
#             if profile is None:
#                 raise ValueError('Cannot run an unspecified process')
#             return self._simulate_run(t=t, profile=profile, override=override)
#         if line is None:
#             if profile is not None:
#                 line = profile.line
#         if line is None:
#             raise ValueError(f'Cannot {command} without specifying either a profile or a line')
#         if command == 'pause':
#             return self._simulate_pause(t=t, line=line)
#         elif command == 'resume':
#             return self._simulate_resume(t=t, line=line)
#         elif command == 'stop':
#             return self._simulate_stop(t=t, line=line)
#         raise ValueError(f'Unknown command: {command}')
#
#     def _step_line(self, line: int) -> ProductionReport:
#         """
#         Steps the line to the time-step `t` assuming that it is already stepped to time-step t-1 given the storage
#
#         Args:
#             line: the line to step
#
#         Returns:
#             ProductionReport
#         """
#         t = self.current_step
#         updates = self.updates.get(t, None)
#         if updates is None:
#             updates = FactoryStatusUpdate.is_empty()
#         command = self.commands[line, t]
#         if command is None:
#             self.schedule[line, :t + 1] = INVALID_STEP
#             return ProductionReport(updates=updates, continuing=None, started=None, finished=None, failure=None
#                                     , line=line)
#         available_storage = self.max_storage - sum(self.storage.values())
#         process_index = command.process
#         missing_inputs = []
#         missing_money = 0
#         failed = False
#         started = command if command.beg == t else None
#         finished = command if command.end == t + 1 else None
#         continuing = command if command.beg != t and command.end != t else None
#         missing_space = 0
#         if updates.balance < 0 and self.wallet < -updates.balance:
#             failed = True
#             missing_money = -updates.balance - self.wallet
#         for product_id, quantity in updates.storage.items():
#             if quantity < 0 and self.storage.get(product_id, 0) < -quantity:
#                 failed = True
#                 missing_inputs.append(MissingInput(product=product_id, quantity=-quantity))
#             elif quantity > 0:
#                 available_storage -= quantity
#                 if available_storage < 0:
#                     failed = True
#                     missing_space -= available_storage
#                     available_storage = 0
#         if failed:
#             failure = ProductionFailure(line=line, command=command, missing_money=missing_money
#                                         , missing_inputs=missing_inputs, missing_space=missing_space)
#             if t == command.beg:
#                 self._cancel_running_command(command)
#             else:
#                 self._simulate_stop(t=t, line=line)
#             self.schedule[line, :t + 1] = INVALID_STEP
#             return ProductionReport(updates=FactoryStatusUpdate.is_empty()
#                                     , continuing=continuing, started=started, finished=finished, failure=failure
#                                     , line=line)
#         self.current_process = process_index
#         self.schedule[line, :t + 1] = INVALID_STEP
#         return ProductionReport(updates=updates, continuing=continuing, started=started, finished=finished, failure=None
#                                 , line=line)
