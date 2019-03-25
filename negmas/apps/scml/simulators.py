import math
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field
from contextlib import contextmanager
from .common import ManufacturingProfile, Job, Factory, NO_PRODUCTION

__all__ = [
    'FactorySimulator',
    'SlowFactorySimulator',
    'FastFactorySimulator',
    'storage_as_array',
    'transaction',
    'temporary_transaction'
]


def storage_as_array(storage: Dict[int, int], n_products: int) -> np.array:
    """
    Converts storage to an array
    Args:
        storage: A dictionary giving quantity for each product index
        n_products: number of products (size of the resulting array)

    Returns:

    """
    a = np.zeros(n_products)
    for k, v in storage.items():
        a[k] = v
    return a


class FactorySimulator(ABC):
    """Simulates a factory allowing for prediction of storage/balance in the future"""

    def __init__(self, initial_wallet: float, initial_storage: Dict[int, int], n_steps: int, n_products: int
                 , profiles: List[ManufacturingProfile], max_storage: Optional[int] = None):
        self._n_steps = n_steps
        self._max_storage = max_storage if max_storage is not None else sys.maxsize
        self._initial_wallet = initial_wallet
        self._initial_storage = np.zeros(n_products)
        for k, v in initial_storage.items():
            self._initial_storage[k] = v
        self._profiles = profiles
        self._n_products = n_products
        self._reserved_storage = np.zeros(shape=(n_products, n_steps))

    def _as_array(self, storage: Dict[int, int]):
        return storage_as_array(storage=storage, n_products=self._n_products)

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
        Returns the cash in wallet up to and including time t
        Args:
            t:

        Returns:

        """

    def wallet_at(self, t: int) -> float:
        return self.wallet_to(t)[-1]

    @abstractmethod
    def storage_to(self, t: int) -> np.array:
        """
        Returns the storage at time t
        
        Args:
            t: 

        Returns:

        """

    def storage_at(self, t: int) -> np.array:
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

    def line_schedules_at(self, t: int) -> np.array:
        return self.line_schedules_to(t)[:, -1]

    def total_storage_to(self, t: int) -> np.array:
        return self.storage_to(t).sum(axis=0)

    def total_storage_at(self, t: int) -> int:
        return self.total_storage_to(t)[-1]

    def reserved_storage_to(self, t: int) -> np.array:
        return self._reserved_storage[:, :t + 1]

    def reserved_storage_at(self, t: int) -> np.array:
        return self._reserved_storage[:, t]

    def available_storage_at(self, t: int) -> np.array:
        return self.storage_at(t) - self.reserved_storage_at(t)

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

    def loans_at(self, t: int) -> float:
        """
        Returns loans at time t
        Args:
            t:

        Returns:

        """
        return self.loans_to(t)[-1]

    def balance_at(self, t: int) -> float:
        """
        Returns the balance fo the factory at time t
        Args:
            t: 

        Returns:

        """
        return self.wallet_at(t) - self.loans_at(t)

    def balance_to(self, t: int) -> np.array:
        """
        Returns the balance fo the factory at times <= t
        Args:
            t:

        Returns:

        """
        return self.wallet_to(t) - self.loans_to(t)

    @property
    @abstractmethod
    def fixed_before(self):
        """Gives the time before which the schedule is fixed"""

    # -------------------------
    # OPERATIONS (UPDATE STATE)
    # -------------------------

    @abstractmethod
    def set_state(self, t: int, storage: np.array, wallet: float, loans: float, line_schedules: np.array) -> None:
        """Sets the current state at the given time-step. It implicitly causes a fix_before(t + 1)"""

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
    def pay(self, payment: float, t: int, ignore_money_shortage: bool = True) -> bool:
        """
        Simulate payment at time t
        
        Args:            
            payment: 
            t:
            ignore_money_shortage: If True, shortage in money will be ignored and the wallet can go negative 

        Returns:

        """

    @abstractmethod
    def transport_to(self, product: int, quantity: int, t: int
                     , ignore_inventory_shortage: bool = True, ignore_space_shortage: bool = True) -> bool:
        """
        Simulates transporting products to/from storage at time t
        
        Args:            
            product:
            quantity:
            t:
            ignore_inventory_shortage: Ignore shortage in the `product` which may lead to negative storage[product]
            ignore_space_shortage:  Ignore the limit on total storage which may lead to total_storage > max_storage

        Returns:

        """

    @abstractmethod
    def buy(self, product: int, quantity: int, price: int, t: int
            , ignore_money_shortage: bool = True, ignore_space_shortage: bool = True) -> bool:
        """
        Buy a given quantity of a product for a given price at some time t
        
        Args:
            product:
            quantity:
            price:
            t: time
            ignore_money_shortage: If True, shortage in money will be ignored and the wallet can go negative
            ignore_space_shortage:  Ignore the limit on total storage which may lead to total_storage > max_storage

        Returns:
            
            - buy cannot ever have inventory shortage

        """

    @abstractmethod
    def sell(self, product: int, quantity: int, price: int, t: int
             , ignore_money_shortage: bool = True, ignore_inventory_shortage: bool = True) -> bool:
        """
        sell a given quantity of a product for a given price at some time t

        Args:
            product:
            quantity:
            price:
            t:
            ignore_money_shortage: If True, shortage in money will be ignored and the wallet can go negative
            ignore_inventory_shortage: Ignore shortage in the `product` which may lead to negative storage[product]

        Returns:

        """

    @abstractmethod
    def schedule(self, job: Job, ignore_inventory_shortage=True, ignore_money_shortage=True, ignore_space_shortage=True
                 , override=True) -> bool:
        """
        Simulates scheduling the given job at its `time` and `line` optionally overriding whatever was already scheduled
        Args:
            job:
            ignore_inventory_shortage: If true shortages in inputs will be ignored
            ignore_money_shortage: If true, shortage in money will be ignored
            ignore_space_shortage: If true, shortage in space will be ignored
            override:

        Returns:
            Success/failure
        """

    def reserve(self, product: int, quantity: int, t: int) -> bool:
        """
        Simulates reserving the given quantity of the given product at times >= t
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
class _Bookmark:
    id: int
    jobs: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list), init=False)
    buy_contracts: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list), init=False)
    sell_contracts: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list), init=False)
    payment_updates: Dict[int, float] = field(default_factory=lambda: defaultdict(float), init=False)
    loans_updates: Dict[int, float] = field(default_factory=lambda: defaultdict(float), init=False)
    storage_updates: Dict[int, Dict[int, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int))
                                                       , init=False)


@dataclass
class _State:
    t: int
    storage: np.array
    wallet: float
    loans: float
    line_schedules: np.array


class SlowFactorySimulator(FactorySimulator):
    """A slow factory simulator that runs an internal factory to find-out what will happen in the future

    Remarks:

        - It is *much* faster to always access the properties/methods of this class in ascending time. If that is not
          the case, each time reversal will cause a complete reset.
        - It is recommended to call `fix_before` () to fix the past once a production step is completed. That will speed
          up operations
    """

    def set_state(self, t: int, storage: np.array, wallet: float, loans: float, line_schedules: np.array) -> None:
        for i, s in enumerate(storage):
            d = s - self.storage_at(t)[i]
            if d == 0.0:
                continue
            self._storage_updates[t][i] += d
        d = self.wallet_at(t) - wallet
        if d != 0.0:
            self._payment_updates[t] -= d
        d = self.loans_at(t) - loans
        if d != 0.0:
            self._loans_updates[t] -= d
        expected_schedules = self.line_schedules_at(t)
        for i in range(self.n_lines):
            expected, actual = expected_schedules[i], line_schedules[i]
            if expected == actual:
                continue
            if expected == NO_PRODUCTION:
                raise ValueError(f'Expected no production at time {t} on line {i} but actually process '
                                 f'{actual} is running')
            if expected != actual and actual != NO_PRODUCTION:
                raise ValueError(f'Expected process {expected} at time {t} on line {i} but actually process '
                                 f'{actual} is running')
            self._line_schedules[i, t] = actual 
        self.fix_before(t + 1)
        self._saved_states[t].append(_State(t=t, storage=storage.copy(), wallet=wallet, loans=loans
                                            , line_schedules=line_schedules.copy()))

    def delete_bookmark(self, bookmark_id: int) -> bool:
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            raise ValueError(f'there is no active bookmark to delete')
        self._bookmarks, self._bookmarked_at = self._bookmarks[:-1], self._bookmarked_at[:-1]
        self._active_bookmark = self._bookmarks[-1] if len(self._bookmarks) > 0 else None
        self._active_bookmarked_at = self._bookmarked_at[-1] if len(self._bookmarked_at) > 0 else -1
        return True

    def bookmark(self) -> int:
        bookmark = _Bookmark(id=len(self._bookmarks))
        self._bookmarks.append(bookmark)
        self._bookmarked_at.append(self._factory.next_step)
        self._active_bookmark = bookmark
        self._active_bookmarked_at = self._bookmarked_at[-1]
        return bookmark.id

    def rollback(self, bookmark_id: int) -> bool:
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            raise ValueError(f'there is no active bookmark to rollback')
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
        return True

    @property
    def final_balance(self) -> float:
        self.goto(self.n_steps - 1)
        return self.balance_at(self.n_steps - 1)

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
        self._jobs: Dict[int, List[(Job, bool, bool, bool, bool)]] = defaultdict(list)
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
        self._bookmarks: List[_Bookmark] = []
        self._active_bookmark: Optional[_Bookmark] = None
        self._active_bookmarked_at: int = -1
        self._bookmarked_at: List[int] = []
        self._saved_states: Dict[int, List[_State]] = defaultdict(list)

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
            for job, override, ignore_storage, ignore_money, ignore_space in jobs:
                # @todo use ignore* here
                try:
                    self._factory.schedule(job=job, override=override)
                except ValueError as err:
                    print(err)
            contracts = self._buy_contracts.get(t, [])
            for product, quantity, price in contracts:
                try:
                    self._factory.buy(product=product, quantity=quantity, price=price)
                except ValueError as err:
                    print(err)
            contracts = self._sell_contracts.get(t, [])
            for product, quantity, price in contracts:
                try:
                    self._factory.sell(product=product, quantity=quantity, price=price)
                except ValueError as err:
                    print(err)
            inventory = self._storage_updates.get(step, {})
            for product, quantity in inventory.items():
                try:
                    self._factory.transport_to(product, quantity)
                except ValueError as err:
                    print(err)
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
            for job, override, ignore_storage, ignore_money, ignore_space in jobs:
                # @todo implement ignore_input_shortage inside the factory to use ignore_input_shortage here
                self._factory.schedule(job=job, override=override)
            self._factory.step()
            inventory = self._storage_updates.get(step, None)
            if inventory is not None:
                for product, quantity in inventory.items():
                    self._factory.transport_to(product, quantity)
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

    def pay(self, payment: float, t: int, ignore_money_shortage: bool = True) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._payment_updates[t] += payment
        if self._active_bookmark:
            self._active_bookmark.payment_updates[t] += payment
        return True

    def transport_to(self, product: int, quantity: int, t: int, ignore_inventory_shortage: bool = True, ignore_space_shortage: bool = True) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        s = self._storage_updates[t]
        s[product] += quantity
        if self._active_bookmark:
            s = self._active_bookmark.storage_updates[t]
            s[product] += quantity
        return True

    def schedule(self, job: Job, ignore_inventory_shortage=True, ignore_money_shortage=True, ignore_space_shortage=True
                 , override=True) -> bool:
        t = job.time
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._jobs[t].append((job, override, ignore_inventory_shortage, ignore_money_shortage, ignore_space_shortage))
        if self._active_bookmark:
            self._active_bookmark.jobs[t].append(len(self._jobs[t]))
        return True

    def buy(self, product: int, quantity: int, price: int, t: int
            , ignore_money_shortage: bool = True, ignore_space_shortage: bool = True) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._buy_contracts[t].append((product, quantity, price))
        if self._active_bookmark:
            self._active_bookmark.buy_contracts[t].append(len(self._buy_contracts[t]))
        return True

    def sell(self, product: int, quantity: int, price: int, t: int, ignore_money_shortage: bool = True, ignore_inventory_shortage: bool = True) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._sell_contracts[t].append((product, quantity, price))
        if self._active_bookmark:
            self._active_bookmark.sell_contracts[t].append(len(self._sell_contracts[t]))
        return True

    @property
    def fixed_before(self):
        return self._fixed_before


@dataclass
class _FullBookmark:
    id: int
    wallet: np.array
    loans: np.array
    storage: np.array
    line_schedules: np.array
    has_jobs: np.array


class FastFactorySimulator(FactorySimulator):

    def _as_array(self, storage: Dict[int, int]) -> np.array:
        a = np.zeros(self._n_products)
        for k, v in storage.items():
            a[k] = v
        return a

    def __init__(self, initial_wallet: float, initial_storage: Dict[int, int], n_steps: int, n_products: int
                 , profiles: List[ManufacturingProfile], max_storage: Optional[int]):
        super().__init__(initial_wallet=initial_wallet, initial_storage=initial_storage, n_steps=n_steps
                         , n_products=n_products, profiles=profiles, max_storage=max_storage)
        self._wallet = np.ones(n_steps) * initial_wallet
        self._loans = np.zeros(n_steps)
        self._storage = np.repeat(self._as_array(initial_storage).reshape((n_products, 1)), n_steps, axis=1)
        self._total_storage = self._storage.sum(axis=0)
        factory = Factory(initial_storage=initial_storage, initial_wallet=initial_wallet, profiles=profiles
                          , max_storage=max_storage)
        self._profiles = factory.profiles
        self._n_lines = factory.n_lines
        self._line_schedules = np.ones(shape=(self._n_lines, self._n_steps)) * NO_PRODUCTION
        self._has_jobs = np.zeros(shape=(self._n_lines, self._n_steps), dtype=bool)
        self._fixed_before = 0
        self._bookmarks: List[_FullBookmark] = []
        self._active_bookmark: Optional[_FullBookmark] = None

    def init(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    @property
    def fixed_before(self):
        return self._fixed_before

    @property
    def n_lines(self):
        return self._n_lines

    @property
    def final_balance(self) -> float:
        return self._wallet[-1] - self._loans[-1]

    def wallet_to(self, t: int) -> np.array:
        return self._wallet[:t + 1]

    def storage_to(self, t: int) -> np.array:
        return self._storage[:, :t + 1]

    def line_schedules_to(self, t: int) -> np.array:
        return self._line_schedules[:, : t + 1]

    def loans_to(self, t: int) -> np.array:
        return self._loans[: t + 1]

    def add_loan(self, total: float, t: int) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        self._loans[t:] += total
        return True

    def pay(self, payment: float, t: int, ignore_money_shortage: bool = True) -> bool:
        # @todo add minimum balance
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        b = self._wallet[t:]
        b -= payment
        if b.min() < 0:
            b += payment
            return False
        return True

    def transport_to(self, product: int, quantity: int, t: int, ignore_inventory_shortage: bool = True, ignore_space_shortage: bool = True) -> bool:
        # @todo add minimum storage
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        s, total = self._storage[product, t:].view(), self._total_storage[t:]
        s += quantity
        total += quantity
        if s.min() < 0 or total.max() > self.max_storage:
            s -= quantity
            total -= quantity
            return False
        return True

    def buy(self, product: int, quantity: int, price: int, t: int
            , ignore_money_shortage: bool = True, ignore_space_shortage: bool = True) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        s, total = self._storage[product, t:].view(), self._total_storage[t:]
        s += quantity
        total += quantity
        b = self._wallet[t:]
        b -= price
        if total.max() > self.max_storage or b.min() < 0:
            s -= quantity
            total -= quantity
            b += price
            return False
        return True

    def sell(self, product: int, quantity: int, price: int, t: int, ignore_money_shortage: bool = True, ignore_inventory_shortage: bool = True) -> bool:
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        s, total = self._storage[product, t:].view(), self._total_storage[t:]
        s -= quantity
        total -= quantity
        b = self._wallet[t:]
        b += price
        if s.min() < 0:
            s += quantity
            total += quantity
            b -= price
            return False
        return True

    def schedule(self, job: Job, ignore_inventory_shortage=True, ignore_money_shortage=True, ignore_space_shortage=True
                 , override=True) -> bool:
        t, job_override = job.time, job.override
        if t < self._fixed_before:
            raise ValueError(f'Cannot run operations in the past (t={t}, fixed before {self._fixed_before})')
        if job_override:
            raise NotImplementedError(f'{self.__class__.__name__} does not support scheduling jobs with overriding')
        # job_line = job.line # only useful for stop/pause/resume that are not supported
        profile = self._profiles[job.profile]
        inputs, outputs, length, cost = profile.process.inputs, profile.process.outputs, profile.n_steps, profile.cost
        line = profile.line

        # confirm that there is no other jobs already scheduled at this exact time:
        if self._has_jobs[line, t]:
            if override:
                raise NotImplementedError(f'{self.__class__.__name__} does not support scheduling more than a single '
                                          f'job at any time-step/line')
            return False

        # confirm that the line is not busy. If it was busy, and we are not overriding, fail.
        if not job_override and np.any(self._line_schedules[line, t:t + length] != NO_PRODUCTION):
            return False

        # confirm that there is enough money to start production
        if (not ignore_money_shortage) and np.any(self._wallet[t:] < cost):
            return False
        # bookmark to be able to rollback at any error
        if job.action == 'run':
            with transaction(self) as bookmark:
                if not self.pay(cost, t):
                    self.rollback(bookmark)
                    return False
                self._line_schedules[line, t:t + length] = profile.process.id
                for i in inputs:
                    it = int(math.floor(i.step * length) + t)
                    p, q = i.product, i.quantity
                    if (not ignore_inventory_shortage) and np.any(self._storage[p, it:] < q):
                        self.rollback(bookmark)
                        return False
                    s, total = self._storage[p, it:].view(), self._total_storage[it:]
                    s -= q
                    total -= q
                for o in outputs:
                    ot = int(math.ceil(o.step * length) + t)
                    p, q = o.product, o.quantity
                    if (not ignore_space_shortage) and np.any(self._total_storage[ot:] + q > self.max_storage):
                        self.rollback(bookmark)
                        return False
                    s, total = self._storage[p, ot:].view(), self._total_storage[ot:]
                    s += q
                    total += q
            return True
        raise NotImplementedError(f'{self.__class__.__name__} does not support scheduling {job.action} jobs')

    def fix_before(self, t: int) -> bool:
        self._fixed_before = t
        return True

    def delete_bookmark(self, bookmark_id: int) -> bool:
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            raise ValueError(f'there is no active bookmark to delete')
        self._bookmarks = self._bookmarks[:-1]
        self._active_bookmark = self._bookmarks[-1] if len(self._bookmarks) > 0 else None
        return True

    def bookmark(self) -> int:
        bookmark = _FullBookmark(id=len(self._bookmarks), wallet=self._wallet.copy(), loans=self._loans.copy()
                                 , storage=self._storage.copy(), line_schedules=self._line_schedules.copy()
                                 , has_jobs=self._has_jobs.copy())
        self._bookmarks.append(bookmark)
        self._active_bookmark = bookmark
        return bookmark.id

    def rollback(self, bookmark_id: int) -> bool:
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            raise ValueError(f'there is no active bookmark to rollback')
        b = self._active_bookmark
        self._wallet, self._loans, self._storage = b.wallet, b.loans, b.storage
        self._line_schedules, self._has_jobs = b.line_schedules, b.has_jobs
        self._total_storage = self._storage.sum(axis=0)
        return True

    def set_state(self, t: int, storage: np.array, wallet: float, loans: float, line_schedules: np.array) -> None:
        self._storage[:, t:] += storage.reshape(self._n_products, 1) - self._storage[:, t].reshape(self._n_products, 1)
        self._wallet[t:] += wallet - self._wallet[t]
        self._loans[t:] += loans - self._loans[t]

        self._line_schedules[:, t] = line_schedules

        # @todo enable this again to confirm that simulation is correct. may be I set_state before the job is run on the simulator
        # expected_schedules = self._line_schedules[:, t]
        # for i in range(self.n_lines):
        #     expected, actual = expected_schedules[i], line_schedules[i]
        #     if expected == actual:
        #         continue
        #     if expected == NO_PRODUCTION:
        #         raise ValueError(f'Expected no production at time {t} on line {i} but actually process '
        #                          f'{actual} is running')
        #     if expected != actual and actual != NO_PRODUCTION:
        #         raise ValueError(f'Expected process {expected} at time {t} on line {i} but actually process '
        #                          f'{actual} is running')
        #     self._line_schedules[i, t] = actual

        self.fix_before(t)

    class Java:
        implements = ['jnegmas.apps.scml.simulators.FactorySimulator']


@contextmanager
def transaction(simulator):
    """Runs the simulated actions then confirms them if they are not rolled back"""
    _bookmark = simulator.bookmark()
    yield _bookmark
    simulator.delete_bookmark(_bookmark)


@contextmanager
def temporary_transaction(simulator):
    """Runs the simulated actions then rolls them back"""
    _bookmark = simulator.bookmark()
    yield _bookmark
    simulator.rollback(_bookmark)
    simulator.delete_bookmark(_bookmark)
