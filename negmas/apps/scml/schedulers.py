import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Collection, List, Optional

import numpy as np
from dataclasses import dataclass, field

from negmas.apps.scml.helpers import zero_runs
from negmas.situated import Contract
from .common import ProductionNeed, Job, Product, Process, ProductManufacturingInfo, SCMLAgreement, \
    ManufacturingProfileCompiled, INVALID_UTILITY
from .simulators import FactorySimulator, NO_PRODUCTION, transaction

__all__ = [
    'ScheduleInfo', 'Scheduler', 'GreedyScheduler'
]


@dataclass
class ScheduleInfo:
    final_balance: float
    """balance at the end of the schedule"""
    valid: bool = True
    """Is this a valid schedule?"""
    start: Optional[int] = None
    """The starting step of this schedule"""
    end: Optional[int] = None
    """The step after the last step in this simulation"""
    # these two attributes are not directly related to the schedule but with the scheduling operation that generated it
    needs: List[ProductionNeed] = field(default_factory=list)
    """The products needed but not still in storage needed to complete this schedule."""
    jobs: List[Job] = field(default_factory=list)
    """The jobs that need to be scheduled"""
    failed_contracts: List[Contract] = field(default_factory=list)
    """A list of contracts that failed to be scheduled."""
    ignored_contracts: List[Contract] = field(default_factory=list)
    """A list of contracts ignored for this schedule because they are in the past."""

    def __str__(self):
        fail = str("fail " + "|".join(str(_) for _ in self.failed_contracts)) if len(self.failed_contracts) > 0 else ""
        ignored = str("ignored " + "|".join(str(_) for _ in self.ignored_contracts)) \
            if len(self.ignored_contracts) > 0 else ""
        jobs = str("jobs " + "|".join(str(_) for _ in self.jobs)) if len(self.jobs) > 0 else ""
        updates, needs = '', ''
        if len(self.needs) > 0:
            needs = "needs :" + "\n\t".join(str(_) for _ in self.needs)

        result = f'{"valid" if self.valid else "invalid"} (ends before {self.end}):'
        for x in (fail, ignored, jobs, updates, needs):
            if len(x) > 0:
                result += '\n' + x
        return result

    def combine(self, other: 'ScheduleInfo') -> None:
        self.valid = self.valid and other.valid
        if other.end is not None and self.end is not None:
            self.end = max(self.end, other.end)
        if other.needs is not None:
            self.needs.extend(other.needs)
        if other.jobs is not None:
            self.jobs.extend(other.jobs)
        if other.failed_contracts is not None:
            self.failed_contracts.extend(other.failed_contracts)
        if other.ignored_contracts is not None:
            self.ignored_contracts.extend(other.ignored_contracts)
        self.final_balance = other.final_balance


class Scheduler(ABC):
    """Base class for all schedulers"""

    def __init__(self, manager_id: str, awi: 'SCMLAWI', max_insurance_premium: float, horizon: Optional[int] = None):
        self.horizon = horizon
        self.n_steps = 0
        self.n_lines = 0
        self.simulator: FactorySimulator = None
        self.products: List[Product] = []
        self.processes: List[Process] = []
        self.profiles: List[ManufacturingProfileCompiled] = []
        self.producing: Dict[int, List[ProductManufacturingInfo]] = {}
        self.manager_id = manager_id
        self.awi = awi
        self.max_insurance_premium = max_insurance_premium

    def bookmark(self) -> int:
        """Sets a bookmark to the current location

        Returns:
            bookmark ID
        """
        return self.simulator.bookmark()

    def rollback(self, bookmark_id: int) -> bool:
        """Rolls back to the given bookmark ID

        Args:
            bookmark_id The bookmark ID returned from bookmark

        Remarks:

            - You can only rollback in the reverse order of bookmarks. If the bookmark ID given here is not the one
              at the top of the bookmarks stack, the rollback will fail (return False)

        """
        return self.simulator.rollback(bookmark_id)

    def delete_bookmark(self, bookmark_id: int) -> bool:
        """Commits everything since the bookmark so it cannot be rolled back

        Args:
            bookmark_id The bookmark ID returned from bookmark

        Remarks:

            - You can only rollback in the reverse order of bookmarks. If the bookmark ID given here is not the one
              at the top of the bookmarks stack, the deletion will fail (return False)

        """
        return self.simulator.delete_bookmark(bookmark_id)

    def init(self, simulator: FactorySimulator, products: List[Product], processes: List[Process]
             , profiles: List[ManufacturingProfileCompiled]
             , producing: Dict[int, List[ProductManufacturingInfo]]):
        """Called by the FactoryManager after it is initialized"""
        self.simulator = simulator
        self.n_lines = self.simulator.n_lines
        self.n_steps = self.simulator.n_steps
        self.products = products
        self.processes = processes
        self.producing = producing
        self.profiles = profiles

    def schedule(self, contracts: Collection[Contract] = ()
                 , assume_no_further_negotiations=False
                 , ensure_storage_for: int = 0
                 , start_at: int = 0) -> ScheduleInfo:
        """
        Schedules a set of contracts and returns either the search_for_schedule or None if infeasible

        Args:
            whatever it has scheduled before. If the state is given, it is taken as the initial state for scheduling
            contracts: The contracts to be scheduled
            assume_no_further_negotiations: whether to assume that more negotiations can take place (to secure
            production needs)
            ensure_storage_for: A minimum time to ensure that products are available in storage before contract delivery
            times (sell contracts).
            start_at: The time at which to start scheduling. No jobs will be scheduled before this time.

        Returns:
            `ScheduleInfo` describing the schedulo and any production needs and updates to be carried out.

        """
        # initialize the state of the scheduler with initial knowledge
        # @todo make sure to take start into account. NOw I always schedule from the start
        start, end = 0, -1
        for contract in contracts:
            if contract.agreement['time'] > end:
                end = contract.agreement['time'] + 1
        if self.horizon is not None and end - start > self.horizon:
            end = start + self.horizon
        if end > self.n_steps:
            end = self.n_steps

        # if we have not new contracts, then we are done
        if len(contracts) == 0:
            return ScheduleInfo(valid=True, start=start, end=end
                                , final_balance=self.simulator.final_balance)

        return self.find_schedule(contracts=contracts, start=start, end=end
                                  , assume_no_further_negotiations=assume_no_further_negotiations
                                  , ensure_storage_for=ensure_storage_for
                                  , start_at=start_at)

    @abstractmethod
    def find_schedule(self, contracts: Collection[Contract]
                      , start: int, end: int
                      , assume_no_further_negotiations=False
                      , ensure_storage_for: int = 0
                      , start_at: int = 0) -> ScheduleInfo:
        """
        Schedules a set of contracts and returns either the search_for_schedule or None if infeasible

        Args:
            start:
            end:
            contracts:
            assume_no_further_negotiations:
            ensure_storage_for:
            start_at: The time at which to start scheduling. No jobs will be scheduled before this time.

        Returns:

        """


class GreedyScheduler(Scheduler):
    """Default scheduler used by the DefaultFactoryManager"""

    def __getstate__(self):
        result = self.__dict__.copy()
        if 'fields' in result.keys():
            del result['fields']

    def __setstate__(self, state):
        self.__dict__ = state
        self.fields = [self.total_unit_cost, self.unit_time, self.production_unit_cost, self.input_unit_cost]

    def __init__(self, manager_id: str, awi: 'SCMLAWI', max_insurance_premium: float, horizon: Optional[int] = None
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
        super().__init__(manager_id=manager_id, horizon=horizon, awi=awi, max_insurance_premium=max_insurance_premium)
        self.add_catalog_prices = add_catalog_prices
        self.strategy = strategy
        self.fields: List[Callable[[ProductManufacturingInfo], float]] = [self.total_unit_cost, self.unit_time
            , self.production_unit_cost, self.input_unit_cost]
        mapper = {'tc': 0, 't': 1, 'pc': 2, 'ic': 3}
        self.field_order: List[int] = []
        sort_fields = profile_sorter.split('>')
        self.producing: Dict[int, List[ProductManufacturingInfo]] = {}
        for field_name in sort_fields:
            if field_name in ('time', 't'):
                self.field_order.append(mapper['t'])
            elif field_name in ('total-cost', 'tc', 'tcost'):
                self.field_order.append(mapper['tc'])
            elif field_name in ('production-cost', 'pc', 'pcost'):
                self.field_order.append(mapper['pc'])
            elif field_name in ('input-cost', 'ic', 'icost'):
                self.field_order.append(mapper['ic'])

    def init(self, simulator: FactorySimulator, products: List[Product], processes: List[Process]
             , profiles: List[ManufacturingProfileCompiled]
             , producing: Dict[int, List[ProductManufacturingInfo]]):
        super().init(simulator=simulator, products=products, processes=processes, producing=producing
                     , profiles=profiles)
        self.producing = {k: sorted(v, key=self._profile_sorter) for k, v in self.producing.items()}

    def _profile_sorter(self, info: ProductManufacturingInfo) -> Any:
        vals = [field(info) for field in self.fields]
        profile = self.profiles[info.profile]
        return tuple([vals[indx] for indx in self.field_order] + [profile.line, profile.process])

    def unit_time(self, info: ProductManufacturingInfo) -> float:
        profile = self.profiles[info.profile]
        return profile.n_steps / info.quantity

    def total_cost(self, info: ProductManufacturingInfo) -> float:
        products = self.products
        profile = self.profiles[info.profile]
        process = self.processes[profile.process]
        production_cost = profile.cost

        def safe(x):
            return 0.0 if x is None else x

        inputs_cost = sum(safe(products[inp.product].catalog_price) * inp.quantity for inp in process.inputs)
        return production_cost + inputs_cost

    def total_unit_cost(self, info: ProductManufacturingInfo) -> float:
        return self.total_cost(info=info) / info.quantity

    def production_cost(self, info: ProductManufacturingInfo) -> float:
        profile = self.profiles[info.profile]
        return profile.cost

    def production_unit_cost(self, info: ProductManufacturingInfo) -> float:
        return self.production_cost(info=info) / info.quantity

    def input_cost(self, info: ProductManufacturingInfo):
        products = self.products
        profile = self.profiles[info.profile]
        process = self.processes[profile.process]

        def safe(x):
            return 0.0 if x is None else x

        return sum(safe(products[inp.product].catalog_price) * inp.quantity for inp in process.inputs)

    def input_unit_cost(self, info: ProductManufacturingInfo) -> float:
        return self.input_cost(info=info) / info.quantity

    # noinspection PyUnusedLocal
    def schedule_contract(self, contract: Contract
                          , assume_no_further_negotiations=False
                          , end: int = None, ensure_storage_for: int = 0
                          , start_at: int = 0) -> ScheduleInfo:
        """
        Schedules this contract if possible and returns information about the resulting schedule

        Args:
            contract:
            assume_no_further_negotiations: If true no further negotiations will be assumed possible
            end:
            ensure_storage_for: The number of steps all needs must be in storage before they are consumed in production
            start_at: No jobs will be scheduled before that time.

        Returns:

        """
        ignore_failures = not assume_no_further_negotiations
        simulator: FactorySimulator = self.simulator
        start = max(simulator.fixed_before, start_at)

        if end is None:
            end = simulator.n_steps
        if contract.agreement is None:
            return ScheduleInfo(end=end, final_balance=self.simulator.balance_at(end - 1))
        agreement: SCMLAgreement
        if isinstance(contract.agreement, dict):
            agreement = SCMLAgreement(**contract.agreement)
        else:
            agreement = contract.agreement  # type: ignore
        t = agreement['time']
        if t < start:
            return ScheduleInfo(end=end, final_balance=INVALID_UTILITY, valid=False, ignored_contracts=[contract])
        q, u = int(agreement['quantity']), agreement['unit_price']
        p = u * q
        pid: int = contract.annotation['cfp'].product
        if contract.annotation['buyer'] == self.manager_id:
            # I am a buyer
            # We do not ignore money shortage for buying. This means that the agent will not buy if the money it needs
            # may partially come from a sell contract that is not considered yet
            if not simulator.buy(product=pid, quantity=q, price=p, t=t, ignore_space_shortage=ignore_failures
                , ignore_money_shortage=ignore_failures):
                return ScheduleInfo(end=end, valid=False, failed_contracts=[contract], final_balance=INVALID_UTILITY)
            if p <= 0:
                return ScheduleInfo(valid=True, end=end, final_balance=self.simulator.balance_at(end - 1))
            insurance = self.awi.evaluate_insurance(contract=contract, t=self.awi.current_step)
            if insurance is not None and insurance / p < self.max_insurance_premium:
                # if it is not possible to buy the insurance, the factory manager will not try to buy it. This is still
                # a valid schedule
                simulator.pay(insurance, t=t)
            return ScheduleInfo(valid=True, end=end, final_balance=self.simulator.balance_at(end - 1))
        elif contract.annotation['seller'] == self.manager_id:
            # I am a seller

            # if enough is available in storage and not reserved, just sell it
            q_needed = q - simulator.available_storage_at(t)[pid]
            if q_needed <= 0:
                if simulator.sell(product=pid, quantity=q, price=p, t=t, ignore_money_shortage=ignore_failures
                    , ignore_inventory_shortage=ignore_failures):
                    return ScheduleInfo(end=end, final_balance=self.simulator.balance_at(end - 1))
                else:
                    return ScheduleInfo(end=end, valid=False, failed_contracts=[contract],
                                        final_balance=INVALID_UTILITY)
            jobs: List[Job] = []
            needs: List[ProductionNeed] = []

            with transaction(simulator) as bookmark:
                some_production = True
                while q_needed > 0 and some_production:
                    some_production = False
                    # I need now to schedule the production needed and calculate all required input products
                    for info in self.producing[pid]:
                        # find if it is possible to use the current process for producing the product
                        profile = self.profiles[info.profile]
                        line, process_index, profile_index = profile.line, profile.process, info.profile
                        q_produced, t_production = info.quantity, info.step
                        current_schedule = simulator.line_schedules_to(t - ensure_storage_for - 1)[line][start:]
                        if len(current_schedule) < t_production:
                            continue
                        locs = zero_runs((current_schedule != NO_PRODUCTION).astype(int))
                        lengths = locs[:, 1] - locs[:, 0]
                        indices = np.array(range(len(lengths)))
                        indices = indices[lengths >= t_production]
                        if len(indices) < 1:
                            continue
                        lengths, locs = lengths[indices], locs[indices]
                        if self.strategy == 'earliest':
                            loc = locs[0, :]
                        elif self.strategy == 'latest':
                            loc = locs[-1, :] - 1
                            loc[0] = loc[1] - t_production + 1
                        elif self.strategy == 'shortest':
                            sorted_lengths = sorted(zip(range(len(lengths)), lengths), key=lambda x: x[1])
                            loc = locs[sorted_lengths[0][0], :]
                        elif self.strategy == 'longest':
                            sorted_lengths = sorted(zip(range(len(lengths)), lengths), key=lambda x: x[1], reverse=True)
                            loc = locs[sorted_lengths[0][0], :]
                        else:
                            raise ValueError(f'Unknown production strategy {self.strategy}')
                        ptime = loc[0] + start
                        job = Job(line=line, action='run', time=ptime, profile=profile_index
                                  , contract=contract, override=False)
                        if not simulator.schedule(job, override=False, ignore_inventory_shortage=ignore_failures
                            , ignore_money_shortage=ignore_failures
                            , ignore_space_shortage=ignore_failures):
                            continue  # should never hit this
                        jobs.append(job)
                        # find the needs
                        new_needs = []
                        process = self.processes[process_index]
                        length = profile.n_steps
                        for i in process.inputs:
                            pind, quantity = i.product, i.quantity
                            # I need the input to be available the step before production
                            step = ptime + int(math.floor(i.step * length)) - 1
                            if step < 0:
                                break
                            available = max(0, self.simulator.available_storage_at(step)[pind] - quantity)
                            if available >= quantity:
                                instore, tobuy = quantity, 0
                            else:
                                instore, tobuy = available, quantity - available
                            if tobuy > 0 or instore > 0:
                                if step < start:
                                    break
                                needs.append(ProductionNeed(product=pind, needed_for=contract
                                                            , quantity_in_storage=instore, quantity_to_buy=tobuy
                                                            , step=step))
                        else:  # all inputs can be secured in time
                            # @todo consider stopping production after the product is available (+ ensure_storage_for) if needed
                            q_needed -= q_produced
                            some_production = True
                            break
                if q_needed <= 0:
                    # add the effect of selling
                    if not simulator.sell(product=pid, quantity=q, price=p, t=t, ignore_money_shortage=ignore_failures
                        , ignore_inventory_shortage=ignore_failures):
                        simulator.rollback(bookmark)
                        return ScheduleInfo(end=end, valid=False, failed_contracts=[contract],
                                            final_balance=INVALID_UTILITY)

                    # add the effect of buying raw materials
                    for need in needs:
                        product_index = need.product
                        product = self.products[product_index]
                        catalog_price = product.catalog_price
                        if catalog_price == 0 or need.quantity_to_buy <= 0:
                            continue
                        price = need.quantity_to_buy * catalog_price
                        simulator.pay(price, t=need.step)

                    # create schedule
                    schedule = ScheduleInfo(jobs=jobs, end=end, needs=needs, failed_contracts=[]
                                            , final_balance=self.simulator.balance_at(end - 1))
                    return schedule
                simulator.rollback(bookmark)
                return ScheduleInfo(valid=False, failed_contracts=[contract], end=end
                                    , final_balance=self.simulator.balance_at(end - 1))
        raise ValueError(f'{self.manager_id} Not a seller of a buyer in Contract: {contract} with '
                         f'annotation: {contract.annotation}')

    def schedule_contracts(self, contracts: Collection[Contract], end: int = None
                           , assume_no_further_negotiations=False
                           , ensure_storage_for: int = 0
                           , start_at: int = 0) -> ScheduleInfo:
        """
        Schedules a set of contracts and returns the `ScheduleInfo`.

        Args:
            contracts: Contracts to schedule
            assume_no_further_negotiations: If true, no further negotiations will be assumed to be possible
            end: The end of the simulation for the schedule (exclusive)
            ensure_storage_for: Ensure that the outcome will be at the storage for at least this time
            start_at: The timestep at which to start scheduling

        Returns:

            ScheduleInfo giving the schedule after these contracts is included. `valid` member can be used to check
            whether this is a valid contract

        """
        simulator = self.simulator
        if end is None:
            end = simulator.n_steps
        result = ScheduleInfo(valid=True, end=end, final_balance=self.simulator.final_balance)
        contracts = sorted(contracts, key=lambda x: x.agreement['time'])
        for contract in contracts:
            new_schedule = self.schedule_contract(contract, end=end, ensure_storage_for=ensure_storage_for
                                                  , assume_no_further_negotiations=assume_no_further_negotiations
                                                  , start_at=start_at)
            result.combine(new_schedule)
            if new_schedule.valid:
                result.final_balance = self.simulator.final_balance
            else:
                result.final_balance = INVALID_UTILITY
        return result

    def find_schedule(self, contracts: Collection[Contract]
                      , start: int, end: int
                      , assume_no_further_negotiations=False
                      , ensure_storage_for: int = 0
                      , start_at: int = 0):

        # Now, schedule the contracts
        schedule = self.schedule_contracts(contracts=contracts, end=end, ensure_storage_for=ensure_storage_for
                                           , assume_no_further_negotiations=assume_no_further_negotiations
                                           , start_at=start_at)

        # Mark the schedule as invalid if it has any production needs and we assume_no_further_negotiations
        if assume_no_further_negotiations and schedule.needs is not None and len(schedule.needs) > 0:
            schedule.valid = False
            return schedule

        return schedule
