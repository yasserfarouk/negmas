import itertools
import math
import sys
from collections import defaultdict
from random import shuffle, random, randint, sample, choices
from typing import Optional, Callable, Type, Sequence, Dict, Tuple, Iterable, Any, Union, Set, \
    Collection, List

import numpy as np

from negmas import MechanismInfo
from negmas.events import Event, EventSource
from negmas.helpers import snake_case, instantiate, unique_name
from negmas.outcomes import Issue
from negmas.situated import AgentWorldInterface, World, Breach, Action, BreachProcessing, Contract, Agent
from .bank import DefaultBank
from .common import *
from .consumers import ScheduleDrivenConsumer, ConsumptionProfile, Consumer
from .factory_managers import GreedyFactoryManager, FactoryManager
from .insurance import DefaultInsuranceCompany
from .miners import ReactiveMiner, MiningProfile, Miner

__all__ = [
    'SCMLWorld'
]


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


class SCMLWorld(World):
    """The `World` class running a simulation of supply chain management."""

    def __init__(self
                 , products: Collection[Product]
                 , processes: Collection[Process]
                 , factories: List[Factory]
                 , consumers: List[Consumer]
                 , miners: List[Miner]
                 , factory_managers: Optional[List[FactoryManager]] = None
                 # timing parameters
                 , n_steps=60
                 , time_limit=60 * 90
                 , neg_n_steps=100
                 , neg_time_limit=3 * 60
                 , neg_step_time_limit=60
                 , negotiation_speed=10
                 # bank parameters
                 , minimum_balance=0
                 , interest_rate=0.1
                 , interest_max=0.3
                 , installment_interest=0.2
                 , interest_time_increment=0.02
                 , balance_at_max_interest=None
                 # loan parameters
                 , loan_installments=1
                 # insurance company parameters
                 , premium=0.1
                 , premium_time_increment=0.1
                 , premium_breach_increment=0.1
                 # breach processing
                 , max_allowed_breach_level=None
                 , breach_processing=BreachProcessing.VICTIM_THEN_PERPETRATOR
                 , breach_penalty_society=0.1
                 , breach_penalty_society_min=0.0
                 , breach_penalty_victim=0.0
                 , breach_move_max_product=True
                 # simulation parameters
                 , initial_wallet_balances: Optional[int] = None
                 , money_resolution=0.5
                 , default_signing_delay=0
                 , transportation_delay: int = 1
                 , transfer_delay: int = 0
                 , start_negotiations_immediately=False
                 , catalog_profit=0.15
                 , avg_process_cost_is_public=True
                 , catalog_prices_are_public=True
                 , strip_annotations=True
                 # general parameters
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
            initial_wallet_balances: If not none, all factories will be forced to have this initial wallet balance
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
            strip_annotations: If true, annotations for all negotiations will be stripped from any information other
            than the following: partners, seller, buyer, cfp
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
                         , neg_step_time_limit=neg_step_time_limit
                         , name=name)
        if balance_at_max_interest is None:
            balance_at_max_interest = initial_wallet_balances
        self.strip_annotations = strip_annotations
        self.contracts: Dict[int, Set[Contract]] = defaultdict(set)
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
        self.bulletin_board.record(section='settings', key='negotiation_step_time_limit', value=neg_step_time_limit)
        self.bulletin_board.record(section='settings', key='negotiation_time_limit', value=neg_time_limit)
        self.bulletin_board.record(section='settings', key='transportation_delay', value=transportation_delay)
        self.avg_process_cost_is_public = avg_process_cost_is_public
        self.catalog_prices_are_public = catalog_prices_are_public
        self.initial_wallet_balances = initial_wallet_balances
        self.products: List[Product] = []
        self.processes: List[Process] = []
        self.factory_managers: List[FactoryManager] = []
        self.miners: List[Miner] = []
        self.consumers: List[Consumer] = []
        self.set_products(products)
        self.set_processes(processes)
        self.factories = factories
        self.set_miners(miners)
        self.set_consumers(consumers)
        self.set_factory_managers(factory_managers)
        self.money_resolution = money_resolution
        # self._remove_processes_not_used_by_factories()
        # self._remove_products_not_used_by_processes()
        if catalog_prices_are_public or avg_process_cost_is_public:
            self._update_dynamic_product_process_info()

        if initial_wallet_balances is not None:
            for factory in self.factories:
                factory._wallet = initial_wallet_balances

        self.f2a: Dict[str, SCMLAgent] = {}
        self.a2f: Dict[str, Factory] = {}
        assert len(self.factories) == len(self.factory_managers), f'{len(self.factories)} factories and {len(self.factory_managers)} managers'
        for factory, agent in zip(self.factories, self.factory_managers):
            self.f2a[factory.id] = agent
            self.a2f[agent.id] = factory
        for agent in self.consumers:
            factory = Factory(initial_storage={}, initial_wallet=0, profiles=[], min_balance=-np.inf)
            factories.append(factory)
            self.f2a[factory.id] = agent
            self.a2f[agent.id] = factory
        for agent in self.miners:
            factory = Factory(initial_storage={}, initial_wallet=0.0, profiles=[], min_storage=-np.inf)
            factories.append(factory)
            self.f2a[factory.id] = agent
            self.a2f[agent.id] = factory

        self.__interested_agents: List[List[SCMLAgent]] = [[]] * len(self.products)
        self.n_new_cfps = 0
        self._transport: Dict[int, List[Tuple[SCMLAgent, int, int]]] = defaultdict(list)
        self._transfer: Dict[int, List[Tuple[SCMLAgent, float]]] = defaultdict(list)
        self.transfer_delay = transfer_delay

        self._n_production_failures = 0

        self.bank = DefaultBank(minimum_balance=minimum_balance, interest_rate=interest_rate, interest_max=interest_max
                                , balance_at_max_interest=balance_at_max_interest,
                                installment_interest=installment_interest
                                , time_increment=interest_time_increment, a2f=self.a2f
                                , name='bank')
        self.join(self.bank, simulation_priority=-1)
        self.insurance_company = DefaultInsuranceCompany(premium=premium,
                                                         premium_breach_increment=premium_breach_increment
                                                         , premium_time_increment=premium_time_increment, a2f=self.a2f
                                                         , name='insurance_company')
        self.join(self.insurance_company)

        for agent in itertools.chain(self.miners, self.consumers, self.factory_managers):  # type: ignore
            agent.init()
        # self.standing_jobs: Dict[int, List[Tuple[Factory, Job]]] = defaultdict(list)

    def join(self, x: 'Agent', simulation_priority: int = 0):
        """Add an agent to the world.

        Args:
            x: The agent to be registered
            simulation_priority: The simulation priority. Entities with lower priorities will be stepped first during

        Returns:

        """
        super().join(x=x, simulation_priority=simulation_priority)

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
    def single_path_world(cls, n_intermediate_levels=0, n_miners=5, n_factories_per_level=5
                          , n_consumers: Union[int, Tuple[int, int], List[int]] = 5
                          , n_steps=200
                          , n_lines_per_factory=10
                          , log_file_name: str = None
                          , agent_names_reveal_type: bool = False
                          , negotiator_type: str = 'negmas.sao.AspirationNegotiator'
                          , miner_type: Union[str, Type[Miner]] = ReactiveMiner
                          , consumer_type: Union[str, Type[Consumer]] = ScheduleDrivenConsumer
                          , max_storage: int = sys.maxsize
                          , manager_kwargs: Dict[str, Any] = None, miner_kwargs: Dict[str, Any] = None
                          , consumption: Union[int, Tuple[int, int]] = (0, 5)
                          , consumer_kwargs: Dict[str, Any] = None
                          , negotiation_speed: Optional[int] = None
                          , manager_types: Sequence[Type[FactoryManager]] = (GreedyFactoryManager,)
                          , n_default_per_level: int = 0
                          , default_factory_manager_type: Type[FactoryManager] = GreedyFactoryManager
                          , randomize: bool = True
                          , initial_wallet_balances=1000
                          , process_cost: Union[float, Tuple[float, float]] = (1.0, 5.0)
                          , process_time: Union[int, Tuple[int, int]] = 1
                          , interest_rate=float('inf')
                          , interest_max=float('inf')
                          , **kwargs):
        """
        Creates a very small world in which only one raw material and one final product. The production graph is a
        series with `n_intermediate_levels` intermediate levels between the single raw material and single final product

        Args:
            randomize: If true, the factory assignment is randomized
            n_default_per_level: The number of `GreedyFactoryManager` objects guaranteed at every level
            default_factory_manager_type: The `FactoryManager` type to use as the base for default_factory_managers. You
            can specify how many of this type exist at everly level by specifying `n_default_per_level`. If
            `n_default_per_level` is zero, this parameter has no effect.
            manager_types: A sequence of factory manager types to control the factories.
            consumer_type: Consumer type to use for all consumers
            miner_type: Miner type to use for all miners
            consumption: Consumption schedule
            n_intermediate_levels: The number of intermediate products
            n_miners: number of miners of the single raw material
            n_factories_per_level: number of factories at every production level
            n_consumers: number of consumers of the final product
            n_steps: number of simulation steps
            n_lines_per_factory: number of lines in each factory
            process_cost: The range of process costs. A uniform distribution will be used
            process_time: The range of process times. A unifrom distribution will be used
            log_file_name: File name to store the logs
            agent_names_reveal_type: If true, agent names will start with a snake_case version of their type name
            negotiator_type: The negotiation factory used to create all negotiators
            max_storage: maximum storage capacity for all factory negmas If None then it is unlimited
            manager_kwargs: keyword arguments to be used for constructing factory negmas
            consumer_kwargs: keyword arguments to be used for constructing consumers
            miner_kwargs: keyword arguments to be used for constructing miners
            negotiation_speed: The number of negotiation steps per simulation step. None means infinite
            interest_max: Maximum interest rate
            interest_rate: Minimum interest rate
            initial_wallet_balances:  initial wallet balances for all factories
            kwargs: Any other parameters are just passed to the world constructor

        Returns:
            SCMLWorld ready to run

        Remarks:

            - Every production level n has one process only that takes n steps to complete


        """
        if manager_kwargs is None:
            manager_kwargs = {}
        if consumer_kwargs is None:
            consumer_kwargs = {}
        if miner_kwargs is None:
            miner_kwargs = {}
        if negotiator_type is not None:
            for args in (manager_kwargs, consumer_kwargs, miner_kwargs):
                if 'negotiator_type' not in args.keys():
                    args['negotiator_type'] = negotiator_type

        products = [Product(id=0, name='p0', catalog_price=1.0, production_level=0, expires_in=0)]
        processes = []
        miners = [instantiate(miner_type, profiles={products[-1].id: MiningProfile(cv=0)}, name=f'm_{i}'
                              , **miner_kwargs) for i in range(n_miners)]
        factories, managers = [], []
        n_steps_profile = 1

        def _s(x):
            return x if x is not None else 0

        for level in range(n_intermediate_levels + 1):
            new_product = Product(name=f'p{level + 1}', catalog_price=None # keep this to the world to calculate _s(products[-1].catalog_price) + level + 1
                                  , production_level=level + 1, id=level + 1, expires_in=0)
            p = Process(name=f'p{level + 1}', inputs=[InputOutput(product=level, quantity=1, step=0.0)]
                        , production_level=level + 1
                        , outputs=[InputOutput(product=level + 1, quantity=1, step=1.0)]
                        , historical_cost= None # keep this to the world to calculate level + 1
                        , id=level)
            processes.append(p)
            products.append(new_product)

        assignable_factories = []

        _DefaultFactoryManager = default_factory_manager_type

        for level in range(n_intermediate_levels + 1):
            default_factories = []
            for j in range(n_factories_per_level):
                profiles = []
                for k in range(n_lines_per_factory):
                    profiles.append(ManufacturingProfile(n_steps=intin(process_time)
                                                         , cost=realin(process_cost)
                                                         , initial_pause_cost=0
                                                         , running_pause_cost=0
                                                         , resumption_cost=0
                                                         , cancellation_cost=0
                                                         , line=k, process=processes[level]))
                factory = Factory(id=f'f{level + 1}_{j}', max_storage=max_storage, profiles=profiles
                                  , initial_storage={}, initial_wallet=initial_wallet_balances)
                factories.append(factory)
                if j >= n_default_per_level:
                    assignable_factories.append((factory, level))
                else:
                    default_factories.append(factory)
            for j, factory in enumerate(default_factories):
                manager_name = unique_name(base='_default__preassigned__', add_time=False, rand_digits=12)
                manager = _DefaultFactoryManager(name=manager_name, **manager_kwargs)
                if agent_names_reveal_type:
                    manager.name = f'_default__preassigned__{manager.type_name}_{level + 1}_{j}'
                managers.append(manager)
        if randomize:
            shuffle(assignable_factories)
        for j, ((factory, level), manager_type) in enumerate(zip(assignable_factories, itertools.cycle(manager_types))):
            manager_name = unique_name(base='', add_time=False, rand_digits=12)
            manager = manager_type(name=manager_name)
            if agent_names_reveal_type:
                manager.name = f'{manager.type_name}_{level + 1}_{j}'
            managers.append(manager)

        def create_schedule():
            if isinstance(consumption, tuple) and len(consumption) == 2:
                return np.random.randint(consumption[0], consumption[1], n_steps).tolist()
            return consumption

        consumers = [
            instantiate(consumer_type, profiles={products[-1].id: ConsumptionProfile(cv=0, schedule=create_schedule())}
                        , name=f'c_{i}', **consumer_kwargs) for i in range(n_consumers)]

        return SCMLWorld(products=products, processes=processes, factories=factories  # type: ignore
                         , consumers=consumers, miners=miners
                         , factory_managers=managers, initial_wallet_balances=initial_wallet_balances, n_steps=n_steps
                         , interest_rate=interest_rate, interest_max=interest_max, log_file_name=log_file_name
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
               , consumer_types: Union[Type[Consumer], List[Type[Consumer]]] = ScheduleDrivenConsumer
               , miner_types: Union[Type[Miner], List[Type[Miner]]] = ReactiveMiner
               , negotiator_type='negmas.sao.AspirationNegotiator'
               , initial_wallet_balance: Union[float, Tuple[float, float]] = 1000
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
            profiles = []
            if lines_are_similar:
                line_processes = sample(processes, intin(n_processes_per_line))
                profiles.extend([ManufacturingProfile(n_steps=intin(n_production_steps), cost=realin(cost_for_line)
                                                      , initial_pause_cost=0
                                                      , running_pause_cost=0
                                                      , resumption_cost=0
                                                      , cancellation_cost=0, line=i, process=_)
                                 for _ in line_processes])
            else:
                lines = []
                for _ in range(intin(n_lines)):
                    line_processes = sample(processes, intin(n_processes_per_line))
                    profiles.extend([ManufacturingProfile(n_steps=intin(n_production_steps), cost=realin(cost_for_line)
                                                          , initial_pause_cost=0
                                                          , running_pause_cost=0
                                                          , resumption_cost=0
                                                          , cancellation_cost=0, line=i, process=_)
                                     for _ in line_processes])
            factories.append(Factory(profiles=profiles, max_storage=intin(max_storage), initial_storage={}
                                     , initial_wallet=realin(initial_wallet_balance)))

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
                         , initial_wallet_balances=None, **kwargs)

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
        producers: Dict[Product, List[Tuple[Process, int]]] = defaultdict(list)
        consumers: Dict[Product, List[Tuple[Process, int]]] = defaultdict(list)

        def production_record(process_: Process, product_: Product):
            for output_ in process_.outputs:
                if output_.product == product_.id:
                    return process_, output_.quantity
            return None

        def consumption_record(process_: Process, product_: Product):
            for input_ in process_.inputs:
                if input_.product == product_.id:
                    return process_, input_.quantity

        # sort products and processes by the production level. We assume here that production levels are well behaved
        products = sorted(self.products, key=lambda x: x.production_level)
        processes = sorted(self.processes, key=lambda x: x.production_level)

        # find the producers and consumers for every product. This builds the production graph and its inverse
        for product in products:
            for process in processes:
                precord = production_record(process, product)
                if precord is not None:
                    producers[product].append(precord)
                crecord = consumption_record(process, product)
                if crecord is not None:
                    consumers[product].append(crecord)

        # find the average manufacturing profile cost for every process
        for factory in self.factories:
            for profile_index, profile in enumerate(factory.profiles):
                process = profile.process
                if self.avg_process_cost_is_public:
                    process_costs[process].append(profile.cost)

        process_costs_avg = {k: sum(v) / len(v) if len(v) > 0 else math.inf for k, v in process_costs.items()}

        # loop over all products finding the processes that can produce it and add a new cost example for this product
        for product in products:
            for process, quantity in producers[product]:
                if quantity == 0:
                    continue

                # find the total input cost for a process that can produce the current product
                input_cost = 0.0
                for input_ in process.inputs:
                    iproduct = self.products[input_.product]
                    if iproduct.catalog_price is None:
                        if self.catalog_prices_are_public or self.avg_process_cost_is_public:
                            raise ValueError(
                                f'Catalog prices should be public but product {product} is needed for process {process}'
                                f' without a catalog price')
                        return
                    input_cost += iproduct.catalog_price * input_.quantity
                # append a new unit price for this product
                product_costs[product].append((input_cost + process_costs_avg[process]) / quantity)

            # now we have the product cost for all processes producing this product. Average them
            if product_costs.get(product, None) is None or len(product_costs[product]) < 1:
                continue
            avg_cost = sum(product_costs[product]) / len(product_costs[product])
            if product.catalog_price is None:
                product.catalog_price = avg_cost * (1 + self.catalog_profit)

        # update the historical cost for processes if needed
        if self.avg_process_cost_is_public:
            for process in self.processes:
                if process.historical_cost is None:
                    process.historical_cost = process_costs_avg[process]
        else:
            for process in self.processes:
                process.historical_cost = None

        # update the catalog prices by averaging all possible ways to create any product.
        if not self.catalog_prices_are_public:
            for product in self.products:
                product.catalog_price = None

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
            self.logerror(f'{str(action)} received from {agent.id} which is {agent.__class__.__name__} not a FactoryManager')
            return False
        line, profile_index = action.params.get('line', None), action.params.get('profile', None)
        t = action.params.get('time', None)
        contract = action.params.get('contract', None)
        time = action.params.get('time', None)
        override = action.params.get('override', True)
        if (profile_index is None and line is None) or time is None or time < 0 or time > self.n_steps - 1:
            if callback is not None:
                callback(action, False)
            self.logerror(f'{str(action)} from {agent.id}: Neither profile index nor line is given or invalid time')
            return False
        factory = self.a2f[agent.id]
        if factory is None:
            if callback is not None:
                callback(action, False)
            self.logerror(
                f'{str(action)} from {agent.id}: Unknown factory')
            return False
        if profile_index is not None:
            profile = factory.profiles[profile_index]
            if line is not None and profile.line != line:
                if callback is not None:
                    callback(action, False)
                self.logerror(f'{str(action)} from {agent.id}: profile\'s line {profile.line} != given line {line}')
                return False
            line = profile.line
        job = Job(action=action.type, profile=profile_index, line=line, time=t, contract=contract, override=override)
        factory.schedule(job=job, override=override)
        if callback is not None:
            self.logdebug(f'{str(action)} from {agent.id}: Executed successfully')
            callback(action, True)
        return True

    def get_private_state(self, agent: 'Agent') -> Any:
        return self.a2f[agent.id]

    def _simulation_step(self):
        """A step of SCML simulation"""

        # run standing jobs
        # -----------------
        # for factory, job in self.standing_jobs.get(self.current_step, []):
        #    factory.schedule_job(job=job, end=self.n_steps)

        # run factories
        # -------------
        for factory in self.factories:
            manager = self.f2a[factory.id]
            reports = factory.step()
            self.logdebug(f'Factory {factory.id}: money={factory.wallet}'
                          f', storage={str(dict(factory.storage))}'
                          f', loans={factory.loans}')
            for report in reports:
                if not report.is_empty:
                    self.logdebug(f'PRODUCTION>> {manager.name}: {str(report)}')
            failures = []
            for report in reports:
                if report.failed:
                    failures.append(report.failure)
            if len(failures) > 0:
                manager.on_production_failure(failures=failures)

        # finish transportation and money transfer
        # -----------------------------------------
        transports = self._transport.get(self.current_step, [])
        for transport in transports:
            manager, product_id, q = transport
            self.a2f[manager.id].transport_to(product_id, q)

        transfers = self._transfer.get(self.current_step, [])
        for transfer in transfers:
            manager, money = transfer
            self.a2f[manager.id].receive(money)

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
        self._n_production_failures = 0
        pass

    def _post_step_stats(self):
        """Saves relevant stats"""
        self._stats['n_cfps'].append(self.n_new_cfps)
        self.n_new_cfps = 0
        # noinspection PyProtectedMember
        cfps = self.bulletin_board._data['cfps']
        self._stats['n_cfps_on_board_after'].append(len(cfps) if cfps else 0)
        market_size = 0
        self._stats[f'_balance_bank'].append(self.bank.wallet)
        self._stats[f'_balance_society'].append(self.penalties)
        self._stats[f'_balance_insurance'].append(self.insurance_company.wallet)
        self._stats[f'_storage_insurance'].append(sum(self.insurance_company.storage.values()))
        internal_market_size = self.bank.wallet + self.penalties + self.insurance_company.wallet
        for a in itertools.chain(self.miners, self.consumers, self.factory_managers):
            self._stats[f'balance_{a.name}'].append(self.a2f[a.id].balance)
            self._stats[f'storage_{a.name}'].append(self.a2f[a.id].total_storage)
            market_size += self.a2f[a.id].balance
        self._stats['market_size'].append(market_size)
        self._stats['production_failures'].append(self._n_production_failures / len(self.factories)
                                                  if len(self.factories) > 0 else np.nan)
        self._stats['_market_size_total'].append(market_size + internal_market_size)

    def _execute_contract(self, contract: Contract) -> Set[Breach]:
        super()._execute_contract(contract=contract)
        partners: Set[SCMLAgent]
        cfp: CFP
        partners, agreement = set(self.agents[_] for _ in contract.partners), contract.agreement
        cfp = contract.annotation['cfp']  # type: ignore
        breaches = set()
        quantity, unit_price = agreement['quantity'], agreement['unit_price']

        # find out the values for vicitm and social penalties
        penalty_victim = agreement.get('penalty', None)
        if penalty_victim is not None and self.breach_penalty_victim is not None:
            # there is a defined penalty, find its value
            penalty_victim = penalty_victim if penalty_victim is not None else 0.0
            penalty_victim += self.breach_penalty_victim if self.breach_penalty_victim is not None else 0.0
        penalty_society = self.breach_penalty_society

        # ask each partner to confirm the execution
        for partner in partners:
            if not partner.confirm_contract_execution(contract=contract):
                self.logdebug(
                    f'{partner.name} refused execution og Contract {contract.id}')
                breaches.add(Breach(contract=contract, perpetrator=partner.id  # type: ignore
                                    , victims=[_.id for _ in list(partners - {partner})]
                                    , level=1.0, type='refusal'))
        if len(breaches) > 0:
            return breaches

        # all partners agreed to execute the agreement -> execute it.
        pind = cfp.product
        buyer_id, seller_id = contract.annotation['buyer'], contract.annotation['seller']
        seller: SCMLAgent
        buyer: SCMLAgent
        buyer, seller = self.agents[buyer_id], self.agents[seller_id]
        seller_factory, buyer_factory = self.a2f[seller.id], self.a2f[buyer.id]
        product_breach, money_breach, penalty_breach_victim, penalty_breach_society = None, None, None, None
        money = unit_price * quantity

        # first we will try to blindly execute the contract and will fall back to the more complex algorithm checking
        # all possibilities only if that failed
        try:
            if money > 0 or quantity > 0:
                self._move_product(buyer=buyer, seller=seller, quantity=quantity, money=money, product_id=pind)
            else:
                self.logdebug(f'Contract {contract.id} has no transfers')
            return breaches
        except ValueError:
            pass

        # check the seller
        available_quantity = seller_factory.storage.get(pind, 0) if not isinstance(seller, Miner) else quantity
        missing_quantity = max(0, quantity - available_quantity)
        if missing_quantity > 0:
            product_breach = missing_quantity / quantity
            penalty_values = []
            for penalty, is_victim in ((penalty_victim, True), (penalty_society, False)):
                if penalty is None:
                    penalty_values.append(0.0)
                    continue
                # find out how much need to be paid for this penalty
                penalty_value = penalty * product_breach

                # society penalty may have a minimum. If so, make sure the penalty value is at least as large as that
                if not is_victim and self.breach_penalty_society_min is not None:
                    penalty_value = max(penalty_value, self.breach_penalty_society_min)
                penalty_values.append(penalty_value)

            # that is how much money is available in the wallet right now. Notice that loans do not count
            seller_balance = seller_factory.wallet

            # if the seller needs more than what she has in her wallet to pay the penalty, try a loan
            # Loans are mandatory for society penalty but the agent can refuse to pay a victim penalty
            if seller_balance < penalty_value:
                self.bank.buy_loan(agent=seller, amount=penalty_value - seller_balance
                                   , n_installments=self.loan_installments, force=not is_victim)

            for penalty, is_victim, penalty_value in ((penalty_victim, True, penalty_values[0])
                                                      , (penalty_society, False, penalty_values[1])):
                if penalty is None:
                    continue
                # if the seller can pay the penalty then pay it and now there is no breach, otherwise pay as much as
                # possible. Notice that for society penalties, it should always be the case that the whole penalty
                # value is available in the wallet by now as loans are forced.
                paid_for_quantity, paid_penalties = 0, 0.0
                if seller_factory.wallet >= penalty_value:
                    # if there is enough money then pay the whole penalty
                    paid_penalties = penalty_value
                else:
                    # if there is not enough money then pay enough for the maximum number of items that can be paid
                    if unit_price == 0:
                        paid_for_quantity = missing_quantity
                    else:
                        paid_for_quantity = int(math.floor(seller_factory.wallet / unit_price))
                    paid_penalties = unit_price * paid_for_quantity
                    missing_quantity_unpaid_for = missing_quantity - paid_for_quantity
                    if is_victim:
                        penalty_breach_victim = missing_quantity_unpaid_for / quantity
                    else:
                        penalty_breach_society = missing_quantity_unpaid_for / quantity

                # actually pay the payable amount
                self.logdebug(f'Penalty: {seller.name} paid {paid_penalties} to {buyer.name if is_victim else "society"}')
                seller_factory.pay(paid_penalties)
                if is_victim:
                    buyer_factory.receive(paid_penalties)
                    # if we agreed on a penalty and the buyer paid it, then clear the product breach
                    quantity -= paid_for_quantity
                    money -= paid_for_quantity * unit_price
                    missing_quantity -= paid_for_quantity
                    if missing_quantity <= 0:
                        product_breach = None
                    else:
                        product_breach = missing_quantity / quantity
                else:
                    # if this is the society penalty, it does not affect the product_breach
                    self.penalties += paid_penalties

        # pay penalties if there are any. Notice that penalties apply only to to seller. It makes no sense to have a
        # penalty on the buyer who already have no money to pay the contract value anyway.
        if penalty_breach_society is not None:
            breaches.add(Breach(contract=contract, perpetrator=seller.id, victims=[]
                                , level=penalty_breach_society, type='penalty_society', step=self.current_step))
        if penalty_breach_victim is not None:
            breaches.add(Breach(contract=contract, perpetrator=seller.id, victims=[buyer.id]
                                , level=penalty_breach_victim, type='penalty_society', step=self.current_step))

        # check the buyer
        available_money = buyer_factory.wallet if not isinstance(buyer, Consumer) else money
        missing_money = max(0.0, money - available_money)
        if missing_money > 0.0:
            # if the buyer cannot pay, then offer him a loan. The loan is always optional
            self.bank.buy_loan(agent=buyer, amount=missing_money, n_installments=self.loan_installments)
            available_money = buyer_factory.wallet
            missing_money = max(0.0, money - available_money)

        # if there is still missing money after the loan is offered, then create a breach
        if missing_money > 0.0:
            money_breach = missing_money / money

        insured_quantity, insured_quantity_cost = 0, 0.0
        insured_money, insured_money_quantity = 0.0, 0

        if product_breach is not None:
            # apply insurances if they exist
            # register the breach independent of insurance
            breaches.add(Breach(contract=contract, perpetrator=seller.id, victims=[buyer.id]
                                , level=product_breach, type='product', step=self.current_step))
            if self.insurance_company.pay_insurance(contract=contract, perpetrator=seller):
                # if the buyer has an insurance against the seller for this contract, then just give him the missing
                #  quantity and proceed as if the contract was for the remaining quantity. Notice that the breach on the
                #  seller is already registered by this time.

                # the insurance company can give  as much as the buyer can buy. No loan is allowed here as the buyer was
                # already offered a loan earlier because surely if they have a deficit they commited a funds breach
                # buyer_deficit = missing_quantity * unit_price - buyer_factory.wallet
                # if buyer_deficit > 0:
                #     self.bank.buy_loan(agent=buyer, amount=buyer_deficit, n_installments=self.loan_installments)
                if unit_price > 0:
                    insured_quantity = min(missing_quantity, int(buyer_factory.wallet / unit_price))
                else:
                    insured_quantity = missing_quantity
                self.logdebug(f'Insurance: {buyer.name} got {insured_quantity} of {self.products[pind].name} '
                              f'from insurance ({missing_quantity} was missing')
                insured_quantity_cost = insured_quantity * unit_price
                buyer_factory.transport_to(product=pind, quantity=insured_quantity)
                buyer_factory.pay(insured_quantity_cost)
                self.insurance_company.storage[pind] -= insured_quantity
                self.insurance_company.wallet += insured_quantity_cost

            # we will only transfer the remaining quantity.
            missing_quantity -= insured_quantity
            quantity -= insured_quantity
            money -= insured_quantity * unit_price

        if money_breach is not None:
            # apply insurances if they exist.
            breaches.add(Breach(contract=contract, perpetrator=buyer.id, victims=[seller.id]
                                , level=money_breach, type='money', step=self.current_step))
            if self.insurance_company.pay_insurance(contract=contract, perpetrator=buyer):
                # if the seller has an insurance against the buyer for this contract, then just give him the missing
                #  money and proceed as if the contract was for the remaining amount. Notice that the breach on the
                #  seller is already registered by this time.

                # the insurance company will provide enough money to buy whatever actually exist of the contract in the
                # seller's storage
                insured_money = min(missing_money, seller_factory.storage.get(pind, 0) * unit_price)
                insured_money_quantity = int(insured_money // unit_price)  # I never come here if unit_price is zero.
                self.logdebug(f'Insurance: {seller.name} got {insured_money} dollars from insurance')
                seller_factory.receive(insured_money)
                seller_factory.transport_from(product=pind, quantity=insured_money_quantity)
                self.insurance_company.wallet -= insured_money
                self.insurance_company.storage[pind] += insured_money_quantity

            # we will only transfer the remaining money.
            money -= insured_money
            missing_money -= insured_money
            quantity -= insured_money_quantity

        if len(breaches) > 0:
            self.logdebug(f'Contract {contract.id} has {len(breaches)} breaches:')
            for breach in breaches:
                self.logdebug(f'{str(breach)}')

        # missing quantity/money is now fully handled. Just remove them from the contract and execute the rest
        quantity -= missing_quantity + (int(missing_money / unit_price) if unit_price != 0.0 else 0.0)
        money -= missing_money + missing_quantity * unit_price

        # confirm that the money and quantity match given the unit price.
        assert money == unit_price * quantity, f'invalid contract!! money {money}, quantity {quantity}' \
            f', unit price {unit_price}, missing quantity {missing_quantity}, missing money {missing_money}' \
            f', breaches: {[str(_) for _ in breaches]}, insured_quantity {insured_quantity}' \
            f', insured_quantity_cost {insured_quantity_cost}, insured_money {insured_money}' \
            f', insured_money_quantity {insured_money_quantity}, original quantity {agreement["quantity"]}' \
            f', original money {agreement["unit_price"] * agreement["quantity"]}'
        # if money > unit_price * quantity:
        #     money = unit_price * quantity
        # if unit_price != 0.0 and quantity > math.floor(money / unit_price):
        #     quantity = int(math.floor(money / unit_price))

        if money > 0 or quantity > 0:
            perpetrators = set([b.perpetrator for b in breaches])
            execute = True
            victims = {}
            if 0 < len(perpetrators) < len(partners):
                victims = set(partners) - set(perpetrators)
                execute = all(victim.confirm_partial_execution(contract=contract, breaches=list(breaches))
                              for victim in victims)
            if execute:
                self._move_product(buyer=buyer, seller=seller, quantity=quantity, money=money, product_id=pind)
            else:
                self.logdebug(f'Contract {contract.id}: one of {[_.id for _ in victims]} refused partial execution.')
        else:
            self.logdebug(f'Contract {contract.id} has no transfers')
        return breaches

    def _move_product(self, buyer: SCMLAgent, seller: SCMLAgent, product_id: int, quantity: int, money: float):
        """Moves as much product and money between the buyer and seller"""
        seller_factory, buyer_factory = self.a2f[seller.id], self.a2f[buyer.id]
        if quantity > 0:
            seller_factory.transport_from(product_id, quantity)
            if self.transportation_delay < 1:
                buyer_factory.transport_to(product_id, quantity)
            else:
                self._transport[self.current_step + self.transportation_delay].append(
                    (buyer, product_id, quantity))
        if money > 0:
            buyer_factory.pay(money)
            if self.transfer_delay < 1:
                seller_factory.receive(money)
            else:
                self._transfer[self.current_step + self.transfer_delay].append((seller, money))
        self.logdebug(f'Moved {quantity} units of {self.products[product_id].name} from {seller.name} to {buyer.name} '
                      f'for {money} dollars')

    def _complete_contract_execution(self, contract: Contract, breaches: List[Breach], resolved: bool):
        pass

    def _move_product_force(self, buyer: SCMLAgent, seller: SCMLAgent, product_id: int, quantity: int, money: float):
        """Moves as much product and money between the buyer and seller"""
        seller_factory, buyer_factory = self.a2f[seller.id], self.a2f[buyer.id]
        if isinstance(seller, Miner):
            available_quantity = quantity
        else:
            available_quantity = min(quantity, seller_factory.storage.get(product_id, 0))
        if isinstance(buyer, Consumer):
            available_money = money
        else:
            available_money = min(money, buyer_factory.wallet)
        self.logdebug(f'Moving {quantity} (available {available_quantity}) units of '
                      f'{self.products[product_id].name} from {seller.name} '
                      f'to {buyer.name} for {money} (available {available_money}) dollars')
        if available_quantity > 0:
            seller_factory.transport_from(product_id, available_quantity)
            if self.transportation_delay < 1:
                buyer_factory.transport_to(product_id, available_quantity)
            else:
                self._transport[self.current_step + self.transportation_delay].append(
                    (buyer, product_id, available_quantity))
        if available_money > 0:
            buyer_factory.pay(available_money)
            if self.transfer_delay < 1:
                seller_factory.receive(available_money)
            else:
                self._transfer[self.current_step + self.transfer_delay].append((seller, available_money))

    def register_interest(self, agent: SCMLAgent, products: List[int]) -> None:
        for product in products:
            self.__interested_agents[product] = list(set(self.__interested_agents[product] + [agent]))

    def unregister_interest(self, agent: SCMLAgent, products: List[int]) -> None:
        for product in products:
            try:
                self.__interested_agents[product].remove(agent)
            except ValueError:
                pass

    def evaluate_insurance(self, contract: Contract, agent: SCMLAgent, t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breachs committed by others

        Args:

            agent: The agent buying the contract
            contract: hypothetical contract
            t: time at which the policy is to be bought. If None, it means current step
        """
        against = [self.agents[_] for _ in contract.partners if _ != agent.id]
        if len(against) != 1:
            raise ValueError('Cannot find partner while evaluating insurance')
        return self.insurance_company.evaluate_insurance(contract=contract, insured=agent, against=against[0], t=t)

    def buy_insurance(self, contract: Contract, agent: SCMLAgent) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        against = [self.agents[_] for _ in contract.partners if _ != agent.id]
        if len(against) != 1:
            raise ValueError('Cannot find partner while evaluating insurance')
        return self.insurance_company.buy_insurance(contract=contract, insured=agent, against=against[0]) is not None

    def _process_annotation(self, annotation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes an annotation stripping any extra information not allowed if necessary"""
        if annotation is None:
            return {}
        if not self.strip_annotations:
            return annotation
        annotation = {k: v for k, v in annotation.items() if k in ('partners', 'cfp', 'buyer', 'seller')}
        return annotation

    def run_negotiation(self, caller: "Agent"
                        , issues: Collection[Issue]
                        , partners: Collection["Agent"]
                        , roles: Collection[str] = None
                        , annotation: Optional[Dict[str, Any]] = None
                        , mechanism_name: str = None
                        , mechanism_params: Dict[str, Any] = None) -> Optional[Tuple[Contract, MechanismInfo]]:
        annotation = self._process_annotation(annotation)
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
        annotation = self._process_annotation(annotation)
        return super().request_negotiation(req_id=req_id, caller=caller, issues=issues, annotation=annotation
                                           , partners=partners, roles=roles, mechanism_name=mechanism_name
                                           , mechanism_params=mechanism_params)

    @property
    def winners(self):
        """The winners of this world (factory managers with maximum wallet balance"""
        if len(self.factory_managers) < 1:
            return []
        if 0.0 in [self.a2f[_.id].initial_balance for _ in self.factory_managers]:
            balances = sorted(((self.a2f[_.id].balance, _) for _ in self.factory_managers), key=lambda x: x[0]
                              , reverse=True)
        else:
            balances = sorted(((self.a2f[_.id].balance / self.a2f[_.id].initial_balance, _)
                               for _ in self.factory_managers), key=lambda x: x[0], reverse=True)

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
            product = cfp.product
            for m in self.__interested_agents[product]:
                if m.id != cfp.publisher:
                    m.on_new_cfp(cfp)
        elif event.type == 'will_remove_record' and event.data['section'] == 'cfps':
            cfp = event.data['value']
            product = cfp.product
            for m in self.__interested_agents[product]:
                if m.id != cfp.publisher:
                    m.on_remove_cfp(cfp)

    def _contract_record(self, contract: Contract) -> Dict[str, Any]:
        c = {
            'id': contract.id,
            'seller_name': self.agents[contract.annotation['seller']].name,
            'buyer_name': self.agents[contract.annotation['buyer']].name,
            'seller_type': self.agents[contract.annotation['seller']].__class__.__name__,
            'buyer_type': self.agents[contract.annotation['buyer']].__class__.__name__,
            'product_name': self.products[contract.annotation['cfp'].product],
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
            'perpetrator': breach.perpetrator,
            'perpetrator_name': breach.perpetrator,
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

    def _contract_execution_time(self, contract: Contract) -> int:
        """
        Returns the time at which the given contract will start execution
        Args:
            contract:

        Returns:

        """
        return contract.agreement['time']

    def _contract_size(self, contract: Contract) -> float:
        return contract.agreement['unit_price'] * contract.agreement['quantity']

    @property
    def business_size(self) -> float:
        """The total business size defined as the total money transferred within the system"""
        return sum(self.stats["activity_level"])

    @property
    def agreement_rate(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = len(self._saved_contracts)
        return n_contracts / n_negs if n_negs != 0 else np.nan

    @property
    def contract_execution_fraction(self) -> float:
        """Fraction of signed contracts successfully executed"""
        n_executed = sum(self.stats['n_contracts_executed'])
        n_contracts = len(self._saved_contracts)
        return n_executed / n_contracts if n_contracts > 0 else np.nan

    @property
    def breach_rate(self) -> float:
        """Fraction of signed contracts that led to breaches"""
        n_breaches = sum(self.stats['n_breaches'])
        n_contracts = len(self._saved_contracts)
        return n_breaches / n_contracts if n_contracts else np.nan
