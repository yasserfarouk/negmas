import copy
import itertools
import logging
import math
import sys
import uuid
from collections import defaultdict
from random import shuffle, random, randint, sample, choices
from typing import (
    Optional,
    Callable,
    Type,
    Sequence,
    Dict,
    Tuple,
    Iterable,
    Any,
    Union,
    Set,
    Collection,
    List,
)

import numpy as np
import yaml

from negmas import AgentMechanismInterface
from .common import DEFAULT_NEGOTIATOR
from negmas.events import Event, EventSource
from negmas.helpers import instantiate, unique_name
from negmas.outcomes import Issue
from negmas.situated import World, Breach, Action, BreachProcessing, Contract, Agent
from .agent import SCMLAgent
from .bank import DefaultBank
from .common import *
from .consumers import ScheduleDrivenConsumer, ConsumptionProfile, Consumer
from .factory_managers import GreedyFactoryManager, FactoryManager
from .insurance import DefaultInsuranceCompany
from .miners import ReactiveMiner, MiningProfile, Miner

CONSUMER_SIMULATION_PRIORITY = 1
MANAGER_SIMULATION_PRIORITY = 2
MINER_SIMULATION_PRIORITY = 3

__all__ = ["SCMLWorld", "Factory"]


def _realin(rng: Union[Tuple[float, float], float]) -> float:
    """
    Selects a random number within a range if given or the input if it was a float

    Args:
        rng: Range or single value

    Returns:

        the real within the given range
    """
    if isinstance(rng, float):
        return rng
    if abs(rng[1] - rng[0]) < 1e-8:
        return rng[0]
    return rng[0] + random() * (rng[1] - rng[0])


def _intin(rng: Union[Tuple[int, int], int]) -> int:
    """
    Selects a random number within a range if given or the input if it was an int

    Args:
        rng: Range or single value

    Returns:

        the int within the given range
    """
    if isinstance(rng, int):
        return rng
    if rng[0] == rng[1]:
        return rng[0]
    return randint(rng[0], rng[1])


class SCMLWorld(World):
    """The `World` class running a simulation of supply chain management."""

    def __init__(
        self,
        products: Collection[Product],
        processes: Collection[Process],
        factories: List[Factory],
        consumers: List[Consumer],
        miners: List[Miner],
        factory_managers: Optional[List[FactoryManager]] = None
        # timing parameters
        ,
        n_steps=100,
        time_limit=60 * 90,
        neg_n_steps=20,
        neg_time_limit=2 * 60,
        neg_step_time_limit=60,
        negotiation_speed=21
        # bank parameters
        ,
        no_bank=False,
        minimum_balance=0,
        interest_rate=0.1,
        interest_max=0.3,
        installment_interest=0.2,
        interest_time_increment=0.02,
        balance_at_max_interest=None
        # loan parameters
        ,
        loan_installments=1
        # insurance company parameters
        ,
        no_insurance=False,
        premium=0.03,
        premium_time_increment=0.03,
        premium_breach_increment=0.001,
        # breach processing
        max_allowed_breach_level=None,
        breach_processing=BreachProcessing.VICTIM_THEN_PERPETRATOR,
        breach_penalty_society=0.1,
        breach_penalty_society_min=0.0,
        breach_penalty_victim=0.0,
        breach_move_max_product=True
        # simulation parameters
        ,
        initial_wallet_balances: Optional[int] = None,
        money_resolution=0.5,
        default_signing_delay=0,
        transportation_delay: int = 0,
        transfer_delay: int = 0,
        start_negotiations_immediately=False,
        catalog_profit=0.15,
        avg_process_cost_is_public=True,
        catalog_prices_are_public=True,
        strip_annotations=True,
        financial_reports_period=10
        # bankruptcy parameters
        ,
        default_price_for_products_without_one=1,
        compensation_fraction=0.5
        # general parameters
        ,
        log_file_name="",
        log_to_screen: bool = False,
        log_file_level=logging.DEBUG,
        log_screen_level=logging.ERROR,
        log_ufuns_file: str = None,
        save_mechanism_state_in_contract=False,
        compact=False,
        save_signed_contracts: bool = True,
        save_cancelled_contracts: bool = True,
        save_negotiations: bool = True,
        save_resolved_breaches: bool = True,
        save_unresolved_breaches: bool = True,
        name: str = None,
    ):
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

        if compact:
            save_mechanism_state_in_contract = False
            log_file_level = logging.ERROR
            log_screen_level = logging.CRITICAL
            save_cancelled_contracts = (
                save_resolved_breaches
            ) = save_negotiations = False
        self.compact = compact
        super().__init__(
            bulletin_board=None,
            n_steps=n_steps,
            time_limit=time_limit,
            negotiation_speed=negotiation_speed,
            neg_n_steps=neg_n_steps,
            neg_time_limit=neg_time_limit,
            breach_processing=breach_processing,
            log_file_name=log_file_name,
            awi_type="negmas.apps.scml.SCMLAWI",
            default_signing_delay=default_signing_delay,
            start_negotiations_immediately=start_negotiations_immediately,
            neg_step_time_limit=neg_step_time_limit,
            mechanisms={"negmas.sao.SAOMechanism": {}},
            log_to_screen=log_to_screen,
            log_file_level=log_file_level,
            log_screen_level=log_screen_level,
            name=name,
            save_signed_contracts=save_signed_contracts,
            save_negotiations=save_negotiations,
            save_cancelled_contracts=save_cancelled_contracts,
            save_resolved_breaches=save_resolved_breaches,
            save_unresolved_breaches=save_unresolved_breaches,
            log_ufuns_file=log_ufuns_file,
        )

        self.compensation_fraction = compensation_fraction
        self.save_mechanism_state_in_contract = save_mechanism_state_in_contract
        self.default_price_for_products_without_one = (
            default_price_for_products_without_one
        )
        self.agents: Dict[str, SCMLAgent] = {}  # just to help static type checkers
        if balance_at_max_interest is None:
            balance_at_max_interest = initial_wallet_balances
        self.strip_annotations = strip_annotations
        self.contracts: Dict[int, Set[Contract]] = defaultdict(set)
        self.bulletin_board.register_listener(event_type="new_record", listener=self)
        self.bulletin_board.register_listener(
            event_type="will_remove_record", listener=self
        )
        self.bulletin_board.add_section("cfps")
        self.bulletin_board.add_section("products")
        self.bulletin_board.add_section("processes")
        self.bulletin_board.add_section("bankruptcy")
        self.bulletin_board.add_section("reports_time")
        self.bulletin_board.add_section("reports_agent")
        self.minimum_balance = minimum_balance
        self.money_resolution = money_resolution
        self.transportation_delay = transportation_delay
        self.breach_penalty_society = breach_penalty_society
        self.breach_move_max_product = breach_move_max_product
        self.breach_penalty_society_min = breach_penalty_society_min
        self.penalties = 0.0
        self.financial_reports_period = financial_reports_period
        self.max_allowed_breach_level = max_allowed_breach_level
        self.catalog_profit = catalog_profit
        self.loan_installments = loan_installments
        self.breach_penalty_victim = breach_penalty_victim
        self.bulletin_board.record(
            section="settings",
            key="breach_penalty_society",
            value=breach_penalty_society,
        )
        self.bulletin_board.record(
            section="settings", key="breach_penalty_victim", value=breach_penalty_victim
        )
        self.bulletin_board.record(
            section="settings",
            key="immediate_negotiations",
            value=start_negotiations_immediately,
        )
        self.bulletin_board.record(
            section="settings",
            key="negotiation_speed_multiple",
            value=negotiation_speed,
        )
        self.bulletin_board.record(
            section="settings", key="negotiation_n_steps", value=neg_n_steps
        )
        self.bulletin_board.record(
            section="settings",
            key="negotiation_step_time_limit",
            value=neg_step_time_limit,
        )
        self.bulletin_board.record(
            section="settings", key="negotiation_time_limit", value=neg_time_limit
        )
        self.bulletin_board.record(
            section="settings", key="transportation_delay", value=transportation_delay
        )
        self.bulletin_board.record(
            section="settings", key="default_signing_delay", value=default_signing_delay
        )
        self.bulletin_board.record(
            section="settings",
            key="breach_penalty_society_min",
            value=breach_penalty_society_min,
        )
        self.bulletin_board.record(
            section="settings",
            key="financial_reports_period",
            value=financial_reports_period,
        )
        self.bulletin_board.record(
            section="settings", key="transfer_delay", value=transfer_delay
        )
        self.bulletin_board.record(section="settings", key="n_steps", value=n_steps)
        self.bulletin_board.record(
            section="settings", key="time_limit", value=time_limit
        )
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

        self._report_receivers: Dict[str, Set[SCMLAgent]] = defaultdict(set)
        # self._remove_processes_not_used_by_factories()
        # self._remove_products_not_used_by_processes()
        if catalog_prices_are_public or avg_process_cost_is_public:
            self._update_dynamic_product_process_info()

        if initial_wallet_balances is not None:
            for factory in self.factories:
                factory._wallet = initial_wallet_balances

        self.f2a: Dict[str, SCMLAgent] = {}
        self.a2f: Dict[str, Factory] = {}
        for factory, agent in zip(self.factories, self.factory_managers):
            self.f2a[factory.id] = agent
            self.a2f[agent.id] = factory
        for agent in self.consumers:
            factory = Factory(
                initial_storage={}, initial_wallet=0, profiles=[], min_balance=-np.inf
            )
            factories.append(factory)
            self.f2a[factory.id] = agent
            self.a2f[agent.id] = factory
        for agent in self.miners:
            factory = Factory(
                initial_storage={}, initial_wallet=0.0, profiles=[], min_storage=-np.inf
            )
            factories.append(factory)
            self.f2a[factory.id] = agent
            self.a2f[agent.id] = factory

        self.__interested_agents: List[List[SCMLAgent]] = [[]] * len(self.products)
        self.n_new_cfps = 0
        self.__n_nullified = 0
        self._transport: Dict[int, List[Tuple[SCMLAgent, int, int]]] = defaultdict(list)
        self._transfer: Dict[int, List[Tuple[SCMLAgent, float]]] = defaultdict(list)
        self.transfer_delay = transfer_delay

        self._n_production_failures = 0

        self.bank = DefaultBank(
            minimum_balance=minimum_balance,
            interest_rate=interest_rate,
            interest_max=interest_max,
            balance_at_max_interest=balance_at_max_interest,
            installment_interest=installment_interest,
            time_increment=interest_time_increment,
            a2f=self.a2f,
            disabled=no_bank,
            name="bank",
        )
        self.join(self.bank, simulation_priority=-1)
        self.insurance_company = DefaultInsuranceCompany(
            premium=premium,
            disabled=no_insurance,
            premium_breach_increment=premium_breach_increment,
            premium_time_increment=premium_time_increment,
            a2f=self.a2f,
            name="insurance_company",
        )
        self.join(self.insurance_company)

        self.all_agents = {}
        for agent in itertools.chain(
            self.miners, self.consumers, self.factory_managers
        ):  # type: ignore
            agent.init_()
            self.all_agents[agent.id] = agent
        # self.standing_jobs: Dict[int, List[Tuple[Factory, Job]]] = defaultdict(list)

    def join(self, x: "Agent", simulation_priority: int = 0):
        """Add an agent to the world.

        Args:
            x: The agent to be registered
            simulation_priority: The simulation priority. Entities with lower priorities will be stepped first during

        Returns:

        """
        super().join(x=x, simulation_priority=simulation_priority)

    def save_config(self, file_name: str) -> None:
        d = {k: v for k, v in self.__dict__.items()}
        d["factory_manager_types"] = {
            manager.id: manager.short_type_name for manager in self.factory_managers
        }
        with open(file_name, "w") as file:
            yaml.safe_dump(d, file)

    def assign_managers(
        self,
        factory_managers=Iterable[Union[str, Type[FactoryManager], FactoryManager]],
        params: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        """
        Assigns existing factories to new factory managers created from the given types and parameters or manager
        objects.

        Args:

            factory_managers: An iterable of  `FactoryManager` objects type names or `FactoryManager` types to assign to
            params: parameters of the newly created managers

        Remarks:

            - factories are assigned in the same order they exist in the local `factories` attribute cycling through
              the input managers or types/params
            - If a `FactoryManager` object is given instead of a type or a string in the `factory_managers` collection,
              and the number of `factory_managers` is less than the number of factories in the world causing this object
              to cycle for more than one factory, it is assigned to the first such factory but then deep copies of it
              with new ids and names are assigned to the rest of the factories. That ensures that each manager has
              exactly one factory and that all factories are assigned exactly one unique manager.
        """
        if params is None:
            params = [dict()]
        # todo add an exit function to agents and call it as they are leaving the world
        self.factory_managers, self.a2f, self.f2a = [], {}, {}
        for factory, (manager_type, manager_params) in zip(
            self.factories,
            itertools.cycle(zip(factory_managers, itertools.cycle(params))),
        ):
            if isinstance(manager_type, FactoryManager):
                manager = manager_type
                if manager.id in self.a2f.keys():
                    manager = copy.deepcopy(manager)
                    manager.id = uuid.uuid4()
                    manager.name = unique_name(
                        manager.name, add_time=False, rand_digits=4
                    )
            else:
                manager = instantiate(manager_type, **manager_params)
            self.join(manager, simulation_priority=MANAGER_SIMULATION_PRIORITY)
            self.factory_managers.append(manager)
            self.a2f[manager.id] = factory
            self.f2a[factory.id] = manager
            manager.init_()

    @classmethod
    def random_small(
        cls,
        n_production_levels: int = 1,
        n_factories: int = 10,
        factory_kwargs: Dict[str, Any] = None,
        miner_kwargs: Dict[str, Any] = None,
        consumer_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        return cls.random(
            n_raw_materials=3,
            raw_material_price=(1, 3),
            n_final_products=3,
            n_production_levels=n_production_levels,
            n_products_per_level=4,
            n_processes_per_level=2,
            n_inputs_per_process=(1, 2),
            bias_toward_last_level_products=1.0,
            quantity_per_input=1,
            quantity_per_output=1,
            process_relative_cost=0.15,
            n_outputs_per_process=1,
            n_lines=2,
            lines_are_similar=True,
            n_processes_per_line=None,
            cost_for_line=(1.0, 5.0),
            n_production_steps=(1, 5),
            n_factories=n_factories,
            n_consumers=1,
            n_products_per_consumer=None,
            n_miners=1,
            n_products_per_miner=None,
            factory_kwargs=factory_kwargs,
            miner_kwargs=miner_kwargs,
            consumer_kwargs=consumer_kwargs,
            **kwargs,
        )

    @classmethod
    def chain_world(
        cls,
        n_intermediate_levels=0,
        n_miners=5,
        n_factories_per_level=5,
        n_consumers: Union[int, Tuple[int, int], List[int]] = 5,
        n_steps=100,
        n_lines_per_factory=10,
        n_max_assignable_factories=None,
        log_file_name: str = None,
        agent_names_reveal_type: bool = False,
        negotiator_type: str = DEFAULT_NEGOTIATOR,
        miner_type: Union[str, Type[Miner]] = ReactiveMiner,
        consumer_type: Union[str, Type[Consumer]] = ScheduleDrivenConsumer,
        max_storage: int = sys.maxsize,
        default_manager_params: Dict[str, Any] = None,
        miner_kwargs: Dict[str, Any] = None,
        consumption: Union[int, Tuple[int, int]] = (0, 5),
        consumer_kwargs: Dict[str, Any] = None,
        negotiation_speed: Optional[int] = None,
        manager_types: Sequence[Type[FactoryManager]] = (GreedyFactoryManager,),
        manager_params: Optional[Sequence[Dict[str, Any]]] = None,
        n_default_per_level: int = 0,
        default_factory_manager_type: Type[FactoryManager] = GreedyFactoryManager,
        randomize: bool = True,
        initial_wallet_balances=1000,
        process_cost: Union[float, Tuple[float, float]] = (1.0, 5.0),
        process_time: Union[int, Tuple[int, int]] = 1,
        interest_rate=float("inf"),
        interest_max=float("inf"),
        **kwargs,
    ):
        """
        Creates a very small world in which only one raw material and one final product. The production graph is a
        series with `n_intermediate_levels` intermediate levels between the single raw material and single final product

        Args:
            n_max_assignable_factories: The maximum number of factories assigned to managers other than the default
            randomize: If true, the factory assignment is randomized
            n_default_per_level: The number of `GreedyFactoryManager` objects guaranteed at every level
            default_factory_manager_type: The `FactoryManager` type to use as the base for default_factory_managers. You
                                          can specify how many of this type exist at every level by specifying
                                          `n_default_per_level`. If `n_default_per_level` is zero, this parameter has
                                          no effect.
            manager_types: A sequence of factory manager types to control the factories.
            manager_params: An optional sequence of dictionaries giving the parameters to pass to `manager_types`.
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
            process_time: The range of process times. A uniform distribution will be used
            log_file_name: File name to store the logs
            agent_names_reveal_type: If true, agent names will start with a snake_case version of their type name
            negotiator_type: The negotiation factory used to create all negotiators
            max_storage: maximum storage capacity for all factory managers If None then it is unlimited
            default_manager_params: keyword arguments to be used for constructing factory managers
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
        if default_manager_params is None:
            default_manager_params = {}
        if consumer_kwargs is None:
            consumer_kwargs = {}
        if miner_kwargs is None:
            miner_kwargs = {}
        if manager_params is None and len(manager_types) > 0:
            manager_params = [dict() for _ in range(len(manager_types))]
        if negotiator_type is not None:
            for args in (default_manager_params, consumer_kwargs, miner_kwargs):
                if "negotiator_type" not in args.keys():
                    args["negotiator_type"] = negotiator_type

        products = [
            Product(
                id=0, name="p0", catalog_price=1.0, production_level=0, expires_in=0
            )
        ]
        processes = []
        miners = [
            instantiate(
                miner_type,
                profiles={products[-1].id: MiningProfile()},
                name=f"m_{i}",
                **miner_kwargs,
            )
            for i in range(n_miners)
        ]
        factories, managers = (
            [],
            [None] * ((n_intermediate_levels + 1) * n_factories_per_level),
        )

        def _s(x):
            return x if x is not None else 0

        for level in range(n_intermediate_levels + 1):
            new_product = Product(
                name=f"p{level + 1}",
                catalog_price=None
                # keep this to the world to calculate _s(products[-1].catalog_price) + level + 1
                ,
                production_level=level + 1,
                id=level + 1,
                expires_in=0,
            )
            p = Process(
                name=f"p{level + 1}",
                inputs=[InputOutput(product=level, quantity=1, step=0.0)],
                production_level=level + 1,
                outputs=[InputOutput(product=level + 1, quantity=1, step=1.0)],
                historical_cost=None,  # keep this to the world to calculate level + 1
                id=level,
            )
            processes.append(p)
            products.append(new_product)

        assignable_factories = []

        _DefaultFactoryManager = default_factory_manager_type

        for level in range(n_intermediate_levels + 1):
            default_factories = []
            for j in range(n_factories_per_level):
                profiles = []
                for k in range(n_lines_per_factory):
                    profiles.append(
                        ManufacturingProfile(
                            n_steps=_intin(process_time),
                            cost=_realin(process_cost),
                            initial_pause_cost=0,
                            running_pause_cost=0,
                            resumption_cost=0,
                            cancellation_cost=0,
                            line=k,
                            process=processes[level],
                        )
                    )
                factory = Factory(
                    id=f"f{level + 1}_{j}",
                    max_storage=max_storage,
                    profiles=profiles,
                    initial_storage={},
                    initial_wallet=initial_wallet_balances,
                )
                factories.append(factory)
                if j >= n_default_per_level and (
                    n_max_assignable_factories is None
                    or len(assignable_factories) < n_max_assignable_factories
                ):
                    assignable_factories.append(j + level * n_factories_per_level)
                else:
                    default_factories.append(j + level * n_factories_per_level)
            for j, indx in enumerate(default_factories):
                manager_name = unique_name(base="_df_", add_time=False, rand_digits=12)
                manager = _DefaultFactoryManager(
                    name=manager_name, **default_manager_params
                )
                if agent_names_reveal_type:
                    manager.name = f"_df_{manager.short_type_name}@{level + 1}_{j}"
                managers[indx] = manager
        if randomize:
            shuffle(assignable_factories)
        for j, (index, (params, manager_type)) in enumerate(
            zip(
                assignable_factories,
                itertools.cycle(zip(manager_params, manager_types)),
            )
        ):
            factory = factories[index]
            manager_name = (
                f"{unique_name(base='', add_time=False, rand_digits=12)}@{factory.id}"
            )
            manager = instantiate(manager_type, name=manager_name, **params)
            if agent_names_reveal_type:
                manager.name = f"{manager.short_type_name}@{factory.id[1:]}"
            managers[index] = manager

        def create_schedule():
            if isinstance(consumption, tuple) and len(consumption) == 2:
                return np.random.randint(
                    consumption[0], consumption[1], n_steps
                ).tolist()
            return consumption

        consumers = [
            instantiate(
                consumer_type,
                profiles={
                    products[-1].id: ConsumptionProfile(schedule=create_schedule())
                },
                name=f"c_{i}",
                **consumer_kwargs,
            )
            for i in range(n_consumers)
        ]
        return SCMLWorld(
            products=products,
            processes=processes,
            factories=factories,
            consumers=consumers,
            miners=miners,
            n_steps=_intin(n_steps),
            factory_managers=managers,
            initial_wallet_balances=initial_wallet_balances,
            interest_rate=interest_rate,
            interest_max=interest_max,
            log_file_name=log_file_name,
            negotiation_speed=negotiation_speed,
            **kwargs,
        )

    @classmethod
    def random(
        cls,
        n_raw_materials: Union[int, Tuple[int, int]] = (5, 10),
        raw_material_price: Union[float, Tuple[float, float]] = (1.0, 30.0),
        n_final_products: Union[int, Tuple[int, int]] = (3, 5),
        n_production_levels: Union[int, Tuple[int, int]] = (3, 5),
        n_products_per_level: Union[int, Tuple[int, int]] = (3, 5),
        n_processes_per_level: Union[int, Tuple[int, int]] = (6, 10),
        n_inputs_per_process: Union[int, Tuple[int, int]] = (2, 5),
        bias_toward_last_level_products: float = 0.0,
        quantity_per_input: Union[int, Tuple[int, int]] = (1, 10),
        input_step: Union[float, Tuple[float, float]] = 0.0,
        quantity_per_output: Union[int, Tuple[int, int]] = (1, 1),
        output_step: Union[float, Tuple[float, float]] = 1.0,
        process_relative_cost: Union[float, Tuple[float, float]] = (0.05, 0.4),
        n_outputs_per_process: Union[int, Tuple[int, int]] = (1, 1),
        n_lines: Union[int, Tuple[int, int]] = (3, 5),
        lines_are_similar: bool = False,
        n_processes_per_line: Union[int, Tuple[int, int]] = None,
        cost_for_line: Union[float, Tuple[float, float]] = (5.0, 50.0),
        n_production_steps: Union[int, Tuple[int, int]] = (2, 10),
        max_storage: Union[int, Tuple[int, int]] = 2000,
        n_factories: Union[int, Tuple[int, int]] = 20,
        n_consumers: Union[int, Tuple[int, int]] = 5,
        n_products_per_consumer: Union[int, Tuple[int, int]] = None,
        n_miners: Union[int, Tuple[int, int]] = 5,
        n_products_per_miner: Optional[Union[int, Tuple[int, int]]] = None,
        factory_manager_types: Union[
            Type[FactoryManager], List[Type[FactoryManager]]
        ] = GreedyFactoryManager,
        consumer_types: Union[
            Type[Consumer], List[Type[Consumer]]
        ] = ScheduleDrivenConsumer,
        miner_types: Union[Type[Miner], List[Type[Miner]]] = ReactiveMiner,
        negotiator_type=DEFAULT_NEGOTIATOR,
        initial_wallet_balance: Union[float, Tuple[float, float]] = 1000,
        factory_kwargs: Dict[str, Any] = None,
        miner_kwargs: Dict[str, Any] = None,
        consumer_kwargs: Dict[str, Any] = None,
        **kwargs,
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
            factory_manager_types: A callable for creating factory managers for the factories
            consumer_types: A callable for creating `Consumer` objects
            miner_types: A callable for creating `Miner` objects
            negotiator_type: A string that can be `eval`uated to a negotiator.
            initial_wallet_balance: The initial balance of all wallets
            factory_kwargs: keyword arguments to be used for constructing factory managers
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
                if "negotiator_type" not in args.keys():
                    args["negotiator_type"] = negotiator_type

        def _sample_product(
            products: list, old_products: list, last_level_products: list, k: int
        ):
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

        products = [
            Product(
                name=f"r_{ind}",
                catalog_price=_realin(raw_material_price),
                production_level=0,
                expires_in=0,
                id=ind,
            )
            for ind in range(_intin(n_raw_materials))
        ]
        raw_materials = products.copy()
        last_level_products = products  # last level of products
        old_products: List[Product] = []  # products not including last level
        processes: List[Process] = []

        def _adjust_level_of_production(new_products, new_processes):
            product_prices: Dict[Product, List[float]] = defaultdict(
                list
            )  # will keep the costs for generating products
            for process in new_processes:
                process.inputs = set(
                    InputOutput(
                        product=_.index,
                        quantity=_intin(quantity_per_input),
                        step=_realin(input_step),
                    )
                    for _ in _sample_product(
                        products=products,
                        old_products=old_products,
                        last_level_products=last_level_products,
                        k=_intin(n_inputs_per_process),
                    )
                )
                process.outputs = set(
                    InputOutput(
                        product=_.index,
                        quantity=_intin(quantity_per_output),
                        step=_realin(output_step),
                    )
                    for _ in sample(new_products, _intin(n_outputs_per_process))
                )
                process.historical_cost = sum(
                    products[_.product].catalog_price * _.quantity
                    for _ in process.inputs
                )
                process.historical_cost *= 1 + _realin(process_relative_cost)
                for output in process.outputs:
                    product_prices[products[output.product]].append(
                        process.historical_cost
                    )

            new_products = [_ for _ in product_prices.keys()]
            for product in new_products:
                product.catalog_price = sum(product_prices[product]) / len(
                    product_prices[product]
                )
            return new_products, new_processes

        n_levels = _intin(n_production_levels)
        if n_levels > 0:
            for level in range(n_levels):
                new_products = [
                    Product(
                        name=f"intermediate_{level}_{ind}",
                        production_level=level + 1,
                        id=len(products) + ind,
                        catalog_price=0,
                        expires_in=0,
                    )
                    for ind in range(_intin(n_products_per_level))
                ]
                new_processes = [
                    Process(
                        name=f"process_{level}_{ind}",
                        production_level=level + 1,
                        id=len(processes) + ind,
                        historical_cost=0,
                        inputs=set(),
                        outputs=set(),
                    )
                    for ind in range(_intin(n_processes_per_level))
                ]
                new_products, new_processes = _adjust_level_of_production(
                    new_products, new_processes
                )
                products += new_products
                old_products += last_level_products
                last_level_products = new_products
                processes += new_processes

            final_products = [
                Product(
                    name=f"f_{ind}",
                    production_level=n_levels + 1,
                    id=len(products) + ind,
                    catalog_price=0,
                    expires_in=0,
                )
                for ind in range(_intin(n_final_products))
            ]
            new_processes = [
                Process(
                    name=f"process_final_{ind}",
                    production_level=n_levels + 1,
                    id=len(processes) + ind,
                    historical_cost=0,
                    inputs=set(),
                    outputs=set(),
                )
                for ind in range(_intin(n_processes_per_level))
            ]

            final_products, new_processes = _adjust_level_of_production(
                final_products, new_processes
            )
            products += final_products
            processes += new_processes
        else:
            final_products = raw_materials

        if n_processes_per_line is None:
            n_processes_per_line = len(processes)

        n_factories = _intin(n_factories)
        factories = []
        for i in range(n_factories):
            profiles = []
            if lines_are_similar:
                line_processes = sample(processes, _intin(n_processes_per_line))
                profiles.extend(
                    [
                        ManufacturingProfile(
                            n_steps=_intin(n_production_steps),
                            cost=_realin(cost_for_line),
                            initial_pause_cost=0,
                            running_pause_cost=0,
                            resumption_cost=0,
                            cancellation_cost=0,
                            line=i,
                            process=_,
                        )
                        for _ in line_processes
                    ]
                )
            else:
                lines = []
                for _ in range(_intin(n_lines)):
                    line_processes = sample(processes, _intin(n_processes_per_line))
                    profiles.extend(
                        [
                            ManufacturingProfile(
                                n_steps=_intin(n_production_steps),
                                cost=_realin(cost_for_line),
                                initial_pause_cost=0,
                                running_pause_cost=0,
                                resumption_cost=0,
                                cancellation_cost=0,
                                line=i,
                                process=_,
                            )
                            for _ in line_processes
                        ]
                    )
            factories.append(
                Factory(
                    profiles=profiles,
                    max_storage=_intin(max_storage),
                    initial_storage={},
                    initial_wallet=_realin(initial_wallet_balance),
                )
            )

        def _ensure_list(x):
            if isinstance(x, Iterable):
                return list(x)
            else:
                return [x]

        miner_types_list, consumer_types_list, factory_manager_types_list = [
            _ensure_list(_)
            for _ in (miner_types, consumer_types, factory_manager_types)
        ]

        factory_managers = [
            current(**factory_kwargs)
            for current in choices(factory_manager_types_list, k=n_factories)
        ]
        miners = [
            current(**miner_kwargs)
            for current in choices(miner_types_list, k=_intin(n_miners))
        ]
        if n_products_per_miner is None:
            n_products_per_miner = len(raw_materials)
        if n_products_per_consumer is None:
            n_products_per_consumer = len(final_products)
        n_products_per_miner = min(n_products_per_miner, len(raw_materials))
        n_products_per_consumer = min(n_products_per_consumer, len(final_products))
        for miner in miners:
            _n = _intin(n_products_per_miner)
            mining_profiles = dict(
                zip(
                    (_.index for _ in sample(raw_materials, _n)),
                    [MiningProfile.random() for _ in range(_n)],
                )
            )
            miner.set_profiles(mining_profiles)
        consumers = [
            current(**consumer_kwargs)
            for current in choices(consumer_types_list, k=_intin(n_consumers))
        ]
        for consumer in consumers:
            _n = _intin(n_products_per_consumer)
            consumer_profiles = dict(
                zip(
                    (_.index for _ in sample(final_products, _n)),
                    [ConsumptionProfile.random() for _ in range(_n)],
                )
            )
            consumer.set_profiles(consumer_profiles)

        return SCMLWorld(
            products=products,
            processes=processes,
            factories=factories,
            consumers=consumers,
            miners=miners,
            factory_managers=factory_managers,
            initial_wallet_balances=None,
            **kwargs,
        )

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

        process_costs_avg = {
            k: sum(v) / len(v) if len(v) > 0 else math.inf
            for k, v in process_costs.items()
        }

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
                        if (
                            self.catalog_prices_are_public
                            or self.avg_process_cost_is_public
                        ):
                            raise ValueError(
                                f"Catalog prices should be public but product {product} is needed for process {process}"
                                f" without a catalog price"
                            )
                        return
                    input_cost += iproduct.catalog_price * input_.quantity
                # append a new unit price for this product
                product_costs[product].append(
                    (input_cost + process_costs_avg[process]) / quantity
                )

            # now we have the product cost for all processes producing this product. Average them
            if (
                product_costs.get(product, None) is None
                or len(product_costs[product]) < 1
            ):
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

        if self.money_resolution is not None:
            for product in self.products:
                if product.catalog_price is not None:
                    product.catalog_price = (
                        math.floor(product.catalog_price / self.money_resolution)
                        * self.money_resolution
                    )
            for process in self.processes:
                if process.historical_cost is not None:
                    process.historical_cost = (
                        math.floor(process.historical_cost / self.money_resolution)
                        * self.money_resolution
                    )

    def set_consumers(self, consumers: List[Consumer]):
        self.consumers = consumers
        for f in consumers:
            self.join(f, simulation_priority=CONSUMER_SIMULATION_PRIORITY)

    def set_miners(self, miners: List[Miner]):
        self.miners = miners
        for f in miners:
            self.join(f, simulation_priority=MINER_SIMULATION_PRIORITY)

    def set_factory_managers(self, factory_managers: Optional[List[FactoryManager]]):
        if factory_managers is None:
            factory_managers = []
        self.factory_managers = factory_managers
        for f in factory_managers:
            self.join(f, simulation_priority=MANAGER_SIMULATION_PRIORITY)

    def set_processes(self, processes: Collection[Process]):
        if processes is None:
            self.processes = []
        else:
            self.processes = list(processes)
        for v in self.processes:
            self.bulletin_board.record("processes", key=str(v), value=v)

    def set_products(self, products: Collection[Product]):
        if products is None:
            self.products = []
        else:
            self.products = list(products)
        for v in self.products:
            self.bulletin_board.record("products", key=str(v), value=v)

    def _contract_execution_order(self, contracts: Collection[Contract]):
        def order(x: Contract):
            o = self.products[x.annotation["cfp"].product].production_level
            if o:
                return o
            return 0

        return sorted(contracts, key=order)

    def execute(
        self,
        action: Action,
        agent: "Agent",
        callback: Callable[[Action, bool], Any] = None,
    ) -> bool:
        if not isinstance(agent, FactoryManager):
            if callback is not None:
                callback(action, False)
            self.logerror(
                f"{str(action)} received from {agent.id} which is {agent.__class__.__name__} not a FactoryManager"
            )
            return False
        factory = self.a2f[agent.id]
        if factory is None:
            if callback is not None:
                callback(action, False)
            self.logerror(f"{str(action)} from {agent.id}: Unknown factory")
            return False
        if action.type == "hide_funds":
            factory.hide_funds(amount=action.params.get("amount", 0.0))
            return True
        elif action.type == "unhide_funds":
            factory.unhide_funds(amount=action.params.get("amount", 0.0))
            return True
        elif action.type in ("hide_product", "hide_products", "hide_inventory"):
            factory.hide_product(
                product=action.params.get("product", -1),
                quantity=action.params.get("quantity", 0),
            )
        elif action.type in ("unhide_product", "unhide_products", "unhide_inventory"):
            factory.unhide_product(
                product=action.params.get("product", -1),
                quantity=action.params.get("quantity", 0),
            )
        elif action.type not in ("run", "start", "stop", "pause", "resume"):
            raise ValueError(f"Unknown action {action.type} received {str(action)}")

        line, profile_index = (
            action.params.get("line", None),
            action.params.get("profile", None),
        )
        t = action.params.get("time", None)
        contract = action.params.get("contract", None)
        time = action.params.get("time", None)
        override = action.params.get("override", True)
        if (
            (profile_index is None and line is None)
            or time is None
            or time < 0
            or time > self.n_steps - 1
        ):
            if callback is not None:
                callback(action, False)
            self.logerror(
                f"{str(action)} from {agent.id}: Neither profile index nor line is given or invalid time"
            )
            return False
        if profile_index is not None:
            profile = factory.profiles[profile_index]
            if line is not None and profile.line != line:
                if callback is not None:
                    callback(action, False)
                self.logerror(
                    f"{str(action)} from {agent.id}: profile's line {profile.line} != given line {line}"
                )
                return False
            line = profile.line
        job = Job(
            action=action.type,
            profile=profile_index,
            line=line,
            time=t,
            contract=contract,
            override=override,
        )
        factory.schedule(job=job, override=override)
        if callback is not None:
            self.logdebug(f"{str(action)} from {agent.id}: Executed successfully")
            callback(action, True)
        return True

    def get_private_state(self, agent: "Agent") -> FactoryState:
        f = self.a2f[agent.id]
        return FactoryState(
            storage=f.storage,
            wallet=f.wallet,
            loans=f.loans,
            max_storage=f.max_storage,
            line_schedules=f.line_schedules,
            hidden_money=f.hidden_money,
            hidden_storage=f.hidden_storage,
            n_lines=f.n_lines,
            profiles=f.profiles,
            next_step=f.next_step,
            commands=f.commands,
            jobs=f.jobs,
        )

    def receive_financial_reports(
        self, agent: SCMLAgent, receive: bool, agents: Optional[List[str]]
    ):
        """Registers interest/disinterest in receiving financial reports"""
        if agents is None:
            agents = self.agents.keys()
        if receive:
            for aid in agents:
                self._report_receivers[aid].add(agent)
        else:
            for aid in agents:
                self._report_receivers[aid].discard(agent)

    def _simulation_step(self):
        """A step of SCML simulation"""

        # publish financial reports
        # -------------------------
        if self.current_step % self.financial_reports_period == 0:
            reports_agent = self.bulletin_board.data["reports_agent"]
            reports_time = self.bulletin_board.data["reports_time"]
            for agent in self.agents.values():
                factory = self.a2f.get(agent.id, None)
                if factory is None:
                    continue
                try:
                    inventory = sum(
                        self.products[product].catalog_price * quantity
                        for product, quantity in factory.storage.items()
                    )
                except ArithmeticError:
                    inventory = None
                report = FinancialReport(
                    agent=agent.id,
                    step=self.current_step,
                    cash=factory.wallet,
                    liabilities=factory.loans,
                    inventory=inventory,
                    credit_rating=self.bank.credit_rating(agent.id),
                )
                if reports_agent.get(agent.id, None) is None:
                    reports_agent[agent.id] = []
                reports_agent[agent.id].append(report)
                if reports_time.get(self.current_step, None) is None:
                    reports_time[self.current_step] = {}
                reports_time[self.current_step][agent.id] = report
                for receiver in self._report_receivers[agent.id]:
                    receiver.on_new_report(report)

        # run standing jobs
        # -----------------
        # for factory, job in self.standing_jobs.get(self.current_step, []):
        #    factory.schedule_job(job=job, end=self.n_steps)

        # run factories
        # -------------
        for factory in self.factories:
            manager = self.f2a[factory.id]
            reports = factory.step()
            if isinstance(manager, FactoryManager):
                self.logdebug(
                    f"{manager.name} (Factory {factory.id}): money={factory.wallet}"
                    f", storage={str(dict(factory.storage))}"
                    f", loans={factory.loans}"
                )
                nonempty = []
                for report in reports:
                    if not report.is_empty:
                        self.logdebug(f"PRODUCTION>> {manager.name}: {str(report)}")
                        if report.finished:
                            nonempty.append(report)
                failures = []
                for report in reports:
                    if report.failed:
                        failures.append(report.failure)
                if len(failures) > 0:
                    manager.on_production_failure(failures=failures)
                else:
                    manager.on_production_success(nonempty)
            else:
                self.logdebug(
                    f"{manager.name}: money={factory.wallet}"
                    f", storage={str(dict(factory.storage))}"
                )

        # finish transportation and money transfer
        # -----------------------------------------
        transports = self._transport.get(self.current_step, [])
        for transport in transports:
            manager, product_id, q = transport
            self.a2f[manager.id].transport_to(product_id, q)
            manager.on_inventory_change(product_id, q, "transport")

        transfers = self._transfer.get(self.current_step, [])
        for transfer in transfers:
            manager, money = transfer
            self.a2f[manager.id].receive(money)
            manager.on_cash_transfer(money, "transfer")

        # remove expired CFPs
        # -------------------
        cfps = self.bulletin_board.query(section="cfps", query=None)
        if cfps is not None:
            # new_cfps = dict()
            toremove = []
            for key, cfp in cfps.items():
                # we remove CFP with a max_time less than *or equal* to current step as all processing for current step
                # should already be complete by now
                if cfp.max_time <= self.current_step:
                    toremove.append(key)
            for key in toremove:
                self.bulletin_board.remove(section="cfps", query=key, query_keys=True)

    def _pre_step_stats(self):
        # noinspection PyProtectedMember
        cfps = self.bulletin_board._data["cfps"]
        self._stats["n_cfps_on_board_before"].append(len(cfps) if cfps else 0)
        self._n_production_failures = 0
        self.__n_nullified = 0
        pass

    def _post_step_stats(self):
        """Saves relevant stats"""
        self._stats["n_cfps"].append(self.n_new_cfps)
        self.n_new_cfps = 0
        # noinspection PyProtectedMember
        cfps = self.bulletin_board._data["cfps"]
        self._stats["n_cfps_on_board_after"].append(len(cfps) if cfps else 0)
        self._stats["n_contracts_nullified"].append(self.__n_nullified)
        market_size = 0
        self._stats[f"_balance_bank"].append(self.bank.wallet)
        self._stats[f"_balance_society"].append(self.penalties)
        self._stats[f"_balance_insurance"].append(self.insurance_company.wallet)
        self._stats[f"_storage_insurance"].append(
            sum(self.insurance_company.storage.values())
        )
        internal_market_size = (
            self.bank.wallet + self.penalties + self.insurance_company.wallet
        )
        for a in itertools.chain(self.miners, self.consumers, self.factory_managers):
            self._stats[f"balance_{a.name}"].append(self.a2f[a.id].balance)
            self._stats[f"storage_{a.name}"].append(self.a2f[a.id].total_storage)
            market_size += self.a2f[a.id].balance
        self._stats["market_size"].append(market_size)
        self._stats["production_failures"].append(
            self._n_production_failures / len(self.factories)
            if len(self.factories) > 0
            else np.nan
        )
        self._stats["_market_size_total"].append(market_size + internal_market_size)

    def _execute_contract(self, contract: Contract) -> Set[Breach]:

        partners, agreement = (
            set(self.agents[_] for _ in contract.partners),
            contract.agreement,
        )
        cfp = contract.annotation["cfp"]  # type: ignore
        breaches = set()
        quantity, unit_price = agreement["quantity"], agreement["unit_price"]
        if quantity < 1 or unit_price < 0.0:
            self.loginfo(
                f"Contract with quantity {quantity} and unit price {unit_price} will be ignored: "
                f"{str(contract)}"
            )
            return breaches
        if unit_price < 1e-7:
            self.logdebug(
                f"Contract with zero unit_price ({unit_price}: {str(contract)}"
            )
        # find out the values for vicitm and social penalties
        penalty_victim = agreement.get("penalty", None)
        if penalty_victim is not None and self.breach_penalty_victim is not None:
            # there is a defined penalty, find its value
            penalty_victim = penalty_victim if penalty_victim is not None else 0.0
            penalty_victim += (
                self.breach_penalty_victim
                if self.breach_penalty_victim is not None
                else 0.0
            )
        penalty_society = self.breach_penalty_society
        penalty_value = 0.0

        # ask each partner to confirm the execution
        for partner in partners:
            if not partner.confirm_contract_execution(contract=contract):
                self.logdebug(
                    f"{partner.name} refused execution og Contract {contract.id}"
                )
                breaches.add(
                    Breach(
                        contract=contract,
                        perpetrator=partner.id,  # type: ignore
                        victims=[_.id for _ in list(partners - {partner})],
                        level=1.0,
                        type="refusal",
                    )
                )
        if len(breaches) > 0:
            return breaches

        # all partners agreed to execute the agreement -> execute it.
        pind = cfp.product
        buyer_id, seller_id = (
            contract.annotation["buyer"],
            contract.annotation["seller"],
        )
        buyer, seller = self.agents[buyer_id], self.agents[seller_id]
        seller_factory, buyer_factory = self.a2f[seller.id], self.a2f[buyer.id]
        product_breach, money_breach, penalty_breach_victim, penalty_breach_society = (
            None,
            None,
            None,
            None,
        )
        money = unit_price * quantity

        # first we will try to blindly execute the contract and will fall back to the more complex algorithm checking
        # all possibilities only if that failed
        try:
            if money > 0 or quantity > 0:
                self._move_product(
                    buyer=buyer,
                    seller=seller,
                    quantity=quantity,
                    money=money,
                    product_id=pind,
                )
            else:
                self.logdebug(f"Contract {contract.id} has no transfers")
            return breaches
        except ValueError:
            pass

        # check the seller
        available_quantity = (
            seller_factory.storage.get(pind, 0)
            if not isinstance(seller, Miner)
            else quantity
        )
        missing_quantity = max(0, quantity - available_quantity)
        if missing_quantity > 0:
            product_breach = missing_quantity / quantity
            penalty_values = []
            for penalty, is_victim in (
                (penalty_victim, True),
                (penalty_society, False),
            ):
                if penalty is None:
                    penalty_values.append(0.0)
                    continue
                # find out how much need to be paid for this penalty
                penalty_value = penalty * product_breach * quantity * unit_price

                # society penalty may have a minimum. If so, make sure the penalty value is at least as large as that
                if not is_victim and self.breach_penalty_society_min is not None:
                    penalty_value = max(penalty_value, self.breach_penalty_society_min)
                penalty_values.append(penalty_value)

            # that is how much money is available in the wallet right now. Notice that loans do not count
            seller_balance = seller_factory.wallet

            # if the seller needs more than what she has in her wallet to pay the penalty, try a loan
            # Loans are mandatory for society penalty but the agent can refuse to pay a victim penalty
            if seller_balance < penalty_value:
                self.bank.buy_loan(
                    agent=seller,
                    amount=penalty_value - seller_balance,
                    n_installments=self.loan_installments,
                    force=True,
                    beneficiary=buyer,
                    contract=contract,
                )

            for penalty, is_victim, penalty_value in (
                (penalty_victim, True, penalty_values[0]),
                (penalty_society, False, penalty_values[1]),
            ):
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
                        paid_for_quantity = int(
                            math.floor(seller_factory.wallet / unit_price)
                        )
                    paid_penalties = unit_price * paid_for_quantity
                    missing_quantity_unpaid_for = missing_quantity - paid_for_quantity
                    if is_victim:
                        penalty_breach_victim = missing_quantity_unpaid_for / quantity
                    else:
                        penalty_breach_society = missing_quantity_unpaid_for / quantity

                # actually pay the payable amount
                self.logdebug(
                    f'Penalty: {seller.name} paid {paid_penalties} to {buyer.name if is_victim else "society"}'
                )
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
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=seller.id,
                    victims=[],
                    level=penalty_breach_society,
                    type="penalty_society",
                    step=self.current_step,
                )
            )
        if penalty_breach_victim is not None:
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=seller.id,
                    victims=[buyer.id],
                    level=penalty_breach_victim,
                    type="penalty",
                    step=self.current_step,
                )
            )

        # check the buyer
        available_money = (
            buyer_factory.wallet if not isinstance(buyer, Consumer) else money
        )
        missing_money = max(0.0, money - available_money)
        if missing_money > 0.0:
            # if the buyer cannot pay, then offer him a loan. The loan is always optional
            self.bank.buy_loan(
                agent=buyer,
                amount=missing_money,
                n_installments=self.loan_installments,
                force=False,
                beneficiary=seller,
                contract=contract,
            )
            available_money = buyer_factory.wallet
            missing_money = max(0.0, money - available_money)

        # if there is still missing money after the loan is offered, then create a breach
        if missing_money > 0.0:
            money_breach = missing_money / money

        if product_breach is not None:
            # register the breach independent of insurance
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=seller.id,
                    victims=[buyer.id],
                    level=product_breach,
                    type="product",
                    step=self.current_step,
                )
            )

        if money_breach is not None:
            # register the breach independent of insurance
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=buyer.id,
                    victims=[seller.id],
                    level=money_breach,
                    type="money",
                    step=self.current_step,
                )
            )

        if len(breaches) > 0:
            return breaches

        # should never arrive here
        assert missing_quantity == 0
        assert missing_money == 0

        if money > 0 or quantity > 0:
            self._move_product(
                buyer=buyer,
                seller=seller,
                quantity=quantity,
                money=money,
                product_id=pind,
            )
        else:
            self.logdebug(f"Contract {contract.id} has no transfers")
        return breaches

    def _move_product(
        self,
        buyer: SCMLAgent,
        seller: SCMLAgent,
        product_id: int,
        quantity: int,
        money: float,
    ):
        """Moves as much product and money between the buyer and seller"""
        seller_factory, buyer_factory = self.a2f[seller.id], self.a2f[buyer.id]
        if quantity > 0:
            seller_factory.transport_from(product_id, quantity)
            seller.on_inventory_change(product_id, -quantity, "contract")
            if self.transportation_delay < 1:
                buyer_factory.transport_to(product_id, quantity)
                buyer.on_inventory_change(product_id, quantity, "contract")
            else:
                self._transport[self.current_step + self.transportation_delay].append(
                    (buyer, product_id, quantity)
                )
        if money > 0:
            buyer_factory.pay(money)
            if self.transfer_delay < 1:
                seller_factory.receive(money)
                seller.on_cash_transfer(money, "contract")
            else:
                self._transfer[self.current_step + self.transfer_delay].append(
                    (seller, money)
                )
        self.logdebug(
            f"Moved {quantity} units of {self.products[product_id].name} from {seller.name} to {buyer.name} "
            f"for {money} dollars"
        )

    def _complete_contract_execution(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ):
        """The resolution can either be None or a contract with the following items:

        The issues can be any or all of the following:

        immediate_quantity: int
        immediate_unit_price: float
        later_quantity: int
        later_unit_price: int
        later_penalty: float
        later_time: int

        """
        partners, agreement = (
            set(self.agents[_] for _ in contract.partners),
            contract.agreement,
        )
        cfp = contract.annotation["cfp"]  # type: ignore
        quantity, unit_price = agreement["quantity"], agreement["unit_price"]
        penalty_victim = agreement.get("penalty", None)
        if penalty_victim is not None and self.breach_penalty_victim is not None:
            # there is a defined penalty, find its value
            penalty_victim = penalty_victim if penalty_victim is not None else 0.0
            penalty_victim += (
                self.breach_penalty_victim
                if self.breach_penalty_victim is not None
                else 0.0
            )
        penalty_society = self.breach_penalty_society

        pind = cfp.product
        buyer_id, seller_id = (
            contract.annotation["buyer"],
            contract.annotation["seller"],
        )
        buyer, seller = self.agents[buyer_id], self.agents[seller_id]
        seller_factory, buyer_factory = self.a2f[seller.id], self.a2f[buyer.id]
        product_breach, money_breach, penalty_breach_victim, penalty_breach_society = (
            None,
            None,
            None,
            None,
        )

        if resolution is None:
            for breach in breaches:
                if breach.type == "product":
                    product_breach = breach.level
                elif breach.type == "money":
                    money_breach = breach.level
                elif breach.type == "penalty":
                    penalty_breach_victim = breach.level
                elif breach.type == "penalty_society":
                    penalty_breach_society = breach.level
        else:
            quantity = resolution.agreement.get("immediate_quantity", quantity)
            unit_price = resolution.agreement.get("immediate_unit_price", unit_price)
            # make and register new contract
            contract.annotation["renegotiation"] = True
            future_agreement = {
                "time": agreement.get("later_time", -1),
                "penalty": agreement.get("later_penalty", -1),
                "quantity": agreement.get("later_quantity", None),
                "unit_price": agreement.get("later_unit_price", 0.0),
            }
            new_contract = Contract(
                partners=contract.partners,
                annotation=contract.annotation,
                issues=contract.issues,
                agreement=SCMLAgreement(**future_agreement),
                concluded_at=self.current_step,
                to_be_signed_at=self.current_step,
                signed_at=self.current_step,
                mechanism_state=None,
                signatures=contract.partners,
            )
            self.on_contract_concluded(new_contract, to_be_signed_at=self.current_step)
            self.on_contract_signed(contract=new_contract)
            for partner in partners:
                partner.on_contract_signed_(contract=new_contract)

        money = unit_price * quantity

        # check the seller
        available_quantity = (
            seller_factory.storage.get(pind, 0)
            if not isinstance(seller, Miner)
            else quantity
        )
        missing_quantity = max(0, quantity - available_quantity)

        # check the buyer
        available_money = (
            buyer_factory.wallet if not isinstance(buyer, Consumer) else money
        )
        missing_money = max(0.0, money - available_money)

        if missing_money > 0.0:
            money_breach = missing_money / money

        insured_quantity, insured_quantity_cost = 0, 0.0
        insured_money, insured_money_quantity = 0.0, 0

        # confirm partial execution
        perpetrators = set([b.perpetrator for b in breaches])
        victims = set()
        if 0 < len(perpetrators) < len(partners):
            victims = set(_.id for _ in partners) - set(perpetrators)
        execute = (
            all(
                self.agents[victim].confirm_partial_execution(
                    contract=contract, breaches=list(breaches)
                )
                for victim in victims
            )
            if len(victims) > 0
            else True
        )

        if product_breach is not None and resolution is None:
            # apply insurances if they exist
            # register the breach independent of insurance
            if self.insurance_company.pay_insurance(
                contract=contract, perpetrator=seller
            ):
                # if the buyer has an insurance against the seller for this contract, then just give him the missing
                #  quantity and proceed as if the contract was for the remaining quantity. Notice that the breach on the
                #  seller is already registered by this time.

                # the insurance company can give  as much as the buyer can buy. No loan is allowed here as the buyer was
                # already offered a loan earlier because surely if they have a deficit they committed a funds breach
                # buyer_deficit = missing_quantity * unit_price - buyer_factory.wallet
                # if buyer_deficit > 0:
                #     self.bank.buy_loan(agent=buyer, amount=buyer_deficit, n_installments=self.loan_installments)
                if unit_price > 0:
                    insured_quantity = min(
                        missing_quantity, int(buyer_factory.wallet / unit_price)
                    )
                else:
                    insured_quantity = missing_quantity
                self.logdebug(
                    f"Insurance: {buyer.name} got {insured_quantity} of {self.products[pind].name} "
                    f"from insurance ({missing_quantity}) was missing"
                )
                insured_quantity_cost = insured_quantity * unit_price
                buyer_factory.transport_to(product=pind, quantity=insured_quantity)
                buyer_factory.pay(insured_quantity_cost)
                buyer.on_inventory_change(pind, insured_quantity, "insurance")
                buyer.on_cash_transfer(-insured_quantity_cost, "insurance")
                self.insurance_company.storage[pind] -= insured_quantity
                self.insurance_company.wallet += insured_quantity_cost

            # we will only transfer the remaining quantity.
            missing_quantity -= insured_quantity
            quantity -= insured_quantity
            money -= insured_quantity * unit_price

        if money_breach is not None and resolution is None:
            # apply insurances if they exist.
            if self.insurance_company.pay_insurance(
                contract=contract, perpetrator=buyer
            ):
                # if the seller has an insurance against the buyer for this contract, then just give him the missing
                #  money and proceed as if the contract was for the remaining amount. Notice that the breach on the
                #  seller is already registered by this time.

                # the insurance company will provide enough money to buy whatever actually exist of the contract in the
                # seller's storage
                insured_money = min(
                    missing_money, seller_factory.storage.get(pind, 0) * unit_price
                )
                insured_money_quantity = int(
                    insured_money // unit_price
                )  # I never come here if unit_price is zero.
                self.logdebug(
                    f"Insurance: {seller.name} got {insured_money} dollars from insurance"
                )
                seller_factory.receive(insured_money)
                seller_factory.transport_from(
                    product=pind, quantity=insured_money_quantity
                )
                seller.on_inventory_change(pind, -insured_money_quantity, "insurance")
                seller.on_cash_transfer(insured_money, "insurance")
                self.insurance_company.wallet -= insured_money
                self.insurance_company.storage[pind] += insured_money_quantity

            # we will only transfer the remaining money.
            money -= insured_money
            missing_money -= insured_money
            quantity -= insured_money_quantity

        # missing quantity/money is now fully handled. Just remove them from the contract and execute the rest
        if missing_money > 0.0 and missing_quantity == 0:
            quantity -= (
                int(math.ceil(missing_money / unit_price)) if unit_price != 0.0 else 0
            )
            money = quantity * unit_price
        elif missing_money <= 0.0 and missing_quantity > 0:
            money -= missing_quantity * unit_price
            quantity = int(math.floor(money / unit_price)) if unit_price != 0.0 else 0
        elif missing_money > 0.0 and missing_quantity > 0:
            money_for_available_quantity = (quantity - missing_quantity) * unit_price
            quantity_for_available_money = int(
                math.floor((money - missing_money) / unit_price)
            )
            money_for_available_money = quantity_for_available_money * unit_price
            available_money = money - missing_money
            money = min(
                (
                    available_money,
                    money_for_available_quantity,
                    money_for_available_money,
                )
            )
            quantity = int(math.floor(money / unit_price)) if unit_price != 0.0 else 0
            money = quantity * unit_price

        # confirm that the money and quantity match given the unit price.
        assert (
            money >= 0.0 and quantity >= 0
        ), f"invalid contract!! negative money ({money}) or quantity ({quantity})"
        assert abs(money - unit_price * quantity) < 1e-5, (
            f"invalid contract!! money {money}, quantity {quantity}"
            f", unit price {unit_price}, missing quantity {missing_quantity}, missing money {missing_money}"
            f", breaches: {[str(_) for _ in breaches]}, insured_quantity {insured_quantity}"
            f", insured_quantity_cost {insured_quantity_cost}, insured_money {insured_money}"
            f', insured_money_quantity {insured_money_quantity}, original quantity {agreement["quantity"]}'
            f', original money {agreement["unit_price"] * agreement["quantity"]}'
        )
        # if money > unit_price * quantity:
        #     money = unit_price * quantity
        # if unit_price != 0.0 and quantity > math.floor(money / unit_price):
        #     quantity = int(math.floor(money / unit_price))

        if money > 0 or quantity > 0:
            if execute:
                self._move_product(
                    buyer=buyer,
                    seller=seller,
                    quantity=quantity,
                    money=money,
                    product_id=pind,
                )
            else:
                self.logdebug(
                    f"Contract {contract.id}: one of {[_.id for _ in victims]} refused partial execution."
                )
        else:
            self.logdebug(f"Contract {contract.id} has no transfers")

    def _move_product_force(
        self,
        buyer: SCMLAgent,
        seller: SCMLAgent,
        product_id: int,
        quantity: int,
        money: float,
    ):
        """Moves as much product and money between the buyer and seller"""
        seller_factory, buyer_factory = self.a2f[seller.id], self.a2f[buyer.id]
        if isinstance(seller, Miner):
            available_quantity = quantity
        else:
            available_quantity = min(
                quantity, seller_factory.storage.get(product_id, 0)
            )
        if isinstance(buyer, Consumer):
            available_money = money
        else:
            available_money = min(money, buyer_factory.wallet)
        self.logdebug(
            f"Moving {quantity} (available {available_quantity}) units of "
            f"{self.products[product_id].name} from {seller.name} "
            f"to {buyer.name} for {money} (available {available_money}) dollars"
        )
        if available_quantity > 0:
            seller_factory.transport_from(product_id, available_quantity)
            seller.on_inventory_change(product_id, -available_quantity, "contract")
            if self.transportation_delay < 1:
                buyer_factory.transport_to(product_id, available_quantity)
                buyer.on_inventory_change(product_id, -available_quantity, "contract")
            else:
                self._transport[self.current_step + self.transportation_delay].append(
                    (buyer, product_id, available_quantity)
                )
        if available_money > 0:
            buyer_factory.pay(available_money)
            if self.transfer_delay < 1:
                seller_factory.receive(available_money)
                seller.on_cash_transfer(available_money, "contract")
            else:
                self._transfer[self.current_step + self.transfer_delay].append(
                    (seller, available_money)
                )

    def register_interest(self, agent: SCMLAgent, products: List[int]) -> None:
        for product in products:
            self.__interested_agents[product] = list(
                set(self.__interested_agents[product] + [agent])
            )

    def unregister_interest(self, agent: SCMLAgent, products: List[int]) -> None:
        for product in products:
            try:
                self.__interested_agents[product].remove(agent)
            except ValueError:
                pass

    def make_bankrupt(
        self,
        agent: SCMLAgent,
        amount: float,
        beneficiary: Agent,
        contract: Optional[Contract],
    ) -> None:
        """Marks the agent as bankrupt"""
        self.bulletin_board.record(
            "bankruptcy", {"time": self.current_step}, key=agent.id
        )
        for receiver in itertools.chain(
            self.miners, self.consumers, self.factory_managers
        ):
            receiver.on_agent_bankrupt(agent.id)
        # liquidate the bankrupt agent
        factory = self.a2f.get(agent.id, None)
        if factory is None:
            return

        # first sell everything the agent has in its factory at catalog prices
        for product_index, quantity in factory.storage.items():
            price = self.products[product_index].catalog_price
            if price is None:
                price = self.default_price_for_products_without_one
            saved_min_storage, factory.min_storage = factory.min_storage, 0
            factory.sell(product=product_index, quantity=quantity, price=price)
            agent.on_inventory_change(product_index, -quantity, "bankruptcy")
            agent.on_cash_transfer(price, "bankruptcy")

        payable = min(factory.wallet, amount)

        # second pay the beneficiary
        if contract is None:
            # beneficiary is the bank
            beneficiary: DefaultBank
            beneficiary.wallet += payable
            factory.pay(payable)
            keep_for_beneficiary = 0
        else:
            # beneficiary is another agent
            keep_for_beneficiary = payable

        # nullify all future contracts
        available = factory.balance - keep_for_beneficiary
        owed = 0.0
        nulled_contracts = []
        for time, contracts in self.contracts.items():
            if time < self.current_step:
                continue
            for contract in contracts:
                if agent.id in contract.partners:
                    victim = [_ for _ in contract.partners if _ != agent.id][0]
                    nulled_contracts.append((victim, contract))
                    owed += (
                        contract.agreement["quantity"]
                        * contract.agreement["unit_price"]
                    )

        # calculate compensation fraction
        if available > owed:
            fraction = self.compensation_fraction
        else:
            fraction = self.compensation_fraction * available / owed

        for victim, contract in nulled_contracts:
            victim = self.all_agents.get(victim, None)
            if victim is None:
                continue
            factory = self.a2f.get(victim.id, None)
            if factory is None:
                continue
            compensation = (
                fraction
                * contract.agreement["quantity"]
                * contract.agreement["unit_price"]
            )
            victim.on_contract_nullified(
                contract=contract, bankrupt_partner=agent.id, compensation=compensation
            )
            factory.receive(compensation)

            self.nullify_contract(contract)

    def nullify_contract(self, contract: Contract):
        self.__n_nullified += 1
        contract.nullified_at = self.current_step

    def evaluate_insurance(
        self, contract: Contract, agent: SCMLAgent, t: int = None
    ) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breachs committed by others

        Args:

            agent: The agent buying the contract
            contract: hypothetical contract
            t: time at which the policy is to be bought. If None, it means current step
        """
        against = [self.agents[_] for _ in contract.partners if _ != agent.id]
        if len(against) != 1:
            raise ValueError("Cannot find partner while evaluating insurance")
        return self.insurance_company.evaluate_insurance(
            contract=contract, insured=agent, against=against[0], t=t
        )

    def buy_insurance(self, contract: Contract, agent: SCMLAgent) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        against = [self.agents[_] for _ in contract.partners if _ != agent.id]
        if len(against) != 1:
            raise ValueError("Cannot find partner while evaluating insurance")
        return (
            self.insurance_company.buy_insurance(
                contract=contract, insured=agent, against=against[0]
            )
            is not None
        )

    def _process_annotation(
        self, annotation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Processes an annotation stripping any extra information not allowed if necessary"""
        if annotation is None:
            return {}
        if not self.strip_annotations:
            return annotation
        annotation = {
            k: v
            for k, v in annotation.items()
            if k in ("partners", "cfp", "buyer", "seller")
        }
        return annotation

    def run_negotiation(
        self,
        caller: "Agent",
        issues: Collection[Issue],
        partners: Collection["Agent"],
        roles: Collection[str] = None,
        annotation: Optional[Dict[str, Any]] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
    ) -> Optional[Tuple[Contract, AgentMechanismInterface]]:
        annotation = self._process_annotation(annotation)
        return super().run_negotiation(
            caller=caller,
            issues=issues,
            annotation=annotation,
            partners=partners,
            roles=roles,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
        )

    def request_negotiation_about(
        self,
        req_id: str,
        caller: "Agent",
        issues: List[Issue],
        partners: List["Agent"],
        roles: List[str] = None,
        annotation: Optional[Dict[str, Any]] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
    ):
        annotation = self._process_annotation(annotation)
        return super().request_negotiation_about(
            req_id=req_id,
            caller=caller,
            issues=issues,
            annotation=annotation,
            partners=partners,
            roles=roles,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
        )

    @property
    def winners(self):
        """The winners of this world (factory managers with maximum wallet balance"""
        if len(self.factory_managers) < 1:
            return []
        if 0.0 in [self.a2f[_.id].initial_balance for _ in self.factory_managers]:
            balances = sorted(
                ((self.a2f[_.id].balance, _) for _ in self.factory_managers),
                key=lambda x: x[0],
                reverse=True,
            )
        else:
            balances = sorted(
                (
                    (self.a2f[_.id].balance / self.a2f[_.id].initial_balance, _)
                    for _ in self.factory_managers
                ),
                key=lambda x: x[0],
                reverse=True,
            )

        max_balance = balances[0][0]
        return [_[1] for _ in balances if _[0] >= max_balance]

    def on_event(self, event: Event, sender: "EventSource") -> None:
        """
        Called whenever an event is raised for which the `World` is registered asa listener

        Args:
            event: The event
            sender: The sender

        Returns:
            None
        """
        if event.type == "new_record" and event.data["section"] == "cfps":
            cfp = event.data["value"]
            product = cfp.product
            for m in self.__interested_agents[product]:
                if m.id != cfp.publisher:
                    m.on_new_cfp(cfp)
        elif event.type == "will_remove_record" and event.data["section"] == "cfps":
            cfp = event.data["value"]
            product = cfp.product
            for m in self.__interested_agents[product]:
                if m.id != cfp.publisher:
                    m.on_remove_cfp(cfp)

    def _contract_record(self, contract: Contract) -> Dict[str, Any]:
        c = {
            "id": contract.id,
            "seller_name": self.agents[contract.annotation["seller"]].name,
            "buyer_name": self.agents[contract.annotation["buyer"]].name,
            "seller_type": self.agents[
                contract.annotation["seller"]
            ].__class__.__name__,
            "buyer_type": self.agents[contract.annotation["buyer"]].__class__.__name__,
            "product_name": self.products[contract.annotation["cfp"].product],
            "delivery_time": contract.agreement["time"],
            "quantity": contract.agreement["quantity"],
            "unit_price": contract.agreement["unit_price"],
            "signed_at": contract.signed_at if contract.signed_at is not None else -1,
            "nullified_at": contract.nullified_at
            if contract.nullified_at is not None
            else -1,
            "concluded_at": contract.concluded_at,
            "penalty": contract.agreement.get("penalty", np.nan),
            "signing_delay": contract.agreement.get("signing_delay", 0),
            "signatures": "|".join(str(_) for _ in contract.signatures),
            "issues": contract.issues if not self.compact else None,
            "seller": contract.annotation["seller"],
            "buyer": contract.annotation["buyer"],
        }
        if not self.compact:
            c.update(contract.annotation)
        c["n_neg_steps"] = contract.mechanism_state.step
        return c

    def _breach_record(self, breach: Breach) -> Dict[str, Any]:
        return {
            "perpetrator": breach.perpetrator,
            "perpetrator_name": breach.perpetrator,
            "level": breach.level,
            "type": breach.type,
            "time": breach.step,
        }

    def on_contract_signed(self, contract: Contract):
        super().on_contract_signed(contract=contract)
        self.contracts[contract.agreement["time"]].add(contract)

    def _get_executable_contracts(self) -> Collection[Contract]:
        """Called at every time-step to get the contracts that are `executable` at this point of the simulation"""
        return self.contracts.get(self.current_step, [])

    def _delete_executed_contracts(self) -> None:
        self.contracts.pop(self.current_step, None)

    def _contract_finalization_time(self, contract: Contract) -> int:
        """
        Returns the time at which the given contract will complete execution
        Args:
            contract:

        Returns:

        """
        return contract.agreement["time"] + self.transportation_delay

    def _contract_execution_time(self, contract: Contract) -> int:
        """
        Returns the time at which the given contract will start execution
        Args:
            contract:

        Returns:

        """
        return contract.agreement["time"]

    def _contract_size(self, contract: Contract) -> float:
        return contract.agreement["unit_price"] * contract.agreement["quantity"]

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
    def cancellation_rate(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = len(self._saved_contracts)
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        return (1.0 - n_signed_contracts / n_contracts) if n_contracts != 0 else np.nan

    @property
    def contract_execution_fraction(self) -> float:
        """Fraction of signed contracts successfully executed"""
        n_executed = sum(self.stats["n_contracts_executed"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        return n_executed / n_signed_contracts if n_signed_contracts > 0 else np.nan

    @property
    def breach_rate(self) -> float:
        """Fraction of signed contracts that led to breaches"""
        n_breaches = sum(self.stats["n_breaches"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        if n_signed_contracts != 0:
            return n_breaches / n_signed_contracts
        return np.nan
