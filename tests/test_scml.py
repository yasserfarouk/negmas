from pprint import pprint
from typing import List, Dict

import numpy as np
import pytest

from negmas.apps.scml import *
from negmas.apps.scml import InputOutput, Job, FactoryStatusUpdate, GreedyScheduler, ProductionFailure
from negmas.situated import Contract


# def test_can_create_a_random_scml_world():
#     world = SCMLWorld.random()
#     assert len(world.products) > 0
#     assert len(world.processes) > 0
#     assert len(world.stats['market_size']) == 0
#     winners = [_.name for _ in world.winners]
#     assert len(winners) == len(world.factory_managers)
#     for x in ('n_contract_executions', 'n_breaches', 'n_cfps_on_board', 'n_cfps'):
#         assert len(world.stats[x]) == 0


# def test_can_run_a_random_small_scml_world():
#     world = SCMLWorld.random_small(log_file_name='', n_steps=200)
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
#     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
#     data.to_csv(f'{logdir()}/contracts.csv')
#
#
# def test_can_run_a_random_small_scml_world_with_delayed_signing():
#     world = SCMLWorld.random_small(log_file_name='', n_steps=20, default_signing_delay=1)
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
#     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
#     data.to_csv(f'{logdir()}/contracts.csv')
#
#
# def test_can_run_a_random_small_scml_world_with_money_resolution():
#     world = SCMLWorld.random_small(log_file_name='', n_steps=20, money_resolution=1)
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
#     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
#     data.to_csv(f'{logdir()}/contracts.csv')


def logdir():
    return '/'.join(__file__.split('/')[:-1])


def test_can_run_a_random_tiny_scml_world():
    world = SCMLWorld.single_path_world(log_file_name='', n_steps=5, n_factories_per_level=1
                                        , consumer_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator', 'consumption_horizon': 2}
                                        # , factory_kwargs={'max_insurance_premium': 100}
                                        , miner_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'}
                                        )
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    winners = [_.name for _ in world.winners]
    # print('Winners: ', winners)
    # print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
    # if len(world.saved_contracts) > 0:
    #     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    #     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    #     data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_no_immediate():
    world = SCMLWorld.single_path_world(log_file_name='', n_steps=20)
    world.run()


def test_can_run_a_random_tiny_scml_world_with_insurance():
    world = SCMLWorld.single_path_world(log_file_name='', n_steps=20
                                        # , factory_kwargs={'max_insurance_premium': 1e6}
                                        , consumer_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'}
                                        , miner_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'})
    world.run()

# @settings(max_examples=50)
# @given(horizon=st.sampled_from([None, 10, 50, 100, 500]),
#        immediate=st.booleans(),
#        signing_delay=st.integers(0, 3),
#        n_factory_levels=st.integers(1, 5),
#        n_factories_per_level=st.integers(1, 5)
#        )
# def test_can_run_a_random_tiny_scml_world_with_n_factories(horizon, immediate, signing_delay
#                                                            , n_factory_levels, n_factories_per_level):
#     n_steps = 500
#     world = SCMLWorld.tiny(n_intermediate_levels=n_factory_levels - 1, log_file_name='', n_steps=n_steps
#                            , n_factories_per_level=n_factories_per_level
#                            , default_signing_delay=signing_delay
#                            , consumer_kwargs={'immediate_cfp_update': immediate, 'consumption_horizon': horizon
#             , 'negotiator_type': 'NiceNegotiator'}
#                            , miner_kwargs={'negotiator_type': 'NiceNegotiator'
#             , 'use_immediate_negotiations': immediate}
#                            , factory_kwargs={'max_insurance_premium': -1
#             , 'use_immediate_negotiations': immediate}
#                            )
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     # assert sum(world.stats['n_contracts_concluded']) > 0
#     # assert sum(world.stats['n_contracts_signed']) > 0
#     # assert len(winners) < len(world.factory_managers)


def test_can_run_a_random_tiny_scml_world_with_linear_production():
    horizon = None
    signing_delay = 0
    n_factory_levels = 0
    n_factories_per_level = 2
    n_steps = 40
    world = SCMLWorld.single_path_world(n_intermediate_levels=n_factory_levels - 1, log_file_name='', n_steps=n_steps
                                        , n_factories_per_level=n_factories_per_level
                                        , default_signing_delay=signing_delay
                                        , consumer_kwargs={'consumption_horizon': horizon
                                              , 'negotiator_type': 'negmas.sao.NiceNegotiator'}
                                        , miner_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'}
                                        )
    world.run()
    assert sum(world.stats['n_contracts_concluded']) > 0
    assert sum(world.stats['n_contracts_signed']) > 0


def test_can_run_a_random_tiny_scml_world_with_no_factory():
    n_steps = 20
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps
                                        , negotiation_speed=None)
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats['n_negotiations'][0] >= n_steps - 1, "at least n negotiations occur where n is the number of " \
                                                            "steps - 1"
    assert world.stats['n_negotiations'][1:] == [0] * (len(world.stats['n_negotiations']) - 1), "All negotiations " \
                                                                                                "happen in step 0"
    assert sum(world.stats['n_contracts_concluded']) >= sum(world.stats['n_contracts_signed']), "some contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_finite_horizon():
    n_steps = 20
    horizon = n_steps // 2
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps
                                        , consumer_kwargs={'consumption_horizon': horizon})
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats['n_negotiations'][0] >= horizon - 1, "at least n negotiations occur where n is the number of " \
                                                            "horizon - 1"
    assert world.stats['n_negotiations'][1:] <= [n_steps] * (len(world.stats['n_negotiations']) - 1)
    assert sum(world.stats['n_contracts_concluded']) >= sum(world.stats['n_contracts_signed']), "all contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_with_delay():
    n_steps = 20
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps, default_signing_delay=1)
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats['n_negotiations'][0] >= n_steps - 1, "at least n negotiations occur where n is the number of " \
                                                            "steps - 1"
    assert world.stats['n_negotiations'][1:] == [0] * (len(world.stats['n_negotiations']) - 1), "All negotiations " \
                                                                                                "happen in step 0"
    assert sum(world.stats['n_contracts_concluded']) >= sum(world.stats['n_contracts_signed']), "all contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_with_delay_no_immediate_neg():
    n_steps = 10
    horizon = 4
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps, default_signing_delay=1
                                        , consumer_kwargs={'immediate_cfp_update': False
            , 'consumption_horizon': horizon})
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats['n_negotiations'][0] >= horizon - 1, "at least n negotiations occur where n is the number of " \
                                                            "horizon - 1"
    assert world.stats['n_negotiations'][1:] <= [n_steps] * (len(world.stats['n_negotiations']) - 1)
    assert sum(world.stats['n_contracts_concluded']) >= world.stats['n_contracts_concluded'][-1] + \
           sum(world.stats['n_contracts_signed']), "some contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


# def test_can_run_a_no_factory_scml_world():
#     world = SCMLWorld.random_small(log_file_name='', n_factories=0
#                                    , n_production_levels=0
#                                    , n_steps=100)
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
#     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
#     data.to_csv(f'{logdir()}/contracts.csv')


@pytest.fixture
def sample_products() -> List[Product]:
    return [Product(name=f'pr_{i}', catalog_price=i + 1, id=i, production_level=i, expires_in=0) for i in range(5)]


@pytest.fixture
def sample_processes(sample_products) -> List[Process]:
    return [Process(id=0, name='p0', inputs={InputOutput(sample_products[0].id, quantity=2, step=0.0),
                                             InputOutput(sample_products[1].id, quantity=3, step=0.0)}
                    , outputs={InputOutput(sample_products[2].id, quantity=1, step=1.0)}
                    , historical_cost=1, production_level=sample_products[1].production_level)
        , Process(id=1, name='p1', inputs={InputOutput(sample_products[1].id, quantity=2, step=0.0),
                                           InputOutput(sample_products[2].id, quantity=3, step=0.0)}
                  , outputs={InputOutput(sample_products[3].id, quantity=1, step=1.0)}
                  , historical_cost=1, production_level=sample_products[2].production_level)
            ]


@pytest.fixture
def sample_profiles(sample_processes) -> Dict[int, ManufacturingProfile]:
    return {2: ManufacturingProfile(n_steps=1, cost=10, cancellation_cost=5
                                    , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)
            , 0: ManufacturingProfile(n_steps=5, cost=5, cancellation_cost=2
                                      , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)
            , 1: ManufacturingProfile(n_steps=3, cost=3, cancellation_cost=0
                                      , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)
            }


@pytest.fixture
def sample_line(sample_profiles, sample_processes) -> Line:
    return Line(profiles=sample_profiles, processes=sample_processes)


@pytest.fixture
def sample_factory(sample_profile, sample_processes, sample_products):
    return Factory(lines=[Line(profiles=sample_profile, processes=sample_processes)]
                   , products=sample_products, processes=sample_processes)


class TestLine:
    
    def test_creation(self, sample_line):
        print(len(sample_line.processes))
        print(len(sample_line.i2p))
        assert len(sample_line.i2p) == len(sample_line.processes) == 2

    def test_init_schedule(self, sample_line):
        sample_line.init_schedule(n_steps=20)
        assert len(sample_line.schedule) == 20
        assert sample_line.schedule.max() == -1
        assert len(sample_line.state.jobs) == 0

    def test_line_start_stop_same_step(self, sample_line, sample_processes):
        sample_line.init_schedule(n_steps=20)
        t = 0
        p = sample_processes[0]
        j0 = Job(process=p.id, time=t, line_name=sample_line.id, command='run', updates={}, contract=None)
        j1 = Job(process=p.id, time=t, line_name=sample_line.id, command='stop', updates={}, contract=None)
        s0 = sample_line.schedule_job(j0)
        assert sample_line.schedule[t] == p.id
        assert len(s0) == 2
        assert tuple(sorted(s0.keys())) == (t, t + sample_line.profiles[p.id].n_steps)
        assert sample_line._state.updates == s0
        s1 = sample_line.schedule_job(j1)
        assert sample_line.schedule[t].max() == -1
        assert len(s1) == 2
        assert tuple(sorted(s1.keys())) == (t, t + sample_line.profiles[p.id].n_steps)
        FactoryStatusUpdate.combine_sets(s0, s1)
        assert len(s0) == 0

    def test_line_start_stop_different_steps(self, sample_line, sample_processes):
        sample_line.init_schedule(n_steps=20)
        t = 0
        t1 = 1
        p = sample_processes[0]
        j0 = Job(process=p.id, time=t, line_name=sample_line.id, command='run', updates={}, contract=None)
        j1 = Job(process=p.id, time=t1, line_name=sample_line.id, command='stop', updates={}, contract=None)
        s0 = sample_line.schedule_job(j0)
        assert sample_line.schedule[t] == p.id
        assert len(s0) == 2
        assert tuple(sorted(s0.keys())) == (t, t + sample_line.profiles[p.id].n_steps)
        assert sample_line._state.updates == s0
        s1 = sample_line.schedule_job(j1)
        assert sample_line.schedule[t].max() == 0
        assert len(s1) == 2
        assert tuple(sorted(s1.keys())) == (t1, t + sample_line.profiles[p.id].n_steps)
        FactoryStatusUpdate.combine_sets(s0, s1)
        assert len(s0) == 2
        assert s0[0].balance == -sample_line.profiles[p.id].cost
        assert s0[1].balance == -sample_line.profiles[p.id].cancellation_cost
        assert (s0[0].storage[_.product] == -_.quantity for _ in p.inputs)
        assert len(s0[1].storage) == 0

    def test_line_stepping_with_success(self, sample_line, sample_processes, sample_products):
        n_steps = 20
        sample_line.init_schedule(n_steps=n_steps)
        t0 = 4
        p = sample_processes[0]
        t1 = t0 + sample_line.profiles[p.id].n_steps
        j0 = Job(process=p.id, time=t0, line_name=sample_line.id, command='run', updates={}, contract=None)
        storage = {sample_products[0].id: 10, sample_products[1].id: 10, sample_products[2].id: 0}
        sample_line.schedule_job(j0)
        for t in range(n_steps):
            result = sample_line.step(t, storage=storage, wallet=1000)
            assert result is not None
            # print(t, result)
            if t == t0:
                assert (result.storage[_.product] == -_.quantity for _ in p.inputs)
                assert result.balance == - sample_line.profiles[p.id].cost
            elif t == t1:
                assert (result.storage[_.product] == _.quantity for _ in p.outputs)
            else:
                assert result.balance == 0
                assert len(result.storage) == 0
            for k, v in result.storage.items():
                storage[k] += v
        assert storage[sample_products[0].id] == 8
        assert storage[sample_products[1].id] == 7
        assert storage[sample_products[2].id] == 1

    def test_line_stepping_with_failure(self, sample_line, sample_processes, sample_products):
        n_steps = 20
        sample_line.init_schedule(n_steps=n_steps)
        t0 = 4
        p = sample_processes[0]
        t1 = t0 + sample_line.profiles[p.id].n_steps
        j0 = Job(process=p.id, time=t0, line_name=sample_line.id, command='run', updates={}, contract=None)
        storage = {sample_products[0].id: 0, sample_products[1].id: 0, sample_products[2].id: 0}
        sample_line.schedule_job(j0)
        for t in range(n_steps):
            result = sample_line.step(t, storage=storage, wallet=1000)
            if t == t0:
                assert isinstance(result, ProductionFailure)
            else:
                assert not isinstance(result, ProductionFailure) and result is not None
                assert result.empty()
                for k, v in result.storage.items():
                    storage[k] += v
        assert storage[sample_products[0].id] == 0
        assert storage[sample_products[1].id] == 0
        assert storage[sample_products[2].id] == 0


class TestGreedyScheduler:

    @classmethod
    def setup_class(cls):
        cls.manager_name = 'test'
        cls.products = [Product(name=f'pr{i}', catalog_price=i + 1, id=i, production_level=i
                                , expires_in=0) for i in range(5)]
        cls.processes = [Process(id=0, name='p0', inputs={InputOutput(0, quantity=2, step=0.0),
                                                          InputOutput(1, quantity=3, step=0.0)}
                                 , outputs={InputOutput(2, quantity=1, step=1.0)}
                                 , historical_cost=1, production_level=1)
            , Process(id=1, name='p1', inputs={InputOutput(2, quantity=3, step=0.0)}
                      , outputs={InputOutput(3, quantity=1, step=1.0)}
                      , historical_cost=1, production_level=2)
            , Process(id=2, name='p2', inputs={InputOutput(2, quantity=3, step=0.0)}
                      , outputs={InputOutput(3, quantity=2, step=1.0)}
                      , historical_cost=3, production_level=2)
            , Process(id=3, name='p3', inputs={InputOutput(3, quantity=3, step=0.0)}
                      , outputs={InputOutput(4, quantity=1, step=1.0)}
                      , historical_cost=3, production_level=3)
                         ]
        cls.profiles = {0: ManufacturingProfile(n_steps=1, cost=10, cancellation_cost=5
                                                , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)
            , 0: ManufacturingProfile(n_steps=5, cost=5, cancellation_cost=2
                                      , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)
            , 1: ManufacturingProfile(n_steps=3, cost=3, cancellation_cost=0
                                      , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)
            , 2: ManufacturingProfile(n_steps=2, cost=5, cancellation_cost=0
                                      , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)
                        }

    @classmethod
    def create_factory(cls, n_lines):
        lines = [Line(profiles=cls.profiles, processes=cls.processes) for _ in range(n_lines)]
        return Factory(lines=lines, products=cls.products, processes=cls.processes)

    @classmethod
    def create_scheduler(cls, n_lines, n_steps, strategy, initial_balance=1000, storage=None):
        factory = cls.create_factory(n_lines)
        if storage:
            factory.init_schedule(n_steps=n_steps, initial_storage=storage, initial_balance=initial_balance)
        return GreedyScheduler(factory=factory, n_steps=n_steps, products=cls.products, processes=cls.processes
                               , manager_id=cls.manager_name, strategy=strategy, awi=None)

    @classmethod
    def buy_contract(cls, t, u, q, p):
        return Contract(agreement=SCMLAgreement(time=t, unit_price=u, quantity=q)
                        , annotation={'cfp': CFP(is_buy=True, product=p, time=t, unit_price=u, quantity=q
                                                 , publisher=cls.manager_name)
                , 'buyer': cls.manager_name, 'seller': 'anyone'})

    @classmethod
    def sell_contract(cls, t, u, q, p):
        return Contract(agreement=SCMLAgreement(time=t, unit_price=u, quantity=q)
                        , annotation={'cfp': CFP(is_buy=True, product=p, time=t, unit_price=u, quantity=q
                                                 , publisher='anyone')
                , 'buyer': 'anyone', 'seller': cls.manager_name})

    def test_initialization(self):
        scheduler = self.create_scheduler(n_lines=3, n_steps=5, strategy='earliest')
        assert len(scheduler.products) == len(self.products)

    def test_scheduling_a_buy_contract(self):
        t = 2
        initial = 1000
        unit_price, quantity = 2, 1
        total = unit_price * quantity
        storage = {0: 10, 1: 10, 2: 0}
        contract = self.buy_contract(p=0, t=t, u=unit_price, q=quantity)
        scheduler = self.create_scheduler(n_lines=1, n_steps=t * 2, storage=storage, strategy='earliest')
        scheduler.schedule([contract])
        factory = scheduler.scheduling_factory
        assert len(factory.predicted_balance) == t + 1
        assert max(abs(factory.predicted_storage[0, :t] - 10)) == 0
        assert max(abs(factory.predicted_storage[0, t:] - 11)) == 0
        assert max(abs(factory.predicted_storage[1, :t] - 10)) == 0
        assert max(abs(factory.predicted_storage[2:].flatten())) == 0
        assert max(abs(factory.predicted_balance[:t] - initial)) == 0
        assert max(abs(factory.predicted_balance[t:] - initial + total)) == 0
        # @todo test the case where end != None

    def test_scheduling_a_sell_contract_no_needs(self):
        t, n_steps = 3, 20
        initial = 1000
        unit_price, quantity = 2, 1
        total = unit_price * quantity
        storage = {0: 10, 1: 10, 2: 0}
        contract = self.sell_contract(p=0, t=t, u=unit_price, q=quantity)
        scheduler = self.create_scheduler(n_lines=1, n_steps=t * 2, storage=storage, strategy='earliest')
        scheduler.schedule([contract])
        factory = scheduler.scheduling_factory

        assert max(abs(factory.predicted_storage[0, :t] - 10)) == 0
        assert max(abs(factory.predicted_storage[0, t:] - 9)) == 0
        assert max(abs(factory.predicted_storage[1, :t] - 10)) == 0
        assert max(abs(factory.predicted_storage[2:].flatten())) == 0
        assert max(abs(factory.predicted_balance[:t] - 1000)) == 0
        assert max(abs(factory.predicted_balance[t:] - 1002)) == 0

    @pytest.mark.parametrize('product_id,t,quantity,valid'
        , [(0, 12, 1, False), (2, 12, 1, True), (2, 12, 2, True), (2, 12, 50, False)]
        , ids=['cannot be produced', 'producing 1 item', 'producing 2 items', 'time is not enough'])
    def test_scheduling_a_sell_contract_with_needs_early(self, product_id, t, quantity, valid):
        products, processes, profiles = self.products, self.processes, self.profiles
        n_steps = 20
        initial = 1000
        unit_price = 2
        total = unit_price * quantity
        process, profile = self.processes[0], self.profiles[0]
        storage = {0: 0, 1: 0, 2: 0}
        contract = self.sell_contract(p=product_id, t=t, u=unit_price, q=quantity)
        scheduler = self.create_scheduler(n_lines=1, n_steps=n_steps, storage=storage, strategy='earliest')
        info = scheduler.schedule([contract])
        factory = scheduler.scheduling_factory

        total_cost = sum(self.products[i.product].catalog_price * i.quantity for i in process.inputs)
        total_cost += profile.cost
        selling_price = contract.agreement['unit_price'] * contract.agreement['quantity']
        production_time = profile.n_steps * contract.agreement['quantity'] // list(process.outputs)[0].quantity

        assert info.valid == valid
        if valid:
            assert len(info.needs) == 2 * quantity
            assert 2 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
            assert 3 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
            if quantity > 1:
                assert 2 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)
                assert 3 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)

            expected_balance = np.array([initial - total_cost] * profile.n_steps +
                                        [initial - quantity * total_cost] * (contract.agreement['time']
                                                                             - profile.n_steps) +
                                        [initial - quantity * total_cost + selling_price])
            expected_storage = np.array([[-2] * profile.n_steps + [-2 * quantity] * (t + 1 - profile.n_steps),
                                         [-3] * profile.n_steps + [-3 * quantity] * (t + 1 - profile.n_steps),
                                         [0] * profile.n_steps + [1] * profile.n_steps +
                                         [quantity] * (t - 2 * profile.n_steps) + [0],
                                         [0] * (t + 1),
                                         [0] * (t + 1)])
            expected_total_storage = expected_storage.sum(axis=0)
            expected_line_schedule = np.array([0] * profile.n_steps * quantity + [-1] * (t + 1
                                                                                         - quantity * profile.n_steps))
        else:
            expected_line_schedule = scheduler.managed_factory.schedule[
                                         list(scheduler.managed_factory.lines.values())[0]][: t+1]
            expected_storage = scheduler.managed_factory.predicted_storage[:, : t+1]
            expected_balance = scheduler.managed_factory.predicted_balance[: t+1]
            expected_total_storage = scheduler.managed_factory.predicted_total_storage[: t+1]

        line_schedule = factory.schedule[list(factory.lines.values())[0]]
        assert np.all(factory.predicted_storage == expected_storage)
        assert np.all(factory.predicted_total_storage == expected_total_storage)
        assert np.all(line_schedule == expected_line_schedule)
        assert np.all(factory.predicted_balance == expected_balance)

    @pytest.mark.parametrize('product_id,t,quantity,valid'
        , [(0, 12, 1, False), (2, 12, 1, True), (2, 12, 2, True), (2, 12, 50, False)]
        , ids=['cannot be produced', 'producing 1 item', 'producing 2 items', 'time is not enough'])
    def test_scheduling_a_sell_contract_with_needs_late(self, product_id, t, quantity, valid):
        products, processes, profiles = self.products, self.processes, self.profiles
        n_steps = 20
        initial = 1000
        unit_price = 2
        total = unit_price * quantity
        process, profile = self.processes[0], self.profiles[0]
        storage = {0: 0, 1: 0, 2: 0}
        contract = self.sell_contract(p=product_id, t=t, u=unit_price, q=quantity)
        scheduler = self.create_scheduler(n_lines=1, n_steps=n_steps, storage=storage, strategy='latest')
        info = scheduler.schedule([contract])
        factory = scheduler.scheduling_factory

        total_cost = sum(self.products[i.product].catalog_price * i.quantity for i in process.inputs)
        total_cost += profile.cost
        selling_price = contract.agreement['unit_price'] * contract.agreement['quantity']
        production_time = profile.n_steps * contract.agreement['quantity'] // list(process.outputs)[0].quantity

        assert info.valid == valid
        if valid:
            assert len(info.needs) == 2 * quantity
            assert 2 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
            assert 3 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
            if quantity > 1:
                assert 2 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)
                assert 3 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)

            def _expected_balance(n_steps=profile.n_steps, total=t + 1):
                s = initial * np.ones(total)
                s[total - 1] += selling_price
                for q in range(quantity):
                    s[-n_steps * (q + 1) - 2:] -= total_cost
                return s.tolist()

            def _expected_storage(n_needed, n_steps=profile.n_steps, total=t+1):
                s = np.zeros(total)
                for q in range(quantity):
                    s[-n_steps * (q + 1) - 2:] -= n_needed
                return s.tolist()

            def _expected_output(total=t+1):
                s = np.zeros(total)
                s[-quantity - 1: -1] = 1
                s[-1] = 0
                return s.tolist()

            def _expected_line(total=t+1, n_steps=profile.n_steps):
                s = -1 * np.ones(total)
                for q in range(quantity):
                    s[-n_steps * (q + 1) - 2:-2] = 0
                return s.tolist()

            expected_balance = _expected_balance()
            expected_storage = np.array([_expected_storage(2), _expected_storage(3), _expected_output()
                                         , [0] * (t + 1), [0] * (t + 1)])
            expected_total_storage = expected_storage.sum(axis=0)
            expected_line_schedule = _expected_line()

            print(f'q={quantity}')
            pprint([str(_) for _ in info.needs])
            pprint(expected_line_schedule)
            pprint(factory.predicted_balance)
            pprint(expected_balance)
            pprint(factory.predicted_storage)
            pprint(expected_storage)
            pprint({k.id: v for k, v in factory.schedule.items()})
            print(f'total_cost: {total_cost}, selling_price: {selling_price}, production_time: {production_time}')
        else:
            expected_line_schedule = scheduler.managed_factory.schedule[
                                         list(scheduler.managed_factory.lines.values())[0]][: t + 1]
            expected_storage = scheduler.managed_factory.predicted_storage[:, : t + 1]
            expected_balance = scheduler.managed_factory.predicted_balance[: t + 1]
            expected_total_storage = scheduler.managed_factory.predicted_total_storage[: t + 1]

        line_schedule = factory.schedule[list(factory.lines.values())[0]]
        assert np.all(factory.predicted_storage == expected_storage)
        assert np.all(factory.predicted_total_storage == expected_total_storage)
        assert np.all(line_schedule == expected_line_schedule)
        assert np.all(factory.predicted_balance == expected_balance)


if __name__ == '__main__':
    pytest.main(args=[__file__])
