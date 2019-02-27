import copy
import itertools
import math
from collections import defaultdict

import pytest
from pytest import fixture
import numpy as np

from negmas import Contract
from negmas.apps.scml import Product, Process, InputOutput, ManufacturingProfile, SCMLAgreement, CFP, GreedyScheduler, \
    ManufacturingProfileCompiled, ProductManufacturingInfo
from negmas.apps.scml.simulators import SlowFactorySimulator
from negmas.apps.scml.world import Factory

n_lines = 5
n_levels = 4
n_products = 5
n_processes = 3
initial_wallet = 100.0
n_steps = 100
max_storage = None
initial_storage = dict(zip(range(n_products * n_levels), range(10, 10 * (n_products * n_levels), 10)))


@fixture(scope='module')
def products():
    return [Product(id=i + l * n_levels, production_level=l, name=f'{l}_{i}', catalog_price=(i + 1) * (l + 1),
                    expires_in=None)
            for i in range(n_products) for l in range(n_levels)]


@fixture(scope='module')
def processes():
    return [Process(id=i + l * n_levels, production_level=l, name=f'p{l}_{i}'
                    , inputs={InputOutput(product=i + (l - 1) * n_levels, quantity=3, step=0.0)
            , InputOutput(product=i + (l - 1) * n_levels, quantity=2, step=0.0)}
                    , outputs={InputOutput(product=i + l * n_levels, quantity=1, step=1.0)
            , InputOutput(product=i + l * n_levels, quantity=2, step=1.0)}
                    , historical_cost=1 + i + l * n_levels)
            for i in range(n_processes) for l in range(1, n_levels)]


@fixture(scope='module')
def profiles(processes):
    return [ManufacturingProfile(n_steps=i + 1, cost=10 * (i + 1), initial_pause_cost=1, running_pause_cost=2
                                 , resumption_cost=3, cancellation_cost=4, line=l, process=p)
            for i, (p, l) in enumerate(zip(processes, itertools.cycle(range(n_lines))))]


@fixture(scope='module')
def n_profiles(profiles):
    return len(profiles)


@fixture
def empty_factory(profiles):
    return Factory(id='factory', profiles=profiles, initial_wallet=initial_wallet, initial_storage={}
                   , max_storage=max_storage)


@fixture
def factory_with_storage(profiles):
    return Factory(id='factory', profiles=profiles, initial_storage=copy.deepcopy(initial_storage)
                   , initial_wallet=initial_wallet, max_storage=max_storage)


@fixture
def slow_simulator(profiles, products):
    return SlowFactorySimulator(initial_wallet=initial_wallet, initial_storage=initial_storage, n_steps=n_steps
                                , n_products=len(products), profiles=profiles, max_storage=max_storage)


# class TestGreedyScheduler:
#     manager_name = 'test'
#     products = [Product(name=f'pr{i}', catalog_price=i + 1, id=i, production_level=i
#                             , expires_in=0) for i in range(5)]
#     processes = [Process(id=0, name='p0', inputs={InputOutput(0, quantity=2, step=0.0),
#                                                       InputOutput(1, quantity=3, step=0.0)}
#                              , outputs={InputOutput(2, quantity=1, step=1.0)}
#                              , historical_cost=1, production_level=1)
#         , Process(id=1, name='p1', inputs={InputOutput(2, quantity=3, step=0.0)}
#                   , outputs={InputOutput(3, quantity=1, step=1.0)}
#                   , historical_cost=1, production_level=2)
#         , Process(id=2, name='p2', inputs={InputOutput(2, quantity=3, step=0.0)}
#                   , outputs={InputOutput(3, quantity=2, step=1.0)}
#                   , historical_cost=3, production_level=2)
#         , Process(id=3, name='p3', inputs={InputOutput(3, quantity=3, step=0.0)}
#                   , outputs={InputOutput(4, quantity=1, step=1.0)}
#                   , historical_cost=3, production_level=3)
#                      ]
#     @classmethod
#     def setup_class(cls):
#         cls.manager_name = 'test'
#         cls.products = [Product(name=f'pr{i}', catalog_price=i + 1, id=i, production_level=i
#                                 , expires_in=0) for i in range(5)]
#         cls.processes = [Process(id=0, name='p0', inputs={InputOutput(0, quantity=2, step=0.0),
#                                                           InputOutput(1, quantity=3, step=0.0)}
#                                  , outputs={InputOutput(2, quantity=1, step=1.0)}
#                                  , historical_cost=1, production_level=1)
#             , Process(id=1, name='p1', inputs={InputOutput(2, quantity=3, step=0.0)}
#                       , outputs={InputOutput(3, quantity=1, step=1.0)}
#                       , historical_cost=1, production_level=2)
#             , Process(id=2, name='p2', inputs={InputOutput(2, quantity=3, step=0.0)}
#                       , outputs={InputOutput(3, quantity=2, step=1.0)}
#                       , historical_cost=3, production_level=2)
#             , Process(id=3, name='p3', inputs={InputOutput(3, quantity=3, step=0.0)}
#                       , outputs={InputOutput(4, quantity=1, step=1.0)}
#                       , historical_cost=3, production_level=3)
#                          ]
#
#     @classmethod
#     def create_profiles(cls,  n_lines):
#         profiles = []
#         for l in range(n_lines):
#             profiles.extend([ManufacturingProfile(n_steps=1, cost=10, cancellation_cost=5
#                               , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0, line=l,
#                               process=cls.processes[0])
#             , ManufacturingProfile(n_steps=5, cost=5, cancellation_cost=2
#                                    , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0, line=l,
#                                    process=cls.processes[0])
#             , ManufacturingProfile(n_steps=3, cost=3, cancellation_cost=0
#                                    , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0, line=l,
#                                    process=cls.processes[1])
#             , ManufacturingProfile(n_steps=2, cost=5, cancellation_cost=0
#                                    , initial_pause_cost=0, running_pause_cost=0, resumption_cost=0, line=l,
#                                    process=cls.processes[1])])
#         return profiles
#
#     @classmethod
#     def create_factory(cls, n_lines, initial_balance, storage):
#         profiles = cls.create_profiles(n_lines)
#         return Factory(id='factory', profiles=profiles, initial_storage=storage
#                        , initial_wallet=initial_balance, max_storage=max_storage)
#
#     @classmethod
#     def create_scheduler(cls, n_lines, n_steps, strategy, initial_balance, storage):
#         if storage is None:
#             storage = {}
#         factory = cls.create_factory(n_lines, initial_balance, storage)
#         profiles = factory.profiles
#         processes = cls.processes
#         p2i = dict(zip(processes, range(len(processes))))
#         compiled_profiles = [ManufacturingProfileCompiled.from_manufacturing_profile(p, process2ind=p2i)
#                              for p in profiles]
#         scheduler = GreedyScheduler(manager_id=cls.manager_name, awi=None, horizon=None, max_insurance_premium=None
#                                     , add_catalog_prices=True, strategy=strategy)
#         simulator = SlowFactorySimulator(initial_wallet=initial_balance, initial_storage=storage
#                                          , n_steps=n_steps, n_products=len(cls.products), profiles=compiled_profiles
#                                          , max_storage=max_storage)
#         producing = defaultdict(list)
#         for index, profile in enumerate(profiles):
#             process = profile.process
#             for outpt in process.outputs:
#                 step = int(math.ceil(outpt.step * profile.n_steps))
#                 producing[outpt.product].append(ProductManufacturingInfo(profile=index
#                                                                               , quantity=outpt.quantity
#                                                                               , step=step))
#
#         scheduler.init(simulator=simulator, products=cls.products, processes=cls.processes
#                        , profiles=compiled_profiles, producing=producing)
#         return scheduler
#
#     @classmethod
#     def buy_contract(cls, t, u, q, p):
#         return Contract(agreement=SCMLAgreement(time=t, unit_price=u, quantity=q)
#                         , annotation={'cfp': CFP(is_buy=True, product=p, time=t, unit_price=u, quantity=q
#                                                  , publisher=cls.manager_name)
#                 , 'buyer': cls.manager_name, 'seller': 'anyone'})
#
#     @classmethod
#     def sell_contract(cls, t, u, q, p):
#         return Contract(agreement=SCMLAgreement(time=t, unit_price=u, quantity=q)
#                         , annotation={'cfp': CFP(is_buy=True, product=p, time=t, unit_price=u, quantity=q
#                                                  , publisher='anyone')
#                 , 'buyer': 'anyone', 'seller': cls.manager_name})
#
#     def test_initialization(self):
#         scheduler = self.create_scheduler(n_lines=3, n_steps=5, strategy='earliest', initial_balance=1000
#                                           , storage={})
#         assert len(scheduler.products) == len(self.products)
#
#     def test_scheduling_a_buy_contract(self):
#         t = 2
#         initial = 1000
#         unit_price, quantity = 2, 1
#         total = unit_price * quantity
#         storage = {0: 10, 1: 10, 2: 0}
#         contract = self.buy_contract(p=0, t=t, u=unit_price, q=quantity)
#         scheduler = self.create_scheduler(n_lines=1, n_steps=t * 2, storage=storage, strategy='earliest'
#                                           , initial_balance=initial)
#         schedule = scheduler.schedule([contract])
#         final = schedule.final_balance
#         assert schedule.valid
#         assert final == initial - total
#         # factory = scheduler.factory_state
#         # assert len(factory.predicted_balance) == t + 1
#         # assert max(abs(factory.predicted_storage[0, :t] - 10)) == 0
#         # assert max(abs(factory.predicted_storage[0, t:] - 11)) == 0
#         # assert max(abs(factory.predicted_storage[1, :t] - 10)) == 0
#         # assert max(abs(factory.predicted_storage[2:].flatten())) == 0
#         # assert max(abs(factory.predicted_balance[:t] - initial)) == 0
#         # assert max(abs(factory.predicted_balance[t:] - initial + total)) == 0
#         # @todo test the case where end != None
#
#     def test_scheduling_a_sell_contract_no_needs(self):
#         t, n_steps = 3, 20
#         initial = 1000
#         unit_price, quantity = 2, 1
#         total = unit_price * quantity
#         storage = {0: 10, 1: 10, 2: 0}
#         contract = self.sell_contract(p=0, t=t, u=unit_price, q=quantity)
#         scheduler = self.create_scheduler(n_lines=1, n_steps=t * 2, storage=storage, strategy='earliest')
#         scheduler.schedule([contract])
#         factory = scheduler.factory_state
#
#         assert max(abs(factory.predicted_storage[0, :t] - 10)) == 0
#         assert max(abs(factory.predicted_storage[0, t:] - 9)) == 0
#         assert max(abs(factory.predicted_storage[1, :t] - 10)) == 0
#         assert max(abs(factory.predicted_storage[2:].flatten())) == 0
#         assert max(abs(factory.predicted_balance[:t] - 1000)) == 0
#         assert max(abs(factory.predicted_balance[t:] - 1002)) == 0
#
#     @pytest.mark.parametrize('product_id,t,quantity,valid'
#         , [(0, 12, 1, False), (2, 12, 1, True), (2, 12, 2, True), (2, 12, 50, False)]
#         , ids=['cannot be produced', 'producing 1 item', 'producing 2 items', 'time is not enough'])
#     def test_scheduling_a_sell_contract_with_needs_early(self, product_id, t, quantity, valid):
#         products, processes, profiles = self.products, self.processes, self.profiles
#         n_steps = 20
#         initial = 1000
#         unit_price = 2
#         total = unit_price * quantity
#         process, profile = self.processes[0], self.profiles[0]
#         storage = {0: 0, 1: 0, 2: 0}
#         contract = self.sell_contract(p=product_id, t=t, u=unit_price, q=quantity)
#         scheduler = self.create_scheduler(n_lines=1, n_steps=n_steps, storage=storage, strategy='earliest')
#         info = scheduler.schedule([contract])
#         factory = scheduler.factory_state
#
#         total_cost = sum(self.products[i.product].catalog_price * i.quantity for i in process.inputs)
#         total_cost += profile.cost
#         selling_price = contract.agreement['unit_price'] * contract.agreement['quantity']
#         production_time = profile.n_steps * contract.agreement['quantity'] // list(process.outputs)[0].quantity
#
#         assert info.valid == valid
#         if valid:
#             assert len(info.needs) == 2 * quantity
#             assert 2 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
#             assert 3 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
#             if quantity > 1:
#                 assert 2 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)
#                 assert 3 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)
#
#             expected_balance = np.array([initial - total_cost] * profile.n_steps +
#                                         [initial - quantity * total_cost] * (contract.agreement['time']
#                                                                              - profile.n_steps) +
#                                         [initial - quantity * total_cost + selling_price])
#             expected_storage = np.array([[-2] * profile.n_steps + [-2 * quantity] * (t + 1 - profile.n_steps),
#                                          [-3] * profile.n_steps + [-3 * quantity] * (t + 1 - profile.n_steps),
#                                          [0] * profile.n_steps + [1] * profile.n_steps +
#                                          [quantity] * (t - 2 * profile.n_steps) + [0],
#                                          [0] * (t + 1),
#                                          [0] * (t + 1)])
#             expected_total_storage = expected_storage.sum(axis=0)
#             expected_line_schedule = np.array([0] * profile.n_steps * quantity + [-1] * (t + 1
#                                                                                          - quantity * profile.n_steps))
#         else:
#             expected_line_schedule = scheduler.managed_factory.schedule[
#                                          list(scheduler.managed_factory.lines.values())[0]][: t + 1]
#             expected_storage = scheduler.managed_factory.predicted_storage[:, : t + 1]
#             expected_balance = scheduler.managed_factory.predicted_balance[: t + 1]
#             expected_total_storage = scheduler.managed_factory.predicted_total_storage[: t + 1]
#
#         line_schedule = factory.schedule[list(factory.lines.values())[0]]
#         assert np.all(factory.predicted_storage == expected_storage)
#         assert np.all(factory.predicted_total_storage == expected_total_storage)
#         assert np.all(line_schedule == expected_line_schedule)
#         assert np.all(factory.predicted_balance == expected_balance)
#
#
#     @pytest.mark.parametrize('product_id,t,quantity,valid'
#         , [(0, 12, 1, False), (2, 12, 1, True), (2, 12, 2, True), (2, 12, 50, False)]
#         , ids=['cannot be produced', 'producing 1 item', 'producing 2 items', 'time is not enough'])
#     def test_scheduling_a_sell_contract_with_needs_late(self, product_id, t, quantity, valid):
#         products, processes, profiles = self.products, self.processes, self.profiles
#         n_steps = 20
#         initial = 1000
#         unit_price = 2
#         total = unit_price * quantity
#         process, profile = self.processes[0], self.profiles[0]
#         storage = {0: 0, 1: 0, 2: 0}
#         contract = self.sell_contract(p=product_id, t=t, u=unit_price, q=quantity)
#         scheduler = self.create_scheduler(n_lines=1, n_steps=n_steps, storage=storage, strategy='latest')
#         info = scheduler.schedule([contract])
#         factory = scheduler.factory_state
#
#         total_cost = sum(self.products[i.product].catalog_price * i.quantity for i in process.inputs)
#         total_cost += profile.cost
#         selling_price = contract.agreement['unit_price'] * contract.agreement['quantity']
#         production_time = profile.n_steps * contract.agreement['quantity'] // list(process.outputs)[0].quantity
#
#         assert info.valid == valid
#         if valid:
#             assert len(info.needs) == 2 * quantity
#             assert 2 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
#             assert 3 in (info.needs[0].quantity_to_buy, info.needs[1].quantity_to_buy)
#             if quantity > 1:
#                 assert 2 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)
#                 assert 3 in (info.needs[2].quantity_to_buy, info.needs[3].quantity_to_buy)
#
#             def _expected_balance(n_steps=profile.n_steps, total=t + 1):
#                 s = initial * np.ones(total)
#                 s[total - 1] += selling_price
#                 for q in range(quantity):
#                     s[-n_steps * (q + 1) - 1:] -= total_cost
#                 return s.tolist()
#
#             def _expected_storage(n_needed, n_steps=profile.n_steps, total=t + 1):
#                 s = np.zeros(total)
#                 for q in range(quantity):
#                     s[-n_steps * (q + 1) - 1:] -= n_needed
#                 return s.tolist()
#
#             def _expected_output(total=t + 1, n_steps=profile.n_steps):
#                 s = np.zeros(total)
#                 for q in range(quantity):
#                     s[-n_steps * q - 1:] += 1
#                 s[-1] = 0
#                 return s.tolist()
#
#             def _expected_line(total=t + 1, n_steps=profile.n_steps):
#                 s = -1 * np.ones(total)
#                 for q in range(quantity):
#                     s[-n_steps * (q + 1) - 1:-1] = 0
#                 return s.tolist()
#
#             expected_balance = _expected_balance()
#             expected_storage = np.array([_expected_storage(2), _expected_storage(3), _expected_output()
#                                             , [0] * (t + 1), [0] * (t + 1)])
#             expected_total_storage = expected_storage.sum(axis=0)
#             expected_line_schedule = _expected_line()
#
#             # print(f'q={quantity}')
#             # pprint([str(_) for _ in info.needs])
#             # print('expected')
#             # pprint(expected_line_schedule)
#             # print('real')
#             # pprint({k.id: v for k, v in factory.schedule.items()})
#             # print('expected')
#             # pprint(expected_balance)
#             # print('real')
#             # pprint(factory.predicted_balance)
#             # print('expected')
#             # pprint(expected_storage)
#             # print('real')
#             # pprint(factory.predicted_storage)
#             #
#             # print(f'total_cost: {total_cost}, selling_price: {selling_price}, production_time: {production_time}')
#         else:
#             expected_line_schedule = scheduler.managed_factory.schedule[
#                                          list(scheduler.managed_factory.lines.values())[0]][: t + 1]
#             expected_storage = scheduler.managed_factory.predicted_storage[:, : t + 1]
#             expected_balance = scheduler.managed_factory.predicted_balance[: t + 1]
#             expected_total_storage = scheduler.managed_factory.predicted_total_storage[: t + 1]
#
#         line_schedule = factory.schedule[list(factory.lines.values())[0]]
#         assert np.all(factory.predicted_storage == expected_storage)
#         assert np.all(factory.predicted_total_storage == expected_total_storage)
#         assert np.all(line_schedule == expected_line_schedule)
#         assert np.all(factory.predicted_balance == expected_balance)
