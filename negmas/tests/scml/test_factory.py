import copy
import itertools
import sys
from collections import defaultdict

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis._strategies import composite
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from pytest import fixture, mark

from negmas.apps.scml import ManufacturingProfile, Product, Process, InputOutput, Job, RunningCommandInfo
from negmas.apps.scml.simulators import SlowFactorySimulator, NO_PRODUCTION, FastFactorySimulator, storage_as_array
from negmas.apps.scml.world import Factory

n_lines = 5
n_levels = 4
n_products = 5
n_processes = 3
initial_wallet = 100.0
n_steps = 100
max_storage = sys.maxsize
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


def test_can_init_factory(empty_factory):
    assert empty_factory._n_lines == n_lines


def test_schedule_and_job(factory_with_storage):
    profile_ind = 0
    factory = factory_with_storage
    t = 10
    line = factory.profiles[profile_ind].line
    job = Job(profile=profile_ind, time=t, line=line, action='run', contract=None, override=True)
    factory.schedule(job=job)
    assert len(factory._jobs) == 1
    assert factory._jobs[(t, factory.profiles[profile_ind].line)] == job
    assert factory._wallet == initial_wallet
    assert len(factory._storage) == len(initial_storage)
    assert factory._storage == initial_storage

    profile = factory.profiles[profile_ind]

    # nothing happens before job time
    for _ in range(t):
        infos = factory.step()
        assert all(info.is_empty for info in infos)
        assert all(command.action == 'none' for command in factory._commands)

    # the command is executed in job time
    for _ in range(profile.n_steps):
        infos = factory.step()
        assert not any(info.failed for info in infos)
        assert all(command.action == 'none' for i, command in enumerate(factory._commands) if i != line)
        if _ == 0:
            assert isinstance(infos[line].started, RunningCommandInfo)
            assert infos[line].started.beg == factory._next_step - 1
            assert infos[line].started.end == profile.n_steps + factory._next_step - 1
        if _ == profile.n_steps - 1:
            assert isinstance(infos[line].finished, RunningCommandInfo)
        if 0 < _ < profile.n_steps - 1:
            assert isinstance(infos[line].continuing, RunningCommandInfo)
            assert not infos[line].finished and not infos[line].started

    # nothing happens after job time
    for _ in range(t):
        infos = factory.step()
        assert all(info.no_production for info in infos)
        assert all(command.action == 'none' for command in factory._commands)


@mark.parametrize('wallet', [initial_wallet, 0], ids=['has_money', 'no_money'])
def test_schedule_and_job_with_failures(empty_factory, wallet):
    profile_ind = 0
    factory = empty_factory
    factory._wallet = wallet
    t = 10
    line = factory.profiles[profile_ind].line
    job = Job(profile=profile_ind, time=t, line=line, action='run', contract=None, override=True)
    factory.schedule(job=job)
    assert len(factory._jobs) == 1
    assert factory._jobs[(t, factory.profiles[profile_ind].line)] == job
    assert factory._wallet == wallet
    assert len(factory._storage) == 0
    profile = factory.profiles[profile_ind]

    # nothing happens before job time
    for _ in range(t):
        infos = factory.step()
        assert all(info.is_empty for info in infos)
        assert all(command.action == 'none' for command in factory._commands)

    # the command is executed in job time
    infos = factory.step()
    assert not any(info.failed for i, info in enumerate(infos) if i != line)
    assert infos[line].failed
    assert len(infos[line].failure.missing_inputs) > 0
    if wallet < profile.cost:
        assert infos[line].failure.missing_money == profile.cost - wallet

    # nothing happens after job time
    for _ in range(2 * t):
        infos = factory.step()
        assert all(info.no_production for info in infos)
        assert all(command.action == 'none' for command in factory._commands)


def test_slow_factory_simulator_can_be_initialized(slow_simulator):
    assert slow_simulator is not None


@given(st.integers(min_value=0, max_value=n_steps * 2))
def test_slow_factory_simulator_can_be_checked_at_any_time(slow_simulator, t):
    assert slow_simulator.wallet_at(t) == initial_wallet
    assert slow_simulator.total_storage_at(t) == sum(initial_storage.values())


def do_simulator_run(simulator, profiles, t, at, profile_ind, override):
    profile = profiles[profile_ind]
    line = profile.line
    cost = profile.cost
    length = profile.n_steps
    total_inputs = sum(_.quantity for _ in profile.process.inputs) if at >= t else 0
    total_outputs = sum(_.quantity for _ in profile.process.outputs) if at >= t + profile.n_steps else 0

    job = Job(profile=profile_ind, time=t, line=line, action='run', contract=None, override=False)
    simulator.schedule(job=job, override=override)
    assert simulator.wallet_at(at) == (initial_wallet if at < t else initial_wallet - cost)
    assert simulator.total_storage_at(at) == sum(initial_storage.values()) - total_inputs + total_outputs
    assert (simulator.line_schedules_at(at)[_] == NO_PRODUCTION for _ in range(n_lines) if _ != line)
    if at < t or at >= t + length:
        assert simulator.line_schedules_at(at)[line] == NO_PRODUCTION

# @todo correct the slow simulator

@mark.parametrize('profile_ind,t,at_,override,simulator_type',
                  [
                      # (0, 10, 'after', True, 'slow'), (0, 10, 'after', False, 'slow'), (0, 10, 'before', True, 'slow'),
                      # (0, 10, 'before', False, 'slow')
                      # , (0, 10, 'just before', True, 'slow'), (0, 10, 'just before', False, 'slow'),
                      # (0, 10, 'just after', True, 'slow')
                      # , (0, 10, 'just after', False, 'slow')
                      # , (0, 10, 'at', True, 'slow'), (0, 10, 'at', False, 'slow'), (0, 10, 'middle', True, 'slow'),
                      # (0, 10, 'middle', False, 'slow')
                      # , (1, 10, 'after', True, 'slow'), (1, 10, 'after', False, 'slow'), (1, 10, 'before', True, 'slow')
                      # , (1, 10, 'before', False, 'slow')
                      # , (1, 10, 'just before', True, 'slow'), (1, 10, 'just before', False, 'slow'),
                      # (1, 10, 'just after', True, 'slow')
                      # , (1, 10, 'just after', False, 'slow')
                      # , (1, 10, 'at', True, 'slow'), (1, 10, 'at', False, 'slow'), (1, 10, 'middle', True, 'slow'),
                      # (1, 10, 'middle', False, 'slow')
                      # , (2, 10, 'after', True, 'slow'), (2, 10, 'after', False, 'slow'), (2, 10, 'before', True, 'slow')
                      # , (2, 10, 'before', False, 'slow')
                      # , (2, 10, 'just before', True, 'slow'), (2, 10, 'just before', False, 'slow'),
                      # (2, 10, 'just after', True, 'slow')
                      # , (2, 10, 'just after', False, 'slow')
                      # , (2, 10, 'at', True, 'slow'), (2, 10, 'at', False, 'slow'), (2, 10, 'middle', True, 'slow'),
                      # (2, 10, 'middle', False, 'slow')
                      # , (3, 10, 'after', True, 'slow'), (3, 10, 'after', False, 'slow'), (3, 10, 'before', True, 'slow')
                      # , (3, 10, 'before', False, 'slow')
                      # , (3, 10, 'just before', True, 'slow'), (3, 10, 'just before', False, 'slow'),
                      # (3, 10, 'just after', True, 'slow')
                      # , (3, 10, 'just after', False, 'slow')
                      # , (3, 10, 'at', True, 'slow'), (3, 10, 'at', False, 'slow'), (3, 10, 'middle', True, 'slow'),
                      # (3, 10, 'middle', False, 'slow'),
                      (0, 0, 'after', True, 'fast'), (0, 0, 'after', False, 'fast'), (0, 0, 'before', True, 'fast'),
                      (0, 0, 'before', False, 'fast')
                      , (0, 0, 'just before', True, 'fast'), (0, 0, 'just before', False, 'fast'),
                      (0, 0, 'just after', True, 'fast')
                      , (0, 0, 'just after', False, 'fast')
                      , (0, 0, 'at', True, 'fast'), (0, 0, 'at', False, 'fast'), (0, 0, 'middle', True, 'fast'),
                      (0, 0, 'middle', False, 'fast')
                      , (1, 0, 'after', True, 'fast'), (1, 0, 'after', False, 'fast'), (1, 0, 'before', True, 'fast')
                      , (1, 0, 'before', False, 'fast')
                      , (1, 0, 'just before', True, 'fast'), (1, 0, 'just before', False, 'fast'),
                      (1, 0, 'just after', True, 'fast')
                      , (1, 0, 'just after', False, 'fast')
                      , (1, 0, 'at', True, 'fast'), (1, 0, 'at', False, 'fast'), (1, 0, 'middle', True, 'fast'),
                      (1, 0, 'middle', False, 'fast')
                      , (2, 0, 'after', True, 'fast'), (2, 0, 'after', False, 'fast'), (2, 0, 'before', True, 'fast')
                      , (2, 0, 'before', False, 'fast')
                      , (2, 0, 'just before', True, 'fast'), (2, 0, 'just before', False, 'fast'),
                      (2, 0, 'just after', True, 'fast')
                      , (2, 0, 'just after', False, 'fast')
                      , (2, 0, 'at', True, 'fast'), (2, 0, 'at', False, 'fast'), (2, 0, 'middle', True, 'fast'),
                      (2, 0, 'middle', False, 'fast')
                      , (3, 0, 'after', True, 'fast'), (3, 0, 'after', False, 'fast'), (3, 0, 'before', True, 'fast')
                      , (3, 0, 'before', False, 'fast')
                      , (3, 0, 'just before', True, 'fast'), (3, 0, 'just before', False, 'fast'),
                      (3, 0, 'just after', True, 'fast')
                      , (3, 0, 'just after', False, 'fast')
                      , (3, 0, 'at', True, 'fast'), (3, 0, 'at', False, 'fast'), (3, 0, 'middle', True, 'fast'),
                      (3, 0, 'middle', False, 'fast')
                      , (0, 10, 'after', True, 'fast'), (0, 10, 'after', False, 'fast'),
                      (0, 10, 'before', True, 'fast'), (0, 10, 'before', False, 'fast')
                      , (0, 10, 'just before', True, 'fast'), (0, 10, 'just before', False, 'fast'),
                      (0, 10, 'just after', True, 'fast')
                      , (0, 10, 'just after', False, 'fast')
                      , (0, 10, 'at', True, 'fast'), (0, 10, 'at', False, 'fast'), (0, 10, 'middle', True, 'fast'),
                      (0, 10, 'middle', False, 'fast')
                      , (1, 10, 'after', True, 'fast'), (1, 10, 'after', False, 'fast'), (1, 10, 'before', True, 'fast')
                      , (1, 10, 'before', False, 'fast')
                      , (1, 10, 'just before', True, 'fast'), (1, 10, 'just before', False, 'fast'),
                      (1, 10, 'just after', True, 'fast')
                      , (1, 10, 'just after', False, 'fast')
                      , (1, 10, 'at', True, 'fast'), (1, 10, 'at', False, 'fast'), (1, 10, 'middle', True, 'fast'),
                      (1, 10, 'middle', False, 'fast')
                      , (2, 10, 'after', True, 'fast'), (2, 10, 'after', False, 'fast'), (2, 10, 'before', True, 'fast')
                      , (2, 10, 'before', False, 'fast')
                      , (2, 10, 'just before', True, 'fast'), (2, 10, 'just before', False, 'fast'),
                      (2, 10, 'just after', True, 'fast')
                      , (2, 10, 'just after', False, 'fast')
                      , (2, 10, 'at', True, 'fast'), (2, 10, 'at', False, 'fast'), (2, 10, 'middle', True, 'fast'),
                      (2, 10, 'middle', False, 'fast')
                      , (3, 10, 'after', True, 'fast'), (3, 10, 'after', False, 'fast'), (3, 10, 'before', True, 'fast')
                      , (3, 10, 'before', False, 'fast')
                      , (3, 10, 'just before', True, 'fast'), (3, 10, 'just before', False, 'fast'),
                      (3, 10, 'just after', True, 'fast')
                      , (3, 10, 'just after', False, 'fast')
                      , (3, 10, 'at', True, 'fast'), (3, 10, 'at', False, 'fast'), (3, 10, 'middle', True, 'fast'),
                      (3, 10, 'middle', False, 'fast')
                      , (0, 0, 'after', True, 'fast'), (0, 0, 'after', False, 'fast'), (0, 0, 'before', True, 'fast'),
                      (0, 0, 'before', False, 'fast')
                      , (0, 0, 'just before', True, 'fast'), (0, 0, 'just before', False, 'fast'),
                      (0, 0, 'just after', True, 'fast')
                      , (0, 0, 'just after', False, 'fast')
                      , (0, 0, 'at', True, 'fast'), (0, 0, 'at', False, 'fast'), (0, 0, 'middle', True, 'fast'),
                      (0, 0, 'middle', False, 'fast')
                      , (1, 0, 'after', True, 'fast'), (1, 0, 'after', False, 'fast'), (1, 0, 'before', True, 'fast')
                      , (1, 0, 'before', False, 'fast')
                      , (1, 0, 'just before', True, 'fast'), (1, 0, 'just before', False, 'fast'),
                      (1, 0, 'just after', True, 'fast')
                      , (1, 0, 'just after', False, 'fast')
                      , (1, 0, 'at', True, 'fast'), (1, 0, 'at', False, 'fast'), (1, 0, 'middle', True, 'fast'),
                      (1, 0, 'middle', False, 'fast')
                      , (2, 0, 'after', True, 'fast'), (2, 0, 'after', False, 'fast'), (2, 0, 'before', True, 'fast')
                      , (2, 0, 'before', False, 'fast')
                      , (2, 0, 'just before', True, 'fast'), (2, 0, 'just before', False, 'fast'),
                      (2, 0, 'just after', True, 'fast')
                      , (2, 0, 'just after', False, 'fast')
                      , (2, 0, 'at', True, 'fast'), (2, 0, 'at', False, 'fast'), (2, 0, 'middle', True, 'fast'),
                      (2, 0, 'middle', False, 'fast')
                      , (3, 0, 'after', True, 'fast'), (3, 0, 'after', False, 'fast'), (3, 0, 'before', True, 'fast')
                      , (3, 0, 'before', False, 'fast')
                      , (3, 0, 'just before', True, 'fast'), (3, 0, 'just before', False, 'fast'),
                      (3, 0, 'just after', True, 'fast')
                      , (3, 0, 'just after', False, 'fast')
                      , (3, 0, 'at', True, 'fast'), (3, 0, 'at', False, 'fast'), (3, 0, 'middle', True, 'fast'),
                      (3, 0, 'middle', False, 'fast')
                  ]
                  )
def test_slow_factory_simulator_with_jobs(products, profiles, profile_ind, t, at_, override, simulator_type):
    simulator_type = SlowFactorySimulator if simulator_type == 'slow' else FastFactorySimulator
    simulator = simulator_type(initial_wallet=initial_wallet, initial_storage=initial_storage, n_steps=n_steps
                               , n_products=len(products), profiles=profiles, max_storage=max_storage)
    profile = profiles[profile_ind]
    length = profile.n_steps
    if at_ == 'before':
        if t - 5 < 0:
            return
        at = t - 5
    elif at_ == 'after':
        if t + length + 3 > n_steps:
            return
        at = t + length + 3
    elif at_ == 'at':
        at = t
    elif at_ == 'just before':
        if t - 1 < 0:
            return
        at = t - 1
    elif at_ == 'just after':
        at = t + length
    elif at_ == 'middle':
        if length == 1:
            return
        at = (t + length) // 2
    else:
        raise ValueError(f'Unknown option {at_}')
    do_simulator_run(simulator, profiles, t, at, profile_ind, override)


@given(profile_ind=st.integers(min_value=0, max_value=3), t=st.integers(min_value=0, max_value=n_steps - 1)
    , at=st.integers(min_value=0, max_value=n_steps - 1), override=st.booleans()
    , simulator_type=st.sampled_from(('fast', ))) # @todo add slow back
def test_slow_factory_simulator_with_jobs_hypothesis(profile_ind, t, at, override, simulator_type):
    products = [Product(id=i + l * n_levels, production_level=l, name=f'{l}_{i}', catalog_price=(i + 1) * (l + 1),
                        expires_in=None)
                for i in range(n_products) for l in range(n_levels)]
    processes = [Process(id=i + l * n_levels, production_level=l, name=f'p{l}_{i}'
                         , inputs={InputOutput(product=i + (l - 1) * n_levels, quantity=3, step=0.0)
            , InputOutput(product=i + (l - 1) * n_levels, quantity=2, step=0.0)}
                         , outputs={InputOutput(product=i + l * n_levels, quantity=1, step=1.0)
            , InputOutput(product=i + l * n_levels, quantity=2, step=1.0)}
                         , historical_cost=1 + i + l * n_levels)
                 for i in range(n_processes) for l in range(1, n_levels)]

    profiles = [ManufacturingProfile(n_steps=i + 1, cost=10 * (i + 1), initial_pause_cost=1, running_pause_cost=2
                                     , resumption_cost=3, cancellation_cost=4, line=l, process=p)
                for i, (p, l) in enumerate(zip(processes, itertools.cycle(range(n_lines))))]

    simulator_type = SlowFactorySimulator if simulator_type == 'slow' else FastFactorySimulator
    simulator = simulator_type(initial_wallet=initial_wallet, initial_storage=initial_storage, n_steps=n_steps
                               , n_products=len(products), profiles=profiles, max_storage=max_storage)
    do_simulator_run(simulator, profiles, t, at, profile_ind, override)


@composite
def storage(draw):
    product = draw(st.integers(0, len(products())))
    quantity = draw(st.integers(-4, 4))
    return product, quantity


sample_products = [Product(id=i + l * n_levels, production_level=l, name=f'{l}_{i}', catalog_price=(i + 1) * (l + 1),
                           expires_in=None)
                   for i in range(n_products) for l in range(n_levels)]
sample_processes = [Process(id=i + l * n_levels, production_level=l, name=f'p{l}_{i}'
                            , inputs={InputOutput(product=i + (l - 1) * n_levels, quantity=3, step=0.0)
        , InputOutput(product=i + (l - 1) * n_levels, quantity=2, step=0.0)}
                            , outputs={InputOutput(product=i + l * n_levels, quantity=1, step=1.0)
        , InputOutput(product=i + l * n_levels, quantity=2, step=1.0)}
                            , historical_cost=1 + i + l * n_levels)
                    for i in range(n_processes) for l in range(1, n_levels)]
sample_profiles = [ManufacturingProfile(n_steps=i + 1, cost=10 * (i + 1), initial_pause_cost=1, running_pause_cost=2
                                        , resumption_cost=3, cancellation_cost=4, line=l, process=p)
                   for i, (p, l) in enumerate(zip(sample_processes, itertools.cycle(range(n_lines))))]


class SimulatorsActAsFactory(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        products, profiles = sample_products, sample_profiles
        self.slow_simulator = SlowFactorySimulator(initial_wallet=initial_wallet, initial_storage=initial_storage
                                                   , n_steps=n_steps
                                                   , n_products=len(products), profiles=profiles,
                                                   max_storage=max_storage)
        self.fast_simulator = FastFactorySimulator(initial_wallet=initial_wallet, initial_storage=initial_storage
                                                   , n_steps=n_steps
                                                   , n_products=len(products), profiles=profiles,
                                                   max_storage=max_storage)
        self.factory = Factory(initial_wallet=initial_wallet, initial_storage=initial_storage
                               , profiles=profiles, max_storage=max_storage)
        self.profiles = self.factory.profiles
        self._payments = defaultdict(list)
        self._transports = defaultdict(list)

    profile_indices = Bundle('profile_indices')
    payments = Bundle('payments')
    times = Bundle('times')
    transports = Bundle('transports')

    @rule(target=profile_indices, k=st.integers(len(sample_profiles)))
    def choose_profile(self, k):
        return k

    @rule(targe=payments, payment=st.floats(-30, 30))
    def choose_payment(self, payment):
        return payment

    @rule(targe=times, payment=st.integers(0, n_steps))
    def choose_time(self, t):
        return t

    @rule(targe=transports, storage=storage())
    def choose_transport(self, t):
        return t

    @rule(profile_index=profile_indices, t=times)
    def run_profile(self, profile_index, t):
        job = Job(profile=profile_index, time=t, line=self.profiles[profile_index].line
                  , action='run', contract=None, override=False)
        end = t + self.profiles[profile_index].n_steps
        if end > n_steps - 1:
            return
        self.factory.schedule(job=job, override=False)
        self.slow_simulator.schedule(job=job, override=False)
        self.fast_simulator.schedule(job=job, override=False)

    @rule(payment=payments, t=times)
    def pay(self, payment, t):
        self._payments[t].append(payment)
        self.slow_simulator.pay(payment, t)
        self.fast_simulator.pay(payment, t)

    @rule(trans=transports, t=times)
    def transport(self, trans, t):
        p, q = trans
        self._transports[t].append(trans)
        self.slow_simulator.transport_to(p, q, t)
        self.fast_simulator.transport_to(p, q, t)

    @rule()
    def run_and_test(self):
        for _ in range(n_steps):
            for payment in self._payments[_]:
                self.factory.pay(payment)
            for p, q in self._transports[_]:
                self.factory.transport_to(p, q)
            self.factory.step()
            assert self.slow_simulator.wallet_at(_) == self.fast_simulator.wallet_at(_) == self.factory.wallet
            assert self.slow_simulator.balance_at(_) == self.fast_simulator.balance_at(_) == self.factory.balance
            assert np.all(self.slow_simulator.storage_at(_) == self.fast_simulator.storage_at(_))
            assert np.all(self.slow_simulator.storage_at(_) == storage_as_array(self.factory.storage
                                                                                , n_products=len(sample_products)))
            assert np.all(self.slow_simulator.line_schedules_at(_) == self.factory.line_schedules)
            assert np.all(self.fast_simulator.line_schedules_at(_) == self.factory.line_schedules)


TestSimulators = SimulatorsActAsFactory.TestCase

if __name__ == '__main__':
    pytest.main(args=[__file__])
