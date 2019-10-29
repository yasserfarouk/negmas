import copy
import itertools
import sys
from collections import defaultdict

import itertools
from collections import defaultdict

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from pytest import fixture

from negmas.apps.scml import ManufacturingProfile, Product, Process, InputOutput, Job
from negmas.apps.scml.simulators import (
    SlowFactorySimulator,
    FastFactorySimulator,
    storage_as_array,
)
from negmas.apps.scml.world import Factory
from negmas.apps.scml.common import NO_PRODUCTION

n_lines = 5
n_levels = 4
n_products = 5
n_processes = 3
initial_wallet = 100.0
n_steps = 100
max_storage = sys.maxsize
initial_storage = dict(
    zip(range(n_products * n_levels), range(10, 10 * (n_products * n_levels), 10))
)


@fixture(scope="module")
def products():
    return [
        Product(
            id=i + l * n_levels,
            production_level=l,
            name=f"{l}_{i}",
            catalog_price=(i + 1) * (l + 1),
            expires_in=None,
        )
        for i in range(n_products)
        for l in range(n_levels)
    ]


@fixture(scope="module")
def processes():
    return [
        Process(
            id=i + l * n_levels,
            production_level=l,
            name=f"p{l}_{i}",
            inputs=[
                InputOutput(product=i + (l - 1) * n_levels, quantity=3, step=0.0),
                InputOutput(product=i + (l - 1) * n_levels, quantity=2, step=0.0),
            ],
            outputs=[
                InputOutput(product=i + l * n_levels, quantity=1, step=1.0),
                InputOutput(product=i + l * n_levels, quantity=2, step=1.0),
            ],
            historical_cost=1 + i + l * n_levels,
        )
        for i in range(n_processes)
        for l in range(1, n_levels)
    ]


@fixture(scope="module")
def profiles(processes):
    return [
        ManufacturingProfile(
            n_steps=i + 1,
            cost=10 * (i + 1),
            initial_pause_cost=1,
            running_pause_cost=2,
            resumption_cost=3,
            cancellation_cost=4,
            line=l,
            process=p,
        )
        for i, (p, l) in enumerate(zip(processes, itertools.cycle(range(n_lines))))
    ]


@fixture(scope="module")
def n_profiles(profiles):
    return len(profiles)


@fixture
def empty_factory(profiles):
    return Factory(
        id="factory",
        profiles=profiles,
        initial_wallet=initial_wallet,
        initial_storage={},
        max_storage=max_storage,
    )


@fixture
def factory_with_storage(profiles):
    return Factory(
        id="factory",
        profiles=profiles,
        initial_storage=copy.deepcopy(initial_storage),
        initial_wallet=initial_wallet,
        max_storage=max_storage,
    )


@fixture
def slow_simulator(profiles, products):
    return SlowFactorySimulator(
        initial_wallet=initial_wallet,
        initial_storage=initial_storage,
        n_steps=n_steps,
        n_products=len(products),
        profiles=profiles,
        max_storage=max_storage,
    )


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
    total_outputs = (
        sum(_.quantity for _ in profile.process.outputs)
        if at >= t + profile.n_steps
        else 0
    )

    job = Job(
        profile=profile_ind,
        time=t,
        line=line,
        action="run",
        contract=None,
        override=False,
    )
    simulator.schedule(job=job, override=override)
    assert simulator.wallet_at(at) == (
        initial_wallet if at < t else initial_wallet - cost
    )
    assert (
        simulator.total_storage_at(at)
        == sum(initial_storage.values()) - total_inputs + total_outputs
    )
    assert (
        simulator.line_schedules_at(at)[_] == NO_PRODUCTION
        for _ in range(n_lines)
        if _ != line
    )
    if at < t or at >= t + length:
        assert simulator.line_schedules_at(at)[line] == NO_PRODUCTION


@given(
    profile_ind=st.integers(min_value=0, max_value=3),
    t=st.integers(min_value=0, max_value=n_steps - 1),
    at=st.integers(min_value=0, max_value=n_steps - 1),
    override=st.booleans(),
    simulator_type=st.sampled_from(("fast",)),
)  # @todo add slow back
def test_slow_factory_simulator_with_jobs_hypothesis(
    profile_ind, t, at, override, simulator_type
):
    products = [
        Product(
            id=i + l * n_levels,
            production_level=l,
            name=f"{l}_{i}",
            catalog_price=(i + 1) * (l + 1),
            expires_in=None,
        )
        for i in range(n_products)
        for l in range(n_levels)
    ]
    processes = [
        Process(
            id=i + l * n_levels,
            production_level=l,
            name=f"p{l}_{i}",
            inputs=[
                InputOutput(product=i + (l - 1) * n_levels, quantity=3, step=0.0),
                InputOutput(product=i + (l - 1) * n_levels, quantity=2, step=0.0),
            ],
            outputs=[
                InputOutput(product=i + l * n_levels, quantity=1, step=1.0),
                InputOutput(product=i + l * n_levels, quantity=2, step=1.0),
            ],
            historical_cost=1 + i + l * n_levels,
        )
        for i in range(n_processes)
        for l in range(1, n_levels)
    ]

    profiles = [
        ManufacturingProfile(
            n_steps=i + 1,
            cost=10 * (i + 1),
            initial_pause_cost=1,
            running_pause_cost=2,
            resumption_cost=3,
            cancellation_cost=4,
            line=l,
            process=p,
        )
        for i, (p, l) in enumerate(zip(processes, itertools.cycle(range(n_lines))))
    ]

    simulator_type = (
        SlowFactorySimulator if simulator_type == "slow" else FastFactorySimulator
    )
    simulator = simulator_type(
        initial_wallet=initial_wallet,
        initial_storage=initial_storage,
        n_steps=n_steps,
        n_products=len(products),
        profiles=profiles,
        max_storage=max_storage,
    )
    do_simulator_run(simulator, profiles, t, at, profile_ind, override)


@composite
def storage(draw):
    product = draw(st.integers(0, len(products())))
    quantity = draw(st.integers(-4, 4))
    return product, quantity


sample_products = [
    Product(
        id=i + l * n_levels,
        production_level=l,
        name=f"{l}_{i}",
        catalog_price=(i + 1) * (l + 1),
        expires_in=None,
    )
    for i in range(n_products)
    for l in range(n_levels)
]
sample_processes = [
    Process(
        id=i + l * n_levels,
        production_level=l,
        name=f"p{l}_{i}",
        inputs=[
            InputOutput(product=i + (l - 1) * n_levels, quantity=3, step=0.0),
            InputOutput(product=i + (l - 1) * n_levels, quantity=2, step=0.0),
        ],
        outputs=[
            InputOutput(product=i + l * n_levels, quantity=1, step=1.0),
            InputOutput(product=i + l * n_levels, quantity=2, step=1.0),
        ],
        historical_cost=1 + i + l * n_levels,
    )
    for i in range(n_processes)
    for l in range(1, n_levels)
]
sample_profiles = [
    ManufacturingProfile(
        n_steps=i + 1,
        cost=10 * (i + 1),
        initial_pause_cost=1,
        running_pause_cost=2,
        resumption_cost=3,
        cancellation_cost=4,
        line=l,
        process=p,
    )
    for i, (p, l) in enumerate(zip(sample_processes, itertools.cycle(range(n_lines))))
]


class SimulatorsActAsFactory(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        products, profiles = sample_products, sample_profiles
        self.slow_simulator = SlowFactorySimulator(
            initial_wallet=initial_wallet,
            initial_storage=initial_storage,
            n_steps=n_steps,
            n_products=len(products),
            profiles=profiles,
            max_storage=max_storage,
        )
        self.fast_simulator = FastFactorySimulator(
            initial_wallet=initial_wallet,
            initial_storage=initial_storage,
            n_steps=n_steps,
            n_products=len(products),
            profiles=profiles,
            max_storage=max_storage,
        )
        self.factory = Factory(
            initial_wallet=initial_wallet,
            initial_storage=initial_storage,
            profiles=profiles,
            max_storage=max_storage,
        )
        self.profiles = self.factory.profiles
        self._payments = defaultdict(list)
        self._transports = defaultdict(list)

    profile_indices = Bundle("profile_indices")
    payments = Bundle("payments")
    times = Bundle("times")
    transports = Bundle("transports")

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
        job = Job(
            profile=profile_index,
            time=t,
            line=self.profiles[profile_index].line,
            action="run",
            contract=None,
            override=False,
        )
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
            assert (
                self.slow_simulator.wallet_at(_)
                == self.fast_simulator.wallet_at(_)
                == self.factory.wallet
            )
            assert (
                self.slow_simulator.balance_at(_)
                == self.fast_simulator.balance_at(_)
                == self.factory.balance
            )
            assert np.all(
                self.slow_simulator.storage_at(_) == self.fast_simulator.storage_at(_)
            )
            assert np.all(
                self.slow_simulator.storage_at(_)
                == storage_as_array(
                    self.factory.storage, n_products=len(sample_products)
                )
            )
            assert np.all(
                self.slow_simulator.line_schedules_at(_) == self.factory.line_schedules
            )
            assert np.all(
                self.fast_simulator.line_schedules_at(_) == self.factory.line_schedules
            )


TestSimulators = SimulatorsActAsFactory.TestCase

if __name__ == "__main__":
    pytest.main(args=[__file__])
