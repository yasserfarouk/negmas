from __future__ import annotations

import random
from typing import Iterable

import pytest

from negmas import (
    LimitedOutcomesAcceptor,
    LimitedOutcomesNegotiator,
    MappingUtilityFunction,
    Mechanism,
    MechanismRoundResult,
    RandomNegotiator,
    SAOMechanism,
)

random.seed(0)


class MyMechanism(Mechanism):
    def __init__(self, dynamic_entry=True, **kwargs):
        super().__init__(outcomes=20, n_steps=10, dynamic_entry=dynamic_entry, **kwargs)
        self.current = 0

    def __call__(self, state) -> MechanismRoundResult:
        # r = random.random()
        # if r > 1.0 / self.n_steps:
        #    return None, False, None
        self.current += 1
        if self.current < 6:
            return MechanismRoundResult(state)
        state.agreement = self.random_outcomes(1)[0]
        return MechanismRoundResult(state)


@pytest.fixture
def mechanism():
    return MyMechanism()


@pytest.fixture
def static_mechanism():
    return MyMechanism(dynamic_entry=False)


@pytest.fixture
def dynamic_mechanism():
    return MyMechanism(dynamic_entry=True)


def test_set_requirements(mechanism):
    c = {"a": True, "b": [1, 2], "c": [1], "d": [1, 2], "e": False, "f": None}
    d = {"g": True}
    mechanism.requirements = c
    # print(mechanism.requirements)
    for k, v in c.items():
        if isinstance(v, Iterable):
            assert [a == b for a, b in zip(mechanism.requirements[k], v)]
        else:
            assert mechanism.requirements[k] == v

    mechanism.add_requirements(d)
    for k, v in c.items():
        if isinstance(v, Iterable):
            assert [a == b for a, b in zip(mechanism.requirements[k], v)]
        else:
            assert mechanism.requirements[k] == v

    for k, v in d.items():
        if isinstance(v, Iterable):
            assert [a == b for a, b in zip(mechanism.requirements[k], v)]
        else:
            assert mechanism.requirements[k] == v


def test_can_work_with_no_requirements(mechanism):
    assert mechanism.can_participate(RandomNegotiator())


def test_can_check_compatibility(mechanism):
    c = {"a": True, "b": [1, 2, 3], "c": 1, "d": None, "f": 3}

    assert mechanism.is_satisfying(c)

    mechanism.add_requirements({"a": False})
    assert not mechanism.is_satisfying(c)

    mechanism.add_requirements({"a": True})
    assert mechanism.is_satisfying(c)

    mechanism.add_requirements({"g": False})
    assert not mechanism.is_satisfying(c)

    mechanism.requirements = c
    assert mechanism.is_satisfying(c)

    mechanism.add_requirements({"g": False})
    assert not mechanism.is_satisfying(c)

    mechanism.remove_requirements(["a", "g"])
    assert mechanism.is_satisfying(c)

    mechanism.requirements = c
    assert mechanism.is_satisfying(c)

    mechanism.add_requirements({"a": not c["a"]})
    assert not mechanism.is_satisfying(c)

    mechanism.add_requirements({"a": None})
    assert mechanism.is_satisfying(c)

    mechanism.add_requirements({"a": not c["a"]})
    assert not mechanism.is_satisfying(c)

    mechanism.add_requirements({"a": [True, False]})
    assert mechanism.is_satisfying(c)

    class HasCapabilities:
        capabilities = c

    assert mechanism.can_participate(HasCapabilities())


def test_different_capability_types(mechanism):
    c = {
        "bvalue": True,
        "svalue": "abc",
        "ivalue": 1,
        "irange": (1, 5),
        "frange": (1.0, 5.0),
        "ilist": [1, 2, 9],
    }
    mechanism.requirements = c

    # single value
    assert mechanism.is_satisfying({**c, "bvalue": True})
    assert not mechanism.is_satisfying({**c, "bvalue": False})
    assert mechanism.is_satisfying({**c, "bvalue": [True, False]})
    assert mechanism.is_satisfying({**c, "svalue": "abc"})
    assert mechanism.is_satisfying({**c, "svalue": ["abc", "cde"]})
    assert not mechanism.is_satisfying({**c, "svalue": "ab"})
    assert not mechanism.is_satisfying({**c, "svalue": ["cc", "ab"]})
    assert mechanism.is_satisfying({**c, "ivalue": 1})
    assert not mechanism.is_satisfying({**c, "ivalue": 5})

    # range
    assert not mechanism.is_satisfying({**c, "irange": [-1, 0]})
    assert not mechanism.is_satisfying({**c, "irange": -1})
    assert not mechanism.is_satisfying({**c, "irange": 0})
    assert not mechanism.is_satisfying({**c, "irange": (-1, 0)})
    assert mechanism.is_satisfying({**c, "irange": (0, 1)})
    assert mechanism.is_satisfying({**c, "irange": [0, 1]})
    assert mechanism.is_satisfying({**c, "irange": 1})
    assert mechanism.is_satisfying({**c, "irange": (0, 2)})
    assert mechanism.is_satisfying({**c, "irange": (2, 4)})
    assert mechanism.is_satisfying({**c, "irange": (4, 5)})
    assert mechanism.is_satisfying({**c, "irange": (4, 6)})
    assert not mechanism.is_satisfying({**c, "irange": (6, 7)})

    assert not mechanism.is_satisfying({**c, "frange": (-1, 0)})
    assert mechanism.is_satisfying({**c, "frange": (0, 1)})
    assert mechanism.is_satisfying({**c, "frange": (0, 2)})
    assert mechanism.is_satisfying({**c, "frange": (2, 4)})
    assert mechanism.is_satisfying({**c, "frange": (4, 5)})
    assert mechanism.is_satisfying({**c, "frange": (4, 6)})
    assert not mechanism.is_satisfying({**c, "frange": (6, 7)})

    # list
    assert not mechanism.is_satisfying({**c, "ilist": (-1, 0)})
    assert mechanism.is_satisfying({**c, "ilist": (0, 1)})
    assert mechanism.is_satisfying({**c, "ilist": (3, 10)})

    assert not mechanism.is_satisfying({**c, "ilist": [-1, 0]})
    assert mechanism.is_satisfying({**c, "ilist": [0, 1]})
    assert not mechanism.is_satisfying({**c, "ilist": [3, 10]})

    assert not mechanism.is_satisfying({**c, "ilist": 0})
    assert mechanism.is_satisfying({**c, "ilist": 9})


def test_can_accept_more_agents():
    mechanism = MyMechanism(max_n_agents=2)
    assert mechanism.can_accept_more_agents() is True
    mechanism.add(RandomNegotiator(), ufun=MappingUtilityFunction(lambda x: 5.0))
    assert mechanism.can_accept_more_agents() is True
    mechanism.add(RandomNegotiator(), ufun=MappingUtilityFunction(lambda x: 5.0))
    assert mechanism.can_accept_more_agents() is False


def test_dynamic_entry(static_mechanism: Mechanism):
    a = RandomNegotiator()
    assert static_mechanism.can_enter(a)
    assert not static_mechanism.can_leave(a)

    static_mechanism.add(a, preferences=lambda x: 5.0)
    static_mechanism.add(RandomNegotiator(), preferences=lambda x: 5.0)


def test_mechanism_fails_on_less_than_two_agents(static_mechanism):
    assert not static_mechanism.run().started
    static_mechanism.add(RandomNegotiator(), ufun=lambda x: 5.0)
    assert not static_mechanism.run().started
    static_mechanism.add(RandomNegotiator(), ufun=lambda x: 5.0)
    static_mechanism.run()
    assert len(static_mechanism.history) > 0 and static_mechanism.state.broken is False


def test_mechanism_fails_on_less_than_two_agents_dynamic(dynamic_mechanism):
    assert not dynamic_mechanism.run().broken
    dynamic_mechanism.add(RandomNegotiator(), ufun=lambda x: 5.0)
    assert not dynamic_mechanism.run().broken
    dynamic_mechanism.add(RandomNegotiator(), ufun=lambda x: 5.0)
    dynamic_mechanism.run()
    assert (
        len(dynamic_mechanism.history) > 0 and dynamic_mechanism.state.broken is False
    )


def test_mechanisms_get_some_rounds():
    lengths = []
    for _ in range(10):
        p = MyMechanism(dynamic_entry=False)
        p.add(RandomNegotiator(), preferences=lambda x: 5.0)
        p.add(RandomNegotiator(), preferences=lambda x: 5.0)
        p.run()
        lengths.append(len(p.history))

    assert not all(_ < 2 for _ in lengths)


# def test_alternating_offers_mechanism():
#     p = SAOMechanism(outcomes=10, n_steps=10, dynamic_entry=False)
#     to_be_offered = [(0,), (1,), (2,)]
#     to_be_accepted = [(2,)]
#     a1 = LimitedOutcomesNegotiator(acceptable_outcomes=to_be_offered, outcomes=10)
#     a2 = LimitedOutcomesAcceptor(acceptable_outcomes=to_be_accepted, outcomes=10)
#     p.add(a1, preferences=MappingUtilityFunction(lambda x: x[0]+1.0))
#     p.add(a2, preferences=MappingUtilityFunction(lambda x: x[0] + 1.0))
#     p.run()
#     a1offers = [s.current_offer for s in p.history if s.current_proposer == a1.id]
#     a2offers = [s.current_offer for s in p.history if s.current_proposer == a2.id]
#     assert len(p.history) > 0
#     assert len(a2offers) == 0, 'acceptor did offer'
#     assert p.agreement is None or p.agreement in to_be_accepted, 'acceptor accepted the correct offer'
#     assert all([_ in to_be_offered for _ in a1offers])


def test_alternating_offers_mechanism_fails_on_no_offerer():
    p = SAOMechanism(
        outcomes=10,
        n_steps=10,
        dynamic_entry=False,
    )
    to_be_offered = [(0,), (1,), (2,)]
    to_be_accepted = [(2,)]
    a1 = LimitedOutcomesAcceptor(acceptable_outcomes=to_be_offered)
    a2 = LimitedOutcomesAcceptor(acceptable_outcomes=to_be_accepted)
    p.add(a1, preferences=MappingUtilityFunction(lambda x: x[0] + 1.0))
    p.add(a2, preferences=MappingUtilityFunction(lambda x: x[0] + 1.0))
    try:
        p.run()
    except RuntimeError:
        pass
    a1offers = [s.current_offer for s in p.history if s.current_proposer == a1.id]
    a2offers = [s.current_offer for s in p.history if s.current_proposer == a2.id]
    assert len(p.history) > 0
    assert len(a2offers) == 0, "acceptor did not offer"
    assert len(a1offers) == 0, "acceptor did not offer"
    assert p.agreement is None, "no agreement"


def test_alternating_offers_mechanism_with_one_agent_run():
    n_outcomes, n_steps = 10, 10
    accepted = [(2,), (3,), (4,), (5,)]
    neg = SAOMechanism(outcomes=n_outcomes, n_steps=n_steps)
    agent = LimitedOutcomesNegotiator(
        acceptable_outcomes=accepted,
        acceptance_probabilities=[1.0] * len(accepted),
    )
    opponent = LimitedOutcomesNegotiator(
        acceptable_outcomes=accepted,
        acceptance_probabilities=[1.0] * len(accepted),
    )
    neg.add(agent)
    neg.add(opponent)
    # assert neg.pareto_frontier(sort_by_welfare=True)[0] == [(1.0,)]
    neg.run()
    assert neg.agreement is not None


def test_same_utility_leads_to_agreement():
    n_outcomes, n_steps = 10, 10
    accepted = [(2,), (3,), (4,), (5,)]
    neg = SAOMechanism(outcomes=n_outcomes, n_steps=n_steps, avoid_ultimatum=False)
    opponent = LimitedOutcomesNegotiator(
        acceptable_outcomes=accepted,
        acceptance_probabilities=[1.0] * len(accepted),
    )
    acceptor = LimitedOutcomesAcceptor(
        acceptable_outcomes=accepted,
        acceptance_probabilities=[1.0] * len(accepted),
    )
    neg.add(opponent)
    neg.add(acceptor)
    # assert neg.pareto_frontier(sort_by_welfare=True)[0] == [(1.0, 1.0)]

    state = neg.run()
    assert state.agreement is not None
    assert state.step < 4


if __name__ == "__main__":
    pytest.main(args=[__file__])
