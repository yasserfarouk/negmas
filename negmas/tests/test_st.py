from __future__ import annotations

from pytest import mark

from negmas import BinaryComparatorNegotiator, MappingUtilityFunction
from negmas.st import HillClimbingSTMechanism, VetoSTMechanism


@mark.parametrize(
    "n_negotiators,n_outcomes,n_steps",
    [(2, 10000, 300), (3, 50, 3), (2, 50, 3), (3, 5, 3)],
)
def test_can_work(n_negotiators, n_outcomes, n_steps):
    mechanism = VetoSTMechanism(outcomes=n_outcomes, n_steps=n_steps)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(
            BinaryComparatorNegotiator(name=f"agent{i}"), preferences=ufuns[i]
        )
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism.state.step == 1
    assert mechanism.state.current_offer is not None
    mechanism.run()
    assert mechanism.agreement is not None


@mark.parametrize("n_negotiators,n_outcomes", [(2, 10), (3, 50), (2, 50), (3, 5)])
def test_hill_climbing_mechanism(n_negotiators, n_outcomes):
    mechanism = HillClimbingSTMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(
            BinaryComparatorNegotiator(name=f"agent{i}"), preferences=ufuns[i]
        )
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism.state.step == 1
    assert mechanism.state.current_offer is not None
    mechanism.run()
    assert mechanism.agreement is not None
