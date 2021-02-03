from pytest import mark

from negmas import (
    MappingUtilityFunction,
    SorterNegotiator,
)
from negmas.ga import GAMechanism


@mark.parametrize("n_negotiators,n_outcomes", [(2, 10), (3, 50), (2, 50), (3, 5)])
def test_ga_mechanism(n_negotiators, n_outcomes):
    mechanism = GAMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(SorterNegotiator(name=f"agent{i}"), ufun=ufuns[i])
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism.state.step == 1
    assert mechanism.dominant_outcomes is not None
    mechanism.run()
