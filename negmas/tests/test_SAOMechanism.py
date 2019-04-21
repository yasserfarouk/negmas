import random

from pytest import mark

from negmas import SAOMechanism, AspirationNegotiator, MappingUtilityFunction


def test_on_negotiation_start():
    mechanism = SAOMechanism(outcomes=10)
    assert mechanism.on_negotiation_start()


@mark.parametrize(["n_negotaitors"], [(2,), (3,)])
def test_round_n_agents(n_negotaitors):
    n_outcomes = 5
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotaitors, outcomes=n_outcomes)
    for i in range(n_negotaitors):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism.state.step == 1
    assert mechanism._current_offer is not None


@mark.parametrize(["n_negotaitors"], [(2,), (3,)])
def test_mechanism_can_run(n_negotaitors):
    n_outcomes = 5
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotaitors, outcomes=n_outcomes)
    for i in range(n_negotaitors):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])
    assert mechanism.state.step == 0
    mechanism.step()
    mechanism.run()
