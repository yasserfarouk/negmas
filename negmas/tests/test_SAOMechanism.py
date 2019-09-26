import random
from typing import List, Optional, Dict

from pytest import mark

from negmas import (
    SAOMechanism,
    AspirationNegotiator,
    MappingUtilityFunction,
    UtilityFunction,
    Issue,
    Outcome,
    UtilityValue,
    PassThroughSAONegotiator,
    SAOSyncController,
    SAOState,
    ResponseType,
)
from negmas.sao import SAOResponse


def test_on_negotiation_start():
    mechanism = SAOMechanism(outcomes=10)
    assert mechanism.on_negotiation_start()


@mark.parametrize(["n_negotiators"], [(2,), (3,)])
def test_round_n_agents(n_negotiators):
    n_outcomes = 5
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism.state.step == 1
    assert mechanism._current_offer is not None


@mark.parametrize(["n_negotiators"], [(2,), (3,)])
def test_mechanism_can_run(n_negotiators):
    n_outcomes = 5
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])
    assert mechanism.state.step == 0
    mechanism.step()
    mechanism.run()


@mark.parametrize("n_negotiators,oia", [(2, True), (3, True), (2, False), (3, False)])
def test_mechanism_runs_with_offering_not_accepting(n_negotiators, oia):
    n_outcomes = 5
    mechanism = SAOMechanism(
        outcomes=n_outcomes, n_steps=3, offering_is_accepting=oia, avoid_ultimatum=False
    )
    ufuns = MappingUtilityFunction.generate_random(1, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[0])
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism._current_proposer.name == "agent0"
    assert mechanism._n_accepting == n_negotiators + int(oia) - 1
    assert (mechanism.agreement is not None) is oia
    if mechanism.agreement is not None:
        return
    mechanism.step()
    assert mechanism._current_proposer.name == "agent0"
    assert mechanism._n_accepting == n_negotiators
    assert mechanism.agreement is not None


@mark.parametrize("n_negotiators,oia", [(2, True), (3, True), (2, False), (3, False)])
def test_mechanism_runall(n_negotiators, oia):
    n_outcomes = 5
    mechanisms = []
    for _ in range(10):
        mechanism = SAOMechanism(
            outcomes=n_outcomes,
            n_steps=random.randint(3, 20),
            offering_is_accepting=oia,
            avoid_ultimatum=False,
        )
        ufuns = MappingUtilityFunction.generate_random(1, outcomes=n_outcomes)
        for i in range(n_negotiators):
            mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[0])
        mechanisms.append(mechanism)

    states = SAOMechanism.runall(mechanisms)
    assert len(states) == 10
    assert not any(_.running for _ in states)


class MySAOSync(SAOSyncController):
    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        responses = {}
        for nid in offers.keys():
            offer, state = offers[nid], states[nid]
            if state.step < 2:
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None)
            else:
                responses[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        return responses


@mark.parametrize("n_negotiations,n_negotiators,oia", [(2, 1, True), (2, 1, False)])
def test_sync_controller(n_negotiations, n_negotiators, oia):
    n_outcomes = 2

    mechanisms = []
    controller = MySAOSync()
    for i in range(n_negotiators):
        mechanisms.append(
            SAOMechanism(
                outcomes=n_outcomes,
                n_steps=5,
                offering_is_accepting=oia,
                avoid_ultimatum=False,
            )
        )
        ufuns = MappingUtilityFunction.generate_random(
            n_negotiators, outcomes=n_outcomes
        )
        for i in range(n_negotiators):
            mechanisms[-1].add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])

        mechanisms[-1].add(controller.create_negotiator())

    states = SAOMechanism.runall(mechanisms)
    assert all(_.agreement is not None for _ in states)
