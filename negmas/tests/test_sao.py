import random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Dict

from hypothesis import given, settings, example
from pytest import mark
import hypothesis.strategies as st

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
    LinearUtilityFunction,
)
from negmas.helpers import unique_name
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


@given(
    n_negotiators=st.integers(2, 4),
    n_issues=st.integers(1, 3),
    presort=st.booleans(),
    randomize_offers=st.booleans(),
)
@settings(deadline=20000, max_examples=100)
@example(n_negotiators=2, n_issues=1, presort=False, randomize_offers=False)
def test_aspiration_continuous_issues(
    n_negotiators, n_issues, presort, randomize_offers
):
    for k in range(5):
        mechanism = SAOMechanism(
            issues=[Issue(values=(0.0, 1.0), name=f"i{i}") for i in range(n_issues)],
            n_steps=10,
        )
        ufuns = [
            LinearUtilityFunction(
                weights=[3.0 * random.random(), 2.0 * random.random()],
                reserved_value=0.0,
            )
            for _ in range(n_negotiators)
        ]
        best_outcome = tuple([1.0] * n_issues)
        worst_outcome = tuple([0.0] * n_issues)
        i = 0
        assert mechanism.add(
            AspirationNegotiator(
                name=f"agent{i}",
                presort=presort,
                randomize_offer=randomize_offers,
                ufun=ufuns[i],
                ufun_max=ufuns[i](best_outcome),
                ufun_min=ufuns[i](worst_outcome),
            )
        ), "Cannot add negotiator"
        for i in range(1, n_negotiators):
            assert mechanism.add(
                AspirationNegotiator(
                    name=f"agent{i}",
                    presort=presort,
                    randomize_offer=randomize_offers,
                    ufun_max=ufuns[i](best_outcome),
                    ufun_min=ufuns[i](worst_outcome),
                ),
                ufun=ufuns[i],
            ), "Cannot add negotiator"
        assert mechanism.state.step == 0
        agents = dict(zip([_.id for _ in mechanism.negotiators], mechanism.negotiators))
        offers = defaultdict(list)
        while not mechanism.state.ended:
            mechanism.step()
            for neg_id, offer in mechanism.state.new_offers:
                assert neg_id in agents.keys()
                neg = agents[neg_id]
                prev = offers[neg_id]
                last_offer = prev[-1] if len(prev) > 0 else float("inf")
                if randomize_offers:
                    assert neg.utility_function(offer) <= neg.utility_function(
                        best_outcome
                    )
                else:
                    assert neg.utility_function(offer) <= last_offer
                    if not presort:
                        assert (
                            -neg.tolerance
                            <= (
                                neg.utility_function(offer)
                                - neg.aspiration(
                                    (mechanism.state.step) / mechanism.n_steps
                                )
                                * neg.utility_function(best_outcome)
                            )
                            < pow(neg.tolerance, 0.5 / neg.n_trials) + neg.tolerance
                        )
                    # else:
                    #     assert -neg.tolerance <= (
                    #         neg.utility_function(offer)
                    #         - neg.aspiration(
                    #             (mechanism.state.step - 1) / mechanism.n_steps
                    #         )
                    #         * neg.utility_function(best_outcome)
                    #     )

                offers[neg_id].append(neg.utility_function(offer))


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


def test_pickling_mechanism(tmp_path):
    import pickle

    file = tmp_path / "mechanism.pck"
    n_outcomes, n_negotiators = 5, 3
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=3,
        offering_is_accepting=True,
        avoid_ultimatum=False,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])

    assert mechanism.state.step == 0
    with open(file, "wb") as f:
        pickle.dump(mechanism, f)
    with open(file, "rb") as f:
        pickle.load(f)
    assert mechanism.state.step == 0
    mechanism.step()
    with open(file, "wb") as f:
        pickle.dump(mechanism, f)
    with open(file, "rb") as f:
        pickle.load(f)
    assert mechanism.state.step == 1


def test_checkpointing_mechanism(tmp_path):
    file = tmp_path
    n_outcomes, n_negotiators = 5, 3
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=3,
        offering_is_accepting=True,
        avoid_ultimatum=False,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])

    assert mechanism.state.step == 0
    file_name = mechanism.checkpoint(file)

    info = SAOMechanism.checkpoint_info(file_name)
    assert isinstance(info["time"], str)
    assert info["step"] == 0
    assert info["type"].endswith("SAOMechanism")
    assert info["id"] == mechanism.id
    assert info["name"] == mechanism.name

    mechanism, info = SAOMechanism.from_checkpoint(file_name, return_info=True)
    assert isinstance(info["time"], str)
    assert info["step"] == 0
    assert info["type"].endswith("SAOMechanism")
    assert info["id"] == mechanism.id
    assert info["name"] == mechanism.name

    assert mechanism.state.step == 0
    mechanism.step()

    file_name = mechanism.checkpoint(file)

    info = SAOMechanism.checkpoint_info(file_name)
    assert isinstance(info["time"], str)
    assert info["step"] == 1
    assert info["type"].endswith("SAOMechanism")
    assert info["id"] == mechanism.id
    assert info["name"] == mechanism.name

    mechanism, info = SAOMechanism.from_checkpoint(file_name, return_info=True)
    assert isinstance(info["time"], str)
    assert info["step"] == 1
    assert info["type"].endswith("SAOMechanism")
    assert info["id"] == mechanism.id
    assert info["name"] == mechanism.name

    mechanism.run()


@given(
    single_checkpoint=st.booleans(),
    checkpoint_every=st.integers(0, 6),
    exist_ok=st.booleans(),
)
@settings(deadline=20000, max_examples=100)
def test_auto_checkpoint(tmp_path, single_checkpoint, checkpoint_every, exist_ok):
    import shutil

    new_folder: Path = tmp_path / unique_name("empty", sep="")
    new_folder.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(new_folder)
    new_folder.mkdir(parents=True, exist_ok=True)
    filename = "mechanism"

    n_outcomes, n_negotiators = 5, 3
    n_steps = 50
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=n_steps,
        offering_is_accepting=True,
        avoid_ultimatum=False,
        checkpoint_every=checkpoint_every,
        checkpoint_folder=new_folder,
        checkpoint_filename=filename,
        extra_checkpoint_info=None,
        exist_ok=exist_ok,
        single_checkpoint=single_checkpoint,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(
            AspirationNegotiator(name=f"agent{i}"),
            ufun=ufuns[i],
            aspiration_type="conceder",
        )

    mechanism.run()

    if 0 < checkpoint_every <= n_steps:
        if single_checkpoint:
            assert len(list(new_folder.glob("*"))) == 2
        else:
            assert len(list(new_folder.glob("*"))) >= 2 * (
                max(1, mechanism.state.step // checkpoint_every)
            )
    elif checkpoint_every > n_steps:
        assert len(list(new_folder.glob("*"))) == 2
    else:
        assert len(list(new_folder.glob("*"))) == 0
