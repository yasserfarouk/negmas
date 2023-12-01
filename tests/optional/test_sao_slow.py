from __future__ import annotations

import itertools
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
from time import sleep
from typing import Sequence

import hypothesis.strategies as st
import numpy as np
import pkg_resources
import pytest
import pytest_check as check
from hypothesis import HealthCheck, example, given, settings
from pytest import mark

import negmas
from negmas import SAOSyncController
from negmas.gb.negotiators.randneg import RandomAlwaysAcceptingNegotiator
from negmas.genius import genius_bridge_is_running
from negmas.helpers import unique_name
from negmas.helpers.strings import shorten
from negmas.helpers.types import get_class
from negmas.outcomes import Outcome, make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearUtilityFunction, MappingUtilityFunction
from negmas.sao import (
    AdditiveParetoFollowingTBNegotiator,
    AspirationNegotiator,
    ConcederTBNegotiator,
    LimitedOutcomesNegotiator,
    MultiplicativeParetoFollowingTBNegotiator,
    RandomNegotiator,
    ResponseType,
    SAOMechanism,
    SAONegotiator,
    SAOResponse,
    SAOState,
    all_negotiator_types,
)
from negmas.sao.negotiators import NaiveTitForTatNegotiator, NiceNegotiator

exception_str = "Custom Exception"

NEGTYPES = all_negotiator_types()

TIME_BASED_NEGOTIATORS = [
    get_class(f"negmas.sao.negotiators.timebased.{x}")
    for x in negmas.sao.negotiators.timebased.__all__
]
TFT_NEGOTIATORS = [
    get_class(f"negmas.sao.negotiators.titfortat.{x}")
    for x in negmas.sao.negotiators.titfortat.__all__
]
ALL_BUILTIN_NEGOTIATORS = [
    get_class(f"negmas.sao.negotiators.{x}")
    for x in negmas.sao.negotiators.__all__
    if x
    not in [
        "SAONegotiator",
        "UtilBasedNegotiator",
        "make_boa",
        "MakeBoa",
        "SAOModularNegotiator",
        "BOANegotiator",
        "ControlledSAONegotiator",
        "MAPNegotiator",
        "CABNegotiator",
        "CANNegotiator",
        "CARNegotiator",
        "WABNegotiator",
        "WANNegotiator",
        "WARNegotiator",
    ]
]


class MyRaisingNegotiator(SAONegotiator):
    def propose(self, state: SAOState) -> Outcome | None:
        _ = state
        raise ValueError(exception_str)


class MySyncController(SAOSyncController):
    def __init__(
        self,
        *args,
        sleep_seconds=0.2,
        accept_after=float("inf"),
        end_after=float("inf"),
        offer_none_after=float("inf"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._sleep_seconds = sleep_seconds
        self.n_counter_all_calls = 0
        self.countered_offers: dict[int, dict[str, list[Outcome | None]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.received_offers: dict[str, dict[int, list[Outcome | None]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.sent_offers: dict[str, dict[int, list[Outcome | None]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.sent_responses: dict[str, dict[int, list[ResponseType]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.wait_states: dict[str, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.accept_after = accept_after
        self.end_after = end_after
        self.offer_none_after = offer_none_after

    def respond(self, negotiator_id, state, source: str | None = None):
        offer = state.current_offer
        response = super().respond(negotiator_id, state, source)
        self.received_offers[negotiator_id][state.step].append(offer)
        if response == ResponseType.WAIT:
            self.wait_states[negotiator_id][state.step] += 1
        else:
            self.sent_responses[negotiator_id][state.step].append(response)
        return response

    def propose(self, negotiator_id, state):
        outcome = super().propose(negotiator_id, state)
        self.sent_offers[negotiator_id][state.step].append(outcome)
        return outcome

    def counter_all(self, offers, states):
        for k, v in offers.items():
            s = states[k]
            self.countered_offers[s.step][k].append(v)

        if self._sleep_seconds:
            if isinstance(self._sleep_seconds, Sequence):
                s = (
                    random.random() * (self._sleep_seconds[1] - self._sleep_seconds[0])
                    + self._sleep_seconds[0]
                )
            else:
                s = self._sleep_seconds
            sleep(s)
        self.n_counter_all_calls += 1
        responses = [
            self.negotiators[_][0].nmi.random_outcomes(1)[0] for _ in states.keys()
        ]
        assert all(_ is not None for _ in responses)
        responses = dict(
            zip(
                states.keys(),
                (SAOResponse(ResponseType.REJECT_OFFER, _) for _ in responses),
            )
        )
        if (
            self.offer_none_after <= self.end_after
            and self.offer_none_after <= self.accept_after
        ):
            for k in responses.keys():
                if states[k].step >= self.offer_none_after:
                    responses[k] = SAOResponse(ResponseType.REJECT_OFFER, None)
        for k in responses.keys():
            if self.accept_after < self.end_after:
                if states[k].step >= self.accept_after:
                    responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                if states[k].step >= self.end_after:
                    responses[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            else:
                if states[k].step >= self.end_after:
                    responses[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                if states[k].step >= self.accept_after:
                    responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        return responses

    def first_offer(self, negotiator_id: str):
        return self.negotiators[negotiator_id][0].nmi.random_outcomes(1)[0]


class InfiniteLoopNegotiator(RandomNegotiator):
    """A negotiator that wastes time"""

    def __init__(self, *args, **kwargs):
        pa, pe, pr = 0.0, 0.0, 1.0
        kwargs["p_acceptance"] = kwargs.get("p_acceptance", pa)
        kwargs["p_ending"] = kwargs.get("p_ending", pe)
        kwargs["p_rejection"] = kwargs.get("p_rejection", pr)
        super().__init__(*args, **kwargs)
        self.__stop = False

    def __call__(self, state):
        _ = state
        while not self.__stop:
            pass

    def stop(self):
        self.__stop = True


class TimeWaster(RandomNegotiator):
    """A negotiator that wastes time"""

    def __init__(
        self,
        *args,
        sleep_seconds: float | tuple[float, float] = 0.2,
        n_waits=0,
        **kwargs,
    ):
        pa, pe, pr = 0.0, 0.0, 1.0
        kwargs["p_acceptance"] = kwargs.get("p_acceptance", pa)
        kwargs["p_ending"] = kwargs.get("p_ending", pe)
        kwargs["p_rejection"] = kwargs.get("p_rejection", pr)
        super().__init__(*args, **kwargs)
        self._sleep_seconds = sleep_seconds
        self.my_offers: dict[int, Outcome | None] = defaultdict(lambda: None)
        self.received_offers: dict[int, Outcome | None] = defaultdict(lambda: None)
        self.my_responses: dict[int, ResponseType | None] = defaultdict(lambda: None)
        self.n_waits = n_waits
        if n_waits:
            self.n_waits = random.randint(0, n_waits)
        self.waited = 0

    def __call__(self, state):
        offer = state.current_offer
        if not self.nmi:
            return None
        if self.waited < self.n_waits and state.step > 0:
            self.waited += 1
            return SAOResponse(ResponseType.WAIT, None)
        if self._sleep_seconds:
            if isinstance(self._sleep_seconds, Sequence):
                s = (
                    random.random() * (self._sleep_seconds[1] - self._sleep_seconds[0])
                    + self._sleep_seconds[0]
                )
            else:
                s = self._sleep_seconds
            sleep(s)
        assert (
            state.step not in self.received_offers
            or self.received_offers[state.step] is None
        )
        assert state.step not in self.my_offers
        assert state.step not in self.my_responses
        self.received_offers[state.step] = offer
        outcome = self.nmi.random_outcome()
        response = SAOResponse(ResponseType.REJECT_OFFER, outcome)
        assert response.outcome is not None
        self.my_offers[state.step] = response.outcome
        self.my_responses[state.step] = response.response
        return response


def test_hidden_time_limit_words():
    n_outcomes, n_steps, tlimit = 100, 10, 1
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=n_steps,
        ignore_negotiator_exceptions=True,
        hidden_time_limit=tlimit,
        step_time_limit=float("inf"),
    )
    ufuns = MappingUtilityFunction.generate_random(
        2, outcomes=mechanism.discrete_outcomes()
    )
    mechanism.add(InfiniteLoopNegotiator(name=f"agent{0}", preferences=ufuns[0]))  # type: ignore
    mechanism.add(InfiniteLoopNegotiator(name=f"agent{1}", preferences=ufuns[1]))  # type: ignore
    mechanism.run()
    assert mechanism.state.agreement is None
    assert mechanism.state.started
    assert mechanism.state.timedout
    assert mechanism.state.step < n_steps
    assert mechanism._start_time is not None
    assert time.perf_counter() - mechanism._start_time >= tlimit
    assert not mechanism.state.waiting
    for negotiator in mechanism.negotiators:
        negotiator.stop()


def test_neg_run_no_waiting():
    n_outcomes, n_steps, waste = 10, 10, 0.5
    mechanism = SAOMechanism(
        outcomes=n_outcomes, n_steps=n_steps, ignore_negotiator_exceptions=True
    )
    ufuns = MappingUtilityFunction.generate_random(
        2, outcomes=mechanism.discrete_outcomes()
    )
    mechanism.add(
        TimeWaster(name=f"agent{0}", sleep_seconds=waste, preferences=ufuns[0])
    )
    mechanism.add(
        TimeWaster(name=f"agent{1}", sleep_seconds=waste, preferences=ufuns[1])
    )
    mechanism.run()
    assert mechanism.state.agreement is None
    assert mechanism.state.started
    assert mechanism.state.timedout
    assert mechanism.state.step == n_steps
    assert not mechanism.state.waiting
    assert len(mechanism.history) == n_steps
    for _, v in mechanism.stats["times"].items():
        assert v >= waste * n_steps


@mark.parametrize(["keep_order"], [(True,), (False,)])
def test_neg_sync_loop(keep_order):
    # from pprint import pprint

    n_outcomes, n_steps = 10, 10
    waste_center = 0.1
    c1 = MySyncController(sleep_seconds=waste_center, name="c1")
    c2 = MySyncController(sleep_seconds=waste_center, name="c2")
    mechanisms = []
    for m in range(2):
        mechanism = SAOMechanism(
            outcomes=n_outcomes,
            n_steps=n_steps,
            ignore_negotiator_exceptions=False,
            name=f"{m}",
        )
        ufuns = MappingUtilityFunction.generate_random(
            2, outcomes=mechanism.discrete_outcomes()
        )
        mechanism.add(
            c1.create_negotiator(preferences=ufuns[0], id=f"0-{m}", name=f"0-{m}")  # type: ignore It will be SAOControlled
        )
        mechanism.add(
            c2.create_negotiator(preferences=ufuns[1], id=f"1-{m}", name=f"1-{m}")  # type: ignore It will be SAOControlled
        )
        mechanisms.append(mechanism)
    SAOMechanism.runall(mechanisms, keep_order=keep_order)

    for mechanism in mechanisms:
        assert mechanism.state.started
        assert mechanism.state.agreement is None
        assert not mechanism.state.has_error
        assert not mechanism.state.broken
        assert mechanism.state.timedout
        assert mechanism.state.step == n_steps
        assert not mechanism.state.waiting
        assert len(mechanism.history) == n_steps
        # assert c1.n_counter_all_calls == n_steps
        # assert c2.n_counter_all_calls == n_steps


@mark.parametrize(["n_negotiators"], [(1,), (2,), (3,)])
def test_neg_run_sync(n_negotiators):
    n_outcomes, n_steps = 10, 10
    waste_edge, waste_center = 0.2, 0.1
    c = MySyncController(sleep_seconds=waste_center)
    mechanisms, edge_names = [], []
    for _ in range(n_negotiators):
        mechanism = SAOMechanism(
            outcomes=n_outcomes,
            n_steps=n_steps,
            ignore_negotiator_exceptions=True,
        )
        ufuns = MappingUtilityFunction.generate_random(
            2, outcomes=mechanism.discrete_outcomes()
        )
        edge_names.append(f"f{0}")
        mechanism.add(
            TimeWaster(
                name=f"agent{0}",
                id=f"a{0}",
                sleep_seconds=waste_edge,
                preferences=ufuns[0],
            )
        )
        mechanism.add(c.create_negotiator(preferences=ufuns[1]))  # type: ignore IT will be the correct type
        mechanisms.append(mechanism)
    SAOMechanism.runall(mechanisms)

    for mechanism in mechanisms:
        assert mechanism.state.started
        assert mechanism.state.agreement is None
        assert not mechanism.state.has_error, f"{mechanism.state.error_details}"
        assert not mechanism.state.broken
        assert mechanism.state.timedout, f"Did not timeout!!\n{mechanism.state}"
        assert mechanism.state.step == n_steps
        assert not mechanism.state.waiting
        assert len(mechanism.history) == n_steps
        delay_center = 0.0
        for k, v in mechanism.stats["times"].items():
            if k in edge_names:
                assert v >= waste_edge * n_steps
            else:
                delay_center += v
        assert delay_center > n_steps * waste_center
        assert c.n_counter_all_calls == n_steps


def test_exceptions_are_saved():
    n_outcomes, n_negotiators = 10, 2
    mechanism = SAOMechanism(
        outcomes=n_outcomes, n_steps=n_outcomes, ignore_negotiator_exceptions=True
    )
    ufuns = MappingUtilityFunction.generate_random(
        n_negotiators, outcomes=mechanism.discrete_outcomes()
    )
    mechanism.add(AspirationNegotiator(name=f"agent{0}"), preferences=ufuns[0])
    mechanism.add(MyRaisingNegotiator(name=f"agent{1}"), preferences=ufuns[1])
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism.state.step == 1
    mechanism.step()
    mechanism.step()
    assert mechanism.state.current_offer is not None
    assert len(mechanism.stats) == 3
    stats = mechanism.stats
    assert "times" in stats.keys()
    assert "exceptions" in stats.keys()
    assert stats["exceptions"] is not None
    assert len(stats["exceptions"]) == 1
    # print(stats["exceptions"][mechanism.negotiators[1].id])
    assert len(stats["exceptions"][mechanism.negotiators[1].id]) == 1
    assert exception_str in stats["exceptions"][mechanism.negotiators[1].id][0]
    assert len(stats["exceptions"][mechanism.negotiators[0].id]) == 0


def test_on_negotiation_start():
    mechanism = SAOMechanism(outcomes=10, n_steps=10)
    assert mechanism.on_negotiation_start()


@mark.parametrize(["n_negotiators"], [(2,), (3,)])
def test_round_n_agents(n_negotiators):
    n_outcomes = 5
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), preferences=ufuns[i])
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism.state.step == 1
    assert mechanism.state.current_offer is not None


@mark.parametrize(["n_negotiators"], [(2,), (3,)])
def test_mechanism_can_run(n_negotiators):
    n_outcomes = 5
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=3)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), preferences=ufuns[i])
    assert mechanism.state.step == 0
    mechanism.step()
    mechanism.run()


@mark.parametrize("n_negotiators,oia", [(2, True), (3, True), (2, False), (3, False)])
def test_mechanism_runs_with_offering_not_accepting(n_negotiators, oia):
    n_outcomes = 5
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=3, offering_is_accepting=oia)
    ufuns = MappingUtilityFunction.generate_random(1, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), preferences=ufuns[0])
    assert mechanism.state.step == 0
    mechanism.step()
    assert mechanism._current_proposer and mechanism._current_proposer.name == "agent0"
    assert mechanism.state.n_acceptances == n_negotiators + int(oia) - 1
    assert (mechanism.agreement is not None) is oia
    if mechanism.agreement is not None:
        return
    mechanism.step()
    assert mechanism._current_proposer.name == "agent0"
    assert mechanism.state.n_acceptances == n_negotiators
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
        )
        ufuns = MappingUtilityFunction.generate_random(1, outcomes=n_outcomes)
        for i in range(n_negotiators):
            mechanism.add(AspirationNegotiator(name=f"agent{i}"), preferences=ufuns[0])
        mechanisms.append(mechanism)

    states = SAOMechanism.runall(mechanisms)
    assert len(states) == 10
    assert not any(_ is not None and _.running for _ in states)


class MySAOSync(SAOSyncController):
    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        responses = {}
        for source in offers.keys():
            _, state = offers[source], states[source]
            if state.step < 2:
                responses[source] = SAOResponse(ResponseType.REJECT_OFFER, None)
            else:
                responses[source] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        return responses


@mark.parametrize(
    "n_steps,n_negotiators,oia",
    [
        (2, 2, True),
        (2, 3, True),
        (30, 2, True),
        (30, 3, True),
        (2, 2, False),
        (2, 3, False),
        (30, 2, False),
        (30, 3, False),
    ],
)
def test_random_offer_guaranteed_acceptance(n_steps, n_negotiators, oia):
    n_outcomes = 10
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=n_steps,
        time_limit=None,
        offering_is_accepting=oia,
        end_on_no_response=False,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for u in ufuns:
        u.reserved_value = float(u.min()) - 0.1
    for i in range(n_negotiators):
        mechanism.add(
            RandomAlwaysAcceptingNegotiator(name=f"agent{i}"), preferences=ufuns[i]
        )

    state = mechanism.run()
    assert state.agreement


@mark.parametrize(
    "n_negs,n_negotiators,oia",
    [
        (2, 2, True),
        (2, 3, True),
        (3, 2, True),
        (1, 2, True),
        (1, 3, True),
        (2, 2, False),
        (2, 3, False),
        (3, 2, False),
        (1, 2, False),
        (1, 3, False),
    ],
)
def test_sync_controller(n_negs, n_negotiators, oia):
    n_outcomes = 2

    mechanisms = []
    controller = MySAOSync()
    for i in range(n_negs):
        mechanisms.append(
            SAOMechanism(
                outcomes=n_outcomes,
                n_steps=30,
                time_limit=None,
                offering_is_accepting=oia,
                end_on_no_response=False,
            )
        )
        ufuns = MappingUtilityFunction.generate_random(
            n_negotiators, outcomes=n_outcomes
        )
        for u in ufuns:
            u.reserved_value = float(u.min()) - 0.1
        for i in range(n_negotiators):
            mechanisms[-1].add(
                RandomAlwaysAcceptingNegotiator(name=f"agent{i}"), preferences=ufuns[i]
            )

        mechanisms[-1].add(controller.create_negotiator())

    states = SAOMechanism.runall(mechanisms)
    assert all(_ is not None and _.agreement is not None for _ in states), str(
        [_.agreement if _ else "No State" for _ in states]
    )


def test_pickling_mechanism(tmp_path):
    import dill as pickle

    file = tmp_path / "mechanism.pck"
    n_outcomes, n_negotiators = 5, 3
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=3,
        offering_is_accepting=True,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), preferences=ufuns[i])

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


@mark.xfail(
    run=False,
    reason="Checkpointing is known to fail with UtilityInverter. As this whole thing will be changed, we may just wait for now",
)
def test_checkpointing_mechanism(tmp_path):
    file = tmp_path
    n_outcomes, n_negotiators = 5, 3
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=3,
        offering_is_accepting=True,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), preferences=ufuns[i])

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
    assert isinstance(mechanism, SAOMechanism)

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
    assert isinstance(mechanism, SAOMechanism)
    assert isinstance(info["time"], str)
    assert info["step"] == 1
    assert info["type"].endswith("SAOMechanism")
    assert info["id"] == mechanism.id
    assert info["name"] == mechanism.name

    mechanism.run()


@mark.parametrize(["n_negs"], [(_,) for _ in range(1, 10)])
def test_sync_controller_gets_all_offers(n_negs):
    from negmas.mechanisms import Mechanism
    from negmas.sao import RandomNegotiator, SAORandomSyncController

    class MyController(SAORandomSyncController):
        def counter_all(self, offers, states):
            assert len(offers) == len(self.active_negotiators)
            responses = super().counter_all(offers, states)
            assert len(responses) == len(offers)
            return responses

    c = MyController()
    negs = tuple(
        SAOMechanism(issues=[make_issue((0.0, 1.0), "price")], n_steps=20)
        for _ in range(n_negs)
    )
    for neg in negs:
        neg.add(RandomNegotiator())
        neg.add(c.create_negotiator())

    Mechanism.runall(negs, True)


# @given(n_negs=st.integers(1, 10), strict=st.booleans())
@mark.parametrize(
    ["n_negs", "strict"], [(_, s) for s in (True, False) for _ in range(1, 10)]
)
def test_single_agreement_gets_one_agreement(n_negs, strict):
    from negmas.mechanisms import Mechanism
    from negmas.sao import AspirationNegotiator, SAOSingleAgreementRandomController

    os = make_os([make_issue((0.0, 1.0), "price")])
    c = SAOSingleAgreementRandomController(strict=strict)

    negs = tuple(
        SAOMechanism(
            outcome_space=os,
            n_steps=50,
            end_on_no_response=False,
        )
        for _ in range(n_negs)
    )
    for i, neg in enumerate(negs):
        neg.add(
            AspirationNegotiator(aspiration_type="linear", name=f"opponent-{i}"),
            preferences=LinearUtilityFunction(weights=[1.0], outcome_space=os),
        )
        neg.add(c.create_negotiator(name=f"against-{i}"))

    Mechanism.runall(negs, True)
    agreements = [neg.state.agreement for neg in negs]
    if strict:
        # assert that the controller never had multiple agreements
        assert sum(_ is not None for _ in agreements) == 1
    else:
        # assert that the controller never accepted twice. It may still have multiple agreements though
        assert (
            len(
                [
                    neg.state.agreement
                    for neg in negs
                    if neg.state.agreement is not None
                    and neg.state.current_proposer is not None
                    and neg.state.current_proposer.startswith("opponent")
                ]
            )
            < 2
        )


@mark.parametrize(["keep_order"], [(False,), (True,)])
def test_loops_are_broken(keep_order):
    """Tests that loops formed by concurrent negotiations are broken for syn controllers"""
    from negmas.mechanisms import Mechanism
    from negmas.sao import SAOSingleAgreementAspirationController

    issues = [make_issue((0.0, 1.0), "price")]

    a, b, c = (
        SAOSingleAgreementAspirationController(
            preferences=MappingUtilityFunction(lambda x: x[0], issues=issues),
            strict=False,
        ),
        SAOSingleAgreementAspirationController(
            preferences=MappingUtilityFunction(lambda x: x[0], issues=issues),
            strict=False,
        ),
        SAOSingleAgreementAspirationController(
            preferences=MappingUtilityFunction(lambda x: x[0], issues=issues),
            strict=False,
        ),
    )

    n1 = SAOMechanism(
        name="ab",
        issues=issues,
        n_steps=50,
    )
    n2 = SAOMechanism(
        name="ac",
        issues=issues,
        n_steps=50,
    )
    n3 = SAOMechanism(
        name="bc",
        issues=issues,
        n_steps=50,
        end_on_no_response=False,
    )

    n1.add(a.create_negotiator(name="a>b"))
    n1.add(b.create_negotiator(name="b>a"))
    n2.add(a.create_negotiator(name="a>c"))
    n2.add(c.create_negotiator(name="c>a"))
    n3.add(b.create_negotiator(name="b>c"))
    n3.add(c.create_negotiator(name="c>b"))
    negs = (n1, n2, n3)
    Mechanism.runall(negs, keep_order)

    agreements = [neg.state.agreement for neg in negs]
    assert not keep_order or sum(_ is not None for _ in agreements) > 0
    # TODO check why sometimes we get no agreements when order is not kept


@given(
    typ=st.sampled_from(NEGTYPES),
)
def test_can_create_all_negotiator_types(typ):
    issues = [make_issue((0.0, 1.0), name="price"), make_issue(10, name="quantity")]
    params = dict(
        preferences=LinearUtilityFunction(
            issues=issues, weights=dict(price=1.0, quantity=1.0)
        )
    )
    assert typ(**params) is not None


@given(
    a=st.sampled_from(NEGTYPES),
    b=st.sampled_from(NEGTYPES),
    w1p=st.floats(-1.0, 1.0),
    w1q=st.floats(-1.0, 1.0),
    w2p=st.floats(-1.0, 1.0),
    w2q=st.floats(-1.0, 1.0),
    r1=st.floats(-1.0, 1.0),
    r2=st.floats(-1.0, 1.0),
)
@settings(deadline=100_000, max_examples=50)
def test_can_run_all_negotiators(a, b, w1p, w1q, w2p, w2q, r1, r2):
    issues = [make_issue((0.0, 1.0), name="price"), make_issue(10, name="quantity")]
    u1 = LinearUtilityFunction(weights=[w1p, w1q], issues=issues, reserved_value=r1)
    u2 = LinearUtilityFunction(weights=[w2p, w2q], issues=issues, reserved_value=r2)
    m = SAOMechanism(n_steps=30, issues=issues)
    m.add(a(preferences=u1))
    m.add(b(), preferences=u2)
    m.run()
    assert not m.running


def test_can_run_asp_tit():
    b, a = AspirationNegotiator, NaiveTitForTatNegotiator
    issues = [make_issue((0.0, 1.0), name="price"), make_issue(10, name="quantity")]
    u1 = LinearUtilityFunction(weights=[0.0, 0.0], issues=issues, reserved_value=0.0)
    u2 = LinearUtilityFunction(weights=[0.0, 0.0], issues=issues, reserved_value=0.0)
    m = SAOMechanism(n_steps=30, issues=issues)
    m.add(a(preferences=u1))
    m.add(b(), preferences=u2)
    m.run()
    assert not m.running


def test_acceptable_outcomes():
    p = SAOMechanism(outcomes=6, n_steps=10)
    p.add(
        LimitedOutcomesNegotiator(name="seller", acceptable_outcomes=[(2,), (3,), (5,)])
    )
    p.add(
        LimitedOutcomesNegotiator(name="buyer", acceptable_outcomes=[(1,), (4,), (3,)])
    )
    state = p.run()
    assert state.agreement == (3,), f"{p.extended_trace}"


class MyNegotiator(SAONegotiator):
    def propose(self, state):
        _ = state
        return (3.0, 2, 1.0)

    def respond(self, state, source=None):
        _ = source
        if state.step < 5:
            return ResponseType.REJECT_OFFER
        return ResponseType.ACCEPT_OFFER


def test_cast_offers_tuple():
    issues = [make_issue(10), make_issue(5), make_issue(3)]
    m = SAOMechanism(
        issues=issues,
        check_offers=True,
        enforce_issue_types=True,
        cast_offers=True,
        n_steps=10,
        end_on_no_response=True,
    )
    m.add(MyNegotiator())
    m.add(MyNegotiator())
    m.run()
    assert m.agreement is not None
    assert all(isinstance(_, int) for _ in m.agreement)
    assert not any(isinstance(_, float) for _ in m.agreement)
    assert m.agreement == (3, 2, 1)


def test_fail_on_incorrect_types_tuple_or_dict():
    issues = [make_issue(10), make_issue(5), make_issue(3)]
    m = SAOMechanism(
        issues=issues,
        check_offers=True,
        enforce_issue_types=True,
        cast_offers=False,
        n_steps=10,
        end_on_no_response=True,
    )
    m.add(MyNegotiator())
    m.add(MyNegotiator())
    m.run()
    assert m.agreement is None


def test_no_check_offers_tuple():
    issues = [make_issue(10), make_issue(5), make_issue(3)]
    for a, b in ((True, False), (False, False), (False, True), (True, True)):
        m = SAOMechanism(
            issues=issues,
            check_offers=False,
            enforce_issue_types=a,
            cast_offers=b,
            n_steps=10,
            end_on_no_response=True,
        )
        m.add(MyNegotiator())
        m.add(MyNegotiator())
        m.run()
        assert m.agreement is not None
        assert isinstance(m.agreement[0], float) and not isinstance(m.agreement[0], int)
        assert not isinstance(m.agreement[1], float) and isinstance(m.agreement[1], int)
        assert isinstance(m.agreement[2], float) and not isinstance(m.agreement[2], int)
        assert m.agreement == (3.0, 2, 1.0)


def test_no_limits_raise_warning():
    from negmas.inout import load_genius_domain_from_folder

    with pytest.warns(UserWarning):
        folder_name = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/cameradomain"
        )

        load_genius_domain_from_folder(
            folder_name
        ).normalize().to_single_issue().make_session()


@given(
    n_steps=st.integers(10, 11),
    n_waits=st.integers(0, 4),
    n_waits2=st.integers(0, 4),
)
@settings(deadline=20000, max_examples=100)
def test_single_mechanism_history_with_waiting(n_steps, n_waits, n_waits2):
    n_outcomes, waste = 5, (0.0, 0.3)
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=n_steps,
        ignore_negotiator_exceptions=False,
    )
    assert mechanism.outcomes
    ufuns = MappingUtilityFunction.generate_random(2, outcomes=mechanism.outcomes)
    mechanism.add(
        TimeWaster(
            name=f"agent{0}", sleep_seconds=waste, preferences=ufuns[0], n_waits=n_waits
        )
    )
    mechanism.add(
        TimeWaster(
            name=f"agent{1}",
            sleep_seconds=waste,
            preferences=ufuns[1],
            n_waits=n_waits2,
        )
    )
    mechanism.run()
    first = mechanism._selected_first
    assert first == 0
    assert mechanism.state.agreement is None
    assert mechanism.state.started
    assert mechanism.state.timedout or (
        n_waits + n_waits2 > 0 and mechanism.state.broken
    )
    assert mechanism.state.step == n_steps or (
        n_waits + n_waits2 > 0
        and mechanism.state.broken
        and mechanism.state.step <= n_steps
    )
    assert not mechanism.state.waiting
    assert len(mechanism.history) == n_steps or (
        n_waits + n_waits2 > 0
        and mechanism.state.broken
        and len(mechanism.history) <= n_steps
    )

    # check history details is correct
    s = [defaultdict(int), defaultdict(int)]
    r = [defaultdict(int), defaultdict(int)]
    h = [defaultdict(int), defaultdict(int)]
    first_offers = []
    for i, n in enumerate(mechanism.negotiators):
        first_offers.append(n.received_offers[0] is None)

        # sent and received match
        for j, w in n.my_offers.items():
            # cannot send mutlipe offers in the same step
            assert j not in s[i].keys()
            # cannot send None
            assert w is not None  # or (j == 0 and not avoid_ultimatum)
            s[i][j] = w[0]
        for j, w in n.received_offers.items():
            # cannot receive multiple ofers in the same step
            assert j not in r[i].keys()
            # canont be asked to start offering except in the first step
            assert w is not None or j == 0
            # this is the first agent to offer, ignore its first step
            if first == i and j == 0:
                # if this is the first agent, its first thing recieved must be None
                assert w is None
                continue
            assert w is not None, f"None outcome agent {i} @ {j} first is {first}"
            r[i][j] = w[0]

    assert any(first_offers) and not all(first_offers)

    # reconstruct history
    neg_map = dict(zip((_.id for _ in mechanism.negotiators), [0, 1]))
    for state in mechanism.history:
        assert isinstance(state, SAOState)
        for _, w in state.new_offers:
            a = neg_map[_]
            # cannot see the same step twice in the history of an agent
            assert state.step not in h[a].keys()
            assert w is not None
            h[a][state.step] = w[0]

    # no gaps in steps and all step sets start with 0 or 1
    for i in range(len(mechanism.negotiators)):
        for x in (r, s, h):
            steps = list(x[i].keys())
            if not steps:
                continue
            assert steps[0] in (0, 1)
            for _ in range(len(steps) - 1):
                assert steps[_] + 1 == steps[_ + 1]

    # pprint([s, r, h])
    # history matches what is stored inside agents
    for i, n in enumerate(mechanism.negotiators):
        for j, w in s[i].items():
            assert j in r[1 - i] or (j == 0)
        for j, w in r[i].items():
            assert j in s[1 - i]

    # s and r will not have matched indices but should have matched values
    s = [list(_.values()) for _ in s]
    r = [list(_.values()) for _ in r]
    h2 = [list(_ for _ in _.values()) for _ in h]
    # history matches what is stored inside agents
    for i, n in enumerate(mechanism.negotiators):
        for j, w in enumerate(s[i]):
            if j < len(r[1 - i]):
                assert r[1 - i][j] == w
            assert h2[i][j] == w

        for j, w in enumerate(r[i]):
            assert s[1 - i][j] == w


@pytest.mark.skip(reason="Known Bug")
@given(
    keep_order=st.booleans(),
    n_first=st.integers(1, 6),
    n_second=st.integers(1, 6),
    end_on_no_response=st.booleans(),
)
@settings(deadline=20000)
def test_neg_sync_loop_receives_all_offers(
    keep_order, n_first, n_second, end_on_no_response
):
    # from pprint import pprint

    n_outcomes, n_steps = 100, 10
    waste_center = 0.01
    c1s = [
        MySyncController(sleep_seconds=waste_center, name="c1") for _ in range(n_first)
    ]
    c2s = [
        MySyncController(sleep_seconds=waste_center, name="c2") for _ in range(n_second)
    ]
    mechanisms = tuple(
        SAOMechanism(
            outcomes=n_outcomes,
            n_steps=n_steps,
            ignore_negotiator_exceptions=False,
            end_on_no_response=end_on_no_response,
            name=f"{i}v{j}",
        )
        for i in range(n_first)
        for j in range(n_second)
    )
    for mechanism in mechanisms:
        assert mechanism.outcomes
        ufuns = MappingUtilityFunction.generate_random(2, outcomes=mechanism.outcomes)
        i, j = tuple(int(_) for _ in mechanism.name.split("v"))
        mechanism.add(
            c1s[i].create_negotiator(
                preferences=ufuns[0], id=f"l{i}>{j}", name=f"l{i}>{j}"
            )
        )
        mechanism.add(
            c2s[j].create_negotiator(
                preferences=ufuns[1], id=f"r{j}>{i}", name=f"r{j}>{i}"
            )
        )

    while True:
        states = SAOMechanism.stepall(mechanisms, keep_order=keep_order)
        if all(not _.running for _ in states):
            break

    for mechanism in mechanisms:
        ls = [len(mechanism.negotiator_offers(n.id)) for n in mechanism.negotiators]
        check.less_equal(abs(ls[0] - ls[1]), 1, str(mechanism.trace))
        check.equal(len(mechanism.offers), 2 * n_steps)
        for n in mechanism.negotiators:
            check.equal(len(mechanism.negotiator_offers(n.id)), n_steps)

    for c, n_partners in itertools.chain(
        zip(c1s, itertools.repeat(n_second)), zip(c2s, itertools.repeat(n_first))
    ):
        steps = set(range(1, n_steps))
        countered_steps = set(c.countered_offers.keys())
        missing_steps = steps.difference(countered_steps)
        check.equal(len(missing_steps), 0, f"Missing steps are {missing_steps}")
        for s in steps:
            partners = c.countered_offers[s]
            check.equal(
                len(partners),
                n_partners,
                f"partners {len(partners)} (of {n_partners}) step {s}.\n{partners}",
            )

    for mechanism in mechanisms:
        check.is_true(mechanism.state.started)
        check.is_none(mechanism.state.agreement)
        check.is_false(mechanism.state.has_error)
        check.is_false(mechanism.state.broken)
        check.is_false(mechanism.state.waiting)
        check.is_true(mechanism.state.timedout)
        check.equal(mechanism.state.step, n_steps)
        check.equal(len(mechanism.history), n_steps)


@given(
    n_outcomes=st.integers(5, 10),
    n_negotiators=st.integers(2, 4),
    n_steps=st.integers(1, 4),
)
@settings(deadline=None)
def test_times_are_calculated(n_outcomes, n_negotiators, n_steps):
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=8)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), preferences=ufuns[i])
    assert mechanism.state.step == 0
    _strt = time.perf_counter()
    for _ in range(n_steps):
        # print(f"Stepping: {_}")
        mechanism.step()
    time.sleep(0.01)
    duration = time.perf_counter() - _strt
    # assert mechanism.current_step == n_steps
    assert mechanism.state.current_offer is not None
    assert len(mechanism.stats) == 3
    stats = mechanism.stats
    assert "round_times" in stats.keys()
    assert 0 < sum(stats["round_times"]) < duration
    assert "times" in stats.keys()
    assert "exceptions" in stats.keys()
    assert stats["times"] is not None
    assert stats["exceptions"] is not None
    assert len(stats["times"]) == n_negotiators
    assert len(stats["exceptions"]) == 0
    for i in range(n_negotiators):
        assert 0 < stats["times"][mechanism.negotiators[i].id] < duration
        assert len(stats["exceptions"][mechanism.negotiators[i].id]) == 0
    assert 0 < sum(stats["times"].values()) < duration


@given(
    n_negotiators=st.integers(2, 4),
    n_issues=st.integers(1, 3),
    presort=st.booleans(),
    stochastic=st.booleans(),
)
@settings(deadline=20000, max_examples=100)
@example(n_negotiators=2, n_issues=1, presort=False, stochastic=False)
def test_aspiration_continuous_issues(n_negotiators, n_issues, presort, stochastic):
    issues = [make_issue(values=(0.0, 1.0), name=f"i{i}") for i in range(n_issues)]
    for _ in range(5):
        mechanism = SAOMechanism(
            issues=issues,
            n_steps=10,
        )
        ufuns = [
            LinearUtilityFunction(
                issues=issues,
                weights=[3.0 * random.random(), 2.0 * random.random()],
                reserved_value=0.0,
            )
            for _ in range(n_negotiators)
        ]
        i = 0
        assert mechanism.add(
            AspirationNegotiator(
                name=f"agent{i}",
                presort=presort,
                stochastic=stochastic,
                preferences=ufuns[i],
            )
        ), "Cannot add negotiator"
        for i in range(1, n_negotiators):
            assert mechanism.add(
                AspirationNegotiator(
                    name=f"agent{i}",
                    presort=presort,
                    stochastic=stochastic,
                ),
                preferences=ufuns[i],
            ), "Cannot add negotiator"
        assert mechanism.state.step == 0
        agents = dict(zip([_.id for _ in mechanism.negotiators], mechanism.negotiators))
        mechanism.run()
        for neg, neg_id in zip(mechanism.negotiators, mechanism.negotiator_ids):
            assert neg_id in agents.keys()
            utils = [neg.ufun(_) for _ in mechanism.negotiator_offers(neg_id)]
            if presort:
                if stochastic:
                    assert all(
                        utils[_] <= utils[0] + neg.tolerance for _ in range(len(utils))
                    ), f"{utils}"
                else:
                    assert all(
                        utils[i] <= utils[j] + neg.tolerance
                        for i, j in zip(range(1, len(utils)), range(0, len(utils) - 1))
                    ), f"{utils}"


@given(
    single_checkpoint=st.booleans(),
    checkpoint_every=st.integers(0, 6),
    exist_ok=st.booleans(),
)
@settings(
    deadline=20000,
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
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
            AspirationNegotiator(
                name=f"agent{i}",
                aspiration_type="conceder",
            ),
            preferences=ufuns[i],
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


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_in_sao_with_time_limit_and_nsteps_raises_warning():
    from negmas.genius import GeniusNegotiator
    from negmas.inout import load_genius_domain_from_folder

    with pytest.warns(UserWarning, match=".*has a .*"):
        folder_name = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/cameradomain"
        )

        d = load_genius_domain_from_folder(folder_name)
        mechanism = d.make_session(n_steps=60, time_limit=180)
        a1 = GeniusNegotiator(
            java_class_name="agents.anac.y2017.ponpokoagent.PonPokoAgent",
            domain_file_name=d.outcome_space.name,
            utility_file_name=d.ufuns[0].name,
        )
        mechanism.add(a1)


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_in_sao_with_time_limit_or_nsteps_raises_no_warning():
    from negmas.genius import GeniusNegotiator
    from negmas.inout import load_genius_domain_from_folder

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        folder_name = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/cameradomain"
        )
        d = load_genius_domain_from_folder(folder_name)
        mechanism = d.make_session(n_steps=None, time_limit=180)
        a1 = GeniusNegotiator(
            java_class_name="agents.anac.y2017.ponpokoagent.PonPokoAgent",
            domain_file_name=d.outcome_space.name,
            utility_file_name=d.ufuns[0].name,
        )
        mechanism.add(a1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        folder_name = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/cameradomain"
        )
        d = load_genius_domain_from_folder(folder_name)
        mechanism = d.make_session(n_steps=60, time_limit=None)
        a1 = GeniusNegotiator(
            java_class_name="agents.anac.y2017.ponpokoagent.PonPokoAgent",
            domain_file_name=d.outcome_space.name,
            utility_file_name=d.ufuns[0].name,
        )
        mechanism.add(a1)


def make_mapping():
    outcome_space = make_os([make_issue(50)])
    outcomes = list(outcome_space.enumerate_or_sample())
    u1 = MappingUtilityFunction(
        dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes)).tolist())),
        outcomes=outcomes,
    )
    u2 = MappingUtilityFunction(
        dict(zip(outcomes, (1 - np.linspace(0.0, 1.0, len(outcomes))).tolist())),
        outcomes=outcomes,
    )
    return u1, u2, outcome_space


def make_linear():
    issues = [make_issue((1, 5)), make_issue((1, 10))]
    outcome_space = make_os(issues)
    u1 = LinearUtilityFunction([-0.75, 0.25], outcome_space=outcome_space)
    u2 = LinearUtilityFunction([0.5, -0.5], outcome_space=outcome_space)
    return u1, u2, outcome_space


def _run_neg(agents, utils, outcome_space):
    neg = SAOMechanism(outcome_space=outcome_space, n_steps=200, time_limit=None)
    for a, u in zip(agents, utils):
        neg.add(a, preferences=u)
    neg.run()
    assert neg.state.agreement is not None, f"No agreement!!\n{neg.trace}"
    for a, u in zip(agents, utils):
        offers = neg.negotiator_offers(a.id)
        _, best = u.extreme_outcomes()
        assert (
            isinstance(a, ConcederTBNegotiator)
            or isinstance(a, AdditiveParetoFollowingTBNegotiator)
            or isinstance(a, MultiplicativeParetoFollowingTBNegotiator)
            or u(offers[0]) >= (u(best) - 1e-4)
        ), f"Did not start with its best offer {best} (u = {u(best)}) but used {offers[0]} (u = {u(offers[0])})"
        if isinstance(a, AspirationNegotiator):
            for i, offer in enumerate(_ for _ in offers):
                assert i == 0 or u(offer) <= u(offers[i - 1]), f"Not always conceding"
                # if not(i == 0 or u(offer) <= u(offers[i - 1])):
                #     import matplotlib.pyplot as plt
                #     neg.plot()
                #     plt.show()
                #     raise AssertionError("Not always conceding")
    return neg


@given(
    typ=st.sampled_from(TIME_BASED_NEGOTIATORS),
    linear=st.booleans(),
    starting=st.booleans(),
    opp=st.sampled_from((None, NaiveTitForTatNegotiator)),
)
@settings(deadline=10_000, max_examples=10)
def test_bilateral_timebased(typ, linear, starting, opp):
    if opp is None and starting:
        return
    if opp is None:
        opp = typ
    if starting:
        a1 = typ(name=f"{typ.__name__}[0]")
        a2 = opp(name=f"{opp.__name__}[1]")
    else:
        a1 = opp(name=f"{opp.__name__}[0]")
        a2 = typ(name=f"{typ.__name__}[1]")
    u1, u2, outcome_space = make_linear() if linear else make_mapping()
    _run_neg((a1, a2), (u1, u2), outcome_space)


def test_bilateral_timebased_example():
    typ = negmas.sao.negotiators.timebased.TimeBasedNegotiator
    linear = False
    starting = False
    opp = None

    if opp is None and starting:
        return
    if opp is None:
        opp = typ
    if starting:
        a1 = typ(name=f"{typ.__name__}[0]")
        a2 = opp(name=f"{opp.__name__}[1]")
    else:
        a1 = opp(name=f"{opp.__name__}[0]")
        a2 = typ(name=f"{typ.__name__}[1]")
    u1, u2, outcome_space = make_linear() if linear else make_mapping()
    _run_neg((a1, a2), (u1, u2), outcome_space)


@given(
    neg_types=st.lists(st.sampled_from(TIME_BASED_NEGOTIATORS), min_size=4, max_size=4)
)
@settings(deadline=10_000, max_examples=20)
def test_multilateral_timebased(neg_types):
    outcome_space = make_os([make_issue(10)])
    outcomes = list(outcome_space.enumerate_or_sample())

    negotiators = [typ(name=f"{typ.__name__}") for typ in neg_types]
    utils = [
        MappingUtilityFunction(
            dict(zip(outcomes, np.random.rand(len(outcomes)).tolist())),
            outcomes=outcomes,
        )
        for _ in neg_types
    ]
    _run_neg(negotiators, utils, outcome_space)


def try_negotiator(
    cls, replace_buyer=True, replace_seller=True, plot=False, n_steps=20
):
    if isinstance(cls, str):
        cls = get_class(cls)
    from negmas.preferences import LinearAdditiveUtilityFunction
    from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

    buyer_cls = cls if replace_buyer else AspirationNegotiator
    seller_cls = cls if replace_seller else AspirationNegotiator

    # create negotiation agenda (issues)
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]

    # create the mechanism
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    # define ufuns
    seller_utility = LinearAdditiveUtilityFunction(
        values={
            "price": IdentityFun(),
            "quantity": LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9),
        },
        weights={"price": 1.0, "quantity": 1.0, "delivery_time": 10.0},
        outcome_space=session.outcome_space,
        reserved_value=0.0,
    )
    buyer_utility = LinearAdditiveUtilityFunction(
        values={
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.0,
    )

    session.add(
        buyer_cls(name=f"buyer-{shorten(buyer_cls.__name__)}"), ufun=buyer_utility
    )
    session.add(
        seller_cls(name=f"seller-{shorten(buyer_cls.__name__)}"), ufun=seller_utility
    )
    session.run()
    if plot:
        session.plot()
    return session


# @pytest.mark.parametrize("negotiator", [MultiplicativeParetoFollowingTBNegotiator])
@given(negotiator=st.sampled_from(ALL_BUILTIN_NEGOTIATORS))
@example(negotiator=NiceNegotiator)
@example(negotiator=NaiveTitForTatNegotiator)
@settings(deadline=100_000)
def test_specific_negotiator_buy_selling(negotiator):
    try_negotiator(negotiator, plot=False)
    # import matplotlib.pyplot as plt
    # plt.show(block=True)


if __name__ == "__main__":
    from rich import print

    print(ALL_BUILTIN_NEGOTIATORS)
