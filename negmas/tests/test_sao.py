import random
import time
from collections import defaultdict
from pathlib import Path
# from pprint import pprint
from time import sleep
from typing import Dict, Sequence

import hypothesis.strategies as st
from hypothesis import HealthCheck, example, given, settings
from pytest import mark

from negmas import Issue, ResponseType
from negmas.helpers import unique_name
from negmas.sao import (
    AspirationNegotiator,
    LimitedOutcomesNegotiator,
    RandomNegotiator,
    SAOMechanism,
    SAOResponse,
    SAOState,
    SAOSyncController,
)
from negmas.utilities import LinearUtilityFunction, MappingUtilityFunction

exception_str = "Custom Exception"


class MyRaisingNegotiator(AspirationNegotiator):
    def counter(self, state, offer):
        raise ValueError(exception_str)


class MySyncController(SAOSyncController):
    #     def respond(self, negotiator_id, state, offer):
    #         response = super().respond(negotiator_id, state, offer)
    #         if response != ResponseType.REJECT_OFFER and response != ResponseType.WAIT:
    #         return response
    #
    #     def propose(self, negotiator_id, state):
    #         offer = super().propose(negotiator_id, state)
    #         if offer is None:
    #         return offer
    #
    def counter_all(self, offers, states):
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
            self.negotiators[_][0].ami.random_outcomes(1)[0] for _ in states.keys()
        ]
        # print(
        #     f"{self.name} COUNTERED {offers} with {responses} @ { {k: v.step for k, v in states.items()} } : w {self.n_waits}, p {self.proposals}, o {self.offers}"
        # )
        return dict(
            zip(
                states.keys(),
                (SAOResponse(ResponseType.REJECT_OFFER, _) for _ in responses),
            )
        )

    def respond(self, negotiator_id, state, offer):
        response = super().respond(negotiator_id, state, offer)
        # print(
        #     f"{self.name} responded to {offer} from {negotiator_id}  @ {state.step} with {response}: w {self.n_waits}, p {self.proposals}, o {self.offers}"
        # )
        return response

    def propose(self, negotiator_id, state):
        proposal = super().propose(negotiator_id, state)
        # print(
        #     f"{self.name} proposed {proposal} through {negotiator_id} @ {state.step}: w {self.n_waits}, p {self.proposals}, o {self.offers}"
        # )
        return proposal

    def __init__(self, *args, sleep_seconds=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._sleep_seconds = sleep_seconds
        self.n_counter_all_calls = 0


class TimeWaster(RandomNegotiator):
    """A negotiator that wastes time"""

    def __init__(self, *args, sleep_seconds=0.2, n_waits=0, **kwargs):
        pa, pe, pr = 0.0, 0.0, 1.0
        kwargs["p_acceptance"] = kwargs.get("p_acceptance", pa)
        kwargs["p_ending"] = kwargs.get("p_ending", pe)
        kwargs["p_rejection"] = kwargs.get("p_rejection", pr)
        super().__init__(*args, **kwargs)
        self._sleep_seconds = sleep_seconds
        self.my_offers = defaultdict(lambda: None)
        self.received_offers = defaultdict(lambda: None)
        self.my_responses = defaultdict(lambda: None)
        self.n_waits = n_waits
        if n_waits:
            self.n_waits = random.randint(0, n_waits)
        self.waited = 0

    def counter(self, state, offer):
        if self.waited < self.n_waits and (
            state.step > 0 or not self.ami.params["avoid_ultimatum"]
        ):
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
        response = super().counter(state, offer)
        assert response.outcome is not None
        self.my_offers[state.step] = response.outcome
        self.my_responses[state.step] = response.response
        return response


@given(
    avoid_ultimatum=st.booleans(),
    n_steps=st.integers(10, 11),
    n_waits=st.integers(0, 4),
    n_waits2=st.integers(0, 4),
)
@settings(deadline=20000, max_examples=100)
@example(
    avoid_ultimatum=False,
    n_steps=10,
    n_waits=0,
    n_waits2=2,
)
def test_single_mechanism_history_with_waiting(
    avoid_ultimatum, n_steps, n_waits, n_waits2
):
    n_outcomes, waste = 5, (0.0, 0.3)
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=n_steps,
        ignore_negotiator_exceptions=False,
        avoid_ultimatum=avoid_ultimatum,
    )
    ufuns = MappingUtilityFunction.generate_random(2, outcomes=mechanism.outcomes)
    mechanism.add(
        TimeWaster(
            name=f"agent{0}", sleep_seconds=waste, ufun=ufuns[0], n_waits=n_waits
        )
    )
    mechanism.add(
        TimeWaster(
            name=f"agent{1}", sleep_seconds=waste, ufun=ufuns[1], n_waits=n_waits2
        )
    )
    mechanism.run()
    first = mechanism._selected_first
    n_negotiators = len(mechanism.negotiators)
    assert (not avoid_ultimatum and first == 0) or (
        avoid_ultimatum and 0 <= first < n_negotiators
    )
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
    ignored_offers = []
    for i, n in enumerate(mechanism.negotiators):
        # print(f"Received, offered, respones for {i} ({n})")
        # print(
        #     list(
        #         zip(
        #             zip(n.received_offers.keys(), n.received_offers.values()),
        #             zip(n.my_offers.keys(), n.my_offers.values()),
        #         )
        #     )
        # )
        # all agents asked to offer first if avoid_ultimatum
        if avoid_ultimatum:
            assert n.received_offers[0] is None
        else:
            first_offers.append(n.received_offers[0] is None)

        # sent and received match
        for j, w in n.my_offers.items():
            # cannot send mutlipe offers in the same step
            assert j not in s[i].keys()
            # cannot send None
            assert w is not None  # or (j == 0 and not avoid_ultimatum)
            if j == 0 and i != first and avoid_ultimatum:
                # cannot have more than n_negotiators - 1 ignored offers
                assert len(ignored_offers) < n_negotiators - 1
                ignored_offers.append(w[0])
                continue
            s[i][j] = w[0]
        for j, w in n.received_offers.items():
            # cannot receive multiple ofers in the same step
            assert j not in r[i].keys()
            # canont be asked to start offering except in the first step
            assert w is not None or j == 0
            # this is the first agent to offer, ignore its first step
            if (first == i and j == 0) or (avoid_ultimatum and j == 0):
                # if this is the first agent, its first thing recieved must be None
                assert w is None
                continue
            assert (
                w is not None
            ), f"None outcome agent {i} @ {j} (avoid={avoid_ultimatum}) first is {first}"
            r[i][j] = w[0]

    # a single agent is asked to offer first if not avoid_ultimatum
    if not avoid_ultimatum:
        assert any(first_offers) and not all(first_offers)
    # every agent is asked to offer and all except one are ignored if avoid_ultimatum
    if avoid_ultimatum:
        assert len(ignored_offers) == n_negotiators - 1

    # reconstruct history
    neg_map = dict(zip((_.id for _ in mechanism.negotiators), [0, 1]))
    for state in mechanism.history:
        for _, w in state.new_offers:
            a = neg_map[_]
            # cannot see the same step twice in the history of an agent
            assert state.step not in h[a].keys()
            h[a][state.step] = w

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
            assert j in s[1 - i] or (j == 0 and not avoid_ultimatum)

    # s and r will not have matched indices but should have matched values
    s = [list(_.values()) for _ in s]
    r = [list(_.values()) for _ in r]
    h = [list(_[0] for _ in _.values()) for _ in h]
    # history matches what is stored inside agents
    for i, n in enumerate(mechanism.negotiators):
        for j, w in enumerate(s[i]):
            if j < len(r[1 - i]):
                assert r[1 - i][j] == w
            assert h[i][j] == w

        for j, w in enumerate(r[i]):
            assert s[1 - i][j] == w


def test_neg_run_no_waiting():
    n_outcomes, n_steps, waste = 10, 10, 0.5
    mechanism = SAOMechanism(
        outcomes=n_outcomes, n_steps=n_steps, ignore_negotiator_exceptions=True
    )
    ufuns = MappingUtilityFunction.generate_random(2, outcomes=mechanism.outcomes)
    mechanism.add(TimeWaster(name=f"agent{0}", sleep_seconds=waste, ufun=ufuns[0]))
    mechanism.add(TimeWaster(name=f"agent{1}", sleep_seconds=waste, ufun=ufuns[1]))
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
            avoid_ultimatum=False,
            name=f"{m}",
        )
        ufuns = MappingUtilityFunction.generate_random(2, outcomes=mechanism.outcomes)
        mechanism.add(c1.create_negotiator(ufun=ufuns[0], id=f"0-{m}", name=f"0-{m}"))
        mechanism.add(c2.create_negotiator(ufun=ufuns[1], id=f"1-{m}", name=f"1-{m}"))
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
            avoid_ultimatum=False,
        )
        ufuns = MappingUtilityFunction.generate_random(2, outcomes=mechanism.outcomes)
        edge_names.append(f"f{0}")
        mechanism.add(
            TimeWaster(
                name=f"agent{0}", id=f"a{0}", sleep_seconds=waste_edge, ufun=ufuns[0]
            )
        )
        mechanism.add(c.create_negotiator(ufun=ufuns[1]))
        mechanisms.append(mechanism)
    SAOMechanism.runall(mechanisms)

    for mechanism in mechanisms:
        assert mechanism.state.started
        assert mechanism.state.agreement is None
        assert not mechanism.state.has_error
        assert not mechanism.state.broken
        assert mechanism.state.timedout, print(f"Did not timeout!!\n{mechanism.state}")
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
        n_negotiators, outcomes=mechanism.outcomes
    )
    mechanism.add(AspirationNegotiator(name=f"agent{0}"), ufun=ufuns[0])
    mechanism.add(MyRaisingNegotiator(name=f"agent{1}"), ufun=ufuns[1])
    assert mechanism.state.step == 0
    mechanism.step()
    mechanism.step()
    mechanism.step()
    assert mechanism.state.step == 1
    assert mechanism._current_offer is not None
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


@given(
    n_outcomes=st.integers(5, 10),
    n_negotiators=st.integers(2, 4),
    n_steps=st.integers(1, 4),
)
def test_times_are_calculated(n_outcomes, n_negotiators, n_steps):
    mechanism = SAOMechanism(outcomes=n_outcomes, n_steps=8)
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])
    assert mechanism.state.step == 0
    _strt = time.perf_counter()
    for _ in range(n_steps):
        # print(f"Stepping: {_}")
        mechanism.step()
    time.sleep(0.01)
    duration = time.perf_counter() - _strt
    # assert mechanism.current_step == n_steps
    assert mechanism._current_offer is not None
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
    for _ in range(5):
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
    import dill as pickle

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
    negs = [
        SAOMechanism(issues=[Issue((0.0, 1.0), "price")], n_steps=20)
        for _ in range(n_negs)
    ]
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

    c = SAOSingleAgreementRandomController(strict=strict)
    negs = [
        SAOMechanism(
            issues=[Issue((0.0, 1.0), "price")], n_steps=50, outcome_type=tuple
        )
        for _ in range(n_negs)
    ]
    for i, neg in enumerate(negs):
        neg.add(
            AspirationNegotiator(aspiration_type="linear", name=f"opponent-{i}"),
            ufun=LinearUtilityFunction(weights=[1.0]),
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

    a, b, c = (
        SAOSingleAgreementAspirationController(
            ufun=MappingUtilityFunction(lambda x: x["price"]), strict=False
        ),
        SAOSingleAgreementAspirationController(
            ufun=MappingUtilityFunction(lambda x: x["price"]), strict=False
        ),
        SAOSingleAgreementAspirationController(
            ufun=MappingUtilityFunction(lambda x: x["price"]), strict=False
        ),
    )

    n1 = SAOMechanism(
        name="ab",
        issues=[Issue((0.0, 1.0), "price")],
        n_steps=50,
        outcome_type=dict,
    )
    n2 = SAOMechanism(
        name="ac",
        issues=[Issue((0.0, 1.0), "price")],
        n_steps=50,
        outcome_type=dict,
    )
    n3 = SAOMechanism(
        name="bc",
        issues=[Issue((0.0, 1.0), "price")],
        n_steps=50,
        outcome_type=dict,
    )

    n1.add(a.create_negotiator(name="a>b"))
    n1.add(b.create_negotiator(name="b>a"))
    n2.add(a.create_negotiator(name="a>c"))
    n2.add(c.create_negotiator(name="c>a"))
    n3.add(b.create_negotiator(name="b>c"))
    n3.add(c.create_negotiator(name="c>b"))
    negs = [n1, n2, n3]
    Mechanism.runall(negs, keep_order)

    agreements = [neg.state.agreement for neg in negs]
    assert not keep_order or sum(_ is not None for _ in agreements) > 0
    # TODO check why sometimes we get no agreements when order is not kept


def test_can_create_all_negotiator_types():
    from negmas.helpers import instantiate

    issues = [Issue((0.0, 1.0), name="price"), Issue(10, name="quantity")]
    for outcome_type in [tuple, dict]:
        outcomes = Issue.enumerate(issues, max_n_outcomes=100, astype=outcome_type)
        neg_types = [
            (
                "RandomNegotiator",
                dict(ufun=LinearUtilityFunction(weights=dict(price=1.0, quantity=1.0))),
            ),
            ("LimitedOutcomesNegotiator", dict()),
            ("LimitedOutcomesAcceptor", dict()),
            (
                "AspirationNegotiator",
                dict(ufun=LinearUtilityFunction(weights=dict(price=1.0, quantity=1.0))),
            ),
            (
                "ToughNegotiator",
                dict(ufun=LinearUtilityFunction(weights=dict(price=1.0, quantity=1.0))),
            ),
            (
                "OnlyBestNegotiator",
                dict(ufun=LinearUtilityFunction(weights=dict(price=1.0, quantity=1.0))),
            ),
            (
                "NaiveTitForTatNegotiator",
                dict(ufun=LinearUtilityFunction(weights=dict(price=1.0, quantity=1.0))),
            ),
            (
                "SimpleTitForTatNegotiator",
                dict(ufun=LinearUtilityFunction(weights=dict(price=1.0, quantity=1.0))),
            ),
            (
                "NiceNegotiator",
                dict(ufun=LinearUtilityFunction(weights=dict(price=1.0, quantity=1.0))),
            ),
        ]
        for neg_type, params in neg_types:
            _ = instantiate("negmas.sao." + neg_type, **params)


@mark.parametrize("asdict", [True, False])
def test_can_run_all_negotiators(asdict):
    from negmas.helpers import instantiate

    issues = [Issue((0.0, 1.0), name="price"), Issue(10, name="quantity")]
    weights = dict(price=1.0, quantity=1.0) if asdict else (1.0, 1.0)
    for outcome_type in [tuple, dict]:
        outcomes = Issue.enumerate(issues, max_n_outcomes=100, astype=outcome_type)
        neg_types = [
            (
                "RandomNegotiator",
                dict(ufun=LinearUtilityFunction(weights=weights)),
            ),
            (
                "AspirationNegotiator",
                dict(ufun=LinearUtilityFunction(weights=weights)),
            ),
            (
                "LimitedOutcomesNegotiator",
                dict(acceptance_probabilities=0.5),
            ),
            (
                "LimitedOutcomesAcceptor",
                dict(acceptance_probabilities=0.5),
            ),
            (
                "ToughNegotiator",
                dict(ufun=LinearUtilityFunction(weights=weights)),
            ),
            (
                "OnlyBestNegotiator",
                dict(ufun=LinearUtilityFunction(weights=weights)),
            ),
            (
                "NaiveTitForTatNegotiator",
                dict(ufun=LinearUtilityFunction(weights=weights)),
            ),
            (
                "SimpleTitForTatNegotiator",
                dict(ufun=LinearUtilityFunction(weights=weights)),
            ),
            (
                "NiceNegotiator",
                dict(ufun=LinearUtilityFunction(weights=weights)),
            ),
        ]
        for i, (neg_type, params) in enumerate(neg_types):
            for n2, p2 in neg_types:
                # print(f"{neg_type} <> {n2}")
                n1 = instantiate("negmas.sao." + neg_type, **params)
                n2 = instantiate("negmas.sao." + n2, **p2)
                m = SAOMechanism(
                    n_steps=30, issues=issues, outcome_type=dict if asdict else tuple
                )
                m.add(n1)
                m.add(n2)
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
    assert state.agreement == (3,)
