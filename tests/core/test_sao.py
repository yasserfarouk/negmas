from __future__ import annotations

import random
from random import choice

import hypothesis.strategies as st
from hypothesis import example, given, settings
from pytest import mark

import negmas
from negmas import (
    PolyAspiration,
    PresortingInverseUtilityFunction,
    all_negotiator_types,
)
from negmas.common import PreferencesChangeType
from negmas.gb.common import ResponseType
from negmas.gb.negotiators.timebased import AspirationNegotiator
from negmas.outcomes import Issue, make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun
from negmas.sao import EndImmediately, NoneOfferingPolicy, RejectAlways, SAOMechanism
from negmas.sao.common import SAOResponse, SAOState
from negmas.sao.negotiators.base import SAONegotiator
from negmas.sao.negotiators.modular.boa import make_boa

NEGTYPES = all_negotiator_types()


class SmartAspirationNegotiator(SAONegotiator):
    _inv = None  # The ufun invertor (finds outcomes in a utility range)
    _partner_first = None  # The best offer of the partner (assumed best for it)
    _min = None  # The minimum of my utility function
    _max = None  # The maximum of my utility function
    _best = None  # The best outcome for me

    def __init__(self, *args, **kwargs):
        # initialize the base SAONegoiator (MUST be done)
        super().__init__(*args, **kwargs)

        # Initialize the aspiration mixin to start at 1.0 and concede slowly
        self._asp = PolyAspiration(1.0, "boulware")

    def on_preferences_changed(self, changes):
        # create an initiaze an invertor for my ufun
        changes = [_ for _ in changes if _.type not in (PreferencesChangeType.Scale,)]
        if not changes:
            return
        self._inv = PresortingInverseUtilityFunction(self.ufun)  # type: ignore
        self._inv.init()

        # find worst and best outcomes for me
        worest, self._best = self.ufun.extreme_outcomes()  # type: ignore

        # and the correponding utility values
        self._min, self._max = self.ufun(worest), self.ufun(self._best)  # type: ignore

        # MUST call parent to avoid being called again for no reason
        super().on_preferences_changed(changes)

    def respond(self, state, source: str):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        # set the partner's first offer when I receive it
        if not self._partner_first:
            self._partner_first = offer

        # accept if the offer is not worse for me than what I would have offered
        return super().respond(state, source)

    def propose(self, state):
        # calculate my current aspiration level (utility level at which I will offer and accept)
        a = (self._max - self._min) * self._asp.utility_at(  # type: ignore
            state.relative_time
        ) + self._min

        # find some outcomes (all if the outcome space is  discrete) above the aspiration level
        outcomes = self._inv.some((a - 1e-6, self._max + 1e-6), False)  # type: ignore
        # If there are no outcomes above the aspiration level, offer my best outcome
        if not outcomes:
            return self._best

        # else if I did not  recieve anything from the partner, offer any outcome above the aspiration level
        if not self._partner_first:
            return choice(outcomes)

        # otherwise, offer the outcome most similar to the partner's first offer (above the aspiration level)
        nearest, ndist = None, float("inf")
        for o in outcomes:
            d = sum((a - b) * (a - b) for a, b in zip(o, self._partner_first))
            if d < ndist:
                nearest, ndist = o, d
        return nearest


def try_negotiator(cls, replace_buyer=True, replace_seller=True, n_steps=100):
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
    seller_utility = LUFun(
        values={  # type: ignore
            "price": IdentityFun(),
            "quantity": LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9),
        },
        weights={"price": 1.0, "quantity": 1.0, "delivery_time": 10.0},
        outcome_space=session.outcome_space,
        reserved_value=15.0,
    ).scale_max(1.0)
    buyer_utility = LUFun(
        values={  # type: ignore
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=10.0,
    ).scale_max(1.0)

    session.add(buyer_cls(name="buyer"), ufun=buyer_utility)  # type: ignore
    session.add(seller_cls(name="seller"), ufun=seller_utility)  # type: ignore
    session.run()
    return session


@given(
    opp=st.sampled_from(NEGTYPES),
    start=st.booleans(),
    rejector=st.sampled_from([EndImmediately, RejectAlways]),
)
@example(
    opp=negmas.sao.negotiators.timebased.AdditiveFirstFollowingTBNegotiator,
    start=True,
    rejector=negmas.sao.components.acceptance.EndImmediately,
)
@settings(deadline=500000)
def test_do_nothing_never_gets_agreements(opp, start, rejector):
    agent = make_boa(acceptance=rejector(), offering=NoneOfferingPolicy())
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(10, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    ufuns = [
        LinearAdditiveUtilityFunction.random(issues=issues),
        LinearAdditiveUtilityFunction.random(issues=issues),
    ]
    session = SAOMechanism(n_steps=1000, issues=issues)
    negs = [opp(), agent] if not start else [agent, opp()]
    for n, u in zip(negs, ufuns):
        session.add(n, preferences=u)

    assert session.run().agreement is None


@mark.parametrize(
    ["factory", "name", "short_name"], [(make_boa, "BOANegotiator", "BOA")]
)
def test_has_correct_type_name(factory, name, short_name):
    x = factory()
    assert x.type_name == name
    assert x.short_type_name == short_name


@mark.repeat(3)
def test_pend_works():
    os = make_os(
        [
            make_issue(10, "price"),
            make_issue(10, "quantity"),
            make_issue(["red", "green", "blue"], "color"),
        ]
    )
    for _ in range(50):
        ufuns = [
            LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
            LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
        ]
        n = 1000
        f = 0.01
        session = SAOMechanism(
            n_steps=None, time_limit=None, pend=f / n, outcome_space=os
        )
        for i, u in enumerate(ufuns):
            neg = AspirationNegotiator()
            assert session.add(neg, preferences=u)  # type: ignore
            assert len(session.negotiators) == (i + 1)

        assert abs(session.expected_relative_time - (f / (n + 1))) < 1e-8
        assert session.expected_remaining_time is None
        assert session.expected_remaining_steps is not None
        assert abs(session.expected_remaining_steps - n / f) < 4
        assert abs(session.relative_time - (f / (n + 1))) < 1e-8
        assert session.remaining_steps is None
        assert session.remaining_time is None
        assert session.state.step <= 10000 * n
        assert not session.state.started
        agreement = session.run().agreement
        assert session.state.started and session.state.ended
        if agreement is not None:
            break
    else:
        raise AssertionError(f"agreement failed in all runs")


def test_pend_per_second_works():
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(10, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    ufuns = [
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
    ]
    n = 10
    session = SAOMechanism(
        n_steps=None, time_limit=None, pend_per_second=1 / n, issues=issues
    )
    for u in ufuns:
        session.add(AspirationNegotiator(), preferences=u)  # type: ignore

    assert session.expected_relative_time < 1e-8
    assert (
        session.expected_remaining_time is not None
        and abs(session.expected_remaining_time - n) < 1e-8
    )
    assert session.expected_remaining_steps is None
    assert session.relative_time < 1e-8
    assert session.remaining_steps is None
    assert session.remaining_time is None
    session.run()
    assert session.state.time <= 100 * n


@mark.parametrize("s", [1, 3, 10, 101, 1000])
def test_nsteps_apply_as_round(s):
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(10, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    ufuns = [
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
    ]
    session = SAOMechanism(n_steps=s, issues=issues)
    for u in ufuns:
        assert session.add(AspirationNegotiator(), preferences=u)  # type: ignore

    assert session.expected_remaining_steps == s
    assert session.remaining_steps == s
    assert session.current_step == 0
    assert abs(session.relative_time - (1.0 / (s + 1))) < 1e-6
    assert session.remaining_time is None
    session.step()
    assert session.current_step == 1
    assert session.expected_remaining_steps == (s - 1)
    assert session.remaining_steps == s - 1
    assert abs(session.relative_time - (2.0 / (s + 1))) < 1e-6
    assert session.remaining_time is None
    session.run()
    ndone = session.current_step
    for nid in session.negotiator_ids:
        assert len(session.negotiator_offers(nid)) in (ndone, ndone - 1)
    assert session.state.step <= s


@mark.parametrize("s", [1, 3, 10, 101, 1000])
def test_nsteps_apply_as_step(s):
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(10, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    ufuns = [
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
    ]
    session = SAOMechanism(n_steps=s, issues=issues, one_offer_per_step=True)
    for u in ufuns:
        assert session.add(AspirationNegotiator(), preferences=u)  # type: ignore

    assert session.expected_remaining_steps == s
    assert session.remaining_steps == s
    assert session.current_step == 0
    assert abs(session.relative_time - (1.0 / (s + 1))) < 1e-6
    assert session.remaining_time is None
    session.step()
    assert session.current_step == 1
    assert session.expected_remaining_steps == (s - 1)
    assert session.remaining_steps == s - 1
    assert abs(session.relative_time - (2.0 / (s + 1))) < 1e-6
    assert session.remaining_time is None
    session.run()
    ndone = session.current_step
    for nid in session.negotiator_ids:
        assert len(session.negotiator_offers(nid)) in (
            int(ndone / 2),
            int(ndone / 2) + 1,
            int((ndone - 1) / 2),
            int((ndone - 1) / 2) + 1,
        )
    assert session.state.step <= s


def test_basic_sao():
    n_steps = 100
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(5, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    os = make_os(issues)
    ufuns = [
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
    ]
    session = SAOMechanism(n_steps=n_steps, outcome_space=os, one_offer_per_step=True)
    agents = [AspirationNegotiator() for _ in range(len(ufuns))]
    for u, a in zip(ufuns, agents):
        assert session.add(a, ufun=u)  # type: ignore
    # offers = [os.random_outcome() for _ in range(n_steps)]
    assert session.expected_remaining_steps == n_steps
    assert session.remaining_steps == n_steps
    assert session.current_step == 0
    assert abs(session.relative_time - (1.0 / (n_steps + 1))) < 1e-6
    assert session.remaining_time is None
    assert not session.state.started and not session.state.running
    for i in range(n_steps):
        if not session.step().running:
            break
        assert (
            session.state.started and session.state.running
        ), f"{session.state=}\n{session.extended_trace=}"
        assert (
            session.current_step == i + 1
        ), f"{session.state=}\n{session.extended_trace=}"
        assert session.expected_remaining_steps == (
            n_steps - i - 1
        ), f"{session.state=}\n{session.extended_trace=}"
        assert (
            session.remaining_steps == n_steps - i - 1
        ), f"{session.state=}\n{session.extended_trace=}"
        assert (
            abs(session.relative_time - ((i + 2) / (n_steps + 1))) < 1e-6
        ), f"{session.state=}\n{session.extended_trace=}"
        assert session.remaining_time is None
    assert session.state.started and not session.state.running
    assert session.state.step <= n_steps


def test_basic_sao_with_action():
    n_steps = 50
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(5, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    os = make_os(issues)
    ufuns = [
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
    ]
    session = SAOMechanism(n_steps=n_steps, outcome_space=os, one_offer_per_step=True)
    agents = [AspirationNegotiator() for _ in range(len(ufuns))]
    ids = [_.id for _ in agents]
    for u, a in zip(ufuns, agents):
        assert session.add(a, ufun=u)  # type: ignore
    offers = [os.random_outcome() for _ in range(n_steps)]
    assert session.expected_remaining_steps == n_steps
    assert session.remaining_steps == n_steps
    assert session.current_step == 0
    assert abs(session.relative_time - (1.0 / (n_steps + 1))) < 1e-6
    assert session.remaining_time is None
    assert not session.state.started and not session.state.running
    for i in range(n_steps):
        action = None
        pass_action = random.random() < 0.5
        if pass_action:
            ids = session.next_negotitor_ids()
            assert len(ids) == 1
            action = {ids[0]: SAOResponse(ResponseType.REJECT_OFFER, offers[i])}
        if not session.step(action).running:
            break
        if pass_action:
            state: SAOState = session.state  # type: ignore
            assert state.current_offer == offers[i]
            assert state.current_proposer == ids[0]
        assert (
            session.state.started and session.state.running
        ), f"{session.state=}\n{session.extended_trace=}"
        assert (
            session.current_step == i + 1
        ), f"{session.state=}\n{session.extended_trace=}"
        assert session.expected_remaining_steps == (
            n_steps - i - 1
        ), f"{session.state=}\n{session.extended_trace=}"
        assert (
            session.remaining_steps == n_steps - i - 1
        ), f"{session.state=}\n{session.extended_trace=}"
        assert (
            abs(session.relative_time - ((i + 2) / (n_steps + 1))) < 1e-6
        ), f"{session.state=}\n{session.extended_trace=}"
        assert session.remaining_time is None
    assert session.state.started and (
        not session.state.running or session.state.step >= n_steps
    ), f"Did not finish running:\n{session.extended_trace}"
    assert (
        session.state.step <= n_steps
    ), f"Ran for too long {session.state.step} but max expected is {n_steps} steps:\n{session.extended_trace}"


class MyNeg(AspirationNegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negstarted = False
        self.negended = False

    def on_negotiation_start(self, *args, **kwargs):
        assert not self.negstarted
        self.negstarted = True
        super().on_negotiation_start(*args, **kwargs)

    def respond(self, *args, **kwargs):
        return ResponseType.REJECT_OFFER

    def on_negotiation_end(self, *args, **kwargs):
        assert not self.negended
        self.negended = True
        super().on_negotiation_end(*args, **kwargs)


def test_hidden_time_works_and_no_call_repetitions():
    time, hidden = 18000, 30
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(5, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    os = make_os(issues)
    ufuns = [
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0),
    ]
    session = SAOMechanism(
        time_limit=time,
        n_steps=None,
        hidden_time_limit=hidden,
        outcome_space=os,
        one_offer_per_step=False,
        ignore_negotiator_exceptions=False,
    )
    agents = [MyNeg() for _ in range(len(ufuns))]
    for u, a in zip(ufuns, agents):
        assert session.add(a, ufun=u)  # type: ignore
    state = session.run()
    assert state.timedout
    assert 0.85 * hidden <= state.time <= hidden * 1.3


def test_smart_asipration():
    state = try_negotiator(SmartAspirationNegotiator)
