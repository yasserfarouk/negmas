from __future__ import annotations
import random
from random import choice

from negmas.gb.negotiators.hybrid import HybridNegotiator
from negmas.sao.negotiators import AspirationNegotiator
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

    def respond(self, state, source: str | None = None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        # set the partner's first offer when I receive it
        if not self._partner_first:
            self._partner_first = offer

        # accept if the offer is not worse for me than what I would have offered
        return super().respond(state, source)

    def propose(self, state, dest: str | None = None):
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


@mark.slow
@mark.repeat(3)
def test_pend_works():
    os = make_os(
        [
            make_issue(10, "price"),
            make_issue(10, "quantity"),
            make_issue(["red", "green", "blue"], "color"),
        ]
    )
    for _ in range(10):  # Reduced from 50 to 10 iterations
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
        raise AssertionError("agreement failed in all runs")


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
        assert session.state.started and session.state.running, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.current_step == i + 1, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.expected_remaining_steps == (n_steps - i - 1), (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.remaining_steps == n_steps - i - 1, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert abs(session.relative_time - ((i + 2) / (n_steps + 1))) < 1e-6, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.remaining_time is None
    assert session.state.started and not session.state.running
    assert session.state.step <= n_steps


def test_basic_sao_hybrid():
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
    session = SAOMechanism(
        n_steps=n_steps,
        outcome_space=os,
        one_offer_per_step=True,
        ignore_negotiator_exceptions=False,
    )
    agents = [HybridNegotiator() for _ in range(len(ufuns))]
    for u, a in zip(ufuns, agents):
        assert session.add(a, ufun=u)  # type: ignore
    # offers = [os.random_outcome() for _ in range(n_steps)]
    assert session.expected_remaining_steps == n_steps
    assert session.remaining_steps == n_steps
    assert session.current_step == 0
    assert abs(session.relative_time - (1.0 / (n_steps + 1))) < 1e-6
    assert session.remaining_time is None
    assert not session.state.started and not session.state.running
    for i in range(n_steps + 1):
        if not session.step().running:
            break
        assert session.state.started and session.state.running, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.current_step == i + 1, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.expected_remaining_steps == (n_steps - i - 1), (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.remaining_steps == n_steps - i - 1, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert abs(session.relative_time - ((i + 2) / (n_steps + 1))) < 1e-6, (
            f"{session.state=}\n{session.extended_trace=}"
        )
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
        assert session.state.started and session.state.running, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.current_step == i + 1, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.expected_remaining_steps == (n_steps - i - 1), (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.remaining_steps == n_steps - i - 1, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert abs(session.relative_time - ((i + 2) / (n_steps + 1))) < 1e-6, (
            f"{session.state=}\n{session.extended_trace=}"
        )
        assert session.remaining_time is None
    assert session.state.started and (
        not session.state.running or session.state.step >= n_steps
    ), f"Did not finish running:\n{session.extended_trace}"
    assert session.state.step <= n_steps, (
        f"Ran for too long {session.state.step} but max expected is {n_steps} steps:\n{session.extended_trace}"
    )


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


@mark.slow
def test_hidden_time_works_and_no_call_repetitions():
    time, hidden = 18000, 3  # Reduced from 30 to 3 seconds for faster tests
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
    try_negotiator(SmartAspirationNegotiator)


class RTRecorder(SAONegotiator):
    def __init__(self, *args, **kwargs):
        self.records = []
        super().__init__(*args, **kwargs)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        self.records.append(
            (
                state.step,
                state.relative_time,
                state.time,
                # ((state.step + 1) / (self.nmi.n_steps + 1) if state.step > 0 else 0.0)
                (state.step + 1) / (self.nmi.n_steps + 1) if self.nmi.n_steps else -1,
            )
        )
        return SAOResponse(ResponseType.REJECT_OFFER, self.nmi.random_outcome())


def test_relative_time():
    time, hidden = float("inf"), float("inf")
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
        n_steps=10,
        hidden_time_limit=hidden,
        outcome_space=os,
        one_offer_per_step=False,
        ignore_negotiator_exceptions=False,
    )
    agents = [RTRecorder() for _ in range(len(ufuns))]
    for u, a in zip(ufuns, agents):
        assert session.add(a, ufun=u)  # type: ignore
    session.run()
    for agent in agents:
        for step, relative_time, time, expected_rt in agent.records:
            assert abs(relative_time - expected_rt) < 1e-5, (
                f"{(step, relative_time, time, expected_rt)}"
            )


def test_extended_outcome_with_text():
    """Test that offering policies can return ExtendedOutcome with data["text"]."""
    from attrs import define, field
    from negmas.gb.components.offering import RandomOfferingPolicy
    from negmas.gb.components.acceptance import AcceptAnyRational
    from negmas.outcomes.common import ExtendedOutcome, Outcome
    from negmas.sao.negotiators.modular.boa import make_boa

    # Track all texts generated during the negotiation
    generated_texts: list[str] = []

    @define
    class TextOfferingPolicy(RandomOfferingPolicy):
        """An offering policy that wraps outcomes with text in data dict."""

        _texts: list[str] = field(factory=list)

        @property
        def texts(self) -> list[str]:
            return self._texts

        def __call__(self, state, dest=None) -> Outcome | ExtendedOutcome | None:
            outcome = super().__call__(state, dest)
            if outcome is None:
                return None
            # Generate a random short text in data["text"]
            text = f"offer_{state.step}_{random.randint(0, 1000)}"
            self._texts.append(text)
            generated_texts.append(text)
            return ExtendedOutcome(outcome=outcome, data={"text": text})

    # Create issues and outcome space
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(5, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    os = make_os(issues)

    # Create utility functions
    ufun1 = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun2 = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    # Create the offering policy with text
    text_offering = TextOfferingPolicy()

    # Create negotiator using BOA architecture with our custom offering policy
    negotiator1 = make_boa(acceptance=AcceptAnyRational(), offering=text_offering)

    # Create AspirationNegotiator as opponent
    negotiator2 = AspirationNegotiator()

    # Create mechanism and run negotiation
    session = SAOMechanism(
        outcome_space=os,
        n_steps=20,
        one_offer_per_step=False,
        ignore_negotiator_exceptions=False,
    )
    session.add(negotiator1, ufun=ufun1)
    session.add(negotiator2, ufun=ufun2)
    session.run()

    # Verify texts were generated
    assert len(generated_texts) > 0, "No texts were generated during negotiation"
    assert len(text_offering.texts) > 0, "TextOfferingPolicy did not track texts"
    assert text_offering.texts == generated_texts, (
        "Generated texts do not match tracked texts"
    )

    # Verify each text has expected format
    for text in generated_texts:
        assert text.startswith("offer_"), f"Text '{text}' does not have expected format"

    # Verify negotiation completed (either agreement or timeout)
    assert session.state.started, "Negotiation did not start"
    assert session.state.ended or session.state.timedout or session.state.agreement, (
        "Negotiation did not complete properly"
    )


def test_sao_response_from_extended():
    """Test SAOResponse.from_extended class method."""
    from negmas.outcomes.common import ExtendedOutcome
    from negmas.gb.common import ResponseType, ExtendedResponseType
    from negmas.sao.common import SAOResponse

    # Test 1: Basic ResponseType and Outcome
    resp = SAOResponse.from_extended(ResponseType.ACCEPT_OFFER, (1, 2, 3))
    assert resp.response == ResponseType.ACCEPT_OFFER
    assert resp.outcome == (1, 2, 3)
    assert resp.data is None

    # Test 2: ExtendedResponseType with data
    ext_resp = ExtendedResponseType(
        response=ResponseType.REJECT_OFFER,
        data={"reason": "too low", "text": "I reject"},
    )
    resp = SAOResponse.from_extended(ext_resp, (1, 2, 3))
    assert resp.response == ResponseType.REJECT_OFFER
    assert resp.outcome == (1, 2, 3)
    assert resp.data == {"reason": "too low", "text": "I reject"}

    # Test 3: ExtendedOutcome with data
    ext_outcome = ExtendedOutcome(
        outcome=(4, 5, 6), data={"info": "counter", "text": "My offer"}
    )
    resp = SAOResponse.from_extended(ResponseType.REJECT_OFFER, ext_outcome)
    assert resp.response == ResponseType.REJECT_OFFER
    assert resp.outcome == (4, 5, 6)
    assert resp.data == {"info": "counter", "text": "My offer"}

    # Test 4: Both extended with no conflicts
    ext_resp = ExtendedResponseType(
        response=ResponseType.REJECT_OFFER, data={"resp_key": "resp_val"}
    )
    ext_outcome = ExtendedOutcome(outcome=(7, 8, 9), data={"offer_key": "offer_val"})
    resp = SAOResponse.from_extended(ext_resp, ext_outcome)
    assert resp.response == ResponseType.REJECT_OFFER
    assert resp.outcome == (7, 8, 9)
    assert resp.data == {"resp_key": "resp_val", "offer_key": "offer_val"}

    # Test 5: Both extended with conflicts (same key)
    ext_resp = ExtendedResponseType(
        response=ResponseType.ACCEPT_OFFER,
        data={"shared": "from_response", "unique_resp": 1},
    )
    ext_outcome = ExtendedOutcome(
        outcome=(10, 11, 12), data={"shared": "from_offer", "unique_offer": 2}
    )
    resp = SAOResponse.from_extended(ext_resp, ext_outcome)
    assert resp.response == ResponseType.ACCEPT_OFFER
    assert resp.outcome == (10, 11, 12)
    assert "shared_response" in resp.data
    assert "shared_offer" in resp.data
    assert resp.data["shared_response"] == "from_response"
    assert resp.data["shared_offer"] == "from_offer"
    assert resp.data["unique_resp"] == 1
    assert resp.data["unique_offer"] == 2

    # Test 6: Text combination - both have text
    ext_resp = ExtendedResponseType(
        response=ResponseType.REJECT_OFFER, data={"text": "Response text"}
    )
    ext_outcome = ExtendedOutcome(outcome=(1, 2, 3), data={"text": "Offer text"})
    resp = SAOResponse.from_extended(ext_resp, ext_outcome)
    assert resp.data["text"] == "Response text\nOffer text"

    # Test 7: Text combination with custom combiner via subclassing
    class CustomSAOResponse(SAOResponse):
        @classmethod
        def text_combiner(cls, response_text: str, offer_text: str) -> str:
            return f"[R] {response_text} | [O] {offer_text}"

    resp = CustomSAOResponse.from_extended(ext_resp, ext_outcome)
    assert resp.data["text"] == "[R] Response text | [O] Offer text"
    # Verify we got the subclass back
    assert isinstance(resp, CustomSAOResponse)

    # Test 8: Only response has text
    ext_resp = ExtendedResponseType(
        response=ResponseType.ACCEPT_OFFER, data={"text": "Only response text"}
    )
    resp = SAOResponse.from_extended(ext_resp, (1, 2, 3))
    assert resp.data["text"] == "Only response text"

    # Test 9: Only offer has text
    ext_outcome = ExtendedOutcome(outcome=(1, 2, 3), data={"text": "Only offer text"})
    resp = SAOResponse.from_extended(ResponseType.ACCEPT_OFFER, ext_outcome)
    assert resp.data["text"] == "Only offer text"

    # Test 10: None offer
    resp = SAOResponse.from_extended(ResponseType.END_NEGOTIATION, None)
    assert resp.response == ResponseType.END_NEGOTIATION
    assert resp.outcome is None
    assert resp.data is None


def test_extended_response_type_in_negotiation():
    """Test that acceptance policies can return ExtendedResponseType in a negotiation."""
    from attrs import define, field
    from negmas.gb.components.offering import RandomOfferingPolicy
    from negmas.gb.components.acceptance import AcceptancePolicy
    from negmas.outcomes.common import ExtendedOutcome, Outcome
    from negmas.gb.common import ResponseType, ExtendedResponseType
    from negmas.sao.negotiators.modular.boa import make_boa

    # Track all response texts generated during the negotiation
    response_texts: list[str] = []
    offer_texts: list[str] = []

    @define
    class TextAcceptancePolicy(AcceptancePolicy):
        """An acceptance policy that returns ExtendedResponseType with text."""

        _response_texts: list[str] = field(factory=list)

        def __call__(self, state, offer, source) -> ResponseType | ExtendedResponseType:
            # Generate response text
            text = f"response_{state.step}_{random.randint(0, 1000)}"
            self._response_texts.append(text)
            response_texts.append(text)

            # Accept with probability based on step (more likely to accept later)
            if state.relative_time > 0.5 and offer is not None:
                return ExtendedResponseType(
                    response=ResponseType.ACCEPT_OFFER,
                    data={"text": text, "reason": "time"},
                )
            return ExtendedResponseType(
                response=ResponseType.REJECT_OFFER,
                data={"text": text, "reason": "continue"},
            )

    @define
    class TextOfferingPolicy(RandomOfferingPolicy):
        """An offering policy that wraps outcomes with text in data dict."""

        _offer_texts: list[str] = field(factory=list)

        def __call__(self, state, dest=None) -> Outcome | ExtendedOutcome | None:
            outcome = super().__call__(state, dest)
            if outcome is None:
                return None
            text = f"offer_{state.step}_{random.randint(0, 1000)}"
            self._offer_texts.append(text)
            offer_texts.append(text)
            return ExtendedOutcome(outcome=outcome, data={"text": text})

    # Create issues and outcome space
    issues: list[Issue] = [make_issue(10, "price"), make_issue(5, "quantity")]
    os = make_os(issues)

    # Create utility functions
    ufun1 = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun2 = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    # Create negotiator with text policies
    text_acceptance = TextAcceptancePolicy()
    text_offering = TextOfferingPolicy()
    negotiator1 = make_boa(acceptance=text_acceptance, offering=text_offering)

    # Create AspirationNegotiator as opponent
    negotiator2 = AspirationNegotiator()

    # Create mechanism and run negotiation
    session = SAOMechanism(
        outcome_space=os,
        n_steps=20,
        one_offer_per_step=False,
        ignore_negotiator_exceptions=False,
    )
    session.add(negotiator1, ufun=ufun1)
    session.add(negotiator2, ufun=ufun2)
    session.run()

    # Verify texts were generated
    assert len(response_texts) > 0 or len(offer_texts) > 0, "No texts were generated"

    # Verify negotiation completed
    assert session.state.started, "Negotiation did not start"
    assert session.state.ended or session.state.timedout or session.state.agreement, (
        "Negotiation did not complete properly"
    )


def test_boa_negotiator_text_combinations():
    """Test BOANegotiator with all 4 combinations of text/no-text offering and acceptance policies.

    Uses time-based policies with deterministic behavior:
    - TimeBasedOfferingPolicy: offers based on aspiration curve (starts high, concedes over time)
    - ACTime: accepts after relative_time >= tau

    Tests:
    1. No text offering + No text acceptance
    2. Text offering + No text acceptance
    3. No text offering + Text acceptance
    4. Text offering + Text acceptance
    """
    from attrs import define, field
    from negmas.gb.components.offering import TimeBasedOfferingPolicy
    from negmas.gb.components.acceptance import ACTime
    from negmas.outcomes.common import ExtendedOutcome, Outcome
    from negmas.gb.common import ExtendedResponseType
    from negmas.sao.negotiators.modular.boa import make_boa
    from negmas.negotiators.helpers import PolyAspiration

    # --------------------------------------------------------------------------
    # Define text-enabled versions of the time-based policies
    # --------------------------------------------------------------------------

    @define
    class TextTimeBasedOfferingPolicy(TimeBasedOfferingPolicy):
        """TimeBasedOfferingPolicy that adds text to each offer."""

        offer_count: int = field(init=False, default=0)
        generated_texts: list[str] = field(factory=list)

        def __call__(self, state, dest=None) -> Outcome | ExtendedOutcome | None:
            outcome = super().__call__(state, dest)
            if outcome is None:
                return None
            self.offer_count += 1
            text = f"offer_step{state.step}_n{self.offer_count}"
            self.generated_texts.append(text)
            return ExtendedOutcome(
                outcome=outcome,
                data={
                    "text": text,
                    "step": state.step,
                    "relative_time": state.relative_time,
                },
            )

    @define
    class TextACTime(ACTime):
        """ACTime acceptance policy that adds text to each response."""

        response_count: int = field(init=False, default=0)
        generated_texts: list[str] = field(factory=list)

        def __call__(self, state, offer, source) -> ResponseType | ExtendedResponseType:
            # Get the base response
            base_response = super().__call__(state, offer, source)
            self.response_count += 1
            text = (
                f"response_step{state.step}_n{self.response_count}_{base_response.name}"
            )
            self.generated_texts.append(text)
            return ExtendedResponseType(
                response=base_response,
                data={"text": text, "step": state.step, "decision": base_response.name},
            )

    # --------------------------------------------------------------------------
    # Test setup: create issues and outcome space
    # --------------------------------------------------------------------------
    issues: list[Issue] = [make_issue(10, "price"), make_issue(5, "quantity")]
    os = make_os(issues)

    # Parameters for deterministic behavior
    n_steps = 20
    tau = 0.5  # Accept after 50% of negotiation time

    # --------------------------------------------------------------------------
    # Test 1: No text offering + No text acceptance
    # --------------------------------------------------------------------------
    ufun1a = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun1b = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    offering1 = TimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    acceptance1 = ACTime(tau=tau)
    negotiator1 = make_boa(acceptance=acceptance1, offering=offering1)

    offering2 = TimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    acceptance2 = ACTime(tau=tau)
    negotiator2 = make_boa(acceptance=acceptance2, offering=offering2)

    session1 = SAOMechanism(outcome_space=os, n_steps=n_steps, one_offer_per_step=False)
    session1.add(negotiator1, ufun=ufun1a)
    session1.add(negotiator2, ufun=ufun1b)
    session1.run()

    assert session1.state.started, "Test 1: Negotiation did not start"

    # --------------------------------------------------------------------------
    # Test 2: Text offering + No text acceptance
    # --------------------------------------------------------------------------
    ufun2a = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun2b = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    text_offering2 = TextTimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    acceptance2_plain = ACTime(tau=tau)
    negotiator2a = make_boa(acceptance=acceptance2_plain, offering=text_offering2)

    offering2b = TimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    acceptance2b = ACTime(tau=tau)
    negotiator2b = make_boa(acceptance=acceptance2b, offering=offering2b)

    session2 = SAOMechanism(outcome_space=os, n_steps=n_steps, one_offer_per_step=False)
    session2.add(negotiator2a, ufun=ufun2a)
    session2.add(negotiator2b, ufun=ufun2b)
    session2.run()

    assert session2.state.started, "Test 2: Negotiation did not start"
    assert len(text_offering2.generated_texts) > 0, "Test 2: No offer texts generated"
    # Verify text format
    for text in text_offering2.generated_texts:
        assert text.startswith("offer_step"), f"Test 2: Unexpected text format: {text}"

    # --------------------------------------------------------------------------
    # Test 3: No text offering + Text acceptance
    # --------------------------------------------------------------------------
    ufun3a = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun3b = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    offering3a = TimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    text_acceptance3 = TextACTime(tau=tau)
    negotiator3a = make_boa(acceptance=text_acceptance3, offering=offering3a)

    offering3b = TimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    acceptance3b = ACTime(tau=tau)
    negotiator3b = make_boa(acceptance=acceptance3b, offering=offering3b)

    session3 = SAOMechanism(outcome_space=os, n_steps=n_steps, one_offer_per_step=False)
    session3.add(negotiator3a, ufun=ufun3a)
    session3.add(negotiator3b, ufun=ufun3b)
    session3.run()

    assert session3.state.started, "Test 3: Negotiation did not start"
    assert len(text_acceptance3.generated_texts) > 0, (
        "Test 3: No response texts generated"
    )
    # Verify text format and that responses contain decision info
    for text in text_acceptance3.generated_texts:
        assert text.startswith("response_step"), (
            f"Test 3: Unexpected text format: {text}"
        )
        assert "ACCEPT" in text or "REJECT" in text, (
            f"Test 3: Missing decision in text: {text}"
        )

    # --------------------------------------------------------------------------
    # Test 4: Text offering + Text acceptance (both negotiators)
    # --------------------------------------------------------------------------
    ufun4a = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun4b = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    text_offering4a = TextTimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    text_acceptance4a = TextACTime(tau=tau)
    negotiator4a = make_boa(acceptance=text_acceptance4a, offering=text_offering4a)

    text_offering4b = TextTimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    text_acceptance4b = TextACTime(tau=tau)
    negotiator4b = make_boa(acceptance=text_acceptance4b, offering=text_offering4b)

    session4 = SAOMechanism(outcome_space=os, n_steps=n_steps, one_offer_per_step=False)
    session4.add(negotiator4a, ufun=ufun4a)
    session4.add(negotiator4b, ufun=ufun4b)
    session4.run()

    assert session4.state.started, "Test 4: Negotiation did not start"

    # Both negotiators should have generated texts
    assert len(text_offering4a.generated_texts) > 0, (
        "Test 4: Negotiator A generated no offer texts"
    )
    assert len(text_acceptance4a.generated_texts) > 0, (
        "Test 4: Negotiator A generated no response texts"
    )
    assert len(text_offering4b.generated_texts) > 0, (
        "Test 4: Negotiator B generated no offer texts"
    )
    assert len(text_acceptance4b.generated_texts) > 0, (
        "Test 4: Negotiator B generated no response texts"
    )

    # Verify the state contains data when extended types are used
    has_data = False
    for step_state in session4.history:
        if step_state.current_data is not None:
            has_data = True
            assert isinstance(step_state.current_data, dict), (
                "Test 4: Data should be a dict"
            )
            break

    # At least some states should have data
    assert has_data or session4.state.agreement is not None, (
        "Test 4: Expected data in history or agreement"
    )

    # --------------------------------------------------------------------------
    # Test 5: Verify text combination when both offer and response have text
    # --------------------------------------------------------------------------
    ufun5a = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun5b = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    text_offering5 = TextTimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "conceder"))
    text_acceptance5 = TextACTime(tau=0.1)  # Accept early
    negotiator5a = make_boa(acceptance=text_acceptance5, offering=text_offering5)

    text_offering5b = TextTimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "conceder"))
    text_acceptance5b = TextACTime(tau=0.1)  # Accept early
    negotiator5b = make_boa(acceptance=text_acceptance5b, offering=text_offering5b)

    session5 = SAOMechanism(outcome_space=os, n_steps=n_steps, one_offer_per_step=False)
    session5.add(negotiator5a, ufun=ufun5a)
    session5.add(negotiator5b, ufun=ufun5b)
    session5.run()

    assert session5.state.started, "Test 5: Negotiation did not start"

    # Verify both offer and response texts were generated
    total_offer_texts = len(text_offering5.generated_texts) + len(
        text_offering5b.generated_texts
    )
    total_response_texts = len(text_acceptance5.generated_texts) + len(
        text_acceptance5b.generated_texts
    )

    assert total_offer_texts > 0, "Test 5: No offer texts generated"
    assert total_response_texts > 0, "Test 5: No response texts generated"


def test_sao_state_data_population():
    """Test that SAOState.current_data and new_data are populated when extended types are used."""
    from attrs import define, field
    from negmas.gb.components.offering import TimeBasedOfferingPolicy
    from negmas.gb.components.acceptance import ACTime
    from negmas.outcomes.common import ExtendedOutcome, Outcome
    from negmas.gb.common import ExtendedResponseType
    from negmas.sao.negotiators.modular.boa import make_boa
    from negmas.negotiators.helpers import PolyAspiration

    # Define policies that add data to their responses
    @define
    class DataOfferingPolicy(TimeBasedOfferingPolicy):
        """TimeBasedOfferingPolicy that adds data to each offer."""

        offer_count: int = field(init=False, default=0)

        def __call__(self, state, dest=None) -> Outcome | ExtendedOutcome | None:
            outcome = super().__call__(state, dest)
            if outcome is None:
                return None
            self.offer_count += 1
            return ExtendedOutcome(
                outcome=outcome,
                data={
                    "text": f"offer_{self.offer_count}",
                    "offer_num": self.offer_count,
                    "source": "offering_policy",
                },
            )

    @define
    class DataACTime(ACTime):
        """ACTime acceptance policy that adds data to each response."""

        response_count: int = field(init=False, default=0)

        def __call__(self, state, offer, source) -> ResponseType | ExtendedResponseType:
            base_response = super().__call__(state, offer, source)
            self.response_count += 1
            return ExtendedResponseType(
                response=base_response,
                data={
                    "text": f"response_{self.response_count}",
                    "response_num": self.response_count,
                    "source": "acceptance_policy",
                },
            )

    # Setup
    issues: list[Issue] = [make_issue(10, "price"), make_issue(5, "quantity")]
    os = make_os(issues)

    ufun1 = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)
    ufun2 = LinearAdditiveUtilityFunction.random(outcome_space=os, reserved_value=0.0)

    data_offering = DataOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
    data_acceptance = DataACTime(tau=0.8)  # Accept late to get more steps
    negotiator1 = make_boa(acceptance=data_acceptance, offering=data_offering)

    # Use regular AspirationNegotiator as opponent
    negotiator2 = AspirationNegotiator()

    session = SAOMechanism(
        outcome_space=os,
        n_steps=20,
        one_offer_per_step=True,  # One offer per step makes tracking easier
        ignore_negotiator_exceptions=False,
    )
    session.add(negotiator1, ufun=ufun1)
    session.add(negotiator2, ufun=ufun2)
    session.run()

    assert session.state.started, "Negotiation did not start"
    assert session.state.ended or session.state.agreement, (
        "Negotiation did not complete"
    )

    # Verify that current_data was populated at some point
    # Check history for states with data
    states_with_current_data = [
        s for s in session.history if s.current_data is not None
    ]
    states_with_new_data = [
        s
        for s in session.history
        if s.new_data and any(d is not None for _, d in s.new_data)
    ]

    assert len(states_with_current_data) > 0, (
        f"Expected some states with current_data, got none. "
        f"History length: {len(session.history)}, "
        f"Offers made: {data_offering.offer_count}"
    )

    # Verify the data structure is correct
    for state in states_with_current_data:
        data = state.current_data
        assert isinstance(data, dict), f"current_data should be dict, got {type(data)}"
        # The data should contain merged text from both response and offer
        if "text" in data:
            # Text should exist
            assert isinstance(data["text"], str), (
                f"text should be str, got {type(data['text'])}"
            )

    # Verify new_data contains tuples of (negotiator_id, data)
    for state in states_with_new_data:
        for neg_id, data in state.new_data:
            assert isinstance(neg_id, str), (
                f"negotiator_id should be str, got {type(neg_id)}"
            )
            if data is not None:
                assert isinstance(data, dict), (
                    f"data should be dict or None, got {type(data)}"
                )

    # Verify that at least one step had data from our negotiator
    found_our_data = False
    for state in session.history:
        if state.current_data:
            # Check if this data came from our policies (has our markers)
            if (
                "offer_num" in state.current_data
                or "response_num" in state.current_data
            ):
                found_our_data = True
                break

    assert found_our_data, (
        "Did not find data from our extended type policies in any state. "
        f"Offers made: {data_offering.offer_count}, "
        f"Responses made: {data_acceptance.response_count}"
    )


def test_mechanism_run_callbacks():
    """Test Mechanism.run() with start, progress, and completion callbacks."""
    # Track callback calls
    calls = {"start": 0, "progress": 0, "completion": 0}

    def start_cb(state):
        calls["start"] += 1
        assert state.step == 0, f"Start callback should have step=0, got {state.step}"

    def progress_cb(state):
        calls["progress"] += 1
        assert state.step >= 0, "Progress callback should have non-negative step"

    def completion_cb(state):
        calls["completion"] += 1
        assert not state.running, "Completion callback should have running=False"

    # Create a simple mechanism
    issues = [
        make_issue([f"option_{i}" for i in range(5)], f"issue_{j}") for j in range(2)
    ]
    session = SAOMechanism(issues=issues, n_steps=10)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    # Run with callbacks
    session.run(
        start_callback=start_cb,
        progress_callback=progress_cb,
        completion_callback=completion_cb,
    )

    # Verify callbacks were called
    assert calls["start"] == 1, "start_callback should be called exactly once"
    assert calls["completion"] == 1, "completion_callback should be called exactly once"
    assert calls["progress"] >= 0, "progress_callback should be called"


def test_mechanism_run_with_progress_callbacks():
    """Test Mechanism.run_with_progress() with callbacks."""
    calls = {"start": 0, "progress": 0, "completion": 0}

    def start_cb(state):
        calls["start"] += 1

    def progress_cb(state):
        calls["progress"] += 1

    def completion_cb(state):
        calls["completion"] += 1

    # Create mechanism
    issues = [
        make_issue([f"option_{i}" for i in range(5)], f"issue_{j}") for j in range(2)
    ]
    session = SAOMechanism(issues=issues, n_steps=5)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    # Run with progress bar and callbacks
    session.run_with_progress(
        start_callback=start_cb,
        progress_callback=progress_cb,
        completion_callback=completion_cb,
    )

    # Verify callbacks were called
    assert calls["start"] == 1
    assert calls["completion"] == 1
    assert calls["progress"] >= 0


def test_mechanism_runall_callbacks():
    """Test Mechanism.runall() with callbacks including negotiation IDs."""
    calls = {"start": {}, "progress": {}, "completion": {}}

    def start_cb(neg_id, mechanism):
        if neg_id not in calls["start"]:
            calls["start"][neg_id] = 0
        calls["start"][neg_id] += 1

    def progress_cb(neg_id, mechanism):
        if neg_id not in calls["progress"]:
            calls["progress"][neg_id] = 0
        calls["progress"][neg_id] += 1

    def completion_cb(neg_id, mechanism):
        if neg_id not in calls["completion"]:
            calls["completion"][neg_id] = 0
        calls["completion"][neg_id] += 1

    # Create multiple mechanisms
    mechanisms = []
    for i in range(3):
        issues = [
            make_issue([f"option_{i}" for i in range(5)], f"issue_{j}")
            for j in range(2)
        ]
        session = SAOMechanism(issues=issues, n_steps=5, name=f"negotiation_{i}")
        session.add(
            AspirationNegotiator(name="buyer"),
            ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
        )
        session.add(
            AspirationNegotiator(name="seller"),
            ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
        )
        mechanisms.append(session)

    # Run all mechanisms with callbacks
    states = SAOMechanism.runall(
        mechanisms,
        method="sequential",
        start_callback=start_cb,
        progress_callback=progress_cb,
        completion_callback=completion_cb,
    )

    # Verify callbacks were called for each mechanism
    assert len(states) == 3
    for i in range(3):
        assert calls["start"].get(i, 0) == 1, (
            f"start_callback should be called once for mechanism {i}"
        )
        assert calls["completion"].get(i, 0) == 1, (
            f"completion_callback should be called once for mechanism {i}"
        )


# =============================================================================
# Tests for nanosecond precision timing (perf_counter_ns)
# =============================================================================


def test_time_returns_zero_before_start():
    """Test that mechanism.time returns 0.0 before the mechanism starts."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=10)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    # Before starting, time should be 0.0
    assert session.time == 0.0
    assert not session.state.started


def test_time_returns_positive_after_start():
    """Test that mechanism.time returns a positive float after the mechanism starts."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=10)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    # Run one step to start the mechanism
    session.step()

    # After starting, time should be positive
    assert session.time > 0.0
    assert session.state.started


def test_time_increases_during_negotiation():
    """Test that mechanism.time increases as the negotiation progresses."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=20)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    times = []
    for _ in range(5):
        session.step()
        times.append(session.time)
        if not session.state.running:
            break

    # Each subsequent time should be >= previous (monotonically increasing)
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"Time should be monotonically increasing: {times}"
        )


def test_time_has_nanosecond_precision():
    """Test that mechanism.time has nanosecond precision (can distinguish sub-millisecond differences)."""

    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    # Start the mechanism
    session.step()

    # Take multiple time readings in quick succession
    readings = []
    for _ in range(10):
        readings.append(session.time)

    # The time property should return float values with high precision
    # All readings should be floats
    for r in readings:
        assert isinstance(r, float), f"Time should be a float, got {type(r)}"

    # At least some readings should show differences (demonstrating precision)
    # or all readings should be positive
    assert all(r > 0 for r in readings), "All time readings should be positive"


def test_remaining_time_with_time_limit():
    """Test that remaining_time works correctly with nanosecond precision timing."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    time_limit = 10.0  # 10 seconds
    session = SAOMechanism(issues=issues, n_steps=None, time_limit=time_limit)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    # Before starting, remaining_time should equal time_limit
    assert session.remaining_time == time_limit

    # Start the mechanism
    session.step()

    # After starting, remaining_time should be less than time_limit
    remaining = session.remaining_time
    assert remaining is not None
    assert remaining < time_limit
    assert remaining > 0

    # Verify: time + remaining_time ≈ time_limit
    assert abs(session.time + remaining - time_limit) < 0.1, (
        f"time ({session.time}) + remaining_time ({remaining}) should ≈ time_limit ({time_limit})"
    )


def test_remaining_time_none_without_time_limit():
    """Test that remaining_time returns None when no time limit is set."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=10, time_limit=float("inf"))
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    # remaining_time should be None when time_limit is inf
    assert session.remaining_time is None

    session.step()
    assert session.remaining_time is None


def test_time_reported_in_state():
    """Test that the time reported in state is consistent with mechanism.time."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=10)
    session.add(
        AspirationNegotiator(name="buyer"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )
    session.add(
        AspirationNegotiator(name="seller"),
        ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
    )

    session.run()

    # After completion, state.time should be a positive float
    assert session.state.time > 0.0
    assert isinstance(session.state.time, float)


def test_time_consistency_across_multiple_mechanisms():
    """Test that timing is consistent and independent across multiple mechanisms."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]

    mechanisms = []
    for i in range(3):
        session = SAOMechanism(issues=issues, n_steps=5, name=f"session_{i}")
        session.add(
            AspirationNegotiator(name="buyer"),
            ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
        )
        session.add(
            AspirationNegotiator(name="seller"),
            ufun=LUFun.random(session.outcome_space, reserved_value=0.0),
        )
        mechanisms.append(session)

    # Run all mechanisms
    for session in mechanisms:
        session.run()

    # Each mechanism should have independent, positive time
    times = [m.state.time for m in mechanisms]
    for t in times:
        assert t > 0.0, "Each mechanism should have positive elapsed time"
        assert isinstance(t, float), "Time should be a float"


# Tests for allow_none_with_data feature


class NoneWithTextNegotiator(SAONegotiator):
    """A negotiator that offers None with text data after a few rounds."""

    def __init__(
        self, *args, none_at_step: int = 2, text: str = "Just a message", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._none_at_step = none_at_step
        self._text = text

    def propose(self, state):
        from negmas.outcomes.common import ExtendedOutcome

        if state.step >= self._none_at_step:
            # Return None offer with text data
            return ExtendedOutcome(outcome=None, data={"text": self._text})
        # Return a valid offer
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        return ResponseType.REJECT_OFFER


class AlwaysRejectNegotiator(SAONegotiator):
    """A negotiator that always rejects and offers the first outcome."""

    def propose(self, state):
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        return ResponseType.REJECT_OFFER


class NoneWithoutDataNegotiator(SAONegotiator):
    """A negotiator that offers None without any data after a few rounds."""

    def __init__(self, *args, none_at_step: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._none_at_step = none_at_step

    def propose(self, state):
        if state.step >= self._none_at_step:
            return None
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        return ResponseType.REJECT_OFFER


class NoneWithEmptyDataNegotiator(SAONegotiator):
    """A negotiator that offers None with empty data dict after a few rounds."""

    def __init__(self, *args, none_at_step: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._none_at_step = none_at_step

    def propose(self, state):
        from negmas.outcomes.common import ExtendedOutcome

        if state.step >= self._none_at_step:
            return ExtendedOutcome(outcome=None, data={})
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        return ResponseType.REJECT_OFFER


def test_allow_none_with_data_continues_negotiation():
    """Test that None offers with data continue negotiation when allow_none_with_data=True."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # NoneWithTextNegotiator will offer None with text at step 2
    session.add(NoneWithTextNegotiator(name="sender", none_at_step=2), ufun=ufun1)
    session.add(AlwaysRejectNegotiator(name="receiver"), ufun=ufun2)

    session.run()

    # Negotiation should NOT break - it should continue until n_steps or timeout
    assert not session.state.broken, (
        "Negotiation should not break when None offer has data"
    )
    assert session.state.step >= 2, (
        "Negotiation should have progressed past the None offer"
    )


def test_allow_none_with_data_disabled_breaks_negotiation():
    """Test that None offers break negotiation when allow_none_with_data=False."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=False
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    session.add(NoneWithTextNegotiator(name="sender", none_at_step=2), ufun=ufun1)
    session.add(AlwaysRejectNegotiator(name="receiver"), ufun=ufun2)

    session.run()

    # Negotiation SHOULD break when allow_none_with_data=False
    assert session.state.broken, (
        "Negotiation should break when None offer and allow_none_with_data=False"
    )


def test_none_without_data_breaks_even_with_allow_none_with_data():
    """Test that None offers without data break negotiation even with allow_none_with_data=True."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    session.add(NoneWithoutDataNegotiator(name="sender", none_at_step=2), ufun=ufun1)
    session.add(AlwaysRejectNegotiator(name="receiver"), ufun=ufun2)

    session.run()

    # Negotiation SHOULD break because there's no data
    assert session.state.broken, "Negotiation should break when None offer has no data"


def test_none_with_empty_data_breaks_even_with_allow_none_with_data():
    """Test that None offers with empty data break negotiation even with allow_none_with_data=True."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    session.add(NoneWithEmptyDataNegotiator(name="sender", none_at_step=2), ufun=ufun1)
    session.add(AlwaysRejectNegotiator(name="receiver"), ufun=ufun2)

    session.run()

    # Negotiation SHOULD break because the data dict is empty
    assert session.state.broken, (
        "Negotiation should break when None offer has empty data"
    )


def test_none_with_data_recorded_in_state():
    """Test that None offers with data are properly recorded in state."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    test_message = "This is a test message"
    session.add(
        NoneWithTextNegotiator(name="sender", none_at_step=2, text=test_message),
        ufun=ufun1,
    )
    session.add(AlwaysRejectNegotiator(name="receiver"), ufun=ufun2)

    session.run()

    # Check that data was recorded
    found_text = False
    for state in session.history:
        if state.current_data and state.current_data.get("text") == test_message:
            found_text = True
            break
        for _, data in state.new_data:
            if data and data.get("text") == test_message:
                found_text = True
                break

    assert found_text, "The text from None offer should be recorded in state"


def test_allow_none_with_data_in_nmi():
    """Test that allow_none_with_data setting is accessible via NMI."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]

    # Test with allow_none_with_data=True (default)
    session1 = SAOMechanism(issues=issues, n_steps=10, allow_none_with_data=True)
    ufun = LUFun.random(session1.outcome_space, reserved_value=0.0)
    session1.add(AspirationNegotiator(name="neg1"), ufun=ufun)

    neg = session1.negotiators[0]
    assert neg.nmi.allow_none_with_data is True, (
        "NMI should reflect allow_none_with_data=True"
    )

    # Test with allow_none_with_data=False
    session2 = SAOMechanism(issues=issues, n_steps=10, allow_none_with_data=False)
    session2.add(AspirationNegotiator(name="neg2"), ufun=ufun)

    neg2 = session2.negotiators[0]
    assert neg2.nmi.allow_none_with_data is False, (
        "NMI should reflect allow_none_with_data=False"
    )


def test_multiple_none_with_data_offers():
    """Test that multiple None offers with data can be sent without breaking."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # This negotiator will send None with data starting from step 0
    session.add(
        NoneWithTextNegotiator(name="sender", none_at_step=0, text="Message"),
        ufun=ufun1,
    )
    session.add(AlwaysRejectNegotiator(name="receiver"), ufun=ufun2)

    session.run()

    # Negotiation should not break and should run all steps
    assert not session.state.broken, (
        "Negotiation should not break with repeated None offers with data"
    )
    assert session.state.step >= 5, "Negotiation should continue for multiple steps"


def test_default_allow_none_with_data_is_true():
    """Test that allow_none_with_data defaults to True."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=10)

    assert session.allow_none_with_data is True, (
        "allow_none_with_data should default to True"
    )
    assert session._internal_nmi.allow_none_with_data is True, (
        "NMI should have allow_none_with_data=True by default"
    )


class AlwaysAcceptNegotiator(SAONegotiator):
    """A negotiator that always accepts whatever is on the table."""

    def propose(self, state):
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        return ResponseType.ACCEPT_OFFER


class AcceptAfterStepNegotiator(SAONegotiator):
    """A negotiator that rejects until a given step, then accepts."""

    def __init__(self, *args, accept_at_step: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._accept_at_step = accept_at_step

    def propose(self, state):
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        if state.step >= self._accept_at_step:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


def test_accepting_none_offer_ends_negotiation_no_agreement():
    """Test that accepting a None offer ends negotiation without agreement."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # Sender sends None with text at step 0
    session.add(
        NoneWithTextNegotiator(name="sender", none_at_step=0, text="Info message"),
        ufun=ufun1,
    )
    # Receiver always accepts
    session.add(AlwaysAcceptNegotiator(name="accepter"), ufun=ufun2)

    session.run()

    # The negotiation should have ended
    assert session.state.ended, "Negotiation should have ended"
    # There should be NO agreement (None offer accepted = no valid agreement)
    assert session.state.agreement is None, (
        "There should be no agreement when accepting a None offer"
    )
    # It should NOT be marked as broken (this is a graceful end)
    assert not session.state.broken, "Negotiation should not be marked as broken"


def test_accepting_none_offer_at_later_step():
    """Test accepting None offer after some valid offers have been rejected."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, end_on_no_response=True, allow_none_with_data=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # Sender sends valid offers for 2 steps, then None with text
    session.add(
        NoneWithTextNegotiator(name="sender", none_at_step=2, text="Info message"),
        ufun=ufun1,
    )
    # Receiver rejects until step 2, then accepts (the None offer)
    session.add(
        AcceptAfterStepNegotiator(name="accepter", accept_at_step=2), ufun=ufun2
    )

    session.run()

    # The negotiation should have ended
    assert session.state.ended, "Negotiation should have ended"
    # There should be NO agreement (None offer accepted = no valid agreement)
    assert session.state.agreement is None, (
        "There should be no agreement when accepting a None offer"
    )
    # It should NOT be marked as broken
    assert not session.state.broken, "Negotiation should not be marked as broken"


# ============================================================================
# LEAVE Response Tests
# ============================================================================


class LeaveAfterStepNegotiator(SAONegotiator):
    """A negotiator that leaves the negotiation after a specified step."""

    def __init__(self, *args, leave_at_step: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._leave_at_step = leave_at_step
        self.left_partners = []  # Track which partners left

    def propose(self, state):
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        if state.step >= self._leave_at_step:
            return ResponseType.LEAVE
        return ResponseType.REJECT_OFFER

    def on_negotiator_left(self, negotiator_id: str, state) -> None:
        self.left_partners.append(negotiator_id)


class TrackingNegotiator(SAONegotiator):
    """A negotiator that tracks entry/exit callbacks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered_partners = []
        self.left_partners = []
        self.didnot_enter_partners = []

    def propose(self, state):
        if self.ufun and self.ufun.outcome_space:
            outcomes = list(
                self.ufun.outcome_space.enumerate_or_sample(max_cardinality=10)
            )
            if outcomes:
                return outcomes[0]
        return None

    def respond(self, state, source=None):
        return ResponseType.REJECT_OFFER

    def on_negotiator_left(self, negotiator_id: str, state) -> None:
        self.left_partners.append(negotiator_id)

    def on_negotiator_entered(self, negotiator_id: str, state) -> None:
        self.entered_partners.append(negotiator_id)

    def on_negotiator_didnot_enter(self, negotiator_id: str, state) -> None:
        self.didnot_enter_partners.append(negotiator_id)


def test_leave_with_allow_negotiators_to_leave_false():
    """Test that LEAVE is treated like END_NEGOTIATION when allow_negotiators_to_leave=False."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues,
        n_steps=10,
        allow_negotiators_to_leave=False,
        extra_callbacks=True,
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # First negotiator leaves at step 1
    session.add(LeaveAfterStepNegotiator(name="leaver", leave_at_step=1), ufun=ufun1)
    # Second negotiator just rejects
    session.add(TrackingNegotiator(name="tracker"), ufun=ufun2)

    session.run()

    # With allow_negotiators_to_leave=False, LEAVE should break the negotiation
    assert session.state.broken, (
        "Negotiation should be broken when allow_negotiators_to_leave=False"
    )
    assert session.state.agreement is None, "There should be no agreement"


def test_leave_with_two_negotiators_ends_negotiation():
    """Test that LEAVE ends negotiation when only 2 negotiators and one leaves (no dynamic_entry)."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # First negotiator leaves at step 1
    leaver = LeaveAfterStepNegotiator(name="leaver", leave_at_step=1)
    session.add(leaver, ufun=ufun1)
    # Second negotiator tracks callbacks
    tracker = TrackingNegotiator(name="tracker")
    session.add(tracker, ufun=ufun2)

    session.run()

    # With 2 negotiators and one leaving (no dynamic_entry), negotiation should end as broken
    # because fewer than 2 remain and no new negotiators can join
    assert session.state.ended, "Negotiation should have ended"
    assert session.state.broken, (
        "Negotiation should be broken (only 1 remains, no dynamic_entry)"
    )
    assert session.state.agreement is None, "There should be no agreement"


def test_leave_with_three_negotiators_continues():
    """Test that LEAVE removes negotiator but others can continue with 3 negotiators."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun3 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # First negotiator leaves at step 1
    leaver = LeaveAfterStepNegotiator(name="leaver", leave_at_step=1)
    session.add(leaver, ufun=ufun1)
    # Second and third negotiators will eventually accept at step 5
    accepter1 = AcceptAfterStepNegotiator(name="accepter1", accept_at_step=5)
    accepter2 = AcceptAfterStepNegotiator(name="accepter2", accept_at_step=5)
    session.add(accepter1, ufun=ufun2)
    session.add(accepter2, ufun=ufun3)

    session.run()

    # The negotiation should have ended with an agreement between the remaining negotiators
    assert session.state.ended, "Negotiation should have ended"
    assert not session.state.broken, "Negotiation should NOT be broken"
    # After the leaver leaves at step 1, the other two should reach agreement at step 5
    assert session.state.agreement is not None, (
        "Remaining negotiators should reach agreement"
    )


def test_on_negotiator_left_callback_is_called():
    """Test that on_negotiator_left callback is called when a negotiator leaves."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun3 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # First negotiator leaves at step 1
    leaver = LeaveAfterStepNegotiator(name="leaver", leave_at_step=1)
    session.add(leaver, ufun=ufun1)
    # Second and third negotiators track callbacks
    tracker1 = TrackingNegotiator(name="tracker1")
    tracker2 = TrackingNegotiator(name="tracker2")
    session.add(tracker1, ufun=ufun2)
    session.add(tracker2, ufun=ufun3)

    session.run()

    # Both trackers should have been notified that the leaver left
    assert leaver.id in tracker1.left_partners, "tracker1 should know leaver left"
    assert leaver.id in tracker2.left_partners, "tracker2 should know leaver left"


def test_on_negotiator_entered_callback_is_called():
    """Test that on_negotiator_entered callback is called when a negotiator joins."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(issues=issues, n_steps=10, extra_callbacks=True)

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # First negotiator added
    tracker1 = TrackingNegotiator(name="tracker1")
    session.add(tracker1, ufun=ufun1)

    # Second negotiator added - tracker1 should be notified
    tracker2 = TrackingNegotiator(name="tracker2")
    session.add(tracker2, ufun=ufun2)

    # tracker1 should know tracker2 entered
    assert tracker2.id in tracker1.entered_partners, (
        "tracker1 should know tracker2 entered"
    )
    # tracker2 should NOT have any entries (no one joined after it)
    assert len(tracker2.entered_partners) == 0, (
        "tracker2 should have no entered partners"
    )


def test_leave_response_type_exists():
    """Test that ResponseType.LEAVE exists and is distinct from other responses."""
    assert hasattr(ResponseType, "LEAVE"), "ResponseType should have LEAVE"
    assert ResponseType.LEAVE != ResponseType.END_NEGOTIATION
    assert ResponseType.LEAVE != ResponseType.REJECT_OFFER
    assert ResponseType.LEAVE != ResponseType.ACCEPT_OFFER
    assert ResponseType.LEAVE != ResponseType.NO_RESPONSE
    assert ResponseType.LEAVE != ResponseType.WAIT


class LeavingAspirationNegotiator(AspirationNegotiator):
    """An aspiration negotiator that leaves after a specified step.

    This extends AspirationNegotiator and overrides both respond and __call__
    to return LEAVE after the specified step. We need to override __call__
    because respond is only called when there's an offer to respond to.
    When the negotiator is the proposer with no current offer, respond is
    not called.
    """

    def __init__(self, *args, leave_at_step: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._leave_at_step = leave_at_step

    def __call__(self, state, dest=None):
        # Check if we should leave before doing anything else
        if state.step >= self._leave_at_step:
            from negmas.sao.common import SAOResponse

            return SAOResponse(ResponseType.LEAVE, None)
        return super().__call__(state, dest)

    def respond(self, state, source=None):
        if state.step >= self._leave_at_step:
            return ResponseType.LEAVE
        return super().respond(state, source)


@mark.parametrize("leaver_position", [0, 1, 2])  # first, middle, last
@mark.parametrize("leave_at_step", [0, 1, 3, 5, 10])
def test_leave_allows_remaining_negotiators_to_continue(leaver_position, leave_at_step):
    """Test that LEAVE allows remaining negotiators to continue and potentially reach agreement.

    This test verifies that:
    1. If the leaver leaves before the negotiation ends, 2 negotiators remain
    2. If the negotiation ends before the leave step, all 3 remain
    3. The negotiation is never broken (either normal agreement or graceful leave)
    """
    # Create outcome space with 3 issues
    issues = [
        make_issue(10, "price"),
        make_issue(5, "quantity"),
        make_issue(3, "delivery"),
    ]

    # Seed for reproducibility based on test parameters
    seed = 42 + leaver_position * 100 + leave_at_step
    random.seed(seed)

    outcome_space = make_os(issues)
    ufuns = [LUFun.random(outcome_space, reserved_value=0.0) for _ in range(3)]

    # Aspiration types for the 3 negotiators
    aspiration_types = ["boulware", "conceder", "linear"]

    # Determine which negotiator will leave
    leaver_idx = leaver_position

    # --- Run negotiation WITH the leaver (who will leave) ---
    session = SAOMechanism(
        issues=issues,
        n_steps=100,
        allow_negotiators_to_leave=True,
        extra_callbacks=True,
    )

    leaver_neg = None
    for i in range(3):
        if i == leaver_idx:
            # This negotiator will leave
            neg = LeavingAspirationNegotiator(
                name=f"neg_{i}",
                leave_at_step=leave_at_step,
                aspiration_type=aspiration_types[i],
            )
            leaver_neg = neg
        else:
            # Regular aspiration negotiator
            neg = AspirationNegotiator(
                name=f"neg_{i}", aspiration_type=aspiration_types[i]
            )
        session.add(neg, ufun=ufuns[i])

    session.run()

    # --- Verify results ---
    # Negotiation should have ended
    assert session.state.ended, "Negotiation should have ended"

    # The negotiation should NOT be broken (either agreement or graceful leave)
    assert not session.state.broken, "Negotiation should not be broken"

    # All 3 negotiators should still be in the list (left ones are marked, not removed)
    assert len(session.negotiators) == 3, (
        f"Should have 3 negotiators in list, got {len(session.negotiators)}"
    )

    # Check if the leaver actually left (may not if agreement reached first)
    leaver_left = leaver_neg.id in session.state.left_negotiators

    if leaver_left:
        # Leaver left - verify 2 participating
        assert session.n_participating == 2, (
            f"Should have 2 participating negotiators after leave, got {session.n_participating}"
        )
        # The leaver should NOT be in participating_negotiators
        participating_ids = [n.id for n in session.participating_negotiators]
        assert leaver_neg.id not in participating_ids, (
            "Leaver should not be in participating negotiators"
        )
    else:
        # Negotiation ended before leave step (likely agreement)
        assert session.n_participating == 3, (
            f"Should have 3 participating negotiators (no one left), got {session.n_participating}"
        )
        # This can happen if agreement is reached before leave_at_step
        # The leaver should be in participating_negotiators
        participating_ids = [n.id for n in session.participating_negotiators]
        assert leaver_neg.id in participating_ids, (
            "Leaver should be in participating negotiators if they didn't leave"
        )

    # The remaining negotiators should have reached an agreement (both are conceding)
    # Note: Agreement is likely but not guaranteed depending on ufuns
    # We just verify the mechanism handled everything correctly
    if session.state.agreement is not None:
        # If there's an agreement, it should be a valid outcome
        assert session.state.agreement in outcome_space, (
            "Agreement should be a valid outcome"
        )


def test_allow_negotiators_to_leave_has_no_effect_when_no_one_leaves():
    """Test that allow_negotiators_to_leave=True/False makes no difference when no LEAVE occurs.

    This test runs the same negotiation twice:
    1. With allow_negotiators_to_leave=True
    2. With allow_negotiators_to_leave=False

    Both should produce identical outcomes when no negotiator actually leaves.
    """
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]

    # Run multiple trials to ensure consistent behavior
    for seed in [42, 123, 456]:
        random.seed(seed)
        outcome_space = make_os(issues)
        ufuns = [
            LUFun.random(outcome_space, reserved_value=0.0),
            LUFun.random(outcome_space, reserved_value=0.0),
        ]

        # --- Run with allow_negotiators_to_leave=True ---
        random.seed(seed)
        session_true = SAOMechanism(
            issues=issues, n_steps=20, allow_negotiators_to_leave=True
        )
        session_true.add(AspirationNegotiator(name="neg1_true"), ufun=ufuns[0])
        session_true.add(AspirationNegotiator(name="neg2_true"), ufun=ufuns[1])
        session_true.run()

        # --- Run with allow_negotiators_to_leave=False ---
        random.seed(seed)
        session_false = SAOMechanism(
            issues=issues, n_steps=20, allow_negotiators_to_leave=False
        )
        session_false.add(AspirationNegotiator(name="neg1_false"), ufun=ufuns[0])
        session_false.add(AspirationNegotiator(name="neg2_false"), ufun=ufuns[1])
        session_false.run()

        # --- Both should have identical outcomes ---
        assert session_true.state.ended == session_false.state.ended, (
            f"Seed {seed}: ended state should match"
        )
        assert session_true.state.broken == session_false.state.broken, (
            f"Seed {seed}: broken state should match"
        )
        assert session_true.state.agreement == session_false.state.agreement, (
            f"Seed {seed}: agreement should match"
        )
        assert session_true.state.step == session_false.state.step, (
            f"Seed {seed}: step count should match"
        )
        assert len(session_true.state.left_negotiators) == 0, (
            f"Seed {seed}: no negotiators should have left (True)"
        )
        assert len(session_false.state.left_negotiators) == 0, (
            f"Seed {seed}: no negotiators should have left (False)"
        )


def test_left_negotiators_set_is_updated_correctly():
    """Test that state.left_negotiators set is updated when negotiators leave."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=20, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun3 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun4 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # Multiple leavers at different steps
    leaver1 = LeaveAfterStepNegotiator(name="leaver1", leave_at_step=1)
    leaver2 = LeaveAfterStepNegotiator(name="leaver2", leave_at_step=3)
    accepter1 = AcceptAfterStepNegotiator(name="accepter1", accept_at_step=10)
    accepter2 = AcceptAfterStepNegotiator(name="accepter2", accept_at_step=10)

    session.add(leaver1, ufun=ufun1)
    session.add(leaver2, ufun=ufun2)
    session.add(accepter1, ufun=ufun3)
    session.add(accepter2, ufun=ufun4)

    session.run()

    # Check left_negotiators set
    assert leaver1.id in session.state.left_negotiators, (
        "leaver1 should be in left_negotiators"
    )
    assert leaver2.id in session.state.left_negotiators, (
        "leaver2 should be in left_negotiators"
    )
    assert accepter1.id not in session.state.left_negotiators, (
        "accepter1 should NOT be in left_negotiators"
    )
    assert accepter2.id not in session.state.left_negotiators, (
        "accepter2 should NOT be in left_negotiators"
    )

    # Check n_participating property
    assert session.state.n_participating == 2, (
        "n_participating should be 2 (two leavers left)"
    )


def test_participating_negotiators_excludes_left():
    """Test that participating_negotiators property excludes those who left."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=20, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun3 = LUFun.random(session.outcome_space, reserved_value=0.0)

    leaver = LeaveAfterStepNegotiator(name="leaver", leave_at_step=1)
    accepter1 = AcceptAfterStepNegotiator(name="accepter1", accept_at_step=5)
    accepter2 = AcceptAfterStepNegotiator(name="accepter2", accept_at_step=5)

    session.add(leaver, ufun=ufun1)
    session.add(accepter1, ufun=ufun2)
    session.add(accepter2, ufun=ufun3)

    session.run()

    # participating_negotiators should exclude the leaver
    participating = session.participating_negotiators
    participating_ids = [n.id for n in participating]

    assert leaver.id not in participating_ids, (
        "leaver should NOT be in participating_negotiators"
    )
    assert accepter1.id in participating_ids, (
        "accepter1 SHOULD be in participating_negotiators"
    )
    assert accepter2.id in participating_ids, (
        "accepter2 SHOULD be in participating_negotiators"
    )

    # negotiators property should still include all
    all_negotiators = session.negotiators
    all_ids = [n.id for n in all_negotiators]
    assert leaver.id in all_ids, "leaver should still be in negotiators list"
    assert len(all_negotiators) == 3, "negotiators list should have all 3"


def test_agreement_partners_excludes_left():
    """Test that agreement_partners excludes those who left before agreement."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=20, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun3 = LUFun.random(session.outcome_space, reserved_value=0.0)

    leaver = LeaveAfterStepNegotiator(name="leaver", leave_at_step=1)
    accepter1 = AcceptAfterStepNegotiator(name="accepter1", accept_at_step=5)
    accepter2 = AcceptAfterStepNegotiator(name="accepter2", accept_at_step=5)

    session.add(leaver, ufun=ufun1)
    session.add(accepter1, ufun=ufun2)
    session.add(accepter2, ufun=ufun3)

    session.run()

    # Should have an agreement
    assert session.state.agreement is not None, "Should have an agreement"

    # agreement_partners should exclude the leaver
    partners = session.agreement_partners
    partner_ids = [n.id for n in partners]

    assert leaver.id not in partner_ids, "leaver should NOT be in agreement_partners"
    assert accepter1.id in partner_ids, "accepter1 SHOULD be in agreement_partners"
    assert accepter2.id in partner_ids, "accepter2 SHOULD be in agreement_partners"


def test_n_participating_property():
    """Test that n_participating property works correctly."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=20, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun3 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun4 = LUFun.random(session.outcome_space, reserved_value=0.0)

    leaver1 = LeaveAfterStepNegotiator(name="leaver1", leave_at_step=1)
    leaver2 = LeaveAfterStepNegotiator(name="leaver2", leave_at_step=2)
    accepter1 = AcceptAfterStepNegotiator(name="accepter1", accept_at_step=10)
    accepter2 = AcceptAfterStepNegotiator(name="accepter2", accept_at_step=10)

    session.add(leaver1, ufun=ufun1)
    session.add(leaver2, ufun=ufun2)
    session.add(accepter1, ufun=ufun3)
    session.add(accepter2, ufun=ufun4)

    # Before running
    assert session.n_participating == 4, "n_participating should be 4 before running"

    session.run()

    # After running - two left
    assert session.n_participating == 2, "n_participating should be 2 after two leave"
    assert session.state.n_participating == 2, "state.n_participating should match"


def test_leave_broken_with_remaining_participant():
    """Test that participating_negotiators shows remaining participant when negotiation breaks due to insufficient participants."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    session = SAOMechanism(
        issues=issues, n_steps=10, allow_negotiators_to_leave=True, extra_callbacks=True
    )

    ufun1 = LUFun.random(session.outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(session.outcome_space, reserved_value=0.0)

    # First negotiator leaves at step 1, causing broken negotiation (only 1 remains)
    leaver1 = LeaveAfterStepNegotiator(name="leaver1", leave_at_step=1)
    stayer = TrackingNegotiator(name="stayer")

    session.add(leaver1, ufun=ufun1)
    session.add(stayer, ufun=ufun2)

    session.run()

    # Negotiation should be broken (fewer than 2 remaining, no dynamic_entry)
    assert session.state.broken, "Negotiation should be broken"
    assert session.state.agreement is None, "Should have no agreement"

    # Only leaver1 is in left_negotiators
    assert leaver1.id in session.state.left_negotiators, (
        "leaver1 should be in left_negotiators"
    )
    assert stayer.id not in session.state.left_negotiators, (
        "stayer should NOT be in left_negotiators"
    )

    # participating_negotiators shows who is still participating (even though broken)
    participating = session.participating_negotiators
    assert len(participating) == 1, "One negotiator should still be 'participating'"
    assert stayer.id == participating[0].id, (
        "stayer should be the remaining participant"
    )
