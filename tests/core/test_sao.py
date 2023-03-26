from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import example, given, settings
from pytest import mark

import negmas
from negmas import all_negotiator_types
from negmas.gb.negotiators.timebased import AspirationNegotiator
from negmas.outcomes import Issue, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import EndImmediately, NoneOfferingPolicy, RejectAlways, SAOMechanism
from negmas.sao.negotiators.modular.boa import make_boa

NEGTYPES = all_negotiator_types()


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


def test_pend_works():
    issues: list[Issue] = [
        make_issue(10, "price"),
        make_issue(10, "quantity"),
        make_issue(["red", "green", "blue"], "color"),
    ]
    ufuns = [
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
        LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0),
    ]
    n = 1000
    f = 0.01
    session = SAOMechanism(n_steps=None, time_limit=None, pend=f / n, issues=issues)
    for u in ufuns:
        session.add(AspirationNegotiator(), preferences=u)  # type: ignore

    assert abs(session.expected_relative_time - (f / (n + 1))) < 1e-8
    assert session.expected_remaining_time is None
    assert session.expected_remaining_steps is not None
    assert abs(session.expected_remaining_steps - n / f) < 4
    assert abs(session.relative_time - (f / (n + 1))) < 1e-8
    assert session.remaining_steps is None
    assert session.remaining_time is None
    assert session.run().agreement is not None
    assert session.state.step <= 10000 * n


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
