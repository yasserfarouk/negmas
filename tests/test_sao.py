from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import example, given, settings
from pytest import mark

import negmas
from negmas import all_negotiator_types
from negmas.outcomes import Issue, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import EndImmediately, NoneOfferingPolicy, RejectAlways, SAOMechanism
from negmas.sao.negotiators.modular.boa import make_boa

NEGTYPES = all_negotiator_types()


@given(
    opp=st.sampled_from(NEGTYPES),
    start=st.booleans(),
    rejector=st.sampled_from([EndImmediately, RejectAlways]),
    avoid_ultimatum=st.booleans(),
)
@example(
    opp=negmas.sao.negotiators.timebased.AdditiveFirstFollowingTBNegotiator,
    start=True,
    rejector=negmas.sao.components.acceptance.EndImmediately,
    avoid_ultimatum=True,
)
@settings(deadline=500000)
def test_do_nothing_never_gets_agreements(opp, start, rejector, avoid_ultimatum):
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
    session = SAOMechanism(n_steps=1000, issues=issues, avoid_ultimatum=avoid_ultimatum)
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
