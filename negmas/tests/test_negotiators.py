from __future__ import annotations

import random
from random import choice

import numpy as np
import pytest

from negmas import (
    AspirationNegotiator,
    BestOutcomeOnlyNegotiator,
    FirstOfferOrientedTBNegotiator,
    NaiveTitForTatNegotiator,
    SAOController,
    SAOMechanism,
    TopFractionNegotiator,
    ToughNegotiator,
)
from negmas.outcomes.base_issue import make_issue
from negmas.preferences import MappingUtilityFunction, RandomUtilityFunction
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

random.seed(0)
np.random.seed(0)


def test_tough_asp_negotiator():
    a1 = ToughNegotiator()
    a2 = AspirationNegotiator(aspiration_type="conceder")
    outcomes = [(_,) for _ in range(10)]
    u1 = MappingUtilityFunction(
        dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes)).tolist())),
        outcomes=outcomes,
    )
    u2 = MappingUtilityFunction(
        dict(zip(outcomes, (1 - np.linspace(0.0, 1.0, len(outcomes))).tolist())),
        outcomes=outcomes,
    )
    neg = SAOMechanism(outcomes=outcomes, n_steps=100)
    neg.add(a1, preferences=u1)
    neg.add(a2, preferences=u2)
    neg.run()
    a1offers = neg.negotiator_offers(a1.id)
    a2offers = neg.negotiator_offers(a2.id)
    assert a1._offerable_outcomes is None
    if len(a1offers) > 0:
        assert len(set(a1offers)) == 1 and a1offers[-1] == (9,)
    assert len(set(a2offers)) >= 0


def test_tough_tit_for_tat_negotiator():
    a1 = ToughNegotiator()
    a2 = NaiveTitForTatNegotiator()
    outcomes = [(_,) for _ in range(10)]
    u1 = MappingUtilityFunction(
        dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes)).tolist())),
        outcomes=outcomes,
    )
    u2 = MappingUtilityFunction(
        dict(zip(outcomes, (1 - np.linspace(0.0, 1.0, len(outcomes))).tolist())),
        outcomes=outcomes,
    )
    neg = SAOMechanism(outcomes=outcomes, n_steps=100)
    neg.add(a1, preferences=u1)
    neg.add(a2, preferences=u2)
    neg.run()
    a1offers = neg.negotiator_offers(a1.id)
    a2offers = neg.negotiator_offers(a2.id)
    # print(a1offers)
    # print(a2offers)
    assert a1._offerable_outcomes is None
    if len(a1offers) > 0:
        assert len(set(a1offers)) == 1 and a1offers[-1] == (9,)
    assert len(set(a2offers)) >= 0


def test_asp_negotaitor():
    a1 = AspirationNegotiator(name="a1")
    a2 = AspirationNegotiator(name="a2")
    outcomes = [(_,) for _ in range(10)]
    u1 = MappingUtilityFunction(
        dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes)).tolist())),
        outcomes=outcomes,
    )
    u2 = MappingUtilityFunction(
        dict(zip(outcomes, (1 - np.linspace(0.0, 1.0, len(outcomes))).tolist())),
        outcomes=outcomes,
    )
    neg = SAOMechanism(outcomes=outcomes, n_steps=100)
    neg.add(a1, preferences=u1)
    neg.add(a2, preferences=u2)
    neg.run()
    a1offers = neg.negotiator_offers(a1.id)
    a2offers = neg.negotiator_offers(a2.id)
    assert a1offers[0] == (9,)
    assert a2offers[0] == (0,)
    for i, offer in enumerate(_[0] for _ in a1offers):
        assert i == 0 or offer <= a1offers[i - 1][0]
    for i, offer in enumerate(_[0] for _ in a2offers):
        assert i == 0 or offer >= a2offers[i - 1][0]
    assert neg.state.agreement is not None
    assert neg.state.agreement in ((4,), (5,))


def test_tit_for_tat_negotiators_agree_in_the_middle():
    a1 = NaiveTitForTatNegotiator(name="a1")
    a2 = NaiveTitForTatNegotiator(name="a2")
    outcomes = [(_,) for _ in range(10)]
    u1 = MappingUtilityFunction(
        dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes)).tolist())),
        outcomes=outcomes,
    )
    u2 = MappingUtilityFunction(
        dict(zip(outcomes, (1 - np.linspace(0.0, 1.0, len(outcomes))).tolist())),
        outcomes=outcomes,
    )
    neg = SAOMechanism(outcomes=outcomes, n_steps=100, avoid_ultimatum=False)
    neg.add(a1, preferences=u1)
    neg.add(a2, preferences=u2)
    neg.run()
    a1offers = neg.negotiator_offers(a1.id)
    a2offers = neg.negotiator_offers(a2.id)
    # print(a1offers)
    # print(a2offers)
    assert a1offers[0] == (9,)
    assert a2offers[0] == (0,)
    for i, offer in enumerate(_[0] for _ in a1offers):
        assert i == 0 or offer <= a1offers[i - 1][0]
    for i, offer in enumerate(_[0] for _ in a2offers):
        assert i == 0 or offer >= a2offers[i - 1][0]
    assert neg.state.agreement is not None
    assert neg.state.agreement in ((4,), (5,))


def test_top_only_negotiator():
    outcomes = [(_,) for _ in range(10)]
    a1 = BestOutcomeOnlyNegotiator(name="a1")
    a2 = BestOutcomeOnlyNegotiator(name="a2")
    u1 = 22.0 - np.linspace(0.0, 22.0, len(outcomes))
    neg = SAOMechanism(outcomes=outcomes, n_steps=10, avoid_ultimatum=False)
    neg.add(
        a1,
        preferences=MappingUtilityFunction(
            dict(zip(outcomes, u1)), outcome_space=neg.outcome_space
        ),
    )
    neg.add(
        a2,
        preferences=MappingUtilityFunction(
            dict(zip(outcomes, 22 - u1)), outcome_space=neg.outcome_space
        ),
    )
    neg.run()
    assert neg.state.timedout
    first = neg.negotiator_offers(neg.negotiator_ids[0])
    second = neg.negotiator_offers(neg.negotiator_ids[1])
    assert len(set(first)) == 1
    assert len(set(second)) == 1
    assert first[0] == (0,)
    assert second[0] == (9,)


class TestTitForTatNegotiator:
    def test_propose(self):
        outcomes = [(_,) for _ in range(10)]
        a1 = NaiveTitForTatNegotiator(name="a1", initial_concession="min")
        a2 = BestOutcomeOnlyNegotiator(name="a2")
        u1 = 22.0 - np.linspace(0.0, 22.0, len(outcomes))
        neg = SAOMechanism(outcomes=outcomes, n_steps=10, avoid_ultimatum=False)
        neg.add(
            a1,
            preferences=MappingUtilityFunction(
                dict(zip(outcomes, u1)), outcome_space=neg.outcome_space
            ),
        )
        neg.add(
            a2,
            preferences=MappingUtilityFunction(
                dict(zip(outcomes, 22 - u1)), outcome_space=neg.outcome_space
            ),
        )
        neg.step()
        proposal = neg.negotiator_offers(neg.negotiators[0].id)[0]
        assert proposal == (0,), "Proposes top first"

        neg.step()
        proposal = neg.negotiator_offers(neg.negotiators[0].id)[1]
        assert proposal == (1,), "Proposes second second if min concession is set"

        a1 = NaiveTitForTatNegotiator(name="a1")
        a2 = BestOutcomeOnlyNegotiator(name="a1")
        u1 = [50.0] * 3 + (22 - np.linspace(10.0, 22.0, len(outcomes) - 3)).tolist()
        neg = SAOMechanism(outcomes=outcomes, n_steps=10, avoid_ultimatum=False)
        neg.add(
            a1,
            preferences=MappingUtilityFunction(lambda x: u1[x[0]], outcomes=outcomes),
        )
        neg.add(
            a2,
            preferences=MappingUtilityFunction(
                lambda x: 22 - u1[x[0]], outcomes=outcomes
            ),
        )

        neg.step()
        proposal = neg.negotiator_offers(neg.negotiators[0].id)[-1]
        assert proposal == (0,), "Proposes top first"

        neg.step()
        proposal = neg.negotiator_offers(neg.negotiators[0].id)[-1]
        assert proposal == (
            3,
        ), "Proposes first item with utility less than the top if concession is min"


def test_tit_for_tat_against_asp_negotiators():
    a1 = NaiveTitForTatNegotiator(name="a1")
    a2 = AspirationNegotiator(name="a2")
    outcomes = [(_,) for _ in range(10)]
    u1 = MappingUtilityFunction(
        dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes)).tolist())),
        outcomes=outcomes,
    )
    u2 = MappingUtilityFunction(
        dict(zip(outcomes, (1 - np.linspace(0.0, 1.0, len(outcomes))).tolist())),
        outcomes=outcomes,
    )
    neg = SAOMechanism(
        outcomes=outcomes, n_steps=10, avoid_ultimatum=False, time_limit=None
    )
    neg.add(a1, preferences=u1)
    neg.add(a2, preferences=u2)
    neg.run()
    a1offers = neg.negotiator_offers(a1.id)
    a2offers = neg.negotiator_offers(a2.id)
    assert a1offers[0] == (9,)
    # assert a2offers[0] == (0,)
    for i, offer in enumerate(_[0] for _ in a1offers):
        assert i == 0 or offer <= a1offers[i - 1][0]
    for i, offer in enumerate(_[0] for _ in a2offers):
        assert i == 0 or offer >= a2offers[i - 1][0]
    assert neg.state.agreement is not None
    assert neg.state.agreement in ((1,), (2,), (3,), (4,), (5,), (6,))


def test_best_only_asp_negotiator():
    a1 = TopFractionNegotiator(min_utility=0.9, top_fraction=0.1)
    a2 = AspirationNegotiator(aspiration_type="conceder")
    outcomes = [(_,) for _ in range(20)]
    u1 = MappingUtilityFunction(
        dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes)).tolist())),
        outcomes=outcomes,
    )
    u2 = MappingUtilityFunction(
        dict(zip(outcomes, (1 - np.linspace(0.0, 1.0, len(outcomes))).tolist())),
        outcomes=outcomes,
    )
    neg = SAOMechanism(outcomes=outcomes, n_steps=200)
    neg.add(a1, preferences=u1)
    neg.add(a2, preferences=u2)
    neg.run()
    a1offers = neg.negotiator_offers(a1.id)
    a2offers = neg.negotiator_offers(a2.id)
    assert a1._offerable_outcomes is None
    if len(a1offers) > 0:
        assert (
            len(set(a1offers)) <= 2
            and min(u1(_) for _ in a1offers if _ is not None) >= 0.9
        )
    assert len(set(a2offers)) >= 1


def test_controller():
    n_sessions = 5
    c = SAOController(
        default_negotiator_type="negmas.sao.AspirationNegotiator",
        default_negotiator_params={"aspiration_type": "conceder"},
    )
    sessions = [SAOMechanism(outcomes=10, n_steps=10) for _ in range(n_sessions)]
    for session in sessions:
        session.add(
            AspirationNegotiator(aspiration_type="conceder"),
            preferences=RandomUtilityFunction(outcome_space=session.outcome_space),
        )
        session.add(
            c.create_negotiator(),
            preferences=RandomUtilityFunction(outcome_space=session.outcome_space),
        )
    completed: list[int] = []
    while len(completed) < n_sessions:
        for i, session in enumerate(sessions):
            if i in completed:
                continue
            state = session.step()
            if state.broken or state.timedout or state.agreement is not None:
                completed.append(i)
    # we are just checking that the controller runs. No need to assert anything


def test_negotiator_checkpoint():
    pass


def test_smart_aspiration():
    # create negotiation agenda (issues)
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]

    # create the mechanism
    session = SAOMechanism(issues=issues, n_steps=20)

    # define ufuns
    seller_utility = LinearAdditiveUtilityFunction(
        values={
            "price": IdentityFun(),
            "quantity": LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9),
        },
        weights={"price": 1.0, "quantity": 1.0, "delivery_time": 10.0},
        outcome_space=session.outcome_space,
    ).scale_max(1.0)
    buyer_utility = LinearAdditiveUtilityFunction(
        values={
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
    ).scale_max(1.0)

    session.add(FirstOfferOrientedTBNegotiator(name="buyer"), ufun=buyer_utility)
    session.add(FirstOfferOrientedTBNegotiator(name="seller"), ufun=seller_utility)
    session.run()


if __name__ == "__main__":
    pytest.main(args=[__file__])
