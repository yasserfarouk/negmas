from typing import List
import random
import numpy as np
import pytest
from negmas import (
    ToughNegotiator,
    SAOMechanism,
    AspirationNegotiator,
    OnlyBestNegotiator,
    SAOController,
    NaiveTitForTatNegotiator,
)
from negmas.utilities import RandomUtilityFunction

random.seed(0)
np.random.seed(0)


def test_tough_asp_negotiator():
    a1 = ToughNegotiator()
    a2 = AspirationNegotiator(aspiration_type="conceder")
    outcomes = [(_,) for _ in range(10)]
    u1 = np.linspace(0.0, 1.0, len(outcomes))
    u2 = 1.0 - u1
    neg = SAOMechanism(outcomes=outcomes, n_steps=100)
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    neg.run()
    a1offers = [s.current_offer for s in neg.history if s.current_proposer == a1.id]
    a2offers = [s.current_offer for s in neg.history if s.current_proposer == a2.id]
    assert a1._offerable_outcomes is None
    if len(a1offers) > 0:
        assert len(set(a1offers)) == 1 and a1offers[-1] == (9,)
    assert len(set(a2offers)) >= 0


def test_tough_tit_for_tat_negotiator():
    a1 = ToughNegotiator()
    a2 = NaiveTitForTatNegotiator()
    outcomes = [(_,) for _ in range(10)]
    u1 = np.linspace(0.0, 1.0, len(outcomes))
    u2 = 1.0 - u1
    neg = SAOMechanism(outcomes=outcomes, n_steps=100)
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    neg.run()
    a1offers = [s.current_offer for s in neg.history if s.current_proposer == a1.id]
    a2offers = [s.current_offer for s in neg.history if s.current_proposer == a2.id]
    print(a1offers)
    print(a2offers)
    assert a1._offerable_outcomes is None
    if len(a1offers) > 0:
        assert len(set(a1offers)) == 1 and a1offers[-1] == (9,)
    assert len(set(a2offers)) >= 0


def test_asp_negotaitor():
    a1 = AspirationNegotiator(assume_normalized=True, name="a1")
    a2 = AspirationNegotiator(assume_normalized=False, name="a2")
    outcomes = [(_,) for _ in range(10)]
    u1 = np.linspace(0.0, 1.0, len(outcomes))
    u2 = 1.0 - u1
    neg = SAOMechanism(outcomes=outcomes, n_steps=100)
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    neg.run()
    a1offers = [s.current_offer for s in neg.history if s.current_proposer == a1.id]
    a2offers = [s.current_offer for s in neg.history if s.current_proposer == a2.id]
    assert a1offers[0] == (9,)
    assert a2offers[0] == (0,)
    for i, offer in enumerate(_[0] for _ in a1offers):
        assert i == 0 or offer <= a1offers[i - 1][0]
    for i, offer in enumerate(_[0] for _ in a2offers):
        assert i == 0 or offer >= a2offers[i - 1][0]
    assert neg.state.agreement is not None
    assert neg.state.agreement in ((4,), (5,))


def test_tit_for_tat_negotiators():
    a1 = NaiveTitForTatNegotiator(name="a1")
    a2 = NaiveTitForTatNegotiator(name="a2")
    outcomes = [(_,) for _ in range(10)]
    u1 = np.linspace(0.0, 1.0, len(outcomes))
    u2 = 1.0 - u1
    neg = SAOMechanism(outcomes=outcomes, n_steps=100, avoid_ultimatum=False)
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    neg.run()
    a1offers = [s.current_offer for s in neg.history if s.current_proposer == a1.id]
    a2offers = [s.current_offer for s in neg.history if s.current_proposer == a2.id]
    print(a1offers)
    print(a2offers)
    assert a1offers[0] == (9,)
    assert a2offers[0] == (0,)
    for i, offer in enumerate(_[0] for _ in a1offers):
        assert i == 0 or offer <= a1offers[i - 1][0]
    for i, offer in enumerate(_[0] for _ in a2offers):
        assert i == 0 or offer >= a2offers[i - 1][0]
    assert neg.state.agreement is not None
    assert neg.state.agreement in ((4,), (5,))


class TestTitForTatNegotiator:
    def test_propose(self):
        outcomes = [(_,) for _ in range(10)]
        a1 = NaiveTitForTatNegotiator(name="a1", initial_concession="min")
        u1 = 22.0 - np.linspace(0.0, 22.0, len(outcomes))
        neg = SAOMechanism(outcomes=outcomes, n_steps=10, avoid_ultimatum=False)
        neg.add(a1, ufun=u1)

        proposal = a1.propose_(neg.state)
        assert proposal == (0,), "Proposes top first"
        proposal = a1.propose_(neg.state)
        assert proposal == (1,), "Proposes second second if min concession is set"

        a1 = NaiveTitForTatNegotiator(name="a1")
        u1 = [50.0] * 3 + (22 - np.linspace(10.0, 22.0, len(outcomes) - 3)).tolist()
        neg = SAOMechanism(outcomes=outcomes, n_steps=10, avoid_ultimatum=False)
        neg.add(a1, ufun=u1)

        proposal = a1.propose_(neg.state)
        assert proposal == (0,), "Proposes top first"
        proposal = a1.propose_(neg.state)
        assert proposal == (
            3,
        ), "Proposes first item with utility less than the top if concession is min"


def test_tit_for_tat_against_asp_negotiators():
    a1 = NaiveTitForTatNegotiator(name="a1")
    a2 = AspirationNegotiator(name="a2")
    outcomes = [(_,) for _ in range(10)]
    u1 = np.linspace(0.0, 1.0, len(outcomes))
    u2 = 1.0 - u1
    neg = SAOMechanism(outcomes=outcomes, n_steps=10, avoid_ultimatum=False)
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    neg.run()
    a1offers = [s.current_offer for s in neg.history if s.current_proposer == a1.id]
    a2offers = [s.current_offer for s in neg.history if s.current_proposer == a2.id]
    assert a1offers[0] == (9,)
    assert a2offers[0] == (0,)
    for i, offer in enumerate(_[0] for _ in a1offers):
        assert i == 0 or offer <= a1offers[i - 1][0]
    for i, offer in enumerate(_[0] for _ in a2offers):
        assert i == 0 or offer >= a2offers[i - 1][0]
    assert neg.state.agreement is not None
    assert neg.state.agreement in ((3,), (4,), (5,), (6,))


def test_best_only_asp_negotiator():
    a1 = OnlyBestNegotiator(min_utility=0.9, top_fraction=0.1)
    a2 = AspirationNegotiator(aspiration_type="conceder")
    outcomes = [(_,) for _ in range(20)]
    u1 = np.linspace(0.0, 1.0, len(outcomes))
    u2 = 1.0 - u1
    neg = SAOMechanism(outcomes=outcomes, n_steps=200)
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    neg.run()
    a1offers = [s.current_offer for s in neg.history if s.current_proposer == a1.id]
    a2offers = [s.current_offer for s in neg.history if s.current_proposer == a2.id]
    assert a1._offerable_outcomes is None
    if len(a1offers) > 0:
        assert (
            len(set(a1offers)) <= 2
            and min([u1[_[0]] for _ in a1offers if _ is not None]) >= 0.9
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
            ufun=RandomUtilityFunction(outcomes=session.outcomes),
        )
        session.add(
            c.create_negotiator(), ufun=RandomUtilityFunction(outcomes=session.outcomes)
        )
    completed: List[int] = []
    while len(completed) < n_sessions:
        for i, session in enumerate(sessions):
            if i in completed:
                continue
            state = session.step()
            if state.broken or state.timedout or state.agreement is not None:
                completed.append(i)
    # we are just asserting that the controller runs


if __name__ == "__main__":
    pytest.main(args=[__file__])
