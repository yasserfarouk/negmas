import os
import numpy as np
import pytest
import pathlib
from hypothesis import given, settings
import hypothesis.strategies as st
from negmas import Negotiator, ToughNegotiator, MappingUtilityFunction, SAOMechanism, \
    AspirationNegotiator, OnlyBestNegotiator
from pprint import pprint



@pytest.fixture()
def dummyagent():
    class MyNegotiator(Negotiator):

        def ufun(self, negotiation_id=None):
            return None

        def respond_(self, state, offer):
            return None

        def isin(self, negotiation_id):
            return False

        def evaluate(self, offer):
            return 0.0

        def propose_(self, state):
            return None

        def enter(self, negotiation, ufun):
            return False

    return MyNegotiator()


def test_agent_has_good_name(dummyagent):
    x = dummyagent

    assert x.isin(None) is False


def test_tough_asp_negotiator():
        a1 = ToughNegotiator(dynamic_ufun=False)
        a2 = AspirationNegotiator(dynamic_ufun=False, aspiration_type='conceder')
        outcomes = [(_,) for _ in range(10)]
        u1 = np.linspace(0.0, 1.0, len(outcomes))
        u2 = 1.0 - u1
        neg = SAOMechanism(outcomes=outcomes, n_steps=20)
        neg.add(a1, ufun=u1)
        neg.add(a2, ufun=u2)
        neg.run()
        a1offers = [s.current_offer for s in neg.history if s.current_offerer == a1.id]
        a2offers = [s.current_offer for s in neg.history if s.current_offerer == a2.id]
        assert a1.offerable_outcomes is None
        if len(a1offers) > 0:
            assert len(set(a1offers)) == 1 and a1offers[-1] == (9,)
        assert len(set(a2offers)) >= 0


def test_best_only_asp_negotiator():
    a1 = OnlyBestNegotiator(dynamic_ufun=False, min_utility=0.9, top_fraction=0.1)
    a2 = AspirationNegotiator(dynamic_ufun=False, aspiration_type='conceder')
    outcomes = [(_,) for _ in range(20)]
    u1 = np.linspace(0.0, 1.0, len(outcomes))
    u2 = 1.0 - u1
    neg = SAOMechanism(outcomes=outcomes, n_steps=200)
    neg.add(a1, ufun=u1)
    neg.add(a2, ufun=u2)
    neg.run()
    a1offers = [s.current_offer for s in neg.history if s.current_offerer == a1.id]
    a2offers = [s.current_offer for s in neg.history if s.current_offerer == a2.id]
    assert a1.offerable_outcomes is None
    if len(a1offers) > 0:
        assert len(set(a1offers)) <= 2 and min([u1[_[0]] for _ in a1offers if _ is not None]) >= 0.9
    assert len(set(a2offers)) >= 1


if __name__ == '__main__':
    pytest.main(args=[__file__])
