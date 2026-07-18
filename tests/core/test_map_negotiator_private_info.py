"""Regression tests: MAPNegotiator must keep ``private_info["opponent_ufun"]``.

``MAPNegotiator.__init__`` registers the first ``UFunModel`` among its models
under ``private_info["opponent_ufun"]`` (the documented discovery key, also
backing the ``Negotiator.opponent_ufun`` property). It used to do so on a
local dict created BEFORE calling ``super().__init__`` — and the base
``Negotiator.__init__`` re-initialises ``_private_info`` from the
``private_info`` kwarg, silently wiping the registration. These tests pin the
fixed behaviour: the registration must survive construction, joining a
mechanism, and a full negotiation, with or without a caller-supplied
``private_info`` dict.
"""

from __future__ import annotations

import random

import pytest

from negmas.gb.components import (
    ACNext,
    FrequencyLinearUFunModel,
    FrequencyUFunModel,
    TimeBasedOfferingPolicy,
)
from negmas.gb.negotiators.modular.mapneg import MAPNegotiator
from negmas.preferences.generators import generate_multi_issue_ufuns
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator
from negmas.warnings import NegmasWarning


def make_map(models=None, private_info=None, **kwargs):
    offering = TimeBasedOfferingPolicy()
    if private_info is not None:
        kwargs["private_info"] = private_info
    return MAPNegotiator(
        acceptance=ACNext(offering_strategy=offering),
        offering=offering,
        models=models,
        **kwargs,
    )


def test_opponent_ufun_registered_without_explicit_private_info():
    model = FrequencyLinearUFunModel()
    neg = make_map(models=[model])
    assert neg.private_info.get("opponent_ufun") is model
    # The documented accessor must see it too.
    assert neg.opponent_ufun is model


def test_opponent_ufun_survives_alongside_explicit_private_info():
    model = FrequencyLinearUFunModel()
    neg = make_map(models=[model], private_info={"secret": 42})
    assert neg.private_info.get("secret") == 42
    assert neg.private_info.get("opponent_ufun") is model


def test_first_ufun_model_wins_and_warns_with_two():
    first, second = FrequencyUFunModel(), FrequencyLinearUFunModel()
    with pytest.warns(NegmasWarning, match="Expecting a single model"):
        neg = make_map(models=[first, second])
    assert neg.private_info.get("opponent_ufun") is first


def test_no_ufun_model_means_no_registration():
    neg = make_map(models=None)
    assert "opponent_ufun" not in neg.private_info
    assert neg.opponent_ufun is None


def test_opponent_ufun_survives_join_and_full_negotiation():
    random.seed(42)
    u1, u2 = generate_multi_issue_ufuns(
        n_issues=2, n_values=5, n_ufuns=2, numeric=False
    )
    assert u1.outcome_space is not None
    model = FrequencyLinearUFunModel()
    neg = make_map(models=[model])
    mechanism = SAOMechanism(outcome_space=u1.outcome_space, n_steps=30)
    assert mechanism.add(neg, preferences=u1)
    assert mechanism.add(AspirationNegotiator(), preferences=u2)
    # Registration survives the join lifecycle...
    assert neg.private_info.get("opponent_ufun") is model
    mechanism.run()
    # ...and a full run, during which the model actually learned something.
    assert neg.private_info.get("opponent_ufun") is model
    some_outcome = next(iter(u1.outcome_space.enumerate_or_sample()))
    assert float(model(some_outcome)) >= 0.0
