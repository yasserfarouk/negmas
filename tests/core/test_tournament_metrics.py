"""Tests that cartesian tournaments record the full negotiation metrics
(session + per-negotiator), not just optimality."""

from __future__ import annotations

import math

import pytest

from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import AffineFun, IdentityFun
from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.tournaments.neg.simple.cartesian import (
    NEGOTIATOR_METRIC_COLS,
    SESSION_METRIC_COLS,
    cartesian_tournament,
)


def _make_scenario() -> Scenario:
    issues = [make_issue(6, "price"), make_issue(4, "quantity")]
    os_ = make_os(issues, name="os")
    u1 = LinearAdditiveUtilityFunction(
        values=[IdentityFun(), IdentityFun()], outcome_space=os_, reserved_value=2.0
    )
    u2 = LinearAdditiveUtilityFunction(
        values=[AffineFun(slope=-1, bias=5), AffineFun(slope=-1, bias=3)],
        outcome_space=os_,
        reserved_value=1.0,
    )
    return Scenario(outcome_space=os_, ufuns=(u1, u2))


def test_cartesian_tournament_records_full_metrics():
    res = cartesian_tournament(
        competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
        scenarios=[_make_scenario()],
        n_repetitions=1,
        verbosity=0,
    )
    df = res.details
    # session metrics are scalar columns
    for f in SESSION_METRIC_COLS:
        assert f"session_{f}" in df.columns, f"missing session_{f}"
    # per-negotiator metrics are list-valued columns
    for f in NEGOTIATOR_METRIC_COLS:
        assert f"neg_{f}" in df.columns, f"missing neg_{f}"

    # list columns hold one entry per negotiator
    sample = df["neg_dominance"].iloc[0]
    assert hasattr(sample, "__len__") and len(sample) == 2

    # session welfare is a finite scalar
    assert not math.isnan(float(df["session_welfare"].iloc[0]))


def test_cartesian_scores_expand_per_negotiator_metrics():
    res = cartesian_tournament(
        competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
        scenarios=[_make_scenario()],
        n_repetitions=1,
        verbosity=0,
    )
    scores = res.scores
    # per-negotiator metric columns are expanded to scalars in the score table
    assert "neg_dominance" in scores.columns
    assert all(
        not hasattr(v, "__len__") or isinstance(v, str)
        for v in scores["neg_dominance"].tolist()
    )
    # session metric columns are copied verbatim (scalar) into each score row
    assert "session_welfare" in scores.columns


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
