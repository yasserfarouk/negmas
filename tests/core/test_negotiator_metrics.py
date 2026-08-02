"""Tests for :class:`NegotiatorMetrics` / :func:`calc_negotiator_metrics`.

``NegotiatorMetrics`` holds only the per-negotiator view of a negotiation.
Negotiator-independent quantities live in ``SessionMetrics`` and are tested in
``test_session_metrics.py``.
"""

from __future__ import annotations

import math

import pytest

from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.ops import calc_negotiator_metrics, compare_ufuns


def _isclose(a: float, b: float, **kw) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, **kw)


def test_negotiator_agreement_with_trace():
    # Bilateral, party 0 concedes over its own offers (round-robin from party 0).
    trace = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
    m = calc_negotiator_metrics(
        trace, (0.6, 0.4), (0.0, 0.0), index=0, utility_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    # Own offers of party 0 (indices 0, 2, 4): self-utility [1.0, 0.8, 0.6].
    assert _isclose(m.utility, 0.6)
    assert _isclose(m.utility_agreed, 0.6)
    assert _isclose(m.advantage, 0.6)
    assert _isclose(m.advantage_agreed, 0.6)
    assert _isclose(m.opponent_advantage, 0.4)
    assert _isclose(m.opponent_advantage_agreed, 0.4)
    assert _isclose(m.surplus_share, 0.6)
    assert _isclose(m.total_concession, 0.4)
    assert _isclose(m.concession_rate, 0.2)  # (1.0 - 0.6) / (3 - 1)
    assert _isclose(m.temporal_patience, 1.0 / 3.0)  # first drop at own-offer 1 / 3
    assert m.dominance == 1.0  # adv0 > adv1
    assert m.rationality == 1.0
    assert _isclose(m.own_gain, -0.4)
    assert _isclose(m.concession_toward_counterparty, 0.4)
    assert _isclose(m.copling_raio, 0.4 / (0.4 + 1e-9))
    assert m.produced_any_offers == 1.0
    assert math.isnan(m.valid_offer_fraction)  # no outcomes/os supplied
    assert math.isnan(m.opp_kendall_optimality)


def test_negotiator_other_party_perspective():
    trace = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
    m1 = calc_negotiator_metrics(
        trace, (0.6, 0.4), (0.0, 0.0), index=1, utility_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    assert _isclose(m1.utility, 0.4)
    assert _isclose(m1.advantage, 0.4)
    assert _isclose(m1.opponent_advantage, 0.6)
    assert m1.dominance == -1.0  # party 1 loses on advantage
    assert _isclose(m1.surplus_share, 0.4)


def test_negotiator_no_agreement_falls_back_to_reserved():
    m = calc_negotiator_metrics(
        (), None, (0.2, 0.1), index=0, utility_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    assert _isclose(m.utility, 0.2)  # realised = reserved
    assert math.isnan(m.utility_agreed)
    assert math.isnan(m.advantage_agreed)
    assert math.isnan(m.total_concession)
    assert m.rationality == 1.0  # walking away is rational
    assert math.isnan(m.produced_any_offers)  # no trace at all
    assert math.isnan(m.surplus_share)  # total surplus 0


def test_negotiator_irrational_acceptance_and_no_ranges():
    m = calc_negotiator_metrics((), (0.05, 0.95), (0.3, 0.0), index=0)
    assert m.rationality == 0.0  # 0.05 < 0.3 reservation
    # advantage without ranges -> raw surplus 0.05 - 0.3 = -0.25
    assert _isclose(m.advantage, -0.25)


def test_negotiator_trace_parties_override_attribution():
    # Mark every offer as party 1's (party 0 made no own offers).
    trace = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4)]
    m = calc_negotiator_metrics(
        trace,
        (0.6, 0.4),
        (0.0, 0.0),
        index=0,
        utility_ranges=[(0.0, 1.0), (0.0, 1.0)],
        trace_parties=[1, 1, 1],
    )
    assert math.isnan(m.total_concession)  # party 0 made no own offers
    assert m.produced_any_offers == 0.0


def test_negotiator_dom_loss_and_tie():
    # party 0 worse than party 1 -> loss (-1)
    m = calc_negotiator_metrics(
        (), (0.2, 0.8), (0.0, 0.0), index=0, utility_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    assert m.dominance == -1.0
    # exact tie -> 0
    m2 = calc_negotiator_metrics(
        (), (0.5, 0.5), (0.0, 0.0), index=0, utility_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    assert m2.dominance == 0.0


def test_negotiator_opp_model_and_valid_offer_fraction_with_outcomes():
    issues = [make_issue([0, 1, 2, 3, 4], "price"), make_issue([0, 1, 2], "quality")]
    os_ = make_os(issues, name="os")
    u0 = LinearAdditiveUtilityFunction(
        {"price": lambda x: x / 4.0, "quality": lambda x: x / 2.0},
        issues=issues,
        reserved_value=0.0,
        name="u0",
    )
    u1 = LinearAdditiveUtilityFunction(
        {"price": lambda x: 1 - x / 4.0, "quality": lambda x: x / 2.0},
        issues=issues,
        reserved_value=0.0,
        name="u1",
    )
    ufuns = (u0, u1)
    ranges = [u0.minmax(issues=issues), u1.minmax(issues=issues)]
    offers = [(0, 2), (4, 2), (2, 2)]
    trace_utils = tuple((float(u0(o)), float(u1(o))) for o in offers)
    agreement = offers[-1]
    agreement_utils = (float(u0(agreement)), float(u1(agreement)))
    m = calc_negotiator_metrics(
        trace_utils,
        agreement_utils,
        (0.0, 0.0),
        index=0,
        utility_ranges=ranges,
        trace_offers=offers,
        outcome_space=os_,
        opponent_ufun=u0,
        ufuns=ufuns,
    )
    assert _isclose(
        m.opp_kendall_optimality,
        float(compare_ufuns(u0, u1, method="kendall_optimality", outcome_space=os_)),
    )
    assert _isclose(
        m.opp_ndcg, float(compare_ufuns(u0, u1, method="ndcg", outcome_space=os_))
    )
    assert _isclose(
        m.opp_euclidean,
        float(compare_ufuns(u0, u1, method="euclidean", outcome_space=os_)),
    )
    # party 0 own offers (round-robin indices 0, 2) are both in-space.
    assert _isclose(m.valid_offer_fraction, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
