"""Tests for :class:`SessionMetrics` / :func:`calc_session_metrics`.

``SessionMetrics`` holds only the negotiator-independent view of a negotiation
(outcome + timing). Per-negotiator quantities live in ``NegotiatorMetrics`` and
are tested in ``test_negotiator_metrics.py``.
"""

from __future__ import annotations

import math

import pytest

from attrs import fields

from negmas.preferences.ops import (
    NegotiatorMetrics,
    SessionMetrics,
    calc_session_metrics,
)


def _isclose(a: float, b: float, **kw) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, **kw)


def test_session_agreement_with_ranges():
    # Bilateral agreement (0.6, 0.4) with symmetric [0, 1] ranges.
    m = calc_session_metrics(
        (0.6, 0.4),
        (0.0, 0.0),
        utility_ranges=[(0.0, 1.0), (0.0, 1.0)],
        relative_time=0.4,
        last_step=5,
        pareto_utils=[(1.0, 0.0), (0.0, 1.0), (0.6, 0.4)],
    )
    assert _isclose(m.welfare, 0.5)
    assert _isclose(m.welfare_agreed, 0.5)
    assert _isclose(m.utility_sum, 1.0)
    assert m.agreement_reached == 1.0
    assert _isclose(m.utility_gap, 0.2)  # 0.6 - 0.4
    assert _isclose(m.advantage_gap, 0.2)  # adv0=0.6, adv1=0.4
    assert _isclose(m.gains_from_trade, 1.0)
    assert _isclose(m.surplus_efficiency, 1.0)  # T = T_max = 1.0
    assert _isclose(m.relative_rounds, 0.4)
    assert _isclose(m.agreement_speed, 0.6)  # 1 - relative_time
    assert _isclose(m.relative_efficiency, -0.2 / 5.0)


def test_session_no_agreement_falls_back_to_reserved():
    m = calc_session_metrics(None, (0.2, 0.1), utility_ranges=[(0.0, 1.0), (0.0, 1.0)])
    assert m.agreement_reached == 0.0
    assert _isclose(m.welfare, (0.2 + 0.1) / 2)  # realised = reserved
    assert math.isnan(m.welfare_agreed)
    assert m.agreement_speed == 0.0  # no agreement -> zero speed
    assert _isclose(m.gains_from_trade, 0.0)  # realised == reserved


def test_session_advantage_gap_without_ranges_uses_raw_surplus():
    # No ranges: advantage falls back to raw surplus u - r.
    m = calc_session_metrics((0.6, 0.4), (0.0, 0.1))
    # adv0 = 0.6 - 0.0 = 0.6, adv1 = 0.4 - 0.1 = 0.3 -> gap 0.3
    assert _isclose(m.advantage_gap, 0.3)
    assert _isclose(m.utility_gap, 0.2)


def test_session_surplus_efficiency_fallback_to_ranges():
    # No pareto_utils: denominator falls back to sum(hi - r).
    m = calc_session_metrics(
        (0.6, 0.4), (0.0, 0.0), utility_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    # total surplus = 1.0, t_max (ranges) = 2.0 -> 0.5
    assert _isclose(m.surplus_efficiency, 0.5)
    # Without ranges either -> nan
    m2 = calc_session_metrics((0.6, 0.4), (0.0, 0.0))
    assert math.isnan(m2.surplus_efficiency)


def test_session_relative_efficiency_uses_last_step():
    m = calc_session_metrics(
        (0.9, 0.1), (0.0, 0.0), utility_ranges=[(0.0, 1.0), (0.0, 1.0)], last_step=8
    )
    assert _isclose(m.relative_efficiency, -(0.9 - 0.1) / 8.0)


def test_session_is_negotiator_independent():
    """Every session metric is symmetric: swapping the two parties (and their
    ranges/reserved values) must not change any session metric."""
    a = calc_session_metrics(
        (0.7, 0.3),
        (0.1, 0.2),
        utility_ranges=[(0.0, 1.0), (0.0, 1.0)],
        relative_time=0.5,
        last_step=4,
    )
    b = calc_session_metrics(
        (0.3, 0.7),
        (0.2, 0.1),
        utility_ranges=[(0.0, 1.0), (0.0, 1.0)],
        relative_time=0.5,
        last_step=4,
    )
    for f in fields(SessionMetrics):
        assert _isclose(getattr(a, f.name), getattr(b, f.name)), f.name


def test_session_and_negotiator_fields_are_disjoint():
    session_fields = {f.name for f in fields(SessionMetrics)}
    negotiator_fields = {f.name for f in fields(NegotiatorMetrics)}
    assert session_fields & negotiator_fields == set()


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
