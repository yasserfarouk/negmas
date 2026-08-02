"""Tests for :class:`FullMetrics` / :func:`calc_full_metrics` and the metric
helpers exposed on :class:`~negmas.inout.Scenario`, :class:`~negmas.mechanisms.Mechanism`
and :class:`~negmas.mechanisms.CompletedRun`.
"""

from __future__ import annotations

import math
import tempfile

import pytest

from attrs import fields

from negmas.inout import Scenario
from negmas.mechanisms import CompletedRun
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.ops import (
    FullMetrics,
    NegotiatorMetrics,
    OutcomeOptimality,
    SessionMetrics,
    calc_full_metrics,
)
from negmas.preferences.value_fun import AffineFun, IdentityFun
from negmas.sao import AspirationNegotiator, SAOMechanism


def _isclose(a: float, b: float, **kw) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, **kw)


def _make_scenario() -> Scenario:
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    os_ = make_os(issues, name="os")
    u1 = LinearAdditiveUtilityFunction(
        values=[IdentityFun(), IdentityFun()], outcome_space=os_, reserved_value=2.0
    )
    u2 = LinearAdditiveUtilityFunction(
        values=[AffineFun(slope=-1, bias=9), AffineFun(slope=-1, bias=4)],
        outcome_space=os_,
        reserved_value=1.0,
    )
    return Scenario(outcome_space=os_, ufuns=(u1, u2))


def _run(scenario: Scenario, n_steps: int = 40) -> SAOMechanism:
    m = SAOMechanism(outcome_space=scenario.outcome_space, n_steps=n_steps)
    for u in scenario.ufuns:
        m.add(AspirationNegotiator(), preferences=u)
    m.run()
    return m


def test_calc_full_metrics_bundles_the_three_views():
    trace = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
    fm = calc_full_metrics(
        trace,
        (0.6, 0.4),
        (0.0, 0.0),
        utility_ranges=[(0.0, 1.0), (0.0, 1.0)],
        relative_time=0.4,
        last_step=5,
        pareto_utils=[(1.0, 0.0), (0.0, 1.0), (0.6, 0.4)],
    )
    assert isinstance(fm, FullMetrics)
    assert isinstance(fm.session, SessionMetrics)
    assert isinstance(fm.optimality, OutcomeOptimality)
    assert len(fm.negotiators) == 2
    assert all(isinstance(n, NegotiatorMetrics) for n in fm.negotiators)
    # session and negotiator views are internally consistent with the calculators
    assert _isclose(fm.session.welfare, 0.5)
    assert _isclose(fm.negotiators[0].utility, 0.6)
    assert _isclose(fm.negotiators[1].utility, 0.4)
    # optimality is nan without scenario stats
    assert math.isnan(fm.optimality.pareto_optimality)


def test_calc_full_metrics_optimality_with_stats():
    scenario = _make_scenario()
    stats = scenario.calc_stats()
    ufuns = scenario.ufuns
    agreement = (4, 2)
    agreement_utils = tuple(float(u(agreement)) for u in ufuns)
    reserved = tuple(u.reserved_value for u in ufuns)
    fm = calc_full_metrics([], agreement_utils, reserved, ufuns=ufuns, stats=stats)
    assert not math.isnan(fm.optimality.pareto_optimality)
    assert 0.0 <= fm.optimality.pareto_optimality <= 1.0


def test_scenario_metric_wrappers():
    scenario = _make_scenario()
    agreement = (4, 2)
    sm = scenario.calc_session_metrics(agreement=agreement, relative_time=0.5)
    assert isinstance(sm, SessionMetrics)
    assert sm.agreement_reached == 1.0
    nm = scenario.calc_negotiator_metrics(index=0, agreement=agreement)
    assert isinstance(nm, NegotiatorMetrics)
    assert not math.isnan(nm.utility)
    fm = scenario.calc_full_metrics(agreement=agreement)
    assert isinstance(fm, FullMetrics)
    assert not math.isnan(fm.optimality.pareto_optimality)


def test_mechanism_metric_methods():
    scenario = _make_scenario()
    m = _run(scenario)
    sm = m.calc_session_metrics()
    assert isinstance(sm, SessionMetrics)
    nm0 = m.calc_negotiator_metrics(0)
    nm_by_id = m.calc_negotiator_metrics(m.negotiator_ids[0])
    assert isinstance(nm0, NegotiatorMetrics)
    assert _isclose(nm0.utility, nm_by_id.utility)
    fm = m.calc_full_metrics()
    assert isinstance(fm, FullMetrics)
    assert len(fm.negotiators) == 2
    # session welfare from the mechanism matches the standalone session metrics
    assert _isclose(fm.session.welfare, sm.welfare)
    if m.agreement is not None:
        assert not math.isnan(fm.optimality.pareto_optimality)


def test_mechanism_metrics_require_preferences_and_space():
    # A mechanism whose negotiators have no preferences cannot compute metrics.
    m = SAOMechanism(outcome_space=_make_scenario().outcome_space, n_steps=3)
    m.add(AspirationNegotiator())
    m.add(AspirationNegotiator())
    with pytest.raises(ValueError):
        m.calc_session_metrics()


def test_completed_run_calc_and_roundtrip():
    scenario = _make_scenario()
    m = _run(scenario)

    # to_completed_run can compute the full metrics eagerly
    cr = m.to_completed_run(calc_metrics=True)
    assert cr.full_metrics is not None
    welfare = cr.full_metrics.session.welfare

    # recompute lazily on a run created without metrics -> same session welfare
    cr2 = m.to_completed_run()
    assert cr2.full_metrics is None
    fm2 = cr2.calc_full_metrics()
    assert cr2.full_metrics is not None  # stored by default
    assert _isclose(fm2.session.welfare, welfare)

    # save/load round-trip preserves the metrics
    with tempfile.TemporaryDirectory() as d:
        path = cr.save(d, "run1")
        loaded = CompletedRun.load(path)
        assert loaded.full_metrics is not None
        assert _isclose(loaded.full_metrics.session.welfare, welfare)
        assert len(loaded.full_metrics.negotiators) == 2
        # metrics can also be recomputed from the loaded scenario + history
        recomputed = loaded.calc_full_metrics()
        assert _isclose(recomputed.session.welfare, welfare)


def test_completed_run_without_metrics_saved():
    scenario = _make_scenario()
    m = _run(scenario)
    cr = m.to_completed_run(calc_metrics=True)
    with tempfile.TemporaryDirectory() as d:
        path = cr.save(d, "run1", save_metrics=False)
        loaded = CompletedRun.load(path)
        assert loaded.full_metrics is None
        # but optimality (agreement_stats) is still recomputable/available
        recomputed = loaded.calc_full_metrics()
        assert isinstance(recomputed, FullMetrics)


def test_full_metrics_to_dict_flattens_all_metrics():
    scenario = _make_scenario()
    m = _run(scenario)
    fm = m.calc_full_metrics()
    d = fm.to_dict()
    # every value is a scalar
    assert all(isinstance(v, (int, float)) for v in d.values())
    # counts: n_optimality + n_session + n_negotiator * n_negotiators
    n_opt = len(fields(OutcomeOptimality))
    n_ses = len(fields(SessionMetrics))
    n_neg = len(fields(NegotiatorMetrics))
    assert len(d) == n_opt + n_ses + n_neg * len(fm.negotiators)
    # optimality and session fields use their own (unprefixed) names
    assert _isclose(d["pareto_optimality"], fm.optimality.pareto_optimality)
    assert _isclose(d["welfare"], fm.session.welfare)
    # per-negotiator fields default to neg{i}_ prefix
    assert _isclose(d["neg0_advantage"], fm.negotiators[0].advantage)
    assert _isclose(d["neg1_advantage"], fm.negotiators[1].advantage)


def test_full_metrics_to_dict_with_negotiator_names():
    scenario = _make_scenario()
    m = _run(scenario)
    fm = m.calc_full_metrics()
    d = fm.to_dict(negotiator_names=["buyer", "seller"])
    assert "buyer_advantage" in d
    assert "seller_advantage" in d
    assert "neg0_advantage" not in d
    assert _isclose(d["buyer_advantage"], fm.negotiators[0].advantage)
    assert _isclose(d["seller_advantage"], fm.negotiators[1].advantage)
    # optimality/session still unprefixed
    assert "welfare" in d and "pareto_optimality" in d


def test_full_metrics_components_disjoint():
    session_fields = {f.name for f in fields(SessionMetrics)}
    negotiator_fields = {f.name for f in fields(NegotiatorMetrics)}
    assert session_fields.isdisjoint(negotiator_fields)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
