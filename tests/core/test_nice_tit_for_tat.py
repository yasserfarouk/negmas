"""Tests for the Nice Tit for Tat negotiator (Baarslag, Hindriks & Jonker, 2013).

Covers:
- The agent runs in principle inside a `SAOMechanism`.
- Its bidding strategy concedes as expected when the opponent concedes,
  assuming a *correct* opponent model (the oracle `PeekingOpponentModel`).
- The ACcombi acceptance condition accepts when the opponent's offer beats the
  next planned offer and near the deadline, and rejects otherwise.
- The default opponent model and the configurable bargaining target.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from negmas.gb.common import ResponseType
from negmas.gb.components.acceptance import ACCombi
from negmas.gb.components.models.ufun import (
    FrequencyLinearUFunModel,
    FrequencyUFunModel,
    PeekingOpponentModel,
)
from negmas.gb.negotiators.titfortat import NiceTitForTatNegotiator
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.ops import nash_points, pareto_frontier
from negmas.preferences.value_fun import AffineFun, IdentityFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator, BoulwareTBNegotiator


def _random_ufuns(issues):
    """Two independent random linear-additive ufuns (non-deterministic)."""
    return (
        LUFun.random(issues=issues, reserved_value=0.0),
        LUFun.random(issues=issues, reserved_value=0.0),
    )


def _zerosum_ufuns(issues):
    """A deterministic (near) zero-sum pair: agent utility grows with the issue
    values, opponent utility shrinks with them. This makes the Nash point
    well-defined (the normalized midpoint) and guarantees that an opponent who
    offers the agent increasingly better outcomes is strictly conceding.
    """
    agent_ufun = LUFun(
        values=[IdentityFun() for _ in issues], issues=issues, reserved_value=0.0
    )
    opp_ufun = LUFun(
        values=[AffineFun(-1, bias=9) for _ in issues],
        issues=issues,
        reserved_value=0.0,
    )
    return agent_ufun, opp_ufun


def test_nice_tit_for_tat_runs_in_principle():
    """The negotiator runs inside a SAOMechanism without errors and either
    reaches an agreement or runs the negotiation to completion."""
    issues = [make_issue(8, "a"), make_issue(8, "b"), make_issue(4, "c")]
    agent_ufun, opp_ufun = _random_ufuns(issues)
    p = SAOMechanism(issues=issues, n_steps=40)
    p.add(NiceTitForTatNegotiator(name="ntft"), preferences=agent_ufun)
    p.add(AspirationNegotiator(name="asp"), preferences=opp_ufun)
    state = p.run()
    assert len(p.history) > 0
    if state.agreement is not None:
        # any agreement must be a rational (non-negative) outcome for the agent
        assert float(agent_ufun(state.agreement)) >= 0.0


def test_default_opponent_model_is_frequency_linear():
    """Without an explicit model, the negotiator uses the linear-additive
    frequency learner (the paper's Bayesian assumption)."""
    ntft = NiceTitForTatNegotiator(name="ntft")
    assert isinstance(ntft.opponent_model, FrequencyLinearUFunModel)


def test_can_pass_opponent_model_type():
    """An opponent-model *type* can be passed and is instantiated as the default."""
    ntft = NiceTitForTatNegotiator(name="ntft", opponent_model_type=FrequencyUFunModel)
    assert isinstance(ntft.opponent_model, FrequencyUFunModel)


def test_can_pass_opponent_model_instance():
    """A ready opponent-model instance can be passed and is exposed verbatim."""
    issues = [make_issue(6, "a"), make_issue(6, "b")]
    agent_ufun, opp_ufun = _zerosum_ufuns(issues)
    model = PeekingOpponentModel(ufun=opp_ufun)
    ntft = NiceTitForTatNegotiator(name="ntft", opponent_model=model)
    ntft.ufun = agent_ufun
    assert ntft.opponent_model is model


def test_concedes_as_opponent_concedes_with_oracle_model():
    """With a *correct* (oracle) opponent model, when the opponent concedes
    (offers the agent increasingly better outcomes), the Nice Tit for Tat
    bidding strategy reciprocates by *lowering its demanded utility* — i.e. it
    concedes toward the Nash point rather than staying put."""
    issues = [make_issue(10, "a"), make_issue(10, "b")]
    agent_ufun, opp_ufun = _zerosum_ufuns(issues)
    model = PeekingOpponentModel(ufun=opp_ufun)
    ntft = NiceTitForTatNegotiator(name="ntft", opponent_model=model, target="nash")
    ntft.ufun = agent_ufun  # wires on_preferences_changed to components
    offering = ntft._offering
    assert offering is not None

    # Outcomes sorted by the agent's utility (ascending). An opponent that
    # "concedes" walks down this list, offering the agent ever better outcomes.
    outcomes = list(agent_ufun.outcome_space.enumerate_or_sample(max_cardinality=10000))  # type: ignore[union-attr]
    outcomes.sort(key=lambda o: float(agent_ufun.eval_normalized(o)))
    n = 10
    seq = [outcomes[int(i * (len(outcomes) - 1) / (n - 1))] for i in range(n)]

    targets = []
    proposals = []
    for i, offer in enumerate(seq):
        state = SimpleNamespace(step=i, relative_time=i / (n - 1))
        offering.before_responding(state, offer)
        proposal = offering(state)
        assert proposal is not None, f"refused to propose at step {i}"
        targets.append(offering._target_util)
        proposals.append(float(agent_ufun.eval_normalized(proposal)))

    # The demanded utility strictly drops: the first opponent offer (agent's
    # worst = opponent's best) leaves the agent at its ideal, while the last
    # opponent offer (agent's best = opponent's worst) pulls the agent down to
    # the Nash point.
    assert targets[-1] < targets[0] - 1e-6
    # Monotonic concession (the Nash-aimed target only drops as the opponent's
    # remaining gap to Nash shrinks, for a correct model).
    for a, b in zip(targets, targets[1:]):
        assert b <= a + 1e-9
    # The agent's actual proposals stay within their demanded band and never
    # fall below the Nash target.
    for t, u in zip(targets, proposals):
        assert u >= t - 1e-6


def test_accombi_acceptance():
    """ACcombi accepts the opponent's offer when it beats our next planned
    offer, rejects when a gap remains early, and accepts near the deadline
    (ACtime)."""
    issues = [make_issue(10, "a"), make_issue(10, "b")]
    agent_ufun, opp_ufun = _zerosum_ufuns(issues)
    model = PeekingOpponentModel(ufun=opp_ufun)
    ntft = NiceTitForTatNegotiator(name="ntft", opponent_model=model)
    ntft.ufun = agent_ufun
    offering = ntft._offering
    acceptance = ntft._acceptance
    assert offering is not None
    assert isinstance(acceptance, ACCombi)

    os_ = agent_ufun.outcome_space
    best = agent_ufun.best(os_)
    worst = agent_ufun.worst(os_)

    # The agent's best outcome is always at least as good as our next planned
    # offer -> accept (ACnext).
    s_best = SimpleNamespace(step=0, relative_time=0.0)
    offering.before_responding(s_best, best)
    assert acceptance(s_best, best, "opp") == ResponseType.ACCEPT_OFFER

    # The agent's worst outcome, offered early with a gap remaining -> reject.
    s_worst = SimpleNamespace(step=10, relative_time=0.0)
    offering.before_responding(s_worst, worst)
    assert acceptance(s_worst, worst, "opp") == ResponseType.REJECT_OFFER

    # Near the deadline, ACtime accepts even the worst offer.
    s_late = SimpleNamespace(step=100, relative_time=0.999)
    assert acceptance(s_late, worst, "opp") == ResponseType.ACCEPT_OFFER


@pytest.mark.parametrize(
    "target",
    ["nash", "kalai", "kalai_smorodinsky", "max_welfare", "max_relative_welfare"],
)
def test_bargaining_targets_run(target):
    """Every supported bargaining target runs end-to-end without error."""
    issues = [make_issue(8, "a"), make_issue(8, "b")]
    agent_ufun, opp_ufun = _random_ufuns(issues)
    p = SAOMechanism(issues=issues, n_steps=30)
    p.add(NiceTitForTatNegotiator(name="ntft", target=target), preferences=agent_ufun)
    p.add(AspirationNegotiator(name="asp"), preferences=opp_ufun)
    p.run()
    assert len(p.history) > 0


@pytest.mark.parametrize(
    "target",
    ["nash", "kalai", "kalai_smorodinsky", "max_welfare", "max_relative_welfare"],
)
def test_target_point_sampler_matches_pareto_frontier(target):
    """The Pareto frontier feeding the bargaining-point calculators is now
    sourced from the shared `ParetoSampler` (so the frontier is built once per
    round instead of twice). On a finite space this must give the *identical*
    target point as the exact `pareto_frontier` fallback path — the sampler's
    reserve-filtered frontier equals `pareto_frontier`'s rational frontier.

    Uses a non-trivial reserve on both sides so the reserve filtering is
    actually exercised.
    """
    issues = [make_issue(6, "a"), make_issue(6, "b"), make_issue(4, "c")]
    make_os(issues)
    u1 = LUFun(
        values=[IdentityFun(), IdentityFun(), IdentityFun()],
        issues=issues,
        weights=[0.4, 0.4, 0.2],
    ).normalize()
    u2 = LUFun(
        values=[AffineFun(-1, bias=5), AffineFun(-1, bias=5), IdentityFun()],
        issues=issues,
        weights=[0.4, 0.4, 0.2],
    ).normalize()
    u1.reserved_value = 0.35
    u2.reserved_value = 0.25

    # sampler-sourced target point (the default path)
    a = NiceTitForTatNegotiator(
        name="s", opponent_model=PeekingOpponentModel(ufun=u2), target=target
    )
    a.ufun = u1
    tp_sampler = a._offering._target_point(u1, a.opponent_model)

    # exact-enumeration fallback (force the sampler off)
    b = NiceTitForTatNegotiator(
        name="f", opponent_model=PeekingOpponentModel(ufun=u2), target=target
    )
    b.ufun = u1
    b._offering._sampler_failed = True
    tp_fallback = b._offering._target_point(u1, b.opponent_model)

    assert tp_sampler is not None and tp_fallback is not None
    for x, y in zip(tp_sampler, tp_fallback):
        assert abs(x - y) < 1e-9, f"{target}: {tp_sampler} != {tp_fallback}"


def _conflict_scenario():
    """A scenario with genuine tension (an interior Nash point).

    Two issues conflict (agent wants high values, opponent wants low) and one is
    aligned, so the Nash bargaining point is a win-win *interior* outcome — not
    either agent's ideal — which lets us check that the agent actually concedes
    toward it. Both ufuns are normalized to ``[0, 1]`` with a reserve of 0.3.
    """
    issues = [make_issue(7, "a"), make_issue(7, "b"), make_issue(5, "c")]
    os_ = make_os(issues)
    u1 = LUFun(
        values=[IdentityFun(), IdentityFun(), IdentityFun()],
        issues=issues,
        weights=[0.4, 0.4, 0.2],
        reserved_value=0.0,
    ).normalize()
    u2 = LUFun(
        values=[AffineFun(-1, bias=6), AffineFun(-1, bias=6), IdentityFun()],
        issues=issues,
        weights=[0.4, 0.4, 0.2],
        reserved_value=0.0,
    ).normalize()
    u1.reserved_value = 0.3
    u2.reserved_value = 0.3
    return os_, issues, u1, u2


@pytest.mark.parametrize("opp_cls", [AspirationNegotiator, BoulwareTBNegotiator])
def test_default_agent_reaches_nash_against_conceding_opponent(opp_cls):
    """End-to-end: the *default* Nice Tit for Tat agent (learned
    `FrequencyLinearUFunModel` + `BruteForceParetoSampler`) reaches a rational,
    near-Nash agreement against a conceding opponent.

    This exercises the whole pipeline the unit tests do not: real concession, the
    `ParetoSampler` step-iv path with the default (non-additive, enumerable)
    learned model, and the ACcombi acceptance condition.
    """
    os_, issues, u1, u2 = _conflict_scenario()

    fr, _ = pareto_frontier([u1, u2], issues=issues)
    nash = nash_points([u1, u2], fr, outcome_space=os_)
    assert nash, "scenario should have a Nash point"
    (nash_u1, nash_u2), _ = nash[0]

    m = SAOMechanism(outcome_space=os_, n_steps=100)
    m.add(NiceTitForTatNegotiator(name="nice"), ufun=u1)  # all-default config
    m.add(opp_cls(name="opp"), ufun=u2)
    m.run()

    agreement = m.agreement
    assert agreement is not None, "should reach an agreement with a conceding opponent"
    ua, ub = float(u1(agreement)), float(u2(agreement))
    # rational for both: at or above each reserved value
    assert ua >= u1.reserved_value - 1e-6
    assert ub >= u2.reserved_value - 1e-6
    # near Nash (a win-win), not at the agent's own extreme
    assert abs(ua - nash_u1) <= 0.15, (
        f"agent utility {ua:.3f} far from Nash {nash_u1:.3f}"
    )
    assert abs(ub - nash_u2) <= 0.15, (
        f"opp utility {ub:.3f} far from Nash {nash_u2:.3f}"
    )


def _clean_conflict_scenario():
    """A conflict scenario normalized to ``[0, 1]`` with reserve ``0`` (so the
    Nash point sits at the symmetric interior). Two issues conflict and one is
    aligned; the Nash bargaining point is ``(0.571, 0.571)``."""
    issues = [make_issue(7, "a"), make_issue(7, "b"), make_issue(5, "c")]
    os_ = make_os(issues)
    u1 = LUFun(
        values=[IdentityFun(), IdentityFun(), IdentityFun()],
        issues=issues,
        weights=[0.4, 0.4, 0.2],
        reserved_value=0.0,
    ).scale_max(1.0)
    u2 = LUFun(
        values=[AffineFun(-1, bias=6), AffineFun(-1, bias=6), IdentityFun()],
        issues=issues,
        weights=[0.4, 0.4, 0.2],
        reserved_value=0.0,
    ).scale_max(1.0)
    return os_, issues, u1, u2


def test_mirror_match_concedes_and_reaches_agreement():
    """Regression: two *default* Nice Tit for Tat agents must concede over time
    and reach a (near-Nash) agreement rather than deadlocking at maximum utility
    until the deadline. This guards the concession bonus that fixes the mutual
    non-concession deadlock.
    """
    os_, issues, u1, u2 = _clean_conflict_scenario()
    fr, _ = pareto_frontier([u1, u2], issues=issues)
    (nash_u1, nash_u2), _ = nash_points([u1, u2], fr, outcome_space=os_)[0]

    m = SAOMechanism(outcome_space=os_, n_steps=80)
    m.add(NiceTitForTatNegotiator(name="A"), ufun=u1)
    m.add(NiceTitForTatNegotiator(name="B"), ufun=u2)
    m.run()

    assert m.agreement is not None, "mirror match must reach an agreement"
    ua, ub = float(u1(m.agreement)), float(u2(m.agreement))
    assert ua >= u1.reserved_value - 1e-6 and ub >= u2.reserved_value - 1e-6
    # a mirror of a Nash-aiming agent should land near the Nash point
    assert abs(ua - nash_u1) <= 0.2 and abs(ub - nash_u2) <= 0.2, (
        f"agreement ({ua:.3f}, {ub:.3f}) far from Nash ({nash_u1:.3f}, {nash_u2:.3f})"
    )


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
