"""Behavioral / scenario tests for the Nice Tit for Tat negotiator.

These correspond to the reference scripts under ``coding_agents/``:

* ``compare_ntft_genius.py``  -> ``test_matches_genius_on_cameradomain``
* ``ntft_vs_timebased.py``    -> ``test_vs_time_based_opponents``
* ``ntft_oracle_mirror.py``   -> ``test_oracle_mirror_concedes_and_agrees``

The Genius comparison is the important one: our `NiceTitForTatNegotiator` should
reach essentially the same agreement as the reference Genius ``NiceTitForTat``
(Baarslag) on the same scenario. It is skipped automatically when the Genius
bridge is not running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import negmas
from negmas.genius import genius_bridge_is_running
from negmas.gb.components.models.ufun import PeekingOpponentModel
from negmas.gb.negotiators.titfortat import NiceTitForTatNegotiator
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.discounted import DiscountedUtilityFunction
from negmas.preferences.ops import nash_points, pareto_frontier
from negmas.preferences.value_fun import AffineFun, IdentityFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
)

CAMERA = (
    Path(negmas.__file__).parent.parent.parent
    / "tests/data/scenarios/anac/y2013/cameradomain"
)


def _strip(u):
    while isinstance(u, DiscountedUtilityFunction):
        u = u.ufun
    return u


def _clean_scenario():
    """Conflict scenario normalized to ``[0, 1]`` with reserve ``0`` (Nash at the
    symmetric interior point ``(0.571, 0.571)``)."""
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
    return os_, u1, u2


# ---------------------------------------------------------------------------
# Genius comparison (the important one)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent comparison",
)
def test_matches_genius_on_cameradomain():
    """Our Nice Tit for Tat should reach essentially the same agreement as the
    reference Genius ``NiceTitForTat`` in a mirror match on the cameradomain
    (discounting stripped)."""
    from negmas.genius.gnegotiators.y2011 import NiceTitForTat as GeniusNiceTFT

    def run(make_a, make_b):
        sc = Scenario.load(CAMERA)
        os_ = sc.outcome_space
        u1, u2 = [_strip(u) for u in sc.ufuns]
        n1, n2 = u1.normalize(), u2.normalize()
        m = SAOMechanism(outcome_space=os_, n_steps=60)
        m.add(make_a(), ufun=u1)
        m.add(make_b(), ufun=u2)
        m.run()
        if m.agreement is None:
            return None
        return float(n1(m.agreement)), float(n2(m.agreement))

    ours = run(
        lambda: NiceTitForTatNegotiator(name="ours-A"),
        lambda: NiceTitForTatNegotiator(name="ours-B"),
    )
    genius = run(
        lambda: GeniusNiceTFT(name="genius-A"), lambda: GeniusNiceTFT(name="genius-B")
    )

    assert ours is not None, "our Nice Tit for Tat mirror should reach agreement"
    assert genius is not None, "Genius Nice Tit for Tat mirror should reach agreement"
    # behaves similarly: the two agreements are close in normalized utility space
    assert abs(ours[0] - genius[0]) <= 0.2 and abs(ours[1] - genius[1]) <= 0.2, (
        f"ours {tuple(round(x, 3) for x in ours)} vs "
        f"genius {tuple(round(x, 3) for x in genius)}"
    )


# ---------------------------------------------------------------------------
# vs time-based opponents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opp_cls", [BoulwareTBNegotiator, LinearTBNegotiator, ConcederTBNegotiator]
)
def test_vs_time_based_reaches_rational_agreement(opp_cls):
    """Against any time-based opponent, Nice Tit for Tat reaches a rational
    agreement (never conceding below its reserved value)."""
    os_, u1, u2 = _clean_scenario()
    m = SAOMechanism(outcome_space=os_, n_steps=100)
    m.add(NiceTitForTatNegotiator(name="ntft"), ufun=u1)
    m.add(opp_cls(name="opp"), ufun=u2)
    m.run()
    assert m.agreement is not None
    assert float(u1(m.agreement)) >= u1.reserved_value - 1e-6
    assert float(u2(m.agreement)) >= u2.reserved_value - 1e-6


def test_reciprocates_toughness_exploits_pushover():
    """Tit-for-tat signature: Nice Tit for Tat gets at least as much against a
    fast conceder (pushover) as against a tough Boulware opponent — it accepts
    the conceder's over-generous offers rather than giving utility back."""

    def util_vs(opp_cls):
        os_, u1, u2 = _clean_scenario()
        m = SAOMechanism(outcome_space=os_, n_steps=100)
        m.add(NiceTitForTatNegotiator(name="ntft"), ufun=u1)
        m.add(opp_cls(name="opp"), ufun=u2)
        m.run()
        assert m.agreement is not None
        return float(u1(m.agreement))

    vs_conceder = util_vs(ConcederTBNegotiator)
    vs_boulware = util_vs(BoulwareTBNegotiator)
    assert vs_conceder >= vs_boulware - 1e-6, (
        f"expected to do at least as well vs Conceder ({vs_conceder:.3f}) "
        f"as vs Boulware ({vs_boulware:.3f})"
    )


# ---------------------------------------------------------------------------
# oracle (PeekingOpponentModel) mirror
# ---------------------------------------------------------------------------


def test_oracle_mirror_concedes_and_agrees():
    """Two Nice Tit for Tat agents, each with a `PeekingOpponentModel` that knows
    the opponent's exact ufun, reach a rational agreement, and at least one side
    concedes its offers toward its Nash utility (rather than both holding at the
    maximum)."""
    os_, u1, u2 = _clean_scenario()
    fr, _ = pareto_frontier([u1, u2], issues=os_.issues)
    (nash_u1, nash_u2), _ = nash_points([u1, u2], fr, outcome_space=os_)[0]

    a_offers: list[float] = []
    from negmas.gb.components.offering import NiceTitForTatOfferingPolicy as _P

    orig = _P.__call__

    def wrap(self, state, dest=None):
        o = orig(self, state, dest=dest)
        if self.negotiator and self.negotiator.name == "A" and o is not None:
            a_offers.append(float(u1(o)))
        return o

    _P.__call__ = wrap
    try:
        m = SAOMechanism(outcome_space=os_, n_steps=100)
        a = NiceTitForTatNegotiator(
            name="A", opponent_model=PeekingOpponentModel(ufun=u2)
        )
        b = NiceTitForTatNegotiator(
            name="B", opponent_model=PeekingOpponentModel(ufun=u1)
        )
        m.add(a, ufun=u1)
        m.add(b, ufun=u2)
        m.run()
    finally:
        _P.__call__ = orig

    assert m.agreement is not None, "oracle mirror should reach agreement"
    assert float(u1(m.agreement)) >= u1.reserved_value - 1e-6
    assert float(u2(m.agreement)) >= u2.reserved_value - 1e-6
    # A concedes over the negotiation (its offers are not flat at the maximum)
    assert a_offers and min(a_offers) < max(a_offers) - 1e-6, (
        "the conceding side should lower its demanded utility over time"
    )
    # a symmetric clean scenario: the agreement should be near the Nash point
    ua, ub = float(u1(m.agreement)), float(u2(m.agreement))
    assert abs(ua - nash_u1) <= 0.25 and abs(ub - nash_u2) <= 0.25


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
