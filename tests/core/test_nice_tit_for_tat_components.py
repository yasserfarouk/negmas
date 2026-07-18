"""Exhaustive unit tests for the ACcombi acceptance condition and the new
opponent models that back the Nice Tit for Tat agent.

These test the components *individually* (not through a full negotiation):

- `ACCombi`: ACnext (with ``a``/``b`` scaling, equality, ``None`` next offer),
  ACtime (deadline), and the ``None``-offer rejection path.
- `FrequencyLinearUFunModel`: neutral before observation, most-offered value
  scores higher, equal frequencies → equal scores, range ``[0, 1]``,
  concentration-based weighting, and continuous-issue discretization.
- `FrequencyUFunModel`: neutral before observation, exact-outcome frequency,
  unseen → ``0``, max → ``1``, ``None`` handling.
- `PeekingOpponentModel`: neutral without a ufun, exact delegation to the
  wrapped ufun, and late ``ufun`` assignment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from negmas.common import PreferencesChange
from negmas.gb.common import ResponseType
from negmas.gb.components.acceptance import ACCombi
from negmas.gb.components.models.ufun import (
    FrequencyLinearUFunModel,
    FrequencyUFunModel,
    PeekingOpponentModel,
)
from negmas.gb.negotiators.titfortat import NiceTitForTatNegotiator
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _zerosum_ufuns(issues):
    """Deterministic near-zero-sum pair: agent util = sum of issue values
    (normalized to (a+b)/18 for the 10x10 case), opponent util shrinks with them.
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


class _FakeOffering:
    """An offering policy whose ``propose`` always returns a fixed outcome."""

    def __init__(self, outcome, negotiator):
        self.outcome = outcome
        self.negotiator = negotiator

    def propose(self, state):
        return self.outcome


def _make_accombi(agent_ufun, opp_ufun, *, a=1.0, b=0.0, t=0.99, next_offer=None):
    """Build a wired ``ACCombi`` whose next planned offer is pinned to
    ``next_offer`` (so ACnext can be tested precisely)."""
    ntft = NiceTitForTatNegotiator(
        name="ntft", opponent_model=PeekingOpponentModel(ufun=opp_ufun), a=a, b=b, t=t
    )
    ntft.ufun = agent_ufun  # wires on_preferences_changed to components
    acc = ntft._acceptance
    assert isinstance(acc, ACCombi)
    acc.offering_strategy = _FakeOffering(next_offer, ntft)
    return acc, ntft


def _make_model(model_class, agent_ufun, **kwargs):
    """Build an opponent model wired to a negotiator (so
    ``on_preferences_changed`` sets up its issues).

    Note: assigning ``ntft.ufun`` directly does not broadcast
    ``on_preferences_changed`` to the components (the mechanism normally does
    that via ``join``), so we broadcast explicitly to wire the model.
    """
    model = model_class(**kwargs)
    ntft = NiceTitForTatNegotiator(name="ntft", opponent_model=model)
    ntft.ufun = agent_ufun
    ntft.on_preferences_changed([PreferencesChange()])
    return model, ntft


# --------------------------------------------------------------------------- #
# ACCombi
# --------------------------------------------------------------------------- #


@pytest.fixture
def disc_issues():
    return [make_issue(10, "a"), make_issue(10, "b")]


@pytest.fixture
def disc_ufuns(disc_issues):
    return _zerosum_ufuns(disc_issues)


def _util(agent_ufun, outcome):
    return float(agent_ufun.eval_normalized(outcome))


def test_accombi_acnext_accepts_offer_better_than_next(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    # next planned offer = (4,5) with util 0.5
    acc, _ = _make_accombi(au, ou, t=0.99, next_offer=(4, 5))
    state = SimpleNamespace(step=0, relative_time=0.0)
    # best (util 1.0) >= 0.5 -> accept
    assert acc(state, (9, 9), "opp") == ResponseType.ACCEPT_OFFER
    # worst (util 0.0) < 0.5 -> reject
    assert acc(state, (0, 0), "opp") == ResponseType.REJECT_OFFER


def test_accombi_acnext_accepts_on_equality(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    acc, _ = _make_accombi(au, ou, t=0.99, next_offer=(4, 5))
    state = SimpleNamespace(step=0, relative_time=0.0)
    # offer == next planned offer -> accept (>=)
    assert acc(state, (4, 5), "opp") == ResponseType.ACCEPT_OFFER


def test_accombi_actime_accepts_anything_past_threshold(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    # tough next offer (the best), threshold t=0.5
    acc, _ = _make_accombi(au, ou, t=0.5, next_offer=(9, 9))
    before = SimpleNamespace(step=0, relative_time=0.49)
    at = SimpleNamespace(step=10, relative_time=0.5)
    after = SimpleNamespace(step=20, relative_time=0.9)
    # before the deadline: worst < best(next) -> reject (ACnext fails, ACtime not yet)
    assert acc(before, (0, 0), "opp") == ResponseType.REJECT_OFFER
    # at/after the deadline: ACtime accepts even the worst
    assert acc(at, (0, 0), "opp") == ResponseType.ACCEPT_OFFER
    assert acc(after, (0, 0), "opp") == ResponseType.ACCEPT_OFFER
    # and the best too
    assert acc(after, (9, 9), "opp") == ResponseType.ACCEPT_OFFER


def test_accombi_none_offer_rejected_even_past_deadline(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    acc, _ = _make_accombi(au, ou, t=0.0, next_offer=(9, 9))
    state = SimpleNamespace(step=0, relative_time=1.0)  # well past any threshold
    # the None-offer guard precedes ACtime -> always reject None
    assert acc(state, None, "opp") == ResponseType.REJECT_OFFER


def test_accombi_none_next_offer_accepts_rational(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    acc, _ = _make_accombi(au, ou, t=0.99, next_offer=None)
    state = SimpleNamespace(step=0, relative_time=0.0)
    # next planned offer is None -> accept iff u(offer) >= reserved_value (0.0)
    assert _util(au, (0, 0)) == pytest.approx(0.0)
    assert acc(state, (0, 0), "opp") == ResponseType.ACCEPT_OFFER
    assert acc(state, (9, 9), "opp") == ResponseType.ACCEPT_OFFER


def test_accombi_scaling_a_makes_acceptance_easier(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    # next = (4,5) util 0.5; offer = (3,3) util 1/3 ~ 0.333
    offer = (3, 3)
    assert _util(au, offer) == pytest.approx(6 / 18)
    # a=1: 0.333 < 0.5 -> reject
    acc1, _ = _make_accombi(au, ou, a=1.0, t=0.99, next_offer=(4, 5))
    state = SimpleNamespace(step=0, relative_time=0.0)
    assert acc1(state, offer, "opp") == ResponseType.REJECT_OFFER
    # a=2: 2*0.333 = 0.667 >= 0.5 -> accept
    acc2, _ = _make_accombi(au, ou, a=2.0, t=0.99, next_offer=(4, 5))
    assert acc2(state, offer, "opp") == ResponseType.ACCEPT_OFFER


def test_accombi_offset_b_makes_acceptance_easier(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    offer = (3, 3)
    # a=1, b=0 -> reject (0.333 < 0.5)
    acc0, _ = _make_accombi(au, ou, a=1.0, b=0.0, t=0.99, next_offer=(4, 5))
    state = SimpleNamespace(step=0, relative_time=0.0)
    assert acc0(state, offer, "opp") == ResponseType.REJECT_OFFER
    # a=1, b=0.3 -> 0.333 + 0.3 = 0.633 >= 0.5 -> accept
    accb, _ = _make_accombi(au, ou, a=1.0, b=0.3, t=0.99, next_offer=(4, 5))
    assert accb(state, offer, "opp") == ResponseType.ACCEPT_OFFER


def test_accombi_negative_b_makes_acceptance_harder(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    # next = (3,3) util 0.333; offer = (4,5) util 0.5
    offer = (4, 5)
    assert _util(au, offer) == pytest.approx(0.5)
    # a=1, b=0 -> 0.5 >= 0.333 -> accept
    acc0, _ = _make_accombi(au, ou, a=1.0, b=0.0, t=0.99, next_offer=(3, 3))
    state = SimpleNamespace(step=0, relative_time=0.0)
    assert acc0(state, offer, "opp") == ResponseType.ACCEPT_OFFER
    # a=1, b=-0.3 -> 0.5 - 0.3 = 0.2 < 0.333 -> reject
    accn, _ = _make_accombi(au, ou, a=1.0, b=-0.3, t=0.99, next_offer=(3, 3))
    assert accn(state, offer, "opp") == ResponseType.REJECT_OFFER


def test_accombi_rejects_when_no_ufun(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    # build then clear preferences so negotiator.ufun is None
    acc, ntft = _make_accombi(au, ou, t=0.0, next_offer=(9, 9))
    ntft._preferences = None  # type: ignore[attr-defined]
    state = SimpleNamespace(step=0, relative_time=1.0)
    assert acc(state, (9, 9), "opp") == ResponseType.REJECT_OFFER


# --------------------------------------------------------------------------- #
# FrequencyLinearUFunModel
# --------------------------------------------------------------------------- #


def _state():
    return SimpleNamespace(step=0)


def test_freq_linear_neutral_before_observation(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyLinearUFunModel, au)
    for o in [(0, 0), (9, 9), (4, 5)]:
        assert model.eval(o) == 0.5
        assert model.eval_normalized(o) == 0.5
    assert model.eval_normalized(None) == 0.0


def test_freq_linear_most_offered_value_scores_higher(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyLinearUFunModel, au)
    # opponent offers a=9 nine times, a=0 once
    for _i in range(9):
        model.before_responding(_state(), (9, 5))
    model.before_responding(_state(), (0, 5))
    # holding b=5, the outcome with a=9 should outrank a=0
    assert model.eval((9, 5)) > model.eval((0, 5))


def test_freq_linear_range_is_unit_interval(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyLinearUFunModel, au)
    offers = [(0, 0), (9, 9), (3, 7), (1, 8), (5, 5), (2, 4), (6, 1)]
    for o in offers:
        model.before_responding(_state(), o)
    os_ = au.outcome_space
    for o in os_.enumerate():
        v = model.eval(o)
        assert 0.0 <= v <= 1.0


def test_freq_linear_equal_frequencies_yield_equal_scores(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyLinearUFunModel, au)
    # offer each a-value exactly once, b fixed at 5 -> equal a-value scores
    for a in range(10):
        model.before_responding(_state(), (a, 5))
    scores = [model.eval((a, 5)) for a in range(10)]
    assert max(scores) == pytest.approx(min(scores))


def test_freq_linear_concentration_increases_weight(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyLinearUFunModel, au)
    # issue a: always 9 (fully concentrated); issue b: spread across all values
    for a in range(10):
        for _i in range(3):
            model.before_responding(_state(), (9, a))
    # weight of 'a' (concentrated) > weight of 'b' (spread)
    assert model._weights["a"] > model._weights["b"]
    # both weights in [0, 1]
    assert 0.0 <= model._weights["a"] <= 1.0
    assert 0.0 <= model._weights["b"] <= 1.0


def test_freq_linear_continuous_issues_are_discretized():
    issues = [make_issue((0.0, 1.0), "x"), make_issue((0.0, 1.0), "y")]
    au = LUFun(
        values=[IdentityFun() for _ in issues], issues=issues, reserved_value=0.0
    )
    model, _ = _make_model(FrequencyLinearUFunModel, au, levels=5)
    # continuous issues are discretized to a finite grid; model must not crash
    # and must return values in [0, 1].
    for _i in range(5):
        model.before_responding(_state(), (0.05, 0.5))
    for _i in range(2):
        model.before_responding(_state(), (0.95, 0.5))
    v_lo = model.eval((0.05, 0.5))
    v_hi = model.eval((0.95, 0.5))
    assert 0.0 <= v_lo <= 1.0
    assert 0.0 <= v_hi <= 1.0
    # the region offered more often (near 0.0) outscores the rare region (near 1.0)
    assert v_lo > v_hi


# --------------------------------------------------------------------------- #
# FrequencyUFunModel
# --------------------------------------------------------------------------- #


def test_freq_ufun_neutral_before_observation(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyUFunModel, au)
    assert model.eval((9, 9)) == 0.5
    assert model.eval((0, 0)) == 0.5
    assert model.eval_normalized(None) == 0.0


def test_freq_ufun_exact_outcome_frequency(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyUFunModel, au)
    for _i in range(5):
        model.before_responding(_state(), (9, 9))
    model.before_responding(_state(), (0, 0))
    assert model.eval((9, 9)) == pytest.approx(1.0)  # max frequency
    assert model.eval((0, 0)) == pytest.approx(1 / 5)
    assert model.eval((5, 5)) == pytest.approx(0.0)  # never offered
    assert model.eval((9, 9)) > model.eval((0, 0)) > model.eval((5, 5))


def test_freq_ufun_eval_none_returns_neutral(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyUFunModel, au)
    model.before_responding(_state(), (9, 9))
    # eval(None) hits the neutral branch
    assert model.eval(None) == 0.5
    assert model.eval_normalized(None) == 0.0


def test_freq_ufun_continuous_outcomes_bucketed():
    issues = [make_issue((0.0, 1.0), "x")]
    au = LUFun(values=[IdentityFun()], issues=issues, reserved_value=0.0)
    model, _ = _make_model(FrequencyUFunModel, au, levels=10)
    # two nearby continuous offers should fall in the same discretization bucket
    # and accumulate frequency together, beating a far offer.
    for _i in range(4):
        model.before_responding(_state(), (0.05,))
    model.before_responding(_state(), (0.06,))
    model.before_responding(_state(), (0.95,))
    assert model.eval((0.05,)) >= model.eval((0.95,))


def test_freq_ufun_range_is_unit_interval(disc_issues, disc_ufuns):
    au, _ = disc_ufuns
    model, _ = _make_model(FrequencyUFunModel, au)
    for o in [(0, 0), (9, 9), (3, 7), (1, 8)]:
        model.before_responding(_state(), o)
    for o in au.outcome_space.enumerate():
        assert 0.0 <= model.eval(o) <= 1.0


# --------------------------------------------------------------------------- #
# PeekingOpponentModel
# --------------------------------------------------------------------------- #


def test_peeking_neutral_without_ufun(disc_issues, disc_ufuns):
    _, ou = disc_ufuns
    m = PeekingOpponentModel()
    assert m.eval((9, 9)) == 0.5
    assert m.eval_normalized((9, 9)) == 0.5
    assert m.eval_normalized(None) == 0.0


def test_peeking_delegates_to_wrapped_ufun(disc_issues, disc_ufuns):
    _, ou = disc_ufuns
    m = PeekingOpponentModel(ufun=ou)
    for o in [(0, 0), (9, 9), (4, 5), (3, 3)]:
        assert m.eval(o) == pytest.approx(float(ou.eval_normalized(o)))
        assert m.eval_normalized(o) == pytest.approx(float(ou.eval_normalized(o)))


def test_peeking_forwards_above_reserve_and_expected_limits(disc_issues, disc_ufuns):
    _, ou = disc_ufuns
    m = PeekingOpponentModel(ufun=ou)
    o = (4, 5)
    assert m.eval_normalized(o, above_reserve=False) == pytest.approx(
        float(ou.eval_normalized(o, False, True))
    )
    assert m.eval_normalized(
        o, above_reserve=True, expected_limits=False
    ) == pytest.approx(float(ou.eval_normalized(o, True, False)))


def test_peeking_supports_late_ufun_assignment(disc_issues, disc_ufuns):
    _, ou = disc_ufuns
    m = PeekingOpponentModel()
    assert m.eval((4, 5)) == 0.5
    m.ufun = ou
    assert m.eval((4, 5)) == pytest.approx(float(ou.eval_normalized((4, 5))))


def test_peeking_none_offer(disc_issues, disc_ufuns):
    _, ou = disc_ufuns
    m = PeekingOpponentModel(ufun=ou)
    assert m.eval_normalized(None) == 0.0


# --------------------------------------------------------------------------- #
# wiring: opponent model is exposed as the negotiator's opponent_model /
# private_info, regardless of how it was provided.
# --------------------------------------------------------------------------- #


def test_negotiator_exposes_model_and_private_info(disc_issues, disc_ufuns):
    au, ou = disc_ufuns
    model = PeekingOpponentModel(ufun=ou)
    ntft = NiceTitForTatNegotiator(name="ntft", opponent_model=model)
    ntft.ufun = au
    ntft.on_preferences_changed([PreferencesChange()])
    assert ntft.opponent_model is model
    # The model is registered as the negotiator's opponent model and, once the
    # mechanism wires private_info (via join), as ``opponent_ufun``; here we
    # just check the model is reachable through the negotiator.
    assert ntft.opponent_model.ufun is ou


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
