"""Tests for the opponent models recovered from Baarslag et al. (2016), *Learning
about the opponent in automated bilateral negotiation* (JAAMAS 30:849-898).

Covers the models added to fill the gaps in the survey's Table-2 taxonomy:

- §5.3.1 issue-weight models: `ConcessionRatioUFunModel`, `ValueDifferenceUFunModel`,
  `KDEWeightUFunModel`.
- §5.3.2 classification: `LuceProfileClassifierModel`.
- §5.3.4 heuristics: `CandidateEliminationModel`.
- §5.1.1 reservation value: `ConcessionExtrapolatingReservationModel`,
  `BayesianReservationValueModel`.
- §5.4 offering strategy: `PolynomialOfferingModel`, `DerivativeOfferingModel`,
  `MarkovChainOfferingModel`.

*AI Generated tests.*
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from negmas.gb.components.models import (
    CandidateEliminationModel,
    ConcessionRatioUFunModel,
    KDEWeightUFunModel,
    LuceProfileClassifierModel,
    ValueDifferenceUFunModel,
)
from negmas.gb.negotiators.titfortat import NiceTitForTatNegotiator
from negmas.models import (
    BayesianReservationValueModel,
    ConcessionExtrapolatingReservationModel,
    DerivativeOfferingModel,
    MarkovChainOfferingModel,
    PolynomialOfferingModel,
)
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import BoulwareTBNegotiator

WEIGHT_MODELS = [ConcessionRatioUFunModel, ValueDifferenceUFunModel, KDEWeightUFunModel]


def _issues():
    return [make_issue(6, "a"), make_issue(6, "b")]


def _agent_ufun(issues):
    return LUFun(
        values=[IdentityFun() for _ in issues], issues=issues, reserved_value=0.0
    )


def _wire(model, issues):
    """Attach ``model`` to a negotiator, set its ufun, and set the model up."""
    ntft = NiceTitForTatNegotiator(name="h", opponent_model=model)
    ntft.ufun = _agent_ufun(issues)
    model.on_preferences_changed([])
    return model


def _feed(model, offers):
    for i, offer in enumerate(offers):
        state = SimpleNamespace(step=i, relative_time=i / max(1, len(offers) - 1))
        model.before_responding(state, offer)


# --------------------------------------------------------------------------- #
# §5.3.1 issue-weight models
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("Model", WEIGHT_MODELS)
def test_weight_model_learns_held_issue_is_important(Model):
    """An issue the opponent holds constant while conceding on the other gets a
    higher learned weight, and its held value gets a higher estimated utility."""
    issues = _issues()
    model = _wire(Model(), issues)
    # Opponent holds issue 'a' at 4 and concedes across issue 'b'.
    _feed(model, [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)])
    assert model._weights["a"] > model._weights["b"]
    # The held value (a=4) is estimated as strongly preferred by the opponent.
    assert float(model.eval((4, 0))) > float(model.eval((0, 0)))


@pytest.mark.parametrize("Model", WEIGHT_MODELS)
def test_weight_model_eval_is_in_unit_range(Model):
    issues = _issues()
    model = _wire(Model(), issues)
    _feed(model, [(1, 5), (2, 4), (3, 3), (4, 2)])
    for outcome in [(0, 0), (5, 5), (3, 3), (4, 2)]:
        v = float(model.eval(outcome))
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("Model", WEIGHT_MODELS)
def test_weight_model_neutral_before_any_offer(Model):
    issues = _issues()
    model = _wire(Model(), issues)
    assert float(model.eval((3, 3))) == pytest.approx(0.5)


def test_value_difference_uses_magnitude_not_just_change():
    """Carbonneau & Vahidov: a big jump concedes more than a small one, so an
    issue with larger normalized moves ends up with a smaller weight."""
    issues = _issues()
    model = _wire(ValueDifferenceUFunModel(), issues)
    # 'a' moves by 1 each step; 'b' moves by 5 each step (bigger concession).
    _feed(model, [(0, 0), (1, 5), (2, 0), (3, 5), (4, 0)])
    assert model._weights["a"] > model._weights["b"]


# --------------------------------------------------------------------------- #
# §5.3.2 classification (Luce)
# --------------------------------------------------------------------------- #


def test_luce_classifier_identifies_the_right_profile():
    """Given two candidate profiles, offers consistent with one drive the
    posterior toward it and make ``eval`` track it."""
    issues = _issues()
    # Profile A prefers HIGH 'a'; profile B prefers LOW 'a'.
    prof_high = LUFun(
        values=[IdentityFun(), IdentityFun()], issues=issues, reserved_value=0.0
    )
    prof_low = LUFun(
        values=[AffineFun(-1, bias=5), IdentityFun()], issues=issues, reserved_value=0.0
    )
    model = LuceProfileClassifierModel(profiles=[prof_high, prof_low])
    _wire(model, issues)
    # Opponent keeps offering HIGH 'a' outcomes → consistent with prof_high.
    _feed(model, [(5, 3), (5, 4), (4, 5), (5, 5), (4, 4)])
    post = model._posterior()
    assert post[0] > post[1]  # prof_high wins
    # eval tracks the favored profile: it prefers high 'a'.
    assert float(model.eval((5, 3))) > float(model.eval((0, 3)))


def test_luce_classifier_neutral_without_profiles():
    issues = _issues()
    model = LuceProfileClassifierModel(profiles=[])
    _wire(model, issues)
    assert float(model.eval((3, 3))) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# §5.3.4 candidate elimination
# --------------------------------------------------------------------------- #


def test_candidate_elimination_ranks_seen_above_rejected():
    issues = _issues()
    model = _wire(CandidateEliminationModel(), issues)
    # Opponent offers (positives) always keep a=5.
    _feed(model, [(5, 0), (5, 1), (5, 2)])
    # A value only ever in a rejected offer is a negative instance.
    model.note_rejected((0, 0))
    # a=5 seen (acceptable) scores above a=0 (rejected).
    assert float(model.eval((5, 4))) > float(model.eval((0, 4)))
    # Unseen values default to the neutral 0.5 general boundary.
    assert 0.0 <= float(model.eval((2, 2))) <= 1.0


def test_candidate_elimination_neutral_before_evidence():
    issues = _issues()
    model = _wire(CandidateEliminationModel(), issues)
    assert float(model.eval((3, 3))) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# §5.1.1 reservation value
# --------------------------------------------------------------------------- #


def test_concession_extrapolating_reservation():
    model = ConcessionExtrapolatingReservationModel(min_observations=3)
    # Linear concession: u = 1 - 0.5 t  → at deadline t=1, u = 0.5.
    for t in (0.0, 0.2, 0.4, 0.6):
        model.update(t, 1.0 - 0.5 * t)
    assert model.predict_reservation() == pytest.approx(0.5, abs=0.05)


def test_concession_extrapolating_reservation_needs_data():
    model = ConcessionExtrapolatingReservationModel(min_observations=3)
    model.update(0.0, 1.0)
    import math

    assert math.isnan(model.predict_reservation())


def test_bayesian_reservation_value():
    model = BayesianReservationValueModel(candidates=21, sigma=0.1)
    # Concession consistent with reservation value 0.3: u = 0.3 + 0.7 (1 - t).
    for t in (0.0, 0.25, 0.5, 0.75, 0.9):
        model.update(t, 0.3 + 0.7 * (1.0 - t))
    assert model.predict_reservation() == pytest.approx(0.3, abs=0.1)


# --------------------------------------------------------------------------- #
# §5.4 offering-strategy forecasters
# --------------------------------------------------------------------------- #


def test_polynomial_offering_model_fits_curve():
    model = PolynomialOfferingModel(degree=2)
    for t in (0.0, 0.2, 0.4, 0.6, 0.8):
        model.update(t, 1.0 - t * t)  # u = 1 - t^2
    assert model.predict_utility(0.5) == pytest.approx(0.75, abs=0.05)


def test_polynomial_offering_model_fallback():
    model = PolynomialOfferingModel(degree=3)
    model.update(0.1, 0.9)
    assert model.predict_utility(0.5) == pytest.approx(0.9)  # last observed


def test_derivative_offering_model_extrapolates_and_time_influence():
    model = DerivativeOfferingModel(min_observations=3)
    for t in (0.0, 0.2, 0.4):
        model.update(t, 1.0 - 0.5 * t)  # linear decline
    # Continues downward: at t=0.6, u ≈ 0.7.
    assert model.predict_utility(0.6) == pytest.approx(0.7, abs=0.05)
    # A perfectly monotone concession is fully time-consistent.
    assert model.time_influence() == pytest.approx(1.0)


def test_markov_chain_offering_model():
    model = MarkovChainOfferingModel(n_states=10)
    for i, u in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]):
        model.update(i / 5.0, u)
    nxt = model.predict_next_utility()
    assert 0.0 <= nxt <= 1.0
    fut = model.predict_utility(1.5)
    assert 0.0 <= fut <= 1.0


# --------------------------------------------------------------------------- #
# integration + exposure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("Model", WEIGHT_MODELS + [CandidateEliminationModel])
def test_model_runs_in_a_real_negotiation(Model):
    issues = _issues()
    agent_ufun = _agent_ufun(issues)
    opp_ufun = LUFun(
        values=[AffineFun(-1, bias=5) for _ in issues],
        issues=issues,
        reserved_value=0.0,
    )
    model = Model()
    p = SAOMechanism(issues=issues, n_steps=30)
    p.add(
        NiceTitForTatNegotiator(name="ntft", opponent_model=model),
        preferences=agent_ufun,
    )
    p.add(BoulwareTBNegotiator(name="opp"), preferences=opp_ufun)
    p.run()
    assert model._issues is not None  # on_preferences_changed fired in the run
    assert 0.0 <= float(model.eval((3, 3))) <= 1.0


def test_new_models_exposed_at_top_level():
    import negmas

    for name in [
        "ConcessionRatioUFunModel",
        "ValueDifferenceUFunModel",
        "KDEWeightUFunModel",
        "LuceProfileClassifierModel",
        "CandidateEliminationModel",
    ]:
        assert hasattr(negmas, name), name


def test_new_ufun_models_registered_as_model_type():
    from negmas import component_registry

    for name in [
        "ConcessionRatioUFunModel",
        "ValueDifferenceUFunModel",
        "KDEWeightUFunModel",
        "LuceProfileClassifierModel",
        "CandidateEliminationModel",
    ]:
        cls = component_registry.get_class(name)
        assert cls is not None, name
        assert component_registry.get_by_class(cls).component_type == "model"


def test_new_component_types_registered():
    from negmas import component_registry

    expected = {
        "reservation-model": {
            "ConcessionExtrapolatingReservationModel",
            "BayesianReservationValueModel",
        },
        "deadline-model": {"ConcessionExtrapolatingDeadlineModel"},
        "offering-model": {
            "TimeSeriesOfferingModel",
            "PolynomialOfferingModel",
            "DerivativeOfferingModel",
            "MarkovChainOfferingModel",
        },
    }
    for ctype, names in expected.items():
        registered = {
            info.short_name
            for info in component_registry.query(component_type=ctype).values()
        }
        assert names <= registered, (ctype, names - registered)
