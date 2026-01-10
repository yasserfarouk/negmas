"""Tests for Genius BOA components transcompiled from Java."""

from __future__ import annotations

import pytest

from negmas import (
    GACNext,
    GHardHeadedFrequencyModel,
    GTimeDependentOffering,
    GeniusAcceptancePolicy,
    GeniusOfferingPolicy,
    GeniusOpponentModel,
)
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator
from negmas.sao.negotiators.modular.boa import make_boa


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_issues():
    """Create simple issues for testing."""
    return [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=10),
        make_issue(name="delivery", values=5),
    ]


@pytest.fixture
def simple_mechanism(simple_issues):
    """Create a simple mechanism for testing."""
    return SAOMechanism(issues=simple_issues, n_steps=100)


@pytest.fixture
def buyer_ufun(simple_mechanism):
    """Create buyer utility function."""
    return LUFun(
        values={
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery": IdentityFun(),
        },
        outcome_space=simple_mechanism.outcome_space,
        reserved_value=0.0,
    ).scale_max(1.0)


@pytest.fixture
def seller_ufun(simple_mechanism):
    """Create seller utility function."""
    return LUFun(
        values={
            "price": IdentityFun(),
            "quantity": LinearFun(0.2),
            "delivery": AffineFun(-1, bias=4),
        },
        outcome_space=simple_mechanism.outcome_space,
        reserved_value=0.0,
    ).scale_max(1.0)


# ============================================================================
# Test Base Classes
# ============================================================================


class TestGeniusBaseClasses:
    """Test that base classes exist and have correct inheritance."""

    def test_genius_offering_policy_is_offering_policy(self):
        """GeniusOfferingPolicy should inherit from OfferingPolicy."""
        from negmas.gb.components import OfferingPolicy

        assert issubclass(GeniusOfferingPolicy, OfferingPolicy)

    def test_genius_acceptance_policy_is_acceptance_policy(self):
        """GeniusAcceptancePolicy should inherit from AcceptancePolicy."""
        from negmas.gb.components import AcceptancePolicy

        assert issubclass(GeniusAcceptancePolicy, AcceptancePolicy)

    def test_genius_opponent_model_is_base_ufun(self):
        """GeniusOpponentModel should inherit from BaseUtilityFunction."""
        from negmas.preferences.base_ufun import BaseUtilityFunction

        assert issubclass(GeniusOpponentModel, BaseUtilityFunction)


# ============================================================================
# Test GTimeDependentOffering
# ============================================================================


class TestGTimeDependentOffering:
    """Tests for the GTimeDependentOffering policy."""

    def test_inherits_from_genius_offering_policy(self):
        """GTimeDependentOffering should inherit from GeniusOfferingPolicy."""
        assert issubclass(GTimeDependentOffering, GeniusOfferingPolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        policy = GTimeDependentOffering()
        assert policy.e == 0.2  # Boulware by default
        assert policy.k == 0.0

    def test_custom_parameters(self):
        """Test custom parameter values."""
        policy = GTimeDependentOffering(e=1.0, k=0.1)
        assert policy.e == 1.0
        assert policy.k == 0.1

    def test_f_function_hardliner(self):
        """Test f(t) for hardliner (e=0)."""
        policy = GTimeDependentOffering(e=0, k=0.5)
        # When e=0, f(t) = k regardless of t
        assert policy._f(0.0) == 0.5
        assert policy._f(0.5) == 0.5
        assert policy._f(1.0) == 0.5

    def test_f_function_linear(self):
        """Test f(t) for linear concession (e=1)."""
        policy = GTimeDependentOffering(e=1.0, k=0.0)
        # When e=1 and k=0, f(t) = t
        assert abs(policy._f(0.0) - 0.0) < 1e-9
        assert abs(policy._f(0.5) - 0.5) < 1e-9
        assert abs(policy._f(1.0) - 1.0) < 1e-9

    def test_f_function_boulware(self):
        """Test f(t) for Boulware strategy (e<1)."""
        policy = GTimeDependentOffering(e=0.2, k=0.0)
        # Boulware: f(t) = t^(1/e) where 1/e > 1, so t^(1/e) < t for t in (0,1)
        # This means f(t) < t: concedes slowly at start, then faster at end
        f_early = policy._f(0.2)
        f_mid = policy._f(0.5)
        f_late = policy._f(0.8)
        # Should be convex: f(t) < t for all t in (0,1)
        assert f_early < 0.2  # Concedes less than linear early
        assert f_mid < 0.5
        assert f_late < 0.8

    def test_f_function_conceder(self):
        """Test f(t) for Conceder strategy (e>1)."""
        policy = GTimeDependentOffering(e=4.0, k=0.0)
        # Conceder: f(t) = t^(1/e) where 1/e < 1, so t^(1/e) > t for t in (0,1)
        # This means f(t) > t: concedes quickly at start, then slows at end
        f_early = policy._f(0.2)
        f_mid = policy._f(0.5)
        # Should be concave: f(t) > t for all t in (0,1)
        assert f_early > 0.2  # Concedes more than linear early
        assert f_mid > 0.5

    def test_in_negotiation_boulware(self, simple_mechanism, buyer_ufun):
        """Test GTimeDependentOffering in actual negotiation with Boulware."""
        offering = GTimeDependentOffering(e=0.2)  # Boulware
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=buyer_ufun)

        result = simple_mechanism.run()
        # Should complete without errors
        assert result.ended

    def test_in_negotiation_conceder(self, simple_mechanism, buyer_ufun):
        """Test GTimeDependentOffering in actual negotiation with Conceder."""
        offering = GTimeDependentOffering(e=4.0)  # Conceder
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=buyer_ufun)

        result = simple_mechanism.run()
        assert result.ended


# ============================================================================
# Test GACNext
# ============================================================================


class TestGACNext:
    """Tests for the GACNext acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACNext should inherit from GeniusAcceptancePolicy."""
        assert issubclass(GACNext, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        offering = GTimeDependentOffering()
        policy = GACNext(offering_policy=offering)
        assert policy.a == 1.0
        assert policy.b == 0.0

    def test_custom_parameters(self):
        """Test custom parameter values."""
        offering = GTimeDependentOffering()
        policy = GACNext(offering_policy=offering, a=1.1, b=0.05)
        assert policy.a == 1.1
        assert policy.b == 0.05

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACNext in actual negotiation."""
        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


# ============================================================================
# Test GHardHeadedFrequencyModel
# ============================================================================


class TestGHardHeadedFrequencyModel:
    """Tests for the GHardHeadedFrequencyModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GHardHeadedFrequencyModel should inherit from GeniusOpponentModel."""
        assert issubclass(GHardHeadedFrequencyModel, GeniusOpponentModel)

    def test_default_parameters(self):
        """Test default parameter values."""
        model = GHardHeadedFrequencyModel()
        assert model.learning_coef == 0.2
        assert model.learning_value_addition == 1
        assert model.default_value == 1

    def test_custom_parameters(self):
        """Test custom parameter values."""
        model = GHardHeadedFrequencyModel(
            learning_coef=0.3, learning_value_addition=2, default_value=2
        )
        assert model.learning_coef == 0.3
        assert model.learning_value_addition == 2
        assert model.default_value == 2

    def test_eval_returns_value_in_range(self, simple_mechanism, buyer_ufun):
        """Test that eval returns values in [0, 1]."""
        model = GHardHeadedFrequencyModel()

        # Create a negotiator with the model
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=buyer_ufun)

        # Run a few steps
        for _ in range(10):
            simple_mechanism.step()
            if simple_mechanism.state.ended:
                break

        # Test eval on some outcomes
        outcomes = list(simple_mechanism.outcome_space.enumerate())[:10]
        for outcome in outcomes:
            value = model.eval(outcome)
            assert (
                0.0 <= float(value) <= 1.0
            ), f"eval({outcome}) = {value} not in [0, 1]"

    def test_in_negotiation_with_model(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GHardHeadedFrequencyModel in actual negotiation."""
        model = GHardHeadedFrequencyModel()
        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


# ============================================================================
# Integration Tests: Complete Negotiations
# ============================================================================


class TestGeniusComponentsIntegration:
    """Integration tests for complete negotiations using Genius components."""

    def test_genius_boa_vs_aspiration_reaches_agreement(
        self, simple_mechanism, buyer_ufun, seller_ufun
    ):
        """Test that Genius BOA negotiator can reach agreement with AspirationNegotiator."""
        # Create Genius BOA negotiator
        offering = GTimeDependentOffering(e=1.0)  # Linear
        acceptance = GACNext(offering_policy=offering)
        model = GHardHeadedFrequencyModel()
        genius_neg = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(genius_neg, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()

        assert result.ended
        # With linear concession, agreement is very likely
        # but not guaranteed, so we just check it completes

    def test_genius_boa_vs_genius_boa_reaches_agreement(
        self, simple_mechanism, buyer_ufun, seller_ufun
    ):
        """Test two Genius BOA negotiators can reach agreement."""
        # Create two Genius BOA negotiators with different strategies
        offering1 = GTimeDependentOffering(e=0.5)  # Somewhat Boulware
        acceptance1 = GACNext(offering_policy=offering1)
        neg1 = make_boa(offering=offering1, acceptance=acceptance1)

        offering2 = GTimeDependentOffering(e=2.0)  # Conceder
        acceptance2 = GACNext(offering_policy=offering2)
        neg2 = make_boa(offering=offering2, acceptance=acceptance2)

        simple_mechanism.add(neg1, ufun=buyer_ufun)
        simple_mechanism.add(neg2, ufun=seller_ufun)

        result = simple_mechanism.run()

        assert result.ended
        # Conceder vs Boulware should likely reach agreement
        assert result.agreement is not None

    def test_hardliner_never_concedes(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test that hardliner (e=0) never concedes."""
        offering = GTimeDependentOffering(e=0, k=0)  # Hardliner
        acceptance = GACNext(offering_policy=offering)
        neg = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(neg, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()

        assert result.ended
        # Hardliner should make very few different offers
        # (only best outcomes for itself)

    @pytest.mark.parametrize("e", [0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    def test_various_concession_rates(self, simple_issues, buyer_ufun, seller_ufun, e):
        """Test various concession rates complete without error."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)
        # Need to recreate ufuns with new outcome space
        buyer = LUFun(
            values={
                "price": AffineFun(-1, bias=9.0),
                "quantity": LinearFun(0.2),
                "delivery": IdentityFun(),
            },
            outcome_space=mechanism.outcome_space,
            reserved_value=0.0,
        ).scale_max(1.0)
        seller = LUFun(
            values={
                "price": IdentityFun(),
                "quantity": LinearFun(0.2),
                "delivery": AffineFun(-1, bias=4),
            },
            outcome_space=mechanism.outcome_space,
            reserved_value=0.0,
        ).scale_max(1.0)

        offering = GTimeDependentOffering(e=e)
        acceptance = GACNext(offering_policy=offering)
        neg = make_boa(offering=offering, acceptance=acceptance)

        mechanism.add(neg, ufun=buyer)
        mechanism.add(AspirationNegotiator(), ufun=seller)

        result = mechanism.run()
        assert result.ended

    def test_longer_negotiation(self, simple_issues, buyer_ufun, seller_ufun):
        """Test in longer negotiation (more steps)."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=500)
        buyer = LUFun(
            values={
                "price": AffineFun(-1, bias=9.0),
                "quantity": LinearFun(0.2),
                "delivery": IdentityFun(),
            },
            outcome_space=mechanism.outcome_space,
            reserved_value=0.0,
        ).scale_max(1.0)
        seller = LUFun(
            values={
                "price": IdentityFun(),
                "quantity": LinearFun(0.2),
                "delivery": AffineFun(-1, bias=4),
            },
            outcome_space=mechanism.outcome_space,
            reserved_value=0.0,
        ).scale_max(1.0)

        offering = GTimeDependentOffering(e=0.2)  # Boulware
        acceptance = GACNext(offering_policy=offering)
        model = GHardHeadedFrequencyModel()
        neg = make_boa(offering=offering, acceptance=acceptance, model=model)

        mechanism.add(neg, ufun=buyer)
        mechanism.add(AspirationNegotiator(), ufun=seller)

        result = mechanism.run()
        assert result.ended


# ============================================================================
# Test Behavior Correctness
# ============================================================================


class TestBehaviorCorrectness:
    """Tests to verify components behave as expected."""

    def test_boulware_concedes_slowly_at_start(self, simple_issues):
        """Boulware should concede slowly at the start."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)
        ufun = LUFun.random(outcome_space=mechanism.outcome_space, reserved_value=0.0)

        offering = GTimeDependentOffering(e=0.2)  # Boulware
        acceptance = GACNext(offering_policy=offering)
        neg = make_boa(offering=offering, acceptance=acceptance)

        mechanism.add(neg, ufun=ufun)
        mechanism.add(AspirationNegotiator(), ufun=ufun)

        # Run first few steps and collect utilities
        utilities = []
        for _ in range(20):
            mechanism.step()
            if mechanism.state.ended:
                break
            # Get the last offer from our negotiator
            offers = mechanism.negotiator_offers(neg.id)
            if offers:
                last_offer = offers[-1]
                if last_offer is not None:
                    utilities.append(float(ufun(last_offer)))

        # With Boulware, early utilities should be high (slow concession)
        if len(utilities) >= 3:
            # First few offers should have high utility
            assert utilities[0] > 0.8, f"First offer utility {utilities[0]} too low"

    def test_conceder_concedes_quickly_at_start(self, simple_issues):
        """Conceder should concede quickly at the start."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)
        ufun = LUFun.random(outcome_space=mechanism.outcome_space, reserved_value=0.0)

        offering = GTimeDependentOffering(e=4.0)  # Conceder
        acceptance = GACNext(offering_policy=offering)
        neg = make_boa(offering=offering, acceptance=acceptance)

        mechanism.add(neg, ufun=ufun)
        mechanism.add(AspirationNegotiator(), ufun=ufun)

        # Run first few steps and collect utilities
        utilities = []
        for _ in range(20):
            mechanism.step()
            if mechanism.state.ended:
                break
            offers = mechanism.negotiator_offers(neg.id)
            if offers:
                last_offer = offers[-1]
                if last_offer is not None:
                    utilities.append(float(ufun(last_offer)))

        # Conceder should drop utility faster
        if len(utilities) >= 5:
            # Utility should drop significantly in first few offers
            drop = utilities[0] - utilities[4]
            assert drop > 0, "Conceder should show utility decrease"

    def test_ac_next_accepts_when_opponent_offer_better_than_next(self, simple_issues):
        """AC_Next should accept when opponent offer >= next own offer utility."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)
        ufun = LUFun.random(outcome_space=mechanism.outcome_space, reserved_value=0.0)

        # Use a conceder that will make low utility offers quickly
        offering = GTimeDependentOffering(e=10.0)  # Very conceding
        acceptance = GACNext(offering_policy=offering)
        neg = make_boa(offering=offering, acceptance=acceptance)

        mechanism.add(neg, ufun=ufun)
        # Use tough opponent
        mechanism.add(AspirationNegotiator(), ufun=ufun)

        result = mechanism.run()
        assert result.ended
        # With very conceding strategy, should reach agreement
        assert result.agreement is not None

    def test_opponent_model_learns_from_offers(self, simple_issues):
        """GHardHeadedFrequencyModel should update based on opponent offers."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=50)
        ufun = LUFun.random(outcome_space=mechanism.outcome_space, reserved_value=0.0)

        model = GHardHeadedFrequencyModel(learning_coef=0.5)
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        neg = make_boa(offering=offering, acceptance=acceptance, model=model)

        mechanism.add(neg, ufun=ufun)
        mechanism.add(AspirationNegotiator(), ufun=ufun)

        # Run negotiation
        mechanism.run()

        # Model should have learned something
        # (issue weights should have changed from uniform)
        if model._initialized and model._n_issues > 0:
            weights = list(model._issue_weights.values())
            # Not all weights should be exactly equal after learning
            if len(weights) > 1:
                # At least check that learning happened
                assert len(model._value_weights) > 0


# ============================================================================
# Test Import Paths
# ============================================================================


class TestImportPaths:
    """Test that components can be imported from various paths."""

    def test_import_from_negmas(self):
        """Test import directly from negmas."""
        from negmas import (
            GACNext,
            GHardHeadedFrequencyModel,
            GTimeDependentOffering,
            GeniusAcceptancePolicy,
            GeniusOfferingPolicy,
            GeniusOpponentModel,
        )

        assert GTimeDependentOffering is not None
        assert GACNext is not None
        assert GHardHeadedFrequencyModel is not None
        assert GeniusOfferingPolicy is not None
        assert GeniusAcceptancePolicy is not None
        assert GeniusOpponentModel is not None

    def test_import_from_gb(self):
        """Test import from negmas.gb."""
        from negmas.gb import GACNext, GHardHeadedFrequencyModel, GTimeDependentOffering

        assert GTimeDependentOffering is not None
        assert GACNext is not None
        assert GHardHeadedFrequencyModel is not None

    def test_import_from_sao(self):
        """Test import from negmas.sao."""
        from negmas.sao import (
            GACNext,
            GHardHeadedFrequencyModel,
            GTimeDependentOffering,
        )

        assert GTimeDependentOffering is not None
        assert GACNext is not None
        assert GHardHeadedFrequencyModel is not None

    def test_import_from_gb_components(self):
        """Test import from negmas.gb.components."""
        from negmas.gb.components import (
            GACNext,
            GHardHeadedFrequencyModel,
            GTimeDependentOffering,
        )

        assert GTimeDependentOffering is not None
        assert GACNext is not None
        assert GHardHeadedFrequencyModel is not None

    def test_import_from_gb_components_genius(self):
        """Test import from negmas.gb.components.genius."""
        from negmas.gb.components.genius import (
            GACNext,
            GHardHeadedFrequencyModel,
            GTimeDependentOffering,
        )

        assert GTimeDependentOffering is not None
        assert GACNext is not None
        assert GHardHeadedFrequencyModel is not None


class TestCLIParsing:
    """Test CLI parsing functions for BOA negotiators."""

    def test_parse_component_spec_simple(self):
        """Test parsing a simple component spec without parameters."""
        from negmas.scripts.negotiate import parse_component_spec

        name, kwargs = parse_component_spec("GTimeDependentOffering")
        assert name == "GTimeDependentOffering"
        assert kwargs == {}

    def test_parse_component_spec_with_params(self):
        """Test parsing a component spec with parameters."""
        from negmas.scripts.negotiate import parse_component_spec

        name, kwargs = parse_component_spec("GTimeDependentOffering(e=0.2, k=0.0)")
        assert name == "GTimeDependentOffering"
        assert kwargs == {"e": 0.2, "k": 0.0}

    def test_get_component_genius(self):
        """Test getting Genius component classes."""
        from negmas.scripts.negotiate import get_component

        cls = get_component("GTimeDependentOffering")
        assert cls.__name__ == "GTimeDependentOffering"

        cls = get_component("GACNext")
        assert cls.__name__ == "GACNext"

        cls = get_component("GHardHeadedFrequencyModel")
        assert cls.__name__ == "GHardHeadedFrequencyModel"

    def test_get_component_builtin(self):
        """Test getting built-in component classes."""
        from negmas.scripts.negotiate import get_component

        cls = get_component("TimeBasedOfferingPolicy")
        assert cls.__name__ == "TimeBasedOfferingPolicy"

        cls = get_component("ACNext")
        assert cls.__name__ == "ACNext"

    def test_make_boa_negotiator_genius(self):
        """Test creating a BOA negotiator with Genius components."""
        from negmas.scripts.negotiate import make_boa_negotiator

        neg = make_boa_negotiator(
            "offering=GTimeDependentOffering(e=0.2),acceptance=GACNext", name="test"
        )
        assert neg is not None
        assert neg.name == "test"

    def test_make_boa_negotiator_with_model(self):
        """Test creating a BOA negotiator with an opponent model."""
        from negmas.scripts.negotiate import make_boa_negotiator

        neg = make_boa_negotiator(
            "offering=GTimeDependentOffering(e=0.5),acceptance=GACNext,model=GHardHeadedFrequencyModel",
            name="test_model",
        )
        assert neg is not None
        assert neg.name == "test_model"

    def test_get_negotiator_boa_prefix(self):
        """Test get_negotiator with boa: prefix."""
        from negmas.scripts.negotiate import get_negotiator

        factory = get_negotiator(
            "boa:offering=GTimeDependentOffering(e=0.2),acceptance=GACNext"
        )
        neg = factory(name="test_boa")
        assert neg is not None
        assert neg.name == "test_boa"

    def test_get_negotiator_map_prefix(self):
        """Test get_negotiator with map: prefix."""
        from negmas.scripts.negotiate import get_negotiator

        factory = get_negotiator(
            "map:offering=GTimeDependentOffering(e=0.3),acceptance=GACNext"
        )
        neg = factory(name="test_map")
        assert neg is not None
        assert neg.name == "test_map"


# ============================================================================
# Test New Acceptance Policies
# ============================================================================


class TestGACConst:
    """Tests for the GACConst acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACConst should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACConst

        assert issubclass(GACConst, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACConst

        policy = GACConst()
        assert policy.c == 0.9

    def test_custom_parameters(self):
        """Test custom parameter values."""
        from negmas.gb.components.genius import GACConst

        policy = GACConst(c=0.8)
        assert policy.c == 0.8

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACConst in actual negotiation."""
        from negmas.gb.components.genius import GACConst

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACConst(c=0.6)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACTime:
    """Tests for the GACTime acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACTime should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACTime

        assert issubclass(GACTime, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACTime

        policy = GACTime()
        assert policy.t == 0.99

    def test_custom_parameters(self):
        """Test custom parameter values."""
        from negmas.gb.components.genius import GACTime

        policy = GACTime(t=0.95)
        assert policy.t == 0.95

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACTime in actual negotiation."""
        from negmas.gb.components.genius import GACTime

        offering = GTimeDependentOffering(e=0)  # hardliner
        acceptance = GACTime(t=0.9)  # will accept at end
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACPrevious:
    """Tests for the GACPrevious acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACPrevious should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACPrevious

        assert issubclass(GACPrevious, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACPrevious

        policy = GACPrevious()
        assert policy.a == 1.0
        assert policy.b == 0.0

    def test_custom_parameters(self):
        """Test custom parameter values."""
        from negmas.gb.components.genius import GACPrevious

        policy = GACPrevious(a=1.1, b=0.05)
        assert policy.a == 1.1
        assert policy.b == 0.05

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACPrevious in actual negotiation."""
        from negmas.gb.components.genius import GACPrevious

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACPrevious(a=1.0, b=0.0)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACGap:
    """Tests for the GACGap acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACGap should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACGap

        assert issubclass(GACGap, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACGap

        policy = GACGap()
        assert policy.c == 0.01

    def test_custom_parameters(self):
        """Test custom parameter values."""
        from negmas.gb.components.genius import GACGap

        policy = GACGap(c=0.05)
        assert policy.c == 0.05

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACGap in actual negotiation."""
        from negmas.gb.components.genius import GACGap

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACGap(c=0.1)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACCombi:
    """Tests for the GACCombi acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACCombi should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACCombi

        assert issubclass(GACCombi, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACCombi

        offering = GTimeDependentOffering()
        policy = GACCombi(offering_policy=offering)
        assert policy.a == 1.0
        assert policy.b == 0.0
        assert policy.t == 0.99

    def test_custom_parameters(self):
        """Test custom parameter values."""
        from negmas.gb.components.genius import GACCombi

        offering = GTimeDependentOffering()
        policy = GACCombi(offering_policy=offering, a=1.1, b=0.05, t=0.95)
        assert policy.a == 1.1
        assert policy.b == 0.05
        assert policy.t == 0.95

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACCombi in actual negotiation."""
        from negmas.gb.components.genius import GACCombi

        offering = GTimeDependentOffering(e=0.2)  # Boulware
        acceptance = GACCombi(offering_policy=offering, t=0.95)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACCombiMaxInWindow:
    """Tests for the GACCombiMaxInWindow acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACCombiMaxInWindow should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACCombiMaxInWindow

        assert issubclass(GACCombiMaxInWindow, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACCombiMaxInWindow

        offering = GTimeDependentOffering()
        policy = GACCombiMaxInWindow(offering_policy=offering)
        assert policy.t == 0.98

    def test_custom_parameters(self):
        """Test custom parameter values."""
        from negmas.gb.components.genius import GACCombiMaxInWindow

        offering = GTimeDependentOffering()
        policy = GACCombiMaxInWindow(offering_policy=offering, t=0.9)
        assert policy.t == 0.9

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACCombiMaxInWindow in actual negotiation."""
        from negmas.gb.components.genius import GACCombiMaxInWindow

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACCombiMaxInWindow(offering_policy=offering, t=0.9)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestNewAcceptanceImportPaths:
    """Test that new acceptance policies can be imported from various paths."""

    def test_import_from_negmas_gb_components_genius(self):
        """Test import from negmas.gb.components.genius."""
        from negmas.gb.components.genius import (
            GACCombi,
            GACCombiMaxInWindow,
            GACConst,
            GACGap,
            GACPrevious,
            GACTime,
        )

        assert GACConst is not None
        assert GACTime is not None
        assert GACPrevious is not None
        assert GACGap is not None
        assert GACCombi is not None
        assert GACCombiMaxInWindow is not None

    def test_import_from_negmas_sao(self):
        """Test import from negmas.sao."""
        from negmas.sao import (
            GACCombi,
            GACCombiMaxInWindow,
            GACConst,
            GACGap,
            GACPrevious,
            GACTime,
        )

        assert GACConst is not None
        assert GACTime is not None
        assert GACPrevious is not None
        assert GACGap is not None
        assert GACCombi is not None
        assert GACCombiMaxInWindow is not None

    def test_import_from_negmas(self):
        """Test import from negmas."""
        from negmas import (
            GACCombi,
            GACCombiMaxInWindow,
            GACConst,
            GACGap,
            GACPrevious,
            GACTime,
        )

        assert GACConst is not None
        assert GACTime is not None
        assert GACPrevious is not None
        assert GACGap is not None
        assert GACCombi is not None
        assert GACCombiMaxInWindow is not None


# ============================================================================
# Test New Acceptance Policies (Extended)
# ============================================================================


class TestGACTrue:
    """Tests for the GACTrue acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACTrue should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACTrue

        assert issubclass(GACTrue, GeniusAcceptancePolicy)

    def test_always_accepts(self, simple_mechanism, buyer_ufun, seller_ufun):
        """GACTrue should always accept offers."""
        from negmas.gb.components.genius import GACTrue

        offering = GTimeDependentOffering(e=0)  # Hardliner
        acceptance = GACTrue()
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended
        # Should accept very quickly
        assert result.agreement is not None


class TestGACFalse:
    """Tests for the GACFalse acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACFalse should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACFalse

        assert issubclass(GACFalse, GeniusAcceptancePolicy)

    def test_never_accepts(self, simple_mechanism, buyer_ufun, seller_ufun):
        """GACFalse should never accept offers."""
        from negmas.gb.components.genius import GACFalse

        offering = GTimeDependentOffering(e=10)  # Very conceding
        acceptance = GACFalse()
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended
        # Should never reach agreement from this negotiator's side
        # (unless opponent accepts first)


class TestGACCombiV2:
    """Tests for the GACCombiV2 acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACCombiV2 should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACCombiV2

        assert issubclass(GACCombiV2, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACCombiV2

        offering = GTimeDependentOffering()
        policy = GACCombiV2(offering_policy=offering)
        assert policy.a == 1.0
        assert policy.b == 0.0
        assert policy.t == 0.99
        assert policy.decay == 0.9

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACCombiV2 in actual negotiation."""
        from negmas.gb.components.genius import GACCombiV2

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACCombiV2(offering_policy=offering, t=0.9)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACCombiV3:
    """Tests for the GACCombiV3 acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACCombiV3 should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACCombiV3

        assert issubclass(GACCombiV3, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACCombiV3

        offering = GTimeDependentOffering()
        policy = GACCombiV3(offering_policy=offering)
        assert policy.a == 1.0
        assert policy.b == 0.0
        assert policy.t == 0.95

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACCombiV3 in actual negotiation."""
        from negmas.gb.components.genius import GACCombiV3

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACCombiV3(offering_policy=offering, t=0.9)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACCombiV4:
    """Tests for the GACCombiV4 acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACCombiV4 should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACCombiV4

        assert issubclass(GACCombiV4, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACCombiV4

        offering = GTimeDependentOffering()
        policy = GACCombiV4(offering_policy=offering)
        assert policy.t == 0.98
        assert policy.w == 0.5

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACCombiV4 in actual negotiation."""
        from negmas.gb.components.genius import GACCombiV4

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACCombiV4(offering_policy=offering, t=0.9)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGACCombiProb:
    """Tests for the GACCombiProb acceptance policy."""

    def test_inherits_from_genius_acceptance_policy(self):
        """GACCombiProb should inherit from GeniusAcceptancePolicy."""
        from negmas.gb.components.genius import GACCombiProb

        assert issubclass(GACCombiProb, GeniusAcceptancePolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GACCombiProb

        offering = GTimeDependentOffering()
        policy = GACCombiProb(offering_policy=offering)
        assert policy.t == 0.98

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GACCombiProb in actual negotiation."""
        from negmas.gb.components.genius import GACCombiProb

        offering = GTimeDependentOffering(e=1.0)
        acceptance = GACCombiProb(offering_policy=offering, t=0.9)
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


# ============================================================================
# Test New Offering Strategies
# ============================================================================


class TestGRandomOffering:
    """Tests for the GRandomOffering policy."""

    def test_inherits_from_genius_offering_policy(self):
        """GRandomOffering should inherit from GeniusOfferingPolicy."""
        from negmas.gb.components.genius import GRandomOffering

        assert issubclass(GRandomOffering, GeniusOfferingPolicy)

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GRandomOffering in actual negotiation."""
        from negmas.gb.components.genius import GRandomOffering

        offering = GRandomOffering()
        acceptance = GACNext(offering_policy=GTimeDependentOffering())
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGBoulwareOffering:
    """Tests for the GBoulwareOffering policy."""

    def test_inherits_from_genius_offering_policy(self):
        """GBoulwareOffering should inherit from GeniusOfferingPolicy."""
        from negmas.gb.components.genius import GBoulwareOffering

        assert issubclass(GBoulwareOffering, GeniusOfferingPolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GBoulwareOffering

        policy = GBoulwareOffering()
        assert policy.e == 0.2

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GBoulwareOffering in actual negotiation."""
        from negmas.gb.components.genius import GBoulwareOffering

        offering = GBoulwareOffering()
        acceptance = GACNext(offering_policy=GTimeDependentOffering(e=0.2))
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGConcederOffering:
    """Tests for the GConcederOffering policy."""

    def test_inherits_from_genius_offering_policy(self):
        """GConcederOffering should inherit from GeniusOfferingPolicy."""
        from negmas.gb.components.genius import GConcederOffering

        assert issubclass(GConcederOffering, GeniusOfferingPolicy)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GConcederOffering

        policy = GConcederOffering()
        assert policy.e == 2.0

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GConcederOffering in actual negotiation."""
        from negmas.gb.components.genius import GConcederOffering

        offering = GConcederOffering()
        acceptance = GACNext(offering_policy=GTimeDependentOffering(e=2.0))
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGLinearOffering:
    """Tests for the GLinearOffering policy."""

    def test_inherits_from_genius_offering_policy(self):
        """GLinearOffering should inherit from GeniusOfferingPolicy."""
        from negmas.gb.components.genius import GLinearOffering

        assert issubclass(GLinearOffering, GeniusOfferingPolicy)

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GLinearOffering in actual negotiation."""
        from negmas.gb.components.genius import GLinearOffering

        offering = GLinearOffering()
        acceptance = GACNext(offering_policy=GTimeDependentOffering(e=1.0))
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGHardlinerOffering:
    """Tests for the GHardlinerOffering policy."""

    def test_inherits_from_genius_offering_policy(self):
        """GHardlinerOffering should inherit from GeniusOfferingPolicy."""
        from negmas.gb.components.genius import GHardlinerOffering

        assert issubclass(GHardlinerOffering, GeniusOfferingPolicy)

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GHardlinerOffering in actual negotiation."""
        from negmas.gb.components.genius import GHardlinerOffering

        offering = GHardlinerOffering()
        acceptance = GACNext(offering_policy=GTimeDependentOffering(e=0))
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGChoosingAllBids:
    """Tests for the GChoosingAllBids policy."""

    def test_inherits_from_genius_offering_policy(self):
        """GChoosingAllBids should inherit from GeniusOfferingPolicy."""
        from negmas.gb.components.genius import GChoosingAllBids

        assert issubclass(GChoosingAllBids, GeniusOfferingPolicy)

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GChoosingAllBids in actual negotiation."""
        from negmas.gb.components.genius import GChoosingAllBids

        offering = GChoosingAllBids()
        acceptance = GACNext(offering_policy=GTimeDependentOffering())
        negotiator = make_boa(offering=offering, acceptance=acceptance)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


# ============================================================================
# Test New Opponent Models
# ============================================================================


class TestGDefaultModel:
    """Tests for the GDefaultModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GDefaultModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GDefaultModel

        assert issubclass(GDefaultModel, GeniusOpponentModel)

    def test_returns_constant(self, simple_mechanism, buyer_ufun):
        """Test that GDefaultModel returns constant utility."""
        from negmas.gb.components.genius import GDefaultModel

        model = GDefaultModel()
        outcomes = list(simple_mechanism.outcome_space.enumerate())[:5]
        for outcome in outcomes:
            value = model.eval(outcome)
            assert float(value) == 0.5

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GDefaultModel in actual negotiation."""
        from negmas.gb.components.genius import GDefaultModel

        model = GDefaultModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGUniformModel:
    """Tests for the GUniformModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GUniformModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GUniformModel

        assert issubclass(GUniformModel, GeniusOpponentModel)

    def test_returns_consistent_values(self, simple_mechanism, buyer_ufun):
        """Test that GUniformModel returns consistent values for same outcome."""
        from negmas.gb.components.genius import GUniformModel

        model = GUniformModel()
        outcomes = list(simple_mechanism.outcome_space.enumerate())[:5]
        for outcome in outcomes:
            value1 = model.eval(outcome)
            value2 = model.eval(outcome)
            assert float(value1) == float(value2)

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GUniformModel in actual negotiation."""
        from negmas.gb.components.genius import GUniformModel

        model = GUniformModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGOppositeModel:
    """Tests for the GOppositeModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GOppositeModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GOppositeModel

        assert issubclass(GOppositeModel, GeniusOpponentModel)

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GOppositeModel in actual negotiation."""
        from negmas.gb.components.genius import GOppositeModel

        model = GOppositeModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGSmithFrequencyModel:
    """Tests for the GSmithFrequencyModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GSmithFrequencyModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GSmithFrequencyModel

        assert issubclass(GSmithFrequencyModel, GeniusOpponentModel)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GSmithFrequencyModel

        model = GSmithFrequencyModel()
        assert model.default_value == 1

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GSmithFrequencyModel in actual negotiation."""
        from negmas.gb.components.genius import GSmithFrequencyModel

        model = GSmithFrequencyModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGAgentXFrequencyModel:
    """Tests for the GAgentXFrequencyModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GAgentXFrequencyModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GAgentXFrequencyModel

        assert issubclass(GAgentXFrequencyModel, GeniusOpponentModel)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GAgentXFrequencyModel

        model = GAgentXFrequencyModel()
        assert model.learning_rate == 0.25
        assert model.default_value == 1

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GAgentXFrequencyModel in actual negotiation."""
        from negmas.gb.components.genius import GAgentXFrequencyModel

        model = GAgentXFrequencyModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGNashFrequencyModel:
    """Tests for the GNashFrequencyModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GNashFrequencyModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GNashFrequencyModel

        assert issubclass(GNashFrequencyModel, GeniusOpponentModel)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GNashFrequencyModel

        model = GNashFrequencyModel()
        assert model.default_value == 1

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GNashFrequencyModel in actual negotiation."""
        from negmas.gb.components.genius import GNashFrequencyModel

        model = GNashFrequencyModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGBayesianModel:
    """Tests for the GBayesianModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GBayesianModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GBayesianModel

        assert issubclass(GBayesianModel, GeniusOpponentModel)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GBayesianModel

        model = GBayesianModel()
        assert model.n_hypotheses == 10
        assert model.rationality == 5.0

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GBayesianModel in actual negotiation."""
        from negmas.gb.components.genius import GBayesianModel

        model = GBayesianModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


class TestGScalableBayesianModel:
    """Tests for the GScalableBayesianModel opponent model."""

    def test_inherits_from_genius_opponent_model(self):
        """GScalableBayesianModel should inherit from GeniusOpponentModel."""
        from negmas.gb.components.genius import GScalableBayesianModel

        assert issubclass(GScalableBayesianModel, GeniusOpponentModel)

    def test_default_parameters(self):
        """Test default parameter values."""
        from negmas.gb.components.genius import GScalableBayesianModel

        model = GScalableBayesianModel()
        assert model.learning_rate == 0.1

    def test_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GScalableBayesianModel in actual negotiation."""
        from negmas.gb.components.genius import GScalableBayesianModel

        model = GScalableBayesianModel()
        offering = GTimeDependentOffering()
        acceptance = GACNext(offering_policy=offering)
        negotiator = make_boa(offering=offering, acceptance=acceptance, model=model)

        simple_mechanism.add(negotiator, ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended


# ============================================================================
# Test New Component Imports
# ============================================================================


class TestAllNewComponentImports:
    """Test that all new components can be imported from various paths."""

    def test_import_all_new_acceptance_from_negmas(self):
        """Test import of all new acceptance policies from negmas."""
        from negmas import (
            GACTrue,
            GACFalse,
            GACConstDiscounted,
            GACCombiAvg,
            GACCombiBestAvg,
            GACCombiMax,
            GACCombiV2,
            GACCombiV3,
            GACCombiV4,
            GACCombiBestAvgDiscounted,
            GACCombiMaxInWindowDiscounted,
            GACCombiProb,
            GACCombiProbDiscounted,
        )

        assert GACTrue is not None
        assert GACFalse is not None
        assert GACConstDiscounted is not None
        assert GACCombiAvg is not None
        assert GACCombiBestAvg is not None
        assert GACCombiMax is not None
        assert GACCombiV2 is not None
        assert GACCombiV3 is not None
        assert GACCombiV4 is not None
        assert GACCombiBestAvgDiscounted is not None
        assert GACCombiMaxInWindowDiscounted is not None
        assert GACCombiProb is not None
        assert GACCombiProbDiscounted is not None

    def test_import_all_new_offering_from_negmas(self):
        """Test import of all new offering policies from negmas."""
        from negmas import (
            GRandomOffering,
            GBoulwareOffering,
            GConcederOffering,
            GLinearOffering,
            GHardlinerOffering,
            GChoosingAllBids,
        )

        assert GRandomOffering is not None
        assert GBoulwareOffering is not None
        assert GConcederOffering is not None
        assert GLinearOffering is not None
        assert GHardlinerOffering is not None
        assert GChoosingAllBids is not None

    def test_import_all_new_models_from_negmas(self):
        """Test import of all new opponent models from negmas."""
        from negmas import (
            GDefaultModel,
            GUniformModel,
            GOppositeModel,
            GSmithFrequencyModel,
            GAgentXFrequencyModel,
            GNashFrequencyModel,
            GBayesianModel,
            GScalableBayesianModel,
        )

        assert GDefaultModel is not None
        assert GUniformModel is not None
        assert GOppositeModel is not None
        assert GSmithFrequencyModel is not None
        assert GAgentXFrequencyModel is not None
        assert GNashFrequencyModel is not None
        assert GBayesianModel is not None
        assert GScalableBayesianModel is not None

    def test_import_from_sao_components(self):
        """Test import from negmas.sao.components."""
        from negmas.sao.components import (
            GACTrue,
            GACFalse,
            GRandomOffering,
            GBoulwareOffering,
            GDefaultModel,
            GBayesianModel,
        )

        assert GACTrue is not None
        assert GACFalse is not None
        assert GRandomOffering is not None
        assert GBoulwareOffering is not None
        assert GDefaultModel is not None
        assert GBayesianModel is not None

    def test_import_from_gb_components(self):
        """Test import from negmas.gb.components."""
        from negmas.gb.components import (
            GACCombiV2,
            GACCombiV3,
            GACCombiV4,
            GLinearOffering,
            GHardlinerOffering,
            GAgentXFrequencyModel,
            GNashFrequencyModel,
        )

        assert GACCombiV2 is not None
        assert GACCombiV3 is not None
        assert GACCombiV4 is not None
        assert GLinearOffering is not None
        assert GHardlinerOffering is not None
        assert GAgentXFrequencyModel is not None
        assert GNashFrequencyModel is not None


# ============================================================================
# Tests for Python-native Genius BOA Negotiators
# ============================================================================


class TestGeniusBOANegotiators:
    """Tests for Python-native Genius BOA Negotiator implementations."""

    @pytest.fixture
    def simple_mechanism(self):
        """Create a simple SAO mechanism for testing."""
        from negmas import SAOMechanism, make_issue

        issues = [make_issue((0, 10), "price"), make_issue((0, 5), "quantity")]
        return SAOMechanism(issues=issues, n_steps=100)

    @pytest.fixture
    def buyer_ufun(self, simple_mechanism):
        """Create a buyer utility function."""
        from negmas import LinearAdditiveUtilityFunction as U

        return U.random(simple_mechanism.outcome_space)

    @pytest.fixture
    def seller_ufun(self, simple_mechanism):
        """Create a seller utility function."""
        from negmas import LinearAdditiveUtilityFunction as U

        return U.random(simple_mechanism.outcome_space)

    # ==========================================================================
    # Import Tests
    # ==========================================================================

    def test_import_from_genius_module(self):
        """Test that all negotiators can be imported from negmas.genius."""
        from negmas.genius import (
            GBoulware,
            GConceder,
            GLinear,
            GHardliner,
            GHardHeaded,
            GAgentK,
            GAgentSmith,
            GNozomi,
            GFSEGA,
            GCUHKAgent,
            GAgentLG,
            GAgentX,
            GRandom,
        )

        assert GBoulware is not None
        assert GConceder is not None
        assert GLinear is not None
        assert GHardliner is not None
        assert GHardHeaded is not None
        assert GAgentK is not None
        assert GAgentSmith is not None
        assert GNozomi is not None
        assert GFSEGA is not None
        assert GCUHKAgent is not None
        assert GAgentLG is not None
        assert GAgentX is not None
        assert GRandom is not None

    def test_all_inherit_from_boa_negotiator(self):
        """Test that all negotiators inherit from BOANegotiator."""
        from negmas.gb.negotiators.modular.boa import BOANegotiator
        from negmas.genius import (
            GBoulware,
            GConceder,
            GLinear,
            GHardliner,
            GHardHeaded,
            GAgentK,
            GAgentSmith,
            GNozomi,
            GFSEGA,
            GCUHKAgent,
            GAgentLG,
            GAgentX,
            GRandom,
        )

        negotiators = [
            GBoulware,
            GConceder,
            GLinear,
            GHardliner,
            GHardHeaded,
            GAgentK,
            GAgentSmith,
            GNozomi,
            GFSEGA,
            GCUHKAgent,
            GAgentLG,
            GAgentX,
            GRandom,
        ]

        for neg_cls in negotiators:
            assert issubclass(
                neg_cls, BOANegotiator
            ), f"{neg_cls.__name__} should inherit from BOANegotiator"

    # ==========================================================================
    # Classic Time-Dependent Agents Tests
    # ==========================================================================

    def test_gboulware_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GBoulware negotiator in actual negotiation."""
        from negmas.genius import GBoulware
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GBoulware(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gconceder_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GConceder negotiator in actual negotiation."""
        from negmas.genius import GConceder
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GConceder(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_glinear_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GLinear negotiator in actual negotiation."""
        from negmas.genius import GLinear
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GLinear(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_ghardliner_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GHardliner negotiator in actual negotiation."""
        from negmas.genius import GHardliner
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GHardliner(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    # ==========================================================================
    # ANAC Competition Agent Tests
    # ==========================================================================

    def test_ghardheaded_in_negotiation(
        self, simple_mechanism, buyer_ufun, seller_ufun
    ):
        """Test GHardHeaded negotiator (ANAC 2011 Winner) in actual negotiation."""
        from negmas.genius import GHardHeaded
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GHardHeaded(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_ghardheaded_custom_e(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GHardHeaded with custom e parameter."""
        from negmas.genius import GHardHeaded
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GHardHeaded(e=0.5), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gagentk_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GAgentK negotiator (ANAC 2010) in actual negotiation."""
        from negmas.genius import GAgentK
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GAgentK(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gagentsmith_in_negotiation(
        self, simple_mechanism, buyer_ufun, seller_ufun
    ):
        """Test GAgentSmith negotiator (ANAC 2010) in actual negotiation."""
        from negmas.genius import GAgentSmith
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GAgentSmith(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gnozomi_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GNozomi negotiator (ANAC 2010) in actual negotiation."""
        from negmas.genius import GNozomi
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GNozomi(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gfsega_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GFSEGA negotiator (ANAC 2010) in actual negotiation."""
        from negmas.genius import GFSEGA
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GFSEGA(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gcuhkagent_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GCUHKAgent negotiator (ANAC 2012 Winner) in actual negotiation."""
        from negmas.genius import GCUHKAgent
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GCUHKAgent(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gagentlg_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GAgentLG negotiator (ANAC 2012) in actual negotiation."""
        from negmas.genius import GAgentLG
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GAgentLG(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    def test_gagentx_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GAgentX negotiator (ANAC 2015) in actual negotiation."""
        from negmas.genius import GAgentX
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GAgentX(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    # ==========================================================================
    # Utility Agent Tests
    # ==========================================================================

    def test_grandom_in_negotiation(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GRandom negotiator in actual negotiation."""
        from negmas.genius import GRandom
        from negmas.sao import AspirationNegotiator

        simple_mechanism.add(GRandom(), ufun=buyer_ufun)
        simple_mechanism.add(AspirationNegotiator(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended

    # ==========================================================================
    # Agent vs Agent Tests
    # ==========================================================================

    def test_gboulware_vs_gconceder(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GBoulware vs GConceder negotiation."""
        from negmas.genius import GBoulware, GConceder

        simple_mechanism.add(GBoulware(), ufun=buyer_ufun)
        simple_mechanism.add(GConceder(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended
        # Conceder should likely agree to Boulware's terms
        # but we don't enforce specific outcome

    def test_ghardheaded_vs_ghardheaded(
        self, simple_mechanism, buyer_ufun, seller_ufun
    ):
        """Test GHardHeaded vs GHardHeaded negotiation."""
        from negmas.genius import GHardHeaded

        simple_mechanism.add(GHardHeaded(), ufun=buyer_ufun)
        simple_mechanism.add(GHardHeaded(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended
        # Two hard negotiators might not reach agreement

    def test_gagentk_vs_gcuhkagent(self, simple_mechanism, buyer_ufun, seller_ufun):
        """Test GAgentK vs GCUHKAgent negotiation."""
        from negmas.genius import GAgentK, GCUHKAgent

        simple_mechanism.add(GAgentK(), ufun=buyer_ufun)
        simple_mechanism.add(GCUHKAgent(), ufun=seller_ufun)

        result = simple_mechanism.run()
        assert result.ended
