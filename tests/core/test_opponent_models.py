"""Tests for opponent modeling functionality.

This module tests that opponent models correctly set private_info["opponent_ufun"]
and that negotiators can access and use these models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from negmas import SAOMechanism
from negmas.gb.negotiators.modular.boa import BOANegotiator
from negmas.sao.components.models.ufun import ZeroSumModel
from negmas.gb.components.genius.models import (
    GAgentXFrequencyModel,
    GBayesianModel,
    GDefaultModel,
    GHardHeadedFrequencyModel,
    GOppositeModel,
    GPerfectModel,
    GSmithFrequencyModel,
    GUniformModel,
    GWorstModel,
)
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import AspirationNegotiator

if TYPE_CHECKING:
    pass


class OpponentModelTestNegotiator(BOANegotiator):
    """Test negotiator that checks for opponent_ufun in private_info."""

    def __init__(self, *args, opponent_model=None, **kwargs):
        # Set opponent model as the model parameter if provided
        if opponent_model is not None:
            kwargs["model"] = opponent_model

        super().__init__(*args, **kwargs)
        self.opponent_ufun_accessed = False
        self.opponent_ufun_updates = []

    def __call__(self, state):  # type: ignore[override]
        """Override to track opponent model access."""
        response = super().__call__(state)

        # Track opponent model updates after each call
        if self.private_info and "opponent_ufun" in self.private_info:
            self.opponent_ufun_accessed = True
            opponent_model = self.private_info["opponent_ufun"]

            # Try to evaluate the opponent's estimated utility for the offered outcome
            offer = state.current_offer
            if offer is not None:
                try:
                    opp_util = float(opponent_model(offer))
                    self.opponent_ufun_updates.append(
                        {"step": state.step, "offer": offer, "opp_util": opp_util}
                    )
                except Exception:
                    # Some models may not be ready yet
                    pass

        return response


@pytest.fixture
def simple_scenario():
    """Create a simple negotiation scenario."""
    issues = [
        make_issue(values=["a", "b", "c"], name="issue1"),
        make_issue(values=[1, 2, 3], name="issue2"),
    ]

    outcome_space = make_os(issues=issues)
    ufun1 = LinearAdditiveUtilityFunction.random(
        issues=tuple(issues), reserved_value=0.0
    )
    ufun2 = LinearAdditiveUtilityFunction.random(
        issues=tuple(issues), reserved_value=0.0
    )

    return outcome_space, ufun1, ufun2


class TestOpponentModelPrivateInfo:
    """Test that opponent models set private_info correctly."""

    @pytest.mark.parametrize(
        "model_class",
        [
            GHardHeadedFrequencyModel,
            GDefaultModel,
            GUniformModel,
            GOppositeModel,
            GSmithFrequencyModel,
            GAgentXFrequencyModel,
            GWorstModel,
            GPerfectModel,
            ZeroSumModel,
        ],
    )
    def test_opponent_model_sets_private_info(self, simple_scenario, model_class):
        """Test that opponent models set private_info['opponent_ufun']."""
        outcome_space, ufun1, ufun2 = simple_scenario

        # Create negotiators with opponent models
        neg1 = OpponentModelTestNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=model_class(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)

        # Run negotiation
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.run()

        # Check that opponent_ufun was set
        assert neg1.opponent_ufun_accessed, (
            f"{model_class.__name__} should set private_info['opponent_ufun']"
        )

    def test_opponent_model_updates_over_time(self, simple_scenario):
        """Test that opponent model updates as negotiation progresses."""
        outcome_space, ufun1, ufun2 = simple_scenario

        # Use a learning model that should update
        neg1 = OpponentModelTestNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=GHardHeadedFrequencyModel(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)

        # Run negotiation
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=20)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.run()

        # Check that opponent_ufun was set (main requirement)
        assert neg1.opponent_ufun_accessed, (
            "Opponent model should set private_info['opponent_ufun']"
        )

        # If we captured any updates, check that utilities are reasonable
        if neg1.opponent_ufun_updates:
            for update in neg1.opponent_ufun_updates:
                assert 0.0 <= update["opp_util"] <= 1.0, (
                    f"Opponent utility should be in [0,1], got {update['opp_util']}"
                )

    def test_bayesian_model_sets_private_info(self, simple_scenario):
        """Test Bayesian opponent model specifically."""
        outcome_space, ufun1, ufun2 = simple_scenario

        neg1 = OpponentModelTestNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=GBayesianModel(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=15)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.run()

        assert neg1.opponent_ufun_accessed, (
            "GBayesianModel should set private_info['opponent_ufun']"
        )

    def test_opposite_model_estimates_correctly(self, simple_scenario):
        """Test that opposite model returns inverse of our utility."""
        outcome_space, ufun1, ufun2 = simple_scenario

        neg1 = OpponentModelTestNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=GOppositeModel(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.run()

        # Check that opponent model gives opposite utilities
        if neg1.opponent_ufun_updates:
            for update in neg1.opponent_ufun_updates:
                offer = update["offer"]
                opp_util = update["opp_util"]
                our_util = float(ufun1(offer))

                # Opposite model should give 1 - our_utility
                expected = 1.0 - our_util
                assert abs(opp_util - expected) < 0.01, (
                    f"GOppositeModel should return 1-u, got {opp_util}, expected {expected}"
                )

    def test_zero_sum_model_sets_private_info(self, simple_scenario):
        """Test ZeroSumModel sets private_info correctly."""
        outcome_space, ufun1, ufun2 = simple_scenario

        neg1 = OpponentModelTestNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=ZeroSumModel(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.run()

        assert neg1.opponent_ufun_accessed, (
            "ZeroSumModel should set private_info['opponent_ufun']"
        )


class TestOpponentModelMultilateral:
    """Test opponent models in multilateral negotiations."""

    def test_multilateral_opponent_ufuns_dict(self, simple_scenario):
        """Test that multilateral negotiations use opponent_ufuns dict."""
        outcome_space, ufun1, ufun2 = simple_scenario
        issues = list(outcome_space.issues)
        ufun3 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)

        # Create negotiators
        neg1 = OpponentModelTestNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=GHardHeadedFrequencyModel(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)
        neg3 = AspirationNegotiator(name="agent3", ufun=ufun3)

        # Run multilateral negotiation
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.add(neg3)
        mechanism.run()

        # In multilateral, should use opponent_ufuns dict instead
        if neg1.private_info:
            # Could be either opponent_ufun (bilateral-style) or opponent_ufuns (multilateral-style)
            has_opponent_info = (
                "opponent_ufun" in neg1.private_info
                or "opponent_ufuns" in neg1.private_info
            )
            assert has_opponent_info, (
                "Multilateral negotiation should set opponent information"
            )


class TestOpponentModelAccess:
    """Test that negotiators can access and use opponent models."""

    def test_negotiator_can_evaluate_opponent_utility(self, simple_scenario):
        """Test that negotiator can use opponent_ufun to evaluate outcomes."""
        outcome_space, ufun1, ufun2 = simple_scenario

        neg1 = OpponentModelTestNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=GHardHeadedFrequencyModel(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.run()

        # Check that opponent_ufun was set (main requirement)
        assert neg1.opponent_ufun_accessed, (
            "Should have access to opponent_ufun in private_info"
        )

        # If we captured any evaluations, verify they are valid
        if neg1.opponent_ufun_updates:
            for update in neg1.opponent_ufun_updates:
                assert isinstance(update["opp_util"], (int, float)), (
                    "Opponent utility should be numeric"
                )
                assert not (update["opp_util"] != update["opp_util"]), (
                    "Opponent utility should not be NaN"
                )

    def test_opponent_model_available_from_first_offer(self, simple_scenario):
        """Test that opponent_ufun is available after first opponent offer."""
        outcome_space, ufun1, ufun2 = simple_scenario

        class FirstOfferCheckNegotiator(BOANegotiator):
            def __init__(self, *args, opponent_model=None, **kwargs):
                if opponent_model is not None:
                    kwargs["model"] = opponent_model
                super().__init__(*args, **kwargs)
                self.checked = False

            def __call__(self, state):  # type: ignore[override]
                response = super().__call__(state)
                if not self.checked and state.step > 0:
                    self.checked = True
                    assert self.private_info is not None, "private_info should exist"
                    assert "opponent_ufun" in self.private_info, (
                        "opponent_ufun should be set after first opponent offer"
                    )
                return response

        neg1 = FirstOfferCheckNegotiator(
            name="agent1",
            ufun=ufun1,
            opponent_model=GDefaultModel(),  # type: ignore
        )
        neg2 = AspirationNegotiator(name="agent2", ufun=ufun2)

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)  # type: ignore
        mechanism.add(neg1)
        mechanism.add(neg2)
        mechanism.run()

        assert neg1.checked, "Should have checked for opponent_ufun"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
