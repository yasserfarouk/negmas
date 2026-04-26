"""Test common_range parameter for Scenario.normalize()."""

from __future__ import annotations

import pytest
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearUtilityFunction


def make_test_scenario():
    """Create a simple test scenario with two agents."""
    issues = (make_issue(10, "x"), make_issue(5, "y"))

    # Create linear utility functions with different preferences
    ufun1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
    ufun2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

    return Scenario(
        outcome_space=make_os(issues, name="TestScenario"), ufuns=(ufun1, ufun2)
    )

    # Create ufuns with different ranges
    ufun1 = U.random(issues=issues, reserved_value=0.0, normalized=False)
    ufun2 = U.random(issues=issues, reserved_value=0.0, normalized=False)

    return Scenario(
        outcome_space=make_os(issues, name="TestScenario"), ufuns=(ufun1, ufun2)
    )


def test_common_range_true_normalization():
    """Test that common_range=True normalizes to a common scale."""
    scenario = make_test_scenario()

    # Normalize with common range
    scenario.normalize(common_range=True)

    # Check that it's normalized with common scale
    assert scenario.is_normalized((0.0, 1.0), common_range=True)

    # At least one ufun should reach 0 and at least one should reach 1
    mins = [u.minmax()[0] for u in scenario.ufuns]
    maxs = [u.minmax()[1] for u in scenario.ufuns]

    assert min(mins) < 0.01  # At least one reaches min
    assert max(maxs) > 0.99  # At least one reaches max
    assert all(0.0 <= m <= 1.0 for m in mins)  # All within range
    assert all(0.0 <= m <= 1.0 for m in maxs)  # All within range


def test_common_range_false_normalization():
    """Test that common_range=False normalizes each ufun independently."""
    scenario = make_test_scenario()

    # Normalize independently
    scenario.normalize(common_range=False)

    # Check that it's normalized independently
    assert scenario.is_normalized((0.0, 1.0), common_range=False)

    # Each ufun should span the full range [0, 1]
    for ufun in scenario.ufuns:
        mn, mx = ufun.minmax()
        assert abs(mn - 0.0) < 0.01
        assert abs(mx - 1.0) < 0.01


def test_common_range_default_is_true():
    """Test that the default behavior is common_range=True."""
    scenario1 = make_test_scenario()
    scenario2 = make_test_scenario()

    # Normalize with default (should be common_range=True)
    scenario1.normalize()

    # Normalize explicitly with common_range=True
    scenario2.normalize(common_range=True)

    # Both should be normalized with common scale
    assert scenario1.is_normalized((0.0, 1.0), common_range=True)
    assert scenario2.is_normalized((0.0, 1.0), common_range=True)


def test_independent_parameter_deprecated_but_works():
    """Test that the old 'independent' parameter still works but shows deprecation warning."""
    scenario = make_test_scenario()

    # Using independent parameter should work but raise a deprecation warning
    with pytest.warns(match="independent.*deprecated"):
        scenario.normalize(independent=True)

    # Should be normalized independently
    assert scenario.is_normalized((0.0, 1.0), common_range=False)


def test_cannot_specify_both_parameters():
    """Test that specifying both parameters raises an error."""
    scenario = make_test_scenario()

    with pytest.raises(ValueError, match="Cannot specify both"):
        scenario.normalize(independent=True, common_range=False)

    with pytest.raises(ValueError, match="Cannot specify both"):
        scenario.is_normalized(independent=True, common_range=False)


def test_common_range_true_vs_independent_false():
    """Test that common_range=True is equivalent to independent=False."""
    scenario1 = make_test_scenario()
    scenario2 = make_test_scenario()

    # Normalize one with common_range=True
    scenario1.normalize(common_range=True)

    # Normalize other with independent=False (deprecated way)
    with pytest.warns(match="independent.*deprecated"):
        scenario2.normalize(independent=False)

    # Both should be normalized with common scale
    assert scenario1.is_normalized((0.0, 1.0), common_range=True)
    assert scenario2.is_normalized((0.0, 1.0), common_range=True)

    # Check that both have similar min/max values
    mins1 = [u.minmax()[0] for u in scenario1.ufuns]
    mins2 = [u.minmax()[0] for u in scenario2.ufuns]
    maxs1 = [u.minmax()[1] for u in scenario1.ufuns]
    maxs2 = [u.minmax()[1] for u in scenario2.ufuns]

    # Not necessarily identical due to randomness, but should both reach bounds
    assert min(mins1) < 0.01 and min(mins2) < 0.01
    assert max(maxs1) > 0.99 and max(maxs2) > 0.99


def test_common_range_false_vs_independent_true():
    """Test that common_range=False is equivalent to independent=True."""
    scenario1 = make_test_scenario()
    scenario2 = make_test_scenario()

    # Set the same seed for reproducibility
    import random

    random.seed(42)
    scenario1 = make_test_scenario()

    random.seed(42)
    scenario2 = make_test_scenario()

    # Normalize one with common_range=False
    scenario1.normalize(common_range=False)

    # Normalize other with independent=True (deprecated way)
    with pytest.warns(match="independent.*deprecated"):
        scenario2.normalize(independent=True)

    # Both should be normalized independently
    assert scenario1.is_normalized((0.0, 1.0), common_range=False)
    assert scenario2.is_normalized((0.0, 1.0), common_range=False)

    # Each ufun should span full range in both scenarios
    for ufun in scenario1.ufuns:
        mn, mx = ufun.minmax()
        assert abs(mn - 0.0) < 0.01
        assert abs(mx - 1.0) < 0.01

    for ufun in scenario2.ufuns:
        mn, mx = ufun.minmax()
        assert abs(mn - 0.0) < 0.01
        assert abs(mx - 1.0) < 0.01


def test_is_normalized_common_range_parameter():
    """Test is_normalized with common_range parameter."""
    scenario = make_test_scenario()

    # Not normalized initially
    assert not scenario.is_normalized((0.0, 1.0), common_range=True)
    assert not scenario.is_normalized((0.0, 1.0), common_range=False)

    # Normalize with common range
    scenario.normalize(common_range=True)

    # Should pass common range check
    assert scenario.is_normalized((0.0, 1.0), common_range=True)

    # Should not pass independent check (not all ufuns span full range)
    assert not scenario.is_normalized((0.0, 1.0), common_range=False)

    # Normalize independently
    scenario.normalize(common_range=False)

    # Should pass independent check
    assert scenario.is_normalized((0.0, 1.0), common_range=False)

    # Should also pass common range check (independent is a subset of common range)
    assert scenario.is_normalized((0.0, 1.0), common_range=True)


def test_common_range_with_custom_bounds():
    """Test common_range parameter with custom normalization bounds."""
    scenario = make_test_scenario()

    # Normalize to custom range with common scale
    scenario.normalize(to=(0.5, 2.0), common_range=True)

    # Check normalization
    assert scenario.is_normalized((0.5, 2.0), common_range=True)

    mins = [u.minmax()[0] for u in scenario.ufuns]
    maxs = [u.minmax()[1] for u in scenario.ufuns]

    assert min(mins) < 0.51  # At least one reaches min
    assert max(maxs) > 1.99  # At least one reaches max
    assert all(0.5 <= m <= 2.0 for m in mins)
    assert all(0.5 <= m <= 2.0 for m in maxs)


def test_common_range_false_with_custom_bounds():
    """Test common_range=False with custom normalization bounds."""
    scenario = make_test_scenario()

    # Normalize to custom range independently
    scenario.normalize(to=(0.5, 2.0), common_range=False)

    # Check normalization
    assert scenario.is_normalized((0.5, 2.0), common_range=False)

    # Each ufun should span [0.5, 2.0]
    for ufun in scenario.ufuns:
        mn, mx = ufun.minmax()
        assert abs(mn - 0.5) < 0.01
        assert abs(mx - 2.0) < 0.01


if __name__ == "__main__":
    # Run tests manually
    print("Testing common_range=True normalization...")
    test_common_range_true_normalization()
    print("✓ Pass")

    print("\nTesting common_range=False normalization...")
    test_common_range_false_normalization()
    print("✓ Pass")

    print("\nTesting default is common_range=True...")
    test_common_range_default_is_true()
    print("✓ Pass")

    print("\nTesting independent parameter deprecated but works...")
    test_independent_parameter_deprecated_but_works()
    print("✓ Pass")

    print("\nTesting cannot specify both parameters...")
    test_cannot_specify_both_parameters()
    print("✓ Pass")

    print("\nTesting common_range=True vs independent=False...")
    test_common_range_true_vs_independent_false()
    print("✓ Pass")

    print("\nTesting common_range=False vs independent=True...")
    test_common_range_false_vs_independent_true()
    print("✓ Pass")

    print("\nTesting is_normalized with common_range parameter...")
    test_is_normalized_common_range_parameter()
    print("✓ Pass")

    print("\nTesting common_range with custom bounds...")
    test_common_range_with_custom_bounds()
    print("✓ Pass")

    print("\nTesting common_range=False with custom bounds...")
    test_common_range_false_with_custom_bounds()
    print("✓ Pass")

    print("\n✅ All tests passed!")
