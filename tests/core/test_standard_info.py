"""Tests for standard scenario information calculation."""

from __future__ import annotations

import pytest

from negmas import make_issue, make_os
from negmas.inout import Scenario
from negmas.preferences import LinearUtilityFunction
from negmas.preferences.ops import calc_standard_info


def test_calc_standard_info_basic():
    """Test basic calculation of standard info."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os)

    assert info["n_negotiators"] == 2
    assert info["n_outcomes"] == 3
    assert info["n_issues"] == 1
    assert "rational_fraction" in info
    assert "opposition_level" in info


def test_calc_standard_info_multi_issue():
    """Test with multiple issues."""
    issues = [make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0, 0.5], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5, 1.0], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os)

    assert info["n_negotiators"] == 2
    assert info["n_outcomes"] == 6  # 3 * 2
    assert info["n_issues"] == 2


def test_calc_standard_info_no_rational_fraction():
    """Test with calc_rational_fraction=False."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os, calc_rational_fraction=False)

    assert info["n_negotiators"] == 2
    assert info["n_outcomes"] == 3
    assert info["n_issues"] == 1
    assert "rational_fraction" not in info
    assert "opposition_level" in info


def test_calc_standard_info_with_reserved_values():
    """Test rational fraction calculation with reserved values."""
    issues = [make_issue([0, 1, 2, 3, 4], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=2.0)

    info = calc_standard_info([u1, u2], outcome_space=os)

    # Only outcomes where both utilities > reserved value are rational
    # u1(x) = x, u2(x) = x
    # Rational: u1(x) > 1.0 AND u2(x) > 2.0
    # So x must be > 2.0, which means x in {3, 4}
    # Rational fraction should be 2/5 = 0.4
    assert info["rational_fraction"] == pytest.approx(0.4, abs=0.01)


def test_scenario_calc_standard_info():
    """Test Scenario.calc_standard_info method."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    scenario = Scenario(outcome_space=os, ufuns=(u1, u2))

    # Initially info should be empty or not contain standard info
    assert "n_negotiators" not in scenario.info

    # Calculate standard info
    info = scenario.calc_standard_info()

    # Check return value
    assert info["n_negotiators"] == 2
    assert info["n_outcomes"] == 3
    assert info["n_issues"] == 1

    # Check that scenario.info was updated
    assert scenario.info["n_negotiators"] == 2
    assert scenario.info["n_outcomes"] == 3
    assert scenario.info["n_issues"] == 1
    assert "rational_fraction" in scenario.info
    assert "opposition_level" in scenario.info


def test_calc_standard_info_zero_sum():
    """Test opposition level for zero-sum scenario."""
    from negmas.preferences.crisp.mapping import MappingUtilityFunction

    issues = [make_issue(list(range(10)), "x")]
    os = make_os(issues)

    # Zero-sum utilities
    u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
    u2 = MappingUtilityFunction(lambda x: 9 - x[0], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os)

    # Zero-sum scenarios should have high opposition level
    assert info["opposition_level"] is not None
    assert info["opposition_level"] > 0.5


def test_calc_standard_info_same_ufun():
    """Test opposition level when both negotiators have same utility."""
    from negmas.preferences.crisp.mapping import MappingUtilityFunction

    issues = [make_issue(list(range(10)), "x")]
    os = make_os(issues)

    # Same utilities
    u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
    u2 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os)

    # Same utilities should have opposition level close to 0
    assert info["opposition_level"] is not None
    assert info["opposition_level"] == pytest.approx(0.0, abs=0.01)


def test_calc_standard_info_error_no_outcome_space():
    """Test that error is raised when no outcome space is available."""
    u1 = LinearUtilityFunction(weights=[1.0])
    u2 = LinearUtilityFunction(weights=[0.5])

    with pytest.raises(ValueError, match="outcome space"):
        calc_standard_info([u1, u2])


def test_calc_standard_info_error_empty_ufuns():
    """Test that error is raised with empty ufuns list."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)

    with pytest.raises(ValueError, match="Must pass the ufuns"):
        calc_standard_info([], outcome_space=os)


def test_calc_standard_info_error_mismatched_outcome_spaces():
    """Test that error is raised when ufuns have different outcome spaces."""
    issues1 = [make_issue([0, 1, 2], "x")]
    issues2 = [make_issue([0, 1, 2, 3], "y")]
    os1 = make_os(issues1)
    os2 = make_os(issues2)

    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os1)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os2)

    with pytest.raises(ValueError, match="different outcome space"):
        calc_standard_info([u1, u2], outcome_space=os1)


def test_scenario_calc_standard_info_preserves_existing_info():
    """Test that calc_standard_info preserves existing info dict entries."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    scenario = Scenario(
        outcome_space=os,
        ufuns=(u1, u2),
        info={"custom_key": "custom_value", "another_key": 42},
    )

    # Calculate standard info
    scenario.calc_standard_info()

    # Check that existing keys are preserved
    assert scenario.info["custom_key"] == "custom_value"
    assert scenario.info["another_key"] == 42

    # Check that new keys are added
    assert "n_negotiators" in scenario.info
    assert "n_outcomes" in scenario.info


def test_calc_standard_info_large_outcome_space():
    """Test with a larger outcome space to ensure it handles size reasonably."""
    issues = [
        make_issue(list(range(10)), "x"),
        make_issue(list(range(10)), "y"),
        make_issue(list(range(10)), "z"),
    ]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0, 0.5, 0.3], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.3, 0.5, 1.0], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os)

    assert info["n_negotiators"] == 2
    assert info["n_outcomes"] == 1000  # 10 * 10 * 10
    assert info["n_issues"] == 3
    assert "rational_fraction" in info
    assert "opposition_level" in info


def test_calc_standard_info_opposition_level_failure_handling():
    """Test that failures in opposition_level calculation are handled gracefully."""
    # This test uses a pathological case that might cause opposition_level to fail
    issues = [make_issue([0], "x")]  # Single outcome
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os)

    # Should still return basic info even if opposition fails
    assert info["n_negotiators"] == 2
    assert info["n_outcomes"] == 1
    assert info["n_issues"] == 1
    # opposition_level might be None or a valid value depending on implementation
    assert "opposition_level" in info


def test_calc_standard_info_with_explicit_outcomes():
    """Test providing explicit outcomes list."""
    issues = [make_issue([0, 1, 2, 3, 4], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    # Provide subset of outcomes
    outcomes = [(0,), (1,), (2,)]

    info = calc_standard_info([u1, u2], outcome_space=os, outcomes=outcomes)

    # n_outcomes should still be the full cardinality, not len(outcomes)
    assert info["n_outcomes"] == 5
    assert info["n_negotiators"] == 2
    # But rational_fraction and opposition_level use the provided outcomes
    assert "rational_fraction" in info
