"""Tests for standard scenario information calculation."""

from __future__ import annotations

import pytest

from negmas import make_issue, make_os
from negmas.inout import Scenario
from negmas.preferences import LinearUtilityFunction
from negmas.preferences.ops import (
    calc_standard_info,
    calc_scenario_stats,
    rational_fraction,
)


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
    """Test with calc_rational=False."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    info = calc_standard_info([u1, u2], outcome_space=os, calc_rational=False)

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

    # Only outcomes where BOTH utilities > reserved value are rational
    # u1(x) = x, u2(x) = x
    # Rational: u1(x) > 1.0 AND u2(x) > 2.0
    # So x must be > 2.0, which means x in {3, 4}
    # Rational fraction should be 2/5 = 0.4
    assert info["rational_fraction"] == pytest.approx(0.4, abs=0.01)


def test_calc_standard_info_rational_requires_all():
    """Test that rational fraction requires ALL negotiators to have positive utility."""
    issues = [make_issue([0, 1, 2, 3, 4, 5], "x")]
    os = make_os(issues)

    # u1: utility = x, reserved = 1.0  -> rational when x > 1 (x in {2,3,4,5})
    # u2: utility = x, reserved = 3.0  -> rational when x > 3 (x in {4,5})
    # u3: utility = x, reserved = 4.0  -> rational when x > 4 (x in {5})
    # Rational for ALL: x > 4 -> only x=5 is rational
    # Rational fraction should be 1/6 ≈ 0.167

    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=3.0)
    u3 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=4.0)

    info = calc_standard_info([u1, u2, u3], outcome_space=os)

    # Only x=5 satisfies all three conditions
    assert info["rational_fraction"] == pytest.approx(1 / 6, abs=0.01)


def test_calc_standard_info_rational_different_ufuns():
    """Test rational fraction with different utility functions."""
    issues = [make_issue([0, 1, 2, 3, 4], "x")]
    os = make_os(issues)

    # u1: utility = 2*x, reserved = 4.0  -> rational when 2*x > 4 -> x > 2 (x in {3,4})
    # u2: utility = x, reserved = 2.0    -> rational when x > 2 (x in {3,4})
    # Both rational: x > 2 -> x in {3,4}
    # Rational fraction should be 2/5 = 0.4

    u1 = LinearUtilityFunction(weights=[2.0], outcome_space=os, reserved_value=4.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=2.0)

    info = calc_standard_info([u1, u2], outcome_space=os)

    assert info["rational_fraction"] == pytest.approx(0.4, abs=0.01)


def test_calc_standard_info_no_rational_outcomes():
    """Test when no outcomes are rational for all negotiators."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)

    # u1: utility = x, reserved = 10.0  -> never rational (max utility is 2)
    # u2: utility = x, reserved = 1.0   -> rational for x in {2}
    # Both rational: none (u1 is never satisfied)
    # Rational fraction should be 0.0

    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=10.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.0)

    info = calc_standard_info([u1, u2], outcome_space=os)

    assert info["rational_fraction"] == 0.0


def test_calc_standard_info_all_outcomes_rational():
    """Test when all outcomes are rational for all negotiators."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)

    # u1: utility = x, reserved = -1.0  -> always rational (all x >= 0 > -1)
    # u2: utility = x, reserved = -1.0  -> always rational
    # Both rational: all outcomes
    # Rational fraction should be 1.0

    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=-1.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=-1.0)

    info = calc_standard_info([u1, u2], outcome_space=os)

    assert info["rational_fraction"] == 1.0


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


def test_scenario_stats_to_dict_pareto_control():
    """Test ScenarioStats.to_dict() with granular pareto frontier control."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[0.5], outcome_space=os)

    stats = calc_scenario_stats([u1, u2])

    # Test include_pareto_frontier=True (default)
    d = stats.to_dict(include_pareto_frontier=True)
    assert len(d["pareto_utils"]) > 0
    assert len(d["pareto_outcomes"]) > 0

    # Test include_pareto_frontier=False
    d = stats.to_dict(include_pareto_frontier=False)
    assert len(d["pareto_utils"]) == 0
    assert len(d["pareto_outcomes"]) == 0

    # Test include_pareto_utils=False only
    d = stats.to_dict(include_pareto_utils=False, include_pareto_outcomes=True)
    assert len(d["pareto_utils"]) == 0
    assert len(d["pareto_outcomes"]) > 0

    # Test include_pareto_outcomes=False only
    d = stats.to_dict(include_pareto_utils=True, include_pareto_outcomes=False)
    assert len(d["pareto_utils"]) > 0
    assert len(d["pareto_outcomes"]) == 0

    # Test explicit parameters override include_pareto_frontier
    d = stats.to_dict(
        include_pareto_frontier=False,
        include_pareto_utils=True,
        include_pareto_outcomes=True,
    )
    assert len(d["pareto_utils"]) > 0
    assert len(d["pareto_outcomes"]) > 0


# Tests specifically for rational_fraction() function


def test_rational_fraction_basic():
    """Test basic calculation of rational fraction."""
    issues = [make_issue([0, 1, 2, 3, 4], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=2.0)

    frac = rational_fraction([u1, u2], outcome_space=os)

    # Only outcomes where BOTH utilities > reserved value are rational
    # u1(x) = x, u2(x) = x
    # Rational: u1(x) > 1.0 AND u2(x) > 2.0
    # So x must be > 2.0, which means x in {3, 4}
    # Rational fraction should be 2/5 = 0.4
    assert frac == pytest.approx(0.4, abs=0.01)


def test_rational_fraction_requires_all():
    """Test that rational fraction requires ALL negotiators to have positive utility."""
    issues = [make_issue([0, 1, 2, 3, 4, 5], "x")]
    os = make_os(issues)

    # u1: utility = x, reserved = 1.0  -> rational when x > 1 (x in {2,3,4,5})
    # u2: utility = x, reserved = 3.0  -> rational when x > 3 (x in {4,5})
    # u3: utility = x, reserved = 4.0  -> rational when x > 4 (x in {5})
    # Rational for ALL: x > 4 -> only x=5 is rational
    # Rational fraction should be 1/6 ≈ 0.167

    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=3.0)
    u3 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=4.0)

    frac = rational_fraction([u1, u2, u3], outcome_space=os)

    # Only x=5 satisfies all three conditions
    assert frac == pytest.approx(1 / 6, abs=0.01)


def test_rational_fraction_different_ufuns():
    """Test rational fraction with different utility functions."""
    issues = [make_issue([0, 1, 2, 3, 4], "x")]
    os = make_os(issues)

    # u1: utility = 2*x, reserved = 4.0  -> rational when 2*x > 4 -> x > 2 (x in {3,4})
    # u2: utility = x, reserved = 2.0    -> rational when x > 2 (x in {3,4})
    # Both rational: x > 2 -> x in {3,4}
    # Rational fraction should be 2/5 = 0.4

    u1 = LinearUtilityFunction(weights=[2.0], outcome_space=os, reserved_value=4.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=2.0)

    frac = rational_fraction([u1, u2], outcome_space=os)

    assert frac == pytest.approx(0.4, abs=0.01)


def test_rational_fraction_no_rational_outcomes():
    """Test when no outcomes are rational for all negotiators."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)

    # u1: utility = x, reserved = 10.0  -> never rational (max utility is 2)
    # u2: utility = x, reserved = 1.0   -> rational for x in {2}
    # Both rational: none (u1 is never satisfied)
    # Rational fraction should be 0.0

    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=10.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.0)

    frac = rational_fraction([u1, u2], outcome_space=os)

    assert frac == 0.0


def test_rational_fraction_all_rational():
    """Test when all outcomes are rational for all negotiators."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)

    # u1: utility = x, reserved = -1.0  -> always rational (all x >= 0 > -1)
    # u2: utility = x, reserved = -1.0  -> always rational
    # Both rational: all outcomes
    # Rational fraction should be 1.0

    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=-1.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=-1.0)

    frac = rational_fraction([u1, u2], outcome_space=os)

    assert frac == 1.0


def test_rational_fraction_with_explicit_outcomes():
    """Test providing explicit outcomes list."""
    issues = [make_issue([0, 1, 2, 3, 4], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.5)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=1.5)

    # Provide subset of outcomes
    outcomes = [(2,), (3,), (4,)]

    frac = rational_fraction([u1, u2], outcomes=outcomes, outcome_space=os)

    # u1(x) > 1.5 AND u2(x) > 1.5 means x > 1.5, so x in {2,3,4}
    # All provided outcomes satisfy this
    # Rational fraction should be 3/5 = 0.6 (still uses cardinality from outcome_space)
    assert frac == pytest.approx(0.6, abs=0.01)


def test_rational_fraction_infer_outcome_space():
    """Test that outcome space is inferred from first ufun."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=-1.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=-1.0)

    # Don't provide outcome_space explicitly
    frac = rational_fraction([u1, u2])

    # All outcomes should be rational
    assert frac == 1.0


def test_rational_fraction_error_empty_ufuns():
    """Test that error is raised with empty ufuns list."""
    issues = [make_issue([0, 1, 2], "x")]
    os = make_os(issues)

    with pytest.raises(ValueError, match="Must pass the ufuns"):
        rational_fraction([], outcome_space=os)


def test_rational_fraction_error_no_outcome_space():
    """Test that error is raised when no outcome space is available."""
    u1 = LinearUtilityFunction(weights=[1.0])
    u2 = LinearUtilityFunction(weights=[0.5])

    with pytest.raises(ValueError, match="outcome space"):
        rational_fraction([u1, u2])


def test_rational_fraction_multi_issue():
    """Test with multiple issues."""
    issues = [make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
    os = make_os(issues)

    # u1: utility = x + 0.5*y, reserved = 1.5
    # u2: utility = 0.5*x + y, reserved = 1.0
    u1 = LinearUtilityFunction(weights=[1.0, 0.5], outcome_space=os, reserved_value=1.5)
    u2 = LinearUtilityFunction(weights=[0.5, 1.0], outcome_space=os, reserved_value=1.0)

    frac = rational_fraction([u1, u2], outcome_space=os)

    # Should return a value between 0 and 1
    assert 0.0 <= frac <= 1.0


def test_rational_fraction_four_negotiators():
    """Test with four negotiators to ensure it handles multiple parties."""
    issues = [make_issue([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "x")]
    os = make_os(issues)

    # All negotiators have same utility function
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=3.0)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=4.0)
    u3 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=5.0)
    u4 = LinearUtilityFunction(weights=[1.0], outcome_space=os, reserved_value=6.0)

    frac = rational_fraction([u1, u2, u3, u4], outcome_space=os)

    # All must be satisfied: x > 6.0 -> x in {7,8,9}
    # Rational fraction should be 3/10 = 0.3
    assert frac == pytest.approx(0.3, abs=0.01)


def test_rational_fraction_default_reserved_values():
    """Test with default reserved values (should be 0.0 or similar)."""
    issues = [make_issue([0, 1, 2, 3, 4], "x")]
    os = make_os(issues)

    # Create ufuns without explicitly setting reserved_value
    u1 = LinearUtilityFunction(weights=[1.0], outcome_space=os)
    u2 = LinearUtilityFunction(weights=[1.0], outcome_space=os)

    frac = rational_fraction([u1, u2], outcome_space=os)

    # Should return a value between 0 and 1
    assert 0.0 <= frac <= 1.0
