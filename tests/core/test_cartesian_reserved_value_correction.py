"""Tests for reserved value correction in cartesian tournaments."""

from __future__ import annotations

import math
import warnings


from negmas.outcomes import make_issue, make_os
from negmas.inout import Scenario
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.tournaments.neg.simple.cartesian import _check_and_correct_reserved_values
from negmas.warnings import NegmasUnexpectedValueWarning


class TestReservedValueCorrection:
    """Test suite for reserved value correction in cartesian tournaments."""

    def test_correction_with_minus_inf_default_eps(self):
        """Test that -inf reserved value is corrected to ufun.min() - 0.0 by default."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os)
        u1.reserved_value = float("-inf")
        u2.reserved_value = 0.5

        scenario = Scenario(outcome_space=os, ufuns=[u1, u2])

        # Correction should happen and warning should be raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, "test_scenario")
            assert len(w) == 1
            assert issubclass(w[0].category, NegmasUnexpectedValueWarning)
            assert "corrected" in str(w[0].message).lower()
            assert "-inf" in str(w[0].message)

        # Reserved value should be corrected to min - 0.0 (i.e., just min)
        expected = float(u1.min()) - 0.0
        assert u1.reserved_value == expected
        assert math.isfinite(u1.reserved_value)

        # u2 should not be affected
        assert u2.reserved_value == 0.5

    def test_correction_with_inf(self):
        """Test that +inf reserved value is corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("inf")

        scenario = Scenario(outcome_space=os, ufuns=[u1])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, "test_scenario", eps=0.01)
            assert len(w) == 1

        expected = float(u1.min()) - 0.01
        assert u1.reserved_value == expected

    def test_correction_with_nan(self):
        """Test that NaN reserved value is corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("nan")

        scenario = Scenario(outcome_space=os, ufuns=[u1])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, "test_scenario", eps=0.001)
            assert len(w) == 1
            assert "nan" in str(w[0].message).lower()

        expected = float(u1.min()) - 0.001
        assert u1.reserved_value == expected
        assert math.isfinite(u1.reserved_value)

    def test_correction_with_custom_eps(self):
        """Test that custom eps value is used correctly."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("-inf")

        scenario = Scenario(outcome_space=os, ufuns=[u1])

        eps = 0.05
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, "test_scenario", eps=eps)
            assert len(w) == 1
            assert str(eps) in str(w[0].message)

        expected = float(u1.min()) - eps
        assert u1.reserved_value == expected

    def test_multiple_problematic_ufuns(self):
        """Test correction when multiple ufuns have problematic reserved values."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os)
        u3 = MappingUtilityFunction(lambda x: x[0] / 2, outcome_space=os)

        u1.reserved_value = float("-inf")
        u2.reserved_value = float("nan")
        u3.reserved_value = 0.5  # Normal value

        scenario = Scenario(outcome_space=os, ufuns=[u1, u2, u3])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, "test_scenario", eps=0.01)
            assert len(w) == 1
            # Warning should mention both corrected ufuns
            msg = str(w[0].message)
            assert "ufun[0]" in msg
            assert "ufun[1]" in msg
            assert "ufun[2]" not in msg  # u3 should not be mentioned

        # Check corrections
        assert u1.reserved_value == float(u1.min()) - 0.01
        assert u2.reserved_value == float(u2.min()) - 0.01
        assert u3.reserved_value == 0.5  # Unchanged

    def test_no_warning_for_normal_reserved_values(self):
        """Test that normal reserved values don't trigger warnings or corrections."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os)

        u1.reserved_value = 0.3
        u2.reserved_value = 0.8

        scenario = Scenario(outcome_space=os, ufuns=[u1, u2])

        old_rv1 = u1.reserved_value
        old_rv2 = u2.reserved_value

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, "test_scenario")
            assert len(w) == 0

        # Values should be unchanged
        assert u1.reserved_value == old_rv1
        assert u2.reserved_value == old_rv2

    def test_correction_with_none_reserved_value(self):
        """Test that None reserved values ARE corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = None

        scenario = Scenario(outcome_space=os, ufuns=[u1])

        # None should be corrected and warning should be raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, "test_scenario", eps=0.01)
            assert len(w) == 1
            assert "None" in str(w[0].message)

        # Reserved value should be corrected
        expected = float(u1.min()) - 0.01
        assert u1.reserved_value == expected
        assert math.isfinite(u1.reserved_value)

    def test_warning_message_includes_scenario_name(self):
        """Test that warning message includes the scenario name."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("-inf")

        scenario = Scenario(outcome_space=os, ufuns=[u1])

        scenario_name = "my_special_scenario"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_and_correct_reserved_values(scenario, scenario_name)
            assert len(w) == 1
            assert scenario_name in str(w[0].message)

    def test_correction_with_zero_min(self):
        """Test correction works correctly when ufun.min() is 0."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        # u1.min() will be 0
        u1.reserved_value = float("-inf")

        scenario = Scenario(outcome_space=os, ufuns=[u1])

        assert u1.min() == 0  # Verify assumption

        eps = 0.001
        _check_and_correct_reserved_values(scenario, "test", eps=eps)

        assert u1.reserved_value == -eps

    def test_correction_with_negative_min(self):
        """Test correction works correctly when ufun.min() is negative."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        # This ufun returns negative values
        u1 = MappingUtilityFunction(lambda x: x[0] - 2, outcome_space=os)
        # u1.min() will be -2
        u1.reserved_value = float("nan")

        scenario = Scenario(outcome_space=os, ufuns=[u1])

        assert u1.min() == -2  # Verify assumption

        eps = 0.1
        _check_and_correct_reserved_values(scenario, "test", eps=eps)

        assert u1.reserved_value == -2.1  # -2 - 0.1
