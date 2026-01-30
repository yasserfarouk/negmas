"""Tests for reserved value normalization in utility functions and scenarios."""

from __future__ import annotations

import math
import warnings


from negmas.common import DEFAULT_RESERVED_VALUE_PENALTY
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import normalize
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.ops import correct_reserved_value
from negmas.warnings import NegmasUnexpectedValueWarning


class TestCorrectReservedValue:
    """Test suite for the centralized correct_reserved_value() function."""

    def test_normal_value_unchanged(self):
        """Test that normal finite values are not corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        ufun.reserved_value = 0.5

        corrected, was_corrected = correct_reserved_value(0.5, ufun, warn=False)
        assert corrected == 0.5
        assert not was_corrected

    def test_none_corrected(self):
        """Test that None reserved value is corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        corrected, was_corrected = correct_reserved_value(
            None, ufun, eps=0.0, warn=False
        )
        assert was_corrected
        assert math.isfinite(corrected)
        assert corrected == float(ufun.min()) - 0.0

    def test_inf_corrected(self):
        """Test that +inf reserved value is corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        corrected, was_corrected = correct_reserved_value(
            float("inf"), ufun, eps=0.0, warn=False
        )
        assert was_corrected
        assert math.isfinite(corrected)
        assert corrected == float(ufun.min()) - 0.0

    def test_minus_inf_corrected(self):
        """Test that -inf reserved value is corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        corrected, was_corrected = correct_reserved_value(
            float("-inf"), ufun, eps=0.0, warn=False
        )
        assert was_corrected
        assert math.isfinite(corrected)
        assert corrected == float(ufun.min()) - 0.0

    def test_nan_corrected(self):
        """Test that NaN reserved value is corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        corrected, was_corrected = correct_reserved_value(
            float("nan"), ufun, eps=0.0, warn=False
        )
        assert was_corrected
        assert math.isfinite(corrected)
        assert corrected == float(ufun.min()) - 0.0

    def test_custom_eps(self):
        """Test that custom epsilon is used in correction."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        corrected, was_corrected = correct_reserved_value(
            None, ufun, eps=0.5, warn=False
        )
        assert was_corrected
        assert corrected == float(ufun.min()) - 0.5

    def test_default_eps(self):
        """Test that DEFAULT_RESERVED_VALUE_PENALTY is used when eps=None."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        corrected, was_corrected = correct_reserved_value(
            None, ufun, eps=None, warn=False
        )
        assert was_corrected
        assert corrected == float(ufun.min()) - DEFAULT_RESERVED_VALUE_PENALTY

    def test_warning_emitted(self):
        """Test that warning is emitted when warn=True."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            correct_reserved_value(None, ufun, eps=0.0, warn=True)
            assert len(w) == 1
            assert issubclass(w[0].category, NegmasUnexpectedValueWarning)
            assert "not finite" in str(w[0].message).lower()

    def test_no_warning_when_warn_false(self):
        """Test that no warning is emitted when warn=False."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            correct_reserved_value(None, ufun, eps=0.0, warn=False)
            assert len(w) == 0


class TestNormalizeOpsWithReservedValues:
    """Test suite for normalize() function in ops.py with reserved value correction."""

    def test_normalize_without_correction(self):
        """Test that normalize works without reserved value correction."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        ufun.reserved_value = 0.5

        normalized = normalize(
            ufun, to=(0.0, 1.0), normalize_reserved_values=False, outcome_space=os
        )
        # Reserved value should not be modified
        # Note: After normalization, reserved value gets scaled too
        assert math.isfinite(normalized.reserved_value)

    def test_normalize_with_correction(self):
        """Test that normalize corrects non-finite reserved values when requested."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        ufun.reserved_value = float("inf")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = normalize(
                ufun,
                to=(0.0, 1.0),
                normalize_reserved_values=True,
                reserved_value_penalty=0.0,
                outcome_space=os,
            )
            # Should have warned about correction
            assert len(w) == 1
            assert issubclass(w[0].category, NegmasUnexpectedValueWarning)

        # Reserved value should be finite after correction and normalization
        assert math.isfinite(normalized.reserved_value)


class TestUtilityFunctionNormalizeWithReservedValues:
    """Test suite for UtilityFunction.normalize() with reserved value correction."""

    def test_normalize_method_without_correction(self):
        """Test ufun.normalize() without reserved value correction."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        ufun.reserved_value = 0.5

        normalized = ufun.normalize(to=(0.0, 1.0), normalize_reserved_values=False)
        assert math.isfinite(normalized.reserved_value)

    def test_normalize_method_with_correction(self):
        """Test ufun.normalize() with reserved value correction."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        ufun.reserved_value = float("-inf")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = ufun.normalize(
                to=(0.0, 1.0),
                normalize_reserved_values=True,
                reserved_value_penalty=0.0,
            )
            assert len(w) == 1

        assert math.isfinite(normalized.reserved_value)

    def test_normalize_for_method_with_correction(self):
        """Test ufun.normalize_for() with reserved value correction."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        ufun = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        ufun.reserved_value = float("nan")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = ufun.normalize_for(
                to=(0.0, 1.0),
                outcome_space=os,
                normalize_reserved_values=True,
                reserved_value_penalty=0.0,
            )
            assert len(w) == 1

        assert math.isfinite(normalized.reserved_value)


class TestScenarioNormalizeWithReservedValues:
    """Test suite for Scenario.normalize() with reserved value correction."""

    def test_scenario_normalize_without_correction(self):
        """Test Scenario.normalize() without reserved value correction."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os)
        u1.reserved_value = 0.5
        u2.reserved_value = 0.3

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        normalized_scenario = scenario.normalize(
            to=(0.0, 1.0), normalize_reserved_values=False
        )

        # Both reserved values should be finite (already were)
        for ufun in normalized_scenario.ufuns:
            assert math.isfinite(ufun.reserved_value)

    def test_scenario_normalize_with_correction(self):
        """Test Scenario.normalize() with reserved value correction."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os)
        u1.reserved_value = float("inf")
        u2.reserved_value = float("-inf")

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized_scenario = scenario.normalize(
                to=(0.0, 1.0),
                normalize_reserved_values=True,
                reserved_value_penalty=0.0,
            )
            # Should have 2 warnings (one for each ufun)
            assert len(w) == 2
            for warning in w:
                assert issubclass(warning.category, NegmasUnexpectedValueWarning)

        # Both reserved values should be finite after correction
        for ufun in normalized_scenario.ufuns:
            assert math.isfinite(ufun.reserved_value)

    def test_scenario_normalize_mixed_reserved_values(self):
        """Test Scenario.normalize() with mixed finite and non-finite reserved values."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os)
        u3 = MappingUtilityFunction(lambda x: x[0] * 0.5, outcome_space=os)
        u1.reserved_value = 0.5  # Finite
        u2.reserved_value = float("nan")  # Non-finite
        u3.reserved_value = None  # Non-finite

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2, u3))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized_scenario = scenario.normalize(
                to=(0.0, 1.0),
                normalize_reserved_values=True,
                reserved_value_penalty=0.0,
            )
            # Should have 2 warnings (for u2 and u3, not u1)
            assert len(w) == 2

        # All reserved values should be finite after correction
        for ufun in normalized_scenario.ufuns:
            assert math.isfinite(ufun.reserved_value)


class TestScenarioIsNormalizedWithFiniteReservedValue:
    """Test suite for Scenario.is_normalized() with finite_reserved_value parameter."""

    def test_is_normalized_without_finite_check(self):
        """Test is_normalized() without checking finite reserved values."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0] / 2, outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: (2 - x[0]) / 2, outcome_space=os)
        u1.reserved_value = float("inf")  # Non-finite but not checked

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        # Should pass because we're not checking finite_reserved_value
        assert scenario.is_normalized(to=(None, 1.0), finite_reserved_value=False)

    def test_is_normalized_with_finite_check_passes(self):
        """Test is_normalized() with finite reserved values passes."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0] / 2, outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: (2 - x[0]) / 2, outcome_space=os)
        u1.reserved_value = 0.5  # Finite
        u2.reserved_value = 0.3  # Finite

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert scenario.is_normalized(to=(None, 1.0), finite_reserved_value=True)

    def test_is_normalized_with_finite_check_fails_inf(self):
        """Test is_normalized() with inf reserved value fails when checking."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0] / 2, outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: (2 - x[0]) / 2, outcome_space=os)
        u1.reserved_value = float("inf")  # Non-finite

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert not scenario.is_normalized(to=(None, 1.0), finite_reserved_value=True)

    def test_is_normalized_with_finite_check_fails_none(self):
        """Test is_normalized() with None reserved value fails when checking."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0] / 2, outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: (2 - x[0]) / 2, outcome_space=os)
        u1.reserved_value = None  # Non-finite

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert not scenario.is_normalized(to=(None, 1.0), finite_reserved_value=True)

    def test_is_normalized_with_finite_check_fails_nan(self):
        """Test is_normalized() with NaN reserved value fails when checking."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0] / 2, outcome_space=os)
        u2 = MappingUtilityFunction(lambda x: (2 - x[0]) / 2, outcome_space=os)
        u1.reserved_value = float("nan")  # Non-finite

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert not scenario.is_normalized(to=(None, 1.0), finite_reserved_value=True)
