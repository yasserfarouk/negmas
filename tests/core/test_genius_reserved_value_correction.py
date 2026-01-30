"""Tests for reserved value correction in Genius XML export."""

from __future__ import annotations

import math
import warnings

from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.warnings import NegmasUnexpectedValueWarning


class TestGeniusReservedValueCorrection:
    """Test suite for reserved value correction when exporting to Genius XML."""

    def test_correction_with_minus_inf(self):
        """Test that -inf reserved value is corrected in XML export."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("-inf")

        # Export to XML should correct the value and warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xml_str = u1.to_xml_str(issues=issues)
            # Should have a warning about correction
            assert len(w) == 1
            assert issubclass(w[0].category, NegmasUnexpectedValueWarning)
            assert "not finite" in str(w[0].message).lower()
            assert "-inf" in str(w[0].message)

        # XML should contain the corrected value (ufun.min() - 0.0)
        expected_rv = float(u1.min()) - 0.0
        assert f'<reservation value="{expected_rv}"' in xml_str

    def test_correction_with_inf(self):
        """Test that +inf reserved value is corrected in XML export."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("inf")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xml_str = u1.to_xml_str(issues=issues)
            assert len(w) == 1
            assert "inf" in str(w[0].message).lower()

        expected_rv = float(u1.min()) - 0.0
        assert f'<reservation value="{expected_rv}"' in xml_str

    def test_correction_with_nan(self):
        """Test that NaN reserved value is corrected in XML export."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("nan")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xml_str = u1.to_xml_str(issues=issues)
            assert len(w) == 1
            assert "nan" in str(w[0].message).lower()

        expected_rv = float(u1.min()) - 0.0
        assert f'<reservation value="{expected_rv}"' in xml_str

    def test_correction_with_none(self):
        """Test that None reserved value is corrected in XML export."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xml_str = u1.to_xml_str(issues=issues)
            assert len(w) == 1
            assert "None" in str(w[0].message)

        expected_rv = float(u1.min()) - 0.0
        assert f'<reservation value="{expected_rv}"' in xml_str

    def test_normal_reserved_value_unchanged(self):
        """Test that normal reserved values are not corrected."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = 0.5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xml_str = u1.to_xml_str(issues=issues)
            # No warnings for normal values
            assert len(w) == 0

        # Should have the original value
        assert '<reservation value="0.5"' in xml_str

    def test_corrected_value_is_finite(self):
        """Test that corrected values are always finite."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)

        problematic_values = [float("-inf"), float("inf"), float("nan"), None]

        for rv in problematic_values:
            u = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
            u.reserved_value = rv

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                xml_str = u.to_xml_str(issues=issues)

            # Extract the reservation value from XML
            if '<reservation value="' in xml_str:
                start = xml_str.index('<reservation value="') + len(
                    '<reservation value="'
                )
                end = xml_str.index('"', start)
                exported_rv = float(xml_str[start:end])
                assert math.isfinite(exported_rv), (
                    f"Exported rv {exported_rv} for original {rv} is not finite"
                )

    def test_negative_min_correction(self):
        """Test correction works when ufun.min() is negative."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        # This ufun has negative values
        u1 = MappingUtilityFunction(lambda x: x[0] - 2, outcome_space=os)
        u1.reserved_value = float("-inf")

        assert u1.min() == -2  # Verify assumption

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xml_str = u1.to_xml_str(issues=issues)
            assert len(w) == 1

        # Should be corrected to -2.0 (min - 0.0)
        assert '<reservation value="-2.0"' in xml_str

    def test_zero_min_correction(self):
        """Test correction works when ufun.min() is 0."""
        issues = [make_issue([0, 1, 2], "x")]
        os = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=os)
        u1.reserved_value = float("nan")

        assert u1.min() == 0  # Verify assumption

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            xml_str = u1.to_xml_str(issues=issues)

        # Should be corrected to 0.0 (min - 0.0)
        assert (
            '<reservation value="0.0"' in xml_str or '<reservation value="0"' in xml_str
        )

    def test_to_genius_file_with_correction(self):
        """Test that to_genius() also corrects reserved values."""
        import tempfile
        import os

        issues = [make_issue([0, 1, 2], "x")]
        outcome_space = make_os(issues)
        u1 = MappingUtilityFunction(lambda x: x[0], outcome_space=outcome_space)
        u1.reserved_value = float("-inf")

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            temp_file = f.name

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                u1.to_genius(temp_file, issues=issues)
                assert len(w) == 1

            # Read the file and check it has corrected value
            with open(temp_file, "r") as f:
                content = f.read()

            expected_rv = float(u1.min()) - 0.0
            assert f'<reservation value="{expected_rv}"' in content
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
