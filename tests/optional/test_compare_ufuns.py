"""
Test cases for compare_ufuns function.

These tests should be added to tests/core/test_preferences.py or a similar test file.
"""

import numpy as np
import pytest

from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.ops import compare_ufuns


class TestCompareUfuns:
    """Tests for the compare_ufuns function."""

    def test_kendall_identical(self):
        """Identical ufuns should have kendall correlation of 1."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)

        result = compare_ufuns(ufun1, ufun1, method="kendall", issues=issues)
        assert abs(result - 1.0) < 1e-6

    def test_kendall_opposite(self):
        """Opposite ufuns should have kendall correlation of -1."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)
        ufun2 = LinearAdditiveUtilityFunction({"price": lambda x: -x}, issues=issues)

        result = compare_ufuns(ufun1, ufun2, method="kendall", issues=issues)
        assert abs(result - (-1.0)) < 1e-6

    def test_euclidean_identical(self):
        """Identical ufuns should have euclidean distance of 0."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)

        result = compare_ufuns(ufun1, ufun1, method="euclidean", issues=issues)
        assert abs(result) < 1e-6

    def test_euclidean_opposite(self):
        """Opposite ufuns should have non-zero euclidean distance."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)
        ufun2 = LinearAdditiveUtilityFunction({"price": lambda x: -x}, issues=issues)

        result = compare_ufuns(ufun1, ufun2, method="euclidean", issues=issues)
        assert 0.0 < result <= 1.0

    def test_ndcg_identical(self):
        """Identical ufuns should have ndcg score of 1."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)

        result = compare_ufuns(ufun1, ufun1, method="ndcg", issues=issues)
        assert abs(result - 1.0) < 1e-6

    def test_ndcg_range(self):
        """NDCG score should be in [0, 1]."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)
        ufun2 = LinearAdditiveUtilityFunction({"price": lambda x: -x}, issues=issues)

        result = compare_ufuns(ufun1, ufun2, method="ndcg", issues=issues)
        assert 0.0 <= result <= 1.0

    def test_custom_callable(self):
        """Custom callable should work correctly."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)
        ufun2 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)

        # Custom function that returns the mean absolute difference
        def mean_abs_diff(v1, v2):
            return float(np.mean(np.abs(v1 - v2)))

        result = compare_ufuns(ufun1, ufun2, method=mean_abs_diff, issues=issues)
        assert abs(result) < 1e-6  # Should be 0 for identical ufuns

    def test_normalize_with_different_scales(self):
        """Normalization should make scale-invariant for euclidean distance."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction(
            {"price": lambda x: x * 100}, issues=issues
        )  # 100-500
        ufun2 = LinearAdditiveUtilityFunction(
            {"price": lambda x: x}, issues=issues
        )  # 1-5

        # Without normalization, distance should be large
        result_no_norm = compare_ufuns(
            ufun1, ufun2, method="euclidean", normalize=False, issues=issues
        )
        assert result_no_norm > 0.1

        # With normalization, distance should be 0 (same relative ordering)
        result_norm = compare_ufuns(
            ufun1, ufun2, method="euclidean", normalize=True, issues=issues
        )
        assert abs(result_norm) < 1e-6

    def test_max_samples(self):
        """max_samples parameter should limit the number of sampled outcomes."""
        # Create a large outcome space
        issues = [make_issue(list(range(1000)), "x")]
        ufun1 = LinearAdditiveUtilityFunction({"x": lambda x: x}, issues=issues)
        ufun2 = LinearAdditiveUtilityFunction({"x": lambda x: -x}, issues=issues)

        # With max_samples=10, should only sample 10 outcomes
        result = compare_ufuns(
            ufun1, ufun2, method="kendall", max_samples=10, issues=issues
        )
        # Should still get strong negative correlation
        assert result < 0

    def test_with_outcomes_parameter(self):
        """Should work with explicit outcomes parameter."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)
        ufun2 = LinearAdditiveUtilityFunction({"price": lambda x: -x}, issues=issues)

        # Provide explicit outcomes
        outcomes = [(1,), (2,), (3,)]
        result = compare_ufuns(ufun1, ufun2, method="kendall", outcomes=outcomes)
        assert abs(result - (-1.0)) < 1e-6

    def test_invalid_method(self):
        """Should raise ValueError for invalid method."""
        issues = [make_issue([1, 2, 3, 4, 5], "price")]
        ufun1 = LinearAdditiveUtilityFunction({"price": lambda x: x}, issues=issues)

        with pytest.raises(ValueError, match="Unknown method"):
            compare_ufuns(ufun1, ufun1, method="invalid", issues=issues)  # type: ignore

    def test_with_explicit_outcomes(self):
        """Should work when outcomes are explicitly provided."""
        from negmas.preferences.crisp.mapping import MappingUtilityFunction

        # Create ufuns with outcome space
        ufun1 = MappingUtilityFunction({(1,): 0.5, (2,): 1.0})
        ufun2 = MappingUtilityFunction({(1,): 1.0, (2,): 0.5})

        # Should work when outcomes are provided explicitly
        result = compare_ufuns(ufun1, ufun2, method="kendall", outcomes=[(1,), (2,)])
        assert abs(result - (-1.0)) < 1e-6

    def test_multi_issue(self):
        """Should work with multiple issues."""
        issues = [
            make_issue([1, 2, 3], "price"),
            make_issue(["a", "b", "c"], "quality"),
        ]
        ufun1 = LinearAdditiveUtilityFunction(
            {"price": lambda x: x, "quality": {"a": 0.0, "b": 0.5, "c": 1.0}},
            issues=issues,
        )
        ufun2 = LinearAdditiveUtilityFunction(
            {"price": lambda x: -x, "quality": {"a": 1.0, "b": 0.5, "c": 0.0}},
            issues=issues,
        )

        # Should be able to compare
        result = compare_ufuns(ufun1, ufun2, method="kendall", issues=issues)
        # Should have negative correlation
        assert result < 0


if __name__ == "__main__":
    # Run tests manually for debugging
    test = TestCompareUfuns()
    test.test_kendall_identical()
    test.test_kendall_opposite()
    test.test_euclidean_identical()
    test.test_euclidean_opposite()
    test.test_ndcg_identical()
    test.test_ndcg_range()
    test.test_custom_callable()
    test.test_normalize_with_different_scales()
    test.test_max_samples()
    test.test_with_outcomes_parameter()
    test.test_multi_issue()
    print("All manual tests passed!")
