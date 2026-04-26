"""Tests for edge cases in reserved value handling (-inf and NaN)."""

from __future__ import annotations

import math

import pytest

from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import MappingUtilityFunction
from negmas.preferences.ops import kalai_points, ks_points, nash_points
from negmas.tournaments.neg.simple.cartesian import make_scores


class TestBargainingSolutionsWithInfReservedValues:
    """Test bargaining solution functions handle -inf and NaN reserved values correctly."""

    def test_ks_points_with_minus_inf_reserved_value(self):
        """Test Kalai-Smorodinsky solution with -inf reserved value."""
        issues = [make_issue(list(range(10)), "x")]
        os = make_os(issues)

        # Create ufuns with -inf reserved value
        u1 = MappingUtilityFunction(
            lambda x: float(x[0]), reserved_value=float("-inf"), outcome_space=os
        )
        u2 = MappingUtilityFunction(
            lambda x: 9.0 - float(x[0]), reserved_value=float("-inf"), outcome_space=os
        )

        # Get Pareto frontier utilities
        frontier = [(float(i), 9.0 - float(i)) for i in range(10)]

        # Should not raise and should return valid results
        results = ks_points([u1, u2], frontier)

        assert len(results) > 0
        for outcome_utils, idx in results:
            assert len(outcome_utils) == 2
            assert all(math.isfinite(u) for u in outcome_utils)

    def test_kalai_points_with_minus_inf_reserved_value(self):
        """Test Kalai solution with -inf reserved value."""
        issues = [make_issue(list(range(10)), "x")]
        os = make_os(issues)

        u1 = MappingUtilityFunction(
            lambda x: float(x[0]), reserved_value=float("-inf"), outcome_space=os
        )
        u2 = MappingUtilityFunction(
            lambda x: 9.0 - float(x[0]), reserved_value=float("-inf"), outcome_space=os
        )

        frontier = [(float(i), 9.0 - float(i)) for i in range(10)]

        results = kalai_points([u1, u2], frontier)

        assert len(results) > 0
        for outcome_utils, idx in results:
            assert len(outcome_utils) == 2
            assert all(math.isfinite(u) for u in outcome_utils)

    def test_nash_points_with_minus_inf_reserved_value(self):
        """Test Nash solution with -inf reserved value."""
        issues = [make_issue(list(range(10)), "x")]
        os = make_os(issues)

        u1 = MappingUtilityFunction(
            lambda x: float(x[0]), reserved_value=float("-inf"), outcome_space=os
        )
        u2 = MappingUtilityFunction(
            lambda x: 9.0 - float(x[0]), reserved_value=float("-inf"), outcome_space=os
        )

        frontier = [(float(i), 9.0 - float(i)) for i in range(10)]

        results = nash_points([u1, u2], frontier)

        assert len(results) > 0
        for outcome_utils, idx in results:
            assert len(outcome_utils) == 2
            assert all(math.isfinite(u) for u in outcome_utils)

    def test_ks_points_with_nan_reserved_value(self):
        """Test Kalai-Smorodinsky solution with NaN reserved value."""
        issues = [make_issue(list(range(10)), "x")]
        os = make_os(issues)

        # Create ufuns with NaN reserved value
        u1 = MappingUtilityFunction(
            lambda x: float(x[0]), reserved_value=float("nan"), outcome_space=os
        )
        u2 = MappingUtilityFunction(
            lambda x: 9.0 - float(x[0]), reserved_value=float("nan"), outcome_space=os
        )

        frontier = [(float(i), 9.0 - float(i)) for i in range(10)]

        # Should not raise and should return valid results
        results = ks_points([u1, u2], frontier)

        assert len(results) > 0
        for outcome_utils, idx in results:
            assert len(outcome_utils) == 2
            assert all(math.isfinite(u) for u in outcome_utils)

    def test_kalai_points_with_nan_reserved_value(self):
        """Test Kalai solution with NaN reserved value."""
        issues = [make_issue(list(range(10)), "x")]
        os = make_os(issues)

        u1 = MappingUtilityFunction(
            lambda x: float(x[0]), reserved_value=float("nan"), outcome_space=os
        )
        u2 = MappingUtilityFunction(
            lambda x: 9.0 - float(x[0]), reserved_value=float("nan"), outcome_space=os
        )

        frontier = [(float(i), 9.0 - float(i)) for i in range(10)]

        results = kalai_points([u1, u2], frontier)

        assert len(results) > 0
        for outcome_utils, idx in results:
            assert len(outcome_utils) == 2
            assert all(math.isfinite(u) for u in outcome_utils)

    def test_nash_points_with_nan_reserved_value(self):
        """Test Nash solution with NaN reserved value."""
        issues = [make_issue(list(range(10)), "x")]
        os = make_os(issues)

        u1 = MappingUtilityFunction(
            lambda x: float(x[0]), reserved_value=float("nan"), outcome_space=os
        )
        u2 = MappingUtilityFunction(
            lambda x: 9.0 - float(x[0]), reserved_value=float("nan"), outcome_space=os
        )

        frontier = [(float(i), 9.0 - float(i)) for i in range(10)]

        results = nash_points([u1, u2], frontier)

        assert len(results) > 0
        for outcome_utils, idx in results:
            assert len(outcome_utils) == 2
            assert all(math.isfinite(u) for u in outcome_utils)

    def test_mixed_reserved_values(self):
        """Test with one -inf and one normal reserved value."""
        issues = [make_issue(list(range(10)), "x")]
        os = make_os(issues)

        u1 = MappingUtilityFunction(
            lambda x: float(x[0]), reserved_value=float("-inf"), outcome_space=os
        )
        u2 = MappingUtilityFunction(
            lambda x: 9.0 - float(x[0]), reserved_value=0.5, outcome_space=os
        )

        frontier = [(float(i), 9.0 - float(i)) for i in range(10)]

        # All three functions should handle mixed cases
        ks_results = ks_points([u1, u2], frontier)
        kalai_results = kalai_points([u1, u2], frontier)
        nash_results = nash_points([u1, u2], frontier)

        assert len(ks_results) > 0
        assert len(kalai_results) > 0
        assert len(nash_results) > 0

        # All results should have finite utilities
        for results in [ks_results, kalai_results, nash_results]:
            for outcome_utils, idx in results:
                assert all(math.isfinite(u) for u in outcome_utils)


class TestTournamentScoringWithInfReservedValues:
    """Test tournament scoring (advantage calculation) with -inf and NaN reserved values."""

    def test_advantage_calculation_with_minus_inf_reserved_value(self):
        """Test advantage calculation when reserved value is -inf."""
        # Create a mock record with -inf reserved value
        record = {
            "utilities": (5.0, 3.0),
            "partners": ["Agent1", "Agent2"],
            "reserved_values": (float("-inf"), 0.0),
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [10.0, 10.0],
            "min_utils": [0.0, 0.0],
            "negotiator_times": [0.1, 0.2],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": None,
            "scenario": "TestScenario",
        }

        scores = make_scores(record)

        assert len(scores) == 2
        # For agent with -inf reserved value, advantage should be (u - min) = 5.0 - 0.0 = 5.0
        assert math.isfinite(scores[0]["advantage"])
        assert scores[0]["advantage"] == 5.0
        # For agent with normal reserved value, advantage should be (u - r) / (max - r) = (3.0 - 0.0) / (10.0 - 0.0) = 0.3
        assert math.isfinite(scores[1]["advantage"])
        assert scores[1]["advantage"] == pytest.approx(0.3)

    def test_advantage_calculation_with_nan_reserved_value(self):
        """Test advantage calculation when reserved value is NaN."""
        record = {
            "utilities": (7.0, 4.0),
            "partners": ["Agent1", "Agent2"],
            "reserved_values": (float("nan"), 2.0),
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [10.0, 10.0],
            "min_utils": [0.0, 0.0],
            "negotiator_times": [0.1, 0.2],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": None,
            "scenario": "TestScenario",
        }

        scores = make_scores(record)

        assert len(scores) == 2
        # For agent with NaN reserved value, advantage should be (u - min) = 7.0 - 0.0 = 7.0
        assert math.isfinite(scores[0]["advantage"])
        assert scores[0]["advantage"] == 7.0
        # For agent with normal reserved value, advantage should be (u - r) / (max - r) = (4.0 - 2.0) / (10.0 - 2.0) = 0.25
        assert math.isfinite(scores[1]["advantage"])
        assert scores[1]["advantage"] == pytest.approx(0.25)

    def test_advantage_calculation_both_inf_reserved_values(self):
        """Test advantage calculation when both reserved values are -inf."""
        record = {
            "utilities": (6.0, 8.0),
            "partners": ["Agent1", "Agent2"],
            "reserved_values": (float("-inf"), float("-inf")),
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [10.0, 10.0],
            "min_utils": [2.0, 3.0],
            "negotiator_times": [0.1, 0.2],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": None,
            "scenario": "TestScenario",
        }

        scores = make_scores(record)

        assert len(scores) == 2
        # Both should use (u - min) formula
        assert math.isfinite(scores[0]["advantage"])
        assert scores[0]["advantage"] == 6.0 - 2.0  # 4.0
        assert math.isfinite(scores[1]["advantage"])
        assert scores[1]["advantage"] == 8.0 - 3.0  # 5.0

    def test_advantage_with_nan_min_utils(self):
        """Test advantage calculation when min_utils is NaN (fallback case)."""
        record = {
            "utilities": (5.0, 7.0),
            "partners": ["Agent1", "Agent2"],
            "reserved_values": (float("-inf"), float("-inf")),
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [10.0, 10.0],
            "min_utils": [float("nan"), float("nan")],
            "negotiator_times": [0.1, 0.2],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": None,
            "scenario": "TestScenario",
        }

        scores = make_scores(record)

        assert len(scores) == 2
        # When min is NaN, advantage should be 0.0 for both
        assert scores[0]["advantage"] == 0.0
        assert scores[1]["advantage"] == 0.0

    def test_advantage_normal_case_still_works(self):
        """Test that normal advantage calculation still works correctly."""
        record = {
            "utilities": (7.0, 3.0),
            "partners": ["Agent1", "Agent2"],
            "reserved_values": (2.0, 1.0),
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [10.0, 8.0],
            "min_utils": [0.0, 0.0],
            "negotiator_times": [0.1, 0.2],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": None,
            "scenario": "TestScenario",
        }

        scores = make_scores(record)

        assert len(scores) == 2
        # Normal formula: (u - r) / (max - r)
        assert scores[0]["advantage"] == pytest.approx(
            (7.0 - 2.0) / (10.0 - 2.0)
        )  # 0.625
        assert scores[1]["advantage"] == pytest.approx(
            (3.0 - 1.0) / (8.0 - 1.0)
        )  # ~0.286
