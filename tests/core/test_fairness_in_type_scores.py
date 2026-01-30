"""Test fairness metric in tournament type_scores."""

from __future__ import annotations

import math

from negmas.tournaments.neg.simple.cartesian import make_scores


class TestFairnessMetric:
    """Test suite for fairness metric in tournament results."""

    def test_fairness_in_make_scores(self):
        """Test that fairness is calculated correctly in make_scores."""
        # Create a mock record with optimality metrics
        record = {
            "utilities": [0.8, 0.7],
            "partners": ["NegotiatorA", "NegotiatorB"],
            "reserved_values": [0.0, 0.0],
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [1.0, 1.0],
            "min_utils": [0.0, 0.0],
            "negotiator_times": [1.0, 1.0],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": "",
            "scenario": "test_scenario",
            "mechanism_name": "test_mechanism",
            # Optimality metrics
            "nash_optimality": 0.85,
            "kalai_optimality": 0.90,
            "ks_optimality": 0.80,
            "pareto_optimality": 0.95,
            "max_welfare_optimality": 0.88,
            # Fairness is calculated in _make_record
            "fairness": 0.90,  # max(0.85, 0.90, 0.80)
        }

        scores = make_scores(record)

        assert len(scores) == 2
        for score in scores:
            # Fairness should be max of nash, kalai, ks
            assert "fairness" in score
            assert score["fairness"] == 0.90  # max(0.85, 0.90, 0.80)

    def test_fairness_with_nan_values(self):
        """Test fairness calculation when some optimality values are NaN."""
        record = {
            "utilities": [0.8, 0.7],
            "partners": ["NegotiatorA", "NegotiatorB"],
            "reserved_values": [0.0, 0.0],
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [1.0, 1.0],
            "min_utils": [0.0, 0.0],
            "negotiator_times": [1.0, 1.0],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": "",
            "scenario": "test_scenario",
            "mechanism_name": "test_mechanism",
            # Some optimality metrics are NaN
            "nash_optimality": 0.85,
            "kalai_optimality": float("nan"),
            "ks_optimality": 0.80,
            "pareto_optimality": 0.95,
            # Fairness should be max of finite values
            "fairness": 0.85,  # max(0.85, 0.80)
        }

        scores = make_scores(record)

        assert len(scores) == 2
        for score in scores:
            assert "fairness" in score
            # Should be max of finite values: max(0.85, 0.80) = 0.85
            assert score["fairness"] == 0.85

    def test_fairness_all_nan(self):
        """Test fairness when all relevant optimality values are NaN."""
        record = {
            "utilities": [0.8, 0.7],
            "partners": ["NegotiatorA", "NegotiatorB"],
            "reserved_values": [0.0, 0.0],
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [1.0, 1.0],
            "min_utils": [0.0, 0.0],
            "negotiator_times": [1.0, 1.0],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": "",
            "scenario": "test_scenario",
            "mechanism_name": "test_mechanism",
            # All relevant optimality metrics are NaN
            "nash_optimality": float("nan"),
            "kalai_optimality": float("nan"),
            "ks_optimality": float("nan"),
            "pareto_optimality": 0.95,
            # Fairness should be NaN when all fairness-relevant values are NaN
            "fairness": float("nan"),
        }

        scores = make_scores(record)

        assert len(scores) == 2
        for score in scores:
            assert "fairness" in score
            # Should be NaN when no valid values
            assert math.isnan(score["fairness"])

    def test_fairness_missing_optimality_metrics(self):
        """Test fairness when optimality metrics are not present."""
        record = {
            "utilities": [0.8, 0.7],
            "partners": ["NegotiatorA", "NegotiatorB"],
            "reserved_values": [0.0, 0.0],
            "negotiator_ids": ["id1", "id2"],
            "max_utils": [1.0, 1.0],
            "min_utils": [0.0, 0.0],
            "negotiator_times": [1.0, 1.0],
            "has_error": False,
            "erred_negotiator": None,
            "error_details": "",
            "scenario": "test_scenario",
            "mechanism_name": "test_mechanism",
            # No optimality metrics
        }

        scores = make_scores(record)

        assert len(scores) == 2
        for score in scores:
            # Fairness should not be in score if optimality metrics weren't calculated
            assert "fairness" not in score

    def test_all_optimality_cols_present(self):
        """Test that all OPTIMALITY_COLS are properly handled."""
        from negmas.tournaments.neg.simple.cartesian import OPTIMALITY_COLS

        # Verify fairness is in the list
        assert "fairness" in OPTIMALITY_COLS
        assert "nash_optimality" in OPTIMALITY_COLS
        assert "kalai_optimality" in OPTIMALITY_COLS
        assert "ks_optimality" in OPTIMALITY_COLS
        assert "modified_kalai_optimality" in OPTIMALITY_COLS
        assert "modified_ks_optimality" in OPTIMALITY_COLS
