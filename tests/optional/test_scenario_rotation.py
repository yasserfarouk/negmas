"""Comprehensive tests for Scenario.rotate_ufuns() method.

Tests verify that utility function rotation works correctly while maintaining
stats integrity and properly handling info dictionaries.
"""

from __future__ import annotations

import pytest

from negmas import make_issue
from negmas.inout import Scenario
from negmas.outcomes import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U


def create_test_scenario(n_ufuns: int = 3, with_stats: bool = True):
    """Helper to create a test scenario with n utility functions.

    Args:
        n_ufuns: Number of utility functions to create
        with_stats: If True, calculate stats for the scenario

    Returns:
        A Scenario with the specified properties
    """
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    os = make_os(issues)

    # Create distinguishable ufuns with different reserved values
    ufuns = tuple(
        U.random(issues=issues, reserved_value=float(i) / 10) for i in range(n_ufuns)
    )

    scenario = Scenario(
        outcome_space=os,
        ufuns=ufuns,
        info={"agent_names": [f"agent{i}" for i in range(n_ufuns)]},
        name="test_scenario",
    )

    if with_stats:
        scenario.calc_stats()

    return scenario


class TestBasicRotation:
    """Test basic rotation functionality."""

    def test_rotate_right_by_one(self):
        """Test rotating right by 1 position: (u0, u1, u2) -> (u2, u0, u1)."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        rotated = scenario.rotate_ufuns(n=1)

        # Check ufuns are rotated correctly
        assert rotated.ufuns[0] is original_ufuns[2]
        assert rotated.ufuns[1] is original_ufuns[0]
        assert rotated.ufuns[2] is original_ufuns[1]

        # Original should be unchanged
        assert scenario.ufuns == original_ufuns

    def test_rotate_left_by_one(self):
        """Test rotating left by 1 position: (u0, u1, u2) -> (u1, u2, u0)."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        rotated = scenario.rotate_ufuns(n=-1)

        # Check ufuns are rotated correctly
        assert rotated.ufuns[0] is original_ufuns[1]
        assert rotated.ufuns[1] is original_ufuns[2]
        assert rotated.ufuns[2] is original_ufuns[0]

    def test_rotate_by_two(self):
        """Test rotating by 2 positions: (u0, u1, u2, u3) -> (u2, u3, u0, u1)."""
        scenario = create_test_scenario(n_ufuns=4, with_stats=False)
        original_ufuns = scenario.ufuns

        rotated = scenario.rotate_ufuns(n=2)

        assert rotated.ufuns[0] is original_ufuns[2]
        assert rotated.ufuns[1] is original_ufuns[3]
        assert rotated.ufuns[2] is original_ufuns[0]
        assert rotated.ufuns[3] is original_ufuns[1]

    def test_rotate_by_zero(self):
        """Test that rotating by 0 returns a deep copy."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        rotated = scenario.rotate_ufuns(n=0)

        # Should have the same number of ufuns in the same order
        assert len(rotated.ufuns) == len(original_ufuns)
        # Check reserved values match (ufuns are in same order)
        for i, (orig, rot) in enumerate(zip(original_ufuns, rotated.ufuns)):
            assert orig.reserved_value == rot.reserved_value, (
                f"Ufun {i} reserved value mismatch"
            )
        # But different scenario object
        assert rotated is not scenario

    def test_rotate_by_length(self):
        """Test that rotating by len(ufuns) returns a deep copy (full rotation)."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        rotated = scenario.rotate_ufuns(n=3)

        # Should have ufuns rotated full circle (back to same order)
        assert len(rotated.ufuns) == len(original_ufuns)
        # Check reserved values match (ufuns are in same order after full rotation)
        for i, (orig, rot) in enumerate(zip(original_ufuns, rotated.ufuns)):
            assert orig.reserved_value == rot.reserved_value, (
                f"Ufun {i} reserved value mismatch"
            )
        # But different scenario object
        assert rotated is not scenario

    def test_rotate_by_more_than_length(self):
        """Test that rotation wraps around correctly (modulo behavior)."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        # Rotate by 4 should be same as rotate by 1
        rotated = scenario.rotate_ufuns(n=4)

        assert rotated.ufuns[0] is original_ufuns[2]
        assert rotated.ufuns[1] is original_ufuns[0]
        assert rotated.ufuns[2] is original_ufuns[1]

    def test_rotate_negative_more_than_length(self):
        """Test that negative rotation wraps around correctly."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        # Rotate by -4 should be same as rotate by -1
        rotated = scenario.rotate_ufuns(n=-4)

        assert rotated.ufuns[0] is original_ufuns[1]
        assert rotated.ufuns[1] is original_ufuns[2]
        assert rotated.ufuns[2] is original_ufuns[0]


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_scenario(self):
        """Test rotating a scenario with no ufuns."""
        issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
        os = make_os(issues)
        scenario = Scenario(outcome_space=os, ufuns=())

        rotated = scenario.rotate_ufuns(n=1)

        assert rotated.ufuns == ()
        assert rotated is not scenario

    def test_single_ufun(self):
        """Test rotating a scenario with only one ufun."""
        scenario = create_test_scenario(n_ufuns=1, with_stats=False)
        original_ufun = scenario.ufuns[0]

        rotated = scenario.rotate_ufuns(n=1)

        # Should still have one ufun (only one to rotate, ends up in same position)
        assert len(rotated.ufuns) == 1
        # Reserved value should match
        assert rotated.ufuns[0].reserved_value == original_ufun.reserved_value

    def test_two_ufuns(self):
        """Test rotating a scenario with two ufuns."""
        scenario = create_test_scenario(n_ufuns=2, with_stats=False)
        u0, u1 = scenario.ufuns

        rotated = scenario.rotate_ufuns(n=1)

        assert rotated.ufuns[0] is u1
        assert rotated.ufuns[1] is u0

    def test_outcome_space_preserved(self):
        """Test that outcome space is preserved (not copied)."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)

        rotated = scenario.rotate_ufuns(n=1)

        # Outcome space should be the same object (shared reference)
        assert rotated.outcome_space is scenario.outcome_space

    def test_mechanism_type_preserved(self):
        """Test that mechanism type and params are preserved."""
        from negmas import SAOMechanism

        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.mechanism_type = SAOMechanism
        scenario.mechanism_params = {"n_steps": 100}

        rotated = scenario.rotate_ufuns(n=1)

        assert rotated.mechanism_type is SAOMechanism
        assert rotated.mechanism_params == {"n_steps": 100}
        # Params should be deep copied (different object)
        assert rotated.mechanism_params is not scenario.mechanism_params

    def test_name_preserved(self):
        """Test that scenario name is preserved."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        assert scenario.name == "test_scenario"

        rotated = scenario.rotate_ufuns(n=1)

        assert rotated.name == "test_scenario"

    def test_source_cleared(self):
        """Test that source path is cleared in rotated scenario."""
        from pathlib import Path

        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.source = Path("/tmp/original_scenario.yml")

        rotated = scenario.rotate_ufuns(n=1)

        assert rotated.source is None


class TestStatsRotation:
    """Test that stats are correctly rotated."""

    def test_utility_ranges_rotated(self):
        """Test that utility_ranges list is rotated."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        assert scenario.stats is not None

        original_ranges = scenario.stats.utility_ranges

        rotated = scenario.rotate_ufuns(n=1)

        # Ranges should be rotated: (r0, r1, r2) -> (r2, r0, r1)
        assert rotated.stats.utility_ranges[0] == original_ranges[2]
        assert rotated.stats.utility_ranges[1] == original_ranges[0]
        assert rotated.stats.utility_ranges[2] == original_ranges[1]

    def test_opposition_unchanged(self):
        """Test that opposition scalar is unchanged."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        original_opposition = scenario.stats.opposition

        rotated = scenario.rotate_ufuns(n=1)

        assert rotated.stats.opposition == original_opposition

    def test_pareto_utils_rotated(self):
        """Test that pareto_utils tuples are rotated.

        pareto_utils is a tuple of tuples, where each inner tuple represents
        utilities for all negotiators at a specific pareto point.
        Each inner tuple should be rotated.
        """
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        assert scenario.stats is not None
        assert scenario.stats.pareto_utils is not None

        original_pareto = scenario.stats.pareto_utils

        rotated = scenario.rotate_ufuns(n=1)

        # Each utility tuple in the pareto frontier should be rotated
        # e.g., (u0, u1, u2) -> (u2, u0, u1) for each point
        assert len(rotated.stats.pareto_utils) == len(original_pareto)
        for original_point, rotated_point in zip(
            original_pareto, rotated.stats.pareto_utils
        ):
            assert rotated_point[0] == original_point[2]
            assert rotated_point[1] == original_point[0]
            assert rotated_point[2] == original_point[1]

    def test_pareto_outcomes_unchanged(self):
        """Test that pareto_outcomes list is unchanged."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        assert scenario.stats is not None

        original_outcomes = scenario.stats.pareto_outcomes

        rotated = scenario.rotate_ufuns(n=1)

        # Outcomes should be the same (not rotated)
        assert rotated.stats.pareto_outcomes == original_outcomes
        # But should be a different list object
        assert rotated.stats.pareto_outcomes is not original_outcomes

    def test_nash_utils_rotated(self):
        """Test that nash_utils list of tuples is rotated."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        assert scenario.stats is not None

        original_nash = scenario.stats.nash_utils
        if not original_nash:
            pytest.skip("No nash solutions found for this scenario")

        rotated = scenario.rotate_ufuns(n=1)

        # Each utility tuple in the list should be rotated
        for i, rotated_utils in enumerate(rotated.stats.nash_utils):
            original_utils = original_nash[i]
            assert rotated_utils[0] == original_utils[2]
            assert rotated_utils[1] == original_utils[0]
            assert rotated_utils[2] == original_utils[1]

    def test_nash_outcomes_unchanged(self):
        """Test that nash_outcomes list is unchanged."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        assert scenario.stats is not None

        original_outcomes = scenario.stats.nash_outcomes

        rotated = scenario.rotate_ufuns(n=1)

        # Outcomes should be the same (not rotated)
        assert rotated.stats.nash_outcomes == original_outcomes
        # But should be a different list object
        assert rotated.stats.nash_outcomes is not original_outcomes

    def test_all_solution_concepts_rotated(self):
        """Test that all solution concept utilities are rotated correctly."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        assert scenario.stats is not None

        rotated = scenario.rotate_ufuns(n=1)

        # Check all _utils fields that contain tuples
        utils_fields = [
            "nash_utils",
            "kalai_utils",
            "modified_kalai_utils",
            "max_welfare_utils",
            "max_relative_welfare_utils",
            "ks_utils",
            "modified_ks_utils",
        ]

        for field in utils_fields:
            original_list = getattr(scenario.stats, field)
            rotated_list = getattr(rotated.stats, field)

            if not original_list:
                continue

            # Each tuple in the list should be rotated
            for i, rotated_utils in enumerate(rotated_list):
                original_utils = original_list[i]
                if len(original_utils) == 3:
                    assert rotated_utils[0] == original_utils[2]
                    assert rotated_utils[1] == original_utils[0]
                    assert rotated_utils[2] == original_utils[1]

    def test_all_outcomes_unchanged(self):
        """Test that all solution concept outcomes are unchanged."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        assert scenario.stats is not None

        rotated = scenario.rotate_ufuns(n=1)

        # Check all _outcomes fields
        outcomes_fields = [
            "pareto_outcomes",
            "nash_outcomes",
            "kalai_outcomes",
            "modified_kalai_outcomes",
            "max_welfare_outcomes",
            "max_relative_welfare_outcomes",
            "ks_outcomes",
            "modified_ks_outcomes",
        ]

        for field in outcomes_fields:
            original_outcomes = getattr(scenario.stats, field)
            rotated_outcomes = getattr(rotated.stats, field)

            # Outcomes should be equal (not rotated)
            assert rotated_outcomes == original_outcomes
            # But should be different list objects
            assert rotated_outcomes is not original_outcomes

    def test_scenario_without_stats(self):
        """Test rotating a scenario without stats."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        assert scenario.stats is None

        rotated = scenario.rotate_ufuns(n=1)

        # Stats should still be None
        assert rotated.stats is None


class TestInfoRotation:
    """Test info dictionary rotation behavior."""

    def test_info_rotated_when_enabled(self):
        """Test that info entries matching ufun count are rotated when rotate_info=True."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.info["agent_names"] = ["alice", "bob", "charlie"]
        scenario.info["agent_ids"] = [10, 20, 30]

        rotated = scenario.rotate_ufuns(n=1, rotate_info=True)

        # Lists matching ufun count should be rotated
        assert rotated.info["agent_names"] == ["charlie", "alice", "bob"]
        assert rotated.info["agent_ids"] == [30, 10, 20]

    def test_info_not_rotated_when_disabled(self):
        """Test that info entries are not rotated when rotate_info=False."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.info["agent_names"] = ["alice", "bob", "charlie"]

        rotated = scenario.rotate_ufuns(n=1, rotate_info=False)

        # Should stay the same
        assert rotated.info["agent_names"] == ["alice", "bob", "charlie"]

    def test_info_different_length_not_rotated(self):
        """Test that info entries with different lengths are not rotated."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.info["agent_names"] = ["alice", "bob"]  # Length 2, not 3
        scenario.info["config"] = {"key": "value"}  # Not a list

        rotated = scenario.rotate_ufuns(n=1, rotate_info=True)

        # Should stay the same (length doesn't match ufun count)
        assert rotated.info["agent_names"] == ["alice", "bob"]
        assert rotated.info["config"] == {"key": "value"}

    def test_info_tuple_rotated(self):
        """Test that info tuples are also rotated."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.info["agent_tuple"] = ("alice", "bob", "charlie")

        rotated = scenario.rotate_ufuns(n=1, rotate_info=True)

        # Tuples should also be rotated
        assert rotated.info["agent_tuple"] == ("charlie", "alice", "bob")
        assert isinstance(rotated.info["agent_tuple"], tuple)

    def test_info_deep_copied(self):
        """Test that info dict is deep copied."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.info["nested"] = {"key": [1, 2, 3]}

        rotated = scenario.rotate_ufuns(n=1)

        # Info should be different object
        assert rotated.info is not scenario.info
        # Nested structures should also be different objects
        assert rotated.info["nested"] is not scenario.info["nested"]
        assert rotated.info["nested"]["key"] is not scenario.info["nested"]["key"]

    def test_empty_info(self):
        """Test rotating with empty or None info."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.info = {}

        rotated = scenario.rotate_ufuns(n=1)

        assert rotated.info == {}


class TestMultipleRotations:
    """Test multiple consecutive rotations."""

    def test_double_rotation(self):
        """Test that rotating twice gives expected result."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        # Rotate twice by 1
        rotated1 = scenario.rotate_ufuns(n=1)
        rotated2 = rotated1.rotate_ufuns(n=1)

        # Should be equivalent to rotating by 2
        assert rotated2.ufuns[0] is original_ufuns[1]
        assert rotated2.ufuns[1] is original_ufuns[2]
        assert rotated2.ufuns[2] is original_ufuns[0]

    def test_rotation_and_reverse(self):
        """Test that rotating forward then backward returns to original order."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        # Rotate right then left
        rotated = scenario.rotate_ufuns(n=1)
        back = rotated.rotate_ufuns(n=-1)

        # Should be back to original order
        assert back.ufuns == original_ufuns

    def test_full_cycle(self):
        """Test rotating through a full cycle."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns

        # Rotate 3 times by 1 (full cycle)
        rotated = scenario
        for _ in range(3):
            rotated = rotated.rotate_ufuns(n=1)

        # Should be back to original order
        assert rotated.ufuns == original_ufuns


class TestDifferentScenarioSizes:
    """Test rotation with different numbers of ufuns."""

    @pytest.mark.parametrize("n_ufuns", [2, 3, 4, 5, 10])
    def test_various_sizes(self, n_ufuns):
        """Test rotation works for scenarios with different numbers of ufuns."""
        scenario = create_test_scenario(n_ufuns=n_ufuns, with_stats=False)
        original_ufuns = scenario.ufuns

        rotated = scenario.rotate_ufuns(n=1)

        # Check rotation is correct: (u0, u1, ..., un) -> (un, u0, u1, ..., un-1)
        assert len(rotated.ufuns) == n_ufuns
        assert rotated.ufuns[0] is original_ufuns[-1]
        for i in range(1, n_ufuns):
            assert rotated.ufuns[i] is original_ufuns[i - 1]

    @pytest.mark.parametrize("n_ufuns", [2, 3, 4, 5])
    def test_various_sizes_with_stats(self, n_ufuns):
        """Test rotation with stats works for different sizes."""
        scenario = create_test_scenario(n_ufuns=n_ufuns, with_stats=True)

        rotated = scenario.rotate_ufuns(n=1)

        # Check stats are present and rotated
        assert rotated.stats is not None
        assert len(rotated.stats.utility_ranges) == n_ufuns

        # Check utility ranges are rotated
        original_ranges = scenario.stats.utility_ranges
        assert rotated.stats.utility_ranges[0] == original_ranges[-1]
        for i in range(1, n_ufuns):
            assert rotated.stats.utility_ranges[i] == original_ranges[i - 1]


class TestOriginalUnchanged:
    """Test that original scenario is never modified."""

    def test_ufuns_unchanged(self):
        """Test that original ufuns are unchanged after rotation."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        original_ufuns = scenario.ufuns
        original_u0 = scenario.ufuns[0]

        _ = scenario.rotate_ufuns(n=1)

        # Original should be unchanged
        assert scenario.ufuns is original_ufuns
        assert scenario.ufuns[0] is original_u0

    def test_stats_unchanged(self):
        """Test that original stats are unchanged after rotation."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=True)
        original_stats = scenario.stats
        original_ranges = scenario.stats.utility_ranges

        _ = scenario.rotate_ufuns(n=1)

        # Original should be unchanged
        assert scenario.stats is original_stats
        assert scenario.stats.utility_ranges == original_ranges

    def test_info_unchanged(self):
        """Test that original info is unchanged after rotation."""
        scenario = create_test_scenario(n_ufuns=3, with_stats=False)
        scenario.info["agent_names"] = ["alice", "bob", "charlie"]
        original_info = scenario.info
        original_names = scenario.info["agent_names"]

        _ = scenario.rotate_ufuns(n=1, rotate_info=True)

        # Original should be unchanged
        assert scenario.info is original_info
        assert scenario.info["agent_names"] == original_names
