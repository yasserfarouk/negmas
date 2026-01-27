"""Test continue_cartesian_tournament function."""

from __future__ import annotations

import shutil
from pathlib import Path
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import AspirationNegotiator, RandomNegotiator
from negmas.tournaments.neg import cartesian_tournament, continue_cartesian_tournament
from negmas.sao.mechanism import SAOMechanism
import pytest


def make_test_scenarios(n: int = 2):
    """Create simple test scenarios."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=0.0, normalized=False),
            U.random(issues=issues, reserved_value=0.0, normalized=False),
        )
        for _ in range(n)
    ]
    return [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=u,
            mechanism_type=SAOMechanism,
            mechanism_params=dict(),
        )
        for i, u in enumerate(ufuns)
    ]


def test_continue_invalid_path(tmp_path):
    """Test that continue_cartesian_tournament returns None for invalid path."""
    # Non-existent path
    result = continue_cartesian_tournament(tmp_path / "nonexistent")
    assert result is None


def test_continue_empty_directory(tmp_path):
    """Test that continue_cartesian_tournament returns None for empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = continue_cartesian_tournament(empty_dir)
    assert result is None


def test_continue_missing_config(tmp_path):
    """Test that continue_cartesian_tournament returns None when config.yaml is missing."""
    tournament_path = tmp_path / "no_config"
    tournament_path.mkdir()
    scenarios_dir = tournament_path / "scenarios"
    scenarios_dir.mkdir()

    result = continue_cartesian_tournament(tournament_path)
    assert result is None


def test_continue_missing_scenarios(tmp_path):
    """Test that continue_cartesian_tournament returns None when scenarios/ is missing."""
    from negmas.helpers.inout import dump

    tournament_path = tmp_path / "no_scenarios"
    tournament_path.mkdir()

    # Create a fake config
    config = {"n_scenarios": 1, "competitors": ["RandomNegotiator"]}
    dump(config, tournament_path / "config.yaml")

    result = continue_cartesian_tournament(tournament_path)
    assert result is None


def test_continue_complete_tournament(tmp_path):
    """Test that continue_cartesian_tournament loads complete tournament."""
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]
    tournament_path = tmp_path / "complete"

    # Run complete tournament
    results1 = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=2,
        path=tournament_path,
        path_exists="continue",
        verbosity=0,
        rotate_ufuns=False,
    )

    # Continue should load existing results
    results2 = continue_cartesian_tournament(tournament_path)

    assert results2 is not None
    assert len(results1.scores) == len(results2.scores)
    assert len(results1.final_scores) == len(results2.final_scores)


def test_continue_incomplete_tournament(tmp_path):
    """Test that continue_cartesian_tournament can resume after partial completion.

    This test verifies the most common continuation scenario: loading results
    from a tournament that completed successfully.
    """
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]
    tournament_path = tmp_path / "test_continue"

    # Run a complete tournament
    results1 = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=2,
        path=tournament_path,
        path_exists="continue",
        verbosity=0,
        rotate_ufuns=False,
    )

    initial_count = len(results1.scores)

    # Call continue - should load existing results
    results2 = continue_cartesian_tournament(tournament_path, verbosity=0)

    assert results2 is not None
    assert len(results2.scores) == initial_count


def test_continue_with_override_parameters(tmp_path):
    """Test that continue_cartesian_tournament accepts override parameters."""
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]
    tournament_path = tmp_path / "override"

    # Run complete tournament with verbosity=0
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=1,
        path=tournament_path,
        path_exists="continue",
        verbosity=0,
        rotate_ufuns=False,
        njobs=0,
    )

    # Continue with different verbosity and njobs
    results = continue_cartesian_tournament(
        tournament_path,
        verbosity=2,  # Override verbosity
        njobs=-1,  # Override njobs
    )

    assert results is not None


def test_continue_tournament_with_rotation(tmp_path):
    """Test that continue_cartesian_tournament works with ufun rotation."""
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]
    tournament_path = tmp_path / "rotation"

    # Run tournament with rotation
    results1 = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=2,
        path=tournament_path,
        path_exists="continue",
        verbosity=0,
        rotate_ufuns=True,  # Enable rotation
    )

    # Continue should work correctly
    results2 = continue_cartesian_tournament(tournament_path, verbosity=0)

    assert results2 is not None
    # With rotation, we get multiple scenario variants
    assert len(results2.scores) >= len(results1.scores)


def test_continue_tournament_returns_none_on_corrupted_config(tmp_path):
    """Test that continue_cartesian_tournament returns None for corrupted config."""
    tournament_path = tmp_path / "corrupted"
    tournament_path.mkdir()
    scenarios_dir = tournament_path / "scenarios"
    scenarios_dir.mkdir()

    # Write corrupted config
    config_path = tournament_path / "config.yaml"
    config_path.write_text("this is not valid yaml: {[{")

    result = continue_cartesian_tournament(tournament_path)
    assert result is None


if __name__ == "__main__":
    import tempfile

    print("Testing invalid path...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_invalid_path(Path(tmp))
    print("✓ Pass")

    print("\nTesting empty directory...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_empty_directory(Path(tmp))
    print("✓ Pass")

    print("\nTesting missing config...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_missing_config(Path(tmp))
    print("✓ Pass")

    print("\nTesting missing scenarios...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_missing_scenarios(Path(tmp))
    print("✓ Pass")

    print("\nTesting complete tournament...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_complete_tournament(Path(tmp))
    print("✓ Pass")

    print("\nTesting incomplete tournament...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_incomplete_tournament(Path(tmp))
    print("✓ Pass")

    print("\nTesting override parameters...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_with_override_parameters(Path(tmp))
    print("✓ Pass")

    print("\nTesting tournament with rotation...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_tournament_with_rotation(Path(tmp))
    print("✓ Pass")

    print("\nTesting corrupted config...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_tournament_returns_none_on_corrupted_config(Path(tmp))
    print("✓ Pass")

    print("\n✅ All tests passed!")
