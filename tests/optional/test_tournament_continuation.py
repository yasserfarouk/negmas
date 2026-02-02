"""Test tournament continuation functionality."""

from __future__ import annotations

from pathlib import Path
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import AspirationNegotiator, RandomNegotiator
from negmas.tournaments.neg import cartesian_tournament
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


def test_path_exists_fail(tmp_path):
    """Test that path_exists='fail' raises error when directory exists."""
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]

    # First run should succeed
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=1,
        path=tmp_path / "test_tournament",
        path_exists="fail",
        verbosity=0,
    )
    assert results is not None

    # Second run should fail
    with pytest.raises(FileExistsError):
        cartesian_tournament(
            competitors=competitors,
            scenarios=scenarios,
            n_steps=10,
            n_repetitions=1,
            path=tmp_path / "test_tournament",
            path_exists="fail",
            verbosity=0,
        )


def test_path_exists_overwrite(tmp_path):
    """Test that path_exists='overwrite' deletes and restarts."""
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]

    # First run
    results1 = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=2,
        path=tmp_path / "test_tournament",
        path_exists="overwrite",
        verbosity=0,
        rotate_ufuns=False,
    )

    # Get the number of results
    n_results_1 = len(results1.scores)

    # Second run with overwrite - should start from scratch
    results2 = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=2,
        path=tmp_path / "test_tournament",
        path_exists="overwrite",
        verbosity=0,
        rotate_ufuns=False,
    )

    # Should have same number of results (fresh start)
    n_results_2 = len(results2.scores)
    assert n_results_1 == n_results_2


def test_path_exists_continue(tmp_path):
    """Test that path_exists='continue' resumes incomplete tournament."""
    scenarios = make_test_scenarios(2)
    competitors = [RandomNegotiator, AspirationNegotiator]
    tournament_path = tmp_path / "test_tournament"

    # First run - complete tournament
    results_full = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=3,
        path=tournament_path / "full",
        path_exists="continue",
        verbosity=0,
        rotate_ufuns=False,
        randomize_runs=False,
    )

    total_expected = len(results_full.scores)

    # Simulate partial run by running just 1 repetition
    results_partial = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=1,
        path=tournament_path / "partial",
        path_exists="continue",
        verbosity=0,
        rotate_ufuns=False,
        randomize_runs=False,
    )

    partial_count = len(results_partial.scores)

    # Continue the tournament with full repetitions
    results_continued = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=3,
        path=tournament_path / "partial",
        path_exists="continue",
        verbosity=1,
        rotate_ufuns=False,
        randomize_runs=False,
    )

    # Should have all results now
    final_count = len(results_continued.scores)
    assert final_count == total_expected
    assert final_count > partial_count


def test_file_naming_convention(tmp_path):
    """Test that file naming follows the expected convention."""
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]
    tournament_path = tmp_path / "test_naming"

    # Run with rotation and storage_optimization="speed" to keep results/ folder
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=1,
        path=tournament_path,
        path_exists="continue",
        verbosity=0,
        rotate_ufuns=True,
        storage_optimization="speed",  # Keep results/ folder
    )

    # Check that results/ directory has files
    results_dir = tournament_path / "results"
    assert results_dir.exists()

    # Get all JSON files
    json_files = list(results_dir.glob("*.json"))
    assert len(json_files) > 0

    # Check naming convention: scenario_negotiator1_negotiator2_..._rep_runid.json
    for file in json_files:
        name_parts = file.stem.split("_")
        # Should have at least: scenario, 2 negotiators, rep, run_id = 5 parts
        assert len(name_parts) >= 5
        # Rep should be a number
        rep = name_parts[-2]
        assert rep.isdigit()
        # run_id should be a hash (numeric string)
        run_id = name_parts[-1]
        assert run_id.lstrip("-").isdigit()


def test_continue_returns_existing_complete_tournament(tmp_path):
    """Test that continuing a complete tournament returns existing results."""
    scenarios = make_test_scenarios(1)
    competitors = [RandomNegotiator, AspirationNegotiator]
    tournament_path = tmp_path / "test_complete"

    # First run - complete
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

    # Second run - should return existing results immediately
    results2 = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=10,
        n_repetitions=2,
        path=tournament_path,
        path_exists="continue",
        verbosity=1,
        rotate_ufuns=False,
    )

    # Should have same results
    assert len(results1.scores) == len(results2.scores)
    assert len(results1.final_scores) == len(results2.final_scores)


if __name__ == "__main__":
    import tempfile

    print("Testing path_exists='fail'...")
    with tempfile.TemporaryDirectory() as tmp:
        test_path_exists_fail(Path(tmp))
    print("✓ Pass")

    print("\nTesting path_exists='overwrite'...")
    with tempfile.TemporaryDirectory() as tmp:
        test_path_exists_overwrite(Path(tmp))
    print("✓ Pass")

    print("\nTesting path_exists='continue'...")
    with tempfile.TemporaryDirectory() as tmp:
        test_path_exists_continue(Path(tmp))
    print("✓ Pass")

    print("\nTesting file naming convention...")
    with tempfile.TemporaryDirectory() as tmp:
        test_file_naming_convention(Path(tmp))
    print("✓ Pass")

    print("\nTesting continue returns existing complete tournament...")
    with tempfile.TemporaryDirectory() as tmp:
        test_continue_returns_existing_complete_tournament(Path(tmp))
    print("✓ Pass")

    print("\n✅ All tests passed!")
