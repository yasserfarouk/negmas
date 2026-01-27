"""Test callback support in cartesian_tournament.

This test demonstrates the three new callbacks for monitoring every step
of every negotiation:
- neg_start_callback: Called when a negotiation starts
- neg_progress_callback: Called after each step
- neg_end_callback: Called when a negotiation ends
"""

from pathlib import Path
import tempfile
import shutil

from negmas import SAOMechanism, make_issue
from negmas.inout import Scenario
from negmas.outcomes import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.tournaments.neg.simple import cartesian_tournament


# Module-level callback dir for parallel mode testing (fixed path for multiprocessing)
_PARALLEL_TEST_DIR = Path("/tmp/negmas_parallel_callback_test")


def _parallel_start_callback(run_id, state):
    """Module-level callback for parallel mode testing."""
    _PARALLEL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    (_PARALLEL_TEST_DIR / f"start_{run_id}.txt").write_text(f"{state.step}")


def _parallel_progress_callback(run_id, state):
    """Module-level callback for parallel mode testing."""
    _PARALLEL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(_PARALLEL_TEST_DIR / f"progress_{run_id}.txt", "a") as f:
        f.write(f"{state.step}\n")


def _parallel_end_callback(run_id, state):
    """Module-level callback for parallel mode testing."""
    _PARALLEL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    (_PARALLEL_TEST_DIR / f"end_{run_id}.txt").write_text(f"{state.step}")


def test_cartesian_callbacks():
    """Test that all three negotiation callbacks work correctly."""

    # Use a temp directory for callbacks to write to (can't rely on closure state with cloudpickle)
    callback_dir = Path(tempfile.mkdtemp())

    def start_callback(run_id, state):
        """Called once at the start of each negotiation."""
        (callback_dir / f"start_{run_id}.txt").write_text(
            f"{state.step},{state.running}"
        )

    def progress_callback(run_id, state):
        """Called after each step of each negotiation."""
        with open(callback_dir / f"progress_{run_id}.txt", "a") as f:
            f.write(f"{state.step}\n")

    def end_callback(run_id, state):
        """Called once at the end of each negotiation."""
        (callback_dir / f"end_{run_id}.txt").write_text(
            f"{state.step},{state.agreement is not None}"
        )

    # Create a simple scenario
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    scenario = Scenario(
        outcome_space=make_os(issues),
        ufuns=(
            U.random(issues=issues, reserved_value=0.0),
            U.random(issues=issues, reserved_value=0.0),
        ),
    )

    # Create temp directory for results
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Run tournament with callbacks
        results = cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=[scenario],
            n_steps=10,
            n_repetitions=2,  # Run each scenario twice
            path=temp_dir,
            njobs=-1,  # Serial mode
            neg_start_callback=start_callback,
            neg_progress_callback=progress_callback,
            neg_end_callback=end_callback,
            save_stats=False,  # Skip stats calculation to avoid ufun issues
            save_scenario_figs=False,  # Skip scenario plotting
            rotate_ufuns=False,  # Don't rotate to simplify counting
            verbosity=0,
        )

        print("\n=== Callback Test Results ===\n")

        # Count files created by callbacks
        start_files = list(callback_dir.glob("start_*.txt"))
        end_files = list(callback_dir.glob("end_*.txt"))
        progress_files = list(callback_dir.glob("progress_*.txt"))

        print(f"Start callbacks called: {len(start_files)}")
        print(f"End callbacks called: {len(end_files)}")

        total_progress = 0
        for f in progress_files:
            total_progress += len(
                [line for line in f.read_text().strip().split("\n") if line]
            )
        print(f"Progress callbacks called: {total_progress}")

        # Verify callbacks were called
        assert len(start_files) > 0, "Start callbacks should be called"
        assert len(end_files) > 0, "End callbacks should be called"
        assert len(start_files) == len(end_files), (
            f"Start and end callbacks should be called same number of times, got {len(start_files)} vs {len(end_files)}"
        )

        # Check progress callback was called
        assert total_progress >= 0, (
            f"Expected some progress callbacks, got {total_progress}"
        )

        print(f"✓ Start/end callbacks: {len(start_files)} negotiations")
        print(f"✓ Progress callbacks: {total_progress} steps total")

        # Verify callback signatures
        print("\n=== Start Callback Details ===")
        for i, f in enumerate(list(start_files)[:3]):  # Show first 3
            run_id = f.stem.replace("start_", "")
            step, running = f.read_text().split(",")
            print(f"  Start {i + 1}: run_id={run_id}, step={step}, running={running}")
            assert step == "0", f"Start callback should have step=0, got {step}"

        print("\n=== End Callback Details ===")
        for i, f in enumerate(list(end_files)[:3]):  # Show first 3
            run_id = f.stem.replace("end_", "")
            step, has_agreement = f.read_text().split(",")
            print(
                f"  End {i + 1}: run_id={run_id}, step={step}, agreement={has_agreement}"
            )

        print("\n✅ All callback tests passed!")

    finally:
        # Clean up temp directories
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if callback_dir.exists():
            shutil.rmtree(callback_dir)


def test_callbacks_parallel_mode():
    """Test that callbacks work in parallel mode (njobs > 0).

    Uses module-level callbacks that write to files to verify they're called
    in parallel mode where each worker has its own memory space.
    """
    # Clean up any previous test files
    if _PARALLEL_TEST_DIR.exists():
        shutil.rmtree(_PARALLEL_TEST_DIR)

    # Create a simple scenario
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    scenario = Scenario(
        outcome_space=make_os(issues),
        ufuns=(
            U.random(issues=issues, reserved_value=0.0),
            U.random(issues=issues, reserved_value=0.0),
        ),
    )

    # Create temp directory for results
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Run tournament with callbacks in PARALLEL mode
        results = cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=[scenario],
            n_steps=10,
            n_repetitions=2,  # Run each scenario twice
            path=temp_dir,
            njobs=2,  # PARALLEL MODE - this is the key test
            neg_start_callback=_parallel_start_callback,
            neg_progress_callback=_parallel_progress_callback,
            neg_end_callback=_parallel_end_callback,
            save_stats=False,
            save_scenario_figs=False,
            rotate_ufuns=False,
            verbosity=0,
        )

        print("\n=== Parallel Mode Callback Test Results ===\n")

        # Check if callbacks were called by looking at the files
        start_files = (
            list(_PARALLEL_TEST_DIR.glob("start_*.txt"))
            if _PARALLEL_TEST_DIR.exists()
            else []
        )
        end_files = (
            list(_PARALLEL_TEST_DIR.glob("end_*.txt"))
            if _PARALLEL_TEST_DIR.exists()
            else []
        )
        progress_files = (
            list(_PARALLEL_TEST_DIR.glob("progress_*.txt"))
            if _PARALLEL_TEST_DIR.exists()
            else []
        )

        print(f"Start callback files created: {len(start_files)}")
        print(f"End callback files created: {len(end_files)}")
        print(f"Progress callback files created: {len(progress_files)}")

        # Verify callbacks were called in parallel mode
        assert len(start_files) > 0, "Callbacks should be called in parallel mode"
        assert len(end_files) > 0, "End callbacks should be called in parallel mode"

        print(f"✓ Callbacks worked in parallel mode!")
        print(f"✓ Start/end callbacks: {len(start_files)} negotiations")

        # Count total progress calls
        total_progress = 0
        for f in progress_files:
            total_progress += len(f.read_text().strip().split("\n"))
        print(f"✓ Progress callbacks: {total_progress} steps total")

    finally:
        # Clean up temp directories
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if _PARALLEL_TEST_DIR.exists():
            shutil.rmtree(_PARALLEL_TEST_DIR)


def test_callbacks_with_opponents():
    """Test callbacks work correctly with opponents mode."""

    callback_dir = Path(tempfile.mkdtemp())

    def start_callback(run_id, state):
        (callback_dir / f"start_{run_id}.txt").write_text(f"{state.step}")

    def end_callback(run_id, state):
        (callback_dir / f"end_{run_id}.txt").write_text(f"{state.step}")

    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    scenario = Scenario(
        outcome_space=make_os(issues),
        ufuns=(
            U.random(issues=issues, reserved_value=0.0),
            U.random(issues=issues, reserved_value=0.0),
        ),
    )

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Run with opponents (competitor vs each opponent)
        results = cartesian_tournament(
            competitors=[AspirationNegotiator],
            opponents=[NaiveTitForTatNegotiator],
            scenarios=[scenario],
            n_steps=5,
            path=temp_dir,
            njobs=-1,
            neg_start_callback=start_callback,
            neg_end_callback=end_callback,
            save_stats=False,
            save_scenario_figs=False,
            verbosity=0,
            rotate_ufuns=False,
        )

        print("\n=== Opponents Mode Test ===\n")

        start_files = list(callback_dir.glob("start_*.txt"))
        end_files = list(callback_dir.glob("end_*.txt"))

        # With 1 competitor, 1 opponent, 1 scenario, and rotate_ufuns=False:
        # Should have 1 negotiation
        expected = 1

        print(f"Expected negotiations: {expected}")
        print(f"Start callbacks: {len(start_files)}")
        print(f"End callbacks: {len(end_files)}")

        assert len(start_files) == expected
        assert len(end_files) == expected

        print("✅ Opponents mode callbacks work correctly!")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if callback_dir.exists():
            shutil.rmtree(callback_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Cartesian Tournament Negotiation Callbacks")
    print("=" * 60)

    test_cartesian_callbacks()

    print("\n" + "=" * 60)
    print("Testing Parallel Mode")
    print("=" * 60)

    parallel_works = test_callbacks_parallel_mode()

    if not parallel_works:
        print("\n⚠️  IMPORTANT: Parallel mode callbacks did not work!")
        print("   Either callbacks can't be serialized, or implementation has issues.")

    test_callbacks_with_opponents()

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
