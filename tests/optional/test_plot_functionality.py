#!/usr/bin/env python3
"""
Test script for new plot functionality in Scenario class.

Tests:
1. plot() with ufun_indices parameter
2. save_plots() for 2 ufuns scenario
3. save_plots() for >2 ufuns scenario
4. dumpas() with save_plot=True
5. update() with save_plot=True
"""

import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from negmas import Scenario, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction


def create_test_scenario(n_ufuns=2):
    """Create a test scenario with n utility functions."""
    issues = [
        make_issue(values=["low", "medium", "high"], name="price"),
        make_issue(values=[1, 2, 3, 4, 5], name="quality"),
    ]

    ufuns = []
    for i in range(n_ufuns):
        # Create different utility functions
        weights = {"price": 0.3 + i * 0.1, "quality": 0.7 - i * 0.1}
        values = {
            "price": {"low": 1.0 - i * 0.2, "medium": 0.5, "high": 0.0 + i * 0.2},
            "quality": {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0},
        }
        ufun = LinearAdditiveUtilityFunction(
            weights=weights, values=values, issues=issues, name=f"agent_{i}"
        )
        ufuns.append(ufun)

    from negmas.outcomes import make_os

    return Scenario(
        outcome_space=make_os(issues, name="test_domain"),
        ufuns=tuple(ufuns),
        name="test_scenario",
    )


def test_plot_with_indices():
    """Test plot() method with ufun_indices parameter."""
    print("\n=== Test 1: plot() with ufun_indices ===")

    # Test with 2 ufuns (default behavior)
    scenario = create_test_scenario(n_ufuns=2)
    print(f"Created scenario with {len(scenario.ufuns)} ufuns")

    # Default plot (should plot first two)
    print("Testing default plot (indices 0, 1)...")
    fig = scenario.plot()
    assert fig is not None, "Plot should return a figure"
    print("✓ Default plot works")

    # Test with 3 ufuns
    scenario = create_test_scenario(n_ufuns=3)
    print(f"\nCreated scenario with {len(scenario.ufuns)} ufuns")

    # Plot different pairs
    print("Testing plot with indices (0, 1)...")
    fig = scenario.plot(ufun_indices=(0, 1))
    assert fig is not None
    print("✓ Plot (0, 1) works")

    print("Testing plot with indices (1, 2)...")
    fig = scenario.plot(ufun_indices=(1, 2))
    assert fig is not None
    print("✓ Plot (1, 2) works")

    print("Testing plot with indices (0, 2)...")
    fig = scenario.plot(ufun_indices=(0, 2))
    assert fig is not None
    print("✓ Plot (0, 2) works")

    # Test error cases
    print("\nTesting error cases...")
    try:
        scenario.plot(ufun_indices=(0, 0))
        assert False, "Should raise ValueError for same indices"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    try:
        scenario.plot(ufun_indices=(0, 5))
        assert False, "Should raise IndexError for invalid indices"
    except IndexError as e:
        print(f"✓ Correctly raised IndexError: {e}")

    print("\n✓ All plot() tests passed!")


def test_save_plots_2_ufuns():
    """Test save_plots() with 2 ufuns."""
    print("\n=== Test 2: save_plots() with 2 ufuns ===")

    scenario = create_test_scenario(n_ufuns=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Saving plots to: {tmpdir}")

        saved_files = scenario.save_plots(tmpdir, ext="png")
        print(f"Saved {len(saved_files)} file(s):")
        for f in saved_files:
            print(
                f"  - {f.name} (exists: {f.exists()}, size: {f.stat().st_size} bytes)"
            )

        # Should save single file as _plot.png
        assert len(saved_files) == 1, f"Expected 1 file, got {len(saved_files)}"
        assert saved_files[0].name == "_plot.png", (
            f"Expected '_plot.png', got '{saved_files[0].name}'"
        )
        assert saved_files[0].exists(), "Plot file should exist"
        assert saved_files[0].stat().st_size > 0, "Plot file should not be empty"

        # Verify no _plots subfolder was created
        plots_folder = tmpdir / "_plots"
        assert not plots_folder.exists(), "_plots folder should not exist for 2 ufuns"

        print("✓ save_plots() with 2 ufuns works correctly!")


def test_save_plots_multiple_ufuns():
    """Test save_plots() with >2 ufuns."""
    print("\n=== Test 3: save_plots() with >2 ufuns ===")

    for n in [3, 4]:
        print(f"\nTesting with {n} ufuns...")
        scenario = create_test_scenario(n_ufuns=n)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            print(f"Saving plots to: {tmpdir}")

            saved_files = scenario.save_plots(tmpdir, ext="png")
            print(f"Saved {len(saved_files)} file(s):")
            for f in saved_files:
                print(
                    f"  - {f.name} (exists: {f.exists()}, size: {f.stat().st_size} bytes)"
                )

            # Should save n files in _plots/ subfolder
            assert len(saved_files) == n, f"Expected {n} files, got {len(saved_files)}"

            # Verify _plots subfolder exists
            plots_folder = tmpdir / "_plots"
            assert plots_folder.exists(), "_plots folder should exist"
            assert plots_folder.is_dir(), "_plots should be a directory"

            # Verify all files exist and are in _plots folder
            expected_pairs = [(i, (i + 1) % n) for i in range(n)]
            for i, (idx1, idx2) in enumerate(expected_pairs):
                expected_name = f"agent_{idx1}-agent_{idx2}.png"
                assert saved_files[i].name == expected_name, (
                    f"Expected '{expected_name}', got '{saved_files[i].name}'"
                )
                assert saved_files[i].parent.name == "_plots", (
                    "File should be in _plots folder"
                )
                assert saved_files[i].exists(), (
                    f"Plot file {expected_name} should exist"
                )
                assert saved_files[i].stat().st_size > 0, (
                    f"Plot file {expected_name} should not be empty"
                )

            print(f"✓ save_plots() with {n} ufuns works correctly!")

    print("\n✓ All save_plots() tests passed!")


def test_dumpas_with_save_plot():
    """Test dumpas() with save_plot=True."""
    print("\n=== Test 4: dumpas() with save_plot=True ===")

    scenario = create_test_scenario(n_ufuns=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Dumping scenario to: {tmpdir}")

        # Test with save_plot=True
        scenario.dumpas(
            tmpdir, type="yml", save_plot=True, save_stats=False, save_info=False
        )

        # Check that domain and ufun files were created
        files = list(tmpdir.glob("*.yml"))
        print(f"Created {len(files)} YAML file(s):")
        for f in files:
            print(f"  - {f.name}")

        # Check that plot was created
        plot_file = tmpdir / "_plot.png"
        assert plot_file.exists(), "Plot file should be created with save_plot=True"
        assert plot_file.stat().st_size > 0, "Plot file should not be empty"
        print(f"  - {plot_file.name} (size: {plot_file.stat().st_size} bytes)")

        print("✓ dumpas() with save_plot=True works!")

    # Test with 3 ufuns
    print("\nTesting with 3 ufuns...")
    scenario = create_test_scenario(n_ufuns=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Dumping scenario to: {tmpdir}")

        scenario.dumpas(
            tmpdir, type="yml", save_plot=True, save_stats=False, save_info=False
        )

        # Check that _plots folder was created with 3 plots
        plots_folder = tmpdir / "_plots"
        assert plots_folder.exists(), "_plots folder should be created"
        plot_files = list(plots_folder.glob("*.png"))
        assert len(plot_files) == 3, f"Expected 3 plot files, got {len(plot_files)}"
        print(f"Created {len(plot_files)} plot file(s) in _plots/:")
        for f in plot_files:
            print(f"  - {f.name} (size: {f.stat().st_size} bytes)")

        print("✓ dumpas() with save_plot=True for 3 ufuns works!")


def test_update_with_save_plot():
    """Test update() with save_plot=True."""
    print("\n=== Test 5: update() with save_plot=True ===")

    scenario = create_test_scenario(n_ufuns=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Saving scenario to: {tmpdir}")

        # First save without plot
        scenario.dumpas(
            tmpdir, type="yml", save_plot=False, save_stats=False, save_info=False
        )

        # Load it back
        from negmas import Scenario

        scenario2 = Scenario.load(tmpdir)
        print(f"Loaded scenario: name='{scenario2.name}', source={scenario2.source}")

        # Update with save_plot=True
        print("Calling update() with save_plot=True...")
        result = scenario2.update(save_plot=True, save_stats=False, save_info=False)
        assert result == True, "update() should return True"

        # Check that plot was created
        plot_file = tmpdir / "_plot.png"
        assert plot_file.exists(), (
            "Plot file should be created after update(save_plot=True)"
        )
        assert plot_file.stat().st_size > 0, "Plot file should not be empty"
        print(
            f"✓ Plot created: {plot_file.name} (size: {plot_file.stat().st_size} bytes)"
        )

        print("✓ update() with save_plot=True works!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Plot Functionality")
    print("=" * 60)

    try:
        test_plot_with_indices()
        test_save_plots_2_ufuns()
        test_save_plots_multiple_ufuns()
        test_dumpas_with_save_plot()
        test_update_with_save_plot()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
