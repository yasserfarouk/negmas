#!/usr/bin/env python3
"""
Simple test for plot functionality - tests the API without generating actual plots.
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


def test_plot_method_signature():
    """Test that plot() method has correct signature."""
    print("\n=== Test 1: plot() method signature ===")

    scenario = create_test_scenario(n_ufuns=3)
    print(f"Created scenario with {len(scenario.ufuns)} ufuns")

    # Check that plot method exists and accepts ufun_indices parameter
    import inspect

    sig = inspect.signature(scenario.plot)
    params = list(sig.parameters.keys())
    print(f"plot() parameters: {params}")

    assert "ufun_indices" in params, "plot() should have ufun_indices parameter"
    print("✓ plot() has ufun_indices parameter")

    # Check default value
    default = sig.parameters["ufun_indices"].default
    assert default is None, f"ufun_indices default should be None, got {default}"
    print("✓ ufun_indices defaults to None")

    print("\n✓ plot() method signature test passed!")


def test_save_plots_method():
    """Test that save_plots() method exists and has correct signature."""
    print("\n=== Test 2: save_plots() method ===")

    scenario = create_test_scenario(n_ufuns=2)

    # Check that save_plots method exists
    assert hasattr(scenario, "save_plots"), "Scenario should have save_plots method"
    print("✓ save_plots() method exists")

    # Check signature
    import inspect

    sig = inspect.signature(scenario.save_plots)
    params = list(sig.parameters.keys())
    print(f"save_plots() parameters: {params}")

    assert "folder" in params, "save_plots() should have folder parameter"
    assert "ext" in params, "save_plots() should have ext parameter"
    print("✓ save_plots() has correct parameters")

    # Check default value for ext
    default_ext = sig.parameters["ext"].default
    assert default_ext == "png", f"ext default should be 'png', got {default_ext}"
    print("✓ ext defaults to 'png'")

    print("\n✓ save_plots() method test passed!")


def test_dumpas_signature():
    """Test that dumpas() has save_plot parameter."""
    print("\n=== Test 3: dumpas() method signature ===")

    scenario = create_test_scenario(n_ufuns=2)

    import inspect

    sig = inspect.signature(scenario.dumpas)
    params = list(sig.parameters.keys())
    print(f"dumpas() parameters: {params}")

    assert "save_plot" in params, "dumpas() should have save_plot parameter"
    assert "plot_extension" in params, "dumpas() should have plot_extension parameter"
    assert "plot_kwargs" in params, "dumpas() should have plot_kwargs parameter"
    print("✓ dumpas() has all plot-related parameters")

    # Check defaults
    assert sig.parameters["save_plot"].default == False
    assert sig.parameters["plot_extension"].default == "png"
    assert sig.parameters["plot_kwargs"].default is None
    print("✓ dumpas() plot parameters have correct defaults")

    print("\n✓ dumpas() method signature test passed!")


def test_update_signature():
    """Test that update() has save_plot parameter."""
    print("\n=== Test 4: update() method signature ===")

    scenario = create_test_scenario(n_ufuns=2)

    import inspect

    sig = inspect.signature(scenario.update)
    params = list(sig.parameters.keys())
    print(f"update() parameters: {params}")

    assert "save_plot" in params, "update() should have save_plot parameter"
    assert "plot_extension" in params, "update() should have plot_extension parameter"
    assert "plot_kwargs" in params, "update() should have plot_kwargs parameter"
    print("✓ update() has all plot-related parameters")

    # Check defaults
    assert sig.parameters["save_plot"].default == False
    assert sig.parameters["plot_extension"].default == "png"
    assert sig.parameters["plot_kwargs"].default is None
    print("✓ update() plot parameters have correct defaults")

    print("\n✓ update() method signature test passed!")


def test_basic_save_and_load():
    """Test basic save/load without plots."""
    print("\n=== Test 5: Basic save/load (backward compatibility) ===")

    scenario = create_test_scenario(n_ufuns=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Saving scenario to: {tmpdir}")

        # Save without plots (default behavior)
        scenario.dumpas(tmpdir, type="yml", save_stats=False, save_info=False)

        # Check that domain and ufun files were created
        files = list(tmpdir.glob("*.yml"))
        print(f"Created {len(files)} YAML file(s): {[f.name for f in files]}")
        assert len(files) >= 2, "Should have domain and ufun files"

        # Check that NO plot was created (save_plot=False by default)
        plot_file = tmpdir / "_plot.png"
        assert not plot_file.exists(), (
            "Plot file should NOT be created with save_plot=False (default)"
        )
        print("✓ No plot created when save_plot=False (default)")

        # Load it back
        scenario2 = Scenario.load(tmpdir)
        print(f"Loaded scenario: name='{scenario2.name}', source={scenario2.source}")
        # Note: When loading from folder, name becomes the folder name (temp dir basename in this case)
        assert scenario2.name is not None
        assert len(scenario2.ufuns) == 2
        print("✓ Scenario loaded successfully")

        # Test update without save_plot
        result = scenario2.update(save_stats=False, save_info=False)
        assert result == True
        print("✓ update() works without save_plot")

        # Verify still no plot
        assert not plot_file.exists(), "Plot file should still not exist"
        print("✓ No plot created when save_plot=False in update()")

    print("\n✓ Basic save/load test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Plot Functionality (API only)")
    print("=" * 60)

    try:
        test_plot_method_signature()
        test_save_plots_method()
        test_dumpas_signature()
        test_update_signature()
        test_basic_save_and_load()

        print("\n" + "=" * 60)
        print("✓ ALL API TESTS PASSED!")
        print("=" * 60)
        print("\nNote: Actual plot generation not tested (requires plotly/kaleido)")
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
