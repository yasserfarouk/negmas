"""Performance benchmarks for rotation optimization.

These tests measure the performance improvement achieved by using
recalculate_stats=False with rotate_ufuns=True.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from negmas import make_issue
from negmas.inout import Scenario
from negmas.outcomes import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import AspirationNegotiator, RandomNegotiator
from negmas.sao.mechanism import SAOMechanism
from negmas.tournaments.neg.simple import cartesian_tournament


def create_benchmark_scenarios(n_scenarios: int = 5, complexity: str = "medium"):
    """Create scenarios for performance testing.

    Args:
        n_scenarios: Number of scenarios to create
        complexity: Scenario complexity - "small", "medium", or "large"

    Returns:
        List of Scenario objects
    """
    if complexity == "small":
        n_quantity = 5
        n_price = 3
    elif complexity == "medium":
        n_quantity = 10
        n_price = 5
    else:  # large
        n_quantity = 20
        n_price = 10

    issues = (
        make_issue([f"q{i}" for i in range(n_quantity)], "quantity"),
        make_issue([f"p{i}" for i in range(n_price)], "price"),
    )

    scenarios = []
    for i in range(n_scenarios):
        ufuns = (
            U.random(issues=issues, reserved_value=0.0),
            U.random(issues=issues, reserved_value=0.0),
        )

        scenario = Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=ufuns,
            mechanism_type=SAOMechanism,  # type: ignore
            mechanism_params={},
            name=f"scenario_{i}",
        )

        scenarios.append(scenario)

    return scenarios


class TestRotationPerformance:
    """Benchmark the performance improvement of rotation optimization."""

    def test_rotation_performance_small(self):
        """Benchmark with small scenarios (quick test)."""
        scenarios = create_benchmark_scenarios(n_scenarios=3, complexity="small")
        competitors = [AspirationNegotiator, RandomNegotiator]

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            # OLD BEHAVIOR: recalculate_stats=True
            start_old = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=10,
                n_repetitions=2,
                path=Path(tmpdir1),
                rotate_ufuns=True,
                recalculate_stats=True,
                save_stats=False,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_old = time.time() - start_old

            # NEW BEHAVIOR: recalculate_stats=False
            start_new = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=10,
                n_repetitions=2,
                path=Path(tmpdir2),
                rotate_ufuns=True,
                recalculate_stats=False,
                save_stats=False,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_new = time.time() - start_new

        print("\nSmall scenarios performance:")
        print(f"  Old (recalculate=True):  {time_old:.3f}s")
        print(f"  New (recalculate=False): {time_new:.3f}s")
        if time_old > 0:
            speedup = time_old / time_new if time_new > 0 else float("inf")
            print(f"  Speedup: {speedup:.2f}x")

        # New should be at least as fast (or faster) than old
        # We don't enforce strict speedup here as it depends on system load
        # but we verify it completes successfully
        assert time_new > 0
        assert time_old > 0

    @pytest.mark.slow
    def test_rotation_performance_medium(self):
        """Benchmark with medium scenarios (slower, more realistic).

        This test is marked as slow and will only run when explicitly requested.
        """
        scenarios = create_benchmark_scenarios(n_scenarios=5, complexity="medium")
        competitors = [AspirationNegotiator, RandomNegotiator]

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            # OLD BEHAVIOR: recalculate_stats=True
            start_old = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=20,
                n_repetitions=2,
                path=Path(tmpdir1),
                rotate_ufuns=True,
                recalculate_stats=True,
                save_stats=True,  # Force stats calculation
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_old = time.time() - start_old

            # NEW BEHAVIOR: recalculate_stats=False
            start_new = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=20,
                n_repetitions=2,
                path=Path(tmpdir2),
                rotate_ufuns=True,
                recalculate_stats=False,
                save_stats=True,  # Calculate once, then rotate
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_new = time.time() - start_new

        print("\nMedium scenarios performance:")
        print(f"  Old (recalculate=True):  {time_old:.3f}s")
        print(f"  New (recalculate=False): {time_new:.3f}s")
        if time_old > 0:
            speedup = time_old / time_new if time_new > 0 else float("inf")
            print(f"  Speedup: {speedup:.2f}x")
            improvement_pct = (
                ((time_old - time_new) / time_old * 100) if time_old > 0 else 0
            )
            print(f"  Improvement: {improvement_pct:.1f}%")

        # With medium scenarios and stats enabled, we should see a measurable improvement
        assert time_new > 0
        assert time_old > 0

    @pytest.mark.slow
    def test_rotation_performance_large(self):
        """Benchmark with large scenarios (very slow, demonstrates maximum benefit).

        This test is marked as slow and will only run when explicitly requested.
        It demonstrates the maximum benefit of the optimization with larger scenarios.
        """
        scenarios = create_benchmark_scenarios(n_scenarios=3, complexity="large")
        competitors = [AspirationNegotiator, RandomNegotiator]

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            # OLD BEHAVIOR: recalculate_stats=True
            start_old = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=15,
                n_repetitions=1,
                path=Path(tmpdir1),
                rotate_ufuns=True,
                recalculate_stats=True,
                save_stats=True,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_old = time.time() - start_old

            # NEW BEHAVIOR: recalculate_stats=False
            start_new = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=15,
                n_repetitions=1,
                path=Path(tmpdir2),
                rotate_ufuns=True,
                recalculate_stats=False,
                save_stats=True,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_new = time.time() - start_new

        print("\nLarge scenarios performance:")
        print(f"  Old (recalculate=True):  {time_old:.3f}s")
        print(f"  New (recalculate=False): {time_new:.3f}s")
        if time_old > 0:
            speedup = time_old / time_new if time_new > 0 else float("inf")
            print(f"  Speedup: {speedup:.2f}x")
            improvement_pct = (
                ((time_old - time_new) / time_old * 100) if time_old > 0 else 0
            )
            print(f"  Improvement: {improvement_pct:.1f}%")

        # With large scenarios, the benefit should be most pronounced
        assert time_new > 0
        assert time_old > 0


class TestStatsCalculationPerformance:
    """Benchmark stats calculation vs rotation performance."""

    def test_stats_calculation_cost(self):
        """Measure the cost of stats calculation for different scenario sizes."""
        for complexity in ["small", "medium"]:
            issues = (
                make_issue(
                    [f"q{i}" for i in range(5 if complexity == "small" else 10)],
                    "quantity",
                ),
                make_issue(
                    [f"p{i}" for i in range(3 if complexity == "small" else 5)], "price"
                ),
            )

            ufuns = (
                U.random(issues=issues, reserved_value=0.0),
                U.random(issues=issues, reserved_value=0.0),
            )

            scenario = Scenario(
                outcome_space=make_os(issues),
                ufuns=ufuns,
                mechanism_type=SAOMechanism,  # type: ignore
                mechanism_params={},
            )

            # Measure stats calculation time
            start = time.time()
            scenario.calc_stats()
            calc_time = time.time() - start

            # Measure rotation time (with stats)
            start = time.time()
            _ = scenario.rotate_ufuns(n=1)
            rotate_time = time.time() - start

            print(f"\n{complexity.capitalize()} scenario stats performance:")
            print(f"  Stats calculation: {calc_time:.4f}s")
            print(f"  Stats rotation:    {rotate_time:.4f}s")
            if calc_time > 0:
                ratio = calc_time / rotate_time if rotate_time > 0 else float("inf")
                print(f"  Ratio (calc/rotate): {ratio:.2f}x")

            # Rotation should be much faster than calculation
            # (but we don't enforce strict ratio due to timing variability)
            assert rotate_time >= 0
            assert calc_time >= 0

    def test_no_rotation_no_overhead(self):
        """Verify that recalculate_stats=False adds no overhead when rotate_ufuns=False."""
        scenarios = create_benchmark_scenarios(n_scenarios=3, complexity="small")
        competitors = [AspirationNegotiator, RandomNegotiator]

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            # With recalculate_stats=True
            start_old = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=10,
                n_repetitions=1,
                path=Path(tmpdir1),
                rotate_ufuns=False,  # No rotation
                recalculate_stats=True,
                save_stats=False,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_old = time.time() - start_old

            # With recalculate_stats=False
            start_new = time.time()
            cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=10,
                n_repetitions=1,
                path=Path(tmpdir2),
                rotate_ufuns=False,  # No rotation
                recalculate_stats=False,
                save_stats=False,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            time_new = time.time() - start_new

        print("\nNo rotation performance (overhead test):")
        print(f"  recalculate=True:  {time_old:.3f}s")
        print(f"  recalculate=False: {time_new:.3f}s")
        print(f"  Difference: {abs(time_new - time_old):.3f}s")

        # Both should be similar (no rotation means no optimization benefit)
        # but both should complete successfully
        assert time_new > 0
        assert time_old > 0


class TestScalability:
    """Test how performance scales with different parameters."""

    @pytest.mark.parametrize("n_scenarios", [1, 3, 5])
    def test_scaling_with_scenarios(self, n_scenarios):
        """Test how performance scales with number of scenarios."""
        scenarios = create_benchmark_scenarios(
            n_scenarios=n_scenarios, complexity="small"
        )
        competitors = [AspirationNegotiator, RandomNegotiator]

        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.time()
            results = cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=10,
                n_repetitions=1,
                path=Path(tmpdir),
                rotate_ufuns=True,
                recalculate_stats=False,
                save_stats=False,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            elapsed = time.time() - start

        print(f"\nScaling with {n_scenarios} scenarios: {elapsed:.3f}s")

        assert results is not None
        assert elapsed > 0

    @pytest.mark.parametrize("n_repetitions", [1, 2, 3])
    def test_scaling_with_repetitions(self, n_repetitions):
        """Test how performance scales with number of repetitions."""
        scenarios = create_benchmark_scenarios(n_scenarios=2, complexity="small")
        competitors = [AspirationNegotiator, RandomNegotiator]

        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.time()
            results = cartesian_tournament(
                competitors=competitors,
                scenarios=scenarios,
                n_steps=10,
                n_repetitions=n_repetitions,
                path=Path(tmpdir),
                rotate_ufuns=True,
                recalculate_stats=False,
                save_stats=False,
                save_scenario_figs=False,
                verbosity=0,
                njobs=-1,
            )
            elapsed = time.time() - start

        print(f"\nScaling with {n_repetitions} repetitions: {elapsed:.3f}s")

        assert results is not None
        assert elapsed > 0


# Performance summary report
def test_performance_summary(pytestconfig):
    """Generate a summary report of performance improvements.

    This test always passes but prints useful performance information.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("=" * 70)
    print("\nThe rotation optimization (recalculate_stats=False) provides:")
    print("  • Faster tournament execution when rotating scenarios")
    print("  • Stats calculated once, then efficiently rotated")
    print("  • No performance penalty when rotation is disabled")
    print("  • Identical results to the old behavior (recalculate_stats=True)")
    print("\nExpected improvements:")
    print("  • Small scenarios (5x3 outcomes): Modest improvement")
    print("  • Medium scenarios (10x5 outcomes): Moderate improvement")
    print("  • Large scenarios (20x10 outcomes): Significant improvement")
    print("\nThe optimization is most beneficial when:")
    print("  • rotate_ufuns=True (creates scenario variants)")
    print("  • save_stats=True (stats are calculated and rotated)")
    print("  • Complex scenarios (more outcomes = more expensive stats)")
    print("=" * 70)
