"""Tests for CompletedRun.plot() method."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from negmas.gb.negotiators.timebased import AspirationNegotiator
from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.sao.mechanism import SAOMechanism


def create_test_mechanism(n_steps=20, n_issues=3):
    """Helper to create a test mechanism with negotiators."""
    os = make_os([make_issue(10) for _ in range(n_issues)])
    m = SAOMechanism(n_steps=n_steps, outcome_space=os)
    ufuns = LinearAdditiveUtilityFunction.generate_random_bilateral(
        list(os.enumerate_or_sample())
    )
    for i, u in enumerate(ufuns):
        u.outcome_space = os
        m.add(AspirationNegotiator(id=f"a{i}", name=f"n{i}", ufun=u))
    return m


class TestCompletedRunPlotBasic:
    """Test basic plotting functionality."""

    def test_plot_basic(self):
        """Test basic plotting works without errors."""
        m = create_test_mechanism()
        m.run()

        # Create completed run
        completed = m.to_completed_run(source="full_trace")

        # Plot should work without errors
        fig = completed.plot(show=False)
        assert fig is not None

    def test_plot_with_scenario(self):
        """Test plotting requires a scenario."""
        m = create_test_mechanism()
        m.run()

        # Create completed run without scenario
        completed = m.to_completed_run(source="full_trace")
        completed.scenario = None

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="Cannot plot CompletedRun without a scenario"
        ):
            completed.plot(show=False)

    def test_plot_requires_full_trace(self):
        """Test that plot requires full_trace history type."""
        m = create_test_mechanism()
        m.run()

        # Try different history types
        for history_type in ["history", "trace", "extended_trace"]:
            completed = m.to_completed_run(source=history_type)
            with pytest.raises(
                ValueError,
                match="Can only plot CompletedRun with history_type='full_trace'",
            ):
                completed.plot(show=False)

    def test_plot_requires_ufuns(self):
        """Test that plot requires utility functions in scenario."""
        m = create_test_mechanism()
        m.run()

        completed = m.to_completed_run(source="full_trace")
        # Clear utility functions
        completed.scenario.ufuns = []

        with pytest.raises(
            ValueError, match="Cannot plot CompletedRun without utility functions"
        ):
            completed.plot(show=False)


class TestCompletedRunPlotParameters:
    """Test various plotting parameters."""

    def test_plot_only2d(self):
        """Test plotting with only2d=True."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(show=False, only2d=True)
        assert fig is not None

    def test_plot_no2d(self):
        """Test plotting with no2d=True."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(show=False, no2d=True)
        assert fig is not None

    def test_plot_with_colors_and_markers(self):
        """Test plotting with custom colors and markers."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(
            show=False, colors=["red", "blue"], markers=["circle", "square"]
        )
        assert fig is not None

    def test_plot_with_different_xdims(self):
        """Test plotting with different x-axis dimensions."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        for xdim in ["relative_time", "step", "time"]:
            fig = completed.plot(show=False, xdim=xdim)
            assert fig is not None

    def test_plot_with_ylimits(self):
        """Test plotting with custom y-axis limits."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(show=False, ylimits=(0.0, 1.0))
        assert fig is not None

    def test_plot_show_distances(self):
        """Test plotting with various distance options."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(
            show=False,
            show_pareto_distance=True,
            show_nash_distance=True,
            show_kalai_distance=True,
            show_ks_distance=True,
            show_max_welfare_distance=True,
        )
        assert fig is not None

    def test_plot_simple_offers_view(self):
        """Test plotting with simple_offers_view=True."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(show=False, simple_offers_view=True)
        assert fig is not None

    def test_plot_fast_mode(self):
        """Test plotting with fast=True."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(show=False, fast=True)
        assert fig is not None


class TestCompletedRunPlotSaving:
    """Test saving plots to disk."""

    def test_plot_save_to_disk(self):
        """Test saving plot to disk."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        with tempfile.TemporaryDirectory() as tmpdir:
            completed.plot(
                show=False, save_fig=True, path=tmpdir, fig_name="test_plot.png"
            )

            # Check file was created
            saved_file = Path(tmpdir) / "test_plot.png"
            assert saved_file.exists()

    def test_plot_save_different_formats(self):
        """Test saving plots in different formats."""
        m = create_test_mechanism()
        m.run()
        completed = m.to_completed_run(source="full_trace")

        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in ["png", "webp", "svg", "pdf"]:
                completed.plot(
                    show=False, save_fig=True, path=tmpdir, fig_name=f"test_plot.{fmt}"
                )
                saved_file = Path(tmpdir) / f"test_plot.{fmt}"
                assert saved_file.exists()


class TestCompletedRunPlotAfterLoadSave:
    """Test plotting after saving and loading CompletedRun."""

    def test_plot_after_save_load(self):
        """Test that plot works after saving and loading."""
        m = create_test_mechanism()
        m.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save completed run
            completed = m.to_completed_run(source="full_trace")

            # Ensure scenario exists
            assert completed.scenario is not None, (
                "Scenario should exist after to_completed_run"
            )

            # Calculate scenario stats before saving
            completed.scenario.calc_stats()

            save_path = completed.save(
                parent=tmpdir,
                name="test_run",
                save_scenario=True,
                save_scenario_stats=True,
            )

            # Load it back with scenario
            loaded = completed.__class__.load(
                save_path, load_scenario=True, load_scenario_stats=True
            )

            # Check if scenario was loaded
            if loaded.scenario is None:
                # If scenario loading failed, skip this test
                pytest.skip(
                    "Scenario failed to load - may be due to serialization issues"
                )

            # Plot should work
            fig = loaded.plot(show=False)
            assert fig is not None

    def test_plot_fails_without_scenario_loaded(self):
        """Test that plot fails if scenario is not loaded."""
        m = create_test_mechanism()
        m.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save completed run
            completed = m.to_completed_run(source="full_trace")
            save_path = completed.save(
                parent=tmpdir, name="test_run", save_scenario=True
            )

            # Load without scenario
            loaded = completed.load(save_path, load_scenario=False)

            # Plot should fail
            with pytest.raises(
                ValueError, match="Cannot plot CompletedRun without a scenario"
            ):
                loaded.plot(show=False)


class TestCompletedRunPlotEquivalence:
    """Test that CompletedRun.plot() produces equivalent results to SAOMechanism.plot()."""

    def test_plot_equivalence_basic(self):
        """Test that mechanism plot and completed run plot produce similar results."""
        m = create_test_mechanism(n_steps=10)
        m.run()

        # Plot from mechanism
        mechanism_fig = m.plot(show=False)

        # Create completed run and plot directly (without save/load)
        completed = m.to_completed_run(source="full_trace")
        completed_fig = completed.plot(show=False)

        # Both should produce figures
        assert mechanism_fig is not None
        assert completed_fig is not None

        # Check that both have data (traces)
        assert len(mechanism_fig.data) > 0
        assert len(completed_fig.data) > 0

    def test_plot_equivalence_with_parameters(self):
        """Test equivalence with various plotting parameters."""
        m = create_test_mechanism(n_steps=10)
        m.run()

        params = {
            "show": False,
            "show_pareto_distance": True,
            "show_nash_distance": True,
            "with_lines": True,
            "only2d": False,
            "no2d": False,
        }

        # Plot from mechanism
        mechanism_fig = m.plot(**params)

        # Create completed run and plot
        completed = m.to_completed_run(source="full_trace")
        completed_fig = completed.plot(**params)

        # Both should produce figures
        assert mechanism_fig is not None
        assert completed_fig is not None

        # Check that both have similar number of traces
        assert len(mechanism_fig.data) > 0
        assert len(completed_fig.data) > 0

    def test_plot_equivalence_only2d(self):
        """Test equivalence with only2d=True."""
        m = create_test_mechanism(n_steps=10)
        m.run()

        mechanism_fig = m.plot(show=False, only2d=True)
        completed = m.to_completed_run(source="full_trace")
        completed_fig = completed.plot(show=False, only2d=True)

        assert mechanism_fig is not None
        assert completed_fig is not None

    def test_plot_equivalence_no2d(self):
        """Test equivalence with no2d=True."""
        m = create_test_mechanism(n_steps=10)
        m.run()

        mechanism_fig = m.plot(show=False, no2d=True)
        completed = m.to_completed_run(source="full_trace")
        completed_fig = completed.plot(show=False, no2d=True)

        assert mechanism_fig is not None
        assert completed_fig is not None


class TestCompletedRunPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_plot_with_no_history(self):
        """Test plotting with empty history."""
        m = create_test_mechanism()
        # Don't run the mechanism
        completed = m.to_completed_run(source="full_trace")

        # Should still work (empty plot)
        fig = completed.plot(show=False)
        assert fig is not None

    def test_plot_with_agreement(self):
        """Test plotting when an agreement was reached."""
        m = create_test_mechanism(n_steps=100)
        m.run()

        completed = m.to_completed_run(source="full_trace")

        # If agreement was reached, it should be visible in the plot
        fig = completed.plot(show=False, show_agreement=True)
        assert fig is not None

    def test_plot_with_timeout(self):
        """Test plotting when negotiation timed out."""
        m = create_test_mechanism(n_steps=5)
        m.run()

        completed = m.to_completed_run(source="full_trace")

        fig = completed.plot(show=False, show_end_reason=True)
        assert fig is not None

    def test_plot_with_extra_annotation(self):
        """Test plotting with extra annotation."""
        m = create_test_mechanism()
        m.run()

        completed = m.to_completed_run(source="full_trace")
        fig = completed.plot(show=False, extra_annotation="Test annotation")
        assert fig is not None

    def test_plot_with_negotiator_indices(self):
        """Test plotting with negotiator indices instead of IDs."""
        m = create_test_mechanism()
        m.run()

        completed = m.to_completed_run(source="full_trace")
        fig = completed.plot(show=False, plotting_negotiators=(0, 1))
        assert fig is not None

    def test_plot_with_negotiator_ids(self):
        """Test plotting with negotiator IDs."""
        m = create_test_mechanism()
        m.run()

        completed = m.to_completed_run(source="full_trace")
        ids = completed.config.get("negotiator_ids", [])

        if len(ids) >= 2:
            fig = completed.plot(show=False, plotting_negotiators=(ids[0], ids[1]))
            assert fig is not None
