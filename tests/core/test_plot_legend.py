"""Tests for plot legend positioning in different plot modes."""

import pytest
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.outcomes import make_issue
from negmas.plots.util import _get_legend_config


@pytest.fixture
def negotiation_session():
    """Create a simple negotiation session for testing plots."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    ufun1 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
    ufun2 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)

    neg1 = AspirationNegotiator(name="Buyer")
    neg2 = AspirationNegotiator(name="Seller")

    session = SAOMechanism(issues=issues, n_steps=20)
    session.add(neg1, ufun=ufun1)
    session.add(neg2, ufun=ufun2)
    session.run()
    return session


class TestLegendConfigHelper:
    """Tests for the _get_legend_config helper function."""

    def test_legend_config_both_plots(self):
        """When both 2D and offer plots are shown, legend should be horizontal below."""
        config = _get_legend_config(only2d=False, no2d=False)
        assert config["orientation"] == "h", (
            "Legend should be horizontal for both plots"
        )
        assert config["y"] < 0, "Legend should be below the figure (negative y)"
        assert config["xanchor"] == "center", "Legend should be centered horizontally"
        # traceorder="normal" is required for horizontal orientation to work
        # with subplot figures that have legendgrouptitle set on traces
        assert config.get("traceorder") == "normal", (
            "Legend should have traceorder='normal' for horizontal layout"
        )

    def test_legend_config_only2d(self):
        """When only 2D plot is shown, legend should be vertical on the right."""
        config = _get_legend_config(only2d=True, no2d=False)
        assert config["orientation"] == "v", "Legend should be vertical for only2d"
        assert config["x"] > 1, "Legend should be to the right of the figure (x > 1)"
        assert config["xanchor"] == "left", "Legend should anchor from left"

    def test_legend_config_no2d(self):
        """When 2D plot is hidden, legend should be vertical on the right."""
        config = _get_legend_config(only2d=False, no2d=True)
        assert config["orientation"] == "v", "Legend should be vertical for no2d"
        assert config["x"] > 1, "Legend should be to the right of the figure (x > 1)"
        assert config["xanchor"] == "left", "Legend should anchor from left"


class TestPlotLegendPositioning:
    """Tests for legend positioning in actual plots."""

    def test_plot_both_legend_horizontal_below(self, negotiation_session):
        """Default plot (both 2D and offers) should have horizontal legend below."""
        fig = negotiation_session.plot(show=False)
        assert fig is not None, "Plot should return a figure"

        legend = fig.layout.legend
        assert legend.orientation == "h", "Legend should be horizontal"
        assert legend.y < 0, "Legend should be below the figure"
        assert legend.xanchor == "center", "Legend should be centered"

    def test_plot_only2d_legend_vertical_right(self, negotiation_session):
        """Only 2D plot should have vertical legend on the right."""
        fig = negotiation_session.plot(show=False, only2d=True)
        assert fig is not None, "Plot should return a figure"

        legend = fig.layout.legend
        assert legend.orientation == "v", "Legend should be vertical"
        assert legend.x > 1, "Legend should be to the right of the figure"
        assert legend.xanchor == "left", "Legend should anchor from left"

    def test_plot_no2d_legend_vertical_right(self, negotiation_session):
        """No 2D plot (only offers) should have vertical legend on the right."""
        fig = negotiation_session.plot(show=False, no2d=True)
        assert fig is not None, "Plot should return a figure"

        legend = fig.layout.legend
        assert legend.orientation == "v", "Legend should be vertical"
        assert legend.x > 1, "Legend should be to the right of the figure"
        assert legend.xanchor == "left", "Legend should anchor from left"


class TestPlotLegendConsistency:
    """Tests to ensure legend configuration is consistent across plot functions."""

    def test_mechanism_plot_and_offline_plot_same_config(self, negotiation_session):
        """plot_mechanism_run and plot_offline_run should use same legend config."""
        from negmas.plots.util import plot_offline_run

        # Get figure from mechanism plot
        fig_mechanism = negotiation_session.plot(show=False)

        # Get figure from offline plot
        fig_offline = plot_offline_run(
            trace=negotiation_session.full_trace,
            ids=negotiation_session.negotiator_ids,
            ufuns=[n.ufun for n in negotiation_session.negotiators],
            agreement=negotiation_session.agreement,
            timedout=negotiation_session.state.timedout,
            broken=negotiation_session.state.broken,
            has_error=negotiation_session.state.has_error,
            show=False,
        )

        # Both should have same legend configuration
        assert (
            fig_mechanism.layout.legend.orientation
            == fig_offline.layout.legend.orientation
        )
        assert fig_mechanism.layout.legend.x == fig_offline.layout.legend.x
        assert fig_mechanism.layout.legend.y == fig_offline.layout.legend.y

    def test_all_modes_produce_valid_figures(self, negotiation_session):
        """All plot modes should produce valid figures with legends."""
        # Default (both)
        fig_both = negotiation_session.plot(show=False)
        assert fig_both is not None
        assert fig_both.layout.showlegend is True

        # Only 2D
        fig_2d = negotiation_session.plot(show=False, only2d=True)
        assert fig_2d is not None
        assert fig_2d.layout.showlegend is True

        # No 2D
        fig_no2d = negotiation_session.plot(show=False, no2d=True)
        assert fig_no2d is not None
        assert fig_no2d.layout.showlegend is True
