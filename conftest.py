"""Session-wide pytest configuration.

Ensures tests never try to *display* a plot. Showing a figure with
``show=True`` calls ``fig.show()`` (plotly's default renderer is ``browser``,
which opens a browser tab) or ``plt.show()`` (a blocking GUI window). On a
headless CI runner these block forever -- and because several tests call
``mechanism.plot(show=True)`` inside an assert-failure message, a single
assertion failure on such a runner would hang the whole job until it times out
(observed as a multi-hour Windows CI hang) instead of reporting the failure.

Suppressing display globally here keeps individual tests free of per-call
``show=False`` and turns any such failure into a fast, readable one.
"""

from __future__ import annotations

import os

# Force a non-interactive matplotlib backend before pyplot is imported anywhere
# so plt.show() is a harmless no-op for the whole session.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


@pytest.fixture(autouse=True)
def _no_blocking_plot_display(monkeypatch):
    """Neutralize plot display so no test can block on a GUI/browser window."""
    try:
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda *a, **k: None, raising=False)
    except Exception:
        pass
    try:
        import plotly.io as pio

        monkeypatch.setattr(pio, "show", lambda *a, **k: None, raising=False)
    except Exception:
        pass
