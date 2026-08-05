"""Post-tournament game-theoretic analysis: EGTA-based evaluation, Sequential
Elimination Ranking, replicator dynamics and related equilibrium/ranking
methods.

This package is organized in three layers:

    1. :mod:`~negmas.tournaments.analysis.loaders` -- adapters that convert a
       tournament result (in memory or on disk, cartesian or situated) into a
       common long-format DataFrame.
    2. :mod:`~negmas.tournaments.analysis.payoff` -- builds the empirical
       payoff matrix (:class:`~negmas.tournaments.analysis.payoff.PayoffTable`)
       that every method below consumes.
    3. Core computations, each optionally paired with a plotting helper in
       :mod:`~negmas.tournaments.analysis.plotting`:
       :mod:`~negmas.tournaments.analysis.equilibria` (pure/mixed Nash
       equilibria, regret, Nash-averaging),
       :mod:`~negmas.tournaments.analysis.ranking` (Sequential Elimination
       Ranking, Tournament Evaluation), and
       :mod:`~negmas.tournaments.analysis.dynamics` (replicator dynamics).
    4. :mod:`~negmas.tournaments.analysis.significance` -- pairwise
       statistical-significance testing (paired/unpaired t-test, rank-sum,
       Kolmogorov-Smirnov) with Bonferroni/Holm/Benjamini-Hochberg
       corrections, and per-strategy normality diagnostics, operating
       directly on the common records format rather than the payoff matrix.
    5. :mod:`~negmas.tournaments.analysis.report` -- :func:`~negmas.tournaments.analysis.report.analyze_tournament`
       runs every method above in one call and optionally saves results/plots
       to disk; this is what ``cartesian_tournament(..., run_analysis=True)``
       calls internally, and can also be used fully post-hoc on any
       in-memory or on-disk tournament result.

Mixed-strategy Nash equilibrium computation additionally requires the
optional ``negmas[egta]`` extra (``nashpy``/``pygambit``); everything else in
this package only needs negmas's core dependencies (numpy/scipy/pandas/
matplotlib/networkx).

AI Assistance Disclosure: this package (``negmas.tournaments.analysis``) was
implemented with AI assistance -- see the "AI Assistance Disclosure" section
of the project README.
"""

from __future__ import annotations

from . import (
    dynamics,
    equilibria,
    loaders,
    payoff,
    plotting,
    ranking,
    report,
    significance,
)
from .dynamics import *
from .equilibria import *
from .loaders import *
from .payoff import *
from .plotting import *
from .ranking import *
from .report import *
from .significance import *

__all__ = (
    loaders.__all__
    + payoff.__all__
    + equilibria.__all__
    + ranking.__all__
    + dynamics.__all__
    + plotting.__all__
    + significance.__all__
    + report.__all__
    + [
        "loaders",
        "payoff",
        "equilibria",
        "ranking",
        "dynamics",
        "plotting",
        "significance",
        "report",
    ]
)
