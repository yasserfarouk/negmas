"""Regression test for ``SimpleTournamentResults.from_records`` when only
``results`` (and not pre-computed ``scores``) are supplied.

``make_scores`` returns one dict *per negotiator* (a list). Every other call
site flattens it with ``scores += make_scores(...)``. The ``results``-only
branch of ``from_records`` must do the same: if it builds the scores frame from
a list-of-lists instead, the frame has list-valued cells and no ``strategy``
column, and the ``type_scores`` ``groupby("strategy")`` raises ``KeyError``.
"""

from __future__ import annotations

from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.gb.negotiators.hybrid import HybridNegotiator
from negmas.gb.negotiators.micro import MiCRONegotiator
from negmas.sao.mechanism import SAOMechanism
from negmas.tournaments.neg import cartesian_tournament
from negmas.tournaments.neg.simple.cartesian import SimpleTournamentResults


def _make_scenarios(n: int = 2):
    issues = (
        make_issue([f"q{i}" for i in range(4)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    return [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=(
                U.random(issues=issues, reserved_value=0.0, normalized=False),
                U.random(issues=issues, reserved_value=0.0, normalized=False),
            ),
            mechanism_type=SAOMechanism,
            mechanism_params=dict(),
        )
        for i in range(n)
    ]


def test_from_records_results_only_flattens_scores(tmp_path):
    """Rebuilding from ``results`` alone must yield a proper per-negotiator
    scores frame with a ``strategy`` column and a computable ``type_scores``."""
    tresults = cartesian_tournament(
        competitors=[MiCRONegotiator, HybridNegotiator],
        scenarios=_make_scenarios(n=2),
        n_steps=8,
        n_repetitions=1,
        njobs=-1,
        path=tmp_path / "t",
        verbosity=0,
        rotate_ufuns=False,
    )
    records = tresults.details
    assert len(records) > 0

    # The branch under test: scores=None, only results given.
    rebuilt = SimpleTournamentResults.from_records(results=records, scores=None)

    # one row per scored negotiator per negotiation (2 negotiators each here)
    assert "strategy" in rebuilt.scores.columns
    assert len(rebuilt.scores) == 2 * len(records)
    # groupby("strategy") inside from_records must have succeeded
    assert rebuilt.scores_summary is not None
    assert len(rebuilt.scores_summary) > 0
    # both competitor types should appear as strategies
    strategies = set(rebuilt.scores["strategy"].tolist())
    assert len(strategies) >= 2
