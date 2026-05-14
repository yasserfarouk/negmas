"""Smoke tests for the parallel/serial refactor in cartesian_tournament.

These verify behaviour parity after wiring negmas.helpers.parallel into the
Cartesian tournament loop. We run a tiny tournament both serially
(``njobs=-1``) and in parallel (``njobs=2``) and check that the two
configurations produce the same set of records.
"""

from __future__ import annotations

import pytest

from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.gb.negotiators.hybrid import HybridNegotiator
from negmas.gb.negotiators.micro import MiCRONegotiator
from negmas.sao.mechanism import SAOMechanism
from negmas.tournaments.neg import cartesian_tournament


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


@pytest.mark.parametrize("njobs", [-1, 2])
def test_cartesian_tournament_runs(tmp_path, njobs):
    """cartesian_tournament must complete without hanging in either mode."""
    scenarios = _make_scenarios(n=2)
    competitors = [MiCRONegotiator, HybridNegotiator]
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=8,
        n_repetitions=1,
        njobs=njobs,
        path=tmp_path / f"tournament_njobs_{njobs}",
        verbosity=0,
        rotate_ufuns=False,
    )
    assert results is not None
    assert len(results.scores) > 0


def test_cartesian_tournament_serial_and_parallel_same_shape(tmp_path):
    """Same configuration should produce the same number of records serially
    and in parallel."""
    scenarios = _make_scenarios(n=2)
    competitors = [MiCRONegotiator, HybridNegotiator]

    serial = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=8,
        n_repetitions=1,
        njobs=-1,
        path=tmp_path / "serial",
        verbosity=0,
        rotate_ufuns=False,
    )
    parallel = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=8,
        n_repetitions=1,
        njobs=2,
        path=tmp_path / "parallel",
        verbosity=0,
        rotate_ufuns=False,
    )
    assert len(serial.scores) == len(parallel.scores)
    assert set(serial.scores.columns) == set(parallel.scores.columns)
    # details table should also match in length and shape
    assert len(serial.details) == len(parallel.details)
    assert set(serial.details.columns) == set(parallel.details.columns)
    # the same set of negotiator pairings should appear in both runs
    def _pair_counts(df):
        from collections import Counter
        return Counter(tuple(sorted(p)) for p in df["partners"].tolist())
    assert _pair_counts(serial.details) == _pair_counts(parallel.details)
