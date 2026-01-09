from __future__ import annotations
from pathlib import Path
from time import sleep
from pytest import mark
import pandas as pd
from negmas.gb.common import ResponseType
from negmas.helpers.inout import is_nonzero_file
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import AspirationNegotiator, RandomNegotiator
from negmas.sao.common import SAOResponse
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators.base import SAONegotiator
from negmas.tournaments.neg import cartesian_tournament
import pytest
from ..switches import NEGMAS_FASTRUN

from negmas.tournaments.neg.simple.cartesian import (
    TOURNAMENT_DIRS,
    TOURNAMENT_FILES,
    SimpleTournamentResults,
)


class TimeWaster(SAONegotiator):
    def __call__(self, state, dest: str | None = None) -> SAOResponse:
        sleep(10000 * 60 * 60)
        return SAOResponse(ResponseType.REJECT_OFFER, self.nmi.random_outcome())


@pytest.mark.skip(
    "Can be used in the future to test breaking negotiations with infinite loops. Currently it will hang at the end"
)
def test_can_run_cartesian_simple_tournament_with_no_infinities():
    n = 1
    rotate_ufuns = True
    n_repetitions = 1
    issues = (
        make_issue([f"q{i}" for i in range(10)], "quantity"),
        make_issue([f"p{i}" for i in range(5)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=u,
            mechanism_type=SAOMechanism,  # type: ignore
            mechanism_params=dict(),
        )
        for i, u in enumerate(ufuns)
    ]
    competitors = [RandomNegotiator, TimeWaster]
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=10000000, hidden_time_limit=3),
        n_repetitions=n_repetitions,
        verbosity=4,
        rotate_ufuns=rotate_ufuns,
        # plot_fraction=0.5,
        path=None,
    )
    scores = results.scores
    assert (
        len(scores)
        == len(scenarios)
        * (int(rotate_ufuns) + 1)  # two variations if rotate_ufuns else one
        * len(competitors)
        * len(competitors)
        * n_repetitions
        * 2  # two scores per run
    )

    assert (
        len(results.details)
        == len(scenarios)
        * (int(rotate_ufuns) + 1)  # two variations if rotate_ufuns else one
        * len(competitors)
        * len(competitors)
        * n_repetitions
    )
    assert len(results.final_scores) == len(competitors)


def test_can_run_cartesian_simple_tournament_n_reps():
    n = 2
    rotate_ufuns = True
    n_repetitions = 4
    issues = (
        make_issue([f"q{i}" for i in range(10)], "quantity"),
        make_issue([f"p{i}" for i in range(5)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=u,
            mechanism_type=SAOMechanism,  # type: ignore
            mechanism_params=dict(),
        )
        for i, u in enumerate(ufuns)
    ]
    competitors = [RandomNegotiator, AspirationNegotiator]
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=10),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=rotate_ufuns,
        # plot_fraction=0.5,
        path=None,
    )
    scores = results.scores
    assert (
        len(scores)
        == len(scenarios)
        * (int(rotate_ufuns) + 1)  # two variations if rotate_ufuns else one
        * len(competitors)
        * len(competitors)
        * n_repetitions
        * 2  # two scores per run
    )

    assert (
        len(results.details)
        == len(scenarios)
        * (int(rotate_ufuns) + 1)  # two variations if rotate_ufuns else one
        * len(competitors)
        * len(competitors)
        * n_repetitions
    )
    assert len(results.final_scores) == len(competitors)


@mark.slow
@mark.skipif(
    NEGMAS_FASTRUN, reason="Testing loading, saving and combining tournaments is slow"
)
def test_load_save_combine(tmp_path: Path):
    path = tmp_path / "mytournament"
    path.mkdir(parents=True, exist_ok=True)
    n = 2
    rotate_ufuns = True
    n_repetitions = 4
    issues = (
        make_issue([f"q{i}" for i in range(10)], "quantity"),
        make_issue([f"p{i}" for i in range(5)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=u,
            mechanism_type=SAOMechanism,  # type: ignore
            mechanism_params=dict(),
        )
        for i, u in enumerate(ufuns)
    ]
    competitors = [RandomNegotiator, AspirationNegotiator]
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=10),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=rotate_ufuns,
        # plot_fraction=0.5,
        path=path,
    )
    final_scores = results.final_scores.copy()
    for d in TOURNAMENT_DIRS:
        dpath = path / d
        assert dpath.exists() and dpath.is_dir()
    for f in TOURNAMENT_FILES:
        fpath = path / f
        assert fpath.exists() and fpath.is_file() and is_nonzero_file(fpath)

    def compare_frames(df1: pd.DataFrame, df2: pd.DataFrame):
        assert len(df1) == len(df2), f"{len(df1)=}, {len(df2)=}"
        r1 = df1.to_dict("records")
        r2 = df2.to_dict("records")
        for a, b in zip(r1, r2, strict=True):
            for k, v in a.items():
                if isinstance(v, float):
                    assert abs(b[k] - v) < 1e-3, f"{k}: df1={v} df2={b[k]}"
                else:
                    assert b[k] == v, f"{k}: df1={v} df2={b[k]}"
        return True

    r = SimpleTournamentResults.load(path, must_have_details=True)
    assert compare_frames(
        final_scores, r.final_scores
    ), f"{final_scores}\n{r.final_scores}"
    results.save(tmp_path / "t1")
    r = SimpleTournamentResults.load(tmp_path / "t1", must_have_details=True)
    assert compare_frames(
        final_scores, r.final_scores
    ), f"{final_scores}\n{r.final_scores}"
    r, _ = SimpleTournamentResults.combine((path, tmp_path / "t1"))
    assert compare_frames(
        final_scores, r.final_scores
    ), f"{final_scores}\n{r.final_scores}"
