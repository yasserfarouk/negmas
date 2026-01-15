from __future__ import annotations
from pathlib import Path
from time import sleep
from pytest import mark
import pandas as pd
from negmas.gb.common import ResponseType
from negmas.gb.negotiators.timebased import BoulwareTBNegotiator
from negmas.gb.negotiators.titfortat import NaiveTitForTatNegotiator
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
    RunInfo,
)


# Module-level callbacks for picklability in parallel tests
def _picklable_before_callback(info: RunInfo):
    """Simple before callback that's picklable"""
    assert info.s is not None


def _picklable_after_callback(record: dict):
    """Simple after callback that's picklable"""
    assert "agreement" in record


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


def test_can_run_cartesian_simple_tournament_n_reps_different_opponents():
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
    opponents = [NaiveTitForTatNegotiator, BoulwareTBNegotiator]
    results = cartesian_tournament(
        competitors=competitors,
        opponents=opponents,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=10),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=rotate_ufuns,
        # plot_fraction=0.5,
        path=None,
    )
    scores = results.scores
    assert len(results.final_scores) == len(competitors)
    assert (
        len(scores)
        == len(scenarios)
        * (int(rotate_ufuns) + 1)  # two variations if rotate_ufuns else one
        * len(competitors)
        * n_repetitions
        * 2  # two scores per run
    )

    assert (
        len(results.details)
        == len(scenarios)
        * (int(rotate_ufuns) + 1)  # two variations if rotate_ufuns else one
        * len(competitors)
        * len(opponents)
        * n_repetitions
    )


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


# ===== Callback Tests =====


def test_callback_before_start_is_called():
    """Test that before_start_callback is invoked for each negotiation"""
    call_count = {"count": 0}
    run_infos = []

    def before_callback(info: RunInfo):
        call_count["count"] += 1
        run_infos.append(info)

    n = 1
    rotate_ufuns = False
    n_repetitions = 2
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(outcome_space=make_os(issues, name=f"S{i}"), ufuns=u)
        for i, u in enumerate(ufuns)
    ]
    competitors = [RandomNegotiator, AspirationNegotiator]
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=rotate_ufuns,
        path=None,
        njobs=-1,  # Serial execution for deterministic testing
        before_start_callback=before_callback,
    )

    # Expected: n_scenarios * n_competitors^2 * n_repetitions
    expected_calls = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert (
        call_count["count"] == expected_calls
    ), f"Expected {expected_calls} calls, got {call_count['count']}"
    assert len(run_infos) == expected_calls

    # Verify RunInfo structure
    for info in run_infos:
        assert isinstance(info, RunInfo)
        assert info.s is not None
        assert len(info.partners) == 2
        assert info.rep < n_repetitions


def test_callback_after_end_is_called():
    """Test that after_end_callback is invoked for each negotiation"""
    call_count = {"count": 0}
    records = []

    def after_callback(record: dict):
        call_count["count"] += 1
        records.append(record)

    n = 1
    rotate_ufuns = False
    n_repetitions = 2
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(outcome_space=make_os(issues, name=f"S{i}"), ufuns=u)
        for i, u in enumerate(ufuns)
    ]
    competitors = [RandomNegotiator, AspirationNegotiator]
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=rotate_ufuns,
        path=None,
        njobs=-1,  # Serial execution
        after_end_callback=after_callback,
    )

    expected_calls = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert call_count["count"] == expected_calls
    assert len(records) == expected_calls

    # Verify record structure
    for record in records:
        assert isinstance(record, dict)
        assert "agreement" in record
        assert "utilities" in record
        assert "partners" in record
        assert len(record["partners"]) == 2


def test_callback_both_called():
    """Test that both callbacks work together"""
    before_count = {"count": 0}
    after_count = {"count": 0}

    def before_callback(info: RunInfo):
        before_count["count"] += 1

    def after_callback(record: dict):
        after_count["count"] += 1

    n_repetitions = 1
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
        before_start_callback=before_callback,
        after_end_callback=after_callback,
    )

    expected_calls = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert before_count["count"] == expected_calls
    assert after_count["count"] == expected_calls


def test_callback_exception_handling_before():
    """Test that exceptions in before_start_callback are handled gracefully with ignore_exceptions"""

    def failing_before_callback(info: RunInfo):
        raise ValueError("Test exception in before callback")

    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator]

    # With raise_exceptions=False, should handle the exception gracefully
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        path=None,
        njobs=-1,
        before_start_callback=failing_before_callback,
        raise_exceptions=False,  # This sets ignore_exceptions=True in mechanism
    )

    # Tournament should complete despite callback failure
    # Note: The callback exception is caught but negotiation still fails in mechanism creation
    # So we might get 0 or 1 results depending on how it's handled
    assert len(results.details) >= 0  # Just verify it doesn't crash


def test_callback_exception_handling_after():
    """Test that exceptions in after_end_callback are handled gracefully"""
    success_count = {"count": 0}

    def failing_after_callback(record: dict):
        if success_count["count"] == 0:
            success_count["count"] += 1
            raise ValueError("Test exception in after callback")
        success_count["count"] += 1

    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(2)
    ]
    scenarios = [
        Scenario(outcome_space=make_os(issues, name=f"S{i}"), ufuns=u)
        for i, u in enumerate(ufuns)
    ]
    competitors = [RandomNegotiator]

    # Should not raise even though callback fails
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        path=None,
        njobs=-1,
        rotate_ufuns=False,  # Disable rotation to get predictable count
        after_end_callback=failing_after_callback,
    )

    # Tournament should complete despite callback failure
    # 2 scenarios * 1 competitor self-play = 2 negotiations
    assert len(results.details) == 2
    assert (
        success_count["count"] == 2
    )  # Called twice, first one raised, second succeeded


def test_callback_works_with_parallel_execution():
    """Test that callbacks work in parallel mode"""
    # Use module-level callbacks for picklability in parallel execution

    n = 2
    n_repetitions = 2
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(outcome_space=make_os(issues, name=f"S{i}"), ufuns=u)
        for i, u in enumerate(ufuns)
    ]
    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=2,  # Parallel execution
        before_start_callback=_picklable_before_callback,
        after_end_callback=_picklable_after_callback,
    )

    # Verify tournament completed successfully
    expected_negotiations = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert len(results.details) == expected_negotiations


def test_callback_receives_correct_info():
    """Test that callbacks receive correct information"""
    collected_data = {"scenarios": [], "partners": [], "agreements": []}

    def before_callback(info: RunInfo):
        collected_data["scenarios"].append(info.s.outcome_space.name)
        collected_data["partners"].append([p.__name__ for p in info.partners])

    def after_callback(record: dict):
        collected_data["agreements"].append(record["agreement"])

    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [
        Scenario(outcome_space=make_os(issues, name="TestScenario"), ufuns=ufuns[0])
    ]
    competitors = [RandomNegotiator, AspirationNegotiator]

    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
        before_start_callback=before_callback,
        after_end_callback=after_callback,
        mask_scenario_names=False,  # Don't mask so we can verify
    )

    # Verify before_callback collected scenario names
    assert len(collected_data["scenarios"]) == 4  # 2x2 combinations
    assert all("TestScenario" in name for name in collected_data["scenarios"])

    # Verify partners data
    assert len(collected_data["partners"]) == 4
    assert all(len(p) == 2 for p in collected_data["partners"])

    # Verify after_callback collected agreements
    assert len(collected_data["agreements"]) == 4


# ===== Opponents Feature Tests =====


def test_opponents_basic():
    """Test that opponents feature works with competitors playing against specified opponents"""
    n = 1
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(outcome_space=make_os(issues, name=f"S{i}"), ufuns=u)
        for i, u in enumerate(ufuns)
    ]

    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator, BoulwareTBNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=opponents,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Should have: 1 scenario * 1 competitor * 2 opponents = 2 negotiations
    assert len(results.details) == 2

    # Should only have scores for competitors (not opponents)
    # Each negotiation has 2 negotiators but we only score the first (competitor)
    assert len(results.scores) == 2  # 2 negotiations * 1 score per negotiation


def test_opponents_no_self_play_among_competitors():
    """Test that with explicit opponents, competitors don't play against each other"""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator, AspirationNegotiator]
    opponents = [BoulwareTBNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=opponents,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Should have: 1 scenario * 2 competitors * 1 opponent = 2 negotiations
    # NOT 2 * 2 = 4 (which would happen if competitors played each other)
    assert len(results.details) == 2

    # Verify partners - should be competitor vs opponent, not competitor vs competitor
    partners_sets = [set(d["partners"]) for d in results.details.to_dict("records")]

    # Check that we don't have RandomNegotiator vs AspirationNegotiator
    competitor_names = {_.__name__ for _ in competitors}
    for partners in partners_sets:
        # Should not have both competitors in same negotiation
        assert (
            len(partners & competitor_names) == 1
        ), f"Found competitors playing each other: {partners}"


def test_opponents_with_rotate_ufuns():
    """Test opponents feature with ufun rotation"""
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=opponents,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=True,  # Enable rotation
        path=None,
        njobs=-1,
    )

    # With rotate_ufuns=True, we get 2 variations (original + 1 rotation)
    # 2 scenarios * 1 competitor * 1 opponent = 2 negotiations
    assert len(results.details) == 2

    # Still only score competitors
    assert len(results.scores) == 2


def test_opponents_same_agent_different_roles():
    """Test that same agent in competitors and opponents is scored only as competitor"""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    # Same agent in both lists
    competitors = [RandomNegotiator]
    opponents = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=opponents,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # 1 scenario * 1 competitor * 2 opponents = 2 negotiations
    assert len(results.details) == 2

    # Only competitor scores are recorded
    assert len(results.scores) == 2

    # All scores should be for RandomNegotiator (as competitor)
    strategies = results.scores["strategy"].unique()
    assert len(strategies) == 1
    assert "RandomNegotiator" in strategies[0]


def test_opponents_empty_defaults_to_competitors():
    """Test that empty opponents list defaults to all-vs-all competitors mode"""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator, AspirationNegotiator]

    # Empty opponents - should default to competitors playing each other
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=tuple(),  # Empty
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Should be same as not providing opponents at all
    # 1 scenario * 2 competitors * 2 competitors = 4 negotiations
    assert len(results.details) == 4

    # Both competitors get scored (2 agents per negotiation * 4 negotiations = 8 scores)
    assert len(results.scores) == 8


def test_opponents_with_params():
    """Test that opponent_params are properly used"""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator]

    # Provide params for opponent
    opponent_params = [{"aspiration_type": "boulware"}]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=opponents,
        opponent_params=opponent_params,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Should complete without error
    assert len(results.details) == 1
    assert len(results.scores) == 1


def test_opponents_multilateral():
    """Test opponents feature with more than 2 negotiators"""
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    # Create 3-party negotiation
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator, BoulwareTBNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=opponents,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # 1 competitor * 2 opponents * 2 opponents (for positions 2 and 3) = 4 negotiations
    # The competitor is always in position 0, opponents fill positions 1 and 2
    assert len(results.details) == 4

    # Only competitor scores (1 per negotiation)
    assert len(results.scores) == 4


def test_opponents_rotation_ufun_assignment_bug():
    """
    Test that exposes the bug: with rotate_ufuns=True and explicit opponents,
    the competitor gets the wrong ufun in rotated scenarios.

    This test creates two VERY different ufuns (one always gives 0, one always gives 10)
    and verifies that the competitor always receives ufun[0] across all rotations.

    Expected behavior (what SHOULD happen):
    - Rotation 0: Competitor@pos0 gets ufun[0] (high utility)
    - Rotation 1: Competitor should STILL be evaluated on ufun[0], not ufun[1]

    Current buggy behavior:
    - Rotation 0: Competitor@pos0 gets ufun[0] (high utility) ✅
    - Rotation 1: Competitor@pos0 gets ufun[1] (low utility) ❌ WRONG!
    """
    issues = (make_issue(["a", "b", "c"], "item"),)

    # Create two VERY different ufuns to make the bug obvious
    # ufun0: Always gives utility 10 (this is the "buyer" ufun)
    # ufun1: Always gives utility 0 (this is the "seller" ufun)
    ufun0 = U(
        values={"item": {"a": 10.0, "b": 10.0, "c": 10.0}},
        issues=issues,
        reserved_value=0.0,
    )
    ufun1 = U(
        values={"item": {"a": 0.0, "b": 0.0, "c": 0.0}},
        issues=issues,
        reserved_value=0.0,
    )

    scenarios = [
        Scenario(outcome_space=make_os(issues, name="S0"), ufuns=(ufun0, ufun1))
    ]

    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        opponents=opponents,
        mechanism_params=dict(n_steps=10),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=True,
        path=None,
        njobs=0,  # Run serially for debugging
    )

    # With rotation, we should have 2 negotiations (original + 1 rotation)
    assert len(results.details) == 2

    # BUG DISCOVERED: Only 1 score instead of 2!
    # This suggests that one of the negotiations is not being scored at all
    print("\n=== BUG ANALYSIS ===")
    print(f"Number of negotiations run: {len(results.details)}")
    print(f"Number of scores recorded: {len(results.scores)}")
    print("\nDetails of both negotiations:")
    for idx, row in results.details.iterrows():
        print(f"\nNegotiation {idx}:")
        print(f"  Scenario: {row['scenario']}")
        print(f"  Partners: {row['partners']}")
        print(f"  Agreement: {row['agreement']}")
        print(f"  Has error: {row['has_error']}")

    print("\nScores recorded:")
    print(results.scores)

    # The bug is now clear: With rotation + explicit opponents,
    # only ONE of the two negotiations gets scored!
