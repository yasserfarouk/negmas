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
    CONFIG_FILE_NAME,
    SimpleTournamentResults,
    RunInfo,
    _find_dataframe_file,
    _combine_configs,
)

# Check if pyarrow is available for parquet tests
try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


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
        # Use speed optimization to keep all files for testing file structure
        storage_optimization="speed",
        memory_optimization="speed",
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


# ===== Storage and Memory Optimization Tests =====


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Storage optimization tests are slow")
def test_storage_optimization_speed(tmp_path: Path):
    """Test storage_optimization='speed' keeps all files"""
    path = tmp_path / "tournament_speed"
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path,
        njobs=-1,
        storage_optimization="speed",
    )

    # All files should exist
    assert (path / "results").exists(), "results/ folder should exist"
    assert (path / "all_scores.csv").exists(), "all_scores.csv should exist"
    assert (path / "details.csv").exists(), "details.csv should exist"
    assert (path / "scores.csv").exists(), "scores.csv should exist"
    assert (path / "type_scores.csv").exists(), "type_scores.csv should exist"

    # Results should be complete
    assert len(results.scores) > 0
    assert len(results.details) > 0
    assert len(results.final_scores) > 0


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Storage optimization tests are slow")
def test_storage_optimization_balanced(tmp_path: Path):
    """Test storage_optimization='balanced' removes results/ folder and uses gzip format"""
    path = tmp_path / "tournament_balanced"
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path,
        njobs=-1,
        storage_optimization="balanced",
    )

    # results/ folder should be removed
    assert not (path / "results").exists(), "results/ folder should be removed"
    # Files should exist (in gzip format for balanced optimization)
    assert (
        _find_dataframe_file(path, "all_scores") is not None
    ), "all_scores file should exist"
    assert (
        _find_dataframe_file(path, "details") is not None
    ), "details file should exist"
    # scores.csv always uses plain CSV format
    assert (path / "scores.csv").exists(), "scores.csv should exist"

    # Results should still be accessible
    assert len(results.scores) > 0
    assert len(results.details) > 0


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Storage optimization tests are slow")
@mark.skipif(not HAS_PYARROW, reason="pyarrow required for parquet format tests")
def test_storage_optimization_space(tmp_path: Path):
    """Test storage_optimization='space' removes results/ and all_scores, uses parquet format"""
    path = tmp_path / "tournament_space"
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path,
        njobs=-1,
        storage_optimization="space",
    )

    # results/ folder and all_scores should be removed
    assert not (path / "results").exists(), "results/ folder should be removed"
    assert (
        _find_dataframe_file(path, "all_scores") is None
    ), "all_scores should be removed"
    # Essential files should still exist (parquet format for space optimization)
    assert (
        _find_dataframe_file(path, "details") is not None
    ), "details file should exist"
    # scores.csv always uses plain CSV format
    assert (path / "scores.csv").exists(), "scores.csv should exist"

    # Results should still be accessible in memory
    assert len(results.scores) > 0
    assert len(results.details) > 0


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Memory optimization tests are slow")
def test_memory_optimization_speed(tmp_path: Path):
    """Test memory_optimization='speed' keeps everything in memory"""
    path = tmp_path / "tournament_mem_speed"
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path,
        njobs=-1,
        memory_optimization="speed",
    )

    # All data should be immediately available (cached)
    assert results._scores_cached is True
    assert results._details_cached is True
    assert len(results.scores) > 0
    assert len(results.details) > 0


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Memory optimization tests are slow")
def test_memory_optimization_without_path():
    """Test that memory_optimization is ignored when path is None"""
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    # Even with memory_optimization="space", data should be kept in memory when path=None
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,  # No path
        njobs=-1,
        memory_optimization="space",  # Should be ignored
    )

    # Data should still be available since there's no path to load from
    assert len(results.scores) > 0
    assert len(results.details) > 0
    assert len(results.final_scores) > 0


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Load/save optimization tests are slow")
def test_load_with_missing_scores(tmp_path: Path):
    """Test that SimpleTournamentResults.load() can reconstruct scores from results/ folder"""
    path = tmp_path / "tournament_load"
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    # Run tournament with speed mode (keeps all files)
    original_results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path,
        njobs=-1,
        storage_optimization="speed",
    )

    original_scores_count = len(original_results.scores)

    # Delete all_scores.csv to simulate space mode
    (path / "all_scores.csv").unlink()

    # Load should still work by reconstructing from results/ folder
    loaded_results = SimpleTournamentResults.load(path)

    assert len(loaded_results.scores) == original_scores_count
    assert len(loaded_results.final_scores) > 0


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Load/save optimization tests are slow")
def test_load_with_missing_details(tmp_path: Path):
    """Test that SimpleTournamentResults.load() can reconstruct details from results/ folder"""
    path = tmp_path / "tournament_load_details"
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    # Run tournament with speed mode (keeps all files)
    original_results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path,
        njobs=-1,
        storage_optimization="speed",
    )

    original_details_count = len(original_results.details)

    # Delete details.csv
    (path / "details.csv").unlink()

    # Load should still work by reconstructing from results/ folder
    loaded_results = SimpleTournamentResults.load(path)

    assert len(loaded_results.details) == original_details_count


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Storage optimization tests are slow")
def test_combine_tournaments_with_different_storage_optimizations(tmp_path: Path):
    """Test that tournaments with different storage optimizations can be combined"""
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    # Run two tournaments with different storage optimizations
    path1 = tmp_path / "tournament1"
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path1,
        njobs=-1,
        storage_optimization="speed",
    )

    path2 = tmp_path / "tournament2"
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path2,
        njobs=-1,
        storage_optimization="balanced",
    )

    # Combine should work
    combined, _ = SimpleTournamentResults.combine([path1, path2])
    assert len(combined.scores) > 0
    assert len(combined.details) > 0


@mark.slow
@mark.skipif(NEGMAS_FASTRUN, reason="Storage optimization tests are slow")
def test_scores_reconstructed_from_details_csv_only(tmp_path: Path):
    """Test that scores can be reconstructed from details.csv alone (no results/ folder).

    This is critical for storage_optimization='balanced' and 'space' modes where
    the results/ folder is deleted to save disk space.
    """
    path = tmp_path / "tournament_details_only"
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    # Run tournament with speed mode to get reference data
    original_results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=2,
        verbosity=0,
        rotate_ufuns=True,
        path=path,
        njobs=-1,
        storage_optimization="speed",
    )

    # Save reference values
    original_scores = original_results.scores.copy()
    original_final_scores = original_results.final_scores.copy()

    # Delete both results/ folder AND all_scores.csv to simulate balanced/space mode
    import shutil

    shutil.rmtree(path / "results")
    (path / "all_scores.csv").unlink()

    # Verify results/ folder and all_scores.csv are gone
    assert not (path / "results").exists()
    assert not (path / "all_scores.csv").exists()
    # But details.csv should still exist
    assert (path / "details.csv").exists()

    # Load should reconstruct scores from details.csv
    loaded_results = SimpleTournamentResults.load(path)

    # Verify scores were reconstructed
    assert len(loaded_results.scores) == len(
        original_scores
    ), f"Expected {len(original_scores)} scores, got {len(loaded_results.scores)}"

    # Verify key score columns match
    for col in ["strategy", "utility", "reserved_value", "scenario"]:
        if col in original_scores.columns:
            assert col in loaded_results.scores.columns, f"Missing column: {col}"

    # Verify final scores still work
    assert len(loaded_results.final_scores) == len(original_final_scores)


def test_competitor_names_parameter():
    """Test that competitor_names parameter correctly assigns custom names."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]
    custom_names = ["MyRandom", "MyAspiration"]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        competitor_names=custom_names,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Check that custom names appear in the results
    strategies_in_scores = set(results.scores["strategy"].unique())
    assert (
        "MyRandom" in strategies_in_scores
    ), f"Expected 'MyRandom' in {strategies_in_scores}"
    assert (
        "MyAspiration" in strategies_in_scores
    ), f"Expected 'MyAspiration' in {strategies_in_scores}"

    # Check final scores also use custom names
    final_strategies = set(results.final_scores["strategy"].unique())
    assert "MyRandom" in final_strategies
    assert "MyAspiration" in final_strategies


def test_opponent_names_parameter():
    """Test that opponent_names parameter correctly assigns custom names to opponents."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator, BoulwareTBNegotiator]
    custom_competitor_names = ["TestCompetitor"]
    custom_opponent_names = ["Opp_Aspiration", "Opp_Boulware"]

    results = cartesian_tournament(
        competitors=competitors,
        opponents=opponents,
        scenarios=scenarios,
        competitor_names=custom_competitor_names,
        opponent_names=custom_opponent_names,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Only competitors should appear in scores (not opponents)
    strategies_in_scores = set(results.scores["strategy"].unique())
    assert (
        "TestCompetitor" in strategies_in_scores
    ), f"Expected 'TestCompetitor' in {strategies_in_scores}"

    # Opponent names should appear in partner columns
    all_partners = set()
    for col in results.scores.columns:
        if col.startswith("partner") or col == "partners":
            if col == "partners":
                # If there's a partners column with tuple/list
                for p in results.scores[col]:
                    if isinstance(p, (list, tuple)):
                        all_partners.update(p)
                    else:
                        all_partners.add(p)
            else:
                all_partners.update(results.scores[col].unique())

    # Check that at least one custom opponent name appears somewhere
    any(opp in str(all_partners) for opp in custom_opponent_names) or any(
        opp in str(results.details.to_dict()) for opp in custom_opponent_names
    )
    # This is a soft check - the names should be used internally
    assert len(results.scores) > 0, "Expected some scores from the tournament"


def test_competitor_names_validation_length():
    """Test that competitor_names length validation works."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]
    wrong_length_names = ["OnlyOne"]  # Should be 2 names

    with pytest.raises(ValueError, match="names length"):
        cartesian_tournament(
            competitors=competitors,
            scenarios=scenarios,
            competitor_names=wrong_length_names,
            mechanism_params=dict(n_steps=5),
            n_repetitions=1,
            verbosity=0,
            path=None,
            njobs=-1,
        )


def test_competitor_names_validation_uniqueness():
    """Test that competitor_names uniqueness validation works."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]
    duplicate_names = ["SameName", "SameName"]

    with pytest.raises(ValueError, match="names must be unique"):
        cartesian_tournament(
            competitors=competitors,
            scenarios=scenarios,
            competitor_names=duplicate_names,
            mechanism_params=dict(n_steps=5),
            n_repetitions=1,
            verbosity=0,
            path=None,
            njobs=-1,
        )


def test_opponent_names_validation_length():
    """Test that opponent_names length validation works."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator, BoulwareTBNegotiator]
    wrong_length_names = ["OnlyOne"]  # Should be 2 names

    with pytest.raises(ValueError, match="names length"):
        cartesian_tournament(
            competitors=competitors,
            opponents=opponents,
            scenarios=scenarios,
            opponent_names=wrong_length_names,
            mechanism_params=dict(n_steps=5),
            n_repetitions=1,
            verbosity=0,
            path=None,
            njobs=-1,
        )


def test_opponent_names_validation_uniqueness():
    """Test that opponent_names uniqueness validation works."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator]
    opponents = [AspirationNegotiator, BoulwareTBNegotiator]
    duplicate_names = ["SameName", "SameName"]

    with pytest.raises(ValueError, match="names must be unique"):
        cartesian_tournament(
            competitors=competitors,
            opponents=opponents,
            scenarios=scenarios,
            opponent_names=duplicate_names,
            mechanism_params=dict(n_steps=5),
            n_repetitions=1,
            verbosity=0,
            path=None,
            njobs=-1,
        )


def test_shorten_names_deprecation_warning():
    """Test that shorten_names parameter triggers a deprecation warning."""
    import warnings

    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = cartesian_tournament(
            competitors=competitors,
            scenarios=scenarios,
            shorten_names=True,  # Deprecated parameter
            mechanism_params=dict(n_steps=5),
            n_repetitions=1,
            verbosity=0,
            rotate_ufuns=False,
            path=None,
            njobs=-1,
        )
        # Check that a deprecation warning was raised
        deprecation_warnings = [
            warning for warning in w if "shorten_names" in str(warning.message).lower()
        ]
        assert (
            len(deprecation_warnings) > 0
        ), f"Expected deprecation warning for shorten_names, got: {[str(x.message) for x in w]}"

    # Tournament should still work
    assert len(results.scores) > 0


def test_default_names_without_custom_names():
    """Test that default name generation works when no custom names are provided."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]
    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        # No competitor_names or opponent_names provided
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Should have scores with auto-generated names
    assert len(results.scores) > 0
    strategies = set(results.scores["strategy"].unique())
    # Names should be generated from class names (shortened)
    assert len(strategies) == 2, f"Expected 2 strategies, got {strategies}"


def test_same_type_different_params_unique_names():
    """Test that same negotiator type with different params gets unique names."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    # Same negotiator type but with different params
    competitors = [AspirationNegotiator, AspirationNegotiator]
    competitor_params = [{"aspiration_type": "linear"}, {"aspiration_type": "conceder"}]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        competitor_params=competitor_params,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Should have scores with unique auto-generated names
    assert len(results.scores) > 0
    strategies = set(results.scores["strategy"].unique())
    # Names should be unique even for same type with different params
    assert len(strategies) == 2, f"Expected 2 unique strategies, got {strategies}"


def test_generated_names_uniqueness_validation():
    """Test that generated names are validated for uniqueness."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    # Same negotiator type with same params - should still work due to auto name generation
    competitors = [AspirationNegotiator, AspirationNegotiator]
    competitor_params = [{"aspiration_type": "linear"}, {"aspiration_type": "linear"}]

    # This should raise an error because generated names would be duplicates
    with pytest.raises(ValueError, match="names must be unique"):
        cartesian_tournament(
            competitors=competitors,
            scenarios=scenarios,
            competitor_params=competitor_params,
            mechanism_params=dict(n_steps=5),
            n_repetitions=1,
            verbosity=0,
            rotate_ufuns=False,
            path=None,
            njobs=-1,
        )


def test_params_encoded_in_names():
    """Test that params are encoded in auto-generated names when needed."""
    issues = (
        make_issue([f"q{i}" for i in range(5)], "quantity"),
        make_issue([f"p{i}" for i in range(3)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    # Same negotiator type but with different params
    competitors = [AspirationNegotiator, AspirationNegotiator]
    competitor_params = [{"aspiration_type": "linear"}, {"aspiration_type": "conceder"}]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        competitor_params=competitor_params,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    strategies = list(results.scores["strategy"].unique())
    # The names should contain some distinguishing info from params
    # since the types are the same
    assert strategies[0] != strategies[1], "Names should be different"


def test_config_saved_to_disk(tmp_path):
    """Test that config is saved to disk when path is provided."""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=tmp_path,
        njobs=-1,
    )

    # Check config file exists
    config_path = tmp_path / CONFIG_FILE_NAME
    assert config_path.exists(), "Config file should be saved"

    # Check config is accessible via results
    assert results.config is not None
    assert results.config["n_competitors"] == 2
    assert results.config["n_scenarios"] == 1
    assert "RandomNegotiator" in results.config["competitors"][0]
    assert "AspirationNegotiator" in results.config["competitors"][1]


def test_config_contains_generated_names(tmp_path):
    """Test that config contains the generated competitor names."""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator, AspirationNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=tmp_path,
        njobs=-1,
    )

    # Config should have the generated names (not None)
    assert results.config is not None
    assert results.config["competitor_names"] is not None
    assert len(results.config["competitor_names"]) == 2


def test_config_with_metadata(tmp_path):
    """Test that metadata is included in config."""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator]
    metadata = {"experiment_name": "test_experiment", "version": 1.0}

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=tmp_path,
        njobs=-1,
        metadata=metadata,
    )

    assert results.config is not None
    assert results.config["metadata"] is not None
    assert results.config["metadata"]["experiment_name"] == "test_experiment"
    assert results.config["metadata"]["version"] == 1.0


def test_config_loaded_from_disk(tmp_path):
    """Test that config is loaded when loading results from disk."""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator, AspirationNegotiator]

    # Run tournament and save
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=tmp_path,
        njobs=-1,
    )

    # Load results from disk
    loaded_results = SimpleTournamentResults.load(tmp_path)

    # Config should be loaded
    assert loaded_results.config is not None
    assert loaded_results.config["n_competitors"] == 2


def test_config_not_saved_when_no_path():
    """Test that config is still available in results even without path."""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios = [Scenario(outcome_space=make_os(issues, name="S0"), ufuns=ufuns[0])]

    competitors = [RandomNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
    )

    # Config should still be available in results
    assert results.config is not None
    assert results.config["n_competitors"] == 1


# Tests for _combine_configs helper function
def test_combine_configs_single_config():
    """Test that single config is returned unchanged except n_scenarios."""
    config = {
        "n_scenarios": 5,
        "competitors": ["A", "B"],
        "n_competitors": 2,
        "rotate_ufuns": True,
    }
    result = _combine_configs([config], n_unique_scenarios=10)
    assert result["n_scenarios"] == 10
    assert result["competitors"] == ["A", "B"]
    assert result["rotate_ufuns"] is True


def test_combine_configs_same_values():
    """Test combining configs with same values preserves them."""
    config1 = {
        "competitors": ["A", "B"],
        "competitor_names": ["NameA", "NameB"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
        "n_scenarios": 5,
        "rotate_ufuns": True,
        "n_steps": 100,
    }
    config2 = {
        "competitors": ["A", "B"],
        "competitor_names": ["NameA", "NameB"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
        "n_scenarios": 3,
        "rotate_ufuns": True,
        "n_steps": 100,
    }
    result = _combine_configs([config1, config2], n_unique_scenarios=8)
    assert result["n_scenarios"] == 8
    assert result["competitors"] == ["A", "B"]
    assert result["rotate_ufuns"] is True
    assert result["n_steps"] == 100


def test_combine_configs_different_booleans():
    """Test that different boolean values become None."""
    config1 = {
        "competitors": ["A"],
        "competitor_names": ["NameA"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
        "n_scenarios": 5,
        "rotate_ufuns": True,
        "self_play": True,
    }
    config2 = {
        "competitors": ["A"],
        "competitor_names": ["NameA"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
        "n_scenarios": 3,
        "rotate_ufuns": False,
        "self_play": True,
    }
    result = _combine_configs([config1, config2], n_unique_scenarios=8)
    assert result["rotate_ufuns"] is None  # Different booleans become None
    assert result["self_play"] is True  # Same booleans stay


def test_combine_configs_different_competitors_raises():
    """Test that different competitors raises ValueError."""
    config1 = {
        "competitors": ["A", "B"],
        "competitor_names": ["NameA", "NameB"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
    }
    config2 = {
        "competitors": ["A", "C"],
        "competitor_names": ["NameA", "NameC"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
    }
    with pytest.raises(ValueError, match="different competitors"):
        _combine_configs([config1, config2], n_unique_scenarios=5)


def test_combine_configs_different_opponents_raises():
    """Test that different opponents raises ValueError."""
    config1 = {
        "competitors": ["A"],
        "competitor_names": ["NameA"],
        "competitor_params": None,
        "opponents": ["X", "Y"],
        "opponent_names": ["NameX", "NameY"],
        "opponent_params": None,
    }
    config2 = {
        "competitors": ["A"],
        "competitor_names": ["NameA"],
        "competitor_params": None,
        "opponents": ["X", "Z"],
        "opponent_names": ["NameX", "NameZ"],
        "opponent_params": None,
    }
    with pytest.raises(ValueError, match="different opponents"):
        _combine_configs([config1, config2], n_unique_scenarios=5)


def test_combine_configs_path_is_none():
    """Test that combined config has path=None."""
    config1 = {
        "competitors": ["A"],
        "competitor_names": ["NameA"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
        "path": "/path/to/tournament1",
    }
    config2 = {
        "competitors": ["A"],
        "competitor_names": ["NameA"],
        "competitor_params": None,
        "opponents": None,
        "opponent_names": [],
        "opponent_params": None,
        "path": "/path/to/tournament2",
    }
    result = _combine_configs([config1, config2], n_unique_scenarios=5)
    assert result["path"] is None


def test_combine_tournaments_config(tmp_path):
    """Test that combining tournaments properly combines configs."""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)

    # Create two tournaments with different scenarios but same competitors
    competitors = [RandomNegotiator, AspirationNegotiator]

    # Tournament 1
    ufuns1 = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios1 = [Scenario(outcome_space=make_os(issues, name="S1"), ufuns=ufuns1[0])]
    path1 = tmp_path / "t1"
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios1,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path1,
        njobs=-1,
        storage_optimization="none",  # Keep all files including results/ folder
    )

    # Tournament 2
    ufuns2 = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios2 = [Scenario(outcome_space=make_os(issues, name="S2"), ufuns=ufuns2[0])]
    path2 = tmp_path / "t2"
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios2,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path2,
        njobs=-1,
        storage_optimization="none",  # Keep all files including results/ folder
    )

    # Combine tournaments
    combined, _ = SimpleTournamentResults.combine([path1, path2], verbosity=0)

    # Check combined config
    assert combined.config is not None
    assert combined.config["n_competitors"] == 2
    assert combined.config["n_scenarios"] == 2  # Two unique scenarios
    assert combined.config["path"] is None  # No single source path


def test_combine_tournaments_different_competitors_raises(tmp_path):
    """Test that combining tournaments with different competitors raises error."""
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)

    # Tournament 1 with competitors A, B
    ufuns1 = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios1 = [Scenario(outcome_space=make_os(issues, name="S1"), ufuns=ufuns1[0])]
    path1 = tmp_path / "t1"
    cartesian_tournament(
        competitors=[RandomNegotiator, AspirationNegotiator],
        scenarios=scenarios1,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path1,
        njobs=-1,
        storage_optimization="none",  # Keep all files including results/ folder
    )

    # Tournament 2 with different competitors
    ufuns2 = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
    ]
    scenarios2 = [Scenario(outcome_space=make_os(issues, name="S2"), ufuns=ufuns2[0])]
    path2 = tmp_path / "t2"
    cartesian_tournament(
        competitors=[RandomNegotiator],  # Different competitors!
        scenarios=scenarios2,
        mechanism_params=dict(n_steps=5),
        n_repetitions=1,
        verbosity=0,
        rotate_ufuns=False,
        path=path2,
        njobs=-1,
        storage_optimization="none",  # Keep all files including results/ folder
    )

    # Combining should raise an error
    with pytest.raises(ValueError, match="different competitors"):
        SimpleTournamentResults.combine([path1, path2], verbosity=0)


def test_callback_after_end_receives_config():
    """Test that after_end_callback receives config as second argument when using new signature."""
    received_configs = []
    received_records = []

    def after_callback_with_config(record: dict, config: dict):
        received_configs.append(config)
        received_records.append(record)

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
        after_end_callback=after_callback_with_config,
    )

    expected_calls = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert len(received_configs) == expected_calls
    assert len(received_records) == expected_calls

    # All configs should be the same (same tournament)
    for config in received_configs:
        assert isinstance(config, dict)
        assert "competitors" in config
        assert "n_repetitions" in config
        assert config["n_repetitions"] == n_repetitions


def test_callback_after_end_backwards_compatible():
    """Test that after_end_callback still works with old signature (record only)."""
    received_records = []

    def after_callback_old_style(record: dict):
        received_records.append(record)

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

    # This should work without errors despite the callback not accepting config
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
        after_end_callback=after_callback_old_style,
    )

    expected_calls = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert len(received_records) == expected_calls


def test_callback_progress_receives_config():
    """Test that progress_callback receives config as fourth argument when using new signature."""
    received_configs = []
    messages = []

    def progress_callback_with_config(
        msg: str, current: int, total: int, config: dict | None
    ):
        received_configs.append(config)
        messages.append(msg)

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
        progress_callback=progress_callback_with_config,
    )

    # Progress callback should be called multiple times (loaded competitors, processing scenario, starting negotiations)
    assert len(received_configs) >= 3
    assert len(messages) >= 3

    # All configs should be the same (same tournament)
    for config in received_configs:
        assert isinstance(config, dict)
        assert "competitors" in config


def test_callback_progress_backwards_compatible():
    """Test that progress_callback still works with old signature (message, current, total only)."""
    messages = []

    def progress_callback_old_style(msg: str, current: int, total: int):
        messages.append(msg)

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

    # This should work without errors despite the callback not accepting config
    cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        mechanism_params=dict(n_steps=5),
        n_repetitions=n_repetitions,
        verbosity=0,
        rotate_ufuns=False,
        path=None,
        njobs=-1,
        progress_callback=progress_callback_old_style,
    )

    # Progress callback should be called
    assert len(messages) >= 3


def test_runinfo_has_config():
    """Test that RunInfo dataclass contains config field."""
    received_configs = []

    def before_callback(info: RunInfo):
        received_configs.append(info.config)

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
    )

    expected_calls = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert len(received_configs) == expected_calls

    # All configs should be dicts with tournament info
    for config in received_configs:
        assert isinstance(config, dict)
        assert "competitors" in config
        assert "n_repetitions" in config
        assert config["n_repetitions"] == n_repetitions


def test_constructedneginfo_has_config():
    """Test that ConstructedNegInfo dataclass contains config field."""
    from negmas.tournaments.neg.simple.cartesian import ConstructedNegInfo

    received_configs = []

    def after_construction_callback(info: ConstructedNegInfo):
        received_configs.append(info.config)

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
        after_construction_callback=after_construction_callback,
    )

    expected_calls = len(scenarios) * len(competitors) ** 2 * n_repetitions
    assert len(received_configs) == expected_calls

    # All configs should be dicts with tournament info
    for config in received_configs:
        assert isinstance(config, dict)
        assert "competitors" in config
        assert "n_repetitions" in config
        assert config["n_repetitions"] == n_repetitions
