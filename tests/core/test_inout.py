from __future__ import annotations
import os
from importlib.resources import files
from os import walk
from pathlib import Path

import pytest

from negmas import load_genius_domain_from_folder
from negmas.inout import Scenario, STATS_FILE_NAME
from negmas.outcomes import enumerate_issues
from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.ops import ScenarioStats
from negmas.sao import AspirationNegotiator

MAX_CARDINALITY = 10_000

try:
    # try to raise the maximum limit for open files
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (50000, -1))
except Exception:
    pass

GENIUSWEB_FOLDERS = Path(str(files("negmas").joinpath("tests/data/geniusweb")))


@pytest.fixture
def scenarios_folder():
    return str(files("negmas").joinpath("tests/data/scenarios"))


# todo: get these to work
SCENARIOS_TO_IGNORE = [
    "S-1NIKFRT-3",
    "IntegerDomain",
    "S-1NIKFRT-2",
    "S-1NAGUNL-128",
    "S-1NAGUNL-255",
    "50issueswithRV",
    "50issues",
    "30issuesDiscounted",
    "30issueswithRV",
    "30issues",
    "50issuesDiscounted",
    "30issuesDiscountedwithRV",
    "50issuesDiscountedwithRV",
    "AgentHp2",
    "web_service",
    "four_issues",
    "AMPOvsCity",
    "laptopdomain",
    "S-1NIKFRT-1",
    "S-1NAGUNL-114",
    "laptopdomainNoBayes",
    "inheritancedomain",
    "10issues",
    "10issuesDiscounted",
    "10issueswithRV",
    "10issuesDiscountedwithRV",
    "group9-vacation",
    "group9-killer_robot",
    "FitnessA",
    "FitnessB",
    "FitnessC",
    "Grandma",
    "YXAgent",
    # "kite",
    # "group5-car_domain",
    # "group2-new_sporthal",
    # "group12-symposium",
    # "group11-car_purchase",
    # "group9-vacation",
    # "group8-holiday",
]


def get_all_scenarios():
    base = Path(__file__).parent.parent / "data" / "scenarios"
    data = []
    for root, dirs, filenames in walk(base):
        if len(filenames) == 0 or len(dirs) != 0:
            continue
        if root.split("/")[-1] in SCENARIOS_TO_IGNORE:
            continue
        data.append(root)
    return data


def test_reading_writing_linear_preferences(tmp_path):
    from negmas.preferences import LinearAdditiveUtilityFunction, UtilityFunction

    base_folder = str(files("negmas").joinpath("tests/data/Laptop"))
    domain = load_genius_domain_from_folder(base_folder)
    ufuns, issues = domain.ufuns, domain.issues
    for ufun in ufuns:
        assert isinstance(ufun, LinearAdditiveUtilityFunction)
        dst = tmp_path / "tmp.xml"
        UtilityFunction.to_genius(ufun, issues=issues, file_name=dst)
        print(str(dst))
        ufun2, _ = UtilityFunction.from_genius(dst, issues=issues)
        try:
            os.unlink(dst)
        except Exception:
            pass
        assert isinstance(ufun2, LinearAdditiveUtilityFunction)
        for outcome in enumerate_issues(issues):
            assert abs(ufun2(outcome) - ufun(outcome)) < 1e-3


def test_importing_file_without_exceptions(scenarios_folder):
    folder_name = scenarios_folder + "/other/S-1NIKFRT-1"
    load_genius_domain_from_folder(folder_name)


def test_simple_run_with_aspiration_agents():
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    assert os.path.exists(file_name)
    domain = Scenario.from_genius_folder(Path(file_name))
    assert domain
    domain.to_single_issue()
    mechanism = domain.make_session(AspirationNegotiator, n_steps=100, time_limit=30)
    assert mechanism is not None
    mechanism.run()


@pytest.mark.parametrize(
    ("r0", "r1", "n_above"),
    (
        (0.0, 0.0, 11),
        (-0.1, -0.1, 11),
        (0.9, 0.9, 0),
        (0.0, 0.9, 2),
        (0.9, 0.0, 2),
        (0.0, 0.95, 1),
        (0.95, 0.0, 1),
    ),
)
def test_enumerate_discrete_rational(r0, r1, n_above):
    domain = Scenario.from_genius_folder(  # type: ignore
        Path(__file__).parent.parent
        / "data"
        / "scenarios"
        / "anac"
        / "y2013"
        / "Fifty2013",
        safe_parsing=False,
        ignore_discount=True,
    )
    assert domain
    domain.scale_max()
    domain: Scenario
    ufuns = domain.ufuns
    ufuns[0].reserved_value = r0
    ufuns[1].reserved_value = r1
    outcomes = list(domain.outcome_space.enumerate_or_sample())
    assert len(outcomes) == 11
    assert (
        len(
            list(
                domain.outcome_space.enumerate_or_sample_rational(
                    preferences=domain.ufuns,
                    aggregator=lambda x: True,  # type: ignore
                )
            )
        )
        == 0
    )
    assert (
        len(
            list(
                domain.outcome_space.enumerate_or_sample_rational(
                    preferences=domain.ufuns,
                    aggregator=lambda x: False,  # type: ignore
                )
            )
        )
        == 11
    )
    assert (
        len(
            list(
                domain.outcome_space.enumerate_or_sample_rational(
                    preferences=domain.ufuns, aggregator=any
                )
            )
        )
        == n_above
    )


def test_load_geniusweb_example_reserved_outcome():
    domain = GENIUSWEB_FOLDERS / "Fitness"
    scenario = Scenario.from_geniusweb_folder(domain, use_reserved_outcome=True)
    assert scenario is not None
    assert isinstance(scenario.outcome_space, DiscreteCartesianOutcomeSpace)
    assert len(scenario.outcome_space.issues) == 5
    for i, (name, vals) in enumerate(
        (("type", 5), ("duration", 4), ("distance", 4), ("intensity", 4), ("price", 4))
    ):
        assert scenario.outcome_space.issues[i].name == name
        assert scenario.outcome_space.issues[i].cardinality == vals
    assert scenario.outcome_space.name == "fitness"
    assert len(scenario.ufuns) == 2
    assert all(isinstance(_, LinearAdditiveUtilityFunction) for _ in scenario.ufuns)
    assert all(
        (_.reserved_value is None or _.reserved_value == float("-inf"))
        and _.reserved_outcome is not None
        for _ in scenario.ufuns
    )
    assert scenario.ufuns[0].name == "fitness1"
    assert scenario.ufuns[1].name == "fitness2"


def test_load_geniusweb_example_reserved_value():
    domain = GENIUSWEB_FOLDERS / "Fitness"
    scenario = Scenario.from_geniusweb_folder(domain, use_reserved_outcome=False)
    assert scenario is not None
    assert isinstance(scenario.outcome_space, DiscreteCartesianOutcomeSpace)
    assert len(scenario.outcome_space.issues) == 5
    for i, (name, vals) in enumerate(
        (("type", 5), ("duration", 4), ("distance", 4), ("intensity", 4), ("price", 4))
    ):
        assert scenario.outcome_space.issues[i].name == name
        assert scenario.outcome_space.issues[i].cardinality == vals
    assert scenario.outcome_space.name == "fitness"
    assert len(scenario.ufuns) == 2
    assert all(isinstance(_, LinearAdditiveUtilityFunction) for _ in scenario.ufuns)
    assert all(
        _.reserved_value is not None and _.reserved_outcome is None
        for _ in scenario.ufuns
    )
    assert scenario.ufuns[0].name == "fitness1"
    assert scenario.ufuns[1].name == "fitness2"


# @pytest.mark.parametrize("folder_name", get_all_scenarios())
# def test_importing_all_single_issue_without_exceptions(folder_name):
#     # def test_encoding_decoding_all_without_discounting():
#     # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2012/FitnessC"
#     d2 = Domain.from_genius_folder(folder_name, safe_parsing=False)
#     d2.discretize()
#     assert isinstance(d2.outcome_space, DiscreteCartesianOutcomeSpace)
#     n = d2.outcome_space.cardinality
#     if n < 10_000:
#         d2.to_single_issue()
#         assert d2.outcome_space.cardinality == n or d2.outcome_space.cardinality == float("inf")


def test_scenario_calc_stats():
    """Test that calc_stats computes and stores stats correctly."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None
    assert scenario.stats is None

    stats = scenario.calc_stats()

    assert stats is not None
    assert isinstance(stats, ScenarioStats)
    assert scenario.stats is stats
    assert stats.opposition is not None
    assert len(stats.utility_ranges) == len(scenario.ufuns)
    assert len(stats.pareto_utils) > 0
    assert len(stats.pareto_outcomes) > 0


def test_scenario_save_and_load_stats(tmp_path):
    """Test that stats can be saved via dumpas and loaded back."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate stats
    stats = scenario.calc_stats()
    assert stats is not None

    # Save the scenario with stats
    scenario.dumpas(tmp_path, type="yml")

    # Verify stats file was created with correct name
    stats_file = tmp_path / STATS_FILE_NAME
    assert stats_file.exists(), f"Stats file {stats_file} should exist"

    # Load stats back
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2 is not None
    assert scenario2.stats is None

    scenario2.load_stats(tmp_path)
    assert scenario2.stats is not None

    # Verify loaded stats match original
    assert scenario2.stats.opposition == pytest.approx(stats.opposition, rel=1e-6)
    assert len(scenario2.stats.pareto_utils) == len(stats.pareto_utils)


def test_scenario_load_stats_file(tmp_path):
    """Test load_stats_file method."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate and save stats
    stats = scenario.calc_stats()
    scenario.dumpas(tmp_path, type="yml")

    stats_file = tmp_path / STATS_FILE_NAME
    assert stats_file.exists()

    # Load stats from specific file
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2.stats is None

    scenario2.load_stats_file(stats_file)
    assert scenario2.stats is not None
    assert scenario2.stats.opposition == pytest.approx(stats.opposition, rel=1e-6)


def test_scenario_load_stats_nonexistent(tmp_path):
    """Test that load_stats handles missing files gracefully."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None
    assert scenario.stats is None

    # Load from folder without stats file
    scenario.load_stats(tmp_path)
    assert scenario.stats is None

    # Load from nonexistent file
    scenario.load_stats_file(tmp_path / "nonexistent.yaml")
    assert scenario.stats is None


def test_scenario_load_with_stats(tmp_path):
    """Test that Scenario.load loads stats when available."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))

    # First create a scenario with stats and save it
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None
    scenario.calc_stats()
    scenario.dumpas(tmp_path, type="yml")

    # Now load from the saved location (which has stats)
    scenario_with_stats = Scenario.from_genius_folder(Path(file_name))
    assert scenario_with_stats is not None
    scenario_with_stats.load_stats(tmp_path)
    assert scenario_with_stats.stats is not None

    # Load without loading stats
    scenario_no_stats = Scenario.from_genius_folder(Path(file_name))
    assert scenario_no_stats is not None
    assert scenario_no_stats.stats is None


def test_scenario_calc_stats_attributes():
    """Test that calc_stats computes all expected attributes."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    stats = scenario.calc_stats()

    # Check all ScenarioStats attributes are populated
    assert isinstance(stats.opposition, float)
    assert 0.0 <= stats.opposition <= 1.0

    assert isinstance(stats.utility_ranges, list)
    for r in stats.utility_ranges:
        assert len(r) == 2
        assert r[0] <= r[1]

    assert isinstance(stats.pareto_utils, tuple)
    assert isinstance(stats.pareto_outcomes, list)
    assert len(stats.pareto_utils) == len(stats.pareto_outcomes)

    assert isinstance(stats.nash_utils, list)
    assert isinstance(stats.nash_outcomes, list)

    assert isinstance(stats.kalai_utils, list)
    assert isinstance(stats.kalai_outcomes, list)

    assert isinstance(stats.max_welfare_utils, list)
    assert isinstance(stats.max_welfare_outcomes, list)
