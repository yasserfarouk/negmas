from __future__ import annotations

import os
from os import walk
from pathlib import Path

import pkg_resources
import pytest

from negmas import load_genius_domain_from_folder
from negmas.inout import Scenario
from negmas.outcomes import enumerate_issues
from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.sao import AspirationNegotiator

MAX_CARDINALITY = 10_000

try:
    # try to raise the maximum limit for open files
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (50000, -1))
except:
    pass

GENIUSWEB_FOLDERS = Path(
    pkg_resources.resource_filename("negmas", resource_name="tests/data/geniusweb")
)


@pytest.fixture
def scenarios_folder():
    return pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/scenarios"
    )


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
    for root, dirs, files in walk(base):
        if len(files) == 0 or len(dirs) != 0:
            continue
        if root.split("/")[-1] in SCENARIOS_TO_IGNORE:
            continue
        data.append(root)
    return data


def test_reading_writing_linear_preferences(tmp_path):
    from negmas.preferences import LinearAdditiveUtilityFunction, UtilityFunction

    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )
    domain = load_genius_domain_from_folder(
        base_folder,
    )
    ufuns, issues = domain.ufuns, domain.issues
    for ufun in ufuns:
        assert isinstance(ufun, LinearAdditiveUtilityFunction)
        dst = tmp_path / "tmp.xml"
        UtilityFunction.to_genius(ufun, issues=issues, file_name=dst)
        print(str(dst))
        ufun2, _ = UtilityFunction.from_genius(dst, issues=issues)
        try:
            os.unlink(dst)
        except:
            pass
        assert isinstance(ufun2, LinearAdditiveUtilityFunction)
        for outcome in enumerate_issues(issues):
            assert abs(ufun2(outcome) - ufun(outcome)) < 1e-3


def test_importing_file_without_exceptions(scenarios_folder):
    folder_name = scenarios_folder + "/other/S-1NIKFRT-1"
    load_genius_domain_from_folder(folder_name)


def test_simple_run_with_aspiration_agents():
    file_name = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )
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
                    preferences=domain.ufuns, aggregator=lambda x: True  # type: ignore
                )
            )
        )
        == 0
    )
    assert (
        len(
            list(
                domain.outcome_space.enumerate_or_sample_rational(
                    preferences=domain.ufuns, aggregator=lambda x: False  # type: ignore
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
