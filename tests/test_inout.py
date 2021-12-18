import os
from os import walk
from pathlib import Path

import pkg_resources
import pytest

from negmas import load_genius_domain_from_folder
from negmas.genius import genius_bridge_is_running
from negmas.inout import Domain
from negmas.outcomes import enumerate_issues
from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace
from negmas.preferences.discounted import DiscountedUtilityFunction
from negmas.preferences.nonlinear import HyperRectangleUtilityFunction
from negmas.sao import AspirationNegotiator


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
]


def get_all_scenarios():
    base = Path(__file__).parent / "data" / "scenarios"
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
    domain = Domain.from_genius_folder(Path(file_name))
    domain.to_single_issue()
    mechanism = domain.make_session(AspirationNegotiator, n_steps=100, time_limit=30)
    assert mechanism is not None
    mechanism.run()


def compared_two_domains(domain, domain2):
    from negmas.genius import AgentX, Atlas3

    assert len(domain.issues) == len(domain2.issues)
    for i1, i2 in zip(domain.issues, domain2.issues):
        assert (
            i1.cardinality == i2.cardinality
            and i1.type == i2.type
            and i1.value_type == i2.value_type
        )

    assert len(domain.ufuns) == len(domain2.ufuns)

    for u1, u2 in zip(domain.ufuns, domain2.ufuns):
        while isinstance(u1, DiscountedUtilityFunction):
            u1 = u1.ufun
        while isinstance(u2, DiscountedUtilityFunction):
            u2 = u2.ufun
        if isinstance(u1, HyperRectangleUtilityFunction) or isinstance(
            u2, HyperRectangleUtilityFunction
        ):
            continue
        dm = domain.agenda.to_discrete(5)
        for i, w in enumerate(dm):
            if i > 10_000:
                return
            assert abs(u1(w) - u2(w)) < 1e-3, f"{str(u1)}\n{str(u2)}"

    for ufun in domain.ufuns:
        if isinstance(ufun, HyperRectangleUtilityFunction):
            continue
        m = domain.make_session(n_steps=100, name=domain.agenda.name)
        assert m is not None
        if not genius_bridge_is_running():
            continue
        n1 = Atlas3(domain_file_name=m.name, ufun=ufun)
        n2 = AgentX(domain_file_name=m.name, utility_file_name=ufun.name)
        m.add(n1)
        m.add(n2)
        u1, u2 = n1.ufun, n2.ufun
        outcomes = m.discrete_outcomes(n_max=100)
        for outcome in outcomes:
            assert abs(u1(outcome) - u2(outcome)) < 1e-3
        n1.destroy_java_counterpart()
        n2.destroy_java_counterpart()


@pytest.mark.parametrize("folder_name", get_all_scenarios())
def test_encoding_decoding_all_without_discounting(tmp_path, folder_name):
    # def test_encoding_decoding_all_without_discounting(tmp_path):
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/other/IntegerDomain"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/other/AMPOvsCity"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/other/laptopdomain"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/other/laptopdomainNoBayes"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/other/inheritancedomain"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/other/web_service"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2012/FitnessA"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2012/FitnessB"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2012/FitnessC"
    # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2016/AgentHp2"
    domain = Domain.from_genius_folder(
        folder_name, safe_parsing=False
    ).remove_discounting()
    tmp = tmp_path / "tmp"
    print(f"{str(folder_name)}\n-> {str(tmp)}")
    domain.to_genius_folder(tmp)
    domain2 = Domain.from_genius_folder(tmp).remove_discounting()
    compared_two_domains(domain, domain2)


@pytest.mark.parametrize("folder_name", get_all_scenarios())
def test_encoding_decoding_all_with_discounting(tmp_path, folder_name):
    # def test_encoding_decoding_all_with_discounting(tmp_path):
    # folder_name ="/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2011/Grocery"
    domain = Domain.from_genius_folder(folder_name, safe_parsing=False)
    tmp = tmp_path / "tmp"
    print(f"{str(folder_name)}\n-> {str(tmp)}")
    domain.to_genius_folder(tmp)
    domain2 = Domain.from_genius_folder(tmp)
    compared_two_domains(domain, domain2)


# @pytest.mark.parametrize("folder_name", get_all_scenarios())
# def test_importing_all_single_issue_without_exceptions(folder_name):
#     # def test_encoding_decoding_all_without_discounting():
#     # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2012/FitnessC"
#     d2 = Domain.from_genius_folder(folder_name, safe_parsing=False)
#     d2.discretize()
#     assert isinstance(d2.agenda, DiscreteCartesianOutcomeSpace)
#     n = d2.agenda.cardinality
#     if n < 10_000:
#         d2.to_single_issue()
#         assert d2.agenda.cardinality == n or d2.agenda.cardinality == float("inf")
