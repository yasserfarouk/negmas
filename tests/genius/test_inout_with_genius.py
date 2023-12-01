from __future__ import annotations

import shutil
from os import walk
from pathlib import Path

import pytest
from pytest import mark

from negmas.genius.bridge import genius_bridge_is_running
from negmas.inout import Scenario
from negmas.preferences.crisp.nonlinear import HyperRectangleUtilityFunction
from negmas.preferences.discounted import DiscountedUtilityFunction

MAX_CARDINALITY = 10_000

try:
    # try to raise the maximum limit for open files
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (50000, -1))
except:
    pass

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


def compared_two_domains(domain, domain2):
    from negmas.genius.gnegotiators import AgentX, Atlas3

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
        dm = domain.outcome_space.to_discrete(5)
        for i, w in enumerate(dm):
            if i > MAX_CARDINALITY:
                return
            u1_, u2_ = u1(w), u2(w)
            assert isinstance(u1_, float)
            assert isinstance(u2_, float)
            assert abs(u1_ - u2_) < 1e-3, f"{str(u1)}\n{str(u2)}"

    for ufun in domain.ufuns:
        if isinstance(ufun, HyperRectangleUtilityFunction):
            continue
        m = domain.make_session(n_steps=100, name=domain.outcome_space.name)
        assert m is not None
        if not genius_bridge_is_running():
            continue
        n1 = Atlas3(domain_file_name=m.name, preferences=ufun)
        n2 = AgentX(domain_file_name=m.name, utility_file_name=ufun.name)
        m.add(n1)
        m.add(n2)
        u1, u2 = n1.ufun, n2.ufun
        assert u1 and u2
        outcomes = m.discrete_outcomes(max_cardinality=100)
        for outcome in outcomes:
            u1_, u2_ = u1(outcome), u2(outcome)
            assert isinstance(u1_, float)
            assert isinstance(u2_, float)
            assert abs(u1_ - u2_) < 1e-3
        n1.destroy_java_counterpart()
        n2.destroy_java_counterpart()


def do_enc_dec_trial(tmp, folder_name, with_discounting=True):
    domain = Scenario.from_genius_folder(folder_name, safe_parsing=False)
    assert domain
    domain = domain.remove_discounting()
    print(f"{str(folder_name)}\n-> {str(tmp)}")
    domain.to_genius_folder(tmp)
    domain2 = Scenario.from_genius_folder(tmp)
    assert domain2
    if not with_discounting:
        domain2 = domain2.remove_discounting()
    compared_two_domains(domain, domain2)
    try:
        shutil.rmtree(tmp)
    except:
        pass


@mark.xfail(
    run=False, reason="Known to fail. It is the int/discrete issue ambiguity in Genius"
)
@pytest.mark.parametrize("disc", [True, False])
def test_encoding_decoding_example_AMPOvsCity(tmp_path, disc):
    folder_name = (
        Path(__file__).parent.parent / "data" / "scenarios" / "other" / "AMPOvsCity"
    )
    do_enc_dec_trial(tmp_path / "tmp", folder_name, disc)


@pytest.mark.parametrize("disc", [True, False])
def test_encoding_decoding_example_group_8_holiday(tmp_path, disc):
    folder_name = (
        Path(__file__).parent.parent
        / "data"
        / "scenarios"
        / "anac"
        / "y2015"
        / "group8-holiday"
    )
    do_enc_dec_trial(tmp_path / "tmp", folder_name, disc)


@pytest.mark.parametrize("folder_name", get_all_scenarios())
def test_encoding_decoding_all_without_discounting(tmp_path, folder_name):
    do_enc_dec_trial(tmp_path / "tmp", folder_name, False)


@pytest.mark.parametrize("folder_name", get_all_scenarios())
def test_encoding_decoding_all_with_discounting(tmp_path, folder_name):
    do_enc_dec_trial(tmp_path / "tmp", folder_name, True)
