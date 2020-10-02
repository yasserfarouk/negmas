import os
from os import walk

import pkg_resources
import pytest

from negmas import AspirationNegotiator, load_genius_domain_from_folder
from negmas.genius import genius_bridge_is_running


@pytest.fixture
def scenarios_folder():
    return pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/scenarios"
    )


def test_reading_writing_linear_ufun(tmp_path):
    from negmas.utilities import LinearUtilityAggregationFunction, UtilityFunction
    from negmas.outcomes import Issue

    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )
    neg, agent_info, issues = load_genius_domain_from_folder(
        base_folder, keep_issue_names=True, keep_value_names=True,
    )
    ufuns = [_["ufun"] for _ in agent_info]
    for ufun in ufuns:
        assert isinstance(ufun, LinearUtilityAggregationFunction)
        dst = tmp_path / "tmp.xml"
        print(dst)
        UtilityFunction.to_genius(ufun, issues, dst)
        ufun2, _ = UtilityFunction.from_genius(dst)
        assert isinstance(ufun2, LinearUtilityAggregationFunction)
        for outcome in Issue.enumerate(issues):
            assert abs(ufun2(outcome) - ufun(outcome)) < 1e-3


def test_importing_file_without_exceptions(scenarios_folder):
    folder_name = scenarios_folder + "/other/S-1NIKFRT-1"
    domain = load_genius_domain_from_folder(folder_name, n_discretization=10)
    # print(domain)


def test_convert_dir_no_names(tmpdir):
    from negmas import convert_genius_domain_from_folder

    dst = tmpdir.mkdir("sub")
    src = pkg_resources.resource_filename("negmas", resource_name="tests/data/Laptop")
    dst = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/LaptopConv"
    )

    assert convert_genius_domain_from_folder(
        src_folder_name=src,
        dst_folder_name=dst,
        force_single_issue=True,
        cache_and_discretize_outcomes=False,
        n_discretization=None,
        keep_issue_names=False,
        keep_value_names=False,
        normalize_utilities=True,
    )
    mechanism, agent_info, issues = load_genius_domain_from_folder(
        dst,
        keep_value_names=True,
        keep_issue_names=True,
        normalize_utilities=False,
        force_single_issue=False,
    )
    assert len(issues) == 1
    for k, v in enumerate(issues):
        assert (
            f"{k}:{v}"
            == """0:0: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']"""
        )


def test_simple_run_with_aspiration_agents():
    file_name = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )
    assert os.path.exists(file_name)
    mechanism, agents, issues = load_genius_domain_from_folder(
        file_name,
        n_steps=100,
        time_limit=30,
        force_single_issue=True,
        keep_issue_names=False,
        keep_value_names=False,
        agent_factories=AspirationNegotiator,
    )
    assert mechanism is not None
    state = mechanism.run()


def test_encoding_decoding_all(capsys, scenarios_folder):
    from negmas.genius import AgentX, Atlas3

    # from random import sample

    with capsys.disabled():
        base = scenarios_folder
        for root, dirs, files in walk(base):
            if len(files) == 0 or len(dirs) != 0:
                continue
            # print(f'{nxt:05}: Importing {root}', flush=True)
            m, ufun_info, _ = load_genius_domain_from_folder(root)
            assert m is not None
            if genius_bridge_is_running():
                for info in ufun_info:
                    n1 = Atlas3(
                        domain_file_name=f"{root}/{m.name}.xml", ufun=info["ufun"],
                    )
                    n2 = AgentX(
                        domain_file_name=f"{root}/{m.name}.xml",
                        utility_file_name=info["ufun_name"],
                    )
                    m.add(n1)
                    m.add(n2)
                    u1, u2 = n1.ufun, n2.ufun
                    outcomes = m.discrete_outcomes(n_max=50)
                    for outcome in outcomes:
                        assert abs(u1(outcome) - u2(outcome)) < 1e-3
                    n1.destroy_java_counterpart()
                    n2.destroy_java_counterpart()


def test_importing_all_single_issue_without_exceptions(capsys, scenarios_folder):
    with capsys.disabled():
        base = scenarios_folder
        nxt, success = 0, 0
        for root, dirs, files in walk(base):
            if len(files) == 0 or len(dirs) != 0:
                continue
            try:
                domain, _, _ = load_genius_domain_from_folder(
                    root, force_single_issue=True, max_n_outcomes=10000
                )
            except Exception as x:
                print(f"Failed on {root}")
                raise x
            nxt += 1
            success += domain is not None
            # print(f'{success:05}/{nxt:05}: {"Single " if domain is not None else "Multi--"}outcome: {root}', flush=True)


def test_convert_dir_keep_names(tmpdir):
    from negmas import convert_genius_domain_from_folder

    dst = tmpdir.mkdir("sub")
    src = pkg_resources.resource_filename("negmas", resource_name="tests/data/Laptop")
    dst = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/LaptopConv"
    )
    assert convert_genius_domain_from_folder(
        src_folder_name=src,
        dst_folder_name=dst,
        force_single_issue=True,
        cache_and_discretize_outcomes=True,
        n_discretization=10,
        keep_issue_names=True,
        keep_value_names=True,
        normalize_utilities=True,
    )
    mechanism, agent_info, issues = load_genius_domain_from_folder(dst)
    assert len(issues) == 1
    for k, v in enumerate(issues):
        assert (
            f"{k}:{v}"
            == """0:Laptop-Harddisk-External Monitor: ["Dell+60 Gb+19'' LCD", "Dell+60 Gb+20'' LCD", "Dell+60 Gb+23'' LCD", "Dell+80 Gb+19'' LCD", "Dell+80 Gb+20'' LCD", "Dell+80 Gb+23'' LCD", "Dell+120 Gb+19'' LCD", "Dell+120 Gb+20'' LCD", "Dell+120 Gb+23'' LCD", "Macintosh+60 Gb+19'' LCD", "Macintosh+60 Gb+20'' LCD", "Macintosh+60 Gb+23'' LCD", "Macintosh+80 Gb+19'' LCD", "Macintosh+80 Gb+20'' LCD", "Macintosh+80 Gb+23'' LCD", "Macintosh+120 Gb+19'' LCD", "Macintosh+120 Gb+20'' LCD", "Macintosh+120 Gb+23'' LCD", "HP+60 Gb+19'' LCD", "HP+60 Gb+20'' LCD", "HP+60 Gb+23'' LCD", "HP+80 Gb+19'' LCD", "HP+80 Gb+20'' LCD", "HP+80 Gb+23'' LCD", "HP+120 Gb+19'' LCD", "HP+120 Gb+20'' LCD", "HP+120 Gb+23'' LCD"]"""
        )
