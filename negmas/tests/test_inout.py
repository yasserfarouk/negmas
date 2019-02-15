from os import walk
import os

import pkg_resources
import pytest

from negmas import load_genius_domain_from_folder, AspirationNegotiator


@pytest.fixture
def scenarios_folder():
    return pkg_resources.resource_filename('negmas', resource_name='tests/data/scenarios')


def test_importing_file_without_exceptions(scenarios_folder):
    folder_name = scenarios_folder + '/other/S-1NIKFRT-1'
    domain = load_genius_domain_from_folder(folder_name
                                            , n_discretization=10
                                            )
    # print(domain)


def test_convert_dir_keep_names(tmpdir):
    from negmas import convert_genius_domain_from_folder
    dst = tmpdir.mkdir("sub")
    src = pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
    dst = pkg_resources.resource_filename('negmas', resource_name='tests/data/LaptopConv')
    assert convert_genius_domain_from_folder(src_folder_name=src
                                      , dst_folder_name=dst
                                      , force_single_issue=True
                                      , cache_and_discretize_outcomes=True
                                      , n_discretization=10
                                      , keep_issue_names=True
                                      , keep_value_names=True
                                      , normalize_utilities=True)
    mechanism, agent_info, issues = load_genius_domain_from_folder(dst)
    assert len(issues) == 1
    for k, v in enumerate(issues):
        assert f'{k}:{v}' == '''0:Laptop-Harddisk-External Monitor: ["Dell+60 Gb+19'' LCD", "Dell+60 Gb+20'' LCD", "Dell+60 Gb+23'' LCD", "Dell+80 Gb+19'' LCD", "Dell+80 Gb+20'' LCD", "Dell+80 Gb+23'' LCD", "Dell+120 Gb+19'' LCD", "Dell+120 Gb+20'' LCD", "Dell+120 Gb+23'' LCD", "Macintosh+60 Gb+19'' LCD", "Macintosh+60 Gb+20'' LCD", "Macintosh+60 Gb+23'' LCD", "Macintosh+80 Gb+19'' LCD", "Macintosh+80 Gb+20'' LCD", "Macintosh+80 Gb+23'' LCD", "Macintosh+120 Gb+19'' LCD", "Macintosh+120 Gb+20'' LCD", "Macintosh+120 Gb+23'' LCD", "HP+60 Gb+19'' LCD", "HP+60 Gb+20'' LCD", "HP+60 Gb+23'' LCD", "HP+80 Gb+19'' LCD", "HP+80 Gb+20'' LCD", "HP+80 Gb+23'' LCD", "HP+120 Gb+19'' LCD", "HP+120 Gb+20'' LCD", "HP+120 Gb+23'' LCD"]'''


def test_convert_dir_no_names(tmpdir):
    from negmas import convert_genius_domain_from_folder
    dst = tmpdir.mkdir("sub")
    src = pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
    dst = pkg_resources.resource_filename('negmas', resource_name='tests/data/LaptopConv')

    assert convert_genius_domain_from_folder(src_folder_name=src
                                             , dst_folder_name=dst
                                             , force_single_issue=True
                                             , cache_and_discretize_outcomes=True
                                             , n_discretization=10
                                             , keep_issue_names=False
                                             , keep_value_names=False
                                             , normalize_utilities=True)
    mechanism, agent_info, issues = load_genius_domain_from_folder(dst)
    assert len(issues) == 1
    for k, v in enumerate(issues):
        assert f'{k}:{v}' == '''0:0: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']'''


def test_simple_run_with_aspiration_agents():
    file_name = pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
    assert os.path.exists(file_name)
    mechanism, agents, issues = load_genius_domain_from_folder(
        file_name, n_steps=100, time_limit=30
        , force_single_issue=True, keep_issue_names=False
        , keep_value_names=False, agent_factories=AspirationNegotiator)
    assert mechanism is not None
    state = mechanism.run()

