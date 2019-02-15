import os
import numpy as np
import pkg_resources
import pytest
import pathlib
from hypothesis import given, settings
import hypothesis.strategies as st

from negmas import GeniusNegotiator, load_genius_domain, load_genius_domain_from_folder, init_genius_connection

dom_folder = pathlib.Path(pkg_resources.resource_filename('negmas'
                                                          , resource_name='tests/data/scenarios/anac/y2010/Travel'))
dom = dom_folder / 'travel_domain.xml'
util1 = dom_folder / 'travel_chox.xml'
util2 = dom_folder / 'travel_fanny.xml'


@pytest.fixture(scope='module')
def init_genius():
    try:
        # init_genius_connection(pkg_resources.resource_filename('negmas', resource_name='external/genius-8.0.4.jar'))
        pass
    except:
        pass


def test_init_genius(init_genius):
    pass


@settings(max_examples=100)
@given(agent_name1=st.sampled_from(GeniusNegotiator.negotiators()),
       agent_name2=st.sampled_from(GeniusNegotiator.negotiators()),
       utils=st.tuples(st.integers(1, 2), st.integers(1, 2)),
       single_issue=st.booleans(),
       keep_issue_names=st.booleans(),
       keep_value_names=st.booleans())
def test_genius_agents_run_using_hypothesis(init_genius, agent_name1, agent_name2, utils, single_issue
                                            , keep_issue_names, keep_value_names):
    from negmas import convert_genius_domain_from_folder
    src = pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
    dst = pkg_resources.resource_filename('negmas', resource_name='tests/data/LaptopConv1D')
    if single_issue:
        assert convert_genius_domain_from_folder(src_folder_name=src
                                                 , dst_folder_name=dst
                                                 , force_single_issue=True
                                                 , cache_and_discretize_outcomes=True
                                                 , n_discretization=10
                                                 )
        base_folder = dst
    else:
        base_folder = src
    neg, agent_info, issues = load_genius_domain_from_folder(base_folder
                                                             , keep_issue_names=keep_issue_names
                                                             , keep_value_names=keep_value_names)
    if neg is None:
        raise ValueError(f'Failed to lead domain from {base_folder}')
    atlas = GeniusNegotiator(java_class_name=agent_name1,
                             domain_file_name=base_folder + '/Laptop-C-domain.xml'
                             , utility_file_name=base_folder + f'/Laptop-C-prof{utils[0]}.xml'
                             , keep_issue_names=keep_issue_names
                             , keep_value_names=keep_value_names)
    agentx = GeniusNegotiator(java_class_name=agent_name2,
                              domain_file_name=base_folder + '/Laptop-C-domain.xml'
                              , utility_file_name=base_folder + f'/Laptop-C-prof{utils[1]}.xml'
                              , keep_issue_names=keep_issue_names
                              , keep_value_names=keep_value_names
                              )
    neg.add(atlas)
    neg.add(agentx)
    # print(f'{agent_name1} <-> {agent_name2}', end = '')
    # print(f': {neg.run(timeout=1)}')


if __name__ == '__main__':
    pytest.main(args=[__file__])
