import hypothesis.strategies as st
import pkg_resources
import pytest
from hypothesis import given, settings
from py4j.protocol import Py4JNetworkError

from negmas import (
    GeniusNegotiator,
    genius_bridge_is_running,
    init_genius_bridge,
    load_genius_domain,
    load_genius_domain_from_folder,
)


# def test_init_genius_bridge():
#     if not genius_bridge_is_running():
#         init_genius_bridge()
#     assert genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
@settings(max_examples=10)
@given(
    agent_name1=st.sampled_from(GeniusNegotiator.robust_negotiators()),
    agent_name2=st.sampled_from(GeniusNegotiator.robust_negotiators()),
    single_issue=st.booleans(),
    keep_issue_names=st.booleans(),
    keep_value_names=st.booleans(),
)
def test_genius_agents_run_using_hypothesis(
    agent_name1, agent_name2, single_issue, keep_issue_names, keep_value_names,
):
    from negmas import convert_genius_domain_from_folder

    utils = (1, 2)
    src = pkg_resources.resource_filename("negmas", resource_name="tests/data/Laptop")
    dst = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/LaptopConv1D"
    )
    if single_issue:
        assert convert_genius_domain_from_folder(
            src_folder_name=src,
            dst_folder_name=dst,
            force_single_issue=True,
            cache_and_discretize_outcomes=True,
            n_discretization=10,
        )
        base_folder = dst
    else:
        base_folder = src
    neg, agent_info, issues = load_genius_domain_from_folder(
        base_folder,
        keep_issue_names=keep_issue_names,
        keep_value_names=keep_value_names,
        time_limit=5,
    )
    if neg is None:
        raise ValueError(f"Failed to lead domain from {base_folder}")
    a1 = GeniusNegotiator(
        java_class_name=agent_name1,
        domain_file_name=base_folder + "/Laptop-C-domain.xml",
        utility_file_name=base_folder + f"/Laptop-C-prof{utils[0]}.xml",
        keep_issue_names=keep_issue_names,
        keep_value_names=keep_value_names,
    )
    a2 = GeniusNegotiator(
        java_class_name=agent_name2,
        domain_file_name=base_folder + "/Laptop-C-domain.xml",
        utility_file_name=base_folder + f"/Laptop-C-prof{utils[1]}.xml",
        keep_issue_names=keep_issue_names,
        keep_value_names=keep_value_names,
    )
    neg._enable_callbacks = True
    neg.add(a1)
    neg.add(a2)
    neg.run()
    # print(f'{agent_name1} <-> {agent_name2}', end = '')
    # print(f': {neg.run(timeout=1)}')


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agents_run_example():
    from random import randint

    agents = ["agents.anac.y2015.Atlas3.Atlas3", "agents.anac.y2015.AgentX.AgentX"]
    for _ in range(5):
        agent_name1 = agents[randint(0, 1)]
        agent_name2 = agents[randint(0, 1)]
        print("{agent_name1} - {agent_name2}")
        utils = (1, 2)

        base_folder = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/Laptop"
        )
        neg, agent_info, issues = load_genius_domain_from_folder(
            base_folder, keep_issue_names=True, keep_value_names=True,
        )
        if neg is None:
            raise ValueError(f"Failed to lead domain from {base_folder}")
        atlas = GeniusNegotiator(
            java_class_name=agent_name1,
            domain_file_name=base_folder + "/Laptop-C-domain.xml",
            utility_file_name=base_folder + f"/Laptop-C-prof{utils[0]}.xml",
            keep_issue_names=True,
            keep_value_names=True,
        )
        agentx = GeniusNegotiator(
            java_class_name=agent_name2,
            domain_file_name=base_folder + "/Laptop-C-domain.xml",
            utility_file_name=base_folder + f"/Laptop-C-prof{utils[1]}.xml",
            keep_issue_names=True,
            keep_value_names=True,
        )
        neg._enable_callbacks = True
        neg.add(atlas)
        neg.add(agentx)
        neg.run()


if __name__ == "__main__":
    pytest.main(args=[__file__])
