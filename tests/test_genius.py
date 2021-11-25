from pathlib import Path

import hypothesis.strategies as st
import pkg_resources
import pytest
from hypothesis import given, settings
from py4j.protocol import Py4JNetworkError

from negmas import (
    AgentBuyong,
    AgentHP2,
    AgentK,
    AgentK2,
    AgentLG,
    AgentM,
    AgentX,
    AgentYK,
    AgreeableAgent2018,
    AspirationNegotiator,
    Atlas3,
    Atlas32016,
    BetaOne,
    BRAMAgent,
    Caduceus,
    CaduceusDC16,
    CUHKAgent,
    DoNA,
    E2Agent,
    Farma,
    Gahboninho,
    Gangster,
    GeniusNegotiator,
    GrandmaAgent,
    Group2,
    HardHeaded,
    IAMhaggler,
    IAMhaggler2011,
    Kawaii,
    KGAgent,
    MengWan,
    MetaAgent,
    MyAgent,
    Ngent,
    NiceTitForTat,
    Nozomi,
    OMACagent,
    ParsAgent,
    ParsCat,
    PhoenixParty,
    PokerFace,
    PonPokoAgent,
    RandomDance,
    Rubick,
    Simpatico,
    Terra,
    TheFawkes,
    TheNegotiator,
    TheNegotiatorReloaded,
    TMFAgent,
    ValueModelAgent,
    WhaleAgent,
    XianFaAgent,
    Yushu,
    YXAgent,
    genius_bridge_is_running,
    load_genius_domain_from_folder,
)
from negmas.genius import GeniusBridge, get_anac_agents
from negmas.genius.ginfo import ALL_PASSING_NEGOTIATORS as ALL_NEGOTIATORS
from negmas.inout import Domain

TIMELIMIT = 120
STEPLIMIT = 1000

AGENTS_WITH_NO_AGREEMENT_ON_SAME_preferences = tuple()

SKIP_IF_NO_BRIDGE = True


@given(
    linear=st.booleans(),
    learning=st.booleans(),
    multilateral=st.booleans(),
    bilateral=st.booleans(),
    reservation=st.booleans(),
    discounting=st.booleans(),
    uncertainty=st.booleans(),
    elicitation=st.booleans(),
)
def test_get_genius_agents_example(
    linear,
    learning,
    multilateral,
    bilateral,
    reservation,
    discounting,
    uncertainty,
    elicitation,
):
    winners = get_anac_agents(bilateral=True, winners_only=True)
    everyone = get_anac_agents(bilateral=True)
    finalists = get_anac_agents(bilateral=True, finalists_only=True)
    # assert len(winners) > 0
    # assert len(everyone) > 0
    # assert len(finalists) > 0
    for x in (winners, finalists, everyone):
        assert all(
            list(
                isinstance(_, tuple)
                and len(_) == 2
                and isinstance(_[0], str)
                and isinstance(_[1], str)
                and len(_[1]) >= len(_[0])
                for _ in x
            )
        ), x


def test_inclusion_of_sets_in_get_agents():
    from negmas.genius.ginfo import GENIUS_INFO

    for year in GENIUS_INFO.keys():
        winners = get_anac_agents(year=year, winners_only=True)
        finalists = get_anac_agents(year=year, finalists_only=True)
        everyone = get_anac_agents(year=year)
        assert not finalists or all([_ in finalists for _ in winners]), set(
            winners
        ).difference(set(finalists))
        # assert not everyone or all([_ in everyone for _ in winners]),set(winners).difference(set(everyone))
        # assert not everyone or all([_ in everyone for _ in finalists]), set(finalists).difference(set(everyone))


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_does_not_freeze():
    from pathlib import Path

    from negmas.genius import GeniusNegotiator

    folder_name = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/cameradomain"
    )
    domain = Domain.from_genius_folder(folder_name)
    a1 = GeniusNegotiator(
        java_class_name="agents.anac.y2017.ponpokoagent.PonPokoAgent",
        domain_file_name=domain.agenda.name,
        utility_file_name=domain.ufuns[0].name,
    )

    a2 = GeniusNegotiator(
        java_class_name="agents.anac.y2016.yxagent.YXAgent",
        domain_file_name=domain.agenda.name,
        utility_file_name=domain.ufuns[1].name,
    )

    mechanism = domain.make_session(n_steps=None, time_limit=TIMELIMIT)
    mechanism.add(a1)
    mechanism.add(a2)
    mechanism.run()
    # print(a1.ufun.__call__(mechanism.agreement), a2.ufun.__call__(mechanism.agreement))
    GeniusBridge.clean()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_old_agent():
    from pathlib import Path

    from negmas.genius import GeniusNegotiator
    from negmas.inout import load_genius_domain_from_folder

    folder_name = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/cameradomain"
    )
    domain = Domain.from_genius_folder(folder_name)
    a1 = GeniusNegotiator(
        java_class_name="agents.anac.y2012.AgentLG.AgentLG",
        domain_file_name=domain.agenda.name,
        utility_file_name=domain.ufuns[0].name,
    )

    a2 = GeniusNegotiator(
        java_class_name="agents.anac.y2016.yxagent.YXAgent",
        domain_file_name=domain.agenda.name,
        utility_file_name=domain.ufuns[1].name,
    )

    mechanism = domain.make_session([a1, a2], n_steps=None, time_limit=TIMELIMIT)
    mechanism.run()
    # print(a1.ufun.__call__(mechanism.agreement), a2.ufun.__call__(mechanism.agreement))
    GeniusBridge.clean()


# def test_init_genius_bridge():
#     if not genius_bridge_is_running():
#         init_genius_bridge()
#     assert genius_bridge_is_running()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
@settings(max_examples=20, deadline=500000)
@given(
    agent_name1=st.sampled_from(GeniusNegotiator.robust_negotiators()),
    agent_name2=st.sampled_from(GeniusNegotiator.robust_negotiators()),
    single_issue=st.booleans(),
)
def test_genius_agents_run_using_hypothesis(
    agent_name1,
    agent_name2,
    single_issue,
):
    from negmas import convert_genius_domain_from_folder

    # TODO remove this limitation.
    src = pkg_resources.resource_filename("negmas", resource_name="tests/data/Laptop")
    dst = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/LaptopConv1D"
    )
    base_folder = dst if single_issue else src
    domain = Domain.from_genius_folder(base_folder)
    a1 = GeniusNegotiator(
        java_class_name=agent_name1,
        ufun=domain.ufuns[0],
    )
    a2 = GeniusNegotiator(
        java_class_name=agent_name2,
        ufun=domain.ufuns[1],
    )
    neg = domain.make_session([a1, a2], n_steps=STEPLIMIT, time_limit=None)
    if neg is None:
        raise ValueError(f"Failed to lead domain from {base_folder}")
    neg._enable_callbacks = True
    neg.run()
    GeniusBridge.clean()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent_gets_preferences():
    agents = ["agents.anac.y2015.Atlas3.Atlas3", "agents.anac.y2015.AgentX.AgentX"]
    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )
    domain = Domain.from_genius_folder(base_folder)
    a1 = GeniusNegotiator(
        java_class_name="agents.anac.y2015.Atlas3.Atlas3",
        domain_file_name=base_folder + "/Laptop-C-domain.xml",
        utility_file_name=base_folder + f"/Laptop-C-prof1.xml",
    )
    assert a1.ufun is not None
    assert not a1._temp_preferences_file
    assert not a1._temp_domain_file
    a2 = GeniusNegotiator(
        java_class_name="agents.anac.y2015.Atlas3.Atlas3",
        domain_file_name=base_folder + "/Laptop-C-domain.xml",
        ufun=domain.ufuns[0],
    )
    neg = domain.make_session(n_steps=None, time_limit=TIMELIMIT)
    neg.add(a1)
    neg.add(a2)
    assert a2.ufun is not None
    assert a2._temp_preferences_file
    assert not a2._temp_domain_file
    neg.run()
    GeniusBridge.clean()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agents_run_example():
    from random import randint

    agents = ["agents.anac.y2015.Atlas3.Atlas3", "agents.anac.y2015.AgentX.AgentX"]
    for _ in range(5):
        agent_name1 = agents[randint(0, 1)]
        agent_name2 = agents[randint(0, 1)]
        # print(f"{agent_name1} - {agent_name2}")
        utils = (1, 2)

        base_folder = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/Laptop"
        )
        domain = Domain.from_genius_folder(base_folder)
        atlas = GeniusNegotiator(
            java_class_name=agent_name1,
            domain_file_name=base_folder + "/Laptop-C-domain.xml",
            utility_file_name=base_folder + f"/Laptop-C-prof{utils[0]}.xml",
        )
        agentx = GeniusNegotiator(
            java_class_name=agent_name2,
            domain_file_name=base_folder + "/Laptop-C-domain.xml",
            utility_file_name=base_folder + f"/Laptop-C-prof{utils[1]}.xml",
        )
        neg = domain.make_session(n_steps=None, time_limit=TIMELIMIT)
        if neg is None:
            raise ValueError(f"Failed to lead domain from {base_folder}")
        neg.add(atlas)
        neg.add(agentx)
        neg.run()

    GeniusBridge.clean()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_agentk_perceives_time():
    n_steps = 80
    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )

    domain = Domain.from_genius_folder(base_folder)
    gagent = AgentK()
    neg = domain.make_session(
        [gagent, AspirationNegotiator()],
        n_steps=n_steps,
        time_limit=float("inf"),
        avoid_ultimatum=True,
    )
    if neg is None:
        raise ValueError(f"Failed to load domain from {base_folder}")
    current_time = 0
    for _ in range(n_steps):
        assert (
            gagent.relative_time >= current_time
        ), f"Failed to get time before step {_}"
        neg.step()
        if neg.ami.state.ended:
            break
        if _ == n_steps - 1:
            assert gagent.relative_time is None, f"Got a time after the last step"
        else:
            assert (
                gagent.relative_time > current_time
            ), f"Failed to get time after step {_}"
        if neg.ami.state.ended:
            break
        current_time = gagent.relative_time

    assert gagent.relative_time >= current_time


@pytest.mark.skipif(
    condition=True or not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_running_genius_mechanism_in_genius(tmp_path):

    base_folder = Path(
        pkg_resources.resource_filename("negmas", resource_name="tests/data/Laptop")
    )
    profiles = [
        "file://" + str(base_folder / "Laptop-C-prof1.xml"),
        "file://" + str(base_folder / "Laptop-C-prof2.xml"),
    ]
    agents = ["agents.anac.y2010.AgentK.Agent_K", "agents.anac.y2010.AgentK.Agent_K"]
    output_file = str(tmp_path)

    gateway = GeniusBridge.gateway()
    gateway.entry_point.run_negotiation(
        "genius.core.protocol.StackedAlternatingOffersProtocol",
        "file://" + str(base_folder / "Laptop-C-domain.xml"),
        ";".join(profiles),
        ";".join(agents),
        output_file,
    )


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
@pytest.mark.parametrize(
    ["a1", "a2", "n_steps", "time_limit"],
    [
        (
            "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
            "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
            100,
            None,
        ),
        (
            "agents.anac.y2011.HardHeaded.KLH",
            "agents.anac.y2012.CUHKAgent.CUHKAgent",
            None,
            20,
        ),
        (
            "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
            "agents.anac.y2010.Nozomi.Nozomi",
            None,
            30,
        ),
        (
            "agents.TimeDependentAgentBoulware",
            "negotiator.parties.BoulwareNegotiationParty",
            100,
            None,
        ),
    ],
)
def test_2genius_together(a1, a2, n_steps, time_limit):
    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Car-A-domain"
    )

    domain = Domain.from_genius_folder(base_folder)
    neg = domain.make_session(
        n_steps=n_steps, time_limit=time_limit, avoid_ultimatum=True
    )
    if neg is None:
        raise ValueError(f"Failed to load domain from {base_folder}")
    neg.add(
        GeniusNegotiator(java_class_name=a1, strict=True, ufun=domain.ufuns[0]),
        strict=True,
    )
    neg.add(
        GeniusNegotiator(java_class_name=a2, strict=True, ufun=domain.ufuns[1]),
        strict=True,
    )
    neg.run()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_caudacius_caudacius():
    n_steps = 100
    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Car-A-domain"
    )

    domain = Domain.from_genius_folder(base_folder)
    neg = domain.make_session(
        n_steps=n_steps, time_limit=float("inf"), avoid_ultimatum=True
    )
    if neg is None:
        raise ValueError(f"Failed to load domain from {base_folder}")
    neg.add(Caduceus(ufun=domain.ufuns[0]), strict=True)
    neg.add(Caduceus(ufun=domain.ufuns[1]), strict=True)
    for _ in range(n_steps):
        neg.step()
        if neg.state.agreement is not None:
            break
        new_offers = [_[1] for _ in neg.state.new_offers]
        assert all(
            _ is not None for _ in new_offers
        ), f"failed at {neg.current_step}: {new_offers}"

    assert not all(
        [len(set(neg.negotiator_offers(_))) == 1 for _ in neg.negotiator_ids]
    ), f"None of the agents conceeded: {neg.trace}"


if __name__ == "__main__":
    pytest.main(args=[__file__])
