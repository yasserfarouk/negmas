from __future__ import annotations

import pkg_resources
import pytest

from negmas.genius.ginfo import ALL_PASSING_NEGOTIATORS as ALL_NEGOTIATORS
from negmas.genius.gnegotiators import (
    ABMPAgent2,
    AgentBuyong,
    AgentHP2,
    AgentK,
    AgentK2,
    AgentLG,
    AgentM,
    AgentX,
    AgentYK,
    AgreeableAgent2018,
    Atlas3,
    Atlas32016,
    BayesianAgent,
    BetaOne,
    BoulwareNegotiationParty,
    BRAMAgent,
    Caduceus,
    CaduceusDC16,
    ConcederNegotiationParty,
    CUHKAgent,
    DoNA,
    E2Agent,
    Farma,
    FuzzyAgent,
    Gahboninho,
    Gangster,
    GeniusNegotiator,
    GrandmaAgent,
    Group2,
    HardHeaded,
    IAMhaggler,
    IAMhaggler2011,
    ImmediateAcceptor,
    Kawaii,
    KGAgent,
    MengWan,
    MetaAgent,
    MyAgent,
    Ngent,
    NiceTitForTat,
    Nozomi,
    OMACagent,
    OptimalBidderSimple,
    ParsAgent,
    ParsCat,
    PhoenixParty,
    PokerFace,
    PonPokoAgent,
    RandomCounterOfferNegotiationParty,
    RandomDance,
    RandomParty,
    RandomParty2,
    Rubick,
    SimilarityAgent,
    Simpatico,
    SimpleAgent,
    Terra,
    TheFawkes,
    TheNegotiator,
    TheNegotiatorReloaded,
    TimeDependentAgentBoulware,
    TimeDependentAgentConceder,
    TimeDependentAgentHardliner,
    TimeDependentAgentLinear,
    TMFAgent,
    ValueModelAgent,
    WhaleAgent,
    XianFaAgent,
    Yushu,
    YXAgent,
)
from negmas.inout import Scenario
from negmas.sao.negotiators import ToughNegotiator

from .switches import NEGMAS_FASTRUN, NEGMAS_RUN_GENIUS

TIMELIMIT = 30
STEPLIMIT = 50

AGENTS_WITH_NO_AGREEMENT_ON_SAME_preferences = (
    "agents.anac.y2015.Mercury.Mercury",
    "parties.in4010.q12015.group19.Group19",
    "parties.in4010.q12015.group6.Group6",
    "agents.anac.y2015.xianfa.XianFaAgent",
)

DOMAINS = [
    "tests/data/Car-A-domain",
    # "tests/data/Laptop",
]

SKIP_CONDITION = NEGMAS_FASTRUN or not NEGMAS_RUN_GENIUS


def do_run(
    agent_factory,
    base_folder,
    opponent_preferences,
    agent_preferences,
    agent_starts,
    opponent_factory,
    n_steps,
    time_limit,
):
    domain = Scenario.from_genius_folder(base_folder)
    if not domain:
        raise ValueError(f"Cannot open domain {base_folder}")
    neg = domain.make_session(
        n_steps=n_steps, time_limit=time_limit, avoid_ultimatum=False
    )
    if neg is None:
        raise ValueError(f"Failed to load domain from {base_folder}")
    opponent = opponent_factory(preferences=domain.ufuns[opponent_preferences])
    theagent = agent_factory(preferences=domain.ufuns[agent_preferences])
    if agent_starts:
        neg.add(theagent)
        neg.add(opponent)
    else:
        neg.add(opponent)
        neg.add(theagent)
    neg.run()
    return neg


def do_test_same_ufun(agent_factory, base_folder, n_steps, time_limit, n_trials=3):
    # check that it will get to an agreement sometimes if the same ufun
    # is used for both agents
    from random import randint

    for _ in range(n_trials):
        indx = randint(0, 1)
        neg = do_run(
            agent_factory,
            base_folder,
            indx,
            indx,
            False,
            agent_factory,
            n_steps,
            time_limit,
        )
        if neg.agreement is not None:
            break
    else:
        assert (
            False
        ), f"failed to get an agreement in {n_trials} trials even using the same ufun\n{neg.trace}"  # type: ignore It makes not sense to have n_trials == 0 so we are safe


def do_test_genius_agent(
    AgentFactory, must_agree_if_same_preferences=True, java_class_name=None
):
    if java_class_name is not None:
        AgentFactory = lambda *args, **kwargs: GeniusNegotiator(
            *args, java_class_name=java_class_name, **kwargs
        )
        agent_class_name = java_class_name
    else:
        agent_class_name = AgentFactory.__name__
    # print(f"Running {AgentClass.__name__}")
    for domain in DOMAINS:
        base_folder = pkg_resources.resource_filename("negmas", resource_name=domain)

        # check that it can run without errors with two different ufuns
        for opponent_type in (ToughNegotiator, Atlas3):
            for starts in (False, True):
                for n_steps, time_limit in ((STEPLIMIT, None), (None, TIMELIMIT)):
                    for ufuns in ((1, 0), (0, 1)):
                        try:
                            do_run(
                                AgentFactory,
                                base_folder,
                                ufuns[0],
                                ufuns[1],
                                starts,
                                opponent_type,
                                n_steps=n_steps,
                                time_limit=time_limit,
                            )
                        except Exception as e:
                            print(
                                f"{agent_class_name} FAILED against {opponent_type.__name__}"
                                f" going {'first' if starts else 'last'} ({n_steps} steps with "
                                f"{time_limit} limit taking ufun {ufuns[1]}."
                            )
                            raise e

        if not must_agree_if_same_preferences or (
            java_class_name is None
            and AgentFactory in AGENTS_WITH_NO_AGREEMENT_ON_SAME_preferences
        ):
            continue
        do_test_same_ufun(AgentFactory, base_folder, STEPLIMIT, None, 3)

    # GeniusBridge.clean()


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
@pytest.mark.parametrize("negotiator", ALL_NEGOTIATORS)
def test_all_negotiators(negotiator):
    do_test_genius_agent(None, java_class_name=negotiator)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_boulware_party():
    do_test_genius_agent(
        None, java_class_name="negotiator.parties.BoulwareNegotiationParty"
    )


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_boulware_agent():
    do_test_genius_agent(None, java_class_name="agents.TimeDependentAgentBoulware")


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_AgentX():
    do_test_genius_agent(AgentX)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_YXAgent():
    do_test_genius_agent(YXAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Caduceus():
    do_test_genius_agent(Caduceus)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_ParsCat():
    do_test_genius_agent(ParsCat)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_ParsAgent():
    do_test_genius_agent(ParsAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_PonPokoAgent():
    do_test_genius_agent(PonPokoAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_RandomDance():
    do_test_genius_agent(RandomDance)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Atlas32016():
    do_test_genius_agent(Atlas32016)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_MyAgent():
    do_test_genius_agent(MyAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Farma():
    do_test_genius_agent(Farma)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_PokerFace():
    do_test_genius_agent(PokerFace)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_AgentBuyong():
    do_test_genius_agent(AgentBuyong)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Kawaii():
    do_test_genius_agent(Kawaii)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Atlas3():
    do_test_genius_agent(Atlas3)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_AgentYK():
    do_test_genius_agent(AgentYK)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Group2():
    do_test_genius_agent(Group2)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_WhaleAgent():
    do_test_genius_agent(WhaleAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_DoNA():
    do_test_genius_agent(DoNA)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_AgentM():
    do_test_genius_agent(AgentM)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_TMFAgent():
    do_test_genius_agent(TMFAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_OMACagent():
    do_test_genius_agent(OMACagent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_AgentLG():
    do_test_genius_agent(AgentLG)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_CUHKAgent():
    do_test_genius_agent(CUHKAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_ValueModelAgent():
    do_test_genius_agent(ValueModelAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_NiceTitForTat():
    do_test_genius_agent(NiceTitForTat)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_TheNegotiator():
    do_test_genius_agent(TheNegotiator)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_AgentK2():
    do_test_genius_agent(AgentK2)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_BRAMAgent():
    do_test_genius_agent(BRAMAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_IAMhaggler2011():
    do_test_genius_agent(IAMhaggler2011)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Gahboninho():
    do_test_genius_agent(Gahboninho)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_HardHeaded():
    do_test_genius_agent(HardHeaded)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_AgentK():
    do_test_genius_agent(AgentK)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Yushu():
    do_test_genius_agent(Yushu)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Nozomi():
    do_test_genius_agent(Nozomi)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_IAMhaggler():
    do_test_genius_agent(IAMhaggler)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Terra():
    do_test_genius_agent(Terra)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Gangster():
    do_test_genius_agent(Gangster)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_TheFawkes():
    do_test_genius_agent(TheFawkes)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_MetaAgent():
    do_test_genius_agent(MetaAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_TheNegotiatorReloaded():
    do_test_genius_agent(TheNegotiatorReloaded)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="Either no-genius or fast-run",
)
def test_Simpatico():
    do_test_genius_agent(Simpatico)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_ConcederNegotiationParty():
    do_test_genius_agent(ConcederNegotiationParty)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_BayesianAgent():
    do_test_genius_agent(BayesianAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_FuzzyAgent():
    do_test_genius_agent(FuzzyAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_ABMPAgent2():
    do_test_genius_agent(ABMPAgent2)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_OptimalBidderSimple():
    do_test_genius_agent(OptimalBidderSimple)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_SimilarityAgent():
    do_test_genius_agent(SimilarityAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_BoulwareNegotiationParty():
    do_test_genius_agent(BoulwareNegotiationParty)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_TimeDependentAgentLinear():
    do_test_genius_agent(TimeDependentAgentLinear)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_TimeDependentAgentHardliner():
    do_test_genius_agent(TimeDependentAgentHardliner)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_TimeDependentAgentConceder():
    do_test_genius_agent(TimeDependentAgentConceder)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_TimeDependentAgentBoulware():
    do_test_genius_agent(TimeDependentAgentBoulware)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_RandomParty2():
    do_test_genius_agent(RandomParty2)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_RandomParty():
    do_test_genius_agent(RandomParty)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_RandomCounterOfferNegotiationParty():
    do_test_genius_agent(RandomCounterOfferNegotiationParty)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_SimpleAgent():
    do_test_genius_agent(SimpleAgent)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="No Genius Gridge, skipping genius-agent tests",
)
def test_ImmediateAcceptor():
    do_test_genius_agent(ImmediateAcceptor)


#### agents after this line are not very robust


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_Rubick():
    do_test_genius_agent(Rubick)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_CaduceusDC16():
    do_test_genius_agent(CaduceusDC16)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_BetaOne():
    do_test_genius_agent(BetaOne)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_AgreeableAgent2018():
    do_test_genius_agent(AgreeableAgent2018)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_MengWan():
    do_test_genius_agent(MengWan)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_GrandmaAgent():
    do_test_genius_agent(GrandmaAgent)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_XianFaAgent():
    do_test_genius_agent(XianFaAgent)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_PhoenixParty():
    do_test_genius_agent(PhoenixParty)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_AgentHP2():
    do_test_genius_agent(AgentHP2)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_KGAgent():
    do_test_genius_agent(KGAgent)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_E2Agent():
    do_test_genius_agent(E2Agent)


@pytest.mark.xfail(
    run=False,
    reason="Genius agent known not to work",
)
def test_Ngent():
    do_test_genius_agent(Ngent)


if __name__ == "__main__":
    pytest.main(args=[__file__])
