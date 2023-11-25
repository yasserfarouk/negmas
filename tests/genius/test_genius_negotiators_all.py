from __future__ import annotations

import pkg_resources
import pytest

from negmas.genius.ginfo import (
    ALL_PASSING_NEGOTIATORS_NO_UNCERTAINTY,
    TESTED_NEGOTIATORS,
)
from negmas.genius.gnegotiators import *
from negmas.genius.negotiator import GeniusNegotiator
from negmas.inout import Scenario
from negmas.sao.negotiators import ToughNegotiator
from tests.switches import NEGMAS_FASTRUN, NEGMAS_RUN_GENIUS

TIMELIMIT = 10
STEPLIMIT = 50

# ALL_NEGOTIATORS = ALL_PASSING_NEGOTIATORS_NO_UNCERTAINTY
ALL_NEGOTIATORS = TESTED_NEGOTIATORS

AGENTS_WITH_NO_AGREEMENT_ON_SAME_preferences = (
    # "agents.anac.y2015.Mercury.Mercury",
    # "parties.in4010.q12015.group19.Group19",
    # "parties.in4010.q12015.group6.Group6",
    # "agents.anac.y2015.xianfa.XianFaAgent",
    "agents.anac.y2016.agentsmith.AgentSmith2016",
    "agents.anac.y2014.AgentTD.AgentTD",
    "agents.anac.y2019.agentgg.AgentGG",
    "agents.anac.y2019.saga.SAGA",
    "agents.anac.y2019.sacra.SACRA",
    "agents.anac.y2016.agentlight.AgentLight",
    "agents.anac.y2017.agentkn.AgentKN",
    "agents.anac.y2011.AgentK2.Agent_K2",
)

DOMAINS = [
    "tests/data/ItexvsCypress",
    "tests/data/Laptop",
]

SKIP_CONDITION = NEGMAS_FASTRUN or not NEGMAS_RUN_GENIUS
# SKIP_CONDITION = False
STRICT_TEST = True


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
    neg = domain.make_session(n_steps=n_steps, time_limit=time_limit)
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
    AgentFactory,
    must_agree_if_same_preferences=True,
    java_class_name=None,
    strict_test=STRICT_TEST,
):
    if java_class_name is not None:
        AgentFactory = lambda *args, **kwargs: GeniusNegotiator(
            *args, java_class_name=java_class_name, strict=strict_test, **kwargs
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


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_FSEGA2019():
    do_test_genius_agent(FSEGA2019)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MINF():
    do_test_genius_agent(MINF)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SACRA():
    do_test_genius_agent(SACRA, must_agree_if_same_preferences=False)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SAGA():
    do_test_genius_agent(SAGA, must_agree_if_same_preferences=False)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SENGOKU():
    do_test_genius_agent(SENGOKU)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ABMPAgent2():
    do_test_genius_agent(ABMPAgent2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Agent33():
    do_test_genius_agent(Agent33)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Agent36():
    do_test_genius_agent(Agent36)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentBuyog():
    do_test_genius_agent(AgentBuyog)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentF():
    do_test_genius_agent(AgentF)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentFSEGA():
    do_test_genius_agent(AgentFSEGA)


# @pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
@pytest.mark.skip("Known failure")
def test_AgentGG():
    do_test_genius_agent(AgentGG)


# @pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
@pytest.mark.skip("Known failure")
def test_AgentGP():
    do_test_genius_agent(AgentGP)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentH():
    do_test_genius_agent(AgentH)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentHerb():
    do_test_genius_agent(AgentHerb)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentHP():
    do_test_genius_agent(AgentHP)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentHP2():
    do_test_genius_agent(AgentHP2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentI():
    do_test_genius_agent(AgentI)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentK():
    do_test_genius_agent(AgentK)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentK2():
    do_test_genius_agent(AgentK2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentKF():
    do_test_genius_agent(AgentKF)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentKN():
    do_test_genius_agent(AgentKN, must_agree_if_same_preferences=False)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentLarry():
    do_test_genius_agent(AgentLarry, must_agree_if_same_preferences=False)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentLG():
    do_test_genius_agent(AgentLG)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentLight():
    do_test_genius_agent(AgentLight, must_agree_if_same_preferences=False)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentM():
    do_test_genius_agent(AgentM)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentMR():
    do_test_genius_agent(AgentMR)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentNP1():
    do_test_genius_agent(AgentNP1)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentQuest():
    do_test_genius_agent(AgentQuest)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentSmith():
    do_test_genius_agent(AgentSmith)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentSmith2016():
    do_test_genius_agent(AgentSmith2016)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentTD():
    do_test_genius_agent(AgentTD)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentTRP():
    do_test_genius_agent(AgentTRP)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentW():
    do_test_genius_agent(AgentW)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentX():
    do_test_genius_agent(AgentX)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentYK():
    do_test_genius_agent(AgentYK)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgreeableAgent2018():
    do_test_genius_agent(AgreeableAgent2018)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AnacSampleAgent():
    do_test_genius_agent(AnacSampleAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AresParty():
    do_test_genius_agent(AresParty)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ArisawaYaki():
    do_test_genius_agent(ArisawaYaki)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Aster():
    do_test_genius_agent(Aster)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ATeamAgent():
    do_test_genius_agent(AteamAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Atlas():
    do_test_genius_agent(Atlas)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Atlas3():
    do_test_genius_agent(Atlas3)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Atlas32016():
    do_test_genius_agent(Atlas32016)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_BayesianAgent():
    do_test_genius_agent(BayesianAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_BetaOne():
    do_test_genius_agent(BetaOne)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Betaone():
    do_test_genius_agent(Betaone)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_BoulwareNegotiationParty():
    do_test_genius_agent(BoulwareNegotiationParty)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_BramAgent():
    do_test_genius_agent(BramAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_BramAgent2():
    do_test_genius_agent(BramAgent2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_BraveCat():
    do_test_genius_agent(BraveCat)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Caduceus():
    do_test_genius_agent(Caduceus)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_CaduceusDC16():
    do_test_genius_agent(CaduceusDC16)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ClockworkAgent():
    do_test_genius_agent(ClockworkAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ConcederNegotiationParty():
    do_test_genius_agent(ConcederNegotiationParty)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ConDAgent():
    do_test_genius_agent(ConDAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_CUHKAgent():
    do_test_genius_agent(CUHKAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_CUHKAgent2015():
    do_test_genius_agent(CUHKAgent2015)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_DandikAgent():
    do_test_genius_agent(DandikAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_DoNA():
    do_test_genius_agent(DoNA)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_DrageKnight():
    do_test_genius_agent(DrageKnight)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_E2Agent():
    do_test_genius_agent(E2Agent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_EAgent():
    do_test_genius_agent(EAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ExpRubick():
    do_test_genius_agent(ExpRubick)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Farma():
    do_test_genius_agent(Farma)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Farma17():
    do_test_genius_agent(Farma17)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Flinch():
    do_test_genius_agent(Flinch)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_FullAgent():
    do_test_genius_agent(FullAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_FunctionalAcceptor():
    do_test_genius_agent(FunctionalAcceptor)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_FuzzyAgent():
    do_test_genius_agent(FuzzyAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_GAgent():
    do_test_genius_agent(GAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Gahboninho():
    do_test_genius_agent(Gahboninho)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Gangester():
    do_test_genius_agent(Gangester)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Gangster():
    do_test_genius_agent(Gangster)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_GaravelAgent():
    do_test_genius_agent(GaravelAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Gin():
    do_test_genius_agent(Gin)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_GrandmaAgent():
    do_test_genius_agent(GrandmaAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Gravity():
    do_test_genius_agent(Gravity)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group1():
    do_test_genius_agent(Group1)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group1BOA():
    do_test_genius_agent(Group1BOA)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AI2014Group2():
    do_test_genius_agent(AI2014Group2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group3():
    do_test_genius_agent(Group3)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group3Q2015():
    do_test_genius_agent(Group3Q2015)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group4():
    do_test_genius_agent(Group4)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group5():
    do_test_genius_agent(Group5)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group6():
    do_test_genius_agent(Group6)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group7():
    do_test_genius_agent(Group7)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group8():
    do_test_genius_agent(Group8)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group9():
    do_test_genius_agent(Group9)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group10():
    do_test_genius_agent(Group10)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group11():
    do_test_genius_agent(Group11)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group12():
    do_test_genius_agent(Group12)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group13():
    do_test_genius_agent(Group13)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group14():
    do_test_genius_agent(Group14)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group15():
    do_test_genius_agent(Group15)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group16():
    do_test_genius_agent(Group16)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group17():
    do_test_genius_agent(Group17)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group18():
    do_test_genius_agent(Group18)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group19():
    do_test_genius_agent(Group19)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group20():
    do_test_genius_agent(Group20)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group21():
    do_test_genius_agent(Group21)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Group22():
    do_test_genius_agent(Group22)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_AgentNeo():
    do_test_genius_agent(AgentNeo)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_GroupY():
    do_test_genius_agent(GroupY)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_HardDealer():
    do_test_genius_agent(HardDealer)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_HardHeaded():
    do_test_genius_agent(HardHeaded)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_IAMcrazyHaggler():
    do_test_genius_agent(IAMcrazyHaggler)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_IAMhaggler():
    do_test_genius_agent(IAMhaggler)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_IAMhaggler2011():
    do_test_genius_agent(IAMhaggler2011)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_IAMhaggler2012():
    do_test_genius_agent(IAMhaggler2012)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Imitator():
    do_test_genius_agent(Imitator)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ImmediateAcceptor():
    do_test_genius_agent(ImmediateAcceptor)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_InoxAgent():
    do_test_genius_agent(InoxAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_IQSun2018():
    do_test_genius_agent(IQSun2018)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_JonnyBlack():
    do_test_genius_agent(JonnyBlack)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_KAgent():
    do_test_genius_agent(KAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_KakeSoba():
    do_test_genius_agent(KakeSoba)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_KGAgent():
    do_test_genius_agent(KGAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Lancelot():
    do_test_genius_agent(Lancelot)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Libra():
    do_test_genius_agent(Libra)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MadAgent():
    do_test_genius_agent(MadAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Mamenchis():
    do_test_genius_agent(Mamenchis)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MaxOops():
    do_test_genius_agent(MaxOops)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MeanBot():
    do_test_genius_agent(MeanBot)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MengWan():
    do_test_genius_agent(MengWan)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Mercury():
    do_test_genius_agent(Mercury)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MetaAgent():
    do_test_genius_agent(MetaAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MetaAgent2012():
    do_test_genius_agent(MetaAgent2012)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MetaAgent2013():
    do_test_genius_agent(MetaAgent2013)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Mosa():
    do_test_genius_agent(Mosa)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_MyAgent():
    do_test_genius_agent(MyAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Ngent():
    do_test_genius_agent(Ngent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_NiceTitForTat():
    do_test_genius_agent(NiceTitForTat)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Nozomi():
    do_test_genius_agent(Nozomi)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_OMACagent():
    do_test_genius_agent(OMACagent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_OptimalBidderSimple():
    do_test_genius_agent(OptimalBidderSimple)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ParsAgent():
    do_test_genius_agent(ParsAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ParsAgent2():
    do_test_genius_agent(ParsAgent2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ParsAgent3():
    do_test_genius_agent(ParsAgent3)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ParsCat():
    do_test_genius_agent(ParsCat)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ParsCat2():
    do_test_genius_agent(ParsCat2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_PhoenixParty():
    do_test_genius_agent(PhoenixParty)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_PNegotiator():
    do_test_genius_agent(PNegotiator)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_PokerFace():
    do_test_genius_agent(PokerFace)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_PonPokoAgent():
    do_test_genius_agent(PonPokoAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_PonPokoRampage():
    do_test_genius_agent(PonPokoRampage)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Q12015Group2():
    do_test_genius_agent(Q12015Group2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_RandomCounterOfferNegotiationParty():
    do_test_genius_agent(RandomCounterOfferNegotiationParty)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_RandomDance():
    do_test_genius_agent(RandomDance)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_RandomParty():
    do_test_genius_agent(RandomParty)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_RandomParty2():
    do_test_genius_agent(RandomParty2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Rubick():
    do_test_genius_agent(Rubick)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Seto():
    do_test_genius_agent(Seto)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ShahAgent():
    do_test_genius_agent(ShahAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Shiboy():
    do_test_genius_agent(Shiboy)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SimilarityAgent():
    do_test_genius_agent(SimilarityAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Simpatico():
    do_test_genius_agent(Simpatico)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SimpleAgent():
    do_test_genius_agent(SimpleAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SimpleAgent2017():
    do_test_genius_agent(SimpleAgent2017)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SlavaAgent():
    do_test_genius_agent(SlavaAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SMACAgent():
    do_test_genius_agent(SMACAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Sobut():
    do_test_genius_agent(Sobut)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SolverAgent():
    do_test_genius_agent(SolverAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Sontag():
    do_test_genius_agent(Sontag)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SouthamptonAgent():
    do_test_genius_agent(SouthamptonAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_SYAgent():
    do_test_genius_agent(SYAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TaxiBox():
    do_test_genius_agent(TaxiBox)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Terra():
    do_test_genius_agent(Terra)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TheFawkes():
    do_test_genius_agent(TheFawkes)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TheNegotiator():
    do_test_genius_agent(TheNegotiator)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TheNegotiatorReloaded():
    do_test_genius_agent(TheNegotiatorReloaded)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TheNewDeal():
    do_test_genius_agent(TheNewDeal)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TimeDependentAgentBoulware():
    do_test_genius_agent(TimeDependentAgentBoulware)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TimeDependentAgentConceder():
    do_test_genius_agent(TimeDependentAgentConceder)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TimeDependentAgentHardliner():
    do_test_genius_agent(TimeDependentAgentHardliner)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TimeDependentAgentLinear():
    do_test_genius_agent(TimeDependentAgentLinear)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TMFAgent():
    do_test_genius_agent(TMFAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TucAgent():
    do_test_genius_agent(TucAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TUDelftGroup2():
    do_test_genius_agent(TUDelftGroup2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_TUDMixedStrategyAgent():
    do_test_genius_agent(TUDMixedStrategyAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_UtilityBasedAcceptor():
    do_test_genius_agent(UtilityBasedAcceptor)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_ValueModelAgent():
    do_test_genius_agent(ValueModelAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_WhaleAgent():
    do_test_genius_agent(WhaleAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_WinkyAgent():
    do_test_genius_agent(WinkyAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_XianFaAgent():
    do_test_genius_agent(XianFaAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Y2015Group2():
    do_test_genius_agent(Y2015Group2)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Yeela():
    do_test_genius_agent(Yeela)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Yushu():
    do_test_genius_agent(Yushu)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_YXAgent():
    do_test_genius_agent(YXAgent)


@pytest.mark.skipif(condition=SKIP_CONDITION, reason="Either no genius or fast run")
def test_Kawaii():
    do_test_genius_agent(Kawaii)


if __name__ == "__main__":
    pytest.main(args=[__file__])
