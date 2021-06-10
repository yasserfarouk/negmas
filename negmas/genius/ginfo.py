"""
Keeps information about ANAC competitions and Genius Agents
"""
import itertools
from typing import List, Set, Tuple

__all__ = [
    "GENIUS_INFO",
    "AGENT_BASED_NEGOTIATORS",
    "PARTY_BASED_NEGOTIATORS",
    "ALL_NEGOTIATORS",
    "TEST_FAILING_NEGOTIATORS",
    "ALL_PASSING_NEGOTIATORS",
    "TESTED_NEGOTIATORS",
    "get_genius_agents",
]
GENIUS_INFO = {
    2010: {
        "winners": [
            [("AgentK", "agents.anac.y2010.AgentK.Agent_K")],
            [("Yushu", "agents.anac.y2010.Yushu.Yushu")],
            [("Nozomi", "agents.anac.y2010.Nozomi.Nozomi")],
            [("IAMhaggler", "agents.anac.y2010.Southampton.IAMhaggler")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": False,
        "discounting": True,
        "uncertainty": False,
        "elicitation": False,
    },
    2011: {
        "winners": [
            [("HardHeaded", "agents.anac.y2011.HardHeaded.KLH")],
            [("Gahboninho", "agents.anac.y2011.Gahboninho.Gahboninho")],
            [("IAMhaggler2011", "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011")],
        ],
        "finalists": [
            [("HardHeaded", "agents.anac.y2011.HardHeaded.KLH")],
            [("Gahboninho", "agents.anac.y2011.Gahboninho.Gahboninho")],
            [("IAMhaggler2011", "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011")],
            [("BRAMAgent", "agents.anac.y2011.BramAgent.BRAMAgent")],
            [("AgentK2", "agents.anac.y2011.AgentK2.Agent_K2")],
            [("TheNegotiator", "agents.anac.y2011.TheNegotiator.TheNegotiator")],
            [("NiceTitForTat", "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat")],
            [("ValueModelAgent", "agents.anac.y2011.ValueModelAgent.ValueModelAgent")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": False,
        "discounting": True,
        "uncertainty": False,
        "elicitation": False,
    },
    2012: {
        "winners": [
            [
                (
                    "CUHKAgent",
                    "agents.anac.y2012.CUHKAgent.CUHKAgent",
                )
            ],
            [("AgentLG", "agents.anac.y2012.AgentLG.AgentLG")],
            [
                (
                    "OMACagent",
                    "agents.anac.y2012.OMACagent.OMACagent",
                ),
                (
                    "TheNegotiatorReloaded",
                    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
                ),
            ],
        ],
        "best_discounted": [
            (
                "CUHKAgent",
                "agents.anac.y2012.CUHKAgent.CUHKAgent",
            )
        ],
        "best_undiscounted": [
            (
                "TheNegotiatorReloaded",
                "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
            ),
        ],
        "winners_welfare": [
            [
                (
                    "IAMhaggler2012",
                    "agents.anac.y2012.IAMhaggler2012.agents2011.SouthamptonAgent",
                )
            ],
            [
                (
                    "TheNegotiatorReloaded",
                    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
                )
            ],
            [("MetaAgent2012", "agents.anac.y2012.MetaAgent.MetaAgent")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": True,
        "discounting": True,
        "uncertainty": False,
        "elicitation": False,
    },
    2013: {
        "winners": [
            [("TheFawkes", "agents.anac.y2013.TheFawkes.TheFawkes")],
            [
                (
                    "MetaAgent",
                    "agents.anac.y2013.MetaAgent.MetaAgent2013",
                )
            ],
            [("TMFAgent", "agents.anac.y2013.TMFAgent.TMFAgent")],
        ],
        "linear": True,
        "learning": True,
        "multilateral": False,
        "bilateral": True,
        "reservation": True,
        "discounting": True,
        "uncertainty": False,
        "elicitation": False,
    },
    2014: {
        "winners": [
            [("AgentM", "agents.anac.y2014.AgentM.AgentM")],
            [("DoNA", "agents.anac.y2014.DoNA.DoNA")],
            [("Gangester", "agents.anac.y2014.Gangster.Gangster")],
        ],
        "winners_welfare": [
            [("AgentM", "agents.anac.y2014.AgentM.AgentM")],
            [("Gangester", "agents.anac.y2014.Gangster.Gangster")],
            [("E2Agent", "agents.anac.y2014.E2Agent.AnacSampleAgent")],
        ],
        "finalists": [
            [("AgentM", "agents.anac.y2014.AgentM.AgentM")],
            [("DoNA", "agents.anac.y2014.DoNA.DoNA")],
            [("Gangester", "agents.anac.y2014.Gangster.Gangster")],
            [("WhaleAgent", "agents.anac.y2014.AgentWhale.WhaleAgent")],
            [("Group2", "agents.ai2014.group2.Group2")],
            [("E2Agent", "agents.anac.y2014.E2Agent.AnacSampleAgent")],
            [("KGAgent", "agents.anac.y2014.KGAgent.KGAgent")],
            [("AgentYK", "agents.anac.y2014.AgentYK.AgentYK")],
            [("BraveCat", "agents.anac.y2014.BraveCat.BraveCat")],
        ],
        "linear": False,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": None,
        "discounting": None,
        "uncertainty": False,
        "elicitation": False,
    },
    2015: {
        "winners": [
            [("Atlas3", "agents.anac.y2015.Atlas3.Atlas3")],
            [("ParsAgent", "agents.anac.y2015.ParsAgent.ParsAgent")],
            [("RandomDance", "agents.anac.y2015.RandomDance.RandomDance")],
        ],
        "finalists": [
            [("Atlas3", "agents.anac.y2015.Atlas3.Atlas3")],
            [("ParsAgent", "agents.anac.y2015.ParsAgent.ParsAgent")],
            [("RandomDance", "agents.anac.y2015.RandomDance.RandomDance")],
            [("Kawaii", "agents.anac.y2015.fairy.kawaii")],
            [
                (
                    "AgentBuyong",
                    "agents.anac.y2015.agentBuyogV2.AgentBuyogMain",
                )
            ],
            [("PhoenixParty", "agents.anac.y2015.Phoenix.PhoenixParty")],
            [("XianFaAgent", "agents.anac.y2015.xianfa.XianFaAgent")],
            [("PokerFace", "agents.anac.y2015.pokerface.PokerFace")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": True,
        "bilateral": False,
        "reservation": None,
        "discounting": None,
        "uncertainty": False,
        "elicitation": False,
    },
    2016: {
        "winners": [
            [("Caduceus", "agents.anac.y2016.caduceus.Caduceus")],
            [
                ("YXAgent", "agents.anac.y2016.yxagent.YXAgent"),
                ("ParsCat", "agents.anac.y2016.parscat.ParsCat"),
            ],
        ],
        "finalists": [
            [("Caduceus", "agents.anac.y2016.caduceus.Caduceus")],
            [("YXAgent", "agents.anac.y2016.yxagent.YXAgent")],
            [("ParsCat", "agents.anac.y2016.parscat.ParsCat")],
            [("Farma", "agents.anac.y2017.farma.Farma17")],
            [("MyAgent", "agents.anac.y2016.myagent.MyAgent")],
            [("Atlas32016", "agents.anac.y2016.atlas3.Atlas32016")],
            [("Ngent", "agents.anac.y2016.ngent.Ngent")],
            [("GrandmaAgent", "agents.anac.y2016.grandma.GrandmaAgent")],
            [("AgentHP2", "agents.anac.y2016.agenthp2.AgentHP2_main")],
            [("Terra", "agents.anac.y2016.terra.Terra")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": True,
        "bilateral": False,
        "reservation": None,
        "discounting": None,
        "uncertainty": False,
        "elicitation": False,
    },
    2017: {
        "winners": [
            [("PonPokoAgent", "agents.anac.y2017.ponpokoagent.PonPokoAgent")],
            [("CaduceusDC16", "agents.anac.y2017.caduceusdc16.CaduceusDC16")],
            [("Rubick", "agents.anac.y2017.rubick.Rubick")],
        ],
        "linear": True,
        "learning": True,
        "multilateral": True,
        "bilateral": False,
        "reservation": None,
        "discounting": None,
        "uncertainty": False,
        "elicitation": False,
    },
    2018: {
        "winners": [
            [
                (
                    "AgreeableAgent2018",
                    "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
                )
            ],
            [("MengWan", "agents.anac.y2018.meng_wan.Agent36")],
            [("BetaOne", "agents.anac.y2018.beta_one.Group2")],
        ],
        "linear": True,
        "learning": True,
        "multilateral": True,
        "bilateral": False,
        "reservation": None,
        "discounting": None,
        "uncertainty": False,
        "elicitation": False,
    },
    # 2019: {
    #     "winners": [["AgentGG"],[ "KakeSoba"],[ "SAGA"]],
    #     "linear": True,
    #     "learning": True,
    #     "multilateral": True,
    #     "bilateral": False,
    #     "reservation": None,
    #     "discounting": None,
    #     "uncertainty": True,
    #     "elicitation": False,
    # },
    # 2020: {
    #     "winners": [],
    #     "linear": True,
    #     "learning": True,
    #     "multilateral": True,
    #     "bilateral": False,
    #     "reservation": None,
    #     "discounting": None,
    #     "uncertainty": True,
    #     "elicitation": True,
    # },
}


AGENT_BASED_NEGOTIATORS = [
    "agents.ABMPAgent",
    "agents.ABMPAgent2",
    "agents.BayesianAgent",
    "agents.BayesianAgentForAuction",
    "agents.DecUtilAgent",
    "agents.FuzzyAgent",
    "agents.OptimalBidder",
    "agents.OptimalBidderU",
    "agents.QOAgent",
    "agents.RandomIncreasingUtilAgent",
    "agents.SimilarityAgent",
    "agents.anac.y2010.AgentFSEGA.AgentFSEGA",
    "agents.anac.y2010.AgentK.Agent_K",
    "agents.anac.y2010.AgentSmith.AgentSmith",
    "agents.anac.y2010.Nozomi.Nozomi",
    "agents.anac.y2010.Southampton.SouthamptonAgent",
    "agents.anac.y2010.Yushu.Yushu",
    "agents.anac.y2011.AgentK2.Agent_K2",
    "agents.anac.y2011.BramAgent.BRAMAgent",
    "agents.anac.y2011.Gahboninho.Gahboninho",
    "agents.anac.y2011.HardHeaded.KLH",
    "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
    "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat",
    "agents.anac.y2011.TheNegotiator.TheNegotiator",
    "agents.anac.y2011.ValueModelAgent.ValueModelAgent",
    "agents.anac.y2012.AgentLG.AgentLG",
    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
    "agents.anac.y2012.AgentMR.AgentMR",
    "agents.anac.y2012.BRAMAgent2.BRAMAgent2",
    "agents.anac.y2012.CUHKAgent.CUHKAgent",
    "agents.anac.y2012.MetaAgent.MetaAgent",
    "agents.anac.y2012.OMACagent.OMACagent",
    "agents.anac.y2013.AgentKF.AgentKF",
    "agents.anac.y2013.GAgent.AgentI",
    "agents.anac.y2013.MetaAgent.MetaAgent2013",
    "agents.anac.y2013.SlavaAgent.SlavaAgent",
    "agents.anac.y2013.TMFAgent.TMFAgent",
    "agents.anac.y2013.TheFawkes.TheFawkes",
    "agents.anac.y2014.AgentM.AgentM",
    "agents.anac.y2014.AgentQuest.AgentQuest",
    "agents.anac.y2014.AgentTD.AgentTD",
    "agents.anac.y2014.AgentTRP.AgentTRP",
    "agents.anac.y2014.AgentWhale.WhaleAgent",
    "agents.anac.y2014.AgentYK.AgentYK",
    "agents.anac.y2014.ArisawaYaki.ArisawaYaki",
    "agents.anac.y2014.Aster.Aster",
    "agents.anac.y2014.Atlas.Atlas",
    "agents.anac.y2014.BraveCat.BraveCat",
    "agents.anac.y2014.DoNA.DoNA",
    "agents.anac.y2014.E2Agent.AnacSampleAgent",
    "agents.anac.y2014.Flinch.Flinch",
    "agents.anac.y2014.Gangster.Gangster",
    "agents.anac.y2014.KGAgent.KGAgent",
    "agents.anac.y2014.SimpaticoAgent.Simpatico",
    "agents.anac.y2014.Sobut.Sobut",
    "agents.anac.y2017.geneking.GeneKing",
]
"""Genius agents based on the Agent base-class. These are the oldest agents"""

PARTY_BASED_NEGOTIATORS = [
    "agents.ai2014.group1.Group1",
    "agents.ai2014.group10.Group10",
    "agents.ai2014.group11.Group11",
    "agents.ai2014.group12.Group12",
    "agents.ai2014.group2.Group2",
    "agents.ai2014.group3.Group3",
    "agents.ai2014.group4.Group4",
    "agents.ai2014.group5.Group5",
    "agents.ai2014.group6.Group6",
    "agents.ai2014.group7.Group7",
    "agents.ai2014.group9.Group9",
    "agents.anac.y2015.AgentNeo.Groupn",
    "agents.anac.y2015.AgentX.AgentX",
    "agents.anac.y2015.AresParty.AresParty",
    "agents.anac.y2015.Atlas3.Atlas3",
    "agents.anac.y2015.DrageKnight.DrageKnight",
    "agents.anac.y2015.JonnyBlack.JonnyBlack",
    "agents.anac.y2015.Mercury.Mercury",
    "agents.anac.y2015.ParsAgent.ParsAgent",
    "agents.anac.y2015.Phoenix.PhoenixParty",
    "agents.anac.y2015.RandomDance.RandomDance",
    "agents.anac.y2015.SENGOKU.SENGOKU",
    "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent",
    "agents.anac.y2015.agentBuyogV2.AgentBuyogMain",
    "agents.anac.y2015.agenth.AgentH",
    "agents.anac.y2015.cuhkagent2015.CUHKAgent2015",
    "agents.anac.y2015.fairy.kawaii",
    "agents.anac.y2015.group2.Group2",
    "agents.anac.y2015.meanBot.MeanBot",
    "agents.anac.y2015.pokerface.PokerFace",
    "agents.anac.y2016.agentsmith.AgentSmith2016",
    "agents.anac.y2016.atlas3.Atlas32016",
    "agents.anac.y2016.caduceus.Caduceus",
    "agents.anac.y2016.clockworkagent.ClockworkAgent",
    "agents.anac.y2016.farma.Farma",
    "agents.anac.y2016.grandma.GrandmaAgent",
    "agents.anac.y2016.myagent.MyAgent",
    "agents.anac.y2016.ngent.Ngent",
    "agents.anac.y2016.pars2.ParsAgent2",
    "agents.anac.y2016.parscat.ParsCat",
    "agents.anac.y2016.syagent.SYAgent",
    "agents.anac.y2016.terra.Terra",
    "agents.anac.y2016.yxagent.YXAgent",
    "agents.anac.y2017.agentf.AgentF",
    "agents.anac.y2017.caduceusdc16.CaduceusDC16",
    "agents.anac.y2017.farma.Farma17",
    "agents.anac.y2017.ponpokoagent.PonPokoAgent",
    "agents.anac.y2017.tangxun.taxibox",
    "agents.anac.y2017.tucagent.TucAgent",
]
"""Genius agents based on the Party base-class. These are the newest agents"""

TEST_FAILING_NEGOTIATORS = [
    "agents.BayesianAgentForAuction",
    "agents.DecUtilAgent",
    "agents.FuzzyAgent",
    "agents.OptimalBidder",
    "agents.OptimalBidderU",
    "agents.QOAgent",
    "agents.RandomIncreasingUtilAgent",
    "agents.SimilarityAgent",
    "agents.anac.y2014.AgentTD.AgentTD",
    "agents.anac.y2017.geneking.GeneKing",
    "agents.anac.y2017.rubick.Rubick",
    "agents.anac.y2017.caduceusdc16.CaduceusDC16",
    "agents.anac.y2018.beta_one.Group2",
    "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
    "agents.anac.y2016.agenthp2.AgentHP2_main",
    "agents.anac.y2014.KGAgent.KGAgent",
    "agents.anac.y2018.meng_wan.Agent36",
    "agents.anac.y2014.E2Agent.AnacSampleAgent",
    "agents.anac.y2010.Southampton.SouthamptonAgent",

]
"""Agetns taht fail simple tests making them less robust over the bridge"""

TESTED_NEGOTIATORS = list(set(["agents.anac.y2015.AgentX.AgentX"] + list(
    itertools.chain(
        *[
            list(_[1] for _ in itertools.chain(*v["winners"]))
            for year, v in GENIUS_INFO.items()
            if v["multilateral"] and not v["learning"]
        ]
    )
)) - set(TEST_FAILING_NEGOTIATORS))
"""Some of the most tested negotaitors"""

ALL_NEGOTIATORS = PARTY_BASED_NEGOTIATORS + AGENT_BASED_NEGOTIATORS
"""All Genius Negotiators accessible from NegMAS"""

ALL_PASSING_NEGOTIATORS = list(set(ALL_NEGOTIATORS) - set(TEST_FAILING_NEGOTIATORS))
"""All negotiators that passed simple tests showing they work on the bridge"""

def get_genius_agents(
    *,
    year: int = None,
    linear: bool = None,
    learning: bool = None,
    multilateral: bool = None,
    bilateral: bool = None,
    reservation: bool = None,
    discounting: bool = None,
    uncertainty: bool = None,
    elicitation: bool = None,
    winners_only: bool = False,
    finalists_only: bool = False,
) -> List[Tuple[str, str]]:
    """
    Get Genius agents matching some given criteria

    Returns:
        a list of 2-item tuples giving agent name and java class name.

    Remarks:
        - For all criteria other than winners_only, and finalists_only, passing
          None means no constraints otherwise the game on the given year must
          match the criterion value (True or False) to get agents from this year.
        - This function uses a heuristic and may not be completely accurate. Use
          with caution.
    """

    def get_agents(year, d) -> List[Tuple[str, str]]:
        lst = tuple(
            d.get("winners", [[]])
            if winners_only
            else d.get("finalists", [])
            if finalists_only
            else [(_.split(".")[-1], _) for _ in ALL_NEGOTIATORS if str(year) in _]
        )
        if winners_only or finalists_only:
            return set(itertools.chain(*lst))
        return set(lst)

    agents: Set[Tuple[str, str]] = set()
    if year is None:
        for y in GENIUS_INFO.keys():
            agents = agents.union(
                get_genius_agents(
                    year=y,
                    linear=linear,
                    learning=learning,
                    multilateral=multilateral,
                    bilateral=bilateral,
                    reservation=reservation,
                    discounting=discounting,
                    uncertainty=uncertainty,
                    elicitation=elicitation,
                    winners_only=winners_only,
                    finalists_only=finalists_only,
                )
            )
        return list(agents)

    d = GENIUS_INFO.get(year, None)
    if not d:
        return []
    if linear is None or (linear is not None and d["linear"] == linear):
        agents = get_agents(year, d)
    if learning is None or (learning is not None and d["learning"] == learning):
        agents = agents.intersection(get_agents(year, d))
    if bilateral is None or (bilateral is not None and d["bilateral"] == bilateral):
        agents = agents.intersection(get_agents(year, d))
    if multilateral is None or (
        multilateral is not None and d["multilateral"] == multilateral
    ):
        agents = agents.intersection(get_agents(year, d))
    if discounting is None or (
        discounting is not None and d["discounting"] == discounting
    ):
        agents = agents.intersection(get_agents(year, d))
    if reservation is None or (
        reservation is not None and d["reservation"] == reservation
    ):
        agents = agents.intersection(get_agents(year, d))
    if uncertainty is None or (
        uncertainty is not None and d["uncertainty"] == uncertainty
    ):
        agents = agents.intersection(get_agents(year, d))
    if elicitation is None or (
        elicitation is not None and d["elicitation"] == elicitation
    ):
        agents = agents.intersection(get_agents(year, d))
    return list(agents)
