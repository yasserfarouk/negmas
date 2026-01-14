"""Keeps information about ANAC competitions and Genius Agents."""

from __future__ import annotations

import itertools
from importlib.util import find_spec

from negmas.helpers.strings import shortest_unique_names

__all__ = [
    "GENIUS_INFO",
    "ANAC_INFO",
    "AGENT_BASED_NEGOTIATORS",
    "PARTY_BASED_NEGOTIATORS",
    "ALL_NEGOTIATORS",
    "TEST_FAILING_NEGOTIATORS",
    "ALL_PASSING_NEGOTIATORS",
    "TESTED_NEGOTIATORS",
    "ALL_GENIUS_BASIC_NEGOTIATORS",
    "ALL_GENIUS_INVALID_NEGOTIATORS",
    "ALL_GENIUS_NEGOTIATORS",
    "get_anac_agents",
]

# Check if optional agent libraries are installed
_HAS_ANL_AGENTS = find_spec("anl_agents") is not None
_HAS_GENIUSWEB_BRIDGE = find_spec("negmas_geniusweb_bridge") is not None


def _anl_path(path: str) -> str | None:
    """Return the path if anl_agents is installed, otherwise None."""
    return path if _HAS_ANL_AGENTS else None


def _geniusweb_path(path: str) -> str | None:
    """Return the path if negmas_geniusweb_bridge is installed, otherwise None."""
    return path if _HAS_GENIUSWEB_BRIDGE else None


GENIUS_INFO = {
    2010: {
        "winners": [
            [("AgentK", "agents.anac.y2010.AgentK.Agent_K")],
            [("Yushu", "agents.anac.y2010.Yushu.Yushu")],
            [("Nozomi", "agents.anac.y2010.Nozomi.Nozomi")],
            [("IAMhaggler", "agents.anac.y2010.Southampton.IAMhaggler")],
        ],
        "finalists": [
            [("AgentK", "agents.anac.y2010.AgentK.Agent_K")],
            [("Yushu", "agents.anac.y2010.Yushu.Yushu")],
            [("Nozomi", "agents.anac.y2010.Nozomi.Nozomi")],
            [("IAMhaggler", "agents.anac.y2010.Southampton.IAMhaggler")],
        ],
        "participants": [
            [("AgentFSEGA", "agents.anac.y2010.AgentFSEGA.AgentFSEGA")],
            [("AgentK", "agents.anac.y2010.AgentK.Agent_K")],
            [("AgentSmith", "agents.anac.y2010.AgentSmith.AgentSmith")],
            [("Nozomi", "agents.anac.y2010.Nozomi.Nozomi")],
            [("IAMcrazyHaggler", "agents.anac.y2010.Southampton.IAMcrazyHaggler")],
            [("IAMhaggler", "agents.anac.y2010.Southampton.IAMhaggler")],
            [("Yushu", "agents.anac.y2010.Yushu.Yushu")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": False,
        "discounting": True,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
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
            [("BramAgent", "agents.anac.y2011.BramAgent.BRAMAgent")],
            [("AgentK2", "agents.anac.y2011.AgentK2.Agent_K2")],
            [("TheNegotiator", "agents.anac.y2011.TheNegotiator.TheNegotiator")],
            [("NiceTitForTat", "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat")],
            [("ValueModelAgent", "agents.anac.y2011.ValueModelAgent.ValueModelAgent")],
        ],
        "participants": [
            [("AgentK2", "agents.anac.y2011.AgentK2.Agent_K2")],
            [("BramAgent", "agents.anac.y2011.BramAgent.BRAMAgent")],
            [("Gahboninho", "agents.anac.y2011.Gahboninho.Gahboninho")],
            [("HardHeaded", "agents.anac.y2011.HardHeaded.KLH")],
            [("IAMhaggler2011", "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011")],
            [("NiceTitForTat", "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat")],
            [("TheNegotiator", "agents.anac.y2011.TheNegotiator.TheNegotiator")],
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
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2012: {
        "winners": [
            [("CUHKAgent", "agents.anac.y2012.CUHKAgent.CUHKAgent")],
            [("AgentLG", "agents.anac.y2012.AgentLG.AgentLG")],
            [
                ("OMACagent", "agents.anac.y2012.OMACagent.OMACagent"),
                (
                    "TheNegotiatorReloaded",
                    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
                ),
            ],
        ],
        "finalists": [
            [("CUHKAgent", "agents.anac.y2012.CUHKAgent.CUHKAgent")],
            [("OMACagent", "agents.anac.y2012.OMACagent.OMACagent")],
            [
                (
                    "TheNegotiatorReloaded",
                    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
                )
            ],
            [("BramAgent2", "agents.anac.y2011.BramAgent.BRAMAgent")],
            [("MetaAgent", "agents.anac.y2012.MetaAgent.MetaAgent")],
            [("AgentLG", "agents.anac.y2012.AgentLG.AgentLG")],
            [("IAMhaggler2012", "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012")],
            [("AgentMR", "agents.anac.y2012.AgentMR.AgentMR")],
        ],
        "participants": [
            [("AgentLG", "agents.anac.y2012.AgentLG.AgentLG")],
            [("AgentMR", "agents.anac.y2012.AgentMR.AgentMR")],
            [("BramAgent2", "agents.anac.y2011.BramAgent.BRAMAgent")],
            [("CUHKAgent", "agents.anac.y2012.CUHKAgent.CUHKAgent")],
            [("IAMhaggler2012", "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012")],
            [("MetaAgent", "agents.anac.y2012.MetaAgent.MetaAgent")],
            [("OMACagent", "agents.anac.y2012.OMACagent.OMACagent")],
            [
                (
                    "TheNegotiatorReloaded",
                    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
                )
            ],
        ],
        "best_discounted": [("CUHKAgent", "agents.anac.y2012.CUHKAgent.CUHKAgent")],
        "best_undiscounted": [
            (
                "TheNegotiatorReloaded",
                "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
            )
        ],
        "winners_welfare": [
            [("IAMhaggler2012", "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012")],
            [
                (
                    "TheNegotiatorReloaded",
                    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
                )
            ],
            [("MetaAgent", "agents.anac.y2012.MetaAgent.MetaAgent")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": True,
        "discounting": True,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2013: {
        "winners": [
            [("TheFawkes", "agents.anac.y2013.TheFawkes.TheFawkes")],
            [("MetaAgent2013", "agents.anac.y2013.MetaAgent.MetaAgent2013")],
            [("TMFAgent", "agents.anac.y2013.TMFAgent.TMFAgent")],
        ],
        "finalists": [
            [("AgentKF", "agents.anac.y2013.AgentKF.AgentKF")],
            [("TheFawkes", "agents.anac.y2013.TheFawkes.TheFawkes")],
            [("TMFAgent", "agents.anac.y2013.TMFAgent.TMFAgent")],
            [("MetaAgent2013", "agents.anac.y2013.MetaAgent.MetaAgent2013")],
            [("GAgent", "agents.anac.y2013.GAgent.AgentI")],
            [("InoxAgent", "agents.anac.y2013.InoxAgent.InoxAgent")],
            [("SlavaAgent", "agents.anac.y2013.SlavaAgent.SlavaAgent")],
        ],
        "participants": [
            [("AgentKF", "agents.anac.y2013.AgentKF.AgentKF")],
            [("GAgent", "agents.anac.y2013.GAgent.AgentI")],
            [("InoxAgent", "agents.anac.y2013.InoxAgent.InoxAgent")],
            [("MetaAgent2013", "agents.anac.y2013.MetaAgent.MetaAgent2013")],
            [("SlavaAgent", "agents.anac.y2013.SlavaAgent.SlavaAgent")],
            [("TheFawkes", "agents.anac.y2013.TheFawkes.TheFawkes")],
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
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
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
            [("TUDelftGroup2", "agents.anac.y2014.TUDelftGroup2.Group2Agent")],
            [("E2Agent", "agents.anac.y2014.E2Agent.AnacSampleAgent")],
            [("KGAgent", "agents.anac.y2014.KGAgent.KGAgent")],
            [("AgentYK", "agents.anac.y2014.AgentYK.AgentYK")],
            [("BraveCat", "agents.anac.y2014.BraveCat.BraveCat")],
        ],
        "participants": [
            [("AgentM", "agents.anac.y2014.AgentM.AgentM")],
            [("DoNA", "agents.anac.y2014.DoNA.DoNA")],
            [("Gangester", "agents.anac.y2014.Gangster.Gangster")],
            [("WhaleAgent", "agents.anac.y2014.AgentWhale.WhaleAgent")],
            [("TUDelftGroup2", "agents.anac.y2014.TUDelftGroup2.Group2Agent")],
            [("E2Agent", "agents.anac.y2014.E2Agent.AnacSampleAgent")],
            [("KGAgent", "agents.anac.y2014.KGAgent.KGAgent")],
            [("AgentYK", "agents.anac.y2014.AgentYK.AgentYK")],
            [("BraveCat", "agents.anac.y2014.BraveCat.BraveCat")],
            [("AgentQuest", "agents.anac.y2014.AgentQuest.AgentQuest")],
            [("AgentTD", "agents.anac.y2014.AgentTD.AgentTD")],
            [("AgentTRP", "agents.anac.y2014.AgentTRP.AgentTRP")],
            [("ArisawaYaki", "agents.anac.y2014.ArisawaYaki.ArisawaYaki")],
            [("Aster", "agents.anac.y2014.Aster.Aster")],
            [("Atlas", "agents.anac.y2014.Atlas.Atlas")],
            [("Flinch", "agents.anac.y2014.Flinch.Flinch")],
            [("SimpaticoAgent", "agents.anac.y2014.SimpaticoAgent.Simpatico")],
            [("Sobut", "agents.anac.y2014.Sobut.Sobut")],
        ],
        "linear": False,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
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
            [("AgentBuyog", "agents.anac.y2015.agentBuyogV2.AgentBuyogMain")],
            [("PhoenixParty", "agents.anac.y2015.Phoenix.PhoenixParty")],
            [("XianFaAgent", "agents.anac.y2015.xianfa.XianFaAgent")],
            [("PokerFace", "agents.anac.y2015.pokerface.PokerFace")],
        ],
        "participants": [
            [("Atlas3", "agents.anac.y2015.Atlas3.Atlas3")],
            [("ParsAgent", "agents.anac.y2015.ParsAgent.ParsAgent")],
            [("RandomDance", "agents.anac.y2015.RandomDance.RandomDance")],
            [("Kawaii", "agents.anac.y2015.fairy.kawaii")],
            [("AgentBuyog", "agents.anac.y2015.agentBuyogV2.AgentBuyogMain")],
            [("PhoenixParty", "agents.anac.y2015.Phoenix.PhoenixParty")],
            [("XianFaAgent", "agents.anac.y2015.xianfa.XianFaAgent")],
            [("PokerFace", "agents.anac.y2015.pokerface.PokerFace")],
            [("AgentH", "agents.anac.y2015.agenth.AgentH")],
            [("AgentHP", "agents.anac.y2015.AgentHP.AgentHP")],
            [("AgentNeo", "agents.anac.y2015.AgentNeo.Groupn")],
            [("AgentW", "agents.anac.y2015.AgentW.AgentW")],
            [("AgentX", "agents.anac.y2015.AgentX.AgentX")],
            [("AresParty", "agents.anac.y2015.AresParty.AresParty")],
            [("CUHKAgent2015", "agents.anac.y2015.cuhkagent2015.CUHKAgent2015")],
            [("DrageKnight", "agents.anac.y2015.DrageKnight.DrageKnight")],
            [("Kawaii", "agents.anac.y2015.fairy.kawaii")],
            [("Y2015Group2", "agents.anac.y2015.group2.Group2")],
            [("JonnyBlack", "agents.anac.y2015.JonnyBlack.JonnyBlack")],
            [("MeanBot", "agents.anac.y2015.meanBot.MeanBot")],
            [("Mercury", "agents.anac.y2015.Mercury.Mercury")],
            [("PNegotiator", "agents.anac.y2015.pnegotiator.PNegotiator")],
            [("SENGOKU", "agents.anac.y2015.SENGOKU.SENGOKU")],
            [
                (
                    "TUDMixedStrategyAgent",
                    "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent",
                )
            ],
        ],
        "linear": True,
        "learning": False,
        "multilateral": True,
        "bilateral": False,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
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
            [("Farma", "agents.anac.y2016.farma.Farma")],
            [("MyAgent", "agents.anac.y2016.myagent.MyAgent")],
            [("Atlas32016", "agents.anac.y2016.atlas3.Atlas32016")],
            [("Ngent", "agents.anac.y2016.ngent.Ngent")],
            [("GrandmaAgent", "agents.anac.y2016.grandma.GrandmaAgent")],
            [("AgentHP2", "agents.anac.y2016.agenthp2.AgentHP2_main")],
            [("Terra", "agents.anac.y2016.terra.Terra")],
        ],
        "participants": [
            [("Caduceus", "agents.anac.y2016.caduceus.Caduceus")],
            [("YXAgent", "agents.anac.y2016.yxagent.YXAgent")],
            [("ParsCat", "agents.anac.y2016.parscat.ParsCat")],
            [("Farma", "agents.anac.y2016.farma.Farma")],
            [("MyAgent", "agents.anac.y2016.myagent.MyAgent")],
            [("Atlas32016", "agents.anac.y2016.atlas3.Atlas32016")],
            [("Ngent", "agents.anac.y2016.ngent.Ngent")],
            [("GrandmaAgent", "agents.anac.y2016.grandma.GrandmaAgent")],
            [("AgentHP2", "agents.anac.y2016.agenthp2.AgentHP2_main")],
            [("Terra", "agents.anac.y2016.terra.Terra")],
            [("AgentLight", "agents.anac.y2016.agentlight.AgentLight")],
            [("AgentSmith2016", "agents.anac.y2016.agentsmith.AgentSmith2016")],
            [("ClockworkAgent", "agents.anac.y2016.clockworkagent.ClockworkAgent")],
            [("MaxOops", "agents.anac.y2016.maxoops.MaxOops")],
            [("ParsAgent2", "agents.anac.y2016.pars2.ParsAgent2")],
            [("SYAgent", "agents.anac.y2016.syagent.SYAgent")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": True,
        "bilateral": False,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2017: {
        "winners": [
            [("PonPokoAgent", "agents.anac.y2017.ponpokoagent.PonPokoAgent")],
            [("CaduceusDC16", "agents.anac.y2017.caduceusdc16.CaduceusDC16")],
            [("Rubick", "agents.anac.y2017.rubick.Rubick")],
        ],
        "finalists": [
            [("PonPokoAgent", "agents.anac.y2017.ponpokoagent.PonPokoAgent")],
            [("CaduceusDC16", "agents.anac.y2017.caduceusdc16.CaduceusDC16")],
            [("ParsCat2", "agents.anac.y2016.parscat.ParsCat")],
            [("Rubick", "agents.anac.y2017.rubick.Rubick")],
            [("ParsAgent3", "agents.anac.y2017.parsagent3.ShahAgent")],
            [("AgentKN", "agents.anac.y2017.agentkn.AgentKN")],
            [("ParsCat2", "agents.anac.y2016.parscat.ParsCat")],
            [("AgentF", "agents.anac.y2017.agentf.AgentF")],
            [("SimpleAgent2017", "agents.anac.y2017.simpleagent.SimpleAgent")],
            [("Mamenchis", "agents.anac.y2017.mamenchis.Mamenchis")],
        ],
        "participants": [
            [("PonPokoAgent", "agents.anac.y2017.ponpokoagent.PonPokoAgent")],
            [("CaduceusDC16", "agents.anac.y2017.caduceusdc16.CaduceusDC16")],
            [("ParsCat2", "agents.anac.y2016.parscat.ParsCat")],
            [("Rubick", "agents.anac.y2017.rubick.Rubick")],
            [("ParsAgent3", "agents.anac.y2017.parsagent3.ShahAgent")],
            [("AgentKN", "agents.anac.y2017.agentkn.AgentKN")],
            [("AgentF", "agents.anac.y2017.agentf.AgentF")],
            [("SimpleAgent2017", "agents.anac.y2017.simpleagent.SimpleAgent")],
            [("Mamenchis", "agents.anac.y2017.mamenchis.Mamenchis")],
            [("Farma2017", "agents.anac.y2017.farma.Farma17")],
            [("GeneKing", "agents.anac.y2017.geneking.GeneKing")],
            [("Gin", "agents.anac.y2017.gin.Gin")],
            [("Group3", "agents.anac.y2017.group3.Group3")],
            [("Imitator", "agents.anac.y2017.limitator.Imitator")],
            [("MadAgent", "agents.anac.y2017.madagent.MadAgent")],
            [("Mosa", "agents.anac.y2017.mosateam.Mosa")],
            [("TaxiBox", "agents.anac.y2017.tangxun.taxibox")],
            [("TucAgent", "agents.anac.y2017.tucagent.TucAgent")],
        ],
        "winners_nash": [
            [("ParsCat2", "agents.anac.y2016.parscat.ParsCat")],
            [("AgentKN", "agents.anac.y2017.agentkn.AgentKN")],
            [("ParsAgent3", "agents.anac.y2017.parsagent3.ShahAgent")],
        ],
        "linear": True,
        "learning": True,
        "learning_full_exchange": False,
        "multilateral": True,
        "bilateral": False,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
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
        "finalists": [
            [("MengWan", "agents.anac.y2018.meng_wan.Agent36")],
            [("IQSun2018", "agents.anac.y2018.iqson.IQSun2018")],
            [("PonPokoRampage", "agents.anac.y2018.ponpokorampage.PonPokoRampage")],
            [("AgentHerb", "agents.anac.y2018.agentherb.AgentHerb")],
            [("FullAgent", "agents.anac.y2018.fullagent.FullAgent")],
            [("BetaOne", "agents.anac.y2018.beta_one.Group2")],
            [
                (
                    "AgreeableAgent2018",
                    "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
                )
            ],
            [("Shiboy", "agents.anac.y2018.shiboy.Shiboy")],
            [("ConDAgent", "agents.anac.y2018.condagent.ConDAgent")],
            [("Yeela", "agents.anac.y2018.yeela.Yeela")],
            [("Sontag", "agents.anac.y2018.sontag.Sontag")],
            [("Agent33", "agents.anac.y2018.agent33.Agent33")],
            [("AgentNP1", "agents.anac.y2018.agentnp1.AgentNP1")],
        ],
        "participants": [
            [("MengWan", "agents.anac.y2018.meng_wan.Agent36")],
            [("IQSun2018", "agents.anac.y2018.iqson.IQSun2018")],
            [("PonpokoRampage", "agents.anac.y2018.ponpokorampage.PonPokoRampage")],
            [("AgentHerb", "agents.anac.y2018.agentherb.AgentHerb")],
            [("FullAgent", "agents.anac.y2018.fullagent.FullAgent")],
            [("BetaOne", "agents.anac.y2018.beta_one.Group2")],
            [
                (
                    "AgreeableAgent2018",
                    "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
                )
            ],
            [("Shiboy", "agents.anac.y2018.shiboy.Shiboy")],
            [("ConDAgent", "agents.anac.y2018.condagent.ConDAgent")],
            [("Yeela", "agents.anac.y2018.yeela.Yeela")],
            [("Sontag", "agents.anac.y2018.sontag.Sontag")],
            [("Agent33", "agents.anac.y2018.agent33.Agent33")],
            [("AgentNP1", "agents.anac.y2018.agentnp1.AgentNP1")],
            [("AteamAgent", "agents.anac.y2018.ateamagent.ATeamAgent")],
            [("ExpRubick", "agents.anac.y2018.exp_rubick.Exp_Rubick")],
            [("GroupY", "agents.anac.y2018.groupy.GroupY")],
            [("Lancelot", "agents.anac.y2018.lancelot.Lancelot")],
            [("Libra", "agents.anac.y2018.libra.Libra")],
            [("Seto", "agents.anac.y2018.seto.Seto")],
            [("SMACAgent", "agents.anac.y2018.smac_agent.SMAC_Agent")],
        ],
        "winners_welfare": [
            [("AgentHerb", "agents.anac.y2018.agentherb.AgentHerb")],
            [("Agent33", "agents.anac.y2018.agent33.Agent33")],
            [("Sontag", "agents.anac.y2018.sontag.Sontag")],
        ],
        "linear": True,
        "learning": True,
        "learning_full_exchange": True,
        "multilateral": True,
        "bilateral": False,
        "reservation": True,
        "discounting": True,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2019: {
        "winners": [
            [("AgentGG", "agents.anac.y2019.agentgg.AgentGG")],
            [("KakeSoba", "agents.anac.y2019.kakesoba.KakeSoba")],
            [("SAGA", "agents.anac.y2019.saga.SAGA")],
        ],
        "winners_nash": [
            [("WinkyAgent", "agents.anac.y2019.winkyagent.winkyAgent")],
            [("FSEGA2019", "agents.anac.y2019.fsega2019.agent.FSEGA2019")],
            [("AgentGP", "agents.anac.y2019.agentgp.AgentGP")],
        ],
        "finalists": [
            [("AgentGG", "agents.anac.y2019.agentgg.AgentGG")],
            [("KakeSoba", "agents.anac.y2019.kakesoba.KakeSoba")],
            [("SAGA", "agents.anac.y2019.saga.SAGA")],
            [("WinkyAgent", "agents.anac.y2019.winkyagent.winkyAgent")],
            [("FSEGA2019", "agents.anac.y2019.fsega2019.agent.FSEGA2019")],
            [("AgentGP", "agents.anac.y2019.agentgp.AgentGP")],
        ],
        "participants": [
            [("AgentGG", "agents.anac.y2019.agentgg.AgentGG")],
            [("KakeSoba", "agents.anac.y2019.kakesoba.KakeSoba")],
            [("SAGA", "agents.anac.y2019.saga.SAGA")],
            [("WinkyAgent", "agents.anac.y2019.winkyagent.winkyAgent")],
            [("FSEGA2019", "agents.anac.y2019.fsega2019.agent.FSEGA2019")],
            [("AgentGP", "agents.anac.y2019.agentgp.AgentGP")],
            [("AgentLarry", "agents.anac.y2019.agentlarry.AgentLarry")],
            [("DandikAgent", "agents.anac.y2019.dandikagent.dandikAgent")],
            [("EAgent", "agents.anac.y2019.eagent.EAgent")],
            [("GaravelAgent", "agents.anac.y2019.garavelagent.GaravelAgent")],
            [("Gravity", "agents.anac.y2019.gravity.Gravity")],
            [("HardDealer", "agents.anac.y2019.harddealer.HardDealer")],
            [("KAgent", "agents.anac.y2019.kagent.KAgent")],
            [("MINF", "agents.anac.y2019.minf.MINF")],
            [("PodAgent", "agents.anac.y2019.podagent.Group1_BOA")],
            [("SACRA", "agents.anac.y2019.sacra.SACRA")],
            [("SolverAgent", "agents.anac.y2019.solveragent.SolverAgent")],
            [("TheNewDeal", "agents.anac.y2019.thenewdeal.TheNewDeal")],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": True,
        "discounting": False,
        "uncertainty": True,
        "elicitation": False,
        "geniusweb": False,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2020: {
        # ANAC 2020 - Finalists listed alphabetically (no official ranking)
        "winners": [],  # No official ranking published
        "finalists": [
            [
                (
                    "AgentKT",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAgentKT"),
                )
            ],
            [
                (
                    "AhBuNeAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAhBuNeAgent"),
                )
            ],
            [("Angel", _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAngel"))],
            [
                (
                    "HammingAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWHammingAgent"),
                )
            ],
            [
                (
                    "ShineAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWShineAgent"),
                )
            ],
        ],
        "participants": [
            [
                (
                    "AgentKT",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAgentKT"),
                )
            ],
            [
                (
                    "AgentP1DAMO",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAgentP1DAMO"),
                )
            ],
            [
                (
                    "AgentXX",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAgentXX"),
                )
            ],
            [
                (
                    "AhBuNeAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAhBuNeAgent"),
                )
            ],
            [
                (
                    "Anaconda",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAnaconda"),
                )
            ],
            [("Angel", _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAngel"))],
            [
                (
                    "AzarAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWAzarAgent"),
                )
            ],
            [
                (
                    "BlingBling",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWBlingBling"),
                )
            ],
            [
                (
                    "DUOAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWDUOAgent"),
                )
            ],
            [
                (
                    "ForArisa",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWForArisa"),
                )
            ],
            [
                (
                    "HammingAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWHammingAgent"),
                )
            ],
            [
                (
                    "NiceAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWNiceAgent"),
                )
            ],
            [
                (
                    "ShineAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2020.GWShineAgent"),
                )
            ],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": False,
        "reservation": True,
        "discounting": False,
        "uncertainty": True,
        "elicitation": True,
        "geniusweb": True,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2021: {
        # ANAC 2021 - Winners: AlphaBIU, MatrixAlienAgent, TripleAgent
        "winners": [
            [
                (
                    "AlphaBIU",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWAlphaBIU"),
                )
            ],
            [
                (
                    "MatrixAlienAgent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anac2021.GWMatrixAlienAgent"
                    ),
                )
            ],
            [
                (
                    "TripleAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWTripleAgent"),
                )
            ],
        ],
        "finalists": [
            [
                (
                    "AlphaBIU",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWAlphaBIU"),
                )
            ],
            [
                (
                    "MatrixAlienAgent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anac2021.GWMatrixAlienAgent"
                    ),
                )
            ],
            [
                (
                    "TripleAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWTripleAgent"),
                )
            ],
        ],
        "participants": [
            [
                (
                    "AgentFO2021",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWAgentFO2021"),
                )
            ],
            [
                (
                    "AlphaBIU",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWAlphaBIU"),
                )
            ],
            [
                (
                    "GamblerAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWGamblerAgent"),
                )
            ],
            [
                (
                    "MatrixAlienAgent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anac2021.GWMatrixAlienAgent"
                    ),
                )
            ],
            [
                (
                    "TheDiceHaggler2021",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anac2021.GWTheDiceHaggler2021"
                    ),
                )
            ],
            [
                (
                    "TripleAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anac2021.GWTripleAgent"),
                )
            ],
        ],
        "linear": True,
        "learning": True,
        "multilateral": False,
        "bilateral": False,
        "reservation": True,
        "discounting": False,
        "uncertainty": True,
        "elicitation": True,
        "geniusweb": True,
        "java": True,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2022: {
        # ANL 2022 - Winners: DreamTeam109Agent, ChargingBoul (Individual Utility)
        #                     DreamTeam109Agent, Agent007 (Social Welfare)
        "winners": [
            [
                (
                    "DreamTeam109Agent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2022.GWDreamTeam109Agent"
                    ),
                )
            ],
            [
                (
                    "ChargingBoul",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWChargingBoul"),
                )
            ],
        ],
        "winners_welfare": [
            [
                (
                    "DreamTeam109Agent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2022.GWDreamTeam109Agent"
                    ),
                )
            ],
            [
                (
                    "Agent007",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWAgent007"),
                )
            ],
        ],
        "finalists": [
            [
                (
                    "Agent007",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWAgent007"),
                )
            ],
            [
                (
                    "ChargingBoul",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWChargingBoul"),
                )
            ],
            [
                (
                    "DreamTeam109Agent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2022.GWDreamTeam109Agent"
                    ),
                )
            ],
        ],
        "participants": [
            [
                (
                    "Agent007",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWAgent007"),
                )
            ],
            [
                (
                    "Agent4410",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWAgent4410"),
                )
            ],
            [
                (
                    "AgentFish",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWAgentFish"),
                )
            ],
            [
                (
                    "AgentFO2",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWAgentFO2"),
                )
            ],
            [
                (
                    "BIU_agent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWBIUAgent"),
                )
            ],
            [
                (
                    "ChargingBoul",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWChargingBoul"),
                )
            ],
            [
                (
                    "CompromisingAgent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2022.GWCompromisingAgent"
                    ),
                )
            ],
            [
                (
                    "DreamTeam109Agent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2022.GWDreamTeam109Agent"
                    ),
                )
            ],
            [
                (
                    "GEAAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWGEAAgent"),
                )
            ],
            [
                (
                    "LearningAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWLearningAgent"),
                )
            ],
            [
                (
                    "LuckyAgent2022",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWLuckyAgent2022"),
                )
            ],
            [
                (
                    "MiCROAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWMiCROAgent"),
                )
            ],
            [
                (
                    "Pinar_Agent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWPinarAgent"),
                )
            ],
            [
                (
                    "ProcrastinAgent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2022.GWProcrastinAgent"
                    ),
                )
            ],
            [("RGAgent", _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWRGAgent"))],
            [
                (
                    "SmartAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWSmartAgent"),
                )
            ],
            [
                (
                    "SuperAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWSuperAgent"),
                )
            ],
            [
                (
                    "ThirdAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2022.GWThirdAgent"),
                )
            ],
            [
                (
                    "Tjaronchery10Agent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2022.GWTjaronchery10Agent"
                    ),
                )
            ],
        ],
        "linear": True,
        "learning": True,
        "multilateral": False,
        "bilateral": False,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": True,
        "geniusweb": True,
        "java": False,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2023: {
        # ANL 2023 - Winners: ExploitAgent, MiCRO2023 (Individual Utility)
        #                     AntHeartAgent, SmartAgent (Social Welfare)
        "winners": [
            [
                (
                    "ExploitAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWExploitAgent"),
                )
            ],
            [
                (
                    "MiCRO2023",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWMiCRO2023"),
                )
            ],
        ],
        "winners_welfare": [
            [
                (
                    "AntHeartAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWAntHeartAgent"),
                )
            ],
            [
                (
                    "SmartAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWSmartAgent"),
                )
            ],
        ],
        "finalists": [
            [
                (
                    "ExploitAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWExploitAgent"),
                )
            ],
            [
                (
                    "MiCRO2023",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWMiCRO2023"),
                )
            ],
            [
                (
                    "AntHeartAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWAntHeartAgent"),
                )
            ],
            [
                (
                    "SmartAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWSmartAgent"),
                )
            ],
        ],
        "participants": [
            [
                (
                    "AgentFO3",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWAgentFO3"),
                )
            ],
            [
                (
                    "AmbitiousAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWAmbitiousAgent"),
                )
            ],
            [
                (
                    "AntAllianceAgent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2023.GWAntAllianceAgent"
                    ),
                )
            ],
            [
                (
                    "AntHeartAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWAntHeartAgent"),
                )
            ],
            [
                (
                    "ColmanAnacondotAgent2",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2023.GWColmanAnacondotAgent2"
                    ),
                )
            ],
            [
                (
                    "ExploitAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWExploitAgent"),
                )
            ],
            [
                (
                    "GotAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWGotAgent"),
                )
            ],
            [
                (
                    "HybridAgent2023",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2023.GWHybridAgent2023"
                    ),
                )
            ],
            [
                (
                    "KBTimeDiffAgent",
                    _geniusweb_path(
                        "negmas_geniusweb_bridge.anl2023.GWKBTimeDiffAgent"
                    ),
                )
            ],
            [
                (
                    "MiCRO2023",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWMiCRO2023"),
                )
            ],
            [
                (
                    "MSCAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWMSCAgent"),
                )
            ],
            [
                (
                    "PopularAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWPopularAgent"),
                )
            ],
            [
                (
                    "SmartAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWSmartAgent"),
                )
            ],
            [
                (
                    "SpaghettiAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWSpaghettiAgent"),
                )
            ],
            [
                (
                    "TripleEAgent",
                    _geniusweb_path("negmas_geniusweb_bridge.anl2023.GWTripleEAgent"),
                )
            ],
        ],
        "linear": True,
        "learning": True,
        "multilateral": False,
        "bilateral": False,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": True,
        "geniusweb": True,
        "java": False,
        "multideal": False,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
    2024: {
        "winners": [
            [("Shochan", _anl_path("anl_agents.anl2024.takafam.Shochan"))],
            [("UOAgent", _anl_path("anl_agents.anl2024.team_moto.UOAgent"))],
            [
                (
                    "AgentRenting2024",
                    _anl_path("anl_agents.anl2024.team_renting.AgentRenting2024"),
                )
            ],
        ],
        "winners_nash": [
            [("Shochan", _anl_path("anl_agents.anl2024.takafam.Shochan"))]
        ],
        "finalists": [
            [("Shochan", _anl_path("anl_agents.anl2024.takafam.Shochan"))],
            [("UOAgent", _anl_path("anl_agents.anl2024.team_moto.UOAgent"))],
            [
                (
                    "AgentRenting2024",
                    _anl_path("anl_agents.anl2024.team_renting.AgentRenting2024"),
                )
            ],
            [("AntiAgent", _anl_path("anl_agents.anl2024.antiagents.AntiAgent"))],
            [
                (
                    "HardChaosNegotiator",
                    _anl_path("anl_agents.anl2024.team_199.HardChaosNegotiator"),
                )
            ],
            [("KosAgent", _anl_path("anl_agents.anl2024.team_205.KosAgent"))],
            [("Nayesian2", _anl_path("anl_agents.anl2024.susumu.Nayesian2"))],
            [("CARCAgent", _anl_path("anl_agents.anl2024.carc.CARCAgent"))],
            [("BidBot", _anl_path("anl_agents.anl2024.team_240.BidBot"))],
            [("AgentNyan", _anl_path("anl_agents.anl2024.team_123.AgentNyan"))],
        ],
        "participants": [
            [("Shochan", _anl_path("anl_agents.anl2024.takafam.Shochan"))],
            [("UOAgent", _anl_path("anl_agents.anl2024.team_moto.UOAgent"))],
            [
                (
                    "AgentRenting2024",
                    _anl_path("anl_agents.anl2024.team_renting.AgentRenting2024"),
                )
            ],
            [("AntiAgent", _anl_path("anl_agents.anl2024.antiagents.AntiAgent"))],
            [
                (
                    "HardChaosNegotiator",
                    _anl_path("anl_agents.anl2024.team_199.HardChaosNegotiator"),
                )
            ],
            [("KosAgent", _anl_path("anl_agents.anl2024.team_205.KosAgent"))],
            [("Nayesian2", _anl_path("anl_agents.anl2024.susumu.Nayesian2"))],
            [("CARCAgent", _anl_path("anl_agents.anl2024.carc.CARCAgent"))],
            [("BidBot", _anl_path("anl_agents.anl2024.team_240.BidBot"))],
            [("AgentNyan", _anl_path("anl_agents.anl2024.team_123.AgentNyan"))],
            [("TAKAgent", _anl_path("anl_agents.anl2024.team_232.TAKAgent"))],
            [("Ardabot", _anl_path("anl_agents.anl2024.team_234.Ardabot"))],
            [("Group6", _anl_path("anl_agents.anl2024.team_235.Group6"))],
            [("Group7", _anl_path("anl_agents.anl2024.team_236.Group7"))],
            [("Goldie", _anl_path("anl_agents.anl2024.tulsa_eulers.Goldie"))],
            [("Ilan", _anl_path("anl_agents.anl2024.team_186.Ilan"))],
            [("AgentKB", _anl_path("anl_agents.anl2024.teamkb.AgentKB"))],
            [("KatlaNirAgent", _anl_path("anl_agents.anl2024.tipsonly.KatlaNirAgent"))],
            [("group5", _anl_path("anl_agents.anl2024.team_twistin.group5"))],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": False,
        "multideal": False,
        "known_opponent_ufun": True,
        "known_opponent_reserved_value": False,
    },
    2025: {
        "winners": [
            [
                ("RUFL", _anl_path("anl_agents.anl2025.team_271.RUFL")),
                (
                    "SacAgent",
                    _anl_path("anl_agents.anl2025.university_of_tehran.SacAgent"),
                ),
            ],
            [("UfunATAgent", _anl_path("anl_agents.anl2025.team_305.UfunATAgent"))],
        ],
        "finalists": [
            [
                (
                    "SacAgent",
                    _anl_path("anl_agents.anl2025.university_of_tehran.SacAgent"),
                )
            ],
            [("ProbaBot", _anl_path("anl_agents.anl2025.team_273.ProbaBot"))],
            [("RUFL", _anl_path("anl_agents.anl2025.team_271.RUFL"))],
            [("KDY", _anl_path("anl_agents.anl2025.tema_kdy.KDY"))],
            [
                (
                    "JeemNegotiator",
                    _anl_path("anl_agents.anl2025.team_291.JeemNegotiator"),
                )
            ],
            [("Astrat3m", _anl_path("anl_agents.anl2025.chongqingagent.Astrat3m"))],
            [("A4E", _anl_path("anl_agents.anl2025.team_298.A4E"))],
            [("OzUAgent", _anl_path("anl_agents.anl2025.team_156.OzUAgent"))],
            [
                (
                    "SmartNegotiator",
                    _anl_path("anl_agents.anl2025.team_278.SmartNegotiator"),
                )
            ],
            [("CARC2025", _anl_path("anl_agents.anl2025.carc.CARC2025"))],
            [("UfunATAgent", _anl_path("anl_agents.anl2025.team_305.UfunATAgent"))],
            [("Wagent", _anl_path("anl_agents.anl2025.team_300.Wagent"))],
        ],
        "participants": [
            [
                (
                    "SacAgent",
                    _anl_path("anl_agents.anl2025.university_of_tehran.SacAgent"),
                )
            ],
            [("ProbaBot", _anl_path("anl_agents.anl2025.team_273.ProbaBot"))],
            [("RUFL", _anl_path("anl_agents.anl2025.team_271.RUFL"))],
            [("KDY", _anl_path("anl_agents.anl2025.tema_kdy.KDY"))],
            [
                (
                    "JeemNegotiator",
                    _anl_path("anl_agents.anl2025.team_291.JeemNegotiator"),
                )
            ],
            [("Astrat3m", _anl_path("anl_agents.anl2025.chongqingagent.Astrat3m"))],
            [("A4E", _anl_path("anl_agents.anl2025.team_298.A4E"))],
            [("OzUAgent", _anl_path("anl_agents.anl2025.team_156.OzUAgent"))],
            [
                (
                    "SmartNegotiator",
                    _anl_path("anl_agents.anl2025.team_278.SmartNegotiator"),
                )
            ],
            [("CARC2025", _anl_path("anl_agents.anl2025.carc.CARC2025"))],
            [("UfunATAgent", _anl_path("anl_agents.anl2025.team_305.UfunATAgent"))],
            [("Wagent", _anl_path("anl_agents.anl2025.team_300.Wagent"))],
            [("EOHAgent", _anl_path("anl_agents.anl2025.eoh.EOHAgent"))],
            [("TheMemorizer", _anl_path("anl_agents.anl2025.team_307.TheMemorizer"))],
            [("RivAgent", _anl_path("anl_agents.anl2025.natures.RivAgent"))],
            [("KAgent", _anl_path("anl_agents.anl2025.team_309.KAgent"))],
            [("StarGold15", _anl_path("anl_agents.anl2025.team_287.StarGold15"))],
        ],
        "linear": True,
        "learning": False,
        "multilateral": False,
        "bilateral": True,
        "reservation": True,
        "discounting": False,
        "uncertainty": False,
        "elicitation": False,
        "geniusweb": False,
        "java": False,
        "multideal": True,
        "known_opponent_ufun": False,
        "known_opponent_reserved_value": False,
    },
}

# Add backwards compatibility alias: genius10 -> participants
for _year_data in GENIUS_INFO.values():
    if "participants" in _year_data and "genius10" not in _year_data:
        _year_data["genius10"] = _year_data["participants"]

# Alias for GENIUS_INFO
ANAC_INFO = GENIUS_INFO


ALL_GENIUS_INVALID_NEGOTIATORS = [
    "agents.UIAgent",
    "agents.UIAgentExtended",
    "negotiator.parties.CounterOfferHumanNegotiationParty",
    "agents.FunctionalAcceptor",
]
"""Genius agents that are known not to work (most likely because they require
human interaction)"""

ALL_GENIUS_BASIC_NEGOTIATORS = [
    "agents.ImmediateAcceptor",
    "agents.SimpleAgent",
    "negotiator.parties.RandomCounterOfferNegotiationParty",
    "negotiator.parties.RandomParty",
    "negotiator.parties.RandomParty2",
    "agents.TimeDependentAgentBoulware",
    "agents.TimeDependentAgentConceder",
    "agents.TimeDependentAgentHardliner",
    "agents.TimeDependentAgentLinear",
    "negotiator.parties.BoulwareNegotiationParty",
    "negotiator.parties.ConcederNegotiationParty",
    "agents.SimilarityAgent",
    "agents.FuzzyAgent",
    "agents.OptimalBidderSimple",
    "agents.BayesianAgent",
    "agents.ABMPAgent2",
]
"""Basic negotiators defined in Genius that are not based on any ANAC
submissions."""

ALL_GENIUS_NEGOTIATORS = [
    "agents.ABMPAgent2",
    "agents.BayesianAgent",
    "agents.FunctionalAcceptor",
    "agents.FuzzyAgent",
    "agents.ImmediateAcceptor",
    "agents.OptimalBidderSimple",
    "agents.SimilarityAgent",
    "agents.SimpleAgent",
    "agents.TimeDependentAgentBoulware",
    "agents.TimeDependentAgentConceder",
    "agents.TimeDependentAgentHardliner",
    "agents.TimeDependentAgentLinear",
    "agents.UtilityBasedAcceptor",
    "negotiator.parties.BoulwareNegotiationParty",
    "negotiator.parties.ConcederNegotiationParty",
    "negotiator.parties.RandomCounterOfferNegotiationParty",
    "negotiator.parties.RandomParty",
    "negotiator.parties.RandomParty2",
    "parties.in4010.q12015.group1.Group1",
    "parties.in4010.q12015.group2.Group2",
    "parties.in4010.q12015.group3.Group3",
    "parties.in4010.q12015.group4.Group4",
    "parties.in4010.q12015.group5.Group5",
    "parties.in4010.q12015.group6.Group6",
    "parties.in4010.q12015.group7.Group7",
    "parties.in4010.q12015.group8.Group8",
    "parties.in4010.q12015.group9.Group9",
    "parties.in4010.q12015.group10.Group10",
    "parties.in4010.q12015.group11.Group11",
    "parties.in4010.q12015.group12.Group12",
    "parties.in4010.q12015.group13.Group13",
    "parties.in4010.q12015.group14.Group14",
    "parties.in4010.q12015.group15.Group15",
    "parties.in4010.q12015.group16.Group16",
    "parties.in4010.q12015.group17.Group17",
    "parties.in4010.q12015.group18.Group18",
    "parties.in4010.q12015.group19.Group19",
    "parties.in4010.q12015.group20.Group20",
    "parties.in4010.q12015.group21.Group21",
    "parties.in4010.q12015.group22.Group22",
    "agents.anac.y2010.AgentFSEGA.AgentFSEGA",
    "agents.anac.y2010.AgentK.Agent_K",
    "agents.anac.y2010.AgentSmith.AgentSmith",
    "agents.anac.y2010.Nozomi.Nozomi",
    "agents.anac.y2010.Southampton.IAMcrazyHaggler",
    "agents.anac.y2010.Southampton.IAMhaggler",
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
    "agents.anac.y2012.AgentMR.AgentMR",
    "agents.anac.y2012.BRAMAgent2.BRAMAgent2",
    "agents.anac.y2012.CUHKAgent.CUHKAgent",
    "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012",
    "agents.anac.y2012.MetaAgent.MetaAgent",
    "agents.anac.y2012.OMACagent.OMACagent",
    "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
    "agents.anac.y2013.AgentKF.AgentKF",
    "agents.anac.y2013.GAgent.AgentI",
    "agents.anac.y2013.InoxAgent.InoxAgent",
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
    "agents.anac.y2014.TUDelftGroup2.Group2Agent",
    "agents.anac.y2015.AgentHP.AgentHP",
    "agents.anac.y2015.AgentNeo.Groupn",
    "agents.anac.y2015.AgentW.AgentW",
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
    "agents.anac.y2015.pnegotiator.PNegotiator",
    "agents.anac.y2015.pokerface.PokerFace",
    "agents.anac.y2015.xianfa.XianFaAgent",
    "agents.anac.y2016.agenthp2.AgentHP2_main",
    "agents.anac.y2016.agentlight.AgentLight",
    "agents.anac.y2016.agentsmith.AgentSmith2016",
    "agents.anac.y2016.atlas3.Atlas32016",
    "agents.anac.y2016.caduceus.Caduceus",
    "agents.anac.y2016.clockworkagent.ClockworkAgent",
    "agents.anac.y2016.farma.Farma",
    "agents.anac.y2016.grandma.GrandmaAgent",
    "agents.anac.y2016.maxoops.MaxOops",
    "agents.anac.y2016.myagent.MyAgent",
    "agents.anac.y2016.ngent.Ngent",
    "agents.anac.y2016.pars2.ParsAgent2",
    "agents.anac.y2016.parscat.ParsCat",
    "agents.anac.y2016.syagent.SYAgent",
    "agents.anac.y2016.terra.Terra",
    "agents.anac.y2016.yxagent.YXAgent",
    "agents.anac.y2017.agentf.AgentF",
    "agents.anac.y2017.agentkn.AgentKN",
    "agents.anac.y2017.caduceusdc16.CaduceusDC16",
    "agents.anac.y2017.mamenchis.Mamenchis",
    "agents.anac.y2017.parsagent3.ShahAgent",
    "agents.anac.y2017.ponpokoagent.PonPokoAgent",
    "agents.anac.y2017.rubick.Rubick",
    "agents.anac.y2017.simpleagent.SimpleAgent",
    "agents.anac.y2018.agent33.Agent33",
    "agents.anac.y2018.agentherb.AgentHerb",
    "agents.anac.y2018.agentnp1.AgentNP1",
    "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
    "agents.anac.y2018.ateamagent.ATeamAgent",
    "agents.anac.y2018.beta_one.Group2",
    "agents.anac.y2018.condagent.ConDAgent",
    "agents.anac.y2018.exp_rubick.Exp_Rubick",
    "agents.anac.y2018.fullagent.FullAgent",
    "agents.anac.y2018.groupy.GroupY",
    "agents.anac.y2018.iqson.IQSun2018",
    "agents.anac.y2018.lancelot.Lancelot",
    "agents.anac.y2018.libra.Libra",
    "agents.anac.y2018.meng_wan.Agent36",
    "agents.anac.y2018.ponpokorampage.PonPokoRampage",
    "agents.anac.y2018.seto.Seto",
    "agents.anac.y2018.shiboy.Shiboy",
    "agents.anac.y2018.smac_agent.SMAC_Agent",
    "agents.anac.y2018.sontag.Sontag",
    "agents.anac.y2018.yeela.Yeela",
    "agents.anac.y2019.agentgg.AgentGG",
    "agents.anac.y2019.agentgp.AgentGP",
    "agents.anac.y2019.agentlarry.AgentLarry",
    "agents.anac.y2019.dandikagent.dandikAgent",
    "agents.anac.y2019.eagent.EAgent",
    "agents.anac.y2019.fsega2019.agent.FSEGA2019",
    "agents.anac.y2019.garavelagent.GaravelAgent",
    "agents.anac.y2019.gravity.Gravity",
    "agents.anac.y2019.harddealer.HardDealer",
    "agents.anac.y2019.kagent.KAgent",
    "agents.anac.y2019.kakesoba.KakeSoba",
    "agents.anac.y2019.minf.MINF",
    "agents.anac.y2019.podagent.Group1_BOA",
    "agents.anac.y2019.sacra.SACRA",
    "agents.anac.y2019.saga.SAGA",
    "agents.anac.y2019.solveragent.SolverAgent",
    "agents.anac.y2019.thenewdeal.TheNewDeal",
    "agents.anac.y2019.winkyagent.winkyAgent",
]
"""All Negotiators Accessible through Genius UI."""


ALL_NON_SAOP_GENIUS_NEGOTIATORS = [
    "parties.AlternatingMultipleOffers.RandomAmopParty",
    "parties.AlternatingMultipleOffers.RandomAmopPartyMajority",
    "parties.simplemediator.RandomFlippingMediator",
    "parties.simplemediator.FixedOrderFlippingMediator",
    "parties.simplemediator.HillClimber",
    "parties.simplemediator.Annealer",
    "parties.simplemediator.FeedbackParty",
    "parties.feedbackmediator.FeedbackMediator",
    "parties.FeedbackHillClimber",
]
"""All Genius negotiators for protocols other than SAOP."""

ALL_GENIUS_AMOP_NEGOTIATORS = [
    "parties.AlternatingMultipleOffers.RandomAmopParty",
    "parties.AlternatingMultipleOffers.RandomAmopPartyMajority",
]
"""All Genius negotiators for the Alternating Multiple Offers Protocol."""

ALL_GENIUS_SIMPLE_MEDIATOR_NEGOTIATORS = [
    "parties.simplemediator.RandomFlippingMediator",
    "parties.simplemediator.FixedOrderFlippingMediator",
    "parties.simplemediator.HillClimber",
    "parties.simplemediator.Annealer",
    "parties.simplemediator.FeedbackParty",
]
"""All Genius negotiators for the Simple Mediator Protocol."""

ALL_GENIUS_FEEDBACK_MEDIATOR_NEGOTIATORS = ["parties.feedbackmediator.FeedbackMediator"]
"""All Genius negotiators for the Feedback Mediator Protocol."""

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
"""Genius agents based on the Agent base-class.

These are the oldest agents
"""

PARTY_BASED_NEGOTIATORS = [
    # "agents.ai2014.group1.Group1",
    # "agents.ai2014.group10.Group10",
    # "agents.ai2014.group11.Group11",
    # "agents.ai2014.group12.Group12",
    # "agents.ai2014.group2.Group2",
    # "agents.ai2014.group3.Group3",
    # "agents.ai2014.group4.Group4",
    # "agents.ai2014.group5.Group5",
    # "agents.ai2014.group6.Group6",
    # "agents.ai2014.group7.Group7",
    # "agents.ai2014.group9.Group9",
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
    "agents.anac.y2017.gin.Gin",
    "agents.anac.y2017.group3.Group3",
    "agents.anac.y2017.limitator.Imitator",
    "agents.anac.y2017.madagent.MadAgent",
    "agents.anac.y2017.mosateam.Mosa",
    "agents.anac.y2017.tangxun.taxibox",
    "agents.anac.y2017.tucagent.TucAgent",
    "agents.anac.y2018.agent33.Agent33",
    "agents.anac.y2018.agentherb.AgentHerb",
    "agents.anac.y2018.agentnp1.AgentNP1",
    "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
    "agents.anac.y2018.ateamagent.ATeamAgent",
    "agents.anac.y2018.beta_one.Group2",
    "agents.anac.y2018.condagent.ConDAgent",
    "agents.anac.y2018.exp_rubick.Exp_Rubick",
    "agents.anac.y2018.fullagent.FullAgent",
    "agents.anac.y2018.groupy.GroupY",
    "agents.anac.y2018.iqson.IQSun2018",
    "agents.anac.y2018.lancelot.Lancelot",
    "agents.anac.y2018.libra.Libra",
    "agents.anac.y2018.meng_wan.Agent36",
    "agents.anac.y2018.ponpokorampage.PonPokoRampage",
    "agents.anac.y2018.seto.Seto",
    "agents.anac.y2018.shiboy.Shiboy",
    "agents.anac.y2018.smac_agent.SMAC_Agent",
    "agents.anac.y2018.sontag.Sontag",
    "agents.anac.y2018.yeela.Yeela",
    "agents.anac.y2019.agentgg.AgentGG",
    "agents.anac.y2019.agentgp.AgentGP",
    "agents.anac.y2019.agentlarry.AgentLarry",
    "agents.anac.y2019.dandikagent.dandikAgent",
    "agents.anac.y2019.eagent.EAgent",
    "agents.anac.y2019.fsega2019.agent.FSEGA2019",
    "agents.anac.y2019.garavelagent.GaravelAgent",
    "agents.anac.y2019.gravity.Gravity",
    "agents.anac.y2019.harddealer.HardDealer",
    "agents.anac.y2019.kagent.KAgent",
    "agents.anac.y2019.kakesoba.KakeSoba",
    "agents.anac.y2019.minf.MINF",
    "agents.anac.y2019.podagent.Group1_BOA",
    "agents.anac.y2019.sacra.SACRA",
    "agents.anac.y2019.saga.SAGA",
    "agents.anac.y2019.solveragent.SolverAgent",
    "agents.anac.y2019.thenewdeal.TheNewDeal",
    "agents.anac.y2019.winkyagent.winkyAgent",
]
"""Genius agents based on the Party base-class.

These are the newest agents
"""

TEST_FAILING_NEGOTIATORS = [
    "agents.ABMPAgent2",  # failed some but not all tests
    "agents.BayesianAgentForAuction",
    "agents.DecUtilAgent",
    "agents.FuzzyAgent",  # fails most test but not all
    "agents.OptimalBidder",
    "agents.OptimalBidderSimple",  # failes most tests but not all
    "agents.OptimalBidderU",
    "agents.QOAgent",
    "agents.RandomIncreasingUtilAgent",
    "agents.SimilarityAgent",  # failed some but not all tests
    "agents.anac.y2010.Southampton.SouthamptonAgent",
    "agents.anac.y2013.MetaAgent.MetaAgent2013",  # failed a few tests
    # "agents.anac.y2014.AgentTD.AgentTD",
    # "agents.anac.y2014.ArisawaYaki.ArisawaYaki",
    # "agents.anac.y2014.Aster.Aster",
    # "agents.anac.y2014.Atlas.Atlas",
    # "agents.anac.y2014.E2Agent.AnacSampleAgent",
    # "agents.anac.y2014.KGAgent.KGAgent",
    # "agents.anac.y2014.TUDelftGroup2.Group2Agent",
    # "agents.anac.y2014.simpleagent.SimpleAgent",
    "agents.anac.y2015.AgentHP.AgentHP",
    "agents.anac.y2015.AgentW.AgentW",
    "agents.anac.y2015.Mercury.Mercury",
    "agents.anac.y2015.Phoenix.PhoenixParty",
    "agents.anac.y2015.cuhkagent2015.CUHKAgent2015",
    "agents.anac.y2015.pnegotiator.PNegotiator",
    "agents.anac.y2015.xianfa.XianFaAgent",
    "agents.anac.y2016.agenthp2.AgentHP2_main",
    "agents.anac.y2016.agentlight.AgentLight",
    "agents.anac.y2016.agentlight.SgentLight",
    "agents.anac.y2016.grandma.GrandmaAgent",
    "agents.anac.y2016.maxoops.MaxOops",
    "agents.anac.y2016.ngent.Ngent",
    "agents.anac.y2017.agentkn.AgentKN",
    "agents.anac.y2017.caduceusdc16.CaduceusDC16",  # failed all tests
    "agents.anac.y2017.geneking.GeneKing",
    "agents.anac.y2017.mamenchis.Mamenchis",
    "agents.anac.y2017.parsagent3.ShahAgent",
    "agents.anac.y2017.rubick.Rubick",  # failed all tests
    "agents.anac.y2017.simpleagent.SimpleAgent",
    "parties.in4010.q12015.group1.Group1",
    "parties.in4010.q12015.group2.Group2",
    "parties.in4010.q12015.group3.Group3",
    "parties.in4010.q12015.group4.Group4",
    "parties.in4010.q12015.group5.Group5",
    "parties.in4010.q12015.group6.Group6",
    "parties.in4010.q12015.group7.Group7",
    "parties.in4010.q12015.group8.Group8",
    "parties.in4010.q12015.group9.Group9",
    "parties.in4010.q12015.group10.Group10",
    "parties.in4010.q12015.group11.Group11",
    "parties.in4010.q12015.group12.Group12",
    "parties.in4010.q12015.group13.Group13",
    "parties.in4010.q12015.group14.Group14",
    "parties.in4010.q12015.group15.Group15",
    "parties.in4010.q12015.group16.Group16",
    "parties.in4010.q12015.group17.Group17",
    "parties.in4010.q12015.group18.Group18",
    "parties.in4010.q12015.group19.Group19",
    "parties.in4010.q12015.group20.Group20",
    "parties.in4010.q12015.group21.Group21",
    "parties.in4010.q12015.group22.Group22",
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
] + [_ for _ in ALL_GENIUS_NEGOTIATORS if "y2018" in _]

"""Agetns taht fail simple tests making them less robust over the bridge"""

GENIUS10_NEGOTIATORS = list(
    set(
        list(
            itertools.chain(
                *(
                    list(_[1] for _ in itertools.chain(*v["genius10"]))
                    for _, v in GENIUS_INFO.items()
                )
            )
        )
    )
)
"""All agents supported by the Genius10 library"""

PARTICIPANT_NEGOTIATORS = list(
    set(
        list(
            itertools.chain(
                *(
                    list(_[1] for _ in itertools.chain(*v["participants"]))
                    for _, v in GENIUS_INFO.items()
                )
            )
        )
    )
)
"""All agents supported by the Genius10 library"""
TESTED_NEGOTIATORS = list(
    set(
        ["agents.anac.y2015.AgentX.AgentX"]
        + list(
            itertools.chain(
                *(
                    list(_[1] for _ in itertools.chain(*v["winners"]))
                    for _, v in GENIUS_INFO.items()
                    if v["multilateral"] and not v["learning"]
                )
            )
        )
    )
    - set(TEST_FAILING_NEGOTIATORS)
)
"""Some of the most tested negotiators."""

ALL_NEGOTIATORS = list(
    set(PARTY_BASED_NEGOTIATORS + AGENT_BASED_NEGOTIATORS + ALL_GENIUS_NEGOTIATORS)
)
"""All Genius Negotiators accessible from NegMAS."""

ALL_PASSING_NEGOTIATORS = list(
    set(ALL_NEGOTIATORS)
    - set(TEST_FAILING_NEGOTIATORS)
    - set(ALL_GENIUS_INVALID_NEGOTIATORS)
)
"""All negotiators that passed simple tests showing they work on the bridge."""


ALL_PASSING_NEGOTIATORS_NO_UNCERTAINTY = list(
    set(ALL_PASSING_NEGOTIATORS) - {_ for _ in ALL_PASSING_NEGOTIATORS if "2019" in _}
)

ALL_ANAC_AGENTS = [_ for _ in ALL_NEGOTIATORS if "anac" in _]
"""All agents submitted to ANAC and registered in Genius."""


def simplify_name(x: str):
    """Simplify name.

    Args:
        x: X.
    """
    for c in ("agent", "Agent", "_main"):
        if x.endswith(c):
            x = x[: -len(c)]
    return x.replace("_", "")


special_names = dict(
    AI2014Group2="agents.ai2014.group2.Group2",
    Agent33="agents.anac.y2018.agent33.Agent33",
    AgentBuyog="agents.anac.y2015.agentBuyogV2.AgentBuyogMain",
    AgentF="agents.anac.y2017.agentf.AgentF",
    AgentFSEGA="agents.anac.y2010.AgentFSEGA.AgentFSEGA",
    AgentGG="agents.anac.y2019.agentgg.AgentGG",
    AgentGP="agents.anac.y2019.agentgp.AgentGP",
    AgentH="agents.anac.y2015.agenth.AgentH",
    AgentHP2="agents.anac.y2016.agenthp2.AgentHP2_main",
    AgentHP="agents.anac.y2015.AgentHP.AgentHP",
    AgentHerb="agents.anac.y2018.agentherb.AgentHerb",
    AgentK2="agents.anac.y2011.AgentK2.Agent_K2",
    AgentK="agents.anac.y2010.AgentK.Agent_K",
    AgentKF="agents.anac.y2013.AgentKF.AgentKF",
    AgentKN="agents.anac.y2017.agentkn.AgentKN",
    AgentLG="agents.anac.y2012.AgentLG.AgentLG",
    AgentLarry="agents.anac.y2019.agentlarry.AgentLarry",
    AgentLight="agents.anac.y2016.agentlight.AgentLight",
    AgentM="agents.anac.y2014.AgentM.AgentM",
    AgentMR="agents.anac.y2012.AgentMR.AgentMR",
    AgentNP1="agents.anac.y2018.agentnp1.AgentNP1",
    AgentNeo="agents.anac.y2015.AgentNeo.Groupn",
    AgentQuest="agents.anac.y2014.AgentQuest.AgentQuest",
    AgentSmith2016="agents.anac.y2016.agentsmith.AgentSmith2016",
    AgentSmith="agents.anac.y2010.AgentSmith.AgentSmith",
    AgentTD="agents.anac.y2014.AgentTD.AgentTD",
    AgentTRP="agents.anac.y2014.AgentTRP.AgentTRP",
    AgentW="agents.anac.y2015.AgentW.AgentW",
    AgentX="agents.anac.y2015.AgentX.AgentX",
    AgentYK="agents.anac.y2014.AgentYK.AgentYK",
    AgreeableAgent2018="agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
    AresParty="agents.anac.y2015.AresParty.AresParty",
    ArisawaYaki="agents.anac.y2014.ArisawaYaki.ArisawaYaki",
    Aster="agents.anac.y2014.Aster.Aster",
    AteamAgent="agents.anac.y2018.ateamagent.ATeamAgent",
    Atlas32016="agents.anac.y2016.atlas3.Atlas32016",
    Atlas3="agents.anac.y2015.Atlas3.Atlas3",
    Atlas="agents.anac.y2014.Atlas.Atlas",
    BetaOne="agents.anac.y2018.beta_one.Group2",
    BramAgent2="agents.anac.y2012.BRAMAgent2.BRAMAgent2",
    BramAgent="agents.anac.y2011.BramAgent.BRAMAgent",
    BraveCat="agents.anac.y2014.BraveCat.BraveCat",
    CUHKAgent2015="agents.anac.y2015.cuhkagent2015.CUHKAgent2015",
    CUHKAgent="agents.anac.y2012.CUHKAgent.CUHKAgent",
    Caduceus="agents.anac.y2016.caduceus.Caduceus",
    CaduceusDC16="agents.anac.y2017.caduceusdc16.CaduceusDC16",
    ClockworkAgent="agents.anac.y2016.clockworkagent.ClockworkAgent",
    ConDAgent="agents.anac.y2018.condagent.ConDAgent",
    DandikAgent="agents.anac.y2019.dandikagent.dandikAgent",
    DoNA="agents.anac.y2014.DoNA.DoNA",
    DrageKnight="agents.anac.y2015.DrageKnight.DrageKnight",
    E2Agent="agents.anac.y2014.E2Agent.AnacSampleAgent",
    EAgent="agents.anac.y2019.eagent.EAgent",
    ExpRubick="agents.anac.y2018.exp_rubick.Exp_Rubick",
    FSEGA2019="agents.anac.y2019.fsega2019.agent.FSEGA2019",
    Farma2017="agents.anac.y2017.farma.Farma17",
    Farma="agents.anac.y2016.farma.Farma",
    Flinch="agents.anac.y2014.Flinch.Flinch",
    FullAgent="agents.anac.y2018.fullagent.FullAgent",
    GAgent="agents.anac.y2013.GAgent.AgentI",
    Gahboninho="agents.anac.y2011.Gahboninho.Gahboninho",
    Gangester="agents.anac.y2014.Gangster.Gangster",
    GaravelAgent="agents.anac.y2019.garavelagent.GaravelAgent",
    GeneKing="agents.anac.y2017.geneking.GeneKing",
    Gin="agents.anac.y2017.gin.Gin",
    GrandmaAgent="agents.anac.y2016.grandma.GrandmaAgent",
    Gravity="agents.anac.y2019.gravity.Gravity",
    Group3="agents.anac.y2017.group3.Group3",
    GroupY="agents.anac.y2018.groupy.GroupY",
    HardDealer="agents.anac.y2019.harddealer.HardDealer",
    HardHeaded="agents.anac.y2011.HardHeaded.KLH",
    IAMcrazyHaggler="agents.anac.y2010.Southampton.IAMcrazyHaggler",
    IAMhaggler2011="agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
    IAMhaggler2012="agents.anac.y2012.IAMhaggler2012.IAMhaggler2012",
    IAMhaggler="agents.anac.y2010.Southampton.IAMhaggler",
    IQSun2018="agents.anac.y2018.iqson.IQSun2018",
    Imitator="agents.anac.y2017.limitator.Imitator",
    InoxAgent="agents.anac.y2013.InoxAgent.InoxAgent",
    JonnyBlack="agents.anac.y2015.JonnyBlack.JonnyBlack",
    KAgent="agents.anac.y2019.kagent.KAgent",
    KGAgent="agents.anac.y2014.KGAgent.KGAgent",
    KakeSoba="agents.anac.y2019.kakesoba.KakeSoba",
    Kawaii="agents.anac.y2015.fairy.kawaii",
    Lancelot="agents.anac.y2018.lancelot.Lancelot",
    Libra="agents.anac.y2018.libra.Libra",
    MINF="agents.anac.y2019.minf.MINF",
    MadAgent="agents.anac.y2017.madagent.MadAgent",
    Mamenchis="agents.anac.y2017.mamenchis.Mamenchis",
    MaxOops="agents.anac.y2016.maxoops.MaxOops",
    MeanBot="agents.anac.y2015.meanBot.MeanBot",
    MengWan="agents.anac.y2018.meng_wan.Agent36",
    Mercury="agents.anac.y2015.Mercury.Mercury",
    MetaAgent2013="agents.anac.y2013.MetaAgent.MetaAgent2013",
    MetaAgent="agents.anac.y2012.MetaAgent.MetaAgent",
    Mosa="agents.anac.y2017.mosateam.Mosa",
    MyAgent="agents.anac.y2016.myagent.MyAgent",
    Ngent="agents.anac.y2016.ngent.Ngent",
    NiceTitForTat="agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat",
    Nozomi="agents.anac.y2010.Nozomi.Nozomi",
    OMACagent="agents.anac.y2012.OMACagent.OMACagent",
    PNegotiator="agents.anac.y2015.pnegotiator.PNegotiator",
    ParsAgent2="agents.anac.y2016.pars2.ParsAgent2",
    ParsAgent3="agents.anac.y2017.parsagent3.ShahAgent",
    ParsAgent="agents.anac.y2015.ParsAgent.ParsAgent",
    ParsCat2="agents.anac.y2016.parscat.ParsCat",
    ParsCat="agents.anac.y2016.parscat.ParsCat",
    PhoenixParty="agents.anac.y2015.Phoenix.PhoenixParty",
    PodAgent="agents.anac.y2019.podagent.Group1_BOA",
    PokerFace="agents.anac.y2015.pokerface.PokerFace",
    PonPokoAgent="agents.anac.y2017.ponpokoagent.PonPokoAgent",
    PonpokoRampage="agents.anac.y2018.ponpokorampage.PonPokoRampage",
    PonPokoRampage="agents.anac.y2018.ponpokorampage.PonPokoRampage",
    RandomDance="agents.anac.y2015.RandomDance.RandomDance",
    Rubick="agents.anac.y2017.rubick.Rubick",
    SACRA="agents.anac.y2019.sacra.SACRA",
    SAGA="agents.anac.y2019.saga.SAGA",
    SENGOKU="agents.anac.y2015.SENGOKU.SENGOKU",
    SMACAgent="agents.anac.y2018.smac_agent.SMAC_Agent",
    SYAgent="agents.anac.y2016.syagent.SYAgent",
    Seto="agents.anac.y2018.seto.Seto",
    Shiboy="agents.anac.y2018.shiboy.Shiboy",
    SimpaticoAgent="agents.anac.y2014.SimpaticoAgent.Simpatico",
    SimpleAgent2017="agents.anac.y2017.simpleagent.SimpleAgent",
    SlavaAgent="agents.anac.y2013.SlavaAgent.SlavaAgent",
    Sobut="agents.anac.y2014.Sobut.Sobut",
    SolverAgent="agents.anac.y2019.solveragent.SolverAgent",
    Sontag="agents.anac.y2018.sontag.Sontag",
    TMFAgent="agents.anac.y2013.TMFAgent.TMFAgent",
    TUDMixedStrategyAgent="agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent",
    TUDelftGroup2="agents.anac.y2014.TUDelftGroup2.Group2Agent",
    TaxiBox="agents.anac.y2017.tangxun.taxibox",
    Terra="agents.anac.y2016.terra.Terra",
    TheFawkes="agents.anac.y2013.TheFawkes.TheFawkes",
    TheNegotiator="agents.anac.y2011.TheNegotiator.TheNegotiator",
    TheNegotiatorReloaded="agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
    TheNewDeal="agents.anac.y2019.thenewdeal.TheNewDeal",
    TucAgent="agents.anac.y2017.tucagent.TucAgent",
    ValueModelAgent="agents.anac.y2011.ValueModelAgent.ValueModelAgent",
    WhaleAgent="agents.anac.y2014.AgentWhale.WhaleAgent",
    WinkyAgent="agents.anac.y2019.winkyagent.winkyAgent",
    XianFaAgent="agents.anac.y2015.xianfa.XianFaAgent",
    Y2015Group2="agents.anac.y2015.group2.Group2",
    YXAgent="agents.anac.y2016.yxagent.YXAgent",
    Yeela="agents.anac.y2018.yeela.Yeela",
    Yushu="agents.anac.y2010.Yushu.Yushu",
)
_names = shortest_unique_names(ALL_GENIUS_NEGOTIATORS)
_names_simpler = [simplify_name(_) for _ in _names]
_names_lower = [simplify_name(_.lower()) for _ in _names_simpler]

NAME_CLASS_MAP = dict(zip(_names_lower, ALL_GENIUS_NEGOTIATORS))
"""Mapping a short name to a Java class."""

NAME_CLASS_MAP.update(dict(zip(_names_simpler, ALL_GENIUS_NEGOTIATORS)))
NAME_CLASS_MAP.update(dict(zip(_names, ALL_GENIUS_NEGOTIATORS)))
"""Mapping a short name to a Java class."""

CLASS_NAME_MAP = dict(zip(ALL_GENIUS_NEGOTIATORS, _names))
"""Mapping a Java class to a short name matching the class name in Java."""

CLASS_NAME_MAP_SIMPLE = dict(zip(ALL_GENIUS_NEGOTIATORS, _names_lower))
"""Mapping a Java class to a short name that is short lowercase and with no 'agent' word or underscores."""

ALL_GENIUS_NEGOTIATORS_SET = set(ALL_GENIUS_NEGOTIATORS)


def get_name(
    java_class: str, allow_dot: bool = True, simple_version: bool = False
) -> str:
    """Returns the name of the agent with the given class."""
    d = CLASS_NAME_MAP_SIMPLE if simple_version else CLASS_NAME_MAP
    name = d[java_class]
    if not allow_dot:
        return name.split(".")[-1]
    return name


def get_java_class(name: str) -> str | None:
    """
    Returns the java class for the agent with this name if known otherwise
    it returns None.

    The name can be either the java class itself or the short agent name
    """
    if name in special_names:
        return special_names[name]
    java_class = NAME_CLASS_MAP.get(
        name,
        NAME_CLASS_MAP.get(
            name.lower(), name if name in ALL_GENIUS_NEGOTIATORS_SET else None
        ),
    )
    return java_class


def get_anac_agents(
    *,
    year: int | None = None,
    linear: bool | None = None,
    learning: bool | None = None,
    multilateral: bool | None = None,
    bilateral: bool | None = None,
    reservation: bool | None = None,
    discounting: bool | None = None,
    uncertainty: bool | None = None,
    elicitation: bool | None = None,
    winners_only: bool = False,
    finalists_only: bool = False,
    genius10: bool = False,
) -> list[tuple[str, str]]:
    """Get Genius agents matching some given criteria.

    Returns:
        a list of 2-item tuples giving agent name and java class name.

    Remarks:
        - For all criteria other than winners_only, and finalists_only, passing
          None means no constraints otherwise the game on the given year must
          match the criterion value (True or False) to get agents from this year.
        - This function uses a heuristic and may not be completely accurate. Use
          with caution.
    """

    def get_agents(year, d) -> set[tuple[str, str]]:
        """Get agents.

        Args:
            year: Year.
            d: D.

        Returns:
            set[tuple[str, str]]: The result.
        """
        if winners_only:
            lst = d.get("winners", [[]])
            lst = list(itertools.chain(*lst))
        elif finalists_only:
            lst = d.get("finalists", [])
            lst = list(itertools.chain(*lst))
        elif genius10:
            lst = d.get("genius10", [])
            lst = list(itertools.chain(*lst))
        else:
            lst = [_ for _ in ALL_NEGOTIATORS if str(year) in _]
            lst = [(_.split(".")[-1], _) for _ in lst]
        if genius10:
            if winners_only:
                _ = d.get("winners", [[]])
                lst = list(set(itertools.chain(*_)).intersection(set(lst)))
            elif finalists_only:
                _ = d.get("finalists", [[]])
                lst = list(set(itertools.chain(*_)).intersection(set(lst)))

        # lst = tuple(lst)
        # if winners_only or finalists_only:
        #     return set(itertools.chain(*lst))
        return set(lst)

    agents: set[tuple[str, ...]] = set()
    if year is None:
        for y in GENIUS_INFO.keys():
            agents = agents.union(
                get_anac_agents(
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
        return list(agents)  # type: ignore

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
    return [_ for _ in agents if _ and len(_) == 2 and _[-1]]


# All agents from genius 9.1.13
# "agents.ABMPAgent2",
# "agents.BayesianAgent",
# "agents.FunctionalAcceptor",
# "agents.FuzzyAgent",
# "agents.ImmediateAcceptor",
# "agents.OptimalBidderSimple",
# "agents.SimilarityAgent",
# "agents.SimpleAgent",
# "agents.TimeDependentAgentBoulware",
# "agents.TimeDependentAgentConceder",
# "agents.TimeDependentAgentHardliner",
# "agents.TimeDependentAgentLinear",
# "agents.UtilityBasedAcceptor",
# "agents.anac.y2010.AgentFSEGA.AgentFSEGA",
# "agents.anac.y2010.AgentK.Agent_K",
# "agents.anac.y2010.AgentSmith.AgentSmith",
# "agents.anac.y2010.Nozomi.Nozomi",
# "agents.anac.y2010.Southampton.IAMcrazyHaggler",
# "agents.anac.y2010.Southampton.IAMhaggler",
# "agents.anac.y2010.Yushu.Yushu",
# "agents.anac.y2011.AgentK2.Agent_K2",
# "agents.anac.y2011.BramAgent.BRAMAgent",
# "agents.anac.y2011.Gahboninho.Gahboninho",
# "agents.anac.y2011.HardHeaded.KLH",
# "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
# "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat",
# "agents.anac.y2011.TheNegotiator.TheNegotiator",
# "agents.anac.y2011.ValueModelAgent.ValueModelAgent",
# "agents.anac.y2012.AgentLG.AgentLG",
# "agents.anac.y2012.AgentMR.AgentMR",
# "agents.anac.y2012.BRAMAgent2.BRAMAgent2",
# "agents.anac.y2012.CUHKAgent.CUHKAgent",
# "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012",
# "agents.anac.y2012.MetaAgent.MetaAgent",
# "agents.anac.y2012.OMACagent.OMACagent",
# "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
# "agents.anac.y2013.AgentKF.AgentKF",
# "agents.anac.y2013.GAgent.AgentI",
# "agents.anac.y2013.InoxAgent.InoxAgent",
# "agents.anac.y2013.MetaAgent.MetaAgent2013",
# "agents.anac.y2013.SlavaAgent.SlavaAgent",
# "agents.anac.y2013.TMFAgent.TMFAgent",
# "agents.anac.y2013.TheFawkes.TheFawkes",
# "agents.anac.y2014.AgentM.AgentM",
# "agents.anac.y2014.AgentQuest.AgentQuest",
# "agents.anac.y2014.AgentTD.AgentTD",
# "agents.anac.y2014.AgentTRP.AgentTRP",
# "agents.anac.y2014.AgentWhale.WhaleAgent",
# "agents.anac.y2014.AgentYK.AgentYK",
# "agents.anac.y2014.ArisawaYaki.ArisawaYaki",
# "agents.anac.y2014.Aster.Aster",
# "agents.anac.y2014.Atlas.Atlas",
# "agents.anac.y2014.BraveCat.BraveCat",
# "agents.anac.y2014.DoNA.DoNA",
# "agents.anac.y2014.E2Agent.AnacSampleAgent",
# "agents.anac.y2014.Flinch.Flinch",
# "agents.anac.y2014.Gangster.Gangster",
# "agents.anac.y2014.KGAgent.KGAgent",
# "agents.anac.y2014.SimpaticoAgent.Simpatico",
# "agents.anac.y2014.Sobut.Sobut",
# "agents.anac.y2014.TUDelftGroup2.Group2Agent",
# "agents.anac.y2015.AgentHP.AgentHP",
# "agents.anac.y2015.AgentNeo.Groupn",
# "agents.anac.y2015.AgentW.AgentW",
# "agents.anac.y2015.AgentX.AgentX",
# "agents.anac.y2015.AresParty.AresParty",
# "agents.anac.y2015.Atlas3.Atlas3",
# "agents.anac.y2015.DrageKnight.DrageKnight",
# "agents.anac.y2015.JonnyBlack.JonnyBlack",
# "agents.anac.y2015.Mercury.Mercury",
# "agents.anac.y2015.ParsAgent.ParsAgent",
# "agents.anac.y2015.Phoenix.PhoenixParty",
# "agents.anac.y2015.RandomDance.RandomDance",
# "agents.anac.y2015.SENGOKU.SENGOKU",
# "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent",
# "agents.anac.y2015.agentBuyogV2.AgentBuyogMain",
# "agents.anac.y2015.agenth.AgentH",
# "agents.anac.y2015.cuhkagent2015.CUHKAgent2015",
# "agents.anac.y2015.fairy.kawaii",
# "agents.anac.y2015.group2.Group2",
# "agents.anac.y2015.meanBot.MeanBot",
# "agents.anac.y2015.pnegotiator.PNegotiator",
# "agents.anac.y2015.pokerface.PokerFace",
# "agents.anac.y2015.xianfa.XianFaAgent",
# "agents.anac.y2016.agenthp2.AgentHP2_main",
# "agents.anac.y2016.agentlight.AgentLight",
# "agents.anac.y2016.agentsmith.AgentSmith2016",
# "agents.anac.y2016.atlas3.Atlas32016",
# "agents.anac.y2016.caduceus.Caduceus",
# "agents.anac.y2016.clockworkagent.ClockworkAgent",
# "agents.anac.y2016.farma.Farma",
# "agents.anac.y2016.grandma.GrandmaAgent",
# "agents.anac.y2016.maxoops.MaxOops",
# "agents.anac.y2016.myagent.MyAgent",
# "agents.anac.y2016.ngent.Ngent",
# "agents.anac.y2016.pars2.ParsAgent2",
# "agents.anac.y2016.parscat.ParsCat",
# "agents.anac.y2016.syagent.SYAgent",
# "agents.anac.y2016.terra.Terra",
# "agents.anac.y2016.yxagent.YXAgent",
# "agents.anac.y2017.agentf.AgentF",
# "agents.anac.y2017.agentkn.AgentKN",
# "agents.anac.y2017.caduceusdc16.CaduceusDC16",
# "agents.anac.y2017.mamenchis.Mamenchis",
# "agents.anac.y2017.parsagent3.ShahAgent",
# "agents.anac.y2017.ponpokoagent.PonPokoAgent",
# "agents.anac.y2017.rubick.Rubick",
# "agents.anac.y2017.simpleagent.SimpleAgent",
# "agents.anac.y2018.agent33.Agent33",
# "agents.anac.y2018.agentherb.AgentHerb",
# "agents.anac.y2018.agentnp1.AgentNP1",
# "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
# "agents.anac.y2018.ateamagent.ATeamAgent",
# "agents.anac.y2018.beta_one.Group2",
# "agents.anac.y2018.condagent.ConDAgent",
# "agents.anac.y2018.exp_rubick.Exp_Rubick",
# "agents.anac.y2018.fullagent.FullAgent",
# "agents.anac.y2018.groupy.GroupY",
# "agents.anac.y2018.iqson.IQSun2018",
# "agents.anac.y2018.lancelot.Lancelot",
# "agents.anac.y2018.libra.Libra",
# "agents.anac.y2018.meng_wan.Agent36",
# "agents.anac.y2018.ponpokorampage.PonPokoRampage",
# "agents.anac.y2018.seto.Seto",
# "agents.anac.y2018.shiboy.Shiboy",
# "agents.anac.y2018.smac_agent.SMAC_Agent",
# "agents.anac.y2018.sontag.Sontag",
# "agents.anac.y2018.yeela.Yeela",
# "negotiator.parties.BoulwareNegotiationParty",
# "negotiator.parties.ConcederNegotiationParty",
# "negotiator.parties.CounterOfferHumanNegotiationParty",
# "negotiator.parties.RandomCounterOfferNegotiationParty",
# "negotiator.parties.RandomParty",
# "negotiator.parties.RandomParty2",

# Parties not used in SAO
# <partyRepItem
#     classPath="parties.AlternatingMultipleOffers.RandomAmopPartyMajority">
# </partyRepItem>
# <partyRepItem classPath="parties.simplemediator.RandomFlippingMediator">
#     <properties>
#         <property>isMediator</property>
#     </properties>
# </partyRepItem>
# <partyRepItem classPath="parties.simplemediator.FixedOrderFlippingMediator">
#     <properties>
#         <property>isMediator</property>
#     </properties>
# </partyRepItem>
# <partyRepItem classPath="parties.simplemediator.HillClimber">
# </partyRepItem>
# <partyRepItem classPath="parties.simplemediator.Annealer">
# </partyRepItem>
# <partyRepItem classPath="parties.simplemediator.FeedbackParty">
# </partyRepItem>
# <partyRepItem classPath="parties.feedbackmediator.FeedbackMediator">
#     <properties>
#         <property>isMediator</property>
#     </properties>
# </partyRepItem>
# <partyRepItem classPath="negotiator.parties.FeedbackHillClimber">
# </partyRepItem>
#
#
#
