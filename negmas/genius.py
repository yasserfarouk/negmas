"""
Genius Negotiator
An agent used to connect to GENIUS agents (ver 8.0.4) and allow them to join negotiation mechanisms

"""
import os
import math
import pathlib
import random
import socket
import subprocess
import tempfile
import time
import typing
import itertools
from typing import List, Optional, Tuple, Union

from py4j.java_gateway import CallbackServerParameters, GatewayParameters, JavaGateway
from py4j.protocol import Py4JNetworkError

from .common import AgentMechanismInterface, MechanismState
from .config import CONFIG_KEY_GENIUS_BRIDGE_JAR, NEGMAS_CONFIG
from .inout import get_domain_issues
from .negotiators import Controller
from .outcomes import Issue, ResponseType
from .sao import SAONegotiator, SAOState, SAOResponse
from .utilities import UtilityFunction, make_discounted_ufun, normalize

DEFAULT_JAVA_PORT = 25337
DEFAULT_PYTHON_PORT = 25338
DEFAULT_GENIUS_NEGOTIAOR_TIMEOUT = 180
if typing.TYPE_CHECKING:
    from .outcomes import Outcome

__all__ = [
    "AgentK",
    "Yushu",
    "Nozomi",
    "IAMhaggler",
    "AgentX",
    "YXAgent",
    "Caduceus",
    "ParsCat",
    "ParsAgent",
    "PonPokoAgent",
    "RandomDance",
    "BetaOne",
    "MengWan",
    "AgreeableAgent2018",
    "Rubick",
    "CaduceusDC16",
    "Terra",
    "AgentHP2",
    "GrandmaAgent",
    "Ngent",
    "Atlas32016",
    "MyAgent",
    "Farma",
    "PokerFace",
    "XianFaAgent",
    "PhoenixParty",
    "AgentBuyong",
    "Kawaii",
    "Atlas3",
    "AgentYK",
    "KGAgent",
    "E2Agent",
    "Group2",
    "WhaleAgent",
    "DoNA",
    "AgentM",
    "TMFAgent",
    "MetaAgent",
    "TheNegotiatorReloaded",
    "OMACagent",
    "AgentLG",
    "CUHKAgent",
    "ValueModelAgent",
    "NiceTitForTat",
    "TheNegotiator",
    "AgentK2",
    "BRAMAgent",
    "IAMhaggler2011",
    "Gahboninho",
    "HardHeaded",
    "GeniusNegotiator",  # Most abstract kind of agent
    "init_genius_bridge",
    "genius_bridge_is_running",
    "GENIUS_INFO",
]

INTERNAL_SEP, ENTRY_SEP, FIELD_SEP = "<<s=s>>", "<<y,y>>", "<<sy>>"

common_gateway: Optional[JavaGateway] = None
common_port: int = 0
java_process = None
python_port: int = 0

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
                    "agents.anac.y2013.MetaAgent.portfolio.CUHKAgent.CUHKAgent",
                )
            ],
            [("AgentLG", "agents.anac.y2013.MetaAgent.portfolio.AgentLG.AgentLG")],
            [
                (
                    "OMACagent",
                    "agents.anac.y2013.MetaAgent.portfolio.OMACagent.OMACagent",
                ),
                (
                    "TheNegotiatorReloaded",
                    "agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded.BOAagent",
                ),
            ],
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
                    "agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded.BOAagent",
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
            [("WhaleAgent", "agents.anac.y2014.AgentWhale.WhaleAgent")],
            [("Group2", "agents.ai2014.group2.Group2")],
            [("E2Agent", "agents.anac.y2014.E2Agent.AnacSampleAgent")],
            [("KGAgent", "agents.anac.y2014.KGAgent.KGAgent")],
            [("AgentYK", "agents.anac.y2014.AgentYK.AgentYK")],
            [("BraveCat", "agents.anac.y2014.BraveCat.necessaryClasses.BOAagent")],
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
            [("Kawaii", "agents.anac.y2015.fairy.kawaii")],
            [
                (
                    "AgentBuyong",
                    "agents.anac.y2016.caduceus.agents.agentBuyong.agentBuyong",
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
    2019: {
        "winners": ["AgentGG", "KakeSoba", "SAGA"],
        "linear": True,
        "learning": True,
        "multilateral": True,
        "bilateral": False,
        "reservation": None,
        "discounting": None,
        "uncertainty": True,
        "elicitation": False,
    },
    2020: {
        "winners": [],
        "linear": True,
        "learning": True,
        "multilateral": True,
        "bilateral": False,
        "reservation": None,
        "discounting": None,
        "uncertainty": True,
        "elicitation": True,
    },
}

all_agent_based_agents = [
    "agents.TestingAgent",
    "agents.SimpleANAC2013Agent",
    "agents.UIAgentExtended",
    "agents.SimilarityAgent",
    "agents.OptimalBidderU",
    "agents.TestAgent",
    "agents.SimpleTFTAgent",
    "agents.BayesianAgentForAuction",
    "agents.ABMPAgent",
    "agents.RandomIncreasingUtilAgent",
    "agents.OptimalBidder",
    "agents.SimpleAgentSavingBidHistory",
    "agents.SimpleAgent",
    "agents.UIAgent",
    "agents.DecUtilAgent",
    "agents.FuzzyAgent",
    "agents.TAgent",
    "agents.anac.y2013.AgentKF.AgentKF",
    "agents.anac.y2013.TheFawkes.TheFawkes",
    "agents.anac.y2013.MetaAgent.portfolio.BRAMAgent2.BRAMAgent2",
    "agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011.SouthamptonAgent",
    "agents.anac.y2013.MetaAgent.portfolio.OMACagent.OMACagent",
    "agents.anac.y2013.MetaAgent.portfolio.AgentLG.AgentLG",
    "agents.anac.y2013.MetaAgent.portfolio.AgentMR.AgentMR",
    "agents.anac.y2013.MetaAgent.portfolio.CUHKAgent.CUHKAgent",
    "agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded.BOAagent",
    "agents.anac.y2013.MetaAgent.MetaAgent2013",
    "agents.anac.y2013.GAgent.AgentI",
    "agents.anac.y2013.SlavaAgent.SlavaAgent",
    "agents.anac.y2013.TMFAgent.TMFAgent",
    "agents.anac.y2014.DoNA.DoNA",
    "agents.anac.y2014.DoNA.ClearDefaultStrategy",
    "agents.anac.y2014.AgentQuest.AgentQuest",
    "agents.anac.y2014.BraveCat.necessaryClasses.BOAagent",
    "agents.anac.y2014.Gangster.Gangster",
    "agents.anac.y2014.Aster.Aster",
    "agents.anac.y2014.AgentTRP.AgentTRP",
    "agents.anac.y2014.E2Agent.AnacSampleAgent",
    "agents.anac.y2014.Sobut.Sobut",
    "agents.anac.y2014.AgentYK.AgentYK",
    "agents.anac.y2014.KGAgent.KGAgent",
    "agents.anac.y2014.SimpaticoAgent.Simpatico",
    "agents.anac.y2014.Atlas.Atlas",
    "agents.anac.y2014.AgentTD.AgentTD",
    "agents.anac.y2014.ArisawaYaki.ArisawaYaki",
    "agents.anac.y2014.Flinch.Flinch",
    "agents.anac.y2014.AgentWhale.WhaleAgent",
    "agents.anac.y2014.AgentM.AgentM",
    "agents.anac.y2012.BRAMAgent2.BRAMAgent2",
    "agents.anac.y2012.IAMhaggler2012.agents2011.SouthamptonAgent",
    "agents.anac.y2012.OMACagent.OMACagent",
    "agents.anac.y2012.AgentLG.AgentLG",
    "agents.anac.y2012.MetaAgent.agents.GYRL.GYRL",
    "agents.anac.y2012.MetaAgent.agents.ShAgent.ShAgent",
    "agents.anac.y2012.MetaAgent.agents.Chameleon.Chameleon",
    "agents.anac.y2012.MetaAgent.agents.SimpleAgentNew.SimpleAgentNew",
    "agents.anac.y2012.MetaAgent.agents.LYY.LYYAgent",
    "agents.anac.y2012.MetaAgent.agents.WinnerAgent.WinnerAgent2",
    "agents.anac.y2012.MetaAgent.agents.DNAgent.DNAgent",
    "agents.anac.y2012.MetaAgent.agents.MrFriendly.MrFriendly",
    "agents.anac.y2012.MetaAgent.MetaAgent",
    "agents.anac.y2012.AgentMR.AgentMR",
    "agents.anac.y2012.CUHKAgent.CUHKAgent",
    "agents.anac.y2017.geneking.GeneKing",
    "agents.anac.y2010.AgentFSEGA.AgentFSEGA",
    "agents.anac.y2010.Yushu.Yushu",
    "agents.anac.y2010.Nozomi.Nozomi",
    "agents.anac.y2010.AgentSmith.AgentSmith",
    "agents.anac.y2010.Southampton.SouthamptonAgentNoExtras",
    "agents.anac.y2010.Southampton.SouthamptonAgentExtrasInterface",
    "agents.anac.y2010.Southampton.SouthamptonAgent",
    "agents.anac.y2010.AgentK.Agent_K",
    "agents.anac.y2011.TheNegotiator.TheNegotiator",
    "agents.anac.y2011.ValueModelAgent.ValueModelAgent",
    "agents.anac.y2011.HardHeaded.KLH",
    "agents.anac.y2011.Gahboninho.Gahboninho",
    "agents.anac.y2011.AgentK2.Agent_K2",
    "agents.anac.y2011.BramAgent.BRAMAgent",
    "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
    "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat",
    "agents.BayesianAgent",
    "agents.ABMPAgent2",
    "agents.QOAgent",
    "agents.SimpleAgt2",
]
all_party_based_agents = [
    "agents.ai2014.group12.Group12",
    "agents.ai2014.group7.Group7",
    "agents.ai2014.group9.Group9",
    "agents.ai2014.group8.Group8",
    "agents.ai2014.group1.Group1",
    "agents.ai2014.group6.Group6",
    "agents.ai2014.group11.Group11",
    "agents.ai2014.group10.Group10",
    "agents.ai2014.group3.Group3",
    "agents.ai2014.group4.Group4",
    "agents.ai2014.group5.Group5",
    "agents.ai2014.group2.Group2",
    "agents.anac.y2015.Mercury.Mercury",
    "agents.anac.y2015.RandomDance.RandomDance",
    "agents.anac.y2015.cuhkagent2015.CUHKAgent2015",
    "agents.anac.y2015.SENGOKU.SENGOKU",
    "agents.anac.y2015.DrageKnight.DrageKnight",
    "agents.anac.y2015.meanBot.MeanBot",
    "agents.anac.y2015.Phoenix.PhoenixParty",
    "agents.anac.y2015.pokerface.PokerFace",
    "agents.anac.y2015.agenth.AgentH",
    "agents.anac.y2015.ParsAgent.ParsAgent",
    "agents.anac.y2015.AgentHP.AgentHP",
    "agents.anac.y2015.JonnyBlack.JonnyBlack",
    "agents.anac.y2015.xianfa.XianFaAgent",
    "agents.anac.y2015.AresParty.AresParty",
    "agents.anac.y2015.fairy.kawaii",
    "agents.anac.y2015.Atlas3.Atlas3",
    "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent",
    "agents.anac.y2015.AgentW.AgentW",
    "agents.anac.y2015.AgentNeo.Groupn",
    "agents.anac.y2015.agentBuyogV2.AgentBuyogMain",
    "agents.anac.y2015.AgentX.AgentX",
    "agents.anac.y2015.pnegotiator.BayesLearner",
    "agents.anac.y2015.pnegotiator.PNegotiator",
    "agents.anac.y2015.group2.Group2",
    "agents.anac.y2017.agentkn.AgentKN",
    "agents.anac.y2017.tucagent.TucAgent",
    "agents.anac.y2017.mosateam.Mosa",
    "agents.anac.y2017.parscat2.ParsCat2",
    "agents.anac.y2017.geneking.GeneKing",
    "agents.anac.y2017.parsagent3.ShahAgent",
    "agents.anac.y2017.mamenchis.Mamenchis",
    "agents.anac.y2017.madagent.MadAgent",
    "agents.anac.y2017.agentf.AgentF",
    "agents.anac.y2017.farma.Farma17",
    "agents.anac.y2017.caduceusdc16.CaduceusDC16",
    "agents.anac.y2017.limitator.Imitator",
    "agents.anac.y2017.simpleagent.SimpleAgent",
    "agents.anac.y2017.rubick.Rubick",
    "agents.anac.y2017.ponpokoagent.PonPokoAgent",
    "agents.anac.y2017.group3.Group3",
    "agents.anac.y2017.gin.Gin",
    "agents.anac.y2017.tangxun.taxibox",
    "agents.anac.y2016.terra.Terra",
    "agents.anac.y2016.maxoops.MaxOops",
    "agents.anac.y2016.grandma.GrandmaAgent",
    "agents.anac.y2016.clockworkagent.ClockworkAgent",
    "agents.anac.y2016.parscat.ParsCat",
    "agents.anac.y2016.caduceus.Caduceus",
    "agents.anac.y2016.caduceus.agents.RandomDance.RandomDance",
    "agents.anac.y2016.caduceus.agents.kawaii.kawaii",
    "agents.anac.y2016.caduceus.agents.Caduceus.Caduceus",
    "agents.anac.y2016.caduceus.agents.ParsAgent.ParsAgent",
    "agents.anac.y2016.caduceus.agents.agentBuyong.agentBuyong",
    "agents.anac.y2016.caduceus.agents.Atlas3.Atlas3",
    "agents.anac.y2016.agentlight.AgentLight",
    "agents.anac.y2016.pars2.ParsAgent2",
    "agents.anac.y2016.yxagent.YXAgent",
    "agents.anac.y2016.farma.Farma",
    "agents.anac.y2016.syagent.SYAgent",
    "agents.anac.y2016.ngent.Ngent",
    "agents.anac.y2016.agenthp2.AgentHP2_main",
    "agents.anac.y2016.atlas3.Atlas32016",
    "agents.anac.y2016.agentsmith.AgentSmith2016",
    "agents.anac.y2016.myagent.MyAgent",
]

agent_based_negotiators = [
    "agents.TestingAgent",
    "agents.SimpleANAC2013Agent",
    "agents.UIAgentExtended",
    "agents.SimilarityAgent",
    "agents.OptimalBidderU",
    "agents.TestAgent",
    "agents.SimpleTFTAgent",
    "agents.BayesianAgentForAuction",
    "agents.ABMPAgent",
    "agents.RandomIncreasingUtilAgent",
    "agents.OptimalBidder",
    "agents.SimpleAgentSavingBidHistory",
    "agents.SimpleAgent",
    "agents.UIAgent",
    "agents.DecUtilAgent",
    "agents.FuzzyAgent",
    "agents.TAgent",
    "agents.anac.y2013.AgentKF.AgentKF",
    "agents.anac.y2013.TheFawkes.TheFawkes",
    "agents.anac.y2013.MetaAgent.portfolio.BRAMAgent2.BRAMAgent2",
    "agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011.SouthamptonAgent",
    "agents.anac.y2013.MetaAgent.portfolio.OMACagent.OMACagent",
    "agents.anac.y2013.MetaAgent.portfolio.AgentLG.AgentLG",
    "agents.anac.y2013.MetaAgent.portfolio.AgentMR.AgentMR",
    "agents.anac.y2013.MetaAgent.portfolio.CUHKAgent.CUHKAgent",
    "agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded.BOAagent",
    "agents.anac.y2013.MetaAgent.MetaAgent2013",
    "agents.anac.y2013.GAgent.AgentI",
    "agents.anac.y2013.SlavaAgent.SlavaAgent",
    "agents.anac.y2013.TMFAgent.TMFAgent",
    "agents.anac.y2014.DoNA.DoNA",
    "agents.anac.y2014.DoNA.ClearDefaultStrategy",
    "agents.anac.y2014.AgentQuest.AgentQuest",
    "agents.anac.y2014.BraveCat.necessaryClasses.BOAagent",
    "agents.anac.y2014.Gangster.Gangster",
    "agents.anac.y2014.Aster.Aster",
    "agents.anac.y2014.AgentTRP.AgentTRP",
    "agents.anac.y2014.E2Agent.AnacSampleAgent",
    "agents.anac.y2014.Sobut.Sobut",
    "agents.anac.y2014.AgentYK.AgentYK",
    "agents.anac.y2014.KGAgent.KGAgent",
    "agents.anac.y2014.SimpaticoAgent.Simpatico",
    "agents.anac.y2014.Atlas.Atlas",
    "agents.anac.y2014.AgentTD.AgentTD",
    "agents.anac.y2014.ArisawaYaki.ArisawaYaki",
    "agents.anac.y2014.Flinch.Flinch",
    "agents.anac.y2014.AgentWhale.WhaleAgent",
    "agents.anac.y2014.AgentM.AgentM",
    "agents.anac.y2012.BRAMAgent2.BRAMAgent2",
    "agents.anac.y2012.IAMhaggler2012.agents2011.SouthamptonAgent",
    "agents.anac.y2012.OMACagent.OMACagent",
    "agents.anac.y2012.AgentLG.AgentLG",
    "agents.anac.y2012.MetaAgent.agents.GYRL.GYRL",
    "agents.anac.y2012.MetaAgent.agents.ShAgent.ShAgent",
    "agents.anac.y2012.MetaAgent.agents.Chameleon.Chameleon",
    "agents.anac.y2012.MetaAgent.agents.SimpleAgentNew.SimpleAgentNew",
    "agents.anac.y2012.MetaAgent.agents.LYY.LYYAgent",
    "agents.anac.y2012.MetaAgent.agents.WinnerAgent.WinnerAgent2",
    "agents.anac.y2012.MetaAgent.agents.DNAgent.DNAgent",
    "agents.anac.y2012.MetaAgent.agents.MrFriendly.MrFriendly",
    "agents.anac.y2012.MetaAgent.MetaAgent",
    "agents.anac.y2012.AgentMR.AgentMR",
    "agents.anac.y2012.CUHKAgent.CUHKAgent",
    "agents.anac.y2017.geneking.GeneKing",
    "agents.anac.y2010.AgentFSEGA.AgentFSEGA",
    "agents.anac.y2010.Yushu.Yushu",
    "agents.anac.y2010.Nozomi.Nozomi",
    "agents.anac.y2010.AgentSmith.AgentSmith",
    "agents.anac.y2010.Southampton.SouthamptonAgentNoExtras",
    "agents.anac.y2010.Southampton.SouthamptonAgentExtrasInterface",
    "agents.anac.y2010.Southampton.SouthamptonAgent",
    "agents.anac.y2010.AgentK.Agent_K",
    "agents.anac.y2011.TheNegotiator.TheNegotiator",
    "agents.anac.y2011.ValueModelAgent.ValueModelAgent",
    "agents.anac.y2011.HardHeaded.KLH",
    "agents.anac.y2011.Gahboninho.Gahboninho",
    "agents.anac.y2011.AgentK2.Agent_K2",
    "agents.anac.y2011.BramAgent.BRAMAgent",
    "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
    "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat",
    "agents.BayesianAgent",
    "agents.ABMPAgent2",
    "agents.QOAgent",
    "agents.SimpleAgt2",
]
party_based_negotiators = [
    "agents.ai2014.group12.Group12",
    "agents.ai2014.group7.Group7",
    "agents.ai2014.group9.Group9",
    "agents.ai2014.group1.Group1",
    "agents.ai2014.group6.Group6",
    "agents.ai2014.group11.Group11",
    "agents.ai2014.group10.Group10",
    "agents.ai2014.group3.Group3",
    "agents.ai2014.group4.Group4",
    "agents.ai2014.group5.Group5",
    "agents.ai2014.group2.Group2",
    "agents.anac.y2015.Mercury.Mercury",
    "agents.anac.y2015.RandomDance.RandomDance",
    "agents.anac.y2015.cuhkagent2015.CUHKAgent2015",
    "agents.anac.y2015.SENGOKU.SENGOKU",
    "agents.anac.y2015.DrageKnight.DrageKnight",
    "agents.anac.y2015.meanBot.MeanBot",
    "agents.anac.y2015.Phoenix.PhoenixParty",
    "agents.anac.y2015.pokerface.PokerFace",
    "agents.anac.y2015.agenth.AgentH",
    "agents.anac.y2015.ParsAgent.ParsAgent",
    "agents.anac.y2015.JonnyBlack.JonnyBlack",
    "agents.anac.y2015.AresParty.AresParty",
    "agents.anac.y2015.fairy.kawaii",
    "agents.anac.y2015.Atlas3.Atlas3",
    "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent",
    "agents.anac.y2015.AgentNeo.Groupn",
    "agents.anac.y2015.agentBuyogV2.AgentBuyogMain",
    "agents.anac.y2015.AgentX.AgentX",
    "agents.anac.y2015.group2.Group2",
    "agents.anac.y2017.tucagent.TucAgent",
    "agents.anac.y2017.agentf.AgentF",
    "agents.anac.y2017.farma.Farma17",
    "agents.anac.y2017.caduceusdc16.CaduceusDC16",
    "agents.anac.y2017.ponpokoagent.PonPokoAgent",
    "agents.anac.y2017.tangxun.taxibox",
    "agents.anac.y2016.terra.Terra",
    "agents.anac.y2016.grandma.GrandmaAgent",
    "agents.anac.y2016.clockworkagent.ClockworkAgent",
    "agents.anac.y2016.parscat.ParsCat",
    "agents.anac.y2016.caduceus.Caduceus",
    "agents.anac.y2016.caduceus.agents.RandomDance.RandomDance",
    "agents.anac.y2016.caduceus.agents.kawaii.kawaii",
    "agents.anac.y2016.caduceus.agents.Caduceus.Caduceus",
    "agents.anac.y2016.caduceus.agents.ParsAgent.ParsAgent",
    "agents.anac.y2016.caduceus.agents.agentBuyong.agentBuyong",
    "agents.anac.y2016.caduceus.agents.Atlas3.Atlas3",
    "agents.anac.y2016.pars2.ParsAgent2",
    "agents.anac.y2016.yxagent.YXAgent",
    "agents.anac.y2016.farma.Farma",
    "agents.anac.y2016.syagent.SYAgent",
    "agents.anac.y2016.ngent.Ngent",
    "agents.anac.y2016.atlas3.Atlas32016",
    "agents.anac.y2016.agentsmith.AgentSmith2016",
    "agents.anac.y2016.myagent.MyAgent",
]
tested_negotiators = ["agents.anac.y2015.AgentX.AgentX",] + list(
    itertools.chain(
        *[
            list(_[1] for _ in itertools.chain(*v["winners"]))
            for year, v in GENIUS_INFO.items()
            if v["multilateral"] and not v["learning"]
        ]
    )
)


def init_genius_bridge(
    path: str = None,
    port: int = 0,
    force: bool = False,
    debug: bool = False,
    timeout: float = 0,
) -> bool:
    """Initializes a genius connection

    Args:
        path: The path to a JAR file that runs negloader
        port: port number to use
        force: Force trial even if an existing bridge is initialized
        debug: If true, passes --debug to the bridge
        timeout: If positive and nonzero, passes it as the global timeout for the bridge. Note that
                 currently, the bridge supports only integer timeout values and the fraction will be
                 truncated.

    Returns:
        True if successful

    """
    global common_gateway
    global common_port
    global java_process
    global python_port

    port = port if port > 0 else DEFAULT_JAVA_PORT
    if genius_bridge_is_running(port):
        return True
    if not force and common_gateway is not None and common_port == port:
        print("Java already initialized")
        return True

    path = NEGMAS_CONFIG.get(CONFIG_KEY_GENIUS_BRIDGE_JAR, None)
    if path is None:
        print(
            "Cannot find the path to genius bridge jar. Download the jar somewhere in your machine and add its path"
            'to ~/negmas/config.json under the key "genius_bridge_jar".\n\nFor example, if you downloaded the jar'
            " to /path/to/your/jar then edit ~/negmas/config.json to read something like\n\n"
            '{\n\t"genius_bridge_jar": "/path/to/your/jar",\n\t.... rest of the config\n}\n\n'
            "You can find the jar at http://www.yasserm.com/scml/genius-8.0.4-bridge.jar"
        )
        return False
    path = pathlib.Path(path).expanduser().absolute()
    if debug:
        params = " --debug"
    else:
        params = ""
    if timeout >= 0:
        params += f" --timeout={int(timeout)}"

    try:
        subprocess.Popen(  # ['java', '-jar',  path, '--die-on-exit', f'{port}']
            f"java -jar {path} --die-on-exit {params} {port}", shell=True
        )
    except (OSError, TimeoutError, RuntimeError, ValueError):
        return False
    time.sleep(0.5)
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=port),
        callback_server_parameters=CallbackServerParameters(port=0),
    )
    python_port = gateway.get_callback_server().get_listening_port()
    gateway.java_gateway_server.resetCallbackClient(
        gateway.java_gateway_server.getCallbackClient().getAddress(), python_port
    )

    common_gateway = gateway
    common_port = port
    return True


class GeniusNegotiator(SAONegotiator):
    """Encapsulates a Genius Negotiator"""

    def __init__(
        self,
        java_class_name: str,
        assume_normalized=True,
        ufun: Optional[UtilityFunction] = None,
        rational_proposal=False,
        parent: Controller = None,
        port: int = None,
        domain_file_name: Union[str, pathlib.Path] = None,
        utility_file_name: Union[str, pathlib.Path] = None,
        keep_issue_names: bool = True,
        keep_value_names: bool = True,
        auto_load_java: bool = False,
        normalize_utility: bool = False,
        normalize_max_only: bool = False,
        can_propose=True,
        genius_bridge_path: str = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.__destroyed = False
        self.__started = False
        self.capabilities["propose"] = can_propose
        self.add_capabilities({"genius": True})
        self.java = None
        self.java_class_name = java_class_name
        self.port = port
        self._normalize_utility = normalize_utility
        self._normalize_max_only = normalize_max_only
        self.connected = self._connect(
            path=genius_bridge_path, port=self.port, auto_load_java=auto_load_java
        )
        self.java_uuid = self._create()
        self.uuid = self.java_uuid
        self.name = self.java_uuid
        self.domain_file_name = str(domain_file_name) if domain_file_name else None
        self.utility_file_name = str(utility_file_name) if utility_file_name else None
        self._my_last_offer = None
        self.keep_issue_names = keep_issue_names
        self._utility_function, self.discount = None, None
        self.issue_names = self.issues = self.issue_index = None
        if domain_file_name is not None:
            # we keep original issues details so that we can create appropriate answers to Java
            self.issues = get_domain_issues(
                domain_file_name=domain_file_name,
                keep_issue_names=True,
                keep_value_names=True,
            )
            self.issue_names = [_.name for _ in self.issues]
            self.issue_index = dict(zip(self.issue_names, range(len(self.issue_names))))
        self.discount = None
        if utility_file_name is not None:
            self._utility_function, self.discount = UtilityFunction.from_genius(
                utility_file_name,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
            )
        # if ufun is not None:
        #     self._utility_function = ufun
        self.base_utility = self._utility_function
        self.__ufun_received = ufun
        self._temp_domain_file = self._temp_ufun_file = False

    @classmethod
    def robust_negotiators(cls) -> List[str]:
        """
        Returns a list of genius agents that were tested and seem to be robustly working with negmas
        """
        return tested_negotiators

    @classmethod
    def negotiators(cls, agent_based=True, party_based=True) -> List[str]:
        """
        Returns a list of all available agents in genius 8.4.0

        Args:
            agent_based: Old agents based on the Java class Negotiator
            party_based: Newer agents based on the Java class AbstractNegotiationParty

        Returns:

        """
        r = []
        if agent_based:
            r += agent_based_negotiators
        if party_based:
            r += party_based_negotiators
        return r

    @classmethod
    def random_negotiator(
        cls,
        agent_based=True,
        party_based=True,
        port: int = None,
        domain_file_name: str = None,
        utility_file_name: str = None,
        keep_issue_names: bool = True,
        keep_value_names: bool = True,
        auto_load_java: bool = False,
        can_propose=True,
        name: str = None,
    ) -> "GeniusNegotiator":
        """
        Returns an agent with a random class name

        Args:
            name: negotiator name
            can_propose: Can this negotiator propose?
            auto_load_java: load the JVM if needed
            keep_value_names: Keep value names if values are strings
            keep_issue_names: Use dictionaries instead of tuples for representing outcomes
            utility_file_name: Name of the utility xml file
            domain_file_name: Name of the domain XML file
            port: port number to use if the JVM is to be started
            agent_based: Old agents based on the Java class Negotiator
            party_based: Newer agents based on the Java class AbstractNegotiationParty


        Returns:
            GeniusNegotiator an agent with a random java class
        """
        agent_names = cls.negotiators(agent_based=agent_based, party_based=party_based)
        agent_name = agent_names[random.randint(0, len(agent_names) - 1)]
        return GeniusNegotiator(
            java_class_name=agent_name,
            port=port,
            domain_file_name=domain_file_name,
            utility_file_name=utility_file_name,
            keep_issue_names=keep_issue_names,
            keep_value_names=keep_value_names,
            auto_load_java=auto_load_java,
            can_propose=can_propose,
            name=name,
        )

    @property
    def is_connected(self):
        return self.connected and self.java is not None

    def _create(self):
        """
        Creates the agent
        Returns:

        Examples:
            >>> if genius_bridge_is_running():
            ...     a = GeniusNegotiator(java_class_name="agents.anac.y2015.Atlas3.Atlas3")
            ...     a.java_uuid.startswith("agents.anac.y2015.Atlas3.Atlas3")
            ... else:
            ...     True
            True
            >>> if genius_bridge_is_running():
            ...     len(a.java_uuid)- len(a.java_class_name) == 36 # length of UUID
            ... else:
            ...     True
            True
            >>> if genius_bridge_is_running():
            ...     a.destroy_java_counterpart()

        """
        aid = self.java.create_agent(self.java_class_name)
        if not aid:
            raise ValueError(f"Cannot initialized {self.java_class_name}")
        return aid

    def _connect(self, path: str, port: int, auto_load_java: bool = False) -> bool:
        """

        Returns:

        Examples:

            - Testing multilateral agent
            >>> if genius_bridge_is_running():
            ...     a = GeniusNegotiator(java_class_name="agents.anac.y2015.Atlas3.Atlas3")
            ...     print(a.java_name)
            ... else:
            ...     print('ANAC2015-6-Atlas')
            ANAC2015-6-Atlas
            >>> if genius_bridge_is_running():
            ...     a.destroy_java_counterpart()

            - Testing bilateral agent
            >>> if genius_bridge_is_running():
            ...    b = GeniusNegotiator(java_class_name="agents.SimpleAgent")
            ...    print(b.java_name)
            ... else:
            ...    print('Agent SimpleAgent')
            Agent SimpleAgent
            >>> if genius_bridge_is_running():
            ...     a.destroy_java_counterpart()

        """
        if port is None:
            if auto_load_java:
                if common_gateway is None:
                    init_genius_bridge(path=path)
                gateway = common_gateway
                self.java = gateway.entry_point
                port = DEFAULT_JAVA_PORT
                return True
            else:
                port = DEFAULT_JAVA_PORT
        gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, auto_close=True)
        )
        if gateway is None:
            self.java = None
            return False
        self.java = gateway.entry_point
        return True

    @property
    def java_name(self):
        if not self.java:
            return None
        return self.java.getName(self.java_uuid)

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        if ufun is None:
            ufun = self.__ufun_received
        result = super().join(ami=ami, state=state, ufun=ufun, role=role)
        if self._normalize_utility:
            self._utility_function = normalize(
                self.ufun,
                outcomes=ami.discrete_outcomes,
                max_only=self._normalize_max_only,
            )
        self.issue_names = [_.name for _ in ami.issues]
        self.issues = ami.issues
        self.issue_index = dict(zip(self.issue_names, range(len(self.issue_names))))
        self.keep_issue_names = self.keep_value_names = ami.outcome_type == dict
        if result and ami.issues is not None and self.domain_file_name is None:
            domain_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.domain_file_name = domain_file.name
            domain_file.write(Issue.to_xml_str(ami.issues))
            domain_file.close()
            self._temp_domain_file = True
        if result and ufun is not None and self.utility_file_name is None:
            utility_file = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
            self.utility_file_name = utility_file.name
            utility_file.write(
                UtilityFunction.to_xml_str(
                    ufun, issues=ami.issues, discount_factor=self.discount
                )
            )
            utility_file.close()
            self._temp_ufun_file = True
        return result

    def test(self) -> str:
        return self.java.test(self.java_class_name)

    def destroy_java_counterpart(self, state=None) -> None:
        if self.__started and not self.__destroyed:
            if self.java is not None:
                self.java.on_negotiation_end(
                    self.java_uuid,
                    None
                    if state is None
                    else self._outcome2str(state.agreement)
                    if state.agreement is not None
                    else None,
                )
                self.java.destroy_agent(self.java_uuid)
            self.__destroyed = True
        # print(self.utility_file_name)
        # print(self.domain_file_name)
        # return
        if self._temp_ufun_file:
            try:
                os.unlink(self.utility_file_name)
            except (FileNotFoundError, PermissionError):
                pass
            self._temp_ufun_file = False

        if self._temp_domain_file:
            try:
                os.unlink(self.domain_file_name)
            except (FileNotFoundError, PermissionError):
                pass
            self._temp_domain_file = False

    def on_negotiation_end(self, state: MechanismState) -> None:
        """called when a negotiation is ended"""
        super().on_negotiation_end(state)
        self.destroy_java_counterpart(state)

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Called when the info starts. Connects to the JVM."""
        super().on_negotiation_start(state=state)
        info = self._ami
        if self.discount is not None and self.discount != 1.0:
            self._utility_function = make_discounted_ufun(
                self._utility_function,
                ami=info,
                discount_per_round=self.discount,
                power_per_round=1.0,
            )
        n_steps = -1 if info.n_steps is None else int(info.n_steps)  # number of steps
        n_seconds = (
            -1
            if info.time_limit is None or math.isinf(info.time_limit)
            else int(info.time_limit)
        )  # time limit
        timeout = (
            info.negotiator_time_limit
            if info.negotiator_time_limit and not math.isinf(info.negotiator_time_limit)
            else info.step_time_limit
            if info.step_time_limit and not math.isinf(info.step_time_limit)
            else info.time_limit
            if info.time_limit and not math.isinf(info.time_limit)
            else DEFAULT_GENIUS_NEGOTIAOR_TIMEOUT
        )
        if timeout is None or math.isinf(timeout) or timeout <= 0:
            timeout = DEFAULT_GENIUS_NEGOTIAOR_TIMEOUT

        if n_steps * n_seconds > 0:
            # n_seconds take precedence
            n_steps = -1
        try:
            self.java.on_negotiation_start(
                self.java_uuid,  # java_uuid
                info.n_negotiators,  # number of agents
                n_steps,
                n_seconds,
                n_seconds > 0,
                self.domain_file_name,  # domain file
                self.utility_file_name,  # Negotiator file
                timeout,
            )
            self.__started = True
        except Exception as e:
            raise ValueError(f"Cannot start negotiation: {str(e)}")

    def cancel(self, reason=None) -> None:
        try:
            self.java.cancel(self.java_uuid)
        except:
            pass

    def counter(self, state: MechanismState, offer: Optional["Outcome"]):
        if offer is not None:
            try:
                self.java.receive_message(
                    self.java_uuid,
                    state.current_proposer,
                    "Offer",
                    self._outcome2str(offer),
                )
            except Exception as e:
                self.destroy_java_counterpart(self.ami.state)
                raise e
        try:
            response, outcome = self.parse(self.java.choose_action(self.java_uuid))
        except Exception as e:
            # self.destroy_java_counterpart(self.ami.state)
            raise e
        self._my_last_offer = outcome
        return SAOResponse(response, outcome)

    def propose(self, state):
        raise ValueError(f"propose should never be called directly on GeniusNegotiator")

    def parse(self, action: str) -> Tuple[Optional[ResponseType], Optional["Outcome"]]:
        """
        Parses an action into a ResponseType and an Outcome (if one is included)
        Args:
            action:

        Returns:

        """
        response, outcome = None, None
        if len(action) < 1:
            return ResponseType.REJECT_OFFER, None
        id, typ_, bid_str = action.split(FIELD_SEP)
        issues = self._ami.issues

        if typ_ in ("Offer",) and (bid_str is not None and len(bid_str) > 0):
            try:
                if self._ami.outcome_type == tuple:
                    outcome = tuple(
                        _.split(INTERNAL_SEP)[1] for _ in bid_str.split(ENTRY_SEP)
                    )
                    outcome = tuple(
                        issue.value_type(v)
                        for i, (issue, v) in enumerate(zip(issues, outcome))
                    )
                else:
                    outcome = {
                        _[0]: _[1]
                        for _ in [
                            _.split(INTERNAL_SEP) for _ in bid_str.split(ENTRY_SEP)
                        ]
                    }
                    issue_map = {i.name: i for i in issues}
                    for k, v in outcome.items():
                        outcome[k] = issue_map[k].value_type(v)
                    if self._ami.outcome_type != dict:
                        outcome = self._ami.outcome_type(
                            **{
                                _[0]: _[1]
                                for _ in [
                                    _.split(INTERNAL_SEP)
                                    for _ in bid_str.split(ENTRY_SEP)
                                ]
                            }
                        )
            except Exception as e:
                print(
                    f"Failed for bid string: {bid_str} of action {action} with exception {str(e)}"
                )

        if typ_ == "Offer":
            response = ResponseType.REJECT_OFFER
        elif typ_ == "Accept":
            response = ResponseType.ACCEPT_OFFER
        elif typ_ == "EndNegotiation":
            response = ResponseType.END_NEGOTIATION
        else:
            raise ValueError(f"Unknown response: {typ_} in action {action}")
        return response, outcome

    def _outcome2str(self, outcome):
        output = ""
        if not isinstance(outcome, dict):
            outcome_dict = dict(zip(self.issue_names, outcome))
        else:
            outcome_dict = outcome
        for i, v in outcome_dict.items():
            output += f"{i}{INTERNAL_SEP}{v}{ENTRY_SEP}"
        output = output[: -len(ENTRY_SEP)]
        return output

    def __str__(self):
        name = super().__str__().split("/")
        return "/".join(name[:-1]) + f"/{self.java_class_name}/" + name[-1]


def genius_bridge_is_running(port: int = None) -> bool:
    """
    Checks whether a Genius Bridge is running. A genius bridge allows you to use `GeniusNegotiator` objects.

    Remarks:

        You can start a Genius Bridge in at least two ways:

        - execute the python function `init_genius_bridge()` in this module
        - run "negmas genius" on the terminal

    """
    if port is None:
        port = DEFAULT_JAVA_PORT
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", port))
        return True
    except ConnectionRefusedError:
        return False
    except IndexError:
        return False
    except Py4JNetworkError:
        return False
    finally:
        s.close()


class Caduceus(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.caduceus.Caduceus"
        super().__init__(**kwargs)


class YXAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.yxagent.YXAgent"
        super().__init__(**kwargs)


class AgentX(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.AgentX.AgentX"
        super().__init__(**kwargs)


class ParsCat(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.parscat.ParsCat"
        super().__init__(**kwargs)


class PonPokoAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.ponpokoagent.PonPokoAgent"
        super().__init__(**kwargs)


class HardHeaded(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.HardHeaded.KLH"
        super().__init__(**kwargs)


class Gahboninho(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.Gahboninho.Gahboninho"
        super().__init__(**kwargs)


class IAMhaggler2011(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011"
        super().__init__(**kwargs)


class BRAMAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.BramAgent.BRAMAgent"
        super().__init__(**kwargs)


class AgentK2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.AgentK2.Agent_K2"
        super().__init__(**kwargs)


class TheNegotiator(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.TheNegotiator.TheNegotiator"
        super().__init__(**kwargs)


class NiceTitForTat(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat"
        super().__init__(**kwargs)


class ValueModelAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.ValueModelAgent.ValueModelAgent"
        super().__init__(**kwargs)


class CUHKAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2013.MetaAgent.portfolio.CUHKAgent.CUHKAgent"
        super().__init__(**kwargs)


class AgentLG(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2013.MetaAgent.portfolio.AgentLG.AgentLG"
        super().__init__(**kwargs)


class OMACagent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2013.MetaAgent.portfolio.OMACagent.OMACagent"
        super().__init__(**kwargs)


class TheNegotiatorReloaded(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded.BOAagent"
        super().__init__(**kwargs)


class MetaAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded.BOAagent"
        super().__init__(**kwargs)


class TMFAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.TMFAgent.TMFAgent"
        super().__init__(**kwargs)


class AgentM(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentM.AgentM"
        super().__init__(**kwargs)


class DoNA(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.DoNA.DoNA"
        super().__init__(**kwargs)


class WhaleAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentWhale.WhaleAgent"
        super().__init__(**kwargs)


class Group2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.ai2014.group2.Group2"
        super().__init__(**kwargs)


class E2Agent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.E2Agent.AnacSampleAgent"
        super().__init__(**kwargs)


class KGAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.KGAgent.KGAgent"
        super().__init__(**kwargs)


class AgentYK(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentYK.AgentYK"
        super().__init__(**kwargs)


class BraveCat(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2014.BraveCat.necessaryClasses.BOAagent"
        super().__init__(**kwargs)


class Atlas3(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.Atlas3.Atlas3"
        super().__init__(**kwargs)


class ParsAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.ParsAgent.ParsAgent"
        super().__init__(**kwargs)


class RandomDance(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.RandomDance.RandomDance"
        super().__init__(**kwargs)


class Kawaii(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.fairy.kawaii"
        super().__init__(**kwargs)


class AgentBuyong(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2016.caduceus.agents.agentBuyong.agentBuyong"
        super().__init__(**kwargs)


class PhoenixParty(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.Phoenix.PhoenixParty"
        super().__init__(**kwargs)


class XianFaAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.xianfa.XianFaAgent"
        super().__init__(**kwargs)


class PokerFace(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.pokerface.PokerFace"
        super().__init__(**kwargs)


class Farma(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.farma.Farma17"
        super().__init__(**kwargs)


class MyAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.myagent.MyAgent"
        super().__init__(**kwargs)


class Atlas32016(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.atlas3.Atlas32016"
        super().__init__(**kwargs)


class Ngent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.ngent.Ngent"
        super().__init__(**kwargs)


class GrandmaAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.grandma.GrandmaAgent"
        super().__init__(**kwargs)


class AgentHP2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.agenthp2.AgentHP2_main"
        super().__init__(**kwargs)


class Terra(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.terra.Terra"
        super().__init__(**kwargs)


class CaduceusDC16(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.caduceusdc16.CaduceusDC16"
        super().__init__(**kwargs)


class Rubick(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.rubick.Rubick"
        super().__init__(**kwargs)


class AgreeableAgent2018(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018"
        super().__init__(**kwargs)


class MengWan(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.meng_wan.Agent36"
        super().__init__(**kwargs)


class BetaOne(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.beta_one.Group2"
        super().__init__(**kwargs)


class AgentK(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.AgentK.Agent_K"
        super().__init__(**kwargs)


class Yushu(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.Yushu.Yushu"
        super().__init__(**kwargs)


class Nozomi(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.Nozomi.Nozomi"
        super().__init__(**kwargs)


class IAMhaggler(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.IAMhaggler"
        super().__init__(**kwargs)
