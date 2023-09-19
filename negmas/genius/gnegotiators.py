from __future__ import annotations

from .negotiator import GeniusNegotiator

__all__ = [
    "AgentBuyog",
    "AgentF",
    "AgentFSEGA",
    "AgentGG",
    "AgentGP",
    "AgentH",
    "AgentHP",
    "AgentHP2",
    "AgentHerb",
    "AgentI",
    "AgentK",
    "AgentK2",
    "AgentKF",
    "AgentKN",
    "AgentLG",
    "AgentLarry",
    "AgentLight",
    "AgentM",
    "AgentMR",
    "AgentNP1",
    "AgentQuest",
    "AgentSmith",
    "AgentSmith2016",
    "AgentTD",
    "AgentTRP",
    "Agent33",
    "Agent36",
    "ABMPAgent2",
    "AgentW",
    "AgentX",
    "AgentYK",
    "AgreeableAgent2018",
    "AnacSampleAgent",
    "AresParty",
    "ArisawaYaki",
    "Aster",
    "AteamAgent",
    "Atlas",
    "Atlas3",
    "Atlas32016",
    "BramAgent",
    "BramAgent2",
    "BayesianAgent",
    "BetaOne",
    "Betaone",
    "BoulwareNegotiationParty",
    "BraveCat",
    "CUHKAgent",
    "CUHKAgent2015",
    "Caduceus",
    "CaduceusDC16",
    "ClockworkAgent",
    "ConDAgent",
    "ConcederNegotiationParty",
    "DandikAgent",
    "DoNA",
    "DrageKnight",
    "E2Agent",
    "EAgent",
    "AI2014Group2",
    "ExpRubick",
    "FSEGA2019",
    "Farma",
    "Farma17",
    "Farma2017",
    "Flinch",
    "FullAgent",
    "FunctionalAcceptor",
    "FuzzyAgent",
    "GAgent",
    "Gahboninho",
    "Gangester",
    "Gangster",
    "GeneKing",
    "GaravelAgent",
    "Gin",
    "GrandmaAgent",
    "Gravity",
    "Group1",
    "Group10",
    "Group11",
    "Group12",
    "Group13",
    "Group14",
    "Group15",
    "Group16",
    "Group17",
    "Group18",
    "Group19",
    "Group1BOA",
    "PodAgent",
    "TUDelftGroup2",
    "Group20",
    "Group21",
    "Group22",
    "Group3Q2015",
    "Group3",
    "Group4",
    "Group5",
    "Group6",
    "Group7",
    "Group8",
    "Group9",
    "GroupY",
    "AgentNeo",
    "HardDealer",
    "HardHeaded",
    "IAMcrazyHaggler",
    "IAMhaggler",
    "IAMhaggler2011",
    "IAMhaggler2012",
    "IQSun2018",
    "Imitator",
    "ImmediateAcceptor",
    "InoxAgent",
    "JonnyBlack",
    "KAgent",
    "KGAgent",
    "KakeSoba",
    "Kawaii",
    "Lancelot",
    "Libra",
    "MINF",
    "MadAgent",
    "Mamenchis",
    "MaxOops",
    "MeanBot",
    "MengWan",
    "Mercury",
    "MetaAgent",
    "MetaAgent2012",
    "MetaAgent2013",
    "Mosa",
    "MyAgent",
    "Ngent",
    "NiceTitForTat",
    "Nozomi",
    "OMACagent",
    "OptimalBidderSimple",
    "PNegotiator",
    "ParsAgent",
    "ParsAgent2",
    "ParsAgent3",
    "ParsCat",
    "ParsCat2",
    "PhoenixParty",
    "PokerFace",
    "PonPokoAgent",
    "PonPokoRampage",
    "Q12015Group2",
    "RandomCounterOfferNegotiationParty",
    "RandomDance",
    "RandomParty",
    "RandomParty2",
    "Rubick",
    "SACRA",
    "SAGA",
    "SENGOKU",
    "SMACAgent",
    "SYAgent",
    "Seto",
    "ShahAgent",
    "Shiboy",
    "SimilarityAgent",
    "Simpatico",
    "SimpleAgent",
    "SimpleAgent2017",
    "SlavaAgent",
    "Sobut",
    "SolverAgent",
    "Sontag",
    "SouthamptonAgent",
    "TMFAgent",
    "TUDMixedStrategyAgent",
    "TaxiBox",
    "Terra",
    "TheFawkes",
    "TheNegotiator",
    "TheNegotiatorReloaded",
    "TheNewDeal",
    "TimeDependentAgentBoulware",
    "TimeDependentAgentConceder",
    "TimeDependentAgentHardliner",
    "TimeDependentAgentLinear",
    "TucAgent",
    "UtilityBasedAcceptor",
    "ValueModelAgent",
    "WhaleAgent",
    "WinkyAgent",
    "XianFaAgent",
    "Y2015Group2",
    "YXAgent",
    "Yeela",
    "Yushu",
]


class ParsAgent3(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.parsagent3.ShahAgent"
        super().__init__(**kwargs)


class MengWan(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.meng_wan.Agent36"
        super().__init__(**kwargs)


class BetaOne(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.beta_one.Group2"
        super().__init__(**kwargs)

class Ateamagent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.ateamagent.ATeamAgent"
        super().__init__(**kwargs)

class AteamAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.ateamagent.ATeamAgent"
        super().__init__(**kwargs)


class Agent33(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.agent33.Agent33"
        super().__init__(**kwargs)


class Agent36(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.meng_wan.Agent36"
        super().__init__(**kwargs)


class AgentF(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.agentf.AgentF"
        super().__init__(**kwargs)


class AgentH(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.agenth.AgentH"
        super().__init__(**kwargs)


class AgentHP(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.AgentHP.AgentHP"
        super().__init__(**kwargs)


class AgentHerb(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.agentherb.AgentHerb"
        super().__init__(**kwargs)


class AgentI(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.GAgent.AgentI"
        super().__init__(**kwargs)


class AgentKN(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.agentkn.AgentKN"
        super().__init__(**kwargs)


class AgentLight(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.agentlight.AgentLight"
        super().__init__(**kwargs)


class AgentNP1(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.agentnp1.AgentNP1"
        super().__init__(**kwargs)


class AgentQuest(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentQuest.AgentQuest"
        super().__init__(**kwargs)


class AgentTD(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentTD.AgentTD"
        super().__init__(**kwargs)


class AgentTRP(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentTRP.AgentTRP"
        super().__init__(**kwargs)


class AgentW(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.AgentW.AgentW"
        super().__init__(**kwargs)


class AgreeableAgent2018(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018"
        super().__init__(**kwargs)


class AnacSampleAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.E2Agent.AnacSampleAgent"
        super().__init__(**kwargs)


class AresParty(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.AresParty.AresParty"
        super().__init__(**kwargs)


class ArisawaYaki(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.ArisawaYaki.ArisawaYaki"
        super().__init__(**kwargs)


class Aster(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.Aster.Aster"
        super().__init__(**kwargs)


class Atlas(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.Atlas.Atlas"
        super().__init__(**kwargs)


class CUHKAgent2015(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.cuhkagent2015.CUHKAgent2015"
        super().__init__(**kwargs)


class ClockworkAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.clockworkagent.ClockworkAgent"
        super().__init__(**kwargs)


class ConDAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.condagent.ConDAgent"
        super().__init__(**kwargs)


class DrageKnight(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.DrageKnight.DrageKnight"
        super().__init__(**kwargs)


class ExpRubick(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.exp_rubick.Exp_Rubick"
        super().__init__(**kwargs)


class Flinch(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.Flinch.Flinch"
        super().__init__(**kwargs)


class FullAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.fullagent.FullAgent"
        super().__init__(**kwargs)


class FunctionalAcceptor(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.FunctionalAcceptor"
        super().__init__(**kwargs)


class Group1(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group1.Group1"
        super().__init__(**kwargs)


class Group10(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group10.Group10"
        super().__init__(**kwargs)


class Group11(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group11.Group11"
        super().__init__(**kwargs)


class Group12(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group12.Group12"
        super().__init__(**kwargs)


class Group13(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group13.Group13"
        super().__init__(**kwargs)


class Group14(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group14.Group14"
        super().__init__(**kwargs)


class Group15(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group15.Group15"
        super().__init__(**kwargs)


class Group16(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group16.Group16"
        super().__init__(**kwargs)


class Group17(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group17.Group17"
        super().__init__(**kwargs)


class Group18(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group18.Group18"
        super().__init__(**kwargs)


class Group19(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group19.Group19"
        super().__init__(**kwargs)


class Group20(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group20.Group20"
        super().__init__(**kwargs)


class Group21(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group21.Group21"
        super().__init__(**kwargs)


class Group22(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group22.Group22"
        super().__init__(**kwargs)


class TUDelftGroup2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.TUDelftGroup2.Group2Agent"
        super().__init__(**kwargs)


class Group3Q2015(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group3.Group3"
        super().__init__(**kwargs)


class Group4(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group4.Group4"
        super().__init__(**kwargs)


class Group5(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group5.Group5"
        super().__init__(**kwargs)


class Group6(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group6.Group6"
        super().__init__(**kwargs)


class Group7(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group7.Group7"
        super().__init__(**kwargs)


class Group8(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group8.Group8"
        super().__init__(**kwargs)


class Group9(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group9.Group9"
        super().__init__(**kwargs)


class GroupY(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.groupy.GroupY"
        super().__init__(**kwargs)


class AgentNeo(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.AgentNeo.Groupn"
        super().__init__(**kwargs)


class IAMcrazyHaggler(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.IAMcrazyHaggler"
        super().__init__(**kwargs)


class IQSun2018(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.iqson.IQSun2018"
        super().__init__(**kwargs)


class JonnyBlack(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.JonnyBlack.JonnyBlack"
        super().__init__(**kwargs)


class HardHeaded(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.HardHeaded.KLH"
        super().__init__(**kwargs)


class Lancelot(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.lancelot.Lancelot"
        super().__init__(**kwargs)


class Libra(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.libra.Libra"
        super().__init__(**kwargs)


class Mamenchis(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.mamenchis.Mamenchis"
        super().__init__(**kwargs)


class MaxOops(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.maxoops.MaxOops"
        super().__init__(**kwargs)


class MeanBot(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.meanBot.MeanBot"
        super().__init__(**kwargs)


class Mercury(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.Mercury.Mercury"
        super().__init__(**kwargs)


class PNegotiator(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.pnegotiator.PNegotiator"
        super().__init__(**kwargs)


class ParsAgent2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.pars2.ParsAgent2"
        super().__init__(**kwargs)


class PonPokoRampage(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.ponpokorampage.PonPokoRampage"
        super().__init__(**kwargs)


class SENGOKU(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.SENGOKU.SENGOKU"
        super().__init__(**kwargs)


class SMACAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.smac_agent.SMAC_Agent"
        super().__init__(**kwargs)


class SYAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.syagent.SYAgent"
        super().__init__(**kwargs)


class Seto(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.seto.Seto"
        super().__init__(**kwargs)


class ShahAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.parsagent3.ShahAgent"
        super().__init__(**kwargs)


class Shiboy(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.shiboy.Shiboy"
        super().__init__(**kwargs)


class Sobut(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.Sobut.Sobut"
        super().__init__(**kwargs)


class Sontag(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.sontag.Sontag"
        super().__init__(**kwargs)


class TUDMixedStrategyAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent"
        super().__init__(**kwargs)


class UtilityBasedAcceptor(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.UtilityBasedAcceptor"
        super().__init__(**kwargs)


class Yeela(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.yeela.Yeela"
        super().__init__(**kwargs)


class Betaone(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2018.beta_one.Group2"
        super().__init__(**kwargs)


class Q12015Group2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "parties.in4010.q12015.group2.Group2"
        super().__init__(**kwargs)


class SimpleAgent2017(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.simpleagent.SimpleAgent"
        super().__init__(**kwargs)


class Group2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.group2.Group2"
        super().__init__(**kwargs)


class Y2015Group2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.group2.Group2"
        super().__init__(**kwargs)


class AgentX(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.AgentX.AgentX"
        super().__init__(**kwargs)


class Gangster(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.Gangster.Gangster"
        super().__init__(**kwargs)


class Simpatico(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.SimpaticoAgent.Simpatico"
        super().__init__(**kwargs)


class ImmediateAcceptor(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.ImmediateAcceptor"
        super().__init__(**kwargs)


class SimpleAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = ["agents.SimpleAgent"]
        super().__init__(**kwargs)


class RandomCounterOfferNegotiationParty(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "negotiator.parties.RandomCounterOfferNegotiationParty"
        super().__init__(**kwargs)


class RandomParty(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "negotiator.parties.RandomParty"
        super().__init__(**kwargs)


class RandomParty2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "negotiator.parties.RandomParty2"
        super().__init__(**kwargs)


class TimeDependentAgentBoulware(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.TimeDependentAgentBoulware"
        super().__init__(**kwargs)


class TimeDependentAgentConceder(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.TimeDependentAgentConceder"
        super().__init__(**kwargs)


class TimeDependentAgentHardliner(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.TimeDependentAgentHardliner"
        super().__init__(**kwargs)


class TimeDependentAgentLinear(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.TimeDependentAgentLinear"
        super().__init__(**kwargs)


class BoulwareNegotiationParty(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "negotiator.parties.BoulwareNegotiationParty"
        super().__init__(**kwargs)


class SimilarityAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.SimilarityAgent"
        super().__init__(**kwargs)


class OptimalBidderSimple(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.OptimalBidderSimple"
        super().__init__(**kwargs)


class ABMPAgent2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.ABMPAgent2"
        super().__init__(**kwargs)


class FuzzyAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.FuzzyAgent"
        super().__init__(**kwargs)


class BayesianAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.BayesianAgent"
        super().__init__(**kwargs)


class ConcederNegotiationParty(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "negotiator.parties.ConcederNegotiationParty"
        super().__init__(**kwargs)


class AgentSmith2016(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.agentsmith.AgentSmith2016"
        super().__init__(**kwargs)


class AgentSmith(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.AgentSmith.AgentSmith"
        super().__init__(**kwargs)


class AgentFSEGA(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.AgentFSEGA.AgentFSEGA"
        super().__init__(**kwargs)


class SouthamptonAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.SouthamptonAgent"
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


class Gahboninho(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.Gahboninho.Gahboninho"
        super().__init__(**kwargs)


class BramAgent(GeniusNegotiator):
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
        kwargs["java_class_name"] = "agents.anac.y2012.CUHKAgent.CUHKAgent"
        super().__init__(**kwargs)


class OMACagent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.OMACagent.OMACagent"
        super().__init__(**kwargs)


class TheNegotiatorReloaded(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded"
        super().__init__(**kwargs)


class BramAgent2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.BramAgent.BRAMAgent"
        super().__init__(**kwargs)


class AgentLG(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.AgentLG.AgentLG"
        super().__init__(**kwargs)


class IAMhaggler(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.IAMhaggler"
        super().__init__(**kwargs)


class IAMhaggler2011(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011"
        super().__init__(**kwargs)


class IAMhaggler2012(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012"
        super().__init__(**kwargs)


class AgentMR(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.AgentMR.AgentMR"
        super().__init__(**kwargs)


class AgentKF(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.AgentKF.AgentKF"
        super().__init__(**kwargs)


class TheFawkes(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.TheFawkes.TheFawkes"
        super().__init__(**kwargs)


class TMFAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.TMFAgent.TMFAgent"
        super().__init__(**kwargs)


class MetaAgent2013(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.MetaAgent.MetaAgent2013"
        super().__init__(**kwargs)


class MetaAgent2012(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.MetaAgent.MetaAgent"
        super().__init__(**kwargs)


class MetaAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.MetaAgent.MetaAgent"
        super().__init__(**kwargs)


class GAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.GAgent.AgentI"
        super().__init__(**kwargs)


class InoxAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.InoxAgent.InoxAgent"
        super().__init__(**kwargs)


class SlavaAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.SlavaAgent.SlavaAgent"
        super().__init__(**kwargs)


class AgentM(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentM.AgentM"
        super().__init__(**kwargs)


class DoNA(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.DoNA.DoNA"
        super().__init__(**kwargs)


class Gangester(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.Gangster.Gangster"
        super().__init__(**kwargs)


class WhaleAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.AgentWhale.WhaleAgent"
        super().__init__(**kwargs)


class AI2014Group2(GeniusNegotiator):
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
        kwargs["java_class_name"] = "agents.anac.y2014.BraveCat.BraveCat"
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


class AgentBuyog(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2015.agentBuyogV2.AgentBuyogMain"
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


class Caduceus(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.caduceus.Caduceus"
        super().__init__(**kwargs)


class YXAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.yxagent.YXAgent"
        super().__init__(**kwargs)


class ParsCat(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.parscat.ParsCat"
        super().__init__(**kwargs)


class Farma(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.farma.Farma"
        super().__init__(**kwargs)


class Farma17(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.farma.Farma17"
        super().__init__(**kwargs)


class Farma2017(GeniusNegotiator):
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


class PonPokoAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.ponpokoagent.PonPokoAgent"
        super().__init__(**kwargs)


class CaduceusDC16(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.caduceusdc16.CaduceusDC16"
        super().__init__(**kwargs)


class Rubick(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.rubick.Rubick"
        super().__init__(**kwargs)


class ParsCat2(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.parscat.ParsCat"
        super().__init__(**kwargs)


class AgentGG(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.agentgg.AgentGG"
        super().__init__(**kwargs)


class AgentGP(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.agentgp.AgentGP"
        super().__init__(**kwargs)


class AgentLarry(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.agentlarry.AgentLarry"
        super().__init__(**kwargs)


class DandikAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.dandikagent.dandikAgent"
        super().__init__(**kwargs)


class EAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.eagent.EAgent"
        super().__init__(**kwargs)


class FSEGA2019(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.fsega2019.agent.FSEGA2019"
        super().__init__(**kwargs)


class GaravelAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.garavelagent.GaravelAgent"
        super().__init__(**kwargs)


class GeneKing(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.geneking.GeneKing"
        super().__init__(**kwargs)


class Gravity(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.gravity.Gravity"
        super().__init__(**kwargs)


class HardDealer(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.harddealer.HardDealer"
        super().__init__(**kwargs)


class KAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.kagent.KAgent"
        super().__init__(**kwargs)


class KakeSoba(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.kakesoba.KakeSoba"
        super().__init__(**kwargs)


class MINF(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.minf.MINF"
        super().__init__(**kwargs)


class Group1BOA(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.podagent.Group1_BOA"
        super().__init__(**kwargs)


class PodAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.podagent.Group1_BOA"
        super().__init__(**kwargs)


class SACRA(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.sacra.SACRA"
        super().__init__(**kwargs)


class SAGA(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.saga.SAGA"
        super().__init__(**kwargs)


class SolverAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.solveragent.SolverAgent"
        super().__init__(**kwargs)


class TheNewDeal(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.thenewdeal.TheNewDeal"
        super().__init__(**kwargs)


class WinkyAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2019.winkyagent.winkyAgent"
        super().__init__(**kwargs)


class Gin(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.gin.Gin"
        super().__init__(**kwargs)


class Group3(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.group3.Group3"
        super().__init__(**kwargs)


class Imitator(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.limitator.Imitator"
        super().__init__(**kwargs)


class MadAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.madagent.MadAgent"
        super().__init__(**kwargs)


class Mosa(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.mosateam.Mosa"
        super().__init__(**kwargs)


class TaxiBox(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.tangxun.taxibox"
        super().__init__(**kwargs)


class TucAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2017.tucagent.TucAgent"
        super().__init__(**kwargs)
