from __future__ import annotations

from .negotiator import GeniusNegotiator

__all__ = [
    "ABMPAgent2",
    "AgentBuyong",
    "AgentHP2",
    "AgentK",
    "AgentK2",
    "AgentLG",
    "AgentM",
    "AgentX",
    "AgentYK",
    "AgreeableAgent2018",
    "Atlas3",
    "Atlas32016",
    "BRAMAgent",
    "BayesianAgent",
    "BetaOne",
    "BoulwareNegotiationParty",
    "BraveCat",
    "CUHKAgent",
    "Caduceus",
    "CaduceusDC16",
    "ConcederNegotiationParty",
    "DoNA",
    "E2Agent",
    "Farma",
    "FuzzyAgent",
    "Gahboninho",
    "Gangster",
    "GrandmaAgent",
    "Group2",
    "HardHeaded",
    "IAMhaggler",
    "IAMhaggler2011",
    "ImmediateAcceptor",
    "KGAgent",
    "Kawaii",
    "MengWan",
    "MetaAgent",
    "MyAgent",
    "Ngent",
    "NiceTitForTat",
    "Nozomi",
    "OMACagent",
    "OptimalBidderSimple",
    "ParsAgent",
    "ParsCat",
    "PhoenixParty",
    "PokerFace",
    "PonPokoAgent",
    "RandomCounterOfferNegotiationParty",
    "RandomDance",
    "RandomParty",
    "RandomParty2",
    "Rubick",
    "SimilarityAgent",
    "Simpatico",
    "SimpleAgent",
    "TMFAgent",
    "Terra",
    "TheFawkes",
    "TheNegotiator",
    "TheNegotiatorReloaded",
    "TimeDependentAgentBoulware",
    "TimeDependentAgentConceder",
    "TimeDependentAgentHardliner",
    "TimeDependentAgentLinear",
    "ValueModelAgent",
    "WhaleAgent",
    "XianFaAgent",
    "YXAgent",
    "Yushu",
]


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
        kwargs["java_class_name"] = "agents.anac.y2012.CUHKAgent.CUHKAgent"
        super().__init__(**kwargs)


class AgentLG(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.AgentLG.AgentLG"
        super().__init__(**kwargs)


class OMACagent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.OMACagent.OMACagent"
        super().__init__(**kwargs)


class TheNegotiatorReloaded(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded"
        super().__init__(**kwargs)


class MetaAgent(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.MetaAgent.MetaAgent2013"
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


class AgentBuyong(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2016.caduceus.agents.agentBuyong.agentBuyong"
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
        kwargs["java_class_name"] = "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018"
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


class Gangster(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.Gangster.Gangster"
        super().__init__(**kwargs)


class Simpatico(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2014.SimpaticoAgent.Simpatico"
        super().__init__(**kwargs)


class TheFawkes(GeniusNegotiator):
    def __init__(self, **kwargs):
        kwargs["java_class_name"] = "agents.anac.y2013.TheFawkes.TheFawkes"
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
        kwargs["java_class_name"] = "negotiator.parties.RandomCounterOfferNegotiationParty"
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
