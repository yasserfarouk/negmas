"""
Genius Negotiator
An agent used to connect to GENIUS agents (ver 8.0.4) and allow them to join negotiation mechanisms

"""
import pathlib

import os
import random
import socket
import subprocess
import time
import typing
from typing import Optional, List, Tuple, Sequence

import pkg_resources
from py4j.java_gateway import JavaGateway, CallbackServerParameters, GatewayParameters
from py4j.protocol import Py4JNetworkError

from negmas import (
    SAONegotiator,
    make_discounted_ufun,
    get_domain_issues,
    NEGMAS_CONFIG,
    CONFIG_KEY_GENIUS_BRIDGE_JAR,
)
from negmas import ResponseType, load_genius_domain
from negmas.common import *
from negmas.utilities import UtilityFunction
import json

DEFAULT_JAVA_PORT = 25337
DEFAULT_PYTHON_PORT = 25338

if typing.TYPE_CHECKING:
    from negmas import Outcome

__all__ = [
    "GeniusNegotiator",  # Most abstract kind of agent
    "init_genius_bridge",
    "genius_bridge_is_running",
]

INTERNAL_SEP, ENTRY_SEP, FIELD_SEP = "<<s=s>>", "<<y,y>>", "<<sy>>"

common_gateway: Optional[JavaGateway] = None
common_port: int = 0
java_process = None
python_port: int = 0


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
    "agents.anac.y2011.Nice_Tit_for_Tat.BilateralAgent",
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

agent_based_negotiators = []
party_based_negotiators = [
    # 'agents.ai2014.group12.Group12',
    # 'agents.ai2014.group7.Group7',
    # 'agents.ai2014.group9.Group9',
    # 'agents.ai2014.group1.Group1',
    # 'agents.ai2014.group6.Group6',
    # 'agents.ai2014.group11.Group11',
    # 'agents.ai2014.group10.Group10',
    # 'agents.ai2014.group3.Group3',
    # 'agents.ai2014.group4.Group4',
    # 'agents.ai2014.group5.Group5',
    # 'agents.ai2014.group2.Group2',
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
    # 'agents.anac.y2015.JonnyBlack.JonnyBlack',
    "agents.anac.y2015.AresParty.AresParty",
    "agents.anac.y2015.fairy.kawaii",
    "agents.anac.y2015.Atlas3.Atlas3",
    "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent",
    "agents.anac.y2015.AgentNeo.Groupn",
    "agents.anac.y2015.agentBuyogV2.AgentBuyogMain",
    "agents.anac.y2015.AgentX.AgentX",
    # 'agents.anac.y2015.pnegotiator.BayesLearner',
    # 'agents.anac.y2015.pnegotiator.PNegotiator',
    # 'agents.anac.y2015.group2.Group2',
    "agents.anac.y2017.tucagent.TucAgent",
    "agents.anac.y2017.agentf.AgentF",
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


def init_genius_bridge(path: str = None, port: int = 0, force: bool = False) -> bool:
    """Initializes a genius connection

    Args:
        path: The path to a JAR file that runs negloader
        port: port number to use
        force: Force trial even if an existing bridge is initialized

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
    try:
        subprocess.Popen(  # ['java', '-jar',  path, '--die-on-exit', f'{port}']
            f"java -jar {path} --die-on-exit {port}", shell=True
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
        port: int = None,
        domain_file_name: str = None,
        utility_file_name: str = None,
        keep_issue_names: bool = True,
        keep_value_names: bool = True,
        auto_load_java: bool = False,
        can_propose=True,
        genius_bridge_path: str = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.capabilities["propose"] = can_propose
        self.add_capabilities({"genius": True})
        self.java = None
        self.java_class_name = java_class_name
        self.port = port
        self.connected = self._connect(
            path=genius_bridge_path, port=self.port, auto_load_java=auto_load_java
        )
        self.java_uuid = self._create()
        self.uuid = self.java_uuid
        self.name = self.java_uuid
        self.domain_file_name = str(domain_file_name)
        self.utility_file_name = str(utility_file_name)
        self._my_last_offer = None
        self.keep_issue_names = keep_issue_names
        self._utility_function, self.discount = None, None
        if domain_file_name is not None:
            # we keep original issues details so that we can create appropriate answers to Java
            self.issues = get_domain_issues(
                domain_file_name=domain_file_name,
                keep_issue_names=True,
                keep_value_names=True,
            )
            self.issue_names = [_.name for _ in self.issues]
            self.issue_index = dict(zip(self.issue_names, range(len(self.issue_names))))
        if utility_file_name is not None:
            self._utility_function, self.discount = UtilityFunction.from_genius(
                utility_file_name,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
            )
        self.base_utility = self._utility_function
        pass

    @classmethod
    def negotiators(cls, agent_based=False, party_based=True) -> List[str]:
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

        """
        return self.java.create_agent(self.java_class_name)

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

            - Testing bilateral agent
            >>> if genius_bridge_is_running():
            ...    b = GeniusNegotiator(java_class_name="agents.SimpleAgent")
            ...    print(b.java_name)
            ... else:
            ...    print('Agent SimpleAgent')
            Agent SimpleAgent

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
        return self.java.getName(self.java_uuid)

    def test(self) -> str:
        return self.java.test(self.java_class_name)

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Called when the info starts. Connects to the JVM.
        """
        super().on_negotiation_start(state=state)
        info = self._ami
        if self.discount is not None and self.discount != 1.0:
            self._utility_function = make_discounted_ufun(
                self._utility_function,
                info=info,
                discount_per_round=self.discount,
                power_per_round=1.0,
            )
        n_steps = -1 if info.n_steps is None else int(info.n_steps)  # number of steps
        n_seconds = (
            -1 if info.time_limit is None else int(info.time_limit)
        )  # time limit
        if n_steps * n_seconds > 0:
            # n_seconds take precedence
            n_steps = -1
        self.java.on_negotiation_start(
            self.java_uuid,  # java_uuid
            info.n_negotiators,  # number of agents
            n_steps,
            n_seconds,
            n_seconds > 0,
            self.domain_file_name,  # domain file
            self.utility_file_name,  # Negotiator file
        )

    def propose(self, state: MechanismState) -> "Outcome":
        if not self.capabilities["propose"]:
            return None
        if self._my_last_offer is None:  # never responded before
            response, outcome = self.parse(self.java.choose_action(self.java_uuid))
            if outcome is None:
                return None
            self._my_last_offer = outcome
            return outcome

        # we offered something before
        tmp = self._my_last_offer
        self._my_last_offer = None
        return tmp

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        action = self.java.choose_action(self.java_uuid)
        response, self._my_last_offer = self.parse(action)
        return response

    def parse(self, action: str) -> Tuple[Optional[ResponseType], Optional["Outcome"]]:
        """
        Parses an action into and a ResponseType and an Outcome (if one is included)
        Args:
            action:

        Returns:

        """
        response, outcome = None, None
        id, typ_, bid_str = action.split(FIELD_SEP)
        if typ_ in ("Offer",) and (bid_str is not None and len(bid_str) > 0):
            try:
                if self.keep_issue_names:
                    outcome = {
                        _[0]: _[1]
                        for _ in [
                            _.split(INTERNAL_SEP) for _ in bid_str.split(ENTRY_SEP)
                        ]
                    }
                else:
                    outcome = tuple(
                        _.split(INTERNAL_SEP)[1] for _ in bid_str.split(ENTRY_SEP)
                    )
            except:
                print(f"Failed for bid string: {bid_str} of action {action}")
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

    def on_partner_proposal(
        self, state: MechanismState, agent_id: str, offer: "Outcome"
    ):
        if agent_id is self.id:
            return
        agent_info = [
            _ for _ in self._ami.participants if _.id != self.id and _.id == agent_id
        ]
        if len(agent_info) == 0:
            return
        agent_info = agent_info[0]
        # if agent_info.type == 'genius_negotiator':
        #    return
        self.java.receive_message(
            self.java_uuid, agent_id, "Offer", self._outcome2str(offer)
        )

    def on_partner_response(
        self,
        state: MechanismState,
        agent_id: str,
        outcome: "Outcome",
        response: "ResponseType",
    ):
        if agent_id is self.id:
            return
        agent_info = [
            _ for _ in self._ami.participants if _.id != self.id and _.id == agent_id
        ]
        if len(agent_info) == 0:
            return
        if outcome is None:
            return
        agent_info = agent_info[0]
        # if agent_info.type == 'genius_negotiator':
        #    return
        bid = self._outcome2str(outcome)
        if response == ResponseType.END_NEGOTIATION:
            resp = "EndNegotiation"
        elif response == ResponseType.ACCEPT_OFFER:
            resp = "Accept"
        elif response == ResponseType.REJECT_OFFER:
            resp = "Reject"
        else:
            return
        self.java.receive_message(self.java_uuid, agent_id, resp, bid)


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
