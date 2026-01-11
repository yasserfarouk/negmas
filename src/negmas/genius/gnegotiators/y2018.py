"""Genius negotiator implementations - ANAC 2018 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "Agent33",
    "Agent36",
    "AgentHerb",
    "AgentNP1",
    "AgreeableAgent2018",
    "AteamAgent",
    "Ateamagent",
    "BetaOne",
    "Betaone",
    "ConDAgent",
    "ExpRubick",
    "FullAgent",
    "GroupY",
    "IQSun2018",
    "Lancelot",
    "Libra",
    "MengWan",
    "PonPokoRampage",
    "SMACAgent",
    "Seto",
    "Shiboy",
    "Sontag",
    "Yeela",
]


class Agent33(GeniusNegotiator):
    """Agent33 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.agent33.Agent33"
        super().__init__(**kwargs)


class Agent36(GeniusNegotiator):
    """Agent36 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.meng_wan.Agent36"
        super().__init__(**kwargs)


class AgentHerb(GeniusNegotiator):
    """AgentHerb implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.agentherb.AgentHerb"
        super().__init__(**kwargs)


class AgentNP1(GeniusNegotiator):
    """AgentNP1 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.agentnp1.AgentNP1"
        super().__init__(**kwargs)


class AgreeableAgent2018(GeniusNegotiator):
    """AgreeableAgent2018 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018"
        super().__init__(**kwargs)


class AteamAgent(GeniusNegotiator):
    """AteamAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.ateamagent.ATeamAgent"
        super().__init__(**kwargs)


class Ateamagent(GeniusNegotiator):
    """Ateamagent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.ateamagent.ATeamAgent"
        super().__init__(**kwargs)


class BetaOne(GeniusNegotiator):
    """BetaOne implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.beta_one.Group2"
        super().__init__(**kwargs)


class Betaone(GeniusNegotiator):
    """Betaone implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.beta_one.Group2"
        super().__init__(**kwargs)


class ConDAgent(GeniusNegotiator):
    """ConDAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.condagent.ConDAgent"
        super().__init__(**kwargs)


class ExpRubick(GeniusNegotiator):
    """ExpRubick implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.exp_rubick.Exp_Rubick"
        super().__init__(**kwargs)


class FullAgent(GeniusNegotiator):
    """FullAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.fullagent.FullAgent"
        super().__init__(**kwargs)


class GroupY(GeniusNegotiator):
    """GroupY implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.groupy.GroupY"
        super().__init__(**kwargs)


class IQSun2018(GeniusNegotiator):
    """IQSun2018 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.iqson.IQSun2018"
        super().__init__(**kwargs)


class Lancelot(GeniusNegotiator):
    """Lancelot implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.lancelot.Lancelot"
        super().__init__(**kwargs)


class Libra(GeniusNegotiator):
    """Libra implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.libra.Libra"
        super().__init__(**kwargs)


class MengWan(GeniusNegotiator):
    """MengWan implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.meng_wan.Agent36"
        super().__init__(**kwargs)


class PonPokoRampage(GeniusNegotiator):
    """PonPokoRampage implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.ponpokorampage.PonPokoRampage"
        super().__init__(**kwargs)


class SMACAgent(GeniusNegotiator):
    """SMACAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.smac_agent.SMAC_Agent"
        super().__init__(**kwargs)


class Seto(GeniusNegotiator):
    """Seto implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.seto.Seto"
        super().__init__(**kwargs)


class Shiboy(GeniusNegotiator):
    """Shiboy implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.shiboy.Shiboy"
        super().__init__(**kwargs)


class Sontag(GeniusNegotiator):
    """Sontag implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.sontag.Sontag"
        super().__init__(**kwargs)


class Yeela(GeniusNegotiator):
    """Yeela implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.yeela.Yeela"
        super().__init__(**kwargs)
