"""Genius negotiator implementations - ANAC 2014 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentM",
    "AgentQuest",
    "AgentTD",
    "AgentTRP",
    "AgentYK",
    "AnacSampleAgent",
    "ArisawaYaki",
    "Aster",
    "Atlas",
    "BraveCat",
    "DoNA",
    "E2Agent",
    "Flinch",
    "Gangester",
    "Gangster",
    "KGAgent",
    "Simpatico",
    "Sobut",
    "TUDelftGroup2",
    "WhaleAgent",
]


class AgentM(GeniusNegotiator):
    """AgentM implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentM.AgentM"
        super().__init__(**kwargs)


class AgentQuest(GeniusNegotiator):
    """AgentQuest implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentQuest.AgentQuest"
        super().__init__(**kwargs)


class AgentTD(GeniusNegotiator):
    """AgentTD implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentTD.AgentTD"
        super().__init__(**kwargs)


class AgentTRP(GeniusNegotiator):
    """AgentTRP implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentTRP.AgentTRP"
        super().__init__(**kwargs)


class AgentYK(GeniusNegotiator):
    """AgentYK implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentYK.AgentYK"
        super().__init__(**kwargs)


class AnacSampleAgent(GeniusNegotiator):
    """AnacSampleAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.E2Agent.AnacSampleAgent"
        super().__init__(**kwargs)


class ArisawaYaki(GeniusNegotiator):
    """ArisawaYaki implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.ArisawaYaki.ArisawaYaki"
        super().__init__(**kwargs)


class Aster(GeniusNegotiator):
    """Aster implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Aster.Aster"
        super().__init__(**kwargs)


class Atlas(GeniusNegotiator):
    """Atlas implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Atlas.Atlas"
        super().__init__(**kwargs)


class BraveCat(GeniusNegotiator):
    """BraveCat implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.BraveCat.BraveCat"
        super().__init__(**kwargs)


class DoNA(GeniusNegotiator):
    """DoNA implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.DoNA.DoNA"
        super().__init__(**kwargs)


class E2Agent(GeniusNegotiator):
    """E2Agent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.E2Agent.AnacSampleAgent"
        super().__init__(**kwargs)


class Flinch(GeniusNegotiator):
    """Flinch implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Flinch.Flinch"
        super().__init__(**kwargs)


class Gangester(GeniusNegotiator):
    """Gangester implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Gangster.Gangster"
        super().__init__(**kwargs)


class Gangster(GeniusNegotiator):
    """Gangster implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Gangster.Gangster"
        super().__init__(**kwargs)


class KGAgent(GeniusNegotiator):
    """KGAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.KGAgent.KGAgent"
        super().__init__(**kwargs)


class Simpatico(GeniusNegotiator):
    """Simpatico implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.SimpaticoAgent.Simpatico"
        super().__init__(**kwargs)


class Sobut(GeniusNegotiator):
    """Sobut implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Sobut.Sobut"
        super().__init__(**kwargs)


class TUDelftGroup2(GeniusNegotiator):
    """TUDelftGroup2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.TUDelftGroup2.Group2Agent"
        super().__init__(**kwargs)


class WhaleAgent(GeniusNegotiator):
    """WhaleAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentWhale.WhaleAgent"
        super().__init__(**kwargs)
