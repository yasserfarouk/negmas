"""Genius negotiator implementations - ANAC 2013 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentI",
    "AgentKF",
    "GAgent",
    "InoxAgent",
    "MetaAgent2013",
    "SlavaAgent",
    "TMFAgent",
    "TheFawkes",
]


class AgentI(GeniusNegotiator):
    """AgentI implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.GAgent.AgentI"
        super().__init__(**kwargs)


class AgentKF(GeniusNegotiator):
    """AgentKF implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.AgentKF.AgentKF"
        super().__init__(**kwargs)


class GAgent(GeniusNegotiator):
    """GAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.GAgent.AgentI"
        super().__init__(**kwargs)


class InoxAgent(GeniusNegotiator):
    """InoxAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.InoxAgent.InoxAgent"
        super().__init__(**kwargs)


class MetaAgent2013(GeniusNegotiator):
    """MetaAgent2013 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.MetaAgent.MetaAgent2013"
        super().__init__(**kwargs)


class SlavaAgent(GeniusNegotiator):
    """SlavaAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.SlavaAgent.SlavaAgent"
        super().__init__(**kwargs)


class TMFAgent(GeniusNegotiator):
    """TMFAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.TMFAgent.TMFAgent"
        super().__init__(**kwargs)


class TheFawkes(GeniusNegotiator):
    """TheFawkes implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.TheFawkes.TheFawkes"
        super().__init__(**kwargs)
