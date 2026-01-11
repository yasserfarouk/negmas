"""Genius negotiator implementations - ANAC 2012 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentLG",
    "AgentMR",
    "CUHKAgent",
    "IAMhaggler2012",
    "MetaAgent",
    "MetaAgent2012",
    "OMACagent",
    "TheNegotiatorReloaded",
]


class AgentLG(GeniusNegotiator):
    """AgentLG implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.AgentLG.AgentLG"
        super().__init__(**kwargs)


class AgentMR(GeniusNegotiator):
    """AgentMR implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.AgentMR.AgentMR"
        super().__init__(**kwargs)


class CUHKAgent(GeniusNegotiator):
    """
    CUHKAgent negotiation agent.

    **ANAC 2012 Winner**.

    CUHKAgent (developed at Chinese University of Hong Kong by Jianye Hao)
    is an adaptive negotiation agent that adjusts its strategy based on
    opponent behavior and time pressure.

    **Offering Strategy:**
        - Time-dependent concession with adaptive threshold adjustment
        - Concession rate adapts based on opponent's toughness degree
        - In large domains: focuses on high-utility bid range
        - Near deadline: considers opponent's best offer as fallback
        - Uses opponent model to select bids favorable to opponent among
          candidates

    **Acceptance Strategy:**
        - Accepts if offer exceeds current utility threshold
        - Accepts if offer exceeds the utility of planned counter-offer
        - Near deadline: more lenient acceptance based on opponent's best offer
        - Adapts acceptance based on predicted maximum achievable utility

    **Opponent Modeling:**
        - Tracks opponent's bidding history to estimate preferences
        - Calculates opponent's concession degree to adapt own strategy
        - Identifies opponent's maximum offered bid for reference
        - Uses opponent model to choose mutually beneficial bids

    References:
        Hao, J., & Leung, H. (2014). CUHKAgent: An Adaptive Negotiation Strategy
        for Bilateral Negotiations over Multiple Items. In Novel Insights in
        Agent-based Complex Automated Negotiation. Studies in Computational
        Intelligence, vol 535. Springer, Tokyo.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.CUHKAgent.CUHKAgent"
        super().__init__(**kwargs)


class IAMhaggler2012(GeniusNegotiator):
    """IAMhaggler2012 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012"
        super().__init__(**kwargs)


class MetaAgent(GeniusNegotiator):
    """MetaAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.MetaAgent.MetaAgent"
        super().__init__(**kwargs)


class MetaAgent2012(GeniusNegotiator):
    """MetaAgent2012 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.MetaAgent.MetaAgent"
        super().__init__(**kwargs)


class OMACagent(GeniusNegotiator):
    """OMACagent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.OMACagent.OMACagent"
        super().__init__(**kwargs)


class TheNegotiatorReloaded(GeniusNegotiator):
    """TheNegotiatorReloaded implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded"
        super().__init__(**kwargs)
