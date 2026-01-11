"""Genius negotiator implementations - ANAC 2010 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentFSEGA",
    "AgentK",
    "AgentSmith",
    "IAMcrazyHaggler",
    "IAMhaggler",
    "Nozomi",
    "SouthamptonAgent",
    "Yushu",
]


class AgentFSEGA(GeniusNegotiator):
    """AgentFSEGA implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.AgentFSEGA.AgentFSEGA"
        super().__init__(**kwargs)


class AgentK(GeniusNegotiator):
    """AgentK implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.AgentK.Agent_K"
        super().__init__(**kwargs)


class AgentSmith(GeniusNegotiator):
    """AgentSmith implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.AgentSmith.AgentSmith"
        super().__init__(**kwargs)


class IAMcrazyHaggler(GeniusNegotiator):
    """IAMcrazyHaggler implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.IAMcrazyHaggler"
        super().__init__(**kwargs)


class IAMhaggler(GeniusNegotiator):
    """
    IAMhaggler negotiation agent.

    **ANAC 2012 Winner** (Nash Product category).

    IAMhaggler (developed by Colin R. Williams at University of Southampton)
    uses Gaussian Process regression to predict opponent behavior and
    optimize concession timing.

    **Offering Strategy:**
        - Uses Gaussian Process to predict when opponent will make maximum
          concession
        - Calculates expected utility surface over time and utility dimensions
        - Target utility based on interpolation toward predicted best agreement
        - Limits concession based on observed opponent bidding range
        - Risk-aware utility function with configurable risk parameter

    **Acceptance Strategy:**
        - Accepts if opponent's offer * multiplier >= target utility
        - Accepts if opponent's offer * multiplier >= maximum aspiration (0.9)
        - Accepts if opponent's offer >= planned bid utility
        - Multiple acceptance thresholds for robustness

    **Opponent Modeling:**
        Gaussian Process regression approach:

        - Tracks opponent utilities over time slots
        - Fits GP model to predict future opponent concessions
        - Calculates probability distribution of opponent's future offers
        - Uses predictions to optimize concession timing
        - Incorporates discounting factor for time-sensitive domains

    References:
        Williams, C.R., Robu, V., Gerding, E.H., & Jennings, N.R. (2012).
        IAMhaggler: A Negotiation Agent for Complex Environments.
        In New Trends in Agent-Based Complex Automated Negotiations.
        Studies in Computational Intelligence, vol 383. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.IAMhaggler"
        super().__init__(**kwargs)


class Nozomi(GeniusNegotiator):
    """Nozomi implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Nozomi.Nozomi"
        super().__init__(**kwargs)


class SouthamptonAgent(GeniusNegotiator):
    """SouthamptonAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.SouthamptonAgent"
        super().__init__(**kwargs)


class Yushu(GeniusNegotiator):
    """Yushu implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Yushu.Yushu"
        super().__init__(**kwargs)
